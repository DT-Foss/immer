#!/usr/bin/env python3
"""Verify a frozen Qwen MLP pilot route against every exact down-projection output."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from safetensors import safe_open
import torch
import torch.nn.functional as F

from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.mlp_pilot_output import (
    PILOT_OUTPUT_KERNEL,
    MlpPilotOutputMetricAccumulator,
    marginal_sparse_down_projection,
    pilot_sparse_down_projection,
)
from immer.runtimes.ooe.mlp_pilot_router import (
    MlpPilotRouterEvaluation,
    MlpPilotRouterFit,
    verify_mlp_pilot_router_evaluation,
)
from immer.runtimes.ooe.prompt_row_roles import PromptRowRoleManifest
from immer.runtimes.ooe.qwen_mlp_evidence import QwenMlpEvidenceBank

from qwen38_mlp_pilot_router import _digest, _persist_exact, _stable_read


REPORT_SCHEMA = "immer.qwen3.8-mlp-pilot-output-verification/v1"
REPORT_NAME = "output-verification.json"


class CliError(RuntimeError):
    pass


def _tensor_name(layer: int) -> str:
    return f"model.language_model.layers.{layer}.mlp.down_proj.weight"


def _finish(
    values: dict[str, MlpPilotOutputMetricAccumulator],
) -> dict[str, dict[str, object]]:
    return {
        name: accumulator.result().to_record() for name, accumulator in values.items()
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    analysis = Path(args.analysis_root).expanduser().absolute()
    fit = MlpPilotRouterFit.from_bytes(_stable_read(analysis / "fit.json"))
    evaluation = MlpPilotRouterEvaluation.from_bytes(
        _stable_read(analysis / "evaluation.json")
    )
    if evaluation.fit_sha256 != fit.sha256:
        raise CliError("energy evaluation differs from the persisted fit")
    row_roles = PromptRowRoleManifest.from_bytes(
        _stable_read(analysis / "holdout-row-roles.json")
    )
    if row_roles.sha256 != evaluation.holdout_row_role_sha256:
        raise CliError("output row roles differ from the energy evaluation")
    bank = QwenMlpEvidenceBank(Path(args.holdout_bank_root).expanduser().absolute())
    if bank.state().split_counts != (25, 10, 5) or not bank.audit().clean:
        raise CliError("output bank is not a clean complete 25/10/5 bank")
    corpus = bank.build_subspace_corpus(
        row_indices_by_prompt=row_roles.row_indices_by_prompt
    )
    if (
        corpus.sha256 != evaluation.holdout_corpus_sha256
        or corpus.model_pin_sha256 != fit.model_pin_sha256
    ):
        raise CliError("output corpus differs from the frozen energy holdout")
    verify_mlp_pilot_router_evaluation(evaluation, fit, corpus)
    row_count = sum(group.row_count for group in corpus.groups)
    values = row_count * corpus.hidden_dimension
    if row_count > args.max_rows or values > args.max_values:
        raise CliError("output verification exceeds its row/value budget")

    weights_root = Path(args.weights_root).expanduser().absolute()
    index_path = weights_root / "model.safetensors.index.json"
    index_raw = _stable_read(index_path, 64 * 1024 * 1024)
    try:
        index = json.loads(index_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CliError("weights index is not JSON") from exc
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise CliError("weights index has no tensor map")
    weight_map = index["weight_map"]
    models = {model.layer: model for model in fit.models}
    pairs = bank.committed_pairs()
    pair_map = {
        (receipt.entry.layer, receipt.entry.prompt_sha256): receipt
        for receipt, _verification in pairs
    }
    global_metrics = {
        "pilot": MlpPilotOutputMetricAccumulator(),
        "marginal": MlpPilotOutputMetricAccumulator(),
    }
    layer_metrics = {
        layer: {
            "pilot": MlpPilotOutputMetricAccumulator(),
            "marginal": MlpPilotOutputMetricAccumulator(),
        }
        for layer in models
    }
    reconstruction_layers = []
    tensor_shards = {}
    torch.set_num_threads(args.threads)
    for layer in sorted(models):
        model = models[layer]
        name = _tensor_name(layer)
        shard = weight_map.get(name)
        if not isinstance(shard, str) or Path(shard).name != shard:
            raise CliError(f"weights index lacks a safe shard for {name}")
        tensor_shards[name] = shard
        with safe_open(
            str(weights_root / shard), framework="pt", device="cpu"
        ) as reader:
            weight = reader.get_tensor(name)
        if weight.shape != (corpus.hidden_dimension, corpus.intermediate_dimension):
            raise CliError(f"down-projection ABI changed at layer {layer}")
        reconstruction_checked = False
        for group in (row for row in corpus.groups if row.layer == layer):
            receipt = pair_map.get((layer, group.prompt_sha256))
            if receipt is None:
                raise CliError("output receipt inventory is incomplete")
            references = {reference.stage: reference for reference in receipt.tensors}
            output = bank.restore_tensor(references["mlp.output"]).reshape(
                -1, corpus.hidden_dimension
            )
            indices = row_roles.row_indices_by_prompt[group.prompt_sha256]
            truth = torch.from_numpy(output[list(indices)].astype(np.float32))
            gate = torch.from_numpy(group.gate_projection.astype(np.float32)).to(
                torch.bfloat16
            )
            up = torch.from_numpy(group.up_projection.astype(np.float32)).to(
                torch.bfloat16
            )
            activated = F.silu(gate) * up
            if not reconstruction_checked:
                if not torch.equal(F.linear(activated[:1], weight).float(), truth[:1]):
                    raise CliError(
                        f"full output reconstruction failed at layer {layer}"
                    )
                reconstruction_layers.append(layer)
                reconstruction_checked = True
            selected = model.select_blocks_from_full(
                group.gate_projection, group.up_projection
            )
            predictions = {
                "pilot": pilot_sparse_down_projection(
                    model, activated, weight, selected
                ),
                "marginal": marginal_sparse_down_projection(model, activated, weight),
            }
            for policy, predicted in predictions.items():
                global_metrics[policy].add(predicted, truth)
                layer_metrics[layer][policy].add(predicted, truth)
    if reconstruction_layers != sorted(models):
        raise CliError("not every fitted layer reconstructed an exact full output")

    global_results = _finish(global_metrics)
    body = {
        "bank_state_sha256": bank.state().sha256,
        "energy_evaluation_sha256": evaluation.sha256,
        "exact_full_reconstruction_layers": reconstruction_layers,
        "fit_sha256": fit.sha256,
        "global_metrics": global_results,
        "kernel": PILOT_OUTPUT_KERNEL,
        "layer_metrics": {
            str(layer): _finish(layer_metrics[layer]) for layer in sorted(layer_metrics)
        },
        "l2_error_reduction_vs_marginal": (
            1.0
            - cast_float(global_results["pilot"]["relative_l2_error"])
            / cast_float(global_results["marginal"]["relative_l2_error"])
        ),
        "model_pin_sha256": fit.model_pin_sha256,
        "output_dimensions": corpus.hidden_dimension,
        "promoted": False,
        "quality_status": "verified_sparse_draft",
        "row_count": row_count,
        "row_role_sha256": row_roles.sha256,
        "selected_fraction": fit.calibration_metrics.selected_fraction,
        "selected_neuron_count": fit.calibration_metrics.selected_neuron_count,
        "tensor_shards": tensor_shards,
        "values": values,
        "weights_index_sha256": hashlib.sha256(index_raw).hexdigest(),
    }
    report = {
        "body": body,
        "body_sha256": _digest(body),
        "schema": REPORT_SCHEMA,
    }
    output_path = (
        analysis / REPORT_NAME
        if args.output is None
        else Path(args.output).expanduser().absolute()
    )
    _persist_exact(output_path, canonical_json_bytes(report) + b"\n")
    return report


def cast_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CliError("metric is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CliError("metric is not finite")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--holdout-bank-root", required=True)
    parser.add_argument("--weights-root", required=True)
    parser.add_argument("--output")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--max-rows", type=int, default=16_384)
    parser.add_argument("--max-values", type=int, default=100_000_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.threads < 1 or args.max_rows < 1 or args.max_values < 1:
        raise CliError("thread and verification budgets must be positive")
    report = run(args)
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
