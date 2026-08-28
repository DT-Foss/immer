#!/usr/bin/env python3
"""Fit a diagonal sparse-MLP residual on generation one, then open generation two."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from safetensors import safe_open
import torch
import torch.nn.functional as F

from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.mlp_pilot_output import (
    MlpPilotOutputMetricAccumulator,
    pilot_sparse_down_projection,
)
from immer.runtimes.ooe.mlp_pilot_residual import (
    PILOT_AFFINE_VERIFIER_SHA256,
    MlpPilotAffineAccumulator,
    MlpPilotAffineEvaluation,
    MlpPilotAffineFit,
)
from immer.runtimes.ooe.mlp_pilot_router import (
    MlpPilotRouterEvaluation,
    MlpPilotRouterFit,
    verify_mlp_pilot_router_evaluation,
    verify_mlp_pilot_router_fit,
)
from immer.runtimes.ooe.prompt_row_roles import PromptRowRoleManifest
from immer.runtimes.ooe.qwen_mlp_evidence import (
    MlpEvidenceReceipt,
    QwenMlpEvidenceBank,
)
from immer.runtimes.ooe.subspace_battery import SubspaceObservationGroup

from qwen38_mlp_pilot_router import _digest, _persist_exact, _stable_read


FIT_NAME = "affine-fit.json"
EVALUATION_NAME = "affine-evaluation.json"
REPORT_NAME = "affine-report.json"
REPORT_SCHEMA = "immer.qwen3.8-mlp-pilot-affine-live-report/v1"


class CliError(RuntimeError):
    pass


def _tensor_name(layer: int) -> str:
    return f"model.language_model.layers.{layer}.mlp.down_proj.weight"


def _pairs(bank: QwenMlpEvidenceBank) -> dict[tuple[int, str], MlpEvidenceReceipt]:
    return {
        (receipt.entry.layer, receipt.entry.prompt_sha256): receipt
        for receipt, _verification in bank.committed_pairs()
    }


def _sparse_group(
    *,
    group: SubspaceObservationGroup,
    bank: QwenMlpEvidenceBank,
    pairs: Mapping[tuple[int, str], MlpEvidenceReceipt],
    roles: PromptRowRoleManifest,
    weight: torch.Tensor,
    model,
) -> tuple[torch.Tensor, torch.Tensor]:
    receipt = pairs.get((group.layer, group.prompt_sha256))
    if receipt is None:
        raise CliError("MLP receipt inventory is incomplete")
    references = {reference.stage: reference for reference in receipt.tensors}
    output = bank.restore_tensor(references["mlp.output"]).reshape(
        -1, group.hidden_dimension
    )
    rows = roles.row_indices_by_prompt[group.prompt_sha256]
    truth = torch.from_numpy(output[list(rows)].astype(np.float32))
    gate = torch.from_numpy(group.gate_projection.astype(np.float32)).to(torch.bfloat16)
    up = torch.from_numpy(group.up_projection.astype(np.float32)).to(torch.bfloat16)
    activated = F.silu(gate) * up
    selected = model.select_blocks_from_full(group.gate_projection, group.up_projection)
    return pilot_sparse_down_projection(model, activated, weight, selected), truth


def _weight_index(weights_root: Path) -> tuple[bytes, Mapping[str, object]]:
    raw = _stable_read(weights_root / "model.safetensors.index.json")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CliError("weights index is not JSON") from exc
    if not isinstance(value, dict) or not isinstance(value.get("weight_map"), dict):
        raise CliError("weights index has no tensor map")
    return raw, value["weight_map"]


def _weight(
    weights_root: Path,
    weight_map: Mapping[str, object],
    layer: int,
    hidden: int,
    intermediate: int,
) -> torch.Tensor:
    name = _tensor_name(layer)
    shard = weight_map.get(name)
    if not isinstance(shard, str) or Path(shard).name != shard:
        raise CliError(f"weights index lacks a safe shard for {name}")
    with safe_open(str(weights_root / shard), framework="pt", device="cpu") as reader:
        result = reader.get_tensor(name)
    if result.shape != (hidden, intermediate):
        raise CliError(f"down-projection ABI changed at layer {layer}")
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.set_num_threads(args.threads)
    analysis = Path(args.analysis_root).expanduser().absolute()
    weights_root = Path(args.weights_root).expanduser().absolute()
    router_fit = MlpPilotRouterFit.from_bytes(_stable_read(analysis / "fit.json"))
    source_roles = PromptRowRoleManifest.from_bytes(
        _stable_read(analysis / "fit-row-roles.json")
    )
    if source_roles.sha256 != router_fit.row_role_sha256:
        raise CliError("source row roles differ from the router fit")
    source_bank = QwenMlpEvidenceBank(
        Path(args.source_bank_root).expanduser().absolute()
    )
    if source_bank.state().split_counts != (25, 10, 5) or not source_bank.audit().clean:
        raise CliError("source bank is not a clean complete 25/10/5 bank")
    source_corpus = source_bank.build_subspace_corpus(
        row_indices_by_prompt=source_roles.row_indices_by_prompt
    )
    if source_corpus.sha256 != router_fit.corpus_sha256:
        raise CliError("source corpus differs from the router fit")
    verify_mlp_pilot_router_fit(router_fit, source_corpus)
    source_rows = sum(group.row_count for group in source_corpus.groups)
    if source_rows > args.max_rows:
        raise CliError("source residual fit exceeds its row budget")
    index_raw, weight_map = _weight_index(weights_root)
    router_models = {model.layer: model for model in router_fit.models}
    source_pairs = _pairs(source_bank)
    affine_models = []
    for layer in sorted(router_models):
        model = router_models[layer]
        weight = _weight(
            weights_root,
            weight_map,
            layer,
            source_corpus.hidden_dimension,
            source_corpus.intermediate_dimension,
        )
        accumulator = MlpPilotAffineAccumulator(source_corpus.hidden_dimension)
        training_groups = []
        for group in (row for row in source_corpus.groups if row.layer == layer):
            sparse, truth = _sparse_group(
                group=group,
                bank=source_bank,
                pairs=source_pairs,
                roles=source_roles,
                weight=weight,
                model=model,
            )
            accumulator.add(sparse, truth)
            training_groups.append(group.sha256)
        affine_models.append(
            accumulator.fit(layer=layer, training_group_sha256s=training_groups)
        )
    affine_fit = MlpPilotAffineFit(
        model_pin_sha256=router_fit.model_pin_sha256,
        router_fit_sha256=router_fit.sha256,
        source_bank_state_sha256=source_bank.state().sha256,
        source_corpus_sha256=source_corpus.sha256,
        source_row_role_sha256=source_roles.sha256,
        weights_index_sha256=hashlib.sha256(index_raw).hexdigest(),
        source_row_count=source_rows,
        models=tuple(affine_models),
    )
    # External generation remains unopened until the complete affine fit exists.
    _persist_exact(analysis / FIT_NAME, affine_fit.to_bytes())

    router_evaluation = MlpPilotRouterEvaluation.from_bytes(
        _stable_read(analysis / "evaluation.json")
    )
    holdout_roles = PromptRowRoleManifest.from_bytes(
        _stable_read(analysis / "holdout-row-roles.json")
    )
    holdout_bank = QwenMlpEvidenceBank(
        Path(args.holdout_bank_root).expanduser().absolute()
    )
    if (
        holdout_bank.state().split_counts != (25, 10, 5)
        or not holdout_bank.audit().clean
    ):
        raise CliError("holdout bank is not a clean complete 25/10/5 bank")
    holdout_corpus = holdout_bank.build_subspace_corpus(
        row_indices_by_prompt=holdout_roles.row_indices_by_prompt
    )
    verify_mlp_pilot_router_evaluation(router_evaluation, router_fit, holdout_corpus)
    holdout_rows = sum(group.row_count for group in holdout_corpus.groups)
    if holdout_rows > args.max_rows:
        raise CliError("holdout residual evaluation exceeds its row budget")
    holdout_pairs = _pairs(holdout_bank)
    affine_by_layer = {model.layer: model for model in affine_fit.models}
    raw_global = MlpPilotOutputMetricAccumulator()
    corrected_global = MlpPilotOutputMetricAccumulator()
    layer_metrics = []
    for layer in sorted(router_models):
        weight = _weight(
            weights_root,
            weight_map,
            layer,
            holdout_corpus.hidden_dimension,
            holdout_corpus.intermediate_dimension,
        )
        raw = MlpPilotOutputMetricAccumulator()
        corrected = MlpPilotOutputMetricAccumulator()
        for group in (row for row in holdout_corpus.groups if row.layer == layer):
            sparse, truth = _sparse_group(
                group=group,
                bank=holdout_bank,
                pairs=holdout_pairs,
                roles=holdout_roles,
                weight=weight,
                model=router_models[layer],
            )
            repaired = affine_by_layer[layer].apply(sparse)
            raw.add(sparse, truth)
            corrected.add(repaired, truth)
            raw_global.add(sparse, truth)
            corrected_global.add(repaired, truth)
        layer_metrics.append((layer, raw.result(), corrected.result()))
    affine_evaluation = MlpPilotAffineEvaluation(
        fit_sha256=affine_fit.sha256,
        router_evaluation_sha256=router_evaluation.sha256,
        holdout_bank_state_sha256=holdout_bank.state().sha256,
        holdout_corpus_sha256=holdout_corpus.sha256,
        holdout_row_role_sha256=holdout_roles.sha256,
        row_count=holdout_rows,
        raw_metrics=raw_global.result(),
        corrected_metrics=corrected_global.result(),
        layer_metrics=tuple(layer_metrics),
    )
    _persist_exact(analysis / EVALUATION_NAME, affine_evaluation.to_bytes())
    body = {
        "corrected_metrics": affine_evaluation.corrected_metrics.to_record(),
        "evaluation_sha256": affine_evaluation.sha256,
        "fit_sha256": affine_fit.sha256,
        "l2_error_reduction": affine_evaluation.l2_error_reduction,
        "model_pin_sha256": affine_fit.model_pin_sha256,
        "promoted": False,
        "quality_status": "verified_affine_sparse_draft",
        "raw_metrics": affine_evaluation.raw_metrics.to_record(),
        "router_evaluation_sha256": router_evaluation.sha256,
        "source_bank_state_sha256": source_bank.state().sha256,
        "source_row_count": source_rows,
        "verifier_sha256": PILOT_AFFINE_VERIFIER_SHA256,
    }
    report = {
        "body": body,
        "body_sha256": _digest(body),
        "schema": REPORT_SCHEMA,
    }
    _persist_exact(analysis / REPORT_NAME, canonical_json_bytes(report) + b"\n")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--source-bank-root", required=True)
    parser.add_argument("--holdout-bank-root", required=True)
    parser.add_argument("--weights-root", required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--max-rows", type=int, default=16_384)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.threads < 1 or args.max_rows < 1:
        raise CliError("thread and row budgets must be positive")
    report = run(args)
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
