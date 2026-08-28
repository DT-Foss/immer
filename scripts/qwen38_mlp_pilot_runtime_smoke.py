#!/usr/bin/env python3
"""Execute one real Qwen MLP row through only pilot and selected weight ranges."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Sequence

import numpy as np
import torch

from immer.knowledge.streamer import Streamer
from immer.runtimes.deepseek_v4.causal_weights import (
    CausalWeightMount,
    LogicalModelIdentity,
)
from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.mlp_pilot_residual import MlpPilotAffineFit
from immer.runtimes.ooe.mlp_pilot_router import MlpPilotRouterFit
from immer.runtimes.ooe.mlp_pilot_runtime import (
    MlpPilotSparseExecutor,
    MlpPilotTransposeManifest,
)
from immer.runtimes.ooe.prompt_row_roles import PromptRowRoleManifest
from immer.runtimes.ooe.qwen_mlp_evidence import QwenMlpEvidenceBank
from immer.runtimes.qwen3_8.pager import Qwen38WeightPager
from immer.runtimes.qwen3_8.kernels import swiglu
from immer.runtimes.qwen3_8.config import (
    OFFICIAL_REPO_ID,
    OFFICIAL_REVISION,
    Qwen38Config,
)
from immer.runtimes.qwen3_8.model import StreamedQwen38

from qwen38_mlp_pilot_router import _digest, _persist_exact, _stable_read


REPORT_NAME = "sparse-runtime-smoke-v4.json"
REPORT_SCHEMA = "immer.qwen3.8-mlp-pilot-sparse-runtime-smoke/v4"


class CliError(RuntimeError):
    pass


def _delta(after: dict, before: dict, key: str) -> int:
    return int(after.get(key, 0)) - int(before.get(key, 0))


def run(args: argparse.Namespace) -> dict[str, object]:
    analysis = Path(args.analysis_root).expanduser().absolute()
    transpose_root = Path(args.transpose_root).expanduser().absolute()
    pilot_root = Path(args.pilot_root).expanduser().absolute()
    weights_root = Path(args.weights_root).expanduser().absolute()
    router_fit = MlpPilotRouterFit.from_bytes(_stable_read(analysis / "fit.json"))
    affine_fit = MlpPilotAffineFit.from_bytes(
        _stable_read(analysis / "affine-fit.json")
    )
    manifest = MlpPilotTransposeManifest.from_bytes(
        _stable_read(transpose_root / "pilot-transpose-manifest.json")
    )
    if (
        manifest.model_pin_sha256 != router_fit.model_pin_sha256
        or manifest.router_fit_sha256 != router_fit.sha256
        or manifest.affine_fit_sha256 != affine_fit.sha256
    ):
        raise CliError("transpose bank differs from the sparse runtime fits")
    entry = next((row for row in manifest.entries if row.layer == args.layer), None)
    if entry is None:
        raise CliError(f"transpose bank has no layer {args.layer}")
    shard_path = transpose_root / entry.shard
    if (
        hashlib.sha256(_stable_read(shard_path, 1024**3)).hexdigest()
        != entry.shard_sha256
    ):
        raise CliError("selected transpose shard hash changed")
    try:
        pilot_manifest = json.loads(
            _stable_read(pilot_root / "pilot-weight-manifest.json")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CliError("pilot weight manifest is not JSON") from exc
    if (
        not isinstance(pilot_manifest, dict)
        or set(pilot_manifest) != {"body", "body_sha256", "schema"}
        or pilot_manifest.get("schema") != "immer.qwen3.8-mlp-pilot-weight-bank/v1"
        or not isinstance(pilot_manifest.get("body"), dict)
        or pilot_manifest.get("body_sha256") != _digest(pilot_manifest["body"])
        or pilot_manifest["body"].get("router_fit_sha256") != router_fit.sha256
        or pilot_manifest["body"].get("affine_fit_sha256") != affine_fit.sha256
        or pilot_manifest["body"].get("transpose_manifest_sha256") != manifest.sha256
    ):
        raise CliError("pilot weight manifest differs from the sparse runtime fits")
    pilot_entry = next(
        (
            row
            for row in pilot_manifest["body"].get("entries", [])
            if isinstance(row, dict) and row.get("layer") == args.layer
        ),
        None,
    )
    if not isinstance(pilot_entry, dict) or not isinstance(
        pilot_entry.get("shard"), str
    ):
        raise CliError(f"pilot weight bank has no layer {args.layer}")
    pilot_shard = pilot_root / pilot_entry["shard"]
    if hashlib.sha256(
        _stable_read(pilot_shard, 128 * 1024**2)
    ).hexdigest() != pilot_entry.get("shard_sha256"):
        raise CliError("selected pilot shard hash changed")

    bank = QwenMlpEvidenceBank(Path(args.holdout_bank_root).expanduser().absolute())
    if bank.state().split_counts != (25, 10, 5) or not bank.audit().clean:
        raise CliError("holdout bank is not clean and complete")
    roles = PromptRowRoleManifest.from_bytes(
        _stable_read(analysis / "holdout-row-roles.json")
    )
    corpus = bank.build_subspace_corpus(
        row_indices_by_prompt=roles.row_indices_by_prompt
    )
    group = next((row for row in corpus.groups if row.layer == args.layer), None)
    if group is None or not 0 <= args.row < group.row_count:
        raise CliError("requested layer/content row is unavailable")
    receipt = next(
        receipt
        for receipt, _verification in bank.committed_pairs()
        if receipt.entry.layer == args.layer
        and receipt.entry.prompt_sha256 == group.prompt_sha256
    )
    references = {reference.stage: reference for reference in receipt.tensors}
    truth_rows = bank.restore_tensor(references["mlp.output"]).reshape(
        -1, corpus.hidden_dimension
    )
    truth = torch.from_numpy(
        truth_rows[roles.row_indices_by_prompt[group.prompt_sha256][args.row]].astype(
            np.float32
        )
    ).reshape(1, -1)
    hidden = torch.from_numpy(
        group.context_states[args.row].astype(np.float32)
    ).reshape(1, -1)

    with tempfile.TemporaryDirectory(prefix="immer-pilot-runtime-") as temporary:
        cache = Path(temporary)
        mount = CausalWeightMount(
            weights_root,
            LogicalModelIdentity(OFFICIAL_REPO_ID, OFFICIAL_REVISION),
            budget_mb=args.source_budget_mb,
        )
        transpose_source = Streamer.from_local(
            transpose_root,
            budget_mb=args.source_budget_mb,
            cache_dir=cache / "transpose",
            use_cache=False,
            max_cache_bytes=256 * 1024**2,
        )
        pilot_source = Streamer.from_local(
            pilot_root,
            budget_mb=args.source_budget_mb,
            cache_dir=cache / "pilot",
            use_cache=False,
            max_cache_bytes=256 * 1024**2,
        )
        weight_pager = Qwen38WeightPager(
            mount.source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=64 * 1024**2,
            close_source=False,
            require_source_identity=True,
            causal_tensor_reader=mount.tensor_reader,
        )
        transpose_pager = Qwen38WeightPager(
            transpose_source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=64 * 1024**2,
            close_source=True,
        )
        pilot_pager = Qwen38WeightPager(
            pilot_source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=64 * 1024**2,
            close_source=True,
        )
        full_pager = Qwen38WeightPager(
            mount.source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=384 * 1024**2,
            close_source=False,
            require_source_identity=True,
            causal_tensor_reader=mount.tensor_reader,
        )
        try:
            executor = MlpPilotSparseExecutor(
                router_fit,
                affine_fit,
                weight_pager,
                transpose_pager,
                pilot_pager=pilot_pager,
                output_dtype=torch.bfloat16,
            )
            model = StreamedQwen38(
                Qwen38Config.from_file(
                    weights_root / "config.json", require_official=True
                ),
                weight_pager,
                mlp_sparse_executor=executor,
                max_batch_size=1,
                max_seq_len=1,
            )
            before_weight = weight_pager.metrics()
            before_transpose = transpose_pager.metrics()
            before_pilot = pilot_pager.metrics()
            before_causal = mount.tensor_reader.metrics()
            started = time.perf_counter()
            predicted = model._mlp(hidden.to(torch.bfloat16), layer=args.layer)
            trace = model.mlp_sparse_last_trace
            if trace is None:
                raise CliError("StreamedQwen38 did not mount the sparse MLP path")
            elapsed = time.perf_counter() - started
            after_weight = weight_pager.metrics()
            after_transpose = transpose_pager.metrics()
            after_pilot = pilot_pager.metrics()
            after_causal = mount.tensor_reader.metrics()
            base = f"model.language_model.layers.{args.layer}.mlp"
            before_full = full_pager.metrics()
            before_full_causal = mount.tensor_reader.metrics()
            full_started = time.perf_counter()
            full_hidden = hidden.to(torch.bfloat16)
            full_gate = full_pager.linear(full_hidden, f"{base}.gate_proj")
            full_up = full_pager.linear(full_hidden, f"{base}.up_proj")
            full_output = full_pager.linear(
                swiglu(full_gate, full_up), f"{base}.down_proj"
            )
            full_elapsed = time.perf_counter() - full_started
            after_full = full_pager.metrics()
            after_full_causal = mount.tensor_reader.metrics()
            if not torch.equal(full_output.float(), truth):
                raise CliError("full causal MLP control differs from captured truth")
        finally:
            weight_pager.close()
            transpose_pager.close()
            pilot_pager.close()
            full_pager.close()
            mount.close()
    difference = predicted.float() - truth
    true_square = float(torch.square(truth).sum())
    predicted_square = float(torch.square(predicted.float()).sum())
    dot = float((predicted.float() * truth).sum())
    original_bytes = _delta(after_weight, before_weight, "logical_weight_bytes")
    transpose_bytes = _delta(after_transpose, before_transpose, "logical_weight_bytes")
    pilot_bytes = _delta(after_pilot, before_pilot, "logical_weight_bytes")
    sparse_bytes = original_bytes + transpose_bytes + pilot_bytes
    full_bytes = 3 * corpus.intermediate_dimension * corpus.hidden_dimension * 2
    measured_full_bytes = _delta(after_full, before_full, "logical_weight_bytes")
    if measured_full_bytes != full_bytes:
        raise CliError("full pager bytes differ from the MLP tensor ABI")
    if sparse_bytes / full_bytes != trace.weight_row_fraction:
        raise CliError("pager byte fraction differs from the sparse trace")
    body = {
        "affine_fit_sha256": affine_fit.sha256,
        "cosine": dot / math.sqrt(predicted_square * true_square),
        "causal_range_reads": _delta(after_causal, before_causal, "read_calls"),
        "elapsed_seconds": elapsed,
        "full_weight_bytes": full_bytes,
        "full_causal_range_reads": _delta(
            after_full_causal, before_full_causal, "read_calls"
        ),
        "full_elapsed_seconds": full_elapsed,
        "layer": args.layer,
        "max_abs_error": float(difference.abs().max()),
        "mounted_streamed_qwen": True,
        "model_pin_sha256": router_fit.model_pin_sha256,
        "output_sha256": hashlib.sha256(
            predicted.detach().float().numpy().astype("<f4").tobytes()
        ).hexdigest(),
        "pilot_manifest_body_sha256": pilot_manifest["body_sha256"],
        "pilot_pager_bytes": pilot_bytes,
        "pilot_pager_row_reads": _delta(after_pilot, before_pilot, "row_reads"),
        "relative_l2_error": math.sqrt(
            float(torch.square(difference).sum()) / true_square
        ),
        "router_fit_sha256": router_fit.sha256,
        "row": args.row,
        "sparse_weight_bytes": sparse_bytes,
        "speedup_full_over_sparse": full_elapsed / elapsed,
        "trace": trace.to_record(),
        "transpose_manifest_sha256": manifest.sha256,
        "transpose_pager_bytes": transpose_bytes,
        "transpose_pager_row_reads": _delta(
            after_transpose, before_transpose, "row_reads"
        ),
        "weight_byte_fraction": sparse_bytes / full_bytes,
        "weight_pager_bytes": original_bytes,
        "weight_pager_row_reads": _delta(after_weight, before_weight, "row_reads"),
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--transpose-root", required=True)
    parser.add_argument("--pilot-root", required=True)
    parser.add_argument("--weights-root", required=True)
    parser.add_argument("--holdout-bank-root", required=True)
    parser.add_argument("--output")
    parser.add_argument("--layer", type=int, default=63)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--source-budget-mb", type=float, default=4096.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (
        args.layer < 0
        or args.row < 0
        or not math.isfinite(args.source_budget_mb)
        or args.source_budget_mb <= 0.0
    ):
        raise CliError("layer, row, and source budget are invalid")
    report = run(args)
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
