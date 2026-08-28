#!/usr/bin/env python3
"""Compare one full versus mounted sparse Qwen decode from the same exact prefix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile
from typing import Sequence

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
from immer.runtimes.qwen3_8.config import (
    OFFICIAL_REPO_ID,
    OFFICIAL_REVISION,
    Qwen38Config,
)
from immer.runtimes.qwen3_8.model import StreamedQwen38
from immer.runtimes.qwen3_8.pager import Qwen38WeightPager
import torch

from qwen38_mlp_pilot_router import _digest, _persist_exact, _stable_read


REPORT_NAME = "sparse-decode-compare-v1.json"
REPORT_SCHEMA = "immer.qwen3.8-mlp-pilot-sparse-decode-compare/v1"


class CliError(RuntimeError):
    pass


def _delta(after: dict, before: dict, key: str) -> int:
    return int(after.get(key, 0)) - int(before.get(key, 0))


def _prompt(
    path: Path, requested_sha256: str | None = None
) -> tuple[str, tuple[int, ...]]:
    try:
        document = json.loads(_stable_read(path))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CliError("prompt registry is not JSON") from exc
    if not isinstance(document, dict):
        raise CliError("prompt registry root is invalid")
    body = document.get("body", document)
    if not isinstance(body, dict) or not isinstance(body.get("prompts"), list):
        raise CliError("prompt registry has no prompts")
    rows = []
    for row in body["prompts"]:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("sha256"), str)
            or not isinstance(row.get("token_ids"), list)
            or not row["token_ids"]
            or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in row["token_ids"]
            )
        ):
            raise CliError("prompt registry row is invalid")
        rows.append((row["sha256"], tuple(row["token_ids"])))
    if requested_sha256 is None:
        return min(rows, key=lambda row: (len(row[1]), row[0]))
    selected = next((row for row in rows if row[0] == requested_sha256), None)
    if selected is None:
        raise CliError("requested prompt SHA is absent from the registry")
    return selected


def _hidden_metrics(candidate: torch.Tensor, truth: torch.Tensor) -> dict[str, float]:
    x = candidate.detach().to(dtype=torch.float64, device="cpu")
    y = truth.detach().to(dtype=torch.float64, device="cpu")
    difference = x - y
    x2 = float(torch.square(x).sum())
    y2 = float(torch.square(y).sum())
    return {
        "cosine": float((x * y).sum()) / math.sqrt(x2 * y2),
        "max_abs_error": float(difference.abs().max()),
        "relative_l2_error": math.sqrt(float(torch.square(difference).sum()) / y2),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    analysis = Path(args.analysis_root).expanduser().absolute()
    transpose_root = Path(args.transpose_root).expanduser().absolute()
    pilot_root = Path(args.pilot_root).expanduser().absolute()
    weights_root = Path(args.weights_root).expanduser().absolute()
    router_fit = MlpPilotRouterFit.from_bytes(_stable_read(analysis / "fit.json"))
    affine_fit = MlpPilotAffineFit.from_bytes(
        _stable_read(analysis / "affine-fit.json")
    )
    transpose_manifest = MlpPilotTransposeManifest.from_bytes(
        _stable_read(transpose_root / "pilot-transpose-manifest.json")
    )
    try:
        pilot_manifest = json.loads(
            _stable_read(pilot_root / "pilot-weight-manifest.json")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CliError("pilot manifest is not JSON") from exc
    if (
        affine_fit.router_fit_sha256 != router_fit.sha256
        or transpose_manifest.router_fit_sha256 != router_fit.sha256
        or not isinstance(pilot_manifest, dict)
        or not isinstance(pilot_manifest.get("body"), dict)
        or pilot_manifest.get("body_sha256") != _digest(pilot_manifest["body"])
        or pilot_manifest["body"].get("router_fit_sha256") != router_fit.sha256
        or pilot_manifest["body"].get("affine_fit_sha256") != affine_fit.sha256
    ):
        raise CliError("sparse runtime artifacts disagree")
    prompt_sha256, prompt_tokens = _prompt(
        Path(args.prompt_registry).expanduser().absolute(), args.prompt_sha256
    )
    if len(prompt_tokens) + 1 > args.max_context_tokens:
        raise CliError("selected prompt exceeds --max-context-tokens")

    with tempfile.TemporaryDirectory(prefix="immer-sparse-decode-") as temporary:
        scratch = Path(temporary)
        mount = CausalWeightMount(
            weights_root,
            LogicalModelIdentity(OFFICIAL_REPO_ID, OFFICIAL_REVISION),
            budget_mb=args.source_budget_mb,
        )
        transpose_source = Streamer.from_local(
            transpose_root,
            budget_mb=args.source_budget_mb,
            cache_dir=scratch / "transpose",
            use_cache=False,
        )
        pilot_source = Streamer.from_local(
            pilot_root,
            budget_mb=args.source_budget_mb,
            cache_dir=scratch / "pilot",
            use_cache=False,
        )
        pager = Qwen38WeightPager(
            mount.source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=384 * 1024**2,
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
        try:
            executor = MlpPilotSparseExecutor(
                router_fit,
                affine_fit,
                pager,
                transpose_pager,
                pilot_pager=pilot_pager,
                output_dtype=torch.bfloat16,
            )
            model = StreamedQwen38(
                Qwen38Config.from_file(
                    weights_root / "config.json", require_official=True
                ),
                pager,
                mlp_sparse_executor=executor,
                max_batch_size=1,
                max_seq_len=args.max_context_tokens,
            )
            prefix_hidden, prefix_evidence = model.hidden_stateful([prompt_tokens])
            prefix_values, prefix_ids = pager.topk_logits(
                prefix_hidden[:, -1, :], k=1, block_rows=args.head_block_rows
            )
            decode_token = int(prefix_ids[0, 0])
            snapshot = scratch / "prefix.snapshot"
            model.save_state(snapshot)

            model.mlp_sparse_executor = None
            before_full_pager = pager.metrics()
            before_full_transpose = transpose_pager.metrics()
            before_full_pilot = pilot_pager.metrics()
            full_hidden, full_evidence = model.decode([[decode_token]])
            after_full_decode_pager = pager.metrics()
            full_values, full_ids = pager.topk_logits(
                full_hidden[:, -1, :], k=10, block_rows=args.head_block_rows
            )
            after_full_transpose = transpose_pager.metrics()
            after_full_pilot = pilot_pager.metrics()

            model.mlp_sparse_executor = executor
            model.load_state(snapshot)
            before_sparse_pager = pager.metrics()
            before_sparse_transpose = transpose_pager.metrics()
            before_sparse_pilot = pilot_pager.metrics()
            sparse_hidden, sparse_evidence = model.decode([[decode_token]])
            after_sparse_decode_pager = pager.metrics()
            sparse_values, sparse_ids = pager.topk_logits(
                sparse_hidden[:, -1, :], k=10, block_rows=args.head_block_rows
            )
            after_sparse_transpose = transpose_pager.metrics()
            after_sparse_pilot = pilot_pager.metrics()

            candidates = tuple(
                sorted(
                    set(int(row) for row in full_ids.reshape(-1).tolist())
                    | set(int(row) for row in sparse_ids.reshape(-1).tolist())
                )
            )
            full_candidate_logits = pager.candidate_logits(
                full_hidden[:, -1, :], candidates
            ).float()
            sparse_candidate_logits = pager.candidate_logits(
                sparse_hidden[:, -1, :], candidates
            ).float()
        finally:
            pager.close()
            transpose_pager.close()
            pilot_pager.close()
            mount.close()
    full_top = tuple(int(row) for row in full_ids[0].tolist())
    sparse_top = tuple(int(row) for row in sparse_ids[0].tolist())
    body = {
        "affine_fit_sha256": affine_fit.sha256,
        "candidate_logit_max_abs_error": float(
            (sparse_candidate_logits - full_candidate_logits).abs().max()
        ),
        "candidate_token_ids": list(candidates),
        "decode_token_id": decode_token,
        "full_decode_seconds": full_evidence.seconds,
        "full_primary_weight_bytes": _delta(
            after_full_decode_pager, before_full_pager, "logical_weight_bytes"
        ),
        "full_top10": list(full_top),
        "full_top10_values": [float(row) for row in full_values[0].tolist()],
        "hidden_metrics": _hidden_metrics(sparse_hidden, full_hidden),
        "model_pin_sha256": router_fit.model_pin_sha256,
        "prefix_seconds": prefix_evidence.seconds,
        "prefix_sha256": prompt_sha256,
        "prefix_top1_value": float(prefix_values[0, 0]),
        "prefix_tokens": len(prompt_tokens),
        "router_fit_sha256": router_fit.sha256,
        "sparse_decode_seconds": sparse_evidence.seconds,
        "sparse_pilot_weight_bytes": _delta(
            after_sparse_pilot, before_sparse_pilot, "logical_weight_bytes"
        ),
        "sparse_primary_weight_bytes": _delta(
            after_sparse_decode_pager,
            before_sparse_pager,
            "logical_weight_bytes",
        ),
        "sparse_top10": list(sparse_top),
        "sparse_top10_overlap": len(set(full_top) & set(sparse_top)),
        "sparse_top10_values": [float(row) for row in sparse_values[0].tolist()],
        "sparse_transpose_weight_bytes": _delta(
            after_sparse_transpose,
            before_sparse_transpose,
            "logical_weight_bytes",
        ),
        "speedup_full_over_sparse": full_evidence.seconds / sparse_evidence.seconds,
        "top1_equal": full_top[0] == sparse_top[0],
        "unused_full_auxiliary_bytes": _delta(
            after_full_transpose, before_full_transpose, "logical_weight_bytes"
        )
        + _delta(after_full_pilot, before_full_pilot, "logical_weight_bytes"),
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
    parser.add_argument("--prompt-registry", required=True)
    parser.add_argument("--prompt-sha256")
    parser.add_argument("--output")
    parser.add_argument("--source-budget-mb", type=float, default=524_288.0)
    parser.add_argument("--max-context-tokens", type=int, default=256)
    parser.add_argument("--head-block-rows", type=int, default=2048)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (
        not math.isfinite(args.source_budget_mb)
        or args.source_budget_mb <= 0.0
        or args.max_context_tokens < 2
        or args.head_block_rows < 1
    ):
        raise CliError("decode budgets are invalid")
    report = run(args)
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
