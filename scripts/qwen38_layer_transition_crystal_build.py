#!/usr/bin/env python3
"""Charge one private layer-63 transition bank from one normal Q4 request."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from immer.runtimes.deepseek_v4.causal_weights import LogicalModelIdentity
from immer.runtimes.qwen3_8.adapter import _open_local_runtime
from immer.runtimes.qwen3_8.config import OFFICIAL_REPO_ID, OFFICIAL_REVISION
from immer.runtimes.qwen3_8.encoding import (
    END_OF_TEXT_TOKEN_ID,
    IM_END_TOKEN_ID,
    Qwen38Tokenizer,
)
from immer.runtimes.qwen3_8.layer_transition_builder import (
    LayerTransitionBuildError,
    capture_exact_layer63,
    current_atlas_revision_sha256,
    publish_layer_transition_crystal,
    restore_promoted_layer63_affine,
)
from immer.runtimes.qwen3_8.layer_transition_crystal import (
    LayerTransitionCrystalIdentity,
    LayerTransitionProjectionIdentity,
)
from immer.runtimes.qwen3_8.native_crsa import Qwen38NativeHeadCrsa


DEFAULT_ROOT = Path("/app/models/Qwen3.8-27B")
DEFAULT_PROJECTION_SEED = (
    "2b188a99b4b36f51bd910866e6d9a007fd02ec7256d6596fc87eb76f0444eccf"
)
RECEIPT_SCHEMA = "immer.qwen3.8-layer-transition-build-receipt/v1"


def _finite_non_negative(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be numeric") from exc
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return result


def _positive(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _text(args: argparse.Namespace) -> str:
    if args.text is not None:
        value = args.text
    else:
        path = Path(args.text_file).expanduser().absolute()
        value = path.read_text(encoding="utf-8")
    value = value.strip()
    if not value:
        raise LayerTransitionBuildError("calibration text must not be empty")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--text-file")
    parser.add_argument("--output-bank", required=True)
    parser.add_argument("--atlas-root", required=True)
    parser.add_argument("--compute-root", required=True)
    parser.add_argument("--compute-crystal-sha256")
    parser.add_argument("--direct-fit", action="store_true")
    parser.add_argument("--bundle", default=str(DEFAULT_ROOT))
    parser.add_argument("--tokenizer")
    parser.add_argument("--q4-root")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--max-new-tokens", type=_positive, default=8)
    parser.add_argument("--max-context-tokens", type=_positive, default=2048)
    parser.add_argument("--sketch-dim", type=_positive, default=16)
    parser.add_argument("--projection-seed", default=DEFAULT_PROJECTION_SEED)
    parser.add_argument("--coverage-guard", type=_finite_non_negative, default=0.0)
    parser.add_argument("--error-guard", type=_finite_non_negative, default=0.0)
    parser.add_argument("--source-budget-mb", type=float, default=4_194_304.0)
    parser.add_argument("--max-resident-mb", type=_positive, default=192)
    parser.add_argument("--q4-threads", type=_positive)
    parser.add_argument("--no-prefix-sinkhorn", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.sketch_dim > 256:
        raise LayerTransitionBuildError("O1-backed sketch_dim cannot exceed 256")
    bundle = Path(args.bundle).expanduser().absolute()
    tokenizer = (
        bundle / "tokenizer.json"
        if args.tokenizer is None
        else Path(args.tokenizer).expanduser().absolute()
    )
    q4_root = (
        bundle / "causal" / "q4-base-v3-mtp"
        if args.q4_root is None
        else Path(args.q4_root).expanduser().absolute()
    )
    atlas_root = Path(args.atlas_root).expanduser().absolute()
    compute_root = Path(args.compute_root).expanduser().absolute()
    graph_sha256, promoted = restore_promoted_layer63_affine(
        compute_root,
        sketch_dim=args.sketch_dim,
        compute_crystal_sha256=args.compute_crystal_sha256,
    )
    if promoted is None and not args.direct_fit:
        raise LayerTransitionBuildError(
            "no promoted layer-63 affine exists; use --direct-fit explicitly"
        )
    atlas_sha256 = current_atlas_revision_sha256(atlas_root)
    prefix_sinkhorn = not args.no_prefix_sinkhorn
    intervention = (
        Qwen38NativeHeadCrsa(alpha=1.0, replace_base_softmax=True)
        if prefix_sinkhorn
        else None
    )
    started = time.perf_counter()
    runtime = _open_local_runtime(
        bundle_path=bundle,
        tokenizer_path=tokenizer,
        identity=LogicalModelIdentity(
            repo_id=OFFICIAL_REPO_ID,
            revision=OFFICIAL_REVISION,
        ),
        require_official_config=True,
        device="cpu",
        compute_dtype="bfloat16",
        source_budget_mb=float(args.source_budget_mb),
        max_resident_bytes=int(args.max_resident_mb * 1024**2),
        max_context_tokens=args.max_context_tokens,
        q4_root=q4_root,
        q4_threads=args.q4_threads,
        native_head_crsa=intervention,
    )
    try:
        text = _text(args)
        rendered = Qwen38Tokenizer.render_no_thinking_prompt(
            args.system_prompt,
            text,
        )
        encoded = runtime.tokenizer.encode(rendered)
        prompt_ids = tuple(int(value) for value in getattr(encoded, "ids", encoded))
        if len(prompt_ids) + args.max_new_tokens > args.max_context_tokens:
            raise LayerTransitionBuildError("prompt plus output exceeds context")
        model = runtime.model
        model_sha256 = model.layer_transition_crystal_model_sha256()
        q4_sha256 = model.layer_transition_crystal_q4_sha256()
        avoided_bytes = model._layer_transition_avoided_q4_bytes()
        projection = LayerTransitionProjectionIdentity(
            hidden_dim=model.config.dim,
            sketch_dim=args.sketch_dim,
            seed_sha256=args.projection_seed,
        )
        identity = LayerTransitionCrystalIdentity(
            model_sha256=model_sha256,
            q4_sha256=q4_sha256,
            graph_revision_sha256=graph_sha256,
            atlas_revision_sha256=atlas_sha256,
            projection=projection,
        )
        source_hidden, target_hidden, generated_ids, evidence = capture_exact_layer63(
            model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=(IM_END_TOKEN_ID, END_OF_TEXT_TOKEN_ID),
        )
        result = publish_layer_transition_crystal(
            bank_path=Path(args.output_bank).expanduser().absolute(),
            identity=identity,
            source_hidden=source_hidden,
            target_hidden=target_hidden,
            packed_weight_bytes_avoided=avoided_bytes,
            promoted=promoted,
            direct_fit=args.direct_fit,
            coverage_guard=args.coverage_guard,
            error_guard=args.error_guard,
        )
        decoded = runtime.tokenizer.decode(generated_ids)
        body = {
            "atlas_revision_sha256": result.atlas_revision_sha256,
            "bank_identity_sha256": result.bank_identity_sha256,
            "bank_path": str(result.bank_path),
            "calibration_rows": result.calibration_rows,
            "crystal_sha256": result.crystal_sha256,
            "direct_fit": promoted is None,
            "error_radius": result.error_radius,
            "generated_token_ids": list(generated_ids),
            "generation_seconds": float(evidence.seconds),
            "graph_revision_sha256": result.graph_revision_sha256,
            "max_observed_error": result.max_observed_error,
            "output_sha256": hashlib.sha256(decoded.encode("utf-8")).hexdigest(),
            "packed_weight_bytes_avoided_per_hit": (result.packed_weight_bytes_avoided),
            "prefix_sinkhorn": prefix_sinkhorn,
            "projection_sha256": projection.projection_sha256,
            "q4_sha256": q4_sha256,
            "model_sha256": model_sha256,
            "sketch_radius": result.sketch_radius,
            "source_compute_crystal_sha256": (result.source_compute_crystal_sha256),
            "source_edge_sha256": result.source_edge_sha256,
            "wall_seconds": time.perf_counter() - started,
        }
        return {
            "body": body,
            "schema": RECEIPT_SCHEMA,
            "sha256": hashlib.sha256(_canonical(body)).hexdigest(),
        }
    finally:
        runtime.close()


def main(argv: list[str] | None = None) -> int:
    try:
        receipt = run(build_parser().parse_args(argv))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(_canonical(receipt).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
