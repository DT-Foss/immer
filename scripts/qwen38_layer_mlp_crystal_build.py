#!/usr/bin/env python3
"""Charge one private layer-63 MLP residual bank from one normal Q4 request."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch

from immer.runtimes.deepseek_v4.causal_weights import LogicalModelIdentity
from immer.runtimes.qwen3_8.adapter import _open_local_runtime
from immer.runtimes.qwen3_8.config import OFFICIAL_REPO_ID, OFFICIAL_REVISION
from immer.runtimes.qwen3_8.encoding import (
    END_OF_TEXT_TOKEN_ID,
    IM_END_TOKEN_ID,
    Qwen38Tokenizer,
)
from immer.runtimes.qwen3_8.layer_mlp_builder import (
    LayerMlpBuildError,
    capture_exact_layer63_mlp,
    current_atlas_revision_sha256,
    current_compute_graph_revision_sha256,
    publish_layer_mlp_residual_crystal,
)
from immer.runtimes.qwen3_8.layer_mlp_crystal import (
    TARGET_LAYER_INDEX,
    Layer63MlpResidualCrystal,
    Layer63MlpResidualCrystalIdentity,
)
from immer.runtimes.qwen3_8.layer_transition_crystal import (
    LayerTransitionProjectionIdentity,
)
from immer.runtimes.qwen3_8.native_crsa import Qwen38NativeHeadCrsa


DEFAULT_ROOT = Path("/app/models/Qwen3.8-27B")
DEFAULT_PROJECTION_SEED = (
    "2b188a99b4b36f51bd910866e6d9a007fd02ec7256d6596fc87eb76f0444eccf"
)
RECEIPT_SCHEMA = "immer.qwen3.8-layer63-mlp-residual-build-receipt/v1"


def _finite_non_negative(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be numeric") from exc
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return result


def _finite_positive(value: str) -> float:
    result = _finite_non_negative(value)
    if result == 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _positive(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _non_negative(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
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
        raise LayerMlpBuildError("calibration text must not be empty")
    return value


def _mlp_packed_bytes(model: object) -> int:
    helper = getattr(model, "_layer_mlp_crystal_avoided_q4_bytes", None)
    if callable(helper):
        value = helper()
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise LayerMlpBuildError("runtime returned an invalid MLP byte count")
        return value
    pager = getattr(model, "pager", None)
    bank = getattr(pager, "q4_bank", None)
    entries = getattr(bank, "entries", None)
    has = getattr(bank, "has", None)
    if not isinstance(entries, dict) or not callable(has):
        raise LayerMlpBuildError("runtime lacks an authenticated packed Q4 inventory")
    base = f"model.language_model.layers.{TARGET_LAYER_INDEX}.mlp"
    names = tuple(f"{base}.{kind}_proj.weight" for kind in ("gate", "up", "down"))
    total = 0
    for name in names:
        if not bool(has(name)) or name not in entries:
            raise LayerMlpBuildError(f"packed Q4 inventory lacks {name}")
        entry = entries[name]
        payload_bytes = (
            entry.get("payload_bytes")
            if isinstance(entry, dict)
            else getattr(entry, "payload_bytes", None)
        )
        if (
            isinstance(payload_bytes, bool)
            or not isinstance(payload_bytes, int)
            or payload_bytes <= 0
        ):
            raise LayerMlpBuildError("packed Q4 MLP payload size is invalid")
        total += payload_bytes
    return total


def _rank_sweep(
    *,
    base: torch.Tensor,
    feature: torch.Tensor,
    target: torch.Tensor,
    identity_fields: dict[str, object],
    selected_sketch_dim: int,
    projection_seed: str,
    packed_weight_bytes_avoided: int,
    ridge: float,
) -> list[dict[str, object]]:
    """Fit several widths from one physical capture without retaining its rows."""

    rows = int(base.shape[0])
    records: list[dict[str, object]] = []
    for sketch_dim in sorted({16, 32, selected_sketch_dim}):
        if sketch_dim > rows - 1:
            continue
        projection = LayerTransitionProjectionIdentity(
            hidden_dim=int(base.shape[-1]),
            sketch_dim=sketch_dim,
            seed_sha256=projection_seed,
        )
        identity = Layer63MlpResidualCrystalIdentity(
            projection=projection,
            **identity_fields,
        )
        crystal = Layer63MlpResidualCrystal.fit(
            identity=identity,
            base_hidden=base,
            mlp_input=feature,
            target_hidden=target,
            packed_weight_bytes_avoided=packed_weight_bytes_avoided,
            ridge=ridge,
        )
        feature64 = feature.detach().to(device="cpu", dtype=torch.float64)
        base64 = base.detach().to(device="cpu", dtype=torch.float64)
        target64 = target.detach().to(device="cpu", dtype=torch.float64)
        projected = feature64 @ projection.matrix()
        predicted = (
            base64
            + crystal.residual_mean
            + (projected - crystal.feature_mean) @ crystal.operator
        )
        errors = torch.linalg.vector_norm(
            predicted.to(dtype=torch.bfloat16).to(dtype=torch.float64) - target64,
            dim=1,
        )
        records.append(
            {
                "error_l2_mean": float(errors.mean().item()),
                "error_l2_rms": float(torch.sqrt(torch.mean(errors.square())).item()),
                "feature_radius": crystal.coverage.feature_radius,
                "max_observed_error": crystal.coverage.max_observed_error,
                "operator_payload_bytes_f64": (
                    crystal.operator.numel() * crystal.operator.element_size()
                    + crystal.feature_mean.numel() * crystal.feature_mean.element_size()
                    + crystal.residual_mean.numel()
                    * crystal.residual_mean.element_size()
                ),
                "sketch_dim": sketch_dim,
            }
        )
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--text-file")
    parser.add_argument("--output-bank", required=True)
    parser.add_argument("--atlas-root", required=True)
    parser.add_argument("--compute-root", required=True)
    parser.add_argument("--direct-fit", action="store_true")
    parser.add_argument("--bundle", default=str(DEFAULT_ROOT))
    parser.add_argument("--tokenizer")
    parser.add_argument("--q4-root")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--max-new-tokens", type=_positive, default=8)
    parser.add_argument("--max-context-tokens", type=_positive, default=2048)
    parser.add_argument("--sketch-dim", type=_positive, default=64)
    parser.add_argument("--projection-seed", default=DEFAULT_PROJECTION_SEED)
    parser.add_argument("--ridge", type=_finite_positive, default=1e-8)
    parser.add_argument("--coverage-guard", type=_finite_non_negative, default=0.0)
    parser.add_argument("--error-guard", type=_finite_non_negative, default=0.0)
    parser.add_argument("--source-budget-mb", type=float, default=4_194_304.0)
    parser.add_argument("--max-resident-mb", type=_positive, default=192)
    parser.add_argument("--q4-resident-budget-mb", type=_non_negative, default=0)
    parser.add_argument("--q4-threads", type=_positive)
    parser.add_argument("--no-prefix-sinkhorn", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if not args.direct_fit:
        raise LayerMlpBuildError("pass --direct-fit to publish a direct ridge action")
    if args.sketch_dim > 256:
        raise LayerMlpBuildError("O1-backed sketch_dim cannot exceed 256")
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
    graph_sha256 = current_compute_graph_revision_sha256(compute_root)
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
        q4_resident_budget_bytes=int(args.q4_resident_budget_mb * 1024**2),
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
            raise LayerMlpBuildError("prompt plus output exceeds context")
        model = runtime.model
        model_sha256 = model.layer_mlp_crystal_model_sha256()
        q4_sha256 = model.layer_mlp_crystal_q4_sha256()
        avoided_bytes = _mlp_packed_bytes(model)
        projection = LayerTransitionProjectionIdentity(
            hidden_dim=model.config.dim,
            sketch_dim=args.sketch_dim,
            seed_sha256=args.projection_seed,
        )
        identity_fields = {
            "model_sha256": model_sha256,
            "q4_sha256": q4_sha256,
            "graph_revision_sha256": graph_sha256,
            "atlas_revision_sha256": atlas_sha256,
        }
        identity = Layer63MlpResidualCrystalIdentity(
            model_sha256=model_sha256,
            q4_sha256=q4_sha256,
            graph_revision_sha256=graph_sha256,
            atlas_revision_sha256=atlas_sha256,
            projection=projection,
        )
        if len(prompt_ids) + args.max_new_tokens < args.sketch_dim + 1:
            raise LayerMlpBuildError(
                "prompt tokens plus max_new_tokens cannot supply "
                "sketch_dim + 1 capture rows"
            )
        base, feature, target, generated_ids, evidence = capture_exact_layer63_mlp(
            model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=(IM_END_TOKEN_ID, END_OF_TEXT_TOKEN_ID),
        )
        rank_sweep = _rank_sweep(
            base=base,
            feature=feature,
            target=target,
            identity_fields=identity_fields,
            selected_sketch_dim=args.sketch_dim,
            projection_seed=args.projection_seed,
            packed_weight_bytes_avoided=avoided_bytes,
            ridge=args.ridge,
        )
        result = publish_layer_mlp_residual_crystal(
            bank_path=Path(args.output_bank).expanduser().absolute(),
            identity=identity,
            base_hidden=base,
            mlp_input=feature,
            target_hidden=target,
            packed_weight_bytes_avoided=avoided_bytes,
            direct_fit=True,
            ridge=args.ridge,
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
            "direct_fit": True,
            "error_radius": result.error_radius,
            "feature_radius": result.feature_radius,
            "feature_rank": result.feature_rank,
            "generated_token_ids": list(generated_ids),
            "generation_seconds": float(evidence.seconds),
            "graph_revision_sha256": result.graph_revision_sha256,
            "max_observed_error": result.max_observed_error,
            "model_sha256": model_sha256,
            "output_sha256": hashlib.sha256(decoded.encode("utf-8")).hexdigest(),
            "packed_weight_bytes_avoided_per_hit": (result.packed_weight_bytes_avoided),
            "prefix_sinkhorn": prefix_sinkhorn,
            "projection_sha256": projection.projection_sha256,
            "q4_resident_budget_bytes": int(args.q4_resident_budget_mb * 1024**2),
            "q4_sha256": q4_sha256,
            "rank_sweep": rank_sweep,
            "ridge": float(args.ridge),
            "sketch_dim": args.sketch_dim,
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
