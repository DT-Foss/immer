#!/usr/bin/env python3
"""Refit a sealed layer-63 MLP O1 state under a new ridge, without Qwen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

from immer.runtimes.qwen3_8.layer_mlp_crystal import (
    Layer63MlpResidualCrystalBank,
    LayerMlpCrystalIntegrityError,
)
from immer.runtimes.qwen3_8.layer_mlp_o1 import Layer63MlpO1Accumulator


RECEIPT_SCHEMA = "immer.qwen3.8-layer63-mlp-o1-ridge-refit-receipt/v1"


def _finite_positive(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be numeric") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return result


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-state", required=True)
    parser.add_argument("--output-state", required=True)
    parser.add_argument("--output-bank", required=True)
    ridge = parser.add_mutually_exclusive_group(required=True)
    ridge.add_argument("--relative-ridge-alpha", type=_finite_positive)
    ridge.add_argument("--ridge", type=_finite_positive)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    source_path = Path(args.source_state).expanduser().absolute()
    output_state_path = Path(args.output_state).expanduser().absolute()
    output_bank_path = Path(args.output_bank).expanduser().absolute()
    source = Layer63MlpO1Accumulator.load(source_path)
    source_before = source.snapshot()
    gram_scale = source.centered_gram_scale()
    if args.relative_ridge_alpha is not None:
        relative_alpha = float(args.relative_ridge_alpha)
        ridge = relative_alpha * gram_scale
        ridge_mode = "relative-centered-gram-trace"
    else:
        relative_alpha = None
        ridge = float(args.ridge)
        ridge_mode = "absolute"
    if not math.isfinite(ridge) or ridge <= 0.0:
        raise LayerMlpCrystalIntegrityError("computed ridge is not finite and positive")
    crystal = source.fork_with_ridge(output_state_path, ridge)
    destination = Layer63MlpO1Accumulator.load(output_state_path)
    destination_snapshot = destination.snapshot()
    source_after = source.snapshot()
    if source_after.state_sha256 != source_before.state_sha256:
        raise LayerMlpCrystalIntegrityError(
            "source O1 state changed during ridge refit"
        )
    if (
        destination_snapshot.source_o1_state_sha256 != source_before.state_sha256
        or destination_snapshot.source_o1_generation != source_before.generation
        or crystal.source_o1_state_sha256 != destination_snapshot.state_sha256
        or crystal.source_o1_generation != destination_snapshot.generation
    ):
        raise LayerMlpCrystalIntegrityError("ridge refit provenance chain is invalid")
    bank = Layer63MlpResidualCrystalBank(output_bank_path, source.identity)
    published_sha256 = bank.publish_latest(crystal)
    if published_sha256 != crystal.crystal_sha256:
        raise LayerMlpCrystalIntegrityError(
            "output bank already contains a newer O1 generation"
        )
    coverage = crystal.coverage
    operator_spectral_norm = float(
        torch.linalg.matrix_norm(crystal.operator, ord=2).item()
    )
    body = {
        "accumulated_rows": destination_snapshot.accumulated_rows,
        "bank_identity_sha256": source.identity.identity_sha256,
        "bank_path": str(output_bank_path),
        "crystal_sha256": crystal.crystal_sha256,
        "destination_generation": destination_snapshot.generation,
        "destination_state_path": str(output_state_path),
        "destination_state_sha256": destination_snapshot.state_sha256,
        "error_radius": coverage.error_radius,
        "feature_radius": coverage.feature_radius,
        "feature_rank": destination_snapshot.feature_rank,
        "gram_scale_trace_per_sketch_dim": gram_scale,
        "max_observed_error": coverage.max_observed_error,
        "observation_batches": destination_snapshot.observation_batches,
        "operator_spectral_norm": operator_spectral_norm,
        "provenance": {
            "destination_source_generation": (
                destination_snapshot.source_o1_generation
            ),
            "destination_source_state_sha256": (
                destination_snapshot.source_o1_state_sha256
            ),
            "published_source_generation": crystal.source_o1_generation,
            "published_source_state_sha256": crystal.source_o1_state_sha256,
        },
        "relative_ridge_alpha": relative_alpha,
        "ridge": ridge,
        "ridge_mode": ridge_mode,
        "sketch_dim": destination_snapshot.sketch_dim,
        "source_generation": source_before.generation,
        "source_state_path": str(source_path),
        "source_state_sha256": source_before.state_sha256,
        "source_unchanged": True,
    }
    return {
        "body": body,
        "schema": RECEIPT_SCHEMA,
        "sha256": hashlib.sha256(_canonical(body)).hexdigest(),
    }


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
