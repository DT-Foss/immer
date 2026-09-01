#!/usr/bin/env python3
"""Publish ready multi-layer Qwen3.8 MLP O1 states without loading Qwen."""

from __future__ import annotations

import argparse
import json
import math
import sys

from immer.runtimes.qwen3_8.layer_mlp_registry import (
    DEFAULT_LAYER_MLP_O1_REGISTRY_BASENAME,
    publish_layer_mlp_o1_registry,
)


def _sorted_layers(value: str) -> tuple[int, ...]:
    try:
        layers = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "layers must be comma-separated integers"
        ) from exc
    if (
        not layers
        or layers != tuple(sorted(set(layers)))
        or any(not 0 <= layer <= 63 for layer in layers)
    ):
        raise argparse.ArgumentTypeError(
            "layers must be sorted unique integers in [0, 63]"
        )
    return layers


def _finite_non_negative(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be numeric") from exc
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return result


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layer63-state",
        "--source-state",
        "--o1-state",
        dest="layer63_state",
        required=True,
        help="sealed legacy layer-63 O1 state and multi-layer filename authority",
    )
    parser.add_argument(
        "--output-root",
        "--output-dir",
        dest="output_root",
        required=True,
        help="directory for immutable banks and the atomic registry manifest",
    )
    parser.add_argument(
        "--layers",
        type=_sorted_layers,
        default=tuple(range(64)),
        help="sorted comma-separated decoder layers (default: 0..63)",
    )
    parser.add_argument(
        "--max-error-radius",
        type=_finite_non_negative,
        default=0.0,
        help="mounted replacement error budget recorded per bank (default: 0)",
    )
    parser.add_argument(
        "--manifest-name",
        default=DEFAULT_LAYER_MLP_O1_REGISTRY_BASENAME,
        help=(
            "path-safe manifest basename inside the output root "
            f"(default: {DEFAULT_LAYER_MLP_O1_REGISTRY_BASENAME})"
        ),
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    summary = publish_layer_mlp_o1_registry(
        args.layer63_state,
        args.output_root,
        layers=args.layers,
        max_error_radius=args.max_error_radius,
        manifest_name=args.manifest_name,
    )
    return summary.to_record()


def main(argv: list[str] | None = None) -> int:
    try:
        result = run(build_parser().parse_args(argv))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(_canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
