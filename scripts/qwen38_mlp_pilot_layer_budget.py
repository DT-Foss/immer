#!/usr/bin/env python3
"""Select and seal a brake-only sparse Qwen MLP action from decode receipts."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Sequence

from immer.runtimes.ooe.mlp_pilot_budget import (
    MlpPilotLayerBudgetConfig,
    fit_mlp_pilot_layer_budget,
    verify_mlp_pilot_layer_budget,
)


class CliError(RuntimeError):
    pass


def _read(path: Path, maximum: int = 8 * 1024 * 1024) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise CliError(f"report is not a regular file: {path}")
    data = path.read_bytes()
    if not data or len(data) > maximum:
        raise CliError(f"report exceeds its byte bound: {path}")
    return data


def _persist(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise CliError(f"refusing to replace a different policy: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def run(args: argparse.Namespace) -> dict[str, object]:
    reports = tuple(
        _read(Path(row).expanduser().absolute()) for row in args.report
    )
    config = MlpPilotLayerBudgetConfig(
        min_reports_per_candidate=args.min_reports_per_candidate,
        min_distinct_prefixes=args.min_distinct_prefixes,
        min_verified_steps=args.min_verified_steps,
        min_top10_overlap=args.min_top10_overlap,
        min_hidden_cosine=args.min_hidden_cosine,
        max_hidden_relative_l2=args.max_hidden_relative_l2,
        max_candidate_logit_error=args.max_candidate_logit_error,
        min_byte_saving_fraction=args.min_byte_saving_fraction,
        min_speedup=args.min_speedup,
        max_sparse_layers=args.max_sparse_layers,
    )
    policy = fit_mlp_pilot_layer_budget(reports, config=config)
    verify_mlp_pilot_layer_budget(policy, reports)
    if not policy.selected_layers and not args.allow_fallback:
        reasons = sorted(
            {
                reason
                for candidate in policy.candidates
                for reason in candidate.rejection_reasons
            }
        )
        raise CliError(
            "no admissible sparse action; rejection reasons: " + ", ".join(reasons)
        )
    output = Path(args.output).expanduser().absolute()
    _persist(output, policy.to_bytes())
    return {
        "candidates": [
            {
                "admissible": row.admissible,
                "byte_saving_fraction": row.byte_saving_fraction,
                "layers": list(row.layers),
                "rejection_reasons": list(row.rejection_reasons),
                "speedup": row.speedup,
                "verified_steps": row.verified_steps,
            }
            for row in policy.candidates
        ],
        "output": str(output),
        "policy_sha256": policy.sha256,
        "scope_prefix_sha256s": list(policy.scope_prefix_sha256s),
        "selected_layers": list(policy.selected_layers),
        "verified_steps": policy.verified_steps,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-reports-per-candidate", type=int, default=1)
    parser.add_argument("--min-distinct-prefixes", type=int, default=1)
    parser.add_argument("--min-verified-steps", type=int, default=4)
    parser.add_argument("--min-top10-overlap", type=int, default=8)
    parser.add_argument("--min-hidden-cosine", type=float, default=0.90)
    parser.add_argument("--max-hidden-relative-l2", type=float, default=0.45)
    parser.add_argument("--max-candidate-logit-error", type=float, default=64.0)
    parser.add_argument("--min-byte-saving-fraction", type=float, default=0.01)
    parser.add_argument("--min-speedup", type=float, default=0.0)
    parser.add_argument("--max-sparse-layers", type=int, default=64)
    parser.add_argument("--allow-fallback", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for field in (
        "min_hidden_cosine",
        "max_hidden_relative_l2",
        "max_candidate_logit_error",
        "min_byte_saving_fraction",
        "min_speedup",
    ):
        if not math.isfinite(float(getattr(args, field))):
            raise CliError(f"{field} must be finite")
    report = run(args)
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
