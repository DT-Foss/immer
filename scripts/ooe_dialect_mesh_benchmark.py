#!/usr/bin/env python3
"""Run the cross-dialect executable-language mechanism benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from immer.runtimes.ooe.dialect_intelligence import (
    run_dialect_mesh_intelligence_benchmark,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", type=_positive_int, default=5)
    parser.add_argument("--dialects", type=_positive_int, default=5)
    parser.add_argument("--contexts-per-dialect", type=_positive_int, default=3)
    parser.add_argument("--train-episodes", type=_positive_int, default=4_000)
    parser.add_argument("--programs-per-pair", type=_positive_int, default=200)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_dialect_mesh_intelligence_benchmark(
        seeds=args.seeds,
        dialects=args.dialects,
        contexts_per_dialect=args.contexts_per_dialect,
        train_episodes=args.train_episodes,
        programs_per_pair=args.programs_per_pair,
    )
    rendered = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
