#!/usr/bin/env python3
"""Run the continual executable-language growth mechanism benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from immer.runtimes.ooe.language_intelligence import (
    run_language_growth_intelligence_benchmark,
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
    parser.add_argument("--base-episodes", type=_positive_int, default=5_000)
    parser.add_argument("--growth-episodes", type=_positive_int, default=5_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_language_growth_intelligence_benchmark(
        seeds=args.seeds,
        base_episodes=args.base_episodes,
        growth_episodes=args.growth_episodes,
    )
    rendered = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
