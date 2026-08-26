#!/usr/bin/env python3
"""Run the fixed harvested-operator contextual-routing benchmark."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from immer.runtimes.ooe.harvest_algebra_intelligence import (
    run_harvest_algebra_intelligence_benchmark,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20_260_826)
    parser.add_argument("--episodes", type=int, default=180)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_harvest_algebra_intelligence_benchmark(
        seed=args.seed,
        episodes=args.episodes,
    )
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
