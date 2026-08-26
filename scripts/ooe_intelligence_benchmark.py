#!/usr/bin/env python3
"""Run the deterministic Markov-OoE intelligence mechanism matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.intelligence import run_intelligence_benchmark


def _positive(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if result < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _write_atomic(path: Path, report: dict[str, Any]) -> None:
    destination = path.expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(report) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".pending",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20_260_826)
    parser.add_argument("--tasks", type=_positive, default=250)
    parser.add_argument("--state-size", type=_positive, default=17)
    parser.add_argument("--replicas", type=_positive, default=12)
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_intelligence_benchmark(
        seed=args.seed,
        task_count=args.tasks,
        state_size=args.state_size,
        replicas=args.replicas,
    )
    if args.output:
        _write_atomic(Path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
