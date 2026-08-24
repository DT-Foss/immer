#!/usr/bin/env python3
"""Compare authenticated Qwen3.8 DeltaNet probes with the 27B Cohen-d formula."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from immer.runtimes.qwen3_8 import (
    Qwen38DeltaNetProbeError,
    compare_probe_documents,
    verify_probe_document,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _load(path: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()

    def pairs(entries: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in entries:
            if key in result:
                raise Qwen38DeltaNetProbeError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    try:
        document = json.loads(
            source.read_text(encoding="utf-8"), object_pairs_hook=pairs
        )
    except Qwen38DeltaNetProbeError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Qwen38DeltaNetProbeError(f"cannot read probe: {source}") from exc
    return verify_probe_document(document)


def _atomic_write(path: str, document: dict[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".pending", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical(document) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def _nonnegative_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return value


def compare(args: argparse.Namespace) -> dict[str, Any]:
    return compare_probe_documents(
        [_load(path) for path in args.reasoning],
        [_load(path) for path in args.knowledge],
        protection_threshold=args.protection_threshold,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    comparison = subparsers.add_parser("compare")
    comparison.add_argument("--reasoning", action="append", required=True)
    comparison.add_argument("--knowledge", action="append", required=True)
    comparison.add_argument(
        "--protection-threshold", type=_nonnegative_float, default=1.0
    )
    comparison.add_argument("--output", required=True)
    comparison.set_defaults(handler=compare)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        document = args.handler(args)
        output = _atomic_write(args.output, document)
    except Qwen38DeltaNetProbeError as exc:
        raise SystemExit(f"Qwen DeltaNet probe failed: {exc}") from exc
    print(json.dumps({"output": str(output), "sha256": document["sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
