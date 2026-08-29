#!/usr/bin/env python3
"""Build the local mmap Q4/Q8 execution plane for causal Qwen3.8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from immer.runtimes.deepseek_v4.causal_weights import (
    CausalWeightMount,
    LogicalModelIdentity,
)
from immer.runtimes.qwen3_8.bundle import verify_qwen38_causal_mount
from immer.runtimes.qwen3_8.config import OFFICIAL_REPO_ID, OFFICIAL_REVISION
from immer.runtimes.qwen3_8.pager import Qwen38WeightPager
from immer.runtimes.qwen3_8.q4 import (
    Q4_BASE_POLICY,
    Q4_BALANCED_POLICY,
    Q4BankBuilder,
    Q4BankError,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the verified local Qwen3.8 text matrices into a causal-bound "
            "Q4_0 bank with Q8_0 embedding/head. The original BF16 bundle is unchanged."
        )
    )
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-budget-mb", type=float, default=65_536)
    parser.add_argument("--max-resident-mb", type=int, default=256)
    parser.add_argument("--row-chunk", type=int, default=128)
    parser.add_argument("--threads", type=int)
    parser.add_argument(
        "--policy",
        choices=("base", "balanced"),
        default="base",
        help="balanced keeps recurrent Attention and residual Down matrices at Q8",
    )
    parser.add_argument(
        "--reuse-bank",
        type=Path,
        help="hardlink tensors whose source-bound format already matches",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="print exact source/output bytes from the real inventory without conversion",
    )
    parser.add_argument(
        "--verbose-rows",
        action="store_true",
        help="also emit row-chunk progress instead of compact tensor milestones",
    )
    return parser


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


def run(args: argparse.Namespace) -> int:
    bundle = args.bundle.expanduser().resolve()
    output = args.output.expanduser().resolve()
    mount = CausalWeightMount(
        bundle,
        LogicalModelIdentity(OFFICIAL_REPO_ID, OFFICIAL_REVISION),
        budget_mb=args.source_budget_mb,
    )
    pager = None
    started = time.perf_counter()
    try:
        receipt = verify_qwen38_causal_mount(mount, require_official_config=True)
        pager = Qwen38WeightPager(
            mount.source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=args.max_resident_mb * 1024**2,
            close_source=False,
            require_source_identity=True,
            causal_tensor_reader=mount.tensor_reader,
        )
        builder = Q4BankBuilder(
            output,
            pager=pager,
            bundle_receipt=receipt,
            row_chunk=args.row_chunk,
            threads=args.threads,
            format_policy=(
                Q4_BASE_POLICY if args.policy == "base" else Q4_BALANCED_POLICY
            ),
            reuse_root=args.reuse_bank,
        )
        plan = builder.plan()
        if args.plan:
            _print(plan)
            return 0

        def progress(row):
            if args.verbose_rows or row.get("event") == "q4_tensor_complete" and (
                row.get("tensor_count", 0) <= 2
                or row.get("tensor_count", 0) % 16 == 0
            ):
                _print(row)

        document = builder.build(progress=progress)
        body = document["body"]
        _print(
            {
                "status": "complete",
                "schema": document["schema"],
                "manifest_sha256": document["sha256"],
                "tensor_count": body["tensor_count"],
                "source_bf16_bytes": body["source_bf16_bytes"],
                "payload_bytes": body["payload_bytes"],
                "payload_fraction": (
                    body["payload_bytes"] / body["source_bf16_bytes"]
                ),
                "seconds": time.perf_counter() - started,
                "output": str(output),
            }
        )
        return 0
    finally:
        if pager is not None:
            pager.close()
        mount.close()


def main() -> int:
    try:
        return run(_parser().parse_args())
    except (OSError, Q4BankError, TypeError, ValueError) as exc:
        _print(
            {
                "status": "error",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
