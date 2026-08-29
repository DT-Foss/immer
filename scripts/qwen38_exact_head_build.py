#!/usr/bin/env python3
"""Build the exact residual-PQ LM-head rail from local weights only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from immer.runtimes.deepseek_v4.causal_weights import (
    CausalWeightMount,
    LogicalModelIdentity,
)
from immer.runtimes.qwen3_8.bundle import verify_qwen38_causal_mount
from immer.runtimes.qwen3_8.config import (
    OFFICIAL_REPO_ID,
    OFFICIAL_REVISION,
    Qwen38Config,
)
from immer.runtimes.qwen3_8.exact_head import ExactHeadConfig, ExactHeadIndex
from immer.runtimes.qwen3_8.pager import Qwen38WeightPager


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an exact weight-only Qwen3.8 LM-head search index"
    )
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--page-rows", type=int, default=2048)
    parser.add_argument("--subspace-width", type=int, default=32)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--fanout", type=int, default=16)
    parser.add_argument("--kmeans-iterations", type=int, default=4)
    parser.add_argument("--assignment-chunk-rows", type=int, default=4096)
    parser.add_argument("--sample-rows", type=int, default=4096)
    parser.add_argument("--source-budget-mb", type=float, default=8192.0)
    parser.add_argument("--max-resident-mb", type=int, default=192)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = ExactHeadConfig(
        subspace_width=args.subspace_width,
        codebook_size=args.codebook_size,
        page_rows=args.page_rows,
        fanout=args.fanout,
        kmeans_iterations=args.kmeans_iterations,
        assignment_chunk_rows=args.assignment_chunk_rows,
    )
    identity = LogicalModelIdentity(OFFICIAL_REPO_ID, OFFICIAL_REVISION)
    with CausalWeightMount(
        args.bundle,
        identity,
        budget_mb=args.source_budget_mb,
    ) as mount:
        bundle = verify_qwen38_causal_mount(
            mount,
            require_official_config=True,
        )
        model_config = Qwen38Config.from_file(
            mount.weights_root / "config.json",
            require_official=True,
        )
        head_name = (
            "model.language_model.embed_tokens.weight"
            if model_config.tie_word_embeddings
            else "lm_head.weight"
        )
        pager = Qwen38WeightPager(
            mount.source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=args.max_resident_mb * 1024**2,
            close_source=False,
            require_source_identity=True,
            causal_tensor_reader=mount.tensor_reader,
        )
        try:
            index = ExactHeadIndex.build_from_pager(
                pager,
                name=head_name,
                config=config,
                sample_rows=args.sample_rows,
            )
            receipt = index.save(args.output)
            report = {
                "bundle": bundle,
                "config": config.to_record(),
                "head_name": head_name,
                "pager": pager.metrics(),
                "receipt": receipt.to_record(),
                "schema": "immer.qwen3.8-exact-head-build-report/v1",
            }
            print(json.dumps(report, ensure_ascii=True, sort_keys=True))
        finally:
            pager.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
