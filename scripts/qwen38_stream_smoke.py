#!/usr/bin/env python3
"""Execute the first real Qwen3.8 layers from pinned remote safetensors.

This is a correctness smoke, not a retrieval shortcut and not a benchmark. It
reads the official text config, embeds explicit token IDs, and executes an
exact prefix of the 64-layer decoder with IMMER's own DeltaNet/attention math.
Only one BF16 matrix is materialized at a time.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any
import urllib.error
import urllib.request

import torch

from immer.knowledge.streamer import Streamer
from immer.runtimes.qwen3_8 import (
    OFFICIAL_REPO_ID,
    OFFICIAL_REVISION,
    Qwen38Config,
    Qwen38WeightPager,
    StreamedQwen38,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = ROOT / "artifacts" / "private" / "qwen3.8-cache"
DEFAULT_OUTPUT = ROOT / "results" / "qwen38_stream_smoke.json"
RESULT_SCHEMA = "immer.qwen3.8-stream-smoke/v1"
MIB = 1024**2


class CliError(RuntimeError):
    """The bounded Qwen smoke cannot satisfy its execution contract."""


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _positive_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def _token_ids(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in raw.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("token IDs must be comma-separated integers") from exc
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("token IDs must be non-negative")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--token-ids",
        type=_token_ids,
        default=(248044,),
        help="comma-separated token IDs (default: Qwen BOS)",
    )
    parser.add_argument(
        "--layers",
        type=_positive_int,
        default=1,
        help="execute this many leading decoder layers",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument(
        "--compute-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--budget-mb", type=_positive_float, default=1024.0)
    parser.add_argument("--cache-mb", type=_positive_float, default=1024.0)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--config",
        help="explicit config.json (required with --source-dir)",
    )
    parser.add_argument(
        "--source-dir",
        help="local safetensors fixture/checkpoint instead of the pinned remote source",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate config and the complete text tensor inventory without payload reads",
    )
    return parser


def _official_config() -> Qwen38Config:
    url = (
        f"https://huggingface.co/{OFFICIAL_REPO_ID}/resolve/"
        f"{OFFICIAL_REVISION}/config.json"
    )
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            document = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise CliError("cannot fetch the pinned Qwen3.8 config") from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CliError("pinned Qwen3.8 config is invalid JSON") from exc
    if not isinstance(document, Mapping):
        raise CliError("pinned Qwen3.8 config root is not an object")
    return Qwen38Config.from_mapping(document)


def _config(args: argparse.Namespace) -> Qwen38Config:
    if args.config:
        return Qwen38Config.from_file(
            args.config,
            require_official=not bool(args.source_dir),
        )
    if args.source_dir:
        raise CliError("--config is required with --source-dir")
    return _official_config()


def _source(args: argparse.Namespace) -> Streamer:
    cache_bytes = int(args.cache_mb * MIB)
    if args.source_dir:
        return Streamer.from_local(
            args.source_dir,
            budget_mb=args.budget_mb,
            cache_dir=args.cache_dir,
            max_cache_bytes=cache_bytes,
        )
    return Streamer(
        OFFICIAL_REPO_ID,
        revision=OFFICIAL_REVISION,
        budget_mb=args.budget_mb,
        cache_dir=args.cache_dir,
        max_cache_bytes=cache_bytes,
    )


def _atomic_json(path: str | Path, document: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        body = (
            json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        ).encode("utf-8")
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
    except (OSError, TypeError, ValueError) as exc:
        raise CliError(f"cannot prepare result: {destination}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except OSError as exc:
        raise CliError(f"cannot write result: {destination}") from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def _activation_summary(hidden: torch.Tensor) -> dict[str, Any]:
    values = hidden.detach().to(device="cpu", dtype=torch.float32)
    finite = torch.isfinite(values)
    return {
        "shape": list(values.shape),
        "dtype": str(hidden.dtype).removeprefix("torch."),
        "device": str(hidden.device),
        "all_finite": bool(finite.all().item()),
        "rms": float(values.square().mean().sqrt()),
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _metric(metrics: Mapping[str, Any], name: str) -> int:
    value = metrics.get(name, 0)
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def run(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    config = _config(args)
    if args.layers > config.n_layers:
        raise CliError(f"--layers exceeds decoder depth {config.n_layers}")
    if any(token >= config.vocab_size for token in args.token_ids):
        raise CliError("token ID outside the Qwen vocabulary")

    source = _source(args)
    pager: Qwen38WeightPager | None = None
    started = time.perf_counter()
    try:
        pager = Qwen38WeightPager(
            source,
            device=args.device,
            compute_dtype=args.compute_dtype,
            require_source_identity=not bool(args.source_dir),
        )
        model = StreamedQwen38(
            config,
            pager,
            max_batch_size=1,
            max_seq_len=max(1, len(args.token_ids)),
        )
        preflight = model.checkpoint_preflight()
        before = dict(source.metrics())
        hidden = None
        completed = 0
        if not args.dry_run:
            ids = torch.tensor([args.token_ids], dtype=torch.long)
            hidden = model.embed_batch(ids)
            for layer in range(args.layers):
                hidden, _ = model.forward_prefill_layer(hidden, ids, layer=layer)
                completed += 1
                pager.release()
                sys.stderr.write(
                    json.dumps(
                        {
                            "event": "qwen_layer_complete",
                            "layer": layer,
                            "layers": args.layers,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                sys.stderr.flush()
        after = dict(source.metrics())
        report = {
            "schema": RESULT_SCHEMA,
            "source": {
                "repo": getattr(source, "repo_id", None),
                "revision": getattr(source, "revision", None),
            },
            "protocol": {
                "token_ids": list(args.token_ids),
                "requested_layers": args.layers,
                "dry_run": bool(args.dry_run),
                "text_only": True,
                "vision": False,
                "mtp": False,
                "weight_order": "one BF16 matrix at a time",
            },
            "preflight": preflight,
            "execution": {
                "completed_layers": completed,
                "seconds": time.perf_counter() - started,
                "source_body_bytes": _metric(after, "network_or_source_body_bytes")
                - _metric(before, "network_or_source_body_bytes"),
                "pager": pager.metrics(),
                "activation": None if hidden is None else _activation_summary(hidden),
            },
        }
        if hidden is not None and not report["execution"]["activation"]["all_finite"]:
            raise CliError("Qwen activation contains non-finite values")
        output = _atomic_json(args.output, report)
        return report, output
    finally:
        if pager is not None:
            pager.close()
        source.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report, output = run(args)
    except (CliError, ValueError, KeyError) as exc:
        sys.stderr.write(f"qwen38_stream_smoke: error: {exc}\n")
        return 2
    print(
        json.dumps(
            {
                "output": str(output),
                "completed_layers": report["execution"]["completed_layers"],
                "source_body_bytes": report["execution"]["source_body_bytes"],
                "seconds": report["execution"]["seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
