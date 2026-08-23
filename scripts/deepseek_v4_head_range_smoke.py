#!/usr/bin/env python3
"""Exact DeepSeek-V4 full-head transport-envelope A/B.

The benchmark reuses a sealed layer-42 activation, applies the official HC
head/final norm once, and then scans the same 129,280 LM-head rows twice:

* baseline: one exact source leaf per unchanged 1,024-row compute block;
* candidate: up to eight adjacent leaves per 64-MiB transport envelope.

Decode, FP32 linear, stable top-k, and reduction order stay unchanged.  The
run fails closed unless the complete blockwise logit stream, top-k values and
token IDs are bit-identical and both arms transfer exactly the same bytes.
This is a transport smoke, not a language-quality or multi-core claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import stat
import statistics
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from immer.knowledge import Streamer  # noqa: E402
from immer.runtimes.deepseek_v4 import (  # noqa: E402
    DeepSeekV4Config,
    DeepSeekWeightPager,
    StreamedDeepSeekV4,
    runtime_dependency_versions,
    runtime_source_manifest,
)

OFFICIAL_SOURCE = "deepseek-ai/DeepSeek-V4-Flash-0731"
OFFICIAL_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
DEFAULT_CONFIG = ROOT / "artifacts" / "private" / "deepseek-v4-reference" / "config.json"
DEFAULT_ACTIVATION_MANIFEST = (
    ROOT
    / "artifacts"
    / "private"
    / "deepseek-v4-mmlu-off-4-exact-v4-window2x3"
    / "manifest.json"
)
DEFAULT_PREPARATION_CACHE = (
    ROOT / "artifacts" / "private" / "deepseek-v4-cache"
)
DEFAULT_TEMP_CACHE_PARENT = ROOT / "artifacts" / "private"
DEFAULT_OUTPUT = ROOT / "results" / "deepseek-v4-head-range-network-smoke.json"
RESULT_SCHEMA = "immer.deepseek-v4-head-range-smoke/v1"
_PINNED_REVISION = re.compile(r"[0-9a-fA-F]{40,64}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_SOURCE_DELTA_KEYS = (
    "network_or_source_body_bytes",
    "range_logical_leaves",
    "range_logical_leaf_bytes",
    "range_requests",
    "range_bytes_requested",
    "range_source_requests",
    "range_source_bytes",
    "failed_requests",
    "cache_hits",
    "cache_misses",
    "cache_writes",
    "cache_bytes_written",
    "transport_requests",
    "transport_retries",
)
_PAGER_DELTA_KEYS = (
    "head_rows",
    "head_logical_leaves",
    "head_transport_batches",
    "head_transport_envelopes",
    "head_transport_source_bytes",
    "head_planned_range_calls_avoided",
    "head_transport_fallbacks",
    "head_transport_fallback_leaves",
)


class HeadSmokeError(RuntimeError):
    """The input, resource receipt, or exactness gate is invalid."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(document: Any) -> bytes:
    return json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_digest(document: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(document)).hexdigest()


def _public_path(value: str | Path) -> str:
    """Return a checkout-relative path or a non-identifying external marker."""

    resolved = Path(value).expanduser().resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return "<external>"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        metadata = candidate.lstat()
    except OSError as exc:
        raise HeadSmokeError(f"cannot inspect {label}: {candidate}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise HeadSmokeError(f"{label} is not a regular file: {candidate}")
    return candidate.resolve()


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    source = _regular_file(path, label)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HeadSmokeError(f"cannot read {label}: {source}") from exc
    if not isinstance(document, dict):
        raise HeadSmokeError(f"{label} must contain a JSON object")
    return document


def _seal_report(document: Mapping[str, Any]) -> dict[str, Any]:
    if "report_sha256" in document:
        raise HeadSmokeError("report is already sealed")
    sealed = dict(document)
    sealed["report_sha256"] = _canonical_digest(sealed)
    return sealed


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    target = expanded.parent.resolve() / expanded.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise HeadSmokeError(f"refusing to replace output symlink: {target}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            payload = (
                json.dumps(
                    document,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n"
            ).encode("utf-8")
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _positive_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def _batch_blocks(raw: str) -> int:
    value = _positive_int(raw)
    if value > DeepSeekWeightPager.HEAD_TRANSPORT_MAX_RANGE_BATCH_BLOCKS:
        raise argparse.ArgumentTypeError(
            "must not exceed the pager's maximum head range batch width"
        )
    return value


def _trial_schedule(pairs: int) -> tuple[tuple[str, str], ...]:
    if isinstance(pairs, bool) or not isinstance(pairs, int) or pairs <= 0:
        raise ValueError("pairs must be a positive integer")
    return tuple(
        ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
        for index in range(pairs)
    )


def _metric_delta(
    before: Mapping[str, Any], after: Mapping[str, Any], keys: Sequence[str]
) -> dict[str, int]:
    result: dict[str, int] = {}
    for key in keys:
        try:
            start = int(before.get(key, 0))
            stop = int(after.get(key, 0))
        except (TypeError, ValueError) as exc:
            raise HeadSmokeError(f"non-integer metric {key!r}") from exc
        if stop < start:
            raise HeadSmokeError(f"metric {key!r} decreased during the arm")
        result[key] = stop - start
    return result


def _torch_tensor_bytes(tensor: Any) -> bytes:
    import torch

    value = tensor.detach().contiguous().to("cpu")
    if value.dtype == torch.bfloat16:
        return value.view(torch.uint16).numpy().tobytes(order="C")
    return value.numpy().tobytes(order="C")


def _tensor_digest(tensor: Any) -> str:
    material = {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "payload_sha256": hashlib.sha256(_torch_tensor_bytes(tensor)).hexdigest(),
    }
    return _canonical_digest(material)


def _synchronize(device: Any) -> None:
    import torch

    if str(device).startswith("mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _load_config(path: Path) -> tuple[DeepSeekV4Config, dict[str, Any]]:
    source = _regular_file(path, "config")
    raw = source.read_bytes()
    try:
        document = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise HeadSmokeError(f"invalid config JSON: {source}") from exc
    if not isinstance(document, Mapping):
        raise HeadSmokeError("config root must be an object")
    return DeepSeekV4Config.from_mapping(document), {
        "path": _public_path(source),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _validate_manifest(
    path: Path,
    *,
    config: DeepSeekV4Config,
    config_sha256: str,
    source: str,
    revision: str,
    variant: str,
    bucket: int,
    row: int,
) -> tuple[Any, dict[str, Any]]:
    """Load one sealed final activation and select a real last-token row."""

    import torch
    from safetensors import safe_open

    manifest_path = _regular_file(path, "activation manifest")
    document = _read_json_object(manifest_path, "activation manifest")
    if set(document) != {"schema", "version", "kind", "body", "body_sha256"}:
        raise HeadSmokeError("activation manifest has an unexpected envelope")
    body = document.get("body")
    if not isinstance(body, dict) or document.get("body_sha256") != _canonical_digest(
        body
    ):
        raise HeadSmokeError("activation manifest body seal is invalid")
    identity = body.get("identity")
    if not isinstance(identity, Mapping):
        raise HeadSmokeError("activation manifest identity is missing")
    model_identity = identity.get("model")
    if not isinstance(model_identity, Mapping):
        raise HeadSmokeError("activation manifest model identity is missing")
    if (
        model_identity.get("repo_id") != source
        or model_identity.get("revision") != revision
        or model_identity.get("config_sha256") != config_sha256
    ):
        raise HeadSmokeError("activation source/config identity does not match the A/B")

    state = body.get("state")
    checkpoints = state.get("checkpoints") if isinstance(state, Mapping) else None
    if not isinstance(checkpoints, list):
        raise HeadSmokeError("activation manifest checkpoints are missing")
    matches = [
        checkpoint
        for checkpoint in checkpoints
        if isinstance(checkpoint, Mapping)
        and checkpoint.get("variant") == variant
        and checkpoint.get("bucket") == bucket
    ]
    if len(matches) != 1:
        raise HeadSmokeError("activation checkpoint selection is not unique")
    checkpoint = dict(matches[0])
    if checkpoint.get("generation_layer") != config.n_layers - 1:
        raise HeadSmokeError("activation is not from the final decoder layer")
    digest = checkpoint.get("sha256")
    filename = checkpoint.get("file")
    if (
        not isinstance(digest, str)
        or _DIGEST.fullmatch(digest) is None
        or not isinstance(filename, str)
        or Path(filename).name != filename
        or filename != f"{digest}.safetensors"
    ):
        raise HeadSmokeError("activation checkpoint file identity is invalid")
    object_path = _regular_file(
        manifest_path.parent / "objects" / filename, "activation object"
    )
    if _sha256_file(object_path) != digest:
        raise HeadSmokeError("activation object SHA-256 mismatch")
    if checkpoint.get("file_bytes") != object_path.stat().st_size:
        raise HeadSmokeError("activation object byte count mismatch")

    with safe_open(object_path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if keys != ["hidden"]:
            raise HeadSmokeError(f"activation object has unexpected tensors: {keys}")
        hidden = handle.get_tensor("hidden")
    expected_shape = tuple(int(value) for value in checkpoint.get("shape", ()))
    if tuple(hidden.shape) != expected_shape or hidden.dtype != torch.bfloat16:
        raise HeadSmokeError("activation tensor shape/dtype mismatch")
    if hidden.numel() * hidden.element_size() != checkpoint.get("tensor_bytes"):
        raise HeadSmokeError("activation tensor byte count mismatch")
    if tuple(hidden.shape[2:]) != (config.hc_mult, config.dim):
        raise HeadSmokeError("activation tensor does not match config HC dimensions")

    plan = body.get("plan")
    buckets = plan.get("buckets") if isinstance(plan, Mapping) else None
    bucket_rows = [
        value
        for value in buckets or ()
        if isinstance(value, Mapping) and value.get("index") == bucket
    ]
    if len(bucket_rows) != 1:
        raise HeadSmokeError("activation plan bucket selection is not unique")
    bucket_plan = bucket_rows[0]
    valid_lengths = bucket_plan.get("valid_lengths")
    item_indices = bucket_plan.get("item_indices")
    if (
        not isinstance(valid_lengths, list)
        or not isinstance(item_indices, list)
        or len(valid_lengths) != hidden.shape[0]
        or len(item_indices) != hidden.shape[0]
        or isinstance(row, bool)
        or not 0 <= row < hidden.shape[0]
    ):
        raise HeadSmokeError("activation plan row selection is invalid")
    valid_length = valid_lengths[row]
    if (
        isinstance(valid_length, bool)
        or not isinstance(valid_length, int)
        or not 1 <= valid_length <= hidden.shape[1]
    ):
        raise HeadSmokeError("activation valid length is invalid")
    evidence = {
        "manifest_path": _public_path(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "manifest_body_sha256": document["body_sha256"],
        "object_path": _public_path(object_path),
        "object_sha256": digest,
        "checkpoint": checkpoint,
        "selected_bucket": bucket,
        "selected_bucket_row": row,
        "selected_item_index": item_indices[row],
        "selected_sequence_position": valid_length - 1,
    }
    return hidden, evidence


def _build_source(
    args: argparse.Namespace,
    *,
    cache_dir: Path,
    budget_mb: float,
    max_cache_bytes: int,
) -> tuple[Streamer, str]:
    if args.source != OFFICIAL_SOURCE:
        raise HeadSmokeError(
            "exact head smoke is bound to the official checkpoint source; "
            "local or alternate sources cannot satisfy its sealed activation provenance"
        )
    common = {
        "revision": args.revision,
        "budget_mb": budget_mb,
        "cache_dir": cache_dir,
        "use_cache": True,
        "max_cache_bytes": max_cache_bytes,
        "verbose": False,
    }
    if _PINNED_REVISION.fullmatch(args.revision) is None:
        raise HeadSmokeError("remote source requires an immutable revision digest")
    return Streamer(args.source, **common), args.source


def _prepare_hidden(
    args: argparse.Namespace,
    *,
    config: DeepSeekV4Config,
    hidden: Any,
    activation: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Apply official post-layer transformations outside the timed A/B."""

    cache = args.preparation_cache_dir.expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    source, source_label = _build_source(
        args,
        cache_dir=cache,
        budget_mb=args.preparation_budget_mb,
        max_cache_bytes=int(args.preparation_cache_gb * 1024**3),
    )
    try:
        source.inventory()
        before = source.metrics()
        pager = DeepSeekWeightPager(
            source,
            device=args.device,
            compute_dtype=args.dtype,
            expert_prefetch=False,
        )
        model = StreamedDeepSeekV4(
            config,
            pager,
            max_batch_size=int(hidden.shape[0]),
            max_seq_len=int(hidden.shape[1]),
        )
        final = model.finalize_hidden(hidden)
        row = int(activation["selected_bucket_row"])
        position = int(activation["selected_sequence_position"])
        selected = final[row, position].unsqueeze(0).detach().to("cpu")
        if tuple(selected.shape) != (1, config.dim):
            raise HeadSmokeError("prepared head input has an invalid shape")
        if not bool(selected.float().isfinite().all().item()):
            raise HeadSmokeError("prepared head input contains non-finite values")
        pager.release()
        after = source.metrics()
        return selected, {
            "source": source_label,
            "source_metrics_delta": _metric_delta(before, after, _SOURCE_DELTA_KEYS),
            "hidden_sha256": _tensor_digest(selected),
            "hidden_dtype": str(selected.dtype).removeprefix("torch."),
            "hidden_shape": list(selected.shape),
        }
    finally:
        source.close()


def _head_manifest(meta: Mapping[str, Any]) -> dict[str, Any]:
    try:
        shape = [int(value) for value in meta["shape"]]
        offsets = [int(value) for value in meta["offset_in_shard"]]
        result = {
            "name": str(meta["name"]),
            "dtype": str(meta["dtype"]).upper(),
            "shape": shape,
            "shard": str(meta["shard"]),
            "data_start": int(meta["data_start"]),
            "offset_in_shard": offsets,
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise HeadSmokeError("LM-head metadata is incomplete") from exc
    if len(shape) != 2 or len(offsets) != 2 or offsets[1] <= offsets[0]:
        raise HeadSmokeError("LM-head metadata shape/offsets are invalid")
    result["payload_bytes"] = offsets[1] - offsets[0]
    result["sha256"] = _canonical_digest(result)
    return result


def _logit_stream_digest(blocks: Sequence[tuple[int, Any]]) -> tuple[str, int]:
    import torch

    digest = hashlib.sha256()
    rows = 0
    expected_start = 0
    for start, logits in blocks:
        if start != expected_start or logits.ndim != 2:
            raise HeadSmokeError("instrumented logit blocks are out of order")
        payload = _torch_tensor_bytes(logits.to(dtype=torch.float32))
        shape = tuple(int(value) for value in logits.shape)
        digest.update(start.to_bytes(8, "big"))
        digest.update(len(shape).to_bytes(4, "big"))
        for dimension in shape:
            digest.update(dimension.to_bytes(8, "big"))
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        rows += shape[-1]
        expected_start += shape[-1]
    digest.update(len(blocks).to_bytes(8, "big"))
    return digest.hexdigest(), rows


def _run_arm(
    args: argparse.Namespace,
    *,
    hidden: Any,
    arm: str,
    cache_dir: Path,
) -> dict[str, Any]:
    import torch

    batch_blocks = 1 if arm == "baseline" else args.candidate_batch_blocks
    source, source_label = _build_source(
        args,
        cache_dir=cache_dir,
        budget_mb=args.source_budget_mb,
        max_cache_bytes=int(args.cache_budget_gb * 1024**3),
    )
    try:
        source.inventory()
        meta = source.find(args.head_name)
        head = _head_manifest(meta)
        expected_payload = int(np.prod(head["shape"], dtype=np.int64)) * 2
        if head["dtype"] != "BF16" or head["payload_bytes"] != expected_payload:
            raise HeadSmokeError("A/B requires the official contiguous BF16 LM head")
        pager = DeepSeekWeightPager(
            source,
            device=args.device,
            compute_dtype=args.dtype,
            expert_prefetch=False,
        )
        source_before = source.metrics()
        pager_before = pager.metrics()
        observed_blocks: list[tuple[int, Any]] = []

        def observe(start: int, logits: Any) -> None:
            observed_blocks.append((int(start), logits))

        _synchronize(pager.device)
        started = time.perf_counter()
        values, ids = pager.topk_logits(
            hidden,
            k=args.top_k,
            block_rows=args.block_rows,
            transport_range_batch_blocks=batch_blocks,
            name=args.head_name,
            instrument_block_observer=observe,
        )
        _synchronize(pager.device)
        seconds = time.perf_counter() - started
        source_after = source.metrics()
        pager_after = pager.metrics()
        logit_sha256, logit_rows = _logit_stream_digest(observed_blocks)
        values_cpu = values.detach().to("cpu", dtype=torch.float32)
        ids_cpu = ids.detach().to("cpu", dtype=torch.long)
        if not bool(values_cpu.isfinite().all().item()):
            raise HeadSmokeError("LM-head top-k contains non-finite logits")
        pager.release()
        source.close()
        closed_metrics = source.metrics()
        return {
            "arm": arm,
            "transport_range_batch_blocks": batch_blocks,
            "seconds": seconds,
            "head": head,
            "logit_stream_sha256": logit_sha256,
            "logit_rows": logit_rows,
            "logit_blocks": len(observed_blocks),
            "topk_values": values_cpu.tolist(),
            "topk_token_ids": ids_cpu.tolist(),
            "topk_values_sha256": _tensor_digest(values_cpu),
            "topk_token_ids_sha256": _tensor_digest(ids_cpu),
            "source": source_label,
            "source_metrics_delta": _metric_delta(
                source_before, source_after, _SOURCE_DELTA_KEYS
            ),
            "pager_metrics_delta": _metric_delta(
                pager_before, pager_after, _PAGER_DELTA_KEYS
            ),
            "source_identity": {
                "revision_is_pinned": bool(closed_metrics.get("revision_is_pinned")),
                "revision_is_mutable": bool(closed_metrics.get("revision_is_mutable")),
                "inventory_source_fingerprint": closed_metrics.get(
                    "inventory_source_fingerprint"
                ),
                "transport_policy": closed_metrics.get("transport_policy"),
                "transport_connection_limit": closed_metrics.get(
                    "transport_connection_limit"
                ),
                "transport_peak_leases": closed_metrics.get(
                    "transport_peak_leases"
                ),
                "transport_active_leases": closed_metrics.get(
                    "transport_active_leases"
                ),
                "transport_closed": closed_metrics.get("transport_closed"),
            },
        }
    finally:
        source.close()


def _validate_pair(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    exact_fields = (
        "head",
        "logit_stream_sha256",
        "logit_rows",
        "logit_blocks",
        "topk_values_sha256",
        "topk_token_ids_sha256",
        "topk_values",
        "topk_token_ids",
    )
    mismatches = [key for key in exact_fields if baseline.get(key) != candidate.get(key)]
    if mismatches:
        raise HeadSmokeError(f"LM-head exactness mismatch: {mismatches}")
    base_source = baseline["source_metrics_delta"]
    candidate_source = candidate["source_metrics_delta"]
    if base_source["network_or_source_body_bytes"] != candidate_source[
        "network_or_source_body_bytes"
    ]:
        raise HeadSmokeError("A/B arms transferred different source body bytes")
    if base_source["range_source_bytes"] != candidate_source["range_source_bytes"]:
        raise HeadSmokeError("A/B arms have different physical source range bytes")
    if base_source["failed_requests"] or candidate_source["failed_requests"]:
        raise HeadSmokeError("A/B contains failed source requests")
    if base_source["transport_retries"] or candidate_source["transport_retries"]:
        raise HeadSmokeError("A/B contains transport retries")
    baseline_requests = int(base_source["range_source_requests"])
    candidate_requests = int(candidate_source["range_source_requests"])
    if candidate_requests >= baseline_requests:
        raise HeadSmokeError(
            "candidate did not reduce physical source range requests"
        )
    if candidate["pager_metrics_delta"]["head_transport_fallbacks"]:
        raise HeadSmokeError("candidate used a scalar head transport fallback")
    for identity in (baseline["source_identity"], candidate["source_identity"]):
        if (
            not identity["revision_is_pinned"]
            or identity["revision_is_mutable"]
            or identity["transport_active_leases"] != 0
            or identity["transport_closed"] is not True
        ):
            raise HeadSmokeError("source identity/transport did not close cleanly")
    return {
        "bit_identical": True,
        "source_bytes_equal": True,
        "baseline_source_requests": baseline_requests,
        "candidate_source_requests": candidate_requests,
        "source_requests_avoided": baseline_requests - candidate_requests,
        "request_reduction_fraction": (
            (baseline_requests - candidate_requests) / baseline_requests
        ),
        "speedup_baseline_over_candidate": (
            float(baseline["seconds"]) / float(candidate["seconds"])
        ),
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile needs at least one value")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _summarize(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_arm = {
        arm: [float(row[arm]["seconds"]) for row in trials]
        for arm in ("baseline", "candidate")
    }
    speedups = [float(row["comparison"]["speedup_baseline_over_candidate"]) for row in trials]
    return {
        "pairs": len(trials),
        "candidate_wins": sum(value > 1.0 for value in speedups),
        "speedups_baseline_over_candidate": speedups,
        "median_speedup_baseline_over_candidate": statistics.median(speedups),
        "baseline": {
            "mean_seconds": statistics.fmean(by_arm["baseline"]),
            "median_seconds": statistics.median(by_arm["baseline"]),
            "p90_seconds": _percentile(by_arm["baseline"], 0.9),
        },
        "candidate": {
            "mean_seconds": statistics.fmean(by_arm["candidate"]),
            "median_seconds": statistics.median(by_arm["candidate"]),
            "p90_seconds": _percentile(by_arm["candidate"], 0.9),
        },
        "performance_claim": (
            "mechanism_smoke_only; fewer than four paired trials"
            if len(trials) < 4
            else "paired latency evidence; inspect dispersion and tails"
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_at = _utc_now()
    config, config_meta = _load_config(args.config)
    hidden, activation = _validate_manifest(
        args.activation_manifest,
        config=config,
        config_sha256=config_meta["sha256"],
        source=args.source,
        revision=args.revision,
        variant=args.activation_variant,
        bucket=args.activation_bucket,
        row=args.activation_row,
    )
    prepared, preparation = _prepare_hidden(
        args, config=config, hidden=hidden, activation=activation
    )
    del hidden

    temporary_parent = args.temp_cache_parent.expanduser().resolve()
    temporary_parent.mkdir(parents=True, exist_ok=True)
    head_payload_bytes = config.vocab_size * config.dim * 2
    required_free_bytes = head_payload_bytes + 256 * 1024**2
    free_bytes = shutil.disk_usage(temporary_parent).free
    if free_bytes < required_free_bytes:
        raise HeadSmokeError(
            "insufficient temporary disk for one cold exact head cache: "
            f"{free_bytes} < {required_free_bytes} bytes"
        )
    trials: list[dict[str, Any]] = []
    for pair_index, order in enumerate(_trial_schedule(args.pairs)):
        arms: dict[str, dict[str, Any]] = {}
        for arm in order:
            with tempfile.TemporaryDirectory(
                prefix=f".deepseek-head-{pair_index}-{arm}-",
                dir=temporary_parent,
            ) as raw_cache:
                arms[arm] = _run_arm(
                    args,
                    hidden=prepared,
                    arm=arm,
                    cache_dir=Path(raw_cache),
                )
        comparison = _validate_pair(arms["baseline"], arms["candidate"])
        trials.append(
            {
                "pair": pair_index,
                "execution_order": list(order),
                "baseline": arms["baseline"],
                "candidate": arms["candidate"],
                "comparison": comparison,
            }
        )

    runtime_sources = runtime_source_manifest(
        project_files=("scripts/deepseek_v4_head_range_smoke.py",)
    )
    runtime_dependencies = runtime_dependency_versions()
    report = _seal_report(
        {
            "schema": RESULT_SCHEMA,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "status": "passed",
            "verdict": (
                "exact LM-head transport-envelope mechanism proven; production "
                "default remains one leaf until repeated latency evidence"
            ),
            "protocol": {
                "pairs": args.pairs,
                "trial_schedule": [list(value) for value in _trial_schedule(args.pairs)],
                "baseline_transport_range_batch_blocks": 1,
                "candidate_transport_range_batch_blocks": (
                    args.candidate_batch_blocks
                ),
                "compute_block_rows": args.block_rows,
                "top_k": args.top_k,
                "raw_resident_limit_bytes": (
                    DeepSeekWeightPager.HEAD_TRANSPORT_RESIDENT_LIMIT_BYTES
                ),
                "same_hidden_for_all_arms": True,
                "separate_cold_leaf_cache_per_arm": True,
                "temporary_cache_required_free_bytes": required_free_bytes,
                "temporary_cache_free_bytes_at_preflight": free_bytes,
                "compute_order": "unchanged sequential row blocks",
                "exactness_gate": (
                    "full blockwise FP32 logit stream + top-k values/IDs bit-identical"
                ),
            },
            "summary": _summarize(trials),
            "trials": trials,
            "provenance": {
                "source": args.source,
                "revision": args.revision,
                "config": config_meta,
                "activation": activation,
                "preparation": preparation,
                "runtime_sources": runtime_sources,
                "runtime_source_sha256": _canonical_digest(runtime_sources),
                "runtime_dependencies": runtime_dependencies,
                "runtime_dependency_sha256": _canonical_digest(runtime_dependencies),
                "harness_sha256": _sha256_file(Path(__file__).resolve()),
                "hardware": {
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                    "python": platform.python_version(),
                },
                "execution": {
                    "device": args.device,
                    "compute_dtype": args.dtype,
                    "head_transport_policy": DeepSeekWeightPager.HEAD_TRANSPORT_POLICY,
                    "head_transport_default_range_batch_blocks": (
                        DeepSeekWeightPager.HEAD_TRANSPORT_DEFAULT_RANGE_BATCH_BLOCKS
                    ),
                    "head_transport_max_range_batch_blocks": (
                        DeepSeekWeightPager.HEAD_TRANSPORT_MAX_RANGE_BATCH_BLOCKS
                    ),
                    "head_transport_resident_limit_bytes": (
                        DeepSeekWeightPager.HEAD_TRANSPORT_RESIDENT_LIMIT_BYTES
                    ),
                },
            },
        }
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=OFFICIAL_SOURCE)
    parser.add_argument("--revision", default=OFFICIAL_REVISION)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--activation-manifest", type=Path, default=DEFAULT_ACTIVATION_MANIFEST
    )
    parser.add_argument("--activation-variant", default="off")
    parser.add_argument("--activation-bucket", type=int, default=0)
    parser.add_argument("--activation-row", type=int, default=0)
    parser.add_argument("--head-name", default="head.weight")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="mps")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--block-rows", type=_positive_int, default=1024)
    parser.add_argument("--candidate-batch-blocks", type=_batch_blocks, default=8)
    parser.add_argument("--top-k", type=_positive_int, default=16)
    parser.add_argument("--pairs", type=_positive_int, default=1)
    parser.add_argument("--source-budget-mb", type=_positive_float, default=2048.0)
    parser.add_argument("--cache-budget-gb", type=_positive_float, default=1.25)
    parser.add_argument(
        "--preparation-cache-dir", type=Path, default=DEFAULT_PREPARATION_CACHE
    )
    parser.add_argument(
        "--preparation-budget-mb", type=_positive_float, default=512.0
    )
    parser.add_argument(
        "--preparation-cache-gb", type=_positive_float, default=12.0
    )
    parser.add_argument(
        "--temp-cache-parent", type=Path, default=DEFAULT_TEMP_CACHE_PARENT
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run(args)
        _atomic_write_json(args.output_json, report)
    except Exception as exc:
        print(
            json.dumps(
                {"status": "error", "error": f"{type(exc).__name__}: {exc}"},
                sort_keys=True,
            ),
            file=os.sys.stderr,
        )
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"geschrieben: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
