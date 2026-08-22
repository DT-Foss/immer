#!/usr/bin/env python3
"""Reproduce an official DeepSeek-V4 exact-prefetch-window A/B.

This is a bounded transport/latency microbenchmark, not a model-quality claim.
It evaluates the same routed experts, input, routing weights, and serial FP32
accumulation with exact prefetch disabled and enabled.  Any value or bit-level
output mismatch fails the run before a success report is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from immer.knowledge import Streamer
from immer.runtimes.deepseek_v4 import (
    DeepSeekV4Config,
    DeepSeekWeightPager,
    runtime_dependency_versions,
    runtime_source_manifest,
)


ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_SOURCE = "deepseek-ai/DeepSeek-V4-Flash-0731"
OFFICIAL_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
DEFAULT_CACHE = ROOT / "artifacts" / "private" / "deepseek-v4-cache"
DEFAULT_OUTPUT = ROOT / "results" / "deepseek-v4-exact-prefetch-window-smoke.json"
RESULT_SCHEMA = "immer.deepseek-v4-exact-prefetch-smoke/v2"
_PINNED_REVISION = re.compile(r"[0-9a-fA-F]{40,64}")
_PREFETCH_COUNTERS = (
    "expert_calls",
    "coalesced_expert_calls",
    "expert_source_ranges",
    "expert_prefetch_submitted",
    "expert_prefetch_consumed",
    "expert_prefetch_failures",
    "expert_prefetch_sync_fallbacks",
    "expert_prefetch_payload_bytes",
    "expert_prefetch_wait_ns",
    "expert_prefetch_ready_before_consume",
    "expert_prefetch_cancelled",
)


class SmokeError(RuntimeError):
    """The requested microbenchmark is invalid or its exactness gate failed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_digest(document: Any) -> str:
    payload = json.dumps(
        document,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _seal_report(document: Mapping[str, Any]) -> dict[str, Any]:
    if "report_sha256" in document:
        raise SmokeError("report payload already contains report_sha256")
    result = dict(document)
    result["report_sha256"] = _canonical_digest(result)
    return result


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    target = expanded.parent.resolve() / expanded.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise SmokeError(f"refusing to replace output symlink: {target}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
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
        if os.path.exists(temporary):
            os.unlink(temporary)


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _nonnegative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def _positive_even_int(raw: str) -> int:
    value = _positive_int(raw)
    if value % 2:
        raise argparse.ArgumentTypeError("must be even for a balanced A/B")
    return value


def _nonnegative_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return value


def _positive_float(raw: str) -> float:
    value = _nonnegative_float(raw)
    if value == 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _bounded_rows(raw: str) -> int:
    value = _positive_int(raw)
    if value > 32:
        raise argparse.ArgumentTypeError("must not exceed 32")
    return value


def _parse_experts(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in raw.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "experts must be comma-separated integers"
        ) from exc
    if len(values) < 2 or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError(
            "experts must contain at least two non-negative integers"
        )
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("experts must be distinct")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=OFFICIAL_SOURCE)
    parser.add_argument("--revision", default=OFFICIAL_REVISION)
    parser.add_argument("--config")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--source-budget-mb", type=_positive_int, default=4096)
    parser.add_argument("--cache-budget-gb", type=_nonnegative_float, default=12.0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "bfloat16"), default="auto")
    parser.add_argument("--layer", type=_nonnegative_int, default=3)
    parser.add_argument("--experts", type=_parse_experts, default=(0, 1, 2))
    parser.add_argument("--route-weight", type=_positive_float, default=0.25)
    parser.add_argument("--rows", type=_bounded_rows, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-rounds", type=_positive_int, default=1)
    parser.add_argument("--trials", type=_positive_even_int, default=20)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser


def _trial_schedule(trials: int) -> tuple[str, ...]:
    if isinstance(trials, bool) or not isinstance(trials, int) or trials <= 0:
        raise SmokeError("trial count must be a positive integer")
    if trials % 2:
        raise SmokeError("trial count must be even for a balanced A/B")
    result: list[str] = []
    for pair in range(trials // 2):
        result.extend(("off", "on") if pair % 2 == 0 else ("on", "off"))
    return tuple(result)


def _summarize_trials(
    trials: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {"off": [], "on": []}
    for index, row in enumerate(trials):
        mode = row.get("mode")
        seconds = row.get("seconds")
        if mode not in grouped:
            raise SmokeError(f"trial {index} has invalid mode {mode!r}")
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(float(seconds))
            or float(seconds) <= 0
        ):
            raise SmokeError(f"trial {index} has invalid elapsed time")
        grouped[str(mode)].append(float(seconds))
    if not grouped["off"] or len(grouped["off"]) != len(grouped["on"]):
        raise SmokeError("A/B trials must contain equal non-empty mode counts")

    modes: dict[str, dict[str, float | int | list[float]]] = {}
    for mode, values in grouped.items():
        modes[mode] = {
            "count": len(values),
            "seconds": values,
            "mean_seconds": statistics.fmean(values),
            "median_seconds": statistics.median(values),
            "minimum_seconds": min(values),
            "maximum_seconds": max(values),
        }
    off_mean = float(modes["off"]["mean_seconds"])
    on_mean = float(modes["on"]["mean_seconds"])
    return {
        "modes": modes,
        "mean_speedup_off_over_on": off_mean / on_mean,
        "mean_latency_reduction_fraction": 1.0 - on_mean / off_mean,
    }


def _local_source_path(value: str) -> Path | None:
    raw = value.removeprefix("local:") if value.startswith("local:") else value
    candidate = Path(raw).expanduser()
    explicit = (
        value.startswith("local:")
        or candidate.is_absolute()
        or raw.startswith(("./", "../"))
    )
    if explicit:
        return candidate.resolve()
    return candidate.resolve() if candidate.is_dir() else None


def _build_source(args: argparse.Namespace) -> tuple[Streamer, str]:
    common = {
        "revision": args.revision,
        "budget_mb": args.source_budget_mb,
        "cache_dir": Path(args.cache_dir).expanduser().resolve(),
        "use_cache": not args.no_cache,
        "max_cache_bytes": int(args.cache_budget_gb * 1024**3),
        "verbose": False,
    }
    local = _local_source_path(args.source)
    if local is not None:
        if not local.is_dir():
            raise SmokeError(f"local checkpoint directory does not exist: {local}")
        return Streamer.from_local(local, **common), f"local:{local}"
    if _PINNED_REVISION.fullmatch(args.revision) is None:
        raise SmokeError("remote source requires an immutable revision digest")
    return Streamer(args.source, **common), args.source


def _load_config(
    args: argparse.Namespace, source: Streamer
) -> tuple[DeepSeekV4Config, dict[str, Any]]:
    if args.config is None:
        raw = source.reader.fetch_file("config.json")
        location = "source:config.json"
    else:
        path = Path(args.config).expanduser().resolve()
        raw = path.read_bytes()
        location = str(path)
    try:
        document = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SmokeError(f"invalid DeepSeek-V4 config JSON at {location}") from exc
    if not isinstance(document, Mapping):
        raise SmokeError("DeepSeek-V4 config root must be an object")
    return DeepSeekV4Config.from_mapping(document), {
        "location": location,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _expert_manifest(
    source: Streamer, bases: Sequence[str]
) -> tuple[list[dict[str, Any]], str]:
    result: list[dict[str, Any]] = []
    for base in bases:
        for role in ("w1", "w2", "w3"):
            for suffix in ("weight", "scale"):
                name = f"{base}.{role}.{suffix}"
                try:
                    meta = source.find(name)
                except KeyError as exc:
                    raise SmokeError(
                        f"official expert tensor is missing: {name}"
                    ) from exc
                result.append(
                    {
                        "name": name,
                        "shard": meta["shard"],
                        "dtype": str(meta["dtype"]).upper(),
                        "shape": [int(value) for value in meta["shape"]],
                        "offset_in_shard": [
                            int(value) for value in meta["offset_in_shard"]
                        ],
                    }
                )
    return result, _canonical_digest(result)


def _synchronize(pager: DeepSeekWeightPager) -> None:
    if pager.device.type != "mps":
        return
    synchronize = getattr(getattr(pager.torch, "mps", None), "synchronize", None)
    if not callable(synchronize):
        raise SmokeError("MPS timing requires torch.mps.synchronize()")
    synchronize()


def _execute_experts(
    pager: DeepSeekWeightPager,
    hidden: Any,
    bases: Sequence[str],
    *,
    route_weight: float,
    swiglu_limit: float,
) -> Any:
    torch = pager.torch
    weights = torch.full(
        (hidden.shape[0], 1),
        float(route_weight),
        dtype=torch.float32,
        device=hidden.device,
    )
    output = torch.zeros_like(hidden, dtype=torch.float32)
    window = pager.prefetch_expert_window(bases) if bases else None
    try:
        for base in bases:
            payload = (
                pager.consume_expert_window(window, base)
                if window is not None
                else None
            )
            try:
                claimed = payload
                payload = None
                try:
                    expert = pager.expert(
                        hidden,
                        base,
                        route_weight=weights,
                        swiglu_limit=swiglu_limit,
                        prefetched_payload=claimed,
                    )
                finally:
                    del claimed
            finally:
                if payload is not None:
                    pager.discard_expert_payload(payload)
            output += expert.float()
    except BaseException:
        if window is not None and not window.closed:
            pager.close_expert_window(window, cancel=True)
        raise
    if window is not None:
        pager.close_expert_window(window)
    return output.to(hidden.dtype)


def _tensor_bytes(tensor: Any) -> bytes:
    import torch

    flat = tensor.detach().to("cpu").contiguous().view(torch.uint8)
    return flat.numpy().tobytes(order="C")


def _tensor_digest(tensor: Any) -> str:
    return hashlib.sha256(_tensor_bytes(tensor)).hexdigest()


def _bit_equal(left: Any, right: Any) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    import torch

    left_bytes = left.detach().to("cpu").contiguous().view(torch.uint8)
    right_bytes = right.detach().to("cpu").contiguous().view(torch.uint8)
    return bool(left_bytes.equal(right_bytes))


def _counter_snapshot(pager: DeepSeekWeightPager) -> dict[str, int]:
    metrics = pager.metrics()
    return {key: int(metrics[key]) for key in _PREFETCH_COUNTERS}


def _counter_delta(
    before: Mapping[str, int], after: Mapping[str, int]
) -> dict[str, int]:
    return {key: int(after[key]) - int(before[key]) for key in _PREFETCH_COUNTERS}


def _pager_projection(pager: DeepSeekWeightPager) -> dict[str, Any]:
    metrics = pager.metrics()
    return {
        **{key: int(metrics[key]) for key in _PREFETCH_COUNTERS},
        "expert_prefetch_peak_bytes": int(metrics["expert_prefetch_peak_bytes"]),
        "expert_prefetch_max_outstanding": int(
            metrics["expert_prefetch_max_outstanding"]
        ),
        "expert_prefetch_policy": metrics["expert_prefetch_policy"],
        "expert_prefetch_payload_limit_bytes": int(
            metrics["expert_prefetch_payload_limit_bytes"]
        ),
        "expert_prefetch_transport_policy": metrics["expert_prefetch_transport_policy"],
        "expert_prefetch_workers": int(metrics["expert_prefetch_workers"]),
        "expert_prefetch_active_read_limit": int(
            metrics["expert_prefetch_active_read_limit"]
        ),
        "expert_prefetch_max_outstanding_limit": int(
            metrics["expert_prefetch_max_outstanding_limit"]
        ),
        "expert_prefetch_max_experts": int(metrics["expert_prefetch_max_experts"]),
        "expert_prefetch_resident_limit_bytes": int(
            metrics["expert_prefetch_resident_limit_bytes"]
        ),
        "expert_prefetch_draining": bool(metrics["expert_prefetch_draining"]),
        "device": metrics["device"],
        "compute_dtype": metrics["compute_dtype"],
        "quantized_accumulation_policy": metrics["quantized_accumulation_policy"],
    }


def _timed_trial(
    mode: str,
    pager: DeepSeekWeightPager,
    hidden: Any,
    bases: Sequence[str],
    *,
    route_weight: float,
    swiglu_limit: float,
) -> tuple[Any, float, dict[str, int], int]:
    before = _counter_snapshot(pager)
    source_before = int(pager.source.bytes_moved())
    _synchronize(pager)
    started = time.perf_counter()
    output = _execute_experts(
        pager,
        hidden,
        bases,
        route_weight=route_weight,
        swiglu_limit=swiglu_limit,
    )
    _synchronize(pager)
    seconds = time.perf_counter() - started
    after = _counter_snapshot(pager)
    if not math.isfinite(seconds) or seconds <= 0:
        raise SmokeError(f"{mode} trial produced an invalid elapsed time")
    return (
        output,
        seconds,
        _counter_delta(before, after),
        int(pager.source.bytes_moved()) - source_before,
    )


def _assert_exact(
    output: Any, reference: Any, *, label: str
) -> tuple[bool, bool, bool]:
    import torch

    finite = bool(output.float().isfinite().all().item())
    value_equal = bool(torch.equal(output, reference))
    bit_equal = _bit_equal(output, reference)
    if not finite:
        raise SmokeError(f"{label}: output contains a non-finite value")
    if not value_equal or not bit_equal:
        raise SmokeError(
            f"{label}: exactness mismatch "
            f"(torch.equal={value_equal}, bit_equal={bit_equal})"
        )
    return finite, value_equal, bit_equal


def _arguments(args: argparse.Namespace) -> dict[str, Any]:
    result = vars(args).copy()
    result["experts"] = list(args.experts)
    for key in ("cache_dir", "config", "output"):
        if result.get(key) is not None:
            result[key] = str(Path(result[key]).expanduser().resolve())
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_at = _utc_now()
    source, source_label = _build_source(args)
    try:
        return _run_with_source(
            args,
            source=source,
            source_label=source_label,
            started_at=started_at,
        )
    finally:
        source.close()


def _run_with_source(
    args: argparse.Namespace,
    *,
    source: Streamer,
    source_label: str,
    started_at: str,
) -> dict[str, Any]:
    config, config_meta = _load_config(args, source)
    requested = tuple(int(value) for value in args.experts)
    if args.layer >= config.n_layers:
        raise SmokeError(
            f"layer {args.layer} is outside checkpoint depth {config.n_layers}"
        )
    if any(expert >= config.n_routed_experts for expert in requested):
        raise SmokeError(f"expert ID is outside [0, {config.n_routed_experts})")
    experts = tuple(sorted(requested))
    bases = tuple(f"layers.{args.layer}.ffn.experts.{expert}" for expert in experts)
    source.inventory()
    expert_manifest, expert_manifest_sha = _expert_manifest(source, bases)
    expert_payload_bytes = {
        base: sum(
            int(row["offset_in_shard"][1]) - int(row["offset_in_shard"][0])
            for row in expert_manifest
            if str(row["name"]).startswith(f"{base}.")
        )
        for base in bases
    }
    if any(
        value > DeepSeekWeightPager.EXPERT_PREFETCH_PAYLOAD_LIMIT_BYTES
        for value in expert_payload_bytes.values()
    ):
        raise SmokeError("selected expert exceeds the exact-prefetch payload limit")
    expected_prefetch_peak = max(
        sum(expert_payload_bytes[base] for base in bases[start : start + 3])
        for start in range(len(bases))
    )
    source_metrics_before = source.metrics()
    body_before_warmup = int(source.bytes_moved())

    off = DeepSeekWeightPager(
        source,
        device=args.device,
        compute_dtype=args.dtype,
        expert_prefetch=False,
    )
    on = DeepSeekWeightPager(
        source,
        device=args.device,
        compute_dtype=args.dtype,
        expert_prefetch=True,
    )
    if off.device != on.device or off.compute_dtype != on.compute_dtype:
        raise SmokeError("A/B pager device or dtype resolution diverged")

    torch = off.torch
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    hidden = torch.randn(
        (args.rows, config.dim), generator=generator, dtype=torch.float32
    ).to(device=off.device, dtype=off.compute_dtype)
    input_sha256 = _tensor_digest(hidden)
    reference = None
    warmup_records: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    try:
        for round_index in range(args.warmup_rounds):
            for mode, pager in (("off", off), ("on", on)):
                output, seconds, delta, source_delta = _timed_trial(
                    mode,
                    pager,
                    hidden,
                    bases,
                    route_weight=args.route_weight,
                    swiglu_limit=config.swiglu_limit,
                )
                if reference is None:
                    if not bool(output.float().isfinite().all().item()):
                        raise SmokeError("warmup reference output is non-finite")
                    reference = output.detach().clone()
                finite, value_equal, bit_equal = _assert_exact(
                    output, reference, label=f"warmup {round_index} {mode}"
                )
                warmup_records.append(
                    {
                        "round": round_index,
                        "mode": mode,
                        "seconds": seconds,
                        "finite": finite,
                        "torch_equal": value_equal,
                        "bit_equal": bit_equal,
                        "output_sha256": _tensor_digest(output),
                        "pager_delta": delta,
                        "source_body_bytes_delta": source_delta,
                    }
                )
                del output

        if reference is None:
            raise SmokeError("warmup did not produce a reference output")
        body_after_warmup = int(source.bytes_moved())
        for index, mode in enumerate(_trial_schedule(args.trials)):
            pager = off if mode == "off" else on
            output, seconds, delta, source_delta = _timed_trial(
                mode,
                pager,
                hidden,
                bases,
                route_weight=args.route_weight,
                swiglu_limit=config.swiglu_limit,
            )
            finite, value_equal, bit_equal = _assert_exact(
                output, reference, label=f"trial {index} {mode}"
            )
            expert_count = len(bases)
            expected_prefetch = expert_count if mode == "on" else 0
            if (
                delta["expert_calls"] != expert_count
                or delta["coalesced_expert_calls"] != expert_count
                or delta["expert_source_ranges"] != 2 * expert_count
                or delta["expert_prefetch_submitted"] != expected_prefetch
                or delta["expert_prefetch_consumed"] != expected_prefetch
                or delta["expert_prefetch_failures"]
                or delta["expert_prefetch_sync_fallbacks"]
                or delta["expert_prefetch_cancelled"]
            ):
                raise SmokeError(
                    f"trial {index} {mode}: exact prefetch counter contract failed"
                )
            trials.append(
                {
                    "index": index,
                    "mode": mode,
                    "seconds": seconds,
                    "finite": finite,
                    "torch_equal": value_equal,
                    "bit_equal": bit_equal,
                    "output_sha256": _tensor_digest(output),
                    "pager_delta": delta,
                    "source_body_bytes_delta": source_delta,
                }
            )
            del output
        body_after_trials = int(source.bytes_moved())
    finally:
        off.release()
        on.release()

    off_metrics = _pager_projection(off)
    on_metrics = _pager_projection(on)
    if on_metrics["expert_prefetch_draining"]:
        raise SmokeError("exact prefetch worker is still draining after the run")
    if on_metrics["expert_prefetch_peak_bytes"] != expected_prefetch_peak:
        raise SmokeError(
            "exact prefetch payload peak does not match the selected expert window"
        )
    expected_outstanding = min(
        len(bases), DeepSeekWeightPager.EXPERT_PREFETCH_MAX_OUTSTANDING
    )
    if (
        on_metrics["expert_prefetch_max_outstanding"] != expected_outstanding
        or on_metrics["expert_prefetch_active_read_limit"]
        != DeepSeekWeightPager.EXPERT_PREFETCH_WORKERS
    ):
        raise SmokeError("exact prefetch queue/active-read contract failed")
    summary = _summarize_trials(trials)
    runtime_sources = runtime_source_manifest()
    runtime_dependencies = runtime_dependency_versions()
    harness_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source.close()
    source_metrics_after = source.metrics()
    if (
        int(source_metrics_after.get("transport_active_leases", 0)) != 0
        or int(source_metrics_after.get("transport_peak_leases", 0))
        > DeepSeekWeightPager.EXPERT_PREFETCH_ACTIVE_READ_LIMIT
    ):
        raise SmokeError("source transport lease bound failed")
    if str(source_metrics_after.get("transport_policy", "")).startswith(
        "requests-session-pool-"
    ) and not bool(source_metrics_after.get("transport_closed")):
        raise SmokeError("persistent source transport did not close")
    report = _seal_report(
        {
            "schema": RESULT_SCHEMA,
            "status": "ok",
            "scope": "official_checkpoint_expert_transport_latency_only",
            "quality_or_end_to_end_performance_claim": False,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "arguments": _arguments(args),
            "protocol": {
                "trial_schedule": list(_trial_schedule(args.trials)),
                "warmup_schedule": ["off", "on"] * args.warmup_rounds,
                "same_input_for_all_executions": True,
                "same_source_and_revision_for_both_modes": True,
                "expert_execution_order": "ascending_official_modulelist_order",
                "accumulation": "serial_fp32_add_then_cast_to_compute_dtype",
                "route_weight_dtype": "float32",
                "output_gate": "finite_and_torch_equal_and_bit_equal",
                "timing_boundary": "device_synchronize_around_expert_execution",
            },
            "exactness": {
                "all_outputs_finite": all(row["finite"] for row in trials),
                "all_outputs_torch_equal": all(row["torch_equal"] for row in trials),
                "all_outputs_bit_equal": all(row["bit_equal"] for row in trials),
                "reference_output_sha256": _tensor_digest(reference),
            },
            "provenance": {
                "source": source_label,
                "revision": args.revision,
                "revision_is_pinned": bool(
                    source_metrics_after.get("revision_is_pinned")
                ),
                "revision_is_mutable": bool(
                    source_metrics_after.get("revision_is_mutable")
                ),
                "inventory_source_fingerprint": source_metrics_after.get(
                    "inventory_source_fingerprint"
                ),
                "config": config_meta,
                "expert_tensor_manifest": expert_manifest,
                "expert_tensor_manifest_sha256": expert_manifest_sha,
                "expert_payload_bytes": expert_payload_bytes,
                "runtime_sources": runtime_sources,
                "runtime_source_sha256": _canonical_digest(runtime_sources),
                "runtime_dependencies": runtime_dependencies,
                "runtime_dependency_sha256": _canonical_digest(runtime_dependencies),
                "harness_sha256": harness_sha256,
                "input_sha256": input_sha256,
                "hardware": {
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                    "python": platform.python_version(),
                    "torch": str(torch.__version__),
                },
                "execution": {
                    "device": str(off.device),
                    "compute_dtype": str(off.compute_dtype).removeprefix("torch."),
                    "activation_quantization": bool(
                        off.simulate_activation_quantization
                    ),
                    "quantized_accumulation_policy": (
                        DeepSeekWeightPager.QUANTIZED_ACCUMULATION_POLICY
                    ),
                    "expert_prefetch_policy": (
                        DeepSeekWeightPager.EXPERT_PREFETCH_POLICY
                    ),
                    "expert_prefetch_payload_limit_bytes": (
                        DeepSeekWeightPager.EXPERT_PREFETCH_PAYLOAD_LIMIT_BYTES
                    ),
                    "expert_prefetch_transport_policy": (
                        DeepSeekWeightPager.EXPERT_PREFETCH_TRANSPORT_POLICY
                    ),
                    "expert_prefetch_workers": (
                        DeepSeekWeightPager.EXPERT_PREFETCH_WORKERS
                    ),
                    "expert_prefetch_active_read_limit": (
                        DeepSeekWeightPager.EXPERT_PREFETCH_ACTIVE_READ_LIMIT
                    ),
                    "expert_prefetch_max_outstanding": (
                        DeepSeekWeightPager.EXPERT_PREFETCH_MAX_OUTSTANDING
                    ),
                    "expert_prefetch_max_experts": (
                        DeepSeekWeightPager.EXPERT_PREFETCH_MAX_EXPERTS
                    ),
                    "expert_prefetch_resident_limit_bytes": (
                        DeepSeekWeightPager.EXPERT_PREFETCH_RESIDENT_LIMIT_BYTES
                    ),
                    "source_transport_policy": str(
                        source_metrics_after.get("transport_policy", "unreported")
                    ),
                    "source_transport_connection_limit": int(
                        source_metrics_after.get("transport_connection_limit", 0)
                    ),
                    "expected_prefetch_peak_bytes": expected_prefetch_peak,
                    "requested_experts": list(requested),
                    "executed_experts": list(experts),
                    "expert_bases": list(bases),
                    "input_shape": list(hidden.shape),
                },
            },
            "warmup": {
                "rounds": args.warmup_rounds,
                "source_body_bytes_delta": body_after_warmup - body_before_warmup,
                "executions": warmup_records,
            },
            "trials": trials,
            "summary": summary,
            "metrics": {
                "pager": {"off": off_metrics, "on": on_metrics},
                "source": {
                    "before": source_metrics_before,
                    "after": source_metrics_after,
                    "body_bytes_before_warmup": body_before_warmup,
                    "body_bytes_after_warmup": body_after_warmup,
                    "body_bytes_after_trials": body_after_trials,
                    "warmup_body_bytes_delta": (body_after_warmup - body_before_warmup),
                    "measured_body_bytes_delta": (
                        body_after_trials - body_after_warmup
                    ),
                },
            },
        }
    )
    _atomic_write_json(Path(args.output), report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run(args)
    except Exception as exc:
        print(
            f"deepseek-v4-prefetch-smoke: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    receipt = {
        "schema": RESULT_SCHEMA,
        "status": report["status"],
        "output": str(Path(args.output).expanduser().resolve()),
        "report_sha256": report["report_sha256"],
        "bit_equal": report["exactness"]["all_outputs_bit_equal"],
        "mean_speedup_off_over_on": report["summary"]["mean_speedup_off_over_on"],
    }
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
