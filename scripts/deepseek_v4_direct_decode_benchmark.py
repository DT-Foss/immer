#!/usr/bin/env python3
"""Benchmark one contextual DeepSeek-V4 decode step from a shared KV prefix.

The prefix is computed once and saved with a transport-neutral identity that
still binds every model-value determinant. Baseline, real-Markov, and shuffled
placebo arms restore that exact state, execute one native decode token, and
must produce bit-identical hidden values and official router choices.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

from immer.knowledge import AccessTraceRecorder, Streamer
from immer.runtimes.deepseek_v4 import (
    DeepSeekV4Config,
    DeepSeekWeightPager,
    StreamedDeepSeekV4,
)
from immer.runtimes.deepseek_v4.causal_prefetch import CheckpointIdentity
from immer.runtimes.deepseek_v4.route_model import (
    RouteModelArtifactError,
    load_route_model_artifact,
)

ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_SOURCE = "deepseek-ai/DeepSeek-V4-Flash-0731"
OFFICIAL_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
DEFAULT_CACHE = ROOT / "artifacts" / "private" / "deepseek-v4-cache"
INPUT_SCHEMA = "immer.deepseek-v4-direct-decode-input/v1"
RESULT_SCHEMA = "immer.deepseek-v4-direct-decode/v1"
COMPARISON_SCHEMA = "immer.deepseek-v4-direct-decode-comparison/v1"
REPLICATED_COMPARISON_SCHEMA = (
    "immer.deepseek-v4-direct-decode-replicated-comparison/v1"
)


class DirectDecodeError(RuntimeError):
    """A shared-prefix decode benchmark contract was violated."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DirectDecodeError("value is not canonical JSON") from exc


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _atomic_bytes(path: str | os.PathLike[str], encoded: bytes) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json(path: str | os.PathLike[str], document: Mapping[str, Any]) -> None:
    _atomic_bytes(path, _canonical(dict(document)))


def _strict_json(path: str | os.PathLike[str]) -> Any:
    source = Path(path).expanduser().resolve()

    def pairs(entries: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in entries:
            if key in result:
                raise DirectDecodeError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise DirectDecodeError(f"non-finite JSON value: {value}")

    try:
        return json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=invalid_constant,
        )
    except DirectDecodeError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DirectDecodeError(f"cannot read JSON: {source}") from exc


def build_input_document(
    prefix_token_ids: Sequence[int], decode_token_id: int
) -> dict[str, Any]:
    prefix = tuple(prefix_token_ids)
    if not prefix:
        raise DirectDecodeError("prefix token IDs must not be empty")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in prefix
    ):
        raise DirectDecodeError("prefix token IDs must be non-negative integers")
    if (
        isinstance(decode_token_id, bool)
        or not isinstance(decode_token_id, int)
        or decode_token_id < 0
    ):
        raise DirectDecodeError("decode token ID must be a non-negative integer")
    identity = {
        "decode_token_id": decode_token_id,
        "prefix_token_ids": list(prefix),
        "schema": INPUT_SCHEMA,
    }
    return {**identity, "sha256": _sha256(identity)}


def load_input_document(path: str | os.PathLike[str]) -> dict[str, Any]:
    document = _strict_json(path)
    if not isinstance(document, dict) or set(document) != {
        "decode_token_id",
        "prefix_token_ids",
        "schema",
        "sha256",
    }:
        raise DirectDecodeError("decode input has unknown or missing fields")
    expected = build_input_document(
        document["prefix_token_ids"], document["decode_token_id"]
    )
    if document != expected:
        raise DirectDecodeError("decode input digest does not match")
    return document


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


def _build_source(
    args: argparse.Namespace,
    recorder: AccessTraceRecorder | None,
) -> Streamer:
    common = {
        "revision": args.revision,
        "budget_mb": args.budget_mb,
        "cache_dir": Path(args.cache_dir).expanduser().resolve(),
        "use_cache": not args.no_cache,
        "max_cache_bytes": int(args.max_cache_gb * 1024**3),
        "verbose": False,
        "access_observer": recorder,
    }
    local = _local_source_path(args.source)
    if local is not None:
        if not local.is_dir():
            raise DirectDecodeError(f"local source directory does not exist: {local}")
        return Streamer.from_local(local, **common)
    return Streamer(args.source, **common)


def _load_config(source: Streamer) -> DeepSeekV4Config:
    try:
        raw = source.reader.fetch_file("config.json")
        document = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise DirectDecodeError("cannot load DeepSeek-V4 config") from exc
    if not isinstance(document, Mapping):
        raise DirectDecodeError("DeepSeek-V4 config root must be an object")
    return DeepSeekV4Config.from_mapping(document)


def _checkpoint(args: argparse.Namespace, source: Streamer) -> CheckpointIdentity:
    source.inventory()
    fingerprint = source.metrics().get("inventory_source_fingerprint")
    if not isinstance(fingerprint, str):
        raise DirectDecodeError("source inventory fingerprint is unavailable")
    try:
        return CheckpointIdentity(
            repo_id=args.logical_repo_id,
            revision=args.revision,
            inventory_fingerprint=fingerprint,
        )
    except ValueError as exc:
        raise DirectDecodeError("source checkpoint identity is invalid") from exc


def _model(
    args: argparse.Namespace,
    source: Streamer,
    config: DeepSeekV4Config,
    *,
    route_predictor: Any | None,
) -> StreamedDeepSeekV4:
    pager = DeepSeekWeightPager(
        source,
        device=args.device,
        compute_dtype=args.dtype,
        simulate_activation_quantization=not args.no_activation_quantization,
        expert_prefetch=not args.no_expert_prefetch,
        expert_reservoir_budget_bytes=args.expert_reservoir_budget_mb * 1024**2,
        expert_reservoir_workers=args.expert_reservoir_workers,
    )
    kwargs: dict[str, Any] = {}
    if route_predictor is not None:
        kwargs = {
            "route_predictor": route_predictor,
            "route_prefetch_alpha": args.route_prefetch_alpha,
            "route_prefetch_direct_max_rows": args.route_prefetch_direct_max_rows,
            "route_prefetch_k": args.route_prefetch_k,
            "route_prefetch_min_confidence": args.route_prefetch_min_confidence,
            "route_prefetch_window_rows": args.route_prefetch_window_rows,
        }
    return StreamedDeepSeekV4(
        config,
        pager,
        max_batch_size=1,
        max_seq_len=args.max_seq_len,
        **kwargs,
    )


def _tensor_sha256(value: Any) -> str:
    import torch

    tensor = value.detach().to(device="cpu").contiguous()
    raw = tensor.view(torch.uint8).numpy().reshape(-1).tobytes()
    return hashlib.sha256(raw).hexdigest()


def _source_bytes(source: Streamer) -> int:
    value = source.metrics().get("network_or_source_body_bytes", 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise DirectDecodeError("source byte counter is invalid")
    return int(value)


def _trace_receipt(
    recorder: AccessTraceRecorder | None,
    path: str | None,
) -> dict[str, Any]:
    if recorder is None:
        return {"enabled": False}
    metrics = recorder.metrics()
    if metrics["dropped_capacity"] or metrics["dropped_identity"]:
        raise DirectDecodeError("access trace dropped operations")
    trace = recorder.snapshot()
    trace.verify()
    if path is not None:
        _atomic_bytes(path, trace.to_bytes())
    return {
        "enabled": True,
        "leaves": metrics["leaves"],
        "operations": metrics["operations"],
        "path": None if path is None else str(Path(path).expanduser().resolve()),
        "sha256": trace.sha256,
    }


def _result(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(identity), "sha256": _sha256(identity)}


def _progress(arm: str, sink: list[dict[str, Any]] | None = None):
    def emit(row: Mapping[str, Any]) -> None:
        experts = row.get("experts")
        flattened = (
            [expert for selected in experts for expert in selected]
            if isinstance(experts, list)
            else []
        )
        record = {
            "arm": arm,
            "event": "layer_complete",
            "layer": row.get("layer"),
            "layers": row.get("layers"),
            "seconds": row.get("seconds"),
            "source_body_bytes": row.get("source_body_bytes"),
            "unique_experts": len(set(flattened)),
        }
        if sink is not None:
            sink.append(record)
        print(json.dumps(record, sort_keys=True), file=sys.stderr, flush=True)

    return emit


def prepare_prefix(args: argparse.Namespace) -> dict[str, Any]:
    inputs = load_input_document(args.input)
    prefix = tuple(int(value) for value in inputs["prefix_token_ids"])
    if len(prefix) >= args.max_seq_len:
        raise DirectDecodeError("prefix leaves no room for the decode token")
    recorder = None if args.no_access_trace else AccessTraceRecorder()
    source = _build_source(args, recorder)
    model: StreamedDeepSeekV4 | None = None
    try:
        config = _load_config(source)
        if max((*prefix, int(inputs["decode_token_id"]))) >= config.vocab_size:
            raise DirectDecodeError("decode input exceeds checkpoint vocabulary")
        model = _model(args, source, config, route_predictor=None)
        before = _source_bytes(source)
        started = time.perf_counter()
        layer_receipts: list[dict[str, Any]] = []
        scope = (
            recorder.scope(phase="shared_prefix", arm="prepare")
            if recorder is not None
            else nullcontext()
        )
        with scope:
            hidden, evidence = model.prefill(
                [prefix],
                tokenwise=False,
                progress=_progress("prepare", layer_receipts),
            )
        seconds = time.perf_counter() - started
        after = _source_bytes(source)
        snapshot = model.save_state(args.snapshot, transport_neutral=True)
        trace = _trace_receipt(recorder, args.access_trace)
        identity = {
            "arm": "prepare",
            "checkpoint": _checkpoint(args, source).as_record(),
            "evidence": asdict(evidence[0]),
            "hidden_sha256": _tensor_sha256(hidden),
            "input_sha256": inputs["sha256"],
            "layers": layer_receipts,
            "pager": model.pager.metrics(),
            "schema": RESULT_SCHEMA,
            "seconds": seconds,
            "snapshot": {
                **snapshot,
                "manifest": str(Path(args.snapshot).expanduser().resolve()),
            },
            "source_body_bytes": after - before,
            "trace": trace,
        }
        return _result(identity)
    finally:
        if model is not None:
            model.reset_state(release=True)
            model.pager.close()
        source.close()


def decode_arm(args: argparse.Namespace) -> dict[str, Any]:
    inputs = load_input_document(args.input)
    prefix = tuple(int(value) for value in inputs["prefix_token_ids"])
    if len(prefix) >= args.max_seq_len:
        raise DirectDecodeError("prefix leaves no room for the decode token")
    recorder = None if args.no_access_trace else AccessTraceRecorder()
    source = _build_source(args, recorder)
    model: StreamedDeepSeekV4 | None = None
    try:
        config = _load_config(source)
        checkpoint = _checkpoint(args, source)
        predictor = None
        artifact_sha256 = None
        role = "baseline"
        if args.route_model is not None:
            try:
                artifact = load_route_model_artifact(
                    args.route_model,
                    expected_checkpoint=checkpoint,
                    expected_role=args.route_model_role,
                )
            except (OSError, RouteModelArtifactError) as exc:
                raise DirectDecodeError("cannot load route model") from exc
            predictor = artifact.predictor
            artifact_sha256 = artifact.sha256
            role = args.route_model_role
        model = _model(args, source, config, route_predictor=predictor)
        restored = model.load_state(args.snapshot, transport_neutral=True)
        if model.next_position != len(prefix):
            raise DirectDecodeError("shared prefix cursor does not match decode input")
        before = _source_bytes(source)
        started = time.perf_counter()
        layer_receipts: list[dict[str, Any]] = []
        scope = (
            recorder.scope(phase="decode", arm=role)
            if recorder is not None
            else nullcontext()
        )
        with scope:
            hidden, evidence = model.decode(
                [[int(inputs["decode_token_id"])]],
                progress=_progress(role, layer_receipts),
            )
        seconds = time.perf_counter() - started
        after = _source_bytes(source)
        trace = _trace_receipt(recorder, args.access_trace)
        identity = {
            "arm": role,
            "checkpoint": checkpoint.as_record(),
            "evidence": asdict(evidence),
            "hidden_sha256": _tensor_sha256(hidden),
            "input_sha256": inputs["sha256"],
            "layers": layer_receipts,
            "pager": model.pager.metrics(),
            "route_model_artifact_sha256": artifact_sha256,
            "route_prefetch": model.route_prefetch_metrics(),
            "schema": RESULT_SCHEMA,
            "seconds": seconds,
            "snapshot": {
                "manifest": str(Path(args.snapshot).expanduser().resolve()),
                "restore": restored,
            },
            "source_body_bytes": after - before,
            "trace": trace,
        }
        return _result(identity)
    finally:
        if model is not None:
            model.reset_state(release=True)
            model.pager.close()
        source.close()


def _load_result(path: str | os.PathLike[str], expected_arm: str) -> dict[str, Any]:
    document = _strict_json(path)
    if not isinstance(document, dict) or document.get("schema") != RESULT_SCHEMA:
        raise DirectDecodeError("decode result schema is invalid")
    identity = {key: value for key, value in document.items() if key != "sha256"}
    if document.get("sha256") != _sha256(identity):
        raise DirectDecodeError("decode result digest does not match")
    if document.get("arm") != expected_arm:
        raise DirectDecodeError("decode result arm does not match")
    return document


def compare_arms(args: argparse.Namespace) -> dict[str, Any]:
    documents = {
        "baseline": _load_result(args.baseline, "baseline"),
        "real_markov": _load_result(args.real, "real_markov"),
        "placebo_markov": _load_result(args.placebo, "placebo_markov"),
    }

    def invariant(document: Mapping[str, Any]) -> dict[str, Any]:
        evidence = document.get("evidence")
        if not isinstance(evidence, Mapping):
            raise DirectDecodeError("decode result lacks evidence")
        return {
            "checkpoint": document.get("checkpoint"),
            "evidence": {
                key: value
                for key, value in evidence.items()
                if key not in {"seconds", "source_body_bytes"}
            },
            "hidden_sha256": document.get("hidden_sha256"),
            "input_sha256": document.get("input_sha256"),
        }

    expected = invariant(documents["baseline"])
    for name in ("real_markov", "placebo_markov"):
        if invariant(documents[name]) != expected:
            raise DirectDecodeError(f"{name} changed hidden values or router choices")
    real_config = documents["real_markov"].get("route_prefetch")
    placebo_config = documents["placebo_markov"].get("route_prefetch")
    if not isinstance(real_config, Mapping) or not isinstance(placebo_config, Mapping):
        raise DirectDecodeError("predictor arm lacks route-prefetch metrics")
    config_keys = {
        "alpha",
        "direct_max_rows",
        "k",
        "min_confidence",
        "window_rows",
    }
    if {key: real_config.get(key) for key in config_keys} != {
        key: placebo_config.get(key) for key in config_keys
    }:
        raise DirectDecodeError("real and placebo route settings differ")

    arms: dict[str, Any] = {}
    for name, document in documents.items():
        pager = document.get("pager")
        if not isinstance(pager, Mapping):
            raise DirectDecodeError("decode result lacks pager metrics")
        route = document.get("route_prefetch")
        if not isinstance(route, Mapping):
            raise DirectDecodeError("decode result lacks route scheduler metrics")
        raw_layers = document.get("layers", [])
        if not isinstance(raw_layers, list):
            raise DirectDecodeError("decode layer receipts must be a list")
        if raw_layers:
            layer_ids = [
                row.get("layer") for row in raw_layers if isinstance(row, Mapping)
            ]
            if layer_ids != list(range(len(raw_layers))):
                raise DirectDecodeError(
                    "decode layer receipts are incomplete or unordered"
                )
            causal_rows = [row for row in raw_layers if int(row["layer"]) >= 22]
            causal_seconds: float | None = sum(
                float(row["seconds"]) for row in causal_rows
            )
            causal_bytes: int | None = sum(
                int(row["source_body_bytes"]) for row in causal_rows
            )
        else:
            causal_seconds = None
            causal_bytes = None
        arms[name] = {
            "causal_layer_seconds": causal_seconds,
            "causal_layer_source_body_bytes": causal_bytes,
            "confidence_skips": int(route.get("confidence_skips", 0)),
            "direct_bindings": int(route.get("direct_bindings", 0)),
            "expert_prefetch_wait_ns": int(pager.get("expert_prefetch_wait_ns", 0)),
            "reservoir_failures": int(pager.get("expert_reservoir_failures", 0)),
            "reservoir_hit_payload_bytes": int(
                pager.get("expert_reservoir_hit_payload_bytes", 0)
            ),
            "reservoir_hits": int(pager.get("expert_reservoir_usable_hits", 0)),
            "reservoir_misses": int(pager.get("expert_reservoir_misses", 0)),
            "reservoir_payload_bytes": int(
                pager.get("expert_reservoir_payload_bytes", 0)
            ),
            "reservoir_ready_hits": int(pager.get("expert_reservoir_ready_hits", 0)),
            "reservoir_source_bytes": int(
                pager.get("expert_reservoir_source_bytes", 0)
            ),
            "reservoir_submitted": int(pager.get("expert_reservoir_submitted", 0)),
            "reservoir_wait_ns": int(pager.get("expert_reservoir_wait_ns", 0)),
            "reservoir_wasted": int(pager.get("expert_reservoir_wasted", 0)),
            "reservoir_wasted_payload_bytes": int(
                pager.get("expert_reservoir_wasted_payload_bytes", 0)
            ),
            "seconds": float(document["seconds"]),
            "source_body_bytes": int(document["source_body_bytes"]),
            "trace_sha256": document.get("trace", {}).get("sha256"),
        }
        submitted = arms[name]["reservoir_submitted"]
        hits = arms[name]["reservoir_hits"]
        misses = arms[name]["reservoir_misses"]
        arms[name]["reservoir_precision"] = hits / submitted if submitted else None
        arms[name]["reservoir_demand_coverage"] = (
            hits / (hits + misses) if hits + misses else None
        )

    baseline = arms["baseline"]

    def contrast(arm: Mapping[str, Any]) -> dict[str, Any]:
        causal_seconds = arm["causal_layer_seconds"]
        baseline_causal_seconds = baseline["causal_layer_seconds"]
        causal_bytes = arm["causal_layer_source_body_bytes"]
        baseline_causal_bytes = baseline["causal_layer_source_body_bytes"]
        return {
            "causal_layer_seconds_delta": (
                None
                if causal_seconds is None or baseline_causal_seconds is None
                else float(causal_seconds) - float(baseline_causal_seconds)
            ),
            "causal_layer_seconds_ratio": (
                None
                if causal_seconds is None or baseline_causal_seconds in (None, 0)
                else float(causal_seconds) / float(baseline_causal_seconds)
            ),
            "causal_layer_source_body_bytes_delta": (
                None
                if causal_bytes is None or baseline_causal_bytes is None
                else int(causal_bytes) - int(baseline_causal_bytes)
            ),
            "seconds_delta": float(arm["seconds"]) - float(baseline["seconds"]),
            "seconds_ratio": (
                float(arm["seconds"]) / float(baseline["seconds"])
                if float(baseline["seconds"])
                else None
            ),
            "source_body_bytes_delta": int(arm["source_body_bytes"])
            - int(baseline["source_body_bytes"]),
            "source_body_bytes_ratio": (
                int(arm["source_body_bytes"]) / int(baseline["source_body_bytes"])
                if int(baseline["source_body_bytes"])
                else None
            ),
        }

    identity = {
        "arms": arms,
        "contrasts": {
            "placebo_vs_baseline": contrast(arms["placebo_markov"]),
            "real_vs_baseline": contrast(arms["real_markov"]),
            "real_vs_placebo": {
                "causal_layer_seconds_delta": (
                    None
                    if arms["real_markov"]["causal_layer_seconds"] is None
                    or arms["placebo_markov"]["causal_layer_seconds"] is None
                    else arms["real_markov"]["causal_layer_seconds"]
                    - arms["placebo_markov"]["causal_layer_seconds"]
                ),
                "causal_layer_seconds_ratio": (
                    None
                    if arms["real_markov"]["causal_layer_seconds"] is None
                    or not arms["placebo_markov"]["causal_layer_seconds"]
                    else arms["real_markov"]["causal_layer_seconds"]
                    / arms["placebo_markov"]["causal_layer_seconds"]
                ),
                "seconds_delta": arms["real_markov"]["seconds"]
                - arms["placebo_markov"]["seconds"],
                "seconds_ratio": (
                    arms["real_markov"]["seconds"] / arms["placebo_markov"]["seconds"]
                    if arms["placebo_markov"]["seconds"]
                    else None
                ),
                "source_body_bytes_delta": arms["real_markov"]["source_body_bytes"]
                - arms["placebo_markov"]["source_body_bytes"],
            },
        },
        "invariant_sha256": _sha256(expected),
        "schema": COMPARISON_SCHEMA,
    }
    return _result(identity)


def compare_replicates(args: argparse.Namespace) -> dict[str, Any]:
    paths = (tuple(args.baseline), tuple(args.real), tuple(args.placebo))
    if len({len(values) for values in paths}) != 1 or len(paths[0]) < 2:
        raise DirectDecodeError(
            "replicated comparison requires two or more matched three-arm cycles"
        )
    cycles = [
        compare_arms(argparse.Namespace(baseline=baseline, real=real, placebo=placebo))
        for baseline, real, placebo in zip(*paths, strict=True)
    ]
    invariant_sha256 = {cycle["invariant_sha256"] for cycle in cycles}
    if len(invariant_sha256) != 1:
        raise DirectDecodeError("replicated cycles do not share output identity")

    arms: dict[str, Any] = {}
    for arm in ("baseline", "real_markov", "placebo_markov"):
        rows = [cycle["arms"][arm] for cycle in cycles]
        stable_metrics = {
            key: {row[key] for row in rows}
            for key in (
                "confidence_skips",
                "direct_bindings",
                "reservoir_hits",
                "reservoir_misses",
                "reservoir_submitted",
                "reservoir_wasted",
            )
        }
        if any(len(values) != 1 for values in stable_metrics.values()):
            raise DirectDecodeError(f"{arm} route counters changed across cycles")
        causal_seconds = [
            row["causal_layer_seconds"]
            for row in rows
            if row["causal_layer_seconds"] is not None
        ]
        arms[arm] = {
            "causal_layer_seconds": causal_seconds,
            "causal_layer_seconds_mean": (
                sum(causal_seconds) / len(causal_seconds) if causal_seconds else None
            ),
            "route_counters": {
                key: next(iter(values)) for key, values in stable_metrics.items()
            },
            "seconds": [row["seconds"] for row in rows],
            "seconds_mean": sum(row["seconds"] for row in rows) / len(rows),
            "source_body_bytes": [row["source_body_bytes"] for row in rows],
            "source_body_bytes_mean": sum(row["source_body_bytes"] for row in rows)
            / len(rows),
        }

    paired: dict[str, Any] = {}
    for name, left, right in (
        ("real_vs_baseline", "real_markov", "baseline"),
        ("placebo_vs_baseline", "placebo_markov", "baseline"),
        ("real_vs_placebo", "real_markov", "placebo_markov"),
    ):
        deltas = [
            cycles[index]["arms"][left]["seconds"]
            - cycles[index]["arms"][right]["seconds"]
            for index in range(len(cycles))
        ]
        paired[name] = {
            "seconds_deltas": deltas,
            "seconds_mean_delta": sum(deltas) / len(deltas),
            "seconds_mean_ratio": (
                arms[left]["seconds_mean"] / arms[right]["seconds_mean"]
                if arms[right]["seconds_mean"]
                else None
            ),
        }

    identity = {
        "arms": arms,
        "cycle_count": len(cycles),
        "cycles": [
            {
                "contrasts": cycle["contrasts"],
                "sha256": cycle["sha256"],
            }
            for cycle in cycles
        ],
        "invariant_sha256": next(iter(invariant_sha256)),
        "paired": paired,
        "schema": REPLICATED_COMPARISON_SCHEMA,
    }
    return _result(identity)


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _positive_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def _unit_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("must be inside [0, 1]")
    return value


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--access-trace")
    parser.add_argument("--no-access-trace", action="store_true")
    parser.add_argument("--source", default=OFFICIAL_SOURCE)
    parser.add_argument("--logical-repo-id", default=OFFICIAL_SOURCE)
    parser.add_argument("--revision", default=OFFICIAL_REVISION)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--max-cache-gb", type=_positive_float, default=12.0)
    parser.add_argument("--budget-mb", type=_positive_int, default=196608)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="mps")
    parser.add_argument("--dtype", choices=("auto", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-seq-len", type=_positive_int, default=64)
    parser.add_argument("--no-activation-quantization", action="store_true")
    parser.add_argument("--no-expert-prefetch", action="store_true")
    parser.add_argument("--expert-reservoir-budget-mb", type=_positive_int, default=64)
    parser.add_argument("--expert-reservoir-workers", type=_positive_int, default=2)
    parser.add_argument("--route-prefetch-window-rows", type=_positive_int, default=2)
    parser.add_argument("--route-prefetch-k", type=_positive_int, default=3)
    parser.add_argument("--route-prefetch-alpha", type=_positive_float, default=1.0)
    parser.add_argument(
        "--route-prefetch-direct-max-rows", type=_positive_int, default=8
    )
    parser.add_argument(
        "--route-prefetch-min-confidence", type=_unit_float, default=0.125
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-prefix")
    _add_runtime_arguments(prepare)
    prepare.set_defaults(handler=prepare_prefix)

    decode = subparsers.add_parser("decode-arm")
    _add_runtime_arguments(decode)
    decode.add_argument("--route-model")
    decode.add_argument("--route-model-role", choices=("real_markov", "placebo_markov"))
    decode.set_defaults(handler=decode_arm)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--real", required=True)
    compare.add_argument("--placebo", required=True)
    compare.add_argument("--output", required=True)
    compare.set_defaults(handler=compare_arms)

    replicated = subparsers.add_parser("compare-replicates")
    replicated.add_argument("--baseline", action="append", required=True)
    replicated.add_argument("--real", action="append", required=True)
    replicated.add_argument("--placebo", action="append", required=True)
    replicated.add_argument("--output", required=True)
    replicated.set_defaults(handler=compare_replicates)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "decode-arm":
        if (args.route_model is None) != (args.route_model_role is None):
            raise SystemExit("decode-arm requires route model and role together")
    try:
        document = args.handler(args)
        _write_json(args.output, document)
    except DirectDecodeError as exc:
        raise SystemExit(f"direct decode benchmark failed: {exc}") from exc
    print(
        json.dumps(
            {
                "arm": document.get("arm"),
                "output": str(Path(args.output).expanduser().resolve()),
                "seconds": document.get("seconds"),
                "sha256": document["sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
