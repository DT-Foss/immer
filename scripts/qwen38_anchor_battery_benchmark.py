#!/usr/bin/env python3
"""Charge and benchmark exact semantic-state compute batteries on local Qwen3.8.

The sealed input contains token IDs only.  ``charge`` performs the expensive
prefix prefill outside the demand path and commits a native continuation
snapshot.  ``compare`` keeps one authenticated model open while executing a
sealed AB/BA (or balanced ABBA/BAAB) schedule.  Every arm starts from released
state, and output is committed only when final hidden tensors and serialized
native state payloads are bit-identical.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import statistics
import sys
import tempfile
import time
from types import ModuleType
from typing import Any

import torch

from immer.runtimes.qwen3_8 import (
    LogicalModelIdentity,
    OFFICIAL_REPO_ID,
    OFFICIAL_REVISION,
)
from immer.runtimes.qwen3_8.semantic_state_cache import (
    AnchorReceipt,
    SemanticStateAnchorCache,
)
from immer.runtimes.qwen3_8.snapshot import QWEN38_SNAPSHOT_SCHEMA


ROOT = Path(__file__).resolve().parent.parent
BASE_SCRIPT = ROOT / "scripts" / "qwen35_live_k2_smoke.py"
DEFAULT_BUNDLE = Path("/app/models/Qwen3.8-27B")
DEFAULT_CACHE = ROOT / "artifacts" / "private" / "qwen3.8-anchor-battery" / "cache"
INPUT_SCHEMA = "immer.qwen3.8-anchor-battery-benchmark-input/v1"
CHARGE_SCHEMA = "immer.qwen3.8-anchor-battery-charge/v1"
RESULT_SCHEMA = "immer.qwen3.8-anchor-battery-benchmark/v1"
MAX_INPUT_BYTES = 1024 * 1024
MIB = 1024**2
_SCHEDULES = {
    "AB": ("baseline", "battery"),
    "BA": ("battery", "baseline"),
    "ABBA": ("baseline", "battery", "battery", "baseline"),
    "BAAB": ("battery", "baseline", "baseline", "battery"),
}


class AnchorBatteryBenchmarkError(RuntimeError):
    """The sealed compute-battery experiment cannot be completed exactly."""


def _load_script(name: str, path: Path) -> ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AnchorBatteryBenchmarkError(f"cannot import {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = _load_script("qwen35_live_k2_smoke", BASE_SCRIPT)


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
        raise AnchorBatteryBenchmarkError("document is not canonical JSON") from exc


def _canonical_ascii(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AnchorBatteryBenchmarkError("document is not canonical JSON") from exc


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    document = dict(value)
    if "sha256" in document:
        raise AnchorBatteryBenchmarkError("document already contains a seal")
    document["sha256"] = _sha256(document)
    return document


def _tokens(value: object, label: str, *, allow_empty: bool) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise AnchorBatteryBenchmarkError(f"{label} must be a JSON token-ID list")
    if not allow_empty and not value:
        raise AnchorBatteryBenchmarkError(f"{label} must not be empty")
    if len(value) > 1_048_576:
        raise AnchorBatteryBenchmarkError(f"{label} exceeds its token bound")
    result: list[int] = []
    for token in value:
        if (
            isinstance(token, bool)
            or not isinstance(token, int)
            or token < 0
            or token > 2**63 - 1
        ):
            raise AnchorBatteryBenchmarkError(f"{label} contains an invalid token ID")
        result.append(token)
    return tuple(result)


def _validate_input(value: object) -> dict[str, Any]:
    required = {
        "attention_mode",
        "boundary_kind",
        "prefix_token_ids",
        "schedule",
        "schema",
        "sha256",
        "suffix_token_ids",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise AnchorBatteryBenchmarkError("benchmark input schema is invalid")
    document = dict(value)
    seal = document.pop("sha256")
    if document.get("schema") != INPUT_SCHEMA or seal != _sha256(document):
        raise AnchorBatteryBenchmarkError("benchmark input seal is invalid")
    if document.get("attention_mode") not in {"off", "native-crsa"}:
        raise AnchorBatteryBenchmarkError("attention mode is invalid")
    if document.get("boundary_kind") not in {
        "turn",
        "tool-call",
        "tool-output",
        "thinking",
        "custom",
    }:
        raise AnchorBatteryBenchmarkError("boundary kind is invalid")
    schedule = document.get("schedule")
    if not isinstance(schedule, list) or tuple(schedule) not in _SCHEDULES.values():
        raise AnchorBatteryBenchmarkError("schedule must be AB, BA, ABBA, or BAAB")
    prefix = _tokens(document.get("prefix_token_ids"), "prefix", allow_empty=False)
    suffix = _tokens(document.get("suffix_token_ids"), "suffix", allow_empty=True)
    if len(prefix) + len(suffix) > 1_048_576:
        raise AnchorBatteryBenchmarkError("combined context exceeds its token bound")
    return {**document, "sha256": seal}


def _absolute_without_symlink_resolution(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return Path(os.path.abspath(os.fspath(expanded)))


def _signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_regular_bytes(path: Path, label: str, *, max_bytes: int) -> bytes:
    target = _absolute_without_symlink_resolution(path)
    try:
        linked_before = os.lstat(target)
    except OSError as exc:
        raise AnchorBatteryBenchmarkError(f"cannot inspect {label}: {target}") from exc
    if stat.S_ISLNK(linked_before.st_mode) or not stat.S_ISREG(linked_before.st_mode):
        raise AnchorBatteryBenchmarkError(f"{label} is not a non-symlink regular file")
    flags = os.O_RDONLY | int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(target, flags)
    except OSError as exc:
        raise AnchorBatteryBenchmarkError(f"cannot open {label}: {target}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or (before.st_dev, before.st_ino)
            != (linked_before.st_dev, linked_before.st_ino)
            or before.st_size > max_bytes
        ):
            raise AnchorBatteryBenchmarkError(f"{label} is not a bounded regular file")
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, min(1024**2, max_bytes + 1 - total)):
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise AnchorBatteryBenchmarkError(f"{label} exceeds its byte bound")
        after = os.fstat(descriptor)
        linked_after = os.lstat(target)
        if _signature(before) != _signature(after) or _signature(after) != _signature(
            linked_after
        ):
            raise AnchorBatteryBenchmarkError(f"{label} changed while reading")
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _stable_regular_sha256(
    path: Path,
    label: str,
    *,
    expected_bytes: int,
    max_bytes: int,
) -> str:
    target = _absolute_without_symlink_resolution(path)
    try:
        linked_before = os.lstat(target)
    except OSError as exc:
        raise AnchorBatteryBenchmarkError(f"cannot inspect {label}: {target}") from exc
    if stat.S_ISLNK(linked_before.st_mode) or not stat.S_ISREG(linked_before.st_mode):
        raise AnchorBatteryBenchmarkError(f"{label} is not a non-symlink regular file")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes <= 0
        or expected_bytes > max_bytes
    ):
        raise AnchorBatteryBenchmarkError(f"{label} byte count is outside its bound")
    flags = os.O_RDONLY | int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(target, flags)
    except OSError as exc:
        raise AnchorBatteryBenchmarkError(f"cannot open {label}: {target}") from exc
    digest = hashlib.sha256()
    total = 0
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or (before.st_dev, before.st_ino)
            != (linked_before.st_dev, linked_before.st_ino)
            or before.st_size != expected_bytes
        ):
            raise AnchorBatteryBenchmarkError(f"{label} byte count or identity changed")
        while chunk := os.read(descriptor, 1024**2):
            digest.update(chunk)
            total += len(chunk)
            if total > expected_bytes:
                raise AnchorBatteryBenchmarkError(
                    f"{label} exceeds its sealed byte count"
                )
        after = os.fstat(descriptor)
        linked_after = os.lstat(target)
        if (
            total != expected_bytes
            or _signature(before) != _signature(after)
            or _signature(after) != _signature(linked_after)
        ):
            raise AnchorBatteryBenchmarkError(f"{label} changed while hashing")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _json_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AnchorBatteryBenchmarkError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _stable_json(
    path: Path,
    label: str,
    *,
    max_bytes: int = MAX_INPUT_BYTES,
    require_canonical: bool = False,
    ascii_canonical: bool = False,
) -> dict[str, Any]:
    raw = _stable_regular_bytes(path, label, max_bytes=max_bytes)
    try:
        value = json.loads(raw, object_pairs_hook=_json_no_duplicates)
    except AnchorBatteryBenchmarkError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AnchorBatteryBenchmarkError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise AnchorBatteryBenchmarkError(f"{label} must be a JSON object")
    if require_canonical:
        expected = (
            _canonical_ascii(value) if ascii_canonical else _canonical(value)
        ) + b"\n"
        if raw != expected:
            raise AnchorBatteryBenchmarkError(f"{label} is not canonical JSON")
    return value


def _atomic_new_json(path: Path, document: Mapping[str, Any]) -> Path:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    body = _canonical(dict(document)) + b"\n"
    descriptor = -1
    temporary = ""
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".pending", dir=destination.parent
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as exc:
        raise AnchorBatteryBenchmarkError(
            f"refusing to overwrite: {destination}"
        ) from exc
    except OSError as exc:
        raise AnchorBatteryBenchmarkError(
            f"cannot write output: {destination}"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return destination


def _tensor_receipt(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().contiguous().to(device="cpu")
    raw = tensor.view(torch.uint8).numpy().tobytes()
    return {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "nbytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "shape": list(tensor.shape),
    }


def _source_bytes(owner: object) -> int:
    pager = getattr(owner, "pager", None)
    source = getattr(pager, "source", None)
    metrics = getattr(source, "metrics", None)
    values = dict(metrics()) if callable(metrics) else {}
    value = values.get("network_or_source_body_bytes", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AnchorBatteryBenchmarkError("source byte counter is invalid")
    return value


def _evidence_rows(values: Sequence[object]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for value in values:
        convert = getattr(value, "to_dict", None)
        row = convert() if callable(convert) else value
        if not isinstance(row, Mapping):
            raise AnchorBatteryBenchmarkError("native CRSA evidence is invalid")
        rows.append(dict(row))
    return {"events": rows, "event_count": len(rows), "sha256": _sha256(rows)}


def _snapshot_audit(model: object, directory: Path, name: str) -> dict[str, Any]:
    target = directory / name / "state.json"
    target.parent.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    receipt = model.save_state(target)
    seconds = time.perf_counter() - started
    required = {
        "manifest_body_sha256",
        "payload_sha256",
        "payload_bytes",
        "tensor_count",
        "tensor_bytes",
    }
    if not isinstance(receipt, Mapping) or not required.issubset(receipt):
        raise AnchorBatteryBenchmarkError("native state audit receipt is incomplete")
    manifest_raw = _stable_regular_bytes(
        target, "native state audit manifest", max_bytes=MAX_INPUT_BYTES
    )
    try:
        manifest = json.loads(manifest_raw, object_pairs_hook=_json_no_duplicates)
    except AnchorBatteryBenchmarkError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AnchorBatteryBenchmarkError(
            "native state audit manifest is invalid JSON"
        ) from exc
    if (
        not isinstance(manifest, Mapping)
        or manifest_raw != _canonical_ascii(manifest) + b"\n"
    ):
        raise AnchorBatteryBenchmarkError(
            "native state audit manifest is not canonical JSON"
        )
    if manifest.get("schema") != QWEN38_SNAPSHOT_SCHEMA:
        raise AnchorBatteryBenchmarkError(
            "native state audit snapshot schema is invalid"
        )
    body = manifest.get("body")
    if not isinstance(body, Mapping):
        raise AnchorBatteryBenchmarkError("native state audit body is missing")
    body_sha256 = hashlib.sha256(_canonical_ascii(body)).hexdigest()
    if (
        manifest.get("body_sha256") != body_sha256
        or receipt.get("manifest_body_sha256") != body_sha256
    ):
        raise AnchorBatteryBenchmarkError("native state audit body seal mismatch")
    tensors = body.get("tensors")
    if (
        not isinstance(tensors, list)
        or body.get("tensor_count") != len(tensors)
        or receipt.get("tensor_count") != len(tensors)
        or body.get("tensor_bytes") != receipt.get("tensor_bytes")
    ):
        raise AnchorBatteryBenchmarkError("native state tensor table is missing")
    payload = body.get("payload")
    if not isinstance(payload, Mapping) or set(payload) != {"bytes", "file", "sha256"}:
        raise AnchorBatteryBenchmarkError("native state payload descriptor is invalid")
    payload_name = payload.get("file")
    payload_bytes = payload.get("bytes")
    payload_sha256 = payload.get("sha256")
    if (
        not isinstance(payload_name, str)
        or not payload_name
        or Path(payload_name).name != payload_name
        or "/" in payload_name
        or "\\" in payload_name
        or isinstance(payload_bytes, bool)
        or not isinstance(payload_bytes, int)
        or payload_bytes <= 0
        or not isinstance(payload_sha256, str)
        or len(payload_sha256) != 64
        or any(character not in "0123456789abcdef" for character in payload_sha256)
    ):
        raise AnchorBatteryBenchmarkError("native state payload descriptor is unsafe")
    actual_payload_sha256 = _stable_regular_sha256(
        target.parent / payload_name,
        "native state audit payload",
        expected_bytes=payload_bytes,
        max_bytes=max(payload_bytes, 1),
    )
    if (
        actual_payload_sha256 != payload_sha256
        or receipt.get("payload_sha256") != actual_payload_sha256
        or receipt.get("payload_bytes") != payload_bytes
    ):
        raise AnchorBatteryBenchmarkError("native state audit payload seal mismatch")
    crsa_names = sorted(
        str(row.get("name"))
        for row in tensors
        if isinstance(row, Mapping)
        and str(row.get("name", "")).endswith("crsa_log_usage")
    )
    return {
        "audit_seconds": seconds,
        "manifest_body_sha256": body_sha256,
        "manifest_bytes": len(manifest_raw),
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "native_crsa_usage_tensor_count": len(crsa_names),
        "native_crsa_usage_tensor_names_sha256": _sha256(crsa_names),
        "payload_bytes": payload_bytes,
        "payload_sha256": actual_payload_sha256,
        "tensor_bytes": int(receipt["tensor_bytes"]),
        "tensor_count": int(receipt["tensor_count"]),
    }


def _head_scan(model: object, hidden: torch.Tensor) -> dict[str, Any]:
    started = time.perf_counter()
    values, token_ids = model.pager.topk_logits(
        hidden[:, -1], k=1, name=model.output_head_name
    )
    return {
        "seconds": time.perf_counter() - started,
        "token_ids": _tensor_receipt(token_ids),
        "values": _tensor_receipt(values),
    }


def charge_anchor(
    owner: object,
    cache: SemanticStateAnchorCache,
    input_document: Mapping[str, Any],
    *,
    native_evidence: list[object],
) -> dict[str, Any]:
    """Charge one prefix and return idle-only cost accounting."""

    document = _validate_input(input_document)
    prefix = tuple(document["prefix_token_ids"])
    model = owner.model
    model.reset_state(release=True)
    source_before = _source_bytes(owner)
    evidence_before = len(native_evidence)
    started = time.perf_counter()
    hidden, _forwards = model.prefill([list(prefix)], reset=True)
    prefill_seconds = time.perf_counter() - started
    source_after = _source_bytes(owner)
    store_started = time.perf_counter()
    anchor = cache.store(
        model,
        prefix,
        boundary_kind=document["boundary_kind"],
        seed_hidden=hidden[:, -1:],
    )
    store_seconds = time.perf_counter() - store_started
    result = {
        "anchor": anchor.to_document(),
        "cache_bytes": anchor.cache_bytes,
        "final_hidden": _tensor_receipt(hidden[:, -1:]),
        "model_source_bytes": source_after - source_before,
        "native_crsa": _evidence_rows(native_evidence[evidence_before:]),
        "prefill_seconds": prefill_seconds,
        "store_seconds": store_seconds,
        "total_seconds": prefill_seconds + store_seconds,
    }
    model.reset_state(release=True)
    return result


def _run_arm(
    route: str,
    owner: object,
    cache: SemanticStateAnchorCache,
    document: Mapping[str, Any],
    *,
    native_evidence: list[object],
    audit_root: Path,
    arm_index: int,
) -> dict[str, Any]:
    model = owner.model
    prefix = tuple(document["prefix_token_ids"])
    suffix = tuple(document["suffix_token_ids"])
    full = (*prefix, *suffix)
    model.reset_state(release=True)
    source_before = _source_bytes(owner)
    evidence_before = len(native_evidence)
    restore_seconds = 0.0
    suffix_seconds = 0.0
    head: dict[str, Any] | None = None
    anchor: AnchorReceipt | None = None
    restored: Any | None = None

    if route == "baseline":
        started = time.perf_counter()
        hidden, _forwards = model.prefill([list(full)], reset=True)
        compute_seconds = time.perf_counter() - started
        final_hidden = hidden[:, -1:]
        if not suffix:
            head = _head_scan(model, final_hidden)
            compute_seconds += float(head["seconds"])
    elif route == "battery":
        started = time.perf_counter()
        restored = cache.restore_deepest(model, full)
        restore_seconds = time.perf_counter() - started
        if restored is None or restored.anchor.prefix_length != len(prefix):
            raise AnchorBatteryBenchmarkError(
                "deepest native anchor is missing or wrong"
            )
        anchor = restored.anchor
        if suffix:
            started = time.perf_counter()
            hidden, _forwards = model.prefill([list(suffix)], reset=False)
            suffix_seconds = time.perf_counter() - started
            final_hidden = hidden[:, -1:]
        else:
            if restored.seed_hidden is None:
                raise AnchorBatteryBenchmarkError(
                    "exact-prefix anchor has no seed hidden"
                )
            final_hidden = restored.seed_hidden
            head = _head_scan(model, final_hidden)
            suffix_seconds = float(head["seconds"])
        compute_seconds = restore_seconds + suffix_seconds
    else:  # pragma: no cover - guarded by sealed schedule validation.
        raise AnchorBatteryBenchmarkError("unknown benchmark route")

    source_after = _source_bytes(owner)
    state = _snapshot_audit(model, audit_root, f"arm-{arm_index:02d}-{route}")
    model_source_bytes = source_after - source_before
    if model_source_bytes < 0:
        raise AnchorBatteryBenchmarkError("model source byte counter moved backwards")
    snapshot_restore_bytes = 0
    if anchor is not None:
        assert restored is not None
        snapshot_restore_bytes = (
            anchor.snapshot_manifest_bytes
            + anchor.snapshot_payload_bytes
            + (anchor.seed_hidden_bytes if restored.exact_prefix else 0)
        )
    arm = {
        "anchor": None
        if anchor is None
        else {
            "cache_bytes": anchor.cache_bytes,
            "hit_count": anchor.hit_count,
            "prefix_length": anchor.prefix_length,
            "prefix_sha256": anchor.prefix_sha256,
            "receipt_sha256": anchor.receipt_sha256,
        },
        "demand_seconds": compute_seconds,
        "final_hidden": _tensor_receipt(final_hidden),
        "final_state": state,
        "head_scan": head,
        "model_source_bytes": model_source_bytes,
        "native_crsa": _evidence_rows(native_evidence[evidence_before:]),
        "restore_and_verify_seconds": restore_seconds,
        "route": route,
        "snapshot_restore_bytes": snapshot_restore_bytes,
        "suffix_or_head_seconds": suffix_seconds,
        "total_read_bytes": model_source_bytes + snapshot_restore_bytes,
    }
    model.reset_state(release=True)
    return arm


def compare_arms(
    owner: object,
    cache: SemanticStateAnchorCache,
    input_document: Mapping[str, Any],
    *,
    native_evidence: list[object],
    audit_root: Path,
) -> dict[str, Any]:
    """Execute the sealed schedule on one long-lived authenticated runtime."""

    document = _validate_input(input_document)
    arms = [
        _run_arm(
            route,
            owner,
            cache,
            document,
            native_evidence=native_evidence,
            audit_root=audit_root,
            arm_index=index,
        )
        for index, route in enumerate(document["schedule"])
    ]
    baselines = [row for row in arms if row["route"] == "baseline"]
    batteries = [row for row in arms if row["route"] == "battery"]
    if not baselines or not batteries:
        raise AnchorBatteryBenchmarkError("schedule lacks an A/B arm")
    reference_hidden = baselines[0]["final_hidden"]["sha256"]
    reference_payload = baselines[0]["final_state"]["payload_sha256"]
    reference_body = baselines[0]["final_state"]["manifest_body_sha256"]
    reference_head = baselines[0]["head_scan"]
    for arm in arms:
        if arm["final_hidden"]["sha256"] != reference_hidden:
            raise AnchorBatteryBenchmarkError(
                "final hidden tensors are not bit-identical"
            )
        if arm["final_state"]["payload_sha256"] != reference_payload:
            raise AnchorBatteryBenchmarkError(
                "serialized final-state payloads are not bit-identical"
            )
        if arm["final_state"]["manifest_body_sha256"] != reference_body:
            raise AnchorBatteryBenchmarkError(
                "authenticated final-state manifests are not bit-identical"
            )
        if arm["head_scan"] != reference_head and (
            arm["head_scan"] is None or reference_head is None
        ):
            raise AnchorBatteryBenchmarkError("empty-suffix head scan presence differs")
        if arm["head_scan"] is not None and reference_head is not None:
            if (
                arm["head_scan"]["token_ids"]["sha256"]
                != reference_head["token_ids"]["sha256"]
                or arm["head_scan"]["values"]["sha256"]
                != reference_head["values"]["sha256"]
            ):
                raise AnchorBatteryBenchmarkError("restored LM-head scan is not exact")

    baseline_seconds = statistics.median(row["demand_seconds"] for row in baselines)
    battery_seconds = statistics.median(row["demand_seconds"] for row in batteries)
    baseline_model_bytes = statistics.median(
        row["model_source_bytes"] for row in baselines
    )
    battery_model_bytes = statistics.median(
        row["model_source_bytes"] for row in batteries
    )
    baseline_total_bytes = statistics.median(
        row["total_read_bytes"] for row in baselines
    )
    battery_total_bytes = statistics.median(
        row["total_read_bytes"] for row in batteries
    )
    saved = baseline_seconds - battery_seconds
    return {
        "arms": arms,
        "exactness": {
            "final_hidden_bit_exact": True,
            "final_state_manifest_body_bit_exact": True,
            "final_state_payload_bit_exact": True,
            "head_scan_bit_exact": reference_head is not None,
            "reference_hidden_sha256": reference_hidden,
            "reference_state_payload_sha256": reference_payload,
            "reference_state_manifest_body_sha256": reference_body,
        },
        "peak_demand": {
            "anchor_charge_excluded": True,
            "baseline_median_model_source_bytes": baseline_model_bytes,
            "baseline_median_seconds": baseline_seconds,
            "baseline_median_total_read_bytes": baseline_total_bytes,
            "battery_median_model_source_bytes": battery_model_bytes,
            "battery_median_seconds": battery_seconds,
            "battery_median_total_read_bytes": battery_total_bytes,
            "saved_seconds": saved,
            "speedup": None
            if battery_seconds == 0.0
            else baseline_seconds / battery_seconds,
            "total_read_bytes_saved": baseline_total_bytes - battery_total_bytes,
        },
    }


def _positive_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _csv_tokens(raw: str, *, allow_empty: bool) -> list[int]:
    if not raw.strip():
        if allow_empty:
            return []
        raise argparse.ArgumentTypeError("token list must not be empty")
    try:
        values = [int(part.strip()) for part in raw.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "token IDs must be comma-separated integers"
        ) from exc
    _tokens(values, "token list", allow_empty=allow_empty)
    return values


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target-bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
    parser.add_argument(
        "--compute-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--source-budget-mb", type=_positive_float, default=1_048_576.0)
    parser.add_argument("--max-resident-mb", type=_positive_float, default=384.0)
    parser.add_argument("--max-context-tokens", type=_positive_int, default=2048)
    parser.add_argument("--cache-max-mb", type=_positive_float, default=2048.0)
    parser.add_argument("--output", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="write a sealed token-only input")
    prepare.add_argument("--prefix-token-ids", required=True)
    prepare.add_argument("--suffix-token-ids", default="")
    prepare.add_argument(
        "--attention-mode", choices=("off", "native-crsa"), default="native-crsa"
    )
    prepare.add_argument(
        "--boundary-kind",
        choices=("turn", "tool-call", "tool-output", "thinking", "custom"),
        default="turn",
    )
    prepare.add_argument("--schedule", choices=tuple(_SCHEDULES), default="AB")
    prepare.add_argument("--output", required=True)
    for name in ("charge", "compare"):
        command = commands.add_parser(name)
        command.add_argument("--input", required=True)
        _add_runtime_arguments(command)
    return parser


def _open_owner(
    args: argparse.Namespace, document: Mapping[str, Any], evidence: list[object]
):
    return base._open_model(
        role="target",
        bundle=Path(args.target_bundle).expanduser().resolve(),
        identity=LogicalModelIdentity(OFFICIAL_REPO_ID, OFFICIAL_REVISION),
        source_budget_mb=args.source_budget_mb,
        device=args.device,
        compute_dtype=args.compute_dtype,
        max_resident_bytes=int(args.max_resident_mb * MIB),
        max_context_tokens=args.max_context_tokens,
        attention_mode=document["attention_mode"],
        native_evidence=evidence,
        require_production_profile=True,
        require_official_target=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        document = _seal(
            {
                "attention_mode": args.attention_mode,
                "boundary_kind": args.boundary_kind,
                "prefix_token_ids": _csv_tokens(
                    args.prefix_token_ids, allow_empty=False
                ),
                "schedule": list(_SCHEDULES[args.schedule]),
                "schema": INPUT_SCHEMA,
                "suffix_token_ids": _csv_tokens(
                    args.suffix_token_ids, allow_empty=True
                ),
            }
        )
        _validate_input(document)
        output = _atomic_new_json(Path(args.output), document)
        print(
            json.dumps(
                {"output": str(output), "sha256": document["sha256"]}, sort_keys=True
            )
        )
        return 0

    input_document = _validate_input(
        _stable_json(
            Path(args.input),
            "benchmark input",
            require_canonical=True,
        )
    )
    if (
        len(input_document["prefix_token_ids"])
        + len(input_document["suffix_token_ids"])
        > args.max_context_tokens
    ):
        raise AnchorBatteryBenchmarkError(
            "sealed context exceeds runtime context bound"
        )
    cache = SemanticStateAnchorCache(
        Path(args.cache), max_bytes=int(args.cache_max_mb * MIB)
    )
    evidence: list[object] = []
    owner = _open_owner(args, input_document, evidence)
    try:
        common = {
            "attention_mode": input_document["attention_mode"],
            "bundle": owner.bundle_receipt,
            "input_sha256": input_document["sha256"],
            "preflight": owner.preflight_receipt,
            "runtime_authentication_seconds": owner.verify_seconds
            + owner.preflight_seconds,
        }
        if args.command == "charge":
            idle = charge_anchor(owner, cache, input_document, native_evidence=evidence)
            result = _seal(
                {
                    **common,
                    "idle_charge": idle,
                    "peak_demand_executed": False,
                    "schema": CHARGE_SCHEMA,
                    "status": "charged",
                }
            )
        else:
            with tempfile.TemporaryDirectory(prefix=".qwen-anchor-audit-") as temporary:
                comparison = compare_arms(
                    owner,
                    cache,
                    input_document,
                    native_evidence=evidence,
                    audit_root=Path(temporary),
                )
            result = _seal(
                {
                    **common,
                    **comparison,
                    "idle_charge": {
                        "executed_in_this_command": False,
                        "excluded_from_peak_demand": True,
                    },
                    "schema": RESULT_SCHEMA,
                    "status": "exact-positive",
                }
            )
        output = _atomic_new_json(Path(args.output), result)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "sha256": result["sha256"],
                    "status": result["status"],
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        owner.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AnchorBatteryBenchmarkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
