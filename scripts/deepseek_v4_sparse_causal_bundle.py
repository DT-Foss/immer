#!/usr/bin/env python3
"""Build a trace-complete sparse local DeepSeek-V4 ``weights/ + causal/`` bundle.

The logical shard sizes and tensor offsets remain byte-identical to the pinned
checkpoint. Only verified access-trace leaves consume physical disk blocks;
all other shard regions are sparse holes. Causal expert bindings are created
only when every byte of that expert is materialized, so unseen experts fail
closed instead of reading a hole.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import tempfile
from typing import Any
import uuid

from immer.knowledge import AccessTrace, AccessTraceError, Streamer
from immer.runtimes.deepseek_v4 import (
    CausalWeightMount,
    DeepSeekWeightPager,
    ExpertSourceRange,
    ExpertTensorLayout,
    LogicalModelIdentity,
    OfficialExpertRangePlan,
    TensorRangePlan,
)

ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_SOURCE = "deepseek-ai/DeepSeek-V4-Flash-0731"
OFFICIAL_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
OFFICIAL_LAYOUT_FINGERPRINT = (
    "61600c552f3e52ae382b3eca0370001905fc4e2ecdd2da5f3e66fa95809206c7"
)
DEFAULT_CACHE = ROOT / "artifacts" / "private" / "deepseek-v4-cache"
BUNDLE_SCHEMA = "immer.deepseek-v4-sparse-causal-bundle/v1"
APPEND_PENDING_SCHEMA = "immer.deepseek-v4-expert-append-pending/v1"
APPEND_RECEIPT_SCHEMA = "immer.deepseek-v4-expert-append-receipt/v1"
APPEND_VERIFY_SCHEMA = "immer.deepseek-v4-expert-append-verification/v1"
TRACE_EXPERT_PLAN_SCHEMA = "immer.deepseek-v4-trace-expert-plan/v1"
TRACE_EXPERT_INGEST_SCHEMA = "immer.deepseek-v4-trace-expert-ingest/v1"
GENERAL_BUNDLE_SCHEMA = "immer.deepseek-v4-general-causal-bundle/v1"
GENERAL_DENSE_COVERAGE_CAPABILITY = "deepseek-v4-general-dense-coverage/v1"
DENSE_PLAN_SCHEMA = "immer.deepseek-v4-dense-promotion-plan/v1"
DENSE_TRANSACTION_SCHEMA = "immer.deepseek-v4-dense-promotion-transaction/v1"
DENSE_PENDING_SCHEMA = "immer.deepseek-v4-dense-promotion-pending/v1"
DENSE_RECEIPT_SCHEMA = "immer.deepseek-v4-dense-promotion-receipt/v1"
DENSE_VERIFY_SCHEMA = "immer.deepseek-v4-dense-promotion-verification/v1"
_APPEND_DIRECTORY = "expert-appends"
_APPEND_LOCK = ".append.lock"
_APPEND_PENDING = "pending.json"
_DENSE_DIRECTORY = "dense-promotion"
_DENSE_PENDING = "pending.json"
_ZERO_SHA256 = "0" * 64
_OFFICIAL_DENSE_TENSOR_COUNT = 1564
_EXPERT_TENSOR = re.compile(
    r"^layers\.(0|[1-9][0-9]*)\.ffn\.experts\."
    r"(0|[1-9][0-9]*)\.(w[123])\.(weight|scale)$"
)
_EXPERT_PARTS = frozenset(
    (f"w{index}.{kind}" for index in (1, 2, 3) for kind in ("scale", "weight"))
)
_REMOTE_RANGE_RESERVATION_OVERHEAD_BYTES = 16 * 1024
_DENSE_DTYPE_BYTES = {
    "BOOL": 1,
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "U8": 1,
    "U16": 2,
    "U32": 4,
    "U64": 8,
}


class SparseBundleError(RuntimeError):
    """The sparse local bundle cannot be proven complete for its trace."""


class InjectedAppendCrash(SparseBundleError):
    """Test-only process interruption at a durable append boundary."""


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
        raise SparseBundleError("value is not canonical JSON") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(value: object) -> str:
    return _sha256_bytes(_canonical(value))


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _strict_json(path: Path) -> Any:
    def pairs(entries: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in entries:
            if key in result:
                raise SparseBundleError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)
    except SparseBundleError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SparseBundleError(f"cannot read JSON: {path}") from exc


def _json_pairs(entries: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in entries:
        if key in result:
            raise SparseBundleError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _read_all(descriptor: int, *, maximum: int = 64 * 1024 * 1024) -> bytes:
    size = os.fstat(descriptor).st_size
    if size < 1 or size > maximum:
        raise SparseBundleError("JSON document size is outside its bound")
    chunks: list[bytes] = []
    cursor = 0
    while cursor < size:
        chunk = os.pread(descriptor, min(1024 * 1024, size - cursor), cursor)
        if not chunk:
            raise SparseBundleError("JSON document returned a short read")
        chunks.append(chunk)
        cursor += len(chunk)
    return b"".join(chunks)


def _read_json_at(directory: int, name: str) -> tuple[Any, bytes]:
    flags = os.O_RDONLY | int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(name, flags, dir_fd=directory)
    except OSError as exc:
        raise SparseBundleError(f"cannot open JSON document {name!r}") from exc
    try:
        opened = os.fstat(descriptor)
        linked = os.stat(name, dir_fd=directory, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(linked.st_mode)
            or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
        ):
            raise SparseBundleError(f"JSON document {name!r} is not a stable file")
        encoded = _read_all(descriptor)
    finally:
        os.close(descriptor)
    try:
        return (
            json.loads(encoded.decode("utf-8"), object_pairs_hook=_json_pairs),
            encoded,
        )
    except SparseBundleError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SparseBundleError(f"cannot decode JSON document {name!r}") from exc


def _optional_json_at(directory: int, name: str) -> tuple[Any, bytes] | None:
    try:
        return _read_json_at(directory, name)
    except SparseBundleError as exc:
        try:
            os.stat(name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            return None
        raise exc


def _write_all(descriptor: int, encoded: bytes) -> None:
    cursor = 0
    while cursor < len(encoded):
        written = os.write(descriptor, encoded[cursor:])
        if written <= 0:
            raise SparseBundleError("short durable metadata write")
        cursor += written


def _atomic_json_at(directory: int, name: str, document: Mapping[str, Any]) -> None:
    if Path(name).name != name or name in (".", ".."):
        raise SparseBundleError("metadata name is not a safe basename")
    encoded = _canonical(document)
    temporary = f".{name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, flags, 0o600, dir_fd=directory)
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise SparseBundleError("temporary metadata is not a regular file")
        os.close(descriptor)
        descriptor = None
        os.replace(
            temporary,
            name,
            src_dir_fd=directory,
            dst_dir_fd=directory,
        )
        os.fsync(directory)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory)
        except FileNotFoundError:
            pass


def _open_directory_at(directory: int, name: str, *, create: bool = False) -> int:
    if Path(name).name != name or name in (".", ".."):
        raise SparseBundleError("directory name is not a safe basename")
    if create:
        try:
            os.mkdir(name, 0o700, dir_fd=directory)
        except FileExistsError:
            pass
        except OSError as exc:
            raise SparseBundleError(f"cannot create directory {name!r}") from exc
    flags = os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0))
    flags |= int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(name, flags, dir_fd=directory)
    except OSError as exc:
        raise SparseBundleError(f"cannot open directory {name!r}") from exc
    try:
        opened = os.fstat(descriptor)
        linked = os.stat(name, dir_fd=directory, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(linked.st_mode)
            or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
        ):
            raise SparseBundleError(f"directory {name!r} is not stable")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_plain_root(path: Path) -> int:
    try:
        linked = path.lstat()
    except OSError as exc:
        raise SparseBundleError(f"bundle root is missing: {path}") from exc
    if stat.S_ISLNK(linked.st_mode) or not stat.S_ISDIR(linked.st_mode):
        raise SparseBundleError("bundle root must be a non-symlink directory")
    flags = os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0))
    flags |= int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    descriptor = os.open(path, flags)
    opened = os.fstat(descriptor)
    if (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino):
        os.close(descriptor)
        raise SparseBundleError("bundle root changed while opening it")
    return descriptor


def _assert_root_stable(path: Path, descriptor: int) -> None:
    try:
        linked = path.lstat()
    except OSError as exc:
        raise SparseBundleError("bundle root disappeared during append") from exc
    opened = os.fstat(descriptor)
    if (
        stat.S_ISLNK(linked.st_mode)
        or not stat.S_ISDIR(linked.st_mode)
        or (linked.st_dev, linked.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        raise SparseBundleError("bundle root changed during append")


@contextmanager
def _append_lock(root: Path) -> Iterable[tuple[int, int, int]]:
    root_descriptor = _open_plain_root(root)
    append_descriptor: int | None = None
    weights_descriptor: int | None = None
    lock_descriptor: int | None = None
    try:
        weights_descriptor = _open_directory_at(root_descriptor, "weights")
        append_descriptor = _open_directory_at(
            root_descriptor,
            _APPEND_DIRECTORY,
            create=True,
        )
        flags = os.O_RDWR | os.O_CREAT
        flags |= int(getattr(os, "O_CLOEXEC", 0))
        flags |= int(getattr(os, "O_NOFOLLOW", 0))
        lock_descriptor = os.open(
            _APPEND_LOCK,
            flags,
            0o600,
            dir_fd=append_descriptor,
        )
        opened = os.fstat(lock_descriptor)
        linked = os.stat(
            _APPEND_LOCK,
            dir_fd=append_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(linked.st_mode)
            or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
        ):
            raise SparseBundleError("append lock is not a stable regular file")
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        _assert_root_stable(root, root_descriptor)
        try:
            yield root_descriptor, append_descriptor, weights_descriptor
        finally:
            _assert_root_stable(root, root_descriptor)
    finally:
        if lock_descriptor is not None:
            try:
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            finally:
                os.close(lock_descriptor)
        if weights_descriptor is not None:
            os.close(weights_descriptor)
        if append_descriptor is not None:
            os.close(append_descriptor)
        os.close(root_descriptor)


def _journal_event(
    history: Sequence[Mapping[str, Any]],
    state: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(state, str) or not state:
        raise SparseBundleError("journal state is invalid")
    body = {
        "payload": dict(payload),
        "previous_sha256": (
            _ZERO_SHA256 if not history else str(history[-1]["sha256"])
        ),
        "sequence": len(history),
        "state": state,
    }
    return {**body, "sha256": _sha256(body)}


def _verify_journal(history: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(history, list) or not history:
        raise SparseBundleError("append journal history is invalid")
    verified: list[dict[str, Any]] = []
    previous = _ZERO_SHA256
    for index, raw in enumerate(history):
        if not isinstance(raw, Mapping) or set(raw) != {
            "payload",
            "previous_sha256",
            "sequence",
            "sha256",
            "state",
        }:
            raise SparseBundleError("append journal record schema is invalid")
        body = {key: raw[key] for key in raw if key != "sha256"}
        if (
            raw.get("sequence") != index
            or raw.get("previous_sha256") != previous
            or raw.get("sha256") != _sha256(body)
            or not isinstance(raw.get("payload"), Mapping)
            or not isinstance(raw.get("state"), str)
        ):
            raise SparseBundleError("append journal SHA chain is invalid")
        verified.append(dict(raw))
        previous = str(raw["sha256"])
    return tuple(verified)


def _pending_document(
    transaction: Mapping[str, Any], history: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    body = {"history": list(history), "transaction": dict(transaction)}
    return {
        "body": body,
        "schema": APPEND_PENDING_SCHEMA,
        "sha256": _sha256(body),
    }


def _verify_pending(
    document: object,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    if not isinstance(document, Mapping) or set(document) != {
        "body",
        "schema",
        "sha256",
    }:
        raise SparseBundleError("append pending document schema is invalid")
    body = document.get("body")
    if (
        document.get("schema") != APPEND_PENDING_SCHEMA
        or not isinstance(body, Mapping)
        or set(body) != {"history", "transaction"}
        or document.get("sha256") != _sha256(body)
        or not isinstance(body.get("transaction"), Mapping)
    ):
        raise SparseBundleError("append pending document identity is invalid")
    history = _verify_journal(body.get("history"))
    return dict(body["transaction"]), history


@dataclass(frozen=True, slots=True)
class CachedRange:
    shard: str
    start: int
    stop: int
    payload: Path
    sha256: str


class VerifiedRangeCache:
    def __init__(self, root: Path, *, repo_id: str, revision: str) -> None:
        self.root = root
        self.repo_id = repo_id
        self.revision = revision
        self.by_shard: dict[str, list[CachedRange]] = defaultdict(list)
        self._verified: set[Path] = set()
        if not root.is_dir():
            raise SparseBundleError(f"range cache is missing: {root}")
        for metadata in root.glob("*.json"):
            document = _strict_json(metadata)
            if not isinstance(document, Mapping) or document.get("schema") != (
                "immer.range-cache/v1"
            ):
                continue
            contract = document.get("contract")
            payload = metadata.with_suffix(".bin")
            if (
                not isinstance(contract, Mapping)
                or contract.get("kind") != "range"
                or contract.get("repo") != repo_id
                or contract.get("revision") != revision
                or not payload.is_file()
            ):
                continue
            start = int(contract["start"])
            stop = int(contract["end"]) + 1
            size = int(document.get("size", -1))
            digest = document.get("sha256")
            if (
                start < 0
                or stop <= start
                or stop - start != size
                or payload.stat().st_size != size
                or not isinstance(digest, str)
                or len(digest) != 64
            ):
                raise SparseBundleError(f"invalid cached range metadata: {metadata}")
            shard = str(contract["filename"])
            self.by_shard[shard].append(
                CachedRange(shard, start, stop, payload, digest)
            )
        for ranges in self.by_shard.values():
            ranges.sort(key=lambda row: (row.start, row.stop, row.payload.name))

    def resolve(self, shard: str, offset: int, length: int) -> tuple[CachedRange, int]:
        stop = offset + length
        for row in self.by_shard.get(shard, ()):
            if row.start <= offset and row.stop >= stop:
                return row, offset - row.start
        raise SparseBundleError(f"cache does not cover {shard}[{offset}:{stop}]")

    def read(self, shard: str, offset: int, length: int) -> bytes:
        row, relative = self.resolve(shard, offset, length)
        if row.payload not in self._verified:
            encoded = row.payload.read_bytes()
            if _sha256_bytes(encoded) != row.sha256:
                raise SparseBundleError(f"cached range SHA-256 mismatch: {row.payload}")
            self._verified.add(row.payload)
        descriptor = os.open(row.payload, os.O_RDONLY)
        try:
            encoded = os.pread(descriptor, length, relative)
        finally:
            os.close(descriptor)
        if len(encoded) != length:
            raise SparseBundleError("cached range returned a short payload")
        return encoded


def _load_inventory(path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    document = _strict_json(path)
    if not isinstance(document, Mapping) or document.get("schema") != (
        "immer.tensor-inventory-cache/v1"
    ):
        raise SparseBundleError("inventory cache schema is invalid")
    inventory = document.get("inventory")
    fingerprint = document.get("source_fingerprint")
    if not isinstance(inventory, dict) or not isinstance(fingerprint, str):
        raise SparseBundleError("inventory cache identity is invalid")
    return inventory, fingerprint, dict(document)


def _load_traces(
    paths: Iterable[Path],
    *,
    repo_id: str,
    revision: str,
    fingerprint: str,
) -> tuple[tuple[AccessTrace, ...], tuple[tuple[str, int, int], ...]]:
    traces: list[AccessTrace] = []
    leaves: set[tuple[str, int, int]] = set()
    for path in paths:
        trace = AccessTrace.from_bytes(path.read_bytes())
        trace.verify()
        if (
            trace.repo_id != repo_id
            or trace.revision != revision
            or trace.inventory_fingerprint != fingerprint
        ):
            raise SparseBundleError(f"access trace identity mismatch: {path}")
        traces.append(trace)
        leaves.update(
            (leaf.shard, leaf.offset, leaf.length)
            for operation in trace.operations
            for leaf in operation.leaves
        )
    if not traces or not leaves:
        raise SparseBundleError("at least one non-empty access trace is required")
    return tuple(traces), tuple(sorted(leaves))


def _header_bytes(
    shard: Mapping[str, Any], tensors: Sequence[Mapping[str, Any]]
) -> bytes:
    header: dict[str, Any] = {}
    if shard.get("st_metadata") is not None:
        header["__metadata__"] = shard["st_metadata"]
    for tensor in sorted(tensors, key=lambda row: int(row["offset_in_shard"][0])):
        header[str(tensor["name"])] = {
            "dtype": str(tensor["dtype"]),
            "shape": [int(value) for value in tensor["shape"]],
            "data_offsets": [int(value) for value in tensor["offset_in_shard"]],
        }
    encoded = json.dumps(
        header,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    capacity = int(shard["data_start"]) - 8
    if len(encoded) > capacity:
        raise SparseBundleError(
            f"reconstructed header exceeds original capacity for {shard['file']}"
        )
    return struct.pack("<Q", capacity) + encoded + b" " * (capacity - len(encoded))


def _interval_covered(
    intervals: Mapping[str, Sequence[tuple[int, int]]],
    shard: str,
    start: int,
    stop: int,
) -> bool:
    cursor = start
    for left, right in intervals.get(shard, ()):
        if right <= cursor:
            continue
        if left > cursor:
            return False
        cursor = max(cursor, right)
        if cursor >= stop:
            return True
    return False


def _copy_config(
    cache_root: Path, weights: Path, *, repo_id: str, revision: str
) -> str:
    for metadata in (cache_root / "files").glob("*.json"):
        document = _strict_json(metadata)
        contract = document.get("contract") if isinstance(document, Mapping) else None
        if (
            isinstance(contract, Mapping)
            and contract.get("filename") == "config.json"
            and contract.get("repo") == repo_id
            and contract.get("revision") == revision
        ):
            payload = metadata.with_suffix(".bin")
            encoded = payload.read_bytes()
            if _sha256_bytes(encoded) != document.get("sha256"):
                raise SparseBundleError("cached config SHA-256 mismatch")
            _atomic_bytes(weights / "config.json", encoded)
            return _sha256_bytes(encoded)
    raise SparseBundleError("verified config.json is absent from the cache")


def _write_sparse_weights(
    weights: Path,
    inventory: Mapping[str, Any],
    leaves: Sequence[tuple[str, int, int]],
    cache: VerifiedRangeCache,
) -> tuple[list[dict[str, Any]], dict[str, list[tuple[int, int]]]]:
    tensors_by_shard: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for tensor in inventory.get("tensors", []):
        tensors_by_shard[str(tensor["shard"])].append(tensor)
    leaves_by_shard: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for shard, offset, length in leaves:
        leaves_by_shard[shard].append((offset, offset + length))
    for intervals in leaves_by_shard.values():
        intervals.sort()

    shard_receipts: list[dict[str, Any]] = []
    for shard in inventory.get("shards", []):
        filename = str(shard["file"])
        path = weights / filename
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        descriptor = os.open(path, flags, 0o600)
        try:
            size = int(shard["size"])
            os.ftruncate(descriptor, size)
            header = _header_bytes(shard, tensors_by_shard[filename])
            if os.pwrite(descriptor, header, 0) != len(header):
                raise SparseBundleError(f"short header write for {filename}")
            payload_bytes = 0
            for offset, stop in leaves_by_shard.get(filename, ()):
                encoded = cache.read(filename, offset, stop - offset)
                if os.pwrite(descriptor, encoded, offset) != len(encoded):
                    raise SparseBundleError(f"short payload write for {filename}")
                payload_bytes += len(encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        metadata = path.stat()
        shard_receipts.append(
            {
                "file": filename,
                "header_bytes": len(header),
                "logical_bytes": metadata.st_size,
                "materialized_leaf_bytes": payload_bytes,
                "physical_bytes": metadata.st_blocks * 512,
            }
        )
    return shard_receipts, leaves_by_shard


def _write_index(weights: Path, inventory: Mapping[str, Any]) -> str:
    document = {
        "metadata": {"total_size": int(inventory["index_total_size"])},
        "weight_map": {
            str(tensor["name"]): str(tensor["shard"]) for tensor in inventory["tensors"]
        },
    }
    encoded = _canonical(document)
    _atomic_bytes(weights / "model.safetensors.index.json", encoded)
    return _sha256_bytes(encoded)


def _covered_experts(
    inventory: Mapping[str, Any],
    intervals: Mapping[str, Sequence[tuple[int, int]]],
) -> dict[int, tuple[int, ...]]:
    parts: dict[tuple[int, int], set[str]] = defaultdict(set)
    for tensor in inventory["tensors"]:
        match = _EXPERT_TENSOR.fullmatch(str(tensor["name"]))
        if match is None:
            continue
        shard = str(tensor["shard"])
        start = int(tensor["data_start"]) + int(tensor["offset_in_shard"][0])
        stop = int(tensor["data_start"]) + int(tensor["offset_in_shard"][1])
        if _interval_covered(intervals, shard, start, stop):
            parts[(int(match.group(1)), int(match.group(2)))].add(
                f"{match.group(3)}.{match.group(4)}"
            )
    by_layer: dict[int, list[int]] = defaultdict(list)
    for (layer, expert), observed in parts.items():
        if observed == _EXPERT_PARTS:
            by_layer[layer].append(expert)
    return {
        layer: tuple(sorted(experts)) for layer, experts in sorted(by_layer.items())
    }


def _coordinate_records(
    coordinates: Iterable[tuple[int, int]],
) -> list[dict[str, int]]:
    return [
        {"expert_id": expert_id, "layer": layer}
        for layer, expert_id in sorted(set(coordinates))
    ]


def _inventory_expert_resources(
    inventory: Mapping[str, Any],
) -> dict[tuple[int, int], dict[str, int]]:
    tensors = inventory.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise SparseBundleError("bundle pinned inventory tensor table is invalid")
    parts: dict[tuple[int, int], dict[str, int]] = defaultdict(dict)
    for tensor in tensors:
        if not isinstance(tensor, Mapping):
            raise SparseBundleError("bundle pinned inventory tensor row is invalid")
        match = _EXPERT_TENSOR.fullmatch(str(tensor.get("name", "")))
        if match is None:
            continue
        offsets = tensor.get("offset_in_shard")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in offsets
            )
            or offsets[0] < 0
            or offsets[1] <= offsets[0]
        ):
            raise SparseBundleError("bundle pinned expert tensor range is invalid")
        coordinate = (int(match.group(1)), int(match.group(2)))
        part = f"{match.group(3)}.{match.group(4)}"
        if part in parts[coordinate]:
            raise SparseBundleError("bundle pinned inventory duplicates expert parts")
        parts[coordinate][part] = offsets[1] - offsets[0]
    resources: dict[tuple[int, int], dict[str, int]] = {}
    for coordinate, observed in parts.items():
        if set(observed) != _EXPERT_PARTS:
            raise SparseBundleError("bundle pinned inventory has incomplete experts")
        resources[coordinate] = {
            "maximum_part_bytes": max(observed.values()),
            "parts": len(observed),
            "payload_bytes": sum(observed.values()),
        }
    if not resources:
        raise SparseBundleError("bundle pinned inventory has no complete experts")
    return resources


def _manifest_expert_coordinates(
    manifest: Mapping[str, Any],
    inventory_coordinates: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    covered = manifest.get("covered_experts")
    binding_count = manifest.get("causal_bindings")
    if (
        not isinstance(covered, Mapping)
        or not covered
        or isinstance(binding_count, bool)
        or not isinstance(binding_count, int)
        or binding_count < 1
    ):
        raise SparseBundleError("bundle covered-expert table is invalid")
    coordinates: set[tuple[int, int]] = set()
    for raw_layer, raw_experts in covered.items():
        if (
            not isinstance(raw_layer, str)
            or re.fullmatch(r"0|[1-9][0-9]*", raw_layer) is None
            or not isinstance(raw_experts, list)
            or not raw_experts
            or any(
                isinstance(expert_id, bool)
                or not isinstance(expert_id, int)
                or expert_id < 0
                for expert_id in raw_experts
            )
            or raw_experts != sorted(set(raw_experts))
        ):
            raise SparseBundleError("bundle covered-expert table is noncanonical")
        layer = int(raw_layer)
        coordinates.update((layer, expert_id) for expert_id in raw_experts)
    if len(coordinates) != binding_count or not coordinates.issubset(
        inventory_coordinates
    ):
        raise SparseBundleError("bundle covered experts disagree with inventory")
    return coordinates


def _sealed_trace_coverage(
    paths: Iterable[Path],
    *,
    repo_id: str,
    revision: str,
    fingerprint: str,
    inventory: Mapping[str, Any],
) -> tuple[
    tuple[dict[str, Any], ...],
    dict[str, tuple[tuple[int, int], ...]],
    tuple[tuple[str, int, int], ...],
]:
    raw_shards = inventory.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise SparseBundleError("bundle pinned inventory shard table is invalid")
    shard_sizes: dict[str, int] = {}
    for shard in raw_shards:
        if not isinstance(shard, Mapping):
            raise SparseBundleError("bundle pinned inventory shard row is invalid")
        name = shard.get("file")
        size = shard.get("size")
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name in (".", "..")
            or name in shard_sizes
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise SparseBundleError("bundle pinned inventory shard identity is invalid")
        shard_sizes[name] = size

    evidence_by_sha256: dict[str, dict[str, Any]] = {}
    leaves: set[tuple[str, int, int]] = set()
    supplied = tuple(paths)
    if not supplied:
        raise SparseBundleError("at least one sealed access trace is required")
    for path in supplied:
        encoded = _stable_file_bytes(path)
        try:
            trace = AccessTrace.from_bytes(encoded)
            trace.verify()
        except AccessTraceError as exc:
            raise SparseBundleError(f"invalid sealed access trace: {path}") from exc
        if (
            trace.repo_id != repo_id
            or trace.revision != revision
            or trace.inventory_fingerprint != fingerprint
        ):
            raise SparseBundleError(f"access trace identity mismatch: {path}")
        if not trace.operations:
            raise SparseBundleError(f"access trace is empty: {path}")
        trace_leaves: set[tuple[str, int, int]] = set()
        for operation in trace.operations:
            for leaf in operation.leaves:
                shard_size = shard_sizes.get(leaf.shard)
                if (
                    Path(leaf.shard).name != leaf.shard
                    or leaf.shard in (".", "..")
                    or shard_size is None
                    or leaf.offset + leaf.length > shard_size
                ):
                    raise SparseBundleError(
                        f"access trace leaf is outside pinned inventory: {path}"
                    )
                trace_leaves.add((leaf.shard, leaf.offset, leaf.length))
        if not trace_leaves:
            raise SparseBundleError(f"access trace is empty: {path}")
        prior = evidence_by_sha256.get(trace.sha256)
        evidence = {
            "operation_count": len(trace.operations),
            "sha256": trace.sha256,
            "unique_leaf_count": len(trace_leaves),
            "unique_leaves_sha256": _sha256(
                [
                    {"length": length, "offset": offset, "shard": shard}
                    for shard, offset, length in sorted(trace_leaves)
                ]
            ),
        }
        if prior is not None and prior != evidence:
            raise SparseBundleError("duplicate access trace SHA conflicts")
        evidence_by_sha256[trace.sha256] = evidence
        leaves.update(trace_leaves)

    intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for shard, offset, length in leaves:
        intervals[shard].append((offset, offset + length))
    merged = {
        shard: _merge_intervals(ranges) for shard, ranges in sorted(intervals.items())
    }
    return (
        tuple(evidence_by_sha256[digest] for digest in sorted(evidence_by_sha256)),
        merged,
        tuple(sorted(leaves)),
    )


def _revision_record(value: tuple[int, str]) -> dict[str, Any]:
    sequence, digest = value
    if (
        isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence < 0
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        raise SparseBundleError("causal graph revision is invalid")
    return {"sequence": sequence, "sha256": digest}


def _expert_plan_record(plan: OfficialExpertRangePlan) -> dict[str, Any]:
    return {
        "base": plan.base,
        "expert_id": plan.expert_id,
        "layer": plan.layer,
        "payload_bytes": plan.payload_bytes,
        "ranges": [
            {
                "absolute_offset": source_range.absolute_offset,
                "length": source_range.length,
                "shard": source_range.shard,
                "tensors": [
                    {
                        "absolute_offset": tensor.absolute_offset,
                        "dtype": tensor.dtype,
                        "length": tensor.length,
                        "name": tensor.name,
                        "range_offset": tensor.range_offset,
                        "shape": list(tensor.shape),
                    }
                    for tensor in source_range.tensors
                ],
            }
            for source_range in plan.ranges
        ],
    }


def _expert_plan_from_record(value: object) -> OfficialExpertRangePlan:
    if not isinstance(value, Mapping) or set(value) != {
        "base",
        "expert_id",
        "layer",
        "payload_bytes",
        "ranges",
    }:
        raise SparseBundleError("stored append expert plan schema is invalid")
    raw_ranges = value.get("ranges")
    if not isinstance(raw_ranges, list):
        raise SparseBundleError("stored append expert ranges are invalid")
    ranges: list[ExpertSourceRange] = []
    try:
        for raw_range in raw_ranges:
            if not isinstance(raw_range, Mapping) or set(raw_range) != {
                "absolute_offset",
                "length",
                "shard",
                "tensors",
            }:
                raise SparseBundleError("stored append source range is invalid")
            raw_tensors = raw_range.get("tensors")
            if not isinstance(raw_tensors, list):
                raise SparseBundleError("stored append tensor table is invalid")
            tensors = tuple(
                ExpertTensorLayout(
                    name=tensor["name"],
                    dtype=tensor["dtype"],
                    shape=tuple(tensor["shape"]),
                    absolute_offset=tensor["absolute_offset"],
                    length=tensor["length"],
                    range_offset=tensor["range_offset"],
                )
                for tensor in raw_tensors
                if isinstance(tensor, Mapping)
                and set(tensor)
                == {
                    "absolute_offset",
                    "dtype",
                    "length",
                    "name",
                    "range_offset",
                    "shape",
                }
            )
            if len(tensors) != len(raw_tensors):
                raise SparseBundleError("stored append tensor table is invalid")
            ranges.append(
                ExpertSourceRange(
                    shard=raw_range["shard"],
                    absolute_offset=raw_range["absolute_offset"],
                    length=raw_range["length"],
                    tensors=tensors,
                )
            )
        plan = OfficialExpertRangePlan(
            base=value["base"],
            layer=value["layer"],
            expert_id=value["expert_id"],
            ranges=tuple(ranges),
            payload_bytes=value["payload_bytes"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SparseBundleError("stored append expert plan is invalid") from exc
    if _expert_plan_record(plan) != dict(value):
        raise SparseBundleError("stored append expert plan is noncanonical")
    _expert_leaf_records((plan,))
    return plan


def _expert_leaf_records(
    plans: Sequence[OfficialExpertRangePlan],
) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    occupied: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for plan_index, plan in enumerate(plans):
        expected = {
            f"{plan.base}.w{index}.{kind}"
            for index in (1, 2, 3)
            for kind in ("scale", "weight")
        }
        observed = {
            tensor.name
            for source_range in plan.ranges
            for tensor in source_range.tensors
        }
        if observed != expected or len(observed) != 6:
            raise SparseBundleError(
                f"expert plan for {plan.base} does not contain exactly six leaves"
            )
        for range_index, source_range in enumerate(plan.ranges):
            if Path(source_range.shard).name != source_range.shard:
                raise SparseBundleError("expert shard must be a root-level basename")
            for tensor_index, tensor in enumerate(source_range.tensors):
                stop = tensor.absolute_offset + tensor.length
                for prior_start, prior_stop in occupied[source_range.shard]:
                    if prior_start < stop and tensor.absolute_offset < prior_stop:
                        raise SparseBundleError("expert leaf plans overlap")
                occupied[source_range.shard].append((tensor.absolute_offset, stop))
                descriptor = {
                    "absolute_offset": tensor.absolute_offset,
                    "dtype": tensor.dtype,
                    "expert_id": plan.expert_id,
                    "layer": plan.layer,
                    "length": tensor.length,
                    "name": tensor.name,
                    "plan_index": plan_index,
                    "range_index": range_index,
                    "shape": list(tensor.shape),
                    "shard": source_range.shard,
                    "tensor_index": tensor_index,
                }
                records.append(
                    {
                        **descriptor,
                        "leaf_sha256": _sha256(descriptor),
                    }
                )
    return tuple(records)


def _parse_expert_coordinate(raw: str) -> tuple[int, int]:
    if not isinstance(raw, str):
        raise argparse.ArgumentTypeError("expert coordinate must be LAYER:EXPERT")
    match = re.fullmatch(r"(0|[1-9][0-9]*):(0|[1-9][0-9]*)", raw)
    if match is None:
        raise argparse.ArgumentTypeError("expert coordinate must be LAYER:EXPERT")
    return int(match.group(1)), int(match.group(2))


def _normalize_coordinates(
    values: Iterable[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    unique: set[tuple[int, int]] = set()
    ordered: list[tuple[int, int]] = []
    for raw_layer, raw_expert in values:
        if (
            isinstance(raw_layer, bool)
            or not isinstance(raw_layer, int)
            or raw_layer < 0
            or isinstance(raw_expert, bool)
            or not isinstance(raw_expert, int)
            or raw_expert < 0
        ):
            raise SparseBundleError("expert coordinates must be non-negative integers")
        coordinate = (raw_layer, raw_expert)
        if coordinate not in unique:
            unique.add(coordinate)
            ordered.append(coordinate)
    if not ordered:
        raise SparseBundleError("at least one explicit expert coordinate is required")
    return tuple(ordered)


def _verify_sparse_bundle_manifest(document: object) -> dict[str, Any]:
    if not isinstance(document, dict) or document.get("schema") != BUNDLE_SCHEMA:
        raise SparseBundleError("bundle manifest schema is invalid")
    identity = {key: value for key, value in document.items() if key != "sha256"}
    if document.get("sha256") != _sha256(identity):
        raise SparseBundleError("bundle manifest digest does not match")
    return dict(document)


def _verify_general_bundle_manifest(document: object) -> dict[str, Any]:
    if (
        not isinstance(document, Mapping)
        or set(document) != {"body", "schema", "sha256"}
        or document.get("schema") != GENERAL_BUNDLE_SCHEMA
    ):
        raise SparseBundleError("general bundle manifest schema is invalid")
    body = document.get("body")
    if (
        not isinstance(body, Mapping)
        or set(body)
        != {
            "base_bundle",
            "capabilities",
            "dense_coverage",
            "layout_fingerprint",
            "logical_model",
            "weights_layout",
        }
        or document.get("sha256") != _sha256(body)
    ):
        raise SparseBundleError("general bundle manifest identity is invalid")
    capabilities = body.get("capabilities")
    if capabilities != {
        "general_dense_weight_coverage": GENERAL_DENSE_COVERAGE_CAPABILITY
    }:
        raise SparseBundleError("general dense coverage capability is invalid")
    if body.get("weights_layout") != "nested/v1":
        raise SparseBundleError("general bundle weights layout is invalid")
    base = _verify_sparse_bundle_manifest(body.get("base_bundle"))
    if body.get("logical_model") != base.get("logical_model") or body.get(
        "layout_fingerprint"
    ) != base.get("layout_fingerprint"):
        raise SparseBundleError("general bundle base identity is inconsistent")
    coverage = body.get("dense_coverage")
    if (
        not isinstance(coverage, Mapping)
        or set(coverage)
        != {
            "graph_post_bind",
            "materialization_bytes",
            "payload_ledger_sha256",
            "plans_sha256",
            "receipt_sha256",
            "required_tensor_count",
            "required_tensor_names_sha256",
            "transaction_id",
        }
        or not isinstance(coverage.get("materialization_bytes"), int)
        or int(coverage["materialization_bytes"]) < 0
        or not isinstance(coverage.get("required_tensor_count"), int)
        or int(coverage["required_tensor_count"]) < 1
    ):
        raise SparseBundleError("general bundle dense coverage record is invalid")
    for key in (
        "payload_ledger_sha256",
        "plans_sha256",
        "receipt_sha256",
        "required_tensor_names_sha256",
        "transaction_id",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(coverage.get(key))) is None:
            raise SparseBundleError("general bundle dense coverage digest is invalid")
    graph_post = coverage.get("graph_post_bind")
    if not isinstance(graph_post, Mapping) or set(graph_post) != {
        "sequence",
        "sha256",
    }:
        raise SparseBundleError("general bundle graph revision is invalid")
    _revision_record((graph_post["sequence"], graph_post["sha256"]))
    return dict(document)


def _bundle_documents_at(
    root_descriptor: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    document, _encoded = _read_json_at(root_descriptor, "bundle.json")
    if isinstance(document, Mapping) and document.get("schema") == BUNDLE_SCHEMA:
        return _verify_sparse_bundle_manifest(document), None
    promoted = _verify_general_bundle_manifest(document)
    body = promoted["body"]
    assert isinstance(body, Mapping)
    return _verify_sparse_bundle_manifest(body["base_bundle"]), promoted


def _load_bundle_manifest_at(
    root_descriptor: int,
    *,
    repo_id: str,
    revision: str,
    expected_layout: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    document, _promoted = _bundle_documents_at(root_descriptor)
    logical = document.get("logical_model")
    if logical != {"repo_id": repo_id, "revision": revision}:
        raise SparseBundleError("bundle logical model identity does not match request")
    layout = document.get("layout_fingerprint")
    if (
        not isinstance(layout, str)
        or re.fullmatch(r"[0-9a-f]{64}", layout) is None
        or (expected_layout is not None and layout != expected_layout)
    ):
        raise SparseBundleError("bundle layout fingerprint does not match request")
    encoded = _canonical(document)
    manifest_identity = {
        "file_sha256": _sha256_bytes(encoded),
        "manifest_sha256": document["sha256"],
        "schema": document["schema"],
    }
    return document, manifest_identity


def _assert_no_dense_pending_at(root_descriptor: int) -> None:
    try:
        linked = os.stat(
            _DENSE_DIRECTORY,
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(linked.st_mode):
        raise SparseBundleError("dense promotion store is not a directory")
    descriptor = _open_directory_at(root_descriptor, _DENSE_DIRECTORY)
    try:
        pending = _optional_json_at(descriptor, _DENSE_PENDING)
        if pending is not None:
            _verify_dense_pending(pending[0])
            raise SparseBundleError(
                "dense promotion is pending; reconcile it before expert append"
            )
    finally:
        os.close(descriptor)


def _open_remote_source(args: argparse.Namespace) -> Streamer:
    use_cache = bool(getattr(args, "remote_cache_dir", None))
    return Streamer(
        args.repo_id,
        revision=args.revision,
        budget_mb=float(args.budget_mb),
        cache_dir=getattr(args, "remote_cache_dir", None),
        use_cache=use_cache,
        max_cache_bytes=(
            int(float(args.remote_cache_limit_mb) * 1024 * 1024) if use_cache else None
        ),
    )


def _adopt_bundle_inventory(
    source: Streamer,
    weights_descriptor: int,
    manifest: Mapping[str, Any],
) -> None:
    document, encoded = _read_json_at(weights_descriptor, "inventory.pinned.json")
    if (
        _sha256_bytes(encoded) != manifest.get("pinned_inventory_sha256")
        or not isinstance(document, Mapping)
        or document.get("schema") != "immer.tensor-inventory-cache/v1"
        or document.get("source_fingerprint") != manifest.get("layout_fingerprint")
        or not isinstance(document.get("inventory"), Mapping)
    ):
        raise SparseBundleError("bundle pinned inventory identity is invalid")
    inventory = document["inventory"]
    if (
        inventory.get("repo") != manifest["logical_model"]["repo_id"]
        or inventory.get("revision") != manifest["logical_model"]["revision"]
    ):
        raise SparseBundleError("bundle pinned inventory logical model is invalid")
    try:
        source.adopt_pinned_inventory(
            inventory,
            expected_fingerprint=str(manifest["layout_fingerprint"]),
        )
    except Exception as exc:
        raise SparseBundleError(
            "official remote Streamer rejected the pinned bundle inventory"
        ) from exc


def _plan_remote_experts(
    source: Streamer,
    coordinates: Sequence[tuple[int, int]],
    *,
    repo_id: str,
    revision: str,
    layout_fingerprint: str,
) -> tuple[tuple[OfficialExpertRangePlan, ...], dict[str, Any]]:
    inventory = source.inventory()
    metrics = source.metrics()
    if repo_id == OFFICIAL_SOURCE and metrics.get("revision_is_pinned") is not True:
        raise SparseBundleError("official remote Streamer revision is not immutable")
    if (
        metrics.get("repo_id") != repo_id
        or metrics.get("revision") != revision
        or metrics.get("inventory_source_fingerprint") != layout_fingerprint
    ):
        raise SparseBundleError("remote Streamer identity does not match the bundle")
    pager = DeepSeekWeightPager(
        source,
        device="cpu",
        compute_dtype="bfloat16",
        expert_prefetch=False,
    )
    plans = tuple(
        pager.plan_expert_ranges(layer, (expert_id,))[0]
        for layer, expert_id in coordinates
    )
    if tuple((plan.layer, plan.expert_id) for plan in plans) != tuple(coordinates):
        raise SparseBundleError("remote expert plan coordinates changed")
    _expert_leaf_records(plans)
    source_identity = {
        "inventory_layout_sha256": _sha256(
            Streamer._inventory_layout_projection(inventory)
        ),
        "layout_fingerprint": layout_fingerprint,
        "repo_id": repo_id,
        "revision": revision,
        "transport": "official-pinned-streamer/v1",
    }
    return plans, source_identity


def _preflight_graph(
    root: Path,
    model: LogicalModelIdentity,
    plans: Sequence[OfficialExpertRangePlan],
    *,
    layout_fingerprint: str,
    budget_mb: float,
) -> dict[str, Any]:
    with CausalWeightMount(root, model, budget_mb=budget_mb) as mount:
        if mount.layout.layout_fingerprint != layout_fingerprint:
            raise SparseBundleError("mounted bundle layout does not match manifest")
        revision = _revision_record(mount.graph.store.revision())
        for plan in plans:
            try:
                existing = mount.resolve_expert_plans(plan.layer, (plan.expert_id,))
            except KeyError:
                continue
            if existing != (plan,):
                raise SparseBundleError(
                    f"causal graph conflicts at {plan.layer}:{plan.expert_id}"
                )
        return revision


def _open_shard_at(weights_descriptor: int, shard: str, *, writable: bool) -> int:
    if Path(shard).name != shard or shard in (".", ".."):
        raise SparseBundleError("shard is not a safe basename")
    flags = os.O_RDWR if writable else os.O_RDONLY
    flags |= int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(shard, flags, dir_fd=weights_descriptor)
    except OSError as exc:
        raise SparseBundleError(f"cannot open sparse shard {shard!r}") from exc
    try:
        opened = os.fstat(descriptor)
        linked = os.stat(shard, dir_fd=weights_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(linked.st_mode)
            or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
        ):
            raise SparseBundleError(f"sparse shard {shard!r} is not stable")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _shard_stats(weights_descriptor: int, shards: Iterable[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for shard in sorted(set(shards)):
        descriptor = _open_shard_at(weights_descriptor, shard, writable=False)
        try:
            metadata = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        rows.append(
            {
                "file": shard,
                "logical_bytes": metadata.st_size,
                "physical_bytes": metadata.st_blocks * 512,
            }
        )
    return {
        "logical_bytes": sum(row["logical_bytes"] for row in rows),
        "physical_bytes": sum(row["physical_bytes"] for row in rows),
        "shards": rows,
    }


def _payload_filename(leaf: Mapping[str, Any]) -> str:
    digest = leaf.get("leaf_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise SparseBundleError("expert leaf identity is invalid")
    return f"{digest}.bin"


def _atomic_payload_at(directory: int, name: str, payload: memoryview) -> None:
    temporary = f".{name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, flags, 0o600, dir_fd=directory)
        cursor = 0
        while cursor < len(payload):
            written = os.write(descriptor, payload[cursor : cursor + 1024 * 1024])
            if written <= 0:
                raise SparseBundleError("short staged expert write")
            cursor += written
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_size != len(payload):
            raise SparseBundleError("staged expert payload is invalid")
        os.close(descriptor)
        descriptor = None
        os.replace(
            temporary,
            name,
            src_dir_fd=directory,
            dst_dir_fd=directory,
        )
        os.fsync(directory)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory)
        except FileNotFoundError:
            pass


def _hash_file_at(directory: int, name: str, expected_length: int) -> str:
    flags = os.O_RDONLY | int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(name, flags, dir_fd=directory)
    except OSError as exc:
        raise SparseBundleError(f"staged payload {name!r} is missing") from exc
    try:
        opened = os.fstat(descriptor)
        linked = os.stat(name, dir_fd=directory, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(linked.st_mode)
            or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
            or opened.st_size != expected_length
        ):
            raise SparseBundleError("staged payload is not a stable exact file")
        digest = hashlib.sha256()
        cursor = 0
        while cursor < expected_length:
            chunk = os.pread(
                descriptor,
                min(1024 * 1024, expected_length - cursor),
                cursor,
            )
            if not chunk:
                raise SparseBundleError("staged payload returned a short read")
            digest.update(chunk)
            cursor += len(chunk)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _stage_remote_payloads(
    source: Streamer,
    stage_descriptor: int,
    leaves: Sequence[Mapping[str, Any]],
    *,
    resident_limit_bytes: int,
    staging_limit_bytes: int,
    source_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    total = sum(int(leaf["length"]) for leaf in leaves)
    largest = max(int(leaf["length"]) for leaf in leaves)
    if largest > resident_limit_bytes:
        raise SparseBundleError("one expert leaf exceeds the resident memory bound")
    if total > staging_limit_bytes:
        raise SparseBundleError("expert payload exceeds the staging disk bound")
    available = (
        os.fstatvfs(stage_descriptor).f_bavail * os.fstatvfs(stage_descriptor).f_frsize
    )
    if total + 8 * 1024 * 1024 > available:
        raise SparseBundleError("insufficient free disk for the complete expert stage")

    staged: list[dict[str, Any]] = []
    for leaf in leaves:
        raw = source.raw_bytes(
            str(leaf["shard"]),
            int(leaf["absolute_offset"]),
            int(leaf["length"]),
        )
        try:
            view = raw if isinstance(raw, memoryview) else memoryview(raw)
        except TypeError as exc:
            raise SparseBundleError(
                "remote Streamer returned a non-buffer leaf"
            ) from exc
        if len(view) != int(leaf["length"]):
            raise SparseBundleError("remote Streamer returned a partial expert leaf")
        filename = _payload_filename(leaf)
        _atomic_payload_at(stage_descriptor, filename, view)
        digest = hashlib.sha256(view).hexdigest()
        if _hash_file_at(stage_descriptor, filename, len(view)) != digest:
            raise SparseBundleError("staged expert readback digest does not match")
        staged.append(
            {
                "filename": filename,
                "leaf_sha256": leaf["leaf_sha256"],
                "length": len(view),
                "sha256": digest,
            }
        )
    metrics = source.metrics()
    if (
        metrics.get("repo_id") != source_identity["repo_id"]
        or metrics.get("revision") != source_identity["revision"]
        or metrics.get("inventory_source_fingerprint")
        != source_identity["layout_fingerprint"]
    ):
        raise SparseBundleError("remote Streamer identity changed during download")
    return tuple(staged)


def _verify_staged_payloads(
    stage_descriptor: int,
    leaves: Sequence[Mapping[str, Any]],
    staged: object,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(staged, list) or len(staged) != len(leaves):
        raise SparseBundleError("staged expert table is incomplete")
    verified: list[dict[str, Any]] = []
    for leaf, raw in zip(leaves, staged, strict=True):
        if not isinstance(raw, Mapping) or set(raw) != {
            "filename",
            "leaf_sha256",
            "length",
            "sha256",
        }:
            raise SparseBundleError("staged expert record schema is invalid")
        if (
            raw.get("leaf_sha256") != leaf.get("leaf_sha256")
            or raw.get("filename") != _payload_filename(leaf)
            or raw.get("length") != leaf.get("length")
            or not isinstance(raw.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", str(raw.get("sha256"))) is None
        ):
            raise SparseBundleError("staged expert record identity is invalid")
        observed = _hash_file_at(
            stage_descriptor,
            str(raw["filename"]),
            int(raw["length"]),
        )
        if observed != raw["sha256"]:
            raise SparseBundleError("staged expert payload SHA-256 mismatch")
        verified.append(dict(raw))
    return tuple(verified)


def _pread_exact(descriptor: int, length: int, offset: int) -> bytes:
    encoded = os.pread(descriptor, length, offset)
    if len(encoded) != length:
        raise SparseBundleError("sparse shard returned a short read")
    return encoded


def _hash_descriptor_range(descriptor: int, offset: int, length: int) -> str:
    digest = hashlib.sha256()
    cursor = 0
    while cursor < length:
        chunk = _pread_exact(
            descriptor,
            min(1024 * 1024, length - cursor),
            offset + cursor,
        )
        digest.update(chunk)
        cursor += len(chunk)
    return digest.hexdigest()


def _materialize_local_payload(
    weights_descriptor: int,
    stage_descriptor: int,
    leaves: Sequence[Mapping[str, Any]],
    staged: Sequence[Mapping[str, Any]],
) -> None:
    shard_descriptors: dict[str, int] = {}
    shard_identities: dict[str, tuple[int, int]] = {}
    stage_descriptors: dict[str, int] = {}
    try:
        for leaf in leaves:
            shard = str(leaf["shard"])
            if shard not in shard_descriptors:
                descriptor = _open_shard_at(
                    weights_descriptor,
                    shard,
                    writable=True,
                )
                opened = os.fstat(descriptor)
                shard_descriptors[shard] = descriptor
                shard_identities[shard] = (opened.st_dev, opened.st_ino)
        for payload in staged:
            filename = str(payload["filename"])
            flags = os.O_RDONLY | int(getattr(os, "O_CLOEXEC", 0))
            flags |= int(getattr(os, "O_NOFOLLOW", 0))
            descriptor = os.open(filename, flags, dir_fd=stage_descriptor)
            opened = os.fstat(descriptor)
            linked = os.stat(
                filename,
                dir_fd=stage_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(opened.st_mode)
                or not stat.S_ISREG(linked.st_mode)
                or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
                or opened.st_size != int(payload["length"])
                or _hash_descriptor_range(descriptor, 0, opened.st_size)
                != payload["sha256"]
            ):
                os.close(descriptor)
                raise SparseBundleError("staged payload changed before materialization")
            stage_descriptors[filename] = descriptor

        # Validate every target before the first pwrite. A partially completed
        # prior transaction is compatible iff every nonzero local byte already
        # equals the sealed source byte; any other occupied byte is a conflict.
        for leaf, payload in zip(leaves, staged, strict=True):
            shard = str(leaf["shard"])
            shard_fd = shard_descriptors[shard]
            stage_fd = stage_descriptors[str(payload["filename"])]
            length = int(leaf["length"])
            absolute = int(leaf["absolute_offset"])
            if absolute + length > os.fstat(shard_fd).st_size:
                raise SparseBundleError("expert leaf exceeds sparse shard size")
            cursor = 0
            while cursor < length:
                take = min(1024 * 1024, length - cursor)
                local = _pread_exact(shard_fd, take, absolute + cursor)
                expected = _pread_exact(stage_fd, take, cursor)
                if any(
                    observed not in (0, target)
                    for observed, target in zip(local, expected, strict=True)
                ):
                    raise SparseBundleError(
                        f"occupied nonmatching bytes at {shard}[{absolute}:{absolute + length}]"
                    )
                cursor += take

        for leaf, payload in zip(leaves, staged, strict=True):
            shard_fd = shard_descriptors[str(leaf["shard"])]
            stage_fd = stage_descriptors[str(payload["filename"])]
            cursor = 0
            length = int(leaf["length"])
            absolute = int(leaf["absolute_offset"])
            while cursor < length:
                take = min(1024 * 1024, length - cursor)
                chunk = _pread_exact(stage_fd, take, cursor)
                written = os.pwrite(shard_fd, chunk, absolute + cursor)
                if written != len(chunk):
                    raise SparseBundleError("short sparse expert pwrite")
                cursor += written
        for descriptor in shard_descriptors.values():
            os.fsync(descriptor)
        os.fsync(weights_descriptor)
        for leaf, payload in zip(leaves, staged, strict=True):
            shard = str(leaf["shard"])
            descriptor = shard_descriptors[shard]
            if (
                _hash_descriptor_range(
                    descriptor,
                    int(leaf["absolute_offset"]),
                    int(leaf["length"]),
                )
                != payload["sha256"]
            ):
                raise SparseBundleError("durable expert pwrite digest does not match")
        for shard, descriptor in shard_descriptors.items():
            linked = os.stat(
                shard,
                dir_fd=weights_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(linked.st_mode)
                or (linked.st_dev, linked.st_ino) != shard_identities[shard]
            ):
                raise SparseBundleError("sparse shard path changed during pwrite")
    finally:
        for descriptor in stage_descriptors.values():
            os.close(descriptor)
        for descriptor in shard_descriptors.values():
            os.close(descriptor)


def _hash_local_leaf(
    weights_descriptor: int,
    leaf: Mapping[str, Any],
) -> str:
    descriptor = _open_shard_at(
        weights_descriptor,
        str(leaf["shard"]),
        writable=False,
    )
    try:
        digest = hashlib.sha256()
        cursor = 0
        length = int(leaf["length"])
        absolute = int(leaf["absolute_offset"])
        while cursor < length:
            chunk = _pread_exact(
                descriptor,
                min(1024 * 1024, length - cursor),
                absolute + cursor,
            )
            digest.update(chunk)
            cursor += len(chunk)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _verify_local_payloads(
    weights_descriptor: int,
    leaves: Sequence[Mapping[str, Any]],
    staged: Sequence[Mapping[str, Any]],
) -> None:
    for leaf, payload in zip(leaves, staged, strict=True):
        if _hash_local_leaf(weights_descriptor, leaf) != payload["sha256"]:
            raise SparseBundleError("durable expert payload SHA-256 mismatch")


def _failpoint(args: argparse.Namespace, name: str) -> None:
    if getattr(args, "inject_crash", None) == name:
        raise InjectedAppendCrash(f"injected crash at {name}")


def _receipt_document(
    transaction: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    final = history[-1]
    if final.get("state") != "binding_visible":
        raise SparseBundleError("cannot seal receipt before binding visibility")
    body = {
        "journal": {
            "final_sha256": final["sha256"],
            "history": list(history),
        },
        "result": dict(final["payload"]),
        "transaction": dict(transaction),
    }
    return {
        "body": body,
        "schema": APPEND_RECEIPT_SCHEMA,
        "sha256": _sha256(body),
    }


def _verify_receipt(document: object) -> dict[str, Any]:
    if not isinstance(document, Mapping) or set(document) != {
        "body",
        "schema",
        "sha256",
    }:
        raise SparseBundleError("append receipt schema is invalid")
    body = document.get("body")
    if (
        document.get("schema") != APPEND_RECEIPT_SCHEMA
        or not isinstance(body, Mapping)
        or set(body) != {"journal", "result", "transaction"}
        or document.get("sha256") != _sha256(body)
        or not isinstance(body.get("journal"), Mapping)
        or not isinstance(body.get("result"), Mapping)
        or not isinstance(body.get("transaction"), Mapping)
    ):
        raise SparseBundleError("append receipt identity is invalid")
    journal = body["journal"]
    if set(journal) != {"final_sha256", "history"}:
        raise SparseBundleError("append receipt journal schema is invalid")
    history = _verify_journal(journal.get("history"))
    if (
        history[-1].get("state") != "binding_visible"
        or journal.get("final_sha256") != history[-1].get("sha256")
        or dict(history[-1]["payload"]) != dict(body["result"])
    ):
        raise SparseBundleError("append receipt journal/result binding is invalid")
    return dict(body)


def _validate_transaction_schema(transaction: Mapping[str, Any]) -> None:
    required = {
        "bundle_manifest",
        "coordinates",
        "leaves",
        "leaves_sha256",
        "plans",
        "plans_sha256",
        "remote_source",
        "schema",
        "transaction_id",
    }
    if set(transaction) != required or transaction.get("schema") != (
        "immer.deepseek-v4-expert-append-transaction/v1"
    ):
        raise SparseBundleError("append transaction schema is invalid")
    core = {key: transaction[key] for key in transaction if key != "transaction_id"}
    if transaction.get("transaction_id") != _sha256(core):
        raise SparseBundleError("append transaction digest is invalid")
    leaves = transaction.get("leaves")
    plans = transaction.get("plans")
    coordinates = transaction.get("coordinates")
    if (
        not isinstance(leaves, list)
        or not leaves
        or len(leaves) % 6
        or transaction.get("leaves_sha256") != _sha256(leaves)
        or not isinstance(plans, list)
        or not plans
        or transaction.get("plans_sha256") != _sha256(plans)
        or not isinstance(coordinates, list)
        or len(coordinates) != len(plans)
        or len(leaves) != 6 * len(plans)
    ):
        raise SparseBundleError("append transaction plan/leaf identity is invalid")
    bundle_manifest = transaction.get("bundle_manifest")
    remote_source = transaction.get("remote_source")
    if (
        not isinstance(bundle_manifest, Mapping)
        or set(bundle_manifest) != {"file_sha256", "manifest_sha256", "schema"}
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(bundle_manifest.get(key))) is None
            for key in ("file_sha256", "manifest_sha256")
        )
        or not isinstance(remote_source, Mapping)
        or set(remote_source)
        != {
            "inventory_layout_sha256",
            "layout_fingerprint",
            "repo_id",
            "revision",
            "transport",
        }
        or remote_source.get("transport") != "official-pinned-streamer/v1"
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(remote_source.get(key))) is None
            for key in ("inventory_layout_sha256", "layout_fingerprint")
        )
    ):
        raise SparseBundleError("append transaction source/bundle identity is invalid")

    expected_leaves: list[dict[str, Any]] = []
    for plan_index, (coordinate, plan) in enumerate(
        zip(coordinates, plans, strict=True)
    ):
        if (
            not isinstance(coordinate, Mapping)
            or set(coordinate) != {"expert_id", "layer"}
            or not isinstance(plan, Mapping)
            or set(plan) != {"base", "expert_id", "layer", "payload_bytes", "ranges"}
            or (coordinate.get("layer"), coordinate.get("expert_id"))
            != (plan.get("layer"), plan.get("expert_id"))
            or not isinstance(plan.get("ranges"), list)
        ):
            raise SparseBundleError("append transaction plan coordinate is invalid")
        for range_index, source_range in enumerate(plan["ranges"]):
            if (
                not isinstance(source_range, Mapping)
                or set(source_range)
                != {"absolute_offset", "length", "shard", "tensors"}
                or not isinstance(source_range.get("tensors"), list)
            ):
                raise SparseBundleError("append transaction source range is invalid")
            for tensor_index, tensor in enumerate(source_range["tensors"]):
                if not isinstance(tensor, Mapping) or set(tensor) != {
                    "absolute_offset",
                    "dtype",
                    "length",
                    "name",
                    "range_offset",
                    "shape",
                }:
                    raise SparseBundleError("append transaction tensor plan is invalid")
                descriptor = {
                    "absolute_offset": tensor["absolute_offset"],
                    "dtype": tensor["dtype"],
                    "expert_id": plan["expert_id"],
                    "layer": plan["layer"],
                    "length": tensor["length"],
                    "name": tensor["name"],
                    "plan_index": plan_index,
                    "range_index": range_index,
                    "shape": tensor["shape"],
                    "shard": source_range["shard"],
                    "tensor_index": tensor_index,
                }
                expected_leaves.append(
                    {**descriptor, "leaf_sha256": _sha256(descriptor)}
                )
    if expected_leaves != leaves:
        raise SparseBundleError("append transaction leaves do not derive from plans")
    for index, leaf in enumerate(leaves):
        if not isinstance(leaf, Mapping):
            raise SparseBundleError("append transaction leaf is invalid")
        descriptor = {key: leaf[key] for key in leaf if key != "leaf_sha256"}
        plan_index = leaf.get("plan_index")
        if (
            isinstance(plan_index, bool)
            or not isinstance(plan_index, int)
            or plan_index not in range(len(plans))
            or leaf.get("leaf_sha256") != _sha256(descriptor)
            or not isinstance(leaf.get("length"), int)
            or int(leaf["length"]) <= 0
            or not isinstance(leaf.get("absolute_offset"), int)
            or int(leaf["absolute_offset"]) < 0
        ):
            raise SparseBundleError(f"append transaction leaf {index} is invalid")


def _verify_result_schema(result: Mapping[str, Any]) -> None:
    required = {
        "binding_receipts",
        "binding_receipts_sha256",
        "graph_post_bind",
        "graph_pre_bind",
        "graph_pre_transaction",
        "growth",
        "staged_payloads",
        "staged_payloads_sha256",
    }
    if set(result) != required:
        raise SparseBundleError("append result schema is invalid")
    bindings = result.get("binding_receipts")
    staged = result.get("staged_payloads")
    if (
        not isinstance(bindings, list)
        or not bindings
        or result.get("binding_receipts_sha256") != _sha256(bindings)
        or not isinstance(staged, list)
        or not staged
        or result.get("staged_payloads_sha256") != _sha256(staged)
    ):
        raise SparseBundleError("append result binding/payload digest is invalid")
    for key in ("graph_pre_transaction", "graph_pre_bind", "graph_post_bind"):
        revision = result.get(key)
        if not isinstance(revision, Mapping) or set(revision) != {"sequence", "sha256"}:
            raise SparseBundleError("append result graph revision is invalid")
        _revision_record((revision["sequence"], revision["sha256"]))
    growth = result.get("growth")
    if not isinstance(growth, Mapping) or set(growth) != {
        "logical_bytes_after",
        "logical_bytes_before",
        "logical_bytes_delta",
        "physical_bytes_after",
        "physical_bytes_before",
        "physical_bytes_delta",
        "shards_after",
        "shards_before",
    }:
        raise SparseBundleError("append result growth receipt is invalid")
    if (
        growth["logical_bytes_delta"]
        != growth["logical_bytes_after"] - growth["logical_bytes_before"]
        or growth["physical_bytes_delta"]
        != growth["physical_bytes_after"] - growth["physical_bytes_before"]
        or growth["logical_bytes_delta"] != 0
        or growth["physical_bytes_delta"] < 0
    ):
        raise SparseBundleError("append result physical/logical growth is invalid")


def _verify_completed_receipt_payload(
    body: Mapping[str, Any],
    weights_descriptor: int,
    root: Path,
    model: LogicalModelIdentity,
    *,
    budget_mb: float,
) -> int:
    transaction = body.get("transaction")
    result = body.get("result")
    if not isinstance(transaction, Mapping) or not isinstance(result, Mapping):
        raise SparseBundleError("append receipt body is incomplete")
    _validate_transaction_schema(transaction)
    _verify_result_schema(result)
    leaves = transaction["leaves"]
    staged = result["staged_payloads"]
    if len(staged) != len(leaves):
        raise SparseBundleError("append receipt lost staged payload hashes")
    for leaf, payload in zip(leaves, staged, strict=True):
        if (
            not isinstance(payload, Mapping)
            or payload.get("leaf_sha256") != leaf.get("leaf_sha256")
            or payload.get("length") != leaf.get("length")
            or payload.get("filename") != _payload_filename(leaf)
            or _hash_local_leaf(weights_descriptor, leaf) != payload.get("sha256")
        ):
            raise SparseBundleError("appended expert data hash does not match receipt")

    plan_records = transaction["plans"]
    coordinates = transaction["coordinates"]
    bindings = result["binding_receipts"]
    if len(bindings) != len(plan_records):
        raise SparseBundleError("append receipt binding count does not match plans")
    with CausalWeightMount(root, model, budget_mb=budget_mb) as mount:
        remote = transaction["remote_source"]
        if not isinstance(
            remote, Mapping
        ) or mount.layout.layout_fingerprint != remote.get("layout_fingerprint"):
            raise SparseBundleError("append receipt layout no longer mounts")
        current_revision = _revision_record(mount.graph.store.revision())
        active_segments = set(mount.graph.store.segments())
        post_revision = result["graph_post_bind"]
        if current_revision["sequence"] < post_revision["sequence"]:
            raise SparseBundleError("causal graph predates an append receipt")
        for coordinate, plan_record, binding in zip(
            coordinates,
            plan_records,
            bindings,
            strict=True,
        ):
            if (
                not isinstance(coordinate, Mapping)
                or set(coordinate) != {"expert_id", "layer"}
                or not isinstance(binding, Mapping)
                or set(binding)
                != {
                    "appended",
                    "expert_id",
                    "layer",
                    "record_index",
                    "segment_sha256",
                }
                or (binding.get("layer"), binding.get("expert_id"))
                != (coordinate.get("layer"), coordinate.get("expert_id"))
                or isinstance(binding.get("record_index"), bool)
                or not isinstance(binding.get("record_index"), int)
                or int(binding["record_index"]) < 0
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(binding.get("segment_sha256")),
                )
                is None
            ):
                raise SparseBundleError("append receipt coordinate binding is invalid")
            segment = str(binding["segment_sha256"])
            if segment not in active_segments:
                raise SparseBundleError("append receipt citation is not active")
            cited = mount.graph.store.record(segment, int(binding["record_index"]))
            if (
                cited.get("schema") != "causal-weight-binding/v1"
                or cited.get("record_type") != "expert_range_plan"
                or cited.get("logical_model") != model.as_record()
                or cited.get("layout_fingerprint") != remote.get("layout_fingerprint")
                or cited.get("plan") != plan_record
            ):
                raise SparseBundleError("append receipt citation record does not match")
            resolved = mount.resolve_expert_plans(
                int(coordinate["layer"]),
                (int(coordinate["expert_id"]),),
            )
            if len(resolved) != 1 or _expert_plan_record(resolved[0]) != plan_record:
                raise SparseBundleError("appended causal binding plan does not match")
    return len(plan_records)


def _stage_payload_from_history(
    history: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    for record in reversed(history):
        if record.get("state") in {"staged", "payload_durable", "binding_visible"}:
            payload = record.get("payload")
            if not isinstance(payload, Mapping):
                break
            staged = payload.get("staged_payloads")
            if isinstance(staged, list):
                return tuple(dict(row) for row in staged if isinstance(row, Mapping))
    raise SparseBundleError("append journal has no complete staged payload table")


def _binding_result(
    receipt: Any,
    *,
    graph_pre_transaction: Mapping[str, Any],
    graph_pre_bind: Mapping[str, Any],
    graph_post_bind: Mapping[str, Any],
    staged: Sequence[Mapping[str, Any]],
    growth_before: Mapping[str, Any],
    growth_after: Mapping[str, Any],
) -> dict[str, Any]:
    bindings = [
        {
            "appended": bool(binding.appended),
            "expert_id": binding.expert_id,
            "layer": binding.layer,
            "record_index": binding.record_index,
            "segment_sha256": binding.segment_sha256,
        }
        for binding in receipt.bindings
    ]
    return {
        "binding_receipts": bindings,
        "binding_receipts_sha256": _sha256(bindings),
        "graph_post_bind": dict(graph_post_bind),
        "graph_pre_bind": dict(graph_pre_bind),
        "graph_pre_transaction": dict(graph_pre_transaction),
        "growth": {
            "logical_bytes_after": growth_after["logical_bytes"],
            "logical_bytes_before": growth_before["logical_bytes"],
            "logical_bytes_delta": (
                growth_after["logical_bytes"] - growth_before["logical_bytes"]
            ),
            "physical_bytes_after": growth_after["physical_bytes"],
            "physical_bytes_before": growth_before["physical_bytes"],
            "physical_bytes_delta": (
                growth_after["physical_bytes"] - growth_before["physical_bytes"]
            ),
            "shards_after": growth_after["shards"],
            "shards_before": growth_before["shards"],
        },
        "staged_payloads": list(staged),
        "staged_payloads_sha256": _sha256(list(staged)),
    }


def _unlink_pending(append_descriptor: int) -> None:
    try:
        linked = os.stat(
            _APPEND_PENDING,
            dir_fd=append_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    if not stat.S_ISREG(linked.st_mode):
        raise SparseBundleError("pending append path is not a regular file")
    os.unlink(_APPEND_PENDING, dir_fd=append_descriptor)
    os.fsync(append_descriptor)


def _cleanup_stage(
    append_descriptor: int,
    stage_name: str,
    staged: Sequence[Mapping[str, Any]],
) -> None:
    try:
        stage_descriptor = _open_directory_at(append_descriptor, stage_name)
    except SparseBundleError:
        return
    try:
        expected = {str(row["filename"]) for row in staged}
        actual = set(os.listdir(stage_descriptor))
        if actual != expected:
            return
        for name in sorted(expected):
            linked = os.stat(name, dir_fd=stage_descriptor, follow_symlinks=False)
            if not stat.S_ISREG(linked.st_mode):
                return
        for name in sorted(expected):
            os.unlink(name, dir_fd=stage_descriptor)
        os.fsync(stage_descriptor)
    finally:
        os.close(stage_descriptor)
    try:
        os.rmdir(stage_name, dir_fd=append_descriptor)
        os.fsync(append_descriptor)
    except OSError:
        pass


def _requested_coordinate_records(
    coordinates: Sequence[tuple[int, int]],
) -> list[dict[str, int]]:
    return [
        {"expert_id": expert_id, "layer": layer} for layer, expert_id in coordinates
    ]


def _require_local_transaction_identity(
    transaction: Mapping[str, Any],
    *,
    coordinates: Sequence[tuple[int, int]],
    manifest_identity: Mapping[str, Any],
    repo_id: str,
    revision: str,
    layout_fingerprint: str,
) -> None:
    _validate_transaction_schema(transaction)
    remote = transaction["remote_source"]
    if (
        transaction.get("coordinates") != _requested_coordinate_records(coordinates)
        or transaction.get("bundle_manifest") != manifest_identity
        or remote.get("repo_id") != repo_id
        or remote.get("revision") != revision
        or remote.get("layout_fingerprint") != layout_fingerprint
    ):
        raise SparseBundleError(
            "local append transaction identity does not match request"
        )


def _local_append_fast_path(
    args: argparse.Namespace,
    root: Path,
    coordinates: Sequence[tuple[int, int]],
    model: LogicalModelIdentity,
) -> dict[str, Any] | None:
    append_path = root / _APPEND_DIRECTORY
    try:
        metadata = append_path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise SparseBundleError("expert append store must be a non-symlink directory")

    with _append_lock(root) as (
        root_descriptor,
        append_descriptor,
        weights_descriptor,
    ):
        manifest, manifest_identity = _load_bundle_manifest_at(
            root_descriptor,
            repo_id=args.repo_id,
            revision=args.revision,
            expected_layout=getattr(args, "layout_fingerprint", None),
        )
        _assert_no_dense_pending_at(root_descriptor)
        receipts_descriptor = _open_directory_at(
            append_descriptor,
            "receipts",
            create=True,
        )
        try:
            matching: list[tuple[str, dict[str, Any], Mapping[str, Any]]] = []
            for name in sorted(os.listdir(receipts_descriptor)):
                if re.fullmatch(r"[0-9a-f]{64}\.json", name) is None:
                    raise SparseBundleError(
                        f"unexpected expert append receipt artifact: {name!r}"
                    )
                document, _encoded = _read_json_at(receipts_descriptor, name)
                body = _verify_receipt(document)
                transaction = body["transaction"]
                if transaction.get("coordinates") != _requested_coordinate_records(
                    coordinates
                ):
                    continue
                _require_local_transaction_identity(
                    transaction,
                    coordinates=coordinates,
                    manifest_identity=manifest_identity,
                    repo_id=args.repo_id,
                    revision=args.revision,
                    layout_fingerprint=str(manifest["layout_fingerprint"]),
                )
                _verify_completed_receipt_payload(
                    body,
                    weights_descriptor,
                    root,
                    model,
                    budget_mb=float(args.budget_mb),
                )
                matching.append((name, body, document))
            if matching:
                plan_hashes = {
                    str(body["transaction"]["plans_sha256"])
                    for _name, body, _document in matching
                }
                if len(plan_hashes) != 1:
                    raise SparseBundleError(
                        "completed receipts conflict for requested experts"
                    )
                _name, body, document = matching[0]
                transaction = body["transaction"]
                pending = _optional_json_at(append_descriptor, _APPEND_PENDING)
                if pending is not None:
                    pending_transaction, _history = _verify_pending(pending[0])
                    if pending_transaction == transaction:
                        _unlink_pending(append_descriptor)
                    elif pending_transaction.get("coordinates") == transaction.get(
                        "coordinates"
                    ):
                        raise SparseBundleError(
                            "pending append conflicts with completed receipt"
                        )
                _cleanup_stage(
                    append_descriptor,
                    f"stage-{transaction['transaction_id']}",
                    body["result"]["staged_payloads"],
                )
                return {
                    "receipt_sha256": document["sha256"],
                    "status": "already-appended",
                    "transaction_id": transaction["transaction_id"],
                }

            pending = _optional_json_at(append_descriptor, _APPEND_PENDING)
            if pending is None:
                return None
            transaction, history = _verify_pending(pending[0])
            if transaction.get("coordinates") != _requested_coordinate_records(
                coordinates
            ):
                raise SparseBundleError("another expert append transaction is pending")
            _require_local_transaction_identity(
                transaction,
                coordinates=coordinates,
                manifest_identity=manifest_identity,
                repo_id=args.repo_id,
                revision=args.revision,
                layout_fingerprint=str(manifest["layout_fingerprint"]),
            )
            state = str(history[-1]["state"])
            if state == "planned":
                return None
            if state not in {"staged", "payload_durable", "binding_visible"}:
                raise SparseBundleError("pending append state is unknown")

            leaves = transaction["leaves"]
            plans = tuple(
                _expert_plan_from_record(record) for record in transaction["plans"]
            )
            if [_expert_plan_record(plan) for plan in plans] != transaction["plans"]:
                raise SparseBundleError("pending append plans failed reconstruction")
            stage_name = f"stage-{transaction['transaction_id']}"
            stage_descriptor = _open_directory_at(append_descriptor, stage_name)
            try:
                staged = _stage_payload_from_history(history)
                staged = _verify_staged_payloads(
                    stage_descriptor,
                    leaves,
                    list(staged),
                )
                planned_payload = history[0]["payload"]
                if (
                    history[0].get("state") != "planned"
                    or not isinstance(planned_payload, Mapping)
                    or not isinstance(planned_payload.get("growth_before"), Mapping)
                    or not isinstance(
                        planned_payload.get("graph_pre_transaction"), Mapping
                    )
                ):
                    raise SparseBundleError("pending append planned state is invalid")
                growth_before = dict(planned_payload["growth_before"])
                graph_pre = dict(planned_payload["graph_pre_transaction"])
                shards = [str(leaf["shard"]) for leaf in leaves]

                if state == "staged":
                    _materialize_local_payload(
                        weights_descriptor,
                        stage_descriptor,
                        leaves,
                        staged,
                    )
                    _verify_local_payloads(weights_descriptor, leaves, staged)
                    growth_after = _shard_stats(weights_descriptor, shards)
                    history = (
                        *history,
                        _journal_event(
                            history,
                            "payload_durable",
                            {
                                "growth_after": growth_after,
                                "staged_payloads": list(staged),
                                "staged_payloads_sha256": _sha256(list(staged)),
                            },
                        ),
                    )
                    _atomic_json_at(
                        append_descriptor,
                        _APPEND_PENDING,
                        _pending_document(transaction, history),
                    )
                    state = "payload_durable"
                    _failpoint(args, "payload-before-binding")
                else:
                    _verify_local_payloads(weights_descriptor, leaves, staged)
                    durable_payload = history[-1]["payload"]
                    if state == "binding_visible":
                        durable_payload = history[-2]["payload"]
                    if not isinstance(durable_payload, Mapping) or not isinstance(
                        durable_payload.get("growth_after"), Mapping
                    ):
                        raise SparseBundleError("pending durable state is invalid")
                    growth_after = dict(durable_payload["growth_after"])

                if state == "payload_durable":
                    with CausalWeightMount(
                        root,
                        model,
                        budget_mb=float(args.budget_mb),
                    ) as mount:
                        if (
                            mount.layout.layout_fingerprint
                            != manifest["layout_fingerprint"]
                        ):
                            raise SparseBundleError(
                                "mounted layout changed before reconciliation"
                            )
                        graph_pre_bind = _revision_record(mount.graph.store.revision())
                        binding_receipt = mount.bind_plans(plans)
                        graph_post_bind = _revision_record(mount.graph.store.revision())
                    _failpoint(args, "binding-before-journal")
                    result = _binding_result(
                        binding_receipt,
                        graph_pre_transaction=graph_pre,
                        graph_pre_bind=graph_pre_bind,
                        graph_post_bind=graph_post_bind,
                        staged=staged,
                        growth_before=growth_before,
                        growth_after=growth_after,
                    )
                    history = (
                        *history,
                        _journal_event(history, "binding_visible", result),
                    )
                    _atomic_json_at(
                        append_descriptor,
                        _APPEND_PENDING,
                        _pending_document(transaction, history),
                    )
                    state = "binding_visible"

                if state != "binding_visible":
                    raise SparseBundleError(
                        "pending append reconciliation is incomplete"
                    )
                receipt_document = _receipt_document(transaction, history)
                receipt_name = f"{transaction['transaction_id']}.json"
                _atomic_json_at(
                    receipts_descriptor,
                    receipt_name,
                    receipt_document,
                )
                _failpoint(args, "receipt-before-cleanup")
                _unlink_pending(append_descriptor)
                _cleanup_stage(append_descriptor, stage_name, staged)
                return {
                    "appended_bindings": sum(
                        bool(row["appended"])
                        for row in history[-1]["payload"]["binding_receipts"]
                    ),
                    "receipt_sha256": receipt_document["sha256"],
                    "status": "appended",
                    "transaction_id": transaction["transaction_id"],
                }
            finally:
                os.close(stage_descriptor)
        finally:
            os.close(receipts_descriptor)


def append_experts(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.bundle).expanduser().absolute()
    coordinates = _normalize_coordinates(args.expert)
    model = LogicalModelIdentity(repo_id=args.repo_id, revision=args.revision)
    resident_limit_bytes = int(float(args.resident_limit_mb) * 1024 * 1024)
    staging_limit_bytes = int(float(args.staging_limit_mb) * 1024 * 1024)

    preflight_root = _open_plain_root(root)
    try:
        _load_bundle_manifest_at(
            preflight_root,
            repo_id=args.repo_id,
            revision=args.revision,
            expected_layout=getattr(args, "layout_fingerprint", None),
        )
    finally:
        os.close(preflight_root)
    local = _local_append_fast_path(args, root, coordinates, model)
    if local is not None:
        return local

    with _append_lock(root) as (
        root_descriptor,
        append_descriptor,
        weights_descriptor,
    ):
        manifest, manifest_identity = _load_bundle_manifest_at(
            root_descriptor,
            repo_id=args.repo_id,
            revision=args.revision,
            expected_layout=getattr(args, "layout_fingerprint", None),
        )
        _assert_no_dense_pending_at(root_descriptor)
        source = _open_remote_source(args)
        try:
            _adopt_bundle_inventory(source, weights_descriptor, manifest)
            plans, source_identity = _plan_remote_experts(
                source,
                coordinates,
                repo_id=args.repo_id,
                revision=args.revision,
                layout_fingerprint=str(manifest["layout_fingerprint"]),
            )
            leaves = _expert_leaf_records(plans)
            plan_records = [_expert_plan_record(plan) for plan in plans]
            transaction_core = {
                "bundle_manifest": manifest_identity,
                "coordinates": [
                    {"expert_id": expert_id, "layer": layer}
                    for layer, expert_id in coordinates
                ],
                "leaves": list(leaves),
                "leaves_sha256": _sha256(list(leaves)),
                "plans": plan_records,
                "plans_sha256": _sha256(plan_records),
                "remote_source": source_identity,
                "schema": "immer.deepseek-v4-expert-append-transaction/v1",
            }
            transaction_id = _sha256(transaction_core)
            transaction = {**transaction_core, "transaction_id": transaction_id}
            stage_name = f"stage-{transaction_id}"
            receipt_name = f"{transaction_id}.json"
            receipts_descriptor = _open_directory_at(
                append_descriptor,
                "receipts",
                create=True,
            )
            try:
                try:
                    completed, _encoded = _read_json_at(
                        receipts_descriptor,
                        receipt_name,
                    )
                except SparseBundleError as exc:
                    try:
                        os.stat(
                            receipt_name,
                            dir_fd=receipts_descriptor,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        completed = None
                    else:
                        raise exc
                try:
                    pending_document, _pending_encoded = _read_json_at(
                        append_descriptor,
                        _APPEND_PENDING,
                    )
                except SparseBundleError as exc:
                    try:
                        os.stat(
                            _APPEND_PENDING,
                            dir_fd=append_descriptor,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        pending_document = None
                    else:
                        raise exc

                if completed is not None:
                    completed_body = _verify_receipt(completed)
                    if dict(completed_body["transaction"]) != transaction:
                        raise SparseBundleError("completed receipt identity conflicts")
                    _verify_completed_receipt_payload(
                        completed_body,
                        weights_descriptor,
                        root,
                        model,
                        budget_mb=float(args.budget_mb),
                    )
                    if pending_document is not None:
                        pending_transaction, _history = _verify_pending(
                            pending_document
                        )
                        if pending_transaction != transaction:
                            raise SparseBundleError(
                                "pending append conflicts with completed receipt"
                            )
                        _unlink_pending(append_descriptor)
                    _cleanup_stage(
                        append_descriptor,
                        stage_name,
                        completed_body["result"]["staged_payloads"],
                    )
                    return {
                        "receipt_sha256": completed["sha256"],
                        "status": "already-appended",
                        "transaction_id": transaction_id,
                    }

                graph_pre = _preflight_graph(
                    root,
                    model,
                    plans,
                    layout_fingerprint=str(manifest["layout_fingerprint"]),
                    budget_mb=float(args.budget_mb),
                )
                shards = [str(leaf["shard"]) for leaf in leaves]
                if pending_document is None:
                    growth_before = _shard_stats(weights_descriptor, shards)
                    history: tuple[dict[str, Any], ...] = (
                        _journal_event(
                            (),
                            "planned",
                            {
                                "graph_pre_transaction": graph_pre,
                                "growth_before": growth_before,
                            },
                        ),
                    )
                    _atomic_json_at(
                        append_descriptor,
                        _APPEND_PENDING,
                        _pending_document(transaction, history),
                    )
                else:
                    pending_transaction, history = _verify_pending(pending_document)
                    if pending_transaction != transaction:
                        raise SparseBundleError(
                            "another expert append transaction is pending"
                        )
                    first_payload = history[0]["payload"]
                    if (
                        history[0].get("state") != "planned"
                        or not isinstance(first_payload, Mapping)
                        or not isinstance(first_payload.get("growth_before"), Mapping)
                        or not isinstance(
                            first_payload.get("graph_pre_transaction"), Mapping
                        )
                    ):
                        raise SparseBundleError(
                            "pending append planned state is invalid"
                        )
                    growth_before = dict(first_payload["growth_before"])
                    graph_pre = dict(first_payload["graph_pre_transaction"])

                state = str(history[-1]["state"])
                stage_descriptor = _open_directory_at(
                    append_descriptor,
                    stage_name,
                    create=True,
                )
                try:
                    if state == "planned":
                        staged = _stage_remote_payloads(
                            source,
                            stage_descriptor,
                            leaves,
                            resident_limit_bytes=resident_limit_bytes,
                            staging_limit_bytes=staging_limit_bytes,
                            source_identity=source_identity,
                        )
                        history = (
                            *history,
                            _journal_event(
                                history,
                                "staged",
                                {
                                    "staged_payloads": list(staged),
                                    "staged_payloads_sha256": _sha256(list(staged)),
                                },
                            ),
                        )
                        _atomic_json_at(
                            append_descriptor,
                            _APPEND_PENDING,
                            _pending_document(transaction, history),
                        )
                        state = "staged"
                    else:
                        staged = _stage_payload_from_history(history)
                        _verify_staged_payloads(
                            stage_descriptor,
                            leaves,
                            list(staged),
                        )

                    if state == "staged":
                        staged = _verify_staged_payloads(
                            stage_descriptor,
                            leaves,
                            list(staged),
                        )
                        _materialize_local_payload(
                            weights_descriptor,
                            stage_descriptor,
                            leaves,
                            staged,
                        )
                        _verify_local_payloads(weights_descriptor, leaves, staged)
                        growth_after = _shard_stats(weights_descriptor, shards)
                        history = (
                            *history,
                            _journal_event(
                                history,
                                "payload_durable",
                                {
                                    "growth_after": growth_after,
                                    "staged_payloads": list(staged),
                                    "staged_payloads_sha256": _sha256(list(staged)),
                                },
                            ),
                        )
                        _atomic_json_at(
                            append_descriptor,
                            _APPEND_PENDING,
                            _pending_document(transaction, history),
                        )
                        state = "payload_durable"
                        _failpoint(args, "payload-before-binding")
                    else:
                        _verify_local_payloads(weights_descriptor, leaves, staged)
                        durable_payload = history[-1]["payload"]
                        if state == "binding_visible":
                            durable_payload = history[-2]["payload"]
                        if not isinstance(durable_payload, Mapping) or not isinstance(
                            durable_payload.get("growth_after"), Mapping
                        ):
                            raise SparseBundleError(
                                "pending append durable state is invalid"
                            )
                        growth_after = dict(durable_payload["growth_after"])

                    if state == "payload_durable":
                        with CausalWeightMount(
                            root,
                            model,
                            budget_mb=float(args.budget_mb),
                        ) as mount:
                            if (
                                mount.layout.layout_fingerprint
                                != manifest["layout_fingerprint"]
                            ):
                                raise SparseBundleError(
                                    "mounted layout changed before binding"
                                )
                            graph_pre_bind = _revision_record(
                                mount.graph.store.revision()
                            )
                            binding_receipt = mount.bind_plans(plans)
                            graph_post_bind = _revision_record(
                                mount.graph.store.revision()
                            )
                        _failpoint(args, "binding-before-journal")
                        result = _binding_result(
                            binding_receipt,
                            graph_pre_transaction=graph_pre,
                            graph_pre_bind=graph_pre_bind,
                            graph_post_bind=graph_post_bind,
                            staged=staged,
                            growth_before=growth_before,
                            growth_after=growth_after,
                        )
                        history = (
                            *history,
                            _journal_event(history, "binding_visible", result),
                        )
                        _atomic_json_at(
                            append_descriptor,
                            _APPEND_PENDING,
                            _pending_document(transaction, history),
                        )
                        state = "binding_visible"

                    if state != "binding_visible":
                        raise SparseBundleError(
                            f"unsupported pending append state: {state}"
                        )
                    receipt_document = _receipt_document(transaction, history)
                    _atomic_json_at(
                        receipts_descriptor,
                        receipt_name,
                        receipt_document,
                    )
                    _failpoint(args, "receipt-before-cleanup")
                    _unlink_pending(append_descriptor)
                    _cleanup_stage(append_descriptor, stage_name, staged)
                    return {
                        "appended_bindings": sum(
                            bool(row["appended"])
                            for row in history[-1]["payload"]["binding_receipts"]
                        ),
                        "receipt_sha256": receipt_document["sha256"],
                        "status": "appended",
                        "transaction_id": transaction_id,
                    }
                finally:
                    os.close(stage_descriptor)
            finally:
                os.close(receipts_descriptor)
        finally:
            source.close()


def _verify_append_store(
    root: Path,
    model: LogicalModelIdentity,
    *,
    repo_id: str,
    revision: str,
    expected_layout: str | None,
    budget_mb: float,
) -> dict[str, Any]:
    append_path = root / _APPEND_DIRECTORY
    try:
        metadata = append_path.lstat()
    except FileNotFoundError:
        return {
            "appended_bindings": 0,
            "coordinates": [],
            "physical_growth_bytes": 0,
            "pending": None,
            "receipts": [],
        }
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise SparseBundleError("expert append store must be a non-symlink directory")

    receipts: list[dict[str, Any]] = []
    appended_coordinates: set[tuple[int, int]] = set()
    physical_growth = 0
    pending_report: dict[str, Any] | None = None
    with _append_lock(root) as (
        root_descriptor,
        append_descriptor,
        weights_descriptor,
    ):
        _manifest, manifest_identity = _load_bundle_manifest_at(
            root_descriptor,
            repo_id=repo_id,
            revision=revision,
            expected_layout=expected_layout,
        )
        try:
            receipts_descriptor = _open_directory_at(
                append_descriptor,
                "receipts",
            )
        except SparseBundleError:
            receipts_descriptor = None
            try:
                os.stat(
                    "receipts",
                    dir_fd=append_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise
        if receipts_descriptor is not None:
            try:
                for name in sorted(os.listdir(receipts_descriptor)):
                    if re.fullmatch(r"[0-9a-f]{64}\.json", name) is None:
                        raise SparseBundleError(
                            f"unexpected expert append receipt artifact: {name!r}"
                        )
                    document, _encoded = _read_json_at(receipts_descriptor, name)
                    body = _verify_receipt(document)
                    transaction = body["transaction"]
                    if transaction.get("bundle_manifest") != manifest_identity:
                        raise SparseBundleError(
                            "append receipt binds a different bundle manifest"
                        )
                    binding_count = _verify_completed_receipt_payload(
                        body,
                        weights_descriptor,
                        root,
                        model,
                        budget_mb=budget_mb,
                    )
                    for coordinate in transaction["coordinates"]:
                        appended_coordinates.add(
                            (int(coordinate["layer"]), int(coordinate["expert_id"]))
                        )
                    growth = body["result"]["growth"]
                    physical_growth += int(growth["physical_bytes_delta"])
                    receipts.append(
                        {
                            "bindings": binding_count,
                            "receipt_sha256": document["sha256"],
                            "transaction_id": transaction["transaction_id"],
                        }
                    )
            finally:
                os.close(receipts_descriptor)

        try:
            pending_document, _encoded = _read_json_at(
                append_descriptor,
                _APPEND_PENDING,
            )
        except SparseBundleError as exc:
            try:
                os.stat(
                    _APPEND_PENDING,
                    dir_fd=append_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pending_document = None
            else:
                raise exc
        if pending_document is not None:
            transaction, history = _verify_pending(pending_document)
            _validate_transaction_schema(transaction)
            if transaction.get("bundle_manifest") != manifest_identity:
                raise SparseBundleError("pending append binds a different bundle")
            state = str(history[-1]["state"])
            if state not in {
                "planned",
                "staged",
                "payload_durable",
                "binding_visible",
            }:
                raise SparseBundleError("pending append state is unknown")
            if state != "planned":
                stage_name = f"stage-{transaction['transaction_id']}"
                stage_descriptor = _open_directory_at(
                    append_descriptor,
                    stage_name,
                )
                try:
                    staged = _stage_payload_from_history(history)
                    _verify_staged_payloads(
                        stage_descriptor,
                        transaction["leaves"],
                        list(staged),
                    )
                finally:
                    os.close(stage_descriptor)
                if state in {"payload_durable", "binding_visible"}:
                    _verify_local_payloads(
                        weights_descriptor,
                        transaction["leaves"],
                        staged,
                    )
                if state == "binding_visible":
                    _verify_completed_receipt_payload(
                        {
                            "result": history[-1]["payload"],
                            "transaction": transaction,
                        },
                        weights_descriptor,
                        root,
                        model,
                        budget_mb=budget_mb,
                    )
            pending_report = {
                "reconcile_required": state in {"payload_durable", "binding_visible"},
                "state": state,
                "transaction_id": transaction["transaction_id"],
            }
    return {
        "appended_bindings": len(appended_coordinates),
        "coordinates": sorted(appended_coordinates),
        "physical_growth_bytes": physical_growth,
        "pending": pending_report,
        "receipts": receipts,
    }


def _stable_file_bytes(path: Path, *, maximum: int = 256 * 1024 * 1024) -> bytes:
    try:
        linked = path.lstat()
    except OSError as exc:
        raise SparseBundleError(f"required evidence file is missing: {path}") from exc
    if stat.S_ISLNK(linked.st_mode) or not stat.S_ISREG(linked.st_mode):
        raise SparseBundleError("evidence path must be a non-symlink regular file")
    flags = os.O_RDONLY | int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino):
            raise SparseBundleError("evidence file changed while opening")
        size = opened.st_size
        if size < 1 or size > maximum:
            raise SparseBundleError("evidence file size is outside its bound")
        chunks: list[bytes] = []
        cursor = 0
        while cursor < size:
            chunk = os.pread(descriptor, min(1024 * 1024, size - cursor), cursor)
            if not chunk:
                raise SparseBundleError("evidence file returned a short read")
            chunks.append(chunk)
            cursor += len(chunk)
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino) != (
            linked.st_dev,
            linked.st_ino,
        ) or after.st_size != size:
            raise SparseBundleError("evidence file changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _trace_expert_ingest_plan(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.bundle).expanduser().absolute()
    model = LogicalModelIdentity(repo_id=args.repo_id, revision=args.revision)
    root_descriptor = _open_plain_root(root)
    try:
        manifest, manifest_identity = _load_bundle_manifest_at(
            root_descriptor,
            repo_id=args.repo_id,
            revision=args.revision,
            expected_layout=getattr(args, "layout_fingerprint", None),
        )
        _assert_no_dense_pending_at(root_descriptor)
        weights_descriptor = _open_directory_at(root_descriptor, "weights")
        try:
            inventory = _load_pinned_inventory_at(weights_descriptor, manifest)
        finally:
            os.close(weights_descriptor)
    finally:
        os.close(root_descriptor)

    expert_resources = _inventory_expert_resources(inventory)
    inventory_coordinates = set(expert_resources)
    base_coordinates = _manifest_expert_coordinates(
        manifest,
        inventory_coordinates,
    )
    append_state = _verify_append_store(
        root,
        model,
        repo_id=args.repo_id,
        revision=args.revision,
        expected_layout=getattr(args, "layout_fingerprint", None),
        budget_mb=float(args.budget_mb),
    )
    if append_state["pending"] is not None:
        raise SparseBundleError(
            "pending expert append must be reconciled before trace ingest"
        )
    appended_coordinates = {
        (int(layer), int(expert_id)) for layer, expert_id in append_state["coordinates"]
    }
    if not appended_coordinates.issubset(inventory_coordinates):
        raise SparseBundleError("append receipts cite experts outside inventory")
    existing_coordinates = base_coordinates | appended_coordinates

    with CausalWeightMount(root, model, budget_mb=float(args.budget_mb)) as mount:
        if mount.layout.layout_fingerprint != manifest["layout_fingerprint"]:
            raise SparseBundleError("current causal graph layout identity changed")
        by_layer: dict[int, list[int]] = defaultdict(list)
        for layer, expert_id in sorted(existing_coordinates):
            by_layer[layer].append(expert_id)
        try:
            resolved_count = sum(
                len(mount.resolve_expert_plans(layer, expert_ids))
                for layer, expert_ids in sorted(by_layer.items())
            )
        except KeyError as exc:
            raise SparseBundleError(
                "current causal graph lost an expert binding"
            ) from exc
        if resolved_count != len(existing_coordinates):
            raise SparseBundleError("current causal graph lost an expert binding")
        graph_revision = _revision_record(mount.graph.store.revision())

    trace_evidence, intervals, unique_leaves = _sealed_trace_coverage(
        (
            Path(path).expanduser().absolute()
            for path in (getattr(args, "access_trace", None) or ())
        ),
        repo_id=args.repo_id,
        revision=args.revision,
        fingerprint=str(manifest["layout_fingerprint"]),
        inventory=inventory,
    )
    covered = _covered_experts(inventory, intervals)
    derived_coordinates = {
        (layer, expert_id)
        for layer, expert_ids in covered.items()
        for expert_id in expert_ids
    }
    if not derived_coordinates:
        raise SparseBundleError("access traces are incomplete for every routed expert")
    if not derived_coordinates.issubset(inventory_coordinates):
        raise SparseBundleError("trace-derived experts disagree with inventory")
    missing_coordinates = derived_coordinates - existing_coordinates
    derived_payload_bytes = sum(
        expert_resources[coordinate]["payload_bytes"]
        for coordinate in derived_coordinates
    )
    existing_payload_bytes = sum(
        expert_resources[coordinate]["payload_bytes"]
        for coordinate in existing_coordinates
    )
    missing_payload_bytes = sum(
        expert_resources[coordinate]["payload_bytes"]
        for coordinate in missing_coordinates
    )
    missing_leaf_count = sum(
        expert_resources[coordinate]["parts"] for coordinate in missing_coordinates
    )
    minimum_leaf_transfer_budget_bytes = (
        missing_payload_bytes
        + missing_leaf_count * _REMOTE_RANGE_RESERVATION_OVERHEAD_BYTES
    )
    missing_maximum_part_bytes = max(
        (
            expert_resources[coordinate]["maximum_part_bytes"]
            for coordinate in missing_coordinates
        ),
        default=0,
    )
    leaf_records = [
        {"length": length, "offset": offset, "shard": shard}
        for shard, offset, length in unique_leaves
    ]
    body = {
        "append_receipts_sha256": _sha256(append_state["receipts"]),
        "bundle_manifest": manifest_identity,
        "coordinates": {
            "derived": _coordinate_records(derived_coordinates),
            "existing": _coordinate_records(existing_coordinates),
            "missing": _coordinate_records(missing_coordinates),
        },
        "counts": {
            "derived": len(derived_coordinates),
            "derived_payload_bytes": derived_payload_bytes,
            "existing": len(existing_coordinates),
            "existing_payload_bytes": existing_payload_bytes,
            "missing": len(missing_coordinates),
            "missing_leaves": missing_leaf_count,
            "missing_payload_bytes": missing_payload_bytes,
            "traces": len(trace_evidence),
            "unique_leaves": len(unique_leaves),
        },
        "graph_revision": graph_revision,
        "inventory_layout_sha256": _sha256(
            Streamer._inventory_layout_projection(inventory)
        ),
        "layout_fingerprint": manifest["layout_fingerprint"],
        "logical_model": dict(manifest["logical_model"]),
        "resource_requirements": {
            "minimum_available_staging_disk_bytes": (
                missing_payload_bytes + 8 * 1024 * 1024 if missing_coordinates else 0
            ),
            "minimum_leaf_transfer_budget_bytes": minimum_leaf_transfer_budget_bytes,
            "minimum_resident_limit_bytes": missing_maximum_part_bytes,
            "minimum_staging_limit_bytes": missing_payload_bytes,
            "source_budget_requires_inventory_scan_headroom": bool(
                missing_coordinates
            ),
            "source_range_reservation_overhead_bytes_per_leaf": (
                _REMOTE_RANGE_RESERVATION_OVERHEAD_BYTES
            ),
        },
        "trace_evidence": list(trace_evidence),
        "trace_evidence_sha256": _sha256(list(trace_evidence)),
        "unique_leaves_sha256": _sha256(leaf_records),
    }
    return {
        "body": body,
        "schema": TRACE_EXPERT_PLAN_SCHEMA,
        "sha256": _sha256(body),
    }


def _trace_expert_ingest_document(
    plan: Mapping[str, Any],
    *,
    status: str,
    append: Mapping[str, Any] | None,
) -> dict[str, Any]:
    identity = {
        "append": None if append is None else dict(append),
        "plan": dict(plan),
        "schema": TRACE_EXPERT_INGEST_SCHEMA,
        "status": status,
    }
    return {**identity, "sha256": _sha256(identity)}


def append_trace_experts(args: argparse.Namespace) -> dict[str, Any]:
    plan = _trace_expert_ingest_plan(args)
    body = plan["body"]
    missing = tuple(
        (int(record["layer"]), int(record["expert_id"]))
        for record in body["coordinates"]["missing"]
    )
    if getattr(args, "plan_only", False):
        return _trace_expert_ingest_document(
            plan,
            status="planned",
            append=None,
        )
    if not missing:
        return _trace_expert_ingest_document(
            plan,
            status="already-complete",
            append=None,
        )
    append_args = argparse.Namespace(**vars(args))
    append_args.expert = list(missing)
    result = append_experts(append_args)
    final_plan = _trace_expert_ingest_plan(args)
    final_body = final_plan["body"]
    for key in (
        "bundle_manifest",
        "layout_fingerprint",
        "logical_model",
        "trace_evidence_sha256",
        "unique_leaves_sha256",
    ):
        if final_body.get(key) != body.get(key):
            raise SparseBundleError(
                "trace ingest identity changed while experts were appended"
            )
    final_missing = final_body["coordinates"]["missing"]
    final_existing = {
        (int(record["layer"]), int(record["expert_id"]))
        for record in final_body["coordinates"]["existing"]
    }
    if final_missing or not set(missing).issubset(final_existing):
        raise SparseBundleError("trace ingest append postcondition is incomplete")
    return _trace_expert_ingest_document(
        final_plan,
        status=str(result["status"]),
        append={**result, "requested_plan_sha256": plan["sha256"]},
    )


def _dense_plan_record(plan: TensorRangePlan) -> dict[str, Any]:
    return {
        "absolute_offset": plan.absolute_offset,
        "dtype": plan.dtype,
        "length": plan.length,
        "name": plan.name,
        "shape": list(plan.shape),
        "shard": plan.shard,
    }


def _dense_plan_from_record(value: object) -> TensorRangePlan:
    if not isinstance(value, Mapping) or set(value) != {
        "absolute_offset",
        "dtype",
        "length",
        "name",
        "shape",
        "shard",
    }:
        raise SparseBundleError("stored dense tensor plan schema is invalid")
    shape = value.get("shape")
    if (
        not isinstance(shape, list)
        or not shape
        or any(
            isinstance(dimension, bool) or not isinstance(dimension, int)
            for dimension in shape
        )
        or not isinstance(value.get("name"), str)
        or not isinstance(value.get("dtype"), str)
        or not isinstance(value.get("shard"), str)
        or isinstance(value.get("absolute_offset"), bool)
        or not isinstance(value.get("absolute_offset"), int)
        or isinstance(value.get("length"), bool)
        or not isinstance(value.get("length"), int)
    ):
        raise SparseBundleError("stored dense tensor shape is invalid")
    try:
        plan = TensorRangePlan(
            name=str(value["name"]),
            dtype=str(value["dtype"]).upper(),
            shape=tuple(int(dimension) for dimension in shape),
            shard=str(value["shard"]),
            absolute_offset=int(value["absolute_offset"]),
            length=int(value["length"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SparseBundleError("stored dense tensor plan is invalid") from exc
    if (
        not plan.name
        or Path(plan.shard).name != plan.shard
        or plan.shard in (".", "..")
        or plan.absolute_offset < 0
        or plan.length <= 0
        or not plan.shape
        or any(dimension <= 0 for dimension in plan.shape)
        or plan.dtype not in _DENSE_DTYPE_BYTES
    ):
        raise SparseBundleError("stored dense tensor plan coordinate is invalid")
    elements = 1
    for dimension in plan.shape:
        elements *= dimension
    if elements * _DENSE_DTYPE_BYTES[plan.dtype] != plan.length:
        raise SparseBundleError("dense tensor plan length disagrees with shape")
    if _dense_plan_record(plan) != dict(value):
        raise SparseBundleError("stored dense tensor plan is noncanonical")
    return plan


def _is_main_decoder_dense_name(name: str) -> bool:
    return not name.startswith("mtp.") and _EXPERT_TENSOR.fullmatch(name) is None


def _dense_plans_from_inventory(
    inventory: Mapping[str, Any],
    *,
    repo_id: str,
    revision: str,
    layout_fingerprint: str,
) -> tuple[TensorRangePlan, ...]:
    if (
        inventory.get("repo") != repo_id
        or inventory.get("revision") != revision
        or not isinstance(inventory.get("tensors"), list)
    ):
        raise SparseBundleError("pinned inventory logical identity is invalid")
    plans: list[TensorRangePlan] = []
    for raw in inventory["tensors"]:
        if not isinstance(raw, Mapping):
            raise SparseBundleError("pinned inventory tensor record is invalid")
        name = str(raw.get("name", ""))
        if not _is_main_decoder_dense_name(name):
            continue
        offsets = raw.get("offset_in_shard")
        shape = raw.get("shape")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not isinstance(shape, list)
        ):
            raise SparseBundleError("pinned dense tensor metadata is invalid")
        begin, stop = offsets
        data_start = raw.get("data_start")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (begin, stop, data_start)
        ):
            raise SparseBundleError("pinned dense tensor offsets are invalid")
        record = {
            "absolute_offset": int(data_start) + int(begin),
            "dtype": str(raw.get("dtype", "")).upper(),
            "length": int(stop) - int(begin),
            "name": name,
            "shape": list(shape),
            "shard": str(raw.get("shard", "")),
        }
        plans.append(_dense_plan_from_record(record))
    plans.sort(key=lambda plan: plan.name)
    if not plans or len({plan.name for plan in plans}) != len(plans):
        raise SparseBundleError("pinned dense tensor plan set is empty or duplicated")
    records = [_dense_plan_record(plan) for plan in plans]
    names = [plan.name for plan in plans]
    if (
        repo_id == OFFICIAL_SOURCE
        and revision == OFFICIAL_REVISION
        and layout_fingerprint == OFFICIAL_LAYOUT_FINGERPRINT
        and (
            len(plans) != _OFFICIAL_DENSE_TENSOR_COUNT
            or _sha256(names)
            != "11f5a7368aedd920bc1bd9c7d5dc0571a2108a052f7c32ca70b9b020aa90decd"
            or _sha256(records)
            != "1c3d612f06648d4aef3d484c759c6318744e44bd6d59990e7fbe5d1e6d5940c2"
        )
    ):
        raise SparseBundleError("official main-decoder dense contract drifted")
    return tuple(plans)


def _merge_intervals(
    intervals: Iterable[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    merged: list[tuple[int, int]] = []
    for start, stop in sorted(intervals):
        if start < 0 or stop <= start:
            raise SparseBundleError("trace coverage interval is invalid")
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return tuple(merged)


def _trace_paths_for_manifest(
    root: Path,
    args: argparse.Namespace,
    expected_sha256: set[str],
) -> tuple[Path, ...]:
    explicit = tuple(
        Path(path).expanduser().absolute()
        for path in (getattr(args, "access_trace", None) or ())
    )
    candidates = explicit
    if not candidates:
        candidates = tuple(sorted(root.parent.glob("access-*.json")))
    selected: dict[str, Path] = {}
    for path in candidates:
        encoded = _stable_file_bytes(path)
        try:
            trace = AccessTrace.from_bytes(encoded)
            trace.verify()
        except Exception as exc:
            if explicit:
                raise SparseBundleError(
                    f"invalid access trace evidence: {path}"
                ) from exc
            continue
        if trace.sha256 not in expected_sha256:
            if explicit:
                raise SparseBundleError(
                    f"access trace is not cited by the base bundle: {path}"
                )
            continue
        prior = selected.get(trace.sha256)
        if prior is not None and _stable_file_bytes(prior) != encoded:
            raise SparseBundleError("duplicate trace SHA has conflicting bytes")
        selected[trace.sha256] = path
    if set(selected) != expected_sha256:
        missing = sorted(expected_sha256 - set(selected))
        raise SparseBundleError(
            "exact base access traces are required for dense promotion; missing "
            + ",".join(missing)
        )
    return tuple(selected[digest] for digest in sorted(selected))


def _dense_trace_evidence(
    root: Path,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], dict[str, tuple[tuple[int, int], ...]]]:
    raw_expected = manifest.get("trace_sha256")
    if (
        not isinstance(raw_expected, list)
        or not raw_expected
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(value)) is None for value in raw_expected
        )
    ):
        raise SparseBundleError("base bundle trace citation table is invalid")
    expected = {str(value) for value in raw_expected}
    if len(expected) != len(raw_expected):
        raise SparseBundleError("base bundle trace citations are duplicated")
    paths = _trace_paths_for_manifest(root, args, expected)
    intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    evidence: list[dict[str, Any]] = []
    for path in paths:
        trace = AccessTrace.from_bytes(_stable_file_bytes(path))
        trace.verify()
        if (
            trace.repo_id != manifest["logical_model"]["repo_id"]
            or trace.revision != manifest["logical_model"]["revision"]
            or trace.inventory_fingerprint != manifest["layout_fingerprint"]
            or trace.sha256 not in expected
        ):
            raise SparseBundleError("base access trace identity is invalid")
        unique: set[tuple[str, int, int]] = set()
        for operation in trace.operations:
            for leaf in operation.leaves:
                unique.add((leaf.shard, leaf.offset, leaf.length))
                intervals[leaf.shard].append((leaf.offset, leaf.offset + leaf.length))
        evidence.append(
            {
                "operation_count": len(trace.operations),
                "sha256": trace.sha256,
                "unique_leaves": len(unique),
            }
        )
    return (
        tuple(sorted(evidence, key=lambda row: str(row["sha256"]))),
        {shard: _merge_intervals(rows) for shard, rows in intervals.items()},
    )


def _plan_covered_ranges(
    plan: TensorRangePlan,
    intervals: Mapping[str, Sequence[tuple[int, int]]],
) -> tuple[tuple[int, int], ...]:
    start = plan.absolute_offset
    stop = plan.absolute_end
    covered: list[tuple[int, int]] = []
    for left, right in intervals.get(plan.shard, ()):
        overlap_start = max(start, left)
        overlap_stop = min(stop, right)
        if overlap_stop > overlap_start:
            covered.append((overlap_start - start, overlap_stop - overlap_start))
    return tuple(covered)


def _dense_materialization_records(
    plans: Sequence[TensorRangePlan],
    intervals: Mapping[str, Sequence[tuple[int, int]]],
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for plan in plans:
        ranges = _plan_covered_ranges(plan, intervals)
        covered_bytes = sum(length for _offset, length in ranges)
        if covered_bytes == plan.length:
            continue
        plan_record = _dense_plan_record(plan)
        rows.append(
            {
                "base_covered_bytes": covered_bytes,
                "base_covered_ranges": [
                    {"length": length, "relative_offset": offset}
                    for offset, length in ranges
                ],
                "name": plan.name,
                "plan_sha256": _sha256(plan_record),
            }
        )
    return tuple(rows)


def _materialization_categories(names: Sequence[str]) -> dict[str, int]:
    counts = {"compressor_norm": 0, "embedding": 0, "head": 0, "other": 0, "tid2eid": 0}
    for name in names:
        if name == "embed.weight":
            counts["embedding"] += 1
        elif name == "head.weight":
            counts["head"] += 1
        elif name.endswith(".ffn.gate.tid2eid"):
            counts["tid2eid"] += 1
        elif name.endswith(".attn.compressor.norm.weight"):
            counts["compressor_norm"] += 1
        else:
            counts["other"] += 1
    return counts


def _dense_plan_body(
    manifest: Mapping[str, Any],
    manifest_identity: Mapping[str, Any],
    inventory: Mapping[str, Any],
    trace_evidence: Sequence[Mapping[str, Any]],
    plans: Sequence[TensorRangePlan],
    materializations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    plan_records = [_dense_plan_record(plan) for plan in plans]
    by_name = {plan.name: plan for plan in plans}
    materialize_names = [str(row["name"]) for row in materializations]
    materialization_bytes = sum(by_name[name].length for name in materialize_names)
    return {
        "bundle_manifest": dict(manifest_identity),
        "categories": _materialization_categories(materialize_names),
        "layout_fingerprint": manifest["layout_fingerprint"],
        "logical_model": dict(manifest["logical_model"]),
        "materialization_bytes": materialization_bytes,
        "materialization_names": materialize_names,
        "materialization_names_sha256": _sha256(materialize_names),
        "materialization_tensor_count": len(materializations),
        "plans_sha256": _sha256(plan_records),
        "required_tensor_bytes": sum(plan.length for plan in plans),
        "required_tensor_count": len(plans),
        "required_tensor_names_sha256": _sha256([plan.name for plan in plans]),
        "schema": DENSE_PLAN_SCHEMA,
        "trace_evidence": list(trace_evidence),
        "trace_evidence_sha256": _sha256(list(trace_evidence)),
        "inventory_layout_sha256": _sha256(
            Streamer._inventory_layout_projection(inventory)
        ),
    }


def plan_dense_promotion(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.bundle).expanduser().absolute()
    root_descriptor = _open_plain_root(root)
    weights_descriptor: int | None = None
    try:
        weights_descriptor = _open_directory_at(root_descriptor, "weights")
        manifest, manifest_identity = _load_bundle_manifest_at(
            root_descriptor,
            repo_id=args.repo_id,
            revision=args.revision,
            expected_layout=getattr(args, "layout_fingerprint", None),
        )
        inventory_document, encoded = _read_json_at(
            weights_descriptor, "inventory.pinned.json"
        )
        if (
            _sha256_bytes(encoded) != manifest.get("pinned_inventory_sha256")
            or not isinstance(inventory_document, Mapping)
            or inventory_document.get("schema") != "immer.tensor-inventory-cache/v1"
            or inventory_document.get("source_fingerprint")
            != manifest["layout_fingerprint"]
            or not isinstance(inventory_document.get("inventory"), Mapping)
        ):
            raise SparseBundleError("bundle pinned inventory identity is invalid")
        inventory = inventory_document["inventory"]
        plans = _dense_plans_from_inventory(
            inventory,
            repo_id=args.repo_id,
            revision=args.revision,
            layout_fingerprint=str(manifest["layout_fingerprint"]),
        )
        trace_evidence, intervals = _dense_trace_evidence(root, args, manifest)
        materializations = _dense_materialization_records(plans, intervals)
        body = _dense_plan_body(
            manifest,
            manifest_identity,
            inventory,
            trace_evidence,
            plans,
            materializations,
        )
        return {"body": body, "schema": DENSE_PLAN_SCHEMA, "sha256": _sha256(body)}
    finally:
        if weights_descriptor is not None:
            os.close(weights_descriptor)
        os.close(root_descriptor)


def _dense_pending_document(
    transaction: Mapping[str, Any], history: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    body = {"history": list(history), "transaction": dict(transaction)}
    return {"body": body, "schema": DENSE_PENDING_SCHEMA, "sha256": _sha256(body)}


def _verify_dense_pending(
    document: object,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    if not isinstance(document, Mapping) or set(document) != {
        "body",
        "schema",
        "sha256",
    }:
        raise SparseBundleError("dense pending document schema is invalid")
    body = document.get("body")
    if (
        document.get("schema") != DENSE_PENDING_SCHEMA
        or not isinstance(body, Mapping)
        or set(body) != {"history", "transaction"}
        or document.get("sha256") != _sha256(body)
        or not isinstance(body.get("transaction"), Mapping)
    ):
        raise SparseBundleError("dense pending document identity is invalid")
    transaction = dict(body["transaction"])
    _validate_dense_transaction(transaction)
    return transaction, _verify_journal(body.get("history"))


def _dense_receipt_document(
    transaction: Mapping[str, Any], history: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    verified = _verify_journal(list(history))
    if verified[-1]["state"] != "binding_visible":
        raise SparseBundleError("dense receipt journal is not complete")
    result = verified[-1]["payload"]
    body = {
        "journal": {
            "history": list(verified),
            "sha256": _sha256(list(verified)),
        },
        "result": dict(result),
        "transaction": dict(transaction),
    }
    return {"body": body, "schema": DENSE_RECEIPT_SCHEMA, "sha256": _sha256(body)}


def _verify_dense_receipt(document: object) -> dict[str, Any]:
    if not isinstance(document, Mapping) or set(document) != {
        "body",
        "schema",
        "sha256",
    }:
        raise SparseBundleError("dense promotion receipt schema is invalid")
    body = document.get("body")
    if (
        document.get("schema") != DENSE_RECEIPT_SCHEMA
        or not isinstance(body, Mapping)
        or set(body) != {"journal", "result", "transaction"}
        or document.get("sha256") != _sha256(body)
        or not isinstance(body.get("journal"), Mapping)
        or not isinstance(body.get("transaction"), Mapping)
        or not isinstance(body.get("result"), Mapping)
    ):
        raise SparseBundleError("dense promotion receipt identity is invalid")
    journal = body["journal"]
    if set(journal) != {"history", "sha256"} or journal.get("sha256") != _sha256(
        journal.get("history")
    ):
        raise SparseBundleError("dense promotion receipt journal is invalid")
    history = _verify_journal(journal.get("history"))
    if (
        history[-1]["state"] != "binding_visible"
        or history[-1]["payload"] != body["result"]
    ):
        raise SparseBundleError("dense promotion receipt result is not journal-bound")
    transaction = dict(body["transaction"])
    result = dict(body["result"])
    _validate_dense_transaction(transaction)
    _verify_dense_result_schema(result, transaction)
    return {"journal": journal, "result": result, "transaction": transaction}


def _validate_dense_transaction(transaction: Mapping[str, Any]) -> None:
    required = {
        "base_covered_names_sha256",
        "bundle_manifest",
        "inventory_layout_sha256",
        "materialization_bytes",
        "materialization_names_sha256",
        "materializations",
        "materializations_sha256",
        "plans",
        "plans_sha256",
        "remote_source",
        "required_tensor_bytes",
        "required_tensor_count",
        "required_tensor_names_sha256",
        "schema",
        "trace_evidence",
        "trace_evidence_sha256",
        "transaction_id",
    }
    if set(transaction) != required or transaction.get("schema") != (
        DENSE_TRANSACTION_SCHEMA
    ):
        raise SparseBundleError("dense promotion transaction schema is invalid")
    identity = {key: transaction[key] for key in transaction if key != "transaction_id"}
    if transaction.get("transaction_id") != _sha256(identity):
        raise SparseBundleError("dense promotion transaction ID is invalid")
    manifest = transaction.get("bundle_manifest")
    if not isinstance(manifest, Mapping) or set(manifest) != {
        "file_sha256",
        "manifest_sha256",
        "schema",
    }:
        raise SparseBundleError("dense promotion base manifest identity is invalid")
    for key in ("file_sha256", "manifest_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(manifest.get(key))) is None:
            raise SparseBundleError("dense promotion base manifest digest is invalid")
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise SparseBundleError("dense promotion base manifest schema is invalid")

    raw_plans = transaction.get("plans")
    if not isinstance(raw_plans, list) or not raw_plans:
        raise SparseBundleError("dense promotion plan table is invalid")
    plans = tuple(_dense_plan_from_record(record) for record in raw_plans)
    if (
        list(raw_plans) != [_dense_plan_record(plan) for plan in plans]
        or list(plans) != sorted(plans, key=lambda plan: plan.name)
        or len({plan.name for plan in plans}) != len(plans)
        or transaction.get("plans_sha256") != _sha256(raw_plans)
        or transaction.get("required_tensor_names_sha256")
        != _sha256([plan.name for plan in plans])
        or transaction.get("required_tensor_count") != len(plans)
        or transaction.get("required_tensor_bytes")
        != sum(plan.length for plan in plans)
    ):
        raise SparseBundleError("dense promotion plan contract is invalid")
    by_name = {plan.name: plan for plan in plans}

    materializations = transaction.get("materializations")
    if not isinstance(materializations, list):
        raise SparseBundleError("dense promotion materialization table is invalid")
    seen: set[str] = set()
    normalized_materializations: list[dict[str, Any]] = []
    for row in materializations:
        if not isinstance(row, Mapping) or set(row) != {
            "base_covered_bytes",
            "base_covered_ranges",
            "name",
            "plan_sha256",
        }:
            raise SparseBundleError("dense promotion materialization row is invalid")
        name = row.get("name")
        plan = by_name.get(str(name))
        ranges = row.get("base_covered_ranges")
        if plan is None or name in seen or not isinstance(ranges, list):
            raise SparseBundleError(
                "dense promotion materialization identity is invalid"
            )
        seen.add(str(name))
        normalized_ranges: list[dict[str, int]] = []
        cursor = -1
        covered = 0
        for raw_range in ranges:
            if not isinstance(raw_range, Mapping) or set(raw_range) != {
                "length",
                "relative_offset",
            }:
                raise SparseBundleError("dense base coverage range is invalid")
            offset = raw_range.get("relative_offset")
            length = raw_range.get("length")
            if (
                isinstance(offset, bool)
                or not isinstance(offset, int)
                or isinstance(length, bool)
                or not isinstance(length, int)
                or offset < 0
                or length <= 0
                or offset < cursor
                or offset + length > plan.length
            ):
                raise SparseBundleError("dense base coverage range exceeds plan")
            cursor = offset + length
            covered += length
            normalized_ranges.append({"length": length, "relative_offset": offset})
        if (
            covered >= plan.length
            or row.get("base_covered_bytes") != covered
            or row.get("plan_sha256") != _sha256(_dense_plan_record(plan))
        ):
            raise SparseBundleError("dense base coverage receipt is invalid")
        normalized_materializations.append(
            {
                "base_covered_bytes": covered,
                "base_covered_ranges": normalized_ranges,
                "name": plan.name,
                "plan_sha256": row["plan_sha256"],
            }
        )
    materialize_names = [str(row["name"]) for row in materializations]
    base_names = [plan.name for plan in plans if plan.name not in seen]
    if (
        list(materializations) != normalized_materializations
        or transaction.get("materializations_sha256") != _sha256(materializations)
        or transaction.get("materialization_names_sha256") != _sha256(materialize_names)
        or transaction.get("base_covered_names_sha256") != _sha256(base_names)
        or transaction.get("materialization_bytes")
        != sum(by_name[name].length for name in materialize_names)
    ):
        raise SparseBundleError("dense promotion materialization digest is invalid")

    evidence = transaction.get("trace_evidence")
    if (
        not isinstance(evidence, list)
        or not evidence
        or transaction.get("trace_evidence_sha256") != _sha256(evidence)
    ):
        raise SparseBundleError("dense promotion trace evidence is invalid")
    for row in evidence:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"operation_count", "sha256", "unique_leaves"}
            or re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256"))) is None
            or isinstance(row.get("operation_count"), bool)
            or not isinstance(row.get("operation_count"), int)
            or int(row["operation_count"]) < 1
            or isinstance(row.get("unique_leaves"), bool)
            or not isinstance(row.get("unique_leaves"), int)
            or int(row["unique_leaves"]) < 1
        ):
            raise SparseBundleError("dense promotion trace evidence row is invalid")
    remote = transaction.get("remote_source")
    if not isinstance(remote, Mapping) or set(remote) != {
        "inventory_layout_sha256",
        "layout_fingerprint",
        "repo_id",
        "revision",
        "transport",
    }:
        raise SparseBundleError("dense promotion remote source identity is invalid")
    if (
        remote.get("inventory_layout_sha256")
        != transaction.get("inventory_layout_sha256")
        or remote.get("transport") != "official-pinned-streamer/v1"
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(transaction.get(key))) is None
            for key in ("inventory_layout_sha256",)
        )
    ):
        raise SparseBundleError("dense promotion remote source digest is invalid")


def _dense_transaction(
    manifest: Mapping[str, Any],
    manifest_identity: Mapping[str, Any],
    inventory: Mapping[str, Any],
    trace_evidence: Sequence[Mapping[str, Any]],
    plans: Sequence[TensorRangePlan],
    materializations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    plan_records = [_dense_plan_record(plan) for plan in plans]
    names = [plan.name for plan in plans]
    materialize_names = [str(row["name"]) for row in materializations]
    materialize_set = set(materialize_names)
    inventory_layout_sha256 = _sha256(Streamer._inventory_layout_projection(inventory))
    core = {
        "base_covered_names_sha256": _sha256(
            [name for name in names if name not in materialize_set]
        ),
        "bundle_manifest": dict(manifest_identity),
        "inventory_layout_sha256": inventory_layout_sha256,
        "materialization_bytes": sum(
            plan.length for plan in plans if plan.name in materialize_set
        ),
        "materialization_names_sha256": _sha256(materialize_names),
        "materializations": list(materializations),
        "materializations_sha256": _sha256(list(materializations)),
        "plans": plan_records,
        "plans_sha256": _sha256(plan_records),
        "remote_source": {
            "inventory_layout_sha256": inventory_layout_sha256,
            "layout_fingerprint": manifest["layout_fingerprint"],
            "repo_id": manifest["logical_model"]["repo_id"],
            "revision": manifest["logical_model"]["revision"],
            "transport": "official-pinned-streamer/v1",
        },
        "required_tensor_bytes": sum(plan.length for plan in plans),
        "required_tensor_count": len(plans),
        "required_tensor_names_sha256": _sha256(names),
        "schema": DENSE_TRANSACTION_SCHEMA,
        "trace_evidence": list(trace_evidence),
        "trace_evidence_sha256": _sha256(list(trace_evidence)),
    }
    transaction = {**core, "transaction_id": _sha256(core)}
    _validate_dense_transaction(transaction)
    return transaction


def _dense_stage_filename(materialization: Mapping[str, Any]) -> str:
    digest = materialization.get("plan_sha256")
    if re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None:
        raise SparseBundleError("dense materialization plan digest is invalid")
    return f"dense-{digest}.bin"


def _clean_dense_stage_temps(stage_descriptor: int) -> None:
    pattern = re.compile(r"\.dense-[0-9a-f]{64}\.bin\.[0-9a-f]{32}\.tmp")
    for name in os.listdir(stage_descriptor):
        if pattern.fullmatch(name) is None:
            continue
        linked = os.stat(name, dir_fd=stage_descriptor, follow_symlinks=False)
        if not stat.S_ISREG(linked.st_mode):
            raise SparseBundleError("dense stage temporary is not a regular file")
        os.unlink(name, dir_fd=stage_descriptor)
    os.fsync(stage_descriptor)


def _reset_unjournaled_dense_stage(
    stage_descriptor: int, transaction: Mapping[str, Any]
) -> None:
    expected = {_dense_stage_filename(row) for row in transaction["materializations"]}
    temporary = re.compile(r"\.dense-[0-9a-f]{64}\.bin\.[0-9a-f]{32}\.tmp")
    actual = set(os.listdir(stage_descriptor))
    unexpected = {
        name
        for name in actual
        if name not in expected and temporary.fullmatch(name) is None
    }
    if unexpected:
        raise SparseBundleError("unjournaled dense stage has unexpected artifacts")
    for name in sorted(actual):
        linked = os.stat(name, dir_fd=stage_descriptor, follow_symlinks=False)
        if not stat.S_ISREG(linked.st_mode):
            raise SparseBundleError("unjournaled dense stage target is not regular")
    for name in sorted(actual):
        os.unlink(name, dir_fd=stage_descriptor)
    os.fsync(stage_descriptor)


def _stage_one_dense_tensor(
    source: Streamer,
    stage_descriptor: int,
    plan: TensorRangePlan,
    materialization: Mapping[str, Any],
    *,
    chunk_bytes: int,
) -> dict[str, Any]:
    filename = _dense_stage_filename(materialization)
    temporary = f".{filename}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    descriptor: int | None = None
    digest = hashlib.sha256()
    try:
        descriptor = os.open(temporary, flags, 0o600, dir_fd=stage_descriptor)
        cursor = 0
        while cursor < plan.length:
            take = min(chunk_bytes, plan.length - cursor)
            raw = source.raw_bytes(plan.shard, plan.absolute_offset + cursor, take)
            try:
                view = raw if isinstance(raw, memoryview) else memoryview(raw)
            except TypeError as exc:
                raise SparseBundleError(
                    "remote Streamer returned a non-buffer dense range"
                ) from exc
            if len(view) != take:
                raise SparseBundleError(
                    "remote Streamer returned a partial dense range"
                )
            written = 0
            while written < take:
                count = os.write(descriptor, view[written:])
                if count <= 0:
                    raise SparseBundleError("short staged dense write")
                written += count
            digest.update(view)
            cursor += take
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_size != plan.length:
            raise SparseBundleError("staged dense payload is invalid")
        os.close(descriptor)
        descriptor = None
        os.replace(
            temporary,
            filename,
            src_dir_fd=stage_descriptor,
            dst_dir_fd=stage_descriptor,
        )
        os.fsync(stage_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=stage_descriptor)
        except FileNotFoundError:
            pass
    observed = _hash_file_at(stage_descriptor, filename, plan.length)
    if observed != digest.hexdigest():
        raise SparseBundleError("staged dense readback digest does not match")
    return {
        "filename": filename,
        "length": plan.length,
        "name": plan.name,
        "plan_sha256": materialization["plan_sha256"],
        "sha256": observed,
    }


def _stage_dense_payloads(
    source: Streamer,
    stage_descriptor: int,
    transaction: Mapping[str, Any],
    *,
    resident_limit_bytes: int,
    staging_limit_bytes: int,
) -> tuple[dict[str, Any], ...]:
    _validate_dense_transaction(transaction)
    _preflight_dense_stage_bounds(
        stage_descriptor,
        transaction,
        resident_limit_bytes=resident_limit_bytes,
        staging_limit_bytes=staging_limit_bytes,
    )
    _clean_dense_stage_temps(stage_descriptor)
    plans = {
        plan.name: plan
        for plan in (_dense_plan_from_record(record) for record in transaction["plans"])
    }
    staged = tuple(
        _stage_one_dense_tensor(
            source,
            stage_descriptor,
            plans[str(row["name"])],
            row,
            chunk_bytes=min(resident_limit_bytes, 64 * 1024 * 1024),
        )
        for row in transaction["materializations"]
    )
    expected = {str(row["filename"]) for row in staged}
    actual = set(os.listdir(stage_descriptor))
    if actual != expected:
        raise SparseBundleError("dense stage contains unexpected artifacts")
    metrics = source.metrics()
    remote = transaction["remote_source"]
    if (
        metrics.get("repo_id") != remote["repo_id"]
        or metrics.get("revision") != remote["revision"]
        or metrics.get("inventory_source_fingerprint") != remote["layout_fingerprint"]
    ):
        raise SparseBundleError("remote Streamer identity changed during dense stage")
    return staged


def _preflight_dense_stage_bounds(
    stage_descriptor: int,
    transaction: Mapping[str, Any],
    *,
    resident_limit_bytes: int,
    staging_limit_bytes: int,
) -> None:
    if resident_limit_bytes < 1:
        raise SparseBundleError("dense resident memory bound is invalid")
    total = int(transaction["materialization_bytes"])
    if total > staging_limit_bytes:
        raise SparseBundleError("dense payload exceeds the staging disk bound")
    available = (
        os.fstatvfs(stage_descriptor).f_bavail * os.fstatvfs(stage_descriptor).f_frsize
    )
    if total + 64 * 1024 * 1024 > available:
        raise SparseBundleError("insufficient free disk for the complete dense stage")


def _verify_dense_staged_payloads(
    stage_descriptor: int,
    transaction: Mapping[str, Any],
    staged: object,
) -> tuple[dict[str, Any], ...]:
    materializations = transaction["materializations"]
    plans = {
        plan.name: plan
        for plan in (_dense_plan_from_record(record) for record in transaction["plans"])
    }
    if not isinstance(staged, list) or len(staged) != len(materializations):
        raise SparseBundleError("staged dense table is incomplete")
    verified: list[dict[str, Any]] = []
    for materialization, raw in zip(materializations, staged, strict=True):
        plan = plans[str(materialization["name"])]
        if not isinstance(raw, Mapping) or set(raw) != {
            "filename",
            "length",
            "name",
            "plan_sha256",
            "sha256",
        }:
            raise SparseBundleError("staged dense record schema is invalid")
        if (
            raw.get("filename") != _dense_stage_filename(materialization)
            or raw.get("length") != plan.length
            or raw.get("name") != plan.name
            or raw.get("plan_sha256") != materialization["plan_sha256"]
            or re.fullmatch(r"[0-9a-f]{64}", str(raw.get("sha256"))) is None
            or _hash_file_at(stage_descriptor, str(raw["filename"]), plan.length)
            != raw["sha256"]
        ):
            raise SparseBundleError("staged dense payload identity is invalid")
        verified.append(dict(raw))
    if set(os.listdir(stage_descriptor)) != {str(row["filename"]) for row in verified}:
        raise SparseBundleError("dense stage contains unexpected artifacts")
    return tuple(verified)


def _authenticate_base_dense_payloads(
    source: Streamer,
    weights_descriptor: int,
    transaction: Mapping[str, Any],
    *,
    resident_limit_bytes: int,
) -> tuple[dict[str, Any], ...]:
    materialized = {str(row["name"]) for row in transaction["materializations"]}
    chunk_bytes = min(resident_limit_bytes, 64 * 1024 * 1024)
    if chunk_bytes < 1:
        raise SparseBundleError("dense resident memory bound is invalid")
    authenticated: list[dict[str, Any]] = []
    for raw_plan in transaction["plans"]:
        plan = _dense_plan_from_record(raw_plan)
        if plan.name in materialized:
            continue
        descriptor = _open_shard_at(weights_descriptor, plan.shard, writable=False)
        digest = hashlib.sha256()
        try:
            opened = os.fstat(descriptor)
            if plan.absolute_end > opened.st_size:
                raise SparseBundleError("base dense tensor exceeds sparse shard")
            cursor = 0
            while cursor < plan.length:
                take = min(chunk_bytes, plan.length - cursor)
                local = _pread_exact(descriptor, take, plan.absolute_offset + cursor)
                remote = source.raw_bytes(
                    plan.shard, plan.absolute_offset + cursor, take
                )
                try:
                    view = (
                        remote if isinstance(remote, memoryview) else memoryview(remote)
                    )
                except TypeError as exc:
                    raise SparseBundleError(
                        "remote Streamer returned a non-buffer base dense range"
                    ) from exc
                if len(view) != take:
                    raise SparseBundleError(
                        "remote Streamer returned a partial base dense range"
                    )
                if local != view:
                    raise SparseBundleError(
                        f"base dense tensor disagrees with official source: {plan.name}"
                    )
                digest.update(view)
                cursor += take
        finally:
            os.close(descriptor)
        authenticated.append(
            {
                "length": plan.length,
                "name": plan.name,
                "plan_sha256": _sha256(raw_plan),
                "sha256": digest.hexdigest(),
            }
        )
    return tuple(authenticated)


def _verify_base_dense_payloads(
    weights_descriptor: int,
    transaction: Mapping[str, Any],
    payloads: object,
) -> tuple[dict[str, Any], ...]:
    materialized = {str(row["name"]) for row in transaction["materializations"]}
    base_plans = [
        raw_plan
        for raw_plan in transaction["plans"]
        if str(raw_plan["name"]) not in materialized
    ]
    if not isinstance(payloads, list) or len(payloads) != len(base_plans):
        raise SparseBundleError("base dense authentication table is incomplete")
    verified: list[dict[str, Any]] = []
    for raw_plan, raw in zip(base_plans, payloads, strict=True):
        plan = _dense_plan_from_record(raw_plan)
        if not isinstance(raw, Mapping) or set(raw) != {
            "length",
            "name",
            "plan_sha256",
            "sha256",
        }:
            raise SparseBundleError("base dense authentication row is invalid")
        if (
            raw.get("length") != plan.length
            or raw.get("name") != plan.name
            or raw.get("plan_sha256") != _sha256(raw_plan)
            or re.fullmatch(r"[0-9a-f]{64}", str(raw.get("sha256"))) is None
            or _hash_local_leaf(weights_descriptor, raw_plan) != raw["sha256"]
        ):
            raise SparseBundleError(
                f"base dense authenticated hash mismatch for {plan.name}"
            )
        verified.append(dict(raw))
    return tuple(verified)


def _dense_stage_tables(
    history: Sequence[Mapping[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    for record in reversed(history):
        if record.get("state") not in {
            "staged",
            "payload_durable",
            "binding_visible",
        }:
            continue
        payload = record.get("payload")
        if not isinstance(payload, Mapping):
            break
        staged = payload.get("staged_payloads")
        base = payload.get("base_payloads")
        if (
            isinstance(staged, list)
            and isinstance(base, list)
            and payload.get("staged_payloads_sha256") == _sha256(staged)
            and payload.get("base_payloads_sha256") == _sha256(base)
        ):
            return (
                tuple(dict(row) for row in staged if isinstance(row, Mapping)),
                tuple(dict(row) for row in base if isinstance(row, Mapping)),
            )
    raise SparseBundleError("dense journal has no complete authentication tables")


def _materialize_dense_payloads(
    weights_descriptor: int,
    stage_descriptor: int,
    transaction: Mapping[str, Any],
    staged: Sequence[Mapping[str, Any]],
) -> None:
    verified = _verify_dense_staged_payloads(
        stage_descriptor, transaction, list(staged)
    )
    plans = {
        plan.name: plan
        for plan in (_dense_plan_from_record(record) for record in transaction["plans"])
    }
    materializations = transaction["materializations"]
    shard_descriptors: dict[str, int] = {}
    shard_identities: dict[str, tuple[int, int]] = {}
    stage_descriptors: dict[str, int] = {}
    try:
        for row in materializations:
            plan = plans[str(row["name"])]
            if plan.shard not in shard_descriptors:
                descriptor = _open_shard_at(
                    weights_descriptor, plan.shard, writable=True
                )
                opened = os.fstat(descriptor)
                shard_descriptors[plan.shard] = descriptor
                shard_identities[plan.shard] = (opened.st_dev, opened.st_ino)
                if plan.absolute_end > opened.st_size:
                    raise SparseBundleError("dense tensor exceeds sparse shard size")
        for payload in verified:
            name = str(payload["filename"])
            flags = os.O_RDONLY | int(getattr(os, "O_CLOEXEC", 0))
            flags |= int(getattr(os, "O_NOFOLLOW", 0))
            descriptor = os.open(name, flags, dir_fd=stage_descriptor)
            opened = os.fstat(descriptor)
            linked = os.stat(name, dir_fd=stage_descriptor, follow_symlinks=False)
            if (
                not stat.S_ISREG(opened.st_mode)
                or not stat.S_ISREG(linked.st_mode)
                or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
                or opened.st_size != int(payload["length"])
                or _hash_descriptor_range(descriptor, 0, opened.st_size)
                != payload["sha256"]
            ):
                os.close(descriptor)
                raise SparseBundleError(
                    "staged dense payload changed before materialization"
                )
            stage_descriptors[name] = descriptor

        # Every trace-covered byte inside a partial tensor is already sealed by
        # the base bundle. It must agree with the official staged tensor before
        # any pwrite; only the uncovered remainder is replaceable.
        for row, payload in zip(materializations, verified, strict=True):
            plan = plans[str(row["name"])]
            shard_fd = shard_descriptors[plan.shard]
            stage_fd = stage_descriptors[str(payload["filename"])]
            for covered in row["base_covered_ranges"]:
                cursor = 0
                offset = int(covered["relative_offset"])
                length = int(covered["length"])
                while cursor < length:
                    take = min(1024 * 1024, length - cursor)
                    local = _pread_exact(
                        shard_fd, take, plan.absolute_offset + offset + cursor
                    )
                    official = _pread_exact(stage_fd, take, offset + cursor)
                    if local != official:
                        raise SparseBundleError(
                            f"base dense bytes disagree with official source for {plan.name}"
                        )
                    cursor += take

        for row, payload in zip(materializations, verified, strict=True):
            plan = plans[str(row["name"])]
            shard_fd = shard_descriptors[plan.shard]
            stage_fd = stage_descriptors[str(payload["filename"])]
            cursor = 0
            while cursor < plan.length:
                take = min(1024 * 1024, plan.length - cursor)
                chunk = _pread_exact(stage_fd, take, cursor)
                written = os.pwrite(shard_fd, chunk, plan.absolute_offset + cursor)
                if written != len(chunk):
                    raise SparseBundleError("short sparse dense pwrite")
                cursor += written
        for descriptor in shard_descriptors.values():
            os.fsync(descriptor)
        os.fsync(weights_descriptor)
        for row, payload in zip(materializations, verified, strict=True):
            plan = plans[str(row["name"])]
            if (
                _hash_descriptor_range(
                    shard_descriptors[plan.shard],
                    plan.absolute_offset,
                    plan.length,
                )
                != payload["sha256"]
            ):
                raise SparseBundleError("durable dense pwrite digest does not match")
        for shard, descriptor in shard_descriptors.items():
            linked = os.stat(shard, dir_fd=weights_descriptor, follow_symlinks=False)
            if (
                not stat.S_ISREG(linked.st_mode)
                or (linked.st_dev, linked.st_ino) != shard_identities[shard]
            ):
                raise SparseBundleError("sparse shard path changed during dense pwrite")
    finally:
        for descriptor in stage_descriptors.values():
            os.close(descriptor)
        for descriptor in shard_descriptors.values():
            os.close(descriptor)


def _dense_payload_ledger(
    weights_descriptor: int,
    transaction: Mapping[str, Any],
    staged: Sequence[Mapping[str, Any]],
    base_payloads: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    staged_by_name = {str(row["name"]): row for row in staged}
    base_by_name = {str(row["name"]): row for row in base_payloads}
    materialized = {str(row["name"]) for row in transaction["materializations"]}
    if set(staged_by_name) != materialized:
        raise SparseBundleError("dense staged payloads do not cover materializations")
    expected_base = {
        str(record["name"])
        for record in transaction["plans"]
        if str(record["name"]) not in materialized
    }
    if set(base_by_name) != expected_base:
        raise SparseBundleError("base dense authentication is incomplete")
    ledger: list[dict[str, Any]] = []
    for raw_plan in transaction["plans"]:
        plan = _dense_plan_from_record(raw_plan)
        digest = _hash_local_leaf(weights_descriptor, raw_plan)
        staged_row = staged_by_name.get(plan.name)
        source_row = staged_row or base_by_name.get(plan.name)
        if source_row is None or digest != source_row["sha256"]:
            raise SparseBundleError("authenticated dense tensor digest changed")
        ledger.append(
            {
                "length": plan.length,
                "name": plan.name,
                "plan_sha256": _sha256(raw_plan),
                "sha256": digest,
                "source": (
                    "official-pinned-streamer/v1"
                    if plan.name in materialized
                    else "official-pinned-streamer-verified-base/v1"
                ),
            }
        )
    return tuple(ledger)


def _verify_dense_payload_ledger(
    weights_descriptor: int,
    transaction: Mapping[str, Any],
    ledger: object,
    *,
    staged: Sequence[Mapping[str, Any]] | None = None,
    base_payloads: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], ...]:
    plans = transaction["plans"]
    materialized = {str(row["name"]) for row in transaction["materializations"]}
    if not isinstance(ledger, list) or len(ledger) != len(plans):
        raise SparseBundleError("dense payload ledger is incomplete")
    source_hashes: dict[str, str] | None = None
    if staged is not None or base_payloads is not None:
        source_hashes = {
            str(row["name"]): str(row["sha256"])
            for row in (*(staged or ()), *(base_payloads or ()))
        }
        if len(source_hashes) != len(plans):
            raise SparseBundleError("dense source authentication table is incomplete")
    verified: list[dict[str, Any]] = []
    for raw_plan, raw in zip(plans, ledger, strict=True):
        plan = _dense_plan_from_record(raw_plan)
        expected_source = (
            "official-pinned-streamer/v1"
            if plan.name in materialized
            else "official-pinned-streamer-verified-base/v1"
        )
        if not isinstance(raw, Mapping) or set(raw) != {
            "length",
            "name",
            "plan_sha256",
            "sha256",
            "source",
        }:
            raise SparseBundleError("dense payload ledger row is invalid")
        if (
            raw.get("length") != plan.length
            or raw.get("name") != plan.name
            or raw.get("plan_sha256") != _sha256(raw_plan)
            or raw.get("source") != expected_source
            or re.fullmatch(r"[0-9a-f]{64}", str(raw.get("sha256"))) is None
            or (
                source_hashes is not None
                and source_hashes.get(plan.name) != raw.get("sha256")
            )
            or _hash_local_leaf(weights_descriptor, raw_plan) != raw["sha256"]
        ):
            raise SparseBundleError(
                f"dense payload ledger hash mismatch for {plan.name}"
            )
        verified.append(dict(raw))
    return tuple(verified)


def _preflight_dense_graph(
    root: Path,
    model: LogicalModelIdentity,
    plans: Sequence[TensorRangePlan],
    *,
    layout_fingerprint: str,
    budget_mb: float,
) -> dict[str, Any]:
    with CausalWeightMount(root, model, budget_mb=budget_mb) as mount:
        if mount.layout.layout_fingerprint != layout_fingerprint:
            raise SparseBundleError("mounted dense layout does not match manifest")
        revision = _revision_record(mount.graph.store.revision())
        for plan in plans:
            try:
                existing = mount.resolve_tensor_plan(plan.name)
            except KeyError:
                continue
            if existing != plan:
                raise SparseBundleError(
                    f"causal graph conflicts for dense tensor {plan.name}"
                )
        return revision


def _dense_binding_result(
    receipt: Any,
    *,
    graph_pre_transaction: Mapping[str, Any],
    graph_pre_bind: Mapping[str, Any],
    graph_post_bind: Mapping[str, Any],
    staged: Sequence[Mapping[str, Any]],
    base_payloads: Sequence[Mapping[str, Any]],
    ledger: Sequence[Mapping[str, Any]],
    growth_before: Mapping[str, Any],
    growth_after: Mapping[str, Any],
) -> dict[str, Any]:
    bindings = [
        {
            "appended": bool(binding.appended),
            "name": binding.name,
            "record_index": binding.record_index,
            "segment_sha256": binding.segment_sha256,
        }
        for binding in receipt.bindings
    ]
    return {
        "base_payloads": list(base_payloads),
        "base_payloads_sha256": _sha256(list(base_payloads)),
        "binding_receipts": bindings,
        "binding_receipts_sha256": _sha256(bindings),
        "graph_post_bind": dict(graph_post_bind),
        "graph_pre_bind": dict(graph_pre_bind),
        "graph_pre_transaction": dict(graph_pre_transaction),
        "growth": {
            "logical_bytes_after": growth_after["logical_bytes"],
            "logical_bytes_before": growth_before["logical_bytes"],
            "logical_bytes_delta": (
                growth_after["logical_bytes"] - growth_before["logical_bytes"]
            ),
            "physical_bytes_after": growth_after["physical_bytes"],
            "physical_bytes_before": growth_before["physical_bytes"],
            "physical_bytes_delta": (
                growth_after["physical_bytes"] - growth_before["physical_bytes"]
            ),
            "shards_after": growth_after["shards"],
            "shards_before": growth_before["shards"],
        },
        "payload_ledger": list(ledger),
        "payload_ledger_sha256": _sha256(list(ledger)),
        "staged_payloads": list(staged),
        "staged_payloads_sha256": _sha256(list(staged)),
    }


def _verify_dense_result_schema(
    result: Mapping[str, Any], transaction: Mapping[str, Any]
) -> None:
    required = {
        "base_payloads",
        "base_payloads_sha256",
        "binding_receipts",
        "binding_receipts_sha256",
        "graph_post_bind",
        "graph_pre_bind",
        "graph_pre_transaction",
        "growth",
        "payload_ledger",
        "payload_ledger_sha256",
        "staged_payloads",
        "staged_payloads_sha256",
    }
    if set(result) != required:
        raise SparseBundleError("dense promotion result schema is invalid")
    bindings = result.get("binding_receipts")
    base_payloads = result.get("base_payloads")
    ledger = result.get("payload_ledger")
    staged = result.get("staged_payloads")
    plans = transaction["plans"]
    materializations = transaction["materializations"]
    base_count = len(plans) - len(materializations)
    if (
        not isinstance(base_payloads, list)
        or len(base_payloads) != base_count
        or result.get("base_payloads_sha256") != _sha256(base_payloads)
        or not isinstance(bindings, list)
        or len(bindings) != len(plans)
        or result.get("binding_receipts_sha256") != _sha256(bindings)
        or not isinstance(ledger, list)
        or len(ledger) != len(plans)
        or result.get("payload_ledger_sha256") != _sha256(ledger)
        or not isinstance(staged, list)
        or len(staged) != len(materializations)
        or result.get("staged_payloads_sha256") != _sha256(staged)
    ):
        raise SparseBundleError("dense promotion result table digest is invalid")
    materialized_names = {str(row["name"]) for row in materializations}
    base_plans = [
        raw_plan
        for raw_plan in plans
        if str(raw_plan["name"]) not in materialized_names
    ]
    for raw_plan, row in zip(base_plans, base_payloads, strict=True):
        plan = _dense_plan_from_record(raw_plan)
        if (
            not isinstance(row, Mapping)
            or set(row) != {"length", "name", "plan_sha256", "sha256"}
            or row.get("length") != plan.length
            or row.get("name") != plan.name
            or row.get("plan_sha256") != _sha256(raw_plan)
            or re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256"))) is None
        ):
            raise SparseBundleError("base dense source receipt is invalid")
    plan_by_name = {str(raw_plan["name"]): raw_plan for raw_plan in plans}
    for materialization, row in zip(materializations, staged, strict=True):
        raw_plan = plan_by_name[str(materialization["name"])]
        plan = _dense_plan_from_record(raw_plan)
        if (
            not isinstance(row, Mapping)
            or set(row) != {"filename", "length", "name", "plan_sha256", "sha256"}
            or row.get("filename") != _dense_stage_filename(materialization)
            or row.get("length") != plan.length
            or row.get("name") != plan.name
            or row.get("plan_sha256") != _sha256(raw_plan)
            or re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256"))) is None
        ):
            raise SparseBundleError("staged dense source receipt is invalid")
    for raw_plan, binding in zip(plans, bindings, strict=True):
        plan = _dense_plan_from_record(raw_plan)
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"appended", "name", "record_index", "segment_sha256"}
            or not isinstance(binding.get("appended"), bool)
            or binding.get("name") != plan.name
            or isinstance(binding.get("record_index"), bool)
            or not isinstance(binding.get("record_index"), int)
            or int(binding["record_index"]) < 0
            or re.fullmatch(r"[0-9a-f]{64}", str(binding.get("segment_sha256"))) is None
        ):
            raise SparseBundleError("dense promotion binding receipt is invalid")
    for key in ("graph_pre_transaction", "graph_pre_bind", "graph_post_bind"):
        revision = result.get(key)
        if not isinstance(revision, Mapping) or set(revision) != {
            "sequence",
            "sha256",
        }:
            raise SparseBundleError("dense promotion graph revision is invalid")
        _revision_record((revision["sequence"], revision["sha256"]))
    growth = result.get("growth")
    if not isinstance(growth, Mapping) or set(growth) != {
        "logical_bytes_after",
        "logical_bytes_before",
        "logical_bytes_delta",
        "physical_bytes_after",
        "physical_bytes_before",
        "physical_bytes_delta",
        "shards_after",
        "shards_before",
    }:
        raise SparseBundleError("dense promotion growth receipt is invalid")
    if (
        growth["logical_bytes_delta"]
        != growth["logical_bytes_after"] - growth["logical_bytes_before"]
        or growth["physical_bytes_delta"]
        != growth["physical_bytes_after"] - growth["physical_bytes_before"]
        or growth["logical_bytes_delta"] != 0
        or growth["physical_bytes_delta"] < 0
    ):
        raise SparseBundleError("dense promotion physical/logical growth is invalid")


def _verify_dense_completed_receipt_payload(
    body: Mapping[str, Any],
    weights_descriptor: int,
    root: Path,
    model: LogicalModelIdentity,
    *,
    budget_mb: float,
) -> dict[str, Any]:
    transaction = body.get("transaction")
    result = body.get("result")
    if not isinstance(transaction, Mapping) or not isinstance(result, Mapping):
        raise SparseBundleError("dense promotion receipt body is incomplete")
    _validate_dense_transaction(transaction)
    _verify_dense_result_schema(result, transaction)
    staged = tuple(dict(row) for row in result["staged_payloads"])
    base_payloads = _verify_base_dense_payloads(
        weights_descriptor, transaction, result["base_payloads"]
    )
    ledger = _verify_dense_payload_ledger(
        weights_descriptor,
        transaction,
        result["payload_ledger"],
        staged=staged,
        base_payloads=base_payloads,
    )
    staged_by_name = {str(row["name"]): row for row in staged}
    ledger_by_name = {str(row["name"]): row for row in ledger}
    for materialization in transaction["materializations"]:
        name = str(materialization["name"])
        if (
            name not in staged_by_name
            or ledger_by_name[name]["sha256"] != staged_by_name[name]["sha256"]
        ):
            raise SparseBundleError("dense materialization lost its source digest")

    plans = tuple(_dense_plan_from_record(record) for record in transaction["plans"])
    bindings = result["binding_receipts"]
    with CausalWeightMount(root, model, budget_mb=budget_mb) as mount:
        remote = transaction["remote_source"]
        if mount.layout.layout_fingerprint != remote["layout_fingerprint"]:
            raise SparseBundleError("dense promotion layout no longer mounts")
        current = _revision_record(mount.graph.store.revision())
        post = result["graph_post_bind"]
        if current["sequence"] < post["sequence"]:
            raise SparseBundleError("causal graph predates dense promotion receipt")
        active_segments = set(mount.graph.store.segments())
        resolved = mount.tensor_reader.resolve_tensor_plans(plan.name for plan in plans)
        if resolved != plans:
            raise SparseBundleError("resolved dense tensor plans do not match receipt")
        for raw_plan, binding in zip(transaction["plans"], bindings, strict=True):
            segment = str(binding["segment_sha256"])
            if segment not in active_segments:
                raise SparseBundleError("dense promotion citation is not active")
            cited = mount.graph.store.record(segment, int(binding["record_index"]))
            if (
                cited.get("schema") != "causal-tensor-binding/v1"
                or cited.get("record_type") != "tensor_range_plan"
                or cited.get("logical_model") != model.as_record()
                or cited.get("layout_fingerprint") != remote["layout_fingerprint"]
                or cited.get("plan") != raw_plan
                or cited.get("plan_sha256") != _sha256(raw_plan)
            ):
                raise SparseBundleError(
                    "dense promotion citation record does not match"
                )
    return {
        "binding_count": len(plans),
        "graph_revision": dict(result["graph_post_bind"]),
        "materialization_bytes": int(transaction["materialization_bytes"]),
        "payload_ledger_sha256": result["payload_ledger_sha256"],
        "plans_sha256": transaction["plans_sha256"],
        "required_tensor_count": int(transaction["required_tensor_count"]),
        "required_tensor_names_sha256": transaction["required_tensor_names_sha256"],
        "transaction_id": transaction["transaction_id"],
    }


def _general_manifest_document(
    base_manifest: Mapping[str, Any],
    receipt_document: Mapping[str, Any],
) -> dict[str, Any]:
    body = _verify_dense_receipt(receipt_document)
    transaction = body["transaction"]
    result = body["result"]
    coverage = {
        "graph_post_bind": dict(result["graph_post_bind"]),
        "materialization_bytes": transaction["materialization_bytes"],
        "payload_ledger_sha256": result["payload_ledger_sha256"],
        "plans_sha256": transaction["plans_sha256"],
        "receipt_sha256": receipt_document["sha256"],
        "required_tensor_count": transaction["required_tensor_count"],
        "required_tensor_names_sha256": transaction["required_tensor_names_sha256"],
        "transaction_id": transaction["transaction_id"],
    }
    manifest_body = {
        "base_bundle": dict(base_manifest),
        "capabilities": {
            "general_dense_weight_coverage": GENERAL_DENSE_COVERAGE_CAPABILITY
        },
        "dense_coverage": coverage,
        "layout_fingerprint": base_manifest["layout_fingerprint"],
        "logical_model": dict(base_manifest["logical_model"]),
        "weights_layout": "nested/v1",
    }
    document = {
        "body": manifest_body,
        "schema": GENERAL_BUNDLE_SCHEMA,
        "sha256": _sha256(manifest_body),
    }
    _verify_general_bundle_manifest(document)
    return document


def _unlink_dense_pending(dense_descriptor: int) -> None:
    try:
        linked = os.stat(_DENSE_PENDING, dir_fd=dense_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(linked.st_mode):
        raise SparseBundleError("pending dense promotion path is not a regular file")
    os.unlink(_DENSE_PENDING, dir_fd=dense_descriptor)
    os.fsync(dense_descriptor)


def _cleanup_dense_stage(
    dense_descriptor: int,
    stage_name: str,
    staged: Sequence[Mapping[str, Any]],
) -> None:
    try:
        stage_descriptor = _open_directory_at(dense_descriptor, stage_name)
    except SparseBundleError:
        try:
            os.stat(stage_name, dir_fd=dense_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise
    try:
        expected = {str(row["filename"]) for row in staged}
        temporary = re.compile(r"\.dense-[0-9a-f]{64}\.bin\.[0-9a-f]{32}\.tmp")
        actual = set(os.listdir(stage_descriptor))
        unexpected = {
            name
            for name in actual
            if name not in expected and temporary.fullmatch(name) is None
        }
        if unexpected:
            raise SparseBundleError("dense stage contains unexpected cleanup artifacts")
        for name in sorted(actual):
            linked = os.stat(name, dir_fd=stage_descriptor, follow_symlinks=False)
            if not stat.S_ISREG(linked.st_mode):
                raise SparseBundleError("dense stage cleanup target is not regular")
        for name in sorted(actual):
            os.unlink(name, dir_fd=stage_descriptor)
        os.fsync(stage_descriptor)
    finally:
        os.close(stage_descriptor)
    os.rmdir(stage_name, dir_fd=dense_descriptor)
    os.fsync(dense_descriptor)


def _load_pinned_inventory_at(
    weights_descriptor: int, manifest: Mapping[str, Any]
) -> Mapping[str, Any]:
    document, encoded = _read_json_at(weights_descriptor, "inventory.pinned.json")
    if (
        _sha256_bytes(encoded) != manifest.get("pinned_inventory_sha256")
        or not isinstance(document, Mapping)
        or document.get("schema") != "immer.tensor-inventory-cache/v1"
        or document.get("source_fingerprint") != manifest["layout_fingerprint"]
        or not isinstance(document.get("inventory"), Mapping)
    ):
        raise SparseBundleError("bundle pinned inventory identity is invalid")
    inventory = document["inventory"]
    if (
        inventory.get("repo") != manifest["logical_model"]["repo_id"]
        or inventory.get("revision") != manifest["logical_model"]["revision"]
    ):
        raise SparseBundleError("bundle pinned inventory logical model is invalid")
    return inventory


def _require_dense_transaction_identity(
    transaction: Mapping[str, Any],
    *,
    manifest_identity: Mapping[str, Any],
    repo_id: str,
    revision: str,
    layout_fingerprint: str,
) -> None:
    _validate_dense_transaction(transaction)
    remote = transaction["remote_source"]
    if (
        transaction.get("bundle_manifest") != manifest_identity
        or remote.get("repo_id") != repo_id
        or remote.get("revision") != revision
        or remote.get("layout_fingerprint") != layout_fingerprint
    ):
        raise SparseBundleError("dense promotion transaction identity does not match")


def _require_dense_inventory_contract(
    transaction: Mapping[str, Any],
    inventory: Mapping[str, Any],
    *,
    repo_id: str,
    revision: str,
    layout_fingerprint: str,
) -> tuple[TensorRangePlan, ...]:
    plans = _dense_plans_from_inventory(
        inventory,
        repo_id=repo_id,
        revision=revision,
        layout_fingerprint=layout_fingerprint,
    )
    records = [_dense_plan_record(plan) for plan in plans]
    if records != transaction.get("plans") or _sha256(
        Streamer._inventory_layout_projection(inventory)
    ) != transaction.get("inventory_layout_sha256"):
        raise SparseBundleError(
            "dense promotion transaction disagrees with pinned inventory"
        )
    return plans


def _dense_planned_payload(
    history: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    first = history[0]
    payload = first.get("payload")
    if (
        first.get("state") != "planned"
        or not isinstance(payload, Mapping)
        or set(payload) != {"graph_pre_transaction", "growth_before"}
        or not isinstance(payload.get("graph_pre_transaction"), Mapping)
        or not isinstance(payload.get("growth_before"), Mapping)
    ):
        raise SparseBundleError("dense promotion planned state is invalid")
    return dict(payload["graph_pre_transaction"]), dict(payload["growth_before"])


def _dense_durable_payload(
    history: Sequence[Mapping[str, Any]], state: str
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    dict[str, Any],
]:
    index = -2 if state == "binding_visible" else -1
    payload = history[index].get("payload")
    if (
        not isinstance(payload, Mapping)
        or not isinstance(payload.get("staged_payloads"), list)
        or not isinstance(payload.get("base_payloads"), list)
        or not isinstance(payload.get("payload_ledger"), list)
        or not isinstance(payload.get("growth_after"), Mapping)
        or payload.get("staged_payloads_sha256") != _sha256(payload["staged_payloads"])
        or payload.get("base_payloads_sha256") != _sha256(payload["base_payloads"])
        or payload.get("payload_ledger_sha256") != _sha256(payload["payload_ledger"])
    ):
        raise SparseBundleError("dense promotion durable state is invalid")
    return (
        tuple(dict(row) for row in payload["staged_payloads"]),
        tuple(dict(row) for row in payload["base_payloads"]),
        tuple(dict(row) for row in payload["payload_ledger"]),
        dict(payload["growth_after"]),
    )


def _scan_dense_receipts(
    receipts_descriptor: int,
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    receipts: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for name in sorted(os.listdir(receipts_descriptor)):
        if re.fullmatch(r"[0-9a-f]{64}\.json", name) is None:
            raise SparseBundleError(
                f"unexpected dense promotion receipt artifact: {name!r}"
            )
        document, _encoded = _read_json_at(receipts_descriptor, name)
        body = _verify_dense_receipt(document)
        if name != f"{body['transaction']['transaction_id']}.json":
            raise SparseBundleError("dense promotion receipt filename is invalid")
        receipts.append((name, body, dict(document)))
    return receipts


def promote_dense(args: argparse.Namespace) -> dict[str, Any]:
    if bool(getattr(args, "plan_only", False)):
        return plan_dense_promotion(args)
    root = Path(args.bundle).expanduser().absolute()
    model = LogicalModelIdentity(repo_id=args.repo_id, revision=args.revision)
    resident_limit_bytes = int(float(args.resident_limit_mb) * 1024 * 1024)
    staging_limit_bytes = int(float(args.staging_limit_mb) * 1024 * 1024)

    preflight_root = _open_plain_root(root)
    try:
        _load_bundle_manifest_at(
            preflight_root,
            repo_id=args.repo_id,
            revision=args.revision,
            expected_layout=getattr(args, "layout_fingerprint", None),
        )
    finally:
        os.close(preflight_root)

    with _append_lock(root) as (
        root_descriptor,
        append_descriptor,
        weights_descriptor,
    ):
        manifest, manifest_identity = _load_bundle_manifest_at(
            root_descriptor,
            repo_id=args.repo_id,
            revision=args.revision,
            expected_layout=getattr(args, "layout_fingerprint", None),
        )
        _base, promoted_manifest = _bundle_documents_at(root_descriptor)
        if _optional_json_at(append_descriptor, _APPEND_PENDING) is not None:
            raise SparseBundleError(
                "expert append is pending; reconcile it before dense promotion"
            )
        inventory = _load_pinned_inventory_at(weights_descriptor, manifest)
        dense_descriptor = _open_directory_at(
            root_descriptor, _DENSE_DIRECTORY, create=True
        )
        receipts_descriptor: int | None = None
        stage_descriptor: int | None = None
        try:
            receipts_descriptor = _open_directory_at(
                dense_descriptor, "receipts", create=True
            )
            receipts = _scan_dense_receipts(receipts_descriptor)
            pending_raw = _optional_json_at(dense_descriptor, _DENSE_PENDING)
            pending: tuple[dict[str, Any], tuple[dict[str, Any], ...]] | None = None
            if pending_raw is not None:
                pending = _verify_dense_pending(pending_raw[0])

            if receipts:
                if len(receipts) != 1:
                    raise SparseBundleError(
                        "multiple dense promotion receipts conflict"
                    )
                _name, completed_body, completed_document = receipts[0]
                transaction = completed_body["transaction"]
                _require_dense_transaction_identity(
                    transaction,
                    manifest_identity=manifest_identity,
                    repo_id=args.repo_id,
                    revision=args.revision,
                    layout_fingerprint=str(manifest["layout_fingerprint"]),
                )
                _require_dense_inventory_contract(
                    transaction,
                    inventory,
                    repo_id=args.repo_id,
                    revision=args.revision,
                    layout_fingerprint=str(manifest["layout_fingerprint"]),
                )
                summary = _verify_dense_completed_receipt_payload(
                    completed_body,
                    weights_descriptor,
                    root,
                    model,
                    budget_mb=float(args.budget_mb),
                )
                if pending is not None and pending[0] != transaction:
                    raise SparseBundleError(
                        "pending dense promotion conflicts with completed receipt"
                    )
                expected_manifest = _general_manifest_document(
                    manifest, completed_document
                )
                if promoted_manifest is None:
                    _atomic_json_at(root_descriptor, "bundle.json", expected_manifest)
                    promoted_manifest = expected_manifest
                elif promoted_manifest != expected_manifest:
                    raise SparseBundleError(
                        "published dense capability conflicts with receipt"
                    )
                if pending is not None:
                    _unlink_dense_pending(dense_descriptor)
                staged = completed_body["result"]["staged_payloads"]
                _cleanup_dense_stage(
                    dense_descriptor,
                    f"stage-{transaction['transaction_id']}",
                    staged,
                )
                return {
                    **summary,
                    "capability": GENERAL_DENSE_COVERAGE_CAPABILITY,
                    "receipt_sha256": completed_document["sha256"],
                    "status": "already-promoted",
                }

            if promoted_manifest is not None:
                raise SparseBundleError(
                    "general dense capability has no completed promotion receipt"
                )

            if pending is None:
                plans = _dense_plans_from_inventory(
                    inventory,
                    repo_id=args.repo_id,
                    revision=args.revision,
                    layout_fingerprint=str(manifest["layout_fingerprint"]),
                )
                trace_evidence, intervals = _dense_trace_evidence(root, args, manifest)
                materializations = _dense_materialization_records(plans, intervals)
                transaction = _dense_transaction(
                    manifest,
                    manifest_identity,
                    inventory,
                    trace_evidence,
                    plans,
                    materializations,
                )
                graph_pre = _preflight_dense_graph(
                    root,
                    model,
                    plans,
                    layout_fingerprint=str(manifest["layout_fingerprint"]),
                    budget_mb=float(args.budget_mb),
                )
                affected_shards = [
                    plan.shard
                    for plan in plans
                    if plan.name in {str(row["name"]) for row in materializations}
                ]
                growth_before = _shard_stats(weights_descriptor, affected_shards)
                history: tuple[dict[str, Any], ...] = (
                    _journal_event(
                        (),
                        "planned",
                        {
                            "graph_pre_transaction": graph_pre,
                            "growth_before": growth_before,
                        },
                    ),
                )
                _atomic_json_at(
                    dense_descriptor,
                    _DENSE_PENDING,
                    _dense_pending_document(transaction, history),
                )
            else:
                transaction, history = pending
                _require_dense_transaction_identity(
                    transaction,
                    manifest_identity=manifest_identity,
                    repo_id=args.repo_id,
                    revision=args.revision,
                    layout_fingerprint=str(manifest["layout_fingerprint"]),
                )
                plans = _require_dense_inventory_contract(
                    transaction,
                    inventory,
                    repo_id=args.repo_id,
                    revision=args.revision,
                    layout_fingerprint=str(manifest["layout_fingerprint"]),
                )
                graph_pre, growth_before = _dense_planned_payload(history)

            state = str(history[-1]["state"])
            if state not in {
                "planned",
                "staged",
                "payload_durable",
                "binding_visible",
            }:
                raise SparseBundleError("pending dense promotion state is unknown")
            stage_name = f"stage-{transaction['transaction_id']}"
            stage_descriptor = _open_directory_at(
                dense_descriptor, stage_name, create=True
            )

            if state == "planned":
                _reset_unjournaled_dense_stage(stage_descriptor, transaction)
                _preflight_dense_stage_bounds(
                    stage_descriptor,
                    transaction,
                    resident_limit_bytes=resident_limit_bytes,
                    staging_limit_bytes=staging_limit_bytes,
                )
                source = _open_remote_source(args)
                try:
                    _adopt_bundle_inventory(source, weights_descriptor, manifest)
                    metrics = source.metrics()
                    remote = transaction["remote_source"]
                    if (
                        metrics.get("repo_id") != remote["repo_id"]
                        or metrics.get("revision") != remote["revision"]
                        or metrics.get("inventory_source_fingerprint")
                        != remote["layout_fingerprint"]
                        or _sha256(
                            Streamer._inventory_layout_projection(source.inventory())
                        )
                        != remote["inventory_layout_sha256"]
                    ):
                        raise SparseBundleError(
                            "remote Streamer identity does not match dense plan"
                        )
                    base_payloads = _authenticate_base_dense_payloads(
                        source,
                        weights_descriptor,
                        transaction,
                        resident_limit_bytes=resident_limit_bytes,
                    )
                    staged = _stage_dense_payloads(
                        source,
                        stage_descriptor,
                        transaction,
                        resident_limit_bytes=resident_limit_bytes,
                        staging_limit_bytes=staging_limit_bytes,
                    )
                finally:
                    source.close()
                history = (
                    *history,
                    _journal_event(
                        history,
                        "staged",
                        {
                            "base_payloads": list(base_payloads),
                            "base_payloads_sha256": _sha256(list(base_payloads)),
                            "staged_payloads": list(staged),
                            "staged_payloads_sha256": _sha256(list(staged)),
                        },
                    ),
                )
                _atomic_json_at(
                    dense_descriptor,
                    _DENSE_PENDING,
                    _dense_pending_document(transaction, history),
                )
                state = "staged"
            else:
                staged, base_payloads = _dense_stage_tables(history)
                staged = _verify_dense_staged_payloads(
                    stage_descriptor, transaction, list(staged)
                )
                base_payloads = _verify_base_dense_payloads(
                    weights_descriptor, transaction, list(base_payloads)
                )

            if state == "staged":
                staged = _verify_dense_staged_payloads(
                    stage_descriptor, transaction, list(staged)
                )
                _materialize_dense_payloads(
                    weights_descriptor, stage_descriptor, transaction, staged
                )
                ledger = _dense_payload_ledger(
                    weights_descriptor, transaction, staged, base_payloads
                )
                affected_shards = [
                    _dense_plan_from_record(record).shard
                    for record in transaction["plans"]
                    if str(record["name"])
                    in {str(row["name"]) for row in transaction["materializations"]}
                ]
                growth_after = _shard_stats(weights_descriptor, affected_shards)
                history = (
                    *history,
                    _journal_event(
                        history,
                        "payload_durable",
                        {
                            "base_payloads": list(base_payloads),
                            "base_payloads_sha256": _sha256(list(base_payloads)),
                            "growth_after": growth_after,
                            "payload_ledger": list(ledger),
                            "payload_ledger_sha256": _sha256(list(ledger)),
                            "staged_payloads": list(staged),
                            "staged_payloads_sha256": _sha256(list(staged)),
                        },
                    ),
                )
                _atomic_json_at(
                    dense_descriptor,
                    _DENSE_PENDING,
                    _dense_pending_document(transaction, history),
                )
                state = "payload_durable"
                _failpoint(args, "payload-before-binding")
            else:
                staged, base_payloads, ledger, growth_after = _dense_durable_payload(
                    history, state
                )
                _verify_dense_staged_payloads(
                    stage_descriptor, transaction, list(staged)
                )
                ledger = _verify_dense_payload_ledger(
                    weights_descriptor,
                    transaction,
                    list(ledger),
                    staged=staged,
                    base_payloads=base_payloads,
                )

            if state == "payload_durable":
                with CausalWeightMount(
                    root, model, budget_mb=float(args.budget_mb)
                ) as mount:
                    if (
                        mount.layout.layout_fingerprint
                        != manifest["layout_fingerprint"]
                    ):
                        raise SparseBundleError(
                            "mounted layout changed before dense binding"
                        )
                    graph_pre_bind = _revision_record(mount.graph.store.revision())
                    binding_receipt = mount.bind_tensor_plans(plans)
                    graph_post_bind = _revision_record(mount.graph.store.revision())
                _failpoint(args, "binding-before-journal")
                result = _dense_binding_result(
                    binding_receipt,
                    graph_pre_transaction=graph_pre,
                    graph_pre_bind=graph_pre_bind,
                    graph_post_bind=graph_post_bind,
                    staged=staged,
                    base_payloads=base_payloads,
                    ledger=ledger,
                    growth_before=growth_before,
                    growth_after=growth_after,
                )
                _verify_dense_result_schema(result, transaction)
                history = (
                    *history,
                    _journal_event(history, "binding_visible", result),
                )
                _atomic_json_at(
                    dense_descriptor,
                    _DENSE_PENDING,
                    _dense_pending_document(transaction, history),
                )
                state = "binding_visible"

            if state != "binding_visible":
                raise SparseBundleError(
                    f"unsupported pending dense promotion state: {state}"
                )
            receipt_document = _dense_receipt_document(transaction, history)
            receipt_name = f"{transaction['transaction_id']}.json"
            _atomic_json_at(receipts_descriptor, receipt_name, receipt_document)
            completed_body = _verify_dense_receipt(receipt_document)
            summary = _verify_dense_completed_receipt_payload(
                completed_body,
                weights_descriptor,
                root,
                model,
                budget_mb=float(args.budget_mb),
            )
            _failpoint(args, "receipt-before-capability")
            general_manifest = _general_manifest_document(manifest, receipt_document)
            _atomic_json_at(root_descriptor, "bundle.json", general_manifest)
            _failpoint(args, "capability-before-cleanup")
            _unlink_dense_pending(dense_descriptor)
            _cleanup_dense_stage(dense_descriptor, stage_name, staged)
            return {
                **summary,
                "appended_bindings": sum(
                    bool(row["appended"])
                    for row in completed_body["result"]["binding_receipts"]
                ),
                "capability": GENERAL_DENSE_COVERAGE_CAPABILITY,
                "receipt_sha256": receipt_document["sha256"],
                "status": "promoted",
            }
        finally:
            if stage_descriptor is not None:
                os.close(stage_descriptor)
            if receipts_descriptor is not None:
                os.close(receipts_descriptor)
            os.close(dense_descriptor)


def _verify_dense_store(
    root: Path,
    model: LogicalModelIdentity,
    base_manifest: Mapping[str, Any],
    promoted_manifest: Mapping[str, Any] | None,
    *,
    repo_id: str,
    revision: str,
    expected_layout: str | None,
    budget_mb: float,
) -> dict[str, Any]:
    dense_path = root / _DENSE_DIRECTORY
    try:
        metadata = dense_path.lstat()
    except FileNotFoundError:
        if promoted_manifest is not None:
            raise SparseBundleError("dense capability has no promotion store")
        return {
            "capability": None,
            "receipt_sha256": None,
            "required_tensor_count": 0,
        }
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise SparseBundleError("dense promotion store must be a non-symlink directory")

    with _append_lock(root) as (
        root_descriptor,
        append_descriptor,
        weights_descriptor,
    ):
        _manifest, manifest_identity = _load_bundle_manifest_at(
            root_descriptor,
            repo_id=repo_id,
            revision=revision,
            expected_layout=expected_layout,
        )
        inventory = _load_pinned_inventory_at(weights_descriptor, base_manifest)
        if _optional_json_at(append_descriptor, _APPEND_PENDING) is not None:
            raise SparseBundleError(
                "expert append is pending while dense capability is active"
            )
        dense_descriptor = _open_directory_at(root_descriptor, _DENSE_DIRECTORY)
        receipts_descriptor: int | None = None
        try:
            if _optional_json_at(dense_descriptor, _DENSE_PENDING) is not None:
                raise SparseBundleError(
                    "pending dense promotion requires reconciliation"
                )
            if promoted_manifest is None:
                raise SparseBundleError(
                    "dense promotion store exists without published capability"
                )
            if set(os.listdir(dense_descriptor)) != {"receipts"}:
                raise SparseBundleError(
                    "dense promotion store has unexpected artifacts"
                )
            receipts_descriptor = _open_directory_at(dense_descriptor, "receipts")
            receipts = _scan_dense_receipts(receipts_descriptor)
            if len(receipts) != 1:
                raise SparseBundleError("dense capability requires exactly one receipt")
            _name, body, document = receipts[0]
            transaction = body["transaction"]
            _require_dense_transaction_identity(
                transaction,
                manifest_identity=manifest_identity,
                repo_id=repo_id,
                revision=revision,
                layout_fingerprint=str(base_manifest["layout_fingerprint"]),
            )
            _require_dense_inventory_contract(
                transaction,
                inventory,
                repo_id=repo_id,
                revision=revision,
                layout_fingerprint=str(base_manifest["layout_fingerprint"]),
            )
            summary = _verify_dense_completed_receipt_payload(
                body,
                weights_descriptor,
                root,
                model,
                budget_mb=budget_mb,
            )
            expected = _general_manifest_document(base_manifest, document)
            if dict(promoted_manifest) != expected:
                raise SparseBundleError(
                    "dense capability manifest is not receipt-bound"
                )
            return {
                **summary,
                "capability": GENERAL_DENSE_COVERAGE_CAPABILITY,
                "receipt_sha256": document["sha256"],
                "schema": DENSE_VERIFY_SCHEMA,
            }
        finally:
            if receipts_descriptor is not None:
                os.close(receipts_descriptor)
            os.close(dense_descriptor)


def build_bundle(args: argparse.Namespace) -> dict[str, Any]:
    inventory, fingerprint, inventory_document = _load_inventory(
        Path(args.inventory).expanduser().resolve()
    )
    if fingerprint != args.layout_fingerprint:
        raise SparseBundleError(
            "inventory fingerprint does not match the pinned layout"
        )
    traces, leaves = _load_traces(
        (Path(path).expanduser().resolve() for path in args.access_trace),
        repo_id=args.repo_id,
        revision=args.revision,
        fingerprint=fingerprint,
    )
    cache_root = Path(args.cache_dir).expanduser().resolve()
    cache = VerifiedRangeCache(
        cache_root / "ranges", repo_id=args.repo_id, revision=args.revision
    )
    for shard, offset, length in leaves:
        cache.resolve(shard, offset, length)

    target = Path(args.output).expanduser().resolve()
    if target.exists() or target.is_symlink():
        raise SparseBundleError(f"output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.parent / f".{target.name}.pending-{uuid.uuid4().hex}"
    weights = pending / "weights"
    causal = pending / "causal"
    weights.mkdir(parents=True)
    causal.mkdir()
    try:
        config_sha = _copy_config(
            cache_root, weights, repo_id=args.repo_id, revision=args.revision
        )
        pinned_inventory_bytes = _canonical(inventory_document)
        _atomic_bytes(weights / "inventory.pinned.json", pinned_inventory_bytes)
        index_sha = _write_index(weights, inventory)
        shard_receipts, intervals = _write_sparse_weights(
            weights, inventory, leaves, cache
        )
        covered = _covered_experts(inventory, intervals)
        if not covered:
            raise SparseBundleError("trace materialized no complete routed expert")

        source = Streamer.from_local(
            weights,
            repo_id=args.repo_id,
            revision=args.revision,
            pinned_inventory=inventory,
            pinned_fingerprint=fingerprint,
            use_cache=False,
            budget_mb=args.budget_mb,
        )
        try:
            source.inventory()
            observed_fingerprint = source.metrics().get("inventory_source_fingerprint")
            if observed_fingerprint != fingerprint:
                raise SparseBundleError(
                    "reconstructed sparse layout fingerprint does not match source"
                )
            pager = DeepSeekWeightPager(
                source,
                device="cpu",
                compute_dtype="bfloat16",
                expert_prefetch=False,
            )
            plans = tuple(
                plan
                for layer, experts in covered.items()
                for plan in pager.plan_expert_ranges(layer, experts)
            )
        finally:
            source.close()

        model = LogicalModelIdentity(repo_id=args.repo_id, revision=args.revision)
        with CausalWeightMount(
            pending,
            model,
            budget_mb=args.budget_mb,
        ) as mount:
            receipt = mount.bind_plans(plans)
            if len(receipt.bindings) != len(plans):
                raise SparseBundleError("causal binding receipt lost expert plans")
            first = plans[0]
            if mount.resolve_expert_plans(first.layer, (first.expert_id,)) != (first,):
                raise SparseBundleError("causal reader did not replay the first plan")

        physical_bytes = sum(row["physical_bytes"] for row in shard_receipts)
        logical_bytes = sum(row["logical_bytes"] for row in shard_receipts)
        identity = {
            "causal_bindings": len(plans),
            "config_sha256": config_sha,
            "covered_experts": {
                str(layer): list(experts) for layer, experts in covered.items()
            },
            "index_sha256": index_sha,
            "pinned_inventory_sha256": _sha256_bytes(pinned_inventory_bytes),
            "layout_fingerprint": fingerprint,
            "logical_model": model.as_record(),
            "logical_shard_bytes": logical_bytes,
            "materialized_leaf_bytes": sum(length for _, _, length in leaves),
            "physical_shard_bytes": physical_bytes,
            "schema": BUNDLE_SCHEMA,
            "shards": shard_receipts,
            "trace_sha256": [trace.sha256 for trace in traces],
            "unique_leaves": len(leaves),
        }
        manifest = {**identity, "sha256": _sha256(identity)}
        _atomic_bytes(pending / "bundle.json", _canonical(manifest))
        os.replace(pending, target)
        return manifest
    except BaseException:
        if pending.exists():
            shutil.rmtree(pending)
        raise


def verify_bundle(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.bundle).expanduser().absolute()
    root_descriptor = _open_plain_root(root)
    try:
        document, promoted_manifest = _bundle_documents_at(root_descriptor)
        loaded, _manifest_identity = _load_bundle_manifest_at(
            root_descriptor,
            repo_id=args.repo_id,
            revision=args.revision,
            expected_layout=getattr(args, "layout_fingerprint", None),
        )
        if loaded != document:
            raise SparseBundleError("bundle manifest normalization changed identity")
    finally:
        os.close(root_descriptor)
    model = LogicalModelIdentity(repo_id=args.repo_id, revision=args.revision)
    with CausalWeightMount(root, model, budget_mb=args.budget_mb) as mount:
        if mount.layout.layout_fingerprint != document["layout_fingerprint"]:
            raise SparseBundleError(
                "mounted layout fingerprint does not match manifest"
            )
        covered = document.get("covered_experts")
        if not isinstance(covered, Mapping):
            raise SparseBundleError("bundle covered-expert table is invalid")
        resolved = sum(
            len(mount.resolve_expert_plans(int(layer), experts))
            for layer, experts in covered.items()
        )
        if resolved != int(document["causal_bindings"]):
            raise SparseBundleError("bundle causal binding count does not match")
        metrics = mount.reader.metrics()
    appended = _verify_append_store(
        root,
        model,
        repo_id=args.repo_id,
        revision=args.revision,
        expected_layout=getattr(args, "layout_fingerprint", None),
        budget_mb=float(args.budget_mb),
    )
    manifest_coordinates = {
        (int(layer), int(expert_id))
        for layer, expert_ids in document["covered_experts"].items()
        for expert_id in expert_ids
    }
    total_coordinates = manifest_coordinates | {
        tuple(coordinate) for coordinate in appended["coordinates"]
    }
    dense = _verify_dense_store(
        root,
        model,
        document,
        promoted_manifest,
        repo_id=args.repo_id,
        revision=args.revision,
        expected_layout=getattr(args, "layout_fingerprint", None),
        budget_mb=float(args.budget_mb),
    )
    return {
        "append_schema": APPEND_VERIFY_SCHEMA,
        "appended_bindings": appended["appended_bindings"],
        "append_physical_growth_bytes": appended["physical_growth_bytes"],
        "append_receipts": appended["receipts"],
        "causal_bindings": len(total_coordinates),
        "layout_fingerprint": document["layout_fingerprint"],
        "manifest_causal_bindings": resolved,
        "manifest_sha256": document["sha256"],
        "pending_append": appended["pending"],
        "dense_coverage": dense,
        "general_dense_weight_coverage": dense["capability"],
        "reader": metrics,
        "schema": f"{BUNDLE_SCHEMA}:verification/v1",
    }


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--inventory", required=True)
    build.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    build.add_argument("--access-trace", action="append", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--repo-id", default=OFFICIAL_SOURCE)
    build.add_argument("--revision", default=OFFICIAL_REVISION)
    build.add_argument("--layout-fingerprint", default=OFFICIAL_LAYOUT_FINGERPRINT)
    build.add_argument("--budget-mb", type=_positive_float, default=512.0)
    build.set_defaults(handler=build_bundle)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--bundle", required=True)
    verify.add_argument("--repo-id", default=OFFICIAL_SOURCE)
    verify.add_argument("--revision", default=OFFICIAL_REVISION)
    verify.add_argument(
        "--layout-fingerprint",
        default=OFFICIAL_LAYOUT_FINGERPRINT,
    )
    verify.add_argument("--budget-mb", type=_positive_float, default=512.0)
    verify.set_defaults(handler=verify_bundle)

    append = subparsers.add_parser(
        "append-experts",
        help="durably materialize and bind explicit routed experts",
    )
    append.add_argument("--bundle", required=True)
    append.add_argument(
        "--expert",
        action="append",
        required=True,
        type=_parse_expert_coordinate,
        metavar="LAYER:EXPERT",
    )
    append.add_argument("--repo-id", default=OFFICIAL_SOURCE)
    append.add_argument("--revision", default=OFFICIAL_REVISION)
    append.add_argument(
        "--layout-fingerprint",
        default=OFFICIAL_LAYOUT_FINGERPRINT,
    )
    append.add_argument("--budget-mb", type=_positive_float, default=512.0)
    append.add_argument(
        "--resident-limit-mb",
        type=_positive_float,
        default=64.0,
    )
    append.add_argument(
        "--staging-limit-mb",
        type=_positive_float,
        default=256.0,
    )
    append.add_argument("--remote-cache-dir")
    append.add_argument(
        "--remote-cache-limit-mb",
        type=_positive_float,
        default=256.0,
    )
    append.add_argument(
        "--inject-crash",
        choices=(
            "payload-before-binding",
            "binding-before-journal",
            "receipt-before-cleanup",
        ),
        help=argparse.SUPPRESS,
    )
    append.set_defaults(handler=append_experts)

    trace_append = subparsers.add_parser(
        "append-trace-experts",
        help=(
            "derive complete routed experts from sealed access traces and "
            "durably append only missing bindings"
        ),
    )
    trace_append.add_argument("--bundle", required=True)
    trace_append.add_argument(
        "--access-trace",
        action="append",
        required=True,
        help="sealed access trace; repeat to union coverage",
    )
    trace_append.add_argument("--repo-id", default=OFFICIAL_SOURCE)
    trace_append.add_argument("--revision", default=OFFICIAL_REVISION)
    trace_append.add_argument(
        "--layout-fingerprint",
        default=OFFICIAL_LAYOUT_FINGERPRINT,
    )
    trace_append.add_argument("--budget-mb", type=_positive_float, default=512.0)
    trace_append.add_argument(
        "--resident-limit-mb",
        type=_positive_float,
        default=64.0,
    )
    trace_append.add_argument(
        "--staging-limit-mb",
        type=_positive_float,
        default=256.0,
    )
    trace_append.add_argument("--remote-cache-dir")
    trace_append.add_argument(
        "--remote-cache-limit-mb",
        type=_positive_float,
        default=256.0,
    )
    trace_append.add_argument(
        "--plan-only",
        action="store_true",
        help="emit the sealed canonical append plan without mutating the bundle",
    )
    trace_append.set_defaults(handler=append_trace_experts)

    promote = subparsers.add_parser(
        "promote-dense",
        help="materialize and bind the complete main-decoder dense tensor plane",
    )
    promote.add_argument("--bundle", required=True)
    promote.add_argument(
        "--access-trace",
        action="append",
        help=(
            "base trace cited by bundle.json; when omitted, exact cited "
            "access-*.json siblings are discovered"
        ),
    )
    promote.add_argument("--repo-id", default=OFFICIAL_SOURCE)
    promote.add_argument("--revision", default=OFFICIAL_REVISION)
    promote.add_argument(
        "--layout-fingerprint",
        default=OFFICIAL_LAYOUT_FINGERPRINT,
    )
    promote.add_argument(
        "--budget-mb",
        type=_positive_float,
        default=12288.0,
        help=(
            "cumulative source-transfer hard budget; independent of the "
            "resident chunk window"
        ),
    )
    promote.add_argument(
        "--resident-limit-mb",
        type=_positive_float,
        default=64.0,
    )
    promote.add_argument(
        "--staging-limit-mb",
        type=_positive_float,
        default=3072.0,
    )
    promote.add_argument("--remote-cache-dir")
    promote.add_argument(
        "--remote-cache-limit-mb",
        type=_positive_float,
        default=4096.0,
    )
    promote.add_argument(
        "--plan-only",
        action="store_true",
        help="emit the sealed exact promotion plan without mutating the bundle",
    )
    promote.add_argument(
        "--inject-crash",
        choices=(
            "payload-before-binding",
            "binding-before-journal",
            "receipt-before-capability",
            "capability-before-cleanup",
        ),
        help=argparse.SUPPRESS,
    )
    promote.set_defaults(handler=promote_dense)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        document = args.handler(args)
    except SparseBundleError as exc:
        raise SystemExit(f"sparse causal bundle failed: {exc}") from exc
    print(json.dumps(document, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
