"""Durable semantic continuation anchors for the streamed Qwen3.8 runtime.

The cache is deliberately an index around :meth:`StreamedQwen38.save_state`
and :meth:`StreamedQwen38.load_state`.  It never serialises tensors itself and
never stores prompt text or token IDs.  A token-prefix digest selects one
authenticated native snapshot, while a sealed index owns deterministic LRU
accounting and crash recovery.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
from typing import Any, Literal, Protocol

from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors
import torch

from .mtp_carry_snapshot import (
    MtpCarrySidecarDescriptor,
    Qwen35MtpCarrySidecarError,
    read_qwen35_mtp_carry_sidecar,
    write_qwen35_mtp_carry_sidecar,
)
from .mtp_draft import Qwen35MtpCarry
from .snapshot import QWEN38_SNAPSHOT_SCHEMA, Qwen38SnapshotIdentityMismatch


TOKEN_PREFIX_SCHEMA = "immer.qwen3.8-token-prefix/v1"
SEMANTIC_ANCHOR_INDEX_SCHEMA = "immer.qwen3.8-semantic-anchor-index/v1"
_SEMANTIC_ANCHOR_RECEIPT_LEGACY_SCHEMA = "immer.qwen3.8-semantic-anchor/v1"
SEMANTIC_ANCHOR_RECEIPT_SCHEMA = "immer.qwen3.8-semantic-anchor/v2"
SEMANTIC_ANCHOR_GC_SCHEMA = "immer.qwen3.8-semantic-anchor-gc/v1"
SEMANTIC_ANCHOR_SEED_SCHEMA = "immer.qwen3.8-semantic-anchor-seed/v1"

BoundaryKind = Literal[
    "turn",
    "tool-call",
    "tool-output",
    "thinking",
    "custom",
]
BOUNDARY_KINDS = frozenset({"turn", "tool-call", "tool-output", "thinking", "custom"})

_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_MANIFEST_RE = re.compile(r"([0-9a-f]{64})\.json")
_PAYLOAD_RE = re.compile(r"([0-9a-f]{64})\.([0-9a-f]{64})\.npz")
_SEED_RE = re.compile(r"([0-9a-f]{64})\.([0-9a-f]{64})\.seed\.safetensors")
_MTP_CARRY_RE = re.compile(r"([0-9a-f]{64})\.qwen35-mtp-carry")
_MAX_INDEX_BYTES = 16 * 1024**2
_MAX_MTP_CARRY_SIDECAR_BYTES = 2 * 1024**3 + 1024**2
_MAX_TOKEN_COUNT = 1_048_576
_READ_CHUNK_BYTES = 1024 * 1024


class SemanticStateCacheError(RuntimeError):
    """The anchor cache is corrupt, unsafe, conflicting, or unavailable."""


class SemanticStateCacheConflict(SemanticStateCacheError):
    """A prefix already owns a committed or orphaned snapshot manifest."""


class SemanticStateCacheBudgetError(SemanticStateCacheError):
    """One anchor cannot fit inside the configured operational budget."""


class _StateModel(Protocol):
    def save_state(
        self,
        path: str | os.PathLike[str],
        *,
        max_bytes: int = ...,
        max_tensors: int = ...,
        transport_neutral: bool = ...,
    ) -> dict[str, Any]: ...

    def load_state(
        self,
        path: str | os.PathLike[str],
        *,
        max_bytes: int = ...,
        max_tensors: int = ...,
        max_restore_peak_bytes: int = ...,
        transport_neutral: bool = ...,
    ) -> dict[str, Any]: ...

    def reset_state(self, *, release: bool = ...) -> None: ...


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SemanticStateCacheError("anchor metadata is not canonical JSON") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_document(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value))


def _digest(value: Any, label: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise SemanticStateCacheError(f"{label} must be a lowercase SHA-256")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SemanticStateCacheError(f"{label} must be a non-negative integer")
    return value


def _positive_int(value: Any, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result == 0:
        raise SemanticStateCacheError(f"{label} must be positive")
    return result


def _token_tuple(token_ids: Sequence[int]) -> tuple[int, ...]:
    if isinstance(token_ids, (str, bytes, bytearray)) or not isinstance(
        token_ids, Sequence
    ):
        raise TypeError("token_ids must be a sequence of integers")
    if len(token_ids) > _MAX_TOKEN_COUNT:
        raise ValueError(f"token prefix exceeds {_MAX_TOKEN_COUNT} tokens")
    result: list[int] = []
    for value in token_ids:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value > 2**63 - 1
        ):
            raise ValueError("token IDs must be non-negative signed 64-bit integers")
        result.append(value)
    return tuple(result)


def token_prefix_sha256(token_ids: Sequence[int]) -> str:
    """Hash one canonical token-ID prefix without retaining its raw tokens."""

    tokens = _token_tuple(token_ids)
    return _sha256_document({"schema": TOKEN_PREFIX_SCHEMA, "token_ids": list(tokens)})


def semantic_label_sha256(label: str | bytes) -> str:
    """Return the optional opaque label digest stored by an anchor receipt."""

    if isinstance(label, str):
        raw = label.encode("utf-8")
    elif isinstance(label, bytes):
        raw = label
    else:
        raise TypeError("semantic label must be str or bytes")
    return _sha256_bytes(raw)


def _json_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SemanticStateCacheError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _stable_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _stable_regular_bytes(
    path: Path,
    *,
    label: str,
    max_bytes: int,
    expected_bytes: int | None = None,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SemanticStateCacheError(f"cannot open {label}") from exc
    try:
        before = os.fstat(descriptor)
        try:
            linked_before = os.lstat(path)
        except OSError as exc:
            raise SemanticStateCacheError(f"{label} path disappeared") from exc
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(linked_before.st_mode)
            or not _same_inode(before, linked_before)
        ):
            raise SemanticStateCacheError(f"{label} must be a stable regular file")
        if before.st_size < 0 or before.st_size > max_bytes:
            raise SemanticStateCacheError(f"{label} exceeds its byte limit")
        if expected_bytes is not None and before.st_size != expected_bytes:
            raise SemanticStateCacheError(f"{label} byte count mismatch")
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
            if not chunk:
                raise SemanticStateCacheError(f"{label} was truncated while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise SemanticStateCacheError(f"{label} grew while reading")
        after = os.fstat(descriptor)
        try:
            linked_after = os.lstat(path)
        except OSError as exc:
            raise SemanticStateCacheError(f"{label} path disappeared") from exc
        if (
            _stable_signature(before) != _stable_signature(after)
            or _stable_signature(linked_before) != _stable_signature(linked_after)
            or not _same_inode(after, linked_after)
        ):
            raise SemanticStateCacheError(f"{label} changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _stable_regular_sha256(
    path: Path,
    *,
    label: str,
    max_bytes: int,
    expected_bytes: int | None = None,
) -> str:
    """Hash one stable inode without retaining a potentially huge payload."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SemanticStateCacheError(f"cannot open {label}") from exc
    try:
        before = os.fstat(descriptor)
        try:
            linked_before = os.lstat(path)
        except OSError as exc:
            raise SemanticStateCacheError(f"{label} path disappeared") from exc
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(linked_before.st_mode)
            or not _same_inode(before, linked_before)
        ):
            raise SemanticStateCacheError(f"{label} must be a stable regular file")
        if before.st_size < 0 or before.st_size > max_bytes:
            raise SemanticStateCacheError(f"{label} exceeds its byte limit")
        if expected_bytes is not None and before.st_size != expected_bytes:
            raise SemanticStateCacheError(f"{label} byte count mismatch")
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
            if not chunk:
                raise SemanticStateCacheError(f"{label} was truncated while hashing")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise SemanticStateCacheError(f"{label} grew while hashing")
        after = os.fstat(descriptor)
        try:
            linked_after = os.lstat(path)
        except OSError as exc:
            raise SemanticStateCacheError(f"{label} path disappeared") from exc
        if (
            _stable_signature(before) != _stable_signature(after)
            or _stable_signature(linked_before) != _stable_signature(linked_after)
            or not _same_inode(after, linked_after)
        ):
            raise SemanticStateCacheError(f"{label} changed while hashing")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SemanticStateCacheError("cannot fsync anchor cache directory") from exc
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_directory(path: Path, label: str) -> os.stat_result:
    try:
        linked = os.lstat(path)
    except OSError as exc:
        raise SemanticStateCacheError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(linked.st_mode) or not stat.S_ISDIR(linked.st_mode):
        raise SemanticStateCacheError(f"{label} must be a non-symlink directory")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SemanticStateCacheError(f"cannot open {label}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode) or not _same_inode(opened, linked):
            raise SemanticStateCacheError(f"{label} changed during open")
        return opened
    finally:
        os.close(descriptor)


_THREAD_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, threading.RLock] = {}


def _thread_lock(path: Path) -> threading.RLock:
    key = os.path.abspath(os.fspath(path))
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _exclusive_lock(root: Path, lock_path: Path) -> Iterator[None]:
    with _thread_lock(lock_path):
        root_before = _validate_directory(root, "anchor cache root")
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            raise SemanticStateCacheError("anchor cache lock is unavailable") from exc
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            lock_before = os.fstat(descriptor)
            linked_before = os.lstat(lock_path)
            if (
                not stat.S_ISREG(lock_before.st_mode)
                or not stat.S_ISREG(linked_before.st_mode)
                or not _same_inode(lock_before, linked_before)
            ):
                raise SemanticStateCacheError("anchor cache lock is not stable")
            yield
            lock_after = os.fstat(descriptor)
            linked_after = os.lstat(lock_path)
            root_after = _validate_directory(root, "anchor cache root")
            if (
                _stable_signature(lock_before) != _stable_signature(lock_after)
                or not _same_inode(lock_after, linked_after)
                or not _same_inode(root_before, root_after)
            ):
                raise SemanticStateCacheError("anchor cache changed during transaction")
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


@dataclass(frozen=True, slots=True)
class AnchorReceipt:
    """Immutable authenticated metadata for one native Qwen state anchor."""

    prefix_length: int
    prefix_sha256: str
    boundary_kind: BoundaryKind
    semantic_label_sha256: str | None
    snapshot_manifest_name: str
    snapshot_manifest_sha256: str
    snapshot_manifest_bytes: int
    snapshot_body_sha256: str
    snapshot_body_bytes: int
    snapshot_payload_name: str
    snapshot_payload_sha256: str
    snapshot_payload_bytes: int
    seed_hidden_name: str | None
    seed_hidden_sha256: str | None
    seed_hidden_bytes: int
    seed_hidden_tensor_sha256: str | None
    seed_hidden_dtype: str | None
    seed_hidden_shape: tuple[int, ...] | None
    seed_hidden_source_device: str | None
    state_bytes: int
    created_sequence: int
    last_access_sequence: int
    hit_count: int
    transport_neutral: bool
    mtp_carry: MtpCarrySidecarDescriptor | None
    receipt_sha256: str

    def __post_init__(self) -> None:
        _nonnegative_int(self.prefix_length, "prefix length")
        _digest(self.prefix_sha256, "prefix SHA-256")
        if self.boundary_kind not in BOUNDARY_KINDS:
            raise SemanticStateCacheError("unsupported semantic boundary kind")
        _digest(
            self.semantic_label_sha256,
            "semantic label SHA-256",
            optional=True,
        )
        if self.snapshot_manifest_name != f"{self.prefix_sha256}.json":
            raise SemanticStateCacheError("anchor manifest name is not prefix-bound")
        _digest(self.snapshot_manifest_sha256, "snapshot manifest SHA-256")
        _positive_int(self.snapshot_manifest_bytes, "snapshot manifest bytes")
        _digest(self.snapshot_body_sha256, "snapshot body SHA-256")
        _positive_int(self.snapshot_body_bytes, "snapshot body bytes")
        _digest(self.snapshot_payload_sha256, "snapshot payload SHA-256")
        expected_payload = f"{self.prefix_sha256}.{self.snapshot_payload_sha256}.npz"
        if self.snapshot_payload_name != expected_payload:
            raise SemanticStateCacheError("anchor payload name is not content-bound")
        _positive_int(self.snapshot_payload_bytes, "snapshot payload bytes")
        seed_fields = (
            self.seed_hidden_name,
            self.seed_hidden_sha256,
            self.seed_hidden_tensor_sha256,
            self.seed_hidden_dtype,
            self.seed_hidden_shape,
            self.seed_hidden_source_device,
        )
        if self.seed_hidden_name is None:
            if any(value is not None for value in seed_fields[1:]) or (
                self.seed_hidden_bytes != 0
            ):
                raise SemanticStateCacheError("absent seed hidden retains metadata")
        else:
            seed_sha = _digest(self.seed_hidden_sha256, "seed hidden SHA-256")
            _digest(
                self.seed_hidden_tensor_sha256,
                "seed hidden tensor SHA-256",
            )
            if self.seed_hidden_name != (
                f"{self.prefix_sha256}.{seed_sha}.seed.safetensors"
            ):
                raise SemanticStateCacheError("seed hidden name is not content-bound")
            _positive_int(self.seed_hidden_bytes, "seed hidden bytes")
            if self.seed_hidden_dtype not in {"float16", "bfloat16", "float32"}:
                raise SemanticStateCacheError("seed hidden dtype is unsupported")
            if (
                not isinstance(self.seed_hidden_shape, tuple)
                or len(self.seed_hidden_shape) != 3
                or self.seed_hidden_shape[:2] != (1, 1)
                or self.seed_hidden_shape[2] <= 0
            ):
                raise SemanticStateCacheError("seed hidden shape is invalid")
            if (
                not isinstance(self.seed_hidden_source_device, str)
                or not self.seed_hidden_source_device
                or len(self.seed_hidden_source_device) > 64
            ):
                raise SemanticStateCacheError("seed hidden source device is invalid")
        _nonnegative_int(self.state_bytes, "snapshot state bytes")
        created = _positive_int(self.created_sequence, "created sequence")
        accessed = _positive_int(self.last_access_sequence, "last access sequence")
        if accessed < created:
            raise SemanticStateCacheError("anchor access sequence predates creation")
        _nonnegative_int(self.hit_count, "anchor hit count")
        if not isinstance(self.transport_neutral, bool):
            raise SemanticStateCacheError("transport_neutral must be boolean")
        if self.mtp_carry is not None and not isinstance(
            self.mtp_carry, MtpCarrySidecarDescriptor
        ):
            raise SemanticStateCacheError("MTP carry descriptor is invalid")
        _digest(self.receipt_sha256, "anchor receipt SHA-256")
        if self.receipt_sha256 != _sha256_document(self._body_document()):
            raise SemanticStateCacheError("anchor receipt SHA-256 mismatch")

    @property
    def cache_bytes(self) -> int:
        return (
            self.snapshot_manifest_bytes
            + self.snapshot_payload_bytes
            + self.seed_hidden_bytes
            + (0 if self.mtp_carry is None else self.mtp_carry.bytes)
        )

    def _body_document(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "boundary_kind": self.boundary_kind,
            "created_sequence": self.created_sequence,
            "hit_count": self.hit_count,
            "last_access_sequence": self.last_access_sequence,
            "prefix_length": self.prefix_length,
            "prefix_sha256": self.prefix_sha256,
            "schema": (
                _SEMANTIC_ANCHOR_RECEIPT_LEGACY_SCHEMA
                if self.mtp_carry is None
                else SEMANTIC_ANCHOR_RECEIPT_SCHEMA
            ),
            "seed_hidden_bytes": self.seed_hidden_bytes,
            "seed_hidden_dtype": self.seed_hidden_dtype,
            "seed_hidden_name": self.seed_hidden_name,
            "seed_hidden_sha256": self.seed_hidden_sha256,
            "seed_hidden_shape": (
                None if self.seed_hidden_shape is None else list(self.seed_hidden_shape)
            ),
            "seed_hidden_source_device": self.seed_hidden_source_device,
            "seed_hidden_tensor_sha256": self.seed_hidden_tensor_sha256,
            "semantic_label_sha256": self.semantic_label_sha256,
            "snapshot_body_bytes": self.snapshot_body_bytes,
            "snapshot_body_sha256": self.snapshot_body_sha256,
            "snapshot_manifest_bytes": self.snapshot_manifest_bytes,
            "snapshot_manifest_name": self.snapshot_manifest_name,
            "snapshot_manifest_sha256": self.snapshot_manifest_sha256,
            "snapshot_payload_bytes": self.snapshot_payload_bytes,
            "snapshot_payload_name": self.snapshot_payload_name,
            "snapshot_payload_sha256": self.snapshot_payload_sha256,
            "state_bytes": self.state_bytes,
            "transport_neutral": self.transport_neutral,
        }
        if self.mtp_carry is not None:
            body["mtp_carry"] = self.mtp_carry.to_record()
        return body

    def to_document(self) -> dict[str, Any]:
        return {**self._body_document(), "receipt_sha256": self.receipt_sha256}

    @classmethod
    def create(cls, **values: Any) -> "AnchorReceipt":
        values.setdefault("mtp_carry", None)
        if isinstance(values.get("seed_hidden_shape"), list):
            values["seed_hidden_shape"] = tuple(values["seed_hidden_shape"])
        mtp_carry = values.get("mtp_carry")
        if isinstance(mtp_carry, Mapping):
            try:
                mtp_carry = MtpCarrySidecarDescriptor.from_record(mtp_carry)
            except (TypeError, ValueError) as exc:
                raise SemanticStateCacheError("MTP carry descriptor is invalid") from exc
            values["mtp_carry"] = mtp_carry
        if mtp_carry is not None and not isinstance(
            mtp_carry, MtpCarrySidecarDescriptor
        ):
            raise SemanticStateCacheError("MTP carry descriptor is invalid")
        body_values = dict(values)
        body_values.pop("mtp_carry")
        body = {
            "schema": (
                _SEMANTIC_ANCHOR_RECEIPT_LEGACY_SCHEMA
                if mtp_carry is None
                else SEMANTIC_ANCHOR_RECEIPT_SCHEMA
            ),
            **body_values,
        }
        if mtp_carry is not None:
            body["mtp_carry"] = mtp_carry.to_record()
        return cls(**values, receipt_sha256=_sha256_document(body))

    @classmethod
    def from_document(cls, raw: Any) -> "AnchorReceipt":
        if not isinstance(raw, Mapping):
            raise SemanticStateCacheError("anchor receipt must be an object")
        expected = {
            "boundary_kind",
            "created_sequence",
            "hit_count",
            "last_access_sequence",
            "prefix_length",
            "prefix_sha256",
            "receipt_sha256",
            "schema",
            "seed_hidden_bytes",
            "seed_hidden_dtype",
            "seed_hidden_name",
            "seed_hidden_sha256",
            "seed_hidden_shape",
            "seed_hidden_source_device",
            "seed_hidden_tensor_sha256",
            "semantic_label_sha256",
            "snapshot_body_bytes",
            "snapshot_body_sha256",
            "snapshot_manifest_bytes",
            "snapshot_manifest_name",
            "snapshot_manifest_sha256",
            "snapshot_payload_bytes",
            "snapshot_payload_name",
            "snapshot_payload_sha256",
            "state_bytes",
            "transport_neutral",
        }
        schema = raw.get("schema")
        if schema == _SEMANTIC_ANCHOR_RECEIPT_LEGACY_SCHEMA:
            if set(raw) != expected:
                raise SemanticStateCacheError("anchor receipt schema is invalid")
            mtp_carry = None
        elif schema == SEMANTIC_ANCHOR_RECEIPT_SCHEMA:
            if set(raw) != expected | {"mtp_carry"}:
                raise SemanticStateCacheError("anchor receipt schema is invalid")
            try:
                mtp_carry = MtpCarrySidecarDescriptor.from_record(raw["mtp_carry"])
            except (TypeError, ValueError) as exc:
                raise SemanticStateCacheError(
                    "anchor MTP carry descriptor is invalid"
                ) from exc
        else:
            raise SemanticStateCacheError("anchor receipt schema is invalid")
        values = dict(raw)
        del values["schema"]
        values["mtp_carry"] = mtp_carry
        if isinstance(values.get("seed_hidden_shape"), list):
            values["seed_hidden_shape"] = tuple(values["seed_hidden_shape"])
        return cls(**values)

    def accessed(self, sequence: int) -> "AnchorReceipt":
        body = self._body_document()
        body["last_access_sequence"] = sequence
        body["hit_count"] = self.hit_count + 1
        body.pop("schema")
        return AnchorReceipt.create(**body)


@dataclass(frozen=True, slots=True)
class OrphanGcReceipt:
    """Deterministic report for explicit orphan collection."""

    deleted_manifest_names: tuple[str, ...]
    deleted_payload_names: tuple[str, ...]
    reclaimed_bytes: int
    index_generation: int
    receipt_sha256: str

    def __post_init__(self) -> None:
        if tuple(sorted(self.deleted_manifest_names)) != self.deleted_manifest_names:
            raise SemanticStateCacheError("GC manifest names are not canonical")
        if tuple(sorted(self.deleted_payload_names)) != self.deleted_payload_names:
            raise SemanticStateCacheError("GC payload names are not canonical")
        _nonnegative_int(self.reclaimed_bytes, "GC reclaimed bytes")
        _nonnegative_int(self.index_generation, "GC index generation")
        _digest(self.receipt_sha256, "GC receipt SHA-256")
        if self.receipt_sha256 != _sha256_document(self._body_document()):
            raise SemanticStateCacheError("GC receipt SHA-256 mismatch")

    def _body_document(self) -> dict[str, Any]:
        return {
            "deleted_manifest_names": list(self.deleted_manifest_names),
            "deleted_payload_names": list(self.deleted_payload_names),
            "index_generation": self.index_generation,
            "reclaimed_bytes": self.reclaimed_bytes,
            "schema": SEMANTIC_ANCHOR_GC_SCHEMA,
        }

    @classmethod
    def create(
        cls,
        *,
        deleted_manifest_names: Sequence[str],
        deleted_payload_names: Sequence[str],
        reclaimed_bytes: int,
        index_generation: int,
    ) -> "OrphanGcReceipt":
        manifests = tuple(sorted(deleted_manifest_names))
        payloads = tuple(sorted(deleted_payload_names))
        body = {
            "deleted_manifest_names": list(manifests),
            "deleted_payload_names": list(payloads),
            "index_generation": index_generation,
            "reclaimed_bytes": reclaimed_bytes,
            "schema": SEMANTIC_ANCHOR_GC_SCHEMA,
        }
        return cls(
            deleted_manifest_names=manifests,
            deleted_payload_names=payloads,
            reclaimed_bytes=reclaimed_bytes,
            index_generation=index_generation,
            receipt_sha256=_sha256_document(body),
        )


@dataclass(frozen=True, slots=True)
class RestoredAnchor:
    """One native restore plus authenticated seed and MTP carry sidecars."""

    anchor: AnchorReceipt
    query_length: int
    exact_prefix: bool
    seed_hidden: torch.Tensor | None
    mtp_carry: Qwen35MtpCarry | None = None
    mtp_carry_bytes: int = 0

    def __post_init__(self) -> None:
        _nonnegative_int(self.query_length, "anchor query length")
        if self.query_length < self.anchor.prefix_length:
            raise SemanticStateCacheError("restored anchor exceeds query length")
        if self.exact_prefix != (self.query_length == self.anchor.prefix_length):
            raise SemanticStateCacheError("exact-prefix flag is inconsistent")
        if self.seed_hidden is not None:
            if not self.exact_prefix:
                raise SemanticStateCacheError(
                    "suffix restore exposed stale seed hidden"
                )
            if tuple(self.seed_hidden.shape) != self.anchor.seed_hidden_shape:
                raise SemanticStateCacheError("restored seed hidden shape mismatch")
        _nonnegative_int(self.mtp_carry_bytes, "restored MTP carry bytes")
        descriptor = self.anchor.mtp_carry
        if self.mtp_carry is None:
            if descriptor is not None or self.mtp_carry_bytes != 0:
                raise SemanticStateCacheError(
                    "restored anchor omitted its announced MTP carry"
                )
        else:
            if not isinstance(self.mtp_carry, Qwen35MtpCarry):
                raise SemanticStateCacheError("restored MTP carry type is invalid")
            if descriptor is None or self.mtp_carry_bytes != descriptor.bytes:
                raise SemanticStateCacheError(
                    "restored MTP carry byte count is inconsistent"
                )
            if (
                len(self.mtp_carry.history) != self.anchor.prefix_length
                or self.mtp_carry.next_position != self.anchor.prefix_length - 1
            ):
                raise SemanticStateCacheError(
                    "restored MTP carry cursor differs from anchor"
                )

    @property
    def suffix_start(self) -> int:
        return self.anchor.prefix_length


@dataclass(frozen=True, slots=True)
class _SeedDescriptor:
    name: str | None = None
    sha256: str | None = None
    bytes: int = 0
    tensor_sha256: str | None = None
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    source_device: str | None = None


@dataclass(frozen=True, slots=True)
class _IndexState:
    anchors: tuple[AnchorReceipt, ...]
    generation: int
    logical_clock: int

    def by_prefix(self) -> dict[str, AnchorReceipt]:
        return {anchor.prefix_sha256: anchor for anchor in self.anchors}


class SemanticStateAnchorCache:
    """Content-verified, deterministic-LRU native Qwen continuation cache."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        max_bytes: int | None = None,
        snapshot_max_bytes: int = 2 * 1024**3,
        snapshot_max_tensors: int = 2048,
        max_restore_peak_bytes: int = 4 * 1024**3,
        transport_neutral: bool = False,
    ) -> None:
        if max_bytes is not None and (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes <= 0
        ):
            raise ValueError("max_bytes must be a positive integer or None")
        for name, value in (
            ("snapshot_max_bytes", snapshot_max_bytes),
            ("snapshot_max_tensors", snapshot_max_tensors),
            ("max_restore_peak_bytes", max_restore_peak_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(transport_neutral, bool):
            raise TypeError("transport_neutral must be boolean")
        self.root = Path(os.path.abspath(os.fspath(Path(root).expanduser())))
        try:
            linked = os.lstat(self.root)
        except FileNotFoundError:
            self.root.mkdir(parents=True, mode=0o700)
        except OSError as exc:
            raise SemanticStateCacheError("cannot inspect anchor cache root") from exc
        else:
            if stat.S_ISLNK(linked.st_mode) or not stat.S_ISDIR(linked.st_mode):
                raise SemanticStateCacheError(
                    "anchor cache root must be a non-symlink directory"
                )
        _validate_directory(self.root, "anchor cache root")
        self._root_identity = _validate_directory(self.root, "anchor cache root")
        self.snapshots = self.root / "snapshots"
        try:
            linked = os.lstat(self.snapshots)
        except FileNotFoundError:
            self.snapshots.mkdir(mode=0o700)
        except OSError as exc:
            raise SemanticStateCacheError("cannot inspect snapshot directory") from exc
        else:
            if stat.S_ISLNK(linked.st_mode) or not stat.S_ISDIR(linked.st_mode):
                raise SemanticStateCacheError(
                    "snapshot directory must be a non-symlink directory"
                )
        self._snapshots_identity = _validate_directory(
            self.snapshots, "anchor snapshot directory"
        )
        self.index_path = self.root / "index.json"
        self.lock_path = self.root / ".anchor-cache.lock"
        self.max_bytes = max_bytes
        self.snapshot_max_bytes = snapshot_max_bytes
        self.snapshot_max_tensors = snapshot_max_tensors
        self.max_restore_peak_bytes = max_restore_peak_bytes
        self.transport_neutral = transport_neutral
        with self._locked():
            if not self.index_path.exists():
                self._write_index(
                    _IndexState(anchors=(), generation=0, logical_clock=0)
                )
            else:
                self._read_index()

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with _exclusive_lock(self.root, self.lock_path):
            self._assert_cache_directories()
            yield
            self._assert_cache_directories()

    def _assert_cache_directories(self) -> None:
        current_root = _validate_directory(self.root, "anchor cache root")
        current_snapshots = _validate_directory(
            self.snapshots, "anchor snapshot directory"
        )
        if not _same_inode(current_root, self._root_identity) or not _same_inode(
            current_snapshots, self._snapshots_identity
        ):
            raise SemanticStateCacheError("anchor cache containment root changed")

    def _index_body(self, state: _IndexState) -> dict[str, Any]:
        return {
            "anchors": [anchor.to_document() for anchor in state.anchors],
            "generation": state.generation,
            "logical_clock": state.logical_clock,
            "schema": SEMANTIC_ANCHOR_INDEX_SCHEMA,
        }

    def _write_index(self, state: _IndexState) -> None:
        self._assert_cache_directories()
        prefixes = [anchor.prefix_sha256 for anchor in state.anchors]
        if len(prefixes) != len(set(prefixes)):
            raise SemanticStateCacheError("anchor index contains duplicate prefixes")
        canonical_anchors = tuple(
            sorted(state.anchors, key=lambda row: row.prefix_sha256)
        )
        canonical = replace(state, anchors=canonical_anchors)
        body = self._index_body(canonical)
        document = {"body": body, "body_sha256": _sha256_document(body)}
        raw = _canonical_json(document) + b"\n"
        if len(raw) > _MAX_INDEX_BYTES:
            raise SemanticStateCacheError("anchor index exceeds its byte limit")
        try:
            existing = os.lstat(self.index_path)
        except FileNotFoundError:
            existing = None
        except OSError as exc:
            raise SemanticStateCacheError("cannot inspect anchor index") from exc
        if existing is not None and (
            stat.S_ISLNK(existing.st_mode) or not stat.S_ISREG(existing.st_mode)
        ):
            raise SemanticStateCacheError("anchor index must be a regular file")
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.root,
            prefix=".index.json.",
            suffix=".pending",
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.index_path)
            _fsync_directory(self.root)
        finally:
            temporary.unlink(missing_ok=True)
        committed = _stable_regular_bytes(
            self.index_path,
            label="anchor index",
            max_bytes=_MAX_INDEX_BYTES,
            expected_bytes=len(raw),
        )
        if committed != raw:
            raise SemanticStateCacheError("anchor index commit verification failed")

    def _read_index(self) -> _IndexState:
        raw = _stable_regular_bytes(
            self.index_path,
            label="anchor index",
            max_bytes=_MAX_INDEX_BYTES,
        )
        if not raw.endswith(b"\n"):
            raise SemanticStateCacheError("anchor index is not canonical JSONL")
        try:
            document = json.loads(
                raw,
                object_pairs_hook=_json_no_duplicates,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    SemanticStateCacheError(f"invalid JSON constant: {value}")
                ),
            )
        except SemanticStateCacheError:
            raise
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise SemanticStateCacheError("anchor index is invalid JSON") from exc
        if not isinstance(document, Mapping) or set(document) != {
            "body",
            "body_sha256",
        }:
            raise SemanticStateCacheError("anchor index envelope is invalid")
        if raw != _canonical_json(document) + b"\n":
            raise SemanticStateCacheError("anchor index is not canonical")
        body = document.get("body")
        if not isinstance(body, Mapping) or set(body) != {
            "anchors",
            "generation",
            "logical_clock",
            "schema",
        }:
            raise SemanticStateCacheError("anchor index body is invalid")
        if body.get("schema") != SEMANTIC_ANCHOR_INDEX_SCHEMA:
            raise SemanticStateCacheError("anchor index schema mismatch")
        claimed = _digest(document.get("body_sha256"), "anchor index body SHA-256")
        if claimed != _sha256_document(body):
            raise SemanticStateCacheError("anchor index body SHA-256 mismatch")
        raw_anchors = body.get("anchors")
        if not isinstance(raw_anchors, list):
            raise SemanticStateCacheError("anchor index table is invalid")
        anchors = tuple(AnchorReceipt.from_document(row) for row in raw_anchors)
        if anchors != tuple(sorted(anchors, key=lambda row: row.prefix_sha256)):
            raise SemanticStateCacheError("anchor index order is not canonical")
        if len({anchor.prefix_sha256 for anchor in anchors}) != len(anchors):
            raise SemanticStateCacheError("anchor index contains duplicate prefixes")
        generation = _nonnegative_int(body.get("generation"), "index generation")
        logical_clock = _nonnegative_int(body.get("logical_clock"), "logical clock")
        if any(anchor.last_access_sequence > logical_clock for anchor in anchors):
            raise SemanticStateCacheError("anchor access sequence exceeds index clock")
        return _IndexState(
            anchors=anchors,
            generation=generation,
            logical_clock=logical_clock,
        )

    def _snapshot_paths(self, anchor: AnchorReceipt) -> tuple[Path, Path, Path | None]:
        return (
            self.snapshots / anchor.snapshot_manifest_name,
            self.snapshots / anchor.snapshot_payload_name,
            (
                None
                if anchor.seed_hidden_name is None
                else self.snapshots / anchor.seed_hidden_name
            ),
        )

    def _mtp_carry_path(self, anchor: AnchorReceipt) -> Path | None:
        descriptor = anchor.mtp_carry
        return None if descriptor is None else self.snapshots / descriptor.basename

    def _verify_anchor_artifacts(
        self, anchor: AnchorReceipt, *, verify_seed: bool
    ) -> None:
        manifest, payload, seed = self._snapshot_paths(anchor)
        manifest_raw = _stable_regular_bytes(
            manifest,
            label="anchor snapshot manifest",
            max_bytes=max(anchor.snapshot_manifest_bytes, 1),
            expected_bytes=anchor.snapshot_manifest_bytes,
        )
        if _sha256_bytes(manifest_raw) != anchor.snapshot_manifest_sha256:
            raise SemanticStateCacheError("anchor snapshot manifest SHA-256 mismatch")
        payload_sha = _stable_regular_sha256(
            payload,
            label="anchor snapshot payload",
            max_bytes=max(anchor.snapshot_payload_bytes, 1),
            expected_bytes=anchor.snapshot_payload_bytes,
        )
        if payload_sha != anchor.snapshot_payload_sha256:
            raise SemanticStateCacheError("anchor snapshot payload SHA-256 mismatch")
        if verify_seed and seed is not None:
            seed_raw = _stable_regular_bytes(
                seed,
                label="anchor seed hidden",
                max_bytes=max(anchor.seed_hidden_bytes, 1),
                expected_bytes=anchor.seed_hidden_bytes,
            )
            if _sha256_bytes(seed_raw) != anchor.seed_hidden_sha256:
                raise SemanticStateCacheError("anchor seed hidden SHA-256 mismatch")
        descriptor = anchor.mtp_carry
        if descriptor is not None:
            sidecar = self.snapshots / descriptor.basename
            sidecar_sha = _stable_regular_sha256(
                sidecar,
                label="anchor MTP carry sidecar",
                max_bytes=descriptor.bytes,
                expected_bytes=descriptor.bytes,
            )
            if sidecar_sha != descriptor.file_sha256:
                raise SemanticStateCacheError(
                    "anchor MTP carry sidecar SHA-256 mismatch"
                )

    @staticmethod
    def _seed_tensor_sha256(value: torch.Tensor) -> str:
        cpu = value.detach().to(device="cpu").contiguous()
        raw = cpu.view(torch.uint8).numpy().reshape(-1).tobytes()
        header = _canonical_json(
            {
                "dtype": str(cpu.dtype).removeprefix("torch."),
                "schema": SEMANTIC_ANCHOR_SEED_SCHEMA,
                "shape": list(cpu.shape),
            }
        )
        return hashlib.sha256(header + b"\0" + raw).hexdigest()

    def _write_seed_hidden(
        self,
        prefix_sha256: str,
        seed_hidden: torch.Tensor | None,
    ) -> _SeedDescriptor:
        if seed_hidden is None:
            return _SeedDescriptor()
        if not isinstance(seed_hidden, torch.Tensor):
            raise TypeError("seed_hidden must be a torch.Tensor or None")
        if seed_hidden.ndim != 3 or tuple(seed_hidden.shape[:2]) != (1, 1):
            raise ValueError("seed_hidden must have exact shape [1, 1, dim]")
        if seed_hidden.shape[2] <= 0:
            raise ValueError("seed_hidden width must be positive")
        dtype = str(seed_hidden.dtype).removeprefix("torch.")
        if dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("seed_hidden dtype must be float16, bfloat16, or float32")
        if not bool(torch.isfinite(seed_hidden).all().item()):
            raise ValueError("seed_hidden must contain only finite values")
        source_device = str(seed_hidden.device)
        cpu = seed_hidden.detach().to(device="cpu").contiguous()
        tensor_sha = self._seed_tensor_sha256(cpu)
        raw = save_safetensors(
            {"seed_hidden": cpu},
            metadata={
                "prefix_sha256": prefix_sha256,
                "schema": SEMANTIC_ANCHOR_SEED_SCHEMA,
                "tensor_sha256": tensor_sha,
            },
        )
        file_sha = _sha256_bytes(raw)
        name = f"{prefix_sha256}.{file_sha}.seed.safetensors"
        target = self.snapshots / name
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(target, flags, 0o600)
        except FileExistsError:
            existing = _stable_regular_bytes(
                target,
                label="existing seed hidden",
                max_bytes=len(raw),
                expected_bytes=len(raw),
            )
            if existing != raw:
                raise SemanticStateCacheConflict("seed hidden digest collision")
        except OSError as exc:
            raise SemanticStateCacheError("cannot publish seed hidden") from exc
        else:
            try:
                remaining = memoryview(raw)
                while remaining:
                    written = os.write(descriptor, remaining)
                    if written <= 0:
                        raise SemanticStateCacheError("seed hidden write was truncated")
                    remaining = remaining[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            _fsync_directory(self.snapshots)
        return _SeedDescriptor(
            name=name,
            sha256=file_sha,
            bytes=len(raw),
            tensor_sha256=tensor_sha,
            dtype=dtype,
            shape=tuple(int(value) for value in cpu.shape),
            source_device=source_device,
        )

    def _load_seed_hidden(
        self, anchor: AnchorReceipt, model: _StateModel
    ) -> torch.Tensor | None:
        if anchor.seed_hidden_name is None:
            return None
        _manifest, _payload, seed_path = self._snapshot_paths(anchor)
        assert seed_path is not None
        raw = _stable_regular_bytes(
            seed_path,
            label="anchor seed hidden",
            max_bytes=anchor.seed_hidden_bytes,
            expected_bytes=anchor.seed_hidden_bytes,
        )
        if _sha256_bytes(raw) != anchor.seed_hidden_sha256:
            raise SemanticStateCacheError("anchor seed hidden SHA-256 mismatch")
        try:
            tensors = load_safetensors(raw)
        except Exception as exc:
            raise SemanticStateCacheError("cannot decode anchor seed hidden") from exc
        if set(tensors) != {"seed_hidden"}:
            raise SemanticStateCacheError("seed hidden tensor table is invalid")
        hidden = tensors["seed_hidden"]
        if (
            tuple(hidden.shape) != anchor.seed_hidden_shape
            or str(hidden.dtype).removeprefix("torch.") != anchor.seed_hidden_dtype
            or not hidden.is_contiguous()
            or not bool(torch.isfinite(hidden).all().item())
            or self._seed_tensor_sha256(hidden) != anchor.seed_hidden_tensor_sha256
        ):
            raise SemanticStateCacheError("seed hidden tensor contract mismatch")
        pager = getattr(model, "pager", None)
        target_device = getattr(pager, "device", torch.device("cpu"))
        target_dtype = getattr(pager, "compute_dtype", hidden.dtype)
        if str(target_dtype).removeprefix("torch.") != anchor.seed_hidden_dtype:
            raise SemanticStateCacheError(
                "seed hidden dtype differs from restored model"
            )
        try:
            source_device = torch.device(anchor.seed_hidden_source_device)
            resolved_target = torch.device(target_device)
        except (RuntimeError, TypeError) as exc:
            raise SemanticStateCacheError(
                "seed hidden device contract is invalid"
            ) from exc
        if source_device.type != resolved_target.type or (
            source_device.index is not None
            and resolved_target.index is not None
            and source_device.index != resolved_target.index
        ):
            raise SemanticStateCacheError(
                "seed hidden device differs from restored model"
            )
        return hidden.to(device=target_device, dtype=target_dtype).detach().clone()

    @staticmethod
    def _mtp_carries_equal(
        left: Qwen35MtpCarry, right: Qwen35MtpCarry
    ) -> bool:
        if (
            left.schema != right.schema
            or left.identity != right.identity
            or left.history != right.history
            or left.next_position != right.next_position
            or not torch.equal(left.last_target_hidden, right.last_target_hidden)
        ):
            return False
        if left.state is None or right.state is None:
            return left.state is right.state
        return all(
            (left_value is None and right_value is None)
            or (
                isinstance(left_value, torch.Tensor)
                and isinstance(right_value, torch.Tensor)
                and torch.equal(left_value, right_value)
            )
            for left_value, right_value in (
                (left.state.key, right.state.key),
                (left.state.value, right.state.value),
                (left.state.crsa_log_usage, right.state.crsa_log_usage),
            )
        )

    def _write_mtp_carry(
        self,
        prefix_sha256: str,
        tokens: tuple[int, ...],
        mtp_carry: Qwen35MtpCarry | None,
        tokenizer_sha256: str | None,
    ) -> MtpCarrySidecarDescriptor | None:
        if mtp_carry is None:
            return None
        if not isinstance(mtp_carry, Qwen35MtpCarry):
            raise TypeError("mtp_carry must be a Qwen35MtpCarry or None")
        if mtp_carry.history != tokens:
            raise ValueError("MTP carry history must equal the anchor token prefix")
        tokenizer_digest = _digest(tokenizer_sha256, "tokenizer SHA-256")
        assert tokenizer_digest is not None
        try:
            descriptor = write_qwen35_mtp_carry_sidecar(
                self.snapshots,
                mtp_carry,
                tokenizer_sha256=tokenizer_digest,
                prefix_sha256=prefix_sha256,
            )
            verified = read_qwen35_mtp_carry_sidecar(
                self.snapshots,
                descriptor,
                history=tokens,
                tokenizer_sha256=tokenizer_digest,
                expected_identity=mtp_carry.identity,
            )
        except (OSError, TypeError, Qwen35MtpCarrySidecarError) as exc:
            raise SemanticStateCacheError(
                "cannot publish and verify MTP carry sidecar"
            ) from exc
        if not self._mtp_carries_equal(verified, mtp_carry):
            raise SemanticStateCacheError(
                "verified MTP carry sidecar differs from supplied carry"
            )
        return descriptor

    def _load_mtp_carry(
        self,
        anchor: AnchorReceipt,
        tokens: tuple[int, ...],
        *,
        tokenizer_sha256: str | None,
        expected_mtp_identity: object | None,
    ) -> Qwen35MtpCarry | None:
        descriptor = anchor.mtp_carry
        if descriptor is None:
            return None
        tokenizer_digest = _digest(tokenizer_sha256, "tokenizer SHA-256")
        if tokenizer_digest is None or expected_mtp_identity is None:
            raise SemanticStateCacheError(
                "MTP carry restore requires tokenizer and runtime identity"
            )
        try:
            carry = read_qwen35_mtp_carry_sidecar(
                self.snapshots,
                descriptor,
                history=tokens[: anchor.prefix_length],
                tokenizer_sha256=tokenizer_digest,
                expected_identity=expected_mtp_identity,
            )
        except (OSError, TypeError, Qwen35MtpCarrySidecarError) as exc:
            raise SemanticStateCacheError(
                "cannot authenticate anchor MTP carry sidecar"
            ) from exc
        if (
            carry.history != tokens[: anchor.prefix_length]
            or carry.next_position != anchor.prefix_length - 1
        ):
            raise SemanticStateCacheError("anchor MTP carry cursor mismatch")
        return carry

    def _anchor_from_snapshot(
        self,
        *,
        prefix_length: int,
        prefix_sha256: str,
        boundary_kind: BoundaryKind,
        semantic_digest: str | None,
        save_receipt: Mapping[str, Any],
        seed: _SeedDescriptor,
        mtp_carry: MtpCarrySidecarDescriptor | None,
        sequence: int,
    ) -> AnchorReceipt:
        manifest_path = self.snapshots / f"{prefix_sha256}.json"
        manifest_raw = _stable_regular_bytes(
            manifest_path,
            label="new anchor snapshot manifest",
            max_bytes=8 * 1024**2,
        )
        try:
            document = json.loads(
                manifest_raw,
                object_pairs_hook=_json_no_duplicates,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    SemanticStateCacheError(f"invalid JSON constant: {value}")
                ),
            )
        except SemanticStateCacheError:
            raise
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise SemanticStateCacheError(
                "new snapshot manifest is invalid JSON"
            ) from exc
        if (
            not isinstance(document, Mapping)
            or manifest_raw != _canonical_json(document) + b"\n"
        ):
            raise SemanticStateCacheError("new snapshot manifest is not canonical")
        if document.get("schema") != QWEN38_SNAPSHOT_SCHEMA:
            raise SemanticStateCacheError("new snapshot schema mismatch")
        body = document.get("body")
        if not isinstance(body, Mapping):
            raise SemanticStateCacheError("new snapshot body is invalid")
        body_raw = _canonical_json(body)
        body_sha = _sha256_bytes(body_raw)
        if document.get("body_sha256") != body_sha:
            raise SemanticStateCacheError("new snapshot body SHA-256 mismatch")
        state = body.get("state")
        if not isinstance(state, Mapping) or state.get("state_batch_size") != 1:
            raise SemanticStateCacheError("semantic anchors require batch size one")
        if state.get("next_position") != prefix_length:
            raise SemanticStateCacheError("snapshot cursor differs from token prefix")
        if seed.shape is not None:
            identity = body.get("identity")
            config = identity.get("config") if isinstance(identity, Mapping) else None
            execution = (
                identity.get("execution") if isinstance(identity, Mapping) else None
            )
            if (
                not isinstance(config, Mapping)
                or seed.shape[2] != config.get("dim")
                or not isinstance(execution, Mapping)
                or seed.dtype != execution.get("compute_dtype")
            ):
                raise SemanticStateCacheError(
                    "seed hidden differs from native snapshot identity"
                )
        payload = body.get("payload")
        if not isinstance(payload, Mapping):
            raise SemanticStateCacheError("new snapshot payload descriptor is invalid")
        payload_name = payload.get("file")
        payload_sha = _digest(payload.get("sha256"), "snapshot payload SHA-256")
        payload_bytes = _positive_int(payload.get("bytes"), "snapshot payload bytes")
        expected_name = f"{prefix_sha256}.{payload_sha}.npz"
        if payload_name != expected_name:
            raise SemanticStateCacheError("new snapshot payload is not prefix-bound")
        payload_path = self.snapshots / expected_name
        payload_raw = _stable_regular_bytes(
            payload_path,
            label="new anchor snapshot payload",
            max_bytes=payload_bytes,
            expected_bytes=payload_bytes,
        )
        if _sha256_bytes(payload_raw) != payload_sha:
            raise SemanticStateCacheError("new snapshot payload SHA-256 mismatch")
        expected_receipt = {
            "manifest_body_sha256": body_sha,
            "payload_bytes": payload_bytes,
            "payload_sha256": payload_sha,
            "tensor_bytes": body.get("tensor_bytes"),
            "next_position": prefix_length,
            "transport_neutral": self.transport_neutral,
        }
        if any(
            save_receipt.get(key) != value for key, value in expected_receipt.items()
        ):
            raise SemanticStateCacheError("native save_state receipt is inconsistent")
        return AnchorReceipt.create(
            prefix_length=prefix_length,
            prefix_sha256=prefix_sha256,
            boundary_kind=boundary_kind,
            semantic_label_sha256=semantic_digest,
            snapshot_manifest_name=manifest_path.name,
            snapshot_manifest_sha256=_sha256_bytes(manifest_raw),
            snapshot_manifest_bytes=len(manifest_raw),
            snapshot_body_sha256=body_sha,
            snapshot_body_bytes=len(body_raw),
            snapshot_payload_name=expected_name,
            snapshot_payload_sha256=payload_sha,
            snapshot_payload_bytes=payload_bytes,
            seed_hidden_name=seed.name,
            seed_hidden_sha256=seed.sha256,
            seed_hidden_bytes=seed.bytes,
            seed_hidden_tensor_sha256=seed.tensor_sha256,
            seed_hidden_dtype=seed.dtype,
            seed_hidden_shape=seed.shape,
            seed_hidden_source_device=seed.source_device,
            state_bytes=_nonnegative_int(
                body.get("tensor_bytes"), "snapshot state bytes"
            ),
            created_sequence=sequence,
            last_access_sequence=sequence,
            hit_count=0,
            transport_neutral=self.transport_neutral,
            mtp_carry=mtp_carry,
        )

    def store(
        self,
        model: _StateModel,
        token_ids: Sequence[int],
        *,
        boundary_kind: BoundaryKind,
        semantic_label_sha256: str | None = None,
        seed_hidden: torch.Tensor | None = None,
        mtp_carry: Qwen35MtpCarry | None = None,
        tokenizer_sha256: str | None = None,
    ) -> AnchorReceipt:
        """Commit one native snapshot, then atomically make its prefix visible."""

        tokens = _token_tuple(token_ids)
        if not tokens:
            raise ValueError("semantic anchors require a non-empty token prefix")
        if boundary_kind not in BOUNDARY_KINDS:
            raise ValueError("unsupported semantic boundary kind")
        semantic_digest = _digest(
            semantic_label_sha256,
            "semantic label SHA-256",
            optional=True,
        )
        if mtp_carry is not None:
            if not isinstance(mtp_carry, Qwen35MtpCarry):
                raise TypeError("mtp_carry must be a Qwen35MtpCarry or None")
            if mtp_carry.history != tokens:
                raise ValueError(
                    "MTP carry history must equal the anchor token prefix"
                )
            _digest(tokenizer_sha256, "tokenizer SHA-256")
        prefix_digest = token_prefix_sha256(tokens)
        manifest_path = self.snapshots / f"{prefix_digest}.json"
        with self._locked():
            state = self._read_index()
            if prefix_digest in state.by_prefix():
                raise SemanticStateCacheConflict(
                    "token prefix already owns a committed anchor"
                )
            try:
                os.lstat(manifest_path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise SemanticStateCacheError(
                    "cannot inspect prospective anchor manifest"
                ) from exc
            else:
                raise SemanticStateCacheConflict(
                    "token prefix owns an uncommitted snapshot; run gc_orphans"
                )
            save_receipt = model.save_state(
                manifest_path,
                max_bytes=self.snapshot_max_bytes,
                max_tensors=self.snapshot_max_tensors,
                transport_neutral=self.transport_neutral,
            )
            if not isinstance(save_receipt, Mapping):
                raise SemanticStateCacheError("native save_state returned no receipt")
            seed = self._write_seed_hidden(prefix_digest, seed_hidden)
            carry_descriptor = self._write_mtp_carry(
                prefix_digest,
                tokens,
                mtp_carry,
                tokenizer_sha256,
            )
            sequence = state.logical_clock + 1
            anchor = self._anchor_from_snapshot(
                prefix_length=len(tokens),
                prefix_sha256=prefix_digest,
                boundary_kind=boundary_kind,
                semantic_digest=semantic_digest,
                save_receipt=save_receipt,
                seed=seed,
                mtp_carry=carry_descriptor,
                sequence=sequence,
            )
            if self.max_bytes is not None and anchor.cache_bytes > self.max_bytes:
                self._delete_anchor_artifacts(anchor)
                raise SemanticStateCacheBudgetError(
                    "native snapshot exceeds the complete anchor-cache budget"
                )
            inserted = _IndexState(
                anchors=(*state.anchors, anchor),
                generation=state.generation + 1,
                logical_clock=sequence,
            )
            # The snapshot already exists durably.  This index replacement is
            # the only operation that makes the anchor visible.
            self._write_index(inserted)
            self._evict_after_commit(inserted, protected_prefix=prefix_digest)
            return anchor

    def _select_deepest(
        self,
        state: _IndexState,
        tokens: tuple[int, ...],
    ) -> AnchorReceipt | None:
        by_length: dict[int, list[AnchorReceipt]] = {}
        for anchor in state.anchors:
            if anchor.prefix_length <= len(tokens):
                by_length.setdefault(anchor.prefix_length, []).append(anchor)
        for length in sorted(by_length, reverse=True):
            candidate_digest = token_prefix_sha256(tokens[:length])
            matches = [
                row
                for row in by_length[length]
                if row.prefix_sha256 == candidate_digest
            ]
            if len(matches) > 1:
                raise SemanticStateCacheError("anchor prefix conflict in sealed index")
            if matches:
                self._verify_anchor_artifacts(
                    matches[0], verify_seed=len(tokens) == length
                )
                return matches[0]
        return None

    def lookup_deepest(self, token_ids: Sequence[int]) -> AnchorReceipt | None:
        """Find the deepest exact token prefix and durably record one cache hit."""

        tokens = _token_tuple(token_ids)
        with self._locked():
            state = self._read_index()
            anchor = self._select_deepest(state, tokens)
            if anchor is None:
                return None
            return self._commit_hit(state, anchor)

    def restore_deepest(
        self,
        model: _StateModel,
        token_ids: Sequence[int],
        *,
        tokenizer_sha256: str | None = None,
        expected_mtp_identity: object | None = None,
    ) -> RestoredAnchor | None:
        """Restore the deepest exact prefix through native ``load_state`` only."""

        tokens = _token_tuple(token_ids)
        with self._locked():
            state = self._read_index()
            anchor = self._select_deepest(state, tokens)
            if anchor is None:
                return None
            manifest, payload, _seed = self._snapshot_paths(anchor)
            exact = len(tokens) == anchor.prefix_length
            # Exact-query seed authentication must finish before load_state can
            # replace the caller's continuation state.  Suffix queries never
            # consume this auxiliary head seed and therefore never read it.
            seed_hidden = self._load_seed_hidden(anchor, model) if exact else None
            mtp_carry = self._load_mtp_carry(
                anchor,
                tokens,
                tokenizer_sha256=tokenizer_sha256,
                expected_mtp_identity=expected_mtp_identity,
            )
            loaded_completed = False
            try:
                loaded = model.load_state(
                    manifest,
                    max_bytes=self.snapshot_max_bytes,
                    max_tensors=self.snapshot_max_tensors,
                    max_restore_peak_bytes=self.max_restore_peak_bytes,
                    transport_neutral=anchor.transport_neutral,
                )
                loaded_completed = True
                expected = {
                    "manifest_body_sha256": anchor.snapshot_body_sha256,
                    "next_position": anchor.prefix_length,
                    "payload_bytes": anchor.snapshot_payload_bytes,
                    "payload_sha256": anchor.snapshot_payload_sha256,
                    "tensor_bytes": anchor.state_bytes,
                    "transport_neutral": anchor.transport_neutral,
                }
                if not isinstance(loaded, Mapping) or any(
                    loaded.get(key) != value for key, value in expected.items()
                ):
                    raise SemanticStateCacheError(
                        "native load_state receipt differs from anchor receipt"
                    )
                try:
                    loaded_manifest = Path(
                        os.path.abspath(os.fspath(loaded.get("manifest")))
                    )
                    loaded_payload = Path(
                        os.path.abspath(os.fspath(loaded.get("payload")))
                    )
                except (TypeError, ValueError) as exc:
                    raise SemanticStateCacheError(
                        "native load_state artifact paths are invalid"
                    ) from exc
                if loaded_manifest != manifest or loaded_payload != payload:
                    raise SemanticStateCacheError(
                        "native load_state artifact paths differ from anchor"
                    )
                updated = self._commit_hit(state, anchor)
                return RestoredAnchor(
                    anchor=updated,
                    query_length=len(tokens),
                    exact_prefix=exact,
                    seed_hidden=seed_hidden,
                    mtp_carry=mtp_carry,
                    mtp_carry_bytes=(
                        0 if anchor.mtp_carry is None else anchor.mtp_carry.bytes
                    ),
                )
            except Qwen38SnapshotIdentityMismatch:
                current = state.by_prefix().get(anchor.prefix_sha256)
                if current != anchor:
                    raise SemanticStateCacheError(
                        "anchor changed before incompatible eviction"
                    )
                retained = tuple(
                    row
                    for row in state.anchors
                    if row.prefix_sha256 != anchor.prefix_sha256
                )
                self._write_index(
                    _IndexState(
                        anchors=retained,
                        generation=state.generation + 1,
                        logical_clock=state.logical_clock,
                    )
                )
                self._delete_anchor_artifacts(anchor)
                return None
            except Exception:
                if loaded_completed:
                    try:
                        model.reset_state(release=True)
                    except Exception as reset_error:
                        raise SemanticStateCacheError(
                            "failed restore could not reset model state"
                        ) from reset_error
                raise

    def _commit_hit(self, state: _IndexState, anchor: AnchorReceipt) -> AnchorReceipt:
        current = state.by_prefix().get(anchor.prefix_sha256)
        if current != anchor:
            raise SemanticStateCacheError("anchor changed before hit commit")
        sequence = state.logical_clock + 1
        updated = anchor.accessed(sequence)
        anchors = tuple(
            updated if row.prefix_sha256 == anchor.prefix_sha256 else row
            for row in state.anchors
        )
        self._write_index(
            _IndexState(
                anchors=anchors,
                generation=state.generation + 1,
                logical_clock=sequence,
            )
        )
        return updated

    @staticmethod
    def _physical_cache_bytes(anchors: Sequence[AnchorReceipt]) -> int:
        """Count anchor-owned files, deduplicating shared content sidecars."""

        total = 0
        sidecar_bytes: dict[tuple[str, str], int] = {}
        for anchor in anchors:
            total += (
                anchor.snapshot_manifest_bytes
                + anchor.snapshot_payload_bytes
                + anchor.seed_hidden_bytes
            )
            descriptor = anchor.mtp_carry
            if descriptor is None:
                continue
            key = (descriptor.file_sha256, descriptor.basename)
            previous = sidecar_bytes.get(key)
            if previous is None:
                sidecar_bytes[key] = descriptor.bytes
                total += descriptor.bytes
            elif previous != descriptor.bytes:
                raise SemanticStateCacheError(
                    "shared MTP carry sidecar byte counts conflict"
                )
        return total

    def _evict_after_commit(self, state: _IndexState, *, protected_prefix: str) -> None:
        if self.max_bytes is None:
            return
        retained = state.anchors
        total = self._physical_cache_bytes(retained)
        if total <= self.max_bytes:
            return
        ordered = sorted(
            state.anchors,
            key=lambda row: (
                row.last_access_sequence,
                row.created_sequence,
                row.prefix_sha256,
            ),
        )
        evicted: list[AnchorReceipt] = []
        for anchor in ordered:
            if total <= self.max_bytes:
                break
            if anchor.prefix_sha256 == protected_prefix:
                continue
            evicted.append(anchor)
            evicted_prefixes = {row.prefix_sha256 for row in evicted}
            retained = tuple(
                row
                for row in state.anchors
                if row.prefix_sha256 not in evicted_prefixes
            )
            # A shared content-addressed carry remains charged until its final
            # live receipt is removed, while anchor-exclusive files free now.
            total = self._physical_cache_bytes(retained)
        if total > self.max_bytes:
            raise SemanticStateCacheBudgetError(
                "anchor budget cannot be met without evicting the committed anchor"
            )
        # Remove references first; a crash after this commit leaves only safe
        # orphans for explicit collection.
        self._write_index(
            _IndexState(
                anchors=retained,
                generation=state.generation + 1,
                logical_clock=state.logical_clock,
            )
        )
        for anchor in evicted:
            self._delete_anchor_artifacts(anchor)

    def _delete_verified_file(
        self, path: Path, *, expected_sha256: str, expected_bytes: int, label: str
    ) -> int:
        actual_sha = _stable_regular_sha256(
            path,
            label=label,
            max_bytes=expected_bytes,
            expected_bytes=expected_bytes,
        )
        if actual_sha != expected_sha256:
            raise SemanticStateCacheError(f"refusing to delete changed {label}")
        linked = os.lstat(path)
        if not stat.S_ISREG(linked.st_mode) or linked.st_size != expected_bytes:
            raise SemanticStateCacheError(f"refusing to delete unstable {label}")
        path.unlink()
        return expected_bytes

    def _delete_anchor_artifacts(self, anchor: AnchorReceipt) -> int:
        manifest, payload, seed = self._snapshot_paths(anchor)
        carry_path = self._mtp_carry_path(anchor)
        reclaimed = 0
        # Manifest first prevents the native snapshot reader from discovering a
        # payload while an unreferenced cache entry is being removed.
        if manifest.exists():
            reclaimed += self._delete_verified_file(
                manifest,
                expected_sha256=anchor.snapshot_manifest_sha256,
                expected_bytes=anchor.snapshot_manifest_bytes,
                label="anchor manifest",
            )
        if payload.exists():
            reclaimed += self._delete_verified_file(
                payload,
                expected_sha256=anchor.snapshot_payload_sha256,
                expected_bytes=anchor.snapshot_payload_bytes,
                label="anchor payload",
            )
        if seed is not None and seed.exists():
            assert anchor.seed_hidden_sha256 is not None
            reclaimed += self._delete_verified_file(
                seed,
                expected_sha256=anchor.seed_hidden_sha256,
                expected_bytes=anchor.seed_hidden_bytes,
                label="anchor seed hidden",
            )
        if carry_path is not None and carry_path.exists():
            descriptor = anchor.mtp_carry
            assert descriptor is not None
            referenced_carries = {
                row.mtp_carry.basename
                for row in self._read_index().anchors
                if row.mtp_carry is not None
            }
            if descriptor.basename not in referenced_carries:
                reclaimed += self._delete_verified_file(
                    carry_path,
                    expected_sha256=descriptor.file_sha256,
                    expected_bytes=descriptor.bytes,
                    label="anchor MTP carry sidecar",
                )
        _fsync_directory(self.snapshots)
        return reclaimed

    def receipts(self) -> tuple[AnchorReceipt, ...]:
        """Return the sealed index table without changing LRU recency."""

        with self._locked():
            return self._read_index().anchors

    @property
    def total_bytes(self) -> int:
        with self._locked():
            return self._physical_cache_bytes(self._read_index().anchors)

    def gc_orphans(self) -> OrphanGcReceipt:
        """Explicitly remove verified cache-owned files absent from the index."""

        with self._locked():
            state = self._read_index()
            referenced = {
                name
                for anchor in state.anchors
                for name in (
                    anchor.snapshot_manifest_name,
                    anchor.snapshot_payload_name,
                    anchor.seed_hidden_name,
                    (
                        None
                        if anchor.mtp_carry is None
                        else anchor.mtp_carry.basename
                    ),
                )
                if name is not None
            }
            deleted_manifests: list[str] = []
            deleted_payloads: list[str] = []
            reclaimed = 0
            entries = sorted(self.snapshots.iterdir(), key=lambda path: path.name)
            for path in entries:
                name = path.name
                if name in referenced:
                    continue
                try:
                    linked = os.lstat(path)
                except OSError as exc:
                    raise SemanticStateCacheError("cannot inspect orphan") from exc
                if not stat.S_ISREG(linked.st_mode):
                    continue
                manifest_match = _MANIFEST_RE.fullmatch(name)
                payload_match = _PAYLOAD_RE.fullmatch(name)
                seed_match = _SEED_RE.fullmatch(name)
                carry_match = _MTP_CARRY_RE.fullmatch(name)
                if manifest_match is not None:
                    raw = _stable_regular_bytes(
                        path,
                        label="orphan manifest",
                        max_bytes=8 * 1024**2,
                    )
                    try:
                        document = json.loads(raw)
                    except (UnicodeError, json.JSONDecodeError):
                        continue
                    body = (
                        document.get("body") if isinstance(document, Mapping) else None
                    )
                    body_sha = (
                        document.get("body_sha256")
                        if isinstance(document, Mapping)
                        else None
                    )
                    payload_descriptor = (
                        body.get("payload") if isinstance(body, Mapping) else None
                    )
                    expected_prefix = manifest_match.group(1)
                    if (
                        not isinstance(document, Mapping)
                        or document.get("schema") != QWEN38_SNAPSHOT_SCHEMA
                        or raw != _canonical_json(document) + b"\n"
                        or not isinstance(body, Mapping)
                        or body_sha != _sha256_document(body)
                        or not isinstance(payload_descriptor, Mapping)
                        or _PAYLOAD_RE.fullmatch(
                            str(payload_descriptor.get("file", ""))
                        )
                        is None
                        or not str(payload_descriptor.get("file", "")).startswith(
                            f"{expected_prefix}."
                        )
                    ):
                        continue
                    size = len(raw)
                    path.unlink()
                    reclaimed += size
                    deleted_manifests.append(name)
                elif payload_match is not None:
                    expected_sha = payload_match.group(2)
                    actual_sha = _stable_regular_sha256(
                        path,
                        label="orphan payload",
                        max_bytes=self.snapshot_max_bytes + 8 * 1024**2,
                    )
                    if actual_sha != expected_sha:
                        continue
                    size = linked.st_size
                    path.unlink()
                    reclaimed += size
                    deleted_payloads.append(name)
                elif seed_match is not None:
                    expected_sha = seed_match.group(2)
                    raw = _stable_regular_bytes(
                        path,
                        label="orphan seed hidden",
                        max_bytes=8 * 1024**2,
                    )
                    if _sha256_bytes(raw) != expected_sha:
                        continue
                    try:
                        tensors = load_safetensors(raw)
                    except Exception:
                        continue
                    if set(tensors) != {"seed_hidden"}:
                        continue
                    size = len(raw)
                    path.unlink()
                    reclaimed += size
                    deleted_payloads.append(name)
                elif carry_match is not None:
                    expected_sha = carry_match.group(1)
                    actual_sha = _stable_regular_sha256(
                        path,
                        label="orphan MTP carry sidecar",
                        max_bytes=max(
                            self.snapshot_max_bytes + 8 * 1024**2,
                            _MAX_MTP_CARRY_SIDECAR_BYTES,
                        ),
                    )
                    if actual_sha != expected_sha:
                        continue
                    size = linked.st_size
                    path.unlink()
                    reclaimed += size
                    deleted_payloads.append(name)
            if deleted_manifests or deleted_payloads:
                _fsync_directory(self.snapshots)
            return OrphanGcReceipt.create(
                deleted_manifest_names=deleted_manifests,
                deleted_payload_names=deleted_payloads,
                reclaimed_bytes=reclaimed,
                index_generation=state.generation,
            )


__all__ = [
    "AnchorReceipt",
    "BOUNDARY_KINDS",
    "BoundaryKind",
    "OrphanGcReceipt",
    "RestoredAnchor",
    "SEMANTIC_ANCHOR_GC_SCHEMA",
    "SEMANTIC_ANCHOR_INDEX_SCHEMA",
    "SEMANTIC_ANCHOR_RECEIPT_SCHEMA",
    "SemanticStateAnchorCache",
    "SemanticStateCacheBudgetError",
    "SemanticStateCacheConflict",
    "SemanticStateCacheError",
    "TOKEN_PREFIX_SCHEMA",
    "semantic_label_sha256",
    "token_prefix_sha256",
]
