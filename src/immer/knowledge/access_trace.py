"""Canonical access traces and windowed exact-range cache replay.

The trace is a control-plane sidecar.  It records which immutable source
ranges were used, but never claims that replaying those ranges reproduces a
model computation. Replay only verifies or warms the existing Streamer cache.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Protocol, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    from .streamer import Streamer


TRACE_SCHEMA = "immer.access-trace/v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PINNED_REVISION_RE = re.compile(r"[0-9a-fA-F]{40,64}")
_TAG_KEY_RE = re.compile(r"[A-Za-z0-9_.:-]{1,64}")
_MAX_TAGS = 32
_MAX_TAG_STRING_BYTES = 512
_MAX_TAG_DOCUMENT_BYTES = 4096


class AccessTraceError(ValueError):
    """Base error for malformed or incompatible access traces."""


class AccessTraceIntegrityError(AccessTraceError):
    """A trace is non-canonical, malformed, or fails its SHA-256 identity."""


class AccessTraceIdentityError(AccessTraceError):
    """A valid trace belongs to a different immutable tensor source."""


def _canonical_json(document: Any) -> bytes:
    return json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(document: Any) -> str:
    return hashlib.sha256(_canonical_json(document)).hexdigest()


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AccessTraceIntegrityError(f"{label} must be a non-negative integer")
    return value


def _positive_int(value: Any, label: str) -> int:
    value = _nonnegative_int(value, label)
    if value == 0:
        raise AccessTraceIntegrityError(f"{label} must be positive")
    return value


def _validate_tag_value(value: Any, key: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        if (
            isinstance(value, str)
            and len(value.encode("utf-8")) > _MAX_TAG_STRING_BYTES
        ):
            raise AccessTraceIntegrityError(
                f"access tag {key!r} exceeds {_MAX_TAG_STRING_BYTES} UTF-8 bytes"
            )
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        if not -(2**63) <= value < 2**63:
            raise AccessTraceIntegrityError(
                f"access tag {key!r} is outside signed 64-bit range"
            )
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise AccessTraceIntegrityError(f"access tag {key!r} must be a bounded JSON scalar")


def canonical_tags(tags: Mapping[str, Any] | None) -> tuple[tuple[str, Any], ...]:
    """Validate and freeze a small canonical JSON-scalar tag mapping."""

    if tags is None:
        return ()
    if not isinstance(tags, Mapping):
        raise AccessTraceIntegrityError("access tags must be a mapping")
    if len(tags) > _MAX_TAGS:
        raise AccessTraceIntegrityError(f"access tags exceed {_MAX_TAGS} entries")
    normalized: dict[str, Any] = {}
    for raw_key, value in tags.items():
        if not isinstance(raw_key, str) or _TAG_KEY_RE.fullmatch(raw_key) is None:
            raise AccessTraceIntegrityError(f"invalid access tag key {raw_key!r}")
        normalized[raw_key] = _validate_tag_value(value, raw_key)
    if len(_canonical_json(normalized)) > _MAX_TAG_DOCUMENT_BYTES:
        raise AccessTraceIntegrityError(
            f"access tags exceed {_MAX_TAG_DOCUMENT_BYTES} canonical bytes"
        )
    return tuple(sorted(normalized.items()))


@dataclass(frozen=True, slots=True, order=True)
class AccessLeaf:
    """One exact non-empty half-open source range ``[offset, offset+length)``."""

    shard: str
    offset: int
    length: int

    def __post_init__(self) -> None:
        if not isinstance(self.shard, str) or not self.shard or "\x00" in self.shard:
            raise AccessTraceIntegrityError("access leaf shard must be non-empty")
        _nonnegative_int(self.offset, "access leaf offset")
        _positive_int(self.length, "access leaf length")

    def to_document(self) -> dict[str, Any]:
        return {
            "length": self.length,
            "offset": self.offset,
            "shard": self.shard,
        }

    @classmethod
    def from_document(cls, raw: Any) -> AccessLeaf:
        if not isinstance(raw, Mapping) or set(raw) != {"shard", "offset", "length"}:
            raise AccessTraceIntegrityError("access leaf has unknown/missing fields")
        return cls(
            shard=raw["shard"],
            offset=raw["offset"],
            length=raw["length"],
        )


@dataclass(frozen=True, slots=True)
class AccessOperation:
    """One successful Streamer call and its logical leaves."""

    repo_id: str
    revision: str
    inventory_fingerprint: str
    operation: str
    operation_sequence: int
    thread_id: int
    thread_name: str
    leaves: tuple[AccessLeaf, ...]
    source_requests: int
    source_bytes: int
    cache_hits: int | None
    tags: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.repo_id, str) or not self.repo_id:
            raise AccessTraceIntegrityError("access operation repo_id is missing")
        if not isinstance(self.revision, str) or not self.revision:
            raise AccessTraceIntegrityError("access operation revision is missing")
        if (
            not isinstance(self.inventory_fingerprint, str)
            or _SHA256_RE.fullmatch(self.inventory_fingerprint) is None
        ):
            raise AccessTraceIntegrityError(
                "access operation inventory fingerprint must be lowercase SHA-256"
            )
        if self.operation not in {"raw_bytes", "raw_bytes_many"}:
            raise AccessTraceIntegrityError(
                f"unsupported access operation {self.operation!r}"
            )
        _positive_int(self.operation_sequence, "operation_sequence")
        _nonnegative_int(self.thread_id, "thread_id")
        if (
            not isinstance(self.thread_name, str)
            or not self.thread_name
            or len(self.thread_name.encode("utf-8")) > 256
        ):
            raise AccessTraceIntegrityError("thread_name must be 1..256 UTF-8 bytes")
        if not isinstance(self.leaves, tuple) or not self.leaves:
            raise AccessTraceIntegrityError("access operation must contain leaves")
        if not all(isinstance(leaf, AccessLeaf) for leaf in self.leaves):
            raise AccessTraceIntegrityError("access operation contains invalid leaves")
        _nonnegative_int(self.source_requests, "source_requests")
        _nonnegative_int(self.source_bytes, "source_bytes")
        if self.cache_hits is not None:
            _nonnegative_int(self.cache_hits, "cache_hits")
            if self.cache_hits > len(self.leaves):
                raise AccessTraceIntegrityError("cache_hits exceed logical leaves")
        canonical = canonical_tags(dict(self.tags))
        if canonical != self.tags:
            raise AccessTraceIntegrityError("access operation tags are not canonical")

    def to_document(self) -> dict[str, Any]:
        return {
            "cache_hits": self.cache_hits,
            "inventory_fingerprint": self.inventory_fingerprint,
            "leaves": [leaf.to_document() for leaf in self.leaves],
            "operation": self.operation,
            "operation_sequence": self.operation_sequence,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "source_bytes": self.source_bytes,
            "source_requests": self.source_requests,
            "tags": dict(self.tags),
            "thread_id": self.thread_id,
            "thread_name": self.thread_name,
        }

    @classmethod
    def from_document(cls, raw: Any) -> AccessOperation:
        fields = {
            "cache_hits",
            "inventory_fingerprint",
            "leaves",
            "operation",
            "operation_sequence",
            "repo_id",
            "revision",
            "source_bytes",
            "source_requests",
            "tags",
            "thread_id",
            "thread_name",
        }
        if not isinstance(raw, Mapping) or set(raw) != fields:
            raise AccessTraceIntegrityError(
                "access operation has unknown/missing fields"
            )
        leaves = raw["leaves"]
        if not isinstance(leaves, list):
            raise AccessTraceIntegrityError("access operation leaves must be a list")
        return cls(
            repo_id=raw["repo_id"],
            revision=raw["revision"],
            inventory_fingerprint=raw["inventory_fingerprint"],
            operation=raw["operation"],
            operation_sequence=raw["operation_sequence"],
            thread_id=raw["thread_id"],
            thread_name=raw["thread_name"],
            leaves=tuple(AccessLeaf.from_document(leaf) for leaf in leaves),
            source_requests=raw["source_requests"],
            source_bytes=raw["source_bytes"],
            cache_hits=raw["cache_hits"],
            tags=canonical_tags(raw["tags"]),
        )


def _validate_source(repo_id: str, revision: str, fingerprint: str) -> None:
    if not isinstance(repo_id, str) or not repo_id:
        raise AccessTraceIntegrityError("trace repo_id is missing")
    if not isinstance(revision, str) or not revision:
        raise AccessTraceIntegrityError("trace revision is missing")
    if (
        not repo_id.startswith("local:")
        and _PINNED_REVISION_RE.fullmatch(revision) is None
    ):
        raise AccessTraceIntegrityError(
            "remote access traces require a pinned 40..64 hex revision"
        )
    if not isinstance(fingerprint, str) or _SHA256_RE.fullmatch(fingerprint) is None:
        raise AccessTraceIntegrityError(
            "trace inventory fingerprint must be lowercase SHA-256"
        )


@dataclass(frozen=True, slots=True)
class AccessTrace:
    """Immutable canonical trace whose identity covers every operation."""

    repo_id: str
    revision: str
    inventory_fingerprint: str
    operations: tuple[AccessOperation, ...]
    sha256: str = ""
    schema: str = TRACE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TRACE_SCHEMA:
            raise AccessTraceIntegrityError(f"unsupported trace schema {self.schema!r}")
        _validate_source(self.repo_id, self.revision, self.inventory_fingerprint)
        if not isinstance(self.operations, tuple):
            raise AccessTraceIntegrityError("trace operations must be immutable tuple")
        ordered = tuple(
            sorted(self.operations, key=lambda item: item.operation_sequence)
        )
        if ordered != self.operations:
            raise AccessTraceIntegrityError("trace operations are not sequence ordered")
        sequences: set[int] = set()
        for operation in self.operations:
            if not isinstance(operation, AccessOperation):
                raise AccessTraceIntegrityError("trace contains invalid operation")
            if operation.operation_sequence in sequences:
                raise AccessTraceIntegrityError("duplicate operation sequence")
            sequences.add(operation.operation_sequence)
            if (
                operation.repo_id != self.repo_id
                or operation.revision != self.revision
                or operation.inventory_fingerprint != self.inventory_fingerprint
            ):
                raise AccessTraceIntegrityError(
                    "trace operation source identity differs"
                )
        expected = _sha256(self._identity_document())
        if self.sha256 and self.sha256 != expected:
            raise AccessTraceIntegrityError("access trace SHA-256 mismatch")
        object.__setattr__(self, "sha256", expected)

    def _identity_document(self) -> dict[str, Any]:
        return {
            "inventory_fingerprint": self.inventory_fingerprint,
            "operations": [item.to_document() for item in self.operations],
            "repo_id": self.repo_id,
            "revision": self.revision,
            "schema": self.schema,
        }

    def to_document(self) -> dict[str, Any]:
        return {**self._identity_document(), "sha256": self.sha256}

    def to_bytes(self) -> bytes:
        return _canonical_json(self.to_document())

    def to_json(self) -> str:
        return self.to_bytes().decode("utf-8")

    def verify(self) -> None:
        if self.sha256 != _sha256(self._identity_document()):
            raise AccessTraceIntegrityError("access trace SHA-256 mismatch")

    @classmethod
    def from_document(cls, raw: Any) -> AccessTrace:
        fields = {
            "inventory_fingerprint",
            "operations",
            "repo_id",
            "revision",
            "schema",
            "sha256",
        }
        if not isinstance(raw, Mapping) or set(raw) != fields:
            raise AccessTraceIntegrityError("trace has unknown/missing fields")
        operations = raw["operations"]
        if not isinstance(operations, list):
            raise AccessTraceIntegrityError("trace operations must be a list")
        return cls(
            repo_id=raw["repo_id"],
            revision=raw["revision"],
            inventory_fingerprint=raw["inventory_fingerprint"],
            operations=tuple(
                AccessOperation.from_document(operation) for operation in operations
            ),
            sha256=raw["sha256"],
            schema=raw["schema"],
        )

    @classmethod
    def from_bytes(cls, encoded: bytes | bytearray | memoryview) -> AccessTrace:
        try:
            body = bytes(encoded)
            raw = json.loads(body.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError, TypeError) as exc:
            raise AccessTraceIntegrityError("access trace JSON is unreadable") from exc
        trace = cls.from_document(raw)
        if body != trace.to_bytes():
            raise AccessTraceIntegrityError("access trace JSON is not canonical")
        return trace

    @classmethod
    def from_json(cls, encoded: str) -> AccessTrace:
        if not isinstance(encoded, str):
            raise AccessTraceIntegrityError("access trace JSON must be text")
        return cls.from_bytes(encoded.encode("utf-8"))


@runtime_checkable
class AccessObserver(Protocol):
    """Minimal fail-open callback contract consumed by ``Streamer``."""

    def observe(self, operation: AccessOperation) -> bool | None: ...


class AccessTraceRecorder:
    """Thread-safe observer with optional explicit capacity limits.

    Recording is complete by default. Callers that deliberately need a lossy
    diagnostic trace may set either limit and can observe every drop in
    :meth:`metrics`.
    """

    def __init__(
        self,
        *,
        max_operations: int | None = None,
        max_leaves: int | None = None,
    ) -> None:
        self.max_operations = (
            None
            if max_operations is None
            else _positive_int(max_operations, "max_operations")
        )
        self.max_leaves = (
            None if max_leaves is None else _positive_int(max_leaves, "max_leaves")
        )
        self._lock = threading.Lock()
        self._scope_tags: ContextVar[tuple[tuple[str, Any], ...]] = ContextVar(
            f"immer_access_trace_scope_{id(self)}",
            default=(),
        )
        self._operations: list[AccessOperation] = []
        self._leaves = 0
        self._source: tuple[str, str, str] | None = None
        self._dropped_capacity = 0
        self._dropped_identity = 0

    @contextmanager
    def scope(self, **tags: Any) -> Iterator[AccessTraceRecorder]:
        """Attach bounded tags to this request context and copied child contexts."""

        additions = canonical_tags(tags)
        merged = dict(self._scope_tags.get())
        merged.update(additions)
        frozen = canonical_tags(merged)
        token = self._scope_tags.set(frozen)
        try:
            yield self
        finally:
            self._scope_tags.reset(token)

    def observe(self, operation: AccessOperation) -> bool:
        tags = self._scope_tags.get()
        if tags:
            merged = dict(operation.tags)
            merged.update(tags)
            operation = replace(operation, tags=canonical_tags(merged))
        source = (
            operation.repo_id,
            operation.revision,
            operation.inventory_fingerprint,
        )
        with self._lock:
            if self._source is not None and self._source != source:
                self._dropped_identity += 1
                return False
            operations_full = (
                self.max_operations is not None
                and len(self._operations) >= self.max_operations
            )
            leaves_full = (
                self.max_leaves is not None
                and self._leaves + len(operation.leaves) > self.max_leaves
            )
            if operations_full or leaves_full:
                self._dropped_capacity += 1
                return False
            self._source = source
            self._operations.append(operation)
            self._leaves += len(operation.leaves)
            return True

    def snapshot(self) -> AccessTrace:
        with self._lock:
            source = self._source
            operations = tuple(
                sorted(self._operations, key=lambda item: item.operation_sequence)
            )
        if source is None:
            raise AccessTraceIntegrityError("cannot snapshot an empty access trace")
        return AccessTrace(
            repo_id=source[0],
            revision=source[1],
            inventory_fingerprint=source[2],
            operations=operations,
        )

    def metrics(self) -> dict[str, int]:
        with self._lock:
            return {
                "operations": len(self._operations),
                "leaves": self._leaves,
                "dropped_capacity": self._dropped_capacity,
                "dropped_identity": self._dropped_identity,
            }


@dataclass(frozen=True, slots=True)
class ReplayReceipt:
    """Explicit outcome of one complete or explicitly limited cache replay."""

    trace_sha256: str
    unique_leaves: int
    unique_bytes: int
    selected_leaves: int
    selected_bytes: int
    warmed_leaves: int
    warmed_bytes: int
    source_requests: int
    source_bytes: int
    windows: int
    largest_window_leaves: int
    largest_window_bytes: int
    duplicate_leaves: int
    limit_skipped_leaves: int
    limit_skipped_bytes: int
    budget_declined_leaves: int
    budget_declined_bytes: int
    budget_declined: bool


def _streamer_fingerprint(streamer: Streamer) -> str:
    metrics = streamer.metrics()
    fingerprint = metrics.get("inventory_source_fingerprint")
    if fingerprint is None:
        streamer.inventory()
        fingerprint = streamer.metrics().get("inventory_source_fingerprint")
    if not isinstance(fingerprint, str):
        raise AccessTraceIdentityError("streamer inventory fingerprint is unavailable")
    return fingerprint


def replay_access_trace(
    streamer: Streamer,
    trace: AccessTrace,
    *,
    max_leaves: int | None = None,
    max_bytes: int | None = None,
    window_max_leaves: int | None = 4096,
    window_max_bytes: int | None = 256 * 1024 * 1024,
) -> ReplayReceipt:
    """Warm exact trace leaves in configurable sequential working windows.

    Source identity is checked before any payload read.  Hard transfer-budget
    rejection is a soft decline recorded in the receipt; integrity failures
    and source mismatches remain hard failures. ``max_*`` optionally limits the
    *whole* replay; by default the whole trace is consumed. ``window_max_*``
    only controls one cache-fill call and never discards later leaves. Passing
    ``None`` removes the corresponding limit.
    """

    from .streamer import ByteBudgetExceeded

    if not isinstance(trace, AccessTrace):
        raise AccessTraceIntegrityError("trace must be an AccessTrace")
    trace.verify()
    if max_leaves is not None:
        max_leaves = _nonnegative_int(max_leaves, "max_leaves")
    if max_bytes is not None:
        max_bytes = _nonnegative_int(max_bytes, "max_bytes")
    if window_max_leaves is not None:
        window_max_leaves = _positive_int(window_max_leaves, "window_max_leaves")
    if window_max_bytes is not None:
        window_max_bytes = _positive_int(window_max_bytes, "window_max_bytes")
    if streamer.repo_id != trace.repo_id or streamer.revision != trace.revision:
        raise AccessTraceIdentityError(
            "access trace repo/revision does not match streamer"
        )
    if _streamer_fingerprint(streamer) != trace.inventory_fingerprint:
        raise AccessTraceIdentityError(
            "access trace inventory fingerprint does not match streamer"
        )

    ordered: list[AccessLeaf] = []
    seen: set[AccessLeaf] = set()
    logical_leaves = 0
    for operation in trace.operations:
        for leaf in operation.leaves:
            logical_leaves += 1
            if leaf not in seen:
                seen.add(leaf)
                ordered.append(leaf)
    unique_bytes = sum(leaf.length for leaf in ordered)

    selected: list[AccessLeaf] = []
    selected_bytes = 0
    skipped: list[AccessLeaf] = []
    for leaf in ordered:
        leaves_limited = max_leaves is not None and len(selected) >= max_leaves
        bytes_limited = (
            max_bytes is not None and selected_bytes + leaf.length > max_bytes
        )
        if leaves_limited or bytes_limited:
            skipped.append(leaf)
            continue
        selected.append(leaf)
        selected_bytes += leaf.length

    windows: list[list[AccessLeaf]] = []
    current: list[AccessLeaf] = []
    current_bytes = 0
    for leaf in selected:
        shard_changed = bool(current and current[0].shard != leaf.shard)
        leaves_full = (
            window_max_leaves is not None and len(current) >= window_max_leaves
        )
        bytes_full = bool(
            current
            and window_max_bytes is not None
            and current_bytes + leaf.length > window_max_bytes
        )
        if shard_changed or leaves_full or bytes_full:
            windows.append(current)
            current = []
            current_bytes = 0
        current.append(leaf)
        current_bytes += leaf.length
    if current:
        windows.append(current)

    warmed_leaves = 0
    warmed_bytes = 0
    source_requests = 0
    source_bytes = 0
    declined_leaves = 0
    declined_bytes = 0
    with streamer.cache_priority(0):
        for leaves in windows:
            group_bytes = sum(leaf.length for leaf in leaves)
            try:
                result = streamer.raw_bytes_many(
                    leaves[0].shard,
                    ((leaf.offset, leaf.length) for leaf in leaves),
                    resident_limit_bytes=group_bytes,
                    max_gap_bytes=0,
                )
            except ByteBudgetExceeded:
                declined_leaves += len(leaves)
                declined_bytes += group_bytes
                continue
            warmed_leaves += len(leaves)
            warmed_bytes += group_bytes
            source_requests += result.source_requests
            source_bytes += result.source_bytes

    return ReplayReceipt(
        trace_sha256=trace.sha256,
        unique_leaves=len(ordered),
        unique_bytes=unique_bytes,
        selected_leaves=len(selected),
        selected_bytes=selected_bytes,
        warmed_leaves=warmed_leaves,
        warmed_bytes=warmed_bytes,
        source_requests=source_requests,
        source_bytes=source_bytes,
        windows=len(windows),
        largest_window_leaves=max((len(window) for window in windows), default=0),
        largest_window_bytes=max(
            (sum(leaf.length for leaf in window) for window in windows), default=0
        ),
        duplicate_leaves=logical_leaves - len(ordered),
        limit_skipped_leaves=len(skipped),
        limit_skipped_bytes=sum(leaf.length for leaf in skipped),
        budget_declined_leaves=declined_leaves,
        budget_declined_bytes=declined_bytes,
        budget_declined=bool(declined_leaves),
    )
