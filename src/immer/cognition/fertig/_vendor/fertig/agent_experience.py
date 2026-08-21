"""Provider-neutral, append-only experience stream for agent work.

The stream is intentionally a *recording substrate*, not a training method.
It preserves the observable causal chain that a later Intelligence Recipe
miner can learn from: goals, messages, actions, tool results, state deltas,
tests, human corrections and outcomes.  Large payloads live in a
content-addressed blob directory while the ordered event log stays JSONL.

The store has a single-writer contract.  A truncated final JSONL record (for
example after a process crash) is ignored while reading and removed before the
next append; any corruption before the final line fails closed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from types import MappingProxyType
from typing import Iterable, Iterator, Mapping, TypeAlias


EXPERIENCE_SCHEMA = "fertig.agent-experience"
EXPERIENCE_VERSION = 1
MAX_INLINE_BYTES = 8 * 1024 * 1024
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 100_000
MAX_ID_LENGTH = 256

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/+~-]{0,255}")
_SHA256 = re.compile(r"[0-9a-f]{64}")

JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


class ExperienceError(ValueError):
    """Base class for invalid or corrupted experience data."""


class ExperienceCorruptionError(ExperienceError):
    """The persisted stream does not form a valid append-only history."""


class EventKind(str, Enum):
    """Observable event categories shared by agent providers."""

    SESSION_START = "session_start"
    SESSION_END = "session_end"
    GOAL = "goal"
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    OBSERVATION = "observation"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ACTION = "action"
    TERMINAL = "terminal"
    FILE_READ = "file_read"
    FILE_DIFF = "file_diff"
    TEST_RESULT = "test_result"
    HUMAN_FEEDBACK = "human_feedback"
    OUTCOME = "outcome"
    CHECKPOINT = "checkpoint"


class OutcomeStatus(str, Enum):
    """Outcome known from observable evidence."""

    UNKNOWN = "unknown"
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    ABORTED = "aborted"


def _identifier(value: object, field: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not _ID.fullmatch(value):
        suffix = " or null" if optional else ""
        raise ExperienceError(f"{field} must match {_ID.pattern!r}{suffix}")
    return value


def _strict_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ExperienceError(f"{field} must be an integer >= {minimum}")
    return value


def _optional_text(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > MAX_INLINE_BYTES:
        raise ExperienceError(f"{field} must be a bounded string or null")
    return value


def _json_copy(value: object, *, field: str = "payload") -> JSONValue:
    """Validate and detach one bounded JSON value."""

    nodes = 0

    def visit(item: object, depth: int) -> JSONValue:
        nonlocal nodes
        nodes += 1
        if nodes > MAX_JSON_NODES:
            raise ExperienceError(f"{field} exceeds {MAX_JSON_NODES} JSON nodes")
        if depth > MAX_JSON_DEPTH:
            raise ExperienceError(f"{field} exceeds depth {MAX_JSON_DEPTH}")
        if item is None or isinstance(item, (bool, str)):
            return item
        if isinstance(item, int) and not isinstance(item, bool):
            if not -(2**63) <= item < 2**63:
                raise ExperienceError(f"{field} integer exceeds signed 63-bit range")
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ExperienceError(f"{field} contains a non-finite float")
            return item
        if isinstance(item, Mapping):
            result: dict[str, JSONValue] = {}
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ExperienceError(f"{field} object keys must be strings")
                result[key] = visit(child, depth + 1)
            return result
        if isinstance(item, (tuple, list)):
            return [visit(child, depth + 1) for child in item]
        raise ExperienceError(f"{field} contains non-JSON value {type(item).__name__}")

    copied = visit(value, 0)
    encoded = _canonical_bytes(copied)
    if len(encoded) > MAX_INLINE_BYTES:
        raise ExperienceError(
            f"{field} exceeds {MAX_INLINE_BYTES} bytes; store it as a blob"
        )
    return copied


def _freeze_json(value: JSONValue) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(child) for child in value)
    return value


def _thaw_json(value: object) -> JSONValue:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(child) for child in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ExperienceError(f"stored value is not JSON: {type(value).__name__}")


def _canonical_bytes(value: JSONValue | Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _exact_fields(
    value: Mapping[str, object], required: set[str], optional: set[str] = frozenset()
) -> None:
    keys = set(value)
    missing = required - keys
    extra = keys - required - optional
    if missing:
        raise ExperienceError(f"missing fields: {sorted(missing)}")
    if extra:
        raise ExperienceError(f"unknown fields: {sorted(extra)}")


@dataclass(frozen=True, slots=True)
class BlobRef:
    """Reference to one immutable content-addressed blob."""

    digest: str
    size_bytes: int
    media_type: str = "application/octet-stream"
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.digest, str) or not _SHA256.fullmatch(self.digest):
            raise ExperienceError("blob digest must be lowercase SHA-256")
        _strict_int(self.size_bytes, "blob size_bytes")
        if (
            not isinstance(self.media_type, str)
            or not self.media_type
            or len(self.media_type) > 255
        ):
            raise ExperienceError("blob media_type must be a bounded non-empty string")
        _optional_text(self.name, "blob name")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "digest": self.digest,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "BlobRef":
        _exact_fields(value, {"digest", "size_bytes", "media_type", "name"})
        return cls(
            digest=value["digest"],
            size_bytes=value["size_bytes"],
            media_type=value["media_type"],
            name=value["name"],
        )


@dataclass(frozen=True, slots=True)
class StateRef:
    """Observable state represented by a canonical digest and optional blobs."""

    digest: str
    label: str | None = None
    blobs: tuple[BlobRef, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.digest, str) or not _SHA256.fullmatch(self.digest):
            raise ExperienceError("state digest must be lowercase SHA-256")
        _optional_text(self.label, "state label")
        if not isinstance(self.blobs, tuple) or not all(
            isinstance(blob, BlobRef) for blob in self.blobs
        ):
            raise ExperienceError("state blobs must be a tuple of BlobRef")
        digests = [blob.digest for blob in self.blobs]
        if len(digests) != len(set(digests)):
            raise ExperienceError("state blobs must be unique")

    @classmethod
    def from_payload(
        cls,
        value: Mapping[str, object],
        *,
        label: str | None = None,
        blobs: Iterable[BlobRef] = (),
    ) -> "StateRef":
        copied = _json_copy(value, field="state payload")
        assert isinstance(copied, dict)
        return cls(
            hashlib.sha256(_canonical_bytes(copied)).hexdigest(),
            label,
            tuple(blobs),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "digest": self.digest,
            "label": self.label,
            "blobs": [blob.to_dict() for blob in self.blobs],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "StateRef":
        _exact_fields(value, {"digest", "label", "blobs"})
        raw_blobs = value["blobs"]
        if not isinstance(raw_blobs, list):
            raise ExperienceError("state blobs must be an array")
        return cls(
            digest=value["digest"],
            label=value["label"],
            blobs=tuple(_blob_from_object(blob) for blob in raw_blobs),
        )


@dataclass(frozen=True, slots=True)
class Outcome:
    """Outcome attached to an event, backed by observable evidence when known."""

    status: OutcomeStatus
    score: float | None = None
    message: str | None = None
    evidence: tuple[BlobRef, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.status, str):
            try:
                object.__setattr__(self, "status", OutcomeStatus(self.status))
            except ValueError as exc:
                raise ExperienceError("unknown outcome status") from exc
        if not isinstance(self.status, OutcomeStatus):
            raise ExperienceError("outcome status must be OutcomeStatus")
        if self.score is not None and (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not math.isfinite(float(self.score))
            or not 0.0 <= float(self.score) <= 1.0
        ):
            raise ExperienceError("outcome score must be finite in [0, 1] or null")
        if self.score is not None:
            object.__setattr__(self, "score", float(self.score))
        _optional_text(self.message, "outcome message")
        if not isinstance(self.evidence, tuple) or not all(
            isinstance(item, BlobRef) for item in self.evidence
        ):
            raise ExperienceError("outcome evidence must be a tuple of BlobRef")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "status": self.status.value,
            "score": self.score,
            "message": self.message,
            "evidence": [item.to_dict() for item in self.evidence],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "Outcome":
        _exact_fields(value, {"status", "score", "message", "evidence"})
        raw_evidence = value["evidence"]
        if not isinstance(raw_evidence, list):
            raise ExperienceError("outcome evidence must be an array")
        return cls(
            status=value["status"],
            score=value["score"],
            message=value["message"],
            evidence=tuple(_blob_from_object(item) for item in raw_evidence),
        )


def _blob_from_object(value: object) -> BlobRef:
    if not isinstance(value, Mapping):
        raise ExperienceError("blob reference must be an object")
    return BlobRef.from_dict(value)


def _state_from_object(value: object) -> StateRef | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ExperienceError("state reference must be an object or null")
    return StateRef.from_dict(value)


def _outcome_from_object(value: object) -> Outcome | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ExperienceError("outcome must be an object or null")
    return Outcome.from_dict(value)


@dataclass(frozen=True, slots=True)
class AgentExperience:
    """One immutable event in a per-session hash chain."""

    session_id: str
    sequence: int
    timestamp_ns: int
    monotonic_ns: int
    actor: str
    kind: EventKind
    payload: Mapping[str, object]
    previous_hash: str | None
    event_hash: str
    parent_event_id: str | None = None
    span_id: str | None = None
    goal_id: str | None = None
    model_id: str | None = None
    before: StateRef | None = None
    after: StateRef | None = None
    blobs: tuple[BlobRef, ...] = ()
    outcome: Outcome | None = None
    privacy_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.session_id, "session_id")
        _strict_int(self.sequence, "sequence")
        _strict_int(self.timestamp_ns, "timestamp_ns")
        _strict_int(self.monotonic_ns, "monotonic_ns")
        _identifier(self.actor, "actor")
        if isinstance(self.kind, str):
            try:
                object.__setattr__(self, "kind", EventKind(self.kind))
            except ValueError as exc:
                raise ExperienceError("unknown event kind") from exc
        if not isinstance(self.kind, EventKind):
            raise ExperienceError("kind must be EventKind")
        copied = _json_copy(self.payload)
        if not isinstance(copied, dict):
            raise ExperienceError("payload must be a JSON object")
        object.__setattr__(self, "payload", _freeze_json(copied))
        if self.previous_hash is not None and not _SHA256.fullmatch(self.previous_hash):
            raise ExperienceError("previous_hash must be lowercase SHA-256 or null")
        if not isinstance(self.event_hash, str) or not _SHA256.fullmatch(
            self.event_hash
        ):
            raise ExperienceError("event_hash must be lowercase SHA-256")
        for field, value in (
            ("parent_event_id", self.parent_event_id),
            ("span_id", self.span_id),
            ("goal_id", self.goal_id),
            ("model_id", self.model_id),
        ):
            _identifier(value, field, optional=True)
        if self.before is not None and not isinstance(self.before, StateRef):
            raise ExperienceError("before must be StateRef or null")
        if self.after is not None and not isinstance(self.after, StateRef):
            raise ExperienceError("after must be StateRef or null")
        if not isinstance(self.blobs, tuple) or not all(
            isinstance(blob, BlobRef) for blob in self.blobs
        ):
            raise ExperienceError("blobs must be a tuple of BlobRef")
        if self.outcome is not None and not isinstance(self.outcome, Outcome):
            raise ExperienceError("outcome must be Outcome or null")
        if not isinstance(self.privacy_labels, tuple):
            raise ExperienceError("privacy_labels must be a tuple")
        labels: list[str] = []
        for label in self.privacy_labels:
            checked = _identifier(label, "privacy label")
            assert checked is not None
            labels.append(checked)
        if len(labels) != len(set(labels)):
            raise ExperienceError("privacy_labels must be unique")
        if self.sequence == 0 and self.previous_hash is not None:
            raise ExperienceError("first session event cannot have previous_hash")
        if self.sequence > 0 and self.previous_hash is None:
            raise ExperienceError("non-first session event requires previous_hash")
        expected = hashlib.sha256(_canonical_bytes(self._preimage())).hexdigest()
        if expected != self.event_hash:
            raise ExperienceCorruptionError("event_hash does not match event content")

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        sequence: int,
        actor: str,
        kind: EventKind | str,
        payload: Mapping[str, object] | None = None,
        previous_hash: str | None,
        timestamp_ns: int | None = None,
        monotonic_ns: int | None = None,
        parent_event_id: str | None = None,
        span_id: str | None = None,
        goal_id: str | None = None,
        model_id: str | None = None,
        before: StateRef | None = None,
        after: StateRef | None = None,
        blobs: Iterable[BlobRef] = (),
        outcome: Outcome | None = None,
        privacy_labels: Iterable[str] = (),
    ) -> "AgentExperience":
        base = cls.__new__(cls)
        object.__setattr__(base, "session_id", session_id)
        object.__setattr__(base, "sequence", sequence)
        object.__setattr__(
            base,
            "timestamp_ns",
            time.time_ns() if timestamp_ns is None else timestamp_ns,
        )
        object.__setattr__(
            base,
            "monotonic_ns",
            time.monotonic_ns() if monotonic_ns is None else monotonic_ns,
        )
        object.__setattr__(base, "actor", actor)
        object.__setattr__(base, "kind", EventKind(kind))
        copied = _json_copy({} if payload is None else payload)
        assert isinstance(copied, dict)
        object.__setattr__(base, "payload", _freeze_json(copied))
        object.__setattr__(base, "previous_hash", previous_hash)
        object.__setattr__(base, "event_hash", "0" * 64)
        object.__setattr__(base, "parent_event_id", parent_event_id)
        object.__setattr__(base, "span_id", span_id)
        object.__setattr__(base, "goal_id", goal_id)
        object.__setattr__(base, "model_id", model_id)
        object.__setattr__(base, "before", before)
        object.__setattr__(base, "after", after)
        object.__setattr__(base, "blobs", tuple(blobs))
        object.__setattr__(base, "outcome", outcome)
        object.__setattr__(base, "privacy_labels", tuple(privacy_labels))
        digest = hashlib.sha256(_canonical_bytes(base._preimage())).hexdigest()
        object.__setattr__(base, "event_hash", digest)
        base.__post_init__()
        return base

    @property
    def event_id(self) -> str:
        """Content-derived event identity."""

        return self.event_hash

    def _preimage(self) -> dict[str, JSONValue]:
        return {
            "schema": EXPERIENCE_SCHEMA,
            "version": EXPERIENCE_VERSION,
            "session_id": self.session_id,
            "sequence": self.sequence,
            "timestamp_ns": self.timestamp_ns,
            "monotonic_ns": self.monotonic_ns,
            "actor": self.actor,
            "kind": self.kind.value,
            "payload": _thaw_json(self.payload),
            "previous_hash": self.previous_hash,
            "parent_event_id": self.parent_event_id,
            "span_id": self.span_id,
            "goal_id": self.goal_id,
            "model_id": self.model_id,
            "before": None if self.before is None else self.before.to_dict(),
            "after": None if self.after is None else self.after.to_dict(),
            "blobs": [blob.to_dict() for blob in self.blobs],
            "outcome": None if self.outcome is None else self.outcome.to_dict(),
            "privacy_labels": list(self.privacy_labels),
        }

    def to_dict(self) -> dict[str, JSONValue]:
        result = self._preimage()
        result["event_hash"] = self.event_hash
        return result

    def to_json(self) -> str:
        return _canonical_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "AgentExperience":
        required = {
            "schema",
            "version",
            "session_id",
            "sequence",
            "timestamp_ns",
            "monotonic_ns",
            "actor",
            "kind",
            "payload",
            "previous_hash",
            "event_hash",
            "parent_event_id",
            "span_id",
            "goal_id",
            "model_id",
            "before",
            "after",
            "blobs",
            "outcome",
            "privacy_labels",
        }
        _exact_fields(value, required)
        if (
            value["schema"] != EXPERIENCE_SCHEMA
            or value["version"] != EXPERIENCE_VERSION
        ):
            raise ExperienceError("unsupported experience schema/version")
        payload = value["payload"]
        raw_blobs = value["blobs"]
        labels = value["privacy_labels"]
        if not isinstance(payload, Mapping):
            raise ExperienceError("payload must be an object")
        if not isinstance(raw_blobs, list):
            raise ExperienceError("blobs must be an array")
        if not isinstance(labels, list) or not all(
            isinstance(label, str) for label in labels
        ):
            raise ExperienceError("privacy_labels must be an array of strings")
        return cls(
            session_id=value["session_id"],
            sequence=value["sequence"],
            timestamp_ns=value["timestamp_ns"],
            monotonic_ns=value["monotonic_ns"],
            actor=value["actor"],
            kind=value["kind"],
            payload=payload,
            previous_hash=value["previous_hash"],
            event_hash=value["event_hash"],
            parent_event_id=value["parent_event_id"],
            span_id=value["span_id"],
            goal_id=value["goal_id"],
            model_id=value["model_id"],
            before=_state_from_object(value["before"]),
            after=_state_from_object(value["after"]),
            blobs=tuple(_blob_from_object(blob) for blob in raw_blobs),
            outcome=_outcome_from_object(value["outcome"]),
            privacy_labels=tuple(labels),
        )

    @classmethod
    def from_json(cls, raw: str | bytes) -> "AgentExperience":
        try:
            decoded = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ExperienceError("invalid experience JSON") from exc
        if not isinstance(decoded, dict):
            raise ExperienceError("experience JSON must contain an object")
        return cls.from_dict(decoded)


@dataclass(frozen=True, slots=True)
class SessionSummary:
    """Derived overview of one recorded session."""

    session_id: str
    event_count: int
    first_timestamp_ns: int
    last_timestamp_ns: int
    terminal_hash: str
    kinds: Mapping[str, int]
    outcome_statuses: Mapping[str, int]


class AgentExperienceStore:
    """Single-writer JSONL event log with content-addressed blobs."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.events_path = self.root / "experience.jsonl"
        self.blobs_path = self.root / "blobs" / "sha256"
        self._lock = threading.RLock()

    def put_blob(
        self,
        data: bytes | bytearray | memoryview | str,
        *,
        media_type: str = "application/octet-stream",
        name: str | None = None,
    ) -> BlobRef:
        """Store opaque bytes once and return their immutable reference."""

        if isinstance(data, str):
            raw = data.encode("utf-8")
            if media_type == "application/octet-stream":
                media_type = "text/plain; charset=utf-8"
        elif isinstance(data, (bytes, bytearray, memoryview)):
            raw = bytes(data)
        else:
            raise ExperienceError("blob data must be bytes-like or str")
        ref = BlobRef(hashlib.sha256(raw).hexdigest(), len(raw), media_type, name)
        target = self.blobs_path / ref.digest[:2] / ref.digest
        with self._lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.read_bytes() != raw:
                    raise ExperienceCorruptionError(
                        "blob digest collision or corruption"
                    )
                return ref
            fd, temporary = tempfile.mkstemp(prefix=".blob-", dir=target.parent)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(raw)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return ref

    def read_blob(self, ref: BlobRef) -> bytes:
        target = self.blobs_path / ref.digest[:2] / ref.digest
        try:
            raw = target.read_bytes()
        except FileNotFoundError as exc:
            raise ExperienceCorruptionError(f"missing blob {ref.digest}") from exc
        if len(raw) != ref.size_bytes or hashlib.sha256(raw).hexdigest() != ref.digest:
            raise ExperienceCorruptionError(f"blob {ref.digest} failed verification")
        return raw

    def append(
        self,
        *,
        session_id: str,
        actor: str,
        kind: EventKind | str,
        payload: Mapping[str, object] | None = None,
        timestamp_ns: int | None = None,
        monotonic_ns: int | None = None,
        parent_event_id: str | None = None,
        span_id: str | None = None,
        goal_id: str | None = None,
        model_id: str | None = None,
        before: StateRef | None = None,
        after: StateRef | None = None,
        blobs: Iterable[BlobRef] = (),
        outcome: Outcome | None = None,
        privacy_labels: Iterable[str] = (),
    ) -> AgentExperience:
        """Append one event after validating its per-session predecessor."""

        with self._lock:
            events, valid_bytes = self._read_records()
            self._repair_tail(valid_bytes)
            session_events = [
                event for event in events if event.session_id == session_id
            ]
            known_ids = {event.event_id for event in session_events}
            if parent_event_id is not None and parent_event_id not in known_ids:
                raise ExperienceError("parent_event_id is not in this session")
            previous = session_events[-1].event_hash if session_events else None
            event = AgentExperience.create(
                session_id=session_id,
                sequence=len(session_events),
                actor=actor,
                kind=kind,
                payload=payload,
                previous_hash=previous,
                timestamp_ns=timestamp_ns,
                monotonic_ns=monotonic_ns,
                parent_event_id=parent_event_id,
                span_id=span_id,
                goal_id=goal_id,
                model_id=model_id,
                before=before,
                after=after,
                blobs=blobs,
                outcome=outcome,
                privacy_labels=privacy_labels,
            )
            self.root.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("ab") as handle:
                handle.write(_canonical_bytes(event.to_dict()) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            return event

    def events(self, session_id: str | None = None) -> tuple[AgentExperience, ...]:
        """Read and validate all complete records, optionally for one session."""

        with self._lock:
            events, _ = self._read_records()
        if session_id is None:
            return tuple(events)
        _identifier(session_id, "session_id")
        return tuple(event for event in events if event.session_id == session_id)

    def iter_events(self, session_id: str | None = None) -> Iterator[AgentExperience]:
        yield from self.events(session_id)

    def sessions(self) -> tuple[str, ...]:
        return tuple(sorted({event.session_id for event in self.events()}))

    def summary(self, session_id: str) -> SessionSummary:
        events = self.events(session_id)
        if not events:
            raise ExperienceError(f"unknown or empty session {session_id!r}")
        kinds = Counter(event.kind.value for event in events)
        outcomes = Counter(
            event.outcome.status.value for event in events if event.outcome is not None
        )
        return SessionSummary(
            session_id=session_id,
            event_count=len(events),
            first_timestamp_ns=events[0].timestamp_ns,
            last_timestamp_ns=events[-1].timestamp_ns,
            terminal_hash=events[-1].event_hash,
            kinds=MappingProxyType(dict(sorted(kinds.items()))),
            outcome_statuses=MappingProxyType(dict(sorted(outcomes.items()))),
        )

    def _repair_tail(self, valid_bytes: int) -> None:
        if not self.events_path.exists():
            return
        size = self.events_path.stat().st_size
        if valid_bytes == size:
            return
        with self.events_path.open("r+b") as handle:
            handle.truncate(valid_bytes)
            handle.flush()
            os.fsync(handle.fileno())

    def _read_records(self) -> tuple[list[AgentExperience], int]:
        if not self.events_path.exists():
            return [], 0
        raw = self.events_path.read_bytes()
        events: list[AgentExperience] = []
        chains: dict[str, tuple[int, str]] = {}
        offset = 0
        valid_bytes = 0
        lines = raw.splitlines(keepends=True)
        for index, line in enumerate(lines):
            complete = line.endswith(b"\n")
            body = line[:-1] if complete else line
            if not body.strip():
                if index == len(lines) - 1 and not complete:
                    break
                raise ExperienceCorruptionError("empty record in experience stream")
            try:
                event = AgentExperience.from_json(body)
            except ExperienceError:
                if index == len(lines) - 1 and not complete:
                    break
                raise
            expected_sequence, expected_previous = chains.get(
                event.session_id, (0, None)
            )
            if event.sequence != expected_sequence:
                raise ExperienceCorruptionError(
                    f"session {event.session_id!r} has non-contiguous sequence"
                )
            if event.previous_hash != expected_previous:
                raise ExperienceCorruptionError(
                    f"session {event.session_id!r} has broken hash chain"
                )
            events.append(event)
            chains[event.session_id] = (event.sequence + 1, event.event_hash)
            offset += len(line)
            valid_bytes = offset
        return events, valid_bytes


__all__ = [
    "EXPERIENCE_SCHEMA",
    "EXPERIENCE_VERSION",
    "AgentExperience",
    "AgentExperienceStore",
    "BlobRef",
    "EventKind",
    "ExperienceCorruptionError",
    "ExperienceError",
    "Outcome",
    "OutcomeStatus",
    "SessionSummary",
    "StateRef",
]
