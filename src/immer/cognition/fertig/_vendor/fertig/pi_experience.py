"""Empirical bridge from Pi's event stream into agent experience storage.

Only boundaries explicitly emitted by Pi are projected.  The provider payload
and the complete capture remain available verbatim; no goal, phase, outcome,
operator, or training-recipe semantics are inferred here.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .agent_experience import AgentExperienceStore, BlobRef, EventKind
from .pi_trace import PiTrace, RawPiEvent, read_pi_trace


_EXPERIENCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/+~-]{0,255}")


class PiExperienceError(ValueError):
    """A Pi capture cannot be projected without losing its identity."""


@dataclass(frozen=True, slots=True)
class PiIngestResult:
    """Exact accounting for one completed or live Pi import."""

    session_id: str
    raw_event_count: int
    mapped_event_count: int
    mapped_kinds: Mapping[str, int]
    ignored_types: Mapping[str, int]
    capture_blob: BlobRef | None
    trace_sha256: str
    already_present: bool = False

    @property
    def blob_ref(self) -> BlobRef | None:
        return self.capture_blob

    @property
    def blob_hash(self) -> str | None:
        return None if self.capture_blob is None else self.capture_blob.digest


@dataclass(frozen=True, slots=True)
class _Projection:
    kind: EventKind
    actor: str
    model_id: str | None = None


def _projection(event: RawPiEvent) -> _Projection | None:
    if event.type == "session":
        return _Projection(EventKind.SESSION_START, "pi")
    if event.type == "message_end":
        message = event.parsed.get("message")
        if not isinstance(message, Mapping):
            raise PiExperienceError(
                f"message_end on line {event.line_number} has no message object"
            )
        role = message.get("role")
        if role == "user":
            return _Projection(EventKind.USER_MESSAGE, "user")
        if role == "assistant":
            return _Projection(
                EventKind.ASSISTANT_MESSAGE,
                "agent",
                _model_id(message),
            )
        # Pi repeats execution results as role=toolResult messages.  The
        # execution_end channel below is the sole neutral TOOL_RESULT mapping.
        return None
    if event.type == "tool_execution_start":
        return _Projection(EventKind.TOOL_CALL, "pi-tool")
    if event.type == "tool_execution_end":
        return _Projection(EventKind.TOOL_RESULT, "pi-tool")
    if event.type == "agent_end":
        return _Projection(EventKind.TERMINAL, "pi")
    if event.type == "agent_settled":
        return _Projection(EventKind.SESSION_END, "pi")
    return None


def _model_id(message: Mapping[str, Any]) -> str | None:
    provider = message.get("provider")
    model = message.get("model")
    parts = [item for item in (provider, model) if isinstance(item, str) and item]
    candidate = ":".join(parts)
    return candidate if candidate and _EXPERIENCE_ID.fullmatch(candidate) else None


def _observed_session_id(trace: PiTrace) -> str | None:
    session_events = [event for event in trace.events if event.type == "session"]
    if len(session_events) != 1:
        raise PiExperienceError(
            f"capture must contain exactly one Pi session event, found {len(session_events)}"
        )
    ids = {
        event.parsed.get("id")
        for event in session_events
        if isinstance(event.parsed.get("id"), str)
    }
    return next(iter(ids), None)


def _checked_session_id(value: str | None, trace_sha256: str) -> str:
    if value is None:
        return f"pi:{trace_sha256}"
    if not _EXPERIENCE_ID.fullmatch(value):
        raise PiExperienceError("session_id is not valid for AgentExperienceStore")
    return value


def _trace_bytes(trace: PiTrace) -> bytes:
    raw = "".join(event.raw for event in trace.events).encode("utf-8")
    if trace.truncated_tail is not None:
        raw += trace.truncated_tail
    if hashlib.sha256(raw).hexdigest() != trace.summary.sha256:
        raise PiExperienceError("PiTrace no longer reconstructs its source hash")
    return raw


def _event_timestamp_ns(event: RawPiEvent, base_ns: int) -> int:
    candidate: object = event.parsed.get("timestamp")
    message = event.parsed.get("message")
    if candidate is None and isinstance(message, Mapping):
        candidate = message.get("timestamp")
    if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
        return max(0, int(candidate * 1_000_000))
    if isinstance(candidate, str):
        try:
            seconds = datetime.fromisoformat(
                candidate.replace("Z", "+00:00")
            ).timestamp()
            return max(0, int(seconds * 1_000_000_000))
        except ValueError:
            pass
    return base_ns + event.line_number


def _base_timestamp_ns(events: tuple[RawPiEvent, ...]) -> int:
    session = next((event for event in events if event.type == "session"), None)
    if session is None:
        return 0
    return _event_timestamp_ns(session, 0)


def _payload_and_blobs(
    event: RawPiEvent,
    store: AgentExperienceStore,
    *,
    source: str,
    max_inline_chars: int,
) -> tuple[dict[str, object], tuple[BlobRef, ...]]:
    raw_payload = json.dumps(
        event.parsed,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    common: dict[str, object] = {
        "source": source,
        "pi_event_type": event.type,
        "pi_line_number": event.line_number,
    }
    if len(raw_payload) <= max_inline_chars:
        common["provider_payload"] = event.parsed
        return common, ()
    blob = store.put_blob(
        event.raw,
        media_type="application/x-ndjson; charset=utf-8",
        name=f"pi-event-{event.line_number}.jsonl",
    )
    common["provider_payload_blob"] = blob.to_dict()
    return common, (blob,)


def _validate_options(source: str, max_inline_chars: int) -> None:
    if not isinstance(source, str) or not source or len(source) > 256:
        raise PiExperienceError("source must be a bounded non-empty string")
    if (
        isinstance(max_inline_chars, bool)
        or not isinstance(max_inline_chars, int)
        or max_inline_chars < 0
    ):
        raise PiExperienceError("max_inline_chars must be an integer >= 0")


def _expected_counts(trace: PiTrace) -> tuple[Counter[str], Counter[str]]:
    mapped: Counter[str] = Counter()
    ignored: Counter[str] = Counter()
    for event in trace.events:
        projection = _projection(event)
        if projection is None:
            ignored[event.type] += 1
        else:
            mapped[projection.kind.value] += 1
    return mapped, ignored


def ingest_pi_trace(
    trace_or_path: PiTrace | str | Path,
    store: AgentExperienceStore,
    *,
    session_id: str | None = None,
    source: str = "pi",
    max_inline_chars: int = 262_144,
    reject_existing: bool = True,
) -> PiIngestResult:
    """Import one completed Pi capture, preserving its exact bytes as a blob.

    Existing session ids fail closed.  With ``reject_existing=False``, an
    exactly matching, fully imported capture is returned idempotently; a
    partial or different import still fails.
    """

    _validate_options(source, max_inline_chars)
    if not isinstance(store, AgentExperienceStore):
        raise TypeError("store must be AgentExperienceStore")
    if isinstance(trace_or_path, PiTrace):
        trace = trace_or_path
        raw = _trace_bytes(trace)
        capture_name = "pi-capture.jsonl"
    else:
        path = Path(trace_or_path)
        trace = read_pi_trace(path)
        raw = path.read_bytes()
        capture_name = path.name
    trace_sha256 = hashlib.sha256(raw).hexdigest()
    if trace_sha256 != trace.summary.sha256:
        raise PiExperienceError("capture bytes do not match PiTrace summary")
    observed_id = _observed_session_id(trace)
    if session_id is not None and observed_id is not None and session_id != observed_id:
        raise PiExperienceError("session_id override disagrees with Pi session event")
    resolved_id = _checked_session_id(session_id or observed_id, trace_sha256)
    mapped_counts, ignored_counts = _expected_counts(trace)
    mapped_total = sum(mapped_counts.values())

    existing = store.events(resolved_id)
    if existing:
        import_event = next(
            (event for event in existing if event.kind is EventKind.SESSION_START),
            None,
        )
        import_payload = (
            None if import_event is None else import_event.to_dict()["payload"]
        )
        exact = (
            len(existing) == mapped_total
            and isinstance(import_payload, dict)
            and import_payload.get("pi_trace_sha256") == trace_sha256
            and import_payload.get("pi_raw_event_count") == len(trace.events)
        )
        if reject_existing or not exact:
            raise PiExperienceError(
                f"experience session {resolved_id!r} already exists"
            )
        capture_blob = (
            import_event.blobs[0]
            if import_event is not None and import_event.blobs
            else None
        )
        return PiIngestResult(
            resolved_id,
            len(trace.events),
            mapped_total,
            dict(sorted(mapped_counts.items())),
            dict(sorted(ignored_counts.items())),
            capture_blob,
            trace_sha256,
            True,
        )

    capture_blob = store.put_blob(
        raw,
        media_type="application/x-ndjson; charset=utf-8",
        name=capture_name,
    )
    base_ns = _base_timestamp_ns(trace.events)
    for event in trace.events:
        projected = _projection(event)
        if projected is None:
            continue
        payload, event_blobs = _payload_and_blobs(
            event,
            store,
            source=source,
            max_inline_chars=max_inline_chars,
        )
        blobs = event_blobs
        if projected.kind is EventKind.SESSION_START:
            payload["pi_trace_sha256"] = trace_sha256
            payload["pi_raw_event_count"] = len(trace.events)
            payload["pi_mapped_event_count"] = mapped_total
            blobs = (capture_blob, *event_blobs)
        store.append(
            session_id=resolved_id,
            actor=projected.actor,
            kind=projected.kind,
            payload=payload,
            timestamp_ns=_event_timestamp_ns(event, base_ns),
            monotonic_ns=event.line_number,
            model_id=projected.model_id,
            blobs=blobs,
        )
    return PiIngestResult(
        resolved_id,
        len(trace.events),
        mapped_total,
        dict(sorted(mapped_counts.items())),
        dict(sorted(ignored_counts.items())),
        capture_blob,
        trace_sha256,
    )


class PiExperienceSink:
    """Synchronous ``run_pi_captured(on_event=...)`` experience sink."""

    def __init__(
        self,
        store: AgentExperienceStore,
        *,
        session_id: str | None = None,
        source: str = "pi",
        max_inline_chars: int = 262_144,
    ) -> None:
        _validate_options(source, max_inline_chars)
        if not isinstance(store, AgentExperienceStore):
            raise TypeError("store must be AgentExperienceStore")
        self.store = store
        self.requested_session_id = session_id
        self.source = source
        self.max_inline_chars = max_inline_chars
        self.session_id: str | None = None
        self._base_ns = 0
        self._pending: list[RawPiEvent] = []
        self._raw = bytearray()
        self._raw_count = 0
        self._mapped: Counter[str] = Counter()
        self._ignored: Counter[str] = Counter()
        self._capture_blob: BlobRef | None = None
        self._final_result: PiIngestResult | None = None

    def __call__(self, event: RawPiEvent) -> None:
        if self._final_result is not None:
            raise PiExperienceError("sink is already finalized")
        if not isinstance(event, RawPiEvent):
            raise TypeError("PiExperienceSink accepts RawPiEvent instances")
        self._raw.extend(event.raw.encode("utf-8"))
        self._raw_count += 1
        if self.session_id is None:
            self._pending.append(event)
            if event.type != "session":
                return
            observed = event.parsed.get("id")
            if not isinstance(observed, str):
                observed = None
            if (
                self.requested_session_id is not None
                and observed is not None
                and self.requested_session_id != observed
            ):
                raise PiExperienceError(
                    "session_id override disagrees with Pi session event"
                )
            self.session_id = _checked_session_id(
                self.requested_session_id or observed,
                hashlib.sha256(bytes(self._raw)).hexdigest(),
            )
            if self.store.events(self.session_id):
                raise PiExperienceError(
                    f"experience session {self.session_id!r} already exists"
                )
            self._base_ns = _event_timestamp_ns(event, 0)
            pending, self._pending = self._pending, []
            for item in pending:
                self._append_event(item)
            return
        if event.type == "session":
            raise PiExperienceError("live stream contains a second session event")
        self._append_event(event)

    def _append_event(self, event: RawPiEvent) -> None:
        projected = _projection(event)
        if projected is None:
            self._ignored[event.type] += 1
            return
        assert self.session_id is not None
        payload, blobs = _payload_and_blobs(
            event,
            self.store,
            source=self.source,
            max_inline_chars=self.max_inline_chars,
        )
        self.store.append(
            session_id=self.session_id,
            actor=projected.actor,
            kind=projected.kind,
            payload=payload,
            timestamp_ns=_event_timestamp_ns(event, self._base_ns),
            monotonic_ns=event.line_number,
            model_id=projected.model_id,
            blobs=blobs,
        )
        self._mapped[projected.kind.value] += 1

    def finalize(self, *, raw_capture_path: str | Path | None = None) -> PiIngestResult:
        """Seal accounting and store the exact live capture as one blob."""

        if self._final_result is not None:
            return self._final_result
        if self.session_id is None:
            raise PiExperienceError("live stream ended before a Pi session event")
        raw = (
            Path(raw_capture_path).read_bytes()
            if raw_capture_path is not None
            else bytes(self._raw)
        )
        if raw_capture_path is not None and not raw.startswith(bytes(self._raw)):
            raise PiExperienceError("capture path does not match delivered live events")
        self._capture_blob = self.store.put_blob(
            raw,
            media_type="application/x-ndjson; charset=utf-8",
            name=(
                Path(raw_capture_path).name
                if raw_capture_path is not None
                else "pi-live-capture.jsonl"
            ),
        )
        self._final_result = PiIngestResult(
            self.session_id,
            self._raw_count,
            sum(self._mapped.values()),
            dict(sorted(self._mapped.items())),
            dict(sorted(self._ignored.items())),
            self._capture_blob,
            hashlib.sha256(raw).hexdigest(),
        )
        return self._final_result
