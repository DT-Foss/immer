"""Lossless reader and capture helper for Pi's observed JSON event stream.

This module deliberately stays provider-specific and descriptive.  It retains
every complete JSON line exactly as emitted, projects only Pi's completed
``message_end`` records, and joins tool lifecycle records by ``toolCallId``.
It does not infer goals, outcomes, semantic labels, or training examples.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import threading
from typing import Any, BinaryIO


class PiTraceError(ValueError):
    """A complete Pi JSONL record is malformed or violates the observed shape."""


@dataclass(frozen=True, slots=True)
class RawPiEvent:
    """One complete Pi stdout record, retaining its exact source text."""

    line_number: int
    raw: str
    parsed: dict[str, Any]

    @property
    def type(self) -> str:
        """Pi event type from the parsed record."""

        return self.parsed["type"]


@dataclass(frozen=True, slots=True)
class CompletedPiMessage:
    """A message observed at Pi's ``message_end`` boundary."""

    event: RawPiEvent
    message: dict[str, Any]

    @property
    def role(self) -> str | None:
        value = self.message.get("role")
        return value if isinstance(value, str) else None

    @property
    def provider(self) -> str | None:
        value = self.message.get("provider")
        return value if isinstance(value, str) else None

    @property
    def model(self) -> str | None:
        value = self.message.get("model")
        return value if isinstance(value, str) else None

    @property
    def timestamp_ms(self) -> int | float | None:
        value = self.message.get("timestamp")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return value

    @property
    def usage(self) -> Mapping[str, Any] | None:
        value = self.message.get("usage")
        return value if isinstance(value, Mapping) else None


@dataclass(frozen=True, slots=True)
class PiToolExecution:
    """Pi tool lifecycle records joined by their provider-issued call ID."""

    tool_call_id: str
    tool_name: str | None
    start: RawPiEvent | None
    updates: tuple[RawPiEvent, ...]
    end: RawPiEvent | None

    @property
    def args(self) -> Mapping[str, Any] | None:
        if self.start is None:
            return None
        value = self.start.parsed.get("args")
        return value if isinstance(value, Mapping) else None

    @property
    def result(self) -> Mapping[str, Any] | None:
        if self.end is None:
            return None
        value = self.end.parsed.get("result")
        return value if isinstance(value, Mapping) else None

    @property
    def is_error(self) -> bool | None:
        if self.end is None:
            return None
        value = self.end.parsed.get("isError")
        return value if isinstance(value, bool) else None

    @property
    def completed(self) -> bool:
        return self.start is not None and self.end is not None


@dataclass(frozen=True, slots=True)
class PiCompletedTrace:
    """Deterministic subset containing only completed Pi content.

    ``events`` stay in source order and retain their original JSONL text.  The
    subset contains every ``message_end`` plus the start and end records of
    every fully paired tool execution.  Streaming delta/update records are not
    included.
    """

    events: tuple[RawPiEvent, ...]
    messages: tuple[CompletedPiMessage, ...]
    tool_executions: tuple[PiToolExecution, ...]

    def to_jsonl(self) -> str:
        """Return the exact projected source records in deterministic order."""

        return "".join(event.raw for event in self.events)

    def to_bytes(self) -> bytes:
        return self.to_jsonl().encode("utf-8")

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())


@dataclass(frozen=True, slots=True)
class PiTraceSummary:
    """Compact summary derived only from fields present in the Pi stream."""

    event_count: int
    event_types: dict[str, int]
    completed_message_count: int
    message_roles: dict[str, int]
    tool_execution_count: int
    completed_tool_execution_count: int
    unpaired_tool_execution_count: int
    tool_pairing_fidelity: float
    duration_seconds: float | None
    usage: dict[str, Any]
    tools: dict[str, int]
    errors: int
    error_tool_call_ids: tuple[str, ...]
    session_id: str | None
    session_version: int | None
    provider: str | None
    model: str | None
    cwd: str | None
    started_at: str | None
    sha256: str
    raw_bytes: int
    projected_bytes: int
    projection_ratio: float

    @property
    def counts(self) -> dict[str, Any]:
        """All compact count groups in one serialization-friendly mapping."""

        return {
            "events": self.event_count,
            "event_types": dict(self.event_types),
            "completed_messages": self.completed_message_count,
            "message_roles": dict(self.message_roles),
            "tool_executions": self.tool_execution_count,
            "completed_tool_executions": self.completed_tool_execution_count,
            "unpaired_tool_executions": self.unpaired_tool_execution_count,
        }

    @property
    def hash(self) -> str:
        """Alias naming the SHA-256 over the exact source bytes."""

        return self.sha256

    @property
    def trace_hash(self) -> str:
        return self.sha256

    @property
    def error_count(self) -> int:
        return self.errors

    @property
    def size_bytes(self) -> int:
        return self.raw_bytes

    @property
    def ratio(self) -> float:
        return self.projection_ratio

    def __call__(self) -> PiTraceSummary:
        """Allow both ``trace.summary`` and ``trace.summary()`` at call sites."""

        return self


@dataclass(frozen=True, slots=True)
class PiTrace:
    """Lossless raw events plus Pi's two explicit completed projections."""

    events: tuple[RawPiEvent, ...]
    completed_messages: tuple[CompletedPiMessage, ...]
    tool_executions: tuple[PiToolExecution, ...]
    completed_trace: PiCompletedTrace
    summary: PiTraceSummary
    truncated_tail: bytes | None = None

    @property
    def raw_events(self) -> tuple[RawPiEvent, ...]:
        return self.events

    @property
    def messages(self) -> tuple[CompletedPiMessage, ...]:
        return self.completed_messages

    @property
    def completed(self) -> PiCompletedTrace:
        return self.completed_trace

    @property
    def projection(self) -> PiCompletedTrace:
        return self.completed_trace


@dataclass(frozen=True, slots=True)
class PiCaptureResult:
    """Process result with stderr kept separate from the captured Pi trace."""

    argv: tuple[str, ...]
    raw_path: Path
    returncode: int
    stderr: bytes
    trace: PiTrace

    @property
    def stderr_text(self) -> str:
        return self.stderr.decode("utf-8", errors="replace")


@dataclass(slots=True)
class _ToolBuilder:
    tool_call_id: str
    tool_name: str | None = None
    start: RawPiEvent | None = None
    updates: list[RawPiEvent] | None = None
    end: RawPiEvent | None = None

    def __post_init__(self) -> None:
        if self.updates is None:
            self.updates = []

    def build(self) -> PiToolExecution:
        return PiToolExecution(
            tool_call_id=self.tool_call_id,
            tool_name=self.tool_name,
            start=self.start,
            updates=tuple(self.updates or ()),
            end=self.end,
        )


def _format_error(source: Path, line_number: int, detail: str) -> PiTraceError:
    return PiTraceError(f"{source}: malformed Pi event on line {line_number}: {detail}")


def _parse_complete_line(
    raw_bytes: bytes, line_number: int, source: Path
) -> RawPiEvent:
    try:
        raw = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _format_error(source, line_number, f"invalid UTF-8 ({exc})") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _format_error(
            source,
            line_number,
            f"invalid JSON at column {exc.colno} ({exc.msg})",
        ) from exc
    if not isinstance(parsed, dict):
        raise _format_error(source, line_number, "top-level JSON must be an object")
    if not isinstance(parsed.get("type"), str):
        raise _format_error(source, line_number, "event type must be a string")
    return RawPiEvent(line_number=line_number, raw=raw, parsed=parsed)


def _string_field(value: Mapping[str, Any], name: str) -> str | None:
    field = value.get(name)
    return field if isinstance(field, str) else None


def _int_field(value: Mapping[str, Any], name: str) -> int | None:
    field = value.get(name)
    return field if isinstance(field, int) and not isinstance(field, bool) else None


def _message_from_event(event: RawPiEvent, source: Path) -> CompletedPiMessage:
    message = event.parsed.get("message")
    if not isinstance(message, dict):
        raise _format_error(
            source, event.line_number, "message_end.message must be an object"
        )
    return CompletedPiMessage(event=event, message=message)


def _tool_builder_for(
    event: RawPiEvent,
    builders: dict[str, _ToolBuilder],
    source: Path,
) -> _ToolBuilder:
    tool_call_id = event.parsed.get("toolCallId")
    if not isinstance(tool_call_id, str) or not tool_call_id:
        raise _format_error(
            source,
            event.line_number,
            f"{event.type}.toolCallId must be a non-empty string",
        )
    builder = builders.setdefault(tool_call_id, _ToolBuilder(tool_call_id))
    tool_name = event.parsed.get("toolName")
    if isinstance(tool_name, str):
        builder.tool_name = builder.tool_name or tool_name
    return builder


def _project_events(
    events: tuple[RawPiEvent, ...], source: Path
) -> tuple[tuple[CompletedPiMessage, ...], tuple[PiToolExecution, ...]]:
    messages: list[CompletedPiMessage] = []
    builders: dict[str, _ToolBuilder] = {}

    for event in events:
        if event.type == "message_end":
            messages.append(_message_from_event(event, source))
            continue
        if event.type not in {
            "tool_execution_start",
            "tool_execution_update",
            "tool_execution_end",
        }:
            continue
        builder = _tool_builder_for(event, builders, source)
        if event.type == "tool_execution_start":
            if builder.start is not None:
                raise _format_error(
                    source,
                    event.line_number,
                    f"duplicate tool start for {builder.tool_call_id!r}",
                )
            builder.start = event
        elif event.type == "tool_execution_update":
            assert builder.updates is not None
            builder.updates.append(event)
        else:
            if builder.end is not None:
                raise _format_error(
                    source,
                    event.line_number,
                    f"duplicate tool end for {builder.tool_call_id!r}",
                )
            builder.end = event

    return tuple(messages), tuple(builder.build() for builder in builders.values())


def _completed_projection(
    events: tuple[RawPiEvent, ...],
    messages: tuple[CompletedPiMessage, ...],
    tools: tuple[PiToolExecution, ...],
) -> PiCompletedTrace:
    completed_tools = tuple(tool for tool in tools if tool.completed)
    retained_lines = {message.event.line_number for message in messages}
    for tool in completed_tools:
        assert tool.start is not None and tool.end is not None
        retained_lines.add(tool.start.line_number)
        retained_lines.add(tool.end.line_number)
    retained_events = tuple(
        event for event in events if event.line_number in retained_lines
    )
    return PiCompletedTrace(
        events=retained_events,
        messages=messages,
        tool_executions=completed_tools,
    )


def _add_numeric_tree(total: dict[str, Any], value: Mapping[str, Any]) -> None:
    for key, item in value.items():
        if isinstance(item, Mapping):
            nested = total.setdefault(key, {})
            if isinstance(nested, dict):
                _add_numeric_tree(nested, item)
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            previous = total.get(key, 0)
            if isinstance(previous, (int, float)) and not isinstance(previous, bool):
                total[key] = previous + item


def _iso_timestamp_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _summarize(
    events: tuple[RawPiEvent, ...],
    messages: tuple[CompletedPiMessage, ...],
    tools: tuple[PiToolExecution, ...],
    projection: PiCompletedTrace,
    *,
    sha256: str,
    size_bytes: int,
) -> PiTraceSummary:
    event_types = Counter(event.type for event in events)
    message_roles = Counter(
        message.role for message in messages if message.role is not None
    )
    session = next((event.parsed for event in events if event.type == "session"), {})
    started_at = _string_field(session, "timestamp")
    timestamps = [
        float(message.timestamp_ms) / 1000.0
        for message in messages
        if message.timestamp_ms is not None
    ]
    start_seconds = _iso_timestamp_seconds(started_at)
    if start_seconds is None and timestamps:
        start_seconds = min(timestamps)
    end_seconds = max(timestamps) if timestamps else start_seconds
    duration = (
        None
        if start_seconds is None or end_seconds is None
        else max(0.0, end_seconds - start_seconds)
    )

    usage: dict[str, Any] = {}
    providers: list[str] = []
    models: list[str] = []
    for message in messages:
        if message.usage is not None:
            _add_numeric_tree(usage, message.usage)
        if message.provider is not None:
            providers.append(message.provider)
        if message.model is not None:
            models.append(message.model)

    tool_counts = Counter(
        execution.tool_name
        for execution in tools
        if execution.start is not None and execution.tool_name is not None
    )
    error_ids = tuple(
        execution.tool_call_id for execution in tools if execution.is_error is True
    )
    completed_tools = sum(execution.completed for execution in tools)
    unpaired_tools = len(tools) - completed_tools
    pairing_fidelity = completed_tools / len(tools) if tools else 1.0
    projected_bytes = projection.byte_count
    return PiTraceSummary(
        event_count=len(events),
        event_types=dict(sorted(event_types.items())),
        completed_message_count=len(messages),
        message_roles=dict(sorted(message_roles.items())),
        tool_execution_count=len(tools),
        completed_tool_execution_count=completed_tools,
        unpaired_tool_execution_count=unpaired_tools,
        tool_pairing_fidelity=pairing_fidelity,
        duration_seconds=duration,
        usage=usage,
        tools=dict(sorted(tool_counts.items())),
        errors=len(error_ids),
        error_tool_call_ids=error_ids,
        session_id=_string_field(session, "id"),
        session_version=_int_field(session, "version"),
        provider=providers[0] if providers else None,
        model=models[0] if models else None,
        cwd=_string_field(session, "cwd"),
        started_at=started_at,
        sha256=sha256,
        raw_bytes=size_bytes,
        projected_bytes=projected_bytes,
        projection_ratio=projected_bytes / size_bytes if size_bytes else 0.0,
    )


def read_pi_trace(path: str | Path) -> PiTrace:
    """Read one Pi JSONL capture.

    Every complete line must be valid UTF-8 JSON with a string ``type``.  A
    malformed last line is ignored only when it has no newline, matching the
    observable failure mode of a process interrupted during a write.  Its
    bytes remain covered by ``summary.sha256`` and exposed as
    ``truncated_tail``.
    """

    source = Path(path)
    digest = hashlib.sha256()
    size_bytes = 0
    events: list[RawPiEvent] = []
    truncated_tail: bytes | None = None
    with source.open("rb") as handle:
        for line_number, raw_bytes in enumerate(handle, 1):
            digest.update(raw_bytes)
            size_bytes += len(raw_bytes)
            try:
                event = _parse_complete_line(raw_bytes, line_number, source)
            except PiTraceError:
                if not raw_bytes.endswith(b"\n"):
                    truncated_tail = raw_bytes
                    continue
                raise
            events.append(event)

    raw_events = tuple(events)
    messages, tools = _project_events(raw_events, source)
    projection = _completed_projection(raw_events, messages, tools)
    summary = _summarize(
        raw_events,
        messages,
        tools,
        projection,
        sha256=digest.hexdigest(),
        size_bytes=size_bytes,
    )
    return PiTrace(
        events=raw_events,
        completed_messages=messages,
        tool_executions=tools,
        completed_trace=projection,
        summary=summary,
        truncated_tail=truncated_tail,
    )


def load_pi_trace(path: str | Path) -> PiTrace:
    """Alias for :func:`read_pi_trace`."""

    return read_pi_trace(path)


def iter_pi_events(path: str | Path) -> Iterator[RawPiEvent]:
    """Iterate complete events using the same strictness as the full reader."""

    return iter(read_pi_trace(path).events)


def summarize_pi_trace(path: str | Path | PiTrace) -> PiTraceSummary:
    """Return a compact summary for an already-read trace or a path."""

    trace = path if isinstance(path, PiTrace) else read_pi_trace(path)
    return trace.summary


def _as_bytes(value: bytes | str) -> bytes:
    return value if isinstance(value, bytes) else value.encode("utf-8")


def _read_stderr(stream: BinaryIO | Any, output: list[bytes]) -> None:
    try:
        if stream is not None:
            output.append(_as_bytes(stream.read()))
    except (OSError, ValueError):
        return


def _kill(process: Any) -> None:
    try:
        if process.poll() is None:
            process.kill()
    except (AttributeError, OSError, ProcessLookupError):
        return


def run_pi_captured(
    prompt: str,
    raw_path: str | Path,
    *,
    cwd: str | Path,
    provider: str,
    model: str,
    session_dir: str | Path,
    name: str,
    pi_executable: str | Path = "pi",
    timeout_seconds: float | None = None,
    on_event: Callable[[RawPiEvent], None] | None = None,
    process_factory: Callable[..., Any] = subprocess.Popen,
) -> PiCaptureResult:
    """Run Pi in JSON print mode and stream its stdout into ``raw_path``.

    The prompt and every option are passed as distinct argv elements; no shell
    is involved.  ``on_event`` runs synchronously after each complete line has
    been flushed to disk and therefore observes the live process, not a replay.
    Stderr is drained concurrently and returned separately.
    """

    if timeout_seconds is not None and (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be finite and positive")
    if on_event is not None and not callable(on_event):
        raise TypeError("on_event must be callable or None")

    destination_path = Path(raw_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    session_path = Path(session_dir)
    session_path.mkdir(parents=True, exist_ok=True)
    argv = [
        str(pi_executable),
        "--mode",
        "json",
        "--print",
        "--provider",
        provider,
        "--model",
        model,
        "--session-dir",
        str(session_path),
        "--name",
        name,
        prompt,
    ]
    process = process_factory(
        argv,
        cwd=str(Path(cwd)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    if process.stdout is None:
        _kill(process)
        raise RuntimeError("Pi stdout pipe is unavailable")

    stderr_parts: list[bytes] = []
    stderr_reader = threading.Thread(
        target=_read_stderr,
        args=(process.stderr, stderr_parts),
        name="fertig-pi-stderr",
        daemon=True,
    )
    stderr_reader.start()
    timed_out = threading.Event()

    def expire() -> None:
        timed_out.set()
        _kill(process)

    timer = None
    if timeout_seconds is not None:
        timer = threading.Timer(timeout_seconds, expire)
        timer.daemon = True
        timer.start()

    line_number = 0
    returncode: int | None = None
    try:
        with destination_path.open("wb") as destination:
            for line in process.stdout:
                line_number += 1
                raw_bytes = _as_bytes(line)
                destination.write(raw_bytes)
                destination.flush()
                try:
                    event = _parse_complete_line(
                        raw_bytes, line_number, destination_path
                    )
                except PiTraceError:
                    if not raw_bytes.endswith(b"\n"):
                        continue
                    raise
                if on_event is not None:
                    on_event(event)
        returncode = process.wait()
    except BaseException:
        _kill(process)
        try:
            process.wait()
        except (AttributeError, OSError, ProcessLookupError):
            pass
        raise
    finally:
        if timer is not None:
            timer.cancel()
        stderr_reader.join(timeout=5.0)

    stderr = b"".join(stderr_parts)
    if timed_out.is_set():
        raise subprocess.TimeoutExpired(argv, timeout_seconds, stderr=stderr)
    if returncode is None:
        polled = process.poll()
        returncode = int(polled) if polled is not None else 0
    trace = read_pi_trace(destination_path)
    return PiCaptureResult(
        argv=tuple(argv),
        raw_path=destination_path,
        returncode=int(returncode),
        stderr=stderr,
        trace=trace,
    )
