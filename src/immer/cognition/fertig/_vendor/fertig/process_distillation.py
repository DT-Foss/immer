"""Replay-grounded process examples from completed Pi traces.

The data in this module is deliberately descriptive: completed assistant
messages are joined to completed tool executions by Pi's ``toolCallId`` and
kept in source order.  No hidden state, semantic strategy, or whole-model
property is inferred from the resulting corpus.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import Path
from statistics import median, pstdev
from types import MappingProxyType
from typing import Any

from fertig.pi_trace import PiToolExecution, PiTrace, read_pi_trace


SCHEMA = "fertig.process-distillation.v1"


class ProcessDistillationError(ValueError):
    """Observed Pi content cannot be represented without dropping data."""


class TrainingArm(str, Enum):
    """Neutral projections used for future controlled training comparisons."""

    ANSWER_ONLY = "answer_only"
    RAW_COT = "raw_cot"
    TOOL_SEQUENCE = "tool_sequence"
    PROCESS_PAIR = "process_pair"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _blocks(value: Any, *, location: str) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProcessDistillationError(f"{location} must be an array")
    blocks: list[Mapping[str, Any]] = []
    for index, block in enumerate(value):
        if not isinstance(block, Mapping):
            raise ProcessDistillationError(f"{location}[{index}] must be an object")
        blocks.append(_freeze(block))
    return tuple(blocks)


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return _freeze(value) if isinstance(value, Mapping) else None


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


@dataclass(frozen=True, slots=True)
class ObservedInputMessage:
    """One completed user message retained as literal task input."""

    line_number: int
    timestamp_ms: int | float | None
    content: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "line_number": self.line_number,
            "timestamp_ms": self.timestamp_ms,
            "content": _thaw(self.content),
        }


@dataclass(frozen=True, slots=True)
class ToolObservation:
    """One assistant tool call joined to its observed Pi execution and result."""

    turn_index: int
    content_index: int
    tool_call_id: str
    requested_name: str | None
    requested_arguments: Mapping[str, Any] | None
    execution_name: str | None
    execution_arguments: Mapping[str, Any] | None
    result: Mapping[str, Any] | None
    is_error: bool | None
    start_line_number: int | None
    end_line_number: int | None
    result_message_line_number: int | None
    result_message_content: tuple[Mapping[str, Any], ...]
    result_message_is_error: bool | None

    @property
    def paired(self) -> bool:
        return self.start_line_number is not None and self.end_line_number is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "content_index": self.content_index,
            "tool_call_id": self.tool_call_id,
            "requested_name": self.requested_name,
            "requested_arguments": _thaw(self.requested_arguments),
            "execution_name": self.execution_name,
            "execution_arguments": _thaw(self.execution_arguments),
            "result": _thaw(self.result),
            "is_error": self.is_error,
            "paired": self.paired,
            "start_line_number": self.start_line_number,
            "end_line_number": self.end_line_number,
            "result_message_line_number": self.result_message_line_number,
            "result_message_content": _thaw(self.result_message_content),
            "result_message_is_error": self.result_message_is_error,
        }


@dataclass(frozen=True, slots=True)
class TeacherTurn:
    """One immutable completed assistant message with paired tool evidence."""

    index: int
    line_number: int
    timestamp_ms: int | float | None
    provider: str | None
    model: str | None
    stop_reason: str | None
    content: tuple[Mapping[str, Any], ...]
    thinking: tuple[str, ...]
    text: tuple[str, ...]
    tools: tuple[ToolObservation, ...]

    @property
    def thinking_chars(self) -> int:
        return sum(len(value) for value in self.thinking)

    @property
    def text_chars(self) -> int:
        return sum(len(value) for value in self.text)

    @property
    def immediate_tool_errors(self) -> tuple[ToolObservation, ...]:
        return tuple(tool for tool in self.tools if tool.is_error is True)

    @property
    def is_observed_terminal(self) -> bool:
        return self.stop_reason == "stop"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "line_number": self.line_number,
            "timestamp_ms": self.timestamp_ms,
            "provider": self.provider,
            "model": self.model,
            "stop_reason": self.stop_reason,
            "content": _thaw(self.content),
            "thinking": list(self.thinking),
            "text": list(self.text),
            "tools": [tool.to_dict() for tool in self.tools],
        }


@dataclass(frozen=True, slots=True)
class RecoverySpan:
    """Literal errored call followed by a later successful paired call.

    Temporal order is observed.  Causal repair and same-state comparability are
    intentionally not claimed.
    """

    index: int
    error_turn_index: int
    error_tool_index: int
    recovery_turn_index: int
    recovery_tool_index: int
    intervening_turn_indices: tuple[int, ...]

    @property
    def provenance(self) -> str:
        return "literal_error_then_later_success"

    @property
    def observationally_comparable(self) -> bool:
        return False

    @property
    def causal_claim(self) -> bool:
        return False

    def to_dict(self, turns: Sequence[TeacherTurn]) -> dict[str, Any]:
        error = turns[self.error_turn_index].tools[self.error_tool_index]
        recovery = turns[self.recovery_turn_index].tools[self.recovery_tool_index]
        return {
            "index": self.index,
            "error_turn_index": self.error_turn_index,
            "error_tool_index": self.error_tool_index,
            "error_tool_call_id": error.tool_call_id,
            "recovery_turn_index": self.recovery_turn_index,
            "recovery_tool_index": self.recovery_tool_index,
            "recovery_tool_call_id": recovery.tool_call_id,
            "intervening_turn_indices": list(self.intervening_turn_indices),
            "provenance": self.provenance,
            "observationally_comparable": self.observationally_comparable,
            "causal_claim": self.causal_claim,
        }


@dataclass(frozen=True, slots=True)
class TerminalEvidence:
    """Successful tool results immediately before an observed terminal stop.

    This is a structural evidence boundary only; it does not infer that a tool
    semantically verified the task.
    """

    terminal_turn_index: int
    evidence_turn_index: int
    tool_indices: tuple[int, ...]

    @property
    def provenance(self) -> str:
        return "successful_tools_in_immediately_preceding_turn_before_stop"

    @property
    def semantic_verification_claim(self) -> bool:
        return False

    def tools(self, turns: Sequence[TeacherTurn]) -> tuple[ToolObservation, ...]:
        turn = turns[self.evidence_turn_index]
        return tuple(turn.tools[index] for index in self.tool_indices)

    def to_dict(self, turns: Sequence[TeacherTurn]) -> dict[str, Any]:
        return {
            "terminal_turn_index": self.terminal_turn_index,
            "evidence_turn_index": self.evidence_turn_index,
            "tool_indices": list(self.tool_indices),
            "tools": [tool.to_dict() for tool in self.tools(turns)],
            "provenance": self.provenance,
            "semantic_verification_claim": self.semantic_verification_claim,
        }


@dataclass(frozen=True, slots=True)
class ProcessCorpusStats:
    assistant_turns: int
    paired_tools: int
    unpaired_tools: int
    tool_errors: int
    recovery_spans: int
    unresolved_tool_errors: int
    observed_terminal_turns: int
    terminal_evidence_tools: int
    thinking_chars: int
    text_chars: int
    turns_with_thinking: int
    max_thinking_chars: int
    top_two_thinking_chars: int
    top_two_thinking_share: float
    mean_thinking_chars: float
    median_thinking_chars: float
    thinking_cv: float

    @property
    def thinking_burstiness(self) -> float:
        """Share of all observed thinking characters in the two largest turns."""

        return self.top_two_thinking_share

    def to_dict(self) -> dict[str, Any]:
        return {
            "assistant_turns": self.assistant_turns,
            "paired_tools": self.paired_tools,
            "unpaired_tools": self.unpaired_tools,
            "tool_errors": self.tool_errors,
            "recovery_spans": self.recovery_spans,
            "unresolved_tool_errors": self.unresolved_tool_errors,
            "observed_terminal_turns": self.observed_terminal_turns,
            "terminal_evidence_tools": self.terminal_evidence_tools,
            "thinking_chars": self.thinking_chars,
            "text_chars": self.text_chars,
            "turns_with_thinking": self.turns_with_thinking,
            "max_thinking_chars": self.max_thinking_chars,
            "top_two_thinking_chars": self.top_two_thinking_chars,
            "top_two_thinking_share": self.top_two_thinking_share,
            "thinking_burstiness": self.thinking_burstiness,
            "mean_thinking_chars": self.mean_thinking_chars,
            "median_thinking_chars": self.median_thinking_chars,
            "thinking_cv": self.thinking_cv,
        }


@dataclass(frozen=True, slots=True)
class ProcessCorpus:
    """Immutable process corpus derived from one lossless Pi trace."""

    trace_sha256: str
    session_id: str | None
    provider: str | None
    model: str | None
    inputs: tuple[ObservedInputMessage, ...]
    turns: tuple[TeacherTurn, ...]
    recoveries: tuple[RecoverySpan, ...]
    terminal_evidence: tuple[TerminalEvidence, ...]
    stats: ProcessCorpusStats

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "scope": "observed_process_from_one_pi_trace",
            "trace_sha256": self.trace_sha256,
            "session_id": self.session_id,
            "provider": self.provider,
            "model": self.model,
            "inputs": [message.to_dict() for message in self.inputs],
            "turns": [turn.to_dict() for turn in self.turns],
            "recoveries": [span.to_dict(self.turns) for span in self.recoveries],
            "terminal_evidence": [
                evidence.to_dict(self.turns) for evidence in self.terminal_evidence
            ],
            "stats": self.stats.to_dict(),
        }


def _timestamp(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _result_messages(trace: PiTrace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for completed in trace.completed_messages:
        if completed.role != "toolResult":
            continue
        tool_call_id = completed.message.get("toolCallId")
        if isinstance(tool_call_id, str) and tool_call_id not in result:
            result[tool_call_id] = completed
    return result


def _tool_observation(
    *,
    turn_index: int,
    content_index: int,
    block: Mapping[str, Any],
    execution: PiToolExecution | None,
    result_message: Any,
) -> ToolObservation:
    tool_call_id = block.get("id")
    if not isinstance(tool_call_id, str) or not tool_call_id:
        raise ProcessDistillationError(
            "assistant toolCall.id must be a non-empty string"
        )
    message_content: tuple[Mapping[str, Any], ...] = ()
    message_line: int | None = None
    message_error: bool | None = None
    if result_message is not None:
        message_content = _blocks(
            result_message.message.get("content"),
            location=f"toolResult[{tool_call_id}].content",
        )
        message_line = result_message.event.line_number
        observed_error = result_message.message.get("isError")
        message_error = observed_error if isinstance(observed_error, bool) else None
    start = execution.start if execution is not None else None
    end = execution.end if execution is not None else None
    return ToolObservation(
        turn_index=turn_index,
        content_index=content_index,
        tool_call_id=tool_call_id,
        requested_name=_string(block.get("name")),
        requested_arguments=_mapping(block.get("arguments")),
        execution_name=execution.tool_name if execution is not None else None,
        execution_arguments=(
            _mapping(execution.args) if execution is not None else None
        ),
        result=_mapping(execution.result) if execution is not None else None,
        is_error=execution.is_error if execution is not None else None,
        start_line_number=start.line_number if start is not None else None,
        end_line_number=end.line_number if end is not None else None,
        result_message_line_number=message_line,
        result_message_content=message_content,
        result_message_is_error=message_error,
    )


def _teacher_turns(trace: PiTrace) -> tuple[TeacherTurn, ...]:
    executions = {
        execution.tool_call_id: execution for execution in trace.tool_executions
    }
    result_messages = _result_messages(trace)
    seen_tool_calls: set[str] = set()
    turns: list[TeacherTurn] = []
    for completed in trace.completed_messages:
        if completed.role != "assistant":
            continue
        turn_index = len(turns)
        content = _blocks(
            completed.message.get("content"),
            location=f"assistant[{turn_index}].content",
        )
        thinking: list[str] = []
        text: list[str] = []
        tools: list[ToolObservation] = []
        for content_index, block in enumerate(content):
            block_type = block.get("type")
            if block_type == "thinking" and isinstance(block.get("thinking"), str):
                thinking.append(block["thinking"])
            elif block_type == "text" and isinstance(block.get("text"), str):
                text.append(block["text"])
            elif block_type == "toolCall":
                tool_call_id = block.get("id")
                if not isinstance(tool_call_id, str) or not tool_call_id:
                    raise ProcessDistillationError(
                        "assistant toolCall.id must be a non-empty string"
                    )
                if tool_call_id in seen_tool_calls:
                    raise ProcessDistillationError(
                        f"duplicate assistant toolCall.id {tool_call_id!r}"
                    )
                seen_tool_calls.add(tool_call_id)
                tools.append(
                    _tool_observation(
                        turn_index=turn_index,
                        content_index=content_index,
                        block=block,
                        execution=executions.get(tool_call_id),
                        result_message=result_messages.get(tool_call_id),
                    )
                )
        turns.append(
            TeacherTurn(
                index=turn_index,
                line_number=completed.event.line_number,
                timestamp_ms=completed.timestamp_ms,
                provider=completed.provider,
                model=completed.model,
                stop_reason=_string(completed.message.get("stopReason")),
                content=content,
                thinking=tuple(thinking),
                text=tuple(text),
                tools=tuple(tools),
            )
        )
    return tuple(turns)


def _input_messages(trace: PiTrace) -> tuple[ObservedInputMessage, ...]:
    messages: list[ObservedInputMessage] = []
    for completed in trace.completed_messages:
        if completed.role != "user":
            continue
        messages.append(
            ObservedInputMessage(
                line_number=completed.event.line_number,
                timestamp_ms=completed.timestamp_ms,
                content=_blocks(
                    completed.message.get("content"), location="user.content"
                ),
            )
        )
    return tuple(messages)


def _recovery_spans(turns: Sequence[TeacherTurn]) -> tuple[RecoverySpan, ...]:
    spans: list[RecoverySpan] = []
    for turn in turns:
        for error_tool_index, error in enumerate(turn.tools):
            if error.is_error is not True:
                continue
            recovered: tuple[int, int] | None = None
            for later_turn in turns[turn.index + 1 :]:
                for tool_index, tool in enumerate(later_turn.tools):
                    if tool.paired and tool.is_error is False:
                        recovered = (later_turn.index, tool_index)
                        break
                if recovered is not None:
                    break
            if recovered is None:
                continue
            recovery_turn_index, recovery_tool_index = recovered
            spans.append(
                RecoverySpan(
                    index=len(spans),
                    error_turn_index=turn.index,
                    error_tool_index=error_tool_index,
                    recovery_turn_index=recovery_turn_index,
                    recovery_tool_index=recovery_tool_index,
                    intervening_turn_indices=tuple(
                        range(turn.index + 1, recovery_turn_index)
                    ),
                )
            )
    return tuple(spans)


def _terminal_evidence(turns: Sequence[TeacherTurn]) -> tuple[TerminalEvidence, ...]:
    evidence: list[TerminalEvidence] = []
    for terminal in turns:
        if not terminal.is_observed_terminal or terminal.index == 0:
            continue
        preceding = turns[terminal.index - 1]
        successful = tuple(
            index
            for index, tool in enumerate(preceding.tools)
            if tool.paired and tool.is_error is False
        )
        if successful:
            evidence.append(
                TerminalEvidence(
                    terminal_turn_index=terminal.index,
                    evidence_turn_index=preceding.index,
                    tool_indices=successful,
                )
            )
    return tuple(evidence)


def _stats(
    turns: Sequence[TeacherTurn],
    recoveries: Sequence[RecoverySpan],
    terminal_evidence: Sequence[TerminalEvidence],
) -> ProcessCorpusStats:
    tools = tuple(tool for turn in turns for tool in turn.tools)
    thinking_lengths = [turn.thinking_chars for turn in turns]
    thinking_total = sum(thinking_lengths)
    largest = sorted(thinking_lengths, reverse=True)[:2]
    top_two = sum(largest)
    mean = thinking_total / len(thinking_lengths) if thinking_lengths else 0.0
    error_count = sum(tool.is_error is True for tool in tools)
    return ProcessCorpusStats(
        assistant_turns=len(turns),
        paired_tools=sum(tool.paired for tool in tools),
        unpaired_tools=sum(not tool.paired for tool in tools),
        tool_errors=error_count,
        recovery_spans=len(recoveries),
        unresolved_tool_errors=max(0, error_count - len(recoveries)),
        observed_terminal_turns=sum(turn.is_observed_terminal for turn in turns),
        terminal_evidence_tools=sum(
            len(item.tool_indices) for item in terminal_evidence
        ),
        thinking_chars=thinking_total,
        text_chars=sum(turn.text_chars for turn in turns),
        turns_with_thinking=sum(length > 0 for length in thinking_lengths),
        max_thinking_chars=max(thinking_lengths, default=0),
        top_two_thinking_chars=top_two,
        top_two_thinking_share=top_two / thinking_total if thinking_total else 0.0,
        mean_thinking_chars=mean,
        median_thinking_chars=(
            float(median(thinking_lengths)) if thinking_lengths else 0.0
        ),
        thinking_cv=(
            float(pstdev(thinking_lengths) / mean)
            if len(thinking_lengths) > 1 and not math.isclose(mean, 0.0)
            else 0.0
        ),
    )


def build_process_corpus(source: PiTrace | str | Path) -> ProcessCorpus:
    """Build an immutable, replay-grounded corpus from one Pi trace."""

    trace = source if isinstance(source, PiTrace) else read_pi_trace(source)
    turns = _teacher_turns(trace)
    recoveries = _recovery_spans(turns)
    terminal_evidence = _terminal_evidence(turns)
    return ProcessCorpus(
        trace_sha256=trace.summary.sha256,
        session_id=trace.summary.session_id,
        provider=trace.summary.provider,
        model=trace.summary.model,
        inputs=_input_messages(trace),
        turns=turns,
        recoveries=recoveries,
        terminal_evidence=terminal_evidence,
        stats=_stats(turns, recoveries, terminal_evidence),
    )


def _provenance(corpus: ProcessCorpus) -> dict[str, Any]:
    return {
        "trace_sha256": corpus.trace_sha256,
        "session_id": corpus.session_id,
        "provider": corpus.provider,
        "model": corpus.model,
        "observed_only": True,
    }


def _turn_process(turn: TeacherTurn) -> dict[str, Any]:
    return {
        "turn_index": turn.index,
        "line_number": turn.line_number,
        "stop_reason": turn.stop_reason,
        "content": _thaw(turn.content),
        "tools": [tool.to_dict() for tool in turn.tools],
    }


def _answer_records(corpus: ProcessCorpus) -> Iterator[dict[str, Any]]:
    evidence_by_terminal = {
        item.terminal_turn_index: item for item in corpus.terminal_evidence
    }
    for turn in corpus.turns:
        text_blocks = [
            _thaw(block) for block in turn.content if block.get("type") == "text"
        ]
        if not turn.is_observed_terminal or not text_blocks:
            continue
        evidence = evidence_by_terminal.get(turn.index)
        yield {
            "schema": SCHEMA,
            "record_id": f"{corpus.trace_sha256}:answer_only:{turn.index}",
            "arm": TrainingArm.ANSWER_ONLY.value,
            "provenance": _provenance(corpus),
            "input": {"messages": [message.to_dict() for message in corpus.inputs]},
            "target": {
                "assistant_turn_index": turn.index,
                "text": text_blocks,
            },
            "observed_terminal_evidence": (
                evidence.to_dict(corpus.turns) if evidence is not None else None
            ),
        }


def _raw_cot_record(corpus: ProcessCorpus) -> dict[str, Any] | None:
    if not corpus.turns:
        return None
    turns = []
    for turn in corpus.turns:
        content = [
            _thaw(block)
            for block in turn.content
            if block.get("type") in {"thinking", "text"}
        ]
        turns.append(
            {
                "turn_index": turn.index,
                "line_number": turn.line_number,
                "stop_reason": turn.stop_reason,
                "content": content,
            }
        )
    return {
        "schema": SCHEMA,
        "record_id": f"{corpus.trace_sha256}:raw_cot:0",
        "arm": TrainingArm.RAW_COT.value,
        "provenance": _provenance(corpus),
        "input": {"messages": [message.to_dict() for message in corpus.inputs]},
        "target": {"assistant_turns": turns},
    }


def _tool_sequence_record(corpus: ProcessCorpus) -> dict[str, Any] | None:
    tools = [tool.to_dict() for turn in corpus.turns for tool in turn.tools]
    if not tools:
        return None
    return {
        "schema": SCHEMA,
        "record_id": f"{corpus.trace_sha256}:tool_sequence:0",
        "arm": TrainingArm.TOOL_SEQUENCE.value,
        "provenance": _provenance(corpus),
        "input": {"messages": [message.to_dict() for message in corpus.inputs]},
        "target": {"tools": tools},
    }


def _process_pair_records(corpus: ProcessCorpus) -> Iterator[dict[str, Any]]:
    for span in corpus.recoveries:
        error_turn = corpus.turns[span.error_turn_index]
        recovery_turn = corpus.turns[span.recovery_turn_index]
        span_provenance = span.to_dict(corpus.turns)
        yield {
            "schema": SCHEMA,
            "record_id": f"{corpus.trace_sha256}:process_pair:{span.index}",
            "arm": TrainingArm.PROCESS_PAIR.value,
            "provenance": {**_provenance(corpus), **span_provenance},
            "input": {
                "messages": [message.to_dict() for message in corpus.inputs],
                "observed_error_result": error_turn.tools[
                    span.error_tool_index
                ].to_dict(),
            },
            "negative": _turn_process(error_turn),
            "positive": _turn_process(recovery_turn),
            "preference_claim": False,
        }


def _normalize_arms(
    arms: Iterable[TrainingArm | str] | None,
) -> tuple[TrainingArm, ...]:
    if arms is None:
        return tuple(TrainingArm)
    normalized: list[TrainingArm] = []
    for value in arms:
        try:
            arm = value if isinstance(value, TrainingArm) else TrainingArm(value)
        except ValueError as exc:
            raise ValueError(f"unknown training arm {value!r}") from exc
        if arm not in normalized:
            normalized.append(arm)
    return tuple(normalized)


def iter_training_records(
    corpus: ProcessCorpus,
    *,
    arms: Iterable[TrainingArm | str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield deterministic JSON-compatible records for selected comparisons."""

    for arm in _normalize_arms(arms):
        if arm is TrainingArm.ANSWER_ONLY:
            yield from _answer_records(corpus)
        elif arm is TrainingArm.RAW_COT:
            record = _raw_cot_record(corpus)
            if record is not None:
                yield record
        elif arm is TrainingArm.TOOL_SEQUENCE:
            record = _tool_sequence_record(corpus)
            if record is not None:
                yield record
        else:
            yield from _process_pair_records(corpus)


def training_jsonl(
    corpus: ProcessCorpus,
    *,
    arms: Iterable[TrainingArm | str] | None = None,
) -> str:
    """Serialize selected records as deterministic architecture-neutral JSONL."""

    return "".join(
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for record in iter_training_records(corpus, arms=arms)
    )


def write_training_jsonl(
    corpus: ProcessCorpus,
    path: str | Path,
    *,
    arms: Iterable[TrainingArm | str] | None = None,
) -> Path:
    """Write deterministic training records and return the destination path."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(training_jsonl(corpus, arms=arms), encoding="utf-8")
    return destination


def read_training_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Strictly read records produced by :func:`write_training_jsonl`."""

    source = Path(path)
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProcessDistillationError(
                    f"{source}:{line_number}: invalid JSON"
                ) from exc
            if not isinstance(record, dict):
                raise ProcessDistillationError(
                    f"{source}:{line_number}: record must be an object"
                )
            if record.get("schema") != SCHEMA:
                raise ProcessDistillationError(
                    f"{source}:{line_number}: unsupported schema"
                )
            try:
                TrainingArm(record.get("arm"))
            except ValueError as exc:
                raise ProcessDistillationError(
                    f"{source}:{line_number}: unknown training arm"
                ) from exc
            records.append(record)
    return tuple(records)
