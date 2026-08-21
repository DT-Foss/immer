"""Empirical, architecture-neutral fingerprints of observable agent traces.

This module learns no task ontology and defines no training recipe.  It reduces
completed host records to the smallest observable alphabet that is useful for
sequence prediction: user context, agent decisions (tool class or finish), and
tool outcomes (ok or error).  A variable-order count model then measures which
local ordering regularities are present in one or more traces.

The model is deliberately modest: a fingerprint learned from one session is a
fingerprint of that session, not a claim about general intelligence.  The
chronological evaluation reports both the full protocol stream and a
decision-only stream, because merely learning ``tool call -> tool result`` is
not evidence of learning an agent's work strategy.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path
import random
from typing import Any
from urllib.parse import quote
import zlib


MODEL_SCHEMA = "fertig.transition-fingerprint"
MODEL_VERSION = 1
UNKNOWN_TOKEN = "<unknown>"
_SERIAL_PREFIX = b"FTFP\x01"
_MAX_SERIALIZED_JSON = 64 * 1024 * 1024
_STEP_KINDS = frozenset({"context", "decision", "outcome"})


def _bounded_name(value: object, fallback: str) -> str:
    if isinstance(value, str) and value:
        return value[:256]
    return fallback


def _event_type(record: Mapping[str, Any]) -> str:
    value = record.get("type", record.get("kind", ""))
    if hasattr(value, "value"):
        value = value.value
    return value if isinstance(value, str) else ""


def _record_mapping(record: object) -> Mapping[str, Any] | None:
    if isinstance(record, Mapping):
        return record
    parsed = getattr(record, "parsed", None)
    if isinstance(parsed, Mapping):
        return parsed
    to_dict = getattr(record, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, Mapping):
            return value
    kind = getattr(record, "kind", None)
    if kind is None:
        return None
    payload = getattr(record, "payload", {})
    return {
        "kind": getattr(kind, "value", kind),
        "actor": getattr(record, "actor", None),
        "payload": payload if isinstance(payload, Mapping) else {},
        "outcome": getattr(record, "outcome", None),
        "span_id": getattr(record, "span_id", None),
    }


def _materialize_records(records: object) -> list[Mapping[str, Any]]:
    events = getattr(records, "events", records)
    if isinstance(events, (str, bytes, Mapping)) or not isinstance(events, Iterable):
        raise TypeError(
            "records must be an iterable of event records or expose .events"
        )
    materialized: list[Mapping[str, Any]] = []
    for record in events:
        mapped = _record_mapping(record)
        if mapped is not None:
            materialized.append(mapped)
    return materialized


def _content_blocks(message: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    content = message.get("content")
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
        return ()
    return tuple(item for item in content if isinstance(item, Mapping))


def _outcome_status(value: object) -> str:
    if isinstance(value, bool):
        return "error" if value else "ok"
    if isinstance(value, str):
        normalized = value.lower()
        if normalized in {"error", "failure", "failed", "aborted"}:
            return "error"
        if normalized in {"ok", "success", "succeeded", "passed"}:
            return "ok"
    status = getattr(value, "status", None)
    if hasattr(status, "value"):
        status = status.value
    if isinstance(status, str):
        return _outcome_status(status)
    return "unknown"


@dataclass(frozen=True, slots=True)
class ObservableStep:
    """One content-free unit directly supported by a completed host record."""

    index: int
    kind: str
    actor: str
    action: str
    outcome: str | None = None
    correlation_id: str | None = None
    source_type: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _STEP_KINDS:
            raise ValueError(f"unknown observable step kind: {self.kind!r}")
        if self.index < 0:
            raise ValueError("step index must be non-negative")
        if not self.actor or not self.action:
            raise ValueError("step actor and action must be non-empty")
        if self.kind == "outcome" and self.outcome not in {
            "ok",
            "error",
            "unknown",
        }:
            raise ValueError("outcome steps require ok, error, or unknown")

    @property
    def symbol(self) -> str:
        """Stable, human-readable token used by the learned count model."""

        pieces = (self.kind, self.actor, self.action)
        encoded = ":".join(quote(piece, safe="._-/") for piece in pieces)
        if self.outcome is not None:
            encoded += ":" + quote(self.outcome, safe="._-/")
        return encoded

    @property
    def is_decision(self) -> bool:
        return self.kind == "decision"


def canonicalize_trace(records: object) -> list[ObservableStep]:
    """Project completed generic/Pi records into observable ordered steps.

    Pi delta events are ignored: their completed ``message_end`` and tool
    lifecycle records retain the relevant decisions and outcomes without
    making tokenization speed a behavioral feature.  Tool calls/results found
    in completed messages are used only when no matching lifecycle record is
    present, preventing double counting while supporting simpler hosts.
    """

    raw = _materialize_records(records)
    started_ids = {
        record.get("toolCallId")
        for record in raw
        if _event_type(record) == "tool_execution_start"
        and isinstance(record.get("toolCallId"), str)
    }
    ended_ids = {
        record.get("toolCallId")
        for record in raw
        if _event_type(record) == "tool_execution_end"
        and isinstance(record.get("toolCallId"), str)
    }

    steps: list[ObservableStep] = []

    def append(
        kind: str,
        actor: str,
        action: str,
        *,
        outcome: str | None = None,
        correlation_id: object = None,
        source_type: str,
    ) -> None:
        steps.append(
            ObservableStep(
                index=len(steps),
                kind=kind,
                actor=actor,
                action=action,
                outcome=outcome,
                correlation_id=(
                    correlation_id if isinstance(correlation_id, str) else None
                ),
                source_type=source_type,
            )
        )

    for record in raw:
        event_type = _event_type(record)
        if event_type == "message_end":
            message = record.get("message")
            if not isinstance(message, Mapping):
                continue
            role = message.get("role")
            if role == "user":
                append("context", "user", "message", source_type=event_type)
                continue
            if role == "toolResult":
                call_id = message.get("toolCallId")
                if call_id in ended_ids:
                    continue
                tool = _bounded_name(message.get("toolName"), "unknown")
                append(
                    "outcome",
                    "tool",
                    f"tool/{tool}",
                    outcome=_outcome_status(message.get("isError")),
                    correlation_id=call_id,
                    source_type=event_type,
                )
                continue
            if role != "assistant":
                continue
            blocks = _content_blocks(message)
            calls = [block for block in blocks if block.get("type") == "toolCall"]
            for call in calls:
                call_id = call.get("id", call.get("toolCallId"))
                if call_id in started_ids:
                    continue
                tool = _bounded_name(call.get("name", call.get("toolName")), "unknown")
                append(
                    "decision",
                    "agent",
                    f"tool/{tool}",
                    correlation_id=call_id,
                    source_type=event_type,
                )
            stop_reason = message.get("stopReason")
            if (
                not calls
                and isinstance(stop_reason, str)
                and stop_reason
                not in {
                    "pending",
                    "toolUse",
                }
            ):
                append(
                    "decision",
                    "agent",
                    f"finish/{stop_reason}",
                    source_type=event_type,
                )
            continue

        if event_type == "tool_execution_start":
            tool = _bounded_name(record.get("toolName"), "unknown")
            append(
                "decision",
                "agent",
                f"tool/{tool}",
                correlation_id=record.get("toolCallId"),
                source_type=event_type,
            )
            continue
        if event_type == "tool_execution_end":
            tool = _bounded_name(record.get("toolName"), "unknown")
            append(
                "outcome",
                "tool",
                f"tool/{tool}",
                outcome=_outcome_status(record.get("isError")),
                correlation_id=record.get("toolCallId"),
                source_type=event_type,
            )
            continue

        # Provider-neutral AgentExperience event names.
        payload = record.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        if event_type == "user_message":
            append("context", "user", "message", source_type=event_type)
        elif event_type in {"tool_call", "action"}:
            tool = _bounded_name(
                payload.get("tool_name", payload.get("tool", payload.get("action"))),
                "unknown",
            )
            append(
                "decision",
                "agent",
                f"tool/{tool}",
                correlation_id=record.get("span_id"),
                source_type=event_type,
            )
        elif event_type in {"tool_result", "test_result", "outcome"}:
            tool = _bounded_name(
                payload.get("tool_name", payload.get("tool", payload.get("action"))),
                "unknown",
            )
            status_value = payload.get("is_error", payload.get("status"))
            if status_value is None:
                status_value = record.get("outcome")
            append(
                "outcome",
                "tool",
                f"tool/{tool}",
                outcome=_outcome_status(status_value),
                correlation_id=record.get("span_id"),
                source_type=event_type,
            )
        elif event_type in {"session_end", "terminal"}:
            append(
                "decision",
                "agent",
                f"finish/{event_type}",
                source_type=event_type,
            )
    return steps


def decision_steps(steps: Iterable[ObservableStep]) -> tuple[ObservableStep, ...]:
    """Return only agent-side decisions, excluding all result symbols."""

    return tuple(step for step in steps if step.is_decision and step.actor == "agent")


def _tokens(sequence: Iterable[ObservableStep | str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in sequence:
        token = value.symbol if isinstance(value, ObservableStep) else value
        if not isinstance(token, str) or not token:
            raise TypeError(
                "model sequences must contain ObservableStep or non-empty str"
            )
        result.append(token)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class SequenceScore:
    """Prequential next-token score; lower log loss and perplexity are better."""

    observations: int
    total_bits: float
    log_loss_bits: float
    perplexity: float
    accuracy: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "observations": self.observations,
            "total_bits": self.total_bits,
            "log_loss_bits": self.log_loss_bits,
            "perplexity": self.perplexity,
            "accuracy": self.accuracy,
        }


class TransitionFingerprint:
    """Interpolated variable-order transition counts over observable symbols."""

    def __init__(
        self,
        *,
        max_order: int,
        alpha: float,
        backoff_strength: float,
        min_count: int,
        vocabulary: Sequence[str],
        counts: Mapping[tuple[str, ...], Mapping[str, int]],
    ) -> None:
        if (
            isinstance(max_order, bool)
            or not isinstance(max_order, int)
            or max_order < 0
        ):
            raise ValueError("max_order must be a non-negative integer")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if not math.isfinite(backoff_strength) or backoff_strength <= 0:
            raise ValueError("backoff_strength must be finite and positive")
        if (
            isinstance(min_count, bool)
            or not isinstance(min_count, int)
            or min_count < 1
        ):
            raise ValueError("min_count must be a positive integer")
        vocab = tuple(sorted(set(vocabulary) | {UNKNOWN_TOKEN}))
        if not vocab:
            raise ValueError("vocabulary must not be empty")
        normalized: dict[tuple[str, ...], Counter[str]] = {}
        for context, values in counts.items():
            context_tuple = tuple(context)
            if len(context_tuple) > max_order:
                raise ValueError("serialized context exceeds max_order")
            counter: Counter[str] = Counter()
            for token, count in values.items():
                if token not in vocab or not isinstance(count, int) or count <= 0:
                    raise ValueError("counts must be positive integers over vocabulary")
                counter[token] = count
            if counter:
                normalized[context_tuple] = counter
        if () not in normalized:
            raise ValueError("counts require an orderless root context")
        self.max_order = max_order
        self.alpha = float(alpha)
        self.backoff_strength = float(backoff_strength)
        self.min_count = min_count
        self.vocabulary = vocab
        self.counts = normalized

    @classmethod
    def fit(
        cls,
        sequence: Iterable[ObservableStep | str],
        *,
        max_order: int = 4,
        alpha: float = 0.5,
        backoff_strength: float = 3.0,
        min_count: int = 2,
    ) -> TransitionFingerprint:
        """Learn every context count up to ``max_order`` from ordered data."""

        tokens = _tokens(sequence)
        if not tokens:
            raise ValueError("cannot fit an empty sequence")
        if (
            isinstance(max_order, bool)
            or not isinstance(max_order, int)
            or max_order < 0
        ):
            raise ValueError("max_order must be a non-negative integer")
        counts: dict[tuple[str, ...], Counter[str]] = {}
        for index, target in enumerate(tokens):
            for order in range(min(max_order, index) + 1):
                context = tokens[index - order : index] if order else ()
                counts.setdefault(context, Counter())[target] += 1
        return cls(
            max_order=max_order,
            alpha=alpha,
            backoff_strength=backoff_strength,
            min_count=min_count,
            vocabulary=tokens,
            counts=counts,
        )

    def _known(self, token: str) -> str:
        return token if token in self.vocabulary else UNKNOWN_TOKEN

    def distribution(self, context: Sequence[str] = ()) -> dict[str, float]:
        """Interpolated posterior predictive distribution for one next symbol."""

        root = self.counts[()]
        denominator = sum(root.values()) + self.alpha * len(self.vocabulary)
        probabilities = {
            token: (root.get(token, 0) + self.alpha) / denominator
            for token in self.vocabulary
        }
        known_context = tuple(self._known(token) for token in context)
        for order in range(1, min(self.max_order, len(known_context)) + 1):
            suffix = known_context[-order:]
            counter = self.counts.get(suffix)
            if not counter:
                continue
            total = sum(counter.values())
            if total < self.min_count:
                continue
            weight = total / (total + self.backoff_strength)
            for token in self.vocabulary:
                empirical = counter.get(token, 0) / total
                probabilities[token] = (1.0 - weight) * probabilities[
                    token
                ] + weight * empirical
        return probabilities

    def predict(self, context: Sequence[str] = ()) -> tuple[str, float]:
        probabilities = self.distribution(context)
        return max(probabilities.items(), key=lambda item: (item[1], item[0]))

    def score(
        self,
        sequence: Iterable[ObservableStep | str],
        *,
        initial_context: Sequence[str] = (),
    ) -> SequenceScore:
        """Score a suffix online without updating the frozen learned counts."""

        targets = _tokens(sequence)
        if not targets:
            raise ValueError("cannot score an empty sequence")
        history = [self._known(token) for token in initial_context]
        total_bits = 0.0
        correct = 0
        for target in targets:
            known_target = self._known(target)
            probabilities = self.distribution(history)
            probability = probabilities[known_target]
            total_bits -= math.log2(probability)
            prediction = max(
                probabilities.items(), key=lambda item: (item[1], item[0])
            )[0]
            correct += prediction == known_target
            history.append(known_target)
        mean_bits = total_bits / len(targets)
        return SequenceScore(
            observations=len(targets),
            total_bits=total_bits,
            log_loss_bits=mean_bits,
            perplexity=2.0**mean_bits,
            accuracy=correct / len(targets),
        )

    def to_bytes(self) -> bytes:
        """Serialize learned counts deterministically as compact compressed JSON."""

        serialized_counts = []
        for context in sorted(self.counts, key=lambda value: (len(value), value)):
            serialized_counts.append(
                [
                    list(context),
                    [
                        [token, self.counts[context][token]]
                        for token in sorted(self.counts[context])
                    ],
                ]
            )
        payload = {
            "schema": MODEL_SCHEMA,
            "version": MODEL_VERSION,
            "max_order": self.max_order,
            "alpha": self.alpha,
            "backoff_strength": self.backoff_strength,
            "min_count": self.min_count,
            "vocabulary": list(self.vocabulary),
            "counts": serialized_counts,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return _SERIAL_PREFIX + zlib.compress(encoded, level=9)

    @classmethod
    def from_bytes(cls, data: bytes) -> TransitionFingerprint:
        """Restore bytes produced by :meth:`to_bytes`, validating all counts."""

        if not isinstance(data, bytes) or not data.startswith(_SERIAL_PREFIX):
            raise ValueError("not a FERTIG transition fingerprint")
        decompressor = zlib.decompressobj()
        try:
            encoded = decompressor.decompress(
                data[len(_SERIAL_PREFIX) :], _MAX_SERIALIZED_JSON + 1
            )
        except zlib.error as exc:
            raise ValueError("corrupt transition fingerprint") from exc
        if (
            len(encoded) > _MAX_SERIALIZED_JSON
            or decompressor.unconsumed_tail
            or decompressor.unused_data
            or not decompressor.eof
        ):
            raise ValueError("transition fingerprint payload is invalid or too large")
        try:
            payload = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("transition fingerprint JSON is invalid") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("transition fingerprint payload must be an object")
        if (
            payload.get("schema") != MODEL_SCHEMA
            or payload.get("version") != MODEL_VERSION
        ):
            raise ValueError("unsupported transition fingerprint schema/version")
        raw_counts = payload.get("counts")
        if not isinstance(raw_counts, list):
            raise ValueError("transition fingerprint counts must be an array")
        counts: dict[tuple[str, ...], dict[str, int]] = {}
        for entry in raw_counts:
            if not isinstance(entry, list) or len(entry) != 2:
                raise ValueError("invalid transition count entry")
            context, values = entry
            if not isinstance(context, list) or not all(
                isinstance(token, str) for token in context
            ):
                raise ValueError("invalid transition context")
            if not isinstance(values, list):
                raise ValueError("invalid transition target counts")
            counter: dict[str, int] = {}
            for pair in values:
                if (
                    not isinstance(pair, list)
                    or len(pair) != 2
                    or not isinstance(pair[0], str)
                    or isinstance(pair[1], bool)
                    or not isinstance(pair[1], int)
                ):
                    raise ValueError("invalid transition target count")
                counter[pair[0]] = pair[1]
            context_tuple = tuple(context)
            if context_tuple in counts:
                raise ValueError("duplicate transition context")
            counts[context_tuple] = counter
        vocabulary = payload.get("vocabulary")
        if not isinstance(vocabulary, list) or not all(
            isinstance(token, str) for token in vocabulary
        ):
            raise ValueError("invalid transition vocabulary")
        try:
            return cls(
                max_order=payload["max_order"],
                alpha=payload["alpha"],
                backoff_strength=payload["backoff_strength"],
                min_count=payload["min_count"],
                vocabulary=vocabulary,
                counts=counts,
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("transition fingerprint parameters are invalid") from exc

    @classmethod
    def evaluate_holdout(
        cls,
        sequence: Iterable[ObservableStep | str],
        **kwargs: Any,
    ) -> HoldoutEvaluation:
        return evaluate_holdout(sequence, **kwargs)


@dataclass(frozen=True, slots=True)
class HoldoutEvaluation:
    """Chronological comparison; positive gain means ordered counts are better."""

    view: str
    total_steps: int
    split_index: int
    train_steps: int
    test_steps: int
    ordered: SequenceScore
    shuffled: SequenceScore
    unigram: SequenceScore
    gain_vs_shuffled_bits: float
    gain_vs_unigram_bits: float
    ordered_wins: bool
    shuffle_seed: int

    def to_dict(self) -> dict[str, object]:
        return {
            "view": self.view,
            "total_steps": self.total_steps,
            "split_index": self.split_index,
            "train_steps": self.train_steps,
            "test_steps": self.test_steps,
            "ordered": self.ordered.to_dict(),
            "shuffled": self.shuffled.to_dict(),
            "unigram": self.unigram.to_dict(),
            "gain_vs_shuffled_bits": self.gain_vs_shuffled_bits,
            "gain_vs_unigram_bits": self.gain_vs_unigram_bits,
            "ordered_wins": self.ordered_wins,
            "shuffle_seed": self.shuffle_seed,
        }


def evaluate_holdout(
    sequence: Iterable[ObservableStep | str],
    *,
    holdout_fraction: float = 0.2,
    seed: int = 0,
    max_order: int = 4,
    alpha: float = 0.5,
    backoff_strength: float = 3.0,
    min_count: int = 2,
    view: str = "full",
) -> HoldoutEvaluation:
    """Evaluate the untouched chronological suffix against two baselines.

    The shuffled baseline retains exactly the training-prefix marginals but
    destroys their order with a deterministic seed.  The unigram baseline is
    fully orderless.  Neither baseline nor the ordered model sees the suffix
    during fitting.
    """

    if not math.isfinite(holdout_fraction) or not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be finite and between zero and one")
    tokens = _tokens(sequence)
    if len(tokens) < 4:
        raise ValueError("chronological holdout requires at least four steps")
    test_size = max(1, math.ceil(len(tokens) * holdout_fraction))
    split = len(tokens) - test_size
    if split < 2:
        raise ValueError("chronological holdout leaves too few training steps")
    train, test = tokens[:split], tokens[split:]
    fit_kwargs = {
        "max_order": max_order,
        "alpha": alpha,
        "backoff_strength": backoff_strength,
        "min_count": min_count,
    }
    ordered_model = TransitionFingerprint.fit(train, **fit_kwargs)
    initial_context = train[-max_order:] if max_order else ()
    ordered_score = ordered_model.score(test, initial_context=initial_context)

    shuffled_train = list(train)
    random.Random(seed).shuffle(shuffled_train)
    shuffled_model = TransitionFingerprint.fit(shuffled_train, **fit_kwargs)
    shuffled_score = shuffled_model.score(test, initial_context=initial_context)

    unigram_model = TransitionFingerprint.fit(
        train,
        max_order=0,
        alpha=alpha,
        backoff_strength=backoff_strength,
        min_count=min_count,
    )
    unigram_score = unigram_model.score(test)
    gain_shuffled = shuffled_score.log_loss_bits - ordered_score.log_loss_bits
    gain_unigram = unigram_score.log_loss_bits - ordered_score.log_loss_bits
    return HoldoutEvaluation(
        view=view,
        total_steps=len(tokens),
        split_index=split,
        train_steps=len(train),
        test_steps=len(test),
        ordered=ordered_score,
        shuffled=shuffled_score,
        unigram=unigram_score,
        gain_vs_shuffled_bits=gain_shuffled,
        gain_vs_unigram_bits=gain_unigram,
        ordered_wins=(gain_shuffled > 0.0 and gain_unigram > 0.0),
        shuffle_seed=seed,
    )


@dataclass(frozen=True, slots=True)
class RecoveryMotif:
    """Observable error followed by a later agent decision and its result."""

    error_index: int
    error_action: str
    decision_index: int | None
    decision_action: str | None
    result_index: int | None
    result_outcome: str | None
    subsequent_success_observed: bool
    gap: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "error_index": self.error_index,
            "error_action": self.error_action,
            "decision_index": self.decision_index,
            "decision_action": self.decision_action,
            "result_index": self.result_index,
            "result_outcome": self.result_outcome,
            "subsequent_success_observed": self.subsequent_success_observed,
            "gap": self.gap,
        }


def recovery_motifs(
    steps: Iterable[ObservableStep], *, max_gap: int = 8
) -> tuple[RecoveryMotif, ...]:
    """Find error -> next decision -> correlated result motifs.

    ``subsequent_success_observed`` is intentionally literal.  The function
    does not claim that the later action fixed the earlier error or even shared
    its semantic cause.
    """

    if isinstance(max_gap, bool) or not isinstance(max_gap, int) or max_gap < 1:
        raise ValueError("max_gap must be a positive integer")
    ordered = tuple(steps)
    motifs: list[RecoveryMotif] = []
    for position, error in enumerate(ordered):
        if error.kind != "outcome" or error.outcome != "error":
            continue
        decision: ObservableStep | None = None
        decision_position: int | None = None
        for candidate_position in range(
            position + 1, min(len(ordered), position + max_gap + 1)
        ):
            candidate = ordered[candidate_position]
            if candidate.kind == "decision" and candidate.actor == "agent":
                decision = candidate
                decision_position = candidate_position
                break
        result: ObservableStep | None = None
        if decision is not None and decision.action.startswith("tool/"):
            assert decision_position is not None
            for candidate in ordered[
                decision_position + 1 : min(
                    len(ordered), decision_position + max_gap + 1
                )
            ]:
                if candidate.kind != "outcome":
                    continue
                if (
                    decision.correlation_id is not None
                    and candidate.correlation_id == decision.correlation_id
                ) or (
                    decision.correlation_id is None
                    and candidate.action == decision.action
                ):
                    result = candidate
                    break
        motifs.append(
            RecoveryMotif(
                error_index=error.index,
                error_action=error.action,
                decision_index=decision.index if decision is not None else None,
                decision_action=decision.action if decision is not None else None,
                result_index=result.index if result is not None else None,
                result_outcome=result.outcome if result is not None else None,
                subsequent_success_observed=(
                    result is not None and result.outcome == "ok"
                ),
                gap=(decision.index - error.index if decision is not None else None),
            )
        )
    return tuple(motifs)


@dataclass(frozen=True, slots=True)
class SurprisingTransition:
    index: int
    observed: str
    probability: float
    surprise_bits: float

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "index": self.index,
            "observed": self.observed,
            "probability": self.probability,
            "surprise_bits": self.surprise_bits,
        }


def surprising_transitions(
    model: TransitionFingerprint,
    sequence: Iterable[ObservableStep | str],
    *,
    limit: int = 10,
    initial_context: Sequence[str] = (),
    index_offset: int = 0,
) -> tuple[SurprisingTransition, ...]:
    """Rank observed steps by self-information under a frozen fingerprint."""

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    tokens = _tokens(sequence)
    if (
        isinstance(index_offset, bool)
        or not isinstance(index_offset, int)
        or index_offset < 0
    ):
        raise ValueError("index_offset must be a non-negative integer")
    history = [model._known(token) for token in initial_context]
    surprises: list[SurprisingTransition] = []
    for index, target in enumerate(tokens):
        known_target = model._known(target)
        probability = model.distribution(history)[known_target]
        surprises.append(
            SurprisingTransition(
                index=index_offset + index,
                observed=target,
                probability=probability,
                surprise_bits=-math.log2(probability),
            )
        )
        history.append(known_target)
    surprises.sort(key=lambda item: (-item.surprise_bits, item.index))
    return tuple(surprises[:limit])


@dataclass(frozen=True, slots=True)
class TraceLearningReport:
    """Machine-readable result of one empirical trace analysis."""

    observable_steps: int
    decision_steps: int
    full_stream: HoldoutEvaluation
    decision_stream: HoldoutEvaluation
    fingerprint_bytes: int
    vocabulary_size: int
    recovery_motifs: tuple[RecoveryMotif, ...]
    surprising_decisions: tuple[SurprisingTransition, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "scope": "one-session behavioral fingerprint; no generalization claim",
            "observable_steps": self.observable_steps,
            "decision_steps": self.decision_steps,
            "full_stream": self.full_stream.to_dict(),
            "decision_stream": self.decision_stream.to_dict(),
            "fingerprint_bytes": self.fingerprint_bytes,
            "vocabulary_size": self.vocabulary_size,
            "recovery_motifs": [motif.to_dict() for motif in self.recovery_motifs],
            "surprising_decisions": [
                transition.to_dict() for transition in self.surprising_decisions
            ],
        }


def analyze_trace(
    records: object,
    *,
    holdout_fraction: float = 0.2,
    seed: int = 0,
    max_order: int = 4,
) -> TraceLearningReport:
    """Canonicalize, learn, and honestly evaluate both observable views."""

    steps = canonicalize_trace(records)
    decisions = decision_steps(steps)
    full_evaluation = evaluate_holdout(
        steps,
        holdout_fraction=holdout_fraction,
        seed=seed,
        max_order=max_order,
        view="full",
    )
    decision_evaluation = evaluate_holdout(
        decisions,
        holdout_fraction=holdout_fraction,
        seed=seed,
        max_order=max_order,
        view="decisions",
    )
    decision_split = decision_evaluation.split_index
    decision_train = decisions[:decision_split]
    decision_holdout = decisions[decision_split:]
    decision_model = TransitionFingerprint.fit(decision_train, max_order=max_order)
    initial_context = (
        tuple(step.symbol for step in decision_train[-max_order:]) if max_order else ()
    )
    return TraceLearningReport(
        observable_steps=len(steps),
        decision_steps=len(decisions),
        full_stream=full_evaluation,
        decision_stream=decision_evaluation,
        fingerprint_bytes=len(decision_model.to_bytes()),
        vocabulary_size=len(decision_model.vocabulary),
        recovery_motifs=recovery_motifs(steps),
        surprising_decisions=surprising_transitions(
            decision_model,
            decision_holdout,
            limit=10,
            initial_context=initial_context,
            index_offset=decision_split,
        ),
    )


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, Mapping):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            records.append(record)
    return records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Learn a narrow observable transition fingerprint from JSONL"
    )
    parser.add_argument("trace", type=Path, help="completed host JSONL trace")
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-order", type=int, default=4)
    args = parser.parse_args(argv)
    report = analyze_trace(
        _read_jsonl(args.trace),
        holdout_fraction=args.holdout,
        seed=args.seed,
        max_order=args.max_order,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
