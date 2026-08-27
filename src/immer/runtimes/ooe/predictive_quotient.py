"""Exact predictive-state quotient over verified execution learning traces.

The quotient observes only runtime facts already authenticated by the execution
learning bridge.  A state is one exact ``(weight site, source action)`` pair;
an outcome is the action that a pinned executor actually returned together
with its action/quality verifier identities and forward accounting.  No label,
task answer, or caller-authored target enters the construction.

Partition refinement uses empirical joint distributions over
``(verified outcome, successor class)``.  Counts are normalized with
``fractions.Fraction`` and serialized as reduced integer ratios, so refinement,
replay verification, and held-out total variation contain no floating-point
comparison.  The terminal successor is an explicit sealed state.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import json
from typing import Any, cast

from .execution_learning import (
    ControllerActionTraceReceipt,
    ExecutionLearningIntegrityError,
    ExecutionLearningReceipt,
    ZERO_SHA256,
)
from .identity import canonical_json_bytes, require_sha256
from .qwen_bridge import OoeAction, validate_action


PREDICTIVE_TRANSITION_SCHEMA = "immer-ooe-predictive-transition/v1"
PREDICTIVE_QUOTIENT_SCHEMA = "immer-ooe-predictive-quotient/v1"
PREDICTIVE_VALIDATION_SCHEMA = "immer-ooe-predictive-validation/v1"
PREDICTIVE_OUTCOME_SCHEMA = {
    "fields": [
        "action",
        "executor_sha256",
        "action_verifier_sha256",
        "quality_verifier_sha256",
        "qwen_forwards",
        "teacher_baseline_qwen_forwards",
    ],
    "observation": "verified-executed-action",
    "schema": "immer-ooe-predictive-outcome-alphabet/v1",
}
PREDICTIVE_OUTCOME_SCHEMA_SHA256 = hashlib.sha256(
    canonical_json_bytes(PREDICTIVE_OUTCOME_SCHEMA)
).hexdigest()
TERMINAL_STATE = {
    "kind": "terminal",
    "schema": "immer-ooe-predictive-terminal/v1",
}
TERMINAL_STATE_SHA256 = hashlib.sha256(canonical_json_bytes(TERMINAL_STATE)).hexdigest()
CENSORED_STATE = {
    "kind": "censored",
    "schema": "immer-ooe-predictive-censored/v1",
}
CENSORED_STATE_SHA256 = hashlib.sha256(canonical_json_bytes(CENSORED_STATE)).hexdigest()
MAX_PREDICTIVE_TRANSITIONS = 1_000_000
MAX_PREDICTIVE_STATES = 1_000_000
MAX_PREDICTIVE_CLASSES = 1_000_000
MAX_PREDICTIVE_DOCUMENT_BYTES = 256 * 1024 * 1024
_TERMINAL_PARTITION = -1
_CENSORED_PARTITION = -2


class PredictiveQuotientError(ValueError):
    """Predictive quotient inputs cannot satisfy the exact contract."""


class PredictiveQuotientIntegrityError(PredictiveQuotientError):
    """A transition, quotient, validation, or source binding was modified."""


class PredictiveQuotientCoverageError(PredictiveQuotientError):
    """Held-out evidence contains a state absent from the trained quotient."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _uint(value: object, *, field: str, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive " if positive else "non-negative "
        raise PredictiveQuotientError(f"{field} must be a {qualifier}integer")
    return value


def _bounded_sequence(
    values: Sequence[Any],
    *,
    field: str,
    maximum: int,
    allow_empty: bool = False,
) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise PredictiveQuotientError(f"{field} must be a bounded sequence")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise PredictiveQuotientError(f"{field} must be a bounded sequence") from exc
    minimum = 0 if allow_empty else 1
    if not minimum <= len(result) <= maximum:
        raise PredictiveQuotientError(
            f"{field} must contain {minimum}..{maximum} entries"
        )
    return result


def _seal(schema: str, body: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(canonical_json_bytes(dict(body)))
    return {"body": normalized, "schema": schema, "sha256": _digest(normalized)}


def _unseal(
    document: Mapping[str, Any],
    *,
    schema: str,
    fields: frozenset[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(document, Mapping) or set(document) != {
        "body",
        "schema",
        "sha256",
    }:
        raise PredictiveQuotientIntegrityError(f"{label} envelope is invalid")
    if document.get("schema") != schema:
        raise PredictiveQuotientIntegrityError(f"{label} schema is invalid")
    body = document.get("body")
    if not isinstance(body, Mapping) or set(body) != fields:
        raise PredictiveQuotientIntegrityError(f"{label} body is invalid")
    try:
        claimed = require_sha256(document.get("sha256"), field=f"{label}.sha256")
    except ValueError as exc:
        raise PredictiveQuotientIntegrityError(f"{label} seal is invalid") from exc
    if claimed != _digest(body):
        raise PredictiveQuotientIntegrityError(f"{label} SHA-256 mismatch")
    return cast(dict[str, Any], json.loads(canonical_json_bytes(body)))


def _strict_document_bytes(document: Mapping[str, Any], *, label: str) -> bytes:
    data = canonical_json_bytes(dict(document))
    if len(data) > MAX_PREDICTIVE_DOCUMENT_BYTES:
        raise PredictiveQuotientIntegrityError(f"{label} exceeds its byte bound")
    return data


@dataclass(frozen=True, slots=True, order=True)
class PredictiveState:
    """One observable Markov state: exact weight site plus source action."""

    site_identity_sha256: str
    source_action: OoeAction | str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "site_identity_sha256",
            require_sha256(self.site_identity_sha256, field="site_identity_sha256"),
        )
        object.__setattr__(self, "source_action", validate_action(self.source_action))

    def to_dict(self) -> dict[str, Any]:
        return {
            "site_identity_sha256": self.site_identity_sha256,
            "source_action": self.source_action,
        }

    @property
    def sha256(self) -> str:
        return _digest({"schema": "immer-ooe-predictive-state/v1", **self.to_dict()})

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PredictiveState":
        if not isinstance(value, Mapping) or set(value) != {
            "site_identity_sha256",
            "source_action",
        }:
            raise PredictiveQuotientIntegrityError("predictive state is invalid")
        return cls(**dict(value))


@dataclass(frozen=True, slots=True, order=True)
class VerifiedExecutionOutcome:
    """Action-level outcome authenticated by its executor and two verifiers."""

    action: OoeAction | str
    executor_sha256: str
    action_verifier_sha256: str
    quality_verifier_sha256: str
    qwen_forwards: int
    teacher_baseline_qwen_forwards: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", validate_action(self.action))
        for name in (
            "executor_sha256",
            "action_verifier_sha256",
            "quality_verifier_sha256",
        ):
            object.__setattr__(
                self,
                name,
                require_sha256(getattr(self, name), field=name),
            )
        qwen_forwards = _uint(self.qwen_forwards, field="qwen_forwards")
        baseline = _uint(
            self.teacher_baseline_qwen_forwards,
            field="teacher_baseline_qwen_forwards",
        )
        if qwen_forwards > 1_000_000 or baseline > 1_000_000:
            raise PredictiveQuotientError("outcome forward accounting is unbounded")
        object.__setattr__(self, "qwen_forwards", qwen_forwards)
        object.__setattr__(self, "teacher_baseline_qwen_forwards", baseline)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "action_verifier_sha256": self.action_verifier_sha256,
            "executor_sha256": self.executor_sha256,
            "quality_verifier_sha256": self.quality_verifier_sha256,
            "qwen_forwards": self.qwen_forwards,
            "teacher_baseline_qwen_forwards": self.teacher_baseline_qwen_forwards,
        }

    @property
    def sha256(self) -> str:
        return _digest({"schema": PREDICTIVE_OUTCOME_SCHEMA_SHA256, **self.to_dict()})

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VerifiedExecutionOutcome":
        if not isinstance(value, Mapping) or set(value) != set(
            cls.__dataclass_fields__
        ):
            raise PredictiveQuotientIntegrityError("predictive outcome is invalid")
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class PredictiveTransition:
    """One verifier-bound observation and its explicit successor/terminal."""

    state: PredictiveState
    outcome: VerifiedExecutionOutcome
    successor_state: PredictiveState | None
    terminal: bool
    censored: bool
    temporal_index: int
    learning_receipt_sha256: str
    trace_sha256: str
    feature_receipt_sha256: str
    execution_sha256: str
    quality_sha256: str
    teacher_transition_sha256: str
    action_authority_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.state, PredictiveState):
            raise TypeError("state must be a PredictiveState")
        if not isinstance(self.outcome, VerifiedExecutionOutcome):
            raise TypeError("outcome must be a VerifiedExecutionOutcome")
        if self.successor_state is not None and not isinstance(
            self.successor_state, PredictiveState
        ):
            raise TypeError("successor_state must be PredictiveState or None")
        if not isinstance(self.terminal, bool):
            raise TypeError("terminal must be bool")
        if not isinstance(self.censored, bool):
            raise TypeError("censored must be bool")
        if self.terminal and self.censored:
            raise PredictiveQuotientError(
                "a successor cannot be both terminal and censored"
            )
        if (self.terminal or self.censored) != (self.successor_state is None):
            raise PredictiveQuotientError(
                "an absent successor must be explicitly terminal or censored"
            )
        if (
            self.successor_state is not None
            and self.successor_state.source_action != self.outcome.action
        ):
            raise PredictiveQuotientIntegrityError(
                "successor source action differs from executed outcome"
            )
        object.__setattr__(
            self,
            "temporal_index",
            _uint(self.temporal_index, field="temporal_index"),
        )
        for name in (
            "learning_receipt_sha256",
            "trace_sha256",
            "feature_receipt_sha256",
            "execution_sha256",
            "quality_sha256",
            "teacher_transition_sha256",
            "action_authority_sha256",
        ):
            object.__setattr__(
                self,
                name,
                require_sha256(getattr(self, name), field=name),
            )

    def as_record(self) -> dict[str, Any]:
        return {
            "action_authority_sha256": self.action_authority_sha256,
            "censored": self.censored,
            "execution_sha256": self.execution_sha256,
            "feature_receipt_sha256": self.feature_receipt_sha256,
            "learning_receipt_sha256": self.learning_receipt_sha256,
            "outcome": self.outcome.to_dict(),
            "outcome_sha256": self.outcome.sha256,
            "quality_sha256": self.quality_sha256,
            "state": self.state.to_dict(),
            "state_sha256": self.state.sha256,
            "successor_state": (
                None if self.successor_state is None else self.successor_state.to_dict()
            ),
            "successor_state_sha256": (
                TERMINAL_STATE_SHA256
                if self.terminal
                else (
                    CENSORED_STATE_SHA256
                    if self.censored
                    else cast(PredictiveState, self.successor_state).sha256
                )
            ),
            "teacher_transition_sha256": self.teacher_transition_sha256,
            "temporal_index": self.temporal_index,
            "terminal": self.terminal,
            "trace_sha256": self.trace_sha256,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())

    def to_document(self) -> dict[str, Any]:
        return _seal(PREDICTIVE_TRANSITION_SCHEMA, self.as_record())

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "PredictiveTransition":
        body = _unseal(
            document,
            schema=PREDICTIVE_TRANSITION_SCHEMA,
            fields=frozenset(
                {
                    "action_authority_sha256",
                    "censored",
                    "execution_sha256",
                    "feature_receipt_sha256",
                    "learning_receipt_sha256",
                    "outcome",
                    "outcome_sha256",
                    "quality_sha256",
                    "state",
                    "state_sha256",
                    "successor_state",
                    "successor_state_sha256",
                    "teacher_transition_sha256",
                    "temporal_index",
                    "terminal",
                    "trace_sha256",
                }
            ),
            label="predictive transition",
        )
        state = PredictiveState.from_dict(body.pop("state"))
        outcome = VerifiedExecutionOutcome.from_dict(body.pop("outcome"))
        successor_raw = body.pop("successor_state")
        successor = (
            None if successor_raw is None else PredictiveState.from_dict(successor_raw)
        )
        if successor is None and not (body["terminal"] or body["censored"]):
            raise PredictiveQuotientIntegrityError(
                "transition omits a non-terminal/non-censored successor"
            )
        if successor is not None and (body["terminal"] or body["censored"]):
            raise PredictiveQuotientIntegrityError(
                "terminal/censored transition unexpectedly contains a successor"
            )
        if body.pop("state_sha256") != state.sha256:
            raise PredictiveQuotientIntegrityError("transition state SHA mismatch")
        if body.pop("outcome_sha256") != outcome.sha256:
            raise PredictiveQuotientIntegrityError("transition outcome SHA mismatch")
        expected_successor = (
            TERMINAL_STATE_SHA256
            if body["terminal"]
            else (
                CENSORED_STATE_SHA256
                if body["censored"]
                else cast(PredictiveState, successor).sha256
            )
        )
        if body.pop("successor_state_sha256") != expected_successor:
            raise PredictiveQuotientIntegrityError("transition successor SHA mismatch")
        return cls(state=state, outcome=outcome, successor_state=successor, **body)


def derive_execution_transitions(
    receipts: Sequence[ExecutionLearningReceipt],
    traces: Sequence[ControllerActionTraceReceipt],
    *,
    terminal: bool,
) -> tuple[PredictiveTransition, ...]:
    """Derive a chain; the caller must explicitly declare its terminal tail."""

    if not isinstance(terminal, bool):
        raise TypeError("terminal must be an exact bool")

    receipt_rows = _bounded_sequence(
        receipts,
        field="receipts",
        maximum=MAX_PREDICTIVE_TRANSITIONS,
    )
    trace_rows = _bounded_sequence(
        traces,
        field="traces",
        maximum=MAX_PREDICTIVE_TRANSITIONS,
    )
    if len(receipt_rows) != len(trace_rows):
        raise PredictiveQuotientIntegrityError(
            "receipt and trace counts must be identical"
        )
    for receipt in receipt_rows:
        if not isinstance(receipt, ExecutionLearningReceipt):
            raise TypeError("receipts must contain ExecutionLearningReceipt values")
        try:
            replayed = ExecutionLearningReceipt.from_document(receipt.to_document())
        except (ExecutionLearningIntegrityError, ValueError, TypeError) as exc:
            raise PredictiveQuotientIntegrityError(
                "execution-learning receipt failed exact replay"
            ) from exc
        if replayed != receipt:
            raise PredictiveQuotientIntegrityError(
                "execution-learning receipt changed during replay"
            )
    for trace in trace_rows:
        if not isinstance(trace, ControllerActionTraceReceipt):
            raise TypeError("traces must contain ControllerActionTraceReceipt values")
        try:
            replayed_trace = ControllerActionTraceReceipt.from_document(
                trace.to_document()
            )
        except (ExecutionLearningIntegrityError, ValueError, TypeError) as exc:
            raise PredictiveQuotientIntegrityError(
                "controller action trace failed exact replay"
            ) from exc
        if replayed_trace != trace:
            raise PredictiveQuotientIntegrityError(
                "controller action trace changed during replay"
            )

    transitions: list[PredictiveTransition] = []
    previous_trace_sha256 = ZERO_SHA256
    previous_temporal: int | None = None
    previous_snapshot_after: str | None = None
    for index, (receipt, trace) in enumerate(
        zip(receipt_rows, trace_rows, strict=True)
    ):
        if (
            trace.ordinal != index
            or trace.learning_receipt_sha256 != receipt.sha256
            or trace.measurement_sha256 != receipt.measurement.sha256
            or trace.temporal_index != receipt.feature.temporal_index
            or trace.source_action != receipt.transition.source_action
            or trace.target_action != receipt.execution.action
            or trace.previous_trace_sha256 != previous_trace_sha256
            or trace.previous_stream_head_sha256 != previous_trace_sha256
            or (
                previous_temporal is not None
                and trace.temporal_index != previous_temporal + 1
            )
            or (
                previous_snapshot_after is not None
                and trace.controller_snapshot_before_sha256 != previous_snapshot_after
            )
        ):
            raise PredictiveQuotientIntegrityError(
                "execution receipt/trace chain is discontinuous"
            )
        if (
            index
            and receipt.transition.source_action
            != receipt_rows[index - 1].execution.action
        ):
            raise PredictiveQuotientIntegrityError(
                "next source action differs from prior executed action"
            )
        state = PredictiveState(
            site_identity_sha256=receipt.feature.site_identity.sha256,
            source_action=receipt.transition.source_action,
        )
        successor = None
        if index + 1 < len(receipt_rows):
            next_receipt = receipt_rows[index + 1]
            successor = PredictiveState(
                site_identity_sha256=next_receipt.feature.site_identity.sha256,
                source_action=next_receipt.transition.source_action,
            )
        authority = receipt.action_authority
        outcome = VerifiedExecutionOutcome(
            action=receipt.execution.action,
            executor_sha256=authority.executor_sha256,
            action_verifier_sha256=authority.action_verifier_sha256,
            quality_verifier_sha256=authority.quality_verifier_sha256,
            qwen_forwards=receipt.execution.qwen_forwards,
            teacher_baseline_qwen_forwards=(
                receipt.execution.teacher_baseline_qwen_forwards
            ),
        )
        transitions.append(
            PredictiveTransition(
                state=state,
                outcome=outcome,
                successor_state=successor,
                terminal=successor is None and terminal,
                censored=successor is None and not terminal,
                temporal_index=trace.temporal_index,
                learning_receipt_sha256=receipt.sha256,
                trace_sha256=trace.sha256,
                feature_receipt_sha256=receipt.feature.sha256,
                execution_sha256=receipt.execution.sha256,
                quality_sha256=receipt.quality.sha256,
                teacher_transition_sha256=receipt.transition.sha256,
                action_authority_sha256=authority.sha256,
            )
        )
        previous_trace_sha256 = trace.sha256
        previous_temporal = trace.temporal_index
        previous_snapshot_after = trace.controller_snapshot_after_sha256
    return tuple(transitions)


def derive_local_agent_transitions(
    receipts: Sequence[ExecutionLearningReceipt],
    traces: Sequence[ControllerActionTraceReceipt],
) -> tuple[PredictiveTransition, ...]:
    """Derive the Markov state transition executed inside each weight site.

    The authenticated trace order proves that every receipt belongs to one
    continuous runtime stream.  It is not the state topology of a site-local
    ``MarkovPDAgent``: after executing action ``a`` at site ``s``, that agent's
    successor is ``(s, a)`` even when the next Atlas measurement visits another
    site.  This view is therefore the one used to quotient controller agents;
    ``derive_execution_transitions`` remains the global trace-topology view.
    """

    traced = derive_execution_transitions(receipts, traces, terminal=False)
    return tuple(
        replace(
            row,
            successor_state=PredictiveState(
                site_identity_sha256=row.state.site_identity_sha256,
                source_action=row.outcome.action,
            ),
            terminal=False,
            censored=False,
        )
        for row in traced
    )


@dataclass(frozen=True, slots=True, order=True)
class ExactJointProbability:
    outcome_sha256: str
    successor_class_sha256: str
    terminal: bool
    censored: bool
    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "outcome_sha256",
            require_sha256(self.outcome_sha256, field="outcome_sha256"),
        )
        object.__setattr__(
            self,
            "successor_class_sha256",
            require_sha256(
                self.successor_class_sha256,
                field="successor_class_sha256",
            ),
        )
        if not isinstance(self.terminal, bool):
            raise TypeError("terminal must be bool")
        if not isinstance(self.censored, bool):
            raise TypeError("censored must be bool")
        if self.terminal and self.censored:
            raise PredictiveQuotientIntegrityError(
                "joint probability cannot be terminal and censored"
            )
        expected_special = (
            TERMINAL_STATE_SHA256
            if self.terminal
            else CENSORED_STATE_SHA256
            if self.censored
            else None
        )
        if (expected_special is not None) != (
            self.successor_class_sha256
            in {TERMINAL_STATE_SHA256, CENSORED_STATE_SHA256}
        ) or (
            expected_special is not None
            and self.successor_class_sha256 != expected_special
        ):
            raise PredictiveQuotientIntegrityError(
                "joint probability successor identity is inconsistent"
            )
        numerator = _uint(self.numerator, field="numerator", positive=True)
        denominator = _uint(self.denominator, field="denominator", positive=True)
        if numerator > denominator:
            raise PredictiveQuotientError("probability numerator exceeds denominator")
        reduced = Fraction(numerator, denominator)
        if (reduced.numerator, reduced.denominator) != (numerator, denominator):
            raise PredictiveQuotientIntegrityError(
                "probability ratio must be canonically reduced"
            )

    @property
    def fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    def to_dict(self) -> dict[str, Any]:
        return {
            "censored": self.censored,
            "denominator": self.denominator,
            "numerator": self.numerator,
            "outcome_sha256": self.outcome_sha256,
            "successor_class_sha256": self.successor_class_sha256,
            "terminal": self.terminal,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExactJointProbability":
        if not isinstance(value, Mapping) or set(value) != set(
            cls.__dataclass_fields__
        ):
            raise PredictiveQuotientIntegrityError("joint probability is invalid")
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class PredictiveClass:
    class_sha256: str
    member_states: tuple[PredictiveState, ...]
    joint_distribution: tuple[ExactJointProbability, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "class_sha256",
            require_sha256(self.class_sha256, field="class_sha256"),
        )
        members = _bounded_sequence(
            self.member_states,
            field="member_states",
            maximum=MAX_PREDICTIVE_STATES,
        )
        distribution = _bounded_sequence(
            self.joint_distribution,
            field="joint_distribution",
            maximum=MAX_PREDICTIVE_TRANSITIONS,
        )
        if any(not isinstance(row, PredictiveState) for row in members):
            raise TypeError("member_states must contain PredictiveState values")
        if any(not isinstance(row, ExactJointProbability) for row in distribution):
            raise TypeError(
                "joint_distribution must contain ExactJointProbability values"
            )
        members = tuple(sorted(members, key=lambda row: row.sha256))
        distribution = tuple(
            sorted(
                distribution,
                key=lambda row: (
                    row.outcome_sha256,
                    row.successor_class_sha256,
                    row.terminal,
                    row.censored,
                ),
            )
        )
        if len({row.sha256 for row in members}) != len(members):
            raise PredictiveQuotientIntegrityError("class member is duplicated")
        if len(
            {
                (
                    row.outcome_sha256,
                    row.successor_class_sha256,
                    row.terminal,
                    row.censored,
                )
                for row in distribution
            }
        ) != len(distribution):
            raise PredictiveQuotientIntegrityError(
                "class joint probability key is duplicated"
            )
        if sum((row.fraction for row in distribution), Fraction()) != 1:
            raise PredictiveQuotientIntegrityError(
                "class joint distribution does not sum exactly to one"
            )
        object.__setattr__(self, "member_states", members)
        object.__setattr__(self, "joint_distribution", distribution)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_sha256": self.class_sha256,
            "joint_distribution": [row.to_dict() for row in self.joint_distribution],
            "member_states": [row.to_dict() for row in self.member_states],
            "member_state_sha256s": [row.sha256 for row in self.member_states],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PredictiveClass":
        if not isinstance(value, Mapping) or set(value) != {
            "class_sha256",
            "joint_distribution",
            "member_states",
            "member_state_sha256s",
        }:
            raise PredictiveQuotientIntegrityError("predictive class is invalid")
        members = tuple(
            PredictiveState.from_dict(row) for row in value["member_states"]
        )
        if list(value["member_state_sha256s"]) != [row.sha256 for row in members]:
            raise PredictiveQuotientIntegrityError("class member SHA list mismatch")
        return cls(
            class_sha256=value["class_sha256"],
            member_states=members,
            joint_distribution=tuple(
                ExactJointProbability.from_dict(row)
                for row in value["joint_distribution"]
            ),
        )


@dataclass(frozen=True, slots=True, order=True)
class StateClassAssignment:
    state: PredictiveState
    class_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.state, PredictiveState):
            raise TypeError("state must be PredictiveState")
        object.__setattr__(
            self,
            "class_sha256",
            require_sha256(self.class_sha256, field="class_sha256"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_sha256": self.class_sha256,
            "state": self.state.to_dict(),
            "state_sha256": self.state.sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StateClassAssignment":
        if not isinstance(value, Mapping) or set(value) != {
            "class_sha256",
            "state",
            "state_sha256",
        }:
            raise PredictiveQuotientIntegrityError("state assignment is invalid")
        state = PredictiveState.from_dict(value["state"])
        if value["state_sha256"] != state.sha256:
            raise PredictiveQuotientIntegrityError("assignment state SHA mismatch")
        return cls(state=state, class_sha256=value["class_sha256"])


def _transition_sort_key(row: PredictiveTransition) -> bytes:
    return canonical_json_bytes(row.as_record())


def _normalize_transitions(
    transitions: Sequence[PredictiveTransition],
    *,
    require_successor_coverage: bool = True,
) -> tuple[PredictiveTransition, ...]:
    if not isinstance(require_successor_coverage, bool):
        raise TypeError("require_successor_coverage must be bool")
    rows = _bounded_sequence(
        transitions,
        field="transitions",
        maximum=MAX_PREDICTIVE_TRANSITIONS,
    )
    if any(not isinstance(row, PredictiveTransition) for row in rows):
        raise TypeError("transitions must contain PredictiveTransition values")
    normalized = tuple(sorted(rows, key=_transition_sort_key))
    if len({row.sha256 for row in normalized}) != len(normalized):
        raise PredictiveQuotientIntegrityError(
            "predictive transition receipt is duplicated"
        )
    if len({row.learning_receipt_sha256 for row in normalized}) != len(normalized):
        raise PredictiveQuotientIntegrityError(
            "execution-learning receipt was replay-counted"
        )
    if len({row.trace_sha256 for row in normalized}) != len(normalized):
        raise PredictiveQuotientIntegrityError("controller trace was replay-counted")
    sources = {row.state for row in normalized}
    successors = {
        row.successor_state for row in normalized if row.successor_state is not None
    }
    missing = successors - sources
    if require_successor_coverage and missing:
        raise PredictiveQuotientIntegrityError(
            "non-terminal successor has no empirical outgoing distribution"
        )
    if len(sources) > MAX_PREDICTIVE_STATES:
        raise PredictiveQuotientError("predictive state count exceeds its bound")
    return normalized


def _fraction_signature(
    counts: Counter[tuple[str, int]],
) -> tuple[tuple[str, int, int, int], ...]:
    total = sum(counts.values())
    if total < 1:
        raise PredictiveQuotientIntegrityError("state has no observations")
    result = []
    for (outcome_sha256, successor_class), count in counts.items():
        probability = Fraction(count, total)
        result.append(
            (
                outcome_sha256,
                successor_class,
                probability.numerator,
                probability.denominator,
            )
        )
    return tuple(sorted(result, key=canonical_json_bytes))


def _same_partition(
    first: Mapping[PredictiveState, int],
    second: Mapping[PredictiveState, int],
) -> bool:
    forward: dict[int, int] = {}
    reverse: dict[int, int] = {}
    for state, first_class in first.items():
        second_class = second[state]
        known_forward = forward.setdefault(first_class, second_class)
        known_reverse = reverse.setdefault(second_class, first_class)
        if known_forward != second_class or known_reverse != first_class:
            return False
    return True


@dataclass(frozen=True, slots=True)
class _RefinementResult:
    assignments: tuple[StateClassAssignment, ...]
    classes: tuple[PredictiveClass, ...]
    rounds: int


def _successor_partition(
    row: PredictiveTransition,
    partition: Mapping[PredictiveState, int],
) -> int:
    if row.terminal:
        return _TERMINAL_PARTITION
    if row.censored:
        return _CENSORED_PARTITION
    return partition[cast(PredictiveState, row.successor_state)]


def _refine(transitions: Sequence[PredictiveTransition]) -> _RefinementResult:
    rows = _normalize_transitions(transitions)
    states = tuple(sorted({row.state for row in rows}, key=lambda row: row.sha256))
    grouped: dict[PredictiveState, list[PredictiveTransition]] = {
        state: [] for state in states
    }
    for row in rows:
        grouped[row.state].append(row)
    by_state = {state: tuple(grouped[state]) for state in states}
    partition: dict[PredictiveState, int] = {state: 0 for state in states}
    rounds = 0
    while True:
        rounds += 1
        signatures: dict[PredictiveState, tuple[tuple[str, int, int, int], ...]] = {}
        for state in states:
            counts: Counter[tuple[str, int]] = Counter()
            for row in by_state[state]:
                successor_class = _successor_partition(row, partition)
                counts[(row.outcome.sha256, successor_class)] += 1
            signatures[state] = _fraction_signature(counts)
        unique = tuple(sorted(set(signatures.values()), key=canonical_json_bytes))
        signature_class = {signature: index for index, signature in enumerate(unique)}
        updated = {state: signature_class[signatures[state]] for state in states}
        if _same_partition(partition, updated):
            partition = updated
            break
        partition = updated
        if rounds > len(states):
            raise PredictiveQuotientIntegrityError(
                "partition refinement exceeded the finite-state bound"
            )

    final_signatures: dict[int, tuple[tuple[str, int, int, int], ...]] = {}
    for state in states:
        counts = Counter()
        for row in by_state[state]:
            successor_class = _successor_partition(row, partition)
            counts[(row.outcome.sha256, successor_class)] += 1
        signature = _fraction_signature(counts)
        class_index = partition[state]
        prior = final_signatures.get(class_index)
        if prior is not None and prior != signature:
            raise PredictiveQuotientIntegrityError(
                "refined class contains unequal predictive distributions"
            )
        final_signatures[class_index] = signature

    class_ids = {
        index: _digest(
            {
                "final_block_index": index,
                "outcome_schema_sha256": PREDICTIVE_OUTCOME_SCHEMA_SHA256,
                "signature": [
                    {
                        "denominator": denominator,
                        "numerator": numerator,
                        "outcome_sha256": outcome,
                        "successor_block_index": (None if successor < 0 else successor),
                        "terminal": successor == _TERMINAL_PARTITION,
                        "censored": successor == _CENSORED_PARTITION,
                    }
                    for outcome, successor, numerator, denominator in signature
                ],
                "censored_state_sha256": CENSORED_STATE_SHA256,
                "terminal_state_sha256": TERMINAL_STATE_SHA256,
                "schema": "immer-ooe-predictive-class-identity/v1",
            }
        )
        for index, signature in final_signatures.items()
    }
    assignments = tuple(
        StateClassAssignment(state=state, class_sha256=class_ids[partition[state]])
        for state in states
    )
    members_by_class: dict[int, list[PredictiveState]] = {
        index: [] for index in final_signatures
    }
    for state in states:
        members_by_class[partition[state]].append(state)
    classes: list[PredictiveClass] = []
    for class_index in sorted(final_signatures):
        members = tuple(members_by_class[class_index])
        counts: Counter[tuple[str, str, bool, bool]] = Counter()
        for state in members:
            for row in by_state[state]:
                successor_class_sha256 = (
                    TERMINAL_STATE_SHA256
                    if row.terminal
                    else (
                        CENSORED_STATE_SHA256
                        if row.censored
                        else class_ids[
                            partition[cast(PredictiveState, row.successor_state)]
                        ]
                    )
                )
                counts[
                    (
                        row.outcome.sha256,
                        successor_class_sha256,
                        row.terminal,
                        row.censored,
                    )
                ] += 1
        total = sum(counts.values())
        probabilities = []
        for (outcome, successor, terminal, censored), count in counts.items():
            probability = Fraction(count, total)
            probabilities.append(
                ExactJointProbability(
                    outcome_sha256=outcome,
                    successor_class_sha256=successor,
                    terminal=terminal,
                    censored=censored,
                    numerator=probability.numerator,
                    denominator=probability.denominator,
                )
            )
        classes.append(
            PredictiveClass(
                class_sha256=class_ids[class_index],
                member_states=members,
                joint_distribution=tuple(probabilities),
            )
        )
    return _RefinementResult(
        assignments=assignments,
        classes=tuple(sorted(classes, key=lambda row: row.class_sha256)),
        rounds=rounds,
    )


@dataclass(frozen=True, slots=True)
class PredictiveQuotient:
    transitions: tuple[PredictiveTransition, ...]
    assignments: tuple[StateClassAssignment, ...]
    classes: tuple[PredictiveClass, ...]
    refinement_rounds: int
    outcome_schema_sha256: str = PREDICTIVE_OUTCOME_SCHEMA_SHA256
    terminal_state_sha256: str = TERMINAL_STATE_SHA256
    censored_state_sha256: str = CENSORED_STATE_SHA256

    def __post_init__(self) -> None:
        transitions = _normalize_transitions(self.transitions)
        assignments = _bounded_sequence(
            self.assignments,
            field="assignments",
            maximum=MAX_PREDICTIVE_STATES,
        )
        classes = _bounded_sequence(
            self.classes,
            field="classes",
            maximum=MAX_PREDICTIVE_CLASSES,
        )
        if any(not isinstance(row, StateClassAssignment) for row in assignments):
            raise TypeError("assignments must contain StateClassAssignment values")
        if any(not isinstance(row, PredictiveClass) for row in classes):
            raise TypeError("classes must contain PredictiveClass values")
        assignments = tuple(sorted(assignments, key=lambda row: row.state.sha256))
        classes = tuple(sorted(classes, key=lambda row: row.class_sha256))
        if len({row.state.sha256 for row in assignments}) != len(assignments):
            raise PredictiveQuotientIntegrityError("state assignment is duplicated")
        if len({row.class_sha256 for row in classes}) != len(classes):
            raise PredictiveQuotientIntegrityError("predictive class is duplicated")
        rounds = _uint(
            self.refinement_rounds,
            field="refinement_rounds",
            positive=True,
        )
        outcome_schema = require_sha256(
            self.outcome_schema_sha256, field="outcome_schema_sha256"
        )
        terminal = require_sha256(
            self.terminal_state_sha256, field="terminal_state_sha256"
        )
        censored = require_sha256(
            self.censored_state_sha256, field="censored_state_sha256"
        )
        if outcome_schema != PREDICTIVE_OUTCOME_SCHEMA_SHA256:
            raise PredictiveQuotientIntegrityError("predictive outcome schema changed")
        if terminal != TERMINAL_STATE_SHA256:
            raise PredictiveQuotientIntegrityError("terminal state identity changed")
        if censored != CENSORED_STATE_SHA256:
            raise PredictiveQuotientIntegrityError("censored state identity changed")
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "assignments", assignments)
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "refinement_rounds", rounds)
        object.__setattr__(self, "outcome_schema_sha256", outcome_schema)
        object.__setattr__(self, "terminal_state_sha256", terminal)
        object.__setattr__(self, "censored_state_sha256", censored)
        self.verify_or_raise()

    @property
    def state_count(self) -> int:
        return len(self.assignments)

    @property
    def class_count(self) -> int:
        return len(self.classes)

    @property
    def observation_count(self) -> int:
        return len(self.transitions)

    @property
    def compression(self) -> Fraction:
        return Fraction(self.state_count, self.class_count)

    @property
    def state_to_class(self) -> dict[PredictiveState, str]:
        return {row.state: row.class_sha256 for row in self.assignments}

    @property
    def class_by_sha256(self) -> dict[str, PredictiveClass]:
        return {row.class_sha256: row for row in self.classes}

    def as_record(self) -> dict[str, Any]:
        return {
            "assignments": [row.to_dict() for row in self.assignments],
            "class_count": self.class_count,
            "classes": [row.to_dict() for row in self.classes],
            "censored_observation_count": sum(
                1 for row in self.transitions if row.censored
            ),
            "censored_state": CENSORED_STATE,
            "censored_state_sha256": self.censored_state_sha256,
            "compression": {
                "denominator": self.compression.denominator,
                "numerator": self.compression.numerator,
            },
            "learning_receipt_sha256s": sorted(
                row.learning_receipt_sha256 for row in self.transitions
            ),
            "observation_count": self.observation_count,
            "outcome_schema_sha256": self.outcome_schema_sha256,
            "refinement_rounds": self.refinement_rounds,
            "state_count": self.state_count,
            "terminal_observation_count": sum(
                1 for row in self.transitions if row.terminal
            ),
            "terminal_state": TERMINAL_STATE,
            "terminal_state_sha256": self.terminal_state_sha256,
            "trace_sha256s": sorted(row.trace_sha256 for row in self.transitions),
            "transitions": [row.to_document() for row in self.transitions],
        }

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())

    def to_document(self) -> dict[str, Any]:
        document = _seal(PREDICTIVE_QUOTIENT_SCHEMA, self.as_record())
        _strict_document_bytes(document, label="predictive quotient")
        return document

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "PredictiveQuotient":
        _strict_document_bytes(document, label="predictive quotient")
        body = _unseal(
            document,
            schema=PREDICTIVE_QUOTIENT_SCHEMA,
            fields=frozenset(
                {
                    "assignments",
                    "class_count",
                    "classes",
                    "censored_observation_count",
                    "censored_state",
                    "censored_state_sha256",
                    "compression",
                    "learning_receipt_sha256s",
                    "observation_count",
                    "outcome_schema_sha256",
                    "refinement_rounds",
                    "state_count",
                    "terminal_observation_count",
                    "terminal_state",
                    "terminal_state_sha256",
                    "trace_sha256s",
                    "transitions",
                }
            ),
            label="predictive quotient",
        )
        transitions = tuple(
            PredictiveTransition.from_document(row) for row in body.pop("transitions")
        )
        assignments = tuple(
            StateClassAssignment.from_dict(row) for row in body.pop("assignments")
        )
        classes = tuple(PredictiveClass.from_dict(row) for row in body.pop("classes"))
        quotient = cls(
            transitions=transitions,
            assignments=assignments,
            classes=classes,
            refinement_rounds=body["refinement_rounds"],
            outcome_schema_sha256=body["outcome_schema_sha256"],
            terminal_state_sha256=body["terminal_state_sha256"],
            censored_state_sha256=body["censored_state_sha256"],
        )
        if quotient.as_record() != json.loads(
            canonical_json_bytes(
                body
                | {
                    "assignments": [row.to_dict() for row in assignments],
                    "classes": [row.to_dict() for row in classes],
                    "transitions": [row.to_document() for row in transitions],
                }
            )
        ):
            raise PredictiveQuotientIntegrityError(
                "predictive quotient derived fields do not replay exactly"
            )
        return quotient

    def verify_or_raise(self) -> bool:
        expected = _refine(self.transitions)
        if (
            expected.assignments != self.assignments
            or expected.classes != self.classes
            or expected.rounds != self.refinement_rounds
        ):
            raise PredictiveQuotientIntegrityError(
                "predictive quotient mapping does not recompute exactly"
            )
        assigned_states = {row.state for row in self.assignments}
        source_states = {row.state for row in self.transitions}
        if assigned_states != source_states:
            raise PredictiveQuotientIntegrityError(
                "predictive quotient assignment coverage is incomplete"
            )
        return True


def build_predictive_quotient(
    transitions: Sequence[PredictiveTransition],
) -> PredictiveQuotient:
    rows = _normalize_transitions(transitions)
    refined = _refine(rows)
    return PredictiveQuotient(
        transitions=rows,
        assignments=refined.assignments,
        classes=refined.classes,
        refinement_rounds=refined.rounds,
    )


def verify_predictive_quotient(quotient: PredictiveQuotient) -> bool:
    if not isinstance(quotient, PredictiveQuotient):
        raise TypeError("quotient must be a PredictiveQuotient")
    return quotient.verify_or_raise()


@dataclass(frozen=True, slots=True, order=True)
class StateTotalVariation:
    state: PredictiveState
    class_sha256: str
    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if not isinstance(self.state, PredictiveState):
            raise TypeError("state must be PredictiveState")
        object.__setattr__(
            self,
            "class_sha256",
            require_sha256(self.class_sha256, field="class_sha256"),
        )
        numerator = _uint(self.numerator, field="numerator")
        denominator = _uint(self.denominator, field="denominator", positive=True)
        if numerator > denominator:
            raise PredictiveQuotientError("total variation exceeds one")
        reduced = Fraction(numerator, denominator)
        if (reduced.numerator, reduced.denominator) != (numerator, denominator):
            raise PredictiveQuotientIntegrityError(
                "total variation ratio must be canonically reduced"
            )

    @property
    def fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_sha256": self.class_sha256,
            "denominator": self.denominator,
            "numerator": self.numerator,
            "state": self.state.to_dict(),
            "state_sha256": self.state.sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StateTotalVariation":
        if not isinstance(value, Mapping) or set(value) != {
            "class_sha256",
            "denominator",
            "numerator",
            "state",
            "state_sha256",
        }:
            raise PredictiveQuotientIntegrityError("state TV row is invalid")
        state = PredictiveState.from_dict(value["state"])
        if value["state_sha256"] != state.sha256:
            raise PredictiveQuotientIntegrityError("state TV identity mismatch")
        return cls(
            state=state,
            class_sha256=value["class_sha256"],
            numerator=value["numerator"],
            denominator=value["denominator"],
        )


@dataclass(frozen=True, slots=True)
class PredictiveValidationReceipt:
    quotient_sha256: str
    heldout_transitions: tuple[PredictiveTransition, ...]
    state_total_variations: tuple[StateTotalVariation, ...]
    max_total_variation_numerator: int
    max_total_variation_denominator: int
    training_state_count: int
    class_count: int
    heldout_state_count: int
    heldout_class_count: int
    state_coverage_numerator: int
    state_coverage_denominator: int
    class_coverage_numerator: int
    class_coverage_denominator: int
    require_full_state_coverage: bool
    compression_numerator: int
    compression_denominator: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "quotient_sha256",
            require_sha256(self.quotient_sha256, field="quotient_sha256"),
        )
        transitions = _normalize_transitions(
            self.heldout_transitions, require_successor_coverage=False
        )
        tv_rows = _bounded_sequence(
            self.state_total_variations,
            field="state_total_variations",
            maximum=MAX_PREDICTIVE_STATES,
        )
        if any(not isinstance(row, StateTotalVariation) for row in tv_rows):
            raise TypeError(
                "state_total_variations must contain StateTotalVariation values"
            )
        tv_rows = tuple(sorted(tv_rows, key=lambda row: row.state.sha256))
        if len({row.state for row in tv_rows}) != len(tv_rows):
            raise PredictiveQuotientIntegrityError("validation state is duplicated")
        maximum = Fraction(
            _uint(
                self.max_total_variation_numerator,
                field="max_total_variation_numerator",
            ),
            _uint(
                self.max_total_variation_denominator,
                field="max_total_variation_denominator",
                positive=True,
            ),
        )
        if maximum > 1 or (
            maximum.numerator,
            maximum.denominator,
        ) != (
            self.max_total_variation_numerator,
            self.max_total_variation_denominator,
        ):
            raise PredictiveQuotientIntegrityError(
                "maximum TV ratio is invalid or unreduced"
            )
        if maximum != max((row.fraction for row in tv_rows), default=Fraction()):
            raise PredictiveQuotientIntegrityError(
                "maximum TV does not match per-state receipts"
            )
        training_states = _uint(
            self.training_state_count, field="training_state_count", positive=True
        )
        class_count = _uint(self.class_count, field="class_count", positive=True)
        heldout_states = _uint(
            self.heldout_state_count, field="heldout_state_count", positive=True
        )
        heldout_classes = _uint(
            self.heldout_class_count, field="heldout_class_count", positive=True
        )
        if not isinstance(self.require_full_state_coverage, bool):
            raise TypeError("require_full_state_coverage must be bool")
        if heldout_states != len(tv_rows) or heldout_classes != len(
            {row.class_sha256 for row in tv_rows}
        ):
            raise PredictiveQuotientIntegrityError(
                "held-out state/class counts do not match TV rows"
            )
        state_coverage = Fraction(
            _uint(
                self.state_coverage_numerator,
                field="state_coverage_numerator",
                positive=True,
            ),
            _uint(
                self.state_coverage_denominator,
                field="state_coverage_denominator",
                positive=True,
            ),
        )
        class_coverage = Fraction(
            _uint(
                self.class_coverage_numerator,
                field="class_coverage_numerator",
                positive=True,
            ),
            _uint(
                self.class_coverage_denominator,
                field="class_coverage_denominator",
                positive=True,
            ),
        )
        if (
            state_coverage != Fraction(heldout_states, training_states)
            or class_coverage != Fraction(heldout_classes, class_count)
            or (state_coverage.numerator, state_coverage.denominator)
            != (self.state_coverage_numerator, self.state_coverage_denominator)
            or (class_coverage.numerator, class_coverage.denominator)
            != (self.class_coverage_numerator, self.class_coverage_denominator)
        ):
            raise PredictiveQuotientIntegrityError(
                "held-out coverage ratios are invalid"
            )
        if state_coverage > 1 or class_coverage > 1:
            raise PredictiveQuotientIntegrityError(
                "held-out coverage exceeds the trained quotient"
            )
        if self.require_full_state_coverage and state_coverage != 1:
            raise PredictiveQuotientCoverageError(
                "validation policy requires every trained state in held-out evidence"
            )
        compression = Fraction(
            _uint(
                self.compression_numerator,
                field="compression_numerator",
                positive=True,
            ),
            _uint(
                self.compression_denominator,
                field="compression_denominator",
                positive=True,
            ),
        )
        if compression != Fraction(training_states, class_count) or (
            compression.numerator,
            compression.denominator,
        ) != (self.compression_numerator, self.compression_denominator):
            raise PredictiveQuotientIntegrityError(
                "validation compression ratio is invalid"
            )
        object.__setattr__(self, "heldout_transitions", transitions)
        object.__setattr__(self, "state_total_variations", tv_rows)
        object.__setattr__(self, "training_state_count", training_states)
        object.__setattr__(self, "class_count", class_count)
        object.__setattr__(self, "heldout_state_count", heldout_states)
        object.__setattr__(self, "heldout_class_count", heldout_classes)

    @property
    def max_total_variation(self) -> Fraction:
        return Fraction(
            self.max_total_variation_numerator,
            self.max_total_variation_denominator,
        )

    @property
    def compression(self) -> Fraction:
        return Fraction(self.compression_numerator, self.compression_denominator)

    @property
    def state_coverage(self) -> Fraction:
        return Fraction(self.state_coverage_numerator, self.state_coverage_denominator)

    @property
    def class_coverage(self) -> Fraction:
        return Fraction(self.class_coverage_numerator, self.class_coverage_denominator)

    def as_record(self) -> dict[str, Any]:
        return {
            "class_count": self.class_count,
            "compression": {
                "denominator": self.compression_denominator,
                "numerator": self.compression_numerator,
            },
            "coverage": {
                "class": {
                    "denominator": self.class_coverage_denominator,
                    "numerator": self.class_coverage_numerator,
                },
                "heldout_class_count": self.heldout_class_count,
                "heldout_state_count": self.heldout_state_count,
                "require_full_state_coverage": self.require_full_state_coverage,
                "state": {
                    "denominator": self.state_coverage_denominator,
                    "numerator": self.state_coverage_numerator,
                },
            },
            "heldout_learning_receipt_sha256s": sorted(
                row.learning_receipt_sha256 for row in self.heldout_transitions
            ),
            "heldout_observation_count": len(self.heldout_transitions),
            "heldout_trace_sha256s": sorted(
                row.trace_sha256 for row in self.heldout_transitions
            ),
            "heldout_transitions": [
                row.to_document() for row in self.heldout_transitions
            ],
            "max_total_variation": {
                "denominator": self.max_total_variation_denominator,
                "numerator": self.max_total_variation_numerator,
            },
            "quotient_sha256": self.quotient_sha256,
            "state_total_variations": [
                row.to_dict() for row in self.state_total_variations
            ],
            "training_state_count": self.training_state_count,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())

    def to_document(self) -> dict[str, Any]:
        document = _seal(PREDICTIVE_VALIDATION_SCHEMA, self.as_record())
        _strict_document_bytes(document, label="predictive validation")
        return document

    @classmethod
    def from_document(
        cls, document: Mapping[str, Any]
    ) -> "PredictiveValidationReceipt":
        _strict_document_bytes(document, label="predictive validation")
        body = _unseal(
            document,
            schema=PREDICTIVE_VALIDATION_SCHEMA,
            fields=frozenset(
                {
                    "class_count",
                    "compression",
                    "coverage",
                    "heldout_learning_receipt_sha256s",
                    "heldout_observation_count",
                    "heldout_trace_sha256s",
                    "heldout_transitions",
                    "max_total_variation",
                    "quotient_sha256",
                    "state_total_variations",
                    "training_state_count",
                }
            ),
            label="predictive validation",
        )
        transitions = tuple(
            PredictiveTransition.from_document(row)
            for row in body["heldout_transitions"]
        )
        tv_rows = tuple(
            StateTotalVariation.from_dict(row) for row in body["state_total_variations"]
        )
        compression = body["compression"]
        coverage = body["coverage"]
        maximum = body["max_total_variation"]
        if not isinstance(compression, Mapping) or set(compression) != {
            "denominator",
            "numerator",
        }:
            raise PredictiveQuotientIntegrityError(
                "validation compression record is invalid"
            )
        if not isinstance(maximum, Mapping) or set(maximum) != {
            "denominator",
            "numerator",
        }:
            raise PredictiveQuotientIntegrityError("validation TV record is invalid")
        if not isinstance(coverage, Mapping) or set(coverage) != {
            "class",
            "heldout_class_count",
            "heldout_state_count",
            "require_full_state_coverage",
            "state",
        }:
            raise PredictiveQuotientIntegrityError(
                "validation coverage record is invalid"
            )
        state_coverage = coverage["state"]
        class_coverage = coverage["class"]
        if not isinstance(state_coverage, Mapping) or set(state_coverage) != {
            "denominator",
            "numerator",
        }:
            raise PredictiveQuotientIntegrityError(
                "validation state coverage record is invalid"
            )
        if not isinstance(class_coverage, Mapping) or set(class_coverage) != {
            "denominator",
            "numerator",
        }:
            raise PredictiveQuotientIntegrityError(
                "validation class coverage record is invalid"
            )
        receipt = cls(
            quotient_sha256=body["quotient_sha256"],
            heldout_transitions=transitions,
            state_total_variations=tv_rows,
            max_total_variation_numerator=maximum["numerator"],
            max_total_variation_denominator=maximum["denominator"],
            training_state_count=body["training_state_count"],
            class_count=body["class_count"],
            heldout_state_count=coverage["heldout_state_count"],
            heldout_class_count=coverage["heldout_class_count"],
            state_coverage_numerator=state_coverage["numerator"],
            state_coverage_denominator=state_coverage["denominator"],
            class_coverage_numerator=class_coverage["numerator"],
            class_coverage_denominator=class_coverage["denominator"],
            require_full_state_coverage=coverage["require_full_state_coverage"],
            compression_numerator=compression["numerator"],
            compression_denominator=compression["denominator"],
        )
        if receipt.as_record() != body:
            raise PredictiveQuotientIntegrityError(
                "predictive validation derived fields do not replay exactly"
            )
        return receipt


def _distribution_map(
    values: Sequence[ExactJointProbability],
) -> dict[tuple[str, str, bool, bool], Fraction]:
    return {
        (
            row.outcome_sha256,
            row.successor_class_sha256,
            row.terminal,
            row.censored,
        ): row.fraction
        for row in values
    }


def _heldout_state_distribution(
    rows: Sequence[PredictiveTransition],
    assignment: Mapping[PredictiveState, str],
) -> dict[tuple[str, str, bool, bool], Fraction]:
    counts: Counter[tuple[str, str, bool, bool]] = Counter()
    for row in rows:
        successor_class = (
            TERMINAL_STATE_SHA256
            if row.terminal
            else (
                CENSORED_STATE_SHA256
                if row.censored
                else assignment[cast(PredictiveState, row.successor_state)]
            )
        )
        counts[(row.outcome.sha256, successor_class, row.terminal, row.censored)] += 1
    total = sum(counts.values())
    return {key: Fraction(count, total) for key, count in counts.items()}


def _total_variation(
    first: Mapping[tuple[str, str, bool, bool], Fraction],
    second: Mapping[tuple[str, str, bool, bool], Fraction],
) -> Fraction:
    keys = set(first) | set(second)
    return (
        sum(
            (
                abs(first.get(key, Fraction()) - second.get(key, Fraction()))
                for key in keys
            ),
            Fraction(),
        )
        / 2
    )


def validate_predictive_quotient(
    quotient: PredictiveQuotient,
    heldout_transitions: Sequence[PredictiveTransition],
    *,
    require_full_state_coverage: bool = True,
) -> PredictiveValidationReceipt:
    if not isinstance(quotient, PredictiveQuotient):
        raise TypeError("quotient must be a PredictiveQuotient")
    if not isinstance(require_full_state_coverage, bool):
        raise TypeError("require_full_state_coverage must be bool")
    quotient.verify_or_raise()
    rows = _normalize_transitions(heldout_transitions, require_successor_coverage=False)
    assignment = quotient.state_to_class
    for row in rows:
        if row.state not in assignment:
            raise PredictiveQuotientCoverageError(
                "held-out source state is absent from quotient"
            )
        if row.successor_state is not None and row.successor_state not in assignment:
            raise PredictiveQuotientCoverageError(
                "held-out successor state is absent from quotient"
            )
    by_state: dict[PredictiveState, tuple[PredictiveTransition, ...]] = {
        state: tuple(row for row in rows if row.state == state)
        for state in sorted({row.state for row in rows}, key=lambda value: value.sha256)
    }
    if require_full_state_coverage and set(by_state) != set(assignment):
        raise PredictiveQuotientCoverageError(
            "held-out evidence does not cover every trained predictive state"
        )
    classes = quotient.class_by_sha256
    tv_rows = []
    for state, state_rows in by_state.items():
        class_sha256 = assignment[state]
        expected = _distribution_map(classes[class_sha256].joint_distribution)
        observed = _heldout_state_distribution(state_rows, assignment)
        total_variation = _total_variation(expected, observed)
        tv_rows.append(
            StateTotalVariation(
                state=state,
                class_sha256=class_sha256,
                numerator=total_variation.numerator,
                denominator=total_variation.denominator,
            )
        )
    maximum = max((row.fraction for row in tv_rows), default=Fraction())
    compression = quotient.compression
    heldout_state_count = len(tv_rows)
    heldout_class_count = len({row.class_sha256 for row in tv_rows})
    state_coverage = Fraction(heldout_state_count, quotient.state_count)
    class_coverage = Fraction(heldout_class_count, quotient.class_count)
    return PredictiveValidationReceipt(
        quotient_sha256=quotient.sha256,
        heldout_transitions=rows,
        state_total_variations=tuple(tv_rows),
        max_total_variation_numerator=maximum.numerator,
        max_total_variation_denominator=maximum.denominator,
        training_state_count=quotient.state_count,
        class_count=quotient.class_count,
        heldout_state_count=heldout_state_count,
        heldout_class_count=heldout_class_count,
        state_coverage_numerator=state_coverage.numerator,
        state_coverage_denominator=state_coverage.denominator,
        class_coverage_numerator=class_coverage.numerator,
        class_coverage_denominator=class_coverage.denominator,
        require_full_state_coverage=require_full_state_coverage,
        compression_numerator=compression.numerator,
        compression_denominator=compression.denominator,
    )


def verify_predictive_validation(
    quotient: PredictiveQuotient,
    heldout_transitions: Sequence[PredictiveTransition],
    receipt: PredictiveValidationReceipt,
) -> bool:
    if not isinstance(receipt, PredictiveValidationReceipt):
        raise TypeError("receipt must be PredictiveValidationReceipt")
    expected = validate_predictive_quotient(
        quotient,
        heldout_transitions,
        require_full_state_coverage=receipt.require_full_state_coverage,
    )
    if expected != receipt:
        raise PredictiveQuotientIntegrityError(
            "predictive validation receipt does not recompute exactly"
        )
    return True


__all__ = [
    "CENSORED_STATE",
    "CENSORED_STATE_SHA256",
    "MAX_PREDICTIVE_CLASSES",
    "MAX_PREDICTIVE_STATES",
    "MAX_PREDICTIVE_TRANSITIONS",
    "PREDICTIVE_OUTCOME_SCHEMA",
    "PREDICTIVE_OUTCOME_SCHEMA_SHA256",
    "PREDICTIVE_QUOTIENT_SCHEMA",
    "PREDICTIVE_TRANSITION_SCHEMA",
    "PREDICTIVE_VALIDATION_SCHEMA",
    "TERMINAL_STATE",
    "TERMINAL_STATE_SHA256",
    "ExactJointProbability",
    "PredictiveClass",
    "PredictiveQuotient",
    "PredictiveQuotientCoverageError",
    "PredictiveQuotientError",
    "PredictiveQuotientIntegrityError",
    "PredictiveState",
    "PredictiveTransition",
    "PredictiveValidationReceipt",
    "StateClassAssignment",
    "StateTotalVariation",
    "VerifiedExecutionOutcome",
    "build_predictive_quotient",
    "derive_execution_transitions",
    "derive_local_agent_transitions",
    "validate_predictive_quotient",
    "verify_predictive_quotient",
    "verify_predictive_validation",
]
