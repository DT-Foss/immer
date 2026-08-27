"""Verified execution outcomes as the only teacher for the OoE controller.

The bridge closes the production learning loop without accepting a caller-
authored label.  It authenticates an active ``MeasurementReceipt`` in the
live semantic Atlas, enriches the numeric receipt with exact action authority,
executes the registered runtime action, applies a separate named quality
verifier, and derives the teacher target from ``ActionExecution.action``.

Publication is a two-phase transaction.  An immutable intent is written
before the controller snapshot CAS.  Only after that CAS succeeds may the
append-only trace head advance.  A retry completes an interrupted intent and
never executes or learns the same transaction twice.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, cast

from immer.runtimes.qwen3_8.semantic_atlas import (
    MeasurementReceipt,
    SemanticAtlasError,
    SemanticWeightAtlas,
)

from .controller import (
    CONTROLLER_STATE_NAME,
    ActionExecution,
    ActionExecutor,
    ExecutionQualityVerifier,
    OoeController,
    VerifiedTeacherTransition,
)
from .crystal import CrystalStore, CrystalStoreError, ManifestConflictError
from .identity import canonical_json_bytes, require_sha256
from .qwen_bridge import (
    DEFAULT_FEATURE_DIMENSIONS,
    OoeAction,
    QwenOoeFeatureReceipt,
    validate_action,
)


ACTION_AUTHORITY_SCHEMA = "immer-ooe-action-authority/v1"
EXECUTION_QUALITY_SCHEMA = "immer-ooe-execution-quality/v1"
EXECUTION_LEARNING_SCHEMA = "immer-ooe-execution-learning/v1"
CONTROLLER_ACTION_TRACE_SCHEMA = "immer-ooe-controller-action-trace/v1"
EXECUTION_LEARNING_STATE_SCHEMA = "immer-ooe-execution-learning-state/v1"
EXECUTION_LEARNING_INTENT_SCHEMA = "immer-ooe-execution-learning-intent/v1"
ATLAS_MEASUREMENT_VERIFIER_SHA256 = hashlib.sha256(
    canonical_json_bytes(
        {
            "contract": "active-primary+full-atlas-audit+revision-membership",
            "schema": "immer-ooe-atlas-measurement-verifier/v1",
        }
    )
).hexdigest()
ZERO_SHA256 = "0" * 64
MAX_LEARNING_RECEIPTS = 1_000_000
MAX_LEARNING_STATE_BYTES = 64 * 1024 * 1024

_HEAD_STATE = "ooe-execution-learning-head/v1"
_HISTORY_PREFIX = "ooe-execution-learning-history/v1:"
_COMMIT_PREFIX = "ooe-execution-learning-commit/v1:"
_RECEIPT_PREFIX = "ooe-execution-learning-receipt/v1:"
_TRACE_PREFIX = "ooe-execution-learning-trace/v1:"
_INTENT_PREFIX = "ooe-execution-learning-intent/v1:"
_BANK_LOCK = "OOE-EXECUTION-LEARNING-LOCK"
_STATE_NAME_RE = re.compile(rb'"name":"([^"\\]+)"')
_STATE_GENERATION_RE = re.compile(rb'"generation":([0-9]+)')


class ExecutionLearningError(RuntimeError):
    """The verified execution-learning contract cannot be completed."""


class ExecutionLearningIntegrityError(ExecutionLearningError):
    """A learning receipt, trace, authority, or persistent state was modified."""


class ExecutionLearningStaleError(ExecutionLearningIntegrityError):
    """A pin, temporal cursor, source action, or CAS head is stale."""


class ExecutionLearningExecutionError(ExecutionLearningError):
    """The selected concrete action executor failed before learning."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _text(value: object, *, field: str, maximum: int = 1024) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > maximum
    ):
        raise ValueError(f"{field} must be canonical non-empty text")
    return value


def _uint(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _weight(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("learning weight must be positive finite numeric data")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= 1_000_000.0:
        raise ValueError("learning weight must be positive and bounded")
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
        raise ExecutionLearningIntegrityError(f"{label} envelope is invalid")
    if document.get("schema") != schema:
        raise ExecutionLearningIntegrityError(f"{label} schema is invalid")
    body = document.get("body")
    if not isinstance(body, Mapping) or set(body) != fields:
        raise ExecutionLearningIntegrityError(f"{label} body is invalid")
    try:
        claimed = require_sha256(document.get("sha256"), field=f"{label}.sha256")
    except ValueError as exc:
        raise ExecutionLearningIntegrityError(f"{label} seal is invalid") from exc
    if claimed != _digest(body):
        raise ExecutionLearningIntegrityError(f"{label} SHA-256 mismatch")
    return cast(dict[str, Any], json.loads(canonical_json_bytes(body)))


def _canonical_document_bytes(document: Mapping[str, Any], *, label: str) -> bytes:
    try:
        value = canonical_json_bytes(dict(document))
    except (TypeError, ValueError) as exc:
        raise ExecutionLearningIntegrityError(f"{label} is not canonical JSON") from exc
    if len(value) > MAX_LEARNING_STATE_BYTES:
        raise ExecutionLearningIntegrityError(f"{label} exceeds its byte bound")
    return value


def _atlas_authentication_sha256(
    measurement: MeasurementReceipt,
    *,
    atlas_head_sha256: str,
) -> str:
    return _digest(
        {
            "atlas_head_sha256": require_sha256(
                atlas_head_sha256, field="atlas_head_sha256"
            ),
            "atlas_model_pin_sha256": measurement.model_pin.sha256,
            "measurement_atlas_revision_sha256": (
                measurement.atlas_head_revision.sha256
            ),
            "measurement_sha256": measurement.sha256,
            "schema": "immer-ooe-live-atlas-authentication/v1",
            "verifier_sha256": ATLAS_MEASUREMENT_VERIFIER_SHA256,
        }
    )


@dataclass(frozen=True, slots=True)
class ActionAuthorityReceipt:
    """Pinned authority for one concrete executor and its two verifiers."""

    action: OoeAction | str
    model_pin_sha256: str
    weight_graph_revision_sha256: str
    executor_sha256: str
    action_verifier_name: str
    action_verifier_sha256: str
    quality_verifier_name: str
    quality_verifier_sha256: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", validate_action(self.action))
        for name in (
            "model_pin_sha256",
            "weight_graph_revision_sha256",
            "executor_sha256",
            "action_verifier_sha256",
            "quality_verifier_sha256",
            "evidence_sha256",
        ):
            object.__setattr__(
                self,
                name,
                require_sha256(getattr(self, name), field=name),
            )
        object.__setattr__(
            self,
            "action_verifier_name",
            _text(self.action_verifier_name, field="action_verifier_name"),
        )
        object.__setattr__(
            self,
            "quality_verifier_name",
            _text(self.quality_verifier_name, field="quality_verifier_name"),
        )
        if self.action_verifier_name == self.quality_verifier_name:
            raise ValueError("action and quality verifiers must be separately named")

    def as_record(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "action_verifier_name": self.action_verifier_name,
            "action_verifier_sha256": self.action_verifier_sha256,
            "evidence_sha256": self.evidence_sha256,
            "executor_sha256": self.executor_sha256,
            "model_pin_sha256": self.model_pin_sha256,
            "quality_verifier_name": self.quality_verifier_name,
            "quality_verifier_sha256": self.quality_verifier_sha256,
            "weight_graph_revision_sha256": self.weight_graph_revision_sha256,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())

    def to_document(self) -> dict[str, Any]:
        return _seal(ACTION_AUTHORITY_SCHEMA, self.as_record())

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "ActionAuthorityReceipt":
        body = _unseal(
            document,
            schema=ACTION_AUTHORITY_SCHEMA,
            fields=frozenset(cls.__dataclass_fields__),
            label="action authority",
        )
        return cls(**body)

    def assert_current(
        self,
        *,
        model_pin_sha256: str,
        weight_graph_revision_sha256: str,
    ) -> None:
        if self.model_pin_sha256 != require_sha256(
            model_pin_sha256, field="model_pin_sha256"
        ) or self.weight_graph_revision_sha256 != require_sha256(
            weight_graph_revision_sha256,
            field="weight_graph_revision_sha256",
        ):
            raise ExecutionLearningStaleError("action authority pin is stale")


@dataclass(frozen=True, slots=True)
class ExecutionQualityReceipt:
    """Outcome of the separate named verifier over one bound execution."""

    feature_receipt_sha256: str
    execution_sha256: str
    execution_quality_sha256: str
    verifier_name: str
    verifier_sha256: str
    verified: bool

    def __post_init__(self) -> None:
        for name in (
            "feature_receipt_sha256",
            "execution_sha256",
            "execution_quality_sha256",
            "verifier_sha256",
        ):
            object.__setattr__(
                self,
                name,
                require_sha256(getattr(self, name), field=name),
            )
        object.__setattr__(
            self, "verifier_name", _text(self.verifier_name, field="verifier_name")
        )
        if not isinstance(self.verified, bool):
            raise TypeError("verified must be bool")

    def as_record(self) -> dict[str, Any]:
        return {
            "execution_quality_sha256": self.execution_quality_sha256,
            "execution_sha256": self.execution_sha256,
            "feature_receipt_sha256": self.feature_receipt_sha256,
            "verified": self.verified,
            "verifier_name": self.verifier_name,
            "verifier_sha256": self.verifier_sha256,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())

    def to_document(self) -> dict[str, Any]:
        return _seal(EXECUTION_QUALITY_SCHEMA, self.as_record())

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "ExecutionQualityReceipt":
        body = _unseal(
            document,
            schema=EXECUTION_QUALITY_SCHEMA,
            fields=frozenset(cls.__dataclass_fields__),
            label="execution quality",
        )
        return cls(**body)

    def assert_bound(
        self,
        feature: QwenOoeFeatureReceipt,
        execution: ActionExecution,
        authority: ActionAuthorityReceipt,
    ) -> None:
        if (
            self.feature_receipt_sha256 != feature.sha256
            or self.execution_sha256 != execution.sha256
            or self.execution_quality_sha256 != execution.quality_sha256
            or self.verifier_name != authority.quality_verifier_name
            or self.verifier_sha256 != authority.quality_verifier_sha256
        ):
            raise ExecutionLearningIntegrityError(
                "quality receipt does not bind feature, execution, and authority"
            )


@dataclass(frozen=True, slots=True)
class ExecutionLearningReceipt:
    """Complete canonical replay package for one verified learned action."""

    measurement: MeasurementReceipt
    action_authority: ActionAuthorityReceipt
    feature: QwenOoeFeatureReceipt
    execution: ActionExecution
    quality: ExecutionQualityReceipt
    transition: VerifiedTeacherTransition
    atlas_head_sha256: str
    atlas_authentication_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.measurement, MeasurementReceipt):
            raise TypeError("measurement must be a MeasurementReceipt")
        if not isinstance(self.action_authority, ActionAuthorityReceipt):
            raise TypeError("action_authority must be an ActionAuthorityReceipt")
        if not isinstance(self.feature, QwenOoeFeatureReceipt):
            raise TypeError("feature must be a QwenOoeFeatureReceipt")
        if not isinstance(self.execution, ActionExecution):
            raise TypeError("execution must be an ActionExecution")
        if not isinstance(self.quality, ExecutionQualityReceipt):
            raise TypeError("quality must be an ExecutionQualityReceipt")
        if not isinstance(self.transition, VerifiedTeacherTransition):
            raise TypeError("transition must be a VerifiedTeacherTransition")
        object.__setattr__(
            self,
            "atlas_head_sha256",
            require_sha256(self.atlas_head_sha256, field="atlas_head_sha256"),
        )
        object.__setattr__(
            self,
            "atlas_authentication_sha256",
            require_sha256(
                self.atlas_authentication_sha256,
                field="atlas_authentication_sha256",
            ),
        )
        self._validate_exact()

    def _validate_exact(self) -> None:
        authority = self.action_authority
        self.feature.validate_measurement(self.measurement)
        if self.atlas_authentication_sha256 != _atlas_authentication_sha256(
            self.measurement,
            atlas_head_sha256=self.atlas_head_sha256,
        ):
            raise ExecutionLearningIntegrityError(
                "Atlas authentication receipt is not exactly replayable"
            )
        authority.assert_current(
            model_pin_sha256=self.feature.model_pin_sha256,
            weight_graph_revision_sha256=self.feature.weight_graph_revision_sha256,
        )
        required_verifiers = {
            ATLAS_MEASUREMENT_VERIFIER_SHA256,
            self.atlas_authentication_sha256,
            authority.sha256,
            authority.action_verifier_sha256,
            authority.quality_verifier_sha256,
        }
        required_evidence = {
            self.measurement.evidence_sha256,
            self.measurement.access_trace_sha256,
            self.atlas_authentication_sha256,
            authority.sha256,
            authority.evidence_sha256,
        }
        if not required_verifiers.issubset(self.feature.verifier_sha256s):
            raise ExecutionLearningIntegrityError(
                "feature receipt lacks Atlas/action authority verifier hashes"
            )
        if not required_evidence.issubset(self.feature.evidence_sha256s):
            raise ExecutionLearningIntegrityError(
                "feature receipt lacks Atlas/action authority evidence hashes"
            )
        self.execution.assert_bound(self.feature, authority.action)
        if (
            self.execution.executor_sha256 != authority.executor_sha256
            or self.execution.verifier_sha256 != authority.action_verifier_sha256
            or self.execution.evidence_sha256 != authority.evidence_sha256
        ):
            raise ExecutionLearningIntegrityError(
                "execution does not match its action authority"
            )
        self.quality.assert_bound(self.feature, self.execution, authority)
        if not self.execution.quality_verified or not self.quality.verified:
            raise ExecutionLearningIntegrityError(
                "unverified execution cannot become learning evidence"
            )
        expected = VerifiedTeacherTransition(
            feature_receipt_sha256=self.feature.sha256,
            site_identity_sha256=self.feature.site_identity.sha256,
            source_action=self.transition.source_action,
            target_action=self.execution.action,
            verifier_sha256=authority.quality_verifier_sha256,
            evidence_sha256=authority.evidence_sha256,
            quality_sha256=self.quality.sha256,
            verified_quality=True,
            weight=self.transition.weight,
        )
        if self.transition != expected:
            raise ExecutionLearningIntegrityError(
                "teacher target is not derived exactly from execution.action"
            )
        self.transition.assert_bound(self.feature)

    def as_record(self) -> dict[str, Any]:
        return {
            "action_authority": self.action_authority.to_document(),
            "atlas_authentication_sha256": self.atlas_authentication_sha256,
            "atlas_head_sha256": self.atlas_head_sha256,
            "execution": self.execution.to_document(),
            "feature": self.feature.to_document(),
            "measurement": self.measurement.to_document(),
            "quality": self.quality.to_document(),
            "transition": self.transition.to_document(),
        }

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())

    def to_document(self) -> dict[str, Any]:
        return _seal(EXECUTION_LEARNING_SCHEMA, self.as_record())

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "ExecutionLearningReceipt":
        body = _unseal(
            document,
            schema=EXECUTION_LEARNING_SCHEMA,
            fields=frozenset(
                {
                    "action_authority",
                    "atlas_authentication_sha256",
                    "atlas_head_sha256",
                    "execution",
                    "feature",
                    "measurement",
                    "quality",
                    "transition",
                }
            ),
            label="execution learning receipt",
        )
        return cls(
            measurement=MeasurementReceipt.from_document(body["measurement"]),
            action_authority=ActionAuthorityReceipt.from_document(
                body["action_authority"]
            ),
            feature=QwenOoeFeatureReceipt.from_document(body["feature"]),
            execution=ActionExecution.from_document(body["execution"]),
            quality=ExecutionQualityReceipt.from_document(body["quality"]),
            transition=VerifiedTeacherTransition.from_document(body["transition"]),
            atlas_head_sha256=body["atlas_head_sha256"],
            atlas_authentication_sha256=body["atlas_authentication_sha256"],
        )


@dataclass(frozen=True, slots=True)
class ControllerActionTraceReceipt:
    """One append-only temporal/source-action transition after snapshot CAS."""

    transaction_sha256: str
    ordinal: int
    temporal_index: int
    previous_trace_sha256: str
    previous_stream_head_sha256: str
    measurement_sha256: str
    learning_receipt_sha256: str
    source_action: OoeAction | str
    target_action: OoeAction | str
    controller_snapshot_before_sha256: str
    controller_snapshot_after_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "transaction_sha256",
            "previous_trace_sha256",
            "previous_stream_head_sha256",
            "measurement_sha256",
            "learning_receipt_sha256",
            "controller_snapshot_before_sha256",
            "controller_snapshot_after_sha256",
        ):
            object.__setattr__(
                self,
                name,
                require_sha256(getattr(self, name), field=name),
            )
        object.__setattr__(self, "ordinal", _uint(self.ordinal, field="ordinal"))
        object.__setattr__(
            self,
            "temporal_index",
            _uint(self.temporal_index, field="temporal_index"),
        )
        object.__setattr__(self, "source_action", validate_action(self.source_action))
        object.__setattr__(self, "target_action", validate_action(self.target_action))
        if self.previous_trace_sha256 != self.previous_stream_head_sha256:
            raise ValueError("previous trace and stream head must be identical")
        if (
            self.controller_snapshot_before_sha256
            == self.controller_snapshot_after_sha256
        ):
            raise ValueError("learned controller snapshot must advance")

    def as_record(self) -> dict[str, Any]:
        return {
            "controller_snapshot_after_sha256": self.controller_snapshot_after_sha256,
            "controller_snapshot_before_sha256": self.controller_snapshot_before_sha256,
            "learning_receipt_sha256": self.learning_receipt_sha256,
            "measurement_sha256": self.measurement_sha256,
            "ordinal": self.ordinal,
            "previous_stream_head_sha256": self.previous_stream_head_sha256,
            "previous_trace_sha256": self.previous_trace_sha256,
            "source_action": self.source_action,
            "target_action": self.target_action,
            "temporal_index": self.temporal_index,
            "transaction_sha256": self.transaction_sha256,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())

    def to_document(self) -> dict[str, Any]:
        return _seal(CONTROLLER_ACTION_TRACE_SCHEMA, self.as_record())

    @classmethod
    def from_document(
        cls, document: Mapping[str, Any]
    ) -> "ControllerActionTraceReceipt":
        body = _unseal(
            document,
            schema=CONTROLLER_ACTION_TRACE_SCHEMA,
            fields=frozenset(cls.__dataclass_fields__),
            label="controller action trace",
        )
        return cls(**body)


@dataclass(frozen=True, slots=True)
class _ExecutionLearningState:
    generation: int
    previous_state_sha256: str
    stream_head_sha256: str
    receipts: tuple[ExecutionLearningReceipt, ...]
    traces: tuple[ControllerActionTraceReceipt, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "generation", _uint(self.generation, field="generation")
        )
        for name in ("previous_state_sha256", "stream_head_sha256"):
            object.__setattr__(
                self,
                name,
                require_sha256(getattr(self, name), field=name),
            )
        try:
            receipts = tuple(self.receipts)
            traces = tuple(self.traces)
        except TypeError as exc:
            raise ValueError("learning receipts/traces must be sequences") from exc
        if len(receipts) != len(traces) or len(receipts) > MAX_LEARNING_RECEIPTS:
            raise ValueError("learning receipt/trace cardinality is invalid")
        if any(not isinstance(row, ExecutionLearningReceipt) for row in receipts):
            raise TypeError("receipts must contain ExecutionLearningReceipt values")
        if any(not isinstance(row, ControllerActionTraceReceipt) for row in traces):
            raise TypeError("traces must contain ControllerActionTraceReceipt values")
        if self.generation != len(traces):
            raise ValueError("learning generation must equal trace count")
        if len({row.sha256 for row in receipts}) != len(receipts):
            raise ValueError("learning receipt is duplicated")
        if len({row.transaction_sha256 for row in traces}) != len(traces):
            raise ValueError("learning transaction is duplicated")
        previous_trace = ZERO_SHA256
        previous_temporal: int | None = None
        previous_snapshot_after: str | None = None
        for ordinal, (receipt, trace) in enumerate(zip(receipts, traces, strict=True)):
            if (
                trace.ordinal != ordinal
                or trace.previous_trace_sha256 != previous_trace
                or trace.learning_receipt_sha256 != receipt.sha256
                or trace.measurement_sha256 != receipt.measurement.sha256
                or trace.source_action != receipt.transition.source_action
                or trace.target_action != receipt.execution.action
                or (
                    previous_temporal is not None
                    and trace.temporal_index != previous_temporal + 1
                )
                or trace.temporal_index != receipt.feature.temporal_index
                or (
                    previous_snapshot_after is not None
                    and trace.controller_snapshot_before_sha256
                    != previous_snapshot_after
                )
            ):
                raise ExecutionLearningIntegrityError(
                    "learning temporal/source-action trace is discontinuous"
                )
            if ordinal and trace.source_action != traces[ordinal - 1].target_action:
                raise ExecutionLearningIntegrityError(
                    "source action does not continue the prior executed action"
                )
            previous_trace = trace.sha256
            previous_temporal = trace.temporal_index
            previous_snapshot_after = trace.controller_snapshot_after_sha256
        expected_head = ZERO_SHA256 if not traces else traces[-1].sha256
        if self.stream_head_sha256 != expected_head:
            raise ExecutionLearningIntegrityError("learning stream head is invalid")
        if self.generation == 0 and self.previous_state_sha256 != ZERO_SHA256:
            raise ValueError("initial learning state must have zero predecessor")
        if self.generation > 0 and self.previous_state_sha256 == ZERO_SHA256:
            raise ValueError("non-initial learning state requires a predecessor")
        object.__setattr__(self, "receipts", receipts)
        object.__setattr__(self, "traces", traces)

    @classmethod
    def initial(cls) -> "_ExecutionLearningState":
        return cls(0, ZERO_SHA256, ZERO_SHA256, (), ())

    def as_record(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "previous_state_sha256": self.previous_state_sha256,
            "receipts": [row.to_document() for row in self.receipts],
            "stream_head_sha256": self.stream_head_sha256,
            "traces": [row.to_document() for row in self.traces],
        }

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(
            _seal(EXECUTION_LEARNING_STATE_SCHEMA, self.as_record())
        )
        if len(data) > MAX_LEARNING_STATE_BYTES:
            raise ExecutionLearningIntegrityError(
                "learning state exceeds its byte bound"
            )
        return data

    @property
    def sha256(self) -> str:
        return _bytes_sha256(self.to_bytes())

    @classmethod
    def from_bytes(cls, data: bytes) -> "_ExecutionLearningState":
        if not isinstance(data, bytes) or len(data) > MAX_LEARNING_STATE_BYTES:
            raise ExecutionLearningIntegrityError("learning state bytes are invalid")
        try:
            document = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExecutionLearningIntegrityError(
                "learning state is invalid JSON"
            ) from exc
        if canonical_json_bytes(document) != data:
            raise ExecutionLearningIntegrityError(
                "learning state is not canonical JSON"
            )
        body = _unseal(
            document,
            schema=EXECUTION_LEARNING_STATE_SCHEMA,
            fields=frozenset(
                {
                    "generation",
                    "previous_state_sha256",
                    "receipts",
                    "stream_head_sha256",
                    "traces",
                }
            ),
            label="execution learning state",
        )
        return cls(
            generation=body["generation"],
            previous_state_sha256=body["previous_state_sha256"],
            stream_head_sha256=body["stream_head_sha256"],
            receipts=tuple(
                ExecutionLearningReceipt.from_document(row) for row in body["receipts"]
            ),
            traces=tuple(
                ControllerActionTraceReceipt.from_document(row)
                for row in body["traces"]
            ),
        )

    def appended(
        self,
        receipt: ExecutionLearningReceipt,
        trace: ControllerActionTraceReceipt,
    ) -> "_ExecutionLearningState":
        if (
            trace.ordinal != self.generation
            or trace.previous_trace_sha256 != self.stream_head_sha256
            or (self.traces and trace.source_action != self.traces[-1].target_action)
            or (
                self.traces
                and trace.temporal_index != self.traces[-1].temporal_index + 1
            )
        ):
            raise ExecutionLearningStaleError(
                "learning trace does not extend the current stream head"
            )
        return _ExecutionLearningState(
            generation=self.generation + 1,
            previous_state_sha256=self.sha256,
            stream_head_sha256=trace.sha256,
            receipts=self.receipts + (receipt,),
            traces=self.traces + (trace,),
        )


@dataclass(frozen=True, slots=True)
class _ExecutionLearningIntent:
    transaction_sha256: str
    bank_head_before_sha256: str
    receipt: ExecutionLearningReceipt
    trace: ControllerActionTraceReceipt

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "transaction_sha256",
            require_sha256(self.transaction_sha256, field="transaction_sha256"),
        )
        object.__setattr__(
            self,
            "bank_head_before_sha256",
            require_sha256(
                self.bank_head_before_sha256,
                field="bank_head_before_sha256",
            ),
        )
        if not isinstance(self.receipt, ExecutionLearningReceipt):
            raise TypeError("intent receipt must be ExecutionLearningReceipt")
        if not isinstance(self.trace, ControllerActionTraceReceipt):
            raise TypeError("intent trace must be ControllerActionTraceReceipt")
        if (
            self.trace.transaction_sha256 != self.transaction_sha256
            or self.trace.learning_receipt_sha256 != self.receipt.sha256
        ):
            raise ExecutionLearningIntegrityError("learning intent bindings differ")

    def as_record(self) -> dict[str, Any]:
        return {
            "bank_head_before_sha256": self.bank_head_before_sha256,
            "receipt": self.receipt.to_document(),
            "trace": self.trace.to_document(),
            "transaction_sha256": self.transaction_sha256,
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(
            _seal(EXECUTION_LEARNING_INTENT_SCHEMA, self.as_record())
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "_ExecutionLearningIntent":
        try:
            document = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExecutionLearningIntegrityError(
                "learning intent is invalid JSON"
            ) from exc
        if canonical_json_bytes(document) != data:
            raise ExecutionLearningIntegrityError(
                "learning intent is not canonical JSON"
            )
        body = _unseal(
            document,
            schema=EXECUTION_LEARNING_INTENT_SCHEMA,
            fields=frozenset(
                {"bank_head_before_sha256", "receipt", "trace", "transaction_sha256"}
            ),
            label="execution learning intent",
        )
        return cls(
            transaction_sha256=body["transaction_sha256"],
            bank_head_before_sha256=body["bank_head_before_sha256"],
            receipt=ExecutionLearningReceipt.from_document(body["receipt"]),
            trace=ControllerActionTraceReceipt.from_document(body["trace"]),
        )


class ExecutionLearningBank:
    """Crash-safe immutable receipt/trace history with one CAS stream head."""

    def __init__(
        self,
        store: CrystalStore | str | os.PathLike[str],
        *,
        trusted_head_sha256: str | None = None,
    ) -> None:
        self.store = store if isinstance(store, CrystalStore) else CrystalStore(store)
        self.root = Path(self.store.root)
        self.trusted_head_sha256 = (
            None
            if trusted_head_sha256 is None
            else require_sha256(trusted_head_sha256, field="trusted_head_sha256")
        )
        with self._locked():
            try:
                self._head_unlocked()
            except KeyError:
                self._initialize_unlocked()

    @staticmethod
    def _history_name(sha256: str) -> str:
        return _HISTORY_PREFIX + require_sha256(sha256, field="state_sha256")

    @staticmethod
    def _commit_name(sha256: str) -> str:
        return _COMMIT_PREFIX + require_sha256(sha256, field="state_sha256")

    @staticmethod
    def _receipt_name(sha256: str) -> str:
        return _RECEIPT_PREFIX + require_sha256(sha256, field="receipt_sha256")

    @staticmethod
    def _trace_name(sha256: str) -> str:
        return _TRACE_PREFIX + require_sha256(sha256, field="trace_sha256")

    @staticmethod
    def _intent_name(transaction_sha256: str) -> str:
        return _INTENT_PREFIX + require_sha256(
            transaction_sha256, field="transaction_sha256"
        )

    @contextmanager
    def _locked(self):
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.root / _BANK_LOCK, flags, 0o600)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ExecutionLearningIntegrityError(
                    "execution-learning lock is not a regular file"
                )
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _restore_optional(self, name: str) -> bytes | None:
        try:
            return self.store.restore_state(name)
        except KeyError:
            return None
        except CrystalStoreError as exc:
            raise ExecutionLearningIntegrityError(
                "execution-learning storage failed integrity validation"
            ) from exc

    def _publish_immutable(self, name: str, payload: bytes) -> None:
        prior = self._restore_optional(name)
        if prior is not None:
            if prior != payload:
                raise ExecutionLearningIntegrityError(
                    "immutable execution-learning object changed bytes"
                )
            return
        try:
            self.store.publish_state(name, payload)
        except CrystalStoreError as exc:
            raise ExecutionLearningIntegrityError(
                "immutable execution-learning publication failed"
            ) from exc
        if self._restore_optional(name) != payload:
            raise ExecutionLearningIntegrityError(
                "immutable execution-learning object failed roundtrip"
            )

    def _initialize_unlocked(self) -> _ExecutionLearningState:
        initial = _ExecutionLearningState.initial()
        self._publish_immutable(self._history_name(initial.sha256), initial.to_bytes())
        try:
            self.store.publish_state(_HEAD_STATE, initial.to_bytes())
        except CrystalStoreError as exc:
            raise ExecutionLearningIntegrityError(
                "execution-learning head initialization failed"
            ) from exc
        self._publish_immutable(
            self._commit_name(initial.sha256),
            canonical_json_bytes(
                {
                    "schema": EXECUTION_LEARNING_STATE_SCHEMA,
                    "state_sha256": initial.sha256,
                }
            ),
        )
        return initial

    def _head_unlocked(self) -> _ExecutionLearningState:
        payload = self._restore_optional(_HEAD_STATE)
        if payload is None:
            raise KeyError("execution-learning bank is not initialized")
        state = _ExecutionLearningState.from_bytes(payload)
        envelope_generation = self._head_envelope_generation_unlocked()
        if envelope_generation != state.generation + 1:
            raise ExecutionLearningIntegrityError(
                "execution-learning pointer generation proves a resealed rollback"
            )
        history = self._restore_optional(self._history_name(state.sha256))
        if history != payload:
            raise ExecutionLearningIntegrityError(
                "execution-learning head lacks exact immutable history"
            )
        commit_name = self._commit_name(state.sha256)
        commit = self._restore_optional(commit_name)
        expected_commit = canonical_json_bytes(
            {"schema": EXECUTION_LEARNING_STATE_SCHEMA, "state_sha256": state.sha256}
        )
        if commit is not None and commit != expected_commit:
            raise ExecutionLearningIntegrityError(
                "execution-learning commit marker is invalid"
            )
        if (
            self.trusted_head_sha256 is not None
            and self.trusted_head_sha256 != state.sha256
            and self.trusted_head_sha256
            not in {row.previous_state_sha256 for row in (state,)}
        ):
            # Full ancestry is checked during replay below; this fast rejection
            # prevents accepting a foreign one-hop head.
            self._replay_ancestry_unlocked(state, self.trusted_head_sha256)
        self._replay_ancestry_unlocked(state, None)
        self._audit_committed_history_unlocked(
            state,
            recoverable_missing_head_commit=commit is None,
        )
        if commit is None:
            # Recovery point: only a fully validated head may acquire the
            # commit marker whose CAS was interrupted.
            self._publish_immutable(commit_name, expected_commit)
        return state

    def _head_envelope_generation_unlocked(self) -> int:
        path = self.root / "state" / self.store._state_filename(_HEAD_STATE)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ExecutionLearningIntegrityError(
                    "execution-learning head pointer is not a regular file"
                )
            prefix = os.read(descriptor, 4096)
            after = os.fstat(descriptor)
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise ExecutionLearningIntegrityError(
                    "execution-learning head changed during generation audit"
                )
        finally:
            os.close(descriptor)
        name_match = _STATE_NAME_RE.search(prefix)
        generation_match = _STATE_GENERATION_RE.search(prefix)
        if (
            name_match is None
            or generation_match is None
            or name_match.group(1) != _HEAD_STATE.encode("ascii")
        ):
            raise ExecutionLearningIntegrityError(
                "execution-learning head envelope identity is invalid"
            )
        generation = int(generation_match.group(1))
        if generation < 1:
            raise ExecutionLearningIntegrityError(
                "execution-learning head envelope generation is invalid"
            )
        return generation

    def _history_state_names_unlocked(self) -> tuple[str, ...]:
        names: list[str] = []
        state_path = self.root / "state"
        for path in sorted(state_path.iterdir()):
            if not path.name.endswith(".state"):
                continue
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode):
                    raise ExecutionLearningIntegrityError(
                        "execution-learning state inventory contains a non-file"
                    )
                prefix = os.read(descriptor, 4096)
                after = os.fstat(descriptor)
                if (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                ) != (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ):
                    raise ExecutionLearningIntegrityError(
                        "execution-learning state changed during inventory"
                    )
            finally:
                os.close(descriptor)
            match = _STATE_NAME_RE.search(prefix)
            if match is None:
                continue
            try:
                name = match.group(1).decode("ascii")
            except UnicodeDecodeError:
                continue
            if not name.startswith((_HISTORY_PREFIX, _COMMIT_PREFIX)):
                continue
            if path.name != self.store._state_filename(name):
                raise ExecutionLearningIntegrityError(
                    "execution-learning history filename is not name-bound"
                )
            names.append(name)
        return tuple(names)

    def _audit_committed_history_unlocked(
        self,
        head: _ExecutionLearningState,
        *,
        recoverable_missing_head_commit: bool = False,
    ) -> None:
        if not isinstance(recoverable_missing_head_commit, bool):
            raise TypeError("recoverable_missing_head_commit must be bool")
        histories: dict[str, _ExecutionLearningState] = {}
        commits: set[str] = set()
        for name in self._history_state_names_unlocked():
            payload = self._restore_optional(name)
            if payload is None:
                raise ExecutionLearningIntegrityError(
                    "execution-learning history disappeared during inventory"
                )
            prefix, kind = (
                (_HISTORY_PREFIX, "history")
                if name.startswith(_HISTORY_PREFIX)
                else (_COMMIT_PREFIX, "commit")
            )
            address = require_sha256(name[len(prefix) :], field=f"{kind} address")
            if kind == "history":
                state = _ExecutionLearningState.from_bytes(payload)
                if state.sha256 != address:
                    raise ExecutionLearningIntegrityError(
                        "execution-learning history has another content address"
                    )
                histories[address] = state
            else:
                expected = canonical_json_bytes(
                    {
                        "schema": EXECUTION_LEARNING_STATE_SCHEMA,
                        "state_sha256": address,
                    }
                )
                if payload != expected:
                    raise ExecutionLearningIntegrityError(
                        "execution-learning commit has another content address"
                    )
                commits.add(address)
        if not commits.issubset(histories):
            raise ExecutionLearningIntegrityError(
                "execution-learning commit lacks immutable history"
            )
        effective_commits = set(commits)
        if recoverable_missing_head_commit:
            if head.sha256 in effective_commits or head.sha256 not in histories:
                raise ExecutionLearningIntegrityError(
                    "recoverable head commit state is inconsistent"
                )
            effective_commits.add(head.sha256)
        committed = {digest: histories[digest] for digest in effective_commits}
        roots = tuple(row for row in committed.values() if row.generation == 0)
        if len(roots) != 1 or roots[0] != _ExecutionLearningState.initial():
            raise ExecutionLearningIntegrityError(
                "committed execution-learning history lacks one exact root"
            )
        chain = [roots[0]]
        while True:
            children = tuple(
                row
                for row in committed.values()
                if row.generation == chain[-1].generation + 1
                and row.previous_state_sha256 == chain[-1].sha256
            )
            if not children:
                break
            if len(children) != 1:
                raise ExecutionLearningIntegrityError(
                    "committed execution-learning history contains a fork"
                )
            child = children[0]
            if (
                child.receipts[:-1] != chain[-1].receipts
                or child.traces[:-1] != chain[-1].traces
            ):
                raise ExecutionLearningIntegrityError(
                    "committed execution-learning history is not append-only"
                )
            chain.append(child)
        if {row.sha256 for row in chain} != set(committed):
            raise ExecutionLearningIntegrityError(
                "committed execution-learning history is disconnected"
            )
        if chain[-1].sha256 != head.sha256:
            raise ExecutionLearningIntegrityError(
                "execution-learning head is a fully resealed rollback"
            )

    def _replay_ancestry_unlocked(
        self,
        head: _ExecutionLearningState,
        required_sha256: str | None,
    ) -> None:
        seen: set[str] = set()
        current = head
        while True:
            if current.sha256 in seen:
                raise ExecutionLearningIntegrityError(
                    "learning history contains a cycle"
                )
            seen.add(current.sha256)
            if current.generation == 0:
                break
            previous_payload = self._restore_optional(
                self._history_name(current.previous_state_sha256)
            )
            if previous_payload is None:
                raise ExecutionLearningIntegrityError(
                    "learning history predecessor is missing"
                )
            previous = _ExecutionLearningState.from_bytes(previous_payload)
            if (
                current.generation != previous.generation + 1
                or current.receipts[:-1] != previous.receipts
                or current.traces[:-1] != previous.traces
                or current.previous_state_sha256 != previous.sha256
            ):
                raise ExecutionLearningIntegrityError(
                    "learning history is not an append-only extension"
                )
            current = previous
        if current != _ExecutionLearningState.initial():
            raise ExecutionLearningIntegrityError("learning history root is invalid")
        if required_sha256 is not None and required_sha256 not in seen:
            raise ExecutionLearningIntegrityError(
                "learning history does not descend from trusted head"
            )

    def head(self) -> _ExecutionLearningState:
        with self._locked():
            return self._head_unlocked()

    def intent(self, transaction_sha256: str) -> _ExecutionLearningIntent | None:
        with self._locked():
            payload = self._restore_optional(self._intent_name(transaction_sha256))
            return (
                None
                if payload is None
                else _ExecutionLearningIntent.from_bytes(payload)
            )

    def find_receipt(
        self,
        *,
        measurement_sha256: str,
        action_authority_sha256: str,
        action: OoeAction | str,
    ) -> ExecutionLearningReceipt | None:
        """Return an exact committed replay, rejecting identity collisions."""

        measurement_digest = require_sha256(
            measurement_sha256, field="measurement_sha256"
        )
        authority_digest = require_sha256(
            action_authority_sha256, field="action_authority_sha256"
        )
        normalized_action = validate_action(action)
        with self._locked():
            matches = tuple(
                row
                for row in self._head_unlocked().receipts
                if row.measurement.sha256 == measurement_digest
                and row.action_authority.sha256 == authority_digest
                and row.execution.action == normalized_action
            )
        if len(matches) > 1:
            raise ExecutionLearningIntegrityError(
                "measurement/action identity has multiple committed receipts"
            )
        return None if not matches else matches[0]

    def publish_intent(self, intent: _ExecutionLearningIntent) -> None:
        with self._locked():
            current = self._head_unlocked()
            if current.sha256 != intent.bank_head_before_sha256:
                existing = next(
                    (
                        row
                        for row in current.traces
                        if row.transaction_sha256 == intent.transaction_sha256
                    ),
                    None,
                )
                if existing is None:
                    raise ExecutionLearningStaleError(
                        "learning stream changed before intent publication"
                    )
                return
            self._publish_immutable(
                self._intent_name(intent.transaction_sha256), intent.to_bytes()
            )

    def preflight(
        self,
        *,
        transaction_sha256: str,
        temporal_index: int,
        source_action: str,
        controller_snapshot_before_sha256: str,
    ) -> _ExecutionLearningState:
        with self._locked():
            current = self._head_unlocked()
            if any(
                row.transaction_sha256 == transaction_sha256 for row in current.traces
            ):
                return current
            expected_source = (
                validate_action(source_action)
                if not current.traces
                else current.traces[-1].target_action
            )
            if validate_action(source_action) != expected_source:
                raise ExecutionLearningStaleError(
                    "source action does not continue the trace chain"
                )
            if (
                current.traces
                and temporal_index != current.traces[-1].temporal_index + 1
            ):
                raise ExecutionLearningStaleError(
                    "temporal index does not continue the trace chain"
                )
            if current.traces and (
                controller_snapshot_before_sha256
                != current.traces[-1].controller_snapshot_after_sha256
            ):
                raise ExecutionLearningStaleError(
                    "controller snapshot does not continue the trace chain"
                )
            return current

    def commit_intent(
        self,
        intent: _ExecutionLearningIntent,
        *,
        controller_snapshot_sha256: str,
    ) -> ExecutionLearningReceipt:
        persisted_controller = require_sha256(
            controller_snapshot_sha256, field="controller_snapshot_sha256"
        )
        if persisted_controller != intent.trace.controller_snapshot_after_sha256:
            raise ExecutionLearningStaleError(
                "controller snapshot was not published before stream advancement"
            )
        with self._locked():
            current = self._head_unlocked()
            existing_index = next(
                (
                    index
                    for index, row in enumerate(current.traces)
                    if row.transaction_sha256 == intent.transaction_sha256
                ),
                None,
            )
            if existing_index is not None:
                existing = current.receipts[existing_index]
                if existing != intent.receipt:
                    raise ExecutionLearningIntegrityError(
                        "transaction identity resolved another learning receipt"
                    )
                return existing
            if current.sha256 != intent.bank_head_before_sha256:
                raise ExecutionLearningStaleError(
                    "learning stream head changed before commit"
                )
            updated = current.appended(intent.receipt, intent.trace)
            self._publish_immutable(
                self._receipt_name(intent.receipt.sha256),
                _canonical_document_bytes(
                    intent.receipt.to_document(), label="learning receipt"
                ),
            )
            self._publish_immutable(
                self._trace_name(intent.trace.sha256),
                _canonical_document_bytes(intent.trace.to_document(), label="trace"),
            )
            self._publish_immutable(
                self._history_name(updated.sha256), updated.to_bytes()
            )
            try:
                self.store.publish_state(
                    _HEAD_STATE,
                    updated.to_bytes(),
                    expected_sha256=current.sha256,
                )
            except ManifestConflictError as exc:
                raise ExecutionLearningStaleError(
                    "learning stream head CAS conflicted"
                ) from exc
            except CrystalStoreError as exc:
                raise ExecutionLearningIntegrityError(
                    "learning stream publication failed"
                ) from exc
            self._publish_immutable(
                self._commit_name(updated.sha256),
                canonical_json_bytes(
                    {
                        "schema": EXECUTION_LEARNING_STATE_SCHEMA,
                        "state_sha256": updated.sha256,
                    }
                ),
            )
            if self._head_unlocked() != updated:
                raise ExecutionLearningIntegrityError(
                    "learning stream commit failed exact roundtrip"
                )
            return intent.receipt


RegisteredExecutor = tuple[str, ActionExecutor]
RegisteredQualityVerifier = tuple[str, ExecutionQualityVerifier]


class ExecutionLearningBridge:
    """Execute, verify, learn, snapshot, and trace one real runtime action."""

    def __init__(
        self,
        *,
        controller: OoeController,
        atlas: SemanticWeightAtlas,
        bank: ExecutionLearningBank,
        action_authorities: Mapping[str, ActionAuthorityReceipt],
        action_executors: Mapping[str, RegisteredExecutor],
        quality_verifiers: Mapping[str, RegisteredQualityVerifier],
        initial_source_action: OoeAction | str = "qwen_fallback",
        controller_state_name: str = CONTROLLER_STATE_NAME,
    ) -> None:
        if not isinstance(controller, OoeController):
            raise TypeError("controller must be an OoeController")
        if not isinstance(atlas, SemanticWeightAtlas):
            raise TypeError("atlas must be a SemanticWeightAtlas")
        if not isinstance(bank, ExecutionLearningBank):
            raise TypeError("bank must be an ExecutionLearningBank")
        self.controller = controller
        self.atlas = atlas
        self.bank = bank
        self.initial_source_action = validate_action(initial_source_action)
        self.controller_state_name = _text(
            controller_state_name, field="controller_state_name"
        )
        authorities: dict[OoeAction, ActionAuthorityReceipt] = {}
        executors: dict[OoeAction, RegisteredExecutor] = {}
        for action, authority in action_authorities.items():
            normalized = validate_action(action)
            if not isinstance(authority, ActionAuthorityReceipt):
                raise TypeError(
                    "action authorities must be ActionAuthorityReceipt values"
                )
            if authority.action != normalized:
                raise ValueError("action authority is registered under another action")
            authority.assert_current(
                model_pin_sha256=controller.model_pin_sha256,
                weight_graph_revision_sha256=controller.weight_graph_revision_sha256,
            )
            authorities[normalized] = authority
        for action, registered in action_executors.items():
            normalized = validate_action(action)
            if (
                not isinstance(registered, tuple)
                or len(registered) != 2
                or not callable(registered[1])
            ):
                raise TypeError("registered executor must be (sha256, callable)")
            executors[normalized] = (
                require_sha256(registered[0], field="executor_sha256"),
                registered[1],
            )
        verifiers: dict[str, RegisteredQualityVerifier] = {}
        for name, registered in quality_verifiers.items():
            canonical_name = _text(name, field="quality verifier name")
            if (
                not isinstance(registered, tuple)
                or len(registered) != 2
                or not callable(registered[1])
            ):
                raise TypeError(
                    "registered quality verifier must be (sha256, callable)"
                )
            verifiers[canonical_name] = (
                require_sha256(registered[0], field="quality_verifier_sha256"),
                registered[1],
            )
        for action, authority in authorities.items():
            registered_executor = executors.get(action)
            registered_verifier = verifiers.get(authority.quality_verifier_name)
            if (
                registered_executor is None
                or registered_executor[0] != authority.executor_sha256
                or registered_verifier is None
                or registered_verifier[0] != authority.quality_verifier_sha256
            ):
                raise ExecutionLearningIntegrityError(
                    "authority does not match registered executor/quality verifier"
                )
        self._authorities = authorities
        self._executors = executors
        self._quality_verifiers = verifiers
        self._ensure_controller_snapshot()

    def _controller_state_bytes(self) -> bytes:
        try:
            return self.controller.crystal_store.restore_state(
                self.controller_state_name
            )
        except KeyError:
            publication = self.controller.save_snapshot(name=self.controller_state_name)
            if publication.payload_sha256 != _bytes_sha256(
                self.controller.snapshot_bytes()
            ):
                raise ExecutionLearningIntegrityError(
                    "initial controller snapshot publication differs"
                )
            return self.controller.crystal_store.restore_state(
                self.controller_state_name
            )

    def _ensure_controller_snapshot(self) -> str:
        persisted = self._controller_state_bytes()
        in_memory = self.controller.snapshot_bytes()
        if persisted != in_memory:
            raise ExecutionLearningStaleError(
                "in-memory controller differs from its persistent snapshot"
            )
        return _bytes_sha256(persisted)

    def _restore_controller(self) -> None:
        self.controller = OoeController.restore(
            crystal_store=self.controller.crystal_store,
            name=self.controller_state_name,
            atlas_revision_verifier=self.atlas.contains_revision,
            expected_model_pin_sha256=self.controller.model_pin_sha256,
            expected_weight_graph_revision_sha256=(
                self.controller.weight_graph_revision_sha256
            ),
        )

    def _authenticate_measurement(
        self, measurement: MeasurementReceipt
    ) -> tuple[str, str]:
        if not isinstance(measurement, MeasurementReceipt):
            raise TypeError("measurement must be a MeasurementReceipt")
        try:
            self.atlas.verify_or_raise()
            audited_head = self.atlas.revision()
            if measurement.model_pin != self.atlas.model_pin:
                raise ExecutionLearningStaleError(
                    "measurement belongs to another Atlas model pin"
                )
            if not self.atlas.contains_revision(measurement.atlas_head_revision):
                raise ExecutionLearningStaleError(
                    "measurement Atlas revision is not in the live chain"
                )
            matches = tuple(
                row
                for row in self.atlas.query_by_coordinate(
                    measurement.coordinate
                ).measurements
                if row.sha256 == measurement.sha256
            )
            if self.atlas.revision() != audited_head:
                raise ExecutionLearningStaleError(
                    "Atlas head changed during measurement authentication"
                )
        except SemanticAtlasError as exc:
            raise ExecutionLearningIntegrityError(
                "live Atlas authentication failed"
            ) from exc
        if len(matches) != 1 or matches[0].to_document() != measurement.to_document():
            raise ExecutionLearningIntegrityError(
                "measurement is not one exact active Atlas primary"
            )
        if measurement.observation_status == "invalidated":
            raise ExecutionLearningIntegrityError(
                "invalidated Atlas measurement cannot train the controller"
            )
        return (
            _atlas_authentication_sha256(
                measurement, atlas_head_sha256=audited_head.sha256
            ),
            audited_head.sha256,
        )

    @staticmethod
    def _transaction_sha256(
        *,
        measurement: MeasurementReceipt,
        authority: ActionAuthorityReceipt,
        temporal_index: int,
        source_action: str,
        o1_surprise: float,
        o1_learning_progress: float,
        dimensions: int,
        weight: float,
    ) -> str:
        return _digest(
            {
                "action_authority_sha256": authority.sha256,
                "dimensions": dimensions,
                "measurement_sha256": measurement.sha256,
                "o1_learning_progress": o1_learning_progress,
                "o1_surprise": o1_surprise,
                "schema": "immer-ooe-execution-learning-transaction/v1",
                "source_action": validate_action(source_action),
                "temporal_index": temporal_index,
                "weight": weight,
            }
        )

    def _finish_intent(
        self, intent: _ExecutionLearningIntent
    ) -> ExecutionLearningReceipt:
        current_state = self._controller_state_bytes()
        current_sha256 = _bytes_sha256(current_state)
        before = intent.trace.controller_snapshot_before_sha256
        after = intent.trace.controller_snapshot_after_sha256
        if current_sha256 == before:
            clone = OoeController.restore(
                crystal_store=self.controller.crystal_store,
                name=self.controller_state_name,
                atlas_revision_verifier=self.atlas.contains_revision,
                expected_model_pin_sha256=self.controller.model_pin_sha256,
                expected_weight_graph_revision_sha256=(
                    self.controller.weight_graph_revision_sha256
                ),
            )
            clone.ingest_teacher(intent.receipt.feature, intent.receipt.transition)
            if _bytes_sha256(clone.snapshot_bytes()) != after:
                raise ExecutionLearningIntegrityError(
                    "replayed controller snapshot differs from prepared intent"
                )
            try:
                publication = clone.save_snapshot(
                    name=self.controller_state_name,
                    expected_sha256=before,
                )
            except ManifestConflictError as exc:
                raise ExecutionLearningStaleError(
                    "controller snapshot CAS conflicted"
                ) from exc
            if publication.payload_sha256 != after:
                raise ExecutionLearningIntegrityError(
                    "controller snapshot CAS published another payload"
                )
            self.controller = clone
            current_sha256 = after
        elif current_sha256 == after:
            self._restore_controller()
        else:
            raise ExecutionLearningStaleError(
                "persistent controller is neither side of prepared transaction"
            )
        try:
            return self.bank.commit_intent(
                intent, controller_snapshot_sha256=current_sha256
            )
        except Exception:
            # The persistent controller is authoritative after its CAS.  Keep
            # this bridge retry-safe even when trace publication was interrupted.
            self._restore_controller()
            raise

    def learn_from_execution(
        self,
        measurement: MeasurementReceipt,
        *,
        action: OoeAction | str,
        source_action: OoeAction | str | None = None,
        temporal_index: int | None = None,
        o1_surprise: float,
        o1_learning_progress: float,
        dimensions: int = DEFAULT_FEATURE_DIMENSIONS,
        weight: float = 1.0,
    ) -> ExecutionLearningReceipt | None:
        """Learn one action; return ``None`` for a neutral quality rejection."""

        normalized_action = validate_action(action)
        authority = self._authorities.get(normalized_action)
        if authority is None:
            raise ExecutionLearningIntegrityError(
                "selected action has no pinned execution authority"
            )
        learned_weight = _weight(weight)
        committed = self.bank.find_receipt(
            measurement_sha256=measurement.sha256,
            action_authority_sha256=authority.sha256,
            action=normalized_action,
        )
        if committed is not None:
            if (
                committed.measurement.to_document() != measurement.to_document()
                or (
                    source_action is not None
                    and validate_action(source_action)
                    != committed.transition.source_action
                )
                or (
                    temporal_index is not None
                    and _uint(temporal_index, field="temporal_index")
                    != committed.feature.temporal_index
                )
                or committed.feature.o1_surprise != float(o1_surprise)
                or committed.feature.o1_learning_progress != float(o1_learning_progress)
                or committed.feature.feature_dimensions != dimensions
                or committed.transition.weight != learned_weight
            ):
                raise ExecutionLearningStaleError(
                    "replay parameters differ from committed learning evidence"
                )
            self._authenticate_measurement(measurement)
            self._ensure_controller_snapshot()
            return committed
        before_sha256 = self._ensure_controller_snapshot()
        head = self.bank.head()
        expected_source = (
            self.initial_source_action
            if not head.traces
            else head.traces[-1].target_action
        )
        actual_source = (
            expected_source if source_action is None else validate_action(source_action)
        )
        if actual_source != expected_source:
            raise ExecutionLearningStaleError(
                "caller source action does not continue the trace chain"
            )
        requested_temporal = (
            None
            if temporal_index is None
            else _uint(temporal_index, field="temporal_index")
        )
        # A crash can leave the persistent controller exactly one intent ahead
        # of the committed trace.  Probe both the normal next index and the
        # already-published controller index before creating new evidence.
        temporal_candidates = (
            (requested_temporal,)
            if requested_temporal is not None
            else tuple(
                dict.fromkeys(
                    (
                        self.controller.last_temporal_index + 1,
                        self.controller.last_temporal_index,
                    )
                )
            )
        )
        for candidate_temporal in temporal_candidates:
            if candidate_temporal < 0:
                continue
            candidate_transaction = self._transaction_sha256(
                measurement=measurement,
                authority=authority,
                temporal_index=candidate_temporal,
                source_action=actual_source,
                o1_surprise=float(o1_surprise),
                o1_learning_progress=float(o1_learning_progress),
                dimensions=dimensions,
                weight=learned_weight,
            )
            pending = self.bank.intent(candidate_transaction)
            if pending is not None:
                return self._finish_intent(pending)

        expected_temporal = self.controller.last_temporal_index + 1
        actual_temporal = (
            expected_temporal if requested_temporal is None else requested_temporal
        )
        if actual_temporal != expected_temporal:
            raise ExecutionLearningStaleError(
                "temporal index must be the next controller index"
            )
        transaction_sha256 = self._transaction_sha256(
            measurement=measurement,
            authority=authority,
            temporal_index=actual_temporal,
            source_action=actual_source,
            o1_surprise=float(o1_surprise),
            o1_learning_progress=float(o1_learning_progress),
            dimensions=dimensions,
            weight=learned_weight,
        )

        atlas_authentication, atlas_head_sha256 = self._authenticate_measurement(
            measurement
        )
        feature = QwenOoeFeatureReceipt.from_measurement(
            measurement,
            temporal_index=actual_temporal,
            verifier_sha256s=(
                ATLAS_MEASUREMENT_VERIFIER_SHA256,
                atlas_authentication,
                authority.sha256,
                authority.action_verifier_sha256,
                authority.quality_verifier_sha256,
            ),
            evidence_sha256s=(
                atlas_authentication,
                authority.sha256,
                authority.evidence_sha256,
            ),
            o1_surprise=o1_surprise,
            o1_learning_progress=o1_learning_progress,
            dimensions=dimensions,
        )
        # Site identity excludes verifier/evidence authorities by design.  Its
        # coordinate/schema identity therefore survives authority enrichment.
        feature.validate_measurement(measurement)
        executor_sha256, executor = self._executors[normalized_action]
        if executor_sha256 != authority.executor_sha256:
            raise ExecutionLearningIntegrityError("executor registry changed")
        try:
            execution = executor(feature)
        except Exception as exc:
            raise ExecutionLearningExecutionError(
                f"executor for {normalized_action!r} failed"
            ) from exc
        if not isinstance(execution, ActionExecution):
            raise ExecutionLearningIntegrityError(
                "action executor did not return ActionExecution"
            )
        execution.assert_bound(feature, authority.action)
        if (
            execution.executor_sha256 != authority.executor_sha256
            or execution.verifier_sha256 != authority.action_verifier_sha256
            or execution.evidence_sha256 != authority.evidence_sha256
        ):
            raise ExecutionLearningIntegrityError(
                "execution hashes differ from action authority"
            )
        verifier_sha256, verifier = self._quality_verifiers[
            authority.quality_verifier_name
        ]
        if verifier_sha256 != authority.quality_verifier_sha256:
            raise ExecutionLearningIntegrityError("quality verifier registry changed")
        try:
            verification_result = verifier(feature, execution)
        except Exception as exc:
            raise ExecutionLearningExecutionError(
                f"quality verifier {authority.quality_verifier_name!r} failed"
            ) from exc
        if not isinstance(verification_result, bool):
            raise ExecutionLearningIntegrityError(
                "quality verifier must return an exact bool"
            )
        verified = verification_result
        quality = ExecutionQualityReceipt(
            feature_receipt_sha256=feature.sha256,
            execution_sha256=execution.sha256,
            execution_quality_sha256=execution.quality_sha256,
            verifier_name=authority.quality_verifier_name,
            verifier_sha256=authority.quality_verifier_sha256,
            verified=verified,
        )
        quality.assert_bound(feature, execution, authority)
        if not execution.quality_verified or not quality.verified:
            return None
        transition = VerifiedTeacherTransition(
            feature_receipt_sha256=feature.sha256,
            site_identity_sha256=feature.site_identity.sha256,
            source_action=actual_source,
            target_action=execution.action,
            verifier_sha256=authority.quality_verifier_sha256,
            evidence_sha256=authority.evidence_sha256,
            quality_sha256=quality.sha256,
            verified_quality=True,
            weight=learned_weight,
        )
        receipt = ExecutionLearningReceipt(
            measurement=measurement,
            action_authority=authority,
            feature=feature,
            execution=execution,
            quality=quality,
            transition=transition,
            atlas_head_sha256=atlas_head_sha256,
            atlas_authentication_sha256=atlas_authentication,
        )
        preflight_head = self.bank.preflight(
            transaction_sha256=transaction_sha256,
            temporal_index=actual_temporal,
            source_action=actual_source,
            controller_snapshot_before_sha256=before_sha256,
        )
        if preflight_head.sha256 != head.sha256:
            raise ExecutionLearningStaleError(
                "learning stream changed during execution preflight"
            )

        # Derive the exact post-ingest snapshot on an isolated restored clone.
        # The live controller is not touched until every receipt and bank gate
        # above has passed.
        clone = OoeController.restore(
            crystal_store=self.controller.crystal_store,
            name=self.controller_state_name,
            atlas_revision_verifier=self.atlas.contains_revision,
            expected_model_pin_sha256=self.controller.model_pin_sha256,
            expected_weight_graph_revision_sha256=(
                self.controller.weight_graph_revision_sha256
            ),
        )
        clone.ingest_teacher(feature, transition)
        after_sha256 = _bytes_sha256(clone.snapshot_bytes())
        trace = ControllerActionTraceReceipt(
            transaction_sha256=transaction_sha256,
            ordinal=head.generation,
            temporal_index=actual_temporal,
            previous_trace_sha256=head.stream_head_sha256,
            previous_stream_head_sha256=head.stream_head_sha256,
            measurement_sha256=measurement.sha256,
            learning_receipt_sha256=receipt.sha256,
            source_action=actual_source,
            target_action=execution.action,
            controller_snapshot_before_sha256=before_sha256,
            controller_snapshot_after_sha256=after_sha256,
        )
        intent = _ExecutionLearningIntent(
            transaction_sha256=transaction_sha256,
            bank_head_before_sha256=head.sha256,
            receipt=receipt,
            trace=trace,
        )
        self.bank.publish_intent(intent)
        return self._finish_intent(intent)


__all__ = [
    "ACTION_AUTHORITY_SCHEMA",
    "ATLAS_MEASUREMENT_VERIFIER_SHA256",
    "CONTROLLER_ACTION_TRACE_SCHEMA",
    "EXECUTION_LEARNING_SCHEMA",
    "EXECUTION_QUALITY_SCHEMA",
    "ActionAuthorityReceipt",
    "ControllerActionTraceReceipt",
    "ExecutionLearningBank",
    "ExecutionLearningBridge",
    "ExecutionLearningError",
    "ExecutionLearningExecutionError",
    "ExecutionLearningIntegrityError",
    "ExecutionLearningReceipt",
    "ExecutionLearningStaleError",
    "ExecutionQualityReceipt",
]
