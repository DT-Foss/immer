"""Online Qwen-teacher to Markov-OoE controller.

The controller learns only from authenticated Qwen/O1 feature receipts and
verified teacher transitions.  Knowledge is split across replica site agents,
fused through an executed PS-Lifted push-sum, and published as immutable
quantized Crystals.  A mobile Markov token's reservoir and rapidity ledger are
part of the execution gate; novel, ambiguous, stale, or uncovered inputs fall
back to Qwen instead of being guessed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from functools import wraps
import hashlib
import json
import math
import threading
from typing import Any, Literal

import numpy as np

from .agents import AttractorRouter, MarkovPDAgent, MobileMarkovToken
from .consensus import adaptive_ps_lift_matrix, barbell_adjacency, measure_consensus
from .crystal import (
    CrystalManifest,
    CrystalPayload,
    CrystalPublication,
    CrystalStore,
    StatePublication,
)
from .identity import OoeSiteIdentity, canonical_json_bytes, require_sha256
from .math_core import RapidityLedger, array_sha256, normalize_rows
from .qwen_bridge import (
    OOE_ACTIONS,
    OoeAction,
    QwenOoeFeatureReceipt,
    validate_action,
)
from immer.runtimes.qwen3_8.semantic_atlas import GraphRevision


CONTROLLER_STATE_SCHEMA = "immer-ooe-controller-state/v1"
TEACHER_TRANSITION_SCHEMA = "immer-ooe-verified-teacher-transition/v1"
ACTION_EXECUTION_SCHEMA = "immer-ooe-action-execution/v1"
WARM_ACCOUNTING_SCHEMA = "immer-ooe-warm-accounting/v1"
CONTROLLER_RECOVERY_SCHEMA = "immer-ooe-controller-recovery/v1"
COVERAGE_SCHEMA = "immer-ooe-site-coverage/v1"
CONTROLLER_STATE_NAME = "qwen-ooe-controller"
_ACTION_INDEX = {action: index for index, action in enumerate(OOE_ACTIONS)}
_MAX_HISTORY = 1_000_000
_MAX_WARM_TRANSACTIONS = 1_000_000


def _locked(method):
    """Serialize one public controller operation on its instance lock."""

    @wraps(method)
    def synchronized(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return synchronized


class OoeControllerError(RuntimeError):
    """The online OoE controller contract cannot be satisfied."""


class OoeControllerIntegrityError(OoeControllerError):
    """Controller evidence, state, or Crystal identity was modified."""


class OoeControllerStaleError(OoeControllerIntegrityError):
    """Controller input belongs to another pinned model or graph."""


class OoeControllerAmbiguityError(OoeControllerIntegrityError):
    """A prompt lookup resolves to more than one exact weight site."""


class OoePromotionError(OoeControllerError):
    """A replica site lacks evidence required for Crystal promotion."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _uint(value: object, *, field: str, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive " if positive else "non-negative "
        raise ValueError(f"{field} must be a {qualifier}integer")
    return value


def _probability(value: object, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a probability")
    result = float(value)
    lower_ok = result > 0.0 if positive else result >= 0.0
    if not math.isfinite(result) or not lower_ok or result > 1.0:
        raise ValueError(f"{field} must be a probability")
    return result


def _canonical_text(value: object, *, field: str, maximum: int = 1024) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > maximum
    ):
        raise ValueError(f"{field} must be canonical non-empty text")
    return value


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    replicas: int = 12
    replica_fanout: int = 4
    min_coverage_per_source: int = 1
    min_promoted_sources: int = 1
    router_radius: float = 0.75
    router_min_margin: float = 0.02
    token_min_confidence: float = 0.08
    consensus_tolerance: float = 1e-6
    consensus_max_rounds: int = 4096
    quantization_levels: int = 65535
    reservoir_size: int = 16

    def __post_init__(self) -> None:
        replicas = _uint(self.replicas, field="replicas", positive=True)
        if replicas < 4 or replicas > 256 or replicas % 2:
            raise ValueError("replicas must be even and lie in [4, 256]")
        fanout = _uint(self.replica_fanout, field="replica_fanout", positive=True)
        if fanout > replicas:
            raise ValueError("replica_fanout cannot exceed replicas")
        _uint(
            self.min_coverage_per_source,
            field="min_coverage_per_source",
            positive=True,
        )
        promoted_sources = _uint(
            self.min_promoted_sources,
            field="min_promoted_sources",
            positive=True,
        )
        if promoted_sources > len(OOE_ACTIONS):
            raise ValueError("min_promoted_sources exceeds the action alphabet")
        radius = float(self.router_radius)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("router_radius must be positive and finite")
        margin = float(self.router_min_margin)
        if not math.isfinite(margin) or margin < 0.0:
            raise ValueError("router_min_margin must be non-negative and finite")
        _probability(
            self.token_min_confidence,
            field="token_min_confidence",
            positive=True,
        )
        tolerance = float(self.consensus_tolerance)
        if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
            raise ValueError("consensus_tolerance must lie in (0, 1)")
        _uint(
            self.consensus_max_rounds,
            field="consensus_max_rounds",
            positive=True,
        )
        if not 2 <= self.quantization_levels <= 65535:
            raise ValueError("quantization_levels must lie in [2, 65535]")
        if self.reservoir_size < 8 or self.reservoir_size > 4096:
            raise ValueError("reservoir_size must lie in [8, 4096]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VerifiedTeacherTransition:
    """One teacher action whose verifier and evidence are feature-bound."""

    feature_receipt_sha256: str
    site_identity_sha256: str
    source_action: OoeAction | str
    target_action: OoeAction | str
    verifier_sha256: str
    evidence_sha256: str
    quality_sha256: str
    verified_quality: bool = True
    weight: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "feature_receipt_sha256",
            "site_identity_sha256",
            "verifier_sha256",
            "evidence_sha256",
            "quality_sha256",
        ):
            object.__setattr__(
                self,
                name,
                require_sha256(getattr(self, name), field=name),
            )
        object.__setattr__(self, "source_action", validate_action(self.source_action))
        object.__setattr__(self, "target_action", validate_action(self.target_action))
        if self.verified_quality is not True:
            raise ValueError("teacher transition must have verified quality")
        weight = float(self.weight)
        if not math.isfinite(weight) or weight <= 0.0 or weight > 1_000_000.0:
            raise ValueError("teacher transition weight must be positive and bounded")
        object.__setattr__(self, "weight", weight)

    def as_record(self) -> dict[str, Any]:
        return {
            "evidence_sha256": self.evidence_sha256,
            "feature_receipt_sha256": self.feature_receipt_sha256,
            "quality_sha256": self.quality_sha256,
            "site_identity_sha256": self.site_identity_sha256,
            "source_action": self.source_action,
            "target_action": self.target_action,
            "verified_quality": self.verified_quality,
            "verifier_sha256": self.verifier_sha256,
            "weight": self.weight,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())

    def to_document(self) -> dict[str, Any]:
        body = self.as_record()
        return {
            "body": body,
            "schema": TEACHER_TRANSITION_SCHEMA,
            "sha256": _digest(body),
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "VerifiedTeacherTransition":
        if not isinstance(document, Mapping) or set(document) != {
            "body",
            "schema",
            "sha256",
        }:
            raise OoeControllerIntegrityError("teacher transition envelope is invalid")
        if document.get("schema") != TEACHER_TRANSITION_SCHEMA:
            raise OoeControllerIntegrityError("teacher transition schema is invalid")
        body = document.get("body")
        expected = {
            "evidence_sha256",
            "feature_receipt_sha256",
            "quality_sha256",
            "site_identity_sha256",
            "source_action",
            "target_action",
            "verified_quality",
            "verifier_sha256",
            "weight",
        }
        if not isinstance(body, Mapping) or set(body) != expected:
            raise OoeControllerIntegrityError("teacher transition body is invalid")
        if require_sha256(document.get("sha256"), field="sha256") != _digest(body):
            raise OoeControllerIntegrityError("teacher transition SHA-256 mismatch")
        return cls(**dict(body))

    def assert_bound(self, receipt: QwenOoeFeatureReceipt) -> None:
        if not isinstance(receipt, QwenOoeFeatureReceipt):
            raise TypeError("receipt must be a QwenOoeFeatureReceipt")
        if self.feature_receipt_sha256 != receipt.sha256:
            raise OoeControllerIntegrityError("teacher transition receipt mismatch")
        if self.site_identity_sha256 != receipt.site_identity.sha256:
            raise OoeControllerIntegrityError("teacher transition site mismatch")
        if self.verifier_sha256 not in receipt.verifier_sha256s:
            raise OoeControllerIntegrityError("teacher verifier is not feature-bound")
        if self.evidence_sha256 not in receipt.evidence_sha256s:
            raise OoeControllerIntegrityError("teacher evidence is not feature-bound")


@dataclass(frozen=True, slots=True)
class ActionExecution:
    """Hash-sealed output of one concrete warm-path action executor."""

    feature_receipt_sha256: str
    action: OoeAction | str
    executor_sha256: str
    verifier_sha256: str
    evidence_sha256: str
    quality_sha256: str
    result: Any
    quality_verified: bool
    qwen_forwards: int
    teacher_baseline_qwen_forwards: int = 1

    def __post_init__(self) -> None:
        for name in (
            "feature_receipt_sha256",
            "executor_sha256",
            "verifier_sha256",
            "evidence_sha256",
            "quality_sha256",
        ):
            object.__setattr__(
                self,
                name,
                require_sha256(getattr(self, name), field=name),
            )
        object.__setattr__(self, "action", validate_action(self.action))
        if not isinstance(self.quality_verified, bool):
            raise TypeError("quality_verified must be bool")
        qwen_forwards = _uint(self.qwen_forwards, field="qwen_forwards")
        baseline = _uint(
            self.teacher_baseline_qwen_forwards,
            field="teacher_baseline_qwen_forwards",
        )
        if qwen_forwards > 1_000_000 or baseline > 1_000_000:
            raise ValueError("execution forward counts exceed the bounded contract")
        try:
            encoded = canonical_json_bytes(self.result)
            normalized = json.loads(encoded)
        except ValueError as exc:
            raise ValueError("execution result must be canonical JSON") from exc
        if len(encoded) > 1024 * 1024:
            raise ValueError("execution result exceeds one MiB")
        object.__setattr__(self, "result", normalized)
        object.__setattr__(self, "qwen_forwards", qwen_forwards)
        object.__setattr__(self, "teacher_baseline_qwen_forwards", baseline)

    @property
    def saved_qwen_forwards(self) -> int:
        return max(0, self.teacher_baseline_qwen_forwards - self.qwen_forwards)

    def as_record(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "evidence_sha256": self.evidence_sha256,
            "executor_sha256": self.executor_sha256,
            "feature_receipt_sha256": self.feature_receipt_sha256,
            "quality_sha256": self.quality_sha256,
            "quality_verified": self.quality_verified,
            "qwen_forwards": self.qwen_forwards,
            "result": self.result,
            "teacher_baseline_qwen_forwards": self.teacher_baseline_qwen_forwards,
            "verifier_sha256": self.verifier_sha256,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())

    def to_document(self) -> dict[str, Any]:
        body = self.as_record()
        return {
            "body": body,
            "schema": ACTION_EXECUTION_SCHEMA,
            "sha256": _digest(body),
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "ActionExecution":
        if not isinstance(document, Mapping) or set(document) != {
            "body",
            "schema",
            "sha256",
        }:
            raise OoeControllerIntegrityError("action execution envelope is invalid")
        if document.get("schema") != ACTION_EXECUTION_SCHEMA:
            raise OoeControllerIntegrityError("action execution schema is invalid")
        body = document.get("body")
        expected = {
            "action",
            "evidence_sha256",
            "executor_sha256",
            "feature_receipt_sha256",
            "quality_sha256",
            "quality_verified",
            "qwen_forwards",
            "result",
            "teacher_baseline_qwen_forwards",
            "verifier_sha256",
        }
        if not isinstance(body, Mapping) or set(body) != expected:
            raise OoeControllerIntegrityError("action execution body is invalid")
        if require_sha256(document.get("sha256"), field="sha256") != _digest(body):
            raise OoeControllerIntegrityError("action execution SHA-256 mismatch")
        return cls(**dict(body))

    def assert_bound(
        self,
        receipt: QwenOoeFeatureReceipt,
        action: OoeAction | str,
    ) -> None:
        expected_action = validate_action(action)
        if self.feature_receipt_sha256 != receipt.sha256:
            raise OoeControllerIntegrityError("action execution receipt mismatch")
        if self.action != expected_action:
            raise OoeControllerIntegrityError("action execution action mismatch")
        if self.verifier_sha256 not in receipt.verifier_sha256s:
            raise OoeControllerIntegrityError("executor verifier is not feature-bound")
        if self.evidence_sha256 not in receipt.evidence_sha256s:
            raise OoeControllerIntegrityError("executor evidence is not feature-bound")


@dataclass(frozen=True, slots=True)
class WarmAccountingReceipt:
    """Final idempotent disposition of one execution-bound warm transaction."""

    transaction_sha256: str
    decision_binding_sha256: str
    execution_receipt_sha256: str
    disposition: Literal["committed", "rejected"] | str
    saved_qwen_forwards: int

    def __post_init__(self) -> None:
        for field_name in (
            "transaction_sha256",
            "decision_binding_sha256",
            "execution_receipt_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )
        if self.disposition not in ("committed", "rejected"):
            raise ValueError("warm disposition must be committed or rejected")
        saved = _uint(self.saved_qwen_forwards, field="saved_qwen_forwards")
        if self.disposition == "rejected" and saved != 0:
            raise ValueError("a rejected warm transaction cannot save Qwen forwards")
        object.__setattr__(self, "saved_qwen_forwards", saved)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_binding_sha256": self.decision_binding_sha256,
            "disposition": self.disposition,
            "execution_receipt_sha256": self.execution_receipt_sha256,
            "saved_qwen_forwards": self.saved_qwen_forwards,
            "schema": WARM_ACCOUNTING_SCHEMA,
            "transaction_sha256": self.transaction_sha256,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WarmAccountingReceipt":
        expected = {
            "decision_binding_sha256",
            "disposition",
            "execution_receipt_sha256",
            "saved_qwen_forwards",
            "schema",
            "transaction_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise OoeControllerIntegrityError("warm accounting receipt is invalid")
        if value.get("schema") != WARM_ACCOUNTING_SCHEMA:
            raise OoeControllerIntegrityError("warm accounting schema is invalid")
        body = dict(value)
        body.pop("schema")
        return cls(**body)


@dataclass(frozen=True, slots=True)
class ControllerRecoveryReceipt:
    """Proof of one narrow snapshot recovery across promotion-only drift."""

    old_manifest_generation: int
    old_manifest_sha256: str
    new_manifest_generation: int
    new_manifest_sha256: str
    old_state_sha256: str
    new_state_sha256: str
    recovered_site_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        old_generation = _uint(
            self.old_manifest_generation,
            field="old_manifest_generation",
        )
        new_generation = _uint(
            self.new_manifest_generation,
            field="new_manifest_generation",
            positive=True,
        )
        if new_generation <= old_generation:
            raise ValueError("recovery manifest generation did not move forward")
        for field_name in (
            "old_manifest_sha256",
            "new_manifest_sha256",
            "old_state_sha256",
            "new_state_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )
        sites = tuple(
            sorted(
                {
                    require_sha256(value, field="recovered_site_sha256s")
                    for value in self.recovered_site_sha256s
                }
            )
        )
        object.__setattr__(self, "recovered_site_sha256s", sites)

    def to_dict(self) -> dict[str, Any]:
        return {
            "new_manifest_generation": self.new_manifest_generation,
            "new_manifest_sha256": self.new_manifest_sha256,
            "new_state_sha256": self.new_state_sha256,
            "old_manifest_generation": self.old_manifest_generation,
            "old_manifest_sha256": self.old_manifest_sha256,
            "old_state_sha256": self.old_state_sha256,
            "recovered_site_sha256s": list(self.recovered_site_sha256s),
            "schema": CONTROLLER_RECOVERY_SCHEMA,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class SiteCoverageReceipt:
    site_identity: OoeSiteIdentity
    atlas_graph_revisions: tuple[GraphRevision, ...]
    first_temporal_index: int
    last_temporal_index: int
    transition_count: int
    per_source: tuple[int, ...]
    per_target: tuple[int, ...]
    feature_receipt_sha256s: tuple[str, ...]
    transition_sha256s: tuple[str, ...]
    verifier_sha256s: tuple[str, ...]
    evidence_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.site_identity, OoeSiteIdentity):
            raise TypeError("site_identity must be an OoeSiteIdentity")
        try:
            revisions = tuple(self.atlas_graph_revisions)
        except TypeError as exc:
            raise TypeError("atlas_graph_revisions must be a sequence") from exc
        if not revisions or any(
            not isinstance(row, GraphRevision) for row in revisions
        ):
            raise ValueError("atlas_graph_revisions must contain GraphRevision values")
        revisions = tuple(sorted(set(revisions), key=lambda row: row.sequence))
        if len({row.sequence for row in revisions}) != len(revisions):
            raise ValueError("coverage contains an atlas revision fork")
        object.__setattr__(self, "atlas_graph_revisions", revisions)
        first = _uint(self.first_temporal_index, field="first_temporal_index")
        last = _uint(self.last_temporal_index, field="last_temporal_index")
        if last < first:
            raise ValueError("coverage temporal range is reversed")
        count = _uint(self.transition_count, field="transition_count", positive=True)
        if len(self.per_source) != len(OOE_ACTIONS) or len(self.per_target) != len(
            OOE_ACTIONS
        ):
            raise ValueError("coverage action vectors have invalid width")
        per_source = tuple(
            _uint(value, field="per_source entry") for value in self.per_source
        )
        per_target = tuple(
            _uint(value, field="per_target entry") for value in self.per_target
        )
        if sum(per_source) != count or sum(per_target) != count:
            raise ValueError("coverage counts do not sum to transition_count")
        object.__setattr__(self, "per_source", per_source)
        object.__setattr__(self, "per_target", per_target)
        for field_name in (
            "feature_receipt_sha256s",
            "transition_sha256s",
            "verifier_sha256s",
            "evidence_sha256s",
        ):
            values = tuple(
                sorted(
                    {
                        require_sha256(value, field=field_name)
                        for value in getattr(self, field_name)
                    }
                )
            )
            if not values:
                raise ValueError(f"{field_name} must not be empty")
            object.__setattr__(self, field_name, values)

    def as_record(self) -> dict[str, Any]:
        return {
            "atlas_graph_revisions": [
                row.to_document() for row in self.atlas_graph_revisions
            ],
            "evidence_sha256s": list(self.evidence_sha256s),
            "feature_receipt_sha256s": list(self.feature_receipt_sha256s),
            "first_temporal_index": self.first_temporal_index,
            "last_temporal_index": self.last_temporal_index,
            "per_source": list(self.per_source),
            "per_target": list(self.per_target),
            "schema": COVERAGE_SCHEMA,
            "site_identity": self.site_identity.to_dict(),
            "transition_count": self.transition_count,
            "transition_sha256s": list(self.transition_sha256s),
            "verifier_sha256s": list(self.verifier_sha256s),
        }

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())


@dataclass(slots=True)
class ControllerMetrics:
    requests: int = 0
    teacher_calls: int = 0
    saved_qwen_forwards: int = 0
    executed_qwen_forwards: int = 0
    crystal_executions: int = 0
    novelty_abstentions: int = 0
    quality_failures: int = 0
    verified_results: int = 0
    promotions: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OoeDecision:
    action: OoeAction
    origin: Literal["crystal", "teacher", "abstention"]
    reason: str
    feature_receipt_sha256: str
    site_identity_sha256: str
    crystal_sha256: str | None
    confidence: float
    teacher_called: bool
    quality_verified: bool
    routed_site_identity_sha256: str | None = None
    crystal_site_identity_sha256: str | None = None
    placebo_mode: Literal["shuffled-site", "shuffled-crystal"] | None = None
    execution_receipt_sha256: str | None = None
    warm_transaction_sha256: str | None = None
    execution_result: Any = None


def _warm_decision_binding(decision: OoeDecision) -> str:
    return _digest(
        {
            "action": decision.action,
            "confidence": decision.confidence,
            "crystal_sha256": decision.crystal_sha256,
            "crystal_site_identity_sha256": (decision.crystal_site_identity_sha256),
            "execution_receipt_sha256": decision.execution_receipt_sha256,
            "feature_receipt_sha256": decision.feature_receipt_sha256,
            "origin": decision.origin,
            "placebo_mode": decision.placebo_mode,
            "quality_verified": decision.quality_verified,
            "reason": decision.reason,
            "routed_site_identity_sha256": decision.routed_site_identity_sha256,
            "schema": "immer-ooe-warm-decision-binding/v1",
            "site_identity_sha256": decision.site_identity_sha256,
            "teacher_called": decision.teacher_called,
        }
    )


@dataclass(slots=True)
class _WarmTransaction:
    transaction_sha256: str
    decision_binding_sha256: str
    execution: ActionExecution
    status: Literal["pending", "committed", "rejected"] = "pending"
    final_receipt: WarmAccountingReceipt | None = None


@dataclass(slots=True)
class _SiteState:
    identity: OoeSiteIdentity
    agents: list[MarkovPDAgent]
    history: list[tuple[QwenOoeFeatureReceipt, VerifiedTeacherTransition]] = field(
        default_factory=list
    )
    crystal_sha256: str | None = None


Teacher = Callable[[QwenOoeFeatureReceipt, OoeAction], VerifiedTeacherTransition]
ExecutionQualityVerifier = Callable[[QwenOoeFeatureReceipt, ActionExecution], bool]
ActionExecutor = Callable[[QwenOoeFeatureReceipt], ActionExecution]
AtlasRevisionVerifier = Callable[[GraphRevision], bool]


class OoeController:
    """Persistent teacher/student controller for exact Qwen weight sites."""

    def __init__(
        self,
        *,
        model_pin_sha256: str,
        weight_graph_revision_sha256: str,
        atlas_graph_revision: GraphRevision,
        crystal_store: CrystalStore,
        config: ControllerConfig | None = None,
        action_executors: Mapping[str, ActionExecutor] | None = None,
        atlas_revision_verifier: AtlasRevisionVerifier | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self.model_pin_sha256 = require_sha256(
            model_pin_sha256, field="model_pin_sha256"
        )
        self.weight_graph_revision_sha256 = require_sha256(
            weight_graph_revision_sha256,
            field="weight_graph_revision_sha256",
        )
        if not isinstance(atlas_graph_revision, GraphRevision):
            raise TypeError("atlas_graph_revision must be a GraphRevision")
        if atlas_revision_verifier is not None and not callable(
            atlas_revision_verifier
        ):
            raise TypeError("atlas_revision_verifier must be callable")
        self._atlas_initial_revision = atlas_graph_revision
        self._atlas_head = atlas_graph_revision
        self._atlas_events: dict[int, str] = {
            atlas_graph_revision.sequence: atlas_graph_revision.event_sha256
        }
        self._atlas_revision_verifier = atlas_revision_verifier
        if not isinstance(crystal_store, CrystalStore):
            raise TypeError("crystal_store must be a CrystalStore")
        self.crystal_store = crystal_store
        self.config = config or ControllerConfig()
        self.router = AttractorRouter(
            radius=self.config.router_radius,
            min_margin=self.config.router_min_margin,
        )
        half = self.config.replicas // 2
        self._adjacency = barbell_adjacency(half, half)
        self._lifted, self._lift_parameters = adaptive_ps_lift_matrix(
            self._adjacency,
            ps=0.003,
        )
        self._sites: dict[str, _SiteState] = {}
        self._tokens: dict[str, MobileMarkovToken] = {}
        self._warm_transactions: dict[str, _WarmTransaction] = {}
        self._next_warm_transaction = 0
        self._last_temporal_index = -1
        self.metrics = ControllerMetrics()
        executors: dict[OoeAction, ActionExecutor] = {}
        for action, executor in (action_executors or {}).items():
            normalized = validate_action(action)
            if not callable(executor):
                raise TypeError(f"executor for {action!r} must be callable")
            executors[normalized] = executor
        self._executors = executors

    @property
    @_locked
    def site_identity_sha256s(self) -> tuple[str, ...]:
        return tuple(sorted(self._sites))

    @property
    @_locked
    def last_temporal_index(self) -> int:
        return self._last_temporal_index

    @property
    @_locked
    def atlas_graph_revision(self) -> GraphRevision:
        return self._atlas_head

    @property
    @_locked
    def atlas_graph_revision_sha256(self) -> str:
        return self._atlas_head.sha256

    def _accept_atlas_revision(self, revision: GraphRevision) -> None:
        if not isinstance(revision, GraphRevision):
            raise TypeError("revision must be a GraphRevision")
        known = self._atlas_events.get(revision.sequence)
        if known is not None and known != revision.event_sha256:
            raise OoeControllerStaleError(
                "atlas graph fork: one sequence has two event hashes"
            )
        if revision.sequence < self._atlas_head.sequence:
            raise OoeControllerStaleError("atlas graph revision rolled back")
        if revision.sequence == self._atlas_head.sequence:
            if revision.event_sha256 != self._atlas_head.event_sha256:
                raise OoeControllerStaleError("atlas graph head forked")
            return
        if self._atlas_revision_verifier is not None and not bool(
            self._atlas_revision_verifier(revision)
        ):
            raise OoeControllerIntegrityError(
                "atlas revision verifier rejected the forward head"
            )
        self._atlas_events[revision.sequence] = revision.event_sha256
        self._atlas_head = revision

    def _assert_pins(self, receipt: QwenOoeFeatureReceipt) -> None:
        if receipt.model_pin_sha256 != self.model_pin_sha256:
            raise OoeControllerStaleError("Qwen/OoE model pin is stale")
        if receipt.weight_graph_revision_sha256 != self.weight_graph_revision_sha256:
            raise OoeControllerStaleError("Qwen/OoE weight graph revision is stale")

    def _assert_current(self, receipt: QwenOoeFeatureReceipt) -> None:
        self._assert_pins(receipt)
        self._accept_atlas_revision(receipt.atlas_graph_revision)

    def _assert_executable_atlas_revision(self, revision: GraphRevision) -> None:
        """Accept an authenticated historical append-only head for warm replay."""

        if not isinstance(revision, GraphRevision):
            raise TypeError("revision must be a GraphRevision")
        if revision.sequence >= self._atlas_head.sequence:
            self._accept_atlas_revision(revision)
            return
        known = self._atlas_events.get(revision.sequence)
        if known is not None:
            if known != revision.event_sha256:
                raise OoeControllerStaleError(
                    "atlas graph fork: one sequence has two event hashes"
                )
            return
        if self._atlas_revision_verifier is None:
            raise OoeControllerStaleError(
                "historical Atlas revision has no authenticated chain membership"
            )
        if not bool(self._atlas_revision_verifier(revision)):
            raise OoeControllerIntegrityError(
                "atlas revision verifier rejected the historical head"
            )
        self._atlas_events[revision.sequence] = revision.event_sha256

    def _new_site(self, receipt: QwenOoeFeatureReceipt) -> _SiteState:
        return _SiteState(
            identity=receipt.site_identity,
            agents=[
                MarkovPDAgent(len(OOE_ACTIONS), len(OOE_ACTIONS))
                for _ in range(self.config.replicas)
            ],
        )

    def _replica_indices(
        self,
        receipt: QwenOoeFeatureReceipt,
        transition: VerifiedTeacherTransition,
    ) -> tuple[int, ...]:
        scores = []
        for replica in range(self.config.replicas):
            score = hashlib.sha256(
                canonical_json_bytes(
                    {
                        "feature_receipt_sha256": receipt.sha256,
                        "replica": replica,
                        "schema": "immer-ooe-replica-assignment/v1",
                        "transition_sha256": transition.sha256,
                    }
                )
            ).digest()
            scores.append((score, replica))
        scores.sort()
        return tuple(
            sorted(replica for _, replica in scores[: self.config.replica_fanout])
        )

    @_locked
    def ingest_teacher(
        self,
        receipt: QwenOoeFeatureReceipt,
        transition: VerifiedTeacherTransition,
    ) -> None:
        """Append one temporally ordered, verified teacher transition."""

        if not isinstance(receipt, QwenOoeFeatureReceipt):
            raise TypeError("receipt must be a QwenOoeFeatureReceipt")
        if not isinstance(transition, VerifiedTeacherTransition):
            raise TypeError("transition must be a VerifiedTeacherTransition")
        transition.assert_bound(receipt)
        if receipt.temporal_index <= self._last_temporal_index:
            raise OoeControllerIntegrityError(
                "teacher feature receipts must be globally temporally ordered"
            )
        if sum(len(site.history) for site in self._sites.values()) >= _MAX_HISTORY:
            raise OoeControllerError("controller history limit reached")
        site_sha256 = receipt.site_identity.sha256
        site = self._sites.get(site_sha256)
        if site is not None and site.identity != receipt.site_identity:
            raise OoeControllerStaleError("site identity changed")
        self._assert_pins(receipt)
        self._assert_executable_atlas_revision(receipt.atlas_graph_revision)
        if site is None:
            site = self._new_site(receipt)
            self._sites[site_sha256] = site

        source = _ACTION_INDEX[transition.source_action]
        target = _ACTION_INDEX[transition.target_action]
        for replica in self._replica_indices(receipt, transition):
            site.agents[replica].observe(source, target, weight=transition.weight)
        site.history.append((receipt, transition))
        # Every Crystal binds the router calibration.  Moving any centroid
        # invalidates that shared calibration, so every active publication is
        # withdrawn until it is re-promoted against the new attractor map.
        for current in self._sites.values():
            current.crystal_sha256 = None
        self.router.observe(site_sha256, receipt.sketch_array)
        self.router.radius = self.config.router_radius
        self.router.min_margin = self.config.router_min_margin
        self._last_temporal_index = receipt.temporal_index

    @_locked
    def coverage_receipt(self, site_identity_sha256: str) -> SiteCoverageReceipt:
        digest = require_sha256(
            site_identity_sha256,
            field="site_identity_sha256",
        )
        try:
            site = self._sites[digest]
        except KeyError as exc:
            raise KeyError(f"unknown OoE site: {digest}") from exc
        if not site.history:
            raise OoePromotionError("site has no verified teacher history")
        per_source = [0] * len(OOE_ACTIONS)
        per_target = [0] * len(OOE_ACTIONS)
        for _, transition in site.history:
            per_source[_ACTION_INDEX[transition.source_action]] += 1
            per_target[_ACTION_INDEX[transition.target_action]] += 1
        receipts = tuple(row[0] for row in site.history)
        transitions = tuple(row[1] for row in site.history)
        return SiteCoverageReceipt(
            site_identity=site.identity,
            atlas_graph_revisions=tuple(
                receipt.atlas_graph_revision for receipt in receipts
            ),
            first_temporal_index=receipts[0].temporal_index,
            last_temporal_index=receipts[-1].temporal_index,
            transition_count=len(transitions),
            per_source=tuple(per_source),
            per_target=tuple(per_target),
            feature_receipt_sha256s=tuple(row.sha256 for row in receipts),
            transition_sha256s=tuple(row.sha256 for row in transitions),
            verifier_sha256s=tuple(
                digest for receipt in receipts for digest in receipt.verifier_sha256s
            ),
            evidence_sha256s=tuple(
                digest for receipt in receipts for digest in receipt.evidence_sha256s
            ),
        )

    @_locked
    def feature_receipts_for_prompt(
        self,
        question_sha256: str,
        token_sha256: str | None = None,
    ) -> tuple[QwenOoeFeatureReceipt, ...]:
        """Return sealed historical features for one exact prompt identity."""

        question = require_sha256(question_sha256, field="question_sha256")
        token = (
            None
            if token_sha256 is None
            else require_sha256(token_sha256, field="token_sha256")
        )
        matches = [
            receipt
            for site in self._sites.values()
            for receipt, _ in site.history
            if receipt.probe.question_sha256 == question
            and (token is None or receipt.probe.token_sha256 == token)
        ]
        return tuple(
            sorted(
                matches,
                key=lambda row: (
                    row.temporal_index,
                    row.site_identity.sha256,
                    row.sha256,
                ),
            )
        )

    @_locked
    def latest_feature_receipt_for_prompt(
        self,
        question_sha256: str,
        token_sha256: str | None = None,
    ) -> QwenOoeFeatureReceipt | None:
        """Return one latest site-unambiguous feature, or explicitly refuse."""

        matches = self.feature_receipts_for_prompt(
            question_sha256,
            token_sha256,
        )
        if not matches:
            return None
        sites = {receipt.site_identity.sha256 for receipt in matches}
        if len(sites) != 1:
            raise OoeControllerAmbiguityError(
                "prompt resolves to multiple OoE weight sites: "
                + ", ".join(sorted(sites))
            )
        return matches[-1]

    def _calibrate_router(self) -> str:
        samples = [
            (site_sha256, receipt.sketch_array)
            for site_sha256, site in sorted(self._sites.items())
            for receipt, _ in site.history
        ]
        return self.router.calibrate(
            samples,
            radius_quantile=1.0,
            margin_quantile=0.0,
            radius_ceiling=self.config.router_radius,
        )

    @staticmethod
    def _named_hashes(prefix: str, values: Sequence[str]) -> dict[str, str]:
        return {
            f"{prefix}-{index:06d}": digest
            for index, digest in enumerate(sorted(set(values)))
        }

    def _build_crystal_payload(
        self,
        site_digest: str,
        coverage: SiteCoverageReceipt,
        calibration_sha256: str,
    ) -> CrystalPayload:
        site = self._sites[site_digest]
        replica_kernels = np.stack(
            [agent.decision_kernel().reshape(-1) for agent in site.agents]
        )
        result = measure_consensus(
            self._lifted,
            replica_kernels,
            tolerance=self.config.consensus_tolerance,
            max_rounds=self.config.consensus_max_rounds,
            lifted_nodes=self.config.replicas,
            topology="ps-lifted",
        )
        fused = normalize_rows(
            result.estimates[0].reshape(len(OOE_ACTIONS), len(OOE_ACTIONS))
        )
        consensus_receipt = result.receipt.to_dict()
        consensus_receipt.update(
            {
                "fused_kernel_sha256": array_sha256(fused),
                "fusion_entry": 0,
                "adaptive_lift": self._lift_parameters.to_dict(),
                "adaptive_lift_sha256": self._lift_parameters.sha256,
                "replica_count": self.config.replicas,
                "replica_fanout": self.config.replica_fanout,
            }
        )
        return CrystalPayload.from_kernel(
            name=site_digest,
            identity=site.identity,
            kernel=fused,
            coverage_sha256=coverage.sha256,
            calibration_sha256=calibration_sha256,
            verifier_hashes=self._named_hashes("verifier", coverage.verifier_sha256s),
            evidence_hashes=self._named_hashes("evidence", coverage.evidence_sha256s),
            consensus_receipt=consensus_receipt,
            quantization_levels=self.config.quantization_levels,
        )

    @_locked
    def promote(
        self,
        site_identity_sha256: str,
        *,
        coverage_sha256: str,
        verifier_sha256s: Sequence[str],
        expected_store_generation: int | None = None,
    ) -> CrystalPublication:
        """PS-Lifted-fuse replica kernels and atomically publish one Crystal."""

        site_digest = require_sha256(
            site_identity_sha256,
            field="site_identity_sha256",
        )
        coverage = self.coverage_receipt(site_digest)
        if require_sha256(coverage_sha256, field="coverage_sha256") != coverage.sha256:
            raise OoePromotionError("coverage hash does not match verified history")
        supplied_verifiers = tuple(
            sorted(
                {
                    require_sha256(value, field="verifier_sha256s")
                    for value in verifier_sha256s
                }
            )
        )
        if supplied_verifiers != coverage.verifier_sha256s:
            raise OoePromotionError("promotion verifier hashes are incomplete or stale")
        promoted_sources = tuple(
            OOE_ACTIONS[index]
            for index, count in enumerate(coverage.per_source)
            if count >= self.config.min_coverage_per_source
        )
        if len(promoted_sources) < self.config.min_promoted_sources:
            raise OoePromotionError(
                "coverage has only "
                f"{len(promoted_sources)} promotable sources; "
                f"requires {self.config.min_promoted_sources}"
            )
        calibration_sha256 = self._calibrate_router()
        for current in self._sites.values():
            if current.crystal_sha256 is None:
                continue
            current_payload = self.crystal_store.restore(current.crystal_sha256)
            if current_payload.calibration_sha256 != calibration_sha256:
                current.crystal_sha256 = None
        site = self._sites[site_digest]
        payload = self._build_crystal_payload(
            site_digest,
            coverage,
            calibration_sha256,
        )
        publication = self.crystal_store.publish(
            payload,
            expected_generation=expected_store_generation,
        )
        site.crystal_sha256 = publication.payload_sha256
        self.metrics.promotions += 1
        return publication

    @_locked
    def shuffled_crystal_map(self, *, seed: int = 0) -> dict[str, str]:
        """Return a deterministic derangement for explicit placebo execution."""

        active = sorted(
            site_sha256
            for site_sha256, site in self._sites.items()
            if site.crystal_sha256 is not None
        )
        if len(active) < 2:
            raise OoeControllerError(
                "crystal shuffle requires at least two active sites"
            )
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        offset = 1 + int.from_bytes(
            hashlib.sha256(str(seed).encode("ascii")).digest()[:8], "little"
        ) % (len(active) - 1)
        return {
            site: active[(index + offset) % len(active)]
            for index, site in enumerate(active)
        }

    def _token(self, stream_id: str, source: int) -> MobileMarkovToken:
        key = _canonical_text(stream_id, field="stream_id")
        token = self._tokens.get(key)
        if token is None:
            token = MobileMarkovToken.from_state(
                source,
                state_size=len(OOE_ACTIONS),
                reservoir_size=self.config.reservoir_size,
            )
            self._tokens[key] = token
        else:
            token.belief.fill(0.0)
            token.belief[source] = 1.0
        return token

    @_locked
    def decide(
        self,
        receipt: QwenOoeFeatureReceipt,
        source_action: OoeAction | str,
        *,
        stream_id: str = "default",
    ) -> OoeDecision:
        """Execute a promoted Crystal or return an explicit Qwen abstention."""

        return self._decide(
            receipt,
            source_action,
            stream_id=stream_id,
            routed_site_override=None,
            crystal_site_override=None,
            placebo_mode=None,
        )

    def _decide(
        self,
        receipt: QwenOoeFeatureReceipt,
        source_action: OoeAction | str,
        *,
        stream_id: str,
        routed_site_override: str | None,
        crystal_site_override: str | None,
        placebo_mode: Literal["shuffled-site", "shuffled-crystal"] | None,
    ) -> OoeDecision:
        """Shared exact execution path; overrides exist only for placebo calls."""

        if not isinstance(receipt, QwenOoeFeatureReceipt):
            raise TypeError("receipt must be a QwenOoeFeatureReceipt")
        self._assert_pins(receipt)
        source = validate_action(source_action)
        input_site = receipt.site_identity.sha256
        route = self.router.decision(receipt.sketch_array)
        self._assert_executable_atlas_revision(receipt.atlas_graph_revision)
        if not route.accepted or route.label != input_site:
            self.metrics.novelty_abstentions += 1
            return OoeDecision(
                action="qwen_fallback",
                origin="abstention",
                reason=route.reason if not route.accepted else "wrong-site-route",
                feature_receipt_sha256=receipt.sha256,
                site_identity_sha256=input_site,
                crystal_sha256=None,
                confidence=0.0,
                teacher_called=False,
                quality_verified=False,
            )

        routed_site = input_site
        if routed_site_override is not None:
            routed_site = require_sha256(
                routed_site_override,
                field="routed_site_override",
            )
        site = self._sites.get(routed_site)
        if site is None or site.crystal_sha256 is None:
            return OoeDecision(
                action="qwen_fallback",
                origin="abstention",
                reason="unpromoted-site",
                feature_receipt_sha256=receipt.sha256,
                site_identity_sha256=input_site,
                crystal_sha256=None,
                confidence=0.0,
                teacher_called=False,
                quality_verified=False,
                routed_site_identity_sha256=routed_site,
                placebo_mode=placebo_mode,
            )
        source_index = _ACTION_INDEX[source]
        coverage = self.coverage_receipt(routed_site)
        if coverage.per_source[source_index] < self.config.min_coverage_per_source:
            return OoeDecision(
                action="qwen_fallback",
                origin="abstention",
                reason="uncovered-source-action",
                feature_receipt_sha256=receipt.sha256,
                site_identity_sha256=input_site,
                crystal_sha256=None,
                confidence=0.0,
                teacher_called=False,
                quality_verified=False,
                routed_site_identity_sha256=routed_site,
                placebo_mode=placebo_mode,
            )

        crystal_site = routed_site
        if crystal_site_override is not None:
            crystal_site = require_sha256(
                crystal_site_override,
                field="crystal_site_override",
            )
            try:
                placebo_state = self._sites[crystal_site]
            except KeyError as exc:
                raise KeyError(f"unknown placebo crystal site: {crystal_site}") from exc
            if placebo_state.crystal_sha256 is None:
                raise OoeControllerError("placebo crystal site is not promoted")
            crystal_sha256 = placebo_state.crystal_sha256
        else:
            crystal_sha256 = site.crystal_sha256
        assert crystal_sha256 is not None
        payload = self.crystal_store.restore(crystal_sha256)
        if placebo_mode is None and payload.identity != receipt.site_identity:
            raise OoeControllerIntegrityError("Crystal execution identity mismatch")
        if payload.name != crystal_site:
            raise OoeControllerIntegrityError("Crystal name/site mismatch")
        crystal_state = self._sites[crystal_site]
        crystal_coverage = self.coverage_receipt(crystal_site)
        if (
            payload.identity != crystal_state.identity
            or payload.coverage_sha256 != crystal_coverage.sha256
            or self.router.calibration_sha256 is None
            or payload.calibration_sha256 != self.router.calibration_sha256
        ):
            raise OoeControllerIntegrityError(
                "Crystal coverage, identity, or calibration is stale"
            )
        kernel = payload.restore_kernel()
        if kernel.shape != (len(OOE_ACTIONS), len(OOE_ACTIONS)):
            raise OoeControllerIntegrityError("Crystal action kernel shape mismatch")
        distribution = kernel[source_index]
        token = self._token(stream_id, source_index)
        confidence = token.update(
            distribution,
            receipt.sketch_array,
            f"{input_site}->{routed_site}->{crystal_site}",
        )
        if not token.gate(min_confidence=self.config.token_min_confidence):
            self.metrics.novelty_abstentions += 1
            return OoeDecision(
                action="qwen_fallback",
                origin="abstention",
                reason="reservoir-rapidity-gate",
                feature_receipt_sha256=receipt.sha256,
                site_identity_sha256=input_site,
                crystal_sha256=crystal_sha256,
                confidence=confidence,
                teacher_called=False,
                quality_verified=False,
                routed_site_identity_sha256=routed_site,
                crystal_site_identity_sha256=crystal_site,
                placebo_mode=placebo_mode,
            )
        action = OOE_ACTIONS[int(np.argmax(token.belief))]
        return OoeDecision(
            action=action,
            origin="crystal",
            reason=(
                "promoted-crystal"
                if placebo_mode is None
                else f"{placebo_mode}-placebo"
            ),
            feature_receipt_sha256=receipt.sha256,
            site_identity_sha256=input_site,
            crystal_sha256=crystal_sha256,
            confidence=confidence,
            teacher_called=False,
            quality_verified=False,
            routed_site_identity_sha256=routed_site,
            crystal_site_identity_sha256=crystal_site,
            placebo_mode=placebo_mode,
        )

    @_locked
    def decide_placebo(
        self,
        receipt: QwenOoeFeatureReceipt,
        source_action: OoeAction | str,
        *,
        shuffled_sites: Mapping[str, str],
        mode: Literal["shuffled-site", "shuffled-crystal"],
        stream_id: str = "placebo",
    ) -> OoeDecision:
        """Execute an explicit wrong-site or wrong-Crystal placebo mapping."""

        if mode not in ("shuffled-site", "shuffled-crystal"):
            raise ValueError("placebo mode must be shuffled-site or shuffled-crystal")
        site_sha256 = receipt.site_identity.sha256
        try:
            override = shuffled_sites[site_sha256]
        except KeyError as exc:
            raise KeyError(f"placebo map has no entry for site: {site_sha256}") from exc
        if override == site_sha256:
            raise OoeControllerIntegrityError("placebo map is not a derangement")
        return self._decide(
            receipt,
            source_action,
            stream_id=stream_id,
            routed_site_override=(override if mode == "shuffled-site" else None),
            crystal_site_override=(override if mode == "shuffled-crystal" else None),
            placebo_mode=mode,
        )

    @staticmethod
    def _as_warm_abstention(
        candidate: OoeDecision,
        *,
        reason: str,
        execution_receipt_sha256: str | None = None,
    ) -> OoeDecision:
        return OoeDecision(
            action="qwen_fallback",
            origin="abstention",
            reason=reason,
            feature_receipt_sha256=candidate.feature_receipt_sha256,
            site_identity_sha256=candidate.site_identity_sha256,
            crystal_sha256=candidate.crystal_sha256,
            confidence=candidate.confidence,
            teacher_called=False,
            quality_verified=False,
            routed_site_identity_sha256=candidate.routed_site_identity_sha256,
            crystal_site_identity_sha256=candidate.crystal_site_identity_sha256,
            placebo_mode=candidate.placebo_mode,
            execution_receipt_sha256=execution_receipt_sha256,
        )

    def _register_warm_transaction(
        self,
        decision: OoeDecision,
        execution: ActionExecution,
    ) -> OoeDecision:
        if len(self._warm_transactions) >= _MAX_WARM_TRANSACTIONS:
            raise OoeControllerError("warm transaction limit reached")
        ordinal = self._next_warm_transaction
        self._next_warm_transaction += 1
        transaction_sha256 = _digest(
            {
                "execution_receipt_sha256": execution.sha256,
                "feature_receipt_sha256": decision.feature_receipt_sha256,
                "model_pin_sha256": self.model_pin_sha256,
                "ordinal": ordinal,
                "schema": "immer-ooe-warm-transaction/v1",
                "weight_graph_revision_sha256": self.weight_graph_revision_sha256,
            }
        )
        bound = replace(decision, warm_transaction_sha256=transaction_sha256)
        decision_binding_sha256 = _warm_decision_binding(bound)
        if transaction_sha256 in self._warm_transactions:
            raise OoeControllerIntegrityError("warm transaction digest collision")
        self._warm_transactions[transaction_sha256] = _WarmTransaction(
            transaction_sha256=transaction_sha256,
            decision_binding_sha256=decision_binding_sha256,
            execution=execution,
        )
        return bound

    def _warm_transaction_for(self, decision: OoeDecision) -> _WarmTransaction:
        if not isinstance(decision, OoeDecision):
            raise TypeError("decision must be an OoeDecision")
        transaction_sha256 = decision.warm_transaction_sha256
        if transaction_sha256 is None:
            raise OoeControllerIntegrityError(
                "decision has no pending warm transaction"
            )
        transaction_sha256 = require_sha256(
            transaction_sha256,
            field="warm_transaction_sha256",
        )
        try:
            transaction = self._warm_transactions[transaction_sha256]
        except KeyError as exc:
            raise OoeControllerIntegrityError("unknown warm transaction") from exc
        if (
            decision.origin != "crystal"
            or not decision.quality_verified
            or decision.execution_receipt_sha256 != transaction.execution.sha256
            or decision.execution_result != transaction.execution.result
            or _warm_decision_binding(decision) != transaction.decision_binding_sha256
        ):
            raise OoeControllerIntegrityError("warm decision binding mismatch")
        return transaction

    @_locked
    def commit_warm(self, decision: OoeDecision) -> WarmAccountingReceipt:
        """Commit verified warm savings after the wrapper's final adjudication."""

        transaction = self._warm_transaction_for(decision)
        if transaction.status == "rejected":
            raise OoeControllerIntegrityError(
                "cannot commit a rejected warm transaction"
            )
        if transaction.final_receipt is not None:
            return transaction.final_receipt
        receipt = WarmAccountingReceipt(
            transaction_sha256=transaction.transaction_sha256,
            decision_binding_sha256=transaction.decision_binding_sha256,
            execution_receipt_sha256=transaction.execution.sha256,
            disposition="committed",
            saved_qwen_forwards=transaction.execution.saved_qwen_forwards,
        )
        self.metrics.saved_qwen_forwards += receipt.saved_qwen_forwards
        self.metrics.crystal_executions += 1
        self.metrics.verified_results += 1
        transaction.status = "committed"
        transaction.final_receipt = receipt
        return receipt

    @_locked
    def reject_warm(self, decision: OoeDecision) -> WarmAccountingReceipt:
        """Reject a warm result without ever claiming its pending savings."""

        transaction = self._warm_transaction_for(decision)
        if transaction.status == "committed":
            raise OoeControllerIntegrityError(
                "cannot reject a committed warm transaction"
            )
        if transaction.final_receipt is not None:
            return transaction.final_receipt
        receipt = WarmAccountingReceipt(
            transaction_sha256=transaction.transaction_sha256,
            decision_binding_sha256=transaction.decision_binding_sha256,
            execution_receipt_sha256=transaction.execution.sha256,
            disposition="rejected",
            saved_qwen_forwards=0,
        )
        self.metrics.quality_failures += 1
        transaction.status = "rejected"
        transaction.final_receipt = receipt
        return receipt

    @_locked
    def try_warm(
        self,
        receipt: QwenOoeFeatureReceipt,
        source_action: OoeAction | str,
        *,
        quality_verifier: ExecutionQualityVerifier,
        stream_id: str = "default",
    ) -> OoeDecision:
        """Try one warm bypass without invoking or learning from the teacher."""

        if not callable(quality_verifier):
            raise TypeError("quality_verifier must be callable")
        self.metrics.requests += 1
        source = validate_action(source_action)
        candidate = self.decide(receipt, source, stream_id=stream_id)
        if candidate.origin != "crystal":
            return candidate
        if candidate.action == "qwen_fallback":
            return self._as_warm_abstention(
                candidate,
                reason="crystal-requested-qwen",
            )
        executor = self._executors.get(candidate.action)
        if executor is None:
            return self._as_warm_abstention(
                candidate,
                reason="missing-action-executor",
            )
        execution = executor(receipt)
        if not isinstance(execution, ActionExecution):
            raise OoeControllerIntegrityError(
                "action executor must return an ActionExecution"
            )
        execution.assert_bound(receipt, candidate.action)
        self.metrics.executed_qwen_forwards += execution.qwen_forwards
        verified = execution.quality_verified and bool(
            quality_verifier(receipt, execution)
        )
        if not verified:
            self.metrics.quality_failures += 1
            return self._as_warm_abstention(
                candidate,
                reason="execution-quality-repair",
                execution_receipt_sha256=execution.sha256,
            )
        decision = OoeDecision(
            action=candidate.action,
            origin="crystal",
            reason=candidate.reason,
            feature_receipt_sha256=candidate.feature_receipt_sha256,
            site_identity_sha256=candidate.site_identity_sha256,
            crystal_sha256=candidate.crystal_sha256,
            confidence=candidate.confidence,
            teacher_called=False,
            quality_verified=True,
            routed_site_identity_sha256=candidate.routed_site_identity_sha256,
            crystal_site_identity_sha256=candidate.crystal_site_identity_sha256,
            placebo_mode=candidate.placebo_mode,
            execution_receipt_sha256=execution.sha256,
            execution_result=execution.result,
        )
        return self._register_warm_transaction(decision, execution)

    def resolve(
        self,
        receipt: QwenOoeFeatureReceipt,
        source_action: OoeAction | str,
        *,
        teacher: Teacher,
        quality_verifier: ExecutionQualityVerifier,
        stream_id: str = "default",
    ) -> OoeDecision:
        """Try warm execution, then invoke the complete Qwen teacher on miss."""

        if not callable(teacher):
            raise TypeError("teacher must be callable")
        source = validate_action(source_action)
        candidate = self.try_warm(
            receipt,
            source,
            quality_verifier=quality_verifier,
            stream_id=stream_id,
        )
        if candidate.origin == "crystal":
            self.commit_warm(candidate)
            return candidate

        with self._lock:
            self.metrics.teacher_calls += 1
        transition = teacher(receipt, source)
        if not isinstance(transition, VerifiedTeacherTransition):
            raise OoeControllerIntegrityError(
                "teacher must return a VerifiedTeacherTransition"
            )
        transition.assert_bound(receipt)
        if transition.source_action != source:
            raise OoeControllerIntegrityError("teacher source action mismatch")
        self.ingest_teacher(receipt, transition)
        with self._lock:
            self.metrics.verified_results += 1
        return OoeDecision(
            action=transition.target_action,
            origin="teacher",
            reason=candidate.reason,
            feature_receipt_sha256=receipt.sha256,
            site_identity_sha256=receipt.site_identity.sha256,
            crystal_sha256=candidate.crystal_sha256,
            confidence=candidate.confidence,
            teacher_called=True,
            quality_verified=True,
            routed_site_identity_sha256=candidate.routed_site_identity_sha256,
            crystal_site_identity_sha256=candidate.crystal_site_identity_sha256,
            placebo_mode=candidate.placebo_mode,
        )

    def _state_record(self) -> dict[str, Any]:
        manifest = self.crystal_store.manifest()
        sites = []
        for site_sha256, site in sorted(self._sites.items()):
            sites.append(
                {
                    "crystal_sha256": site.crystal_sha256,
                    "history": [
                        {
                            "feature": receipt.to_document(),
                            "transition": transition.to_document(),
                        }
                        for receipt, transition in site.history
                    ],
                    "site_identity_sha256": site_sha256,
                }
            )
        tokens = []
        for stream_id, token in sorted(self._tokens.items()):
            tokens.append(
                {
                    "belief": token.belief.tolist(),
                    "branch_mass": token.branch_mass,
                    "fallbacks": token.fallbacks,
                    "ledger_limit": token.ledger.limit,
                    "ledger_xi": token.ledger.xi,
                    "reservoir": token.reservoir.tolist(),
                    "reservoir_coherence": token._reservoir_coherence,
                    "reservoir_decay": token.reservoir_decay,
                    "route": list(token.route),
                    "stream_id": stream_id,
                }
            )
        warm_transactions = []
        for transaction_sha256, transaction in sorted(self._warm_transactions.items()):
            warm_transactions.append(
                {
                    "decision_binding_sha256": (transaction.decision_binding_sha256),
                    "execution": transaction.execution.to_document(),
                    "final_receipt": (
                        None
                        if transaction.final_receipt is None
                        else transaction.final_receipt.to_dict()
                    ),
                    "status": transaction.status,
                    "transaction_sha256": transaction_sha256,
                }
            )
        return {
            "atlas_current_revision": self._atlas_head.to_document(),
            "atlas_initial_revision": self._atlas_initial_revision.to_document(),
            "atlas_revision_verifier_required": (
                self._atlas_revision_verifier is not None
            ),
            "atlas_seen_revisions": [
                GraphRevision(sequence, event_sha256).to_document()
                for sequence, event_sha256 in sorted(self._atlas_events.items())
            ],
            "config": self.config.to_dict(),
            "crystal_manifest": manifest.to_dict(),
            "crystal_manifest_generation": manifest.generation,
            "crystal_manifest_sha256": manifest.sha256,
            "last_temporal_index": self._last_temporal_index,
            "metrics": self.metrics.to_dict(),
            "model_pin_sha256": self.model_pin_sha256,
            "next_warm_transaction": self._next_warm_transaction,
            "router": {
                "calibration_sha256": self.router.calibration_sha256,
                "min_margin": self.router.min_margin,
                "radius": self.router.radius,
            },
            "schema": CONTROLLER_STATE_SCHEMA,
            "sites": sites,
            "tokens": tokens,
            "warm_transactions": warm_transactions,
            "weight_graph_revision_sha256": self.weight_graph_revision_sha256,
        }

    @_locked
    def snapshot_bytes(self) -> bytes:
        body = self._state_record()
        return canonical_json_bytes(
            {
                "body": body,
                "schema": CONTROLLER_STATE_SCHEMA,
                "sha256": _digest(body),
            }
        )

    @_locked
    def save_snapshot(
        self,
        *,
        name: str = CONTROLLER_STATE_NAME,
        expected_sha256: str | None = None,
    ) -> StatePublication:
        """Atomically persist deterministic state through the CrystalStore."""

        return self.crystal_store.publish_state(
            name,
            self.snapshot_bytes(),
            expected_sha256=expected_sha256,
        )

    @classmethod
    def restore(
        cls,
        *,
        crystal_store: CrystalStore,
        name: str = CONTROLLER_STATE_NAME,
        action_executors: Mapping[str, ActionExecutor] | None = None,
        atlas_revision_verifier: AtlasRevisionVerifier | None = None,
        expected_model_pin_sha256: str | None = None,
        expected_weight_graph_revision_sha256: str | None = None,
        expected_atlas_graph_revision: GraphRevision | None = None,
        _allow_manifest_forward_recovery: bool = False,
    ) -> "OoeController":
        """Restore, re-derive all replica state, and audit active Crystals."""

        raw = crystal_store.restore_state(name)
        try:
            document = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OoeControllerIntegrityError(
                "controller snapshot is invalid JSON"
            ) from exc
        if canonical_json_bytes(document) != raw:
            raise OoeControllerIntegrityError(
                "controller snapshot is not canonical JSON"
            )
        if not isinstance(document, dict) or set(document) != {
            "body",
            "schema",
            "sha256",
        }:
            raise OoeControllerIntegrityError("controller snapshot envelope is invalid")
        if document.get("schema") != CONTROLLER_STATE_SCHEMA:
            raise OoeControllerIntegrityError("controller snapshot schema is invalid")
        body = document.get("body")
        expected_fields = {
            "atlas_current_revision",
            "atlas_initial_revision",
            "atlas_revision_verifier_required",
            "atlas_seen_revisions",
            "config",
            "crystal_manifest",
            "crystal_manifest_generation",
            "crystal_manifest_sha256",
            "last_temporal_index",
            "metrics",
            "model_pin_sha256",
            "next_warm_transaction",
            "router",
            "schema",
            "sites",
            "tokens",
            "warm_transactions",
            "weight_graph_revision_sha256",
        }
        if not isinstance(body, dict) or set(body) != expected_fields:
            raise OoeControllerIntegrityError("controller snapshot body is invalid")
        if require_sha256(document.get("sha256"), field="sha256") != _digest(body):
            raise OoeControllerIntegrityError("controller snapshot SHA-256 mismatch")
        if body.get("schema") != CONTROLLER_STATE_SCHEMA:
            raise OoeControllerIntegrityError("controller state schema is invalid")

        try:
            stored_manifest = CrystalManifest.from_bytes(
                canonical_json_bytes(body["crystal_manifest"])
            )
        except (TypeError, ValueError) as exc:
            raise OoeControllerIntegrityError(
                "snapshot CrystalStore manifest is invalid"
            ) from exc
        if (
            body["crystal_manifest_generation"] != stored_manifest.generation
            or body["crystal_manifest_sha256"] != stored_manifest.sha256
        ):
            raise OoeControllerIntegrityError(
                "snapshot CrystalStore manifest binding is invalid"
            )
        current_manifest = crystal_store.manifest()
        manifest_exact = current_manifest == stored_manifest
        if not manifest_exact:
            if not _allow_manifest_forward_recovery:
                raise OoeControllerIntegrityError(
                    "CrystalStore manifest changed since snapshot"
                )
            if current_manifest.generation <= stored_manifest.generation:
                raise OoeControllerIntegrityError(
                    "recoverable CrystalStore manifest did not move strictly forward"
                )
            if not set(stored_manifest.objects).issubset(current_manifest.objects):
                raise OoeControllerIntegrityError(
                    "recoverable CrystalStore manifest lost immutable objects"
                )
            current_entries = {entry.name: entry for entry in current_manifest.entries}
            for old_entry in stored_manifest.entries:
                current_entry = current_entries.get(old_entry.name)
                if (
                    current_entry is None
                    or current_entry.identity_sha256 != old_entry.identity_sha256
                    or old_entry.payload_sha256 not in current_manifest.objects
                ):
                    raise OoeControllerIntegrityError(
                        "recoverable CrystalStore manifest changed name identity"
                    )
            audit = crystal_store.audit()
            if not audit.clean:
                raise OoeControllerIntegrityError(
                    "recoverable CrystalStore manifest failed object audit"
                )
        expected_bindings = (
            expected_model_pin_sha256,
            expected_weight_graph_revision_sha256,
        )
        actual_bindings = (
            body["model_pin_sha256"],
            body["weight_graph_revision_sha256"],
        )
        for expected, actual, field_name in zip(
            expected_bindings,
            actual_bindings,
            (
                "model_pin_sha256",
                "weight_graph_revision_sha256",
            ),
            strict=True,
        ):
            if (
                expected is not None
                and require_sha256(expected, field=field_name) != actual
            ):
                raise OoeControllerStaleError(f"snapshot {field_name} is stale")
        try:
            initial_atlas = GraphRevision.from_document(body["atlas_initial_revision"])
            current_atlas = GraphRevision.from_document(body["atlas_current_revision"])
            raw_seen_atlas = body["atlas_seen_revisions"]
            if not isinstance(raw_seen_atlas, list):
                raise TypeError("atlas_seen_revisions must be a list")
            seen_atlas = tuple(
                GraphRevision.from_document(value) for value in raw_seen_atlas
            )
        except (TypeError, ValueError) as exc:
            raise OoeControllerIntegrityError(
                "snapshot atlas revision state is invalid"
            ) from exc
        if (
            not seen_atlas
            or tuple(sorted(seen_atlas, key=lambda row: row.sequence)) != seen_atlas
            or len({row.sequence for row in seen_atlas}) != len(seen_atlas)
            or initial_atlas not in seen_atlas
            or seen_atlas[-1] != current_atlas
        ):
            raise OoeControllerIntegrityError(
                "snapshot atlas revision chain is not canonical"
            )
        verifier_required = body["atlas_revision_verifier_required"]
        if not isinstance(verifier_required, bool):
            raise OoeControllerIntegrityError(
                "atlas_revision_verifier_required must be bool"
            )
        if verifier_required and atlas_revision_verifier is None:
            raise OoeControllerIntegrityError(
                "snapshot requires its atlas revision verifier"
            )
        if expected_atlas_graph_revision is not None:
            if not isinstance(expected_atlas_graph_revision, GraphRevision):
                raise TypeError("expected_atlas_graph_revision must be a GraphRevision")
            if expected_atlas_graph_revision != current_atlas:
                raise OoeControllerStaleError("snapshot atlas graph head is stale")
        if atlas_revision_verifier is not None:
            for revision in seen_atlas:
                if not bool(atlas_revision_verifier(revision)):
                    raise OoeControllerIntegrityError(
                        "atlas revision verifier rejected snapshot history"
                    )
        try:
            config = ControllerConfig(**body["config"])
        except (TypeError, ValueError) as exc:
            raise OoeControllerIntegrityError("controller config is invalid") from exc
        controller = cls(
            model_pin_sha256=body["model_pin_sha256"],
            weight_graph_revision_sha256=body["weight_graph_revision_sha256"],
            atlas_graph_revision=initial_atlas,
            crystal_store=crystal_store,
            config=config,
            action_executors=action_executors,
            atlas_revision_verifier=atlas_revision_verifier,
        )
        raw_sites = body["sites"]
        if not isinstance(raw_sites, list):
            raise OoeControllerIntegrityError("controller sites must be a list")
        history = []
        crystal_bindings: dict[str, str | None] = {}
        for raw_site in raw_sites:
            if not isinstance(raw_site, dict) or set(raw_site) != {
                "crystal_sha256",
                "history",
                "site_identity_sha256",
            }:
                raise OoeControllerIntegrityError("controller site record is invalid")
            site_sha256 = require_sha256(
                raw_site["site_identity_sha256"], field="site_identity_sha256"
            )
            if site_sha256 in crystal_bindings:
                raise OoeControllerIntegrityError("controller site is duplicated")
            crystal_digest = raw_site["crystal_sha256"]
            if crystal_digest is not None:
                crystal_digest = require_sha256(crystal_digest, field="crystal_sha256")
            crystal_bindings[site_sha256] = crystal_digest
            if not isinstance(raw_site["history"], list):
                raise OoeControllerIntegrityError("controller site history is invalid")
            for row in raw_site["history"]:
                if not isinstance(row, dict) or set(row) != {"feature", "transition"}:
                    raise OoeControllerIntegrityError(
                        "controller history row is invalid"
                    )
                receipt = QwenOoeFeatureReceipt.from_document(row["feature"])
                transition = VerifiedTeacherTransition.from_document(row["transition"])
                if receipt.site_identity.sha256 != site_sha256:
                    raise OoeControllerIntegrityError("history row has wrong site")
                history.append((receipt.temporal_index, receipt, transition))
        history.sort(key=lambda row: row[0])
        if len({row[0] for row in history}) != len(history):
            raise OoeControllerIntegrityError("history temporal index is duplicated")
        seen_by_sequence = {row.sequence: row.event_sha256 for row in seen_atlas}
        for _, receipt, _ in history:
            revision = receipt.atlas_graph_revision
            if seen_by_sequence.get(revision.sequence) != revision.event_sha256:
                raise OoeControllerIntegrityError(
                    "history atlas revision is absent from the stored chain"
                )
        for _, receipt, transition in history:
            controller.ingest_teacher(receipt, transition)
        controller._atlas_events = dict(seen_by_sequence)
        controller._atlas_head = current_atlas
        if controller._last_temporal_index != body["last_temporal_index"]:
            raise OoeControllerIntegrityError("snapshot temporal head mismatch")
        raw_router = body["router"]
        if not isinstance(raw_router, dict) or set(raw_router) != {
            "calibration_sha256",
            "min_margin",
            "radius",
        }:
            raise OoeControllerIntegrityError("controller router state is invalid")
        stored_calibration = raw_router["calibration_sha256"]
        if stored_calibration is not None:
            stored_calibration = require_sha256(
                stored_calibration,
                field="router.calibration_sha256",
            )
        radius = float(raw_router["radius"])
        margin = float(raw_router["min_margin"])
        if not math.isfinite(radius) or radius <= 0.0:
            raise OoeControllerIntegrityError("controller router radius is invalid")
        if not math.isfinite(margin) or margin < 0.0:
            raise OoeControllerIntegrityError("controller router margin is invalid")
        if stored_calibration is None:
            if controller.router.calibration_sha256 is not None or (
                radius,
                margin,
            ) != (
                controller.router.radius,
                controller.router.min_margin,
            ):
                raise OoeControllerIntegrityError(
                    "uncalibrated router thresholds differ from ControllerConfig"
                )
        else:
            if not history:
                raise OoeControllerIntegrityError(
                    "calibrated router has no calibration samples"
                )
            derived_calibration = controller._calibrate_router()
            if (
                derived_calibration != stored_calibration
                or radius != controller.router.radius
                or margin != controller.router.min_margin
            ):
                raise OoeControllerIntegrityError(
                    "controller router calibration or thresholds cannot be reproduced"
                )

        for site_sha256, crystal_digest in crystal_bindings.items():
            if site_sha256 not in controller._sites:
                raise OoeControllerIntegrityError("snapshot contains an empty site")
            if crystal_digest is None:
                continue
            payload = crystal_store.restore(crystal_digest)
            site = controller._sites[site_sha256]
            coverage = controller.coverage_receipt(site_sha256)
            if (
                payload.name != site_sha256
                or payload.identity != site.identity
                or payload.coverage_sha256 != coverage.sha256
                or payload.calibration_sha256 != controller.router.calibration_sha256
            ):
                raise OoeControllerIntegrityError("snapshot Crystal binding is invalid")
            site.crystal_sha256 = crystal_digest

        raw_tokens = body["tokens"]
        if not isinstance(raw_tokens, list):
            raise OoeControllerIntegrityError("controller tokens must be a list")
        for raw_token in raw_tokens:
            expected_token_fields = {
                "belief",
                "branch_mass",
                "fallbacks",
                "ledger_limit",
                "ledger_xi",
                "reservoir",
                "reservoir_coherence",
                "reservoir_decay",
                "route",
                "stream_id",
            }
            if (
                not isinstance(raw_token, dict)
                or set(raw_token) != expected_token_fields
            ):
                raise OoeControllerIntegrityError("controller token record is invalid")
            stream_id = _canonical_text(raw_token["stream_id"], field="stream_id")
            if stream_id in controller._tokens:
                raise OoeControllerIntegrityError("controller token is duplicated")
            try:
                token = MobileMarkovToken(
                    belief=np.asarray(raw_token["belief"], dtype=np.float64),
                    reservoir=np.asarray(raw_token["reservoir"], dtype=np.float64),
                    ledger=RapidityLedger(
                        raw_token["ledger_xi"], raw_token["ledger_limit"]
                    ),
                    route=list(raw_token["route"]),
                    branch_mass=raw_token["branch_mass"],
                    fallbacks=raw_token["fallbacks"],
                    reservoir_decay=raw_token["reservoir_decay"],
                    _reservoir_coherence=raw_token["reservoir_coherence"],
                )
            except (TypeError, ValueError) as exc:
                raise OoeControllerIntegrityError(
                    "controller token is invalid"
                ) from exc
            controller._tokens[stream_id] = token

        next_warm_transaction = _uint(
            body["next_warm_transaction"],
            field="next_warm_transaction",
        )
        raw_transactions = body["warm_transactions"]
        if not isinstance(raw_transactions, list):
            raise OoeControllerIntegrityError("warm transactions must be a list")
        if len(raw_transactions) > _MAX_WARM_TRANSACTIONS:
            raise OoeControllerIntegrityError("warm transaction limit exceeded")
        for raw_transaction in raw_transactions:
            expected_transaction_fields = {
                "decision_binding_sha256",
                "execution",
                "final_receipt",
                "status",
                "transaction_sha256",
            }
            if (
                not isinstance(raw_transaction, dict)
                or set(raw_transaction) != expected_transaction_fields
            ):
                raise OoeControllerIntegrityError("warm transaction record is invalid")
            transaction_sha256 = require_sha256(
                raw_transaction["transaction_sha256"],
                field="transaction_sha256",
            )
            if transaction_sha256 in controller._warm_transactions:
                raise OoeControllerIntegrityError("warm transaction is duplicated")
            decision_binding_sha256 = require_sha256(
                raw_transaction["decision_binding_sha256"],
                field="decision_binding_sha256",
            )
            execution = ActionExecution.from_document(raw_transaction["execution"])
            status = raw_transaction["status"]
            if status not in ("pending", "committed", "rejected"):
                raise OoeControllerIntegrityError("warm transaction status is invalid")
            raw_final = raw_transaction["final_receipt"]
            final_receipt = (
                None
                if raw_final is None
                else WarmAccountingReceipt.from_dict(raw_final)
            )
            if (status == "pending") != (final_receipt is None):
                raise OoeControllerIntegrityError(
                    "warm transaction disposition is inconsistent"
                )
            if final_receipt is not None and (
                final_receipt.transaction_sha256 != transaction_sha256
                or final_receipt.decision_binding_sha256 != decision_binding_sha256
                or final_receipt.execution_receipt_sha256 != execution.sha256
                or final_receipt.disposition != status
            ):
                raise OoeControllerIntegrityError(
                    "warm transaction final receipt binding is invalid"
                )
            controller._warm_transactions[transaction_sha256] = _WarmTransaction(
                transaction_sha256=transaction_sha256,
                decision_binding_sha256=decision_binding_sha256,
                execution=execution,
                status=status,
                final_receipt=final_receipt,
            )
        if next_warm_transaction < len(controller._warm_transactions):
            raise OoeControllerIntegrityError("warm transaction ordinal rolled back")
        controller._next_warm_transaction = next_warm_transaction

        metrics = body["metrics"]
        if not isinstance(metrics, dict) or set(metrics) != set(
            ControllerMetrics.__dataclass_fields__
        ):
            raise OoeControllerIntegrityError("controller metrics are invalid")
        try:
            valid_metrics = all(
                _uint(value, field=f"metrics.{name}") == value
                for name, value in metrics.items()
            )
        except ValueError as exc:
            raise OoeControllerIntegrityError("controller metrics are invalid") from exc
        if not valid_metrics:
            raise OoeControllerIntegrityError("controller metrics are invalid")
        controller.metrics = ControllerMetrics(**metrics)
        return controller

    @classmethod
    def restore_recoverable(
        cls,
        *,
        crystal_store: CrystalStore,
        name: str = CONTROLLER_STATE_NAME,
        action_executors: Mapping[str, ActionExecutor] | None = None,
        atlas_revision_verifier: AtlasRevisionVerifier | None = None,
        expected_model_pin_sha256: str | None = None,
        expected_weight_graph_revision_sha256: str | None = None,
        expected_atlas_graph_revision: GraphRevision | None = None,
        expected_old_manifest_generation: int | None = None,
        expected_old_manifest_sha256: str | None = None,
        expected_old_state_sha256: str | None = None,
    ) -> tuple["OoeController", ControllerRecoveryReceipt]:
        """Recover only promotion-complete manifest drift from an older state."""

        old_state = crystal_store.restore_state(name)
        old_state_sha256 = hashlib.sha256(old_state).hexdigest()
        try:
            old_document = json.loads(old_state)
            old_body = old_document["body"]
            old_manifest = CrystalManifest.from_bytes(
                canonical_json_bytes(old_body["crystal_manifest"])
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise OoeControllerIntegrityError(
                "recoverable controller snapshot is invalid"
            ) from exc
        if (
            expected_old_manifest_generation is not None
            and _uint(
                expected_old_manifest_generation,
                field="expected_old_manifest_generation",
            )
            != old_manifest.generation
        ):
            raise OoeControllerStaleError(
                "recoverable old manifest generation differs from prepared intent"
            )
        if (
            expected_old_manifest_sha256 is not None
            and require_sha256(
                expected_old_manifest_sha256,
                field="expected_old_manifest_sha256",
            )
            != old_manifest.sha256
        ):
            raise OoeControllerStaleError(
                "recoverable old manifest hash differs from prepared intent"
            )
        if (
            expected_old_state_sha256 is not None
            and require_sha256(
                expected_old_state_sha256,
                field="expected_old_state_sha256",
            )
            != old_state_sha256
        ):
            raise OoeControllerStaleError(
                "recoverable old state hash differs from prepared intent"
            )
        drifted_manifest = crystal_store.manifest()
        if drifted_manifest.generation <= old_manifest.generation:
            raise OoeControllerIntegrityError(
                "recoverable CrystalStore manifest did not move strictly forward"
            )
        controller = cls.restore(
            crystal_store=crystal_store,
            name=name,
            action_executors=action_executors,
            atlas_revision_verifier=atlas_revision_verifier,
            expected_model_pin_sha256=expected_model_pin_sha256,
            expected_weight_graph_revision_sha256=(
                expected_weight_graph_revision_sha256
            ),
            expected_atlas_graph_revision=expected_atlas_graph_revision,
            _allow_manifest_forward_recovery=True,
        )
        if hashlib.sha256(crystal_store.restore_state(name)).hexdigest() != (
            old_state_sha256
        ):
            raise OoeControllerIntegrityError(
                "controller state changed during recovery"
            )

        recovered_sites: list[str] = []
        with controller._lock:
            known_sites = set(controller.site_identity_sha256s)
            calibration_sha256 = controller._calibrate_router()
            eligible: dict[str, tuple[SiteCoverageReceipt, CrystalPayload]] = {}
            for site_sha256 in sorted(known_sites):
                coverage = controller.coverage_receipt(site_sha256)
                covered_sources = sum(
                    count >= controller.config.min_coverage_per_source
                    for count in coverage.per_source
                )
                if covered_sources < controller.config.min_promoted_sources:
                    continue
                eligible[site_sha256] = (
                    coverage,
                    controller._build_crystal_payload(
                        site_sha256,
                        coverage,
                        calibration_sha256,
                    ),
                )

            current_manifest = crystal_store.manifest()
            if current_manifest != drifted_manifest:
                raise OoeControllerIntegrityError(
                    "CrystalStore manifest changed during recovery validation"
                )
            old_entries = {entry.name: entry for entry in old_manifest.entries}
            changed_names: set[str] = set()
            for entry in current_manifest.entries:
                if entry.name not in known_sites:
                    raise OoeControllerIntegrityError(
                        "recoverable manifest declares an unknown site name"
                    )
                site = controller._sites[entry.name]
                if entry.identity_sha256 != site.identity.sha256:
                    raise OoeControllerIntegrityError(
                        "recoverable manifest site identity is invalid"
                    )
                expected = eligible.get(entry.name)
                old_entry = old_entries.get(entry.name)
                is_unchanged_old = (
                    old_entry is not None
                    and old_entry.payload_sha256 == entry.payload_sha256
                    and old_entry.identity_sha256 == entry.identity_sha256
                )
                is_promoted_current = (
                    expected is not None and entry.payload_sha256 == expected[1].sha256
                )
                if not is_unchanged_old and not is_promoted_current:
                    raise OoeControllerIntegrityError(
                        "recoverable manifest payload is neither prepared nor current"
                    )
                payload = crystal_store.restore(entry.payload_sha256)
                if payload.name != entry.name or payload.identity != site.identity:
                    raise OoeControllerIntegrityError(
                        "recoverable manifest payload identity is invalid"
                    )
                if is_promoted_current and payload.to_bytes() != expected[1].to_bytes():
                    raise OoeControllerIntegrityError(
                        "recoverable current payload bytes are invalid"
                    )
                if not is_unchanged_old:
                    changed_names.add(entry.name)
            generation_delta = current_manifest.generation - old_manifest.generation
            if generation_delta != len(changed_names):
                raise OoeControllerIntegrityError(
                    "recoverable manifest generation exceeds prepared promotions"
                )
            allowed_objects = set(old_manifest.objects)
            allowed_objects.update(
                entry.payload_sha256
                for entry in current_manifest.entries
                if entry.name in changed_names
            )
            if set(current_manifest.objects) != allowed_objects:
                raise OoeControllerIntegrityError(
                    "recoverable manifest contains undeclared extra objects"
                )

            for site_sha256, (coverage, _) in sorted(eligible.items()):
                generation = crystal_store.manifest().generation
                controller.promote(
                    site_sha256,
                    coverage_sha256=coverage.sha256,
                    verifier_sha256s=coverage.verifier_sha256s,
                    expected_store_generation=generation,
                )
                recovered_sites.append(site_sha256)
            publication = controller.save_snapshot(
                name=name,
                expected_sha256=old_state_sha256,
            )
        new_manifest = crystal_store.manifest()
        audit = crystal_store.audit()
        if not audit.clean:
            raise OoeControllerIntegrityError(
                "recovered CrystalStore failed final object audit"
            )
        saved_state = crystal_store.restore_state(name)
        if hashlib.sha256(saved_state).hexdigest() != publication.payload_sha256:
            raise OoeControllerIntegrityError(
                "recovered controller state publication mismatch"
            )
        recovered = ControllerRecoveryReceipt(
            old_manifest_generation=old_manifest.generation,
            old_manifest_sha256=old_manifest.sha256,
            new_manifest_generation=new_manifest.generation,
            new_manifest_sha256=new_manifest.sha256,
            old_state_sha256=old_state_sha256,
            new_state_sha256=publication.payload_sha256,
            recovered_site_sha256s=tuple(recovered_sites),
        )
        return controller, recovered


__all__ = [
    "ACTION_EXECUTION_SCHEMA",
    "ActionExecution",
    "ActionExecutor",
    "AtlasRevisionVerifier",
    "CONTROLLER_RECOVERY_SCHEMA",
    "CONTROLLER_STATE_NAME",
    "CONTROLLER_STATE_SCHEMA",
    "COVERAGE_SCHEMA",
    "ControllerConfig",
    "ControllerMetrics",
    "ControllerRecoveryReceipt",
    "ExecutionQualityVerifier",
    "OoeController",
    "OoeControllerAmbiguityError",
    "OoeControllerError",
    "OoeControllerIntegrityError",
    "OoeControllerStaleError",
    "OoeDecision",
    "OoePromotionError",
    "SiteCoverageReceipt",
    "TEACHER_TRANSITION_SCHEMA",
    "Teacher",
    "VerifiedTeacherTransition",
    "WARM_ACCOUNTING_SCHEMA",
    "WarmAccountingReceipt",
]
