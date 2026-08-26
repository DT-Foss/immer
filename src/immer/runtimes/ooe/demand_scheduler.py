"""Verified demand learning for materialized compute routes.

The scheduler observes only authenticated ``MaterializedRoute`` values from an
exact ``ComputeOperatorGraphState``.  Its event log is append-only and stored as
canonical, content-addressed controller state.  UCB exploration, PPM backoff,
co-occurrence prefetch, and retention are therefore reproducible from the same
sealed evidence after a restart.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
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
from typing import Iterator, cast

from .compute_crystals import ComputeCrystalBank, ComputeCrystalError
from .compute_graph import (
    ComputeOperatorGraphState,
    ComputeRoutePlan,
    MaterializedRoute,
)
from .crystal import (
    CrystalStore,
    CrystalStoreError,
    ManifestConflictError,
)
from .identity import canonical_json_bytes, require_sha256


DEMAND_OUTCOME_SCHEMA = "immer-ooe-operator-demand-outcome/v1"
DEMAND_SELECTION_EVENT_SCHEMA = "immer-ooe-operator-demand-selection-event/v1"
DEMAND_OUTCOME_EVENT_SCHEMA = "immer-ooe-operator-demand-outcome-event/v1"
DEMAND_PINS_EVENT_SCHEMA = "immer-ooe-operator-demand-pins-event/v1"
DEMAND_STATE_SCHEMA = "immer-ooe-operator-demand-state/v1"
DEMAND_COMMIT_SCHEMA = "immer-ooe-operator-demand-commit/v1"
DEMAND_TRANSITION_SCHEMA = "immer-ooe-operator-demand-transition/v1"
UCB_SELECTION_SCHEMA = "immer-ooe-operator-demand-ucb1-selection/v1"
PPM_PREDICTION_SCHEMA = "immer-ooe-operator-demand-ppm-prediction/v1"
COOCCURRENCE_SCHEMA = "immer-ooe-operator-demand-cooccurrence/v1"
PREFETCH_SCHEMA = "immer-ooe-operator-demand-prefetch/v1"
RETENTION_SCHEMA = "immer-ooe-operator-demand-retention/v1"
SCHEDULER_CONFIG_SCHEMA = "immer-ooe-operator-demand-config/v1"

_SEALED_RECEIPT_SCHEMAS = frozenset(
    {
        DEMAND_OUTCOME_SCHEMA,
        DEMAND_STATE_SCHEMA,
        DEMAND_COMMIT_SCHEMA,
        DEMAND_TRANSITION_SCHEMA,
        UCB_SELECTION_SCHEMA,
        PPM_PREDICTION_SCHEMA,
        COOCCURRENCE_SCHEMA,
        PREFETCH_SCHEMA,
        RETENTION_SCHEMA,
    }
)

DEMAND_STATE_NAME = "ooe-operator-demand-scheduler/v1"
DEMAND_HISTORY_PREFIX = "ooe-operator-demand-history/v1:"
DEMAND_COMMIT_PREFIX = "ooe-operator-demand-commit/v1:"

MAX_DEMAND_STATE_BYTES = 48 * 1024 * 1024
MAX_DEMAND_EVENTS = 65_536
MAX_DEMAND_ROUTES = 65_536
MAX_CONTEXT_ORDER = 64
MAX_COOCCURRENCE_WINDOW = 1024
MAX_TEXT_BYTES = 1024
MAX_BUDGET = (1 << 63) - 1

_LOCK_NAME = ".operator-demand-scheduler.lock"
_STATE_NAME_RE = re.compile(
    rb'^\{"format":"immer-ooe-controller-state/v1","generation":[1-9][0-9]*,"name":"([^"\\]*)","payload_base64":"'
)


class OperatorDemandError(RuntimeError):
    """Base error for verified route-demand scheduling."""


class OperatorDemandIntegrityError(OperatorDemandError):
    """A receipt, graph binding, history object, or state failed validation."""


class OperatorDemandConflictError(OperatorDemandError):
    """A compare-and-swap precondition or unique evidence binding conflicted."""


class OperatorDemandUnavailableError(OperatorDemandError):
    """No materialized route satisfies the requested graph and ABI binding."""


class OperatorDemandBudgetError(OperatorDemandError):
    """Protected items cannot fit inside the requested exact budget."""


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_json(data: bytes, *, label: str, maximum: int) -> object:
    if not isinstance(data, bytes):
        raise TypeError(f"{label} must be immutable bytes")
    if len(data) > maximum:
        raise OperatorDemandIntegrityError(f"{label} exceeds its byte bound")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise OperatorDemandIntegrityError(f"{label} is not strict JSON") from exc
    if canonical_json_bytes(value) != data:
        raise OperatorDemandIntegrityError(f"{label} is not canonical JSON")
    return value


def _uint(value: object, *, field: str, positive: bool = False) -> int:
    lower = 1 if positive else 0
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not lower <= value <= MAX_BUDGET
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must be a bounded {qualifier} integer")
    return value


def _finite(value: object, *, field: str, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not lower <= result <= upper:
        raise ValueError(f"{field} must lie in [{lower}, {upper}]")
    return 0.0 if result == 0.0 else result


def _text(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > MAX_TEXT_BYTES
    ):
        raise ValueError(f"{field} must be canonical non-empty text")
    return value


def _hashes(
    values: Sequence[str],
    *,
    field: str,
    sorted_unique: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field} must be a sequence")
    result = tuple(require_sha256(item, field=field) for item in values)
    if sorted_unique and result != tuple(sorted(set(result))):
        raise ValueError(f"{field} must be sorted and unique")
    return result


def _sealed_document(schema: str, body: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(body)
    return {"schema": schema, "body": normalized, "body_sha256": _sha256(normalized)}


def _decode_sealed(
    value: object,
    *,
    schema: str,
    label: str,
) -> Mapping[str, object]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema", "body", "body_sha256"}
        or value.get("schema") != schema
    ):
        raise OperatorDemandIntegrityError(f"invalid {label} envelope")
    body = value.get("body")
    if not isinstance(body, Mapping):
        raise OperatorDemandIntegrityError(f"invalid {label} body")
    try:
        claimed = require_sha256(value.get("body_sha256"), field="body_sha256")
    except ValueError as exc:
        raise OperatorDemandIntegrityError(f"invalid {label} body hash") from exc
    if claimed != _sha256(body):
        raise OperatorDemandIntegrityError(f"{label} body hash mismatch")
    return body


def verify_demand_receipt(
    data: bytes,
    *,
    expected_schema: str | None = None,
) -> str:
    """Verify canonical bytes and the self-seal; return the content address."""

    value = _strict_json(
        data, label="operator-demand receipt", maximum=MAX_DEMAND_STATE_BYTES
    )
    if not isinstance(value, Mapping):
        raise OperatorDemandIntegrityError("operator-demand receipt is not an object")
    schema = value.get("schema")
    if not isinstance(schema, str) or schema not in _SEALED_RECEIPT_SCHEMAS:
        raise OperatorDemandIntegrityError("unknown operator-demand receipt schema")
    if expected_schema is not None and schema != expected_schema:
        raise OperatorDemandIntegrityError("operator-demand receipt schema mismatch")
    _decode_sealed(value, schema=schema, label="operator-demand receipt")
    _verify_typed_demand_receipt(data, schema=schema, value=value)
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, slots=True)
class OperatorDemandConfig:
    """Identity-bound deterministic scheduler parameters."""

    exploration: float = 1.0
    max_context_order: int = 8
    cooccurrence_window: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "exploration",
            _finite(
                self.exploration,
                field="exploration",
                lower=0.0,
                upper=1_000_000.0,
            ),
        )
        order = _uint(self.max_context_order, field="max_context_order", positive=True)
        window = _uint(
            self.cooccurrence_window,
            field="cooccurrence_window",
            positive=True,
        )
        if order > MAX_CONTEXT_ORDER:
            raise ValueError(f"max_context_order exceeds {MAX_CONTEXT_ORDER}")
        if window > MAX_COOCCURRENCE_WINDOW:
            raise ValueError(
                f"cooccurrence_window exceeds {MAX_COOCCURRENCE_WINDOW}"
            )
        object.__setattr__(self, "max_context_order", order)
        object.__setattr__(self, "cooccurrence_window", window)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEDULER_CONFIG_SCHEMA,
            "exploration": self.exploration,
            "max_context_order": self.max_context_order,
            "cooccurrence_window": self.cooccurrence_window,
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.to_dict())


def _route_map(state: ComputeOperatorGraphState) -> dict[str, MaterializedRoute]:
    if not isinstance(state, ComputeOperatorGraphState):
        raise TypeError("graph_state must be a ComputeOperatorGraphState")
    return {route.sha256: route for route in state.materialized_routes}


def _require_route(
    state: ComputeOperatorGraphState,
    route_sha256: str,
) -> MaterializedRoute:
    address = require_sha256(route_sha256, field="route_sha256")
    try:
        return _route_map(state)[address]
    except KeyError as exc:
        raise OperatorDemandIntegrityError(
            "route is absent from the exact graph revision"
        ) from exc


def _require_exact_route(
    state: ComputeOperatorGraphState,
    route: MaterializedRoute,
) -> None:
    if not isinstance(route, MaterializedRoute):
        raise TypeError("route must be a MaterializedRoute")
    if _require_route(state, route.sha256) != route:
        raise OperatorDemandIntegrityError("route bytes differ from graph inventory")


@dataclass(frozen=True, slots=True)
class DemandOutcomeReceipt:
    """A verified positive result or an exact route/revision negative result."""

    success: bool
    reward: float
    route_sha256: str
    graph_generation: int
    graph_state_sha256: str
    input_abi_sha256: str
    output_abi_sha256: str
    route_verifier_sha256s: tuple[str, ...]
    route_evidence_sha256s: tuple[str, ...]
    outcome_evidence_sha256: str
    outcome_verifier_sha256: str
    selection_event_sha256: str | None = None
    episode_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise TypeError("success must be a bool")
        reward = _finite(
            self.reward,
            field="reward",
            lower=-1_000_000.0,
            upper=1_000_000.0,
        )
        if self.success and reward <= 0.0:
            raise ValueError("a positive outcome requires positive reward")
        if not self.success and reward >= 0.0:
            raise ValueError("a negative outcome requires negative reward")
        object.__setattr__(self, "reward", reward)
        object.__setattr__(
            self,
            "graph_generation",
            _uint(self.graph_generation, field="graph_generation"),
        )
        for field in (
            "route_sha256",
            "graph_state_sha256",
            "input_abi_sha256",
            "output_abi_sha256",
            "outcome_evidence_sha256",
        ):
            object.__setattr__(
                self,
                field,
                require_sha256(getattr(self, field), field=field),
            )
        object.__setattr__(
            self,
            "route_verifier_sha256s",
            _hashes(
                self.route_verifier_sha256s,
                field="route_verifier_sha256s",
                sorted_unique=True,
            ),
        )
        object.__setattr__(
            self,
            "route_evidence_sha256s",
            _hashes(
                self.route_evidence_sha256s,
                field="route_evidence_sha256s",
                sorted_unique=True,
            ),
        )
        if not self.route_verifier_sha256s or not self.route_evidence_sha256s:
            raise ValueError("route provenance inventories must be non-empty")
        object.__setattr__(
            self,
            "outcome_verifier_sha256",
            require_sha256(
                self.outcome_verifier_sha256,
                field="outcome_verifier_sha256",
            ),
        )
        for field in ("selection_event_sha256", "episode_sha256"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(
                    self, field, require_sha256(value, field=field)
                )

    @classmethod
    def create(
        cls,
        *,
        route: MaterializedRoute,
        graph_state: ComputeOperatorGraphState,
        success: bool,
        reward: float,
        outcome_evidence_sha256: str,
        outcome_verifier_sha256: str,
        selection_event_sha256: str | None = None,
        episode_sha256: str | None = None,
    ) -> "DemandOutcomeReceipt":
        _require_exact_route(graph_state, route)
        return cls(
            success=success,
            reward=reward,
            route_sha256=route.sha256,
            graph_generation=graph_state.generation,
            graph_state_sha256=graph_state.sha256,
            input_abi_sha256=route.input_abi_sha256,
            output_abi_sha256=route.output_abi_sha256,
            route_verifier_sha256s=route.verifier_sha256s,
            route_evidence_sha256s=route.evidence_sha256s,
            outcome_evidence_sha256=outcome_evidence_sha256,
            outcome_verifier_sha256=outcome_verifier_sha256,
            selection_event_sha256=selection_event_sha256,
            episode_sha256=episode_sha256,
        )

    def to_dict(self) -> dict[str, object]:
        body = {
            "success": self.success,
            "reward": self.reward,
            "route_sha256": self.route_sha256,
            "graph_generation": self.graph_generation,
            "graph_state_sha256": self.graph_state_sha256,
            "input_abi_sha256": self.input_abi_sha256,
            "output_abi_sha256": self.output_abi_sha256,
            "route_verifier_sha256s": list(self.route_verifier_sha256s),
            "route_evidence_sha256s": list(self.route_evidence_sha256s),
            "outcome_evidence_sha256": self.outcome_evidence_sha256,
            "outcome_verifier_sha256": self.outcome_verifier_sha256,
            "selection_event_sha256": self.selection_event_sha256,
            "episode_sha256": self.episode_sha256,
        }
        return _sealed_document(DEMAND_OUTCOME_SCHEMA, body)

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> "DemandOutcomeReceipt":
        body = _decode_sealed(
            value, schema=DEMAND_OUTCOME_SCHEMA, label="demand outcome"
        )
        expected = {
            "success",
            "reward",
            "route_sha256",
            "graph_generation",
            "graph_state_sha256",
            "input_abi_sha256",
            "output_abi_sha256",
            "route_verifier_sha256s",
            "route_evidence_sha256s",
            "outcome_evidence_sha256",
            "outcome_verifier_sha256",
            "selection_event_sha256",
            "episode_sha256",
        }
        if set(body) != expected:
            raise OperatorDemandIntegrityError("invalid demand outcome body")
        verifiers = body.get("route_verifier_sha256s")
        evidence = body.get("route_evidence_sha256s")
        if not isinstance(verifiers, list) or not isinstance(evidence, list):
            raise OperatorDemandIntegrityError("invalid route provenance inventory")
        try:
            result = cls(
                success=cast(bool, body.get("success")),
                reward=cast(float, body.get("reward")),
                route_sha256=cast(str, body.get("route_sha256")),
                graph_generation=cast(int, body.get("graph_generation")),
                graph_state_sha256=cast(str, body.get("graph_state_sha256")),
                input_abi_sha256=cast(str, body.get("input_abi_sha256")),
                output_abi_sha256=cast(str, body.get("output_abi_sha256")),
                route_verifier_sha256s=tuple(verifiers),
                route_evidence_sha256s=tuple(evidence),
                outcome_evidence_sha256=cast(
                    str, body.get("outcome_evidence_sha256")
                ),
                outcome_verifier_sha256=cast(
                    str, body.get("outcome_verifier_sha256")
                ),
                selection_event_sha256=cast(
                    str | None, body.get("selection_event_sha256")
                ),
                episode_sha256=cast(str | None, body.get("episode_sha256")),
            )
        except (TypeError, ValueError) as exc:
            raise OperatorDemandIntegrityError(
                "demand outcome validation failed"
            ) from exc
        if result.to_dict() != dict(value):
            raise OperatorDemandIntegrityError(
                "demand outcome failed canonical reconstruction"
            )
        return result

    @classmethod
    def from_bytes(cls, data: bytes) -> "DemandOutcomeReceipt":
        value = _strict_json(data, label="demand outcome", maximum=256 * 1024)
        result = cls.from_dict(value)
        if result.to_bytes() != data:
            raise OperatorDemandIntegrityError("demand outcome bytes changed")
        return result


@dataclass(frozen=True, slots=True)
class UCBScore:
    route_sha256: str
    pulls: int
    reward_sum: float
    score: float | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "route_sha256",
            require_sha256(self.route_sha256, field="route_sha256"),
        )
        object.__setattr__(self, "pulls", _uint(self.pulls, field="pulls"))
        reward = _finite(
            self.reward_sum,
            field="reward_sum",
            lower=-float(MAX_BUDGET),
            upper=float(MAX_BUDGET),
        )
        object.__setattr__(self, "reward_sum", reward)
        if self.score is not None:
            object.__setattr__(
                self,
                "score",
                _finite(
                    self.score,
                    field="score",
                    lower=-float(MAX_BUDGET),
                    upper=float(MAX_BUDGET),
                ),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "route_sha256": self.route_sha256,
            "pulls": self.pulls,
            "reward_sum": self.reward_sum,
            "score": self.score,
        }


@dataclass(frozen=True, slots=True)
class SelectionEvent:
    logical_time: int
    route_sha256: str
    graph_generation: int
    graph_state_sha256: str
    input_abi_sha256: str
    output_abi_sha256: str | None
    exploration: float
    scores: tuple[UCBScore, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "logical_time", _uint(self.logical_time, field="logical_time", positive=True)
        )
        object.__setattr__(
            self,
            "graph_generation",
            _uint(self.graph_generation, field="graph_generation"),
        )
        for field in ("route_sha256", "graph_state_sha256", "input_abi_sha256"):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        if self.output_abi_sha256 is not None:
            object.__setattr__(
                self,
                "output_abi_sha256",
                require_sha256(self.output_abi_sha256, field="output_abi_sha256"),
            )
        object.__setattr__(
            self,
            "exploration",
            _finite(
                self.exploration,
                field="exploration",
                lower=0.0,
                upper=1_000_000.0,
            ),
        )
        scores = tuple(self.scores)
        if not scores or any(not isinstance(item, UCBScore) for item in scores):
            raise ValueError("selection scores must be non-empty UCBScore values")
        if tuple(sorted(scores, key=lambda item: item.route_sha256)) != scores:
            raise ValueError("selection scores must be route-sorted")
        if len({item.route_sha256 for item in scores}) != len(scores):
            raise ValueError("selection scores must be unique")
        if self.route_sha256 not in {item.route_sha256 for item in scores}:
            raise ValueError("selected route is absent from score inventory")
        unseen = [item.route_sha256 for item in scores if item.pulls == 0]
        if unseen:
            expected = min(unseen)
        else:
            if any(item.score is None for item in scores):
                raise ValueError("seen UCB arms require finite scores")
            expected = min(
                scores,
                key=lambda item: (
                    -cast(float, item.score),
                    item.route_sha256,
                ),
            ).route_sha256
        if self.route_sha256 != expected:
            raise ValueError("selection is not the deterministic UCB1 winner")
        object.__setattr__(self, "scores", scores)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": DEMAND_SELECTION_EVENT_SCHEMA,
            "logical_time": self.logical_time,
            "route_sha256": self.route_sha256,
            "graph_generation": self.graph_generation,
            "graph_state_sha256": self.graph_state_sha256,
            "input_abi_sha256": self.input_abi_sha256,
            "output_abi_sha256": self.output_abi_sha256,
            "exploration": self.exploration,
            "scores": [item.to_dict() for item in self.scores],
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class OutcomeEvent:
    logical_time: int
    outcome: DemandOutcomeReceipt

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "logical_time", _uint(self.logical_time, field="logical_time", positive=True)
        )
        if not isinstance(self.outcome, DemandOutcomeReceipt):
            raise TypeError("outcome must be a DemandOutcomeReceipt")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": DEMAND_OUTCOME_EVENT_SCHEMA,
            "logical_time": self.logical_time,
            "outcome": self.outcome.to_dict(),
            "outcome_sha256": self.outcome.sha256,
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class PinsEvent:
    logical_time: int
    graph_generation: int
    graph_state_sha256: str
    route_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "logical_time", _uint(self.logical_time, field="logical_time", positive=True)
        )
        object.__setattr__(
            self,
            "graph_generation",
            _uint(self.graph_generation, field="graph_generation"),
        )
        object.__setattr__(
            self,
            "graph_state_sha256",
            require_sha256(self.graph_state_sha256, field="graph_state_sha256"),
        )
        object.__setattr__(
            self,
            "route_sha256s",
            _hashes(self.route_sha256s, field="route_sha256s", sorted_unique=True),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": DEMAND_PINS_EVENT_SCHEMA,
            "logical_time": self.logical_time,
            "graph_generation": self.graph_generation,
            "graph_state_sha256": self.graph_state_sha256,
            "route_sha256s": list(self.route_sha256s),
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.to_dict())


DemandEvent = SelectionEvent | OutcomeEvent | PinsEvent


def _event_from_dict(value: object) -> DemandEvent:
    if not isinstance(value, Mapping):
        raise OperatorDemandIntegrityError("invalid demand event")
    schema = value.get("schema")
    try:
        if schema == DEMAND_SELECTION_EVENT_SCHEMA:
            if set(value) != {
                "schema",
                "logical_time",
                "route_sha256",
                "graph_generation",
                "graph_state_sha256",
                "input_abi_sha256",
                "output_abi_sha256",
                "exploration",
                "scores",
            }:
                raise ValueError("invalid selection event fields")
            raw_scores = value.get("scores")
            if not isinstance(raw_scores, list):
                raise ValueError("invalid selection scores")
            scores: list[UCBScore] = []
            for raw in raw_scores:
                if not isinstance(raw, Mapping) or set(raw) != {
                    "route_sha256",
                    "pulls",
                    "reward_sum",
                    "score",
                }:
                    raise ValueError("invalid UCB score")
                scores.append(
                    UCBScore(
                        route_sha256=cast(str, raw.get("route_sha256")),
                        pulls=cast(int, raw.get("pulls")),
                        reward_sum=cast(float, raw.get("reward_sum")),
                        score=cast(float | None, raw.get("score")),
                    )
                )
            event: DemandEvent = SelectionEvent(
                logical_time=cast(int, value.get("logical_time")),
                route_sha256=cast(str, value.get("route_sha256")),
                graph_generation=cast(int, value.get("graph_generation")),
                graph_state_sha256=cast(str, value.get("graph_state_sha256")),
                input_abi_sha256=cast(str, value.get("input_abi_sha256")),
                output_abi_sha256=cast(str | None, value.get("output_abi_sha256")),
                exploration=cast(float, value.get("exploration")),
                scores=tuple(scores),
            )
        elif schema == DEMAND_OUTCOME_EVENT_SCHEMA:
            if set(value) != {
                "schema",
                "logical_time",
                "outcome",
                "outcome_sha256",
            }:
                raise ValueError("invalid outcome event fields")
            outcome = DemandOutcomeReceipt.from_dict(value.get("outcome"))
            if require_sha256(
                value.get("outcome_sha256"), field="outcome_sha256"
            ) != outcome.sha256:
                raise ValueError("outcome event digest mismatch")
            event = OutcomeEvent(
                logical_time=cast(int, value.get("logical_time")), outcome=outcome
            )
        elif schema == DEMAND_PINS_EVENT_SCHEMA:
            if set(value) != {
                "schema",
                "logical_time",
                "graph_generation",
                "graph_state_sha256",
                "route_sha256s",
            }:
                raise ValueError("invalid pins event fields")
            raw_routes = value.get("route_sha256s")
            if not isinstance(raw_routes, list):
                raise ValueError("invalid pinned-route inventory")
            event = PinsEvent(
                logical_time=cast(int, value.get("logical_time")),
                graph_generation=cast(int, value.get("graph_generation")),
                graph_state_sha256=cast(str, value.get("graph_state_sha256")),
                route_sha256s=tuple(raw_routes),
            )
        else:
            raise ValueError("unknown demand event schema")
    except (TypeError, ValueError) as exc:
        raise OperatorDemandIntegrityError("demand event validation failed") from exc
    if event.to_dict() != dict(value):
        raise OperatorDemandIntegrityError(
            "demand event failed canonical reconstruction"
        )
    return event


@dataclass(frozen=True, slots=True)
class OperatorDemandState:
    generation: int
    previous_state_sha256: str | None
    config_sha256: str
    events: tuple[DemandEvent, ...]

    def __post_init__(self) -> None:
        generation = _uint(self.generation, field="generation")
        previous = self.previous_state_sha256
        if previous is not None:
            previous = require_sha256(previous, field="previous_state_sha256")
        config = require_sha256(self.config_sha256, field="config_sha256")
        events = tuple(self.events)
        if len(events) > MAX_DEMAND_EVENTS:
            raise ValueError("operator-demand event bound exceeded")
        if generation != len(events):
            raise ValueError("generation must equal append-only event count")
        if generation == 0 and previous is not None:
            raise ValueError("empty demand state cannot have a predecessor")
        if generation > 0 and previous is None:
            raise ValueError("non-empty demand state requires a predecessor")
        for logical_time, event in enumerate(events, start=1):
            if not isinstance(event, (SelectionEvent, OutcomeEvent, PinsEvent)):
                raise TypeError("events must be immutable demand-event values")
            if event.logical_time != logical_time:
                raise ValueError("demand-event logical times must be contiguous")
        _validate_event_semantics(events)
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "previous_state_sha256", previous)
        object.__setattr__(self, "config_sha256", config)
        object.__setattr__(self, "events", events)

    @classmethod
    def empty(cls, config_sha256: str) -> "OperatorDemandState":
        return cls(0, None, config_sha256, ())

    @property
    def logical_clock(self) -> int:
        return self.generation

    @property
    def pinned_route_sha256s(self) -> tuple[str, ...]:
        for event in reversed(self.events):
            if isinstance(event, PinsEvent):
                return event.route_sha256s
        return ()

    def to_document(self) -> dict[str, object]:
        body = {
            "generation": self.generation,
            "previous_state_sha256": self.previous_state_sha256,
            "config_sha256": self.config_sha256,
            "events": [event.to_dict() for event in self.events],
        }
        return _sealed_document(DEMAND_STATE_SCHEMA, body)

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_document())
        if len(data) > MAX_DEMAND_STATE_BYTES:
            raise ValueError("operator-demand state exceeds its byte bound")
        return data

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "OperatorDemandState":
        value = _strict_json(
            data, label="operator-demand state", maximum=MAX_DEMAND_STATE_BYTES
        )
        body = _decode_sealed(
            value, schema=DEMAND_STATE_SCHEMA, label="operator-demand state"
        )
        if set(body) != {
            "generation",
            "previous_state_sha256",
            "config_sha256",
            "events",
        }:
            raise OperatorDemandIntegrityError("invalid operator-demand state body")
        raw_events = body.get("events")
        if not isinstance(raw_events, list):
            raise OperatorDemandIntegrityError("demand events must be a list")
        try:
            state = cls(
                generation=cast(int, body.get("generation")),
                previous_state_sha256=cast(
                    str | None, body.get("previous_state_sha256")
                ),
                config_sha256=cast(str, body.get("config_sha256")),
                events=tuple(_event_from_dict(event) for event in raw_events),
            )
        except (TypeError, ValueError) as exc:
            raise OperatorDemandIntegrityError(
                "operator-demand state validation failed"
            ) from exc
        if state.to_bytes() != data:
            raise OperatorDemandIntegrityError(
                "operator-demand state failed canonical reconstruction"
            )
        return state


def _validate_event_semantics(events: Sequence[DemandEvent]) -> None:
    selections: dict[str, SelectionEvent] = {}
    settled: set[str] = set()
    outcomes: dict[str, DemandOutcomeReceipt] = {}
    evidence: dict[str, str] = {}
    pulls: dict[str, int] = defaultdict(int)
    rewards: dict[str, float] = defaultdict(float)
    for event in events:
        if isinstance(event, SelectionEvent):
            if event.sha256 in selections:
                raise ValueError("duplicate selection event")
            total_pulls = max(1, sum(pulls[item.route_sha256] for item in event.scores))
            for item in event.scores:
                expected_pulls = pulls[item.route_sha256]
                expected_reward = rewards[item.route_sha256]
                if (
                    item.pulls != expected_pulls
                    or item.reward_sum != expected_reward
                ):
                    raise ValueError("UCB score inventory disagrees with prior events")
                expected_score = (
                    None
                    if expected_pulls == 0
                    else expected_reward / expected_pulls
                    + event.exploration
                    * math.sqrt(2.0 * math.log(total_pulls) / expected_pulls)
                )
                if item.score != expected_score:
                    raise ValueError("UCB score is not reproducible from prior events")
            selections[event.sha256] = event
            pulls[event.route_sha256] += 1
            continue
        if not isinstance(event, OutcomeEvent):
            continue
        outcome = event.outcome
        existing = outcomes.get(outcome.sha256)
        if existing is not None:
            raise ValueError("duplicate outcome event")
        outcomes[outcome.sha256] = outcome
        prior = evidence.get(outcome.outcome_evidence_sha256)
        if prior is not None and prior != outcome.sha256:
            raise ValueError("one evidence digest authenticates conflicting outcomes")
        evidence[outcome.outcome_evidence_sha256] = outcome.sha256
        selection_sha = outcome.selection_event_sha256
        if selection_sha is None:
            pulls[outcome.route_sha256] += 1
            rewards[outcome.route_sha256] += outcome.reward
            continue
        selection = selections.get(selection_sha)
        if selection is None:
            raise ValueError("outcome references an unknown or future selection")
        if selection.route_sha256 != outcome.route_sha256:
            raise ValueError("outcome settles another route's selection")
        if (
            selection.graph_generation != outcome.graph_generation
            or selection.graph_state_sha256 != outcome.graph_state_sha256
            or selection.input_abi_sha256 != outcome.input_abi_sha256
            or (
                selection.output_abi_sha256 is not None
                and selection.output_abi_sha256 != outcome.output_abi_sha256
            )
        ):
            raise ValueError("outcome selection revision or ABI binding mismatch")
        if selection_sha in settled:
            raise ValueError("a selection cannot be settled twice")
        settled.add(selection_sha)
        rewards[outcome.route_sha256] += outcome.reward


@dataclass(frozen=True, slots=True)
class ArmStatistics:
    route_sha256: str
    pulls: int
    successes: int
    failures: int
    reward_sum: float
    last_logical_time: int

    @property
    def observations(self) -> int:
        return self.successes + self.failures

    @property
    def mean_reward(self) -> float:
        return 0.0 if self.pulls == 0 else self.reward_sum / self.pulls


def _arm_statistics(state: OperatorDemandState) -> dict[str, ArmStatistics]:
    selected: dict[str, int] = defaultdict(int)
    successes: dict[str, int] = defaultdict(int)
    failures: dict[str, int] = defaultdict(int)
    rewards: dict[str, float] = defaultdict(float)
    last: dict[str, int] = defaultdict(int)
    for event in state.events:
        if isinstance(event, SelectionEvent):
            selected[event.route_sha256] += 1
            last[event.route_sha256] = event.logical_time
        elif isinstance(event, OutcomeEvent):
            outcome = event.outcome
            if outcome.selection_event_sha256 is None:
                selected[outcome.route_sha256] += 1
            if outcome.success:
                successes[outcome.route_sha256] += 1
            else:
                failures[outcome.route_sha256] += 1
            rewards[outcome.route_sha256] += outcome.reward
            last[outcome.route_sha256] = event.logical_time
    routes = set(selected) | set(successes) | set(failures)
    return {
        route: ArmStatistics(
            route_sha256=route,
            pulls=selected[route],
            successes=successes[route],
            failures=failures[route],
            reward_sum=rewards[route],
            last_logical_time=last[route],
        )
        for route in routes
    }


def _positive_outcomes(
    state: OperatorDemandState,
) -> tuple[tuple[int, DemandOutcomeReceipt], ...]:
    return tuple(
        (event.logical_time, event.outcome)
        for event in state.events
        if isinstance(event, OutcomeEvent) and event.outcome.success
    )


def _positive_episode_sequences(
    state: OperatorDemandState,
) -> dict[str, tuple[DemandOutcomeReceipt, ...]]:
    """Reconstruct execution order, independent of asynchronous settlement.

    An explicit episode is required for sequence learning.  Outcomes tied to a
    selection are ordered by the selection's logical time; direct verified
    outcomes use their own append time.  This prevents reversed verifier
    completion from teaching a reversed program and prevents unrelated
    standalone outcomes from becoming a synthetic episode.
    """

    selection_time = {
        event.sha256: event.logical_time
        for event in state.events
        if isinstance(event, SelectionEvent)
    }
    grouped: dict[
        str, list[tuple[int, int, str, DemandOutcomeReceipt]]
    ] = defaultdict(list)
    for event in state.events:
        if not isinstance(event, OutcomeEvent) or not event.outcome.success:
            continue
        outcome = event.outcome
        if outcome.episode_sha256 is None:
            continue
        execution_time = (
            event.logical_time
            if outcome.selection_event_sha256 is None
            else selection_time[outcome.selection_event_sha256]
        )
        grouped[outcome.episode_sha256].append(
            (execution_time, event.logical_time, outcome.sha256, outcome)
        )
    return {
        episode: tuple(row[-1] for row in sorted(rows))
        for episode, rows in grouped.items()
    }


def _blocked_route_sha256s(
    state: OperatorDemandState,
    graph_state: ComputeOperatorGraphState,
    *,
    input_abi_sha256: str,
) -> frozenset[str]:
    """Return routes whose latest exact-revision outcome is a verified failure."""

    input_abi = require_sha256(input_abi_sha256, field="input_abi_sha256")
    latest: dict[str, DemandOutcomeReceipt] = {}
    for event in state.events:
        if not isinstance(event, OutcomeEvent):
            continue
        outcome = event.outcome
        if (
            outcome.graph_generation == graph_state.generation
            and outcome.graph_state_sha256 == graph_state.sha256
            and outcome.input_abi_sha256 == input_abi
        ):
            latest[outcome.route_sha256] = outcome
    return frozenset(
        route_sha for route_sha, outcome in latest.items() if not outcome.success
    )


@dataclass(frozen=True, slots=True)
class DemandStateTransitionReceipt:
    operation: str
    event_sha256: str
    previous_state_sha256: str
    current_state_sha256: str
    generation: int
    changed: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", _text(self.operation, field="operation"))
        for field in (
            "event_sha256",
            "previous_state_sha256",
            "current_state_sha256",
        ):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        object.__setattr__(
            self, "generation", _uint(self.generation, field="generation")
        )
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be a bool")
        if not self.changed and self.previous_state_sha256 != self.current_state_sha256:
            raise ValueError("an idempotent transition cannot change state")

    def to_dict(self) -> dict[str, object]:
        return _sealed_document(
            DEMAND_TRANSITION_SCHEMA,
            {
                "operation": self.operation,
                "event_sha256": self.event_sha256,
                "previous_state_sha256": self.previous_state_sha256,
                "current_state_sha256": self.current_state_sha256,
                "generation": self.generation,
                "changed": self.changed,
            },
        )

    @property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())


@dataclass(frozen=True, slots=True)
class UCBSelectionReceipt:
    event: SelectionEvent
    transition: DemandStateTransitionReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.event, SelectionEvent):
            raise TypeError("event must be a SelectionEvent")
        if not isinstance(self.transition, DemandStateTransitionReceipt):
            raise TypeError("transition must be a DemandStateTransitionReceipt")
        if self.transition.event_sha256 != self.event.sha256:
            raise ValueError("selection transition names another event")
        if not self.transition.changed:
            raise ValueError("every UCB selection is a persisted pull")

    @property
    def route_sha256(self) -> str:
        return self.event.route_sha256

    @property
    def selection_event_sha256(self) -> str:
        return self.event.sha256

    def to_dict(self) -> dict[str, object]:
        return _sealed_document(
            UCB_SELECTION_SCHEMA,
            {
                "event": self.event.to_dict(),
                "event_sha256": self.event.sha256,
                "transition": self.transition.to_dict(),
                "transition_sha256": self.transition.sha256,
            },
        )

    @property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())


@dataclass(frozen=True, slots=True)
class PPMPredictionReceipt:
    scheduler_state_sha256: str
    graph_generation: int
    graph_state_sha256: str
    input_abi_sha256: str
    output_abi_sha256: str | None
    history_route_sha256s: tuple[str, ...]
    matched_terminal_prefix: tuple[str, ...]
    counts: tuple[tuple[str, int], ...]
    evidence_receipt_sha256s: tuple[str, ...]
    selected_route_sha256: str | None
    probability: float
    reason: str

    def __post_init__(self) -> None:
        for field in (
            "scheduler_state_sha256",
            "graph_state_sha256",
            "input_abi_sha256",
        ):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        object.__setattr__(
            self,
            "graph_generation",
            _uint(self.graph_generation, field="graph_generation"),
        )
        if self.output_abi_sha256 is not None:
            object.__setattr__(
                self,
                "output_abi_sha256",
                require_sha256(self.output_abi_sha256, field="output_abi_sha256"),
            )
        object.__setattr__(
            self,
            "history_route_sha256s",
            _hashes(self.history_route_sha256s, field="history_route_sha256s"),
        )
        object.__setattr__(
            self,
            "matched_terminal_prefix",
            _hashes(
                self.matched_terminal_prefix, field="matched_terminal_prefix"
            ),
        )
        counts = tuple(self.counts)
        if counts != tuple(sorted(counts)):
            raise ValueError("PPM counts must be route-sorted")
        normalized: list[tuple[str, int]] = []
        for route_sha, count in counts:
            normalized.append(
                (
                    require_sha256(route_sha, field="route_sha256"),
                    _uint(count, field="count", positive=True),
                )
            )
        if len({route for route, _ in normalized}) != len(normalized):
            raise ValueError("PPM counts must be unique")
        object.__setattr__(self, "counts", tuple(normalized))
        object.__setattr__(
            self,
            "evidence_receipt_sha256s",
            _hashes(
                self.evidence_receipt_sha256s,
                field="evidence_receipt_sha256s",
                sorted_unique=True,
            ),
        )
        if self.selected_route_sha256 is not None:
            object.__setattr__(
                self,
                "selected_route_sha256",
                require_sha256(
                    self.selected_route_sha256, field="selected_route_sha256"
                ),
            )
        probability = _finite(
            self.probability, field="probability", lower=0.0, upper=1.0
        )
        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "reason", _text(self.reason, field="reason"))
        if (self.selected_route_sha256 is None) != (not self.counts):
            raise ValueError("PPM selection and counts must be present together")
        if self.matched_terminal_prefix and tuple(
            self.history_route_sha256s[-len(self.matched_terminal_prefix) :]
        ) != self.matched_terminal_prefix:
            raise ValueError("PPM context is not a terminal prefix of history")
        if self.counts:
            count_by_route = dict(self.counts)
            best = max(count_by_route.values())
            expected_route = min(
                route for route, count in self.counts if count == best
            )
            expected_probability = best / sum(count_by_route.values())
            if (
                self.selected_route_sha256 != expected_route
                or self.probability != expected_probability
            ):
                raise ValueError("PPM winner or probability is inconsistent")
        elif self.probability != 0.0:
            raise ValueError("an abstained PPM receipt must have zero probability")

    def to_dict(self) -> dict[str, object]:
        return _sealed_document(
            PPM_PREDICTION_SCHEMA,
            {
                "scheduler_state_sha256": self.scheduler_state_sha256,
                "graph_generation": self.graph_generation,
                "graph_state_sha256": self.graph_state_sha256,
                "input_abi_sha256": self.input_abi_sha256,
                "output_abi_sha256": self.output_abi_sha256,
                "history_route_sha256s": list(self.history_route_sha256s),
                "matched_terminal_prefix": list(self.matched_terminal_prefix),
                "counts": [
                    {"route_sha256": route, "count": count}
                    for route, count in self.counts
                ],
                "evidence_receipt_sha256s": list(
                    self.evidence_receipt_sha256s
                ),
                "selected_route_sha256": self.selected_route_sha256,
                "probability": self.probability,
                "reason": self.reason,
            },
        )

    @property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())


@dataclass(frozen=True, slots=True)
class CooccurrenceEntry:
    left_route_sha256: str
    right_route_sha256: str
    count: int
    evidence_receipt_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        left = require_sha256(self.left_route_sha256, field="left_route_sha256")
        right = require_sha256(self.right_route_sha256, field="right_route_sha256")
        if not left < right:
            raise ValueError("co-occurrence pairs must be canonical and non-reflexive")
        object.__setattr__(self, "left_route_sha256", left)
        object.__setattr__(self, "right_route_sha256", right)
        object.__setattr__(self, "count", _uint(self.count, field="count", positive=True))
        object.__setattr__(
            self,
            "evidence_receipt_sha256s",
            _hashes(
                self.evidence_receipt_sha256s,
                field="evidence_receipt_sha256s",
                sorted_unique=True,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "left_route_sha256": self.left_route_sha256,
            "right_route_sha256": self.right_route_sha256,
            "count": self.count,
            "evidence_receipt_sha256s": list(self.evidence_receipt_sha256s),
        }


@dataclass(frozen=True, slots=True)
class CooccurrenceReceipt:
    scheduler_state_sha256: str
    graph_generation: int
    graph_state_sha256: str
    window: int
    entries: tuple[CooccurrenceEntry, ...]

    def __post_init__(self) -> None:
        for field in ("scheduler_state_sha256", "graph_state_sha256"):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        object.__setattr__(
            self,
            "graph_generation",
            _uint(self.graph_generation, field="graph_generation"),
        )
        window = _uint(self.window, field="window", positive=True)
        if window > MAX_COOCCURRENCE_WINDOW:
            raise ValueError("co-occurrence window exceeds its bound")
        object.__setattr__(self, "window", window)
        entries = tuple(self.entries)
        if any(not isinstance(item, CooccurrenceEntry) for item in entries):
            raise TypeError("co-occurrence entries have the wrong type")
        if entries != tuple(
            sorted(
                entries,
                key=lambda item: (
                    item.left_route_sha256,
                    item.right_route_sha256,
                ),
            )
        ):
            raise ValueError("co-occurrence entries must be pair-sorted")
        pairs = {
            (item.left_route_sha256, item.right_route_sha256) for item in entries
        }
        if len(pairs) != len(entries):
            raise ValueError("co-occurrence entries must be unique")
        object.__setattr__(self, "entries", entries)

    def to_dict(self) -> dict[str, object]:
        return _sealed_document(
            COOCCURRENCE_SCHEMA,
            {
                "scheduler_state_sha256": self.scheduler_state_sha256,
                "graph_generation": self.graph_generation,
                "graph_state_sha256": self.graph_state_sha256,
                "window": self.window,
                "entries": [entry.to_dict() for entry in self.entries],
            },
        )

    @property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())


@dataclass(frozen=True, slots=True)
class PrefetchItem:
    route_sha256: str
    cooccurrence_count: int
    payload_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "route_sha256",
            require_sha256(self.route_sha256, field="route_sha256"),
        )
        object.__setattr__(
            self,
            "cooccurrence_count",
            _uint(self.cooccurrence_count, field="cooccurrence_count", positive=True),
        )
        object.__setattr__(
            self,
            "payload_bytes",
            _uint(self.payload_bytes, field="payload_bytes", positive=True),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "route_sha256": self.route_sha256,
            "cooccurrence_count": self.cooccurrence_count,
            "payload_bytes": self.payload_bytes,
        }


@dataclass(frozen=True, slots=True)
class PrefetchReceipt:
    scheduler_state_sha256: str
    graph_generation: int
    graph_state_sha256: str
    input_abi_sha256: str
    output_abi_sha256: str | None
    seed_route_sha256s: tuple[str, ...]
    max_items: int
    max_bytes: int
    size_inventory_sha256: str
    cooccurrence_receipt_sha256: str
    selected: tuple[PrefetchItem, ...]
    total_bytes: int

    def __post_init__(self) -> None:
        for field in (
            "scheduler_state_sha256",
            "graph_state_sha256",
            "input_abi_sha256",
            "size_inventory_sha256",
            "cooccurrence_receipt_sha256",
        ):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        object.__setattr__(
            self,
            "graph_generation",
            _uint(self.graph_generation, field="graph_generation"),
        )
        if self.output_abi_sha256 is not None:
            object.__setattr__(
                self,
                "output_abi_sha256",
                require_sha256(self.output_abi_sha256, field="output_abi_sha256"),
            )
        object.__setattr__(
            self,
            "seed_route_sha256s",
            _hashes(
                self.seed_route_sha256s,
                field="seed_route_sha256s",
                sorted_unique=True,
            ),
        )
        max_items = _uint(self.max_items, field="max_items", positive=True)
        max_bytes = _uint(self.max_bytes, field="max_bytes", positive=True)
        selected = tuple(self.selected)
        if len(selected) > max_items:
            raise ValueError("prefetch selection exceeds item budget")
        if any(not isinstance(item, PrefetchItem) for item in selected):
            raise TypeError("prefetch items have the wrong type")
        if len({item.route_sha256 for item in selected}) != len(selected):
            raise ValueError("prefetch selection must be unique")
        if any(item.route_sha256 in self.seed_route_sha256s for item in selected):
            raise ValueError("prefetch cannot select a seed route")
        if selected != tuple(
            sorted(
                selected,
                key=lambda item: (-item.cooccurrence_count, item.route_sha256),
            )
        ):
            raise ValueError("prefetch items must preserve deterministic rank order")
        total = sum(item.payload_bytes for item in selected)
        if total != _uint(self.total_bytes, field="total_bytes") or total > max_bytes:
            raise ValueError("prefetch selection exceeds or misstates byte budget")
        object.__setattr__(self, "max_items", max_items)
        object.__setattr__(self, "max_bytes", max_bytes)
        object.__setattr__(self, "selected", selected)
        object.__setattr__(self, "total_bytes", total)

    @property
    def selected_route_sha256s(self) -> tuple[str, ...]:
        return tuple(item.route_sha256 for item in self.selected)

    def to_dict(self) -> dict[str, object]:
        return _sealed_document(
            PREFETCH_SCHEMA,
            {
                "scheduler_state_sha256": self.scheduler_state_sha256,
                "graph_generation": self.graph_generation,
                "graph_state_sha256": self.graph_state_sha256,
                "input_abi_sha256": self.input_abi_sha256,
                "output_abi_sha256": self.output_abi_sha256,
                "seed_route_sha256s": list(self.seed_route_sha256s),
                "max_items": self.max_items,
                "max_bytes": self.max_bytes,
                "size_inventory_sha256": self.size_inventory_sha256,
                "cooccurrence_receipt_sha256": self.cooccurrence_receipt_sha256,
                "selected": [item.to_dict() for item in self.selected],
                "total_bytes": self.total_bytes,
            },
        )

    @property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())


@dataclass(frozen=True, slots=True)
class RetentionItem:
    route_sha256: str
    reward: float
    logical_age: int
    keep_score: float
    payload_bytes: int
    pinned: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "route_sha256",
            require_sha256(self.route_sha256, field="route_sha256"),
        )
        object.__setattr__(
            self,
            "reward",
            _finite(
                self.reward,
                field="reward",
                lower=-float(MAX_BUDGET),
                upper=float(MAX_BUDGET),
            ),
        )
        object.__setattr__(
            self, "logical_age", _uint(self.logical_age, field="logical_age")
        )
        object.__setattr__(
            self,
            "keep_score",
            _finite(
                self.keep_score,
                field="keep_score",
                lower=0.0,
                upper=float(MAX_BUDGET),
            ),
        )
        object.__setattr__(
            self,
            "payload_bytes",
            _uint(self.payload_bytes, field="payload_bytes", positive=True),
        )
        if not isinstance(self.pinned, bool):
            raise TypeError("pinned must be a bool")

    def to_dict(self) -> dict[str, object]:
        return {
            "route_sha256": self.route_sha256,
            "reward": self.reward,
            "logical_age": self.logical_age,
            "keep_score": self.keep_score,
            "payload_bytes": self.payload_bytes,
            "pinned": self.pinned,
        }


@dataclass(frozen=True, slots=True)
class RetentionReceipt:
    scheduler_state_sha256: str
    graph_generation: int
    graph_state_sha256: str
    alpha: float
    max_items: int
    max_bytes: int
    size_inventory_sha256: str
    ranked: tuple[RetentionItem, ...]
    kept_route_sha256s: tuple[str, ...]
    evicted_route_sha256s: tuple[str, ...]
    total_bytes: int

    def __post_init__(self) -> None:
        for field in (
            "scheduler_state_sha256",
            "graph_state_sha256",
            "size_inventory_sha256",
        ):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        object.__setattr__(
            self,
            "graph_generation",
            _uint(self.graph_generation, field="graph_generation"),
        )
        object.__setattr__(
            self,
            "alpha",
            _finite(self.alpha, field="alpha", lower=0.0, upper=1_000_000.0),
        )
        max_items = _uint(self.max_items, field="max_items", positive=True)
        max_bytes = _uint(self.max_bytes, field="max_bytes", positive=True)
        ranked = tuple(self.ranked)
        if any(not isinstance(item, RetentionItem) for item in ranked):
            raise TypeError("retention items have the wrong type")
        if len({item.route_sha256 for item in ranked}) != len(ranked):
            raise ValueError("retention inventory must be unique")
        if ranked != tuple(
            sorted(
                ranked,
                key=lambda item: (
                    not item.pinned,
                    -item.keep_score,
                    item.route_sha256,
                ),
            )
        ):
            raise ValueError("retention inventory is not deterministically ranked")
        kept = _hashes(
            self.kept_route_sha256s,
            field="kept_route_sha256s",
            sorted_unique=True,
        )
        evicted = _hashes(
            self.evicted_route_sha256s,
            field="evicted_route_sha256s",
            sorted_unique=True,
        )
        if set(kept) & set(evicted) or set(kept) | set(evicted) != {
            item.route_sha256 for item in ranked
        }:
            raise ValueError("retention partition is invalid")
        by_sha = {item.route_sha256: item for item in ranked}
        total = sum(by_sha[route].payload_bytes for route in kept)
        if len(kept) > max_items or total > max_bytes:
            raise ValueError("retention result exceeds its budget")
        if any(item.pinned and item.route_sha256 not in kept for item in ranked):
            raise ValueError("retention evicted a pinned route")
        expected_kept: list[str] = []
        expected_bytes = 0
        for item in ranked:
            if len(expected_kept) >= max_items:
                break
            if expected_bytes + item.payload_bytes > max_bytes:
                continue
            expected_kept.append(item.route_sha256)
            expected_bytes += item.payload_bytes
        if kept != tuple(sorted(expected_kept)):
            raise ValueError("retention partition does not follow deterministic ranking")
        if total != _uint(self.total_bytes, field="total_bytes"):
            raise ValueError("retention byte total mismatch")
        object.__setattr__(self, "max_items", max_items)
        object.__setattr__(self, "max_bytes", max_bytes)
        object.__setattr__(self, "ranked", ranked)
        object.__setattr__(self, "kept_route_sha256s", kept)
        object.__setattr__(self, "evicted_route_sha256s", evicted)
        object.__setattr__(self, "total_bytes", total)

    def to_dict(self) -> dict[str, object]:
        return _sealed_document(
            RETENTION_SCHEMA,
            {
                "scheduler_state_sha256": self.scheduler_state_sha256,
                "graph_generation": self.graph_generation,
                "graph_state_sha256": self.graph_state_sha256,
                "alpha": self.alpha,
                "max_items": self.max_items,
                "max_bytes": self.max_bytes,
                "size_inventory_sha256": self.size_inventory_sha256,
                "ranked": [item.to_dict() for item in self.ranked],
                "kept_route_sha256s": list(self.kept_route_sha256s),
                "evicted_route_sha256s": list(self.evicted_route_sha256s),
                "total_bytes": self.total_bytes,
            },
        )

    @property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())


def _transition_from_document(value: object) -> DemandStateTransitionReceipt:
    body = _decode_sealed(
        value,
        schema=DEMAND_TRANSITION_SCHEMA,
        label="demand transition",
    )
    expected = {
        "operation",
        "event_sha256",
        "previous_state_sha256",
        "current_state_sha256",
        "generation",
        "changed",
    }
    if set(body) != expected:
        raise OperatorDemandIntegrityError("invalid demand transition body")
    try:
        result = DemandStateTransitionReceipt(**dict(body))
    except (TypeError, ValueError) as exc:
        raise OperatorDemandIntegrityError(
            "demand transition validation failed"
        ) from exc
    if result.to_dict() != dict(cast(Mapping[str, object], value)):
        raise OperatorDemandIntegrityError(
            "demand transition failed canonical reconstruction"
        )
    return result


def _verify_typed_demand_receipt(
    data: bytes,
    *,
    schema: str,
    value: Mapping[str, object],
) -> None:
    """Dispatch a self-sealed document through its semantic constructor."""

    try:
        if schema == DEMAND_OUTCOME_SCHEMA:
            reconstructed: object = DemandOutcomeReceipt.from_bytes(data)
        elif schema == DEMAND_STATE_SCHEMA:
            reconstructed = OperatorDemandState.from_bytes(data)
        elif schema == DEMAND_COMMIT_SCHEMA:
            _decode_commit(data)
            return
        elif schema == DEMAND_TRANSITION_SCHEMA:
            reconstructed = _transition_from_document(value)
        elif schema == UCB_SELECTION_SCHEMA:
            body = _decode_sealed(
                value, schema=UCB_SELECTION_SCHEMA, label="UCB selection"
            )
            if set(body) != {
                "event",
                "event_sha256",
                "transition",
                "transition_sha256",
            }:
                raise OperatorDemandIntegrityError("invalid UCB selection body")
            event = _event_from_dict(body.get("event"))
            if not isinstance(event, SelectionEvent):
                raise OperatorDemandIntegrityError(
                    "UCB receipt does not contain a selection event"
                )
            transition = _transition_from_document(body.get("transition"))
            if (
                require_sha256(body.get("event_sha256"), field="event_sha256")
                != event.sha256
                or require_sha256(
                    body.get("transition_sha256"), field="transition_sha256"
                )
                != transition.sha256
            ):
                raise OperatorDemandIntegrityError(
                    "UCB nested receipt address mismatch"
                )
            reconstructed = UCBSelectionReceipt(event, transition)
        elif schema == PPM_PREDICTION_SCHEMA:
            body = _decode_sealed(
                value, schema=PPM_PREDICTION_SCHEMA, label="PPM prediction"
            )
            expected = {
                "scheduler_state_sha256",
                "graph_generation",
                "graph_state_sha256",
                "input_abi_sha256",
                "output_abi_sha256",
                "history_route_sha256s",
                "matched_terminal_prefix",
                "counts",
                "evidence_receipt_sha256s",
                "selected_route_sha256",
                "probability",
                "reason",
            }
            raw_counts = body.get("counts")
            if set(body) != expected or not isinstance(raw_counts, list):
                raise OperatorDemandIntegrityError("invalid PPM prediction body")
            counts: list[tuple[str, int]] = []
            for row in raw_counts:
                if not isinstance(row, Mapping) or set(row) != {
                    "route_sha256",
                    "count",
                }:
                    raise OperatorDemandIntegrityError("invalid PPM count row")
                counts.append(
                    (cast(str, row.get("route_sha256")), cast(int, row.get("count")))
                )
            reconstructed = PPMPredictionReceipt(
                scheduler_state_sha256=cast(
                    str, body.get("scheduler_state_sha256")
                ),
                graph_generation=cast(int, body.get("graph_generation")),
                graph_state_sha256=cast(str, body.get("graph_state_sha256")),
                input_abi_sha256=cast(str, body.get("input_abi_sha256")),
                output_abi_sha256=cast(
                    str | None, body.get("output_abi_sha256")
                ),
                history_route_sha256s=tuple(
                    cast(list[str], body.get("history_route_sha256s"))
                ),
                matched_terminal_prefix=tuple(
                    cast(list[str], body.get("matched_terminal_prefix"))
                ),
                counts=tuple(counts),
                evidence_receipt_sha256s=tuple(
                    cast(list[str], body.get("evidence_receipt_sha256s"))
                ),
                selected_route_sha256=cast(
                    str | None, body.get("selected_route_sha256")
                ),
                probability=cast(float, body.get("probability")),
                reason=cast(str, body.get("reason")),
            )
        elif schema == COOCCURRENCE_SCHEMA:
            body = _decode_sealed(
                value, schema=COOCCURRENCE_SCHEMA, label="co-occurrence"
            )
            expected = {
                "scheduler_state_sha256",
                "graph_generation",
                "graph_state_sha256",
                "window",
                "entries",
            }
            raw_entries = body.get("entries")
            if set(body) != expected or not isinstance(raw_entries, list):
                raise OperatorDemandIntegrityError("invalid co-occurrence body")
            entries = []
            for row in raw_entries:
                if not isinstance(row, Mapping) or set(row) != {
                    "left_route_sha256",
                    "right_route_sha256",
                    "count",
                    "evidence_receipt_sha256s",
                }:
                    raise OperatorDemandIntegrityError(
                        "invalid co-occurrence entry"
                    )
                entries.append(
                    CooccurrenceEntry(
                        left_route_sha256=cast(
                            str, row.get("left_route_sha256")
                        ),
                        right_route_sha256=cast(
                            str, row.get("right_route_sha256")
                        ),
                        count=cast(int, row.get("count")),
                        evidence_receipt_sha256s=tuple(
                            cast(list[str], row.get("evidence_receipt_sha256s"))
                        ),
                    )
                )
            reconstructed = CooccurrenceReceipt(
                scheduler_state_sha256=cast(
                    str, body.get("scheduler_state_sha256")
                ),
                graph_generation=cast(int, body.get("graph_generation")),
                graph_state_sha256=cast(str, body.get("graph_state_sha256")),
                window=cast(int, body.get("window")),
                entries=tuple(entries),
            )
        elif schema == PREFETCH_SCHEMA:
            body = _decode_sealed(value, schema=PREFETCH_SCHEMA, label="prefetch")
            expected = {
                "scheduler_state_sha256",
                "graph_generation",
                "graph_state_sha256",
                "input_abi_sha256",
                "output_abi_sha256",
                "seed_route_sha256s",
                "max_items",
                "max_bytes",
                "size_inventory_sha256",
                "cooccurrence_receipt_sha256",
                "selected",
                "total_bytes",
            }
            raw_selected = body.get("selected")
            if set(body) != expected or not isinstance(raw_selected, list):
                raise OperatorDemandIntegrityError("invalid prefetch body")
            selected = []
            for row in raw_selected:
                if not isinstance(row, Mapping) or set(row) != {
                    "route_sha256",
                    "cooccurrence_count",
                    "payload_bytes",
                }:
                    raise OperatorDemandIntegrityError("invalid prefetch item")
                selected.append(PrefetchItem(**dict(row)))
            reconstructed = PrefetchReceipt(
                scheduler_state_sha256=cast(
                    str, body.get("scheduler_state_sha256")
                ),
                graph_generation=cast(int, body.get("graph_generation")),
                graph_state_sha256=cast(str, body.get("graph_state_sha256")),
                input_abi_sha256=cast(str, body.get("input_abi_sha256")),
                output_abi_sha256=cast(
                    str | None, body.get("output_abi_sha256")
                ),
                seed_route_sha256s=tuple(
                    cast(list[str], body.get("seed_route_sha256s"))
                ),
                max_items=cast(int, body.get("max_items")),
                max_bytes=cast(int, body.get("max_bytes")),
                size_inventory_sha256=cast(
                    str, body.get("size_inventory_sha256")
                ),
                cooccurrence_receipt_sha256=cast(
                    str, body.get("cooccurrence_receipt_sha256")
                ),
                selected=tuple(selected),
                total_bytes=cast(int, body.get("total_bytes")),
            )
        elif schema == RETENTION_SCHEMA:
            body = _decode_sealed(value, schema=RETENTION_SCHEMA, label="retention")
            expected = {
                "scheduler_state_sha256",
                "graph_generation",
                "graph_state_sha256",
                "alpha",
                "max_items",
                "max_bytes",
                "size_inventory_sha256",
                "ranked",
                "kept_route_sha256s",
                "evicted_route_sha256s",
                "total_bytes",
            }
            raw_ranked = body.get("ranked")
            if set(body) != expected or not isinstance(raw_ranked, list):
                raise OperatorDemandIntegrityError("invalid retention body")
            ranked = []
            for row in raw_ranked:
                if not isinstance(row, Mapping) or set(row) != {
                    "route_sha256",
                    "reward",
                    "logical_age",
                    "keep_score",
                    "payload_bytes",
                    "pinned",
                }:
                    raise OperatorDemandIntegrityError("invalid retention item")
                ranked.append(RetentionItem(**dict(row)))
            reconstructed = RetentionReceipt(
                scheduler_state_sha256=cast(
                    str, body.get("scheduler_state_sha256")
                ),
                graph_generation=cast(int, body.get("graph_generation")),
                graph_state_sha256=cast(str, body.get("graph_state_sha256")),
                alpha=cast(float, body.get("alpha")),
                max_items=cast(int, body.get("max_items")),
                max_bytes=cast(int, body.get("max_bytes")),
                size_inventory_sha256=cast(
                    str, body.get("size_inventory_sha256")
                ),
                ranked=tuple(ranked),
                kept_route_sha256s=tuple(
                    cast(list[str], body.get("kept_route_sha256s"))
                ),
                evicted_route_sha256s=tuple(
                    cast(list[str], body.get("evicted_route_sha256s"))
                ),
                total_bytes=cast(int, body.get("total_bytes")),
            )
        else:
            raise OperatorDemandIntegrityError(
                "unsupported operator-demand receipt schema"
            )
    except OperatorDemandIntegrityError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise OperatorDemandIntegrityError(
            "operator-demand receipt semantic validation failed"
        ) from exc
    if not hasattr(reconstructed, "to_bytes") or cast(
        object, reconstructed
    ).to_bytes() != data:
        raise OperatorDemandIntegrityError(
            "operator-demand receipt failed typed reconstruction"
        )


def _commit_bytes(state_sha256: str) -> bytes:
    address = require_sha256(state_sha256, field="state_sha256")
    return canonical_json_bytes(
        _sealed_document(DEMAND_COMMIT_SCHEMA, {"state_sha256": address})
    )


def _decode_commit(data: bytes) -> str:
    value = _strict_json(data, label="demand commit", maximum=4096)
    body = _decode_sealed(value, schema=DEMAND_COMMIT_SCHEMA, label="demand commit")
    if set(body) != {"state_sha256"}:
        raise OperatorDemandIntegrityError("invalid demand commit body")
    try:
        address = require_sha256(body.get("state_sha256"), field="state_sha256")
    except ValueError as exc:
        raise OperatorDemandIntegrityError("invalid demand commit address") from exc
    if _commit_bytes(address) != data:
        raise OperatorDemandIntegrityError(
            "demand commit failed canonical reconstruction"
        )
    return address


def _validate_extension(
    previous: OperatorDemandState,
    current: OperatorDemandState,
) -> None:
    if current.config_sha256 != previous.config_sha256:
        raise OperatorDemandIntegrityError("scheduler configuration changed in history")
    if current.generation != previous.generation + 1:
        raise OperatorDemandIntegrityError("demand history generation is not contiguous")
    if current.previous_state_sha256 != previous.sha256:
        raise OperatorDemandIntegrityError("demand history predecessor mismatch")
    if current.events[:-1] != previous.events:
        raise OperatorDemandIntegrityError("demand history is not append-only")


def _validate_committed_history(
    histories: Mapping[str, OperatorDemandState],
    commits: set[str],
    *,
    config_sha256: str,
) -> OperatorDemandState:
    if len(histories) > MAX_DEMAND_EVENTS + 1 or len(commits) > MAX_DEMAND_EVENTS:
        raise OperatorDemandIntegrityError("demand history exceeds its bound")
    by_generation: dict[int, tuple[str, OperatorDemandState]] = {}
    for digest in commits:
        state = histories.get(digest)
        if state is None or state.sha256 != digest:
            raise OperatorDemandIntegrityError(
                "committed demand state lacks exact immutable history"
            )
        if state.generation < 1:
            raise OperatorDemandIntegrityError("empty demand state cannot be committed")
        existing = by_generation.get(state.generation)
        if existing is not None and existing[0] != digest:
            raise OperatorDemandIntegrityError("committed demand history contains a fork")
        by_generation[state.generation] = (digest, state)
    previous = OperatorDemandState.empty(config_sha256)
    if not by_generation:
        return previous
    maximum = max(by_generation)
    if set(by_generation) != set(range(1, maximum + 1)):
        raise OperatorDemandIntegrityError("committed demand history has a gap")
    for generation in range(1, maximum + 1):
        current = by_generation[generation][1]
        _validate_extension(previous, current)
        previous = current
    return previous


class OperatorDemandScheduler:
    """Persistent verified demand learner for materialized operator routes."""

    def __init__(
        self,
        store: CrystalStore | str | os.PathLike[str],
        *,
        config: OperatorDemandConfig | None = None,
        retry_limit: int = 16,
        trusted_state_sha256: str | None = None,
        trusted_head_resolver: Callable[[], str] | None = None,
    ) -> None:
        self.store = store if isinstance(store, CrystalStore) else CrystalStore(store)
        self.compute_bank = ComputeCrystalBank(self.store)
        self.config = config or OperatorDemandConfig()
        if not isinstance(self.config, OperatorDemandConfig):
            raise TypeError("config must be an OperatorDemandConfig")
        if (
            isinstance(retry_limit, bool)
            or not isinstance(retry_limit, int)
            or not 1 <= retry_limit <= 1024
        ):
            raise ValueError("retry_limit must lie in [1, 1024]")
        self.retry_limit = retry_limit
        self.root = Path(self.store.root)
        self.trusted_state_sha256 = (
            None
            if trusted_state_sha256 is None
            else require_sha256(trusted_state_sha256, field="trusted_state_sha256")
        )
        if trusted_head_resolver is not None and not callable(trusted_head_resolver):
            raise TypeError("trusted_head_resolver must be callable")
        self.trusted_head_resolver = trusted_head_resolver
        self._last_authorized_head_sha256: str | None = None

    @staticmethod
    def history_state_name(state_sha256: str) -> str:
        return DEMAND_HISTORY_PREFIX + require_sha256(
            state_sha256, field="state_sha256"
        )

    @staticmethod
    def commit_state_name(state_sha256: str) -> str:
        return DEMAND_COMMIT_PREFIX + require_sha256(
            state_sha256, field="state_sha256"
        )

    @contextmanager
    def _locked(self) -> Iterator[None]:
        path = self.root / _LOCK_NAME
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as exc:
            raise OperatorDemandIntegrityError("cannot open scheduler lock") from exc
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise OperatorDemandIntegrityError("scheduler lock is not regular")
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            linked = path.lstat()
            if (before.st_dev, before.st_ino) != (linked.st_dev, linked.st_ino):
                raise OperatorDemandIntegrityError("scheduler lock changed while opening")
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _restore_raw(self, name: str) -> bytes:
        try:
            return self.store.restore_state(name)
        except KeyError:
            raise
        except CrystalStoreError as exc:
            raise OperatorDemandIntegrityError(
                "scheduler storage failed integrity"
            ) from exc

    def _history_names_unlocked(self) -> tuple[str, ...]:
        names: list[str] = []
        root_fd = os.open(self.root, self.store._directory_flags())
        try:
            state_fd = os.open("state", self.store._directory_flags(), dir_fd=root_fd)
            try:
                maximum = 4096 + 4 * ((self.store.max_state_bytes + 2) // 3)
                for filename in sorted(os.listdir(state_fd)):
                    if not filename.endswith(".state"):
                        continue
                    try:
                        envelope = self.store._stable_read(
                            state_fd, filename, max_bytes=maximum
                        )
                    except (FileNotFoundError, CrystalStoreError):
                        continue
                    match = _STATE_NAME_RE.match(envelope[:4096])
                    if match is None:
                        continue
                    try:
                        name = match.group(1).decode("ascii")
                    except UnicodeDecodeError:
                        continue
                    if not name.startswith((DEMAND_HISTORY_PREFIX, DEMAND_COMMIT_PREFIX)):
                        continue
                    if filename != self.store._state_filename(name):
                        raise OperatorDemandIntegrityError(
                            "demand history filename is not name-bound"
                        )
                    names.append(name)
            finally:
                os.close(state_fd)
        finally:
            os.close(root_fd)
        if len(names) > 2 * MAX_DEMAND_EVENTS + 1:
            raise OperatorDemandIntegrityError("demand history scan exceeds bound")
        return tuple(names)

    def _history_inventory_unlocked(
        self,
    ) -> tuple[dict[str, OperatorDemandState], set[str]]:
        histories: dict[str, OperatorDemandState] = {}
        commits: set[str] = set()
        for name in self._history_names_unlocked():
            payload = self._restore_raw(name)
            if name.startswith(DEMAND_HISTORY_PREFIX):
                try:
                    address = require_sha256(
                        name[len(DEMAND_HISTORY_PREFIX) :], field="history address"
                    )
                except ValueError as exc:
                    raise OperatorDemandIntegrityError(
                        "demand history state name has an invalid address"
                    ) from exc
                state = OperatorDemandState.from_bytes(payload)
                if state.sha256 != address:
                    raise OperatorDemandIntegrityError(
                        "demand history is stored under another address"
                    )
                histories[address] = state
            else:
                try:
                    address = require_sha256(
                        name[len(DEMAND_COMMIT_PREFIX) :], field="commit address"
                    )
                except ValueError as exc:
                    raise OperatorDemandIntegrityError(
                        "demand commit state name has an invalid address"
                    ) from exc
                if _decode_commit(payload) != address:
                    raise OperatorDemandIntegrityError(
                        "demand commit is stored under another address"
                    )
                commits.add(address)
        return histories, commits

    def _pointer_unlocked(self) -> tuple[OperatorDemandState, str | None]:
        try:
            data = self._restore_raw(DEMAND_STATE_NAME)
        except KeyError:
            return OperatorDemandState.empty(self.config.sha256), None
        state = OperatorDemandState.from_bytes(data)
        if state.config_sha256 != self.config.sha256:
            raise OperatorDemandIntegrityError(
                "persisted scheduler configuration does not match this instance"
            )
        if any(
            isinstance(event, SelectionEvent)
            and event.exploration != self.config.exploration
            for event in state.events
        ):
            raise OperatorDemandIntegrityError(
                "persisted UCB event disagrees with scheduler configuration"
            )
        digest = hashlib.sha256(data).hexdigest()
        if state.sha256 != digest:
            raise OperatorDemandIntegrityError("scheduler pointer address mismatch")
        return state, digest

    def _assert_trusted_head(self, state: OperatorDemandState) -> None:
        anchors: list[str] = []
        if self.trusted_state_sha256 is not None:
            anchors.append(self.trusted_state_sha256)
        if self.trusted_head_resolver is not None:
            try:
                anchors.append(
                    require_sha256(
                        self.trusted_head_resolver(), field="trusted resolved head"
                    )
                )
            except Exception as exc:
                raise OperatorDemandIntegrityError(
                    "trusted scheduler-head resolver failed"
                ) from exc
        if any(anchor != state.sha256 for anchor in dict.fromkeys(anchors)):
            raise OperatorDemandIntegrityError(
                "scheduler head does not equal its trusted anchor"
            )

    def _validated_state_unlocked(
        self, *, assert_trusted_head: bool = True
    ) -> OperatorDemandState:
        current, pointer_digest = self._pointer_unlocked()
        histories, commits = self._history_inventory_unlocked()
        latest = _validate_committed_history(
            histories, commits, config_sha256=self.config.sha256
        )
        if pointer_digest is None:
            if latest.generation != 0:
                raise OperatorDemandIntegrityError(
                    "scheduler pointer was deleted or rolled back"
                )
            if assert_trusted_head:
                self._assert_trusted_head(current)
            return current
        historical = histories.get(current.sha256)
        if historical is None or historical.to_bytes() != current.to_bytes():
            raise OperatorDemandIntegrityError(
                "scheduler pointer lacks immutable history"
            )
        if current.sha256 not in commits:
            _validate_extension(latest, current)
            self._publish_commit_unlocked(current)
            commits.add(current.sha256)
            latest = _validate_committed_history(
                histories, commits, config_sha256=self.config.sha256
            )
        if latest.sha256 != current.sha256:
            raise OperatorDemandIntegrityError(
                "scheduler pointer is a validly resealed rollback"
            )
        node = current
        for expected_generation in range(current.generation, -1, -1):
            if node.generation != expected_generation:
                raise OperatorDemandIntegrityError("scheduler history depth mismatch")
            try:
                payload = self._restore_raw(self.history_state_name(node.sha256))
            except KeyError as exc:
                raise OperatorDemandIntegrityError(
                    "scheduler predecessor history is missing"
                ) from exc
            if payload != node.to_bytes():
                raise OperatorDemandIntegrityError(
                    "scheduler history address mismatch"
                )
            if node.generation == 0:
                break
            previous_sha = node.previous_state_sha256
            if previous_sha is None:
                raise OperatorDemandIntegrityError("scheduler predecessor was lost")
            previous = OperatorDemandState.from_bytes(
                self._restore_raw(self.history_state_name(previous_sha))
            )
            _validate_extension(previous, node)
            node = previous
        if assert_trusted_head:
            self._assert_trusted_head(current)
        return current

    def state(self) -> OperatorDemandState:
        with self._locked():
            return self._validated_state_unlocked()

    def current_anchor_sha256(self) -> str:
        if self._last_authorized_head_sha256 is not None:
            return self._last_authorized_head_sha256
        return self.state().sha256

    def _publish_history_unlocked(self, state: OperatorDemandState) -> None:
        name = self.history_state_name(state.sha256)
        data = state.to_bytes()
        try:
            existing = self._restore_raw(name)
        except KeyError:
            existing = None
        if existing is not None:
            if existing != data:
                raise OperatorDemandIntegrityError(
                    "content-addressed demand history contains other bytes"
                )
            return
        try:
            self.store.publish_state(name, data)
        except (CrystalStoreError, ManifestConflictError) as exc:
            raise OperatorDemandIntegrityError(
                "failed to publish demand history"
            ) from exc
        if self._restore_raw(name) != data:
            raise OperatorDemandIntegrityError(
                "demand history failed immediate verification"
            )

    def _publish_commit_unlocked(self, state: OperatorDemandState) -> None:
        if state.generation == 0:
            raise ValueError("empty scheduler state is never committed")
        name = self.commit_state_name(state.sha256)
        data = _commit_bytes(state.sha256)
        try:
            existing = self._restore_raw(name)
        except KeyError:
            existing = None
        if existing is not None:
            if existing != data:
                raise OperatorDemandIntegrityError(
                    "content-addressed demand commit contains other bytes"
                )
            return
        try:
            self.store.publish_state(name, data)
        except (CrystalStoreError, ManifestConflictError) as exc:
            raise OperatorDemandIntegrityError("failed to publish demand commit") from exc
        if self._restore_raw(name) != data:
            raise OperatorDemandIntegrityError(
                "demand commit failed immediate verification"
            )

    def _append_event(
        self,
        event_factory: Callable[[OperatorDemandState], DemandEvent | None],
        *,
        operation: str,
        duplicate_event_sha256: str
        | Callable[[OperatorDemandState], str]
        | None = None,
        expected_state_sha256: str | None = None,
    ) -> tuple[OperatorDemandState, DemandStateTransitionReceipt]:
        if expected_state_sha256 is not None:
            expected_state_sha256 = require_sha256(
                expected_state_sha256, field="expected_state_sha256"
            )
        for attempt in range(self.retry_limit):
            with self._locked():
                current = self._validated_state_unlocked()
                if (
                    expected_state_sha256 is not None
                    and current.sha256 != expected_state_sha256
                ):
                    raise OperatorDemandConflictError(
                        "scheduler compare-and-swap precondition is stale"
                    )
                event = event_factory(current)
                if event is None:
                    if duplicate_event_sha256 is None:
                        raise AssertionError("idempotent mutation lacks event address")
                    duplicate_sha = (
                        duplicate_event_sha256(current)
                        if callable(duplicate_event_sha256)
                        else duplicate_event_sha256
                    )
                    transition = DemandStateTransitionReceipt(
                        operation=operation,
                        event_sha256=duplicate_sha,
                        previous_state_sha256=current.sha256,
                        current_state_sha256=current.sha256,
                        generation=current.generation,
                        changed=False,
                    )
                    return current, transition
                if event.logical_time != current.logical_clock + 1:
                    raise OperatorDemandIntegrityError(
                        "new event has the wrong logical time"
                    )
                updated = OperatorDemandState(
                    generation=current.generation + 1,
                    previous_state_sha256=current.sha256,
                    config_sha256=current.config_sha256,
                    events=(*current.events, event),
                )
                _validate_extension(current, updated)
                self._publish_history_unlocked(current)
                self._publish_history_unlocked(updated)
                expected = None if current.generation == 0 else current.sha256
                try:
                    publication = self.store.publish_state(
                        DEMAND_STATE_NAME,
                        updated.to_bytes(),
                        expected_sha256=expected,
                    )
                except ManifestConflictError as exc:
                    if expected_state_sha256 is not None or attempt + 1 == self.retry_limit:
                        raise OperatorDemandConflictError(
                            "scheduler state publication conflicted"
                        ) from exc
                    continue
                except CrystalStoreError as exc:
                    raise OperatorDemandIntegrityError(
                        "scheduler state publication failed integrity"
                    ) from exc
                if not publication.changed:
                    raise OperatorDemandIntegrityError(
                        "new scheduler generation was not published"
                    )
                self._publish_commit_unlocked(updated)
                restored = self._validated_state_unlocked(assert_trusted_head=False)
                if restored != updated:
                    raise OperatorDemandIntegrityError(
                        "scheduler failed immediate state verification"
                    )
                if self.trusted_state_sha256 is not None:
                    self.trusted_state_sha256 = restored.sha256
                self._last_authorized_head_sha256 = restored.sha256
                transition = DemandStateTransitionReceipt(
                    operation=operation,
                    event_sha256=event.sha256,
                    previous_state_sha256=current.sha256,
                    current_state_sha256=updated.sha256,
                    generation=updated.generation,
                    changed=True,
                )
                return updated, transition
        raise OperatorDemandConflictError("scheduler exhausted its CAS retry limit")

    @staticmethod
    def resolve_materialized_candidate(
        graph_state: ComputeOperatorGraphState,
        candidate: str | MaterializedRoute | ComputeRoutePlan,
    ) -> MaterializedRoute:
        """Resolve a graph address, exact route, or its authenticated cold plan."""

        if isinstance(candidate, str):
            return _require_route(graph_state, candidate)
        if isinstance(candidate, MaterializedRoute):
            _require_exact_route(graph_state, candidate)
            return candidate
        if not isinstance(candidate, ComputeRoutePlan):
            raise TypeError(
                "candidate must be a route address, MaterializedRoute, or ComputeRoutePlan"
            )
        matches = tuple(
            route
            for route in graph_state.materialized_routes
            if route.planning_graph_generation == candidate.graph_generation
            and route.planning_graph_state_sha256 == candidate.graph_state_sha256
            and route.finite_horizon_plan == candidate.finite_plan
            and route.primitive_edge_sha256s == candidate.primitive_edge_sha256s
        )
        if len(matches) != 1:
            raise OperatorDemandIntegrityError(
                "compute route plan has no unique materialized route in this graph revision"
            )
        return matches[0]

    @staticmethod
    def _eligible_routes(
        graph_state: ComputeOperatorGraphState,
        *,
        input_abi_sha256: str,
        output_abi_sha256: str | None,
    ) -> tuple[MaterializedRoute, ...]:
        input_abi = require_sha256(input_abi_sha256, field="input_abi_sha256")
        output_abi = (
            None
            if output_abi_sha256 is None
            else require_sha256(output_abi_sha256, field="output_abi_sha256")
        )
        routes = tuple(
            sorted(
                (
                    route
                    for route in graph_state.materialized_routes
                    if route.input_abi_sha256 == input_abi
                    and (output_abi is None or route.output_abi_sha256 == output_abi)
                ),
                key=lambda route: route.sha256,
            )
        )
        if len(routes) > MAX_DEMAND_ROUTES:
            raise OperatorDemandIntegrityError("eligible route inventory exceeds bound")
        return routes

    def select_ucb1(
        self,
        graph_state: ComputeOperatorGraphState,
        *,
        input_abi_sha256: str,
        output_abi_sha256: str | None = None,
        candidate_route_sha256s: Sequence[str] | None = None,
        candidate_routes_or_plans: Sequence[
            str | MaterializedRoute | ComputeRoutePlan
        ]
        | None = None,
        expected_state_sha256: str | None = None,
    ) -> UCBSelectionReceipt:
        """Persist one unseen-first UCB1 pull with deterministic SHA tie-breaking."""

        routes = self._eligible_routes(
            graph_state,
            input_abi_sha256=input_abi_sha256,
            output_abi_sha256=output_abi_sha256,
        )
        if (
            candidate_route_sha256s is not None
            and candidate_routes_or_plans is not None
        ):
            raise ValueError("provide only one candidate inventory")
        if candidate_routes_or_plans is not None:
            resolved = tuple(
                self.resolve_materialized_candidate(graph_state, candidate)
                for candidate in candidate_routes_or_plans
            )
            candidates = tuple(sorted({route.sha256 for route in resolved}))
            if len(candidates) != len(resolved):
                raise ValueError("candidate inventory must be unique")
            available = {route.sha256: route for route in routes}
            try:
                routes = tuple(available[address] for address in candidates)
            except KeyError as exc:
                raise OperatorDemandIntegrityError(
                    "candidate route violates graph or ABI binding"
                ) from exc
        elif candidate_route_sha256s is not None:
            candidates = _hashes(
                candidate_route_sha256s,
                field="candidate_route_sha256s",
                sorted_unique=True,
            )
            available = {route.sha256: route for route in routes}
            try:
                routes = tuple(available[address] for address in candidates)
            except KeyError as exc:
                raise OperatorDemandIntegrityError(
                    "candidate route violates graph or ABI binding"
                ) from exc
        if not routes:
            raise OperatorDemandUnavailableError(
                "no materialized route matches the requested ABI"
            )
        result_event: SelectionEvent | None = None

        def create(current: OperatorDemandState) -> SelectionEvent:
            nonlocal result_event
            stats = _arm_statistics(current)
            blocked = _blocked_route_sha256s(
                current,
                graph_state,
                input_abi_sha256=input_abi_sha256,
            )
            eligible_routes = tuple(
                route for route in routes if route.sha256 not in blocked
            )
            if not eligible_routes:
                raise OperatorDemandUnavailableError(
                    "every exact-revision route has verified negative evidence"
                )
            pulls = {
                route.sha256: stats.get(
                    route.sha256,
                    ArmStatistics(route.sha256, 0, 0, 0, 0.0, 0),
                )
                for route in eligible_routes
            }
            total_pulls = max(1, sum(item.pulls for item in pulls.values()))
            scores: list[UCBScore] = []
            unseen: list[str] = []
            for route in eligible_routes:
                item = pulls[route.sha256]
                if item.pulls == 0:
                    score = None
                    unseen.append(route.sha256)
                else:
                    score = item.mean_reward + self.config.exploration * math.sqrt(
                        2.0 * math.log(total_pulls) / item.pulls
                    )
                scores.append(
                    UCBScore(
                        route_sha256=route.sha256,
                        pulls=item.pulls,
                        reward_sum=item.reward_sum,
                        score=score,
                    )
                )
            if unseen:
                selected = min(unseen)
            else:
                selected = min(
                    scores,
                    key=lambda item: (
                        -cast(float, item.score),
                        item.route_sha256,
                    ),
                ).route_sha256
            result_event = SelectionEvent(
                logical_time=current.logical_clock + 1,
                route_sha256=selected,
                graph_generation=graph_state.generation,
                graph_state_sha256=graph_state.sha256,
                input_abi_sha256=input_abi_sha256,
                output_abi_sha256=output_abi_sha256,
                exploration=self.config.exploration,
                scores=tuple(scores),
            )
            return result_event

        _state, transition = self._append_event(
            create,
            operation="ucb1-select",
            expected_state_sha256=expected_state_sha256,
        )
        assert result_event is not None
        return UCBSelectionReceipt(result_event, transition)

    def record_outcome(
        self,
        outcome: DemandOutcomeReceipt,
        graph_state: ComputeOperatorGraphState,
        *,
        expected_state_sha256: str | None = None,
    ) -> DemandStateTransitionReceipt:
        """Append verified result evidence, or return an idempotent duplicate."""

        if not isinstance(outcome, DemandOutcomeReceipt):
            raise TypeError("outcome must be a DemandOutcomeReceipt")
        if (
            outcome.graph_generation != graph_state.generation
            or outcome.graph_state_sha256 != graph_state.sha256
        ):
            raise OperatorDemandIntegrityError(
                "outcome names another graph revision"
            )
        route = _require_route(graph_state, outcome.route_sha256)
        if (
            outcome.input_abi_sha256 != route.input_abi_sha256
            or outcome.output_abi_sha256 != route.output_abi_sha256
            or outcome.route_verifier_sha256s != route.verifier_sha256s
            or outcome.route_evidence_sha256s != route.evidence_sha256s
        ):
            raise OperatorDemandIntegrityError(
                "outcome route ABI or provenance binding mismatch"
            )

        def create(current: OperatorDemandState) -> OutcomeEvent | None:
            prior_outcomes = [
                event.outcome
                for event in current.events
                if isinstance(event, OutcomeEvent)
            ]
            if any(item.sha256 == outcome.sha256 for item in prior_outcomes):
                return None
            if any(
                item.outcome_evidence_sha256 == outcome.outcome_evidence_sha256
                for item in prior_outcomes
            ):
                raise OperatorDemandConflictError(
                    "outcome evidence already authenticates another receipt"
                )
            if outcome.selection_event_sha256 is not None:
                selection_by_sha = {
                    event.sha256: event
                    for event in current.events
                    if isinstance(event, SelectionEvent)
                }
                selection = selection_by_sha.get(outcome.selection_event_sha256)
                if selection is None or selection.route_sha256 != outcome.route_sha256:
                    raise OperatorDemandIntegrityError(
                        "outcome selection binding is unknown or mismatched"
                    )
                if (
                    selection.graph_generation != outcome.graph_generation
                    or selection.graph_state_sha256 != outcome.graph_state_sha256
                    or selection.input_abi_sha256 != outcome.input_abi_sha256
                    or (
                        selection.output_abi_sha256 is not None
                        and selection.output_abi_sha256 != outcome.output_abi_sha256
                    )
                ):
                    raise OperatorDemandIntegrityError(
                        "outcome selection revision or ABI binding mismatch"
                    )
                if any(
                    item.selection_event_sha256 == outcome.selection_event_sha256
                    for item in prior_outcomes
                ):
                    raise OperatorDemandConflictError(
                        "selection already has an outcome"
                    )
            return OutcomeEvent(current.logical_clock + 1, outcome)

        _state, transition = self._append_event(
            create,
            operation="record-positive" if outcome.success else "record-negative",
            duplicate_event_sha256=lambda state: next(
                event.sha256
                for event in state.events
                if isinstance(event, OutcomeEvent)
                and event.outcome.sha256 == outcome.sha256
            ),
            expected_state_sha256=expected_state_sha256,
        )
        return transition

    def set_pins(
        self,
        graph_state: ComputeOperatorGraphState,
        route_sha256s: Sequence[str],
        *,
        expected_state_sha256: str | None = None,
    ) -> DemandStateTransitionReceipt:
        pins = _hashes(route_sha256s, field="route_sha256s", sorted_unique=True)
        for route_sha in pins:
            _require_route(graph_state, route_sha)
        def create(current: OperatorDemandState) -> PinsEvent | None:
            candidate = PinsEvent(
                logical_time=current.logical_clock + 1,
                graph_generation=graph_state.generation,
                graph_state_sha256=graph_state.sha256,
                route_sha256s=pins,
            )
            if current.pinned_route_sha256s == pins:
                return None
            return candidate

        state, transition = self._append_event(
            create,
            operation="set-pins",
            duplicate_event_sha256=lambda state: next(
                (
                    event.sha256
                    for event in reversed(state.events)
                    if isinstance(event, PinsEvent) and event.route_sha256s == pins
                ),
                _sha256(
                    {
                        "schema": "immer-ooe-operator-demand-pins-noop/v1",
                        "route_sha256s": list(pins),
                        "scheduler_state_sha256": state.sha256,
                    }
                ),
            ),
            expected_state_sha256=expected_state_sha256,
        )
        del state
        return transition

    def predict_ppm(
        self,
        graph_state: ComputeOperatorGraphState,
        history_route_sha256s: Sequence[str],
        *,
        input_abi_sha256: str,
        output_abi_sha256: str | None = None,
    ) -> PPMPredictionReceipt:
        current = self.state()
        history = _hashes(
            history_route_sha256s, field="history_route_sha256s"
        )
        route_by_sha = _route_map(graph_state)
        if any(route_sha not in route_by_sha for route_sha in history):
            raise OperatorDemandIntegrityError(
                "PPM history contains a non-materialized route"
            )
        eligible = {
            route.sha256
            for route in self._eligible_routes(
                graph_state,
                input_abi_sha256=input_abi_sha256,
                output_abi_sha256=output_abi_sha256,
            )
        }
        groups = _positive_episode_sequences(current)
        blocked = _blocked_route_sha256s(
            current,
            graph_state,
            input_abi_sha256=input_abi_sha256,
        )
        eligible -= blocked
        selected_counts: dict[str, int] = {}
        selected_evidence: set[str] = set()
        matched: tuple[str, ...] = ()
        maximum = min(len(history), self.config.max_context_order)
        for order in range(maximum, -1, -1):
            context = history[-order:] if order else ()
            counts: dict[str, int] = defaultdict(int)
            evidence: set[str] = set()
            for sequence in groups.values():
                route_sequence = [item.route_sha256 for item in sequence]
                for index, candidate in enumerate(route_sequence):
                    if index < order or candidate not in eligible:
                        continue
                    if tuple(route_sequence[index - order : index]) != context:
                        continue
                    counts[candidate] += 1
                    evidence.add(sequence[index].sha256)
                    evidence.update(
                        item.sha256 for item in sequence[index - order : index]
                    )
            if counts:
                matched = context
                selected_counts = dict(counts)
                selected_evidence = evidence
                break
        if selected_counts:
            best_count = max(selected_counts.values())
            selected = min(
                route
                for route, count in selected_counts.items()
                if count == best_count
            )
            probability = best_count / sum(selected_counts.values())
            reason = "longest-terminal-prefix"
        else:
            selected = None
            probability = 0.0
            reason = "no-verified-prefix-evidence"
        return PPMPredictionReceipt(
            scheduler_state_sha256=current.sha256,
            graph_generation=graph_state.generation,
            graph_state_sha256=graph_state.sha256,
            input_abi_sha256=input_abi_sha256,
            output_abi_sha256=output_abi_sha256,
            history_route_sha256s=history,
            matched_terminal_prefix=matched,
            counts=tuple(sorted(selected_counts.items())),
            evidence_receipt_sha256s=tuple(sorted(selected_evidence)),
            selected_route_sha256=selected,
            probability=probability,
            reason=reason,
        )

    def cooccurrence(
        self,
        graph_state: ComputeOperatorGraphState,
    ) -> CooccurrenceReceipt:
        current = self.state()
        materialized = set(_route_map(graph_state))
        groups = {
            episode: tuple(
                outcome
                for outcome in sequence
                if outcome.route_sha256 in materialized
            )
            for episode, sequence in _positive_episode_sequences(current).items()
        }
        counts: dict[tuple[str, str], int] = defaultdict(int)
        evidence: dict[tuple[str, str], set[str]] = defaultdict(set)
        window = self.config.cooccurrence_window
        for sequence in groups.values():
            for left_index, left in enumerate(sequence):
                limit = min(len(sequence), left_index + window + 1)
                for right in sequence[left_index + 1 : limit]:
                    if left.route_sha256 == right.route_sha256:
                        continue
                    pair = tuple(sorted((left.route_sha256, right.route_sha256)))
                    typed_pair = cast(tuple[str, str], pair)
                    counts[typed_pair] += 1
                    evidence[typed_pair].update((left.sha256, right.sha256))
        entries = tuple(
            CooccurrenceEntry(
                left_route_sha256=pair[0],
                right_route_sha256=pair[1],
                count=count,
                evidence_receipt_sha256s=tuple(sorted(evidence[pair])),
            )
            for pair, count in sorted(counts.items())
        )
        return CooccurrenceReceipt(
            scheduler_state_sha256=current.sha256,
            graph_generation=graph_state.generation,
            graph_state_sha256=graph_state.sha256,
            window=window,
            entries=entries,
        )

    def _size_inventory(
        self,
        graph_state: ComputeOperatorGraphState,
        payload_bytes_by_route: Mapping[str, int] | None,
    ) -> tuple[dict[str, int], str]:
        routes = _route_map(graph_state)
        if payload_bytes_by_route is None:
            sizes = {}
            for route_sha, route in routes.items():
                try:
                    programs = {
                        route.primitive_program_sha256: self.compute_bank.restore_program(
                            route.primitive_program_sha256
                        ),
                        route.executable_program_sha256: self.compute_bank.restore_program(
                            route.executable_program_sha256
                        ),
                    }
                    crystal_addresses = {
                        address
                        for program in programs.values()
                        for address in program.crystal_sha256s
                    }
                    size = len(route.to_bytes())
                    size += sum(len(program.to_bytes()) for program in programs.values())
                    size += sum(
                        len(self.compute_bank.restore_crystal(address).to_bytes())
                        for address in crystal_addresses
                    )
                    if route.charge_basis_sha256 is not None:
                        size += len(
                            self.compute_bank.restore_charge(
                                route.charge_basis_sha256
                            ).to_bytes()
                        )
                except ComputeCrystalError as exc:
                    raise OperatorDemandIntegrityError(
                        "route payload inventory cannot restore its compute artifacts"
                    ) from exc
                sizes[route_sha] = _uint(
                    size, field="payload_bytes", positive=True
                )
        else:
            if set(payload_bytes_by_route) != set(routes):
                raise ValueError(
                    "payload-size inventory must exactly cover the graph routes"
                )
            sizes = {
                require_sha256(route_sha, field="route_sha256"): _uint(
                    size, field="payload_bytes", positive=True
                )
                for route_sha, size in payload_bytes_by_route.items()
            }
        document = {
            "schema": "immer-ooe-route-size-inventory/v1",
            "graph_state_sha256": graph_state.sha256,
            "sizes": [
                {"route_sha256": route, "payload_bytes": size}
                for route, size in sorted(sizes.items())
            ],
        }
        return sizes, _sha256(document)

    def prefetch(
        self,
        graph_state: ComputeOperatorGraphState,
        seed_route_sha256s: Sequence[str],
        *,
        input_abi_sha256: str,
        output_abi_sha256: str | None = None,
        max_items: int,
        max_bytes: int,
        payload_bytes_by_route: Mapping[str, int] | None = None,
    ) -> PrefetchReceipt:
        seeds = _hashes(
            seed_route_sha256s,
            field="seed_route_sha256s",
            sorted_unique=True,
        )
        if not seeds:
            raise ValueError("prefetch requires at least one seed route")
        for route_sha in seeds:
            _require_route(graph_state, route_sha)
        item_budget = _uint(max_items, field="max_items", positive=True)
        byte_budget = _uint(max_bytes, field="max_bytes", positive=True)
        sizes, size_sha = self._size_inventory(
            graph_state, payload_bytes_by_route
        )
        matrix = self.cooccurrence(graph_state)
        current = self.state()
        if current.sha256 != matrix.scheduler_state_sha256:
            matrix = self.cooccurrence(graph_state)
            current = self.state()
            if current.sha256 != matrix.scheduler_state_sha256:
                raise OperatorDemandConflictError(
                    "scheduler changed while deriving prefetch evidence"
                )
        eligible = {
            route.sha256
            for route in self._eligible_routes(
                graph_state,
                input_abi_sha256=input_abi_sha256,
                output_abi_sha256=output_abi_sha256,
            )
        } - set(seeds)
        eligible -= _blocked_route_sha256s(
            current,
            graph_state,
            input_abi_sha256=input_abi_sha256,
        )
        scores: dict[str, int] = defaultdict(int)
        seed_set = set(seeds)
        for entry in matrix.entries:
            if entry.left_route_sha256 in seed_set:
                scores[entry.right_route_sha256] += entry.count
            if entry.right_route_sha256 in seed_set:
                scores[entry.left_route_sha256] += entry.count
        ordered = sorted(
            (
                (route, count)
                for route, count in scores.items()
                if route in eligible and count > 0
            ),
            key=lambda item: (-item[1], item[0]),
        )
        selected: list[PrefetchItem] = []
        total = 0
        for route_sha, count in ordered:
            size = sizes[route_sha]
            if len(selected) >= item_budget:
                break
            if total + size > byte_budget:
                continue
            selected.append(PrefetchItem(route_sha, count, size))
            total += size
        return PrefetchReceipt(
            scheduler_state_sha256=matrix.scheduler_state_sha256,
            graph_generation=graph_state.generation,
            graph_state_sha256=graph_state.sha256,
            input_abi_sha256=input_abi_sha256,
            output_abi_sha256=output_abi_sha256,
            seed_route_sha256s=seeds,
            max_items=item_budget,
            max_bytes=byte_budget,
            size_inventory_sha256=size_sha,
            cooccurrence_receipt_sha256=matrix.sha256,
            selected=tuple(selected),
            total_bytes=total,
        )

    def retention(
        self,
        graph_state: ComputeOperatorGraphState,
        resident_route_sha256s: Sequence[str],
        *,
        alpha: float,
        max_items: int,
        max_bytes: int,
        payload_bytes_by_route: Mapping[str, int] | None = None,
    ) -> RetentionReceipt:
        current = self.state()
        resident = _hashes(
            resident_route_sha256s,
            field="resident_route_sha256s",
            sorted_unique=True,
        )
        if not resident:
            raise ValueError("retention requires a resident inventory")
        for route_sha in resident:
            _require_route(graph_state, route_sha)
        pins = current.pinned_route_sha256s
        if not set(pins).issubset(resident):
            raise OperatorDemandIntegrityError(
                "pinned routes must be present in the resident inventory"
            )
        decay = _finite(alpha, field="alpha", lower=0.0, upper=1_000_000.0)
        item_budget = _uint(max_items, field="max_items", positive=True)
        byte_budget = _uint(max_bytes, field="max_bytes", positive=True)
        sizes, size_sha = self._size_inventory(
            graph_state, payload_bytes_by_route
        )
        stats = _arm_statistics(current)
        pin_set = set(pins)
        ranked = []
        for route_sha in resident:
            item = stats.get(
                route_sha,
                ArmStatistics(route_sha, 0, 0, 0, 0.0, 0),
            )
            age = current.logical_clock - item.last_logical_time
            keep = abs(item.reward_sum) * math.exp(-decay * age)
            ranked.append(
                RetentionItem(
                    route_sha256=route_sha,
                    reward=item.reward_sum,
                    logical_age=age,
                    keep_score=keep,
                    payload_bytes=sizes[route_sha],
                    pinned=route_sha in pin_set,
                )
            )
        ranked.sort(
            key=lambda item: (
                not item.pinned,
                -item.keep_score,
                item.route_sha256,
            )
        )
        pinned_items = [item for item in ranked if item.pinned]
        if len(pinned_items) > item_budget or sum(
            item.payload_bytes for item in pinned_items
        ) > byte_budget:
            raise OperatorDemandBudgetError(
                "pinned routes exceed the exact retention budget"
            )
        kept: list[str] = []
        total = 0
        for item in ranked:
            if len(kept) >= item_budget:
                break
            if total + item.payload_bytes > byte_budget:
                if item.pinned:
                    raise AssertionError("prechecked pinned item no longer fits")
                continue
            kept.append(item.route_sha256)
            total += item.payload_bytes
        kept_set = set(kept)
        evicted = tuple(sorted(set(resident) - kept_set))
        return RetentionReceipt(
            scheduler_state_sha256=current.sha256,
            graph_generation=graph_state.generation,
            graph_state_sha256=graph_state.sha256,
            alpha=decay,
            max_items=item_budget,
            max_bytes=byte_budget,
            size_inventory_sha256=size_sha,
            ranked=tuple(ranked),
            kept_route_sha256s=tuple(sorted(kept)),
            evicted_route_sha256s=evicted,
            total_bytes=total,
        )


__all__ = [
    "ArmStatistics",
    "CooccurrenceEntry",
    "CooccurrenceReceipt",
    "DEMAND_COMMIT_PREFIX",
    "DEMAND_HISTORY_PREFIX",
    "DEMAND_STATE_NAME",
    "DemandOutcomeReceipt",
    "DemandStateTransitionReceipt",
    "OperatorDemandBudgetError",
    "OperatorDemandConfig",
    "OperatorDemandConflictError",
    "OperatorDemandError",
    "OperatorDemandIntegrityError",
    "OperatorDemandScheduler",
    "OperatorDemandState",
    "OperatorDemandUnavailableError",
    "PPMPredictionReceipt",
    "PrefetchItem",
    "PrefetchReceipt",
    "RetentionItem",
    "RetentionReceipt",
    "UCBScore",
    "UCBSelectionReceipt",
    "verify_demand_receipt",
]
