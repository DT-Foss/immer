"""Append-only semantic routing over generic stored compute.

The operator graph gives the Markov planner a semantic view of published
``ComputeCrystal`` objects.  Edges contain only one-step evidence.  Longer
programs are discovered by planning, materialised once, and discharged on
future numerical values through ``ComputeCrystalVM``.

The graph deliberately has no model, prompt, or evaluator dependency.
"""

from __future__ import annotations

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
from typing import Any, Iterator, cast

import numpy as np
from numpy.typing import NDArray

from .compute_crystals import (
    AFFINE_FLOAT64,
    CAUSAL_MIX_FLOAT64,
    MARKOV_FLOAT64,
    MAX_PROGRAM_STEPS,
    PERMUTATION,
    ComputeBankPublication,
    ComputeChargeReceipt,
    ComputeCrystal,
    ComputeCrystalABIError,
    ComputeCrystalBank,
    ComputeCrystalError,
    ComputeCrystalIntegrityError,
    ComputeCrystalVM,
    ComputeExecution,
    ComputeExecutionReceipt,
    ComputeProgram,
    fuse_compatible_chain,
)
from .contraction_ledger import ContractionLedger
from .crystal import CrystalStoreError, ManifestConflictError
from .identity import canonical_json_bytes, require_sha256
from .math_core import array_sha256
from .planning import FiniteHorizonPlan, FiniteHorizonPlanner, PlanDecision, PolicyEntry
from .world_model import ActionConditionedWorldModel, TransitionEvidence

OPERATOR_EDGE_SCHEMA = "immer-ooe-compute-operator-edge/v1"
OPERATOR_GRAPH_STATE_SCHEMA = "immer-ooe-compute-operator-graph-state/v1"
OPERATOR_GRAPH_COMMIT_SCHEMA = "immer-ooe-compute-operator-graph-commit/v1"
MATERIALIZED_ROUTE_SCHEMA = "immer-ooe-materialized-compute-route/v1"
ROUTE_PLAN_SCHEMA = "immer-ooe-compute-route-plan/v1"
ROUTE_CHARGE_SCHEMA = "immer-ooe-compute-route-charge/v1"
ROUTE_DISCHARGE_SCHEMA = "immer-ooe-compute-route-discharge/v1"
FUSION_VERIFICATION_SCHEMA = "immer-ooe-exact-fusion-verification/v1"
OPERATOR_GRAPH_STATE_NAME = "ooe-compute-operator-graph/v1"
OPERATOR_GRAPH_HISTORY_PREFIX = "ooe-compute-operator-graph-history/v1:"
OPERATOR_GRAPH_COMMIT_PREFIX = "ooe-compute-operator-graph-commit/v1:"

MAX_GRAPH_STATE_BYTES = 48 * 1024 * 1024
MAX_GRAPH_EDGES = 65_536
MAX_MATERIALIZED_ROUTES = 65_536
MAX_GRAPH_GENERATIONS = MAX_GRAPH_EDGES + MAX_MATERIALIZED_ROUTES
MAX_ROUTE_RECEIPT_BYTES = 4 * 1024 * 1024
MAX_SEMANTIC_LABEL_BYTES = 1024
MAX_WORK_UNITS = (1 << 63) - 1

_GRAPH_LOCK_NAME = ".compute-operator-graph.lock"
_STATE_NAME_RE = re.compile(
    rb'^\{"format":"immer-ooe-controller-state/v1","generation":[1-9][0-9]*,"name":"([^"\\]*)","payload_base64":"'
)


class ComputeOperatorGraphError(RuntimeError):
    """Base error for semantic compute routing."""


class ComputeOperatorGraphIntegrityError(ComputeOperatorGraphError):
    """Graph state, history, or a materialized route failed authentication."""


class ComputeOperatorGraphConflictError(ComputeOperatorGraphError):
    """A graph compare-and-swap precondition is stale."""


class ComputeRouteUnavailableError(ComputeOperatorGraphError):
    """No verified materialized route can answer a semantic query."""


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


EXACT_FUSION_VERIFIER_SHA256 = _sha256(
    {
        "schema": "immer-ooe-exact-fusion-verifier/v1",
        "operator": "fuse_compatible_chain",
        "criterion": "canonical-byte-equality",
    }
)
CAUSAL_MIX_FUSION_VERIFIER_SHA256 = _sha256(
    {
        "schema": "immer-ooe-causal-mix-fusion-verifier/v1",
        "operator": "fuse_causal_mix_chain",
        "criterion": "canonical-causal-left-action-fusion",
    }
)


def _strict_json(data: bytes, *, label: str, maximum: int) -> object:
    if not isinstance(data, bytes):
        raise TypeError(f"{label} must be immutable bytes")
    if len(data) > maximum:
        raise ComputeOperatorGraphIntegrityError(f"{label} exceeds its byte bound")

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
        raise ComputeOperatorGraphIntegrityError(f"{label} is not strict JSON") from exc
    if canonical_json_bytes(value) != data:
        raise ComputeOperatorGraphIntegrityError(f"{label} is not canonical JSON")
    return value


def _text(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > MAX_SEMANTIC_LABEL_BYTES
    ):
        raise ValueError(f"{field} must be canonical non-empty text")
    return value


def _uint(value: object, *, field: str, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field} must be an integer")
    result = int(value)
    if not minimum <= result <= MAX_WORK_UNITS:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must be a bounded {qualifier} integer")
    return result


def _weight(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("weight must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= 1_000_000.0:
        raise ValueError("weight must lie in (0, 1000000]")
    return result


def _hash_tuple(values: Sequence[str], *, field: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field} must be a sequence")
    return tuple(require_sha256(value, field=field) for value in values)


def _sorted_hash_tuple(values: Sequence[str], *, field: str) -> tuple[str, ...]:
    result = _hash_tuple(values, field=field)
    if tuple(sorted(set(result))) != result:
        raise ValueError(f"{field} must be sorted and unique")
    return result


def _checked_sum(values: Sequence[int], *, field: str) -> int:
    total = 0
    for value in values:
        total += _uint(value, field=field)
        if total > MAX_WORK_UNITS:
            raise ValueError(f"{field} exceeds its bound")
    return total


def _fusion_verification_sha256(
    source_program: ComputeProgram,
    fused_crystal: ComputeCrystal,
    *,
    verifier_sha256: str = EXACT_FUSION_VERIFIER_SHA256,
) -> str:
    """Hash the exact fusion check that authorizes a compute charge."""

    return _sha256(
        {
            "schema": FUSION_VERIFICATION_SCHEMA,
            "verifier_sha256": require_sha256(
                verifier_sha256, field="fusion verifier SHA-256"
            ),
            "source_program_sha256": source_program.sha256,
            "source_crystal_sha256s": list(source_program.crystal_sha256s),
            "fused_crystal_sha256": fused_crystal.sha256,
            "input_abi_sha256": source_program.input_abi.sha256,
            "output_abi_sha256": source_program.output_abi.sha256,
            "canonical_fusion_verified": True,
        }
    )


def _fusion_verifier_for_kinds(kinds: set[str]) -> str:
    if kinds == {CAUSAL_MIX_FLOAT64}:
        return CAUSAL_MIX_FUSION_VERIFIER_SHA256
    if kinds in ({AFFINE_FLOAT64}, {PERMUTATION}, {MARKOV_FLOAT64}):
        return EXACT_FUSION_VERIFIER_SHA256
    raise ComputeOperatorGraphIntegrityError(
        "operator kinds have no registered fusion verifier"
    )


def _contraction_kernels(
    crystals: Sequence[ComputeCrystal],
    kinds: set[str],
) -> tuple[NDArray[np.float64], ...]:
    if kinds != {MARKOV_FLOAT64}:
        raise ComputeOperatorGraphIntegrityError(
            "operator kinds have no contraction-kernel representation"
        )
    kernels = []
    for crystal in crystals:
        dimension = crystal.input_abi.trailing_shape[0]
        recovered = crystal.apply(np.eye(dimension, dtype=np.float64))
        kernels.append(np.asarray(recovered, dtype=np.float64))
    return tuple(kernels)


def _graph_commit_bytes(graph_state_sha256: str) -> bytes:
    address = require_sha256(graph_state_sha256, field="graph_state_sha256")
    body = {"graph_state_sha256": address}
    return canonical_json_bytes(
        {
            "schema": OPERATOR_GRAPH_COMMIT_SCHEMA,
            "body": body,
            "body_sha256": _sha256(body),
        }
    )


def _decode_graph_commit(data: bytes) -> str:
    value = _strict_json(data, label="operator graph commit", maximum=4096)
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema", "body", "body_sha256"}
        or value.get("schema") != OPERATOR_GRAPH_COMMIT_SCHEMA
    ):
        raise ComputeOperatorGraphIntegrityError("invalid operator graph commit marker")
    body = value.get("body")
    if not isinstance(body, Mapping) or set(body) != {"graph_state_sha256"}:
        raise ComputeOperatorGraphIntegrityError(
            "invalid operator graph commit-marker body"
        )
    try:
        claimed = require_sha256(value.get("body_sha256"), field="body_sha256")
        state_sha256 = require_sha256(
            body.get("graph_state_sha256"), field="graph_state_sha256"
        )
    except ValueError as exc:
        raise ComputeOperatorGraphIntegrityError(
            "invalid operator graph commit-marker digest"
        ) from exc
    if claimed != _sha256(body) or _graph_commit_bytes(state_sha256) != data:
        raise ComputeOperatorGraphIntegrityError(
            "operator graph commit marker failed canonical reconstruction"
        )
    return state_sha256


@dataclass(frozen=True, slots=True)
class OperatorEdge:
    """One authenticated semantic transition backed by a published operator."""

    source_state: str
    target_state: str
    crystal_sha256: str
    verifier_sha256: str
    evidence_sha256: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_state", _text(self.source_state, field="source_state")
        )
        object.__setattr__(
            self, "target_state", _text(self.target_state, field="target_state")
        )
        for field in ("crystal_sha256", "verifier_sha256", "evidence_sha256"):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        object.__setattr__(self, "weight", _weight(self.weight))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": OPERATOR_EDGE_SCHEMA,
            "source_state": self.source_state,
            "target_state": self.target_state,
            "crystal_sha256": self.crystal_sha256,
            "verifier_sha256": self.verifier_sha256,
            "evidence_sha256": self.evidence_sha256,
            "weight": self.weight,
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    @property
    def action(self) -> str:
        """The exact action label used by the Markov world model."""

        return self.sha256

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "OperatorEdge":
        expected = {
            "schema",
            "source_state",
            "target_state",
            "crystal_sha256",
            "verifier_sha256",
            "evidence_sha256",
            "weight",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or value.get("schema") != OPERATOR_EDGE_SCHEMA
        ):
            raise ComputeOperatorGraphIntegrityError("invalid operator edge")
        try:
            return cls(
                source_state=cast(str, value.get("source_state")),
                target_state=cast(str, value.get("target_state")),
                crystal_sha256=cast(str, value.get("crystal_sha256")),
                verifier_sha256=cast(str, value.get("verifier_sha256")),
                evidence_sha256=cast(str, value.get("evidence_sha256")),
                weight=cast(float, value.get("weight")),
            )
        except (TypeError, ValueError) as exc:
            raise ComputeOperatorGraphIntegrityError(
                "operator edge validation failed"
            ) from exc

    @classmethod
    def from_bytes(cls, data: bytes) -> "OperatorEdge":
        value = _strict_json(
            data, label="operator edge", maximum=MAX_ROUTE_RECEIPT_BYTES
        )
        edge = cls.from_dict(value)
        if edge.to_bytes() != data:
            raise ComputeOperatorGraphIntegrityError(
                "operator edge failed canonical reconstruction"
            )
        return edge


def _parse_ledger(value: object) -> ContractionLedger | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ComputeOperatorGraphIntegrityError("invalid contraction ledger")
    try:
        return ContractionLedger.from_bytes(canonical_json_bytes(value))
    except (TypeError, ValueError) as exc:
        raise ComputeOperatorGraphIntegrityError(
            "contraction ledger validation failed"
        ) from exc


@dataclass(frozen=True, slots=True)
class MaterializedRoute:
    """An append-only address record for a planned compute composition."""

    source_state: str
    goal_state: str
    primitive_edge_sha256s: tuple[str, ...]
    planning_graph_generation: int
    planning_graph_state_sha256: str
    finite_horizon_plan: FiniteHorizonPlan
    finite_horizon_plan_sha256: str
    world_model_sha256: str
    primitive_program_sha256: str
    executable_program_sha256: str
    fused_crystal_sha256: str | None
    charge_basis_sha256: str | None
    charge_verifier_sha256: str | None
    verification_receipt_sha256: str | None
    input_abi_sha256: str
    output_abi_sha256: str
    equivalent_source_work_units: int
    live_work_units: int
    contraction_ledger: ContractionLedger | None
    verifier_sha256s: tuple[str, ...]
    evidence_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_state", _text(self.source_state, field="source_state")
        )
        object.__setattr__(
            self, "goal_state", _text(self.goal_state, field="goal_state")
        )
        edges = _hash_tuple(self.primitive_edge_sha256s, field="primitive_edge_sha256s")
        if not edges or len(edges) > MAX_PROGRAM_STEPS:
            raise ValueError("a materialized route needs bounded primitive edges")
        object.__setattr__(self, "primitive_edge_sha256s", edges)
        object.__setattr__(
            self,
            "planning_graph_generation",
            _uint(self.planning_graph_generation, field="planning_graph_generation"),
        )
        if not isinstance(self.finite_horizon_plan, FiniteHorizonPlan):
            raise TypeError("finite_horizon_plan must be a FiniteHorizonPlan")
        for field in (
            "planning_graph_state_sha256",
            "finite_horizon_plan_sha256",
            "world_model_sha256",
            "primitive_program_sha256",
            "executable_program_sha256",
            "input_abi_sha256",
            "output_abi_sha256",
        ):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        if self.finite_horizon_plan_sha256 != self.finite_horizon_plan.sha256:
            raise ValueError("finite-horizon plan content address mismatch")
        if self.finite_horizon_plan.expected_actions != edges:
            raise ValueError(
                "finite-horizon plan actions disagree with primitive edges"
            )
        if (
            self.finite_horizon_plan.start_state != self.source_state
            or self.finite_horizon_plan.goal_state != self.goal_state
            or self.finite_horizon_plan.world_model_sha256 != self.world_model_sha256
        ):
            raise ValueError("finite-horizon plan semantic binding mismatch")
        if self.fused_crystal_sha256 is not None:
            object.__setattr__(
                self,
                "fused_crystal_sha256",
                require_sha256(self.fused_crystal_sha256, field="fused_crystal_sha256"),
            )
        charge_fields = (
            self.charge_basis_sha256,
            self.charge_verifier_sha256,
            self.verification_receipt_sha256,
        )
        if any(value is None for value in charge_fields) and any(
            value is not None for value in charge_fields
        ):
            raise ValueError("compute charge address and verifier lineage are atomic")
        for field in (
            "charge_basis_sha256",
            "charge_verifier_sha256",
            "verification_receipt_sha256",
        ):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, require_sha256(value, field=field))
        source_work = _uint(
            self.equivalent_source_work_units,
            field="equivalent_source_work_units",
            positive=True,
        )
        live_work = _uint(self.live_work_units, field="live_work_units", positive=True)
        object.__setattr__(self, "equivalent_source_work_units", source_work)
        object.__setattr__(self, "live_work_units", live_work)
        if self.contraction_ledger is not None and not isinstance(
            self.contraction_ledger, ContractionLedger
        ):
            raise TypeError("contraction_ledger must be a ContractionLedger or None")
        object.__setattr__(
            self,
            "verifier_sha256s",
            _sorted_hash_tuple(self.verifier_sha256s, field="verifier_sha256s"),
        )
        object.__setattr__(
            self,
            "evidence_sha256s",
            _sorted_hash_tuple(self.evidence_sha256s, field="evidence_sha256s"),
        )

    @property
    def historical_work_units(self) -> int:
        return max(0, self.equivalent_source_work_units - self.live_work_units)

    def to_dict(self) -> dict[str, object]:
        ledger = self.contraction_ledger
        return {
            "schema": MATERIALIZED_ROUTE_SCHEMA,
            "source_state": self.source_state,
            "goal_state": self.goal_state,
            "primitive_edge_sha256s": list(self.primitive_edge_sha256s),
            "planning_graph_generation": self.planning_graph_generation,
            "planning_graph_state_sha256": self.planning_graph_state_sha256,
            "finite_horizon_plan": self.finite_horizon_plan.to_dict(),
            "finite_horizon_plan_sha256": self.finite_horizon_plan_sha256,
            "world_model_sha256": self.world_model_sha256,
            "primitive_program_sha256": self.primitive_program_sha256,
            "executable_program_sha256": self.executable_program_sha256,
            "fused_crystal_sha256": self.fused_crystal_sha256,
            "charge_basis_sha256": self.charge_basis_sha256,
            "charge_verifier_sha256": self.charge_verifier_sha256,
            "verification_receipt_sha256": self.verification_receipt_sha256,
            "input_abi_sha256": self.input_abi_sha256,
            "output_abi_sha256": self.output_abi_sha256,
            "equivalent_source_work_units": self.equivalent_source_work_units,
            "live_work_units": self.live_work_units,
            "historical_work_units": self.historical_work_units,
            "contraction_ledger": None if ledger is None else ledger.to_dict(),
            "contraction_ledger_sha256": None if ledger is None else ledger.sha256,
            "verifier_sha256s": list(self.verifier_sha256s),
            "evidence_sha256s": list(self.evidence_sha256s),
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_dict())
        if len(data) > MAX_ROUTE_RECEIPT_BYTES:
            raise ValueError("materialized route exceeds its byte bound")
        return data

    @classmethod
    def from_dict(cls, value: object) -> "MaterializedRoute":
        expected = {
            "schema",
            "source_state",
            "goal_state",
            "primitive_edge_sha256s",
            "planning_graph_generation",
            "planning_graph_state_sha256",
            "finite_horizon_plan",
            "finite_horizon_plan_sha256",
            "world_model_sha256",
            "primitive_program_sha256",
            "executable_program_sha256",
            "fused_crystal_sha256",
            "charge_basis_sha256",
            "charge_verifier_sha256",
            "verification_receipt_sha256",
            "input_abi_sha256",
            "output_abi_sha256",
            "equivalent_source_work_units",
            "live_work_units",
            "historical_work_units",
            "contraction_ledger",
            "contraction_ledger_sha256",
            "verifier_sha256s",
            "evidence_sha256s",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or value.get("schema") != MATERIALIZED_ROUTE_SCHEMA
        ):
            raise ComputeOperatorGraphIntegrityError("invalid materialized route")
        raw_edges = value.get("primitive_edge_sha256s")
        raw_verifiers = value.get("verifier_sha256s")
        raw_evidence = value.get("evidence_sha256s")
        if not all(
            isinstance(item, list) for item in (raw_edges, raw_verifiers, raw_evidence)
        ):
            raise ComputeOperatorGraphIntegrityError(
                "materialized route inventories must be lists"
            )
        ledger = _parse_ledger(value.get("contraction_ledger"))
        claimed_ledger = value.get("contraction_ledger_sha256")
        if (ledger is None) != (claimed_ledger is None):
            raise ComputeOperatorGraphIntegrityError(
                "contraction ledger address is inconsistent"
            )
        if ledger is not None:
            try:
                claimed = require_sha256(
                    claimed_ledger, field="contraction_ledger_sha256"
                )
            except ValueError as exc:
                raise ComputeOperatorGraphIntegrityError(
                    "invalid contraction ledger address"
                ) from exc
            if claimed != ledger.sha256:
                raise ComputeOperatorGraphIntegrityError(
                    "contraction ledger address mismatch"
                )
        try:
            route = cls(
                source_state=cast(str, value.get("source_state")),
                goal_state=cast(str, value.get("goal_state")),
                primitive_edge_sha256s=tuple(cast(list[str], raw_edges)),
                planning_graph_generation=cast(
                    int, value.get("planning_graph_generation")
                ),
                planning_graph_state_sha256=cast(
                    str, value.get("planning_graph_state_sha256")
                ),
                finite_horizon_plan=_finite_plan_from_dict(
                    value.get("finite_horizon_plan")
                ),
                finite_horizon_plan_sha256=cast(
                    str, value.get("finite_horizon_plan_sha256")
                ),
                world_model_sha256=cast(str, value.get("world_model_sha256")),
                primitive_program_sha256=cast(
                    str, value.get("primitive_program_sha256")
                ),
                executable_program_sha256=cast(
                    str, value.get("executable_program_sha256")
                ),
                fused_crystal_sha256=cast(
                    str | None, value.get("fused_crystal_sha256")
                ),
                charge_basis_sha256=cast(str | None, value.get("charge_basis_sha256")),
                charge_verifier_sha256=cast(
                    str | None, value.get("charge_verifier_sha256")
                ),
                verification_receipt_sha256=cast(
                    str | None, value.get("verification_receipt_sha256")
                ),
                input_abi_sha256=cast(str, value.get("input_abi_sha256")),
                output_abi_sha256=cast(str, value.get("output_abi_sha256")),
                equivalent_source_work_units=cast(
                    int, value.get("equivalent_source_work_units")
                ),
                live_work_units=cast(int, value.get("live_work_units")),
                contraction_ledger=ledger,
                verifier_sha256s=tuple(cast(list[str], raw_verifiers)),
                evidence_sha256s=tuple(cast(list[str], raw_evidence)),
            )
        except (TypeError, ValueError) as exc:
            raise ComputeOperatorGraphIntegrityError(
                "materialized route validation failed"
            ) from exc
        if route.historical_work_units != value.get("historical_work_units"):
            raise ComputeOperatorGraphIntegrityError(
                "materialized route derived work was altered"
            )
        if route.to_dict() != dict(value):
            raise ComputeOperatorGraphIntegrityError(
                "materialized route failed canonical reconstruction"
            )
        return route

    @classmethod
    def from_bytes(cls, data: bytes) -> "MaterializedRoute":
        value = _strict_json(
            data, label="materialized route", maximum=MAX_ROUTE_RECEIPT_BYTES
        )
        route = cls.from_dict(value)
        if route.to_bytes() != data:
            raise ComputeOperatorGraphIntegrityError(
                "materialized route failed canonical byte reconstruction"
            )
        return route


@dataclass(frozen=True, slots=True)
class ComputeOperatorGraphState:
    generation: int
    previous_state_sha256: str | None
    edges: tuple[OperatorEdge, ...]
    materialized_routes: tuple[MaterializedRoute, ...]

    def __post_init__(self) -> None:
        generation = _uint(self.generation, field="graph generation")
        previous = self.previous_state_sha256
        if previous is not None:
            previous = require_sha256(previous, field="previous_state_sha256")
        edges = tuple(self.edges)
        routes = tuple(self.materialized_routes)
        if len(edges) > MAX_GRAPH_EDGES or len(routes) > MAX_MATERIALIZED_ROUTES:
            raise ValueError("operator graph exceeds its inventory bound")
        if any(not isinstance(edge, OperatorEdge) for edge in edges):
            raise TypeError("graph edges must be OperatorEdge values")
        if any(not isinstance(route, MaterializedRoute) for route in routes):
            raise TypeError("graph routes must be MaterializedRoute values")
        if tuple(sorted(edges, key=lambda edge: edge.sha256)) != edges:
            raise ValueError("graph edges must be sorted by content address")
        if tuple(sorted(routes, key=lambda route: route.sha256)) != routes:
            raise ValueError("graph routes must be sorted by content address")
        if len({edge.sha256 for edge in edges}) != len(edges):
            raise ValueError("graph edges must be unique")
        if len({edge.evidence_sha256 for edge in edges}) != len(edges):
            raise ValueError("one evidence record cannot authenticate multiple edges")
        if len({route.sha256 for route in routes}) != len(routes):
            raise ValueError("graph routes must be unique")
        if generation == 0 and (previous is not None or edges or routes):
            raise ValueError("generation zero must be the empty graph")
        if generation > 0 and previous is None:
            raise ValueError("non-empty graph generations must bind a predecessor")
        if generation > len(edges) + len(routes):
            raise ValueError("graph generation exceeds its append inventory")
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "previous_state_sha256", previous)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "materialized_routes", routes)

    @classmethod
    def empty(cls) -> "ComputeOperatorGraphState":
        return cls(0, None, (), ())

    def as_record(self) -> dict[str, object]:
        return {
            "generation": self.generation,
            "previous_state_sha256": self.previous_state_sha256,
            "edges": [edge.to_dict() for edge in self.edges],
            "materialized_routes": [
                route.to_dict() for route in self.materialized_routes
            ],
        }

    def to_document(self) -> dict[str, object]:
        body = self.as_record()
        return {
            "schema": OPERATOR_GRAPH_STATE_SCHEMA,
            "body": body,
            "body_sha256": _sha256(body),
        }

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_document())
        if len(data) > MAX_GRAPH_STATE_BYTES:
            raise ValueError("operator graph state exceeds its hard byte limit")
        return data

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "ComputeOperatorGraphState":
        value = _strict_json(
            data, label="compute operator graph", maximum=MAX_GRAPH_STATE_BYTES
        )
        if (
            not isinstance(value, Mapping)
            or set(value) != {"schema", "body", "body_sha256"}
            or value.get("schema") != OPERATOR_GRAPH_STATE_SCHEMA
        ):
            raise ComputeOperatorGraphIntegrityError("invalid operator graph envelope")
        body = value.get("body")
        if not isinstance(body, Mapping) or set(body) != {
            "generation",
            "previous_state_sha256",
            "edges",
            "materialized_routes",
        }:
            raise ComputeOperatorGraphIntegrityError("invalid operator graph body")
        try:
            claimed = require_sha256(value.get("body_sha256"), field="body_sha256")
        except ValueError as exc:
            raise ComputeOperatorGraphIntegrityError(
                "invalid operator graph body hash"
            ) from exc
        if claimed != _sha256(body):
            raise ComputeOperatorGraphIntegrityError(
                "operator graph body hash mismatch"
            )
        raw_edges = body.get("edges")
        raw_routes = body.get("materialized_routes")
        if not isinstance(raw_edges, list) or not isinstance(raw_routes, list):
            raise ComputeOperatorGraphIntegrityError(
                "operator graph inventories must be lists"
            )
        try:
            state = cls(
                generation=cast(int, body.get("generation")),
                previous_state_sha256=cast(
                    str | None, body.get("previous_state_sha256")
                ),
                edges=tuple(OperatorEdge.from_dict(edge) for edge in raw_edges),
                materialized_routes=tuple(
                    MaterializedRoute.from_dict(route) for route in raw_routes
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ComputeOperatorGraphIntegrityError(
                "operator graph validation failed"
            ) from exc
        if state.to_bytes() != data:
            raise ComputeOperatorGraphIntegrityError(
                "operator graph failed canonical reconstruction"
            )
        return state


def _validate_committed_graph_history(
    histories: Mapping[str, ComputeOperatorGraphState],
    commits: set[str],
) -> ComputeOperatorGraphState:
    if (
        len(histories) > MAX_GRAPH_GENERATIONS + 1
        or len(commits) > MAX_GRAPH_GENERATIONS
    ):
        raise ComputeOperatorGraphIntegrityError(
            "operator graph history exceeds its append inventory bound"
        )
    by_generation: dict[int, tuple[str, ComputeOperatorGraphState]] = {}
    for digest in commits:
        state = histories.get(digest)
        if state is None:
            raise ComputeOperatorGraphIntegrityError(
                "committed graph state is missing its immutable history object"
            )
        if state.sha256 != digest:
            raise ComputeOperatorGraphIntegrityError(
                "operator graph history address does not match its bytes"
            )
        if state.generation < 1:
            raise ComputeOperatorGraphIntegrityError(
                "the deterministic empty graph must not be committed"
            )
        existing = by_generation.get(state.generation)
        if existing is not None and existing[0] != digest:
            raise ComputeOperatorGraphIntegrityError(
                "operator graph committed history contains a fork"
            )
        by_generation[state.generation] = (digest, state)

    previous = ComputeOperatorGraphState.empty()
    if not by_generation:
        return previous
    maximum = max(by_generation)
    if set(by_generation) != set(range(1, maximum + 1)):
        raise ComputeOperatorGraphIntegrityError(
            "operator graph committed history contains a generation gap"
        )
    for generation in range(1, maximum + 1):
        _digest, current = by_generation[generation]
        ComputeOperatorGraph._validate_extension(previous, current)
        previous = current
    return previous


def _finite_plan_from_dict(value: object) -> FiniteHorizonPlan:
    if not isinstance(value, Mapping):
        raise ComputeOperatorGraphIntegrityError("route plan has no finite plan")
    expected = {
        "schema",
        "world_model_sha256",
        "counts_sha256",
        "start_state",
        "goal_state",
        "horizon",
        "min_evidence_mass",
        "max_normalized_entropy",
        "min_peak_probability",
        "min_predicted_success",
        "gate_config_sha256",
        "terminal_rewards",
        "predicted_success",
        "minimum_coverage",
        "expected_states",
        "expected_actions",
        "policy",
        "kernel_hashes",
        "verifier_hashes",
        "evidence_hashes",
    }
    if (
        set(value) != expected
        or value.get("schema") != "immer-ooe-finite-horizon-plan/v1"
    ):
        raise ComputeOperatorGraphIntegrityError("invalid finite-horizon plan")
    terminal = value.get("terminal_rewards")
    kernels = value.get("kernel_hashes")
    raw_policy = value.get("policy")
    raw_states = value.get("expected_states")
    raw_actions = value.get("expected_actions")
    raw_verifiers = value.get("verifier_hashes")
    raw_evidence = value.get("evidence_hashes")
    if (
        not isinstance(terminal, Mapping)
        or not isinstance(kernels, Mapping)
        or not isinstance(raw_policy, list)
        or not isinstance(raw_states, list)
        or not isinstance(raw_actions, list)
        or not isinstance(raw_verifiers, list)
        or not isinstance(raw_evidence, list)
    ):
        raise ComputeOperatorGraphIntegrityError("invalid finite-plan collections")
    policies: list[PolicyEntry] = []
    try:
        for entry in raw_policy:
            if not isinstance(entry, Mapping) or set(entry) != {
                "remaining_horizon",
                "state",
                "action",
                "predicted_value",
                "coverage",
                "transition_row_sha256",
            }:
                raise ValueError("invalid policy entry")
            policies.append(PolicyEntry(**dict(entry)))
        plan = FiniteHorizonPlan(
            world_model_sha256=cast(str, value.get("world_model_sha256")),
            counts_sha256=cast(str, value.get("counts_sha256")),
            start_state=cast(str, value.get("start_state")),
            goal_state=cast(str | None, value.get("goal_state")),
            horizon=cast(int, value.get("horizon")),
            min_evidence_mass=cast(float, value.get("min_evidence_mass")),
            max_normalized_entropy=cast(float, value.get("max_normalized_entropy")),
            min_peak_probability=cast(float, value.get("min_peak_probability")),
            min_predicted_success=cast(float, value.get("min_predicted_success")),
            gate_config_sha256=cast(str, value.get("gate_config_sha256")),
            terminal_rewards=tuple(
                (cast(str, key), cast(float, val)) for key, val in terminal.items()
            ),
            predicted_success=cast(float, value.get("predicted_success")),
            minimum_coverage=cast(float, value.get("minimum_coverage")),
            expected_states=tuple(cast(list[str], raw_states)),
            expected_actions=tuple(cast(list[str], raw_actions)),
            policy=tuple(policies),
            kernel_hashes=tuple(
                (cast(str, key), cast(str, val)) for key, val in kernels.items()
            ),
            verifier_hashes=tuple(cast(list[str], raw_verifiers)),
            evidence_hashes=tuple(cast(list[str], raw_evidence)),
        )
    except (TypeError, ValueError) as exc:
        raise ComputeOperatorGraphIntegrityError(
            "finite-horizon plan validation failed"
        ) from exc
    if plan.to_dict() != dict(value):
        raise ComputeOperatorGraphIntegrityError(
            "finite-horizon plan failed canonical reconstruction"
        )
    return plan


@dataclass(frozen=True, slots=True)
class ComputeRoutePlan:
    graph_generation: int
    graph_state_sha256: str
    finite_plan: FiniteHorizonPlan
    primitive_edge_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
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
        if not isinstance(self.finite_plan, FiniteHorizonPlan):
            raise TypeError("finite_plan must be a FiniteHorizonPlan")
        edges = _hash_tuple(self.primitive_edge_sha256s, field="primitive_edge_sha256s")
        if edges != self.finite_plan.expected_actions or not edges:
            raise ValueError("route edges must equal the finite plan action path")
        object.__setattr__(self, "primitive_edge_sha256s", edges)

    @property
    def source_state(self) -> str:
        return self.finite_plan.start_state

    @property
    def goal_state(self) -> str:
        goal = self.finite_plan.goal_state
        if goal is None:
            raise ValueError("compute routes require a concrete goal")
        return goal

    @property
    def world_model_sha256(self) -> str:
        return self.finite_plan.world_model_sha256

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": ROUTE_PLAN_SCHEMA,
            "graph_generation": self.graph_generation,
            "graph_state_sha256": self.graph_state_sha256,
            "finite_plan": self.finite_plan.to_dict(),
            "primitive_edge_sha256s": list(self.primitive_edge_sha256s),
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_dict())
        if len(data) > MAX_ROUTE_RECEIPT_BYTES:
            raise ValueError("route plan exceeds its byte bound")
        return data

    @classmethod
    def from_bytes(cls, data: bytes) -> "ComputeRoutePlan":
        value = _strict_json(
            data, label="compute route plan", maximum=MAX_ROUTE_RECEIPT_BYTES
        )
        if (
            not isinstance(value, Mapping)
            or set(value)
            != {
                "schema",
                "graph_generation",
                "graph_state_sha256",
                "finite_plan",
                "primitive_edge_sha256s",
            }
            or value.get("schema") != ROUTE_PLAN_SCHEMA
        ):
            raise ComputeOperatorGraphIntegrityError("invalid compute route plan")
        raw_edges = value.get("primitive_edge_sha256s")
        if not isinstance(raw_edges, list):
            raise ComputeOperatorGraphIntegrityError("route plan edges must be a list")
        try:
            plan = cls(
                graph_generation=cast(int, value.get("graph_generation")),
                graph_state_sha256=cast(str, value.get("graph_state_sha256")),
                finite_plan=_finite_plan_from_dict(value.get("finite_plan")),
                primitive_edge_sha256s=tuple(raw_edges),
            )
        except (TypeError, ValueError) as exc:
            raise ComputeOperatorGraphIntegrityError(
                "route plan validation failed"
            ) from exc
        if plan.to_bytes() != data:
            raise ComputeOperatorGraphIntegrityError(
                "route plan failed canonical reconstruction"
            )
        return plan


@dataclass(frozen=True, slots=True)
class ComputeRoutePlanDecision:
    plan: ComputeRoutePlan | None
    abstained: bool
    reason: str | None
    graph_state_sha256: str
    world_model_sha256: str | None
    predicted_success: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "graph_state_sha256",
            require_sha256(self.graph_state_sha256, field="graph_state_sha256"),
        )
        if self.world_model_sha256 is not None:
            object.__setattr__(
                self,
                "world_model_sha256",
                require_sha256(self.world_model_sha256, field="world_model_sha256"),
            )
        if not isinstance(self.abstained, bool):
            raise TypeError("abstained must be boolean")
        if (self.plan is None) != self.abstained:
            raise ValueError("route decision plan and abstention disagree")
        if self.abstained and not self.reason:
            raise ValueError("an abstention requires a reason")
        if not self.abstained and self.reason is not None:
            raise ValueError("an accepted plan cannot have an abstention reason")
        if (
            not isinstance(self.predicted_success, (int, float))
            or not 0.0 <= float(self.predicted_success) <= 1.0
        ):
            raise ValueError("predicted_success must lie in [0, 1]")
        object.__setattr__(self, "predicted_success", float(self.predicted_success))


@dataclass(frozen=True, slots=True)
class RouteChargeReceipt:
    route: MaterializedRoute
    graph_generation: int
    graph_state_sha256: str
    graph_changed: bool
    primitive_program_created: bool
    fused_crystal_created: bool
    executable_program_created: bool
    charge_basis_created: bool
    composition_work_units: int

    def __post_init__(self) -> None:
        if not isinstance(self.route, MaterializedRoute):
            raise TypeError("route must be a MaterializedRoute")
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
        for field in (
            "graph_changed",
            "primitive_program_created",
            "fused_crystal_created",
            "executable_program_created",
            "charge_basis_created",
        ):
            if not isinstance(getattr(self, field), bool):
                raise TypeError(f"{field} must be boolean")
        object.__setattr__(
            self,
            "composition_work_units",
            _uint(self.composition_work_units, field="composition_work_units"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": ROUTE_CHARGE_SCHEMA,
            "route": self.route.to_dict(),
            "route_sha256": self.route.sha256,
            "graph_generation": self.graph_generation,
            "graph_state_sha256": self.graph_state_sha256,
            "graph_changed": self.graph_changed,
            "primitive_program_created": self.primitive_program_created,
            "fused_crystal_created": self.fused_crystal_created,
            "executable_program_created": self.executable_program_created,
            "charge_basis_created": self.charge_basis_created,
            "composition_work_units": self.composition_work_units,
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_dict())
        if len(data) > MAX_ROUTE_RECEIPT_BYTES:
            raise ValueError("route charge receipt exceeds its byte bound")
        return data

    @classmethod
    def from_bytes(cls, data: bytes) -> "RouteChargeReceipt":
        value = _strict_json(
            data, label="route charge receipt", maximum=MAX_ROUTE_RECEIPT_BYTES
        )
        expected = {
            "schema",
            "route",
            "route_sha256",
            "graph_generation",
            "graph_state_sha256",
            "graph_changed",
            "primitive_program_created",
            "fused_crystal_created",
            "executable_program_created",
            "charge_basis_created",
            "composition_work_units",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or value.get("schema") != ROUTE_CHARGE_SCHEMA
        ):
            raise ComputeOperatorGraphIntegrityError("invalid route charge receipt")
        route = MaterializedRoute.from_dict(value.get("route"))
        try:
            route_sha = require_sha256(value.get("route_sha256"), field="route_sha256")
            receipt = cls(
                route=route,
                graph_generation=cast(int, value.get("graph_generation")),
                graph_state_sha256=cast(str, value.get("graph_state_sha256")),
                graph_changed=cast(bool, value.get("graph_changed")),
                primitive_program_created=cast(
                    bool, value.get("primitive_program_created")
                ),
                fused_crystal_created=cast(bool, value.get("fused_crystal_created")),
                executable_program_created=cast(
                    bool, value.get("executable_program_created")
                ),
                charge_basis_created=cast(bool, value.get("charge_basis_created")),
                composition_work_units=cast(int, value.get("composition_work_units")),
            )
        except (TypeError, ValueError) as exc:
            raise ComputeOperatorGraphIntegrityError(
                "route charge receipt validation failed"
            ) from exc
        if route_sha != route.sha256 or receipt.to_bytes() != data:
            raise ComputeOperatorGraphIntegrityError(
                "route charge receipt failed canonical reconstruction"
            )
        return receipt


@dataclass(frozen=True, slots=True)
class RouteDischargeReceipt:
    route_sha256: str
    finite_horizon_plan_sha256: str
    world_model_sha256: str
    planning_graph_state_sha256: str
    graph_generation: int
    graph_state_sha256: str
    primitive_program_sha256: str
    executable_program_sha256: str
    fused_crystal_sha256: str | None
    charge_basis_sha256: str | None
    vm_receipt: ComputeExecutionReceipt
    application_count: int
    equivalent_source_work_units: int
    live_work_units: int
    historical_work_released: int
    verifier_sha256s: tuple[str, ...]
    evidence_sha256s: tuple[str, ...]
    contraction_ledger_sha256: str | None

    def __post_init__(self) -> None:
        for field in (
            "route_sha256",
            "finite_horizon_plan_sha256",
            "world_model_sha256",
            "planning_graph_state_sha256",
            "graph_state_sha256",
            "primitive_program_sha256",
            "executable_program_sha256",
        ):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        for field in (
            "fused_crystal_sha256",
            "charge_basis_sha256",
            "contraction_ledger_sha256",
        ):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, require_sha256(value, field=field))
        if not isinstance(self.vm_receipt, ComputeExecutionReceipt):
            raise TypeError("vm_receipt must be a ComputeExecutionReceipt")
        if self.vm_receipt.program_sha256 != self.executable_program_sha256:
            raise ValueError("VM receipt executed a different program")
        if self.vm_receipt.charge_basis_sha256 != self.charge_basis_sha256:
            raise ValueError("VM receipt used a different compute charge basis")
        object.__setattr__(
            self,
            "graph_generation",
            _uint(self.graph_generation, field="graph_generation"),
        )
        applications = _uint(
            self.application_count, field="application_count", positive=True
        )
        source = _uint(
            self.equivalent_source_work_units,
            field="equivalent_source_work_units",
            positive=True,
        )
        live = _uint(self.live_work_units, field="live_work_units", positive=True)
        released = _uint(
            self.historical_work_released, field="historical_work_released"
        )
        if released != max(0, source - live):
            raise ValueError("route savings do not match source and live work")
        if self.vm_receipt.live_discharge_work != live:
            raise ValueError("route live work disagrees with the VM receipt")
        if (
            self.vm_receipt.equivalent_unfused_source_work != source
            or self.vm_receipt.historical_work_released != released
        ):
            raise ValueError("route savings disagree with the authenticated VM receipt")
        object.__setattr__(self, "application_count", applications)
        object.__setattr__(self, "equivalent_source_work_units", source)
        object.__setattr__(self, "live_work_units", live)
        object.__setattr__(self, "historical_work_released", released)
        object.__setattr__(
            self,
            "verifier_sha256s",
            _sorted_hash_tuple(self.verifier_sha256s, field="verifier_sha256s"),
        )
        object.__setattr__(
            self,
            "evidence_sha256s",
            _sorted_hash_tuple(self.evidence_sha256s, field="evidence_sha256s"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": ROUTE_DISCHARGE_SCHEMA,
            "route_sha256": self.route_sha256,
            "finite_horizon_plan_sha256": self.finite_horizon_plan_sha256,
            "world_model_sha256": self.world_model_sha256,
            "planning_graph_state_sha256": self.planning_graph_state_sha256,
            "graph_generation": self.graph_generation,
            "graph_state_sha256": self.graph_state_sha256,
            "primitive_program_sha256": self.primitive_program_sha256,
            "executable_program_sha256": self.executable_program_sha256,
            "fused_crystal_sha256": self.fused_crystal_sha256,
            "charge_basis_sha256": self.charge_basis_sha256,
            "vm_receipt": self.vm_receipt.to_document(),
            "vm_receipt_sha256": self.vm_receipt.sha256,
            "application_count": self.application_count,
            "equivalent_source_work_units": self.equivalent_source_work_units,
            "live_work_units": self.live_work_units,
            "historical_work_released": self.historical_work_released,
            "verifier_sha256s": list(self.verifier_sha256s),
            "evidence_sha256s": list(self.evidence_sha256s),
            "contraction_ledger_sha256": self.contraction_ledger_sha256,
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_dict())
        if len(data) > MAX_ROUTE_RECEIPT_BYTES:
            raise ValueError("route discharge receipt exceeds its byte bound")
        return data

    @classmethod
    def from_bytes(cls, data: bytes) -> "RouteDischargeReceipt":
        value = _strict_json(
            data, label="route discharge receipt", maximum=MAX_ROUTE_RECEIPT_BYTES
        )
        expected = {
            "schema",
            "route_sha256",
            "finite_horizon_plan_sha256",
            "world_model_sha256",
            "planning_graph_state_sha256",
            "graph_generation",
            "graph_state_sha256",
            "primitive_program_sha256",
            "executable_program_sha256",
            "fused_crystal_sha256",
            "charge_basis_sha256",
            "vm_receipt",
            "vm_receipt_sha256",
            "application_count",
            "equivalent_source_work_units",
            "live_work_units",
            "historical_work_released",
            "verifier_sha256s",
            "evidence_sha256s",
            "contraction_ledger_sha256",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or value.get("schema") != ROUTE_DISCHARGE_SCHEMA
        ):
            raise ComputeOperatorGraphIntegrityError("invalid route discharge receipt")
        vm_document = value.get("vm_receipt")
        verifiers = value.get("verifier_sha256s")
        evidence = value.get("evidence_sha256s")
        if (
            not isinstance(vm_document, Mapping)
            or not isinstance(verifiers, list)
            or not isinstance(evidence, list)
        ):
            raise ComputeOperatorGraphIntegrityError(
                "invalid route discharge receipt collections"
            )
        try:
            vm = ComputeExecutionReceipt.from_bytes(canonical_json_bytes(vm_document))
            vm_sha = require_sha256(
                value.get("vm_receipt_sha256"), field="vm_receipt_sha256"
            )
            receipt = cls(
                route_sha256=cast(str, value.get("route_sha256")),
                finite_horizon_plan_sha256=cast(
                    str, value.get("finite_horizon_plan_sha256")
                ),
                world_model_sha256=cast(str, value.get("world_model_sha256")),
                planning_graph_state_sha256=cast(
                    str, value.get("planning_graph_state_sha256")
                ),
                graph_generation=cast(int, value.get("graph_generation")),
                graph_state_sha256=cast(str, value.get("graph_state_sha256")),
                primitive_program_sha256=cast(
                    str, value.get("primitive_program_sha256")
                ),
                executable_program_sha256=cast(
                    str, value.get("executable_program_sha256")
                ),
                fused_crystal_sha256=cast(
                    str | None, value.get("fused_crystal_sha256")
                ),
                charge_basis_sha256=cast(str | None, value.get("charge_basis_sha256")),
                vm_receipt=vm,
                application_count=cast(int, value.get("application_count")),
                equivalent_source_work_units=cast(
                    int, value.get("equivalent_source_work_units")
                ),
                live_work_units=cast(int, value.get("live_work_units")),
                historical_work_released=cast(
                    int, value.get("historical_work_released")
                ),
                verifier_sha256s=tuple(verifiers),
                evidence_sha256s=tuple(evidence),
                contraction_ledger_sha256=cast(
                    str | None, value.get("contraction_ledger_sha256")
                ),
            )
        except (ComputeCrystalIntegrityError, TypeError, ValueError) as exc:
            raise ComputeOperatorGraphIntegrityError(
                "route discharge receipt validation failed"
            ) from exc
        if vm_sha != vm.sha256 or receipt.to_bytes() != data:
            raise ComputeOperatorGraphIntegrityError(
                "route discharge receipt failed canonical reconstruction"
            )
        return receipt


@dataclass(frozen=True, slots=True)
class RouteDischarge:
    output: NDArray[Any]
    receipt: RouteDischargeReceipt
    route: MaterializedRoute


class ComputeOperatorGraph:
    """Crash-safe semantic graph backed by a ``ComputeCrystalBank``.

    Immutable history and commit markers detect partial rollback, forks, and
    crash windows.  A self-contained store cannot distinguish a complete,
    internally consistent replacement from the original.  Persist
    :meth:`current_anchor_sha256` outside the store and supply it through
    ``trusted_graph_state_sha256`` or ``trusted_head_resolver`` to detect that
    stronger attacker model.  Both are authoritative exact-head checks, not
    ancestor allowlists.  A supplied explicit anchor advances after this
    instance completes its own append.  A resolver stays externally
    authoritative and must be updated from the returned graph state/receipt
    after an authorized write before the next read.
    """

    def __init__(
        self,
        bank: ComputeCrystalBank | str | os.PathLike[str],
        *,
        retry_limit: int = 16,
        trusted_graph_state_sha256: str | None = None,
        trusted_head_resolver: Callable[[], str] | None = None,
    ) -> None:
        self.bank = (
            bank if isinstance(bank, ComputeCrystalBank) else ComputeCrystalBank(bank)
        )
        if (
            isinstance(retry_limit, bool)
            or not isinstance(retry_limit, int)
            or not 1 <= retry_limit <= 1024
        ):
            raise ValueError("retry_limit must lie in [1, 1024]")
        self.retry_limit = retry_limit
        self.root = Path(self.bank.root)
        self.trusted_graph_state_sha256 = (
            None
            if trusted_graph_state_sha256 is None
            else require_sha256(
                trusted_graph_state_sha256,
                field="trusted_graph_state_sha256",
            )
        )
        if trusted_head_resolver is not None and not callable(trusted_head_resolver):
            raise TypeError("trusted_head_resolver must be callable")
        self.trusted_head_resolver = trusted_head_resolver
        self._last_authorized_head_sha256: str | None = None

    def _trusted_anchors(self) -> tuple[str, ...]:
        anchors: list[str] = []
        if self.trusted_graph_state_sha256 is not None:
            anchors.append(self.trusted_graph_state_sha256)
        if self.trusted_head_resolver is not None:
            try:
                resolved = self.trusted_head_resolver()
            except Exception as exc:
                raise ComputeOperatorGraphIntegrityError(
                    "trusted graph-head resolver failed"
                ) from exc
            try:
                anchors.append(
                    require_sha256(resolved, field="trusted resolved graph head")
                )
            except ValueError as exc:
                raise ComputeOperatorGraphIntegrityError(
                    "trusted graph-head resolver returned an invalid anchor"
                ) from exc
        return tuple(dict.fromkeys(anchors))

    def _assert_trusted_anchors(
        self,
        head: ComputeOperatorGraphState,
    ) -> None:
        for anchor in self._trusted_anchors():
            if anchor != head.sha256:
                raise ComputeOperatorGraphIntegrityError(
                    "committed graph head does not equal the trusted anchor"
                )

    @contextmanager
    def _locked(self) -> Iterator[None]:
        path = self.root / _GRAPH_LOCK_NAME
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as exc:
            raise ComputeOperatorGraphIntegrityError(
                "cannot open compute-operator graph lock"
            ) from exc
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise ComputeOperatorGraphIntegrityError(
                    "compute-operator graph lock is not a regular file"
                )
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            linked = path.lstat()
            if (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino):
                raise ComputeOperatorGraphIntegrityError(
                    "compute-operator graph lock changed while acquiring it"
                )
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    @staticmethod
    def history_state_name(state_sha256: str) -> str:
        return OPERATOR_GRAPH_HISTORY_PREFIX + require_sha256(
            state_sha256, field="state_sha256"
        )

    @staticmethod
    def commit_state_name(state_sha256: str) -> str:
        return OPERATOR_GRAPH_COMMIT_PREFIX + require_sha256(
            state_sha256, field="state_sha256"
        )

    def _restore_raw(self, name: str) -> bytes:
        try:
            return self.bank.store.restore_state(name)
        except KeyError:
            raise
        except CrystalStoreError as exc:
            raise ComputeOperatorGraphIntegrityError(
                "compute-operator graph state failed storage integrity"
            ) from exc

    def _graph_history_state_names_unlocked(self) -> tuple[str, ...]:
        names: list[str] = []
        root_fd = os.open(self.root, self.bank.store._directory_flags())
        try:
            state_fd = os.open(
                "state", self.bank.store._directory_flags(), dir_fd=root_fd
            )
            try:
                for filename in sorted(os.listdir(state_fd)):
                    if not filename.endswith(".state"):
                        continue
                    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                    try:
                        descriptor = os.open(filename, flags, dir_fd=state_fd)
                    except OSError:
                        continue
                    try:
                        before = os.fstat(descriptor)
                        if not stat.S_ISREG(before.st_mode):
                            continue
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
                            raise ComputeOperatorGraphIntegrityError(
                                "operator graph state changed during history scan"
                            )
                    finally:
                        os.close(descriptor)
                    match = _STATE_NAME_RE.match(prefix)
                    if match is None:
                        continue
                    try:
                        name = match.group(1).decode("ascii")
                    except UnicodeDecodeError:
                        continue
                    if not name.startswith(
                        (OPERATOR_GRAPH_HISTORY_PREFIX, OPERATOR_GRAPH_COMMIT_PREFIX)
                    ):
                        continue
                    if filename != self.bank.store._state_filename(name):
                        raise ComputeOperatorGraphIntegrityError(
                            "operator graph history filename is not name-bound"
                        )
                    names.append(name)
            finally:
                os.close(state_fd)
        finally:
            os.close(root_fd)
        if len(names) > 2 * MAX_GRAPH_GENERATIONS + 1:
            raise ComputeOperatorGraphIntegrityError(
                "operator graph history scan exceeds its append inventory bound"
            )
        return tuple(names)

    def _history_inventory_unlocked(
        self,
    ) -> tuple[dict[str, ComputeOperatorGraphState], set[str]]:
        histories: dict[str, ComputeOperatorGraphState] = {}
        commits: set[str] = set()
        for name in self._graph_history_state_names_unlocked():
            payload = self._restore_raw(name)
            if name.startswith(OPERATOR_GRAPH_HISTORY_PREFIX):
                suffix = name[len(OPERATOR_GRAPH_HISTORY_PREFIX) :]
                try:
                    address = require_sha256(suffix, field="graph history address")
                except ValueError as exc:
                    raise ComputeOperatorGraphIntegrityError(
                        "operator graph history state name is invalid"
                    ) from exc
                state = ComputeOperatorGraphState.from_bytes(payload)
                if state.sha256 != address:
                    raise ComputeOperatorGraphIntegrityError(
                        "operator graph history is stored under another address"
                    )
                histories[address] = state
            else:
                suffix = name[len(OPERATOR_GRAPH_COMMIT_PREFIX) :]
                try:
                    address = require_sha256(suffix, field="graph commit address")
                except ValueError as exc:
                    raise ComputeOperatorGraphIntegrityError(
                        "operator graph commit state name is invalid"
                    ) from exc
                if _decode_graph_commit(payload) != address:
                    raise ComputeOperatorGraphIntegrityError(
                        "operator graph commit marker is stored under another address"
                    )
                commits.add(address)
        return histories, commits

    def _pointer_unlocked(self) -> tuple[ComputeOperatorGraphState, str | None]:
        try:
            data = self._restore_raw(OPERATOR_GRAPH_STATE_NAME)
        except KeyError:
            return ComputeOperatorGraphState.empty(), None
        state = ComputeOperatorGraphState.from_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        if state.sha256 != digest:
            raise ComputeOperatorGraphIntegrityError(
                "operator graph pointer content address mismatch"
            )
        return state, digest

    @staticmethod
    def _validate_extension(
        previous: ComputeOperatorGraphState,
        current: ComputeOperatorGraphState,
    ) -> None:
        if current.generation != previous.generation + 1:
            raise ComputeOperatorGraphIntegrityError(
                "operator graph history generation is not contiguous"
            )
        if current.previous_state_sha256 != previous.sha256:
            raise ComputeOperatorGraphIntegrityError(
                "operator graph history predecessor mismatch"
            )
        old_edges = {edge.sha256: edge.to_bytes() for edge in previous.edges}
        new_edges = {edge.sha256: edge.to_bytes() for edge in current.edges}
        old_routes = {
            route.sha256: canonical_json_bytes(route.to_dict())
            for route in previous.materialized_routes
        }
        new_routes = {
            route.sha256: canonical_json_bytes(route.to_dict())
            for route in current.materialized_routes
        }
        if any(new_edges.get(key) != value for key, value in old_edges.items()):
            raise ComputeOperatorGraphIntegrityError(
                "operator graph edge history is not append-only"
            )
        if any(new_routes.get(key) != value for key, value in old_routes.items()):
            raise ComputeOperatorGraphIntegrityError(
                "operator graph route history is not append-only"
            )
        if len(new_edges) == len(old_edges) and len(new_routes) == len(old_routes):
            raise ComputeOperatorGraphIntegrityError(
                "operator graph generation contains no append"
            )

    def _history_unlocked(
        self, current: ComputeOperatorGraphState
    ) -> tuple[ComputeOperatorGraphState, ...]:
        reverse: list[ComputeOperatorGraphState] = []
        seen: set[str] = set()
        node = current
        while True:
            digest = node.sha256
            if digest in seen:
                raise ComputeOperatorGraphIntegrityError(
                    "operator graph history contains a cycle"
                )
            seen.add(digest)
            try:
                historical = self._restore_raw(self.history_state_name(digest))
            except KeyError as exc:
                raise ComputeOperatorGraphIntegrityError(
                    "operator graph history object is missing"
                ) from exc
            if (
                historical != node.to_bytes()
                or hashlib.sha256(historical).hexdigest() != digest
            ):
                raise ComputeOperatorGraphIntegrityError(
                    "operator graph history content address mismatch"
                )
            reverse.append(node)
            if node.generation == 0:
                break
            previous_digest = node.previous_state_sha256
            if previous_digest is None:
                raise ComputeOperatorGraphIntegrityError(
                    "operator graph history lost its predecessor"
                )
            try:
                previous_data = self._restore_raw(
                    self.history_state_name(previous_digest)
                )
            except KeyError as exc:
                raise ComputeOperatorGraphIntegrityError(
                    "operator graph predecessor is missing"
                ) from exc
            if hashlib.sha256(previous_data).hexdigest() != previous_digest:
                raise ComputeOperatorGraphIntegrityError(
                    "operator graph predecessor address mismatch"
                )
            previous = ComputeOperatorGraphState.from_bytes(previous_data)
            self._validate_extension(previous, node)
            node = previous
        chain = tuple(reversed(reverse))
        if len(chain) != current.generation + 1:
            raise ComputeOperatorGraphIntegrityError(
                "operator graph history depth disagrees with generation"
            )
        return chain

    def _validated_plan_edges(
        self,
        *,
        planning_graph_generation: int,
        planning_graph_state_sha256: str,
        finite: FiniteHorizonPlan,
        primitive_edge_sha256s: tuple[str, ...],
        source_state: str,
        goal_state: str,
        edge_by_sha: Mapping[str, OperatorEdge],
        history_by_sha: Mapping[str, ComputeOperatorGraphState],
    ) -> tuple[OperatorEdge, ...]:
        planning_state = history_by_sha.get(planning_graph_state_sha256)
        if planning_state is None:
            raise ComputeOperatorGraphIntegrityError(
                "planning revision is outside committed graph history"
            )
        if planning_state.generation != planning_graph_generation:
            raise ComputeOperatorGraphIntegrityError(
                "planning generation does not match its graph state"
            )
        if (
            finite.start_state != source_state
            or finite.goal_state != goal_state
            or finite.expected_actions != primitive_edge_sha256s
        ):
            raise ComputeOperatorGraphIntegrityError(
                "finite plan semantic/action binding mismatch"
            )
        planning_world = self._world_model(planning_state)
        if planning_world.sha256 != finite.world_model_sha256:
            raise ComputeOperatorGraphIntegrityError(
                "finite plan world model does not match its graph revision"
            )
        planning_edges = {edge.sha256: edge for edge in planning_state.edges}
        try:
            edges = tuple(planning_edges[digest] for digest in primitive_edge_sha256s)
        except KeyError as exc:
            raise ComputeOperatorGraphIntegrityError(
                "finite plan references an edge absent from its graph revision"
            ) from exc
        if len(finite.expected_states) != len(edges) + 1:
            raise ComputeOperatorGraphIntegrityError("finite plan path length mismatch")
        for index, edge in enumerate(edges):
            if (
                edge.source_state != finite.expected_states[index]
                or edge.target_state != finite.expected_states[index + 1]
                or edge_by_sha.get(edge.sha256) != edge
            ):
                raise ComputeOperatorGraphIntegrityError(
                    "finite plan edge identity/topology mismatch"
                )
        if (
            not edges
            or edges[0].source_state != source_state
            or edges[-1].target_state != goal_state
        ):
            raise ComputeOperatorGraphIntegrityError(
                "finite plan endpoint binding mismatch"
            )
        planner = FiniteHorizonPlanner(
            planning_world,
            min_evidence_mass=finite.min_evidence_mass,
            max_normalized_entropy=finite.max_normalized_entropy,
            min_peak_probability=finite.min_peak_probability,
            min_predicted_success=finite.min_predicted_success,
            max_horizon=MAX_PROGRAM_STEPS,
            max_policy_entries=max(1, MAX_PROGRAM_STEPS * len(planning_world.states)),
        )
        replanned = planner.plan_goal(
            finite.start_state,
            goal_state,
            horizon=finite.horizon,
        )
        canonical = (
            replanned.plan is not None and replanned.plan.to_dict() == finite.to_dict()
        )
        if not canonical:
            exact = self._exact_finite_plan(planning_world, edges)
            if exact.to_dict() != finite.to_dict():
                raise ComputeOperatorGraphIntegrityError(
                    "finite plan is not reproducible from one-step evidence"
                )
        return edges

    @staticmethod
    def _exact_finite_plan(
        world: ActionConditionedWorldModel,
        edges: Sequence[OperatorEdge],
    ) -> FiniteHorizonPlan:
        """Reconstruct one explicit, evidence-gated trajectory exactly."""

        path = tuple(edges)
        if not path:
            raise ValueError("exact operator path must not be empty")
        if len(path) > MAX_PROGRAM_STEPS:
            raise ValueError("exact operator path exceeds the program step bound")
        expected_states = [path[0].source_state]
        policy: list[PolicyEntry] = []
        coverages: list[float] = []
        state_index = {label: index for index, label in enumerate(world.states)}
        for index, edge in enumerate(path):
            if index and path[index - 1].target_state != edge.source_state:
                raise ComputeOperatorGraphIntegrityError(
                    "exact operator path is topologically disconnected"
                )
            prediction = world.predict(
                edge.source_state,
                edge.sha256,
                min_evidence_mass=1.0,
                max_normalized_entropy=0.0,
                min_peak_probability=1.0,
            )
            target_index = state_index.get(edge.target_state)
            if (
                prediction.abstained
                or target_index is None
                or prediction.event_count != 1
                or prediction.evidence_mass != edge.weight
                or prediction.probabilities[target_index] != 1.0
            ):
                raise ComputeOperatorGraphIntegrityError(
                    "exact operator path lacks deterministic one-step evidence"
                )
            policy.append(
                PolicyEntry(
                    remaining_horizon=len(path) - index,
                    state=edge.source_state,
                    action=edge.sha256,
                    predicted_value=1.0,
                    coverage=prediction.coverage,
                    transition_row_sha256=prediction.row_sha256,
                )
            )
            coverages.append(prediction.coverage)
            expected_states.append(edge.target_state)
        gate_config_sha256 = _sha256(
            {
                "schema": "immer-ooe-planning-gates/v1",
                "min_evidence_mass": 1.0,
                "max_normalized_entropy": 0.0,
                "min_peak_probability": 1.0,
                "min_predicted_success": 1.0,
            }
        )
        return FiniteHorizonPlan(
            world_model_sha256=world.sha256,
            counts_sha256=array_sha256(world.counts),
            start_state=path[0].source_state,
            goal_state=path[-1].target_state,
            horizon=len(path),
            min_evidence_mass=1.0,
            max_normalized_entropy=0.0,
            min_peak_probability=1.0,
            min_predicted_success=1.0,
            gate_config_sha256=gate_config_sha256,
            terminal_rewards=((path[-1].target_state, 1.0),),
            predicted_success=1.0,
            minimum_coverage=min(coverages),
            expected_states=tuple(expected_states),
            expected_actions=tuple(edge.sha256 for edge in path),
            policy=tuple(policy),
            kernel_hashes=tuple(
                (action, array_sha256(world.action_kernel(action)))
                for action in world.actions
            ),
            verifier_hashes=world.verifier_hashes,
            evidence_hashes=world.evidence_hashes,
        )

    def _validate_materialized_route(
        self,
        route: MaterializedRoute,
        edge_by_sha: Mapping[str, OperatorEdge],
        history_by_sha: Mapping[str, ComputeOperatorGraphState],
    ) -> None:
        finite = route.finite_horizon_plan
        edges = self._validated_plan_edges(
            planning_graph_generation=route.planning_graph_generation,
            planning_graph_state_sha256=route.planning_graph_state_sha256,
            finite=finite,
            primitive_edge_sha256s=route.primitive_edge_sha256s,
            source_state=route.source_state,
            goal_state=route.goal_state,
            edge_by_sha=edge_by_sha,
            history_by_sha=history_by_sha,
        )
        try:
            crystals = tuple(
                self.bank.restore_crystal(edge.crystal_sha256) for edge in edges
            )
            primitive = self.bank.restore_program(route.primitive_program_sha256)
            executable = self.bank.restore_program(route.executable_program_sha256)
        except ComputeCrystalError as exc:
            raise ComputeOperatorGraphIntegrityError(
                "materialized route artifact restore failed"
            ) from exc
        if primitive.crystal_sha256s != tuple(edge.crystal_sha256 for edge in edges):
            raise ComputeOperatorGraphIntegrityError(
                "primitive program disagrees with semantic edge order"
            )
        try:
            expected_primitive = ComputeProgram.compose(crystals)
        except ComputeCrystalABIError as exc:
            raise ComputeOperatorGraphIntegrityError(
                "materialized route primitive ABI is invalid"
            ) from exc
        if expected_primitive != primitive:
            raise ComputeOperatorGraphIntegrityError(
                "materialized route primitive program is not canonical"
            )
        if (
            primitive.input_abi.sha256 != route.input_abi_sha256
            or primitive.output_abi.sha256 != route.output_abi_sha256
        ):
            raise ComputeOperatorGraphIntegrityError(
                "materialized route ABI address mismatch"
            )
        source_work = _checked_sum(
            [crystal.discharge_work_units for crystal in crystals],
            field="route source work",
        )
        kinds = {crystal.operator_kind for crystal in crystals}
        fusible = len(crystals) >= 2 and kinds in (
            {AFFINE_FLOAT64},
            {PERMUTATION},
            {MARKOV_FLOAT64},
            {CAUSAL_MIX_FLOAT64},
        )
        if fusible:
            if route.fused_crystal_sha256 is None or route.charge_basis_sha256 is None:
                raise ComputeOperatorGraphIntegrityError(
                    "a compatible route lacks its fused operator or compute charge"
                )
            if executable.crystal_sha256s != (route.fused_crystal_sha256,):
                raise ComputeOperatorGraphIntegrityError(
                    "fused route executable program is not one operator"
                )
            fused = self.bank.restore_crystal(route.fused_crystal_sha256)
            if fused.parent_sha256s != tuple(crystal.sha256 for crystal in crystals):
                raise ComputeOperatorGraphIntegrityError(
                    "fused route parent lineage mismatch"
                )
            try:
                charge = self.bank.restore_charge(route.charge_basis_sha256)
            except ComputeCrystalError as exc:
                raise ComputeOperatorGraphIntegrityError(
                    "materialized route compute charge restore failed"
                ) from exc
            fusion_verifier = _fusion_verifier_for_kinds(kinds)
            verification_sha = _fusion_verification_sha256(
                primitive,
                fused,
                verifier_sha256=fusion_verifier,
            )
            expected_charge = ComputeChargeReceipt.create(
                source_program=primitive,
                source_crystals=crystals,
                fused_crystal=fused,
                charge_verifier_sha256=fusion_verifier,
                verification_receipt_sha256=verification_sha,
            )
            if charge.to_bytes() != expected_charge.to_bytes():
                raise ComputeOperatorGraphIntegrityError(
                    "materialized route compute charge basis mismatch"
                )
            if (
                route.charge_verifier_sha256 != fusion_verifier
                or route.verification_receipt_sha256 != verification_sha
            ):
                raise ComputeOperatorGraphIntegrityError(
                    "materialized route fusion verifier lineage mismatch"
                )
            source_work = charge.source_work_units
            live_work = charge.live_work_units
        else:
            if (
                route.fused_crystal_sha256 is not None
                or route.charge_basis_sha256 is not None
                or route.charge_verifier_sha256 is not None
                or route.verification_receipt_sha256 is not None
                or executable != primitive
            ):
                raise ComputeOperatorGraphIntegrityError(
                    "non-fusible route has an invalid executable program"
                )
            live_work = _checked_sum(
                [crystal.discharge_work_units for crystal in crystals],
                field="route live work",
            )
        if source_work != route.equivalent_source_work_units:
            raise ComputeOperatorGraphIntegrityError(
                "materialized route source work mismatch"
            )
        if route.live_work_units != live_work:
            raise ComputeOperatorGraphIntegrityError(
                "materialized route live work mismatch"
            )
        expected_verifiers = tuple(sorted({edge.verifier_sha256 for edge in edges}))
        expected_evidence = tuple(sorted({edge.evidence_sha256 for edge in edges}))
        if (
            route.verifier_sha256s != expected_verifiers
            or route.evidence_sha256s != expected_evidence
        ):
            raise ComputeOperatorGraphIntegrityError(
                "materialized route provenance lineage mismatch"
            )
        contracting = kinds == {MARKOV_FLOAT64}
        if contracting:
            kernels = _contraction_kernels(crystals, kinds)
            expected_ledger = ContractionLedger.from_kernels(kernels)
            if route.contraction_ledger != expected_ledger:
                raise ComputeOperatorGraphIntegrityError(
                    "materialized Markov route contraction ledger mismatch"
                )
            if route.fused_crystal_sha256 is not None:
                fused = self.bank.restore_crystal(route.fused_crystal_sha256)
                (fused_kernel,) = _contraction_kernels((fused,), kinds)
                if not expected_ledger.verifies(fused_kernel):
                    raise ComputeOperatorGraphIntegrityError(
                        "fused stochastic route violates its contraction bound"
                    )
        elif route.contraction_ledger is not None:
            raise ComputeOperatorGraphIntegrityError(
                "non-contracting route carries a contraction ledger"
            )

    def _validated_state_unlocked(
        self, *, assert_trusted_head: bool = True
    ) -> ComputeOperatorGraphState:
        current, pointer_digest = self._pointer_unlocked()
        histories, commits = self._history_inventory_unlocked()
        latest = _validate_committed_graph_history(histories, commits)
        if pointer_digest is None:
            if latest.generation != 0:
                raise ComputeOperatorGraphIntegrityError(
                    "operator graph pointer was deleted or rolled back"
                )
            if assert_trusted_head:
                self._assert_trusted_anchors(current)
            return current
        if current.generation == 0:
            raise ComputeOperatorGraphIntegrityError(
                "the deterministic empty graph must not be persisted as a pointer"
            )
        historical = histories.get(current.sha256)
        if historical is None or historical.to_bytes() != current.to_bytes():
            raise ComputeOperatorGraphIntegrityError(
                "operator graph pointer lacks its exact immutable history object"
            )
        if current.sha256 not in commits:
            self._validate_extension(latest, current)
            self._publish_commit_unlocked(current)
            commits.add(current.sha256)
            latest = _validate_committed_graph_history(histories, commits)
        if latest.sha256 != current.sha256 or latest.generation != current.generation:
            raise ComputeOperatorGraphIntegrityError(
                "operator graph pointer is a validly resealed rollback"
            )
        if assert_trusted_head:
            self._assert_trusted_anchors(current)
        history = self._history_unlocked(current)
        history_by_sha = {state.sha256: state for state in history}
        edge_by_sha = {edge.sha256: edge for edge in current.edges}
        for edge in current.edges:
            try:
                self.bank.restore_crystal(edge.crystal_sha256)
            except ComputeCrystalError as exc:
                raise ComputeOperatorGraphIntegrityError(
                    "operator edge references an unpublished crystal"
                ) from exc
        for route in current.materialized_routes:
            self._validate_materialized_route(route, edge_by_sha, history_by_sha)
        return current

    def state(self) -> ComputeOperatorGraphState:
        with self._locked():
            return self._validated_state_unlocked()

    def current_anchor_sha256(self) -> str:
        """Return the last structurally verified head for external persistence.

        Immediately after this instance writes, this returns that authorized
        new head even while an external resolver still names the predecessor.
        Updating the resolver to this value makes subsequent reads pass the
        exact-head check.
        """

        if self._last_authorized_head_sha256 is not None:
            return self._last_authorized_head_sha256
        return self.state().sha256

    def _publish_history_unlocked(self, state: ComputeOperatorGraphState) -> None:
        data = state.to_bytes()
        name = self.history_state_name(state.sha256)
        try:
            existing = self._restore_raw(name)
        except KeyError:
            existing = None
        if existing is not None:
            if existing != data:
                raise ComputeOperatorGraphIntegrityError(
                    "content-addressed graph history contains different bytes"
                )
            return
        try:
            self.bank.store.publish_state(name, data)
        except ManifestConflictError as exc:
            raise ComputeOperatorGraphConflictError(
                "graph history publication conflicted"
            ) from exc
        except CrystalStoreError as exc:
            raise ComputeOperatorGraphIntegrityError(
                "graph history publication failed integrity"
            ) from exc
        if self._restore_raw(name) != data:
            raise ComputeOperatorGraphIntegrityError(
                "graph history failed immediate verification"
            )

    def _publish_commit_unlocked(self, state: ComputeOperatorGraphState) -> None:
        if state.generation == 0:
            raise ValueError("the deterministic empty graph is never committed")
        data = _graph_commit_bytes(state.sha256)
        name = self.commit_state_name(state.sha256)
        try:
            existing = self._restore_raw(name)
        except KeyError:
            existing = None
        if existing is not None:
            if existing != data:
                raise ComputeOperatorGraphIntegrityError(
                    "content-addressed graph commit contains different bytes"
                )
            return
        try:
            self.bank.store.publish_state(name, data)
        except ManifestConflictError as exc:
            raise ComputeOperatorGraphConflictError(
                "graph commit publication conflicted"
            ) from exc
        except CrystalStoreError as exc:
            raise ComputeOperatorGraphIntegrityError(
                "graph commit publication failed integrity"
            ) from exc
        if self._restore_raw(name) != data:
            raise ComputeOperatorGraphIntegrityError(
                "graph commit failed immediate verification"
            )

    def _append(
        self,
        mutation: Callable[
            [ComputeOperatorGraphState],
            tuple[tuple[OperatorEdge, ...], tuple[MaterializedRoute, ...]],
        ],
        *,
        expected_generation: int | None = None,
        expected_state_sha256: str | None = None,
    ) -> tuple[ComputeOperatorGraphState, bool]:
        if expected_generation is not None:
            expected_generation = _uint(
                expected_generation, field="expected_generation"
            )
        if expected_state_sha256 is not None:
            expected_state_sha256 = require_sha256(
                expected_state_sha256, field="expected_state_sha256"
            )
        for attempt in range(self.retry_limit):
            with self._locked():
                current = self._validated_state_unlocked()
                _logical_digest = current.sha256
                if (
                    expected_generation is not None
                    and current.generation != expected_generation
                ):
                    raise ComputeOperatorGraphConflictError(
                        f"graph generation is {current.generation}, expected {expected_generation}"
                    )
                if (
                    expected_state_sha256 is not None
                    and _logical_digest != expected_state_sha256
                ):
                    raise ComputeOperatorGraphConflictError(
                        "graph state SHA-256 compare-and-swap precondition is stale"
                    )
                edges, routes = mutation(current)
                edges = tuple(sorted(edges, key=lambda edge: edge.sha256))
                routes = tuple(sorted(routes, key=lambda route: route.sha256))
                if edges == current.edges and routes == current.materialized_routes:
                    return current, False
                updated = ComputeOperatorGraphState(
                    generation=current.generation + 1,
                    previous_state_sha256=current.sha256,
                    edges=edges,
                    materialized_routes=routes,
                )
                self._validate_extension(current, updated)
                self._publish_history_unlocked(current)
                self._publish_history_unlocked(updated)
                expected_pointer = None if current.generation == 0 else current.sha256
                try:
                    publication = self.bank.store.publish_state(
                        OPERATOR_GRAPH_STATE_NAME,
                        updated.to_bytes(),
                        expected_sha256=expected_pointer,
                    )
                except ManifestConflictError as exc:
                    if (
                        expected_generation is not None
                        or expected_state_sha256 is not None
                        or attempt + 1 == self.retry_limit
                    ):
                        raise ComputeOperatorGraphConflictError(
                            "operator graph compare-and-swap conflicted"
                        ) from exc
                    continue
                except CrystalStoreError as exc:
                    raise ComputeOperatorGraphIntegrityError(
                        "operator graph publication failed integrity"
                    ) from exc
                if not publication.changed:
                    raise ComputeOperatorGraphIntegrityError(
                        "new graph generation was not published"
                    )
                self._publish_commit_unlocked(updated)
                restored = self._validated_state_unlocked(assert_trusted_head=False)
                if restored != updated:
                    raise ComputeOperatorGraphIntegrityError(
                        "operator graph failed immediate verification"
                    )
                if self.trusted_graph_state_sha256 is not None:
                    self.trusted_graph_state_sha256 = restored.sha256
                self._last_authorized_head_sha256 = restored.sha256
                return restored, True
        raise ComputeOperatorGraphConflictError(
            "operator graph exhausted its CAS retry limit"
        )

    def append_edge(
        self,
        edge: OperatorEdge,
        *,
        expected_generation: int | None = None,
        expected_state_sha256: str | None = None,
    ) -> tuple[ComputeOperatorGraphState, bool]:
        if not isinstance(edge, OperatorEdge):
            raise TypeError("edge must be an OperatorEdge")
        self.bank.restore_crystal(edge.crystal_sha256)

        def mutation(
            state: ComputeOperatorGraphState,
        ) -> tuple[tuple[OperatorEdge, ...], tuple[MaterializedRoute, ...]]:
            by_sha = {item.sha256: item for item in state.edges}
            existing = by_sha.get(edge.sha256)
            if existing is not None:
                if existing != edge:
                    raise ComputeOperatorGraphIntegrityError(
                        "edge content address collision"
                    )
                return state.edges, state.materialized_routes
            if any(
                item.evidence_sha256 == edge.evidence_sha256 for item in state.edges
            ):
                raise ValueError("one evidence record cannot authenticate two edges")
            return (*state.edges, edge), state.materialized_routes

        return self._append(
            mutation,
            expected_generation=expected_generation,
            expected_state_sha256=expected_state_sha256,
        )

    add_edge = append_edge

    def append_edges(
        self,
        edges: Sequence[OperatorEdge],
        *,
        expected_generation: int | None = None,
        expected_state_sha256: str | None = None,
    ) -> tuple[ComputeOperatorGraphState, bool]:
        if isinstance(edges, (str, bytes)) or not isinstance(edges, Sequence):
            raise TypeError("edges must be a sequence")
        additions = tuple(edges)
        if not additions:
            raise ValueError("edges must not be empty")
        if any(not isinstance(edge, OperatorEdge) for edge in additions):
            raise TypeError("edges must contain OperatorEdge values")
        if len({edge.sha256 for edge in additions}) != len(additions):
            raise ValueError("edge batch contains duplicates")
        if len({edge.evidence_sha256 for edge in additions}) != len(additions):
            raise ValueError("edge batch reuses evidence")
        for edge in additions:
            self.bank.restore_crystal(edge.crystal_sha256)

        def mutation(
            state: ComputeOperatorGraphState,
        ) -> tuple[tuple[OperatorEdge, ...], tuple[MaterializedRoute, ...]]:
            current_by_sha = {edge.sha256: edge for edge in state.edges}
            evidence = {edge.evidence_sha256: edge.sha256 for edge in state.edges}
            result = list(state.edges)
            for edge in additions:
                if edge.sha256 in current_by_sha:
                    continue
                collision = evidence.get(edge.evidence_sha256)
                if collision is not None and collision != edge.sha256:
                    raise ValueError(
                        "one evidence record cannot authenticate two edges"
                    )
                result.append(edge)
                evidence[edge.evidence_sha256] = edge.sha256
            return tuple(result), state.materialized_routes

        return self._append(
            mutation,
            expected_generation=expected_generation,
            expected_state_sha256=expected_state_sha256,
        )

    def append_materialized_route(
        self,
        route: MaterializedRoute,
        *,
        expected_generation: int | None = None,
        expected_state_sha256: str | None = None,
    ) -> tuple[ComputeOperatorGraphState, bool]:
        if not isinstance(route, MaterializedRoute):
            raise TypeError("route must be a MaterializedRoute")

        def mutation(
            state: ComputeOperatorGraphState,
        ) -> tuple[tuple[OperatorEdge, ...], tuple[MaterializedRoute, ...]]:
            if route.sha256 in {item.sha256 for item in state.materialized_routes}:
                return state.edges, state.materialized_routes
            history = self._history_unlocked(state) if state.generation else (state,)
            self._validate_materialized_route(
                route,
                {edge.sha256: edge for edge in state.edges},
                {item.sha256: item for item in history},
            )
            return state.edges, (*state.materialized_routes, route)

        return self._append(
            mutation,
            expected_generation=expected_generation,
            expected_state_sha256=expected_state_sha256,
        )

    def _world_model(
        self, state: ComputeOperatorGraphState
    ) -> ActionConditionedWorldModel:
        if not state.edges:
            raise ComputeRouteUnavailableError(
                "operator graph has no one-step evidence"
            )
        states = tuple(
            sorted(
                {
                    label
                    for edge in state.edges
                    for label in (edge.source_state, edge.target_state)
                }
            )
        )
        actions = tuple(edge.sha256 for edge in state.edges)
        required_bytes = 16 * len(actions) * len(states) * len(states)
        model = ActionConditionedWorldModel(
            states,
            actions,
            min_evidence_mass=1.0,
            max_normalized_entropy=0.0,
            min_peak_probability=1.0,
            max_states=max(2, len(states)),
            max_actions=max(1, len(actions)),
            max_bytes=max(16, required_bytes),
            max_provenance=max(1, len(actions)),
        )
        model.observe_many(
            TransitionEvidence(
                source_state=edge.source_state,
                action=edge.sha256,
                target_state=edge.target_state,
                weight=edge.weight,
                verifier_sha256=edge.verifier_sha256,
                evidence_sha256=edge.evidence_sha256,
            )
            for edge in state.edges
        )
        return model

    def build_world_model(self) -> ActionConditionedWorldModel:
        state = self.state()
        return self._world_model(state)

    def plan_route(
        self,
        source_state: str,
        goal_state: str,
        *,
        horizon: int | None = None,
    ) -> ComputeRoutePlanDecision:
        source = _text(source_state, field="source_state")
        goal = _text(goal_state, field="goal_state")
        state = self.state()
        if not state.edges:
            return ComputeRoutePlanDecision(
                plan=None,
                abstained=True,
                reason="no-one-step-evidence",
                graph_state_sha256=state.sha256,
                world_model_sha256=None,
                predicted_success=0.0,
            )
        world = self._world_model(state)
        if source not in world.states:
            return ComputeRoutePlanDecision(
                plan=None,
                abstained=True,
                reason="unknown-source-state",
                graph_state_sha256=state.sha256,
                world_model_sha256=world.sha256,
                predicted_success=0.0,
            )
        if goal not in world.states:
            return ComputeRoutePlanDecision(
                plan=None,
                abstained=True,
                reason="unknown-goal-state",
                graph_state_sha256=state.sha256,
                world_model_sha256=world.sha256,
                predicted_success=0.0,
            )
        if source == goal:
            return ComputeRoutePlanDecision(
                plan=None,
                abstained=True,
                reason="source-already-goal",
                graph_state_sha256=state.sha256,
                world_model_sha256=world.sha256,
                predicted_success=1.0,
            )
        steps = (
            max(1, len(world.states) - 1)
            if horizon is None
            else _uint(horizon, field="horizon")
        )
        if steps > MAX_PROGRAM_STEPS:
            raise ValueError("horizon exceeds the compute-program step bound")
        planner = FiniteHorizonPlanner(
            world,
            min_evidence_mass=1.0,
            max_normalized_entropy=0.0,
            min_peak_probability=1.0,
            min_predicted_success=1.0,
            max_horizon=MAX_PROGRAM_STEPS,
            max_policy_entries=max(1, MAX_PROGRAM_STEPS * len(world.states)),
        )
        decision: PlanDecision = planner.plan_goal(source, goal, horizon=steps)
        if decision.abstained or decision.plan is None:
            return ComputeRoutePlanDecision(
                plan=None,
                abstained=True,
                reason=decision.reason or "unreachable-or-uncovered",
                graph_state_sha256=state.sha256,
                world_model_sha256=world.sha256,
                predicted_success=decision.predicted_success,
            )
        finite = decision.plan
        edge_by_sha = {edge.sha256: edge for edge in state.edges}
        if not finite.expected_actions or finite.expected_states[-1] != goal:
            return ComputeRoutePlanDecision(
                plan=None,
                abstained=True,
                reason="planner-path-does-not-reach-goal",
                graph_state_sha256=state.sha256,
                world_model_sha256=world.sha256,
                predicted_success=decision.predicted_success,
            )
        for index, action in enumerate(finite.expected_actions):
            edge = edge_by_sha.get(action)
            if edge is None or (
                edge.source_state != finite.expected_states[index]
                or edge.target_state != finite.expected_states[index + 1]
            ):
                raise ComputeOperatorGraphIntegrityError(
                    "planner action identity disagrees with operator edge topology"
                )
        route_plan = ComputeRoutePlan(
            graph_generation=state.generation,
            graph_state_sha256=state.sha256,
            finite_plan=finite,
            primitive_edge_sha256s=finite.expected_actions,
        )
        return ComputeRoutePlanDecision(
            plan=route_plan,
            abstained=False,
            reason=None,
            graph_state_sha256=state.sha256,
            world_model_sha256=world.sha256,
            predicted_success=decision.predicted_success,
        )

    def plan_exact_path(
        self,
        primitive_edge_sha256s: Sequence[str],
    ) -> ComputeRoutePlan:
        """Bind one caller-selected ordered edge path to the current graph head.

        Unlike :meth:`plan_route`, this does not choose among alternatives.  It
        authenticates and reconstructs the exact requested trajectory, while
        retaining the same world-model, count, kernel, gate, and provenance
        contract used by every other compute route.
        """

        addresses = _hash_tuple(
            primitive_edge_sha256s,
            field="primitive_edge_sha256s",
        )
        if not addresses:
            raise ValueError("exact operator path must not be empty")
        if len(addresses) > MAX_PROGRAM_STEPS:
            raise ValueError("exact operator path exceeds the program step bound")
        with self._locked():
            state = self._validated_state_unlocked()
            edge_by_sha = {edge.sha256: edge for edge in state.edges}
            try:
                edges = tuple(edge_by_sha[address] for address in addresses)
            except KeyError as exc:
                raise ComputeRouteUnavailableError(
                    "exact operator path references an edge absent from the current graph head"
                ) from exc
            for index, edge in enumerate(edges[1:], start=1):
                if edges[index - 1].target_state != edge.source_state:
                    raise ComputeRouteUnavailableError(
                        f"exact operator path is disconnected at edge {index}"
                    )
            crystals = tuple(
                self.bank.restore_crystal(edge.crystal_sha256) for edge in edges
            )
            # This is a planning-time gate for an explicit path: no graph route is
            # promised if its restored numerical operators cannot execute in order.
            ComputeProgram.compose(crystals)
            world = self._world_model(state)
            finite = self._exact_finite_plan(world, edges)
            return ComputeRoutePlan(
                graph_generation=state.generation,
                graph_state_sha256=state.sha256,
                finite_plan=finite,
                primitive_edge_sha256s=addresses,
            )

    @staticmethod
    def _publication_created(publication: ComputeBankPublication) -> bool:
        return publication.object_created or publication.manifest_changed

    def charge_route(self, plan: ComputeRoutePlan) -> RouteChargeReceipt:
        if not isinstance(plan, ComputeRoutePlan):
            raise TypeError("plan must be a ComputeRoutePlan")
        with self._locked():
            current = self._validated_state_unlocked()
            history = self._history_unlocked(current)
            edges = self._validated_plan_edges(
                planning_graph_generation=plan.graph_generation,
                planning_graph_state_sha256=plan.graph_state_sha256,
                finite=plan.finite_plan,
                primitive_edge_sha256s=plan.primitive_edge_sha256s,
                source_state=plan.source_state,
                goal_state=plan.goal_state,
                edge_by_sha={edge.sha256: edge for edge in current.edges},
                history_by_sha={state.sha256: state for state in history},
            )
        existing = next(
            (
                route
                for route in current.materialized_routes
                if route.finite_horizon_plan_sha256 == plan.finite_plan.sha256
                and route.planning_graph_generation == plan.graph_generation
                and route.planning_graph_state_sha256 == plan.graph_state_sha256
                and route.primitive_edge_sha256s == plan.primitive_edge_sha256s
            ),
            None,
        )
        if existing is not None:
            return RouteChargeReceipt(
                route=existing,
                graph_generation=current.generation,
                graph_state_sha256=current.sha256,
                graph_changed=False,
                primitive_program_created=False,
                fused_crystal_created=False,
                executable_program_created=False,
                charge_basis_created=False,
                composition_work_units=0,
            )
        crystals = tuple(
            self.bank.restore_crystal(edge.crystal_sha256) for edge in edges
        )
        # Compose before publication so a mixed ABI fails without graph mutation.
        primitive_program = ComputeProgram.compose(crystals)
        primitive_publication = self.bank.publish_program(primitive_program)
        primitive_live = _checked_sum(
            [crystal.discharge_work_units for crystal in crystals],
            field="route primitive live work",
        )
        kinds = {crystal.operator_kind for crystal in crystals}
        fused: ComputeCrystal | None = None
        fused_publication: ComputeBankPublication | None = None
        executable_program = primitive_program
        executable_publication = primitive_publication
        charge: ComputeChargeReceipt | None = None
        charge_publication: ComputeBankPublication | None = None
        verification_receipt_sha256: str | None = None
        composition_work = 0
        if len(crystals) >= 2 and kinds in (
            {AFFINE_FLOAT64},
            {PERMUTATION},
            {MARKOV_FLOAT64},
            {CAUSAL_MIX_FLOAT64},
        ):
            fused = fuse_compatible_chain(
                crystals,
                extensions={
                    "compute_operator_graph": {
                        "finite_horizon_plan_sha256": plan.finite_plan.sha256,
                        "primitive_edge_path_sha256": _sha256(
                            list(plan.primitive_edge_sha256s)
                        ),
                    }
                },
            )
            fused_publication = self.bank.publish_crystal(fused)
            executable_program = ComputeProgram.compose((fused,))
            executable_publication = self.bank.publish_program(executable_program)
            fusion_verifier = _fusion_verifier_for_kinds(kinds)
            verification_receipt_sha256 = _fusion_verification_sha256(
                primitive_program,
                fused,
                verifier_sha256=fusion_verifier,
            )
            charge = ComputeChargeReceipt.create(
                source_program=primitive_program,
                source_crystals=crystals,
                fused_crystal=fused,
                charge_verifier_sha256=fusion_verifier,
                verification_receipt_sha256=verification_receipt_sha256,
            )
            charge_publication = self.bank.publish_charge(charge)
            composition_work = primitive_live
            equivalent_work = charge.source_work_units
            live_work = charge.live_work_units
        else:
            equivalent_work = primitive_live
            live_work = primitive_live
        ledger: ContractionLedger | None = None
        if kinds == {MARKOV_FLOAT64}:
            kernels = _contraction_kernels(crystals, kinds)
            ledger = ContractionLedger.from_kernels(kernels)
            if fused is not None:
                (fused_kernel,) = _contraction_kernels((fused,), kinds)
                if not ledger.verifies(fused_kernel):
                    raise ComputeOperatorGraphIntegrityError(
                        "fused stochastic operator violates the contraction ledger"
                    )
        route = MaterializedRoute(
            source_state=plan.source_state,
            goal_state=plan.goal_state,
            primitive_edge_sha256s=plan.primitive_edge_sha256s,
            planning_graph_generation=plan.graph_generation,
            planning_graph_state_sha256=plan.graph_state_sha256,
            finite_horizon_plan=plan.finite_plan,
            finite_horizon_plan_sha256=plan.finite_plan.sha256,
            world_model_sha256=plan.world_model_sha256,
            primitive_program_sha256=primitive_program.sha256,
            executable_program_sha256=executable_program.sha256,
            fused_crystal_sha256=None if fused is None else fused.sha256,
            charge_basis_sha256=None if charge is None else charge.sha256,
            charge_verifier_sha256=(
                None if charge is None else charge.charge_verifier_sha256
            ),
            verification_receipt_sha256=verification_receipt_sha256,
            input_abi_sha256=primitive_program.input_abi.sha256,
            output_abi_sha256=primitive_program.output_abi.sha256,
            equivalent_source_work_units=equivalent_work,
            live_work_units=live_work,
            contraction_ledger=ledger,
            verifier_sha256s=tuple(sorted({edge.verifier_sha256 for edge in edges})),
            evidence_sha256s=tuple(sorted({edge.evidence_sha256 for edge in edges})),
        )
        updated, changed = self.append_materialized_route(route)
        actual = next(
            item for item in updated.materialized_routes if item.sha256 == route.sha256
        )
        return RouteChargeReceipt(
            route=actual,
            graph_generation=updated.generation,
            graph_state_sha256=updated.sha256,
            graph_changed=changed,
            primitive_program_created=self._publication_created(primitive_publication),
            fused_crystal_created=(
                False
                if fused_publication is None
                else self._publication_created(fused_publication)
            ),
            executable_program_created=(
                False
                if executable_program.sha256 == primitive_program.sha256
                else self._publication_created(executable_publication)
            ),
            charge_basis_created=(
                False
                if charge_publication is None
                else self._publication_created(charge_publication)
            ),
            composition_work_units=composition_work,
        )

    materialize_route = charge_route

    def best_route(self, source_state: str, goal_state: str) -> MaterializedRoute:
        source = _text(source_state, field="source_state")
        goal = _text(goal_state, field="goal_state")
        candidates = tuple(
            route
            for route in self.state().materialized_routes
            if route.source_state == source and route.goal_state == goal
        )
        if not candidates:
            raise ComputeRouteUnavailableError(
                f"no materialized route from {source!r} to {goal!r}"
            )
        return min(
            candidates,
            key=lambda route: (
                route.live_work_units,
                0 if route.fused_crystal_sha256 is not None else 1,
                len(route.primitive_edge_sha256s),
                route.sha256,
            ),
        )

    def discharge(
        self,
        source_state: str,
        goal_state: str,
        value: object,
    ) -> RouteDischarge:
        state = self.state()
        candidates = tuple(
            route
            for route in state.materialized_routes
            if route.source_state == source_state and route.goal_state == goal_state
        )
        if not candidates:
            raise ComputeRouteUnavailableError(
                f"no materialized route from {source_state!r} to {goal_state!r}"
            )
        compatible: list[tuple[MaterializedRoute, ComputeProgram, int]] = []
        for candidate in candidates:
            candidate_program = self.bank.restore_program(
                candidate.executable_program_sha256
            )
            try:
                candidate_applications = candidate_program.input_abi.application_count(
                    value
                )
            except ComputeCrystalABIError:
                continue
            compatible.append((candidate, candidate_program, candidate_applications))
        if not compatible:
            raise ComputeCrystalABIError(
                "no materialized semantic route accepts the supplied numerical ABI"
            )
        route, program, applications = min(
            compatible,
            key=lambda item: (
                item[0].live_work_units * item[2],
                0 if item[0].fused_crystal_sha256 is not None else 1,
                len(item[0].primitive_edge_sha256s),
                item[0].sha256,
            ),
        )
        return self._execute_materialized_route(
            state,
            route,
            program,
            applications,
            value,
        )

    def discharge_exact(
        self,
        route_sha256: str,
        value: object,
    ) -> RouteDischarge:
        """Execute exactly one authenticated materialized route by address."""

        address = require_sha256(route_sha256, field="route_sha256")
        state = self.state()
        route = next(
            (
                candidate
                for candidate in state.materialized_routes
                if candidate.sha256 == address
            ),
            None,
        )
        if route is None:
            raise ComputeRouteUnavailableError(
                "exact materialized route is absent from the current graph head"
            )
        program = self.bank.restore_program(route.executable_program_sha256)
        applications = program.input_abi.application_count(value)
        return self._execute_materialized_route(
            state,
            route,
            program,
            applications,
            value,
        )

    def _execute_materialized_route(
        self,
        state: ComputeOperatorGraphState,
        route: MaterializedRoute,
        program: ComputeProgram,
        applications: int,
        value: object,
    ) -> RouteDischarge:
        execution: ComputeExecution = ComputeCrystalVM(self.bank).execute(
            program,
            value,
            charge_basis_sha256=route.charge_basis_sha256,
        )
        equivalent = route.equivalent_source_work_units * applications
        live = route.live_work_units * applications
        if equivalent > MAX_WORK_UNITS or live > MAX_WORK_UNITS:
            raise ValueError("route discharge work exceeds its bound")
        receipt = RouteDischargeReceipt(
            route_sha256=route.sha256,
            finite_horizon_plan_sha256=route.finite_horizon_plan_sha256,
            world_model_sha256=route.world_model_sha256,
            planning_graph_state_sha256=route.planning_graph_state_sha256,
            graph_generation=state.generation,
            graph_state_sha256=state.sha256,
            primitive_program_sha256=route.primitive_program_sha256,
            executable_program_sha256=route.executable_program_sha256,
            fused_crystal_sha256=route.fused_crystal_sha256,
            charge_basis_sha256=route.charge_basis_sha256,
            vm_receipt=execution.receipt,
            application_count=applications,
            equivalent_source_work_units=equivalent,
            live_work_units=live,
            historical_work_released=max(0, equivalent - live),
            verifier_sha256s=route.verifier_sha256s,
            evidence_sha256s=route.evidence_sha256s,
            contraction_ledger_sha256=(
                None
                if route.contraction_ledger is None
                else route.contraction_ledger.sha256
            ),
        )
        return RouteDischarge(output=execution.output, receipt=receipt, route=route)

    query = discharge
    query_exact = discharge_exact


__all__ = [
    "ComputeOperatorGraph",
    "ComputeOperatorGraphConflictError",
    "ComputeOperatorGraphError",
    "ComputeOperatorGraphIntegrityError",
    "ComputeOperatorGraphState",
    "ComputeRoutePlan",
    "ComputeRoutePlanDecision",
    "ComputeRouteUnavailableError",
    "MaterializedRoute",
    "OperatorEdge",
    "RouteChargeReceipt",
    "RouteDischarge",
    "RouteDischargeReceipt",
]
