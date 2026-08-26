"""Execute the longest charged compute prefix and only calculate its live suffix.

``ComputeOperatorGraph`` can materialize any previously discovered sub-route as
one fused ``ComputeCrystal``.  This module turns those independent charges into
partial computation: a future, longer route reuses the deepest matching
materialized prefix and executes only the unseen remainder.

The numerical input is not part of the charge.  A charged affine, permutation,
or Markov prefix therefore applies to arbitrary future values with the same ABI.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .compute_crystals import (
    MAX_PROGRAM_STEPS,
    ComputeCrystalABIError,
    ComputeCrystalIntegrityError,
    ComputeCrystalVM,
    ComputeExecutionReceipt,
    ComputeProgram,
    tensor_sha256,
)
from .compute_graph import (
    ComputeOperatorGraph,
    ComputeOperatorGraphIntegrityError,
    ComputeOperatorGraphState,
    ComputeRoutePlan,
    MaterializedRoute,
    OperatorEdge,
)
from .identity import canonical_json_bytes, require_sha256


RESIDUAL_ROUTE_DISCHARGE_SCHEMA = "immer-ooe-residual-route-discharge/v1"
MAX_RESIDUAL_RECEIPT_BYTES = 4 * 1024 * 1024
MAX_WORK_UNITS = (1 << 63) - 1


class ResidualExecutionError(RuntimeError):
    """Base error for charged-prefix plus live-suffix execution."""


class ResidualExecutionIntegrityError(ResidualExecutionError):
    """A plan, materialized prefix, or receipt failed authentication."""


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _uint(value: object, *, field: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field} must be an integer")
    result = int(value)
    minimum = 1 if positive else 0
    if not minimum <= result <= MAX_WORK_UNITS:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must be a bounded {qualifier} integer")
    return result


def _checked_sum(values: Sequence[int], *, field: str) -> int:
    total = 0
    for value in values:
        total += _uint(value, field=field)
        if total > MAX_WORK_UNITS:
            raise ValueError(f"{field} exceeds its bound")
    return total


def _hashes(values: Sequence[str], *, field: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field} must be a sequence")
    return tuple(require_sha256(value, field=field) for value in values)


def _sorted_hashes(values: Sequence[str], *, field: str) -> tuple[str, ...]:
    result = _hashes(values, field=field)
    if tuple(sorted(set(result))) != result:
        raise ValueError(f"{field} must be sorted and unique")
    return result


def _strict_json(data: bytes) -> object:
    if not isinstance(data, bytes):
        raise TypeError("residual receipt must be immutable bytes")
    if len(data) > MAX_RESIDUAL_RECEIPT_BYTES:
        raise ResidualExecutionIntegrityError("residual receipt exceeds its byte bound")

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
        raise ResidualExecutionIntegrityError(
            "residual receipt is not strict JSON"
        ) from exc
    if canonical_json_bytes(value) != data:
        raise ResidualExecutionIntegrityError(
            "residual receipt is not canonical JSON"
        )
    return value


def _parse_vm_receipt(value: object, *, field: str) -> ComputeExecutionReceipt | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ResidualExecutionIntegrityError(f"{field} is not a receipt document")
    try:
        return ComputeExecutionReceipt.from_bytes(canonical_json_bytes(value))
    except ComputeCrystalIntegrityError as exc:
        raise ResidualExecutionIntegrityError(f"{field} failed authentication") from exc


@dataclass(frozen=True, slots=True)
class ResidualRouteDischargeReceipt:
    """Authenticated accounting for a charged prefix plus a live suffix."""

    plan: ComputeRoutePlan
    graph_generation: int
    graph_state_sha256: str
    prefix_route_sha256: str | None
    prefix_edge_sha256s: tuple[str, ...]
    suffix_edge_sha256s: tuple[str, ...]
    prefix_execution_receipt: ComputeExecutionReceipt | None
    suffix_execution_receipt: ComputeExecutionReceipt | None
    input_sha256: str
    output_sha256: str
    equivalent_source_work_units: int
    live_work_units: int
    historical_work_released: int
    verifier_sha256s: tuple[str, ...]
    evidence_sha256s: tuple[str, ...]

    FORMAT = RESIDUAL_ROUTE_DISCHARGE_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ComputeRoutePlan):
            raise TypeError("plan must be a ComputeRoutePlan")
        generation = _uint(self.graph_generation, field="graph_generation")
        graph_sha = require_sha256(
            self.graph_state_sha256, field="graph_state_sha256"
        )
        if (
            generation != self.plan.graph_generation
            or graph_sha != self.plan.graph_state_sha256
        ):
            raise ValueError("residual receipt graph head differs from its plan")
        prefix = _hashes(self.prefix_edge_sha256s, field="prefix_edge_sha256s")
        suffix = _hashes(self.suffix_edge_sha256s, field="suffix_edge_sha256s")
        if not prefix and not suffix:
            raise ValueError("residual execution needs a non-empty route")
        if len(prefix) + len(suffix) > MAX_PROGRAM_STEPS:
            raise ValueError("residual route exceeds the compute-program step bound")
        if (*prefix, *suffix) != self.plan.primitive_edge_sha256s:
            raise ValueError("residual prefix and suffix do not reconstruct the plan")

        prefix_route = self.prefix_route_sha256
        if prefix_route is not None:
            prefix_route = require_sha256(prefix_route, field="prefix_route_sha256")
        prefix_receipt = self.prefix_execution_receipt
        suffix_receipt = self.suffix_execution_receipt
        if bool(prefix) != (prefix_route is not None and prefix_receipt is not None):
            raise ValueError("charged prefix identity and execution receipt are atomic")
        if not prefix and (prefix_route is not None or prefix_receipt is not None):
            raise ValueError("an empty prefix cannot carry charged execution state")
        if bool(suffix) != (suffix_receipt is not None):
            raise ValueError("live suffix edges and execution receipt are atomic")
        if prefix_receipt is not None and not isinstance(
            prefix_receipt, ComputeExecutionReceipt
        ):
            raise TypeError("prefix_execution_receipt has the wrong type")
        if suffix_receipt is not None and not isinstance(
            suffix_receipt, ComputeExecutionReceipt
        ):
            raise TypeError("suffix_execution_receipt has the wrong type")
        if suffix_receipt is not None and suffix_receipt.charge_basis_sha256 is not None:
            raise ValueError("the residual suffix must report only live computation")

        input_sha = require_sha256(self.input_sha256, field="input_sha256")
        output_sha = require_sha256(self.output_sha256, field="output_sha256")
        if prefix_receipt is not None:
            if prefix_receipt.input_sha256 != input_sha:
                raise ValueError("prefix receipt input differs from residual input")
            if suffix_receipt is not None:
                if prefix_receipt.output_sha256 != suffix_receipt.input_sha256:
                    raise ValueError("prefix output and suffix input do not join")
            elif prefix_receipt.output_sha256 != output_sha:
                raise ValueError("prefix-only output differs from residual output")
        elif suffix_receipt is not None and suffix_receipt.input_sha256 != input_sha:
            raise ValueError("suffix receipt input differs from residual input")
        if suffix_receipt is not None and suffix_receipt.output_sha256 != output_sha:
            raise ValueError("suffix output differs from residual output")

        receipts = tuple(
            receipt
            for receipt in (prefix_receipt, suffix_receipt)
            if receipt is not None
        )
        source = _checked_sum(
            [receipt.equivalent_unfused_source_work for receipt in receipts],
            field="equivalent_source_work_units",
        )
        live = _checked_sum(
            [receipt.live_discharge_work for receipt in receipts],
            field="live_work_units",
        )
        released = _checked_sum(
            [receipt.historical_work_released for receipt in receipts],
            field="historical_work_released",
        )
        if (
            source != _uint(
                self.equivalent_source_work_units,
                field="equivalent_source_work_units",
                positive=True,
            )
            or live
            != _uint(self.live_work_units, field="live_work_units", positive=True)
            or released
            != _uint(
                self.historical_work_released,
                field="historical_work_released",
            )
            or released != max(0, source - live)
        ):
            raise ValueError("residual work accounting disagrees with VM receipts")

        object.__setattr__(self, "graph_generation", generation)
        object.__setattr__(self, "graph_state_sha256", graph_sha)
        object.__setattr__(self, "prefix_route_sha256", prefix_route)
        object.__setattr__(self, "prefix_edge_sha256s", prefix)
        object.__setattr__(self, "suffix_edge_sha256s", suffix)
        object.__setattr__(self, "input_sha256", input_sha)
        object.__setattr__(self, "output_sha256", output_sha)
        object.__setattr__(self, "equivalent_source_work_units", source)
        object.__setattr__(self, "live_work_units", live)
        object.__setattr__(self, "historical_work_released", released)
        object.__setattr__(
            self,
            "verifier_sha256s",
            _sorted_hashes(self.verifier_sha256s, field="verifier_sha256s"),
        )
        object.__setattr__(
            self,
            "evidence_sha256s",
            _sorted_hashes(self.evidence_sha256s, field="evidence_sha256s"),
        )

    @property
    def prefix_length(self) -> int:
        return len(self.prefix_edge_sha256s)

    @property
    def residual_length(self) -> int:
        return len(self.suffix_edge_sha256s)

    @property
    def reused_past_compute(self) -> bool:
        return self.historical_work_released > 0

    def to_dict(self) -> dict[str, object]:
        prefix = self.prefix_execution_receipt
        suffix = self.suffix_execution_receipt
        return {
            "schema": self.FORMAT,
            "plan": self.plan.to_dict(),
            "plan_sha256": self.plan.sha256,
            "graph_generation": self.graph_generation,
            "graph_state_sha256": self.graph_state_sha256,
            "prefix_route_sha256": self.prefix_route_sha256,
            "prefix_edge_sha256s": list(self.prefix_edge_sha256s),
            "suffix_edge_sha256s": list(self.suffix_edge_sha256s),
            "prefix_execution_receipt": (
                None if prefix is None else prefix.to_document()
            ),
            "prefix_execution_receipt_sha256": (
                None if prefix is None else prefix.sha256
            ),
            "suffix_execution_receipt": (
                None if suffix is None else suffix.to_document()
            ),
            "suffix_execution_receipt_sha256": (
                None if suffix is None else suffix.sha256
            ),
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
            "equivalent_source_work_units": self.equivalent_source_work_units,
            "live_work_units": self.live_work_units,
            "historical_work_released": self.historical_work_released,
            "verifier_sha256s": list(self.verifier_sha256s),
            "evidence_sha256s": list(self.evidence_sha256s),
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_dict())
        if len(data) > MAX_RESIDUAL_RECEIPT_BYTES:
            raise ValueError("residual receipt exceeds its byte bound")
        return data

    @classmethod
    def from_bytes(cls, data: bytes) -> "ResidualRouteDischargeReceipt":
        value = _strict_json(data)
        expected = {
            "schema",
            "plan",
            "plan_sha256",
            "graph_generation",
            "graph_state_sha256",
            "prefix_route_sha256",
            "prefix_edge_sha256s",
            "suffix_edge_sha256s",
            "prefix_execution_receipt",
            "prefix_execution_receipt_sha256",
            "suffix_execution_receipt",
            "suffix_execution_receipt_sha256",
            "input_sha256",
            "output_sha256",
            "equivalent_source_work_units",
            "live_work_units",
            "historical_work_released",
            "verifier_sha256s",
            "evidence_sha256s",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or value.get("schema") != cls.FORMAT
        ):
            raise ResidualExecutionIntegrityError("invalid residual receipt envelope")
        plan_document = value.get("plan")
        prefix_edges = value.get("prefix_edge_sha256s")
        suffix_edges = value.get("suffix_edge_sha256s")
        verifiers = value.get("verifier_sha256s")
        evidence = value.get("evidence_sha256s")
        if not all(
            isinstance(item, list)
            for item in (prefix_edges, suffix_edges, verifiers, evidence)
        ) or not isinstance(plan_document, Mapping):
            raise ResidualExecutionIntegrityError(
                "invalid residual receipt collections"
            )
        try:
            plan = ComputeRoutePlan.from_bytes(canonical_json_bytes(plan_document))
            plan_sha = require_sha256(value.get("plan_sha256"), field="plan_sha256")
            prefix = _parse_vm_receipt(
                value.get("prefix_execution_receipt"),
                field="prefix_execution_receipt",
            )
            suffix = _parse_vm_receipt(
                value.get("suffix_execution_receipt"),
                field="suffix_execution_receipt",
            )
            prefix_sha_raw = value.get("prefix_execution_receipt_sha256")
            suffix_sha_raw = value.get("suffix_execution_receipt_sha256")
            prefix_sha = (
                None
                if prefix_sha_raw is None
                else require_sha256(prefix_sha_raw, field="prefix receipt SHA-256")
            )
            suffix_sha = (
                None
                if suffix_sha_raw is None
                else require_sha256(suffix_sha_raw, field="suffix receipt SHA-256")
            )
            receipt = cls(
                plan=plan,
                graph_generation=cast(int, value.get("graph_generation")),
                graph_state_sha256=cast(str, value.get("graph_state_sha256")),
                prefix_route_sha256=cast(
                    str | None, value.get("prefix_route_sha256")
                ),
                prefix_edge_sha256s=tuple(cast(list[str], prefix_edges)),
                suffix_edge_sha256s=tuple(cast(list[str], suffix_edges)),
                prefix_execution_receipt=prefix,
                suffix_execution_receipt=suffix,
                input_sha256=cast(str, value.get("input_sha256")),
                output_sha256=cast(str, value.get("output_sha256")),
                equivalent_source_work_units=cast(
                    int, value.get("equivalent_source_work_units")
                ),
                live_work_units=cast(int, value.get("live_work_units")),
                historical_work_released=cast(
                    int, value.get("historical_work_released")
                ),
                verifier_sha256s=tuple(cast(list[str], verifiers)),
                evidence_sha256s=tuple(cast(list[str], evidence)),
            )
        except (
            ComputeOperatorGraphIntegrityError,
            ComputeCrystalIntegrityError,
            TypeError,
            ValueError,
        ) as exc:
            raise ResidualExecutionIntegrityError(
                "residual receipt validation failed"
            ) from exc
        if (
            plan_sha != plan.sha256
            or prefix_sha != (None if prefix is None else prefix.sha256)
            or suffix_sha != (None if suffix is None else suffix.sha256)
            or receipt.to_bytes() != data
        ):
            raise ResidualExecutionIntegrityError(
                "residual receipt failed canonical reconstruction"
            )
        return receipt


@dataclass(frozen=True, slots=True)
class ResidualRouteExecution:
    output: NDArray[Any]
    receipt: ResidualRouteDischargeReceipt
    prefix_route: MaterializedRoute | None


class ResidualRouteExecutor:
    """Reuse the deepest materialized prefix of an authenticated route plan."""

    def __init__(self, graph: ComputeOperatorGraph) -> None:
        if not isinstance(graph, ComputeOperatorGraph):
            raise TypeError("graph must be a ComputeOperatorGraph")
        self.graph = graph

    def _validate_current_plan(
        self, plan: ComputeRoutePlan
    ) -> ComputeOperatorGraphState:
        if not isinstance(plan, ComputeRoutePlan):
            raise TypeError("plan must be a ComputeRoutePlan")
        state = self.graph.state()
        if (
            state.generation != plan.graph_generation
            or state.sha256 != plan.graph_state_sha256
        ):
            raise ResidualExecutionIntegrityError(
                "route plan is stale for the current append-only graph head"
            )
        decision = self.graph.plan_route(
            plan.source_state,
            plan.goal_state,
            horizon=plan.finite_plan.horizon,
        )
        if (
            decision.abstained
            or decision.plan is None
            or decision.plan.to_bytes() != plan.to_bytes()
        ):
            raise ResidualExecutionIntegrityError(
                "route plan is not reproducible from the current graph"
            )
        return state

    @staticmethod
    def _matches_prefix(route: MaterializedRoute, plan: ComputeRoutePlan) -> bool:
        length = len(route.primitive_edge_sha256s)
        return bool(
            0 < length <= len(plan.primitive_edge_sha256s)
            and route.source_state == plan.source_state
            and route.goal_state == plan.finite_plan.expected_states[length]
            and route.primitive_edge_sha256s
            == plan.primitive_edge_sha256s[:length]
        )

    def execute(
        self,
        plan: ComputeRoutePlan,
        value: object,
        *,
        prefix_route_sha256: str | None = None,
    ) -> ResidualRouteExecution:
        state = self._validate_current_plan(plan)
        requested_prefix = (
            None
            if prefix_route_sha256 is None
            else require_sha256(
                prefix_route_sha256, field="prefix_route_sha256"
            )
        )
        edge_by_sha = {edge.sha256: edge for edge in state.edges}
        edges = tuple(edge_by_sha.get(address) for address in plan.primitive_edge_sha256s)
        if any(edge is None for edge in edges):
            raise ResidualExecutionIntegrityError(
                "route plan references an absent operator edge"
            )
        exact_edges = tuple(cast(OperatorEdge, edge) for edge in edges)
        crystals = tuple(
            self.graph.bank.restore_crystal(edge.crystal_sha256)
            for edge in exact_edges
        )
        primitive = ComputeProgram.compose(crystals)
        primitive.input_abi.validate(value, field="residual route input")

        compatible: list[tuple[MaterializedRoute, int, int]] = []
        for route in state.materialized_routes:
            if not self._matches_prefix(route, plan):
                continue
            # A materialized address without an authenticated charge basis is
            # useful provenance, but it has not stored any executable work.
            # Running it as a "prefix hit" would claim reuse while doing the
            # same primitive work live.
            if route.charge_basis_sha256 is None or route.historical_work_units == 0:
                continue
            program = self.graph.bank.restore_program(route.executable_program_sha256)
            try:
                applications = program.input_abi.application_count(value)
            except ComputeCrystalABIError:
                continue
            compatible.append((route, len(route.primitive_edge_sha256s), applications))
        if requested_prefix is None:
            prefix_route = (
                None
                if not compatible
                else min(
                    compatible,
                    key=lambda item: (
                        -item[1],
                        item[0].live_work_units * item[2],
                        item[0].sha256,
                    ),
                )[0]
            )
        else:
            selected = tuple(
                route
                for route, _length, _applications in compatible
                if route.sha256 == requested_prefix
            )
            if len(selected) != 1:
                raise ResidualExecutionIntegrityError(
                    "requested demand prefix is not one charged compatible plan prefix"
                )
            prefix_route = selected[0]
        prefix_length = (
            0 if prefix_route is None else len(prefix_route.primitive_edge_sha256s)
        )

        current: object = value
        prefix_receipt: ComputeExecutionReceipt | None = None
        if prefix_route is not None:
            prefix_execution = ComputeCrystalVM(self.graph.bank).execute(
                prefix_route.executable_program_sha256,
                current,
                charge_basis_sha256=prefix_route.charge_basis_sha256,
            )
            prefix_receipt = prefix_execution.receipt
            current = prefix_execution.output

        suffix_crystals = crystals[prefix_length:]
        suffix_receipt: ComputeExecutionReceipt | None = None
        if suffix_crystals:
            suffix_program = ComputeProgram.compose(suffix_crystals)
            self.graph.bank.publish_program(suffix_program)
            suffix_execution = ComputeCrystalVM(self.graph.bank).execute(
                suffix_program,
                current,
            )
            suffix_receipt = suffix_execution.receipt
            current = suffix_execution.output

        output = np.asarray(current)
        primitive.output_abi.validate(output, field="residual route output")
        receipts = tuple(
            receipt
            for receipt in (prefix_receipt, suffix_receipt)
            if receipt is not None
        )
        source_work = _checked_sum(
            [receipt.equivalent_unfused_source_work for receipt in receipts],
            field="equivalent source work",
        )
        live_work = _checked_sum(
            [receipt.live_discharge_work for receipt in receipts],
            field="live work",
        )
        historical = _checked_sum(
            [receipt.historical_work_released for receipt in receipts],
            field="historical work",
        )
        receipt = ResidualRouteDischargeReceipt(
            plan=plan,
            graph_generation=state.generation,
            graph_state_sha256=state.sha256,
            prefix_route_sha256=(
                None if prefix_route is None else prefix_route.sha256
            ),
            prefix_edge_sha256s=plan.primitive_edge_sha256s[:prefix_length],
            suffix_edge_sha256s=plan.primitive_edge_sha256s[prefix_length:],
            prefix_execution_receipt=prefix_receipt,
            suffix_execution_receipt=suffix_receipt,
            input_sha256=tensor_sha256(value, primitive.input_abi),
            output_sha256=tensor_sha256(output, primitive.output_abi),
            equivalent_source_work_units=source_work,
            live_work_units=live_work,
            historical_work_released=historical,
            verifier_sha256s=tuple(
                sorted({edge.verifier_sha256 for edge in exact_edges})
            ),
            evidence_sha256s=tuple(
                sorted({edge.evidence_sha256 for edge in exact_edges})
            ),
        )
        return ResidualRouteExecution(
            output=output,
            receipt=receipt,
            prefix_route=prefix_route,
        )


__all__ = [
    "MAX_RESIDUAL_RECEIPT_BYTES",
    "RESIDUAL_ROUTE_DISCHARGE_SCHEMA",
    "ResidualExecutionError",
    "ResidualExecutionIntegrityError",
    "ResidualRouteDischargeReceipt",
    "ResidualRouteExecution",
    "ResidualRouteExecutor",
]
