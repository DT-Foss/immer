"""Production execution loop from learned route demand to verified feedback."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Protocol, cast, runtime_checkable

from numpy.typing import NDArray

from .compute_graph import (
    ComputeOperatorGraph,
    ComputeOperatorGraphState,
    ComputeRoutePlan,
    MaterializedRoute,
)
from .demand_scheduler import (
    DEMAND_OUTCOME_EVENT_SCHEMA,
    DemandOutcomeReceipt,
    DemandStateTransitionReceipt,
    OperatorDemandIntegrityError,
    OperatorDemandScheduler,
    PPMPredictionReceipt,
    UCBSelectionReceipt,
)
from .identity import canonical_json_bytes, require_sha256
from .residual_execution import (
    ResidualRouteDischargeReceipt,
    ResidualRouteExecution,
    ResidualRouteExecutor,
)


DEMAND_EXECUTION_VERIFICATION_SCHEMA = "immer-ooe-demand-execution-verification/v1"
DEMAND_ROUTED_EXECUTION_SCHEMA = "immer-ooe-demand-routed-execution/v1"
DEMAND_EXECUTION_EVIDENCE_SCHEMA = "immer-ooe-demand-execution-evidence/v1"
DEMAND_EXECUTION_ABORT_SCHEMA = "immer-ooe-demand-execution-abort/v1"
DEMAND_REWARD_POLICY = "verified-quality-plus-historical-work-fraction/v1"
MAX_DEMAND_EXECUTION_RECEIPT_BYTES = 8 * 1024 * 1024


class DemandExecutionError(RuntimeError):
    """Base error for learned route execution and feedback."""


class DemandExecutionIntegrityError(DemandExecutionError):
    """A selection, execution, verifier, or feedback binding failed."""


class DemandExecutionUnavailableError(DemandExecutionError):
    """No charged materialized prefix can execute the requested plan."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


DEMAND_EXECUTION_ABORT_VERIFIER_SHA256 = _digest(
    {
        "schema": DEMAND_EXECUTION_ABORT_SCHEMA,
        "criterion": "operational-exception-before-verified-outcome",
        "effect": "neutralize-selection-pull-without-quality-feedback",
    }
)


def _strict_json(data: bytes, *, label: str) -> object:
    if not isinstance(data, bytes):
        raise TypeError(f"{label} must be immutable bytes")
    if len(data) > MAX_DEMAND_EXECUTION_RECEIPT_BYTES:
        raise DemandExecutionIntegrityError(f"{label} exceeds its byte bound")

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
        raise DemandExecutionIntegrityError(f"{label} is not strict JSON") from exc
    if canonical_json_bytes(value) != data:
        raise DemandExecutionIntegrityError(f"{label} is not canonical JSON")
    return value


def _sealed_body(value: object, *, schema: str, label: str) -> Mapping[str, object]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema", "body", "body_sha256"}
        or value.get("schema") != schema
    ):
        raise DemandExecutionIntegrityError(f"invalid {label} envelope")
    body = value.get("body")
    if not isinstance(body, Mapping):
        raise DemandExecutionIntegrityError(f"invalid {label} body")
    try:
        claimed = require_sha256(value.get("body_sha256"), field="body_sha256")
    except ValueError as exc:
        raise DemandExecutionIntegrityError(f"invalid {label} body hash") from exc
    if claimed != _digest(body):
        raise DemandExecutionIntegrityError(f"{label} body hash mismatch")
    return body


def _text(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > 4096
    ):
        raise ValueError(f"{field} must be canonical non-empty text")
    return value


def _reward(receipt: ResidualRouteDischargeReceipt, *, accepted: bool) -> float:
    if not accepted:
        return -1.0
    source = receipt.equivalent_source_work_units
    if source <= 0:
        raise ValueError("verified residual execution has no source-work basis")
    fraction = receipt.historical_work_released / source
    if not 0.0 <= fraction <= 1.0 or not math.isfinite(fraction):
        raise ValueError("historical-work fraction lies outside [0, 1]")
    return 1.0 + fraction


def _outcome_evidence_sha256(
    selection: UCBSelectionReceipt,
    residual: ResidualRouteDischargeReceipt,
    verification: "DemandExecutionVerification",
) -> str:
    return _digest(
        {
            "schema": DEMAND_EXECUTION_EVIDENCE_SCHEMA,
            "selection_event_sha256": selection.selection_event_sha256,
            "residual_receipt_sha256": residual.sha256,
            "verification_sha256": verification.sha256,
        }
    )


def _outcome_event_sha256(
    outcome: DemandOutcomeReceipt,
    logical_time: int,
) -> str:
    return _digest(
        {
            "schema": DEMAND_OUTCOME_EVENT_SCHEMA,
            "logical_time": logical_time,
            "outcome": outcome.to_dict(),
            "outcome_sha256": outcome.sha256,
        }
    )


@dataclass(frozen=True, slots=True)
class DemandExecutionVerification:
    """External quality judgment bound to one exact residual execution."""

    residual_receipt_sha256: str
    output_sha256: str
    accepted: bool
    verifier_sha256: str
    evidence_sha256: str
    reason: str

    def __post_init__(self) -> None:
        for field in (
            "residual_receipt_sha256",
            "output_sha256",
            "verifier_sha256",
            "evidence_sha256",
        ):
            object.__setattr__(
                self,
                field,
                require_sha256(getattr(self, field), field=field),
            )
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be boolean")
        object.__setattr__(self, "reason", _text(self.reason, field="reason"))

    @classmethod
    def create(
        cls,
        execution: ResidualRouteExecution,
        *,
        accepted: bool,
        verifier_sha256: str,
        evidence_sha256: str,
        reason: str,
    ) -> "DemandExecutionVerification":
        if not isinstance(execution, ResidualRouteExecution):
            raise TypeError("execution must be a ResidualRouteExecution")
        return cls(
            residual_receipt_sha256=execution.receipt.sha256,
            output_sha256=execution.receipt.output_sha256,
            accepted=accepted,
            verifier_sha256=verifier_sha256,
            evidence_sha256=evidence_sha256,
            reason=reason,
        )

    def to_dict(self) -> dict[str, object]:
        body = {
            "residual_receipt_sha256": self.residual_receipt_sha256,
            "output_sha256": self.output_sha256,
            "accepted": self.accepted,
            "verifier_sha256": self.verifier_sha256,
            "evidence_sha256": self.evidence_sha256,
            "reason": self.reason,
        }
        return {
            "schema": DEMAND_EXECUTION_VERIFICATION_SCHEMA,
            "body": body,
            "body_sha256": _digest(body),
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "DemandExecutionVerification":
        value = _strict_json(data, label="demand execution verification")
        body = _sealed_body(
            value,
            schema=DEMAND_EXECUTION_VERIFICATION_SCHEMA,
            label="demand execution verification",
        )
        expected = {
            "residual_receipt_sha256",
            "output_sha256",
            "accepted",
            "verifier_sha256",
            "evidence_sha256",
            "reason",
        }
        if set(body) != expected:
            raise DemandExecutionIntegrityError(
                "invalid demand execution verification body"
            )
        try:
            result = cls(**dict(body))
        except (TypeError, ValueError) as exc:
            raise DemandExecutionIntegrityError(
                "demand execution verification failed validation"
            ) from exc
        if result.to_bytes() != data:
            raise DemandExecutionIntegrityError(
                "demand execution verification changed during reconstruction"
            )
        return result


@runtime_checkable
class DemandExecutionVerifier(Protocol):
    def __call__(
        self, execution: ResidualRouteExecution
    ) -> DemandExecutionVerification: ...


def _parse_ppm(value: object) -> PPMPredictionReceipt:
    body = _sealed_body(
        value,
        schema="immer-ooe-operator-demand-ppm-prediction/v1",
        label="PPM prediction",
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
        raise DemandExecutionIntegrityError("invalid PPM prediction body")
    counts = []
    for row in raw_counts:
        if not isinstance(row, Mapping) or set(row) != {
            "route_sha256",
            "count",
        }:
            raise DemandExecutionIntegrityError("invalid PPM prediction count")
        counts.append(
            (cast(str, row.get("route_sha256")), cast(int, row.get("count")))
        )
    try:
        return PPMPredictionReceipt(
            scheduler_state_sha256=cast(str, body.get("scheduler_state_sha256")),
            graph_generation=cast(int, body.get("graph_generation")),
            graph_state_sha256=cast(str, body.get("graph_state_sha256")),
            input_abi_sha256=cast(str, body.get("input_abi_sha256")),
            output_abi_sha256=cast(str | None, body.get("output_abi_sha256")),
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
    except (TypeError, ValueError) as exc:
        raise DemandExecutionIntegrityError("PPM prediction is invalid") from exc


@dataclass(frozen=True, slots=True)
class DemandRoutedExecutionReceipt:
    """One joined proof from demand selection through verified feedback."""

    graph_generation: int
    graph_state_sha256: str
    selected_prefix_route_sha256: str
    ppm_prediction: PPMPredictionReceipt | None
    selection: UCBSelectionReceipt
    residual: ResidualRouteDischargeReceipt
    verification: DemandExecutionVerification
    outcome: DemandOutcomeReceipt
    outcome_transition: DemandStateTransitionReceipt
    episode_sha256: str | None
    reward_policy: str = DEMAND_REWARD_POLICY

    def __post_init__(self) -> None:
        if isinstance(self.graph_generation, bool) or not isinstance(
            self.graph_generation, int
        ) or self.graph_generation < 0:
            raise ValueError("graph_generation must be non-negative")
        graph_sha = require_sha256(
            self.graph_state_sha256, field="graph_state_sha256"
        )
        selected = require_sha256(
            self.selected_prefix_route_sha256,
            field="selected_prefix_route_sha256",
        )
        if not isinstance(self.selection, UCBSelectionReceipt):
            raise TypeError("selection must be a UCBSelectionReceipt")
        if not isinstance(self.residual, ResidualRouteDischargeReceipt):
            raise TypeError("residual must be a ResidualRouteDischargeReceipt")
        if not isinstance(self.verification, DemandExecutionVerification):
            raise TypeError("verification must be a DemandExecutionVerification")
        if not isinstance(self.outcome, DemandOutcomeReceipt):
            raise TypeError("outcome must be a DemandOutcomeReceipt")
        if not isinstance(self.outcome_transition, DemandStateTransitionReceipt):
            raise TypeError(
                "outcome_transition must be a DemandStateTransitionReceipt"
            )
        if self.ppm_prediction is not None and not isinstance(
            self.ppm_prediction, PPMPredictionReceipt
        ):
            raise TypeError("ppm_prediction must be a PPMPredictionReceipt or None")
        episode = self.episode_sha256
        if episode is not None:
            episode = require_sha256(episode, field="episode_sha256")
        policy = _text(self.reward_policy, field="reward_policy")
        if policy != DEMAND_REWARD_POLICY:
            raise ValueError("unknown demand reward policy")

        selection_event = self.selection.event
        if (
            selection_event.graph_generation != self.graph_generation
            or selection_event.graph_state_sha256 != graph_sha
            or self.residual.graph_generation != self.graph_generation
            or self.residual.graph_state_sha256 != graph_sha
            or self.outcome.graph_generation != self.graph_generation
            or self.outcome.graph_state_sha256 != graph_sha
        ):
            raise ValueError("joined demand execution graph identities differ")
        if (
            self.selection.route_sha256 != selected
            or self.residual.prefix_route_sha256 != selected
            or self.outcome.route_sha256 != selected
        ):
            raise ValueError("joined demand execution selected different prefixes")
        if (
            self.verification.residual_receipt_sha256 != self.residual.sha256
            or self.verification.output_sha256 != self.residual.output_sha256
        ):
            raise ValueError("external verification names another execution")
        if (
            self.outcome.success != self.verification.accepted
            or self.outcome.selection_event_sha256
            != self.selection.selection_event_sha256
            or self.outcome.outcome_verifier_sha256
            != self.verification.verifier_sha256
            or self.outcome.outcome_evidence_sha256
            != _outcome_evidence_sha256(
                self.selection, self.residual, self.verification
            )
            or self.outcome.reward
            != _reward(self.residual, accepted=self.verification.accepted)
            or self.outcome.episode_sha256 != episode
        ):
            raise ValueError("demand outcome differs from execution and verifier")
        expected_operation = (
            "record-positive" if self.verification.accepted else "record-negative"
        )
        if (
            not self.outcome_transition.changed
            or self.outcome_transition.operation != expected_operation
            or self.outcome_transition.previous_state_sha256
            != self.selection.transition.current_state_sha256
            or self.outcome_transition.generation
            != self.selection.transition.generation + 1
            or self.outcome_transition.event_sha256
            != _outcome_event_sha256(
                self.outcome,
                self.outcome_transition.generation,
            )
        ):
            raise ValueError("outcome transition did not commit the verified result")
        if self.ppm_prediction is not None:
            if (
                self.ppm_prediction.graph_generation != self.graph_generation
                or self.ppm_prediction.graph_state_sha256 != graph_sha
                or self.ppm_prediction.input_abi_sha256
                != selection_event.input_abi_sha256
                or self.ppm_prediction.scheduler_state_sha256
                != self.selection.transition.previous_state_sha256
            ):
                raise ValueError("PPM prediction differs from selection graph or ABI")
        object.__setattr__(self, "graph_state_sha256", graph_sha)
        object.__setattr__(self, "selected_prefix_route_sha256", selected)
        object.__setattr__(self, "episode_sha256", episode)
        object.__setattr__(self, "reward_policy", policy)

    def to_dict(self) -> dict[str, object]:
        ppm = self.ppm_prediction
        body = {
            "graph_generation": self.graph_generation,
            "graph_state_sha256": self.graph_state_sha256,
            "selected_prefix_route_sha256": self.selected_prefix_route_sha256,
            "ppm_prediction": None if ppm is None else ppm.to_dict(),
            "ppm_prediction_sha256": None if ppm is None else ppm.sha256,
            "selection": self.selection.to_dict(),
            "selection_sha256": self.selection.sha256,
            "residual": self.residual.to_dict(),
            "residual_sha256": self.residual.sha256,
            "verification": self.verification.to_dict(),
            "verification_sha256": self.verification.sha256,
            "outcome": self.outcome.to_dict(),
            "outcome_sha256": self.outcome.sha256,
            "outcome_transition": self.outcome_transition.to_dict(),
            "outcome_transition_sha256": self.outcome_transition.sha256,
            "episode_sha256": self.episode_sha256,
            "reward_policy": self.reward_policy,
        }
        return {
            "schema": DEMAND_ROUTED_EXECUTION_SCHEMA,
            "body": body,
            "body_sha256": _digest(body),
        }

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_dict())
        if len(data) > MAX_DEMAND_EXECUTION_RECEIPT_BYTES:
            raise ValueError("demand routed execution receipt exceeds its byte bound")
        return data

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "DemandRoutedExecutionReceipt":
        value = _strict_json(data, label="demand routed execution")
        body = _sealed_body(
            value,
            schema=DEMAND_ROUTED_EXECUTION_SCHEMA,
            label="demand routed execution",
        )
        expected = {
            "graph_generation",
            "graph_state_sha256",
            "selected_prefix_route_sha256",
            "ppm_prediction",
            "ppm_prediction_sha256",
            "selection",
            "selection_sha256",
            "residual",
            "residual_sha256",
            "verification",
            "verification_sha256",
            "outcome",
            "outcome_sha256",
            "outcome_transition",
            "outcome_transition_sha256",
            "episode_sha256",
            "reward_policy",
        }
        if set(body) != expected:
            raise DemandExecutionIntegrityError(
                "invalid demand routed execution body"
            )
        try:
            ppm_document = body.get("ppm_prediction")
            ppm = None if ppm_document is None else _parse_ppm(ppm_document)
            selection = UCBSelectionReceipt.from_bytes(
                canonical_json_bytes(body.get("selection"))
            )
            residual = ResidualRouteDischargeReceipt.from_bytes(
                canonical_json_bytes(body.get("residual"))
            )
            verification = DemandExecutionVerification.from_bytes(
                canonical_json_bytes(body.get("verification"))
            )
            outcome = DemandOutcomeReceipt.from_dict(body.get("outcome"))
            transition = DemandStateTransitionReceipt.from_bytes(
                canonical_json_bytes(body.get("outcome_transition"))
            )
            receipt = cls(
                graph_generation=cast(int, body.get("graph_generation")),
                graph_state_sha256=cast(str, body.get("graph_state_sha256")),
                selected_prefix_route_sha256=cast(
                    str, body.get("selected_prefix_route_sha256")
                ),
                ppm_prediction=ppm,
                selection=selection,
                residual=residual,
                verification=verification,
                outcome=outcome,
                outcome_transition=transition,
                episode_sha256=cast(str | None, body.get("episode_sha256")),
                reward_policy=cast(str, body.get("reward_policy")),
            )
            nested = (
                ("ppm_prediction_sha256", None if ppm is None else ppm.sha256),
                ("selection_sha256", selection.sha256),
                ("residual_sha256", residual.sha256),
                ("verification_sha256", verification.sha256),
                ("outcome_sha256", outcome.sha256),
                ("outcome_transition_sha256", transition.sha256),
            )
            for field, actual in nested:
                claimed = body.get(field)
                if actual is None:
                    if claimed is not None:
                        raise DemandExecutionIntegrityError(
                            f"{field} must be null with its absent receipt"
                        )
                elif require_sha256(claimed, field=field) != actual:
                    raise DemandExecutionIntegrityError(
                        f"{field} content address mismatch"
                    )
        except DemandExecutionIntegrityError:
            raise
        except (OperatorDemandIntegrityError, TypeError, ValueError) as exc:
            raise DemandExecutionIntegrityError(
                "demand routed execution failed validation"
            ) from exc
        if receipt.to_bytes() != data:
            raise DemandExecutionIntegrityError(
                "demand routed execution changed during reconstruction"
            )
        return receipt


@dataclass(frozen=True, slots=True)
class DemandRoutedExecution:
    output: NDArray[Any]
    receipt: DemandRoutedExecutionReceipt


class DemandRoutedExecutor:
    """Select, execute, verify, and learn one charged-prefix decision."""

    def __init__(
        self,
        *,
        graph: ComputeOperatorGraph,
        scheduler: OperatorDemandScheduler,
        verifier: DemandExecutionVerifier
        | Callable[[ResidualRouteExecution], DemandExecutionVerification],
        ppm_min_probability: float = 0.75,
    ) -> None:
        if not isinstance(graph, ComputeOperatorGraph):
            raise TypeError("graph must be a ComputeOperatorGraph")
        if not isinstance(scheduler, OperatorDemandScheduler):
            raise TypeError("scheduler must be an OperatorDemandScheduler")
        if not callable(verifier):
            raise TypeError("verifier must be callable")
        probability = float(ppm_min_probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("ppm_min_probability must lie in [0, 1]")
        self.graph = graph
        self.scheduler = scheduler
        self.verifier = verifier
        self.ppm_min_probability = probability

    @staticmethod
    def _matches_prefix(route: MaterializedRoute, plan: ComputeRoutePlan) -> bool:
        length = len(route.primitive_edge_sha256s)
        return bool(
            0 < length <= len(plan.primitive_edge_sha256s)
            and route.source_state == plan.source_state
            and route.goal_state == plan.finite_plan.expected_states[length]
            and route.primitive_edge_sha256s
            == plan.primitive_edge_sha256s[:length]
            and route.charge_basis_sha256 is not None
            and route.historical_work_units > 0
        )

    def _candidates(
        self,
        state: ComputeOperatorGraphState,
        plan: ComputeRoutePlan,
        candidate_route_sha256s: Sequence[str] | None,
    ) -> tuple[MaterializedRoute, ...]:
        by_sha = {route.sha256: route for route in state.materialized_routes}
        available = tuple(
            route
            for route in state.materialized_routes
            if self._matches_prefix(route, plan)
        )
        if candidate_route_sha256s is None:
            candidates = available
        else:
            requested = tuple(
                require_sha256(value, field="candidate_route_sha256s")
                for value in candidate_route_sha256s
            )
            if requested != tuple(sorted(set(requested))):
                raise ValueError("candidate_route_sha256s must be sorted and unique")
            try:
                candidates = tuple(by_sha[value] for value in requested)
            except KeyError as exc:
                raise DemandExecutionIntegrityError(
                    "candidate prefix is absent from the graph"
                ) from exc
            if any(not self._matches_prefix(route, plan) for route in candidates):
                raise DemandExecutionIntegrityError(
                    "candidate route is not one charged prefix of the plan"
                )
        if not candidates:
            raise DemandExecutionUnavailableError(
                "plan has no charged materialized prefix"
            )
        return tuple(sorted(candidates, key=lambda route: route.sha256))

    def _abort_selection(
        self,
        selection: UCBSelectionReceipt,
        state: ComputeOperatorGraphState,
        *,
        stage: str,
        exception: BaseException,
    ) -> None:
        exception_type = (
            f"{type(exception).__module__}.{type(exception).__qualname__}"
        )
        evidence = _digest(
            {
                "schema": DEMAND_EXECUTION_ABORT_SCHEMA,
                "selection_event_sha256": selection.selection_event_sha256,
                "graph_state_sha256": state.sha256,
                "stage": stage,
                "exception_type": exception_type,
            }
        )
        self.scheduler.abort_selection(
            selection,
            state,
            abort_evidence_sha256=evidence,
            abort_verifier_sha256=DEMAND_EXECUTION_ABORT_VERIFIER_SHA256,
            reason_code=f"{stage}:{exception_type}",
        )

    def execute(
        self,
        plan: ComputeRoutePlan,
        value: object,
        *,
        candidate_route_sha256s: Sequence[str] | None = None,
        history_route_sha256s: Sequence[str] = (),
        episode_sha256: str | None = None,
    ) -> DemandRoutedExecution:
        if not isinstance(plan, ComputeRoutePlan):
            raise TypeError("plan must be a ComputeRoutePlan")
        state = self.graph.state()
        if (
            state.generation != plan.graph_generation
            or state.sha256 != plan.graph_state_sha256
        ):
            raise DemandExecutionIntegrityError(
                "demand execution plan is stale for the graph head"
            )
        candidates = self._candidates(state, plan, candidate_route_sha256s)
        input_abi = candidates[0].input_abi_sha256
        if any(route.input_abi_sha256 != input_abi for route in candidates):
            raise DemandExecutionIntegrityError(
                "charged prefix candidates disagree on input ABI"
            )
        # Reject an invalid caller tensor before a UCB pull is persisted.
        candidate_program = self.graph.bank.restore_program(
            candidates[0].executable_program_sha256
        )
        candidate_program.input_abi.application_count(value)
        episode = (
            None
            if episode_sha256 is None
            else require_sha256(episode_sha256, field="episode_sha256")
        )
        history = tuple(
            require_sha256(value, field="history_route_sha256s")
            for value in history_route_sha256s
        )
        ppm: PPMPredictionReceipt | None = None
        selection_candidates = tuple(route.sha256 for route in candidates)
        if history:
            ppm = self.scheduler.predict_ppm(
                state,
                history,
                input_abi_sha256=input_abi,
            )
            if (
                ppm.selected_route_sha256 in set(selection_candidates)
                and ppm.probability >= self.ppm_min_probability
            ):
                selection_candidates = (cast(str, ppm.selected_route_sha256),)
        selection = self.scheduler.select_ucb1(
            state,
            input_abi_sha256=input_abi,
            candidate_route_sha256s=tuple(sorted(selection_candidates)),
        )
        stage = "ppm-selection-binding"
        try:
            if (
                ppm is not None
                and ppm.scheduler_state_sha256
                != selection.transition.previous_state_sha256
            ):
                raise DemandExecutionIntegrityError(
                    "scheduler changed between PPM prediction and persisted selection"
                )
            stage = "residual-execution"
            residual_execution = ResidualRouteExecutor(self.graph).execute(
                plan,
                value,
                prefix_route_sha256=selection.route_sha256,
            )
            stage = "external-verification"
            verification = self.verifier(residual_execution)
            if not isinstance(verification, DemandExecutionVerification):
                raise DemandExecutionIntegrityError(
                    "verifier returned no DemandExecutionVerification"
                )
            if (
                verification.residual_receipt_sha256
                != residual_execution.receipt.sha256
                or verification.output_sha256
                != residual_execution.receipt.output_sha256
            ):
                raise DemandExecutionIntegrityError(
                    "verifier judgment belongs to another residual execution"
                )
        except BaseException as exc:
            try:
                self._abort_selection(
                    selection,
                    state,
                    stage=stage,
                    exception=exc,
                )
            except BaseException as abort_exc:
                raise DemandExecutionIntegrityError(
                    "failed to neutralize an operationally aborted selection"
                ) from abort_exc
            raise
        try:
            selected_route = next(
                route
                for route in candidates
                if route.sha256 == selection.route_sha256
            )
            reward = _reward(
                residual_execution.receipt,
                accepted=verification.accepted,
            )
            outcome = DemandOutcomeReceipt.create(
                route=selected_route,
                graph_state=state,
                success=verification.accepted,
                reward=reward,
                outcome_evidence_sha256=_outcome_evidence_sha256(
                    selection,
                    residual_execution.receipt,
                    verification,
                ),
                outcome_verifier_sha256=verification.verifier_sha256,
                selection_event_sha256=selection.selection_event_sha256,
                episode_sha256=episode,
            )
            transition = self.scheduler.record_outcome(
                outcome,
                state,
                expected_state_sha256=selection.transition.current_state_sha256,
            )
        except BaseException as exc:
            try:
                self._abort_selection(
                    selection,
                    state,
                    stage="outcome-commit",
                    exception=exc,
                )
            except BaseException as abort_exc:
                raise DemandExecutionIntegrityError(
                    "failed to settle or neutralize a verified selection"
                ) from abort_exc
            raise
        receipt = DemandRoutedExecutionReceipt(
            graph_generation=state.generation,
            graph_state_sha256=state.sha256,
            selected_prefix_route_sha256=selected_route.sha256,
            ppm_prediction=ppm,
            selection=selection,
            residual=residual_execution.receipt,
            verification=verification,
            outcome=outcome,
            outcome_transition=transition,
            episode_sha256=episode,
        )
        return DemandRoutedExecution(
            output=residual_execution.output,
            receipt=receipt,
        )


__all__ = [
    "DEMAND_EXECUTION_ABORT_SCHEMA",
    "DEMAND_EXECUTION_ABORT_VERIFIER_SHA256",
    "DEMAND_EXECUTION_EVIDENCE_SCHEMA",
    "DEMAND_EXECUTION_VERIFICATION_SCHEMA",
    "DEMAND_REWARD_POLICY",
    "DEMAND_ROUTED_EXECUTION_SCHEMA",
    "DemandExecutionError",
    "DemandExecutionIntegrityError",
    "DemandExecutionUnavailableError",
    "DemandExecutionVerification",
    "DemandExecutionVerifier",
    "DemandRoutedExecution",
    "DemandRoutedExecutionReceipt",
    "DemandRoutedExecutor",
    "MAX_DEMAND_EXECUTION_RECEIPT_BYTES",
]
