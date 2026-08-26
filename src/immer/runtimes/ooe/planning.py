"""Finite-horizon and goal-directed planning over OoE world kernels."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np

from .identity import canonical_json_bytes, require_sha256
from .math_core import array_sha256
from .world_model import (
    ActionConditionedWorldModel,
    TransitionPrediction,
    label_sha256,
)


PLAN_SCHEMA = "immer-ooe-finite-horizon-plan/v1"
PLAN_EXECUTION_SCHEMA = "immer-ooe-plan-execution/v1"
PLANNING_GATE_SCHEMA = "immer-ooe-planning-gates/v1"
DEFAULT_MAX_HORIZON = 4096
DEFAULT_MAX_POLICY_ENTRIES = 1_000_000


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _uint(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return result


def _probability(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a probability")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must lie in [0, 1]")
    return result


def _finite_positive(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be a positive finite number")
    return result


def _planning_gate_dict(
    min_evidence_mass: float,
    max_normalized_entropy: float,
    min_peak_probability: float,
    min_predicted_success: float,
) -> dict[str, Any]:
    return {
        "schema": PLANNING_GATE_SCHEMA,
        "min_evidence_mass": min_evidence_mass,
        "max_normalized_entropy": max_normalized_entropy,
        "min_peak_probability": min_peak_probability,
        "min_predicted_success": min_predicted_success,
    }


def _planning_gate_sha256(
    min_evidence_mass: float,
    max_normalized_entropy: float,
    min_peak_probability: float,
    min_predicted_success: float,
) -> str:
    return _sha256(
        _planning_gate_dict(
            min_evidence_mass,
            max_normalized_entropy,
            min_peak_probability,
            min_predicted_success,
        )
    )


def _text(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > 1024
    ):
        raise ValueError(f"{field} must be canonical non-empty text")
    return value


@dataclass(frozen=True, slots=True)
class PolicyEntry:
    remaining_horizon: int
    state: str
    action: str
    predicted_value: float
    coverage: float
    transition_row_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "remaining_horizon",
            _uint(self.remaining_horizon, field="remaining_horizon"),
        )
        if self.remaining_horizon == 0:
            raise ValueError("policy entries require a positive remaining horizon")
        object.__setattr__(self, "state", _text(self.state, field="state"))
        object.__setattr__(self, "action", _text(self.action, field="action"))
        object.__setattr__(
            self,
            "predicted_value",
            _probability(self.predicted_value, field="predicted_value"),
        )
        object.__setattr__(
            self, "coverage", _probability(self.coverage, field="coverage")
        )
        object.__setattr__(
            self,
            "transition_row_sha256",
            require_sha256(self.transition_row_sha256, field="transition_row_sha256"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "remaining_horizon": self.remaining_horizon,
            "state": self.state,
            "action": self.action,
            "predicted_value": self.predicted_value,
            "coverage": self.coverage,
            "transition_row_sha256": self.transition_row_sha256,
        }


@dataclass(frozen=True, slots=True)
class FiniteHorizonPlan:
    world_model_sha256: str
    counts_sha256: str
    start_state: str
    goal_state: str | None
    horizon: int
    min_evidence_mass: float
    max_normalized_entropy: float
    min_peak_probability: float
    min_predicted_success: float
    gate_config_sha256: str
    terminal_rewards: tuple[tuple[str, float], ...]
    predicted_success: float
    minimum_coverage: float
    expected_states: tuple[str, ...]
    expected_actions: tuple[str, ...]
    policy: tuple[PolicyEntry, ...]
    kernel_hashes: tuple[tuple[str, str], ...]
    verifier_hashes: tuple[str, ...]
    evidence_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "world_model_sha256",
            require_sha256(self.world_model_sha256, field="world_model_sha256"),
        )
        object.__setattr__(
            self,
            "counts_sha256",
            require_sha256(self.counts_sha256, field="counts_sha256"),
        )
        object.__setattr__(self, "horizon", _uint(self.horizon, field="horizon"))
        object.__setattr__(
            self,
            "min_evidence_mass",
            _finite_positive(self.min_evidence_mass, field="min_evidence_mass"),
        )
        for field in (
            "max_normalized_entropy",
            "min_peak_probability",
            "min_predicted_success",
        ):
            object.__setattr__(
                self, field, _probability(getattr(self, field), field=field)
            )
        object.__setattr__(
            self,
            "gate_config_sha256",
            require_sha256(self.gate_config_sha256, field="gate_config_sha256"),
        )
        expected_gate_hash = _planning_gate_sha256(
            self.min_evidence_mass,
            self.max_normalized_entropy,
            self.min_peak_probability,
            self.min_predicted_success,
        )
        if self.gate_config_sha256 != expected_gate_hash:
            raise ValueError("planning gate hash mismatch")
        object.__setattr__(
            self, "start_state", _text(self.start_state, field="start_state")
        )
        if self.goal_state is not None:
            object.__setattr__(
                self, "goal_state", _text(self.goal_state, field="goal_state")
            )
        object.__setattr__(
            self,
            "predicted_success",
            _probability(self.predicted_success, field="predicted_success"),
        )
        if self.predicted_success < self.min_predicted_success:
            raise ValueError("plan success lies below its planning gate")
        object.__setattr__(
            self,
            "minimum_coverage",
            _probability(self.minimum_coverage, field="minimum_coverage"),
        )
        terminal_states: set[str] = set()
        for state, reward in self.terminal_rewards:
            _text(state, field="terminal state")
            if state in terminal_states:
                raise ValueError("terminal reward states must be unique")
            terminal_states.add(state)
            _probability(reward, field="terminal reward")
        if not terminal_states:
            raise ValueError("terminal rewards must not be empty")
        if self.goal_state is not None and self.goal_state not in terminal_states:
            raise ValueError("goal state lacks a terminal reward")
        kernel_actions: set[str] = set()
        for action, digest in self.kernel_hashes:
            _text(action, field="kernel action")
            if action in kernel_actions:
                raise ValueError("kernel actions must be unique")
            kernel_actions.add(action)
            require_sha256(digest, field="kernel hash")
        for provenance in (self.verifier_hashes, self.evidence_hashes):
            if tuple(sorted(set(provenance))) != provenance:
                raise ValueError("provenance hashes must be sorted and unique")
        for digest in (*self.verifier_hashes, *self.evidence_hashes):
            require_sha256(digest, field="provenance hash")
        if len(self.expected_states) != len(self.expected_actions) + 1:
            raise ValueError("expected state/action path lengths differ")
        if len(self.expected_actions) > self.horizon:
            raise ValueError("expected path exceeds the plan horizon")
        if not self.expected_states or self.expected_states[0] != self.start_state:
            raise ValueError("expected path must start at start_state")
        policy_keys: set[tuple[int, str]] = set()
        for entry in self.policy:
            if entry.remaining_horizon > self.horizon:
                raise ValueError("policy entry exceeds the plan horizon")
            key = (entry.remaining_horizon, entry.state)
            if key in policy_keys:
                raise ValueError("policy contains duplicate state/horizon entries")
            policy_keys.add(key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PLAN_SCHEMA,
            "world_model_sha256": self.world_model_sha256,
            "counts_sha256": self.counts_sha256,
            "start_state": self.start_state,
            "goal_state": self.goal_state,
            "horizon": self.horizon,
            "min_evidence_mass": self.min_evidence_mass,
            "max_normalized_entropy": self.max_normalized_entropy,
            "min_peak_probability": self.min_peak_probability,
            "min_predicted_success": self.min_predicted_success,
            "gate_config_sha256": self.gate_config_sha256,
            "terminal_rewards": dict(self.terminal_rewards),
            "predicted_success": self.predicted_success,
            "minimum_coverage": self.minimum_coverage,
            "expected_states": list(self.expected_states),
            "expected_actions": list(self.expected_actions),
            "policy": [entry.to_dict() for entry in self.policy],
            "kernel_hashes": dict(self.kernel_hashes),
            "verifier_hashes": list(self.verifier_hashes),
            "evidence_hashes": list(self.evidence_hashes),
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class PlanDecision:
    plan: FiniteHorizonPlan | None
    abstained: bool
    reason: str | None
    predicted_success: float
    world_model_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "immer-ooe-plan-decision/v1",
            "plan_sha256": None if self.plan is None else self.plan.sha256,
            "abstained": self.abstained,
            "reason": self.reason,
            "predicted_success": self.predicted_success,
            "world_model_sha256": self.world_model_sha256,
        }


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    target_state: str
    verifier_sha256: str
    evidence_sha256: str
    verified: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.target_state, str) or not self.target_state:
            raise ValueError("target_state must be non-empty text")
        object.__setattr__(
            self,
            "verifier_sha256",
            require_sha256(self.verifier_sha256, field="verifier_sha256"),
        )
        object.__setattr__(
            self,
            "evidence_sha256",
            require_sha256(self.evidence_sha256, field="evidence_sha256"),
        )
        if not isinstance(self.verified, bool):
            raise TypeError("verified must be boolean")


@dataclass(frozen=True, slots=True)
class PlanExecutionStep:
    ordinal: int
    source_state: str
    action: str
    target_state: str
    predicted_probability: float
    coverage: float
    state_sha256: str
    action_sha256: str
    transition_sha256: str
    transition_row_sha256: str
    verifier_sha256: str
    evidence_sha256: str
    verified: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "ordinal", _uint(self.ordinal, field="ordinal"))
        for field in ("source_state", "action", "target_state"):
            object.__setattr__(self, field, _text(getattr(self, field), field=field))
        object.__setattr__(
            self,
            "predicted_probability",
            _probability(self.predicted_probability, field="predicted_probability"),
        )
        object.__setattr__(
            self, "coverage", _probability(self.coverage, field="coverage")
        )
        for field in (
            "state_sha256",
            "action_sha256",
            "transition_sha256",
            "transition_row_sha256",
            "verifier_sha256",
            "evidence_sha256",
        ):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        if not isinstance(self.verified, bool):
            raise TypeError("verified must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "source_state": self.source_state,
            "action": self.action,
            "target_state": self.target_state,
            "predicted_probability": self.predicted_probability,
            "coverage": self.coverage,
            "state_sha256": self.state_sha256,
            "action_sha256": self.action_sha256,
            "transition_sha256": self.transition_sha256,
            "transition_row_sha256": self.transition_row_sha256,
            "verifier_sha256": self.verifier_sha256,
            "evidence_sha256": self.evidence_sha256,
            "verified": self.verified,
        }


@dataclass(frozen=True, slots=True)
class PlanExecutionReceipt:
    plan_sha256: str
    world_model_sha256: str
    start_state: str
    goal_state: str | None
    final_state: str
    min_evidence_mass: float
    max_normalized_entropy: float
    min_peak_probability: float
    min_predicted_success: float
    gate_config_sha256: str
    predicted_success: float
    minimum_coverage: float
    success: bool
    reason: str
    steps: tuple[PlanExecutionStep, ...]
    state_hashes: tuple[str, ...]
    action_hashes: tuple[str, ...]
    transition_hashes: tuple[str, ...]
    verifier_hashes: tuple[str, ...]
    evidence_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in ("plan_sha256", "world_model_sha256"):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        for field in ("start_state", "final_state", "reason"):
            object.__setattr__(self, field, _text(getattr(self, field), field=field))
        if self.goal_state is not None:
            object.__setattr__(
                self, "goal_state", _text(self.goal_state, field="goal_state")
            )
        object.__setattr__(
            self,
            "min_evidence_mass",
            _finite_positive(self.min_evidence_mass, field="min_evidence_mass"),
        )
        for field in (
            "max_normalized_entropy",
            "min_peak_probability",
            "min_predicted_success",
        ):
            object.__setattr__(
                self, field, _probability(getattr(self, field), field=field)
            )
        object.__setattr__(
            self,
            "gate_config_sha256",
            require_sha256(self.gate_config_sha256, field="gate_config_sha256"),
        )
        expected_gate_hash = _planning_gate_sha256(
            self.min_evidence_mass,
            self.max_normalized_entropy,
            self.min_peak_probability,
            self.min_predicted_success,
        )
        if self.gate_config_sha256 != expected_gate_hash:
            raise ValueError("execution planning gate hash mismatch")
        object.__setattr__(
            self,
            "predicted_success",
            _probability(self.predicted_success, field="predicted_success"),
        )
        if self.predicted_success < self.min_predicted_success:
            raise ValueError("execution success lies below its planning gate")
        object.__setattr__(
            self,
            "minimum_coverage",
            _probability(self.minimum_coverage, field="minimum_coverage"),
        )
        if not isinstance(self.success, bool):
            raise TypeError("success must be boolean")
        if tuple(step.ordinal for step in self.steps) != tuple(range(len(self.steps))):
            raise ValueError("execution step ordinals must be contiguous")
        if len(self.state_hashes) != len(self.steps) + 1:
            raise ValueError("execution state hash count is inconsistent")
        if len(self.action_hashes) != len(self.steps):
            raise ValueError("execution action hash count is inconsistent")
        if len(self.transition_hashes) != len(self.steps):
            raise ValueError("execution transition hash count is inconsistent")
        for hashes in (
            self.state_hashes,
            self.action_hashes,
            self.transition_hashes,
            self.verifier_hashes,
            self.evidence_hashes,
        ):
            for digest in hashes:
                require_sha256(digest, field="execution hash")
        for hashes in (self.verifier_hashes, self.evidence_hashes):
            if tuple(sorted(set(hashes))) != hashes:
                raise ValueError("execution provenance must be sorted and unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PLAN_EXECUTION_SCHEMA,
            "plan_sha256": self.plan_sha256,
            "world_model_sha256": self.world_model_sha256,
            "start_state": self.start_state,
            "goal_state": self.goal_state,
            "final_state": self.final_state,
            "min_evidence_mass": self.min_evidence_mass,
            "max_normalized_entropy": self.max_normalized_entropy,
            "min_peak_probability": self.min_peak_probability,
            "min_predicted_success": self.min_predicted_success,
            "gate_config_sha256": self.gate_config_sha256,
            "predicted_success": self.predicted_success,
            "minimum_coverage": self.minimum_coverage,
            "success": self.success,
            "reason": self.reason,
            "steps": [step.to_dict() for step in self.steps],
            "state_hashes": list(self.state_hashes),
            "action_hashes": list(self.action_hashes),
            "transition_hashes": list(self.transition_hashes),
            "verifier_hashes": list(self.verifier_hashes),
            "evidence_hashes": list(self.evidence_hashes),
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.to_dict())


class FiniteHorizonPlanner:
    """Dynamic-programming planner with epistemic row gates."""

    def __init__(
        self,
        world_model: ActionConditionedWorldModel,
        *,
        min_evidence_mass: float | None = None,
        max_normalized_entropy: float | None = None,
        min_peak_probability: float | None = None,
        min_predicted_success: float = 0.5,
        max_horizon: int = DEFAULT_MAX_HORIZON,
        max_policy_entries: int = DEFAULT_MAX_POLICY_ENTRIES,
    ) -> None:
        if not isinstance(world_model, ActionConditionedWorldModel):
            raise TypeError("world_model must be an ActionConditionedWorldModel")
        self.world_model = world_model
        self.min_evidence_mass = min_evidence_mass
        self.max_normalized_entropy = max_normalized_entropy
        self.min_peak_probability = min_peak_probability
        self.min_predicted_success = _probability(
            min_predicted_success, field="min_predicted_success"
        )
        self.max_horizon = _uint(max_horizon, field="max_horizon")
        if self.max_horizon == 0:
            raise ValueError("max_horizon must be positive")
        self.max_policy_entries = _uint(max_policy_entries, field="max_policy_entries")
        if self.max_policy_entries == 0:
            raise ValueError("max_policy_entries must be positive")

    def _resolved_gates(self) -> tuple[float, float, float, float, str]:
        min_evidence_mass = _finite_positive(
            self.world_model.min_evidence_mass
            if self.min_evidence_mass is None
            else self.min_evidence_mass,
            field="min_evidence_mass",
        )
        max_normalized_entropy = _probability(
            self.world_model.max_normalized_entropy
            if self.max_normalized_entropy is None
            else self.max_normalized_entropy,
            field="max_normalized_entropy",
        )
        min_peak_probability = _probability(
            self.world_model.min_peak_probability
            if self.min_peak_probability is None
            else self.min_peak_probability,
            field="min_peak_probability",
        )
        min_predicted_success = _probability(
            self.min_predicted_success, field="min_predicted_success"
        )
        gate_hash = _planning_gate_sha256(
            min_evidence_mass,
            max_normalized_entropy,
            min_peak_probability,
            min_predicted_success,
        )
        return (
            min_evidence_mass,
            max_normalized_entropy,
            min_peak_probability,
            min_predicted_success,
            gate_hash,
        )

    def _prediction(
        self,
        state: str,
        action: str,
        *,
        min_evidence_mass: float,
        max_normalized_entropy: float,
        min_peak_probability: float,
    ) -> TransitionPrediction:
        return self.world_model.predict(
            state,
            action,
            min_evidence_mass=min_evidence_mass,
            max_normalized_entropy=max_normalized_entropy,
            min_peak_probability=min_peak_probability,
        )

    def plan_goal(
        self,
        start_state: str,
        goal_state: str,
        *,
        horizon: int,
    ) -> PlanDecision:
        if goal_state not in self.world_model.states:
            raise KeyError(f"unknown goal state: {goal_state!r}")
        return self.plan_terminal_reward(
            start_state,
            {goal_state: 1.0},
            horizon=horizon,
            goal_state=goal_state,
        )

    def plan_terminal_reward(
        self,
        start_state: str,
        terminal_rewards: Mapping[str, float],
        *,
        horizon: int,
        goal_state: str | None = None,
    ) -> PlanDecision:
        if start_state not in self.world_model.states:
            raise KeyError(f"unknown start state: {start_state!r}")
        steps = _uint(horizon, field="horizon")
        if steps > self.max_horizon:
            raise ValueError("horizon exceeds the configured bound")
        if steps * len(self.world_model.states) > self.max_policy_entries:
            raise ValueError("plan policy exceeds the configured entry bound")
        if not isinstance(terminal_rewards, Mapping) or not terminal_rewards:
            raise ValueError("terminal_rewards must be a non-empty mapping")
        reward_by_state: dict[str, float] = {}
        for state, reward in terminal_rewards.items():
            if state not in self.world_model.states:
                raise KeyError(f"unknown terminal state: {state!r}")
            reward_by_state[state] = _probability(reward, field="terminal reward")
        if max(reward_by_state.values()) <= 0.0:
            raise ValueError("at least one terminal reward must be positive")
        if goal_state is not None and goal_state not in reward_by_state:
            raise ValueError("goal_state must be one of the terminal rewards")

        (
            min_evidence_mass,
            max_normalized_entropy,
            min_peak_probability,
            min_predicted_success,
            gate_config_sha256,
        ) = self._resolved_gates()

        before_hash = self.world_model.sha256
        size = len(self.world_model.states)
        state_index = {
            state: index for index, state in enumerate(self.world_model.states)
        }
        rewards = np.zeros(size, dtype=np.float64)
        for state, reward in reward_by_state.items():
            rewards[state_index[state]] = reward

        predictions: dict[tuple[str, str], TransitionPrediction] = {}
        for state in self.world_model.states:
            if rewards[state_index[state]] > 0.0:
                continue
            for action in self.world_model.actions:
                predictions[(state, action)] = self._prediction(
                    state,
                    action,
                    min_evidence_mass=min_evidence_mass,
                    max_normalized_entropy=max_normalized_entropy,
                    min_peak_probability=min_peak_probability,
                )

        values: list[np.ndarray] = [rewards.copy()]
        policy_maps: list[dict[str, PolicyEntry]] = [{}]
        for remaining in range(1, steps + 1):
            previous = values[-1]
            current = rewards.copy()
            policy: dict[str, PolicyEntry] = {}
            for state_index_value, state in enumerate(self.world_model.states):
                if rewards[state_index_value] > 0.0:
                    continue
                best_value = -1.0
                best_prediction: TransitionPrediction | None = None
                best_action: str | None = None
                for action in self.world_model.actions:
                    prediction = predictions[(state, action)]
                    if prediction.abstained:
                        continue
                    candidate = float(np.dot(prediction.probabilities, previous))
                    if candidate > best_value + 1e-15:
                        best_value = candidate
                        best_action = action
                        best_prediction = prediction
                if (
                    best_action is not None
                    and best_prediction is not None
                    and best_value > 0.0
                ):
                    current[state_index_value] = min(1.0, best_value)
                    policy[state] = PolicyEntry(
                        remaining_horizon=remaining,
                        state=state,
                        action=best_action,
                        predicted_value=float(current[state_index_value]),
                        coverage=best_prediction.coverage,
                        transition_row_sha256=best_prediction.row_sha256,
                    )
            values.append(current)
            policy_maps.append(policy)

        predicted = float(values[steps][state_index[start_state]])
        if predicted < min_predicted_success:
            reason = (
                "unreachable-or-uncovered"
                if predicted == 0.0
                else "predicted-success-below-threshold"
            )
            if self.world_model.sha256 != before_hash:
                raise RuntimeError("planning mutated the world model")
            return PlanDecision(
                plan=None,
                abstained=True,
                reason=reason,
                predicted_success=predicted,
                world_model_sha256=before_hash,
            )

        expected_states = [start_state]
        expected_actions: list[str] = []
        path_coverages: list[float] = []
        current_state = start_state
        remaining = steps
        while remaining > 0 and rewards[state_index[current_state]] == 0.0:
            entry = policy_maps[remaining].get(current_state)
            if entry is None:
                break
            prediction = predictions[(current_state, entry.action)]
            downstream = values[remaining - 1]
            scores = np.asarray(prediction.probabilities) * downstream
            target_index = int(np.argmax(scores))
            expected_actions.append(entry.action)
            path_coverages.append(entry.coverage)
            current_state = self.world_model.states[target_index]
            expected_states.append(current_state)
            remaining -= 1

        policy_entries = tuple(
            policy_maps[remaining][state]
            for remaining in range(1, steps + 1)
            for state in self.world_model.states
            if state in policy_maps[remaining]
        )
        kernel_hashes = tuple(
            (action, array_sha256(self.world_model.action_kernel(action)))
            for action in self.world_model.actions
        )
        plan = FiniteHorizonPlan(
            world_model_sha256=before_hash,
            counts_sha256=array_sha256(self.world_model.counts),
            start_state=start_state,
            goal_state=goal_state,
            horizon=steps,
            min_evidence_mass=min_evidence_mass,
            max_normalized_entropy=max_normalized_entropy,
            min_peak_probability=min_peak_probability,
            min_predicted_success=min_predicted_success,
            gate_config_sha256=gate_config_sha256,
            terminal_rewards=tuple(
                (state, reward_by_state[state])
                for state in self.world_model.states
                if state in reward_by_state
            ),
            predicted_success=predicted,
            minimum_coverage=min(path_coverages, default=1.0),
            expected_states=tuple(expected_states),
            expected_actions=tuple(expected_actions),
            policy=policy_entries,
            kernel_hashes=kernel_hashes,
            verifier_hashes=self.world_model.verifier_hashes,
            evidence_hashes=self.world_model.evidence_hashes,
        )
        if self.world_model.sha256 != before_hash:
            raise RuntimeError("planning mutated the world model")
        return PlanDecision(
            plan=plan,
            abstained=False,
            reason=None,
            predicted_success=predicted,
            world_model_sha256=before_hash,
        )

    def execute(
        self,
        plan: FiniteHorizonPlan,
        executor: Callable[[str, str, int], ExecutionOutcome],
    ) -> PlanExecutionReceipt:
        if not isinstance(plan, FiniteHorizonPlan):
            raise TypeError("plan must be a FiniteHorizonPlan")
        if not callable(executor):
            raise TypeError("executor must be callable")
        expected_gate_hash = _planning_gate_sha256(
            plan.min_evidence_mass,
            plan.max_normalized_entropy,
            plan.min_peak_probability,
            plan.min_predicted_success,
        )
        if plan.gate_config_sha256 != expected_gate_hash:
            raise ValueError("planning gate hash mismatch")
        before_hash = self.world_model.sha256
        if plan.world_model_sha256 != before_hash:
            raise ValueError("plan belongs to another world-model revision")
        if array_sha256(self.world_model.counts) != plan.counts_sha256:
            raise ValueError("plan transition counts are stale")
        policy = {
            (entry.remaining_horizon, entry.state): entry for entry in plan.policy
        }
        rewards = dict(plan.terminal_rewards)
        state_index = {
            state: index for index, state in enumerate(self.world_model.states)
        }
        current = plan.start_state
        remaining = plan.horizon
        steps: list[PlanExecutionStep] = []
        states = [current]
        reason = "horizon-exhausted"
        while remaining > 0 and rewards.get(current, 0.0) <= 0.0:
            entry = policy.get((remaining, current))
            if entry is None:
                reason = "policy-abstained"
                break
            prediction = self._prediction(
                current,
                entry.action,
                min_evidence_mass=plan.min_evidence_mass,
                max_normalized_entropy=plan.max_normalized_entropy,
                min_peak_probability=plan.min_peak_probability,
            )
            if (
                prediction.abstained
                or prediction.row_sha256 != entry.transition_row_sha256
            ):
                reason = "transition-became-uncertain"
                break
            outcome = executor(current, entry.action, len(steps))
            if not isinstance(outcome, ExecutionOutcome):
                raise TypeError("executor must return ExecutionOutcome")
            if outcome.target_state not in state_index:
                raise ValueError("executor returned a state outside the world model")
            probability = float(
                prediction.probabilities[state_index[outcome.target_state]]
            )
            transition_hash = _sha256(
                {
                    "plan_sha256": plan.sha256,
                    "ordinal": len(steps),
                    "source_state": current,
                    "action": entry.action,
                    "target_state": outcome.target_state,
                    "transition_row_sha256": prediction.row_sha256,
                    "verifier_sha256": outcome.verifier_sha256,
                    "evidence_sha256": outcome.evidence_sha256,
                }
            )
            step = PlanExecutionStep(
                ordinal=len(steps),
                source_state=current,
                action=entry.action,
                target_state=outcome.target_state,
                predicted_probability=probability,
                coverage=prediction.coverage,
                state_sha256=label_sha256("state", current),
                action_sha256=label_sha256("action", entry.action),
                transition_sha256=transition_hash,
                transition_row_sha256=prediction.row_sha256,
                verifier_sha256=outcome.verifier_sha256,
                evidence_sha256=outcome.evidence_sha256,
                verified=outcome.verified,
            )
            steps.append(step)
            current = outcome.target_state
            states.append(current)
            remaining -= 1
            if not outcome.verified:
                reason = "verifier-rejected"
                break
        else:
            reason = (
                "goal-reached"
                if rewards.get(current, 0.0) > 0.0
                else "horizon-exhausted"
            )
        all_verified = all(step.verified for step in steps)
        success = rewards.get(current, 0.0) > 0.0 and all_verified
        if success:
            reason = "goal-reached"
        receipt = PlanExecutionReceipt(
            plan_sha256=plan.sha256,
            world_model_sha256=before_hash,
            start_state=plan.start_state,
            goal_state=plan.goal_state,
            final_state=current,
            min_evidence_mass=plan.min_evidence_mass,
            max_normalized_entropy=plan.max_normalized_entropy,
            min_peak_probability=plan.min_peak_probability,
            min_predicted_success=plan.min_predicted_success,
            gate_config_sha256=plan.gate_config_sha256,
            predicted_success=plan.predicted_success,
            minimum_coverage=min((step.coverage for step in steps), default=1.0),
            success=success,
            reason=reason,
            steps=tuple(steps),
            state_hashes=tuple(label_sha256("state", state) for state in states),
            action_hashes=tuple(step.action_sha256 for step in steps),
            transition_hashes=tuple(step.transition_sha256 for step in steps),
            verifier_hashes=tuple(sorted({step.verifier_sha256 for step in steps})),
            evidence_hashes=tuple(sorted({step.evidence_sha256 for step in steps})),
        )
        if self.world_model.sha256 != before_hash:
            raise RuntimeError("plan execution mutated the world model")
        return receipt
