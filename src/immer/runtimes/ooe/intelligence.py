"""Deterministic mechanism benchmark for emergent Markov-OoE intelligence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np

from .consensus import barbell_adjacency, complete_adjacency
from .crystal_intelligence import run_crystal_intelligence_benchmark
from .identity import canonical_json_bytes
from .mpo import factorize_action_transitions
from .novelty import HopfieldNoveltyModel
from .options import (
    OptionKernelCatalog,
    VerifiedTrajectory,
    discover_macro_options,
)
from .planning import ExecutionOutcome, FiniteHorizonPlanner
from .reservoir_intelligence import run_reservoir_intelligence_benchmark
from .world_model import (
    ActionConditionedWorldModel,
    RegimeChangeSignal,
    TransitionEvidence,
)


INTELLIGENCE_BENCHMARK_SCHEMA = "immer-ooe-intelligence-benchmark/v1"


def _hash(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _positive_int(value: int, *, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must lie in [1, {maximum}]")
    return value


@dataclass(frozen=True, slots=True)
class ModularPlanningTask:
    start: int
    goal: int
    witness: tuple[str, ...]


class ModularOperatorWorld:
    """Prime-ring operator world with no multi-step teacher labels."""

    actions = ("add1", "add3", "mul2", "affine")

    def __init__(self, state_size: int = 17) -> None:
        size = _positive_int(state_size, name="state_size", maximum=257)
        if size < 5:
            raise ValueError("state_size must be at least five")
        self.state_size = size
        self.states = tuple(f"s{index}" for index in range(size))

    def step(self, state: int, action: str) -> int:
        if action == "add1":
            value = state + 1
        elif action == "add3":
            value = state + 3
        elif action == "mul2":
            value = 2 * state
        elif action == "affine":
            value = 4 * state + 2
        else:
            raise KeyError(f"unknown modular action: {action}")
        return value % self.state_size

    def action_tensor(self) -> np.ndarray:
        tensor = np.zeros(
            (len(self.actions), self.state_size, self.state_size),
            dtype=np.float64,
        )
        for action_index, action in enumerate(self.actions):
            for state in range(self.state_size):
                tensor[action_index, state, self.step(state, action)] = 1.0
        return tensor

    def tasks(
        self,
        *,
        count: int,
        min_length: int,
        max_length: int,
        seed: int,
    ) -> tuple[ModularPlanningTask, ...]:
        total = _positive_int(count, name="count", maximum=100_000)
        lower = _positive_int(min_length, name="min_length", maximum=128)
        upper = _positive_int(max_length, name="max_length", maximum=128)
        if upper < lower:
            raise ValueError("max_length must not be smaller than min_length")
        rng = np.random.default_rng(seed)
        result = []
        for _ in range(total):
            start = int(rng.integers(self.state_size))
            length = int(rng.integers(lower, upper + 1))
            witness = tuple(
                str(value)
                for value in rng.choice(np.asarray(self.actions), size=length)
            )
            goal = start
            for action in witness:
                goal = self.step(goal, action)
            result.append(ModularPlanningTask(start, goal, witness))
        return tuple(result)


def _transition(
    world: ModularOperatorWorld,
    source: int,
    action: str,
    *,
    target_action: str | None = None,
    prefix: str,
) -> TransitionEvidence:
    actual = action if target_action is None else target_action
    identity = f"{prefix}:{source}:{action}:{actual}"
    return TransitionEvidence(
        source_state=world.states[source],
        action=action,
        target_state=world.states[world.step(source, actual)],
        weight=2.0,
        verifier_sha256=_hash({"kind": "verifier", "identity": identity}),
        evidence_sha256=_hash({"kind": "evidence", "identity": identity}),
    )


def _models(
    world: ModularOperatorWorld,
    *,
    replicas: int,
) -> tuple[list[ActionConditionedWorldModel], ActionConditionedWorldModel]:
    count = _positive_int(replicas, name="replicas", maximum=64)
    if count < 4 or count % 2:
        raise ValueError("replicas must be even and at least four")
    distributed = [
        ActionConditionedWorldModel(world.states, world.actions) for _ in range(count)
    ]
    central = ActionConditionedWorldModel(world.states, world.actions)
    for action_index, action in enumerate(world.actions):
        for source in range(world.state_size):
            evidence = _transition(world, source, action, prefix="one-step")
            distributed[(source + 3 * action_index) % count].observe(evidence)
            # Central baseline receives an independently identified copy.  It is
            # never used by PS-Lifted fusion or its evidence accounting.
            central.observe(
                _transition(world, source, action, prefix="central-one-step")
            )
    return distributed, central


def _evaluate_planner(
    planner: FiniteHorizonPlanner,
    world: ModularOperatorWorld,
    tasks: tuple[ModularPlanningTask, ...],
    *,
    label: str,
) -> dict[str, Any]:
    planned = 0
    success = 0
    total_steps = 0
    predicted = []
    for task_index, task in enumerate(tasks):
        decision = planner.plan_goal(
            world.states[task.start],
            world.states[task.goal],
            horizon=len(task.witness),
        )
        if decision.abstained or decision.plan is None:
            continue
        planned += 1
        predicted.append(decision.predicted_success)

        def execute(source: str, action: str, ordinal: int) -> ExecutionOutcome:
            source_index = int(source[1:])
            target = world.states[world.step(source_index, action)]
            return ExecutionOutcome(
                target_state=target,
                verifier_sha256=_hash(
                    {
                        "kind": "live-verifier",
                        "label": label,
                        "task": task_index,
                        "ordinal": ordinal,
                    }
                ),
                evidence_sha256=_hash(
                    {
                        "kind": "live-evidence",
                        "label": label,
                        "task": task_index,
                        "ordinal": ordinal,
                        "source": source,
                        "action": action,
                        "target": target,
                    }
                ),
            )

        receipt = planner.execute(decision.plan, execute)
        success += int(receipt.success)
        total_steps += len(receipt.steps)
    return {
        "label": label,
        "tasks": len(tasks),
        "planned": planned,
        "success": success,
        "planning_coverage": planned / len(tasks),
        "success_rate": success / len(tasks),
        "conditional_success": success / planned if planned else 0.0,
        "executed_steps": total_steps,
        "mean_predicted_success": sum(predicted) / len(predicted) if predicted else 0.0,
    }


def _placebo_model(world: ModularOperatorWorld) -> ActionConditionedWorldModel:
    shuffled = ActionConditionedWorldModel(world.states, world.actions)
    permutation = dict(zip(world.actions, world.actions[1:] + world.actions[:1], strict=True))
    for action in world.actions:
        for source in range(world.state_size):
            shuffled.observe(
                _transition(
                    world,
                    source,
                    action,
                    target_action=permutation[action],
                    prefix="shuffled-action-placebo",
                )
            )
    return shuffled


def _aggregate_local_replicas(
    results: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    if not results:
        raise ValueError("local replica results must be non-empty")
    tasks = sum(int(result["tasks"]) for result in results)
    planned = sum(int(result["planned"]) for result in results)
    success = sum(int(result["success"]) for result in results)
    predicted_mass = sum(
        float(result["mean_predicted_success"]) * int(result["planned"])
        for result in results
    )
    return {
        "label": "local-replica-cohort",
        "replica_count": len(results),
        "tasks": tasks,
        "tasks_per_replica": int(results[0]["tasks"]),
        "planned": planned,
        "success": success,
        "planning_coverage": planned / tasks,
        "success_rate": success / tasks,
        "conditional_success": success / planned if planned else 0.0,
        "executed_steps": sum(int(result["executed_steps"]) for result in results),
        "mean_predicted_success": predicted_mass / planned if planned else 0.0,
        "replicas": list(results),
    }


def _one_step_no_memory_success(
    world: ModularOperatorWorld,
    tasks: tuple[ModularPlanningTask, ...],
) -> dict[str, Any]:
    success = 0
    for task in tasks:
        reachable = task.start == task.goal or any(
            world.step(task.start, action) == task.goal for action in world.actions
        )
        success += int(reachable)
    return {
        "tasks": len(tasks),
        "success": success,
        "success_rate": success / len(tasks),
        "memory_steps": 0,
    }


def _option_experiment() -> dict[str, Any]:
    states = 16
    advance = np.zeros((states, states), dtype=np.float64)
    reset = np.zeros((states, states), dtype=np.float64)
    for state in range(states):
        advance[state, min(state + 1, states - 1)] = 1.0
        reset[state, 0] = 1.0
    kernels = {"advance": advance, "reset": reset}
    world_hash = _hash({"world": "corridor", "states": states})
    graph_hash = _hash({"graph": "corridor-v1"})
    verifiers = {"trajectory": _hash({"verifier": "corridor"})}
    trajectories = []
    for start in (0, 1, 2, 3):
        actions = ("advance",) * 8
        trajectory_states = tuple(range(start, start + 9))
        trajectories.append(
            VerifiedTrajectory.create(
                states=trajectory_states,
                actions=actions,
                source_world_model_sha256=world_hash,
                graph_revision_sha256=graph_hash,
                verifier_hashes=verifiers,
                verification_receipt_sha256=_hash(
                    {"trajectory_receipt": start}
                ),
                outcome_sha256=_hash({"trajectory_outcome": start}),
            )
        )
    discovered = discover_macro_options(
        tuple(trajectories),
        kernels,
        min_support=4,
        min_option_length=2,
        max_option_length=8,
    )
    primitive_catalog = OptionKernelCatalog(kernels)
    option_catalog = OptionKernelCatalog(kernels, discovered.options)
    primitive = primitive_catalog.plan(0, 8, include_options=False)
    hierarchical = option_catalog.plan(0, 8, include_options=True)
    if primitive is None or hierarchical is None:
        raise RuntimeError("corridor option experiment became unreachable")
    contraction = option_catalog.contraction_ledger(hierarchical)
    return {
        "verified_trajectories": len(trajectories),
        "discovered_options": len(discovered.options),
        "discovery_receipt_sha256": discovered.receipt.sha256,
        "primitive_operator_depth": primitive.operator_depth,
        "hierarchical_operator_depth": hierarchical.operator_depth,
        "primitive_length": hierarchical.primitive_length,
        "depth_reduction": primitive.operator_depth - hierarchical.operator_depth,
        "hierarchical_success_probability": hierarchical.success_probability,
        "contraction_ledger_sha256": contraction.sha256,
        "contraction_depth": contraction.depth,
        "contraction_tau_upper_bound": contraction.tau_upper_bound,
        "used_option_sha256s": [
            value.removeprefix("option:")
            for value in hierarchical.operator_ids
            if value.startswith("option:")
        ],
    }


def _mpo_experiment(seed: int) -> dict[str, Any]:
    local_kernels = (
        np.eye(4, dtype=np.float64),
        np.roll(np.eye(4, dtype=np.float64), 1, axis=1),
    )
    structured = np.stack(
        [
            np.kron(local_kernels[action % 2], local_kernels[(action // 2) % 2])
            for action in range(4)
        ]
    )
    model_hash = _hash({"world": "structured-mpo"})
    graph_hash = _hash({"graph": "structured-mpo"})
    verifier_hashes = {"mpo": _hash({"verifier": "mpo"})}
    compressed = factorize_action_transitions(
        structured,
        action_ids=("a0", "a1", "a2", "a3"),
        state_shape=(4, 4),
        source_world_model_sha256=model_hash,
        graph_revision_sha256=graph_hash,
        verifier_hashes=verifier_hashes,
        max_rank=4,
        relative_tolerance=1e-12,
    )
    rng = np.random.default_rng(seed)
    raw = rng.random((4, 16, 16))
    unstructured = raw / raw.sum(axis=2, keepdims=True)
    fallback = factorize_action_transitions(
        unstructured,
        action_ids=("a0", "a1", "a2", "a3"),
        state_shape=(4, 4),
        source_world_model_sha256=_hash({"world": "unstructured-mpo"}),
        graph_revision_sha256=graph_hash,
        verifier_hashes=verifier_hashes,
        max_rank=1,
        relative_tolerance=1e-14,
    )
    return {
        "structured": {
            "mode": compressed.receipt.mode,
            "dense_numeric_bytes": compressed.receipt.dense_numeric_bytes,
            "stored_numeric_bytes": compressed.receipt.stored_numeric_bytes,
            "compression_ratio": float.fromhex(
                compressed.receipt.compression_ratio_hex
            ),
            "relative_error": float.fromhex(
                compressed.receipt.reconstruction_relative_error_hex
            ),
            "receipt_sha256": compressed.receipt.sha256,
        },
        "unstructured": {
            "mode": fallback.receipt.mode,
            "selection_reason": fallback.receipt.selection_reason,
            "exact_fallback": fallback.receipt.exact_fallback,
            "receipt_sha256": fallback.receipt.sha256,
        },
    }


def _novelty_experiment() -> dict[str, Any]:
    model = HopfieldNoveltyModel(beta=8.0)
    for value in ([1.0, 0.05, 0.0], [1.0, -0.05, 0.0], [0.98, 0.1, 0.0]):
        model.observe("alpha", value)
    for value in ([0.0, 1.0, 0.05], [0.05, 1.0, 0.0], [-0.05, 0.98, 0.0]):
        model.observe("beta", value)
    receipt = model.calibrate()
    in_distribution = (
        model.decision([1.0, 0.0, 0.0]),
        model.decision([0.0, 1.0, 0.0]),
    )
    out_of_distribution = (
        model.decision([1.0, 1.0, 0.0]),
        model.decision([0.0, 0.0, 1.0]),
        model.decision([-1.0, -1.0, 0.0]),
    )
    return {
        "calibration_sha256": receipt.sha256,
        "id_accepts": sum(value.accepted for value in in_distribution),
        "id_total": len(in_distribution),
        "ood_false_accepts": sum(value.accepted for value in out_of_distribution),
        "ood_total": len(out_of_distribution),
    }


def _regime_experiment() -> dict[str, Any]:
    states = ("root", "old", "new")
    model = ActionConditionedWorldModel(states, ("choose",))
    for index in range(20):
        model.observe(
            TransitionEvidence(
                "root",
                "choose",
                "old",
                1.0,
                _hash({"old-verifier": index}),
                _hash({"old-evidence": index}),
                epoch=0,
            )
        )
    stale = model.clone()
    change = model.apply_regime_change(
        RegimeChangeSignal(
            signal_sha256=_hash({"change": "old-to-new"}),
            verifier_sha256=_hash({"change-verifier": "old-to-new"}),
            strength=0.95,
            retention=0.05,
            epoch=1,
            name="old-to-new",
        )
    )
    for index in range(5):
        evidence = TransitionEvidence(
            "root",
            "choose",
            "new",
            1.0,
            _hash({"new-verifier": index}),
            _hash({"new-evidence": index}),
            epoch=1,
        )
        model.observe(evidence)
        stale.observe(evidence)
    adapted = model.predict("root", "choose")
    unadapted = stale.predict("root", "choose")
    new_index = states.index("new")
    return {
        "regime_receipt_sha256": change.sha256,
        "adapted_new_probability": adapted.probabilities[new_index],
        "unadapted_new_probability": unadapted.probabilities[new_index],
        "adaptation_gain": (
            adapted.probabilities[new_index] - unadapted.probabilities[new_index]
        ),
        "regime_revision": model.regime_revision,
    }


def run_intelligence_benchmark(
    *,
    seed: int = 20_260_826,
    task_count: int = 250,
    state_size: int = 17,
    replicas: int = 12,
) -> dict[str, Any]:
    """Run the bounded learning/planning/abstraction/adaptation matrix."""

    total_tasks = _positive_int(task_count, name="task_count", maximum=100_000)
    world = ModularOperatorWorld(state_size)
    distributed, central = _models(world, replicas=replicas)
    fused = ActionConditionedWorldModel.fuse_ps_lifted(
        distributed,
        barbell_adjacency(replicas // 2, replicas // 2),
        tolerance=1e-9,
        max_rounds=4096,
        topology="intelligence-barbell-ps-lifted",
    )
    complete_fused = ActionConditionedWorldModel.fuse_ps_lifted(
        distributed,
        complete_adjacency(replicas),
        tolerance=1e-9,
        max_rounds=4096,
        topology="intelligence-complete-ps-lifted",
    )
    tasks = world.tasks(
        count=total_tasks,
        min_length=4,
        max_length=10,
        seed=seed,
    )
    local_replica_results = tuple(
        _evaluate_planner(
            FiniteHorizonPlanner(model, min_predicted_success=0.1),
            world,
            tasks,
            label=f"local-replica-{index}",
        )
        for index, model in enumerate(distributed)
    )
    local_result = _aggregate_local_replicas(local_replica_results)
    ps_result = _evaluate_planner(
        FiniteHorizonPlanner(fused.model, min_predicted_success=0.99),
        world,
        tasks,
        label="ps-lifted",
    )
    central_result = _evaluate_planner(
        FiniteHorizonPlanner(central, min_predicted_success=0.99),
        world,
        tasks,
        label="central-table",
    )
    topology_shift_result = _evaluate_planner(
        FiniteHorizonPlanner(complete_fused.model, min_predicted_success=0.99),
        world,
        tasks,
        label="complete-topology-shift",
    )
    placebo = _placebo_model(world)
    placebo_result = _evaluate_planner(
        FiniteHorizonPlanner(placebo, min_predicted_success=0.99),
        world,
        tasks,
        label="shuffled-action-placebo",
    )
    option_result = _option_experiment()
    mpo_result = _mpo_experiment(seed + 1)
    novelty_result = _novelty_experiment()
    regime_result = _regime_experiment()
    reservoir_result = run_reservoir_intelligence_benchmark(seed=seed)
    crystal_result = run_crystal_intelligence_benchmark(seed=seed)
    body = {
        "central_table": central_result,
        "crystal_compute": crystal_result,
        "ablations": {
            "no_memory": _one_step_no_memory_success(world, tasks),
            "no_options_operator_depth": option_result["primitive_operator_depth"],
            "no_novelty_ood_false_accepts": 3,
        },
        "fusion": {
            "adjacency_sha256": fused.receipt.adjacency_sha256,
            "consensus_rounds": fused.receipt.consensus.rounds,
            "evidence_events": fused.model.total_events,
            "fused_world_model_sha256": fused.model.sha256,
            "receipt_sha256": fused.receipt.sha256,
            "topology": fused.receipt.topology,
        },
        "local_replica": local_result,
        "mpo": mpo_result,
        "novelty": novelty_result,
        "one_step_teacher_observations": len(world.actions) * world.state_size,
        "options": option_result,
        "placebo": placebo_result,
        "ps_lifted": ps_result,
        "regime": regime_result,
        "reservoir": reservoir_result,
        "seed": seed,
        "task_contract": {
            "count": total_tasks,
            "maximum_witness_length": 10,
            "minimum_witness_length": 4,
            "multi_step_teacher_labels": 0,
        },
        "world": {
            "actions": list(world.actions),
            "state_size": world.state_size,
        },
        "topology_shift": {
            "barbell_receipt_sha256": fused.receipt.sha256,
            "barbell_rounds": fused.receipt.consensus.rounds,
            "complete_receipt_sha256": complete_fused.receipt.sha256,
            "complete_rounds": complete_fused.receipt.consensus.rounds,
            "counts_max_absolute_delta": float(
                np.max(np.abs(fused.model.counts - complete_fused.model.counts))
            ),
            "complete_success_rate": topology_shift_result["success_rate"],
        },
    }
    headline = {
        "local_success_rate": local_result["success_rate"],
        "ps_lifted_success_rate": ps_result["success_rate"],
        "central_success_rate": central_result["success_rate"],
        "shuffled_placebo_success_rate": placebo_result["success_rate"],
        "ps_minus_local": ps_result["success_rate"] - local_result["success_rate"],
        "ps_minus_placebo": (
            ps_result["success_rate"] - placebo_result["success_rate"]
        ),
        "option_depth_reduction": body["options"]["depth_reduction"],
        "mpo_compression_ratio": body["mpo"]["structured"]["compression_ratio"],
        "ood_false_accepts": body["novelty"]["ood_false_accepts"],
        "regime_adaptation_gain": body["regime"]["adaptation_gain"],
        "topology_shift_success_rate": body["topology_shift"][
            "complete_success_rate"
        ],
        "no_memory_success_rate": body["ablations"]["no_memory"][
            "success_rate"
        ],
        "reservoir_fused_accuracy": reservoir_result["headline"][
            "fused_accuracy"
        ],
        "reservoir_fused_minus_no_memory": reservoir_result["headline"][
            "fused_minus_no_memory"
        ],
        "reservoir_fused_minus_placebo": reservoir_result["headline"][
            "fused_minus_placebo"
        ],
        "compute_crystal_live_operators": crystal_result["headline"][
            "live_operator_count"
        ],
        "compute_crystal_route_length": crystal_result["headline"][
            "primitive_route_length"
        ],
        "compute_crystal_historical_work_released": crystal_result["headline"][
            "historical_work_released"
        ],
        "algebraic_crystals_length_12_exact": crystal_result["headline"][
            "algebraic_length_12_exact"
        ],
    }
    report_body = {"body": body, "headline": headline}
    return {
        **report_body,
        "schema": INTELLIGENCE_BENCHMARK_SCHEMA,
        "sha256": _hash(report_body),
    }


__all__ = [
    "INTELLIGENCE_BENCHMARK_SCHEMA",
    "ModularOperatorWorld",
    "ModularPlanningTask",
    "run_intelligence_benchmark",
]
