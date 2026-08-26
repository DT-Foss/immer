"""End-to-end benchmark for compositional stored compute and map crystals."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .algebraic_crystals import (
    ACCEPTED,
    REJECTED,
    ExactGroupAccumulator,
    InvariantSupport,
    crystallize_map,
    fit_algebraic_map,
)
from .bvn_crystals import BirkhoffCrystalBank
from .bvn_search import (
    BehavioralElite,
    BehavioralMAPElites,
    ContextualThompsonMutation,
    PermutationAtom,
    fiedler_edge_novelty,
    mixture_from_theta,
)
from .compute_crystals import (
    ComputeCrystal,
    ComputeCrystalBank,
    ComputeCrystalVM,
    ComputeProgram,
)
from .compute_graph import ComputeOperatorGraph, OperatorEdge
from .consensus import barbell_adjacency
from .identity import canonical_json_bytes
from .math_core import spectral_gap, tv_contraction
from .residual_execution import ResidualRouteExecutor


CRYSTAL_INTELLIGENCE_SCHEMA = "immer-ooe-crystal-intelligence/v1"


def _hash(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _support(prefix: str, element: int, learned_phi: float) -> InvariantSupport:
    return InvariantSupport(
        element_sha256=_hash({"element": element, "prefix": prefix}),
        group_element=element,
        learned_phi=learned_phi,
        verifier_sha256=_hash({"kind": "verifier", "prefix": prefix}),
        evidence_sha256=_hash(
            {"element": element, "kind": "evidence", "prefix": prefix}
        ),
        source_receipt_sha256=_hash(
            {"element": element, "kind": "source", "prefix": prefix}
        ),
    )


def _edge(
    source: str,
    target: str,
    crystal: ComputeCrystal,
    ordinal: int,
) -> OperatorEdge:
    return OperatorEdge(
        source_state=source,
        target_state=target,
        crystal_sha256=crystal.sha256,
        verifier_sha256=_hash({"edge": ordinal, "kind": "verifier"}),
        evidence_sha256=_hash({"edge": ordinal, "kind": "evidence"}),
    )


def _compute_experiment(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    dimension = 6
    crystals = []
    for index in range(4):
        matrix = np.eye(dimension, dtype=np.float64)
        matrix += rng.normal(0.0, 0.025, size=(dimension, dimension))
        bias = rng.normal(0.0, 0.1, size=dimension)
        crystals.append(ComputeCrystal.affine(matrix, bias))
    states = ("raw", "centered", "mixed", "projected", "decoded")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "compute-bank"
        bank = ComputeCrystalBank(root)
        for crystal in crystals:
            bank.publish_crystal(crystal)
        graph = ComputeOperatorGraph(bank)
        graph.append_edges(
            tuple(
                _edge(states[index], states[index + 1], crystal, index)
                for index, crystal in enumerate(crystals)
            )
        )
        prefix_decision = graph.plan_route(states[0], states[2])
        if prefix_decision.abstained or prefix_decision.plan is None:
            raise RuntimeError("verified affine prefix unexpectedly abstained")
        prefix_charge = graph.charge_route(prefix_decision.plan)

        decision = graph.plan_route(states[0], states[-1])
        if decision.abstained or decision.plan is None:
            raise RuntimeError("verified affine route unexpectedly abstained")
        if decision.plan.finite_plan.horizon < len(crystals):
            raise RuntimeError("planned horizon cannot contain the affine route")
        # Values are generated only after charging.  They were unavailable to
        # the compiler and therefore cannot be cached answers.
        unseen = np.asarray(
            rng.normal(size=(128, dimension)),
            dtype=np.float64,
        )
        primitive_program = ComputeProgram.compose(crystals)
        bank.publish_program(primitive_program)
        primitive = ComputeCrystalVM(bank).execute(primitive_program, unseen)
        residual = ResidualRouteExecutor(graph).execute(decision.plan, unseen)
        residual_delta = float(np.max(np.abs(primitive.output - residual.output)))
        if residual_delta > 1e-12:
            raise RuntimeError("charged prefix changed residual unseen outputs")

        charge = graph.charge_route(decision.plan)
        warm = graph.discharge(states[0], states[-1], unseen)
        max_delta = float(np.max(np.abs(primitive.output - warm.output)))
        if max_delta > 1e-12:
            raise RuntimeError("charged affine segment changed unseen outputs")

        reopened_bank = ComputeCrystalBank(
            root,
            trusted_manifest_sha256=bank.current_anchor_sha256(),
        )
        reopened = ComputeOperatorGraph(
            reopened_bank,
            trusted_graph_state_sha256=graph.state().sha256,
        )
        replay = reopened.discharge(states[0], states[-1], unseen)
        if not np.array_equal(replay.output, warm.output):
            raise RuntimeError("reopened affine route changed its exact output")

        placebo_root = Path(temporary) / "placebo-bank"
        placebo_bank = ComputeCrystalBank(placebo_root)
        for crystal in crystals:
            placebo_bank.publish_crystal(crystal)
        placebo_graph = ComputeOperatorGraph(placebo_bank)
        placebo_graph.append_edges(
            (
                _edge("raw", "centered", crystals[0], 100),
                _edge("centered", "mixed", crystals[1], 101),
                _edge("decoy", "projected", crystals[2], 102),
                _edge("projected", "decoded", crystals[3], 103),
            )
        )
        placebo = placebo_graph.plan_route("raw", "decoded")
        return {
            "charge_receipt_sha256": charge.sha256,
            "charge_basis_sha256": charge.route.charge_basis_sha256,
            "cold_operator_count": primitive.receipt.executed_operator_count,
            "equivalent_source_work": (
                warm.receipt.equivalent_source_work_units
            ),
            "graph_generation": graph.state().generation,
            "historical_work_released": (
                warm.receipt.historical_work_released
            ),
            "live_operator_count": warm.receipt.vm_receipt.executed_operator_count,
            "live_work": warm.receipt.live_work_units,
            "max_absolute_delta": max_delta,
            "multi_step_teacher_labels": 0,
            "placebo_abstained": placebo.abstained,
            "prefix_charge_receipt_sha256": prefix_charge.sha256,
            "primitive_route_length": len(charge.route.primitive_edge_sha256s),
            "residual_historical_work_released": (
                residual.receipt.historical_work_released
            ),
            "residual_live_operator_count": sum(
                receipt.executed_operator_count
                for receipt in (
                    residual.receipt.prefix_execution_receipt,
                    residual.receipt.suffix_execution_receipt,
                )
                if receipt is not None
            ),
            "residual_max_absolute_delta": residual_delta,
            "residual_prefix_length": residual.receipt.prefix_length,
            "residual_suffix_length": residual.receipt.residual_length,
            "residual_receipt_sha256": residual.receipt.sha256,
            "reopen_receipt_sha256": replay.receipt.sha256,
            "unseen_batch": int(unseen.shape[0]),
            "warm_receipt_sha256": warm.receipt.sha256,
        }


def _algebra_experiment() -> dict[str, Any]:
    additive_support = tuple(
        _support("additive", value, 1.75 * value - 0.4 + 0.001 * math.sin(value))
        for value in range(1, 10)
    )
    multiplicative_support = tuple(
        _support(
            "multiplicative",
            value,
            2.2 * math.log(value) + 0.7 + 0.001 * math.cos(value),
        )
        for value in range(1, 10)
    )
    cyclic_support = tuple(
        _support(
            "cyclic",
            value,
            float(value % 3) + 0.001 * math.sin(value),
        )
        for value in range(1, 13)
    )
    additive_fit = fit_algebraic_map(additive_support)
    multiplicative_fit = fit_algebraic_map(multiplicative_support)
    cyclic_fit = fit_algebraic_map(cyclic_support, candidate_cyclic_orders=(3,))
    for receipt in (additive_fit, multiplicative_fit, cyclic_fit):
        if receipt.outcome != ACCEPTED:
            raise RuntimeError("genuine algebraic map failed admission")

    additive = crystallize_map(
        additive_fit,
        deployment_min=-1_000,
        deployment_max=1_000,
    )
    multiplicative = crystallize_map(
        multiplicative_fit,
        deployment_min=1,
        deployment_max=1_000,
    )
    cyclic = crystallize_map(cyclic_fit, deployment_min=0, deployment_max=2)
    additive_values = (7, -3, 11, 5, -2, 9, 4, -6, 8, 3, 2, 1)
    multiplicative_values = (2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1)
    cyclic_values = (1, 2, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2)
    additive_run = ExactGroupAccumulator(additive).snap_and_execute(
        tuple(additive.learned_phi_of(value) + 1e-5 for value in additive_values)
    )
    multiplicative_run = ExactGroupAccumulator(multiplicative).snap_and_execute(
        tuple(
            multiplicative.learned_phi_of(value) + 1e-5
            for value in multiplicative_values
        )
    )
    cyclic_run = ExactGroupAccumulator(cyclic).snap_and_execute(
        tuple(cyclic.learned_phi_of(value) + 1e-5 for value in cyclic_values)
    )
    placebo_phi = tuple(item.learned_phi for item in additive_support)
    permutation = (4, 1, 7, 0, 6, 2, 8, 3, 5)
    placebo_support = tuple(
        _support("placebo", index + 1, placebo_phi[permutation[index]])
        for index in range(9)
    )
    placebo = fit_algebraic_map(placebo_support)
    if placebo.outcome != REJECTED:
        raise RuntimeError("permuted algebraic placebo was not rejected")
    return {
        "additive": {
            "family": additive.family,
            "fit_score": additive_fit.best_score,
            "length": len(additive_values),
            "result": additive_run.result,
            "expected": sum(additive_values),
            "receipt_sha256": additive_run.sha256,
        },
        "multiplicative": {
            "family": multiplicative.family,
            "fit_score": multiplicative_fit.best_score,
            "length": len(multiplicative_values),
            "result": multiplicative_run.result,
            "expected": math.prod(multiplicative_values),
            "receipt_sha256": multiplicative_run.sha256,
        },
        "cyclic": {
            "family": cyclic.family,
            "fit_score": cyclic_fit.best_score,
            "length": len(cyclic_values),
            "result": cyclic_run.result,
            "expected": sum(cyclic_values) % 3,
            "receipt_sha256": cyclic_run.sha256,
        },
        "placebo": {
            "best_score": placebo.best_score,
            "outcome": placebo.outcome,
            "receipt_sha256": placebo.sha256,
        },
    }


def _operator_search_experiment(seed: int) -> dict[str, Any]:
    """Exercise constructive operator space, diversity, and novelty learning."""

    rng = np.random.default_rng(seed ^ 0xB17C0FFEE)
    dimension = 8
    atoms = tuple(
        PermutationAtom(
            tuple((index + shift) % dimension for index in range(dimension))
        )
        for shift in range(dimension)
    )
    theta = rng.normal(0.0, 1.25, size=len(atoms))
    mixture = mixture_from_theta(theta, atoms)
    kernel = mixture.reconstruct()
    with tempfile.TemporaryDirectory() as temporary:
        bank = ComputeCrystalBank(Path(temporary) / "bvn-bank")
        basis = BirkhoffCrystalBank(bank)
        publication = basis.publish(
            kernel,
            verifier_sha256=_hash({"kind": "bvn-verifier", "seed": seed}),
            evidence_sha256s=tuple(
                sorted(
                    (
                        _hash({"kind": "bvn-evidence", "replica": 0, "seed": seed}),
                        _hash({"kind": "bvn-evidence", "replica": 1, "seed": seed}),
                    )
                )
            ),
        )
        future = rng.normal(size=(96, dimension)).astype(np.float64)
        atom_output = basis.apply_atoms(publication.receipt.sha256, future)
        expected = np.einsum("...i,ij->...j", future, kernel, optimize=False)
        basis_delta = float(np.max(np.abs(atom_output - expected)))

    archive = BehavioralMAPElites(
        (4, 4, 4),
        objective_count=2,
        max_elites_per_cell=2,
    )
    for ordinal in range(64):
        candidate_theta = rng.normal(0.0, 1.5, size=len(atoms))
        candidate = mixture_from_theta(candidate_theta, atoms)
        candidate_kernel = candidate.reconstruct()
        gap = spectral_gap(candidate_kernel)
        contraction = tv_contraction(candidate_kernel)
        dominant = int(np.argmax(candidate.weights)) % 4
        descriptor = (
            dominant,
            min(3, int(max(0.0, gap) * 4.0)),
            min(3, candidate.component_count // 2),
        )
        archive.add(
            BehavioralElite(
                candidate_sha256=_hash(
                    {
                        "kernel_sha256": candidate.kernel_sha256,
                        "ordinal": ordinal,
                        "seed": seed,
                    }
                ),
                descriptor=descriptor,
                objectives=(gap, 1.0 - contraction),
            )
        )

    adjacency = barbell_adjacency(6, 6, bridge_weight=0.05)
    missing = []
    direct = []
    for left in range(adjacency.shape[0]):
        for right in range(left + 1, adjacency.shape[0]):
            receipt = fiedler_edge_novelty(adjacency, left, right)
            target = direct if receipt.direct_edge else missing
            target.append(receipt)
    best_missing = max(missing, key=lambda row: (row.priority(1.0), row.sha256))
    best_direct = max(direct, key=lambda row: (row.priority(1.0), row.sha256))

    bandit = ContextualThompsonMutation(
        ("compose", "permute", "rescale"),
        _hash({"kind": "mutation-bandit", "seed": seed}),
    )
    choices = []
    for _ in range(96):
        choice, bandit = bandit.choose("qwen.operator-graph")
        success = choice.arm == "permute"
        bandit = bandit.observe(
            "qwen.operator-graph", choice.arm, success=success
        )
        choices.append(choice.arm)
    late_window = choices[-32:]
    preferred = sum(value == "permute" for value in late_window) / len(late_window)
    return {
        "archive_coverage": archive.coverage,
        "archive_elite_count": archive.elite_count,
        "archive_occupied_cells": archive.occupied_cells,
        "archive_sha256": archive.sha256,
        "basis_component_count": len(publication.receipt.atom_crystal_sha256s),
        "basis_max_absolute_delta": basis_delta,
        "basis_receipt_sha256": publication.receipt.sha256,
        "basis_storage_bound": (dimension - 1) ** 2 + 1,
        "fiedler_best_direct_priority": best_direct.priority(1.0),
        "fiedler_best_missing_edge": [
            best_missing.left_index,
            best_missing.right_index,
        ],
        "fiedler_best_missing_priority": best_missing.priority(1.0),
        "fiedler_receipt_sha256": best_missing.sha256,
        "mutation_late_preferred_fraction": preferred,
        "mutation_state_sha256": bandit.sha256,
    }


def run_crystal_intelligence_benchmark(
    *,
    seed: int = 20_260_826,
) -> dict[str, Any]:
    compute = _compute_experiment(seed)
    algebra = _algebra_experiment()
    operator_search = _operator_search_experiment(seed)
    headline = {
        "algebraic_families_admitted": sum(
            algebra[name]["fit_score"] >= 0.95
            for name in ("additive", "multiplicative", "cyclic")
        ),
        "algebraic_length_12_exact": all(
            algebra[name]["result"] == algebra[name]["expected"]
            for name in ("additive", "multiplicative", "cyclic")
        ),
        "compute_exact": compute["max_absolute_delta"] <= 1e-12,
        "historical_work_released": compute["historical_work_released"],
        "live_operator_count": compute["live_operator_count"],
        "placebos_rejected": (
            compute["placebo_abstained"]
            and algebra["placebo"]["outcome"] == REJECTED
        ),
        "primitive_route_length": compute["primitive_route_length"],
        "residual_exact": compute["residual_max_absolute_delta"] <= 1e-12,
        "residual_prefix_length": compute["residual_prefix_length"],
        "residual_suffix_length": compute["residual_suffix_length"],
        "bvn_basis_exact": operator_search["basis_max_absolute_delta"] <= 1e-12,
        "bvn_basis_within_bound": (
            operator_search["basis_component_count"]
            <= operator_search["basis_storage_bound"]
        ),
        "fiedler_prefers_missing_bridge": (
            operator_search["fiedler_best_missing_priority"]
            > operator_search["fiedler_best_direct_priority"]
        ),
        "map_elites_occupied_cells": operator_search["archive_occupied_cells"],
        "mutation_preference_learned": (
            operator_search["mutation_late_preferred_fraction"] >= 0.75
        ),
    }
    body = {
        "algebra": algebra,
        "compute": compute,
        "operator_search": operator_search,
        "seed": seed,
    }
    report_body = {"body": body, "headline": headline}
    return {
        **report_body,
        "schema": CRYSTAL_INTELLIGENCE_SCHEMA,
        "sha256": _hash(report_body),
    }


__all__ = [
    "CRYSTAL_INTELLIGENCE_SCHEMA",
    "run_crystal_intelligence_benchmark",
]
