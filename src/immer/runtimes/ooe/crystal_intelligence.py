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
from .compute_crystals import (
    ComputeCrystal,
    ComputeCrystalBank,
    ComputeCrystalVM,
)
from .compute_graph import ComputeOperatorGraph, OperatorEdge
from .identity import canonical_json_bytes


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
        decision = graph.plan_route(states[0], states[-1])
        if decision.abstained or decision.plan is None:
            raise RuntimeError("verified affine route unexpectedly abstained")
        if decision.plan.finite_plan.horizon < len(crystals):
            raise RuntimeError("planned horizon cannot contain the affine route")
        charge = graph.charge_route(decision.plan)

        # Values are generated only after charging.  They were unavailable to
        # the compiler and therefore cannot be cached answers.
        unseen = np.asarray(
            rng.normal(size=(128, dimension)),
            dtype=np.float64,
        )
        primitive = ComputeCrystalVM(bank).execute(
            charge.route.primitive_program_sha256,
            unseen,
        )
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
            "primitive_route_length": len(charge.route.primitive_edge_sha256s),
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


def run_crystal_intelligence_benchmark(
    *,
    seed: int = 20_260_826,
) -> dict[str, Any]:
    compute = _compute_experiment(seed)
    algebra = _algebra_experiment()
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
    }
    body = {"algebra": algebra, "compute": compute, "seed": seed}
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
