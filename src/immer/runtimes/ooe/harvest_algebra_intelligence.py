"""Fixed contextual-routing trial over harvested executable operator families."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .algebra_agents import AlgebraRouterState
from .compute_crystals import (
    AFFINE_FLOAT64,
    MARKOV_FLOAT64,
    PERMUTATION,
    ComputeCrystal,
    ComputeCrystalBank,
)
from .compute_graph import ComputeOperatorGraph, OperatorEdge
from .harvest_algebra_bridge import (
    HarvestedCandidateProfile,
    execute_and_observe_compute_candidate,
    harvest_promotion_to_algebra_candidate,
)
from .identity import canonical_json_bytes
from .operator_harvester import CandidateEvidenceStream, HarvestPromotion


HARVEST_ALGEBRA_INTELLIGENCE_SCHEMA = "immer-ooe-harvest-algebra-intelligence/v1"


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class _ExpectedArrayVerifier:
    def __init__(self, expected: np.ndarray, verifier_sha256: str) -> None:
        self.expected = expected
        self.verifier_sha256 = verifier_sha256

    def verify(self, candidate, execution):
        accepted = bool(np.array_equal(execution.output, self.expected))
        return accepted, canonical_json_bytes(
            {
                "accepted": accepted,
                "candidate_sha256": candidate.sha256,
                "output_sha256": execution.receipt.output_sha256,
                "schema": "immer-ooe-harvest-algebra-benchmark-verifier/v1",
            }
        )


def _catalog(root: Path):
    bank = ComputeCrystalBank(root)
    graph = ComputeOperatorGraph(bank)
    families = ("double", "swap", "mix")
    crystals = (
        ComputeCrystal.affine(
            np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64),
            np.array([1.0, 1.0], dtype=np.float64),
        ),
        ComputeCrystal.permutation((1, 0)),
        ComputeCrystal.markov(np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float64)),
    )
    kinds = (AFFINE_FLOAT64, PERMUTATION, MARKOV_FLOAT64)
    publications = tuple(bank.publish_crystal(crystal) for crystal in crystals)
    discovery = tuple(_hash(f"{family}:discovery") for family in families)
    evidence = tuple(_hash(f"{family}:evidence") for family in families)
    edges = tuple(
        OperatorEdge(
            source_state=f"context/{family}/input",
            target_state=f"context/{family}/output",
            crystal_sha256=crystal.sha256,
            verifier_sha256=verifier,
            evidence_sha256=evidence_sha,
            weight=4.0,
        )
        for family, crystal, verifier, evidence_sha in zip(
            families, crystals, discovery, evidence, strict=True
        )
    )
    initial = graph.state()
    graph_state, _changed = graph.append_edges(
        edges,
        expected_generation=initial.generation,
        expected_state_sha256=initial.sha256,
    )
    candidates = []
    bridge_receipts = []
    for index, (family, kind, crystal, publication, edge) in enumerate(
        zip(families, kinds, crystals, publications, edges, strict=True)
    ):
        observations = tuple(
            sorted(_hash(f"{family}:context:{sample}") for sample in range(4))
        )
        stream = CandidateEvidenceStream(
            group_sha256=_hash(f"{family}:group"),
            operator_kind=kind,
            status="promoted",
            reason="fixed-heldout-verification-passed",
            observation_receipt_sha256s=observations,
            fit_receipt_sha256s=tuple(sorted(observations[:3])),
            holdout_receipt_sha256=observations[-1],
            verifier_sha256=discovery[index],
            crystal_sha256=crystal.sha256,
            evidence_sha256=evidence[index],
        )
        promotion = HarvestPromotion(
            candidate=stream,
            edge=edge,
            bank_publication=publication,
            graph_changed=True,
            graph_state_sha256=graph_state.sha256,
        )
        profile = HarvestedCandidateProfile(
            family=family,
            behavior_descriptor=(index, 0),
            objectives=(4.0, 1.0 / crystal.discharge_work_units),
            policy_sha256=_hash("fixed-three-family-profile/v1"),
            execution_verifier_sha256=_hash(f"{family}:execution-verifier"),
        )
        candidate, _program_publication, bridge = (
            harvest_promotion_to_algebra_candidate(
                promotion,
                graph=graph,
                profile=profile,
            )
        )
        candidates.append(candidate)
        bridge_receipts.append(bridge.sha256)
    return bank, families, crystals, tuple(candidates), tuple(bridge_receipts)


def _run_router(
    *,
    bank: ComputeCrystalBank,
    families: tuple[str, ...],
    crystals: tuple[ComputeCrystal, ...],
    candidates: tuple[Any, ...],
    episodes: int,
    seed: int,
    shuffled_context: bool,
) -> dict[str, Any]:
    seed_label = "placebo" if shuffled_context else "contextual"
    router = AlgebraRouterState.bootstrap(
        candidates,
        seed_sha256=_hash(f"{seed}:{seed_label}:router"),
        bin_counts=(3, 2),
        objective_count=2,
        max_elites_per_cell=2,
    )
    value = np.array([[0.25, 0.75]], dtype=np.float64)
    expected = tuple(crystal.apply(value) for crystal in crystals)
    random = np.random.default_rng(seed)
    successes: list[int] = []
    candidate_choices: list[str] = []
    for episode in range(episodes):
        target = episode % len(families)
        context = (
            families[int(random.integers(0, len(families)))]
            if shuffled_context
            else families[target]
        )
        candidate, selection, advanced = router.choose(context)
        router, _update, execution = execute_and_observe_compute_candidate(
            advanced,
            selection,
            candidate,
            bank=bank,
            value=value,
            verifier=_ExpectedArrayVerifier(
                expected[target], candidate.verifier_sha256
            ),
        )
        successes.append(int(execution.outcome.success))
        candidate_choices.append(candidate.sha256)
    late = min(60, episodes)
    final = min(30, episodes)
    return {
        "candidate_choice_sha256": _hash("".join(candidate_choices)),
        "episodes": episodes,
        "final_successes": sum(successes[-final:]),
        "final_window": final,
        "late_successes": sum(successes[-late:]),
        "late_window": late,
        "router_sha256": router.sha256,
        "successes": sum(successes),
    }


def run_harvest_algebra_intelligence_benchmark(
    *,
    seed: int = 20_260_826,
    episodes: int = 180,
) -> dict[str, Any]:
    """Measure contextual choice against an information-destroying placebo."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if (
        isinstance(episodes, bool)
        or not isinstance(episodes, int)
        or episodes < 60
        or episodes % 3
    ):
        raise ValueError("episodes must be a multiple of three and at least 60")
    with tempfile.TemporaryDirectory(prefix=".harvest-algebra-intelligence-") as tmp:
        bank, families, crystals, candidates, bridges = _catalog(Path(tmp))
        contextual = _run_router(
            bank=bank,
            families=families,
            crystals=crystals,
            candidates=candidates,
            episodes=episodes,
            seed=seed,
            shuffled_context=False,
        )
        placebo = _run_router(
            bank=bank,
            families=families,
            crystals=crystals,
            candidates=candidates,
            episodes=episodes,
            seed=seed,
            shuffled_context=True,
        )
    body = {
        "bridge_receipt_sha256s": list(bridges),
        "candidate_count": len(candidates),
        "contextual": contextual,
        "families": list(families),
        "placebo": placebo,
        "program_runtime": "compute-crystal-vm",
        "seed": seed,
    }
    return {
        "body": body,
        "schema": HARVEST_ALGEBRA_INTELLIGENCE_SCHEMA,
        "sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }


__all__ = [
    "HARVEST_ALGEBRA_INTELLIGENCE_SCHEMA",
    "run_harvest_algebra_intelligence_benchmark",
]
