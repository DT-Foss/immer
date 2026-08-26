from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import unittest

import numpy as np

from immer.runtimes.ooe.bvn_search import (
    BvNSearchIntegrityError,
    BehavioralElite,
    BehavioralMAPElites,
    BirkhoffDecomposition,
    BirkhoffReconstructionReceipt,
    ContextualThompsonMutation,
    DiagonalSuccessStepAdapter,
    MatchedSpectralProposalFilter,
    OperatorGenome,
    PermutationAtom,
    PermutationMixture,
    ThompsonPosterior,
    birkhoff_von_neumann,
    canonical_theta,
    fiedler_edge_novelty,
    fiedler_projector_novelty,
    mixture_from_theta,
    spectral_proposal_features,
    theta_to_doubly_stochastic,
)
from immer.runtimes.ooe.identity import canonical_json_bytes


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _atom(*images: int) -> PermutationAtom:
    return PermutationAtom(tuple(images))


def _cycle_atoms(size: int) -> tuple[PermutationAtom, ...]:
    return tuple(
        PermutationAtom(tuple((row + shift) % size for row in range(size)))
        for shift in range(size)
    )


def _convex_kernel(size: int, weights: tuple[float, ...]) -> np.ndarray:
    return PermutationMixture(_cycle_atoms(size), weights).reconstruct()


class PermutationMixtureTests(unittest.TestCase):
    def test_atom_is_immutable_bijective_and_applies_without_dense_storage(self) -> None:
        atom = _atom(2, 0, 1)
        self.assertEqual(atom.permutation, (2, 0, 1))
        self.assertEqual(atom.dense().tolist(), [[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        value = np.asarray([[10.0, 20.0, 30.0], [-1.0, 4.0, 9.0]])
        self.assertEqual(atom.apply(value)[0].tolist(), [20.0, 30.0, 10.0])
        self.assertTrue(np.array_equal(atom.apply(value), value @ atom.dense()))
        with self.assertRaises(FrozenInstanceError):
            atom.permutation = (0, 1, 2)  # type: ignore[misc]
        with self.assertRaisesRegex(ValueError, "bijection"):
            _atom(0, 0, 2)
        with self.assertRaisesRegex(ValueError, "outside|trailing"):
            atom.apply(np.ones((2, 1)))

    def test_theta_is_strict_gauge_fixed_and_doubly_stochastic(self) -> None:
        # All six S3 vertices are legal even though a decomposition only needs five.
        atoms = (
            _atom(0, 1, 2),
            _atom(0, 2, 1),
            _atom(1, 0, 2),
            _atom(1, 2, 0),
            _atom(2, 0, 1),
            _atom(2, 1, 0),
        )
        theta = np.asarray([3.0, -1.0, 0.5, 2.0, -2.0, 1.0])
        gauge = canonical_theta(theta, atom_count=6)
        self.assertEqual(gauge[0], 0.0)
        self.assertTrue(np.array_equal(gauge, canonical_theta(theta + 9.0, atom_count=6)))
        first = theta_to_doubly_stochastic(theta, atoms)
        second = theta_to_doubly_stochastic(theta + 9.0, atoms)
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.all(first >= 0.0))
        self.assertTrue(np.allclose(first.sum(axis=0), 1.0, atol=1e-15))
        self.assertTrue(np.allclose(first.sum(axis=1), 1.0, atol=1e-15))
        with self.assertRaisesRegex(ValueError, "exact shape"):
            theta_to_doubly_stochastic(theta[:-1], atoms)
        with self.assertRaisesRegex(ValueError, "exact shape"):
            theta_to_doubly_stochastic(np.pad(theta, (0, 1)), atoms)
        with self.assertRaisesRegex(ValueError, "finite"):
            theta_to_doubly_stochastic([0, 1, 2, 3, 4, np.inf], atoms)

        extreme = mixture_from_theta([0.0, -10_000.0], atoms[:2])
        self.assertEqual(extreme.component_count, 1)
        np.testing.assert_array_equal(extreme.reconstruct(), atoms[0].dense())

    def test_mixture_roundtrip_and_resealed_derived_tamper_fail_closed(self) -> None:
        mixture = mixture_from_theta([0.0, 1.0, -1.0], _cycle_atoms(3))
        restored = PermutationMixture.from_bytes(mixture.to_bytes())
        self.assertEqual(restored, mixture)
        self.assertTrue(np.array_equal(restored.reconstruct(), mixture.reconstruct()))

        document = json.loads(mixture.to_bytes())
        document["body"]["weights_hex"][0] = float(0.5).hex()
        document["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(document["body"])
        ).hexdigest()
        with self.assertRaises(BvNSearchIntegrityError):
            PermutationMixture.from_bytes(canonical_json_bytes(document))

    def test_operator_genome_binds_gauge_family_and_ordered_lineage(self) -> None:
        genome = OperatorGenome(
            atoms=_cycle_atoms(3),
            theta=(7.0, 8.0, 6.0),
            map_family="bvn.markov",
            composition_parent_sha256s=(_sha("left"), _sha("right")),
        )
        self.assertEqual(genome.theta, (0.0, 1.0, -1.0))
        self.assertEqual(OperatorGenome.from_bytes(genome.to_bytes()), genome)
        reversed_lineage = replace(
            genome,
            composition_parent_sha256s=tuple(
                reversed(genome.composition_parent_sha256s)
            ),
        )
        self.assertNotEqual(reversed_lineage.sha256, genome.sha256)
        document = json.loads(genome.to_bytes())
        document["body"]["kernel_sha256"] = "0" * 64
        document["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(document["body"])
        ).hexdigest()
        with self.assertRaisesRegex(BvNSearchIntegrityError, "derived"):
            OperatorGenome.from_bytes(canonical_json_bytes(document))


class BirkhoffDecompositionTests(unittest.TestCase):
    def test_dyadic_kernel_has_bit_exact_reconstruction_and_receipt_roundtrip(self) -> None:
        kernel = _convex_kernel(4, (0.5, 0.25, 0.125, 0.125))
        decomposition = birkhoff_von_neumann(kernel)
        receipt = decomposition.verify_source(kernel)
        self.assertTrue(receipt.exact_bitwise)
        self.assertEqual(receipt.max_abs_error, 0.0)
        self.assertTrue(receipt.accepted)
        self.assertEqual(
            BirkhoffReconstructionReceipt.from_bytes(receipt.to_bytes()), receipt
        )
        self.assertLessEqual(decomposition.component_count, (4 - 1) ** 2 + 1)
        self.assertTrue(all(weight > 0.0 for weight in decomposition.weights))

    def test_dense_random_kernels_obey_bound_and_are_deterministic(self) -> None:
        generator = np.random.default_rng(20260826)
        for size in range(2, 9):
            atoms: dict[PermutationAtom, float] = {}
            raw_weights = generator.random(min(30, (size - 1) ** 2 + 1))
            raw_weights /= raw_weights.sum()
            for weight in raw_weights:
                atom = PermutationAtom(tuple(generator.permutation(size)))
                atoms[atom] = atoms.get(atom, 0.0) + float(weight)
            ordered = tuple(sorted(atoms))
            weights = [atoms[atom] for atom in ordered]
            weights[int(np.argmax(weights))] += 1.0 - sum(weights)
            kernel = PermutationMixture(ordered, tuple(weights)).reconstruct()
            first = birkhoff_von_neumann(kernel, tolerance=1e-11)
            second = birkhoff_von_neumann(kernel, tolerance=1e-11)
            self.assertEqual(first.to_bytes(), second.to_bytes())
            self.assertLessEqual(first.component_count, (size - 1) ** 2 + 1)
            self.assertLessEqual(
                first.verify_source(kernel, tolerance=1e-11).max_abs_error,
                1e-11,
            )


    def test_sparse_support_and_singleton_are_constructive(self) -> None:
        singleton = np.ones((1, 1), dtype=np.float64)
        self.assertEqual(birkhoff_von_neumann(singleton).component_count, 1)
        sparse = 0.7 * _atom(0, 1, 2, 3).dense() + 0.3 * _atom(1, 0, 3, 2).dense()
        result = birkhoff_von_neumann(sparse)
        self.assertEqual(result.component_count, 2)
        self.assertTrue(result.verify_source(sparse).exact_bitwise)

    def test_invalid_kernel_and_wrong_source_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "rows"):
            birkhoff_von_neumann([[0.8, 0.2], [0.3, 0.3]])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            birkhoff_von_neumann([[1.1, -0.1], [-0.1, 1.1]])
        with self.assertRaisesRegex(ValueError, "square"):
            birkhoff_von_neumann(np.ones((2, 3)))
        first = _convex_kernel(3, (0.5, 0.3, 0.2))
        second = _convex_kernel(3, (0.4, 0.35, 0.25))
        decomposition = birkhoff_von_neumann(first)
        with self.assertRaisesRegex(BvNSearchIntegrityError, "identity"):
            decomposition.verify_source(second)

    def test_decomposition_roundtrip_and_tamper_checks_bind_reconstruction(self) -> None:
        kernel = _convex_kernel(3, (0.5, 0.25, 0.25))
        decomposition = birkhoff_von_neumann(kernel)
        restored = BirkhoffDecomposition.from_bytes(decomposition.to_bytes())
        self.assertEqual(restored.sha256, decomposition.sha256)
        self.assertTrue(restored.verify_source(kernel).accepted)

        document = json.loads(decomposition.to_bytes())
        document["body"]["atoms"][0]["permutation"] = [2, 1, 0]
        document["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(document["body"])
        ).hexdigest()
        with self.assertRaises(BvNSearchIntegrityError):
            BirkhoffDecomposition.from_bytes(canonical_json_bytes(document))

        receipt = decomposition.verify_source(kernel)
        receipt_document = json.loads(receipt.to_bytes())
        receipt_document["accepted"] = False
        with self.assertRaisesRegex(BvNSearchIntegrityError, "derived"):
            BirkhoffReconstructionReceipt.from_bytes(
                canonical_json_bytes(receipt_document)
            )


class FiedlerEdgeNoveltyTests(unittest.TestCase):
    def test_projector_gap_is_basis_invariant_and_penalizes_existing_edges(self) -> None:
        complete = np.ones((5, 5), dtype=np.float64) - np.eye(5)
        first = fiedler_edge_novelty(complete, 0, 3)
        reversed_endpoints = fiedler_edge_novelty(complete, 3, 0)
        self.assertEqual(first, reversed_endpoints)
        self.assertAlmostEqual(first.novelty, 1.0, places=12)
        self.assertTrue(first.direct_edge)
        self.assertAlmostEqual(first.priority(0.8), 0.24, places=12)

        path = np.zeros((5, 5), dtype=np.float64)
        for index in range(4):
            path[index, index + 1] = path[index + 1, index] = 1.0
        missing = fiedler_edge_novelty(path, 0, 4)
        self.assertFalse(missing.direct_edge)
        self.assertGreater(missing.priority(0.8), 0.0)
        self.assertGreater(missing.priority(0.8), first.priority(0.8))


class MutationSearchStateTests(unittest.TestCase):
    def test_contextual_thompson_replays_and_keeps_contexts_separate(self) -> None:
        state = ContextualThompsonMutation(("permute", "rescale"), _sha("thompson"))
        first, advanced = state.choose("graph-a")
        replay, replay_advanced = state.choose("graph-a")
        self.assertEqual(first, replay)
        self.assertEqual(advanced, replay_advanced)
        self.assertEqual(advanced.decision_index, 1)
        self.assertEqual(state.decision_index, 0)

        trained = state
        for _ in range(20):
            trained = trained.observe("graph-a", "permute", success=True)
            trained = trained.observe("graph-a", "rescale", success=False)
        self.assertEqual(trained.posterior("graph-a", "permute").successes, 20)
        self.assertEqual(trained.posterior("graph-b", "permute").successes, 0)
        choice, _ = trained.choose("graph-a")
        self.assertEqual(choice.arm, "permute")

    def test_thompson_serialization_and_duplicate_context_arm_rejection(self) -> None:
        state = ContextualThompsonMutation(("a", "b"), _sha("seed")).observe(
            "context", "a", success=True
        )
        self.assertEqual(ContextualThompsonMutation.from_bytes(state.to_bytes()), state)
        with self.assertRaisesRegex(ValueError, "unique"):
            ContextualThompsonMutation(
                ("a", "b"),
                _sha("seed"),
                posteriors=(
                    ThompsonPosterior("context", "a", successes=1),
                    ThompsonPosterior("context", "a", successes=2),
                ),
            )
        document = json.loads(state.to_bytes())
        document["body"]["decision_index"] = 99
        with self.assertRaisesRegex(BvNSearchIntegrityError, "hash"):
            ContextualThompsonMutation.from_bytes(canonical_json_bytes(document))

    def test_diagonal_success_step_is_deterministic_and_named_honestly(self) -> None:
        state = DiagonalSuccessStepAdapter(
            mean=(0.0, 0.0, 0.0),
            log_steps=(-1.0, -1.0, -1.0),
            seed_sha256=_sha("diagonal"),
        )
        first = state.propose(context_sha256=_sha("context"))
        second = state.propose(context_sha256=_sha("context"))
        self.assertEqual(first, second)
        success = state.update(first, success=True)
        failure = state.update(first, success=False)
        self.assertEqual(success.mean, first.candidate)
        self.assertEqual(failure.mean, state.mean)
        self.assertTrue(all(a > b for a, b in zip(success.log_steps, state.log_steps)))
        self.assertTrue(all(a < b for a, b in zip(failure.log_steps, state.log_steps)))
        self.assertEqual(
            DiagonalSuccessStepAdapter.from_bytes(success.to_bytes()), success
        )
        with self.assertRaisesRegex(BvNSearchIntegrityError, "belong"):
            success.update(first, success=True)
        damaged = replace(first, candidate=(99.0, *first.candidate[1:]))
        with self.assertRaisesRegex(BvNSearchIntegrityError, "altered"):
            state.update(damaged, success=True)


class BehavioralMAPElitesTests(unittest.TestCase):
    def test_coverage_uses_full_grid_and_cells_keep_bounded_pareto_sets(self) -> None:
        archive = BehavioralMAPElites(
            (2, 3, 4), objective_count=2, max_elites_per_cell=2
        )
        self.assertEqual(archive.coverage_denominator, 24)
        self.assertTrue(archive.add(BehavioralElite(_sha("a"), (1, 2, 3), (1.0, 0.0))))
        self.assertTrue(archive.add(BehavioralElite(_sha("b"), (1, 2, 3), (0.0, 1.0))))
        self.assertFalse(archive.add(BehavioralElite(_sha("c"), (1, 2, 3), (0.0, 0.0))))
        self.assertEqual(archive.elite_count, 2)
        self.assertEqual(archive.occupied_cells, 1)
        self.assertEqual(archive.coverage, 1 / 24)
        self.assertEqual(
            {item.candidate_sha256 for item in archive.cell((1, 2, 3))},
            {_sha("a"), _sha("b")},
        )

    def test_equal_objective_tie_uses_smallest_sha_and_capacity_is_exact(self) -> None:
        archive = BehavioralMAPElites((1,), objective_count=2, max_elites_per_cell=1)
        larger, smaller = sorted((_sha("larger"), _sha("smaller")), reverse=True)
        archive.add(BehavioralElite(larger, (0,), (1.0, 1.0)))
        archive.add(BehavioralElite(smaller, (0,), (1.0, 1.0)))
        self.assertEqual(archive.cell((0,))[0].candidate_sha256, smaller)
        archive.add(BehavioralElite(_sha("tradeoff"), (0,), (2.0, 0.0)))
        self.assertEqual(len(archive.cell((0,))), 1)

    def test_sampling_is_cell_uniform_not_elite_uniform_and_replays(self) -> None:
        archive = BehavioralMAPElites((2,), objective_count=2, max_elites_per_cell=8)
        archive.add(BehavioralElite(_sha("only"), (0,), (1.0, 1.0)))
        for index in range(8):
            archive.add(
                BehavioralElite(
                    _sha(f"many-{index}"),
                    (1,),
                    (float(index), float(7 - index)),
                )
            )
        counts = [0, 0]
        samples = []
        for nonce in range(600):
            selected = archive.sample_cell_uniform(
                seed_sha256=_sha("sample"), nonce=nonce
            )
            counts[selected.descriptor[0]] += 1
            samples.append(selected.candidate_sha256)
        self.assertTrue(240 <= counts[0] <= 360, counts)
        replay = [
            archive.sample_cell_uniform(seed_sha256=_sha("sample"), nonce=nonce).candidate_sha256
            for nonce in range(600)
        ]
        self.assertEqual(samples, replay)

    def test_archive_roundtrip_and_derived_coverage_tamper_rejection(self) -> None:
        archive = BehavioralMAPElites((2, 2), objective_count=1)
        archive.add(BehavioralElite(_sha("elite"), (1, 0), (3.0,)))
        restored = BehavioralMAPElites.from_bytes(archive.to_bytes())
        self.assertEqual(restored.sha256, archive.sha256)
        document = json.loads(archive.to_bytes())
        document["body"]["coverage_denominator"] = 3
        document["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(document["body"])
        ).hexdigest()
        with self.assertRaisesRegex(BvNSearchIntegrityError, "derived"):
            BehavioralMAPElites.from_bytes(canonical_json_bytes(document))


class ProposalFilterTests(unittest.TestCase):
    def test_fiedler_projector_is_deterministic_and_reference_relative(self) -> None:
        adjacency = np.ones((4, 4), dtype=np.float64) - np.eye(4)
        identity = np.eye(4, dtype=np.float64)
        cycle = _atom(1, 2, 3, 0).dense()
        zero = fiedler_projector_novelty(adjacency, identity, (identity,))
        first = fiedler_projector_novelty(adjacency, cycle, (identity,))
        second = fiedler_projector_novelty(adjacency, cycle, (identity,))
        self.assertEqual(zero.novelty, 0.0)
        self.assertGreater(first.novelty, 0.0)
        self.assertEqual(first, second)
        self.assertEqual(first.eigenspace_rank, 3)
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            fiedler_projector_novelty(adjacency, np.eye(3))

    def test_matched_empirical_spectral_filter_rejects_placebo_and_accepts_outlier(self) -> None:
        uniform = np.full((4, 4), 0.25, dtype=np.float64)
        null = tuple(
            weight * np.eye(4) + (1.0 - weight) * uniform
            for weight in (0.15, 0.2, 0.25, 0.3, 0.35)
        )
        proposal_filter = MatchedSpectralProposalFilter.calibrate(
            null, quantile=1.0
        )
        placebo = 0.25 * np.eye(4) + 0.75 * uniform
        structured = _atom(1, 2, 3, 0).dense()
        self.assertFalse(proposal_filter.evaluate(placebo).accepted)
        self.assertTrue(proposal_filter.evaluate(structured).accepted)
        self.assertEqual(
            proposal_filter.evaluate(structured),
            proposal_filter.evaluate(structured),
        )
        self.assertEqual(len(spectral_proposal_features(structured)), 6)

    def test_spectral_filter_roundtrip_tamper_and_mismatch_checks(self) -> None:
        uniform = np.full((3, 3), 1 / 3, dtype=np.float64)
        null = tuple(
            weight * np.eye(3) + (1.0 - weight) * uniform
            for weight in (0.1, 0.2, 0.3)
        )
        proposal_filter = MatchedSpectralProposalFilter.calibrate(null)
        restored = MatchedSpectralProposalFilter.from_bytes(
            proposal_filter.to_bytes()
        )
        self.assertEqual(restored, proposal_filter)
        with self.assertRaisesRegex(ValueError, "dimension"):
            proposal_filter.evaluate(np.eye(4))
        with self.assertRaisesRegex(ValueError, "distinct"):
            MatchedSpectralProposalFilter.calibrate((null[0], null[0], null[1]))

        document = json.loads(proposal_filter.to_bytes())
        document["body"]["threshold_hex"] = float(0.0).hex()
        document["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(document["body"])
        ).hexdigest()
        with self.assertRaises(BvNSearchIntegrityError):
            MatchedSpectralProposalFilter.from_bytes(canonical_json_bytes(document))


if __name__ == "__main__":
    unittest.main()
