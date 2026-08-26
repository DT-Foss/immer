from __future__ import annotations

import json
import math
import unittest

import numpy as np

from immer.runtimes.ooe.consensus import (
    ConsensusMassError,
    TopologyRouter,
    adjacency_from_edges,
    adaptive_lift_parameters,
    adaptive_ps_lift_matrix,
    barbell_adjacency,
    complete_adjacency,
    fiedler_eigenspace,
    fiedler_vector,
    measure_consensus,
    metropolis_matrix,
    ps_lifted_matrix,
    push_sum,
    push_sum_with_receipt,
    rounds_to_consensus,
    validate_adjacency,
)
from immer.runtimes.ooe.math_core import (
    KernelDiagnostics,
    RapidityLedger,
    array_sha256,
    asymmetry_parameter,
    ginibre_s2,
    jensen_gap,
    kernel_diagnostics,
    mobius_compose,
    normalize_rows,
    normalized_entropy,
    sinkhorn_project,
    sinkhorn_rg_cool,
    spectral_gap,
    tv_contraction,
)


class OoEMathCoreTests(unittest.TestCase):
    def test_mobius_is_additive_in_rapidity_and_receipted(self) -> None:
        left, right = 0.37, -0.21
        expected = math.tanh(math.atanh(left) + math.atanh(right))
        self.assertAlmostEqual(mobius_compose(left, right), expected, places=14)

        ledger = RapidityLedger()
        ledger.add(left)
        ledger.add(right)
        self.assertAlmostEqual(ledger.value, expected, places=14)
        self.assertEqual(ledger.to_dict()["schema"], "immer-ooe-rapidity-ledger/v1")
        json.dumps(ledger.to_dict(), allow_nan=False, sort_keys=True)

    def test_rapidity_rejects_nonfinite_input_and_honors_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            RapidityLedger().add(float("nan"))
        ledger = RapidityLedger(limit=0.25)
        ledger.add(0.99)
        self.assertAlmostEqual(ledger.xi, 0.25)
        with self.assertRaisesRegex(ValueError, "within"):
            RapidityLedger(xi=1.0, limit=0.25)

    def test_sinkhorn_projection_and_rg_cooling(self) -> None:
        rng = np.random.default_rng(11)
        source = rng.lognormal(size=(12, 12))
        projected = sinkhorn_project(source, rounds=80)
        self.assertTrue(projected.flags.c_contiguous)
        np.testing.assert_allclose(projected.sum(axis=0), 1.0, atol=1e-10, rtol=0)
        np.testing.assert_allclose(projected.sum(axis=1), 1.0, atol=1e-10, rtol=0)

        monopolized = np.eye(16) * 20.0 + 0.01
        before = sinkhorn_project(monopolized, rounds=30).max()
        cooled = sinkhorn_rg_cool(monopolized, alpha=0.35, rounds=3)
        self.assertLess(cooled.max(), before)
        np.testing.assert_allclose(cooled.sum(0), 1.0, atol=1e-8, rtol=0)
        np.testing.assert_allclose(cooled.sum(1), 1.0, atol=1e-8, rtol=0)

    def test_row_normalization_preserves_zero_row_semantics(self) -> None:
        normalized = normalize_rows(np.array([[0.0, 0.0], [1.0, 3.0]]))
        np.testing.assert_allclose(normalized, [[0.5, 0.5], [0.25, 0.75]])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            normalize_rows(np.array([[1.0, -1.0]]))
        with self.assertRaisesRegex(ValueError, "finite"):
            normalize_rows(np.array([[1.0, np.inf]]))

    def test_diagnostics_are_canonical_and_json_serializable(self) -> None:
        uniform = np.full((5, 5), 0.2)
        self.assertAlmostEqual(tv_contraction(uniform), 0.0, places=14)
        self.assertAlmostEqual(spectral_gap(uniform), 1.0, places=14)
        self.assertAlmostEqual(normalized_entropy(uniform), 1.0, places=14)
        self.assertGreater(jensen_gap(np.array([0.1, 0.4, 0.8])), 0.0)
        self.assertEqual(asymmetry_parameter(np.eye(4)), 0.0)
        self.assertIsNone(ginibre_s2(np.eye(3)))

        receipt = KernelDiagnostics.from_matrix(uniform)
        document = receipt.to_dict()
        self.assertEqual(document, kernel_diagnostics(uniform))
        json.dumps(document, allow_nan=False, sort_keys=True)
        self.assertEqual(document["matrix_sha256"], array_sha256(uniform))
        self.assertEqual(
            array_sha256(np.asfortranarray(uniform)), array_sha256(uniform)
        )

    def test_dense_caps_fail_before_large_working_allocation(self) -> None:
        with self.assertRaisesRegex(ValueError, "working set"):
            sinkhorn_project(np.ones((8, 8)), max_bytes=1_000)
        with self.assertRaisesRegex(ValueError, "max_nodes"):
            sinkhorn_project(np.ones((5, 5)), max_nodes=4)


class OoEConsensusTests(unittest.TestCase):
    def test_barbell_and_weighted_general_adjacency(self) -> None:
        barbell = barbell_adjacency(4, 7, bridge_weight=0.25)
        self.assertEqual(barbell.shape, (11, 11))
        self.assertEqual(barbell[3, 4], 0.25)
        graph = adjacency_from_edges(
            5,
            [(0, 1, 0.5), (1, 2, 2.0), (2, 3), (3, 4), (4, 0)],
        )
        self.assertTrue(graph.flags.c_contiguous)
        self.assertEqual(graph[1, 2], 2.0)
        np.testing.assert_allclose(graph, graph.T)

    def test_adjacency_validation_rejects_invalid_topologies(self) -> None:
        with self.assertRaisesRegex(ValueError, "disconnected"):
            validate_adjacency(
                np.array(
                    [
                        [0.0, 1.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0, 0.0],
                    ]
                )
            )
        with self.assertRaisesRegex(ValueError, "symmetric"):
            validate_adjacency(np.array([[0.0, 1.0], [0.0, 0.0]]))
        with self.assertRaisesRegex(ValueError, "self-loop"):
            adjacency_from_edges(2, [(0, 0)])
        with self.assertRaisesRegex(ValueError, "outside"):
            adjacency_from_edges(2, [(0, 2)])
        with self.assertRaisesRegex(ValueError, "working set"):
            barbell_adjacency(5, 5, max_bytes=1_000)

    def test_fiedler_sign_and_degenerate_basis_are_deterministic(self) -> None:
        graph = complete_adjacency(7)
        eigenvalue_a, basis_a = fiedler_eigenspace(graph)
        eigenvalue_b, basis_b = fiedler_eigenspace(np.asfortranarray(graph))
        self.assertAlmostEqual(eigenvalue_a, 7.0)
        self.assertEqual(eigenvalue_a, eigenvalue_b)
        np.testing.assert_allclose(basis_a, basis_b, atol=1e-12, rtol=0)
        np.testing.assert_allclose(basis_a.T @ basis_a, np.eye(6), atol=1e-12)
        np.testing.assert_allclose(basis_a.sum(axis=0), 0.0, atol=1e-12)
        for column in basis_a.T:
            pivot = int(np.argmax(np.abs(column)))
            self.assertGreater(column[pivot], 0.0)

        _, vector_a = fiedler_vector(barbell_adjacency(5, 8))
        _, vector_b = fiedler_vector(barbell_adjacency(5, 8))
        np.testing.assert_array_equal(vector_a, vector_b)

    def test_metropolis_and_lifted_are_stochastic_at_arbitrary_size(self) -> None:
        adjacency = barbell_adjacency(5, 8)
        reversible = metropolis_matrix(adjacency)
        lifted = ps_lifted_matrix(adjacency)
        self.assertEqual(reversible.shape, (13, 13))
        self.assertEqual(lifted.shape, (26, 26))
        np.testing.assert_allclose(reversible.sum(axis=1), 1.0, atol=1e-12, rtol=0)
        np.testing.assert_allclose(lifted.sum(axis=1), 1.0, atol=1e-12, rtol=0)
        self.assertGreater(asymmetry_parameter(lifted), 0.0)

    def test_ps_lifted_preserves_poc_speedup_on_barbell(self) -> None:
        adjacency = barbell_adjacency(6, 6)
        reversible = metropolis_matrix(adjacency)
        lifted = ps_lifted_matrix(adjacency)
        values = np.arange(12, dtype=np.float64)[:, None]
        reversible_rounds = rounds_to_consensus(
            reversible,
            values,
            tolerance=1e-5,
            max_rounds=2_000,
        )
        lifted_rounds = rounds_to_consensus(
            lifted,
            values,
            tolerance=1e-5,
            max_rounds=2_000,
            lifted_nodes=12,
        )
        self.assertEqual(reversible_rounds, 338)
        self.assertEqual(lifted_rounds, 64)
        self.assertLess(lifted_rounds, reversible_rounds * 0.6)

    def test_adaptive_lift_binds_foss_topology_schedule(self) -> None:
        barbell = barbell_adjacency(6, 6)
        complete = complete_adjacency(12)
        slow = adaptive_lift_parameters(barbell)
        fast = adaptive_lift_parameters(complete)
        self.assertAlmostEqual(
            slow.formula_pc,
            float(
                np.clip(
                    0.85 - 0.05 * math.log(slow.fiedler_eigenvalue),
                    0.5,
                    0.97,
                )
            ),
        )
        self.assertAlmostEqual(
            fast.formula_pc,
            float(
                np.clip(
                    0.85 - 0.05 * math.log(fast.fiedler_eigenvalue),
                    0.5,
                    0.97,
                )
            ),
        )
        self.assertGreater(slow.pc, fast.pc)
        self.assertEqual(
            slow.selection,
            "formula-proposal+spectral-self-calibration",
        )
        self.assertEqual(
            slow.pc,
            max(slow.spectral_candidates, key=lambda item: item[1])[0],
        )
        lifted, bound = adaptive_ps_lift_matrix(barbell)
        self.assertEqual(bound, slow)
        np.testing.assert_allclose(
            lifted.sum(axis=1), 1.0, atol=1e-12, rtol=0.0
        )
        self.assertEqual(len(bound.sha256), 64)
        values = np.arange(12, dtype=np.float64)[:, None]
        adaptive_rounds = rounds_to_consensus(
            lifted,
            values,
            tolerance=1e-5,
            max_rounds=2_000,
            lifted_nodes=12,
        )
        fixed_rounds = rounds_to_consensus(
            ps_lifted_matrix(barbell, pc=0.65, ps=0.003),
            values,
            tolerance=1e-5,
            max_rounds=2_000,
            lifted_nodes=12,
        )
        self.assertLess(adaptive_rounds, fixed_rounds)

    def test_push_sum_recovers_vector_average_and_receipts_it(self) -> None:
        adjacency = barbell_adjacency(5, 5)
        lifted = ps_lifted_matrix(adjacency)
        values = np.column_stack(
            [np.arange(10, dtype=np.float64), np.arange(10, dtype=np.float64) ** 2]
        )
        estimates = push_sum(lifted, values, rounds=250, lifted_nodes=10)
        expected = np.broadcast_to(values.mean(axis=0), estimates.shape)
        np.testing.assert_allclose(estimates, expected, atol=1e-10, rtol=0)

        result = push_sum_with_receipt(
            lifted,
            values,
            rounds=250,
            tolerance=1e-9,
            lifted_nodes=10,
        )
        self.assertTrue(result.receipt.converged)
        self.assertEqual(result.receipt.rounds, 250)
        self.assertGreater(result.receipt.minimum_mass, 0.0)
        json.dumps(result.receipt.to_dict(), allow_nan=False, sort_keys=True)
        self.assertEqual(
            result.receipt.estimates_sha256, array_sha256(result.estimates)
        )

    def test_measure_consensus_returns_first_converged_round(self) -> None:
        adjacency = barbell_adjacency(6, 6)
        lifted = ps_lifted_matrix(adjacency)
        values = np.arange(12, dtype=np.float64)
        result = measure_consensus(
            lifted,
            values,
            tolerance=1e-5,
            max_rounds=100,
            lifted_nodes=12,
            topology="ps-lifted-z2",
        )
        self.assertEqual(result.receipt.rounds, 64)
        self.assertEqual(result.receipt.topology, "ps-lifted-z2")
        self.assertTrue(result.receipt.converged)

    def test_push_sum_fails_on_unreachable_zero_mass(self) -> None:
        transition = np.array([[1.0, 0.0], [1.0, 0.0]])
        with self.assertRaisesRegex(ConsensusMassError, r"\[1\]"):
            push_sum(transition, np.array([2.0, 4.0]), rounds=1)

    def test_topology_router_selects_by_measured_gap_and_receipts(self) -> None:
        router = TopologyRouter(gap_threshold=0.1)
        barbell = barbell_adjacency(6, 6)
        complete = complete_adjacency(12)
        self.assertEqual(router.choose(barbell), "lifted")
        self.assertEqual(router.choose(complete), "reversible")
        decision = router.route(barbell)
        json.dumps(decision.to_dict(), allow_nan=False, sort_keys=True)
        mode, transition, lifted_nodes = router.transition(barbell)
        self.assertEqual(mode, "lifted")
        self.assertEqual(transition.shape, (24, 24))
        self.assertEqual(lifted_nodes, 12)


if __name__ == "__main__":
    unittest.main()
