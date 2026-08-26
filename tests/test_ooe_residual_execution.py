from __future__ import annotations

import hashlib
import json
import tempfile
import unittest

import numpy as np

from immer.runtimes.ooe.compute_crystals import ComputeCrystal, ComputeCrystalBank
from immer.runtimes.ooe.compute_graph import ComputeOperatorGraph, OperatorEdge
from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.residual_execution import (
    ResidualExecutionIntegrityError,
    ResidualRouteDischargeReceipt,
    ResidualRouteExecutor,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class ResidualRouteExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.bank = ComputeCrystalBank(self.temporary.name)
        self.graph = ComputeOperatorGraph(self.bank)

    def _append_chain(
        self,
        crystals: tuple[ComputeCrystal, ...],
        states: tuple[str, ...],
        *,
        offset: int = 0,
    ) -> tuple[OperatorEdge, ...]:
        self.assertEqual(len(states), len(crystals) + 1)
        edges = []
        for index, crystal in enumerate(crystals):
            self.bank.publish_crystal(crystal)
            edges.append(
                OperatorEdge(
                    source_state=states[index],
                    target_state=states[index + 1],
                    crystal_sha256=crystal.sha256,
                    verifier_sha256=_digest(f"verifier:{index + offset}"),
                    evidence_sha256=_digest(f"evidence:{index + offset}"),
                )
            )
        self.graph.append_edges(tuple(edges))
        return tuple(edges)

    def _affine_chain(self) -> tuple[ComputeCrystal, ...]:
        return (
            ComputeCrystal.affine([[2.0]], [1.0]),
            ComputeCrystal.affine([[-3.0]], [4.0]),
            ComputeCrystal.affine([[0.5]], [-2.0]),
            ComputeCrystal.affine([[1.25]], [7.0]),
        )

    @staticmethod
    def _direct(
        crystals: tuple[ComputeCrystal, ...], value: np.ndarray
    ) -> np.ndarray:
        result = value
        for crystal in crystals:
            result = crystal.apply(result)
        return result

    def test_charged_prefix_runs_and_only_unknown_suffix_stays_live(self) -> None:
        crystals = self._affine_chain()
        self._append_chain(crystals, ("s0", "s1", "s2", "s3", "s4"))
        prefix = self.graph.plan_route("s0", "s2").plan
        assert prefix is not None
        charged = self.graph.charge_route(prefix)
        self.assertIsNotNone(charged.route.fused_crystal_sha256)

        full = self.graph.plan_route("s0", "s4").plan
        assert full is not None
        future = np.array([[-91.5], [0.125], [44.0]], dtype=np.float64)
        result = ResidualRouteExecutor(self.graph).execute(full, future)

        np.testing.assert_array_equal(result.output, self._direct(crystals, future))
        self.assertIsNotNone(result.prefix_route)
        self.assertEqual(result.receipt.prefix_length, 2)
        self.assertEqual(result.receipt.residual_length, 2)
        self.assertTrue(result.receipt.reused_past_compute)
        assert result.receipt.prefix_execution_receipt is not None
        assert result.receipt.suffix_execution_receipt is not None
        self.assertEqual(
            result.receipt.prefix_execution_receipt.executed_operator_count, 1
        )
        self.assertEqual(
            result.receipt.suffix_execution_receipt.executed_operator_count, 2
        )
        self.assertGreater(result.receipt.historical_work_released, 0)
        self.assertEqual(
            ResidualRouteDischargeReceipt.from_bytes(result.receipt.to_bytes()),
            result.receipt,
        )

    def test_no_charged_prefix_executes_the_complete_plan_without_false_credit(
        self,
    ) -> None:
        crystals = self._affine_chain()[:3]
        self._append_chain(crystals, ("raw", "a", "b", "answer"))
        plan = self.graph.plan_route("raw", "answer").plan
        assert plan is not None
        value = np.array([[3.0], [7.0]], dtype=np.float64)

        result = ResidualRouteExecutor(self.graph).execute(plan, value)

        np.testing.assert_array_equal(result.output, self._direct(crystals, value))
        self.assertIsNone(result.prefix_route)
        self.assertEqual(result.receipt.prefix_length, 0)
        self.assertEqual(result.receipt.residual_length, 3)
        self.assertFalse(result.receipt.reused_past_compute)
        self.assertEqual(result.receipt.historical_work_released, 0)
        self.assertEqual(
            result.receipt.live_work_units,
            result.receipt.equivalent_source_work_units,
        )

    def test_deepest_available_prefix_wins(self) -> None:
        crystals = self._affine_chain()
        self._append_chain(crystals, ("s0", "s1", "s2", "s3", "s4"))
        one = self.graph.plan_route("s0", "s1").plan
        assert one is not None
        self.graph.charge_route(one)
        three = self.graph.plan_route("s0", "s3").plan
        assert three is not None
        deep = self.graph.charge_route(three).route
        full = self.graph.plan_route("s0", "s4").plan
        assert full is not None

        result = ResidualRouteExecutor(self.graph).execute(
            full, np.array([[12.0]], dtype=np.float64)
        )

        self.assertEqual(result.receipt.prefix_length, 3)
        self.assertEqual(result.receipt.residual_length, 1)
        self.assertEqual(result.receipt.prefix_route_sha256, deep.sha256)

    def test_fully_charged_route_needs_no_residual_program(self) -> None:
        crystals = self._affine_chain()[:3]
        self._append_chain(crystals, ("s0", "s1", "s2", "s3"))
        first_plan = self.graph.plan_route("s0", "s3").plan
        assert first_plan is not None
        route = self.graph.charge_route(first_plan).route
        current_plan = self.graph.plan_route("s0", "s3").plan
        assert current_plan is not None
        value = np.array([[1.0], [9.0]], dtype=np.float64)

        result = ResidualRouteExecutor(self.graph).execute(current_plan, value)

        np.testing.assert_array_equal(result.output, self._direct(crystals, value))
        self.assertEqual(result.receipt.prefix_route_sha256, route.sha256)
        self.assertEqual(result.receipt.prefix_length, 3)
        self.assertEqual(result.receipt.residual_length, 0)
        self.assertIsNone(result.receipt.suffix_execution_receipt)

    def test_permutation_prefix_reuses_past_composition_on_future_int_inputs(
        self,
    ) -> None:
        crystals = (
            ComputeCrystal.permutation([1, 2, 0], dtype="int64"),
            ComputeCrystal.permutation([2, 0, 1], dtype="int64"),
            ComputeCrystal.permutation([0, 2, 1], dtype="int64"),
        )
        self._append_chain(crystals, ("p0", "p1", "p2", "p3"))
        prefix = self.graph.plan_route("p0", "p2").plan
        assert prefix is not None
        self.graph.charge_route(prefix)
        full = self.graph.plan_route("p0", "p3").plan
        assert full is not None
        future = np.array([[10, 20, 30], [-4, 9, 7]], dtype=np.int64)

        result = ResidualRouteExecutor(self.graph).execute(full, future)

        np.testing.assert_array_equal(result.output, self._direct(crystals, future))
        self.assertEqual(result.receipt.prefix_length, 2)
        self.assertGreater(result.receipt.historical_work_released, 0)

    def test_stale_plan_and_tampered_receipt_fail_closed(self) -> None:
        crystals = self._affine_chain()[:2]
        self._append_chain(crystals, ("s0", "s1", "s2"))
        plan = self.graph.plan_route("s0", "s2").plan
        assert plan is not None
        result = ResidualRouteExecutor(self.graph).execute(
            plan, np.array([[1.0]], dtype=np.float64)
        )
        document = json.loads(result.receipt.to_bytes())
        document["historical_work_released"] = 1
        with self.assertRaises(ResidualExecutionIntegrityError):
            ResidualRouteDischargeReceipt.from_bytes(canonical_json_bytes(document))

        identity = ComputeCrystal.affine([[1.0]], [0.0])
        self.bank.publish_crystal(identity)
        self.graph.append_edge(
            OperatorEdge(
                source_state="other-a",
                target_state="other-b",
                crystal_sha256=identity.sha256,
                verifier_sha256=_digest("other-verifier"),
                evidence_sha256=_digest("other-evidence"),
            )
        )
        with self.assertRaisesRegex(ResidualExecutionIntegrityError, "stale"):
            ResidualRouteExecutor(self.graph).execute(
                plan, np.array([[1.0]], dtype=np.float64)
            )


if __name__ == "__main__":
    unittest.main()
