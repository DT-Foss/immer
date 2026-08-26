from __future__ import annotations

import hashlib
import json
import tempfile
import unittest

import numpy as np

from immer.runtimes.ooe.compute_crystals import (
    ComputeCrystal,
    ComputeCrystalABIError,
    ComputeCrystalBank,
)
from immer.runtimes.ooe.compute_graph import ComputeOperatorGraph, OperatorEdge
from immer.runtimes.ooe.demand_execution import (
    DemandExecutionIntegrityError,
    DemandExecutionUnavailableError,
    DemandExecutionVerification,
    DemandRoutedExecutionReceipt,
    DemandRoutedExecutor,
)
from immer.runtimes.ooe.demand_scheduler import (
    DemandOutcomeReceipt,
    DemandStateTransitionReceipt,
    OperatorDemandScheduler,
    OutcomeEvent,
    SelectionAbortEvent,
    UCBSelectionReceipt,
)
from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.residual_execution import ResidualRouteExecution


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class _ExactVerifier:
    def __init__(self, expected: np.ndarray, *, accept: bool = True) -> None:
        self.expected = expected
        self.accept = accept
        self.calls = 0

    def __call__(
        self, execution: ResidualRouteExecution
    ) -> DemandExecutionVerification:
        self.calls += 1
        exact = np.array_equal(execution.output, self.expected)
        accepted = self.accept and exact
        return DemandExecutionVerification.create(
            execution,
            accepted=accepted,
            verifier_sha256=_digest("exact-output-verifier"),
            evidence_sha256=_digest(
                f"exact-output-evidence:{self.calls}:{accepted}"
            ),
            reason="exact-output-match" if accepted else "quality-rejected",
        )


class DemandRoutedExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.bank = ComputeCrystalBank(self.temporary.name)
        self.graph = ComputeOperatorGraph(self.bank)
        self.crystals = (
            ComputeCrystal.affine([[2.0]], [1.0]),
            ComputeCrystal.affine([[-3.0]], [4.0]),
            ComputeCrystal.affine([[0.5]], [-2.0]),
            ComputeCrystal.affine([[1.25]], [7.0]),
        )
        states = ("s0", "s1", "s2", "s3", "s4")
        for index, crystal in enumerate(self.crystals):
            self.bank.publish_crystal(crystal)
            self.graph.append_edge(
                OperatorEdge(
                    source_state=states[index],
                    target_state=states[index + 1],
                    crystal_sha256=crystal.sha256,
                    verifier_sha256=_digest(f"edge-verifier:{index}"),
                    evidence_sha256=_digest(f"edge-evidence:{index}"),
                )
            )
        two_plan = self.graph.plan_route("s0", "s2").plan
        assert two_plan is not None
        self.two = self.graph.charge_route(two_plan).route
        three_plan = self.graph.plan_route("s0", "s3").plan
        assert three_plan is not None
        self.three = self.graph.charge_route(three_plan).route
        self.plan = self.graph.plan_route("s0", "s4").plan
        assert self.plan is not None
        self.graph_state = self.graph.state()
        self.scheduler = OperatorDemandScheduler(self.bank.store)
        self.value = np.array([[-7.25], [0.125], [91.0]], dtype=np.float64)
        expected = self.value
        for crystal in self.crystals:
            expected = crystal.apply(expected)
        self.expected = expected
        self.candidates = tuple(sorted((self.two.sha256, self.three.sha256)))

    def _executor(self, *, accept: bool = True) -> tuple[DemandRoutedExecutor, _ExactVerifier]:
        verifier = _ExactVerifier(self.expected, accept=accept)
        return (
            DemandRoutedExecutor(
                graph=self.graph,
                scheduler=self.scheduler,
                verifier=verifier,
            ),
            verifier,
        )

    def test_select_execute_verify_feedback_is_one_canonical_receipt(self) -> None:
        executor, verifier = self._executor()
        result = executor.execute(
            self.plan,
            self.value,
            candidate_route_sha256s=self.candidates,
            episode_sha256=_digest("episode-1"),
        )

        np.testing.assert_array_equal(result.output, self.expected)
        self.assertEqual(verifier.calls, 1)
        self.assertTrue(result.receipt.verification.accepted)
        self.assertEqual(
            result.receipt.residual.prefix_route_sha256,
            result.receipt.selection.route_sha256,
        )
        self.assertGreater(result.receipt.outcome.reward, 1.0)
        self.assertEqual(self.scheduler.state().generation, 2)
        self.assertEqual(
            DemandRoutedExecutionReceipt.from_bytes(result.receipt.to_bytes()),
            result.receipt,
        )
        self.assertEqual(
            UCBSelectionReceipt.from_bytes(result.receipt.selection.to_bytes()),
            result.receipt.selection,
        )
        self.assertEqual(
            DemandStateTransitionReceipt.from_bytes(
                result.receipt.outcome_transition.to_bytes()
            ),
            result.receipt.outcome_transition,
        )

    def test_ucb_learns_the_prefix_that_releases_more_past_compute(self) -> None:
        executor, _verifier = self._executor()
        first = executor.execute(
            self.plan,
            self.value,
            candidate_route_sha256s=self.candidates,
        )
        second = executor.execute(
            self.plan,
            self.value,
            candidate_route_sha256s=self.candidates,
        )
        self.assertNotEqual(
            first.receipt.selected_prefix_route_sha256,
            second.receipt.selected_prefix_route_sha256,
        )
        third = executor.execute(
            self.plan,
            self.value,
            candidate_route_sha256s=self.candidates,
        )
        by_route = {
            row.receipt.selected_prefix_route_sha256: row.receipt.outcome.reward
            for row in (first, second)
        }
        expected = max(by_route, key=by_route.__getitem__)
        self.assertEqual(third.receipt.selected_prefix_route_sha256, expected)
        self.assertEqual(expected, self.three.sha256)

    def test_verified_failure_blocks_only_the_failed_exact_prefix(self) -> None:
        rejecting, _verifier = self._executor(accept=False)
        failed = rejecting.execute(
            self.plan,
            self.value,
            candidate_route_sha256s=self.candidates,
        )
        self.assertFalse(failed.receipt.outcome.success)
        accepted, _ = self._executor()
        recovered = accepted.execute(
            self.plan,
            self.value,
            candidate_route_sha256s=self.candidates,
        )
        self.assertNotEqual(
            failed.receipt.selected_prefix_route_sha256,
            recovered.receipt.selected_prefix_route_sha256,
        )

    def test_ppm_prediction_can_drive_the_persisted_selection(self) -> None:
        for ordinal in range(2):
            episode = _digest(f"ppm-episode:{ordinal}")
            for index, route in enumerate((self.two, self.three)):
                outcome = DemandOutcomeReceipt.create(
                    route=route,
                    graph_state=self.graph_state,
                    success=True,
                    reward=1.0,
                    outcome_evidence_sha256=_digest(
                        f"ppm-evidence:{ordinal}:{index}"
                    ),
                    outcome_verifier_sha256=_digest("ppm-verifier"),
                    episode_sha256=episode,
                )
                self.scheduler.record_outcome(outcome, self.graph_state)
        executor, _ = self._executor()
        result = executor.execute(
            self.plan,
            self.value,
            candidate_route_sha256s=self.candidates,
            history_route_sha256s=(self.two.sha256,),
        )
        assert result.receipt.ppm_prediction is not None
        self.assertEqual(
            result.receipt.ppm_prediction.selected_route_sha256,
            self.three.sha256,
        )
        self.assertEqual(
            result.receipt.selected_prefix_route_sha256,
            self.three.sha256,
        )

    def test_forged_verifier_binding_does_not_write_an_outcome(self) -> None:
        def forged(
            execution: ResidualRouteExecution,
        ) -> DemandExecutionVerification:
            return DemandExecutionVerification(
                residual_receipt_sha256=_digest("other-residual"),
                output_sha256=execution.receipt.output_sha256,
                accepted=True,
                verifier_sha256=_digest("forged-verifier"),
                evidence_sha256=_digest("forged-evidence"),
                reason="forged-binding",
            )

        executor = DemandRoutedExecutor(
            graph=self.graph,
            scheduler=self.scheduler,
            verifier=forged,
        )
        with self.assertRaisesRegex(DemandExecutionIntegrityError, "another"):
            executor.execute(
                self.plan,
                self.value,
                candidate_route_sha256s=self.candidates,
            )
        state = self.scheduler.state()
        self.assertEqual(state.generation, 2)
        self.assertIsInstance(state.events[-1], SelectionAbortEvent)
        aborted_route = state.events[0].route_sha256
        retry = self.scheduler.select_ucb1(
            self.graph_state,
            input_abi_sha256=self.two.input_abi_sha256,
            candidate_route_sha256s=self.candidates,
        )
        self.assertEqual(retry.route_sha256, aborted_route)

    def test_verifier_exception_is_neutralized_without_negative_feedback(self) -> None:
        def crashing(_execution: ResidualRouteExecution) -> DemandExecutionVerification:
            raise RuntimeError("synthetic verifier crash")

        executor = DemandRoutedExecutor(
            graph=self.graph,
            scheduler=self.scheduler,
            verifier=crashing,
        )
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            executor.execute(
                self.plan,
                self.value,
                candidate_route_sha256s=self.candidates,
            )
        state = self.scheduler.state()
        self.assertEqual(state.generation, 2)
        self.assertIsInstance(state.events[-1], SelectionAbortEvent)
        self.assertFalse(
            any(
                isinstance(event, OutcomeEvent)
                for event in state.events
            )
        )

    def test_invalid_input_abi_is_rejected_before_a_demand_pull(self) -> None:
        executor, _ = self._executor()
        with self.assertRaises(ComputeCrystalABIError):
            executor.execute(
                self.plan,
                self.value.astype(np.float32),
                candidate_route_sha256s=self.candidates,
            )
        self.assertEqual(self.scheduler.state().generation, 0)

    def test_unavailable_prefix_and_resealed_joined_tamper_fail_closed(self) -> None:
        executor, _ = self._executor()
        with self.assertRaises(DemandExecutionUnavailableError):
            executor.execute(
                self.plan,
                self.value,
                candidate_route_sha256s=(),
            )
        result = executor.execute(
            self.plan,
            self.value,
            candidate_route_sha256s=self.candidates,
        )
        document = json.loads(result.receipt.to_bytes())
        document["body"]["selected_prefix_route_sha256"] = _digest("forged-prefix")
        document["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(document["body"])
        ).hexdigest()
        with self.assertRaises(DemandExecutionIntegrityError):
            DemandRoutedExecutionReceipt.from_bytes(canonical_json_bytes(document))

        event_document = json.loads(result.receipt.to_bytes())
        transition = event_document["body"]["outcome_transition"]
        transition["body"]["event_sha256"] = _digest("another-outcome-event")
        transition["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(transition["body"])
        ).hexdigest()
        event_document["body"]["outcome_transition_sha256"] = hashlib.sha256(
            canonical_json_bytes(transition)
        ).hexdigest()
        event_document["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(event_document["body"])
        ).hexdigest()
        with self.assertRaises(DemandExecutionIntegrityError):
            DemandRoutedExecutionReceipt.from_bytes(
                canonical_json_bytes(event_document)
            )

        chain_document = json.loads(result.receipt.to_bytes())
        transition = chain_document["body"]["outcome_transition"]
        transition["body"]["previous_state_sha256"] = _digest("other-state")
        transition["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(transition["body"])
        ).hexdigest()
        chain_document["body"]["outcome_transition_sha256"] = hashlib.sha256(
            canonical_json_bytes(transition)
        ).hexdigest()
        chain_document["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(chain_document["body"])
        ).hexdigest()
        with self.assertRaises(DemandExecutionIntegrityError):
            DemandRoutedExecutionReceipt.from_bytes(
                canonical_json_bytes(chain_document)
            )


if __name__ == "__main__":
    unittest.main()
