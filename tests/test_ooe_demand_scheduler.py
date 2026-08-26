from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import math
from pathlib import Path
import tempfile
import threading
import unittest

from immer.runtimes.ooe.compute_crystals import ComputeCrystal, ComputeCrystalBank
from immer.runtimes.ooe.compute_graph import (
    ComputeOperatorGraph,
    ComputeOperatorGraphState,
    MaterializedRoute,
    OperatorEdge,
)
from immer.runtimes.ooe.demand_scheduler import (
    DEMAND_STATE_NAME,
    DemandOutcomeReceipt,
    OperatorDemandBudgetError,
    OperatorDemandConfig,
    OperatorDemandConflictError,
    OperatorDemandIntegrityError,
    OperatorDemandScheduler,
    OperatorDemandState,
    SelectionAbortEvent,
    verify_demand_receipt,
)
from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe import demand_scheduler as demand_module


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class OperatorDemandSchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.bank = ComputeCrystalBank(self.temporary.name)
        self.graph = ComputeOperatorGraph(self.bank)
        self.plans = []
        sources = ("alpha", "beta", "gamma")
        for index, source in enumerate(sources):
            crystal = ComputeCrystal.affine([[float(index + 1)]], [float(index)])
            self.bank.publish_crystal(crystal)
            self.graph.append_edge(
                OperatorEdge(
                    source_state=source,
                    target_state=f"{source}-terminal",
                    crystal_sha256=crystal.sha256,
                    verifier_sha256=_digest(f"route-verifier-{index}"),
                    evidence_sha256=_digest(f"route-evidence-{index}"),
                )
            )
            decision = self.graph.plan_route(source, f"{source}-terminal")
            assert decision.plan is not None
            self.plans.append(decision.plan)
            self.graph.charge_route(decision.plan)
        self.graph_state = self.graph.state()
        self.routes_by_source = {
            route.source_state: route for route in self.graph_state.materialized_routes
        }
        self.routes = tuple(sorted(self.routes_by_source.values(), key=lambda r: r.sha256))
        self.input_abi = self.routes[0].input_abi_sha256
        self.output_abi = self.routes[0].output_abi_sha256
        self.scheduler = OperatorDemandScheduler(
            self.bank.store,
            config=OperatorDemandConfig(
                exploration=1.0,
                max_context_order=4,
                cooccurrence_window=4,
            ),
        )
        self.outcome_ordinal = 0

    def _make_outcome(
        self,
        route: MaterializedRoute,
        *,
        success: bool = True,
        reward: float | None = None,
        selection_event_sha256: str | None = None,
        episode: str | None = None,
        graph_state: ComputeOperatorGraphState | None = None,
    ) -> DemandOutcomeReceipt:
        ordinal = self.outcome_ordinal
        self.outcome_ordinal += 1
        state = graph_state or self.graph_state
        return DemandOutcomeReceipt.create(
            route=route,
            graph_state=state,
            success=success,
            reward=(1.0 if success else -1.0) if reward is None else reward,
            outcome_evidence_sha256=_digest(f"outcome-evidence-{ordinal}"),
            outcome_verifier_sha256=_digest(f"outcome-verifier-{ordinal}"),
            selection_event_sha256=selection_event_sha256,
            episode_sha256=None if episode is None else _digest(episode),
        )

    def _record_sequence(
        self,
        episode: str,
        routes: tuple[MaterializedRoute, ...],
    ) -> None:
        for route in routes:
            self.scheduler.record_outcome(
                self._make_outcome(route, episode=episode), self.graph_state
            )

    def test_outcome_receipts_are_sealed_immutable_and_provenance_bound(self) -> None:
        route = self.routes_by_source["alpha"]
        positive = self._make_outcome(route, reward=2.5, episode="verified-episode")
        negative = self._make_outcome(route, success=False, reward=-3.0)

        self.assertEqual(DemandOutcomeReceipt.from_bytes(positive.to_bytes()), positive)
        self.assertEqual(verify_demand_receipt(positive.to_bytes()), positive.sha256)
        self.assertEqual(positive.route_verifier_sha256s, route.verifier_sha256s)
        self.assertEqual(positive.route_evidence_sha256s, route.evidence_sha256s)
        self.assertIsNotNone(positive.outcome_verifier_sha256)
        self.assertIsNotNone(negative.outcome_verifier_sha256)
        with self.assertRaises(FrozenInstanceError):
            positive.reward = 99.0  # type: ignore[misc]
        with self.assertRaises(OperatorDemandIntegrityError):
            DemandOutcomeReceipt.from_bytes(positive.to_bytes() + b"\n")
        with self.assertRaises(OperatorDemandIntegrityError):
            verify_demand_receipt(positive.to_bytes() + b"\n")

        invalid = positive.to_dict()
        invalid["body"]["reward"] = -1.0
        invalid["body_sha256"] = demand_module._sha256(invalid["body"])
        with self.assertRaises(OperatorDemandIntegrityError):
            verify_demand_receipt(canonical_json_bytes(invalid))

        forged = replace(
            positive,
            route_evidence_sha256s=(_digest("forged-route-evidence"),),
        )
        with self.assertRaisesRegex(
            OperatorDemandIntegrityError, "provenance binding"
        ):
            self.scheduler.record_outcome(forged, self.graph_state)

    def test_ucb1_is_unseen_first_and_sha_tie_broken(self) -> None:
        first = self.scheduler.select_ucb1(
            self.graph_state, input_abi_sha256=self.input_abi
        )
        self.assertEqual(verify_demand_receipt(first.to_bytes()), first.sha256)
        self.assertEqual(
            verify_demand_receipt(first.transition.to_bytes()),
            first.transition.sha256,
        )
        self.assertEqual(first.route_sha256, min(route.sha256 for route in self.routes))
        self.assertTrue(all(score.score is None for score in first.event.scores))

        route = OperatorDemandScheduler.resolve_materialized_candidate(
            self.graph_state, first.route_sha256
        )
        outcome = self._make_outcome(
            route,
            reward=4.0,
            selection_event_sha256=first.selection_event_sha256,
        )
        self.scheduler.record_outcome(outcome, self.graph_state)
        second = self.scheduler.select_ucb1(
            self.graph_state, input_abi_sha256=self.input_abi
        )
        remaining = sorted(
            item.sha256 for item in self.routes if item.sha256 != first.route_sha256
        )
        self.assertEqual(second.route_sha256, remaining[0])
        self.assertEqual(second.event.logical_time, 3)
        self.assertEqual(
            first.transition.event_sha256, first.selection_event_sha256
        )

    def test_route_plan_and_materialized_route_candidates_resolve_exactly(self) -> None:
        plan = self.plans[0]
        route = OperatorDemandScheduler.resolve_materialized_candidate(
            self.graph_state, plan
        )
        self.assertEqual(route.source_state, "alpha")
        self.assertEqual(
            OperatorDemandScheduler.resolve_materialized_candidate(
                self.graph_state, route
            ),
            route,
        )
        selection = self.scheduler.select_ucb1(
            self.graph_state,
            input_abi_sha256=self.input_abi,
            candidate_routes_or_plans=(plan,),
        )
        self.assertEqual(selection.route_sha256, route.sha256)
        with self.assertRaises(ValueError):
            self.scheduler.select_ucb1(
                self.graph_state,
                input_abi_sha256=self.input_abi,
                candidate_route_sha256s=(route.sha256,),
                candidate_routes_or_plans=(route,),
            )

    def test_positive_and_negative_updates_require_exact_revision_route_and_abi(self) -> None:
        route = self.routes_by_source["beta"]
        negative = self._make_outcome(route, success=False, reward=-2.0)
        recorded = self.scheduler.record_outcome(negative, self.graph_state)
        self.assertTrue(recorded.changed)
        state = self.scheduler.state()
        self.assertEqual(state.generation, 1)

        wrong_abi = replace(negative, input_abi_sha256=_digest("wrong-abi"))
        with self.assertRaisesRegex(OperatorDemandIntegrityError, "ABI"):
            self.scheduler.record_outcome(wrong_abi, self.graph_state)

        crystal = ComputeCrystal.affine([[1.0]], [0.0])
        self.bank.publish_crystal(crystal)
        self.graph.append_edge(
            OperatorEdge(
                "delta",
                "delta-terminal",
                crystal.sha256,
                _digest("delta-verifier"),
                _digest("delta-evidence"),
            )
        )
        newer = self.graph.state()
        with self.assertRaisesRegex(OperatorDemandIntegrityError, "revision"):
            self.scheduler.record_outcome(negative, newer)

    def test_latest_verified_negative_blocks_exact_route_until_positive_recovery(
        self,
    ) -> None:
        beta = self.routes_by_source["beta"]
        gamma = self.routes_by_source["gamma"]
        self.scheduler.record_outcome(
            self._make_outcome(beta, success=False), self.graph_state
        )

        chosen = self.scheduler.select_ucb1(
            self.graph_state,
            input_abi_sha256=self.input_abi,
            candidate_routes_or_plans=(beta, gamma),
        )
        self.assertEqual(chosen.route_sha256, gamma.sha256)
        with self.assertRaisesRegex(
            demand_module.OperatorDemandUnavailableError, "negative"
        ):
            self.scheduler.select_ucb1(
                self.graph_state,
                input_abi_sha256=self.input_abi,
                candidate_routes_or_plans=(beta,),
            )

        self.scheduler.record_outcome(self._make_outcome(beta), self.graph_state)
        recovered = self.scheduler.select_ucb1(
            self.graph_state,
            input_abi_sha256=self.input_abi,
            candidate_routes_or_plans=(beta,),
        )
        self.assertEqual(recovered.route_sha256, beta.sha256)

    def test_duplicate_outcome_is_idempotent_and_evidence_forks_fail(self) -> None:
        route = self.routes_by_source["alpha"]
        outcome = self._make_outcome(route)
        first = self.scheduler.record_outcome(outcome, self.graph_state)
        second = self.scheduler.record_outcome(outcome, self.graph_state)
        self.assertTrue(first.changed)
        self.assertFalse(second.changed)
        self.assertEqual(first.current_state_sha256, second.current_state_sha256)
        self.assertEqual(first.event_sha256, second.event_sha256)
        self.assertEqual(self.scheduler.state().generation, 1)

        conflicting = replace(
            self._make_outcome(route, reward=7.0),
            outcome_evidence_sha256=outcome.outcome_evidence_sha256,
        )
        with self.assertRaisesRegex(OperatorDemandConflictError, "already"):
            self.scheduler.record_outcome(conflicting, self.graph_state)

    def test_one_selection_cannot_be_settled_twice(self) -> None:
        selection = self.scheduler.select_ucb1(
            self.graph_state,
            input_abi_sha256=self.input_abi,
            candidate_routes_or_plans=(self.routes_by_source["alpha"],),
        )
        route = self.routes_by_source["alpha"]
        first = self._make_outcome(
            route, selection_event_sha256=selection.selection_event_sha256
        )
        second = self._make_outcome(
            route,
            reward=2.0,
            selection_event_sha256=selection.selection_event_sha256,
        )
        self.scheduler.record_outcome(first, self.graph_state)
        with self.assertRaisesRegex(OperatorDemandConflictError, "already"):
            self.scheduler.record_outcome(second, self.graph_state)

    def test_operational_abort_neutralizes_pull_and_is_idempotent(self) -> None:
        alpha = self.routes_by_source["alpha"]
        selection = self.scheduler.select_ucb1(
            self.graph_state,
            input_abi_sha256=self.input_abi,
            candidate_routes_or_plans=(alpha,),
        )
        kwargs = {
            "abort_evidence_sha256": _digest("abort-evidence"),
            "abort_verifier_sha256": _digest("abort-verifier"),
            "reason_code": "synthetic-operational-failure",
        }
        first = self.scheduler.abort_selection(
            selection,
            self.graph_state,
            **kwargs,
        )
        second = self.scheduler.abort_selection(
            selection,
            self.graph_state,
            **kwargs,
        )
        self.assertTrue(first.changed)
        self.assertFalse(second.changed)
        state = self.scheduler.state()
        self.assertIsInstance(state.events[-1], SelectionAbortEvent)
        retry = self.scheduler.select_ucb1(
            self.graph_state,
            input_abi_sha256=self.input_abi,
            candidate_routes_or_plans=(alpha,),
        )
        self.assertEqual(retry.event.scores[0].pulls, 0)
        outcome = self._make_outcome(
            alpha,
            selection_event_sha256=selection.selection_event_sha256,
        )
        with self.assertRaisesRegex(OperatorDemandIntegrityError, "aborted"):
            self.scheduler.record_outcome(outcome, self.graph_state)

    def test_selection_outcome_cannot_cross_graph_revisions(self) -> None:
        alpha = self.routes_by_source["alpha"]
        selection = self.scheduler.select_ucb1(
            self.graph_state,
            input_abi_sha256=self.input_abi,
            candidate_routes_or_plans=(alpha,),
        )
        crystal = ComputeCrystal.affine([[1.0]], [0.0])
        self.bank.publish_crystal(crystal)
        self.graph.append_edge(
            OperatorEdge(
                "new-source",
                "new-terminal",
                crystal.sha256,
                _digest("new-verifier"),
                _digest("new-evidence"),
            )
        )
        newer = self.graph.state()
        same_route = next(
            route for route in newer.materialized_routes if route.sha256 == alpha.sha256
        )
        outcome = self._make_outcome(
            same_route,
            selection_event_sha256=selection.selection_event_sha256,
            graph_state=newer,
        )
        with self.assertRaisesRegex(OperatorDemandIntegrityError, "revision"):
            self.scheduler.record_outcome(outcome, newer)

    def test_ppm_uses_longest_terminal_prefix_and_verified_successes_only(self) -> None:
        alpha = self.routes_by_source["alpha"]
        beta = self.routes_by_source["beta"]
        gamma = self.routes_by_source["gamma"]
        self._record_sequence("episode-a", (alpha, beta))
        self._record_sequence("episode-b", (alpha, beta))
        self._record_sequence("episode-c", (alpha, gamma))
        self.scheduler.record_outcome(
            self._make_outcome(beta, success=False, reward=-50.0), self.graph_state
        )

        receipt = self.scheduler.predict_ppm(
            self.graph_state,
            (gamma.sha256, alpha.sha256),
            input_abi_sha256=self.input_abi,
        )
        self.assertEqual(receipt.matched_terminal_prefix, (alpha.sha256,))
        self.assertEqual(receipt.selected_route_sha256, gamma.sha256)
        self.assertNotIn(beta.sha256, dict(receipt.counts))
        self.assertEqual(dict(receipt.counts)[gamma.sha256], 1)
        self.assertEqual(receipt.probability, 1.0)
        self.assertTrue(receipt.evidence_receipt_sha256s)
        self.assertEqual(verify_demand_receipt(receipt.to_bytes()), receipt.sha256)

        with self.assertRaisesRegex(
            OperatorDemandIntegrityError, "non-materialized"
        ):
            self.scheduler.predict_ppm(
                self.graph_state,
                (_digest("unknown-route"),),
                input_abi_sha256=self.input_abi,
            )

    def test_async_outcome_settlement_preserves_selection_execution_order(self) -> None:
        alpha = self.routes_by_source["alpha"]
        beta = self.routes_by_source["beta"]
        episode = "async-episode"
        selected_alpha = self.scheduler.select_ucb1(
            self.graph_state,
            input_abi_sha256=self.input_abi,
            candidate_routes_or_plans=(alpha,),
        )
        selected_beta = self.scheduler.select_ucb1(
            self.graph_state,
            input_abi_sha256=self.input_abi,
            candidate_routes_or_plans=(beta,),
        )
        # The verifier finishes beta first.  Sequence learning must still use
        # the selection/execution order alpha -> beta.
        self.scheduler.record_outcome(
            self._make_outcome(
                beta,
                selection_event_sha256=selected_beta.selection_event_sha256,
                episode=episode,
            ),
            self.graph_state,
        )
        self.scheduler.record_outcome(
            self._make_outcome(
                alpha,
                selection_event_sha256=selected_alpha.selection_event_sha256,
                episode=episode,
            ),
            self.graph_state,
        )

        prediction = self.scheduler.predict_ppm(
            self.graph_state,
            (alpha.sha256,),
            input_abi_sha256=self.input_abi,
        )
        self.assertEqual(prediction.selected_route_sha256, beta.sha256)
        self.assertEqual(prediction.matched_terminal_prefix, (alpha.sha256,))

    def test_standalone_outcomes_do_not_fabricate_sequence_or_cooccurrence(
        self,
    ) -> None:
        alpha = self.routes_by_source["alpha"]
        beta = self.routes_by_source["beta"]
        self.scheduler.record_outcome(self._make_outcome(alpha), self.graph_state)
        self.scheduler.record_outcome(self._make_outcome(beta), self.graph_state)

        prediction = self.scheduler.predict_ppm(
            self.graph_state,
            (alpha.sha256,),
            input_abi_sha256=self.input_abi,
        )
        self.assertIsNone(prediction.selected_route_sha256)
        self.assertFalse(prediction.counts)
        self.assertFalse(self.scheduler.cooccurrence(self.graph_state).entries)

    def test_ppm_backs_off_to_zero_order_and_abstains_on_abi_miss(self) -> None:
        alpha = self.routes_by_source["alpha"]
        beta = self.routes_by_source["beta"]
        self._record_sequence("episode", (alpha, beta))
        zero_order = self.scheduler.predict_ppm(
            self.graph_state, (), input_abi_sha256=self.input_abi
        )
        self.assertEqual(zero_order.matched_terminal_prefix, ())
        self.assertIsNotNone(zero_order.selected_route_sha256)

        miss = self.scheduler.predict_ppm(
            self.graph_state, (), input_abi_sha256=_digest("absent-abi")
        )
        self.assertIsNone(miss.selected_route_sha256)
        self.assertEqual(miss.reason, "no-verified-prefix-evidence")

    def test_verified_cooccurrence_drives_exact_budgeted_prefetch(self) -> None:
        alpha = self.routes_by_source["alpha"]
        beta = self.routes_by_source["beta"]
        gamma = self.routes_by_source["gamma"]
        self._record_sequence("episode-a", (alpha, beta))
        self._record_sequence("episode-b", (alpha, beta))
        self._record_sequence("episode-c", (alpha, gamma))
        matrix = self.scheduler.cooccurrence(self.graph_state)
        pairs = {
            (entry.left_route_sha256, entry.right_route_sha256): entry.count
            for entry in matrix.entries
        }
        self.assertEqual(pairs[tuple(sorted((alpha.sha256, beta.sha256)))], 2)
        self.assertEqual(pairs[tuple(sorted((alpha.sha256, gamma.sha256)))], 1)
        self.assertTrue(all(entry.evidence_receipt_sha256s for entry in matrix.entries))
        self.assertEqual(verify_demand_receipt(matrix.to_bytes()), matrix.sha256)

        sizes = {route.sha256: 10 for route in self.routes}
        sizes[beta.sha256] = 20
        receipt = self.scheduler.prefetch(
            self.graph_state,
            (alpha.sha256,),
            input_abi_sha256=self.input_abi,
            max_items=1,
            max_bytes=15,
            payload_bytes_by_route=sizes,
        )
        self.assertEqual(receipt.selected_route_sha256s, (gamma.sha256,))
        self.assertEqual(receipt.total_bytes, 10)
        self.assertLessEqual(receipt.total_bytes, receipt.max_bytes)
        self.assertEqual(receipt.cooccurrence_receipt_sha256, matrix.sha256)
        self.assertEqual(verify_demand_receipt(receipt.to_bytes()), receipt.sha256)
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            self.scheduler.prefetch(
                self.graph_state,
                (alpha.sha256,),
                input_abi_sha256=self.input_abi,
                max_items=1,
                max_bytes=100,
                payload_bytes_by_route={alpha.sha256: 1},
            )

    def test_default_byte_inventory_counts_programs_crystals_and_charge(self) -> None:
        alpha = self.routes_by_source["alpha"]
        sizes, inventory_sha = self.scheduler._size_inventory(
            self.graph_state, None
        )
        self.assertEqual(len(inventory_sha), 64)
        self.assertGreater(sizes[alpha.sha256], len(alpha.to_bytes()))

        receipt = self.scheduler.retention(
            self.graph_state,
            (alpha.sha256,),
            alpha=0.001,
            max_items=1,
            max_bytes=len(alpha.to_bytes()),
        )
        self.assertFalse(receipt.kept_route_sha256s)
        self.assertEqual(receipt.evicted_route_sha256s, (alpha.sha256,))
        self.assertEqual(receipt.total_bytes, 0)

    def test_ricci_retention_uses_exact_formula_and_never_evicts_pins(self) -> None:
        alpha = self.routes_by_source["alpha"]
        beta = self.routes_by_source["beta"]
        gamma = self.routes_by_source["gamma"]
        self.scheduler.record_outcome(
            self._make_outcome(alpha, reward=10.0), self.graph_state
        )
        self.scheduler.record_outcome(
            self._make_outcome(beta, success=False, reward=-20.0), self.graph_state
        )
        self.scheduler.record_outcome(
            self._make_outcome(gamma, reward=1.0), self.graph_state
        )
        pin = self.scheduler.set_pins(self.graph_state, (gamma.sha256,))
        duplicate = self.scheduler.set_pins(self.graph_state, (gamma.sha256,))
        self.assertTrue(pin.changed)
        self.assertFalse(duplicate.changed)

        sizes = {route.sha256: 10 for route in self.routes}
        receipt = self.scheduler.retention(
            self.graph_state,
            tuple(sorted(sizes)),
            alpha=0.1,
            max_items=1,
            max_bytes=10,
            payload_bytes_by_route=sizes,
        )
        self.assertEqual(receipt.kept_route_sha256s, (gamma.sha256,))
        by_route = {item.route_sha256: item for item in receipt.ranked}
        beta_item = by_route[beta.sha256]
        self.assertAlmostEqual(
            beta_item.keep_score,
            abs(beta_item.reward) * math.exp(-0.1 * beta_item.logical_age),
        )
        self.assertTrue(by_route[gamma.sha256].pinned)
        self.assertEqual(verify_demand_receipt(receipt.to_bytes()), receipt.sha256)

        with self.assertRaises(OperatorDemandBudgetError):
            self.scheduler.retention(
                self.graph_state,
                tuple(sorted(sizes)),
                alpha=0.1,
                max_items=1,
                max_bytes=9,
                payload_bytes_by_route=sizes,
            )
        with self.assertRaisesRegex(OperatorDemandIntegrityError, "pinned"):
            self.scheduler.retention(
                self.graph_state,
                tuple(sorted((alpha.sha256, beta.sha256))),
                alpha=0.1,
                max_items=1,
                max_bytes=10,
                payload_bytes_by_route=sizes,
            )

    def test_state_roundtrip_restart_and_stale_cas(self) -> None:
        initial = self.scheduler.state()
        selection = self.scheduler.select_ucb1(
            self.graph_state, input_abi_sha256=self.input_abi
        )
        current = self.scheduler.state()
        self.assertEqual(OperatorDemandState.from_bytes(current.to_bytes()), current)
        with self.assertRaises(OperatorDemandIntegrityError):
            OperatorDemandState.from_bytes(current.to_bytes() + b"\n")

        restarted = OperatorDemandScheduler(
            self.bank.store, config=self.scheduler.config
        )
        self.assertEqual(restarted.state(), current)
        self.assertEqual(restarted.current_anchor_sha256(), current.sha256)
        self.assertEqual(selection.transition.current_state_sha256, current.sha256)
        with self.assertRaisesRegex(OperatorDemandConflictError, "stale"):
            restarted.select_ucb1(
                self.graph_state,
                input_abi_sha256=self.input_abi,
                expected_state_sha256=initial.sha256,
            )

    def test_concurrent_scheduler_instances_do_not_lose_pulls(self) -> None:
        errors: list[BaseException] = []
        receipts = []

        def select() -> None:
            try:
                instance = OperatorDemandScheduler(
                    self.bank.store, config=self.scheduler.config
                )
                receipts.append(
                    instance.select_ucb1(
                        self.graph_state, input_abi_sha256=self.input_abi
                    )
                )
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        threads = [threading.Thread(target=select) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(receipts), 8)
        state = self.scheduler.state()
        self.assertEqual(state.generation, 8)
        self.assertEqual(len({receipt.selection_event_sha256 for receipt in receipts}), 8)

    def test_validly_resealed_rollback_is_detected(self) -> None:
        first = self.scheduler.select_ucb1(
            self.graph_state, input_abi_sha256=self.input_abi
        )
        first_state = self.scheduler.state()
        route = OperatorDemandScheduler.resolve_materialized_candidate(
            self.graph_state, first.route_sha256
        )
        self.scheduler.record_outcome(
            self._make_outcome(
                route, selection_event_sha256=first.selection_event_sha256
            ),
            self.graph_state,
        )
        current = self.scheduler.state()
        self.bank.store.publish_state(
            DEMAND_STATE_NAME,
            first_state.to_bytes(),
            expected_sha256=current.sha256,
        )
        with self.assertRaisesRegex(OperatorDemandIntegrityError, "rollback"):
            OperatorDemandScheduler(
                self.bank.store, config=self.scheduler.config
            ).state()

    def test_missing_commit_after_pointer_publish_is_recovered(self) -> None:
        self.scheduler.select_ucb1(
            self.graph_state, input_abi_sha256=self.input_abi
        )
        current = self.scheduler.state()
        name = self.scheduler.commit_state_name(current.sha256)
        path = (
            Path(self.bank.store.root)
            / "state"
            / self.bank.store._state_filename(name)
        )
        path.unlink()
        restarted = OperatorDemandScheduler(
            self.bank.store, config=self.scheduler.config
        )
        self.assertEqual(restarted.state(), current)
        self.assertTrue(path.exists())

    def test_committed_history_fork_is_detected(self) -> None:
        self.scheduler.select_ucb1(
            self.graph_state, input_abi_sha256=self.input_abi
        )
        parent = self.scheduler.state()
        self.scheduler.set_pins(
            self.graph_state, (self.routes_by_source["alpha"].sha256,)
        )
        alternate_event = demand_module.PinsEvent(
            logical_time=2,
            graph_generation=self.graph_state.generation,
            graph_state_sha256=self.graph_state.sha256,
            route_sha256s=(self.routes_by_source["beta"].sha256,),
        )
        alternate = OperatorDemandState(
            generation=2,
            previous_state_sha256=parent.sha256,
            config_sha256=parent.config_sha256,
            events=(*parent.events, alternate_event),
        )
        self.bank.store.publish_state(
            self.scheduler.history_state_name(alternate.sha256), alternate.to_bytes()
        )
        self.bank.store.publish_state(
            self.scheduler.commit_state_name(alternate.sha256),
            demand_module._commit_bytes(alternate.sha256),
        )
        with self.assertRaisesRegex(OperatorDemandIntegrityError, "fork"):
            OperatorDemandScheduler(
                self.bank.store, config=self.scheduler.config
            ).state()

    def test_controller_state_tamper_and_configuration_mismatch_fail_closed(self) -> None:
        self.scheduler.select_ucb1(
            self.graph_state, input_abi_sha256=self.input_abi
        )
        with self.assertRaisesRegex(OperatorDemandIntegrityError, "configuration"):
            OperatorDemandScheduler(
                self.bank.store,
                config=OperatorDemandConfig(exploration=2.0),
            ).state()

        filename = self.bank.store._state_filename(DEMAND_STATE_NAME)
        path = Path(self.bank.store.root) / "state" / filename
        envelope = bytearray(path.read_bytes())
        marker = b'"payload_base64":"'
        index = envelope.index(marker) + len(marker)
        envelope[index] = ord("A") if envelope[index] != ord("A") else ord("B")
        path.chmod(0o600)
        path.write_bytes(envelope)
        with self.assertRaises(OperatorDemandIntegrityError):
            self.scheduler.state()

    def test_trusted_exact_head_detects_external_rollback(self) -> None:
        empty_anchor = self.scheduler.state().sha256
        anchored = OperatorDemandScheduler(
            self.bank.store,
            config=self.scheduler.config,
            trusted_state_sha256=empty_anchor,
        )
        anchored.select_ucb1(
            self.graph_state, input_abi_sha256=self.input_abi
        )
        new_anchor = anchored.current_anchor_sha256()
        self.assertNotEqual(new_anchor, empty_anchor)
        exact = OperatorDemandScheduler(
            self.bank.store,
            config=self.scheduler.config,
            trusted_state_sha256=new_anchor,
        )
        self.assertEqual(exact.state().sha256, new_anchor)
        with self.assertRaisesRegex(OperatorDemandIntegrityError, "trusted"):
            OperatorDemandScheduler(
                self.bank.store,
                config=self.scheduler.config,
                trusted_state_sha256=empty_anchor,
            ).state()


if __name__ == "__main__":
    unittest.main()
