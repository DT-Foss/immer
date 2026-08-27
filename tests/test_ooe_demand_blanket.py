from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib
import json
import tempfile
import unittest

import numpy as np

from immer.runtimes.ooe.compute_crystals import ComputeCrystal, ComputeCrystalBank
from immer.runtimes.ooe.compute_graph import ComputeOperatorGraph, OperatorEdge
from immer.runtimes.ooe.conditional_blanket import ConditionalBlanketConfig
from immer.runtimes.ooe.demand_blanket import (
    DEMAND_LAG_BOS,
    DemandBlanketIntegrityError,
    DemandEpisodeSplitReceipt,
    DemandLagBlanketFitReceipt,
    DemandLagBlanketPredictionReceipt,
    DemandLagBlanketValidationReceipt,
    DemandLagCorpusReceipt,
    DemandLagSample,
    build_demand_lag_corpus,
    chronological_episode_group_split,
    fit_demand_lag_blanket,
    predict_demand_lag_blanket,
    validate_demand_lag_blanket,
)
from immer.runtimes.ooe.demand_execution import (
    DemandExecutionIntegrityError,
    DemandExecutionVerification,
    DemandRoutedExecutionReceipt,
    DemandRoutedExecutor,
)
from immer.runtimes.ooe.demand_scheduler import (
    DemandOutcomeReceipt,
    OperatorDemandConfig,
    OperatorDemandScheduler,
)
from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.residual_execution import ResidualRouteExecution


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class _ExactVerifier:
    def __init__(self, expected: np.ndarray) -> None:
        self.expected = expected
        self.calls = 0

    def __call__(
        self, execution: ResidualRouteExecution
    ) -> DemandExecutionVerification:
        self.calls += 1
        return DemandExecutionVerification.create(
            execution,
            accepted=np.array_equal(execution.output, self.expected),
            verifier_sha256=_digest("blanket-exact-verifier"),
            evidence_sha256=_digest(f"blanket-exact-evidence:{self.calls}"),
            reason="exact-output-match",
        )


class DemandLagBlanketTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.bank = ComputeCrystalBank(self.temporary.name)
        self.graph = ComputeOperatorGraph(self.bank)
        self.crystals = (
            ComputeCrystal.affine([[2.0]], [1.0]),
            ComputeCrystal.affine([[-3.0]], [4.0]),
            ComputeCrystal.affine([[0.5]], [-2.0]),
        )
        states = ("s0", "s1", "s2", "s3")
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
        three_plan = self.graph.plan_route("s0", "s3").plan
        assert two_plan is not None and three_plan is not None
        self.two = self.graph.charge_route(two_plan).route
        self.three = self.graph.charge_route(three_plan).route
        self.plan = self.graph.plan_route("s0", "s3").plan
        assert self.plan is not None
        self.graph_state = self.graph.state()
        self.scheduler = OperatorDemandScheduler(
            self.bank.store,
            config=OperatorDemandConfig(max_context_order=3),
        )
        self.value = np.array([[-7.25], [0.125], [91.0]], dtype=np.float64)
        self.expected = self.value
        for crystal in self.crystals:
            self.expected = crystal.apply(self.expected)

    def _synthetic_xor_corpus(self) -> DemandLagCorpusReceipt:
        routes = tuple(sorted((self.two.sha256, self.three.sha256)))
        samples: list[DemandLagSample] = []
        logical_time = 1
        for episode_index in range(24):
            bits = [
                episode_index & 1,
                (episode_index >> 1) & 1,
                (episode_index >> 2) & 1,
            ]
            for position in range(3, 14):
                bits.append(bits[position - 1] ^ bits[position - 3])
            sequence = tuple(routes[bit] for bit in bits)
            episode = _digest(f"xor-episode:{episode_index}")
            for position, target in enumerate(sequence):
                lags = tuple(
                    sequence[position - lag]
                    if position >= lag
                    else DEMAND_LAG_BOS
                    for lag in range(1, 4)
                )
                samples.append(
                    DemandLagSample(
                        logical_time=logical_time,
                        settlement_logical_time=logical_time,
                        episode_sha256=episode,
                        source_sha256=episode,
                        position=position,
                        lag_route_sha256s=lags,
                        target_route_sha256=target,
                        outcome_receipt_sha256=_digest(
                            f"xor-outcome:{logical_time}"
                        ),
                        outcome_verifier_sha256=_digest("xor-verifier"),
                        outcome_evidence_sha256=_digest(
                            f"xor-evidence:{logical_time}"
                        ),
                        selection_event_sha256=None,
                    )
                )
                logical_time += 1
        state = self.scheduler.state()
        return DemandLagCorpusReceipt(
            scheduler_state_sha256=state.sha256,
            scheduler_config_sha256=state.config_sha256,
            graph_generation=self.graph_state.generation,
            graph_state_sha256=self.graph_state.sha256,
            input_abi_sha256=self.two.input_abi_sha256,
            output_abi_sha256=None,
            max_context_order=3,
            bos_category=DEMAND_LAG_BOS,
            route_alphabet_sha256s=routes,
            samples=tuple(samples),
            sample_sha256s=tuple(row.sha256 for row in samples),
            outcome_receipt_sha256s=tuple(
                row.outcome_receipt_sha256 for row in samples
            ),
        )

    def _validation(self) -> DemandLagBlanketValidationReceipt:
        corpus = self._synthetic_xor_corpus()
        split = chronological_episode_group_split(corpus)
        config = ConditionalBlanketConfig(
            target_alphabet=corpus.route_alphabet_sha256s,
            max_subset_size=3,
            max_exhaustive_subsets=100,
            max_pair_checks=10,
            laplace_alpha=Fraction(1),
            max_brier_regret=Fraction(1, 20),
            max_accuracy_regret=Fraction(1, 20),
            max_coverage_regret=Fraction(1),
            dependence_tolerance=Fraction(1, 20),
        )
        fit = fit_demand_lag_blanket(corpus, split, config=config)
        return validate_demand_lag_blanket(
            fit, minimum_holdout_accuracy=Fraction(1, 2)
        )

    def test_async_verified_episode_derivation_uses_selection_chronology(self) -> None:
        episode = _digest("async-episode")
        selected_two = self.scheduler.select_ucb1(
            self.graph_state,
            input_abi_sha256=self.two.input_abi_sha256,
            candidate_routes_or_plans=(self.two,),
        )
        selected_three = self.scheduler.select_ucb1(
            self.graph_state,
            input_abi_sha256=self.two.input_abi_sha256,
            candidate_routes_or_plans=(self.three,),
        )
        for ordinal, (route, selection) in enumerate(
            ((self.three, selected_three), (self.two, selected_two))
        ):
            self.scheduler.record_outcome(
                DemandOutcomeReceipt.create(
                    route=route,
                    graph_state=self.graph_state,
                    success=True,
                    reward=1.0,
                    outcome_evidence_sha256=_digest(
                        f"async-outcome-evidence:{ordinal}"
                    ),
                    outcome_verifier_sha256=_digest("async-verifier"),
                    selection_event_sha256=selection.selection_event_sha256,
                    episode_sha256=episode,
                ),
                self.graph_state,
            )
        # Verified failure and successful standalone evidence are not allowed
        # to fabricate sequence rows.
        self.scheduler.record_outcome(
            DemandOutcomeReceipt.create(
                route=self.two,
                graph_state=self.graph_state,
                success=False,
                reward=-1.0,
                outcome_evidence_sha256=_digest("negative-evidence"),
                outcome_verifier_sha256=_digest("negative-verifier"),
                episode_sha256=_digest("negative-episode"),
            ),
            self.graph_state,
        )
        self.scheduler.record_outcome(
            DemandOutcomeReceipt.create(
                route=self.three,
                graph_state=self.graph_state,
                success=True,
                reward=1.0,
                outcome_evidence_sha256=_digest("standalone-evidence"),
                outcome_verifier_sha256=_digest("standalone-verifier"),
            ),
            self.graph_state,
        )

        corpus = build_demand_lag_corpus(
            self.scheduler,
            self.graph_state,
            input_abi_sha256=self.two.input_abi_sha256,
            max_context_order=3,
        )
        self.assertEqual(
            tuple(row.target_route_sha256 for row in corpus.samples),
            (self.two.sha256, self.three.sha256),
        )
        self.assertEqual(
            corpus.samples[1].lag_route_sha256s,
            (self.two.sha256, DEMAND_LAG_BOS, DEMAND_LAG_BOS),
        )
        self.assertEqual(DemandLagCorpusReceipt.from_bytes(corpus.to_bytes()), corpus)

    def test_xor_selects_noncontiguous_lags_and_roundtrips_all_receipts(self) -> None:
        validation = self._validation()
        fit = validation.fit
        split = fit.split

        self.assertEqual(fit.selected_lags, (1, 3))
        self.assertEqual(fit.generic_fit.status, "sparse")
        self.assertTrue(fit.generic_fit.closure_verified)
        self.assertTrue(validation.accepted)
        self.assertGreater(
            validation.generic_validation.selected_metrics.accuracy,
            validation.generic_validation.random_metrics.accuracy,
        )
        self.assertEqual(
            DemandEpisodeSplitReceipt.from_bytes(split.to_bytes()), split
        )
        self.assertEqual(
            DemandLagBlanketFitReceipt.from_bytes(fit.to_bytes()), fit
        )
        self.assertEqual(
            DemandLagBlanketValidationReceipt.from_bytes(
                validation.to_bytes()
            ),
            validation,
        )
        self.assertFalse(
            set(split.train_episode_sha256s)
            & set(split.calibration_episode_sha256s)
        )
        self.assertFalse(
            set(split.calibration_episode_sha256s)
            & set(split.holdout_episode_sha256s)
        )

    def test_prediction_survives_append_only_head_and_rejects_stale_pins(self) -> None:
        validation = self._validation()
        history = (
            validation.fit.corpus.route_alphabet_sha256s[0],
        ) * 3
        first = predict_demand_lag_blanket(
            validation,
            self.scheduler,
            self.graph_state,
            history,
            input_abi_sha256=self.two.input_abi_sha256,
            minimum_probability=Fraction(3, 4),
        )
        self.assertIsNotNone(first.candidate_route_sha256)
        self.assertTrue(first.context_seen)
        self.assertEqual(
            DemandLagBlanketPredictionReceipt.from_bytes(
                first.to_bytes(), validation=validation
            ),
            first,
        )
        long_history = predict_demand_lag_blanket(
            validation,
            self.scheduler,
            self.graph_state,
            (_digest("irrelevant-old-route"), *history),
            input_abi_sha256=self.two.input_abi_sha256,
            minimum_probability=Fraction(3, 4),
        )
        self.assertEqual(
            long_history.candidate_route_sha256,
            first.candidate_route_sha256,
        )
        self.assertEqual(
            long_history.lag_route_sha256s,
            first.lag_route_sha256s,
        )

        capacity_fit = fit_demand_lag_blanket(
            validation.fit.corpus,
            validation.fit.split,
            config=ConditionalBlanketConfig(
                target_alphabet=(
                    validation.fit.corpus.route_alphabet_sha256s
                ),
                max_subset_size=1,
                max_exhaustive_subsets=4,
                max_pair_checks=10,
                max_brier_regret=Fraction(1),
                max_accuracy_regret=Fraction(1),
                max_coverage_regret=Fraction(1),
                dependence_tolerance=Fraction(1),
            ),
        )
        capacity_validation = validate_demand_lag_blanket(
            capacity_fit, minimum_holdout_accuracy=Fraction()
        )
        self.assertTrue(capacity_fit.generic_fit.capacity_exhausted)
        self.assertFalse(capacity_validation.accepted)
        capacity_prediction = predict_demand_lag_blanket(
            capacity_validation,
            self.scheduler,
            self.graph_state,
            history,
            input_abi_sha256=self.two.input_abi_sha256,
        )
        self.assertIsNone(capacity_prediction.candidate_route_sha256)
        self.assertEqual(capacity_prediction.reason, "validation-rejected")
        # A caller can compute perfectly matching hashes for its own mutated
        # object.  Public authorization must still replay the frozen fit and
        # holdout semantics rather than treating those hashes as capabilities.
        object.__setattr__(capacity_validation, "accepted", True)
        object.__setattr__(
            capacity_validation, "reason", "validated-exact-holdout"
        )
        self.assertEqual(len(capacity_validation.sha256), 64)
        with self.assertRaises(DemandBlanketIntegrityError):
            predict_demand_lag_blanket(
                capacity_validation,
                self.scheduler,
                self.graph_state,
                history,
                input_abi_sha256=self.two.input_abi_sha256,
            )

        foreign_corpus = replace(
            validation.fit.corpus,
            scheduler_state_sha256=_digest("foreign-scheduler-head"),
        )
        foreign_split = chronological_episode_group_split(foreign_corpus)
        foreign_fit = fit_demand_lag_blanket(
            foreign_corpus,
            foreign_split,
            config=validation.fit.generic_fit.config,
        )
        foreign_validation = validate_demand_lag_blanket(
            foreign_fit, minimum_holdout_accuracy=Fraction(1, 2)
        )
        with self.assertRaisesRegex(DemandBlanketIntegrityError, "ancestor"):
            predict_demand_lag_blanket(
                foreign_validation,
                self.scheduler,
                self.graph_state,
                history,
                input_abi_sha256=self.two.input_abi_sha256,
            )

        self.scheduler.select_ucb1(
            self.graph_state,
            input_abi_sha256=self.two.input_abi_sha256,
            candidate_routes_or_plans=(self.two,),
        )
        later = predict_demand_lag_blanket(
            validation,
            self.scheduler,
            self.graph_state,
            history,
            input_abi_sha256=self.two.input_abi_sha256,
        )
        self.assertNotEqual(
            later.scheduler_state_sha256,
            later.fit_scheduler_state_sha256,
        )
        self.assertTrue(
            self.scheduler.is_state_ancestor(later.fit_scheduler_state_sha256)
        )

        with self.assertRaisesRegex(DemandBlanketIntegrityError, "ABI"):
            predict_demand_lag_blanket(
                validation,
                self.scheduler,
                self.graph_state,
                history,
                input_abi_sha256=_digest("wrong-abi"),
            )
        crystal = ComputeCrystal.affine([[1.0]], [0.0])
        self.bank.publish_crystal(crystal)
        self.graph.append_edge(
            OperatorEdge(
                "other",
                "other-terminal",
                crystal.sha256,
                _digest("other-verifier"),
                _digest("other-evidence"),
            )
        )
        with self.assertRaisesRegex(DemandBlanketIntegrityError, "graph"):
            predict_demand_lag_blanket(
                validation,
                self.scheduler,
                self.graph.state(),
                history,
                input_abi_sha256=self.two.input_abi_sha256,
            )

        document = json.loads(first.to_bytes())
        document["body"]["candidate_route_sha256"] = _digest("forged-route")
        document["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(document["body"])
        ).hexdigest()
        with self.assertRaises(DemandBlanketIntegrityError):
            DemandLagBlanketPredictionReceipt.from_bytes(
                canonical_json_bytes(document)
            )

        context_document = json.loads(first.to_bytes())
        context_document["body"]["context_seen"] = not context_document["body"][
            "context_seen"
        ]
        context_document["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(context_document["body"])
        ).hexdigest()
        with self.assertRaisesRegex(
            DemandBlanketIntegrityError, "prediction failed validation"
        ):
            DemandLagBlanketPredictionReceipt.from_bytes(
                canonical_json_bytes(context_document)
            )

        coordinated_context_document = json.loads(first.to_bytes())
        coordinated_context_document["body"]["context_seen"] = False
        coordinated_context_document["body"]["matched_cell_sha256"] = None
        coordinated_context_document["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(coordinated_context_document["body"])
        ).hexdigest()
        with self.assertRaisesRegex(
            DemandBlanketIntegrityError, "prediction cell"
        ):
            DemandLagBlanketPredictionReceipt.from_bytes(
                canonical_json_bytes(coordinated_context_document),
                validation=validation,
            )

    def test_executor_priority_fallback_wrong_prefix_and_restart(self) -> None:
        validation = self._validation()
        alphabet = validation.fit.corpus.route_alphabet_sha256s
        history = (alphabet[0],) * 3
        predicted = predict_demand_lag_blanket(
            validation,
            self.scheduler,
            self.graph_state,
            history,
            input_abi_sha256=self.two.input_abi_sha256,
        ).candidate_route_sha256
        assert predicted is not None
        verifier = _ExactVerifier(self.expected)
        executor = DemandRoutedExecutor(
            graph=self.graph,
            scheduler=self.scheduler,
            verifier=verifier,
            lag_blanket_validation=validation,
        )
        result = executor.execute(
            self.plan,
            self.value,
            candidate_route_sha256s=tuple(sorted(alphabet)),
            history_route_sha256s=history,
            episode_sha256=_digest("runtime-episode"),
        )
        self.assertEqual(result.receipt.selected_prefix_route_sha256, predicted)
        self.assertIsNotNone(result.receipt.blanket_prediction)
        self.assertIsNone(result.receipt.ppm_prediction)
        self.assertEqual(verifier.calls, 1)
        with self.assertRaisesRegex(
            DemandExecutionIntegrityError, "validation authority"
        ):
            DemandRoutedExecutionReceipt.from_bytes(result.receipt.to_bytes())
        self.assertEqual(
            DemandRoutedExecutionReceipt.from_bytes(
                result.receipt.to_bytes(),
                lag_blanket_validation=validation,
            ),
            result.receipt,
        )
        foreign = json.loads(result.receipt.to_bytes())
        nested = foreign["body"]["blanket_prediction"]
        nested["body"]["validation_sha256"] = _digest(
            "foreign-validation"
        )
        nested["body"]["generic_fit_sha256"] = _digest(
            "foreign-generic-fit"
        )
        nested["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(nested["body"])
        ).hexdigest()
        foreign["body"]["blanket_prediction_sha256"] = hashlib.sha256(
            canonical_json_bytes(nested)
        ).hexdigest()
        foreign["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(foreign["body"])
        ).hexdigest()
        with self.assertRaises(DemandExecutionIntegrityError):
            DemandRoutedExecutionReceipt.from_bytes(
                canonical_json_bytes(foreign),
                lag_blanket_validation=validation,
            )

        # A high-confidence route outside the caller's exact prefix inventory
        # cannot execute.  The full UCB mechanism receives the remaining arm.
        other = next(route for route in alphabet if route != predicted)
        second = executor.execute(
            self.plan,
            self.value,
            candidate_route_sha256s=(other,),
            history_route_sha256s=history,
        )
        self.assertEqual(second.receipt.selected_prefix_route_sha256, other)
        self.assertEqual(
            second.receipt.blanket_prediction.candidate_route_sha256,
            predicted,
        )

        stale_history_executor = DemandRoutedExecutor(
            graph=self.graph,
            scheduler=self.scheduler,
            verifier=_ExactVerifier(self.expected),
            lag_blanket_validation=validation,
            lag_blanket_min_probability=Fraction(1),
        )
        stale_history = stale_history_executor.execute(
            self.plan,
            self.value,
            candidate_route_sha256s=tuple(sorted(alphabet)),
            history_route_sha256s=(_digest("stale-unmaterialized-route"),),
        )
        self.assertIsNone(
            stale_history.receipt.blanket_prediction.candidate_route_sha256
        )
        self.assertIsNone(stale_history.receipt.ppm_prediction)
        self.assertEqual(len(stale_history.receipt.selection.event.scores), 2)

        # Exact threshold 1 forces blanket abstention.  With no history the
        # complete UCB arm inventory remains visible rather than being pruned.
        restarted = OperatorDemandScheduler(
            self.bank.store, config=self.scheduler.config
        )
        restored_validation = DemandLagBlanketValidationReceipt.from_bytes(
            validation.to_bytes()
        )
        for episode_index in range(2):
            episode = _digest(f"ppm-fallback-episode:{episode_index}")
            for route_index, route in enumerate((self.two, self.three)):
                restarted.record_outcome(
                    DemandOutcomeReceipt.create(
                        route=route,
                        graph_state=self.graph_state,
                        success=True,
                        reward=1.0,
                        outcome_evidence_sha256=_digest(
                            f"ppm-fallback-evidence:{episode_index}:{route_index}"
                        ),
                        outcome_verifier_sha256=_digest(
                            "ppm-fallback-verifier"
                        ),
                        episode_sha256=episode,
                    ),
                    self.graph_state,
                )
        abstaining = DemandRoutedExecutor(
            graph=self.graph,
            scheduler=restarted,
            verifier=_ExactVerifier(self.expected),
            lag_blanket_validation=restored_validation,
            lag_blanket_min_probability=Fraction(1),
        )
        ppm_fallback = abstaining.execute(
            self.plan,
            self.value,
            candidate_route_sha256s=tuple(sorted(alphabet)),
            history_route_sha256s=(self.two.sha256,),
        )
        self.assertIsNone(
            ppm_fallback.receipt.blanket_prediction.candidate_route_sha256
        )
        self.assertIsNotNone(ppm_fallback.receipt.ppm_prediction)
        self.assertEqual(
            ppm_fallback.receipt.selected_prefix_route_sha256,
            self.three.sha256,
        )

        fallback = abstaining.execute(
            self.plan,
            self.value,
            candidate_route_sha256s=tuple(sorted(alphabet)),
        )
        self.assertIsNone(
            fallback.receipt.blanket_prediction.candidate_route_sha256
        )
        self.assertIsNone(fallback.receipt.ppm_prediction)
        self.assertEqual(len(fallback.receipt.selection.event.scores), 2)


if __name__ == "__main__":
    unittest.main()
