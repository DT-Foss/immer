from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib
from itertools import product
import json
import unittest

from immer.runtimes.ooe.conditional_blanket import (
    CategoricalFeatureAtom,
    CategoricalSample,
    ConditionalBlanketAbstentionError,
    ConditionalBlanketBoundsError,
    ConditionalBlanketConfig,
    ConditionalBlanketFitReceipt,
    ConditionalBlanketIntegrityError,
    ConditionalBlanketLeakageError,
    ConditionalBlanketValidationReceipt,
    fit_conditional_blanket,
    predict_category,
    predict_probabilities,
    validate_conditional_blanket,
    verify_conditional_blanket_fit,
    verify_conditional_blanket_validation,
)
from immer.runtimes.ooe.identity import canonical_json_bytes


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sample(
    temporal_index: int,
    split: str,
    features: dict[str, str],
    target: str,
    *,
    group: str | None = None,
) -> CategoricalSample:
    return CategoricalSample(
        temporal_index=temporal_index,
        group_sha256=_hash(f"group:{split}:{group or temporal_index}"),
        source_receipt_sha256=_hash(f"source:{split}:{temporal_index}"),
        source_revision_sha256=_hash(f"revision:{split}"),
        verifier_sha256=_hash("categorical-verifier"),
        evidence_sha256=_hash(f"evidence:{split}:{temporal_index}"),
        target_category=target,
        features=tuple(
            CategoricalFeatureAtom(name, category)
            for name, category in reversed(tuple(features.items()))
        ),
    )


def _sparse_rows(start: int, split: str, repetitions: int) -> tuple[CategoricalSample, ...]:
    rows = []
    temporal = start
    for repetition in range(repetitions):
        for signal, noise_a, noise_b in product("01", repeat=3):
            rows.append(
                _sample(
                    temporal,
                    split,
                    {"signal": signal, "noise_a": noise_a, "noise_b": noise_b},
                    "yes" if signal == "1" else "no",
                    group=f"{repetition}:{signal}",
                )
            )
            temporal += 1
    return tuple(rows)


def _xor_rows(start: int, split: str, repetitions: int) -> tuple[CategoricalSample, ...]:
    rows = []
    temporal = start
    for repetition in range(repetitions):
        for left, right, noise in product("01", repeat=3):
            rows.append(
                _sample(
                    temporal,
                    split,
                    {"left": left, "right": right, "noise": noise},
                    "yes" if left != right else "no",
                    group=f"{repetition}:{left}",
                )
            )
            temporal += 1
    return tuple(rows)


def _no_signal_rows(
    start: int, split: str, repetitions: int
) -> tuple[CategoricalSample, ...]:
    rows = []
    temporal = start
    for repetition in range(repetitions):
        for first, second, target in product("01", "01", ("no", "yes")):
            rows.append(
                _sample(
                    temporal,
                    split,
                    {"first": first, "second": second},
                    target,
                    group=f"{repetition}:{first}",
                )
            )
            temporal += 1
    return tuple(rows)


def _higher_order_rows(
    start: int, split: str, repetitions: int
) -> tuple[CategoricalSample, ...]:
    rows = []
    temporal = start
    for repetition in range(repetitions):
        for first, second, third, noise in product("01", repeat=4):
            parity = (first == "1") ^ (second == "1") ^ (third == "1")
            rows.append(
                _sample(
                    temporal,
                    split,
                    {
                        "first": first,
                        "second": second,
                        "third": third,
                        "noise": noise,
                    },
                    "yes" if parity else "no",
                    group=f"{repetition}:{first}",
                )
            )
            temporal += 1
    return tuple(rows)


class ExactConditionalBlanketTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ConditionalBlanketConfig(
            target_alphabet=("yes", "no"),
            max_subset_size=3,
            max_exhaustive_subsets=100,
            max_pair_checks=16,
        )

    def test_true_sparse_blanket_and_all_calibration_arms(self) -> None:
        fit = fit_conditional_blanket(
            _sparse_rows(0, "train", 3),
            _sparse_rows(100, "calibration", 2),
            config=self.config,
        )

        self.assertEqual(fit.status, "sparse")
        self.assertEqual(fit.selected_feature_names, ("signal",))
        self.assertEqual(fit.compression, Fraction(3))
        self.assertEqual(fit.feature_reduction, Fraction(2, 3))
        self.assertEqual(fit.candidate_count, 8)
        self.assertGreater(fit.qualifying_candidate_count, 0)
        self.assertTrue(fit.closure_verified)
        self.assertFalse(fit.capacity_exhausted)
        self.assertEqual(fit.full_arm.feature_names, fit.feature_schema)
        self.assertEqual(fit.marginal_arm.feature_names, ())
        self.assertEqual(len(fit.random_arm.feature_names), 1)
        self.assertNotEqual(fit.random_arm.feature_names, fit.selected_feature_names)
        self.assertLess(
            fit.selected_arm.calibration_metrics.brier,
            fit.random_arm.calibration_metrics.brier,
        )
        self.assertLess(
            fit.selected_arm.calibration_metrics.brier,
            fit.marginal_arm.calibration_metrics.brier,
        )
        self.assertEqual(fit.selected_arm.calibration_metrics.accuracy, 1)
        self.assertTrue(verify_conditional_blanket_fit(fit))
        self.assertEqual(ConditionalBlanketFitReceipt.from_bytes(fit.to_bytes()), fit)

    def test_xor_requires_bounded_pair_synergy_with_full_search(self) -> None:
        config = replace(self.config, max_subset_size=3)
        fit = fit_conditional_blanket(
            _xor_rows(0, "xor-train", 4),
            _xor_rows(100, "xor-calibration", 2),
            config=config,
        )

        self.assertEqual(fit.selected_feature_names, ("left", "right"))
        self.assertEqual(fit.status, "sparse")
        self.assertEqual(fit.closure_candidate_count, 0)
        self.assertTrue(fit.closure_verified)
        self.assertFalse(fit.capacity_exhausted)
        self.assertEqual(
            tuple(
                (check.excluded_feature_names, check.residual.fraction)
                for check in fit.selected_dependence_checks
            ),
            ((('noise',), Fraction()),),
        )

    def test_order_and_feature_input_permutations_are_identity_invariant(self) -> None:
        train = _sparse_rows(0, "order-train", 2)
        calibration = _sparse_rows(100, "order-calibration", 2)
        first = fit_conditional_blanket(train, calibration, config=self.config)
        second = fit_conditional_blanket(
            tuple(reversed(train)),
            tuple(reversed(calibration)),
            config=ConditionalBlanketConfig(
                target_alphabet=("no", "yes"),
                max_subset_size=3,
                max_exhaustive_subsets=100,
                max_pair_checks=16,
            ),
        )

        self.assertEqual(first.to_bytes(), second.to_bytes())
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(
            first.candidate_subset_inventory_sha256,
            second.candidate_subset_inventory_sha256,
        )

    def test_chronology_source_and_group_leakage_are_rejected(self) -> None:
        train = _sparse_rows(20, "leak-train", 1)
        calibration = _sparse_rows(10, "leak-calibration", 1)
        with self.assertRaises(ConditionalBlanketLeakageError):
            fit_conditional_blanket(train, calibration, config=self.config)

        calibration = list(_sparse_rows(100, "isolated-calibration", 1))
        calibration[0] = replace(
            calibration[0], source_receipt_sha256=train[0].source_receipt_sha256
        )
        with self.assertRaises(ConditionalBlanketLeakageError):
            fit_conditional_blanket(train, calibration, config=self.config)

        calibration = list(_sparse_rows(100, "group-calibration", 1))
        calibration[0] = replace(calibration[0], group_sha256=train[0].group_sha256)
        with self.assertRaises(ConditionalBlanketLeakageError):
            fit_conditional_blanket(train, calibration, config=self.config)

    def test_group_macro_metrics_do_not_let_large_groups_outvote_small_groups(self) -> None:
        train = tuple(
            _sample(
                index,
                "macro-train",
                {"constant": "x"},
                "yes",
                group="teacher",
            )
            for index in range(10)
        )
        calibration = tuple(
            [
                _sample(
                    100 + index,
                    "macro-calibration",
                    {"constant": "x"},
                    "yes",
                    group="large",
                )
                for index in range(9)
            ]
            + [
                _sample(
                    200,
                    "macro-calibration",
                    {"constant": "x"},
                    "no",
                    group="small",
                )
            ]
        )
        fit = fit_conditional_blanket(
            train,
            calibration,
            config=replace(self.config, max_subset_size=1),
        )

        self.assertEqual(fit.full_arm.calibration_metrics.accuracy, Fraction(1, 2))

    def test_later_holdout_all_arms_unseen_backoff_and_tamper(self) -> None:
        train = _sparse_rows(0, "holdout-train", 2)
        calibration = _sparse_rows(100, "holdout-calibration", 1)
        fit = fit_conditional_blanket(train, calibration, config=self.config)
        holdout = tuple(
            _sample(
                200 + index,
                "holdout",
                {"signal": signal, "noise_a": "unseen", "noise_b": "unseen"},
                "yes" if signal == "1" else "no",
            )
            for index, signal in enumerate(("0", "1", "0", "1"))
        )
        validation = validate_conditional_blanket(fit, holdout)

        self.assertEqual(validation.selected_metrics.context_coverage.fraction, 1)
        self.assertEqual(validation.full_metrics.context_coverage.fraction, 0)
        self.assertEqual(
            predict_probabilities(fit, holdout[0], arm="full"),
            predict_probabilities(fit, holdout[0], arm="marginal"),
        )
        self.assertEqual(
            predict_probabilities(fit, holdout[0]),
            predict_probabilities(fit, holdout[0].feature_map),
        )
        self.assertEqual(
            predict_probabilities(fit, holdout[0]),
            predict_probabilities(fit, holdout[0].features),
        )
        self.assertEqual(validation.compression, Fraction(3))
        self.assertTrue(verify_conditional_blanket_validation(validation))
        self.assertEqual(
            ConditionalBlanketValidationReceipt.from_bytes(validation.to_bytes()),
            validation,
        )
        self.assertNotIn(holdout[0].source_receipt_sha256, fit.to_bytes().decode())
        self.assertIn(
            holdout[0].source_receipt_sha256,
            validation.to_bytes().decode(),
        )

        with self.assertRaises(ConditionalBlanketLeakageError):
            validate_conditional_blanket(fit, (fit.train_samples[0],))
        early = replace(holdout[0], temporal_index=1)
        with self.assertRaises(ConditionalBlanketLeakageError):
            validate_conditional_blanket(fit, (early,))

        document = validation.to_document()
        document["body"]["selected_metrics"]["sample_count"] += 1
        document["sha256"] = hashlib.sha256(
            canonical_json_bytes(document["body"])
        ).hexdigest()
        with self.assertRaises(ConditionalBlanketIntegrityError):
            ConditionalBlanketValidationReceipt.from_bytes(
                canonical_json_bytes(document)
            )

    def test_no_signal_learns_empty_blanket(self) -> None:
        config = replace(self.config, max_subset_size=2)
        fit = fit_conditional_blanket(
            _no_signal_rows(0, "none-train", 3),
            _no_signal_rows(100, "none-calibration", 2),
            config=config,
        )

        self.assertEqual(fit.status, "sparse")
        self.assertEqual(fit.selected_feature_names, ())
        self.assertEqual(fit.selected_arm.calibration_metrics.brier, Fraction(1, 2))
        self.assertEqual(
            fit.selected_arm.calibration_metrics,
            fit.marginal_arm.calibration_metrics,
        )
        self.assertTrue(
            all(
                check.residual.fraction == 0
                for check in fit.selected_dependence_checks
            )
        )

    def test_capacity_boundary_abstains_instead_of_hiding_full_fallback(self) -> None:
        config = replace(self.config, max_subset_size=1)
        fit = fit_conditional_blanket(
            _xor_rows(0, "capacity-train", 4),
            _xor_rows(100, "capacity-calibration", 2),
            config=config,
        )

        self.assertEqual(fit.status, "capacity_exhausted")
        self.assertTrue(fit.capacity_exhausted)
        self.assertTrue(fit.abstained)
        self.assertFalse(fit.closure_verified)
        self.assertEqual(fit.selected_feature_names, fit.feature_schema)
        self.assertEqual(ConditionalBlanketFitReceipt.from_bytes(fit.to_bytes()), fit)
        query = _xor_rows(200, "capacity-query", 1)[0]
        with self.assertRaises(ConditionalBlanketAbstentionError):
            predict_probabilities(fit, query)
        with self.assertRaises(ConditionalBlanketAbstentionError):
            predict_probabilities(fit, query, arm="selected")
        with self.assertRaises(ConditionalBlanketAbstentionError):
            predict_category(fit, query)
        for diagnostic_arm in ("full", "random", "marginal"):
            self.assertEqual(
                sum(predict_probabilities(fit, query, arm=diagnostic_arm)),
                Fraction(1),
            )

    def test_higher_order_synergy_cannot_be_closed_by_pairs_or_one_step(self) -> None:
        config = replace(
            self.config,
            max_subset_size=2,
            max_brier_regret=Fraction(2),
            max_accuracy_regret=Fraction(1),
            max_coverage_regret=Fraction(1),
        )
        fit = fit_conditional_blanket(
            _higher_order_rows(0, "higher-train", 3),
            _higher_order_rows(200, "higher-calibration", 2),
            config=config,
        )

        # Every single and pair is empirically independent of the triple
        # parity target, so the empty subset passes those local checks.  It is
        # still unauthorized because the complete four-feature power set was
        # not enumerated.
        self.assertEqual(fit.selected_feature_names, ())
        self.assertTrue(
            all(check.residual.fraction == 0 for check in fit.selected_dependence_checks)
        )
        self.assertEqual(fit.status, "capacity_exhausted")
        self.assertTrue(fit.abstained)
        self.assertFalse(fit.closure_verified)
        with self.assertRaises(ConditionalBlanketAbstentionError):
            predict_category(fit, _higher_order_rows(400, "higher-query", 1)[0])

    def test_hash_seals_exact_types_and_search_bounds(self) -> None:
        atom = CategoricalFeatureAtom("feature", "value")
        self.assertEqual(CategoricalFeatureAtom.from_bytes(atom.to_bytes()), atom)
        sample = _sample(0, "roundtrip", {"feature": "value"}, "yes")
        self.assertEqual(CategoricalSample.from_bytes(sample.to_bytes()), sample)
        self.assertEqual(
            ConditionalBlanketConfig.from_bytes(self.config.to_bytes()), self.config
        )
        with self.assertRaises(ValueError):
            ConditionalBlanketConfig(
                target_alphabet=("no", "yes"), laplace_alpha=0.5  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            CategoricalFeatureAtom("feature", float("nan"))  # type: ignore[arg-type]

        train = _sparse_rows(0, "bounds-train", 1)
        calibration = _sparse_rows(100, "bounds-calibration", 1)
        with self.assertRaises(ConditionalBlanketBoundsError):
            fit_conditional_blanket(
                train,
                calibration,
                config=replace(self.config, max_exhaustive_subsets=1),
            )

        fit = fit_conditional_blanket(train, calibration, config=self.config)
        document = json.loads(fit.to_bytes())
        document["body"]["candidate_subset_inventory_sha256"] = _hash("forged")
        document["sha256"] = hashlib.sha256(
            canonical_json_bytes(document["body"])
        ).hexdigest()
        with self.assertRaises(ConditionalBlanketIntegrityError):
            ConditionalBlanketFitReceipt.from_bytes(canonical_json_bytes(document))


if __name__ == "__main__":
    unittest.main()
