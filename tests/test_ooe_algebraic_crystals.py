from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import math
import random
import unittest

from immer.runtimes.ooe.algebraic_crystals import (
    ACCEPTED,
    ADDITIVE,
    AMBIGUOUS,
    CYCLIC,
    MULTIPLICATIVE,
    REJECTED,
    AlgebraicCrystalIntegrityError,
    CrystalAdmissionPolicy,
    CrystallizationFitReceipt,
    CrystallizedMap,
    CrystallizedMapStructure,
    ExactGroupAccumulator,
    GroupExecutionReceipt,
    InvariantSupport,
    crystallize_map,
    fit_algebraic_map,
    parse_structure_only,
)
from immer.runtimes.ooe.identity import canonical_json_bytes


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _supports(
    values: list[tuple[int, float]],
    *,
    prefix: str,
) -> tuple[InvariantSupport, ...]:
    return tuple(
        InvariantSupport(
            element_sha256=_sha(f"{prefix}:element:{index}"),
            group_element=element,
            learned_phi=phi,
            verifier_sha256=_sha(f"{prefix}:verifier"),
            evidence_sha256=_sha(f"{prefix}:evidence:{index}"),
            source_receipt_sha256=_sha(f"{prefix}:receipt:{index}"),
        )
        for index, (element, phi) in enumerate(values)
    )


def _linear_supports(
    *,
    count: int = 12,
    outliers: dict[int, float] | None = None,
) -> tuple[InvariantSupport, ...]:
    offsets = outliers or {}
    return _supports(
        [
            (
                element,
                2.3 * element
                - 0.7
                + 0.004 * math.sin(element)
                + offsets.get(element, 0.0),
            )
            for element in range(1, count + 1)
        ],
        prefix=f"linear:{sorted(offsets.items())}",
    )


def _log_supports(*, count: int = 12) -> tuple[InvariantSupport, ...]:
    return _supports(
        [
            (
                element,
                1.7 * math.log(element) + 0.4 + 0.004 * math.cos(element),
            )
            for element in range(1, count + 1)
        ],
        prefix="log",
    )


def _cyclic_supports(*, count: int = 12) -> tuple[InvariantSupport, ...]:
    return _supports(
        [
            (
                element,
                (element % 3) + 0.52 + 0.004 * math.sin(element),
            )
            for element in range(1, count + 1)
        ],
        prefix="cyclic-3",
    )


POLICY = CrystalAdmissionPolicy(
    acceptance_score=0.97,
    ambiguity_score=0.90,
    minimum_margin=0.015,
    minimum_support=6,
)


class AlgebraicCrystalFitTests(unittest.TestCase):
    def test_support_is_immutable_and_canonical(self) -> None:
        support = _linear_supports(count=6)[0]
        restored = InvariantSupport.from_bytes(support.to_bytes())
        self.assertEqual(restored, support)
        self.assertEqual(restored.sha256, support.sha256)
        with self.assertRaises(FrozenInstanceError):
            support.learned_phi = 99.0  # type: ignore[misc]

    def test_noisy_genuine_linear_map_is_admitted_as_additive(self) -> None:
        receipt = fit_algebraic_map(_linear_supports(), policy=POLICY)
        self.assertEqual(receipt.outcome, ACCEPTED)
        self.assertEqual(receipt.best_fit_key, ADDITIVE)
        self.assertGreater(receipt.best_score, 0.999)
        self.assertGreater(receipt.best_vs_second_margin, POLICY.minimum_margin)
        self.assertEqual(receipt.second_fit_key, MULTIPLICATIVE)

    def test_noisy_genuine_log_map_is_admitted_as_multiplicative(self) -> None:
        receipt = fit_algebraic_map(_log_supports(), policy=POLICY)
        self.assertEqual(receipt.outcome, ACCEPTED)
        self.assertEqual(receipt.best_fit_key, MULTIPLICATIVE)
        self.assertGreater(receipt.best_score, 0.999)
        additive = next(item for item in receipt.fits if item.key == ADDITIVE)
        self.assertLess(additive.score, receipt.best_score - POLICY.minimum_margin)

    def test_noisy_cycle_uses_phase_centres_and_circular_resultant(self) -> None:
        receipt = fit_algebraic_map(
            _cyclic_supports(),
            policy=POLICY,
            candidate_cyclic_orders=(3,),
        )
        self.assertEqual(receipt.outcome, ACCEPTED)
        self.assertEqual(receipt.best_fit_key, "cyclic:3")
        fit = receipt.best_fit
        assert fit is not None
        self.assertEqual(fit.family, CYCLIC)
        self.assertEqual(len(fit.phase_centers), 3)
        self.assertTrue(all(value > 0.999 for value in fit.phase_resultants))
        self.assertGreater(fit.aligned_resultant or 0.0, 0.999)
        self.assertGreater(fit.chart_concentration or 0.0, 0.999)

    def test_permuted_placebo_support_is_rejected(self) -> None:
        ys = [2.3 * value - 0.7 for value in range(1, 13)]
        random.Random(7).shuffle(ys)
        placebo = _supports(
            list(zip(range(1, 13), ys, strict=True)),
            prefix="permuted-placebo",
        )
        receipt = fit_algebraic_map(placebo, policy=POLICY)
        self.assertEqual(receipt.outcome, REJECTED)
        self.assertLess(receipt.best_score, 0.5)

    def test_one_unshaped_token_is_removed_by_measured_trimmed_fit(self) -> None:
        receipt = fit_algebraic_map(
            _linear_supports(count=10, outliers={10: 18.0}),
            policy=POLICY,
        )
        self.assertEqual(receipt.outcome, ACCEPTED)
        fit = receipt.best_fit
        assert fit is not None
        self.assertIsNotNone(fit.dropped_support_sha256)
        dropped = next(
            item
            for item in receipt.supports
            if item.sha256 == fit.dropped_support_sha256
        )
        self.assertEqual(dropped.group_element, 10)
        self.assertEqual(len(fit.used_support_sha256s), 9)

    def test_two_unshaped_tokens_cannot_silently_pass_one_point_trim(self) -> None:
        receipt = fit_algebraic_map(
            _linear_supports(count=10, outliers={9: 18.0, 10: -15.0}),
            policy=POLICY,
        )
        self.assertEqual(receipt.outcome, REJECTED)
        self.assertLess(receipt.best_score, POLICY.ambiguity_score)
        fit = receipt.best_fit
        assert fit is not None
        self.assertEqual(len(fit.used_support_sha256s), 9)

    def test_margin_threshold_produces_explicit_ambiguous_outcome(self) -> None:
        policy = CrystalAdmissionPolicy(
            acceptance_score=0.97,
            ambiguity_score=0.90,
            minimum_margin=0.10,
            minimum_support=6,
        )
        receipt = fit_algebraic_map(_linear_supports(), policy=policy)
        self.assertEqual(receipt.outcome, AMBIGUOUS)
        self.assertGreater(receipt.best_score, policy.acceptance_score)
        self.assertLess(receipt.best_vs_second_margin, policy.minimum_margin)
        with self.assertRaisesRegex(ValueError, "accepted"):
            crystallize_map(receipt, deployment_min=-20, deployment_max=100)

    def test_linear_on_log_falsifier_fails_and_log_map_wins(self) -> None:
        receipt = fit_algebraic_map(_log_supports(count=16), policy=POLICY)
        additive = next(item for item in receipt.fits if item.key == ADDITIVE)
        multiplicative = next(
            item for item in receipt.fits if item.key == MULTIPLICATIVE
        )
        self.assertLess(additive.score, POLICY.acceptance_score)
        self.assertGreater(multiplicative.score, 0.999)
        self.assertEqual(receipt.best_fit_key, MULTIPLICATIVE)

    def test_cyclic_order_is_bound_input_not_guessed_from_finite_support(self) -> None:
        without_group_contract = fit_algebraic_map(
            _cyclic_supports(),
            policy=POLICY,
        )
        with_group_contract = fit_algebraic_map(
            _cyclic_supports(),
            policy=POLICY,
            candidate_cyclic_orders=(3,),
        )
        self.assertEqual(without_group_contract.outcome, REJECTED)
        self.assertEqual(with_group_contract.best_fit_key, "cyclic:3")

    def test_number_line_wound_onto_circle_is_a_valid_cyclic_representation(
        self,
    ) -> None:
        wrapped = _supports(
            [
                (element, (element % 3) + 0.001 * math.sin(element))
                for element in range(1, 13)
            ],
            prefix="wrapped-number-line",
        )
        receipt = fit_algebraic_map(
            wrapped,
            policy=POLICY,
            candidate_cyclic_orders=(3,),
        )
        self.assertEqual(receipt.outcome, ACCEPTED)
        self.assertEqual(receipt.best_fit_key, "cyclic:3")

    def test_unwrapped_affine_line_never_becomes_false_cyclic_winner(self) -> None:
        unwrapped = _supports(
            [
                (element, element + 0.001 * math.sin(element))
                for element in range(1, 13)
            ],
            prefix="unwrapped-number-line",
        )
        receipt = fit_algebraic_map(
            unwrapped,
            policy=POLICY,
            candidate_cyclic_orders=(3,),
        )
        self.assertIn(receipt.outcome, (ACCEPTED, AMBIGUOUS))
        self.assertEqual(receipt.best_fit_key, ADDITIVE)
        cyclic = next(item for item in receipt.fits if item.key == "cyclic:3")
        additive = next(item for item in receipt.fits if item.key == ADDITIVE)
        self.assertGreaterEqual(additive.score, cyclic.score)

    def test_fit_is_deterministic_under_input_reordering(self) -> None:
        supports = _linear_supports()
        forward = fit_algebraic_map(supports, policy=POLICY)
        reverse = fit_algebraic_map(tuple(reversed(supports)), policy=POLICY)
        shuffled = list(supports)
        random.Random(991).shuffle(shuffled)
        random_fit = fit_algebraic_map(shuffled, policy=POLICY)
        self.assertEqual(forward.to_bytes(), reverse.to_bytes())
        self.assertEqual(forward.to_bytes(), random_fit.to_bytes())

    def test_receipt_roundtrip_recomputes_every_fit_and_rejects_resealed_tamper(
        self,
    ) -> None:
        receipt = fit_algebraic_map(_linear_supports(), policy=POLICY)
        restored = CrystallizationFitReceipt.from_bytes(receipt.to_bytes())
        self.assertEqual(restored.to_bytes(), receipt.to_bytes())

        document = json.loads(receipt.to_bytes())
        document["body"]["supports"][0]["learned_phi"] += 0.25
        document["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(document["body"])
        ).hexdigest()
        with self.assertRaisesRegex(
            AlgebraicCrystalIntegrityError,
            "recomputed",
        ):
            CrystallizationFitReceipt.from_bytes(canonical_json_bytes(document))


class ExactCrystallizedMapTests(unittest.TestCase):
    def test_additive_inverse_snap_supports_unseen_deployment_range(self) -> None:
        receipt = fit_algebraic_map(_linear_supports(count=8), policy=POLICY)
        crystal = crystallize_map(
            receipt,
            deployment_min=-100,
            deployment_max=100,
        )
        self.assertEqual(crystal.family, ADDITIVE)
        raw = crystal.learned_phi_of(73) + 0.08 * abs(crystal.alpha)
        self.assertAlmostEqual(crystal.inverse(raw), 73.08, places=12)
        self.assertEqual(crystal.snap(raw).group_element, 73)
        self.assertNotIn("acceptance_score", crystal.to_body())
        self.assertNotIn("policy", crystal.to_body())

    def test_nearest_log_lattice_snap_supports_unseen_value_range(self) -> None:
        receipt = fit_algebraic_map(_log_supports(count=8), policy=POLICY)
        crystal = crystallize_map(receipt, deployment_min=1, deployment_max=512)
        raw = crystal.learned_phi_of(256) + 0.001 * abs(crystal.alpha)
        snapped = crystal.snap(raw)
        self.assertEqual(snapped.group_element, 256)
        self.assertEqual(snapped.latent_coordinate, math.log(256))
        self.assertAlmostEqual(crystal.inverse(crystal.learned_phi_of(400)), 400.0)

    def test_cyclic_phase_snap_uses_nearest_learned_class_centre(self) -> None:
        receipt = fit_algebraic_map(
            _cyclic_supports(),
            policy=POLICY,
            candidate_cyclic_orders=(3,),
        )
        crystal = crystallize_map(receipt, deployment_min=0, deployment_max=2)
        for residue, center in enumerate(crystal.phase_centers):
            self.assertEqual(crystal.snap(center + 0.04).group_element, residue)
            self.assertEqual(crystal.snap(center + 3.04).group_element, residue)
            self.assertEqual(crystal.inverse(crystal.learned_phi_of(residue)), residue)

    def test_map_roundtrip_and_resealed_parameter_tamper_are_checked_against_fit(
        self,
    ) -> None:
        receipt = fit_algebraic_map(_linear_supports(), policy=POLICY)
        crystal = crystallize_map(receipt, deployment_min=-50, deployment_max=50)
        restored = CrystallizedMap.from_bytes(
            crystal.to_bytes(),
            fit_receipt=receipt,
        )
        self.assertEqual(restored, crystal)

        structure = parse_structure_only(crystal.to_bytes())
        self.assertIsInstance(structure, CrystallizedMapStructure)
        self.assertEqual(structure.sha256, crystal.sha256)
        self.assertFalse(hasattr(structure, "snap"))
        with self.assertRaisesRegex(
            AlgebraicCrystalIntegrityError,
            "requires exactly one",
        ):
            CrystallizedMap.from_bytes(crystal.to_bytes())

        resolved = CrystallizedMap.from_bytes(
            crystal.to_bytes(),
            fit_receipt_resolver=lambda digest: (
                receipt
                if digest == receipt.sha256
                else (_ for _ in ()).throw(KeyError(digest))
            ),
        )
        self.assertEqual(resolved, crystal)

        document = json.loads(crystal.to_bytes())
        document["body"]["alpha"] *= 2.0
        document["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(document["body"])
        ).hexdigest()
        with self.assertRaisesRegex(
            AlgebraicCrystalIntegrityError,
            "accepted map receipt",
        ):
            CrystallizedMap.from_bytes(
                canonical_json_bytes(document),
                fit_receipt=receipt,
            )
        with self.assertRaisesRegex(
            AlgebraicCrystalIntegrityError,
            "requires exactly one",
        ):
            CrystallizedMap.from_bytes(canonical_json_bytes(document))

    def test_directly_forged_accepted_fit_cannot_create_a_crystal(self) -> None:
        receipt = fit_algebraic_map(_linear_supports(), policy=POLICY)
        forged_fits = tuple(
            replace(item, alpha=(item.alpha or 0.0) * 3.0)
            if item.key == receipt.best_fit_key
            else item
            for item in receipt.fits
        )
        forged = replace(receipt, fits=forged_fits)
        with self.assertRaisesRegex(
            AlgebraicCrystalIntegrityError,
            "recomputed",
        ):
            crystallize_map(forged, deployment_min=-50, deployment_max=50)

    def test_snap_rejects_values_outside_configured_lattice(self) -> None:
        receipt = fit_algebraic_map(_linear_supports(), policy=POLICY)
        crystal = crystallize_map(receipt, deployment_min=1, deployment_max=20)
        with self.assertRaisesRegex(ValueError, "deployment"):
            crystal.snap(crystal.learned_phi_of(30))


class ExactGroupAccumulatorTests(unittest.TestCase):
    def test_signed_additive_length_twelve_is_exact_with_one_live_state(self) -> None:
        receipt = fit_algebraic_map(_linear_supports(count=8), policy=POLICY)
        crystal = crystallize_map(receipt, deployment_min=-20, deployment_max=20)
        values = (3, 7, 2, 5, 4, 8, 1, 6, 9, 2, 3, 7)
        signs = (1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1)
        execution = ExactGroupAccumulator(crystal).snap_and_execute(
            tuple(crystal.learned_phi_of(value) for value in values),
            signs=signs,
        )
        self.assertEqual(
            execution.result,
            sum(sign * value for sign, value in zip(signs, values, strict=True)),
        )
        self.assertEqual(execution.live_steps, 12)
        self.assertEqual(execution.state_size, 1)

    def test_multiplicative_length_twelve_is_exact(self) -> None:
        receipt = fit_algebraic_map(_log_supports(count=8), policy=POLICY)
        crystal = crystallize_map(receipt, deployment_min=1, deployment_max=20)
        values = (2, 3, 2, 5, 2, 3, 2, 2, 3, 2, 5, 2)
        execution = ExactGroupAccumulator(crystal).snap_and_execute(
            tuple(crystal.learned_phi_of(value) for value in values)
        )
        self.assertEqual(execution.result, math.prod(values))
        self.assertEqual(execution.steps[-1].state_after, math.prod(values))

    def test_cyclic_length_twelve_composes_modulo_group_order(self) -> None:
        receipt = fit_algebraic_map(
            _cyclic_supports(),
            policy=POLICY,
            candidate_cyclic_orders=(3,),
        )
        crystal = crystallize_map(receipt, deployment_min=0, deployment_max=2)
        values = tuple(index % 3 for index in range(12))
        signs = (1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1)
        execution = ExactGroupAccumulator(crystal).snap_and_execute(
            tuple(crystal.learned_phi_of(value) for value in values),
            signs=signs,
        )
        expected = sum(
            sign * value for sign, value in zip(signs, values, strict=True)
        ) % 3
        self.assertEqual(execution.result, expected)

    def test_execution_roundtrip_binds_inputs_map_steps_and_rejects_resealed_tamper(
        self,
    ) -> None:
        receipt = fit_algebraic_map(_linear_supports(), policy=POLICY)
        crystal = crystallize_map(receipt, deployment_min=-30, deployment_max=30)
        execution = ExactGroupAccumulator(crystal).snap_and_execute(
            tuple(crystal.learned_phi_of(value) for value in (2, 3, 4, 5)),
            signs=(1, -1, 1, 1),
        )
        restored = GroupExecutionReceipt.from_bytes(
            execution.to_bytes(),
            crystal=crystal,
        )
        self.assertEqual(restored, execution)

        document = json.loads(execution.to_bytes())
        document["body"]["steps"][-1]["state_after"] += 1
        document["body"]["result"] += 1
        document["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(document["body"])
        ).hexdigest()
        with self.assertRaisesRegex(
            AlgebraicCrystalIntegrityError,
            "exact live composition",
        ):
            GroupExecutionReceipt.from_bytes(
                canonical_json_bytes(document),
                crystal=crystal,
            )


if __name__ == "__main__":
    unittest.main()
