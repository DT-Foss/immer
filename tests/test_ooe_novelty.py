from __future__ import annotations

import copy
import math
import unittest

import numpy as np

from immer.runtimes.ooe.novelty import (
    HopfieldNoveltyModel,
    hopfield_energy,
)


class HopfieldNoveltyTests(unittest.TestCase):
    @staticmethod
    def _model() -> HopfieldNoveltyModel:
        model = HopfieldNoveltyModel(beta=8.0)
        for value in ([1.0, 0.05, 0.0], [1.0, -0.05, 0.0], [0.98, 0.1, 0.0]):
            model.observe("alpha", value)
        for value in ([0.0, 1.0, 0.05], [0.05, 1.0, 0.0], [-0.05, 0.98, 0.0]):
            model.observe("beta", value)
        model.calibrate(energy_quantile=1.0, gap_quantile=0.0)
        return model

    def test_energy_matches_stable_formula(self) -> None:
        patterns = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        query = np.array([1.0, 0.0], dtype=np.float64)
        beta = 3.0
        expected = -math.log(math.exp(beta) + 1.0) + 0.5
        self.assertAlmostEqual(
            hopfield_energy(patterns, query, beta=beta),
            expected,
            places=12,
        )

    def test_in_distribution_routes_and_ood_abstains(self) -> None:
        model = self._model()
        alpha = model.decision([1.0, 0.0, 0.0])
        beta = model.decision([0.0, 1.0, 0.0])
        ambiguous = model.decision([1.0, 1.0, 0.0])
        orthogonal = model.decision([0.0, 0.0, 1.0])

        self.assertTrue(alpha.accepted)
        self.assertEqual(alpha.label, "alpha")
        self.assertTrue(beta.accepted)
        self.assertEqual(beta.label, "beta")
        self.assertFalse(ambiguous.accepted)
        self.assertIsNone(ambiguous.label)
        self.assertFalse(orthogonal.accepted)
        self.assertEqual(orthogonal.reason, "energy-threshold")

    def test_calibration_binds_pattern_count_and_observe_invalidates(self) -> None:
        model = self._model()
        before = model.calibration_receipt
        assert before is not None
        self.assertEqual(
            sum(len(hashes) for _, hashes in before.pattern_sha256s),
            6,
        )
        model.observe("alpha", [1.0, 0.02, 0.0])
        self.assertIsNone(model.calibration_receipt)
        after = model.calibrate()
        self.assertNotEqual(before.sha256, after.sha256)
        self.assertEqual(
            sum(len(hashes) for _, hashes in after.pattern_sha256s),
            7,
        )

    def test_runtime_beta_is_immutable_after_calibration(self) -> None:
        model = self._model()
        calibration = model.calibration_sha256
        with self.assertRaises(AttributeError):
            model.beta = 0.01
        self.assertEqual(model.calibration_sha256, calibration)
        self.assertTrue(model.decision([1.0, 0.0, 0.0]).accepted)

    def test_canonical_roundtrip_and_tamper_rejection(self) -> None:
        model = self._model()
        document = model.to_document()
        restored = HopfieldNoveltyModel.from_document(document)
        self.assertEqual(restored.to_document(), document)
        self.assertEqual(restored.decision([1.0, 0.0, 0.0]).label, "alpha")

        tampered = copy.deepcopy(document)
        tampered["body"]["patterns"][0]["values"][0][0] = 0.5
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            HopfieldNoveltyModel.from_document(tampered)

    def test_shuffled_site_patterns_destroy_semantic_route(self) -> None:
        model = HopfieldNoveltyModel(beta=8.0)
        for value in ([1.0, 0.0], [0.98, 0.05]):
            model.observe("wrong-beta", value)
        for value in ([0.0, 1.0], [0.05, 0.98]):
            model.observe("wrong-alpha", value)
        model.calibrate()
        self.assertEqual(model.decision([1.0, 0.0]).label, "wrong-beta")
        self.assertEqual(model.decision([0.0, 1.0]).label, "wrong-alpha")


if __name__ == "__main__":
    unittest.main()
