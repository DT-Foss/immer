from __future__ import annotations

import hashlib
import json
import unittest

from immer.runtimes.ooe.affine_intelligence import (
    run_affine_monoid_intelligence_benchmark,
)
from immer.runtimes.ooe.identity import canonical_json_bytes


class AffineMonoidIntelligenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run_affine_monoid_intelligence_benchmark()

    def test_every_exact_operator_mechanism_and_boundary_passes(self) -> None:
        self.assertTrue(all(self.report["headline"].values()))
        body = self.report["body"]
        self.assertEqual(body["stack"]["tested_depth"], 53)
        self.assertEqual(body["stack"]["capacity"], 64)
        self.assertEqual(body["counter_dfa"]["accepted"], 4)
        self.assertEqual(body["counter_dfa"]["order_and_count_placebos_rejected"], 7)
        self.assertTrue(body["decimal_horner"]["digits_129_dead"])

    def test_fingerprint_candidate_never_crosses_exact_verifier_boundary(self) -> None:
        fingerprint = self.report["body"]["fingerprint"]
        self.assertFalse(fingerprint["unequal_length_candidate"])
        self.assertTrue(fingerprint["modular_collision_candidate"])
        self.assertFalse(fingerprint["modular_collision_exact"])
        self.assertTrue(fingerprint["exact_candidate"])
        self.assertTrue(fingerprint["exact_verified"])

    def test_markov_meta_agent_learns_algebra_context_and_regime(self) -> None:
        agent = self.report["body"]["algebra_agent"]
        self.assertEqual(agent["context_correct"], 120)
        self.assertEqual(agent["context_total"], 120)
        self.assertEqual(agent["shuffled_placebo_correct"], 0)
        self.assertEqual(agent["shuffled_placebo_total"], 90)
        self.assertEqual(agent["regime_recovered"], 30)
        self.assertEqual(agent["regime_total"], 30)
        self.assertEqual(agent["ensemble_lane_count"], 3)
        self.assertTrue(agent["ensemble_replay_exact"])

    def test_report_and_atomic_bundle_are_deterministic_and_self_sealed(self) -> None:
        second = run_affine_monoid_intelligence_benchmark()
        self.assertEqual(second, self.report)
        body = {"body": self.report["body"], "headline": self.report["headline"]}
        self.assertEqual(
            self.report["sha256"],
            hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
        )
        self.assertTrue(self.report["body"]["persistence"]["restored_exact"])
        json.dumps(self.report, allow_nan=False, sort_keys=True)


if __name__ == "__main__":
    unittest.main()
