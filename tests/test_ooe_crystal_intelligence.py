from __future__ import annotations

import hashlib
import json
import warnings
import unittest

from immer.runtimes.ooe.crystal_intelligence import (
    run_crystal_intelligence_benchmark,
)
from immer.runtimes.ooe.identity import canonical_json_bytes


class CrystalIntelligenceBenchmarkTests(unittest.TestCase):
    def test_unknown_values_use_markov_planned_charged_operator(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            report = run_crystal_intelligence_benchmark()
        headline = report["headline"]
        compute = report["body"]["compute"]
        self.assertTrue(headline["compute_exact"])
        self.assertEqual(headline["primitive_route_length"], 4)
        self.assertEqual(headline["live_operator_count"], 1)
        self.assertGreater(headline["historical_work_released"], 0)
        self.assertTrue(headline["residual_exact"])
        self.assertEqual(headline["residual_prefix_length"], 2)
        self.assertEqual(headline["residual_suffix_length"], 2)
        self.assertGreater(compute["residual_historical_work_released"], 0)
        self.assertEqual(compute["residual_live_operator_count"], 3)
        self.assertEqual(compute["multi_step_teacher_labels"], 0)
        self.assertEqual(compute["unseen_batch"], 128)
        self.assertLess(compute["max_absolute_delta"], 1e-12)
        self.assertTrue(compute["placebo_abstained"])
        self.assertEqual(len(compute["charge_basis_sha256"]), 64)

    def test_group_crystals_classify_and_compose_exactly_at_length_twelve(
        self,
    ) -> None:
        report = run_crystal_intelligence_benchmark(seed=77)
        headline = report["headline"]
        algebra = report["body"]["algebra"]
        self.assertEqual(headline["algebraic_families_admitted"], 3)
        self.assertTrue(headline["algebraic_length_12_exact"])
        self.assertTrue(headline["placebos_rejected"])
        self.assertEqual(algebra["placebo"]["outcome"], "rejected")
        for family in ("additive", "multiplicative", "cyclic"):
            self.assertGreater(algebra[family]["fit_score"], 0.99)
            self.assertEqual(algebra[family]["length"], 12)
            self.assertEqual(
                algebra[family]["result"],
                algebra[family]["expected"],
            )

    def test_operator_search_learns_diverse_exact_and_topological_structure(
        self,
    ) -> None:
        report = run_crystal_intelligence_benchmark(seed=20260826)
        headline = report["headline"]
        search = report["body"]["operator_search"]
        self.assertTrue(headline["bvn_basis_exact"])
        self.assertTrue(headline["bvn_basis_within_bound"])
        self.assertTrue(headline["fiedler_prefers_missing_bridge"])
        self.assertTrue(headline["mutation_preference_learned"])
        self.assertGreaterEqual(headline["map_elites_occupied_cells"], 4)
        self.assertGreaterEqual(search["mutation_late_preferred_fraction"], 0.75)
        self.assertGreater(search["fiedler_best_missing_priority"], 0.0)
        self.assertLessEqual(search["basis_max_absolute_delta"], 1e-12)

    def test_report_is_deterministic_json_and_self_sealed(self) -> None:
        first = run_crystal_intelligence_benchmark(seed=123)
        second = run_crystal_intelligence_benchmark(seed=123)
        self.assertEqual(first, second)
        body = {"body": first["body"], "headline": first["headline"]}
        self.assertEqual(
            first["sha256"],
            hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
        )
        json.dumps(first, allow_nan=False, sort_keys=True)


if __name__ == "__main__":
    unittest.main()
