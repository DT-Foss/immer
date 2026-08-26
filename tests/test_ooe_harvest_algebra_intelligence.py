from __future__ import annotations

import unittest

from immer.runtimes.ooe.harvest_algebra_intelligence import (
    HARVEST_ALGEBRA_INTELLIGENCE_SCHEMA,
    run_harvest_algebra_intelligence_benchmark,
)


class HarvestAlgebraIntelligenceTests(unittest.TestCase):
    def test_contextual_markov_router_learns_all_three_harvested_families(self) -> None:
        report = run_harvest_algebra_intelligence_benchmark()
        body = report["body"]
        self.assertEqual(report["schema"], HARVEST_ALGEBRA_INTELLIGENCE_SCHEMA)
        self.assertEqual(body["candidate_count"], 3)
        self.assertEqual(body["program_runtime"], "compute-crystal-vm")
        self.assertEqual(body["families"], ["double", "swap", "mix"])
        self.assertEqual(len(body["bridge_receipt_sha256s"]), 3)
        self.assertEqual(body["contextual"]["successes"], 174)
        self.assertEqual(body["contextual"]["late_successes"], 60)
        self.assertEqual(body["contextual"]["final_successes"], 30)
        self.assertEqual(body["placebo"]["successes"], 49)
        self.assertEqual(body["placebo"]["late_successes"], 13)
        self.assertEqual(body["placebo"]["final_successes"], 5)
        self.assertGreater(
            body["contextual"]["late_successes"],
            4 * body["placebo"]["late_successes"],
        )

    def test_episode_contract_rejects_non_triplet_trials(self) -> None:
        with self.assertRaises(ValueError):
            run_harvest_algebra_intelligence_benchmark(episodes=61)


if __name__ == "__main__":
    unittest.main()
