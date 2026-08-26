from __future__ import annotations

import hashlib
import unittest

from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.language_intelligence import (
    LANGUAGE_GROWTH_INTELLIGENCE_SCHEMA,
    run_language_growth_intelligence_benchmark,
)


class LanguageGrowthIntelligenceTests(unittest.TestCase):
    def test_continual_growth_retains_old_skill_learns_macro_and_resets_shift(
        self,
    ) -> None:
        report = run_language_growth_intelligence_benchmark(
            seeds=1,
            base_episodes=3_000,
            growth_episodes=3_000,
        )
        self.assertEqual(report["schema"], LANGUAGE_GROWTH_INTELLIGENCE_SCHEMA)
        self.assertEqual(
            report["sha256"],
            hashlib.sha256(canonical_json_bytes(report["body"])).hexdigest(),
        )
        aggregate = report["body"]["aggregate"]
        self.assertEqual(aggregate["base_accuracy"], 1.0)
        self.assertEqual(aggregate["retained_old_action_accuracy"], 1.0)
        self.assertEqual(aggregate["overall_accuracy_after_promotion"], 0.75)
        self.assertTrue(aggregate["all_promoted_action_initially_abstains"])
        self.assertEqual(aggregate["promoted_action_learned"], 1.0)
        self.assertGreater(aggregate["episodes_to_promoted_action"], 0.0)
        self.assertLessEqual(aggregate["episodes_to_promoted_action"], 3_000.0)
        self.assertEqual(aggregate["final_accuracy"], 1.0)
        self.assertEqual(aggregate["no_memory_accuracy"], 0.0)
        self.assertEqual(aggregate["shifted_context_global_accuracy"], 1.0)
        self.assertFalse(aggregate["all_shifted_context_evidence_retained"])
        self.assertEqual(aggregate["reward_policy_shift_accuracy"], 0.0)
        self.assertEqual(aggregate["reward_policy_shift_reset_count"], 4.0)
        self.assertTrue(aggregate["all_compiled_constant_discharge"])
        self.assertGreater(aggregate["compiled_historical_work_released"], 0.0)
        self.assertGreaterEqual(aggregate["second_generation_binding_count"], 4.0)
        self.assertTrue(aggregate["all_second_generation_constant_discharge"])
        self.assertGreater(
            aggregate["second_generation_historical_work_released"],
            aggregate["compiled_historical_work_released"],
        )


if __name__ == "__main__":
    unittest.main()
