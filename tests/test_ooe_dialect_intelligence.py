from __future__ import annotations

import hashlib
import unittest

from immer.runtimes.ooe.dialect_intelligence import (
    DIALECT_MESH_INTELLIGENCE_SCHEMA,
    run_dialect_mesh_intelligence_benchmark,
)
from immer.runtimes.ooe.identity import canonical_json_bytes


class DialectMeshIntelligenceTests(unittest.TestCase):
    def test_independent_dialects_translate_and_compile_portable_programs(self) -> None:
        report = run_dialect_mesh_intelligence_benchmark(
            seeds=1,
            dialects=3,
            contexts_per_dialect=3,
            train_episodes=3_000,
            programs_per_pair=50,
        )
        self.assertEqual(report["schema"], DIALECT_MESH_INTELLIGENCE_SCHEMA)
        self.assertEqual(
            report["sha256"],
            hashlib.sha256(canonical_json_bytes(report["body"])).hexdigest(),
        )
        aggregate = report["body"]["aggregate"]
        self.assertEqual(aggregate["unique_surface_dialects"], 3.0)
        self.assertEqual(aggregate["contexts_per_dialect"], 3.0)
        self.assertEqual(aggregate["context_specific_dialects"], 2.0)
        self.assertEqual(aggregate["minimum_distinct_context_mappings"], 3.0)
        self.assertEqual(aggregate["source_globally_stable_action_words"], 3.0)
        self.assertGreaterEqual(
            aggregate["target_dialects_without_global_action_words"], 1.0
        )
        self.assertEqual(aggregate["context_training_episodes_per_learner"], 1_000.0)
        self.assertEqual(aggregate["target_contexts_exercised"], 9.0)
        self.assertEqual(aggregate["translation_receipts"], 18.0)
        self.assertEqual(aggregate["pairwise_programs"], 900.0)
        self.assertEqual(aggregate["translated_program_accuracy"], 1.0)
        self.assertEqual(aggregate["translated_token_accuracy"], 1.0)
        self.assertEqual(aggregate["minimum_target_context_translation_accuracy"], 1.0)
        self.assertTrue(aggregate["all_target_context_translation_exact"])
        self.assertEqual(aggregate["surface_token_overlap_rate"], 1.0)
        self.assertGreater(aggregate["surface_semantic_collision_rate"], 0.0)
        self.assertGreater(aggregate["direct_surface_token_accuracy"], 0.0)
        self.assertLess(aggregate["direct_surface_token_accuracy"], 1.0)
        self.assertLess(aggregate["direct_surface_accuracy"], 1.0)
        self.assertGreater(aggregate["permutation_placebo_token_accuracy"], 0.0)
        self.assertLess(aggregate["permutation_placebo_token_accuracy"], 1.0)
        self.assertLess(aggregate["permutation_placebo_accuracy"], 1.0)
        self.assertEqual(aggregate["portable_localization_accuracy"], 1.0)
        self.assertEqual(aggregate["portable_localizations"], 9.0)
        self.assertTrue(aggregate["all_target_context_localization_exact"])
        self.assertTrue(aggregate["all_binding_change_rejected"])
        self.assertGreater(aggregate["mean_historical_work_released"], 0.0)
        components = report["body"]["seeds"][0]["context_component_sha256s"]
        self.assertEqual(tuple(len(row) for row in components), (3, 3, 3))
        self.assertEqual(len({digest for row in components for digest in row}), 9)
        self.assertEqual(
            report["body"]["controls"],
            {
                "direct_surface": (
                    "identity transfer over a fully shared token inventory"
                ),
                "independent_context_dialects": (
                    "separately learned one-context snapshots merged without averaging"
                ),
                "permutation_placebo": (
                    "non-identity action-word permutation per ordered "
                    "dialect-context pair"
                ),
            },
        )

        repeated = run_dialect_mesh_intelligence_benchmark(
            seeds=1,
            dialects=3,
            contexts_per_dialect=3,
            train_episodes=3_000,
            programs_per_pair=50,
        )
        self.assertEqual(repeated, report)


if __name__ == "__main__":
    unittest.main()
