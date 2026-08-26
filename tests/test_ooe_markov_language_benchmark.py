from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import unittest

from immer.runtimes.ooe.identity import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]


def _load_benchmark():
    path = ROOT / "scripts" / "ooe_markov_language_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "_ooe_markov_language_benchmark", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MarkovLanguageBenchmarkTests(unittest.TestCase):
    def test_sealed_runtime_benchmark_covers_language_grammar_and_word_dag(
        self,
    ) -> None:
        module = _load_benchmark()
        report = module.build_report(
            seeds=1,
            tasks=100,
            primitive_episodes=3_000,
            grammar_episodes=4_000,
            recursive_depth=6,
        )
        self.assertEqual(report["schema"], module.REPORT_SCHEMA)
        self.assertEqual(
            report["sha256"],
            hashlib.sha256(canonical_json_bytes(report["body"])).hexdigest(),
        )
        aggregate = report["body"]["aggregate"]
        for field in (
            "primitive_reward_exact",
            "novel_program_exact",
            "context_dependent_word_exact",
            "grammar_heldout_exact",
            "option_word_exact",
            "cultural_child_exact_surface",
            "self_hosted_macro_exact",
            "recursive_word_exact",
        ):
            self.assertEqual(aggregate[field], 1.0, field)
        self.assertEqual(aggregate["grammar_holistic_heldout"], 0.0)
        self.assertEqual(aggregate["grammar_shuffled_exact"], 0.0)
        self.assertLess(
            aggregate["no_message_fixed_policy_exact"],
            aggregate["novel_program_exact"],
        )
        self.assertLess(
            aggregate["no_message_abstain_exact"],
            aggregate["novel_program_exact"],
        )
        self.assertTrue(aggregate["all_in_vocab_unknown_abstains"])
        self.assertTrue(aggregate["all_recursive_constant_discharge"])
        self.assertEqual(aggregate["first_nested_definition_requests"], 2.0)
        self.assertEqual(aggregate["second_nested_definition_requests"], 0.0)
        self.assertEqual(aggregate["recursive_expanded_actions"], 127.0)
        self.assertEqual(aggregate["recursive_definition_references"], 18.0)
        self.assertGreater(
            aggregate["recursive_equivalent_source_work"],
            aggregate["recursive_live_discharge_work"],
        )
        self.assertGreater(aggregate["recursive_historical_work_released"], 0.0)
        self.assertGreater(aggregate["recursive_execution_speedup"], 0.0)


if __name__ == "__main__":
    unittest.main()
