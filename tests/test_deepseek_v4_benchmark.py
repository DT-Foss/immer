from __future__ import annotations

import unittest

import numpy as np

from immer.runtimes.deepseek_v4.benchmark import (
    BenchmarkContractError,
    BenchmarkProvenance,
    BenchmarkRun,
    BenchmarkTask,
    CacheProfile,
    CacheState,
    DatasetProvenance,
    ItemMetric,
    OutcomeStatus,
    PerformanceMetric,
    calculate_stop_go,
    canonical_digest,
    dataset_digest,
    evaluate_decoder_parity,
    evaluate_gsm8k,
    evaluate_mmlu,
    extract_gsm8k_answer,
    maximum_future_attention_mass,
    summarize_decoder_parity,
    summarize_outcomes,
    summarize_paired_ablation,
)


def _dataset(size: int = 1) -> DatasetProvenance:
    rows = [{"id": f"q{index}", "answer": index % 4} for index in range(size)]
    return DatasetProvenance.from_records(
        "fixture/mixed", "validation", rows, revision="fixture-v1"
    )


def _provenance(*, pinned: bool = True, size: int = 1) -> BenchmarkProvenance:
    return BenchmarkProvenance(
        model_id="deepseek-ai/DeepSeek-V4-Flash-0731",
        model_revision="a" * 40 if pinned else "main",
        inventory_sha256="b" * 64,
        config_sha256="c" * 64,
        tokenizer_sha256="d" * 64,
        dataset=_dataset(size),
        harness_revision="test-harness-v1",
        command=("python", "benchmark.py", "--seed", "7"),
    )


def _item(
    item_id: str,
    seed: int,
    status: OutcomeStatus,
    *,
    task: BenchmarkTask = BenchmarkTask.MMLU,
) -> ItemMetric:
    return ItemMetric(
        item_id,
        task,
        seed,
        status,
        expected=0,
        predicted=0 if status is OutcomeStatus.CORRECT else 1,
        error="decoder crashed" if status is OutcomeStatus.ERROR else None,
    )


class DigestAndProvenanceTests(unittest.TestCase):
    def test_canonical_digest_is_key_order_independent(self) -> None:
        self.assertEqual(
            canonical_digest({"b": 2, "a": 1}), canonical_digest({"a": 1, "b": 2})
        )

    def test_dataset_digest_is_record_order_sensitive(self) -> None:
        first = [{"id": "a"}, {"id": "b"}]
        self.assertNotEqual(
            dataset_digest(first), dataset_digest(list(reversed(first)))
        )
        provenance = DatasetProvenance.from_records("x", "test", first)
        self.assertEqual(provenance.num_items, 2)
        self.assertEqual(provenance.sha256, dataset_digest(first))

    def test_digest_validation_and_revision_pin(self) -> None:
        self.assertTrue(_provenance().revision_is_pinned)
        self.assertFalse(_provenance(pinned=False).revision_is_pinned)
        with self.assertRaisesRegex(BenchmarkContractError, "lowercase SHA-256"):
            DatasetProvenance("x", "test", "BAD", 1)


class OutcomeEvaluatorTests(unittest.TestCase):
    def test_mmlu_accepts_letter_or_index_and_counts_runtime_error(self) -> None:
        correct = evaluate_mmlu("one", "C", 2, seed=4)
        invalid = evaluate_mmlu("two", 0, "I think maybe Z", seed=4)
        failed = evaluate_mmlu("three", 0, None, seed=4, error="timeout")
        summary = summarize_outcomes((correct, invalid, failed))
        self.assertEqual(
            (summary.total, summary.correct, summary.incorrect, summary.errors),
            (3, 1, 1, 1),
        )
        self.assertAlmostEqual(summary.accuracy, 1 / 3)
        self.assertAlmostEqual(summary.coverage, 2 / 3)
        self.assertAlmostEqual(summary.wrong_rate, 1 / 3)

    def test_abstention_is_separate_from_wrong_and_error(self) -> None:
        rows = (
            evaluate_mmlu("correct", "A", "A", seed=0),
            evaluate_mmlu("wrong", "A", "B", seed=0),
            evaluate_mmlu("skip", "A", None, seed=0, abstained=True),
            evaluate_mmlu("error", "A", None, seed=0, error="timeout"),
        )
        summary = summarize_outcomes(rows)
        self.assertEqual(
            (
                summary.total,
                summary.correct,
                summary.incorrect,
                summary.abstained,
                summary.errors,
            ),
            (4, 1, 1, 1, 1),
        )
        self.assertEqual(summary.accuracy, 0.25)
        self.assertEqual(summary.accuracy_attempted, 0.5)
        self.assertEqual(summary.coverage, 0.5)
        with self.assertRaisesRegex(BenchmarkContractError, "both error and abstained"):
            evaluate_gsm8k(
                "invalid", "#### 1", None, seed=0, error="failed", abstained=True
            )

    def test_mmlu_extracts_explicit_answer_without_guessing_free_text(self) -> None:
        self.assertTrue(evaluate_mmlu("x", "B", "The answer is (B).", seed=0).correct)
        self.assertEqual(
            evaluate_mmlu("x", "B", "A and B are discussed", seed=0).status,
            OutcomeStatus.INCORRECT,
        )

    def test_gsm8k_uses_final_numeric_answer_and_decimal_canonicalization(self) -> None:
        self.assertEqual(extract_gsm8k_answer("work 10 then #### 1,200.00"), "1200")
        self.assertEqual(extract_gsm8k_answer("therefore -0.000"), "0")
        self.assertEqual(extract_gsm8k_answer("therefore 42."), "42")
        self.assertEqual(
            extract_gsm8k_answer("30 cars. **Josh makes $120 in 2 weeks.**"),
            "120",
        )
        self.assertEqual(
            extract_gsm8k_answer("work. **Answer: 10 + 5 = 15.**"),
            "15",
        )
        self.assertEqual(
            extract_gsm8k_answer("work. **$30 * 4 = $120 in 2 weeks.**"),
            "120",
        )
        self.assertEqual(
            extract_gsm8k_answer(r"work: 6 * 7. Therefore \boxed{42}"),
            "42",
        )
        result = evaluate_gsm8k("g1", "#### 42", "6 * 7 = 42", seed=9)
        self.assertTrue(result.correct)
        self.assertEqual(result.predicted, "42")

    def test_gsm8k_rejects_numbers_embedded_in_words_units_and_versions(self) -> None:
        for response in ("v2.3.4", "42nd", "#### 7cm"):
            with self.subTest(response=response):
                self.assertIsNone(extract_gsm8k_answer(response))

    def test_unparseable_gsm8k_response_is_wrong_but_runtime_failure_is_error(
        self,
    ) -> None:
        wrong = evaluate_gsm8k("g1", "#### 5", "no numeric answer", seed=0)
        error = evaluate_gsm8k("g2", "#### 5", None, seed=0, error="OOM")
        self.assertEqual(wrong.status, OutcomeStatus.INCORRECT)
        self.assertEqual(error.status, OutcomeStatus.ERROR)


class PerformanceTests(unittest.TestCase):
    def test_snapshot_delta_and_cold_warm_hot_profile(self) -> None:
        before = {
            "network_or_source_body_bytes": 100,
            "cache_bytes_reused": 20,
            "budget": {"requests": 2},
        }
        after = {
            "network_or_source_body_bytes": 350,
            "cache_bytes_reused": 80,
            "budget": {"requests": 5},
        }
        cold = PerformanceMetric.from_snapshots(
            "q", 0, "cold", 10.0, before, after, output_tokens=2
        )
        warm = PerformanceMetric("q", 0, CacheState.WARM, 5, 10, 90, 1, 2)
        hot = PerformanceMetric("q", 0, CacheState.HOT, 2, 0, 100, 1, 2)
        profile = CacheProfile.from_metrics((cold, warm, hot))
        self.assertEqual(
            (cold.source_bytes, cold.cache_bytes, cold.requests), (250, 60, 3)
        )
        self.assertTrue(profile.complete)
        self.assertEqual(profile.cold.total_source_bytes, 250)
        self.assertAlmostEqual(profile.hot.tokens_per_second, 1000.0)

    def test_missing_snapshot_counters_fail_instead_of_becoming_zero(self) -> None:
        with self.assertRaisesRegex(BenchmarkContractError, "missing source bytes"):
            PerformanceMetric.from_snapshots("q", 0, "uncontrolled", 1.0, {}, {})

    def test_decreasing_snapshot_counter_is_rejected(self) -> None:
        with self.assertRaisesRegex(BenchmarkContractError, "decreased"):
            PerformanceMetric.from_snapshots(
                "q",
                0,
                "cold",
                1,
                {"network_or_source_body_bytes": 10},
                {"network_or_source_body_bytes": 9},
            )

    def test_performance_errors_remain_in_summary_denominator(self) -> None:
        rows = (
            PerformanceMetric("a", 0, "cold", 10, 100, 0, 1),
            PerformanceMetric("b", 0, "cold", 30, 50, 0, 1, error="timeout"),
        )
        summary = CacheProfile.from_metrics(rows).cold
        self.assertEqual((summary.total, summary.errors), (2, 1))
        self.assertEqual(summary.mean_latency_ms, 20)
        self.assertAlmostEqual(summary.p99_latency_ms, 29.8)
        self.assertEqual(summary.total_source_bytes, 150)

    def test_uncontrolled_cache_observation_never_claims_a_profile_state(self) -> None:
        profile = CacheProfile.from_metrics(
            (PerformanceMetric("q", 0, "uncontrolled", 10, 100, 0, 1),)
        )
        self.assertFalse(profile.complete)
        self.assertEqual(profile.summaries, (None, None, None))

    def test_cache_profile_requires_the_same_workload_in_every_state(self) -> None:
        profile = CacheProfile.from_metrics(
            (
                PerformanceMetric("cold-item", 0, "cold", 10, 100, 0, 1),
                PerformanceMetric("warm-item", 0, "warm", 5, 0, 100, 1),
                PerformanceMetric("hot-item", 0, "hot", 2, 0, 100, 1),
            )
        )
        self.assertFalse(profile.workload_aligned)
        self.assertFalse(profile.complete)


class DecoderParityTests(unittest.TestCase):
    def test_exact_decoder_parity_includes_alpha0_and_causal_mass(self) -> None:
        logits = np.asarray([[1.0, 3.0, 2.0]], dtype=np.float32)
        alpha = np.asarray([1.0, 2.0], dtype=np.float32)
        attention = np.asarray([[1.0, 0.0], [0.4, 0.6]], dtype=np.float32)
        metric = evaluate_decoder_parity(
            "prompt",
            logits,
            logits.copy(),
            seed=0,
            dtype="float32",
            reference_sequence=[1, 2],
            candidate_sequence=[1, 2],
            alpha0_reference=alpha,
            alpha0_candidate=alpha.copy(),
            attention=attention,
        )
        summary = summarize_decoder_parity((metric,))
        self.assertTrue(metric.passed)
        self.assertEqual(summary.passed, 1)
        self.assertEqual(summary.alpha0_matches, 1)
        self.assertEqual(summary.max_future_attention_mass, 0.0)

    def test_tolerance_and_fallback_fail_even_when_top1_matches(self) -> None:
        metric = evaluate_decoder_parity(
            "prompt",
            [0.0, 1.0],
            [0.0, 1.1],
            seed=0,
            dtype="float32",
            fallbacks=["dense_moe"],
        )
        self.assertTrue(metric.top1_match)
        self.assertFalse(metric.within_logit_tolerance)
        self.assertFalse(metric.passed)

    def test_future_attention_detects_leak(self) -> None:
        attention = np.asarray([[0.9, 0.1], [0.5, 0.5]])
        self.assertEqual(maximum_future_attention_mass(attention), 0.1)
        decode_attention = np.asarray([[0.2, 0.3, 0.5]])
        self.assertEqual(
            maximum_future_attention_mass(decode_attention, query_start=2),
            0.0,
        )

    def test_torch_bfloat16_alpha0_can_be_checked_bitwise(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch optional extra is unavailable")
        logits = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
        alpha = torch.tensor([0.5, -0.0], dtype=torch.bfloat16)
        metric = evaluate_decoder_parity(
            "torch",
            logits,
            logits.clone(),
            seed=0,
            dtype="bf16",
            reference_sequence=[1],
            candidate_sequence=[1],
            alpha0_reference=alpha,
            alpha0_candidate=alpha.clone(),
        )
        self.assertTrue(metric.alpha0_bit_identical)


class PairedAndStopGoTests(unittest.TestCase):
    def test_deterministic_seed_repeats_collapse_to_one_independent_item(self) -> None:
        candidate = [_item("q", seed, OutcomeStatus.CORRECT) for seed in range(5)]
        placebo = [_item("q", seed, OutcomeStatus.INCORRECT) for seed in range(5)]
        summary = summarize_paired_ablation(candidate, placebo)
        self.assertEqual(summary.seeds, (0, 1, 2, 3, 4))
        self.assertEqual(summary.observations, 5)
        self.assertEqual(summary.pairs, 1)
        self.assertEqual((summary.wins, summary.ties, summary.losses), (1, 0, 0))
        self.assertEqual(summary.exact_one_sided_p_value, 0.5)
        self.assertFalse(summary.exceeds_sigma(2.0))
        self.assertFalse(summary.passes_exact_paired_gate(0.02275))
        with self.assertRaisesRegex(BenchmarkContractError, "exactly the same"):
            summarize_paired_ablation(candidate, placebo[:-1])

    def test_exact_paired_gate_uses_unique_items(self) -> None:
        candidate = [_item(f"q{index}", 0, OutcomeStatus.CORRECT) for index in range(6)]
        placebo = [_item(f"q{index}", 0, OutcomeStatus.INCORRECT) for index in range(6)]
        summary = summarize_paired_ablation(candidate, placebo)
        self.assertEqual(summary.pairs, 6)
        self.assertEqual(summary.discordant_pairs, 6)
        self.assertEqual(summary.exact_one_sided_p_value, 1 / 64)
        self.assertTrue(summary.passes_exact_paired_gate(0.02275))

    def test_uncontrolled_byte_evidence_is_retained_but_cannot_pass_gate(self) -> None:
        candidate = [_item("q", 0, OutcomeStatus.CORRECT)]
        baseline = [_item("q", 0, OutcomeStatus.CORRECT)]
        candidate_perf = [PerformanceMetric("q", 0, "uncontrolled", 1, 10, 0, 1)]
        baseline_perf = [PerformanceMetric("q", 0, "uncontrolled", 1, 20, 0, 1)]
        summary = summarize_paired_ablation(
            candidate,
            baseline,
            candidate_performance=candidate_perf,
            baseline_performance=baseline_perf,
            cache_state=CacheState.UNCONTROLLED,
        )
        self.assertEqual(summary.candidate_mean_source_bytes, 10)
        self.assertEqual(summary.baseline_mean_source_bytes, 20)
        self.assertEqual(summary.source_byte_cache_state, CacheState.UNCONTROLLED)
        self.assertFalse(summary.source_byte_comparison_valid)

    def test_stop_go_passes_only_with_all_preregistered_evidence(self) -> None:
        logits = np.asarray([0.0, 1.0], dtype=np.float32)
        alpha = np.asarray([1.0], dtype=np.float32)
        parity = summarize_decoder_parity(
            (
                evaluate_decoder_parity(
                    "p",
                    logits,
                    logits.copy(),
                    seed=0,
                    dtype="f32",
                    reference_sequence=[1],
                    candidate_sequence=[1],
                    alpha0_reference=alpha,
                    alpha0_candidate=alpha.copy(),
                    attention=np.asarray([[1.0]]),
                ),
            )
        )
        selection_real = [
            _item(f"q{index}", seed, OutcomeStatus.CORRECT)
            for index in range(6)
            for seed in range(5)
        ]
        selection_placebo = [
            _item(f"q{index}", seed, OutcomeStatus.INCORRECT)
            for index in range(6)
            for seed in range(5)
        ]
        selection = summarize_paired_ablation(selection_real, selection_placebo)
        real = [_item(f"q{index}", 0, OutcomeStatus.CORRECT) for index in range(6)]
        placebo = [_item(f"q{index}", 0, OutcomeStatus.INCORRECT) for index in range(6)]
        baseline = [_item(f"q{index}", 0, OutcomeStatus.CORRECT) for index in range(6)]
        real_perf = [
            PerformanceMetric(f"q{index}", 0, "cold", 1, 90, 0, 1) for index in range(6)
        ]
        base_perf = [
            PerformanceMetric(f"q{index}", 0, "cold", 1, 100, 0, 1)
            for index in range(6)
        ]
        graft = summarize_paired_ablation(
            real,
            baseline,
            candidate_performance=real_perf,
            baseline_performance=base_perf,
        )
        shuffled = summarize_paired_ablation(real, placebo)
        profile = CacheProfile.from_metrics(
            PerformanceMetric("q", 0, state, 1, 1, 1, 1)
            for state in (CacheState.COLD, CacheState.WARM, CacheState.HOT)
        )
        outcomes = summarize_outcomes(real)
        decision = calculate_stop_go(
            provenance=_provenance(size=1),
            decoder=parity,
            selection=selection,
            graft=graft,
            shuffled=shuffled,
            candidate=outcomes,
            reference=outcomes,
            cache_profile=profile,
        )
        self.assertTrue(decision.go, decision.to_dict())
        self.assertEqual(decision.verdict, "go")

    def test_stop_go_fails_mutable_revision_and_missing_cache_temperatures(
        self,
    ) -> None:
        decision = calculate_stop_go(
            provenance=_provenance(pinned=False),
            decoder=None,
            selection=None,
            graft=None,
            shuffled=None,
            candidate=None,
            reference=None,
            cache_profile=CacheProfile.from_metrics(
                (PerformanceMetric("q", 0, "cold", 1, 0, 0, 0),)
            ),
        )
        self.assertFalse(decision.go)
        failed = {gate.name: gate.reasons for gate in decision.gates if not gate.passed}
        self.assertIn("provenance", failed)
        self.assertIn("cache_profile", failed)


class RunSchemaTests(unittest.TestCase):
    def test_run_digest_is_reproducible_and_summary_counts_errors(self) -> None:
        items = (
            _item("q0", 0, OutcomeStatus.CORRECT),
            _item("q1", 0, OutcomeStatus.ERROR),
        )
        first = BenchmarkRun(_provenance(size=2), items, metadata={"b": 2, "a": 1})
        second = BenchmarkRun(_provenance(size=2), items, metadata={"a": 1, "b": 2})
        self.assertEqual(first.digest(), second.digest())
        self.assertEqual(len(first.run_id), 16)
        self.assertEqual(first.summarize().outcomes.accuracy, 0.5)

    def test_run_rejects_duplicate_item_seed(self) -> None:
        item = _item("q", 0, OutcomeStatus.CORRECT)
        with self.assertRaisesRegex(BenchmarkContractError, "duplicate run item"):
            BenchmarkRun(_provenance(), (item, item))

    def test_run_rejects_partial_pinned_dataset_coverage(self) -> None:
        with self.assertRaisesRegex(BenchmarkContractError, "coverage"):
            BenchmarkRun(
                _provenance(size=2),
                (_item("only-one", 0, OutcomeStatus.CORRECT),),
            )

    def test_run_rejects_partial_performance_coverage(self) -> None:
        items = (
            _item("q0", 0, OutcomeStatus.CORRECT),
            _item("q1", 0, OutcomeStatus.CORRECT),
        )
        with self.assertRaisesRegex(BenchmarkContractError, "performance coverage"):
            BenchmarkRun(
                _provenance(size=2),
                items,
                (PerformanceMetric("q0", 0, "uncontrolled", 1, 1, 0, 1),),
            )


if __name__ == "__main__":
    unittest.main()
