from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "deepseek_v4_benchmark.py"
OFFICIAL_TOKENIZER = (
    ROOT / "artifacts" / "private" / "deepseek-v4-reference" / "tokenizer.json"
)


def _fixture_writer():
    source = ROOT / "tests" / "test_deepseek_v4_poc_script.py"
    spec = importlib.util.spec_from_file_location("_immer_v4_poc_fixture", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load DeepSeek-V4 fixture helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._tiny_checkpoint


def _script_module():
    spec = importlib.util.spec_from_file_location("_immer_v4_benchmark", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load DeepSeek-V4 benchmark runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DeepSeekV4BenchmarkScriptTests(unittest.TestCase):
    def _run(
        self, *arguments: str, timeout: int = 180
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT / "src")
        return subprocess.run(
            [sys.executable, str(SCRIPT), *arguments],
            cwd=ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )

    def test_help_exposes_resume_modes_and_hard_budgets(self) -> None:
        result = self._run("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in (
            "--modes",
            "--seeds",
            "--limit",
            "--resume",
            "--journal",
            "--output",
            "--source-budget-mb",
            "--cache-budget-mb",
            "--max-dataset-mb",
            "--max-prompt-tokens",
            "--thinking-mode",
            "--allow-closed-set-gsm8k",
        ):
            self.assertIn(option, result.stdout)

    def test_offline_fixture_modes_seed_semantics_uncontrolled_cache_and_resume(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            checkpoint = work / "checkpoint"
            _fixture_writer()(checkpoint)
            dataset = work / "mmlu.jsonl"
            rows = [
                {
                    "id": "valid",
                    "prompt_token_ids": [7],
                    "candidate_token_ids": [[0], [1], [2], [3]],
                    "choices": ["zero", "one", "two", "three"],
                    "answer": 0,
                }
            ]
            dataset.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            journal = work / "run.jsonl"
            report = work / "report.json"
            cache = work / "cache"
            common = (
                "--dataset",
                str(dataset),
                "--task",
                "mmlu",
                "--source",
                str(checkpoint),
                "--revision",
                "local-fixture-v1",
                "--modes",
                "off",
                "crsa",
                "softmax",
                "shuffle",
                "--seeds",
                "0",
                "1",
                "2",
                "3",
                "4",
                "--journal",
                str(journal),
                "--output",
                str(report),
                "--cache-dir",
                str(cache),
                "--source-budget-mb",
                "64",
                "--cache-budget-mb",
                "8",
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--no-activation-quantization",
                "--max-prompt-tokens",
                "8",
                "--preflight",
                "exhaustive",
            )
            first = self._run(*common)
            self.assertEqual(first.returncode, 0, first.stderr)
            receipt = json.loads(first.stdout)
            self.assertEqual(receipt["status"], "complete")
            self.assertEqual(receipt["new_item_runs"], 8)
            document = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(
                document["schema"], "immer.deepseek-v4-benchmark-report/v1"
            )
            self.assertEqual(document["completed_item_runs"], 8)
            self.assertEqual(
                document["quantized_accumulation_policy"],
                "mx-block-scaled-fp32/v1",
            )
            self.assertEqual(
                document["attention_qat_policy"],
                "v4-native-fp8-kv+fp4-hadamard-indexer/v1",
            )
            self.assertEqual(document["budgets"]["cache_limit_bytes"], 8 * 1024**2)
            self.assertLessEqual(
                document["budgets"]["source_used_bytes_this_process"],
                document["budgets"]["source_limit_bytes_per_process"],
            )
            self.assertEqual(
                set(document["modes"]), {"off", "crsa", "softmax", "shuffle"}
            )
            self.assertIn("crsa_vs_off", document["paired"])
            self.assertIn("crsa_vs_shuffle_mean_over_seeds", document["paired"])
            for comparison in document["paired"].values():
                self.assertIsNotNone(comparison["candidate_mean_source_bytes"])
                self.assertIsNotNone(comparison["baseline_mean_source_bytes"])
                self.assertEqual(comparison["source_byte_cache_state"], "uncontrolled")
                self.assertFalse(comparison["source_byte_comparison_valid"])
            self.assertTrue(
                document["seed_protocol"][
                    "repeated_deterministic_seeds_are_not_executed_or_counted"
                ]
            )
            self.assertEqual(document["seed_protocol"]["deterministic_seed"], 0)
            self.assertEqual(document["modes"]["off"]["summary"]["total"], 1)
            self.assertEqual(document["modes"]["crsa"]["summary"]["total"], 1)
            self.assertEqual(document["modes"]["softmax"]["summary"]["total"], 1)
            self.assertEqual(document["modes"]["shuffle"]["summary"]["total"], 5)
            for mode in document["modes"].values():
                summary = mode["summary"]
                self.assertEqual(summary["errors"], 0)
                self.assertFalse(mode["cache_profile_complete"])
                self.assertFalse(mode["byte_comparison_across_modes_valid"])
                self.assertEqual(
                    mode["measured_cache_state"]["cache_state"], "uncontrolled"
                )
                self.assertEqual(
                    mode["measured_cache_state"]["total"], summary["total"]
                )
                self.assertEqual(
                    mode["measurement_totals"]["head_rows"], 4 * summary["total"]
                )
                self.assertEqual(
                    mode["measurement_totals"]["materialized_scale_bytes"],
                    6184 * summary["total"],
                )
                self.assertTrue(
                    all(
                        perf["cache_state"] == "uncontrolled"
                        for perf in mode["run"]["performance"]
                    )
                )
                items = mode["run"]["items"]
                valid = items[0]
                self.assertEqual(
                    valid["metadata"]["runtime"]["engine"], "candidate_token_head"
                )
                self.assertEqual(
                    valid["metadata"]["runtime"]["head_row_ranges"], [[0, 4]]
                )
            self.assertEqual(
                document["cache_measurement_protocol"]["state"], "uncontrolled"
            )
            provenance_protocol = document["modes"]["off"]["run"]["provenance"][
                "protocol"
            ]
            self.assertIn(
                "external_pretokenized_prompt/v1",
                provenance_protocol["prompt_protocols"],
            )
            lines_before = journal.read_bytes().splitlines()
            self.assertEqual(len(lines_before), 9)

            resumed = self._run(*common, "--resume")
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            resumed_receipt = json.loads(resumed.stdout)
            self.assertEqual(resumed_receipt["new_item_runs"], 0)
            self.assertEqual(journal.read_bytes().splitlines(), lines_before)

            with journal.open("ab") as handle:
                handle.write(b'{"torn":')
            repaired = self._run(*common, "--resume")
            self.assertEqual(repaired.returncode, 0, repaired.stderr)
            self.assertEqual(json.loads(repaired.stdout)["new_item_runs"], 0)
            self.assertEqual(journal.read_bytes().splitlines(), lines_before)
            self.assertFalse(
                any(path.name.startswith(".report.json.") for path in work.iterdir())
            )

    def test_tiny_source_budget_fails_before_claiming_a_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            checkpoint = work / "checkpoint"
            _fixture_writer()(checkpoint)
            dataset = work / "mmlu.json"
            dataset.write_text(
                json.dumps(
                    [
                        {
                            "id": "q",
                            "prompt_token_ids": [7],
                            "candidate_token_ids": [0, 1],
                            "answer": 0,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            report = work / "report.json"
            base = (
                "--dataset",
                str(dataset),
                "--task",
                "mmlu",
                "--source",
                str(checkpoint),
                "--revision",
                "fixture",
                "--modes",
                "off",
                "--cache-dir",
                str(work / "cache"),
                "--source-budget-mb",
                "0.001",
                "--cache-budget-mb",
                "1",
                "--device",
                "cpu",
            )
            result = self._run(
                *base,
                "--journal",
                str(work / "run.jsonl"),
                "--output",
                str(report),
            )
            self.assertEqual(result.returncode, 2)
            error = json.loads(result.stderr)
            self.assertEqual(error["status"], "error")
            self.assertIn("budget", error["error"].lower())
            self.assertFalse(report.exists())

    def test_runtime_budget_exhaustion_is_an_item_error_in_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            checkpoint = work / "checkpoint"
            _fixture_writer()(checkpoint)
            dataset = work / "mmlu.json"
            dataset.write_text(
                json.dumps(
                    [
                        {
                            "id": "runtime-error",
                            "prompt_token_ids": [7],
                            "candidate_token_ids": [0, 1],
                            "answer": 0,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            report = work / "report.json"
            result = self._run(
                "--dataset",
                str(dataset),
                "--task",
                "mmlu",
                "--source",
                str(checkpoint),
                "--revision",
                "fixture",
                "--modes",
                "off",
                "--journal",
                str(work / "run.jsonl"),
                "--output",
                str(report),
                "--cache-dir",
                str(work / "cache"),
                "--source-budget-mb",
                "0.01",
                "--cache-budget-mb",
                "1",
                "--device",
                "cpu",
                "--preflight",
                "none",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            document = json.loads(report.read_text(encoding="utf-8"))
            summary = document["modes"]["off"]["summary"]
            self.assertEqual((summary["total"], summary["errors"]), (1, 1))
            self.assertEqual(summary["accuracy_total"], 0.0)
            self.assertEqual(summary["error_rate"], 1.0)
            item = document["modes"]["off"]["run"]["items"][0]
            self.assertEqual(item["status"], "error")
            self.assertIn("budget", item["error"].lower())

    def test_invalid_selected_token_id_fails_setup_before_journal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            checkpoint = work / "checkpoint"
            _fixture_writer()(checkpoint)
            dataset = work / "bad.json"
            dataset.write_text(
                json.dumps(
                    [
                        {
                            "id": "bad",
                            "prompt_token_ids": [999],
                            "candidate_token_ids": [0, 1],
                            "answer": 0,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            journal = work / "run.jsonl"
            base = (
                "--dataset",
                str(dataset),
                "--task",
                "mmlu",
                "--source",
                str(checkpoint),
                "--revision",
                "fixture",
                "--modes",
                "off",
                "--journal",
                str(journal),
                "--output",
                str(work / "report.json"),
                "--cache-dir",
                str(work / "cache"),
                "--source-budget-mb",
                "64",
                "--cache-budget-mb",
                "8",
                "--device",
                "cpu",
            )
            result = self._run(*base)
            self.assertEqual(result.returncode, 2)
            self.assertIn("prompt token outside", result.stderr)
            self.assertFalse(journal.exists())

    def test_text_plan_uses_official_thinking_envelope(self) -> None:
        module = _script_module()

        class CapturingTokenizer:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            def encode(self, text: str) -> list[int]:
                self.prompts.append(text)
                if text.endswith("Answer: A"):
                    return [1, 2, 3, 20]
                if text.endswith("Answer: B"):
                    return [1, 2, 3, 21]
                return [1, 2, 3]

        tokenizer = CapturingTokenizer()
        args = types.SimpleNamespace(
            thinking_mode="thinking",
            reasoning_effort="low",
            max_prompt_tokens=32,
            max_new_tokens=4,
            eos_token_ids=[],
            allow_closed_set_gsm8k=False,
        )
        plans = module._prepare_rows(
            [
                {
                    "id": "q",
                    "question": "Capital of France?",
                    "choices": ["Paris", "Berlin"],
                    "answer": 0,
                }
            ],
            module.BenchmarkTask.MMLU,
            tokenizer,
            types.SimpleNamespace(vocab_size=128),
            args,
        )
        encoded = tokenizer.prompts[0]
        self.assertEqual(
            encoded,
            "<｜begin▁of▁sentence｜><｜User｜>Capital of France?\n"
            "A. Paris\nB. Berlin<｜Assistant｜><think>Answer:",
        )
        self.assertEqual(plans["q"].candidate_token_ids, (20, 21))
        self.assertEqual(plans["q"].candidate_surfaces, (" A", " B"))
        self.assertEqual(
            plans["q"].candidate_tokenization,
            "full_prompt_exact_prefix_one_token_suffix_verified",
        )
        self.assertEqual(plans["q"].assistant_generation_prefix, "Answer:")
        self.assertEqual(
            plans["q"].prompt_encoding,
            "official_single_user_message/thinking/low",
        )

    def test_text_candidate_must_preserve_full_prompt_token_prefix(self) -> None:
        module = _script_module()

        class BoundaryMergingTokenizer:
            def encode(self, text: str) -> list[int]:
                if text.endswith("Answer: A"):
                    return [1, 2, 99]
                if text.endswith("Answer: B"):
                    return [1, 2, 3, 21]
                return [1, 2, 3]

        args = types.SimpleNamespace(
            thinking_mode="chat",
            reasoning_effort="low",
            max_prompt_tokens=32,
            max_new_tokens=4,
            eos_token_ids=[],
            allow_closed_set_gsm8k=False,
        )
        with self.assertRaisesRegex(
            module.RunnerError, "retokenizes the prompt prefix"
        ):
            module._prepare_rows(
                [
                    {
                        "id": "boundary",
                        "question": "Q?",
                        "choices": ["x", "y"],
                        "answer": 0,
                    }
                ],
                module.BenchmarkTask.MMLU,
                BoundaryMergingTokenizer(),
                types.SimpleNamespace(vocab_size=128),
                args,
            )

    @unittest.skipUnless(OFFICIAL_TOKENIZER.exists(), "official tokenizer unavailable")
    def test_official_tokenizer_scores_space_letter_after_answer_prefix(self) -> None:
        module = _script_module()
        tokenizer = module.LocalTokenizer(OFFICIAL_TOKENIZER)
        args = types.SimpleNamespace(
            thinking_mode="chat",
            reasoning_effort="low",
            max_prompt_tokens=64,
            max_new_tokens=4,
            eos_token_ids=[],
            allow_closed_set_gsm8k=False,
        )
        plans = module._prepare_rows(
            [
                {
                    "id": "official",
                    "question": "Q?",
                    "choices": ["x", "y"],
                    "answer": 0,
                }
            ],
            module.BenchmarkTask.MMLU,
            tokenizer,
            types.SimpleNamespace(vocab_size=129_280),
            args,
        )
        plan = plans["official"]
        self.assertEqual(plan.candidate_token_ids, (334, 406))
        self.assertNotEqual(plan.candidate_token_ids[0], tokenizer.encode("A")[0])

    @unittest.skipUnless(OFFICIAL_TOKENIZER.exists(), "official tokenizer unavailable")
    def test_canonical_text_rows_auto_discover_official_local_tokenizer(self) -> None:
        module = _script_module()
        defaults = module._parser().parse_args(
            [
                "--dataset",
                "unused",
                "--task",
                "mmlu",
                "--journal",
                "unused.jsonl",
                "--output",
                "unused.json",
            ]
        )
        tokenizer = module._load_tokenizer(
            defaults,
            object(),
            [{"question": "Q?", "choices": ["x", "y"], "answer": 0}],
            module.BenchmarkTask.MMLU,
        )
        self.assertIsNotNone(tokenizer)
        self.assertEqual(Path(tokenizer.location), OFFICIAL_TOKENIZER.resolve())
        self.assertEqual(tokenizer.encode("A"), [35])

    @unittest.skipUnless(
        importlib.util.find_spec("pyarrow.parquet") is not None
        and OFFICIAL_TOKENIZER.exists()
        and (ROOT / "evals" / "mmlu_high_school_geography_test.parquet").exists()
        and (ROOT / "evals" / "gsm8k_test.parquet").exists(),
        "bundled canonical data or tokenizer unavailable",
    )
    def test_default_prompt_bound_covers_bundled_canonical_splits(self) -> None:
        module = _script_module()
        tokenizer = module.LocalTokenizer(OFFICIAL_TOKENIZER)
        parser = module._parser()
        defaults = parser.parse_args(
            [
                "--dataset",
                "unused",
                "--task",
                "mmlu",
                "--journal",
                "unused.jsonl",
                "--output",
                "unused.json",
            ]
        )
        config = types.SimpleNamespace(vocab_size=129_280)
        for filename, task in (
            ("mmlu_high_school_geography_test.parquet", module.BenchmarkTask.MMLU),
            ("gsm8k_test.parquet", module.BenchmarkTask.GSM8K),
        ):
            rows = module._read_dataset(
                ROOT / "evals" / filename, maximum_bytes=64 * 1024**2
            )
            plans = module._prepare_rows(rows, task, tokenizer, config, defaults)
            self.assertEqual(len(plans), len(rows))

    @unittest.skipUnless(
        importlib.util.find_spec("pyarrow.parquet") is not None,
        "pyarrow unavailable",
    )
    def test_standard_parquet_rows_are_read_without_mutating_source(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        module = _script_module()
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "mmlu.parquet"
            table = pa.Table.from_pylist(
                [
                    {
                        "question": "Q?",
                        "choices": ["A0", "A1", "A2", "A3"],
                        "answer": 2,
                    }
                ]
            )
            pq.write_table(table, dataset)
            before = dataset.read_bytes()
            rows = module._read_dataset(dataset, maximum_bytes=dataset.stat().st_size)
            self.assertEqual(rows[0]["id"], "row-000000")
            self.assertEqual(rows[0]["answer"], 2)
            self.assertEqual(rows[0]["choices"], ["A0", "A1", "A2", "A3"])
            self.assertEqual(dataset.read_bytes(), before)

    def test_gsm8k_candidate_tokens_and_limit_execute_offline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            checkpoint = work / "checkpoint"
            _fixture_writer()(checkpoint)
            dataset = work / "gsm8k.json"
            dataset.write_text(
                json.dumps(
                    [
                        {
                            "id": "g0",
                            "prompt_token_ids": [7],
                            "candidate_token_ids": [[4], [5]],
                            "candidate_answers": ["10", "42"],
                            "answer": "reasoning #### 42",
                        },
                        {
                            "id": "excluded-by-limit",
                            "prompt_token_ids": [999],
                            "candidate_token_ids": [[4], [5]],
                            "candidate_answers": ["1", "2"],
                            "answer": "#### 2",
                        },
                    ]
                ),
                encoding="utf-8",
            )
            report = work / "report.json"
            base = (
                "--dataset",
                str(dataset),
                "--task",
                "gsm8k",
                "--limit",
                "1",
                "--source",
                str(checkpoint),
                "--revision",
                "fixture",
                "--modes",
                "off",
                "--cache-dir",
                str(work / "cache"),
                "--source-budget-mb",
                "64",
                "--cache-budget-mb",
                "8",
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--no-activation-quantization",
                "--preflight",
                "sample",
            )
            rejected_journal = work / "rejected.jsonl"
            rejected = self._run(
                *base,
                "--journal",
                str(rejected_journal),
                "--output",
                str(work / "rejected.json"),
            )
            self.assertEqual(rejected.returncode, 2)
            self.assertIn("closed-set diagnostics", rejected.stderr)
            self.assertFalse(rejected_journal.exists())

            result = self._run(
                *base,
                "--allow-closed-set-gsm8k",
                "--journal",
                str(work / "run.jsonl"),
                "--output",
                str(report),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            document = json.loads(report.read_text(encoding="utf-8"))
            summary = document["modes"]["off"]["summary"]
            self.assertEqual((summary["total"], summary["errors"]), (1, 0))
            self.assertIsNone(summary["accuracy_total"])
            self.assertFalse(summary["accuracy_is_canonical"])
            diagnostic = document["modes"]["off"]["protocol_summaries"][
                "gsm8k_closed_set_candidate_token_diagnostic"
            ]
            self.assertEqual(diagnostic["total"], 1)
            item = document["modes"]["off"]["run"]["items"][0]
            self.assertEqual(item["expected"], "42")
            self.assertIn(item["predicted"], ("10", "42"))
            self.assertEqual(
                item["metadata"]["runtime"]["engine"], "candidate_token_head"
            )
            self.assertEqual(
                item["metadata"]["benchmark_protocol"],
                "gsm8k_closed_set_candidate_token_diagnostic",
            )


if __name__ == "__main__":
    unittest.main()
