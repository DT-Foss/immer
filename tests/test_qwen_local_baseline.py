from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from contextlib import redirect_stdout


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen_local_baseline.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("qwen_local_baseline", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import Qwen local baseline script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


baseline = _load_script()


class _FakeBackend:
    model_id = "fake/qwen"
    model_path = "/local/fake-qwen"
    model_revision = "a" * 40
    backend_name = "fake"
    load_seconds = 0.01

    def __init__(self, answers: dict[str, str], failing: str | None = None) -> None:
        self.answers = answers
        self.failing = failing

    def generate(self, question: str, *, max_tokens: int):
        if question == self.failing:
            raise RuntimeError("synthetic failure")
        return baseline.Generation(
            text=f"Reasoning.\n#### {self.answers[question]}",
            prompt_tokens=42,
            completion_tokens=min(8, max_tokens),
            finish_reason="stop",
            peak_memory_gb=1.5,
        )


class _UrlResponse:
    def __init__(self, document: dict, status: int = 200) -> None:
        self.document = document
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.document).encode("utf-8")


class QwenLocalBaselineTests(unittest.TestCase):
    def test_help_and_import_do_not_load_mlx(self) -> None:
        with mock.patch.dict(sys.modules, {"mlx_lm": None, "mlx": None}):
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    baseline._parser().parse_args(["--help"])
        self.assertEqual(caught.exception.code, 0)

    def test_selects_the_exact_eight_historical_fertig_rows(self) -> None:
        items = baseline.select_fixed_items(baseline.DEFAULT_BENCHMARK)
        self.assertEqual(tuple(row.item_id for row in items), baseline.FIXED_ITEM_IDS)
        self.assertEqual(
            tuple(row.gold for row in items),
            ("2", "4", "14000", "1", "450", "3", "16", "2"),
        )
        self.assertEqual(len({row.question for row in items}), 8)

    def test_selects_deterministic_dynamic_abstention_cohort(self) -> None:
        items = baseline.select_items(
            baseline.DEFAULT_BENCHMARK,
            cohort="abstained",
            limit=16,
            offset=0,
        )
        benchmark = json.loads(Path(baseline.DEFAULT_BENCHMARK).read_text())
        expected_all = [
            row["item_id"] for row in benchmark["items"] if row["status"] == "abstained"
        ]
        expected = expected_all[:16]
        self.assertEqual([row.item_id for row in items], expected)
        self.assertEqual(len({row.item_id for row in items}), 16)
        with self.assertRaisesRegex(baseline.CliError, "explicit --limit"):
            baseline.select_items(
                baseline.DEFAULT_BENCHMARK,
                cohort="abstained",
                limit=None,
                offset=0,
            )
        with self.assertRaisesRegex(baseline.CliError, "does not accept"):
            baseline.select_items(
                baseline.DEFAULT_BENCHMARK,
                cohort="fixed",
                limit=8,
                offset=0,
            )
        with self.assertRaisesRegex(baseline.CliError, "must not exceed 64"):
            baseline.select_items(
                baseline.DEFAULT_BENCHMARK,
                cohort="abstained",
                limit=65,
                offset=0,
            )

        holdout = baseline.select_items(
            baseline.DEFAULT_BENCHMARK,
            cohort="abstained",
            limit=16,
            offset=16,
        )
        self.assertEqual(
            [row.item_id for row in holdout],
            expected_all[16:32],
        )
        self.assertTrue(
            {row.item_id for row in items}.isdisjoint(row.item_id for row in holdout)
        )
        with self.assertRaisesRegex(baseline.CliError, "offset"):
            baseline.select_items(
                baseline.DEFAULT_BENCHMARK,
                cohort="fixed",
                limit=None,
                offset=1,
            )

    def test_numeric_extraction_is_canonical_and_uses_final_marker(self) -> None:
        cases = {
            "work 2 + 3 = 5\n#### 14,000.00": "14000",
            "The answer is -0.500": "-0.5",
            "#### +0.0": "0",
            "no number": None,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(baseline.extract_numeric_answer(text), expected)

    def test_fake_backend_writes_complete_json_without_mlx(self) -> None:
        items = baseline.select_fixed_items(baseline.DEFAULT_BENCHMARK)
        answers = {item.question: item.gold for item in items}
        answers[items[1].question] = "999"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result.json"
            args = baseline._parser().parse_args(["--output", str(output)])
            with mock.patch.dict(sys.modules, {"mlx_lm": None, "mlx": None}):
                with mock.patch.object(baseline, "_progress"):
                    report, written = baseline.run(
                        args,
                        backend_factory=lambda _args: _FakeBackend(answers),
                    )

            self.assertEqual(written, output.resolve())
            self.assertEqual(report["summary"]["total"], 8)
            self.assertEqual(report["summary"]["correct"], 7)
            self.assertEqual(report["summary"]["incorrect"], 1)
            self.assertEqual(report["summary"]["accuracy"], 7 / 8)
            self.assertEqual(len(report["items"]), 8)
            self.assertTrue(all(row["latency_seconds"] >= 0 for row in report["items"]))
            self.assertEqual(report["model"]["path"], "<external-local-checkpoint>")
            self.assertEqual(report["benchmark"]["source"], "results/bench_gsm8k.json")
            unsealed = dict(report)
            digest = unsealed.pop("report_sha256")
            self.assertEqual(digest, baseline._canonical_digest(unsealed))
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)

    def test_one_generation_error_is_retained_in_the_denominator(self) -> None:
        items = baseline.select_fixed_items(baseline.DEFAULT_BENCHMARK)
        answers = {item.question: item.gold for item in items}
        with tempfile.TemporaryDirectory() as temporary:
            args = baseline._parser().parse_args(
                ["--output", str(Path(temporary) / "result.json")]
            )
            with mock.patch.object(baseline, "_progress"):
                report, _ = baseline.run(
                    args,
                    backend_factory=lambda _args: _FakeBackend(
                        answers, failing=items[3].question
                    ),
                )
        self.assertEqual(report["summary"]["correct"], 7)
        self.assertEqual(report["summary"]["error"], 1)
        self.assertEqual(report["summary"]["accuracy"], 7 / 8)
        failed = report["items"][3]
        self.assertIsNone(failed["correct"])
        self.assertIn("synthetic failure", failed["error"])

    def test_limit_truncation_is_never_scored_from_an_intermediate_number(self) -> None:
        items = baseline.select_fixed_items(baseline.DEFAULT_BENCHMARK)
        answers = {item.question: item.gold for item in items}

        class TruncatedBackend(_FakeBackend):
            def generate(self, question: str, *, max_tokens: int):
                if question == items[0].question:
                    return baseline.Generation(
                        text="Partial reasoning ends at 14",
                        completion_tokens=max_tokens,
                        finish_reason="length",
                    )
                return super().generate(question, max_tokens=max_tokens)

        with tempfile.TemporaryDirectory() as temporary:
            args = baseline._parser().parse_args(
                ["--output", str(Path(temporary) / "result.json")]
            )
            with mock.patch.object(baseline, "_progress"):
                report, _ = baseline.run(
                    args,
                    backend_factory=lambda _args: TruncatedBackend(answers),
                )

        self.assertEqual(report["summary"]["truncated"], 1)
        self.assertEqual(report["summary"]["correct"], 7)
        self.assertEqual(report["summary"]["parsed_accuracy"], 1.0)
        truncated = report["items"][0]
        self.assertEqual(truncated["predicted"], "14")
        self.assertIsNone(truncated["correct"])

    def test_missing_local_checkpoint_fails_before_mlx_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing"
            with mock.patch.dict(sys.modules, {"mlx_lm": None, "mlx": None}):
                with self.assertRaisesRegex(
                    baseline.CliError, "checkpoint directory does not exist"
                ):
                    baseline.resolve_local_model(str(missing))

    def test_openai_backend_is_deterministic_and_strips_reasoning(self) -> None:
        requests = []

        def open_url(request, timeout):
            requests.append((request, timeout))
            if isinstance(request, str):
                return _UrlResponse({"object": "list", "data": []})
            return _UrlResponse(
                {
                    "choices": [
                        {
                            "message": {"content": "<think>private</think>\n\n#### 4"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 5},
                }
            )

        with mock.patch.object(
            baseline.urllib.request, "urlopen", side_effect=open_url
        ):
            backend = baseline.OpenAIBackend(
                "qwen",
                "Q3",
                base_url="http://127.0.0.1:8780/v1/",
                timeout=30,
                seed=7,
            )
            generated = backend.generate("2+2?", max_tokens=32)

        self.assertEqual(generated.text, "#### 4")
        self.assertEqual(generated.prompt_tokens, 20)
        self.assertEqual(generated.completion_tokens, 5)
        payload = json.loads(requests[1][0].data)
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["seed"], 7)
        self.assertEqual(payload["max_tokens"], 32)
        self.assertEqual(payload["messages"][1]["content"], "2+2?")

    def test_openai_backend_can_bypass_server_chat_template(self) -> None:
        requests = []

        def open_url(request, timeout):
            requests.append((request, timeout))
            if isinstance(request, str):
                return _UrlResponse({"object": "list", "data": []})
            return _UrlResponse(
                {
                    "choices": [{"text": "#### 4", "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 17, "completion_tokens": 3},
                }
            )

        with mock.patch.object(
            baseline.urllib.request, "urlopen", side_effect=open_url
        ):
            backend = baseline.OpenAIBackend(
                "qwen",
                "Q4",
                base_url="http://127.0.0.1:8780/v1",
                timeout=30,
                seed=0,
                prompt_mode="qwen3.8-no-thinking",
            )
            generated = backend.generate("2+2?", max_tokens=32)

        self.assertEqual(generated.text, "#### 4")
        self.assertTrue(requests[1][0].full_url.endswith("/v1/completions"))
        payload = json.loads(requests[1][0].data)
        self.assertNotIn("messages", payload)
        self.assertEqual(payload["stop"], ["<|im_end|>", "<|endoftext|>"])
        self.assertTrue(payload["prompt"].endswith("<think>\n\n</think>\n\n"))
        self.assertIn("<|im_start|>user\n2+2?<|im_end|>", payload["prompt"])

    def test_public_backend_location_never_serializes_machine_addresses(self) -> None:
        self.assertEqual(
            baseline._public_backend_location("http://127.0.0.1:8780/v1"),
            "<loopback-openai-compatible>",
        )
        self.assertEqual(
            baseline._public_backend_location("https://model.internal.example/v1"),
            "<external-openai-compatible>",
        )
        self.assertEqual(
            baseline._public_backend_location(str(baseline.ROOT / "models" / "qwen")),
            "models/qwen",
        )


if __name__ == "__main__":
    unittest.main()
