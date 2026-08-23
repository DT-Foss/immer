from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen38_fertig_draft_verify.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "qwen38_fertig_draft_verify",
        SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import Qwen3.8 FERTIG verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verify_script = _load_script()


class _ByteTokenizer:
    def __init__(self, _path: str | Path) -> None:
        pass

    @staticmethod
    def render_no_thinking_prompt(system: str, user: str) -> str:
        return f"{system}\0{user}\0"

    @staticmethod
    def encode(text: str) -> tuple[int, ...]:
        return tuple(1000 + value for value in text.encode("utf-8"))

    @staticmethod
    def decode(token_ids) -> str:
        return bytes(int(token) - 1000 for token in token_ids).decode("utf-8")

    @staticmethod
    def token_pieces(token_ids) -> tuple[str, ...]:
        return tuple(chr(int(token) - 1000) for token in token_ids)


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    tokenizer = root / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    benchmark_rows = []
    baseline_rows = []
    for index, item_id in enumerate(verify_script.FIXED_ITEM_IDS):
        question = f"synthetic question {index}?"
        gold = str(index)
        text = f"#### {index}"
        prompt = _ByteTokenizer.render_no_thinking_prompt(
            verify_script.BASELINE_SYSTEM_PROMPT,
            question,
        )
        benchmark_rows.append(
            {
                "item_id": item_id,
                "status": "abstained",
                "question": question,
                "gold": f"work\n#### {gold}",
            }
        )
        baseline_rows.append(
            {
                "item_id": item_id,
                "question": question,
                "gold": gold,
                "predicted": gold,
                "correct": True,
                "status": "correct",
                "prompt_tokens": len(_ByteTokenizer.encode(prompt)),
                "completion_tokens": len(_ByteTokenizer.encode(text)),
                "finish_reason": "stop",
                "text": text,
            }
        )
    benchmark = root / "benchmark.json"
    benchmark.write_text(json.dumps({"items": benchmark_rows}), encoding="utf-8")
    drafts = root / "drafts.json"
    drafts.write_text(
        json.dumps(
            {
                "schema": verify_script.BASELINE_SCHEMA,
                "model": {
                    "revision": verify_script.BASELINE_MODEL_REVISION,
                    "backend": "openai-compatible",
                },
                "protocol": {
                    "system_prompt": verify_script.BASELINE_SYSTEM_PROMPT,
                    "temperature": 0.0,
                    "seed": 0,
                    "max_tokens": 32,
                },
                "summary": {
                    "total": 8,
                    "correct": 8,
                    "incorrect": 0,
                    "truncated": 0,
                    "unparseable": 0,
                    "error": 0,
                    "accuracy": 1.0,
                },
                "items": baseline_rows,
            }
        ),
        encoding="utf-8",
    )
    return tokenizer, benchmark, drafts


class Qwen38FertigDraftVerifyTests(unittest.TestCase):
    def test_prepare_only_validates_contextual_tokens_and_writes_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer, benchmark, drafts = _write_fixture(root)
            output = root / "inputs.json"
            args = verify_script._parser().parse_args(
                [
                    "--tokenizer-json",
                    str(tokenizer),
                    "--benchmark",
                    str(benchmark),
                    "--drafts-json",
                    str(drafts),
                    "--result-json",
                    str(output),
                    "--prepare-only",
                ]
            )

            with mock.patch.object(
                verify_script,
                "_verify_tokenizer_digest",
                side_effect=lambda path: path,
            ):
                result, written = verify_script.run(
                    args,
                    tokenizer_factory=_ByteTokenizer,
                )

            self.assertEqual(written, output.resolve())
            self.assertEqual(result["schema"], verify_script.INPUT_SCHEMA)
            self.assertEqual(result["summary"]["items"], 8)
            self.assertEqual(result["summary"]["candidate_accuracy"], 1.0)
            self.assertEqual(
                result["protocol"]["accepted_eos_token_ids"],
                [verify_script.IM_END_TOKEN_ID, verify_script.END_OF_TEXT_TOKEN_ID],
            )
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), result)

    def test_baseline_truncation_or_token_count_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer_path, benchmark, drafts = _write_fixture(root)
            tokenizer = _ByteTokenizer(tokenizer_path)
            document = json.loads(drafts.read_text(encoding="utf-8"))
            document["summary"]["truncated"] = 1
            drafts.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(verify_script.CliError, "incomplete"):
                verify_script._prepare_drafts(
                    benchmark,
                    drafts,
                    tokenizer,
                    max_draft_tokens=32,
                )

            document["summary"]["truncated"] = 0
            document["items"][0]["prompt_tokens"] += 1
            drafts.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(verify_script.CliError, "prompt token"):
                verify_script._prepare_drafts(
                    benchmark,
                    drafts,
                    tokenizer,
                    max_draft_tokens=32,
                )

    def test_verify_passes_exact_padding_stop_set_and_native_resume_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer_path, benchmark, drafts = _write_fixture(root)
            rows = verify_script._prepare_drafts(
                benchmark,
                drafts,
                _ByteTokenizer(tokenizer_path),
                max_draft_tokens=32,
            )
            args = verify_script._parser().parse_args(
                [
                    "--tokenizer-json",
                    str(tokenizer_path),
                    "--benchmark",
                    str(benchmark),
                    "--drafts-json",
                    str(drafts),
                    "--run-dir",
                    str(root / "run"),
                    "--cache-dir",
                    str(root / "cache"),
                    "--device",
                    "cpu",
                    "--dtype",
                    "float32",
                ]
            )
            captured: dict[str, object] = {}
            model = SimpleNamespace(
                config=SimpleNamespace(n_layers=4),
                pager=SimpleNamespace(
                    device=torch.device("cpu"),
                    compute_dtype=torch.float32,
                    source=SimpleNamespace(
                        budget=SimpleNamespace(total=0, limit=0),
                    ),
                ),
                prefill_hidden_shape=lambda batch, sequence: (batch, sequence, 12),
            )

            @contextmanager
            def runtime(_args, *, max_batch_size, max_seq_len):
                captured["runtime_shape"] = (max_batch_size, max_seq_len)
                yield model

            class FakeVerifier:
                def __init__(self, received_model, **kwargs):
                    captured["model"] = received_model
                    captured["constructor"] = kwargs

                def verify(self, prompts, candidates, **kwargs):
                    captured["prompts"] = prompts
                    captured["candidates"] = candidates
                    captured["verify"] = kwargs
                    return "verified"

            with mock.patch.object(
                verify_script,
                "load_resume",
                return_value=None,
            ) as load:
                result = verify_script._verify(
                    args,
                    rows,
                    runtime_factory=runtime,
                    verifier_factory=FakeVerifier,
                )

            self.assertEqual(result, "verified")
            verify_kwargs = captured["verify"]
            self.assertEqual(
                verify_kwargs["padding_token_id"],
                verify_script.END_OF_TEXT_TOKEN_ID,
            )
            self.assertEqual(
                verify_kwargs["eos_token_ids"],
                (verify_script.IM_END_TOKEN_ID, verify_script.END_OF_TEXT_TOKEN_ID),
            )
            expected_length = max(
                len(row.prompt_token_ids) + len(row.draft_token_ids) for row in rows
            )
            self.assertEqual(captured["runtime_shape"], (8, expected_length))
            self.assertEqual(
                load.call_args.kwargs["expected_shape"],
                (8, expected_length, 12),
            )

    def test_tokenizer_digest_and_resumed_source_budget_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            wrong = Path(temporary) / "tokenizer.json"
            wrong.write_text("{}", encoding="utf-8")
            args = verify_script._parser().parse_args(
                ["--tokenizer-json", str(wrong)]
            )
            with self.assertRaisesRegex(verify_script.CliError, "pinned"):
                verify_script._resolve_tokenizer_path(args)

        budget = SimpleNamespace(total=10, limit=999)
        model = SimpleNamespace(
            pager=SimpleNamespace(source=SimpleNamespace(budget=budget))
        )
        args = verify_script._parser().parse_args(["--source-budget-mb", "1"])
        report = verify_script._apply_cumulative_source_budget(
            args,
            model,
            prior_source_bytes=100,
        )
        self.assertEqual(report["total_limit_bytes"], 1024**2)
        self.assertEqual(budget.limit, 1024**2 - 100)
        with self.assertRaisesRegex(verify_script.CliError, "exhausted"):
            verify_script._apply_cumulative_source_budget(
                args,
                model,
                prior_source_bytes=1024**2 + 1,
            )


if __name__ == "__main__":
    unittest.main()
