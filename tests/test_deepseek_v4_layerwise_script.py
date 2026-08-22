from __future__ import annotations

import copy
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as parquet

from test_deepseek_v4_layerwise import _config, _source


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "deepseek_v4_layerwise.py"
SPEC = importlib.util.spec_from_file_location("deepseek_v4_layerwise_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _ContextTokenizer:
    sha256 = "1" * 64

    def __init__(self, _path=None) -> None:
        self.prefix: str | None = None
        self.prefix_ids = (1, 2, 3)

    def encode(self, text: str) -> tuple[int, ...]:
        suffixes = {" A": 11, " B": 12, " C": 13, " D": 14}
        for suffix, token in suffixes.items():
            if text.endswith(suffix) and self.prefix == text[: -len(suffix)]:
                return (*self.prefix_ids, token)
        self.prefix = text
        return self.prefix_ids


class LayerwiseScriptTests(unittest.TestCase):
    def test_accuracy_is_pure_and_returns_separate_summaries(self) -> None:
        result = {
            "modes": {
                "off": {
                    "items": [
                        {"predicted": 1, "expected": 1},
                        {"predicted": 0, "expected": 1},
                    ]
                }
            }
        }
        original = copy.deepcopy(result)
        summaries = MODULE._accuracy(result)
        self.assertEqual(result, original)
        self.assertEqual(
            summaries,
            {"off": {"total": 2, "correct": 1, "accuracy": 0.5}},
        )

    def test_cli_rejects_non_bfloat16_layerwise_dtype(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            MODULE._parser().parse_args(["--dtype", "float32"])
        self.assertEqual(raised.exception.code, 2)

    def test_parquet_rows_are_loaded_with_stable_generated_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mmlu.parquet"
            parquet.write_table(
                pa.table(
                    {
                        "question": ["Q1", "Q2"],
                        "choices": [["a", "b"], ["c", "d"]],
                        "answer": [0, 1],
                    }
                ),
                path,
            )
            rows = MODULE._read_rows(path)
        self.assertEqual([row["id"] for row in rows], ["row-000000", "row-000001"])

    def test_official_prompt_and_contextual_candidates_are_constructed(self) -> None:
        args = MODULE._parser().parse_args([])
        rows = [
            {
                "id": "geo-1",
                "question": "Where?",
                "choices": ["North", "South", "East", "West"],
                "answer": 2,
            }
        ]
        items = MODULE._items(rows, _ContextTokenizer(), _config(), args)
        self.assertEqual(items[0].prompt_token_ids, (1, 2, 3))
        self.assertEqual(items[0].candidate_token_ids, (11, 12, 13, 14))
        self.assertEqual(items[0].expected, 2)

    def test_dry_run_plans_without_creating_activation_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset.parquet"
            dataset.write_bytes(b"fixture")
            run_dir = root / "activations"
            args = MODULE._parser().parse_args(
                [
                    "--dataset",
                    str(dataset),
                    "--run-dir",
                    str(run_dir),
                    "--microbatch-size",
                    "1",
                    "--limit",
                    "1",
                    "--dry-run",
                    "--disk-margin-gb",
                    "0",
                ]
            )
            rows = [
                {
                    "id": "geo-1",
                    "question": "Where?",
                    "choices": ["North", "South", "East", "West"],
                    "answer": 2,
                }
            ]
            with (
                mock.patch.object(MODULE, "_read_rows", return_value=rows),
                mock.patch.object(MODULE, "LocalTokenizer", _ContextTokenizer),
                mock.patch.object(MODULE, "_source", return_value=(_source(), "fixture")),
                mock.patch.object(
                    MODULE, "_config", return_value=(_config(), "2" * 64)
                ),
            ):
                receipt = MODULE.run(args)
            self.assertEqual(receipt["status"], "planned")
            self.assertFalse(run_dir.exists())
            self.assertIn("activation_transaction_peak_bytes", receipt["plan"])
            self.assertIn("official_source_safe_bytes", receipt["plan"])

    def test_complete_receipt_preserves_hashed_result_on_run_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset.parquet"
            dataset.write_bytes(b"fixture")
            run_dir = root / "activations"
            base = [
                "--dataset",
                str(dataset),
                "--run-dir",
                str(run_dir),
                "--microbatch-size",
                "1",
                "--limit",
                "1",
                "--modes",
                "off",
                "--graft-layer",
                "0",
                "--disk-margin-gb",
                "0",
            ]
            rows = [
                {
                    "id": "geo-1",
                    "question": "Where?",
                    "choices": ["North", "South", "East", "West"],
                    "answer": 2,
                }
            ]

            def execute(extra: list[str]) -> dict:
                args = MODULE._parser().parse_args([*base, *extra])
                with (
                    mock.patch.object(MODULE, "_read_rows", return_value=rows),
                    mock.patch.object(MODULE, "LocalTokenizer", _ContextTokenizer),
                    mock.patch.object(
                        MODULE, "_source", return_value=(_source(), "fixture")
                    ),
                    mock.patch.object(
                        MODULE, "_config", return_value=(_config(), "2" * 64)
                    ),
                ):
                    with redirect_stderr(io.StringIO()):
                        return MODULE.run(args)

            first = execute([])
            persisted = json.loads((run_dir / "result.json").read_text())
            self.assertEqual(first["result"], persisted["body"])
            self.assertEqual(first["result_body_sha256"], persisted["body_sha256"])
            self.assertEqual(set(first["summaries"]), {"off"})
            self.assertEqual(set(first["result"]["modes"]["off"]), {"items"})

            resumed = execute(["--resume"])
            self.assertEqual(resumed["result"], persisted["body"])
            self.assertEqual(resumed["result_body_sha256"], persisted["body_sha256"])
            self.assertEqual(resumed["summaries"], first["summaries"])

    def test_help_is_available_without_model_access(self) -> None:
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as raised:
            MODULE._parser().parse_args(["--help"])
        self.assertEqual(raised.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
