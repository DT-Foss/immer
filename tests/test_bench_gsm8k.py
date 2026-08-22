from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from immer.contracts import ExecutionStatus, Result


ROOT = Path(__file__).resolve().parents[1]


def _load_benchmark():
    path = ROOT / "scripts" / "bench_gsm8k.py"
    spec = importlib.util.spec_from_file_location("_test_bench_gsm8k", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _SequenceSolver:
    def __init__(self, outcomes):
        self._outcomes = iter(outcomes)

    def handle(self, _request):
        outcome = next(self._outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _UnprintableOutput:
    def __str__(self) -> str:
        raise ValueError("cannot render")


class BenchGsm8kTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bench = _load_benchmark()

    @staticmethod
    def _record(index: int, gold: str = "10") -> dict[str, str]:
        return {
            "question": f"question {index}",
            "answer": f"worked answer\n#### {gold}",
        }

    def test_item_ids_and_dataset_digest_are_deterministic_and_order_bound(
        self,
    ) -> None:
        records = [self._record(0), self._record(1)]
        first = self.bench.records_digest(records)
        self.assertEqual(first, self.bench.records_digest(records))
        self.assertNotEqual(first, self.bench.records_digest(list(reversed(records))))

        item_id = self.bench.deterministic_item_id(
            0, records[0]["question"], records[0]["answer"]
        )
        self.assertEqual(
            item_id,
            self.bench.deterministic_item_id(
                0, records[0]["question"], records[0]["answer"]
            ),
        )
        self.assertNotEqual(
            item_id,
            self.bench.deterministic_item_id(
                1, records[0]["question"], records[0]["answer"]
            ),
        )

    def test_outcomes_form_partition_and_errors_remain_in_wrong_gate(self) -> None:
        records = [self._record(index) for index in range(7)]
        records[-1] = {"question": "bad gold", "answer": "no marker"}
        solver = _SequenceSolver(
            [
                Result(ExecutionStatus.OK, "fake", output="answer 10"),
                Result(
                    ExecutionStatus.ABSTAINED,
                    "fake",
                    reason="not proven",
                ),
                Result(ExecutionStatus.OK, "fake", output="no number"),
                Result(ExecutionStatus.OK, "fake", output="answer 9"),
                Result(ExecutionStatus.ERROR, "fake", reason="runtime failed"),
                RuntimeError("boom"),
            ]
        )

        items = self.bench.evaluate_records(records, solver)
        self.assertEqual(
            [item["status"] for item in items],
            [
                "correct",
                "abstained",
                "abstained",
                "incorrect",
                "error",
                "error",
                "error",
            ],
        )
        for item in items:
            self.assertEqual(
                {
                    "item_id",
                    "index",
                    "status",
                    "question",
                    "gold",
                    "predicted",
                    "reason",
                },
                set(item),
            )

        summary = self.bench.summarize_items(items)
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["abstained"], 2)
        self.assertEqual(summary["incorrect"], 1)
        self.assertEqual(summary["errors"], 3)
        self.assertEqual(summary["wrong"], 4)
        self.assertFalse(summary["wrong_must_be_zero"])
        self.assertEqual(summary["accuracy_on_attempted"], 0.2)

    def test_result_processing_exception_is_an_error_not_a_dropped_row(self) -> None:
        solver = _SequenceSolver(
            [Result(ExecutionStatus.OK, "fake", output=_UnprintableOutput())]
        )
        items = self.bench.evaluate_records([self._record(0)], solver)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["status"], "error")
        self.assertIn("ValueError: cannot render", items[0]["reason"])

    def test_report_keeps_v1_summary_and_carries_provenance(self) -> None:
        item = {
            "item_id": "gsm8k-test-0000-deadbeefdeadbeef",
            "index": 0,
            "status": "correct",
            "question": "1 + 1?",
            "gold": 2.0,
            "predicted": 2.0,
            "reason": "numeric_match",
        }
        digest = "a" * 64
        dataset = {"sha256": digest, "name": "fixture"}
        harness = {"sha256": "b" * 64, "solver": "fixture solver"}
        report = self.bench.build_report(
            [item], seconds=0.04, dataset=dataset, harness=harness
        )

        self.assertEqual(report["schema"], "immer.benchmark/v1")
        self.assertEqual(report["n"], 1)
        self.assertEqual(report["wrong"], 0)
        self.assertTrue(report["wrong_must_be_zero"])
        self.assertEqual(report["dataset_sha256"], digest)
        self.assertEqual(report["harness_sha256"], "b" * 64)
        self.assertEqual(len(report["items_sha256"]), 64)
        self.assertEqual(report["items"], [item])

    def test_json_and_failure_jsonl_are_atomic_and_complete(self) -> None:
        items = [
            {
                "item_id": "correct",
                "index": 0,
                "status": "correct",
                "question": "q0",
                "gold": 1.0,
                "predicted": 1.0,
                "reason": "numeric_match",
            },
            {
                "item_id": "failed",
                "index": 1,
                "status": "error",
                "question": "q1",
                "gold": 2.0,
                "predicted": None,
                "reason": "solver_exception: RuntimeError: boom",
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report_path = directory / "nested" / "report.json"
            failure_path = directory / "nested" / "failures.jsonl"
            report = {"items": items, "n": 2}

            self.bench.write_report(report_path, report)
            failures = self.bench.write_failures_jsonl(failure_path, items)

            self.assertEqual(json.loads(report_path.read_text()), report)
            failure_rows = [
                json.loads(line)
                for line in failure_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(failures, 1)
            self.assertEqual(failure_rows, [items[1]])
            self.assertEqual(list(report_path.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
