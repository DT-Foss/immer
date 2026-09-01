from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from immer.contracts import ExecutionStatus, Result
from immer.runtimes.qwen3_8.inference_economics import (
    INFERENCE_ECONOMICS_ROLLUP_SCHEMA,
    InferenceEconomicsError,
    InferenceEconomicsLedger,
    InferenceEconomicsReceipt,
    receipt_from_result,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _raw_result(output: str = "answer") -> Result:
    return Result(
        ExecutionStatus.OK,
        "qwen3.8.causal-chat",
        output=output,
        evidence={
            "generation": {
                "forward_passes": 3,
                "generated_tokens": 4,
                "linear_calls": 90,
                "seconds": 2.5,
                "source_body_bytes": 300,
                "time_to_first_token_seconds": 1.25,
            },
            "draft": {
                "accepted_draft_tokens": 2,
                "draft_source_body_bytes": 99,
                "rounds": [
                    {"proposed_token_ids": [7, 8, 9]},
                ],
                "target_source_body_bytes": 500,
            },
            "mlp_page_route": {
                "page_count": 272,
                "route_width": 192,
                "request": {
                    "adaptive_width_pages_saved": 11,
                    "exact_rows": 2,
                },
            },
            "q4": {
                "request": {
                    "logical_weight_bytes": 400,
                    "page_mlp_prefetch_bytes": 50,
                    "page_mlp_selected_pages": 384,
                    "page_mlp_weight_bytes": 200,
                },
            },
            "runtime_metrics": {
                "generation_wall_seconds": 2.75,
                "physical_read_bytes": 123,
                "process_peak_rss_bytes": 1024,
            },
            "runtime_reward": {
                "o1_priority": 4.5,
                "reward": 1.75,
            },
        },
    )


def _wrapped_warm_result() -> Result:
    core = {
        "candidate": "answer",
        "fertig": {"status": "abstained"},
        "kind": "qwen-fertig-chat-receipt/v1",
        "ooe": {
            "accounting": {
                "disposition": "committed",
                "saved_qwen_forwards": 4,
            },
        },
        "output_sha256": _sha("answer"),
        "question_sha256": _sha("question"),
        "qwen": {
            "draft": {
                "accepted_draft_tokens": 2,
                "draft_source_body_bytes": 0,
                "target_source_body_bytes": 0,
            },
            "generation": {
                "forward_passes": 0,
                "generated_tokens": 4,
                "linear_calls": 0,
                "seconds": 0.0,
                "source_body_bytes": 0,
            },
            "status": "ok",
        },
        "route": "ooe_verification_abstained",
    }
    core["receipt_sha256"] = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return Result(
        ExecutionStatus.OK,
        "qwen3.8.fertig-chat",
        output="answer",
        evidence={"receipt": core},
    )


class InferenceEconomicsReceiptTests(unittest.TestCase):
    def test_raw_result_becomes_one_bounded_receipt_without_prompt_text(self) -> None:
        receipt = receipt_from_result(
            _raw_result(),
            question_sha256=_sha("private raw prompt"),
            runtime_profile_sha256=_sha("profile"),
        )
        self.assertEqual(receipt.target_forwards, 3)
        self.assertEqual(receipt.saved_qwen_forwards, 1)
        self.assertFalse(receipt.warm_hit)
        self.assertEqual(receipt.accepted_draft_tokens, 2)
        self.assertEqual(receipt.proposed_draft_tokens, 3)
        self.assertEqual(receipt.target_source_body_bytes, 500)
        self.assertEqual(receipt.avoidable_work_bytes["draft_miss"], 33)
        self.assertEqual(receipt.avoidable_work_bytes["target_fallback"], 500)
        self.assertEqual(receipt.selected_pages, 384)
        self.assertEqual(receipt.saved_pages, 160)
        self.assertEqual(receipt.o1_priority, 4.5)
        document = receipt.to_document()
        self.assertEqual(InferenceEconomicsReceipt.from_document(document), receipt)
        self.assertNotIn(
            b"private raw prompt",
            json.dumps(document, sort_keys=True).encode("utf-8"),
        )

    def test_wrapped_warm_result_records_zero_target_and_saved_forwards(self) -> None:
        receipt = receipt_from_result(
            _wrapped_warm_result(),
            question_sha256=_sha("question"),
            runtime_profile_sha256=_sha("profile"),
        )
        self.assertTrue(receipt.warm_hit)
        self.assertEqual(receipt.target_forwards, 0)
        self.assertEqual(receipt.saved_qwen_forwards, 4)
        self.assertEqual(receipt.accepted_draft_tokens, 2)
        self.assertEqual(receipt.proposed_draft_tokens, 2)
        self.assertEqual(receipt.route, "ooe_verification_abstained")
        forged = _wrapped_warm_result()
        forged_receipt = dict(forged.evidence["receipt"])
        forged_receipt["receipt_sha256"] = "f" * 64
        with self.assertRaises(InferenceEconomicsError):
            receipt_from_result(
                Result(
                    forged.status,
                    forged.component,
                    output=forged.output,
                    evidence={"receipt": forged_receipt},
                ),
                question_sha256=_sha("question"),
                runtime_profile_sha256=_sha("profile"),
            )


class InferenceEconomicsLedgerTests(unittest.TestCase):
    def test_append_restart_idempotence_and_rollup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "economics"
            ledger = InferenceEconomicsLedger(root)
            first = ledger.observe(
                _raw_result("first"),
                question_sha256=_sha("first private prompt"),
                runtime_profile_sha256=_sha("profile"),
            )
            duplicate = ledger.observe(
                _raw_result("first"),
                question_sha256=_sha("first private prompt"),
                runtime_profile_sha256=_sha("profile"),
            )
            second = ledger.observe(
                _wrapped_warm_result(),
                question_sha256=_sha("question"),
                runtime_profile_sha256=_sha("profile"),
                request_sha256=_sha("second-request"),
            )
            self.assertFalse(first.duplicate)
            self.assertTrue(duplicate.duplicate)
            self.assertFalse(second.duplicate)
            self.assertEqual(len(tuple((root / "events").iterdir())), 2)
            snapshot = ledger.snapshot()
            self.assertEqual(snapshot["requests"], 2)
            self.assertEqual(snapshot["target_forwards"], 3)
            self.assertEqual(snapshot["saved_qwen_forwards"], 5)
            self.assertEqual(
                snapshot["largest_avoidable_cost_class"],
                "target_fallback",
            )
            restarted = InferenceEconomicsLedger(root)
            self.assertEqual(restarted.snapshot(), snapshot)
            self.assertEqual(
                [receipt.request_sha256 for receipt in restarted.receipts()],
                [first.receipt.request_sha256, second.receipt.request_sha256],
            )
            state_bytes = b"".join(
                path.read_bytes()
                for path in sorted((root / "events").iterdir())
            )
            self.assertNotIn(b"first private prompt", state_bytes)

    def test_event_tamper_is_hard_and_rollup_is_recoverable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "economics"
            ledger = InferenceEconomicsLedger(root)
            ledger.observe(
                _raw_result(),
                question_sha256=_sha("question"),
                runtime_profile_sha256=_sha("profile"),
            )
            rollup = root / "rollup.json"
            rollup.chmod(0o600)
            rollup.write_bytes(b"{}")
            repaired = InferenceEconomicsLedger(root)
            repaired_rollup = json.loads(rollup.read_bytes())
            self.assertEqual(
                repaired_rollup["schema"],
                INFERENCE_ECONOMICS_ROLLUP_SCHEMA,
            )
            self.assertEqual(repaired.snapshot()["requests"], 1)
            rollup.chmod(0o600)
            rollup.write_bytes(b"{}")
            duplicate = repaired.observe(
                _raw_result(),
                question_sha256=_sha("question"),
                runtime_profile_sha256=_sha("profile"),
            )
            self.assertTrue(duplicate.duplicate)
            self.assertEqual(json.loads(rollup.read_bytes())["body"]["requests"], 1)

            event = next((root / "events").iterdir())
            payload = bytearray(event.read_bytes())
            payload[-1] ^= 1
            event.chmod(0o600)
            event.write_bytes(payload)
            with self.assertRaises(InferenceEconomicsError):
                InferenceEconomicsLedger(root)

    def test_concurrent_duplicate_observation_creates_one_event(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "economics"
            ledger = InferenceEconomicsLedger(root)

            def observe(_index: int):
                return ledger.observe(
                    _raw_result(),
                    question_sha256=_sha("same question"),
                    runtime_profile_sha256=_sha("profile"),
                )

            with ThreadPoolExecutor(max_workers=8) as executor:
                observations = tuple(executor.map(observe, range(16)))
            self.assertEqual(sum(not row.duplicate for row in observations), 1)
            self.assertEqual(len(tuple((root / "events").iterdir())), 1)
            self.assertEqual(ledger.snapshot()["requests"], 1)

    def test_one_request_identity_cannot_bind_two_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ledger = InferenceEconomicsLedger(Path(temporary) / "economics")
            request_sha256 = _sha("request")
            ledger.observe(
                _raw_result("first"),
                question_sha256=_sha("question"),
                runtime_profile_sha256=_sha("profile"),
                request_sha256=request_sha256,
            )
            with self.assertRaises(InferenceEconomicsError):
                ledger.observe(
                    _raw_result("different"),
                    question_sha256=_sha("question"),
                    runtime_profile_sha256=_sha("profile"),
                    request_sha256=request_sha256,
                )

    def test_two_ledger_instances_serialize_one_process_journal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "economics"
            ledgers = (InferenceEconomicsLedger(root), InferenceEconomicsLedger(root))

            def observe(index: int):
                return ledgers[index % 2].observe(
                    _raw_result(f"answer-{index}"),
                    question_sha256=_sha(f"question-{index}"),
                    runtime_profile_sha256=_sha("profile"),
                    request_sha256=_sha(f"request-{index}"),
                )

            with ThreadPoolExecutor(max_workers=8) as executor:
                observations = tuple(executor.map(observe, range(16)))
            self.assertTrue(all(not row.duplicate for row in observations))
            self.assertEqual(InferenceEconomicsLedger(root).snapshot()["requests"], 16)
            self.assertEqual(len(tuple((root / "events").iterdir())), 16)

    def test_symlink_root_is_rejected_before_children_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            target = base / "target"
            target.mkdir()
            link = base / "ledger-link"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaises(InferenceEconomicsError):
                InferenceEconomicsLedger(link)
            self.assertFalse((target / "events").exists())
            self.assertFalse((target / "staging").exists())


if __name__ == "__main__":
    unittest.main()
