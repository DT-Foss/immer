from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from immer.runtimes.qwen3_8.action_bank import (
    INFERENCE_ACTION_BANK_SCHEMA,
    InferenceActionBank,
    InferenceActionBankError,
    InferenceActionReceipt,
)
from immer.runtimes.qwen3_8.inference_economics import InferenceEconomicsReceipt


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _economics(
    request: str,
    *,
    warm: bool = False,
    fertig: bool = False,
    draft: bool = True,
    pages: bool = True,
    target: int = 3,
    saved: int = 1,
) -> InferenceEconomicsReceipt:
    return InferenceEconomicsReceipt(
        request_sha256=_sha(request),
        question_sha256=_sha(f"question:{request}"),
        runtime_profile_sha256=_sha("profile"),
        result_sha256=_sha(f"result:{request}"),
        output_sha256=_sha(f"output:{request}"),
        status="ok",
        component="qwen3.8.fertig-chat",
        route="ooe_verification_abstained" if warm else "qwen_verification_abstained",
        target_forwards=target,
        saved_qwen_forwards=saved,
        generated_tokens=target + saved,
        accepted_draft_tokens=saved if draft else 0,
        proposed_draft_tokens=saved + 1 if draft else 0,
        source_body_bytes=100,
        target_source_body_bytes=90,
        draft_source_body_bytes=10 if draft else 0,
        logical_weight_bytes=500,
        page_mlp_weight_bytes=200 if pages else 0,
        prefetch_bytes=50 if pages else 0,
        physical_read_bytes=25,
        linear_calls=10,
        process_peak_rss_bytes=1024,
        generation_seconds=2.0,
        request_wall_seconds=2.25,
        time_to_first_token_seconds=0.5,
        selected_pages=12 if pages else 0,
        saved_pages=2 if pages else 0,
        o1_priority=1.5,
        runtime_reward=0.25,
        warm_hit=warm,
        fertig_exact=fertig,
        draft_active=draft,
        page_active=pages,
        avoidable_work_bytes={"target_fallback": 500},
    )


class InferenceActionReceiptTests(unittest.TestCase):
    def test_normal_cold_request_becomes_one_content_addressed_action_vector(self) -> None:
        receipt = InferenceActionReceipt.from_economics(_economics("one"))
        self.assertEqual(
            receipt.actions,
            ("dynamic_mlp_pages", "qwen_target", "target_verified_draft"),
        )
        self.assertEqual(receipt.target_forwards, 3)
        self.assertEqual(receipt.saved_qwen_forwards, 1)
        self.assertEqual(
            InferenceActionReceipt.from_document(receipt.to_document()),
            receipt,
        )
        self.assertNotIn("question:one", json.dumps(receipt.to_document()))

    def test_legacy_draft_savings_flag_cannot_impersonate_a_zero_forward_hit(
        self,
    ) -> None:
        economics = _economics("legacy-cold", warm=True, target=3, saved=1)
        receipt = InferenceActionReceipt.from_economics(economics)
        self.assertEqual(
            receipt.actions,
            ("dynamic_mlp_pages", "qwen_target", "target_verified_draft"),
        )
        self.assertEqual(receipt.target_forwards, 3)
        self.assertEqual(receipt.accepted_draft_tokens, 1)
        self.assertEqual(receipt.source_body_bytes, 100)


class InferenceActionBankTests(unittest.TestCase):
    def test_observe_restart_duplicate_and_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "actions"
            bank = InferenceActionBank(root)
            first = bank.observe(_economics("cold"))
            duplicate = bank.observe(_economics("cold"))
            warm = bank.observe(
                _economics(
                    "warm",
                    warm=True,
                    target=0,
                    saved=4,
                )
            )
            snapshot = bank.snapshot()
            restarted = InferenceActionBank(root)
            restarted_snapshot = restarted.snapshot()

        self.assertFalse(first.duplicate)
        self.assertTrue(duplicate.duplicate)
        self.assertFalse(warm.duplicate)
        self.assertEqual(snapshot["schema"], INFERENCE_ACTION_BANK_SCHEMA)
        self.assertEqual(snapshot["requests"], 2)
        self.assertEqual(len(snapshot["signatures"]), 2)
        self.assertEqual(snapshot["signatures"][0]["actions"], ["stored_result"])
        self.assertEqual(snapshot["signatures"][0]["saved_qwen_forwards"], 4)
        self.assertEqual(snapshot["signatures"][0]["accepted_draft_tokens"], 0)
        self.assertEqual(snapshot["signatures"][0]["source_body_bytes"], 0)
        self.assertEqual(restarted_snapshot, snapshot)

    def test_reconcile_recovers_a_missed_derived_event_and_tamper_is_hard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = InferenceActionBank(Path(temporary) / "actions")
            snapshot = bank.reconcile((_economics("one"), _economics("two")))
            self.assertEqual(snapshot["requests"], 2)
            event = next(bank.events.iterdir())
            payload = bytearray(event.read_bytes())
            payload[-1] ^= 1
            event.chmod(0o600)
            event.write_bytes(payload)
            with self.assertRaises(InferenceActionBankError):
                bank.snapshot()


if __name__ == "__main__":
    unittest.main()
