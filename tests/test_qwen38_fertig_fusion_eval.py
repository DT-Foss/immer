from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen38_fertig_fusion_eval.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("qwen38_fertig_fusion_eval", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import Qwen/FERTIG fusion script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fusion = _load_script()


EXACT = (
    "Mira had some marbles. Mira received 11 more marbles and used 4 marbles. "
    "Mira now has 29 marbles. How many marbles did Mira have at first?"
)


def _draft(item_id: str, question: str, answer: str, gold: str) -> dict:
    return {
        "item_id": item_id,
        "question": question,
        "gold": gold,
        "predicted": answer,
        "text": f"#### {answer}",
        "finish_reason": "stop",
    }


def _verified(
    item_id: str,
    question: str,
    answer: str,
    gold: str,
    *,
    full: bool,
    accepted_prefix: int,
) -> dict:
    draft_ids = [794, 220, 16]
    return {
        "item_id": item_id,
        "question": question,
        "gold": gold,
        "answer": answer,
        "text": f"#### {answer}",
        "verification": {
            "draft_token_ids": draft_ids,
            "accepted_prefix_length": accepted_prefix,
            "fully_verified": full,
            "eos_verified": True if full else None,
            "first_mismatch_index": None if full else accepted_prefix,
        },
    }


class Qwen38FertigFusionTests(unittest.TestCase):
    def test_exact_certificate_overrides_a_fully_verified_wrong_draft(self) -> None:
        drafts = {
            "schema": fusion.BASELINE_SCHEMA,
            "items": [_draft("exact", EXACT, "999", "22")],
        }
        verification = {
            "schema": fusion.VERIFICATION_SCHEMA,
            "status": "complete",
            "items": [
                _verified("exact", EXACT, "999", "22", full=True, accepted_prefix=3)
            ],
        }

        report = fusion.fuse_documents(drafts, verification)
        row = report["items"][0]
        self.assertEqual(row["decision"], "exact_ir")
        self.assertEqual(row["answer"], "22")
        self.assertTrue(row["correct"])
        self.assertTrue(row["exact"]["certificate"]["zero_residuals"])

    def test_model_agreement_quarantines_by_default_and_legacy_is_explicit(
        self,
    ) -> None:
        question = "Which unsupported symbolic answer should be surfaced?"
        drafts = {
            "schema": fusion.BASELINE_SCHEMA,
            "items": [_draft("model", question, "7", "7")],
        }
        verification = {
            "schema": fusion.VERIFICATION_SCHEMA,
            "status": "complete",
            "items": [
                _verified("model", question, "7", "7", full=True, accepted_prefix=3)
            ],
        }

        safe = fusion.fuse_documents(drafts, verification)
        row = safe["items"][0]
        self.assertEqual(row["decision"], "model_agreement_quarantine")
        self.assertIsNone(row["answer"])
        self.assertFalse(row["accepted"])
        self.assertFalse(safe["protocol"]["model_agreement_can_answer"])

        legacy = fusion.fuse_documents(
            drafts,
            verification,
            allow_model_verified=True,
        )
        row = legacy["items"][0]
        self.assertEqual(row["decision"], "model_verified")
        self.assertEqual(row["answer"], "7")
        self.assertIsNone(row["exact"]["answer"])

    def test_decision_and_answer_are_invariant_to_gold_label(self) -> None:
        question = "Which unsupported symbolic answer should be surfaced?"

        def run(gold: str) -> tuple[str, str | None]:
            drafts = {
                "schema": fusion.BASELINE_SCHEMA,
                "items": [_draft("model", question, "7", gold)],
            }
            verification = {
                "schema": fusion.VERIFICATION_SCHEMA,
                "status": "complete",
                "items": [
                    _verified(
                        "model",
                        question,
                        "7",
                        gold,
                        full=True,
                        accepted_prefix=3,
                    )
                ],
            }
            row = fusion.fuse_documents(drafts, verification)["items"][0]
            return row["decision"], row["answer"]

        self.assertEqual(run("7"), run("999"))

    def test_non_terminating_rational_answers_are_compared_exactly(self) -> None:
        question = "Which unsupported rational answer should be surfaced?"
        drafts = {
            "schema": fusion.BASELINE_SCHEMA,
            "items": [_draft("rational", question, "1/3", "1/3")],
        }
        verification = {
            "schema": fusion.VERIFICATION_SCHEMA,
            "status": "complete",
            "items": [
                _verified(
                    "rational",
                    question,
                    "1/3",
                    "1/3",
                    full=True,
                    accepted_prefix=3,
                )
            ],
        }

        row = fusion.fuse_documents(drafts, verification, allow_model_verified=True)[
            "items"
        ][0]
        self.assertEqual(row["decision"], "model_verified")
        self.assertEqual(row["answer"], "1/3")
        self.assertTrue(row["correct"])

    def test_rejection_after_and_before_prefix_have_distinct_abstentions(self) -> None:
        questions = ("Unsupported semantic branch?", "Unsupported surface branch?")
        drafts = {
            "schema": fusion.BASELINE_SCHEMA,
            "items": [
                _draft("content", questions[0], "8", "9"),
                _draft("surface", questions[1], "-1", "3"),
            ],
        }
        verification = {
            "schema": fusion.VERIFICATION_SCHEMA,
            "status": "complete",
            "items": [
                _verified(
                    "content", questions[0], "8", "9", full=False, accepted_prefix=2
                ),
                _verified(
                    "surface", questions[1], "-1", "3", full=False, accepted_prefix=1
                ),
            ],
        }

        report = fusion.fuse_documents(drafts, verification)
        self.assertEqual(
            [row["decision"] for row in report["items"]],
            ["content_rejected", "surface_quarantine"],
        )
        self.assertTrue(all(row["answer"] is None for row in report["items"]))

    def test_cli_writes_complete_result(self) -> None:
        drafts = {
            "schema": fusion.BASELINE_SCHEMA,
            "items": [_draft("model", "Unsupported?", "7", "7")],
        }
        verification = {
            "schema": fusion.VERIFICATION_SCHEMA,
            "status": "complete",
            "items": [
                _verified(
                    "model", "Unsupported?", "7", "7", full=True, accepted_prefix=3
                )
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            drafts_path = root / "drafts.json"
            verification_path = root / "verification.json"
            output = root / "result.json"
            drafts_path.write_text(json.dumps(drafts), encoding="utf-8")
            verification_path.write_text(json.dumps(verification), encoding="utf-8")
            self.assertEqual(
                fusion.main(
                    [
                        "--drafts",
                        str(drafts_path),
                        "--verification",
                        str(verification_path),
                        "--output",
                        str(output),
                    ]
                ),
                0,
            )
            result = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["protocol"]["gold_used_for_decisions"])
        self.assertFalse(result["protocol"]["model_verified_is_exact"])
        self.assertFalse(result["protocol"]["model_agreement_can_answer"])
        self.assertEqual(result["summary"]["wrong"], 0)
        self.assertEqual(result["summary"]["answered"], 0)


if __name__ == "__main__":
    unittest.main()
