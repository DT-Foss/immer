from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "deepseek_v4_fertig_fusion_eval.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "deepseek_v4_fertig_fusion_eval", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import DeepSeek/FERTIG fusion script")
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
    content = f" The answer is {answer}."
    return {
        "item_id": item_id,
        "question": question,
        "gold": gold,
        "content": content,
        "token_ids": [10, 20, 30],
        "logprob_tokens": [content],
        "logprob_token_count": 3,
        "response_model": fusion.API_MODEL,
        "system_fingerprint": "fp_test",
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4},
    }


def _verified(
    draft: dict,
    *,
    full: bool,
    draft_verified: bool | None = None,
    eos_verified: bool | None = None,
) -> dict:
    if draft_verified is None:
        draft_verified = full
    if eos_verified is None and full:
        eos_verified = True
    token_count = len(draft["token_ids"])
    accepted = token_count if draft_verified else 1
    mismatch = None if full else (token_count if draft_verified else accepted)
    answer = fusion.extract_gsm8k_answer("Answer:" + draft["content"])
    return {
        "item_id": draft["item_id"],
        "gold": draft["gold"],
        "api_content": "Answer:" + draft["content"],
        "api_response_model": draft["response_model"],
        "api_system_fingerprint": draft["system_fingerprint"],
        "api_answer": answer,
        "api_correct": answer == draft["gold"],
        "locally_fully_verified_content": (
            "Answer:" + draft["content"] if full else None
        ),
        "local_answer": answer if full else None,
        "local_correct": full and answer == draft["gold"],
        "content_verified": draft_verified,
        "eos_verified": eos_verified if draft_verified else None,
        "fully_verified": full,
        "verification": {
            "draft_token_ids": draft["token_ids"],
            "accepted_prefix_length": accepted,
            "first_mismatch_index": mismatch,
            "first_mismatch_target_token_id": None if full else 999,
            "first_mismatch_draft_token_id": None if full else 998,
            "first_mismatch_is_eos": bool(draft_verified and not full),
            "draft_verified": draft_verified,
            "eos_verified": eos_verified if draft_verified else None,
            "fully_verified": full,
        },
    }


def _documents(*pairs: tuple[dict, dict], mode: str = "off") -> tuple[dict, dict, dict]:
    drafts = {
        "schema": fusion.DRAFT_SCHEMA,
        "protocol": {
            "model": fusion.API_MODEL,
            "assistant_prefix": "Answer:",
            "thinking": {"type": "disabled"},
            "temperature": 0,
            "max_tokens": fusion.DRAFT_MAX_TOKENS,
            "logprobs": True,
        },
        "items": [pair[0] for pair in pairs],
    }
    rows = [pair[1] for pair in pairs]
    verification = {
        "schema": fusion.VERIFICATION_SCHEMA,
        "status": "complete",
        "protocol": {
            "checkpoint": fusion.OFFICIAL_SOURCE,
            "revision": fusion.OFFICIAL_REVISION,
            "api_model": fusion.API_MODEL,
            "mode": mode,
            "graft_layer": 21 if mode == "stable-crsa" else None,
            "graft_alpha": 0.01 if mode == "stable-crsa" else None,
        },
        "summary": {
            "total": len(rows),
            "content_verified": sum(row["content_verified"] for row in rows),
            "eos_verified": sum(row["eos_verified"] is True for row in rows),
            "fully_verified": sum(row["fully_verified"] for row in rows),
        },
        "items": rows,
    }
    benchmark_items = [
        {
            "item_id": draft["item_id"],
            "question": draft["question"],
            "gold": draft["gold"],
        }
        for draft in drafts["items"]
    ]
    benchmark = {
        "schema": fusion.BENCHMARK_SCHEMA,
        "items_sha256": fusion._framed_digest(benchmark_items),
        "items": benchmark_items,
    }
    return drafts, verification, benchmark


class DeepSeekV4FertigFusionTests(unittest.TestCase):
    def test_mixed_gain_loss_average_is_quarantined_before_model(self) -> None:
        question = (
            "On the first race, Lee lost $5. On the second race, Lee won $11. "
            "On the third race, Lee lost $16.50. "
            "How much did Lee lose on average that day?"
        )

        def run(gold: str) -> tuple[str, str | None, str | None]:
            draft = _draft("ambiguous", question, "3.5", gold)
            row = fusion.fuse_documents(
                *_documents((draft, _verified(draft, full=True)))
            )["items"][0]
            return row["decision"], row["answer"], row["semantic"]["reason"]

        first = run("3")
        self.assertEqual(first[0], "semantic_ambiguity")
        self.assertIsNone(first[1])
        self.assertIn("under-specified", first[2])
        self.assertEqual(first, run("3.5"))

    def test_unambiguous_average_is_not_quarantined(self) -> None:
        question = (
            "Lee scored 6 points in one game and 8 points in another. "
            "What was Lee's average score?"
        )
        draft = _draft("average", question, "7", "7")
        row = fusion.fuse_documents(*_documents((draft, _verified(draft, full=True))))[
            "items"
        ][0]
        self.assertEqual(row["decision"], "model_verified")
        self.assertFalse(row["semantic"]["ambiguous"])

    def test_exact_certificate_overrides_fully_verified_wrong_draft(self) -> None:
        draft = _draft("exact", EXACT, "999", "22")
        report = fusion.fuse_documents(
            *_documents((draft, _verified(draft, full=True)))
        )
        row = report["items"][0]
        self.assertEqual(row["decision"], "exact_ir")
        self.assertEqual(row["answer"], "22")
        self.assertTrue(row["correct"])
        self.assertTrue(row["exact"]["certificate"]["zero_residuals"])

    def test_fully_verified_model_answer_is_not_labeled_exact(self) -> None:
        draft = _draft("model", "Which unsupported result is requested?", "7", "7")
        row = fusion.fuse_documents(*_documents((draft, _verified(draft, full=True))))[
            "items"
        ][0]
        self.assertEqual(row["decision"], "model_verified")
        self.assertEqual(row["answer"], "7")
        self.assertIsNone(row["exact"]["answer"])

    def test_decision_is_invariant_to_gold(self) -> None:
        def run(gold: str) -> tuple[str, str | None]:
            draft = _draft("model", "Which unsupported result is requested?", "7", gold)
            row = fusion.fuse_documents(
                *_documents((draft, _verified(draft, full=True)))
            )["items"][0]
            return row["decision"], row["answer"]

        self.assertEqual(run("7"), run("999"))

    def test_draft_and_eos_rejections_are_distinct_abstentions(self) -> None:
        draft_rejected = _draft("draft", "Unsupported draft?", "7", "7")
        eos_rejected = _draft("eos", "Unsupported termination?", "8", "8")
        report = fusion.fuse_documents(
            *_documents(
                (
                    draft_rejected,
                    _verified(
                        draft_rejected,
                        full=False,
                        draft_verified=False,
                        eos_verified=None,
                    ),
                ),
                (
                    eos_rejected,
                    _verified(
                        eos_rejected,
                        full=False,
                        draft_verified=True,
                        eos_verified=False,
                    ),
                ),
            )
        )
        self.assertEqual(
            [row["decision"] for row in report["items"]],
            ["draft_rejected", "termination_rejected"],
        )
        self.assertTrue(all(row["answer"] is None for row in report["items"]))

    def test_tampered_source_or_summary_fails_closed(self) -> None:
        draft = _draft("model", "Unsupported?", "7", "7")
        drafts, verification, benchmark = _documents(
            (draft, _verified(draft, full=True))
        )
        verification["protocol"]["revision"] = "mutable"
        with self.assertRaisesRegex(fusion.CliError, "pinned source"):
            fusion.fuse_documents(drafts, verification, benchmark)

        drafts, verification, benchmark = _documents(
            (draft, _verified(draft, full=True))
        )
        verification["summary"]["fully_verified"] = 0
        with self.assertRaisesRegex(fusion.CliError, "summary"):
            fusion.fuse_documents(drafts, verification, benchmark)

        drafts, verification, benchmark = _documents(
            (draft, _verified(draft, full=True))
        )
        verification["items"][0]["content_verified"] = False
        verification["summary"]["content_verified"] = 0
        with self.assertRaisesRegex(fusion.CliError, "top-level"):
            fusion.fuse_documents(drafts, verification, benchmark)

    def test_draft_protocol_and_canonical_question_fail_closed(self) -> None:
        draft = _draft("model", "Unsupported?", "7", "7")
        drafts, verification, benchmark = _documents(
            (draft, _verified(draft, full=True))
        )
        drafts["protocol"]["assistant_prefix"] = "Solution:"
        with self.assertRaisesRegex(fusion.CliError, "pinned contract"):
            fusion.fuse_documents(drafts, verification, benchmark)

        drafts, verification, benchmark = _documents(
            (draft, _verified(draft, full=True))
        )
        benchmark["items"][0]["question"] = "A different problem"
        benchmark["items_sha256"] = fusion._framed_digest(benchmark["items"])
        with self.assertRaisesRegex(fusion.CliError, "canonical question"):
            fusion.fuse_documents(drafts, verification, benchmark)

        drafts, verification, benchmark = _documents(
            (draft, _verified(draft, full=True))
        )
        benchmark["items_sha256"] = "0" * 64
        with self.assertRaisesRegex(fusion.CliError, "digest"):
            fusion.fuse_documents(drafts, verification, benchmark)

    def test_stable_graft_parameters_are_validated_and_preserved(self) -> None:
        draft = _draft("model", "Unsupported?", "7", "7")
        drafts, verification, benchmark = _documents(
            (draft, _verified(draft, full=True)), mode="stable-crsa"
        )
        report = fusion.fuse_documents(drafts, verification, benchmark)
        self.assertEqual(report["source"]["mode"], "stable-crsa")
        self.assertEqual(report["source"]["graft_layer"], 21)
        self.assertEqual(report["source"]["graft_alpha"], 0.01)

        verification["protocol"]["graft_alpha"] = None
        with self.assertRaisesRegex(fusion.CliError, "graft parameters"):
            fusion.fuse_documents(drafts, verification, benchmark)

    def test_cli_writes_complete_result(self) -> None:
        draft = _draft("model", "Unsupported?", "7", "7")
        drafts, verification, benchmark = _documents(
            (draft, _verified(draft, full=True))
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            drafts_path = root / "drafts.json"
            verification_path = root / "verification.json"
            benchmark_path = root / "benchmark.json"
            output = root / "result.json"
            drafts_path.write_text(json.dumps(drafts), encoding="utf-8")
            verification_path.write_text(json.dumps(verification), encoding="utf-8")
            benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
            self.assertEqual(
                fusion.main(
                    [
                        "--drafts",
                        str(drafts_path),
                        "--verification",
                        str(verification_path),
                        "--benchmark",
                        str(benchmark_path),
                        "--output",
                        str(output),
                    ]
                ),
                0,
            )
            result = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["protocol"]["gold_used_for_decisions"])
        self.assertTrue(result["protocol"]["requires_complete_draft_and_eos"])


if __name__ == "__main__":
    unittest.main()
