from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen38_fertig_arm_compare.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("qwen38_fertig_arm_compare", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import Qwen FERTIG arm comparator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


compare = _load_script()


def _item(item_id: str, *, correct: bool, full: bool, prefix: int) -> dict:
    draft_ids = [1, 2, 3]
    return {
        "item_id": item_id,
        "question": f"question {item_id}",
        "gold": "7" if correct else "9",
        "text": "#### 7",
        "answer": "7",
        "candidate_correct": correct,
        "candidate_status": "correct" if correct else "incorrect",
        "finish_reason": "stop",
        "prompt_token_ids": [10, 11],
        "draft_token_ids": draft_ids,
        "verification": {
            "draft_token_ids": draft_ids,
            "draft_verified": full,
            "eos_verified": True if full else None,
            "fully_verified": full,
            "accepted_prefix_length": prefix,
        },
    }


def _arm(mode: str, rows: list[dict], *, seconds: float) -> dict:
    graft = mode == "stable-crsa"
    content_verified = sum(row["verification"]["draft_verified"] for row in rows)
    eos_verified = sum(row["verification"]["eos_verified"] is True for row in rows)
    fully_verified = sum(row["verification"]["fully_verified"] for row in rows)
    candidate_correct = sum(row["candidate_correct"] is True for row in rows)
    verified_correct = sum(
        row["candidate_correct"] is True and row["verification"]["fully_verified"]
        for row in rows
    )
    return {
        "schema": compare.VERIFICATION_SCHEMA,
        "status": "complete",
        "source": {
            "checkpoint": "Qwen/Qwen3.8-27B",
            "revision": "a" * 40,
            "verification": {
                "kind": "complete-causal-bundle/v1",
                "checkpoint_bytes": 100,
                "graph_revision": [1, "b" * 64],
                "layout_fingerprint": "c" * 64,
                "manifest_sha256": "d" * 64,
                "shards": 2,
                "shards_sha256": "e" * 64,
                "tensor_bindings": 10,
                "weights_layout": "flat/v1",
                "seconds": 3.0,
            },
        },
        "protocol": {
            "mode": mode,
            "graft_layer": 27 if graft else None,
            "graft_alpha": 0.01 if graft else None,
            "batch_size": len(rows),
            "padding": "right",
            "padding_token_id": 8,
            "eos_token_id": 9,
            "accepted_eos_token_ids": [9, 8],
            "weight_order": "layer-major, one BF16 matrix at a time",
            "dynamic_cohort": True,
            "item_ids": [row["item_id"] for row in rows],
        },
        "evidence": {
            "batch_size": len(rows),
            "draft_lengths": [3] * len(rows),
            "eos_token_id": 9,
            "fixed_model_batch": True,
            "padded_sequence_length": 5,
            "prompt_lengths": [2] * len(rows),
            "right_prefix_mask": True,
            "verification_rows": 8,
            "seconds": seconds,
            "source_body_bytes": 1000,
            "layer_calls": 64,
            "layer_retry_count": 0,
            "head_scans": 1,
            "head_retry_count": 0,
        },
        "summary": {
            "total": len(rows),
            "candidate_correct": candidate_correct,
            "candidate_incomplete": 0,
            "content_verified": content_verified,
            "eos_verified": eos_verified,
            "fully_verified": fully_verified,
            "verified_correct": verified_correct,
        },
        "items": rows,
    }


class Qwen38FertigArmCompareTests(unittest.TestCase):
    def test_wrong_draft_promotion_is_an_unsafe_regression(self) -> None:
        off = _arm(
            "off",
            [
                _item("correct", correct=True, full=True, prefix=3),
                _item("wrong", correct=False, full=False, prefix=1),
            ],
            seconds=10,
        )
        candidate = _arm(
            "stable-crsa",
            [
                _item("correct", correct=True, full=True, prefix=3),
                _item("wrong", correct=False, full=True, prefix=3),
            ],
            seconds=12,
        )

        result = compare.compare_arms(off, candidate)

        self.assertEqual(result["comparison"]["wrong_agreement_promotions"], 1)
        self.assertEqual(result["comparison"]["correct_promotions"], 0)
        self.assertEqual(result["comparison"]["verdict"], "unsafe_regression")
        self.assertEqual(result["comparison"]["accepted_prefix_delta"], 2)
        self.assertAlmostEqual(
            result["comparison"]["model_speed_ratio_off_over_candidate"],
            10 / 12,
        )
        unsealed = dict(result)
        observed = unsealed.pop("report_sha256")
        self.assertEqual(observed, compare._canonical_digest(unsealed))

    def test_identity_drift_fails_before_comparison(self) -> None:
        off = _arm("off", [_item("one", correct=True, full=True, prefix=3)], seconds=1)
        candidate = _arm(
            "stable-crsa",
            [_item("one", correct=True, full=True, prefix=3)],
            seconds=1,
        )
        candidate["items"][0]["prompt_token_ids"] = [99]
        with self.assertRaisesRegex(compare.CliError, "identity differs"):
            compare.compare_arms(off, candidate)

        candidate = _arm(
            "stable-crsa",
            [_item("one", correct=True, full=True, prefix=3)],
            seconds=1,
        )
        candidate["source"]["verification"]["manifest_sha256"] = "f" * 64
        with self.assertRaisesRegex(compare.CliError, "identity differs"):
            compare.compare_arms(off, candidate)

        candidate = _arm(
            "stable-crsa",
            [_item("one", correct=True, full=True, prefix=3)],
            seconds=1,
        )
        candidate["summary"]["fully_verified"] = 0
        with self.assertRaisesRegex(compare.CliError, "summary field"):
            compare.compare_arms(off, candidate)

    def test_cli_writes_a_sealed_atomic_report(self) -> None:
        off = _arm("off", [_item("one", correct=True, full=True, prefix=3)], seconds=1)
        candidate = _arm(
            "stable-crsa",
            [_item("one", correct=True, full=True, prefix=3)],
            seconds=1.1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            off_path = root / "off.json"
            candidate_path = root / "candidate.json"
            output = root / "nested" / "result.json"
            off_path.write_text(json.dumps(off), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

            code = compare.main(
                [
                    "--off",
                    str(off_path),
                    "--candidate",
                    str(candidate_path),
                    "--output",
                    str(output),
                ]
            )

            self.assertEqual(code, 0)
            document = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(document["comparison"]["verdict"], "neutral")
            self.assertTrue(document["report_sha256"])
            self.assertEqual(list(output.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
