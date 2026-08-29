from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from immer.runtimes.ooe.crystal import CrystalStore
from immer.runtimes.ooe.qwen_warm_templates import ParametricWarmBank


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class ParametricWarmBankTests(unittest.TestCase):
    def test_two_distinct_slots_promote_and_unseen_slot_executes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = _hash("profile")

            def token(question: str) -> str:
                return _hash(f"tokens:{question}")

            bank = ParametricWarmBank(
                CrystalStore(root),
                profile,
                output_character_limit=32,
                prompt_token_verifier=(
                    lambda question, claimed: claimed == token(question)
                ),
            )
            gamma = "Schreibe exakt: GAMMA"
            metadata = {
                "qwen_token_sha256": token(gamma),
                "qwen_warm_runtime_profile_sha256": profile,
            }

            first = bank.observe(
                "Schreibe exakt: ALPHA",
                "ALPHA",
                question_sha256=_hash("question-alpha"),
                cell_payload_sha256=_hash("cell-alpha"),
                teacher_forward_count=4,
            )
            self.assertEqual(first["promoted"], 0)
            self.assertIsNone(bank.try_warm(gamma, metadata))

            second = bank.observe(
                "Schreibe exakt: BETA",
                "BETA",
                question_sha256=_hash("question-beta"),
                cell_payload_sha256=_hash("cell-beta"),
                teacher_forward_count=3,
            )
            self.assertGreaterEqual(second["promoted"], 1)
            attempt = bank.try_warm(gamma, metadata)
            self.assertIsNotNone(attempt)
            assert attempt is not None and attempt.result is not None
            self.assertEqual(attempt.result.output, "GAMMA")
            self.assertTrue(attempt._abstention_authorized)
            receipt = attempt._settler(True)
            self.assertEqual(receipt.disposition, "committed")
            self.assertEqual(receipt.saved_qwen_forwards, 3)

            restarted = ParametricWarmBank(
                CrystalStore(root),
                profile,
                output_character_limit=32,
                prompt_token_verifier=(
                    lambda question, claimed: claimed == token(question)
                ),
            )
            delta = "Schreibe exakt: DELTA"
            replay = restarted.try_warm(
                delta,
                {
                    "qwen_token_sha256": token(delta),
                    "qwen_warm_runtime_profile_sha256": profile,
                },
            )
            self.assertIsNotNone(replay)
            assert replay is not None and replay.result is not None
            self.assertEqual(replay.result.output, "DELTA")
            self.assertEqual(restarted.body["committed_executions"], 1)
            self.assertEqual(restarted.body["saved_qwen_forwards"], 3)

    def test_ambiguous_transforms_wrong_profile_and_missing_token_abstain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile = _hash("profile")

            def token(question: str) -> str:
                return _hash(f"tokens:{question}")

            bank = ParametricWarmBank(
                CrystalStore(temporary),
                profile,
                output_character_limit=32,
                prompt_token_verifier=(
                    lambda question, claimed: claimed == token(question)
                ),
            )
            for slot in ("ALPHA", "BETA"):
                bank.observe(
                    f"copy:{slot}",
                    slot,
                    question_sha256=_hash(f"question:{slot}"),
                    cell_payload_sha256=_hash(f"cell:{slot}"),
                    teacher_forward_count=2,
                )
            gamma = "copy:gamma"
            correct = {
                "qwen_token_sha256": token(gamma),
                "qwen_warm_runtime_profile_sha256": profile,
            }
            self.assertIsNone(bank.try_warm(gamma, correct))
            self.assertIsNone(
                bank.try_warm(
                    "copy:GAMMA",
                    {
                        **correct,
                        "qwen_token_sha256": token("copy:GAMMA"),
                        "qwen_warm_runtime_profile_sha256": _hash("other"),
                    },
                )
            )
            self.assertIsNone(
                bank.try_warm(
                    "copy:GAMMA",
                    {
                        "qwen_token_sha256": _hash("wrong-token-binding"),
                        "qwen_warm_runtime_profile_sha256": profile,
                    },
                )
            )
            self.assertIsNone(
                bank.try_warm(
                    "copy:GAMMA",
                    {"qwen_warm_runtime_profile_sha256": profile},
                )
            )

    def test_repeated_same_slot_does_not_promote(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile = _hash("profile")
            bank = ParametricWarmBank(
                CrystalStore(temporary),
                profile,
                output_character_limit=32,
            )
            for index in range(3):
                bank.observe(
                    "prefix VALUE",
                    "VALUE",
                    question_sha256=_hash(f"question:{index}"),
                    cell_payload_sha256=_hash(f"cell:{index}"),
                    teacher_forward_count=2,
                )
            self.assertEqual(bank.promoted, ())

    def test_unseen_output_cannot_exceed_the_bound_runtime_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile = _hash("profile")

            def token(question: str) -> str:
                return _hash(f"tokens:{question}")

            bank = ParametricWarmBank(
                CrystalStore(temporary),
                profile,
                output_character_limit=3,
                prompt_token_verifier=(
                    lambda question, claimed: claimed == token(question)
                ),
            )
            for slot in ("A", "BB"):
                bank.observe(
                    f"copy:{slot}",
                    slot,
                    question_sha256=_hash(f"question:{slot}"),
                    cell_payload_sha256=_hash(f"cell:{slot}"),
                    teacher_forward_count=3,
                )
            xyz = "copy:XYZ"
            metadata = {
                "qwen_token_sha256": token(xyz),
                "qwen_warm_runtime_profile_sha256": profile,
            }
            allowed = bank.try_warm(xyz, metadata)
            self.assertIsNotNone(allowed)
            too_long = "copy:TOOLONG"
            self.assertIsNone(
                bank.try_warm(
                    too_long,
                    {
                        "qwen_token_sha256": token(too_long),
                        "qwen_warm_runtime_profile_sha256": profile,
                    },
                )
            )


if __name__ == "__main__":
    unittest.main()
