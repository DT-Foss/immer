from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from immer.runtimes.qwen3_8.contextual_continuation import (
    CONTEXTUAL_KEY_DIMENSIONS,
    ContextualCandidateFeedback,
    ContextualCapture,
    ContextualContinuationBank,
    ContextualContinuationIdentity,
    ContextualContinuationIdentityError,
    ContextualContinuationIntegrityError,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _identity(*, runtime: str = "runtime", width: int = 12):
    return ContextualContinuationIdentity(
        runtime_sha256=_hash(runtime),
        model_sha256=_hash("model"),
        q4_sha256=_hash("q4"),
        tokenizer_sha256=_hash("tokenizer"),
        hidden_width=width,
    )


def _hidden(width: int, offset: float = 0.0) -> torch.Tensor:
    return torch.arange(1, width + 1, dtype=torch.float32).reshape(1, 1, width) + offset


class ContextualContinuationBankTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "continuations.json"
        self.identity = _identity()

    def test_projection_is_deterministic_normalized_and_identity_bound(self) -> None:
        bank = ContextualContinuationBank(self.path, self.identity, max_cells=8)
        hidden = _hidden(self.identity.hidden_width)

        first = bank.project(hidden, 41)
        second = bank.project(hidden.clone(), 41)
        other_token = bank.project(hidden, 42)

        self.assertEqual(first, second)
        self.assertEqual(len(first.q8), CONTEXTUAL_KEY_DIMENSIONS)
        self.assertGreaterEqual(first.norm_sq, 112**2)
        self.assertLessEqual(first.norm_sq, 144**2)
        self.assertEqual(first.q8, other_token.q8)
        self.assertNotEqual(first.key_sha256, other_token.key_sha256)
        self.assertFalse(self.path.exists())

        other = ContextualContinuationBank(
            self.root / "other.json",
            _identity(runtime="other"),
            max_cells=8,
        )
        with self.assertRaises(ContextualContinuationIdentityError):
            other.query_key(first)

    def test_capture_persists_only_q8_boundary_and_confirmed_target_tail(self) -> None:
        bank = ContextualContinuationBank(self.path, self.identity, max_cells=8)
        hidden = _hidden(self.identity.hidden_width)
        capture = bank.make_capture(hidden, 7, 1234, (11, 12, 13))

        metrics = bank.settle(captures=(capture,))
        restored = ContextualContinuationBank(
            self.path,
            self.identity,
            max_cells=8,
        )
        candidates = restored.query(hidden, 7)

        self.assertEqual(metrics.cell_count, 1)
        self.assertEqual(metrics.support, 1)
        self.assertEqual(candidates[0].target_tail, (11, 12, 13))
        self.assertAlmostEqual(candidates[0].cosine, 1.0)
        encoded = self.path.read_bytes()
        self.assertNotIn(b"1234", encoded)
        self.assertNotIn(hidden.numpy().tobytes(), encoded)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(
            ContextualContinuationBank.read_identity(self.path),
            self.identity,
        )

    def test_feedback_updates_exact_per_position_counters_atomically(self) -> None:
        bank = ContextualContinuationBank(self.path, self.identity, max_cells=8)
        hidden = _hidden(self.identity.hidden_width)
        capture = bank.make_capture(hidden, 7, 0, (11, 12, 13, 14))
        bank.settle(captures=(capture, capture))
        candidate = bank.query(hidden, 7)[0]

        metrics = bank.settle(
            feedback=(
                ContextualCandidateFeedback(candidate.cell_sha256, 2, 4),
                ContextualCandidateFeedback(candidate.cell_sha256, 1, 3),
            )
        )
        updated = bank.query(hidden, 7)[0]

        self.assertEqual(updated.support, 2)
        self.assertEqual(updated.position_verified, (2, 2, 2, 1))
        self.assertEqual(updated.position_hits, (2, 1, 0, 0))
        self.assertEqual(metrics.feedback_count, 2)
        self.assertEqual(metrics.verified_positions, 7)
        self.assertEqual(metrics.hit_positions, 3)

    def test_query_collapses_equal_tails_before_runner_up_and_margin(self) -> None:
        bank = ContextualContinuationBank(self.path, self.identity, max_cells=8)
        exact = bank.make_capture(_hidden(self.identity.hidden_width), 9, 0, (100, 101))
        duplicate = bank.make_capture(
            -_hidden(self.identity.hidden_width), 9, 1, (100, 101)
        )
        runner_up = bank.make_capture(
            _hidden(self.identity.hidden_width, 100.0), 9, 2, (200,)
        )
        wrong_known = bank.make_capture(
            _hidden(self.identity.hidden_width), 10, 3, (300,)
        )
        bank.settle(captures=(exact, duplicate, runner_up, wrong_known))

        before = self.path.read_bytes()
        candidates = bank.query(_hidden(self.identity.hidden_width), 9)
        after = self.path.read_bytes()

        self.assertEqual(before, after)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0].target_tail, (100, 101))
        self.assertEqual(candidates[0].collapsed_cells, 2)
        self.assertEqual(candidates[0].support, 1)
        self.assertEqual(candidates[0].runner_up_cosine, candidates[1].cosine)
        self.assertAlmostEqual(
            candidates[0].margin,
            candidates[0].cosine - candidates[1].cosine,
        )
        self.assertIsNone(candidates[1].runner_up_cosine)
        self.assertIsNone(candidates[1].margin)

    def test_equal_tail_feedback_stays_with_the_nearest_hidden_cell(self) -> None:
        bank = ContextualContinuationBank(self.path, self.identity, max_cells=8)
        near = bank.make_capture(_hidden(self.identity.hidden_width), 9, 0, (100, 101))
        far = bank.make_capture(-_hidden(self.identity.hidden_width), 9, 1, (100, 101))
        bank.settle(captures=(near, far))
        far_candidate = bank.query_key(far.key)[0]
        bank.settle(
            feedback=(ContextualCandidateFeedback(far_candidate.cell_sha256, 0, 2),)
        )

        near_candidate = bank.query_key(near.key)[0]
        far_candidate = bank.query_key(far.key)[0]

        self.assertEqual(near_candidate.collapsed_cells, 2)
        self.assertEqual(near_candidate.position_verified, (0, 0))
        self.assertEqual(near_candidate.position_hits, (0, 0))
        self.assertEqual(far_candidate.position_verified, (1, 1))
        self.assertEqual(far_candidate.position_hits, (0, 0))

    def test_deterministic_value_then_lru_eviction_is_bounded(self) -> None:
        bank = ContextualContinuationBank(self.path, self.identity, max_cells=2)
        first = bank.make_capture(_hidden(12), 1, 0, (10, 11, 12))
        second = bank.make_capture(_hidden(12, 10.0), 1, 1, (20,))
        bank.settle(captures=(first, second))
        first_candidate = next(
            item
            for item in bank.query_key(first.key)
            if item.target_tail == first.target_tail
        )
        bank.settle(
            feedback=(ContextualCandidateFeedback(first_candidate.cell_sha256, 3, 3),)
        )
        third = bank.make_capture(_hidden(12, -20.0), 1, 2, (30, 31))
        metrics = bank.settle(captures=(third,))

        tails = {candidate.target_tail for candidate in bank.query_key(first.key)}
        self.assertEqual(metrics.cell_count, 2)
        self.assertEqual(metrics.evictions, 1)
        self.assertIn(first.target_tail, tails)
        self.assertIn(third.target_tail, tails)
        self.assertNotIn(second.target_tail, tails)

    def test_concurrent_instances_commit_without_lost_updates(self) -> None:
        banks = [
            ContextualContinuationBank(self.path, self.identity, max_cells=32)
            for _ in range(8)
        ]

        def commit(index: int) -> None:
            banks[index].settle(
                captures=(
                    banks[index].make_capture(
                        _hidden(12, float(index)),
                        index,
                        index,
                        (1000 + index,),
                    ),
                )
            )

        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(commit, range(8)))

        restored = ContextualContinuationBank(self.path, self.identity, max_cells=32)
        metrics = restored.metrics()
        self.assertEqual(metrics.cell_count, 8)
        self.assertEqual(metrics.capture_count, 8)
        self.assertEqual(metrics.settlements, 8)

    def test_identity_capacity_hash_canonical_and_symlink_checks_fail_closed(
        self,
    ) -> None:
        bank = ContextualContinuationBank(self.path, self.identity, max_cells=8)
        capture = bank.make_capture(_hidden(12), 1, 0, (2, 3))
        bank.settle(captures=(capture,))

        with self.assertRaises(ContextualContinuationIdentityError):
            ContextualContinuationBank(
                self.path, _identity(runtime="changed"), max_cells=8
            )
        with self.assertRaises(ContextualContinuationIdentityError):
            ContextualContinuationBank(self.path, self.identity, max_cells=7)

        document = json.loads(self.path.read_bytes())
        document["body"]["clock"] += 1
        self.path.write_bytes(
            json.dumps(document, separators=(",", ":"), sort_keys=True).encode()
        )
        with self.assertRaisesRegex(
            ContextualContinuationIntegrityError, "SHA-256 mismatch"
        ):
            ContextualContinuationBank(self.path, self.identity, max_cells=8)

        self.path.unlink()
        target = self.root / "target.json"
        target.write_bytes(b"{}")
        self.path.symlink_to(target)
        with self.assertRaises(ContextualContinuationIntegrityError):
            ContextualContinuationBank(self.path, self.identity, max_cells=8)

    def test_noncanonical_and_duplicate_json_are_rejected(self) -> None:
        bank = ContextualContinuationBank(self.path, self.identity, max_cells=8)
        bank.settle(captures=(bank.make_capture(_hidden(12), 1, 0, (2,)),))
        document = json.loads(self.path.read_bytes())
        self.path.write_bytes(json.dumps(document, indent=2, sort_keys=True).encode())
        with self.assertRaisesRegex(
            ContextualContinuationIntegrityError, "not canonical JSON"
        ):
            ContextualContinuationBank(self.path, self.identity, max_cells=8)

        self.path.write_bytes(b'{"schema":1,"schema":2}')
        with self.assertRaisesRegex(
            ContextualContinuationIntegrityError, "duplicate JSON key"
        ):
            ContextualContinuationBank(self.path, self.identity, max_cells=8)

    def test_failed_atomic_publication_leaves_memory_and_disk_unchanged(self) -> None:
        bank = ContextualContinuationBank(self.path, self.identity, max_cells=8)
        first = bank.make_capture(_hidden(12), 1, 0, (2,))
        bank.settle(captures=(first,))
        before_bytes = self.path.read_bytes()
        before_metrics = bank.metrics()
        second = bank.make_capture(_hidden(12, 5.0), 1, 1, (3,))

        with (
            patch(
                "immer.runtimes.qwen3_8.contextual_continuation._atomic_write",
                side_effect=ContextualContinuationIntegrityError("disk failed"),
            ),
            self.assertRaisesRegex(ContextualContinuationIntegrityError, "disk failed"),
        ):
            bank.settle(captures=(second,))

        self.assertEqual(self.path.read_bytes(), before_bytes)
        self.assertEqual(bank.metrics(), before_metrics)

    def test_invalid_hidden_tail_feedback_and_unknown_cell_are_rejected(self) -> None:
        bank = ContextualContinuationBank(self.path, self.identity, max_cells=8)
        with self.assertRaisesRegex(ValueError, "exact shape"):
            bank.project(torch.ones(1, 12), 1)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            bad = _hidden(12)
            bad[0, 0, 0] = float("nan")
            bank.project(bad, 1)
        with self.assertRaisesRegex(ValueError, "1..15"):
            ContextualCapture(bank.project(_hidden(12), 1), (), 0)
        with self.assertRaisesRegex(ValueError, "exceeds verified"):
            ContextualCandidateFeedback("a" * 64, 2, 1)
        with self.assertRaisesRegex(
            ContextualContinuationIntegrityError, "unknown contextual cell"
        ):
            bank.settle(feedback=(ContextualCandidateFeedback("a" * 64, 0, 1),))
        self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main()
