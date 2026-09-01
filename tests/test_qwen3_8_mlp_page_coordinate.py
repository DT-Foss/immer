from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from immer.runtimes.qwen3_8.mlp_page_coordinate import (
    MLP_PAGE_COORDINATE_ENVELOPE_SCHEMA,
    MlpPageCoordinateBank,
    MlpPageCoordinateIdentity,
    MlpPageCoordinateIdentityError,
    MlpPageCoordinateIntegrityError,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _identity(runtime: str = "runtime") -> MlpPageCoordinateIdentity:
    return MlpPageCoordinateIdentity(
        runtime_math_sha256=_hash(runtime),
        q4_identity_sha256=_hash("q4-static"),
        page_router_identity_sha256=_hash("page-router-static"),
    )


def _row(position: int, width: int = 8) -> torch.Tensor:
    return (
        torch.arange(width, dtype=torch.float32)
        .add_(position * width)
        .to(dtype=torch.bfloat16)
        .reshape(1, 1, width)
    )


class MlpPageCoordinateBankTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "mlp-coordinates.json"
        self.identity = _identity()

    def _bank(self, *, max_cells: int = 16) -> MlpPageCoordinateBank:
        return MlpPageCoordinateBank(
            self.path,
            self.identity,
            max_cells=max_cells,
            max_state_bytes=1024 * 1024,
        )

    def _stage(
        self,
        bank: MlpPageCoordinateBank,
        position: int,
        *,
        layer: int = 2,
        page_ids: tuple[int, ...] = (7, 2),
        logical_bytes: int = 4096,
        teacher: bool = True,
    ):
        key = bank.make_key(layer, position, _row(position))
        return bank.stage(
            key,
            page_ids,
            selected_width=len(page_ids),
            ranked_page_scores=(9.0, 4.0) if teacher else None,
            total_energy=16.0 if teacher else None,
            logical_page_weight_bytes_saved=logical_bytes,
        )

    @staticmethod
    def _commit_capture(bank: MlpPageCoordinateBank, staged) -> None:
        transaction = bank.begin_transaction()
        transaction.stage_capture(staged.absolute_position, staged)
        transaction.commit(
            accepted_end_position=staged.absolute_position + 1
        )

    def test_key_binds_all_static_identity_coordinates_and_exact_bf16_input(
        self,
    ) -> None:
        bank = self._bank()
        row = _row(3)
        key = bank.make_key(4, 3, row)

        self.assertEqual(key.runtime_math_sha256, self.identity.runtime_math_sha256)
        self.assertEqual(
            key.static_identity_sha256,
            self.identity.static_identity_sha256,
        )
        self.assertNotEqual(key.key_sha256, bank.make_key(5, 3, row).key_sha256)
        self.assertNotEqual(key.key_sha256, bank.make_key(4, 4, row).key_sha256)
        self.assertNotEqual(
            key.key_sha256,
            bank.make_key(4, 3, row.add(1)).key_sha256,
        )
        with self.assertRaisesRegex(ValueError, "row_count == 1"):
            bank.make_key(4, 3, row, row_count=2)
        with self.assertRaisesRegex(ValueError, "K1 shape"):
            bank.make_key(4, 3, row.expand(1, 2, 8), row_count=1)
        with self.assertRaisesRegex(TypeError, "bfloat16"):
            bank.make_key(4, 3, row.float())
        self.assertFalse(self.path.exists())

    def test_teacher_capture_preserves_execution_order_and_peek_is_read_only(
        self,
    ) -> None:
        bank = self._bank()
        staged = self._stage(bank, 0, page_ids=(7, 2))
        self._commit_capture(bank, staged)
        before = self.path.read_bytes()

        hit = self._bank().peek(staged.key)

        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.page_ids, (7, 2))
        self.assertEqual(hit.selected_width, 2)
        self.assertEqual(hit.ranked_page_scores, (9.0, 4.0))
        self.assertEqual(hit.total_energy, 16.0)
        self.assertEqual(hit.logical_page_weight_bytes_saved, 4096)
        self.assertTrue(hit.teacher_evidence)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(bank.metrics().hit_count, 0)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

        ranked_key = bank.make_key(2, 1, _row(1))
        ranked = bank.stage(
            ranked_key,
            (7, 2, 5),
            selected_width=2,
            ranked_page_scores=(9.0, 4.0, 1.0),
            total_energy=16.0,
            logical_page_weight_bytes_saved=4096,
        )
        self._commit_capture(bank, ranked)
        ranked_hit = bank.peek(ranked_key)
        assert ranked_hit is not None
        self.assertEqual(ranked_hit.page_ids, (7, 2, 5))
        self.assertEqual(ranked_hit.selected_page_ids, (7, 2))

    def test_transaction_filters_rejected_captures_hits_and_exact_savings(
        self,
    ) -> None:
        bank = self._bank()
        initial = [self._stage(bank, position) for position in range(3)]
        captures = bank.begin_transaction()
        for staged in initial:
            captures.stage_capture(staged.absolute_position, staged)
        captures.commit(accepted_end_position=3)

        transaction = bank.begin_transaction()
        for position, staged in enumerate(initial):
            hit = transaction.lookup(
                staged.key,
                physical_pages_saved=270 + position,
                logical_page_weight_bytes_saved=4096 + position,
            )
            self.assertIsNotNone(hit)
        accepted_capture = self._stage(bank, 3)
        rejected_capture = self._stage(bank, 4)
        transaction.stage_capture(3, accepted_capture)
        transaction.stage_capture(4, rejected_capture)

        metrics = transaction.commit(accepted_end_position=4)

        self.assertEqual(metrics.hit_count, 3)
        self.assertEqual(metrics.hits, 3)
        self.assertEqual(metrics.physical_pages_saved, 270 + 271 + 272)
        self.assertEqual(
            metrics.logical_page_weight_bytes_saved,
            4096 + 4097 + 4098,
        )
        self.assertEqual(metrics.staged_captures, 5)
        self.assertEqual(metrics.accepted_captures, 4)
        self.assertEqual(metrics.rejected_captures, 1)
        self.assertIsNotNone(bank.peek(accepted_capture.key))
        self.assertIsNone(bank.peek(rejected_capture.key))

        rejected = bank.begin_transaction()
        rejected.stage_hit(
            2,
            bank.peek(initial[2].key),
            physical_pages_saved=999,
            logical_page_weight_bytes_saved=888,
        )
        unchanged = rejected.commit(accepted_end_position=2)
        self.assertEqual(unchanged.hit_count, 3)
        self.assertEqual(unchanged.physical_pages_saved, 813)

    def test_rollback_and_failed_atomic_write_are_side_effect_free(self) -> None:
        bank = self._bank()
        original = self._stage(bank, 0)
        self._commit_capture(bank, original)
        before_bytes = self.path.read_bytes()
        before_metrics = bank.metrics()

        transaction = bank.begin_transaction()
        transaction.lookup(
            original.key,
            physical_pages_saved=270,
            logical_page_weight_bytes_saved=4096,
        )
        transaction.stage_capture(1, self._stage(bank, 1))
        transaction.rollback()
        self.assertEqual(self.path.read_bytes(), before_bytes)
        self.assertEqual(bank.metrics(), before_metrics)

        failed = bank.begin_transaction()
        failed.stage_capture(1, self._stage(bank, 1))
        with (
            patch(
                "immer.runtimes.qwen3_8.mlp_page_coordinate._atomic_write",
                side_effect=MlpPageCoordinateIntegrityError("disk failed"),
            ),
            self.assertRaisesRegex(MlpPageCoordinateIntegrityError, "disk failed"),
        ):
            failed.commit(accepted_end_position=2)
        self.assertFalse(failed.closed)
        failed.rollback()
        self.assertEqual(self.path.read_bytes(), before_bytes)
        self.assertEqual(bank.metrics(), before_metrics)

    def test_invalid_ragged_duplicate_width_and_teacher_evidence_are_rejected(
        self,
    ) -> None:
        bank = self._bank()
        key = bank.make_key(2, 0, _row(0))
        with self.assertRaisesRegex(ValueError, "not ragged"):
            bank.stage(
                key,
                ((1, 2),),
                selected_width=2,
                logical_page_weight_bytes_saved=1,
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            bank.stage(
                key,
                (1, 1),
                selected_width=2,
                logical_page_weight_bytes_saved=1,
            )
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            bank.stage(
                key,
                (1, 2),
                selected_width=3,
                logical_page_weight_bytes_saved=1,
            )
        with self.assertRaisesRegex(ValueError, "present together"):
            bank.stage(
                key,
                (1, 2),
                selected_width=2,
                ranked_page_scores=(2.0, 1.0),
                logical_page_weight_bytes_saved=1,
            )
        with self.assertRaisesRegex(ValueError, "non-increasing"):
            bank.stage(
                key,
                (1, 2),
                selected_width=2,
                ranked_page_scores=(1.0, 2.0),
                total_energy=4.0,
                logical_page_weight_bytes_saved=1,
            )
        with self.assertRaisesRegex(ValueError, "exceed total_energy"):
            bank.stage(
                key,
                (1, 2),
                selected_width=2,
                ranked_page_scores=(3.0, 2.0),
                total_energy=4.0,
                logical_page_weight_bytes_saved=1,
            )
        with self.assertRaisesRegex(ValueError, "row_count == 1"):
            bank.stage(
                key,
                (1, 2),
                selected_width=2,
                logical_page_weight_bytes_saved=1,
                row_count=3,
            )

        dense = bank.stage(
            key,
            torch.tensor([[[3, 1]]], dtype=torch.int64),
            selected_width=2,
            ranked_page_scores=torch.tensor(
                [[[2.0, 1.0]]], dtype=torch.float64
            ),
            total_energy=torch.tensor([[4.0]], dtype=torch.float64),
            logical_page_weight_bytes_saved=1,
        )
        self.assertEqual(dense.page_ids, (3, 1))
        with self.assertRaisesRegex(ValueError, "row_count == 1"):
            bank.stage(
                key,
                torch.tensor([[3, 1], [2, 0]], dtype=torch.int64),
                selected_width=2,
                logical_page_weight_bytes_saved=1,
            )

    def test_conflicting_coordinate_is_rejected_without_mutation(self) -> None:
        bank = self._bank()
        first = self._stage(bank, 0, page_ids=(7, 2))
        self._commit_capture(bank, first)
        conflict = self._stage(bank, 0, page_ids=(7, 3))
        before_bytes = self.path.read_bytes()
        before_metrics = bank.metrics()
        transaction = bank.begin_transaction()
        transaction.stage_capture(0, conflict)

        with self.assertRaisesRegex(
            MlpPageCoordinateIntegrityError,
            "conflicting coordinates",
        ):
            transaction.commit(accepted_end_position=1)

        self.assertEqual(self.path.read_bytes(), before_bytes)
        self.assertEqual(bank.metrics(), before_metrics)

    def test_value_lru_eviction_is_bounded_and_deterministic(self) -> None:
        bank = self._bank(max_cells=2)
        first = self._stage(bank, 0, logical_bytes=100)
        second = self._stage(bank, 1, logical_bytes=100)
        transaction = bank.begin_transaction()
        transaction.stage_capture(0, first)
        transaction.stage_capture(1, second)
        transaction.commit(accepted_end_position=2)
        hit = bank.begin_transaction()
        hit.lookup(
            first.key,
            physical_pages_saved=1,
            logical_page_weight_bytes_saved=1,
        )
        hit.commit(accepted_end_position=1)
        third = self._stage(bank, 2, logical_bytes=100)
        self._commit_capture(bank, third)

        metrics = bank.metrics()
        self.assertEqual(metrics.cell_count, 2)
        self.assertEqual(metrics.evictions, 1)
        self.assertIsNotNone(bank.peek(first.key))
        self.assertIsNone(bank.peek(second.key))
        self.assertIsNotNone(bank.peek(third.key))

    def test_concurrent_instances_commit_without_lost_updates(self) -> None:
        banks = [self._bank(max_cells=32) for _ in range(8)]

        def commit(index: int) -> None:
            staged = self._stage(banks[index], index)
            transaction = banks[index].begin_transaction()
            transaction.stage_capture(index, staged)
            transaction.commit(accepted_end_position=index + 1)

        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(commit, range(8)))

        metrics = self._bank(max_cells=32).metrics()
        self.assertEqual(metrics.cell_count, 8)
        self.assertEqual(metrics.accepted_captures, 8)
        self.assertEqual(metrics.transactions, 8)

    def test_identity_canonical_tamper_duplicate_and_symlink_fail_closed(
        self,
    ) -> None:
        bank = self._bank()
        staged = self._stage(bank, 0)
        self._commit_capture(bank, staged)
        with self.assertRaises(MlpPageCoordinateIdentityError):
            MlpPageCoordinateBank(
                self.path,
                _identity("changed"),
                max_cells=16,
                max_state_bytes=1024 * 1024,
            )

        document = json.loads(self.path.read_bytes())
        self.assertEqual(document["schema"], MLP_PAGE_COORDINATE_ENVELOPE_SCHEMA)
        document["body"]["cells"][0]["payload"]["page_ids"][0] = 99
        document["body_sha256"] = hashlib.sha256(
            _canonical(document["body"])
        ).hexdigest()
        self.path.write_bytes(_canonical(document))
        with self.assertRaisesRegex(
            MlpPageCoordinateIntegrityError,
            "payload SHA-256 mismatch",
        ):
            self._bank()

        self.path.write_bytes(b'{"schema":1,"schema":2}')
        with self.assertRaisesRegex(
            MlpPageCoordinateIntegrityError,
            "duplicate JSON key",
        ):
            self._bank()

        self.path.unlink()
        target = self.root / "target.json"
        target.write_bytes(b"{}")
        self.path.symlink_to(target)
        with self.assertRaises(MlpPageCoordinateIntegrityError):
            self._bank()


if __name__ == "__main__":
    unittest.main()
