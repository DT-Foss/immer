from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from immer.runtimes.qwen3_8.attention_output_crystal import (
    ATTENTION_OUTPUT_CRYSTAL_ENVELOPE_SCHEMA,
    EMPTY_ATTENTION_STATE_SHA256,
    AttentionOutputCrystalBank,
    AttentionOutputCrystalIdentity,
    AttentionOutputCrystalIdentityError,
    AttentionOutputCrystalIntegrityError,
    canonical_attention_state_sha256,
    canonical_bf16_tensor_bytes,
    canonical_bf16_tensor_sha256,
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


def _identity(runtime: str = "runtime-math") -> AttentionOutputCrystalIdentity:
    return AttentionOutputCrystalIdentity(runtime_math_sha256=_hash(runtime))


def _row(position: int, *, width: int = 8) -> torch.Tensor:
    return (
        torch.arange(width, dtype=torch.float32).add_(position * width)
        .to(dtype=torch.bfloat16)
        .reshape(1, 1, width)
    )


def _appended(position: int) -> tuple[torch.Tensor, torch.Tensor]:
    key = torch.arange(8, dtype=torch.float32).add_(position).to(
        dtype=torch.bfloat16
    )
    key = key.reshape(1, 2, 1, 4)
    return key, key.add(10)


class AttentionOutputCrystalBankTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "attention-crystals.json"
        self.identity = _identity()

    def _bank(
        self,
        *,
        max_cells: int = 16,
        max_state_bytes: int = 2 * 1024 * 1024,
    ) -> AttentionOutputCrystalBank:
        return AttentionOutputCrystalBank(
            self.path,
            self.identity,
            max_cells=max_cells,
            max_state_bytes=max_state_bytes,
        )

    def _stage(
        self,
        bank: AttentionOutputCrystalBank,
        position: int,
        *,
        layer: int = 3,
        output_offset: int = 100,
        logical_bytes: int = 4096,
        prior: str | None = None,
    ):
        row = _row(position)
        key = bank.make_key(
            layer,
            position,
            row,
            EMPTY_ATTENTION_STATE_SHA256 if prior is None else prior,
        )
        appended_key, appended_value = _appended(position)
        return bank.stage(
            key,
            post_o_proj=row.add(output_offset),
            appended_rope_key=appended_key,
            appended_value=appended_value,
            logical_projection_bytes=logical_bytes,
            skipped_projection_calls=4,
        )

    def test_canonical_bf16_and_attention_state_hashes_are_exact(self) -> None:
        row = _row(0)
        same_bytes_new_shape = row.reshape(1, 2, 4)

        self.assertEqual(
            canonical_bf16_tensor_bytes(row),
            row.view(torch.uint8).numpy().tobytes(),
        )
        self.assertEqual(
            canonical_bf16_tensor_sha256(row),
            canonical_bf16_tensor_sha256(row.clone()),
        )
        self.assertNotEqual(
            canonical_bf16_tensor_sha256(row),
            canonical_bf16_tensor_sha256(same_bytes_new_shape),
        )
        changed = row.clone()
        changed[0, 0, 0] = 0.5
        self.assertNotEqual(
            canonical_bf16_tensor_sha256(row),
            canonical_bf16_tensor_sha256(changed),
        )

        key, value = _appended(0)
        usage = torch.tensor(
            [[[-float("inf")], [0.0], [1.0], [2.0]]],
            dtype=torch.float32,
        )
        state = SimpleNamespace(key=key, value=value, crsa_log_usage=usage)
        clone = SimpleNamespace(
            key=key.clone(), value=value.clone(), crsa_log_usage=usage.clone()
        )
        self.assertEqual(
            canonical_attention_state_sha256(state),
            canonical_attention_state_sha256(clone),
        )
        clone.value[0, 0, 0, 0] += 1
        self.assertNotEqual(
            canonical_attention_state_sha256(state),
            canonical_attention_state_sha256(clone),
        )
        self.assertEqual(
            EMPTY_ATTENTION_STATE_SHA256,
            canonical_attention_state_sha256(None),
        )

    def test_key_binds_runtime_layer_position_input_and_prior_state(self) -> None:
        bank = self._bank()
        row = _row(2)
        key = bank.make_key(7, 2, row, _hash("prior"))

        self.assertEqual(key.runtime_math_sha256, self.identity.runtime_math_sha256)
        self.assertNotEqual(
            key.key_sha256,
            bank.make_key(8, 2, row, _hash("prior")).key_sha256,
        )
        self.assertNotEqual(
            key.key_sha256,
            bank.make_key(7, 3, row, _hash("prior")).key_sha256,
        )
        self.assertNotEqual(
            key.key_sha256,
            bank.make_key(7, 2, row.add(1), _hash("prior")).key_sha256,
        )
        self.assertNotEqual(
            key.key_sha256,
            bank.make_key(7, 2, row, _hash("other-prior")).key_sha256,
        )
        other = AttentionOutputCrystalBank(
            self.root / "other.json",
            _identity("different-math"),
            max_cells=16,
            max_state_bytes=2 * 1024 * 1024,
        )
        with self.assertRaises(AttentionOutputCrystalIdentityError):
            other.lookup(key)
        self.assertFalse(self.path.exists())

    def test_staging_detaches_and_publish_keeps_only_accepted_prefix(self) -> None:
        bank = self._bank()
        staged = [self._stage(bank, position) for position in range(3)]
        expected_first = _row(0).add(100)
        source = _row(0)
        detached = bank.stage(
            bank.make_key(11, 10, source, _hash("prior-10")),
            post_o_proj=source,
            appended_rope_key=_appended(10)[0],
            appended_value=_appended(10)[1],
            logical_projection_bytes=2048,
        )
        source.fill_(999)

        metrics = bank.publish(staged, accepted_rows=2)
        restored = self._bank()

        self.assertEqual(metrics.cell_count, 2)
        self.assertEqual(metrics.staged_captures, 3)
        self.assertEqual(metrics.accepted_captures, 2)
        self.assertEqual(metrics.rejected_captures, 1)
        first = restored.lookup(staged[0].key)
        second = restored.lookup(staged[1].key)
        rejected = restored.lookup(staged[2].key)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertIsNone(rejected)
        assert first is not None
        self.assertTrue(torch.equal(first.post_o_proj, expected_first))
        self.assertTrue(torch.equal(first.appended_rope_key, _appended(0)[0]))
        self.assertTrue(torch.equal(first.appended_value, _appended(0)[1]))
        self.assertEqual(first.post_o_proj.dtype, torch.bfloat16)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

        bank.publish((detached,), accepted_rows=1)
        hit = bank.lookup(detached.key)
        assert hit is not None
        self.assertFalse(torch.equal(hit.post_o_proj, source))
        self.assertTrue(torch.equal(hit.post_o_proj, _row(0)))

    def test_layer27_crsa_usage_and_evidence_round_trip_exactly(self) -> None:
        bank = self._bank()
        position = 2
        row = _row(position)
        key = bank.make_key(27, position, row, _hash("native-prior"))
        usage = torch.tensor(
            [
                [
                    [-float("inf"), -1.0, 0.0],
                    [-float("inf"), -2.0, 0.25],
                    [-float("inf"), -3.0, 0.5],
                    [-float("inf"), -4.0, 0.75],
                ]
            ],
            dtype=torch.float32,
        )
        evidence = {
            "history_length_after": 3,
            "history_length_before": 2,
            "layer": 27,
            "query_length": 1,
            "query_start": 2,
            "schema": "immer.qwen3.8-native-head-crsa/v1",
        }
        appended_key, appended_value = _appended(position)
        staged = bank.stage(
            key,
            post_o_proj=row.add(7),
            appended_rope_key=appended_key,
            appended_value=appended_value,
            next_crsa_usage=usage,
            crsa_evidence=evidence,
            logical_projection_bytes=8192,
        )
        usage.fill_(123)
        evidence["layer"] = 1
        bank.publish((staged,), accepted_rows=1)

        hit = self._bank().lookup(key)
        assert hit is not None
        self.assertTrue(torch.isneginf(hit.next_crsa_usage[0, :, 0]).all())
        self.assertEqual(hit.next_crsa_usage.dtype, torch.float32)
        self.assertEqual(tuple(hit.next_crsa_usage.shape), (1, 4, 3))
        self.assertEqual(hit.crsa_evidence["layer"], 27)
        self.assertEqual(hit.crsa_evidence["query_start"], 2)
        self.assertEqual(hit.crsa_evidence["history_length_after"], 3)

        wrong_layer = bank.make_key(26, position, row, _hash("native-prior"))
        with self.assertRaisesRegex(ValueError, "only for layer 27"):
            bank.stage(
                wrong_layer,
                post_o_proj=row,
                appended_rope_key=appended_key,
                appended_value=appended_value,
                next_crsa_usage=torch.zeros(1, 4, 3, dtype=torch.float32),
                logical_projection_bytes=1,
            )

    def test_transaction_filters_speculative_captures_hits_and_savings(self) -> None:
        bank = self._bank()
        initial = [self._stage(bank, position) for position in range(3)]
        bank.publish(initial, accepted_rows=3)
        transaction = bank.begin_transaction()
        for position, staged in enumerate(initial):
            hit = transaction.lookup(staged.key)
            self.assertIsNotNone(hit)
            if position == 1:
                transaction.stage_savings(
                    position,
                    skipped_projection_calls=4,
                    logical_projection_bytes_saved=16_384,
                )
            if position == 2:
                transaction.stage_savings(
                    4,
                    skipped_projection_calls=4,
                    logical_projection_bytes_saved=99_999,
                )
        accepted_capture = self._stage(bank, 3)
        rejected_capture = self._stage(bank, 4)
        transaction.stage_capture(3, accepted_capture)
        transaction.stage_capture(4, rejected_capture)

        metrics = transaction.commit(accepted_end_position=4)

        self.assertEqual(metrics.hit_count, 3)
        self.assertEqual(metrics.skipped_projection_calls_saved, 4)
        self.assertEqual(metrics.logical_projection_bytes_saved, 16_384)
        self.assertEqual(metrics.staged_captures, 5)
        self.assertEqual(metrics.accepted_captures, 4)
        self.assertEqual(metrics.rejected_captures, 1)
        self.assertIsNotNone(bank.lookup(accepted_capture.key))
        self.assertIsNone(bank.lookup(rejected_capture.key))
        self.assertEqual(bank.lookup(initial[0].key).cell_hit_count, 1)
        self.assertEqual(bank.lookup(initial[1].key).cell_hit_count, 1)
        self.assertEqual(bank.lookup(initial[2].key).cell_hit_count, 1)

        rejected = bank.begin_transaction()
        rejected.stage_hit(
            2,
            bank.lookup(initial[2].key),
            skipped_projection_calls=4,
            logical_projection_bytes_saved=777,
        )
        unchanged = rejected.commit(accepted_end_position=2)
        self.assertEqual(unchanged.hit_count, 3)
        self.assertEqual(unchanged.skipped_projection_calls_saved, 4)
        self.assertEqual(unchanged.logical_projection_bytes_saved, 16_384)

    def test_transaction_rollback_is_side_effect_free(self) -> None:
        bank = self._bank()
        staged = self._stage(bank, 0)
        bank.publish((staged,), accepted_rows=1)
        before_bytes = self.path.read_bytes()
        before_metrics = bank.metrics()

        transaction = bank.begin_transaction()
        transaction.lookup(
            staged.key,
            skipped_projection_calls=4,
            logical_projection_bytes_saved=4096,
        )
        transaction.stage_capture(1, self._stage(bank, 1))
        transaction.rollback()

        self.assertEqual(self.path.read_bytes(), before_bytes)
        self.assertEqual(bank.metrics(), before_metrics)
        with self.assertRaisesRegex(Exception, "closed"):
            transaction.commit(accepted_end_position=2)

    def test_value_then_lru_eviction_is_deterministic_and_bounded(self) -> None:
        bank = self._bank(max_cells=2)
        first = self._stage(bank, 0, logical_bytes=100)
        second = self._stage(bank, 1, logical_bytes=100)
        bank.publish((first, second), accepted_rows=2)
        transaction = bank.begin_transaction()
        transaction.lookup(first.key)
        transaction.commit(accepted_end_position=1)
        third = self._stage(bank, 2, logical_bytes=100)

        metrics = bank.publish((third,), accepted_rows=1)

        self.assertEqual(metrics.cell_count, 2)
        self.assertEqual(metrics.evictions, 1)
        self.assertIsNotNone(bank.lookup(first.key))
        self.assertIsNone(bank.lookup(second.key))
        self.assertIsNotNone(bank.lookup(third.key))

    def test_concurrent_instances_publish_without_lost_updates(self) -> None:
        banks = [self._bank(max_cells=32) for _ in range(8)]

        def commit(index: int) -> None:
            staged = self._stage(banks[index], index)
            banks[index].publish((staged,), accepted_rows=1)

        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(commit, range(8)))

        restored = self._bank(max_cells=32)
        metrics = restored.metrics()
        self.assertEqual(metrics.cell_count, 8)
        self.assertEqual(metrics.accepted_captures, 8)
        self.assertEqual(metrics.publish_transactions, 8)

    def test_conflicting_exact_transition_and_failed_write_leave_state_unchanged(
        self,
    ) -> None:
        bank = self._bank()
        original = self._stage(bank, 0)
        bank.publish((original,), accepted_rows=1)
        conflicting = self._stage(bank, 0, output_offset=200)
        before_bytes = self.path.read_bytes()
        before_metrics = bank.metrics()

        with self.assertRaisesRegex(
            AttentionOutputCrystalIntegrityError,
            "conflicting payloads",
        ):
            bank.publish((conflicting,), accepted_rows=1)
        self.assertEqual(self.path.read_bytes(), before_bytes)
        self.assertEqual(bank.metrics(), before_metrics)

        new = self._stage(bank, 1)
        with (
            patch(
                "immer.runtimes.qwen3_8.attention_output_crystal._atomic_write",
                side_effect=AttentionOutputCrystalIntegrityError("disk failed"),
            ),
            self.assertRaisesRegex(AttentionOutputCrystalIntegrityError, "disk failed"),
        ):
            bank.publish((new,), accepted_rows=1)
        self.assertEqual(self.path.read_bytes(), before_bytes)
        self.assertEqual(bank.metrics(), before_metrics)

    def test_identity_capacity_canonical_tamper_and_symlink_checks_fail_closed(
        self,
    ) -> None:
        bank = self._bank()
        staged = self._stage(bank, 0)
        bank.publish((staged,), accepted_rows=1)

        with self.assertRaises(AttentionOutputCrystalIdentityError):
            AttentionOutputCrystalBank(
                self.path,
                _identity("changed"),
                max_cells=16,
                max_state_bytes=2 * 1024 * 1024,
            )
        with self.assertRaises(AttentionOutputCrystalIdentityError):
            AttentionOutputCrystalBank(
                self.path,
                self.identity,
                max_cells=15,
                max_state_bytes=2 * 1024 * 1024,
            )

        document = json.loads(self.path.read_bytes())
        self.assertEqual(document["schema"], ATTENTION_OUTPUT_CRYSTAL_ENVELOPE_SCHEMA)
        tensor = document["body"]["cells"][0]["payload"]["post_o_proj"]
        raw = bytearray(base64.b64decode(tensor["data_base64"]))
        raw[0] ^= 1
        tensor["data_base64"] = base64.b64encode(raw).decode("ascii")
        document["body_sha256"] = hashlib.sha256(
            _canonical(document["body"])
        ).hexdigest()
        self.path.write_bytes(_canonical(document))
        with self.assertRaisesRegex(
            AttentionOutputCrystalIntegrityError,
            "data SHA-256 mismatch",
        ):
            self._bank()

        self.path.unlink()
        bank = self._bank()
        bank.publish((self._stage(bank, 0),), accepted_rows=1)
        canonical = self.path.read_bytes()
        self.path.write_bytes(
            json.dumps(json.loads(canonical), indent=2, sort_keys=True).encode()
        )
        with self.assertRaisesRegex(
            AttentionOutputCrystalIntegrityError,
            "not canonical JSON",
        ):
            self._bank()

        self.path.unlink()
        target = self.root / "target.json"
        target.write_bytes(b"{}")
        self.path.symlink_to(target)
        with self.assertRaises(AttentionOutputCrystalIntegrityError):
            self._bank()

    def test_invalid_tensor_shapes_dtypes_crsa_and_boundaries_are_rejected(
        self,
    ) -> None:
        bank = self._bank()
        with self.assertRaisesRegex(TypeError, "bfloat16"):
            bank.make_key(1, 0, torch.ones(1, 1, 8), _hash("prior"))
        with self.assertRaisesRegex(ValueError, "exact shape"):
            bank.make_key(
                1,
                0,
                torch.ones(1, 8, dtype=torch.bfloat16),
                _hash("prior"),
            )
        row = _row(0)
        key = bank.make_key(1, 0, row, _hash("prior"))
        with self.assertRaisesRegex(ValueError, "appended key/value"):
            bank.stage(
                key,
                post_o_proj=row,
                appended_rope_key=torch.ones(1, 2, 1, 4, dtype=torch.bfloat16),
                appended_value=torch.ones(1, 2, 1, 5, dtype=torch.bfloat16),
                logical_projection_bytes=1,
            )
        native = bank.make_key(27, 3, row, _hash("prior"))
        with self.assertRaisesRegex(ValueError, "history"):
            bank.stage(
                native,
                post_o_proj=row,
                appended_rope_key=_appended(0)[0],
                appended_value=_appended(0)[1],
                next_crsa_usage=torch.zeros(1, 4, 3, dtype=torch.float32),
                logical_projection_bytes=1,
            )
        staged = self._stage(bank, 0)
        with self.assertRaisesRegex(ValueError, "accepted_rows"):
            bank.publish((staged,), accepted_rows=2)
        with self.assertRaisesRegex(ValueError, "increasing row positions"):
            bank.publish((self._stage(bank, 1), staged), accepted_rows=1)


if __name__ == "__main__":
    unittest.main()
