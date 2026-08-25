from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from safetensors.torch import save as save_safetensors
import torch

from immer.runtimes.deepseek_v4 import DeepSeekWeightPager, StreamedDeepSeekV4
from immer.runtimes.deepseek_v4.semantic_state_cache import (
    SEMANTIC_ANCHOR_INDEX_SCHEMA,
    SEMANTIC_ANCHOR_SEED_SCHEMA,
    SemanticStateAnchorCache,
    SemanticStateCacheConflict,
    SemanticStateCacheError,
    semantic_label_sha256,
    token_prefix_sha256,
)
from immer.runtimes.deepseek_v4.snapshot import DeepSeekV4SnapshotError

from test_deepseek_v4_model import _CompressedTinyCheckpoint, _config


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


class _DifferentSource(_CompressedTinyCheckpoint):
    def metrics(self) -> dict:
        result = super().metrics()
        result["inventory_source_fingerprint"] = "different-semantic-cache-fixture"
        return result


class _CrashAfterNativeSave:
    def __init__(self, model: StreamedDeepSeekV4) -> None:
        self.model = model

    def save_state(self, *args, **kwargs):
        self.model.save_state(*args, **kwargs)
        raise RuntimeError("simulated crash after native snapshot")


class DeepSeekV4SemanticStateCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix=".deepseek-v4-semantic-cache-test-", dir=Path.cwd()
        )
        self.root = Path(self.temporary.name)
        self.pagers: list[DeepSeekWeightPager] = []

    def tearDown(self) -> None:
        for pager in reversed(self.pagers):
            pager.close()
        self.temporary.cleanup()

    def _model(
        self,
        *,
        source=None,
        max_batch_size: int = 1,
    ) -> StreamedDeepSeekV4:
        checkpoint = source or _CompressedTinyCheckpoint(
            random_weights=True, compress_ratio=4
        )
        pager = DeepSeekWeightPager(
            checkpoint,
            device="cpu",
            compute_dtype="float32",
        )
        self.pagers.append(pager)
        return StreamedDeepSeekV4(
            _config(4),
            pager,
            max_batch_size=max_batch_size,
            max_seq_len=64,
        )

    def _charged(
        self,
        cache: SemanticStateAnchorCache,
        tokens: list[int],
        *,
        boundary: str = "turn",
    ):
        model = self._model()
        hidden, _ = model.prefill([tokens], tokenwise=False)
        return cache.store(
            model,
            tokens,
            boundary_kind=boundary,
            seed_hidden=hidden[:, -1:],
            semantic_label_sha256=semantic_label_sha256(f"fixture:{len(tokens)}"),
        )

    def test_prefix_index_is_deepseek_sealed_and_contains_no_raw_tokens(self) -> None:
        expected = hashlib.sha256(
            b'{"schema":"immer.deepseek-v4-token-prefix/v1","token_ids":[1,4,9]}'
        ).hexdigest()
        self.assertEqual(token_prefix_sha256([1, 4, 9]), expected)
        self.assertEqual(token_prefix_sha256((1, 4, 9)), expected)
        with self.assertRaises(ValueError):
            token_prefix_sha256([True])

        cache = SemanticStateAnchorCache(self.root / "cache")
        receipt = self._charged(cache, [1, 4, 9], boundary="tool-output")
        document = json.loads(cache.index_path.read_text(encoding="utf-8"))
        index_text = cache.index_path.read_text(encoding="utf-8")
        self.assertEqual(document["body"]["schema"], SEMANTIC_ANCHOR_INDEX_SCHEMA)
        self.assertNotIn("qwen", index_text.lower())
        self.assertNotIn("token_ids", index_text)
        self.assertNotIn("fixture:", index_text)
        self.assertNotIn("[1,4,9]", index_text)
        self.assertIn(receipt.prefix_sha256, index_text)

    def test_deepest_prefix_restores_arbitrary_unknown_suffix_bit_exactly(
        self,
    ) -> None:
        cache = SemanticStateAnchorCache(self.root / "cache")
        short = self._charged(cache, [3, 10], boundary="turn")
        deep = self._charged(cache, [3, 10, 17, 24], boundary="thinking")
        suffix = [31, 38, 45]

        uninterrupted = self._model()
        uninterrupted.prefill([[3, 10, 17, 24]], tokenwise=False)
        expected, expected_evidence = uninterrupted.prefill(
            [suffix], tokenwise=True, reset=False
        )

        restored = self._model()
        hit = cache.restore_deepest(restored, [3, 10, 17, 24, *suffix])
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.anchor.prefix_sha256, deep.prefix_sha256)
        self.assertFalse(hit.exact_prefix)
        self.assertIsNone(hit.seed_hidden)
        actual, actual_evidence = restored.prefill(
            [suffix], tokenwise=True, reset=False
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertEqual(
            [(row.start_pos, row.end_pos) for row in actual_evidence],
            [(4, 5), (5, 6), (6, 7)],
        )
        self.assertEqual(
            [(row.start_pos, row.end_pos) for row in actual_evidence],
            [(row.start_pos, row.end_pos) for row in expected_evidence],
        )
        expected_state = uninterrupted.save_state(self.root / "expected.json")
        actual_state = restored.save_state(self.root / "actual.json")
        self.assertEqual(
            actual_state["payload_sha256"], expected_state["payload_sha256"]
        )
        self.assertEqual(actual_state["next_position"], expected_state["next_position"])

        shallow_target = self._model()
        shallow = cache.restore_deepest(shallow_target, [3, 10, 99])
        self.assertIsNotNone(shallow)
        assert shallow is not None
        self.assertEqual(shallow.anchor.prefix_sha256, short.prefix_sha256)
        self.assertEqual(shallow_target.next_position, 2)

    def test_exact_prefix_seed_drives_head_without_replaying_final_token(self) -> None:
        cache = SemanticStateAnchorCache(self.root / "cache")
        prefix = [3, 10, 17, 24]
        source = self._model()
        hidden, _ = source.prefill([prefix], tokenwise=False)
        expected_values, expected_ids = source.pager.topk_logits(hidden[:, -1], k=1)
        anchor = cache.store(
            source,
            prefix,
            boundary_kind="turn",
            seed_hidden=hidden[:, -1:],
        )

        target = self._model()
        restored = cache.restore_deepest(target, prefix)
        self.assertIsNotNone(restored)
        assert restored is not None and restored.seed_hidden is not None
        self.assertTrue(restored.exact_prefix)
        self.assertEqual(restored.anchor.prefix_sha256, anchor.prefix_sha256)
        self.assertEqual(
            restored.anchor.snapshot_final_hidden_sha256,
            source.last_hidden_sha256,
        )
        self.assertEqual(target.last_hidden_sha256, source.last_hidden_sha256)
        self.assertTrue(torch.equal(restored.seed_hidden, hidden[:, -1:]))
        actual_values, actual_ids = target.pager.topk_logits(
            restored.seed_hidden[:, -1], k=1
        )
        self.assertTrue(torch.equal(actual_ids, expected_ids))
        self.assertTrue(torch.equal(actual_values, expected_values))
        self.assertEqual(target.next_position, len(prefix))

    def test_store_proves_batch_one_and_seed_width_from_native_snapshot(self) -> None:
        cache = SemanticStateAnchorCache(self.root / "batch-cache")
        batch_model = self._model(max_batch_size=2)
        batch_model.prefill([[1, 4, 9], [1, 4, 9]], tokenwise=False)
        with self.assertRaisesRegex(SemanticStateCacheError, "batch size one"):
            cache.store(batch_model, [1, 4, 9], boundary_kind="turn")
        self.assertEqual(cache.receipts(), ())

        width_cache = SemanticStateAnchorCache(self.root / "width-cache")
        width_model = self._model()
        width_model.prefill([[1, 4, 9]], tokenwise=False)
        wrong_width = torch.zeros((1, 1, width_model.config.dim + 1))
        with self.assertRaisesRegex(SemanticStateCacheError, "snapshot identity"):
            width_cache.store(
                width_model,
                [1, 4, 9],
                boundary_kind="turn",
                seed_hidden=wrong_width,
            )
        self.assertEqual(width_cache.receipts(), ())

        semantic_cache = SemanticStateAnchorCache(self.root / "semantic-cache")
        semantic_model = self._model()
        hidden, _ = semantic_model.prefill([[1, 4, 9]], tokenwise=False)
        wrong_hidden = hidden[:, -1:].clone()
        wrong_hidden[..., 0] += 1.0
        with self.assertRaisesRegex(SemanticStateCacheError, "final-hidden digest"):
            semantic_cache.store(
                semantic_model,
                [1, 4, 9],
                boundary_kind="turn",
                seed_hidden=wrong_hidden,
            )
        self.assertEqual(semantic_cache.receipts(), ())

    def test_native_identity_mismatch_is_rejected_without_hit(self) -> None:
        cache = SemanticStateAnchorCache(self.root / "cache")
        anchor = self._charged(cache, [1, 4, 9])
        incompatible = self._model(
            source=_DifferentSource(random_weights=True, compress_ratio=4)
        )
        with self.assertRaisesRegex(DeepSeekV4SnapshotError, "identity mismatch"):
            cache.restore_deepest(incompatible, [1, 4, 9, 7])
        self.assertEqual(incompatible.next_position, 0)
        current = cache.receipts()[0]
        self.assertEqual(current.receipt_sha256, anchor.receipt_sha256)
        self.assertEqual(current.hit_count, 0)

    def test_manifest_payload_seed_index_and_symlink_tamper_fail_closed(self) -> None:
        manifest_cache = SemanticStateAnchorCache(self.root / "manifest-cache")
        manifest_anchor = self._charged(manifest_cache, [1, 4, 9])
        manifest = manifest_cache.snapshots / manifest_anchor.snapshot_manifest_name
        manifest.write_bytes(manifest.read_bytes() + b" ")
        with self.assertRaisesRegex(
            SemanticStateCacheError, "byte count mismatch|exceeds its byte limit"
        ):
            manifest_cache.lookup_deepest([1, 4, 9])

        payload_cache = SemanticStateAnchorCache(self.root / "payload-cache")
        payload_anchor = self._charged(payload_cache, [1, 4, 9])
        payload = payload_cache.snapshots / payload_anchor.snapshot_payload_name
        with payload.open("r+b") as stream:
            stream.seek(payload.stat().st_size // 2)
            original = stream.read(1)
            stream.seek(-1, os.SEEK_CUR)
            stream.write(bytes([original[0] ^ 1]))
        with self.assertRaisesRegex(SemanticStateCacheError, "payload SHA-256"):
            payload_cache.lookup_deepest([1, 4, 9])
        self.assertEqual(payload_cache.receipts()[0].hit_count, 0)

        seed_cache = SemanticStateAnchorCache(self.root / "seed-cache")
        seed_anchor = self._charged(seed_cache, [1, 4, 9])
        assert seed_anchor.seed_hidden_name is not None
        seed = seed_cache.snapshots / seed_anchor.seed_hidden_name
        with seed.open("r+b") as stream:
            stream.seek(seed.stat().st_size // 2)
            original = stream.read(1)
            stream.seek(-1, os.SEEK_CUR)
            stream.write(bytes([original[0] ^ 1]))
        with self.assertRaisesRegex(SemanticStateCacheError, "seed hidden SHA-256"):
            seed_cache.lookup_deepest([1, 4, 9])
        self.assertEqual(seed_cache.receipts()[0].hit_count, 0)

        symlink_root = self.root / "cache-link"
        os.symlink(seed_cache.root, symlink_root)
        with self.assertRaisesRegex(SemanticStateCacheError, "non-symlink"):
            SemanticStateAnchorCache(symlink_root)

        document = json.loads(seed_cache.index_path.read_text(encoding="utf-8"))
        row = document["body"]["anchors"][0]
        row["snapshot_manifest_name"] = "../escape.json"
        receipt_body = dict(row)
        receipt_body.pop("receipt_sha256")
        row["receipt_sha256"] = hashlib.sha256(
            _canonical_json(receipt_body)
        ).hexdigest()
        document["body_sha256"] = hashlib.sha256(
            _canonical_json(document["body"])
        ).hexdigest()
        seed_cache.index_path.write_bytes(_canonical_json(document) + b"\n")
        with self.assertRaisesRegex(SemanticStateCacheError, "prefix-bound"):
            SemanticStateAnchorCache(seed_cache.root)

    def test_corrupt_unused_exact_seed_does_not_block_suffix_restore(self) -> None:
        cache = SemanticStateAnchorCache(self.root / "cache")
        anchor = self._charged(cache, [1, 4, 9])
        assert anchor.seed_hidden_name is not None
        seed = cache.snapshots / anchor.seed_hidden_name
        with seed.open("r+b") as stream:
            stream.seek(seed.stat().st_size // 2)
            original = stream.read(1)
            stream.seek(-1, os.SEEK_CUR)
            stream.write(bytes([original[0] ^ 1]))

        uninterrupted = self._model()
        uninterrupted.prefill([[1, 4, 9]], tokenwise=False)
        expected, _ = uninterrupted.prefill([[7, 8]], tokenwise=True, reset=False)
        suffix_target = self._model()
        restored = cache.restore_deepest(suffix_target, [1, 4, 9, 7, 8])
        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertIsNone(restored.seed_hidden)
        actual, _ = suffix_target.prefill([[7, 8]], tokenwise=True, reset=False)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

        exact_target = self._model()
        with self.assertRaisesRegex(SemanticStateCacheError, "seed hidden SHA-256"):
            cache.restore_deepest(exact_target, [1, 4, 9])
        self.assertEqual(exact_target.next_position, 0)
        self.assertEqual(exact_target.attention_state_bytes, 0)
        self.assertEqual(cache.receipts()[0].hit_count, 1)

    def test_same_identity_path_swap_and_hit_failure_clear_loaded_state(self) -> None:
        cache = SemanticStateAnchorCache(self.root / "swap-cache")
        self._charged(cache, [1, 4, 9])
        alternate = self._model()
        alternate.prefill([[2, 3, 8]], tokenwise=False)
        alternate_path = self.root / "alternate.json"
        alternate.save_state(alternate_path)

        target = self._model()
        native_load = target.load_state

        def swapped_load(_path, **kwargs):
            return native_load(alternate_path, **kwargs)

        with mock.patch.object(target, "load_state", side_effect=swapped_load):
            with self.assertRaisesRegex(
                SemanticStateCacheError,
                "receipt differs from anchor|paths differ from anchor",
            ):
                cache.restore_deepest(target, [1, 4, 9, 7])
        self.assertEqual(target.next_position, 0)
        self.assertEqual(target.attention_state_bytes, 0)
        self.assertEqual(cache.receipts()[0].hit_count, 0)

        hit_cache = SemanticStateAnchorCache(self.root / "hit-cache")
        self._charged(hit_cache, [1, 4, 9])
        hit_target = self._model()
        with mock.patch.object(
            hit_cache, "_commit_hit", side_effect=RuntimeError("hit commit failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "hit commit failed"):
                hit_cache.restore_deepest(hit_target, [1, 4, 9, 7])
        self.assertEqual(hit_target.next_position, 0)
        self.assertEqual(hit_target.attention_state_bytes, 0)
        self.assertEqual(hit_cache.receipts()[0].hit_count, 0)

    def test_restore_rejects_nonempty_or_unreleased_target_without_mutation(
        self,
    ) -> None:
        cache = SemanticStateAnchorCache(self.root / "cache")
        self._charged(cache, [1, 4, 9])

        target = self._model()
        target.prefill([[11, 13]], tokenwise=False)
        prior = target.save_state(self.root / "prior.json")
        with self.assertRaisesRegex(SemanticStateCacheError, "empty.*released"):
            cache.restore_deepest(target, [1, 4, 9, 7])
        self.assertEqual(target.next_position, 2)
        after = target.save_state(self.root / "after.json")
        self.assertEqual(after["payload_sha256"], prior["payload_sha256"])
        self.assertEqual(after["next_position"], prior["next_position"])
        self.assertEqual(cache.receipts()[0].hit_count, 0)

        target.reset_state(release=False)
        self.assertEqual(target.next_position, 0)
        self.assertGreater(target.attention_state_bytes, 0)
        with self.assertRaisesRegex(SemanticStateCacheError, "empty.*released"):
            cache.restore_deepest(target, [1, 4, 9, 7])
        target.reset_state(release=True)
        self.assertEqual(target.attention_state_bytes, 0)
        self.assertIsNotNone(cache.restore_deepest(target, [1, 4, 9, 7]))

    def test_durable_hits_drive_deterministic_bounded_lru(self) -> None:
        sizing = SemanticStateAnchorCache(self.root / "sizing")
        sizes = {
            tuple(tokens): self._charged(sizing, tokens).cache_bytes
            for tokens in ([1], [1, 4], [1, 4, 9])
        }
        budget = sizes[(1,)] + sizes[(1, 4, 9)]
        cache = SemanticStateAnchorCache(self.root / "cache", max_bytes=budget)
        first = self._charged(cache, [1])
        second = self._charged(cache, [1, 4])
        touched = cache.lookup_deepest([1, 7])
        self.assertIsNotNone(touched)
        third = self._charged(cache, [1, 4, 9])

        retained = {row.prefix_sha256 for row in cache.receipts()}
        self.assertEqual(retained, {first.prefix_sha256, third.prefix_sha256})
        self.assertNotIn(
            second.snapshot_manifest_name,
            {path.name for path in cache.snapshots.iterdir()},
        )
        self.assertLessEqual(cache.total_bytes, budget)

        reopened = SemanticStateAnchorCache(cache.root, max_bytes=budget)
        self.assertEqual(reopened.receipts(), cache.receipts())

    def test_snapshot_first_crash_is_invisible_until_explicit_orphan_gc(self) -> None:
        cache = SemanticStateAnchorCache(self.root / "cache")
        model = self._model()
        model.prefill([[1, 4, 9]], tokenwise=False)
        with self.assertRaisesRegex(RuntimeError, "simulated crash"):
            cache.store(
                _CrashAfterNativeSave(model),
                [1, 4, 9],
                boundary_kind="custom",
            )
        self.assertEqual(cache.receipts(), ())
        self.assertEqual(len(tuple(cache.snapshots.glob("*.json"))), 1)
        self.assertEqual(len(tuple(cache.snapshots.glob("*.npz"))), 1)
        with self.assertRaisesRegex(SemanticStateCacheConflict, "gc_orphans"):
            cache.store(model, [1, 4, 9], boundary_kind="custom")

        reopened = SemanticStateAnchorCache(cache.root)
        gc = reopened.gc_orphans()
        self.assertEqual(len(gc.deleted_manifest_names), 1)
        self.assertEqual(len(gc.deleted_payload_names), 1)
        self.assertGreater(gc.reclaimed_bytes, 0)
        self.assertEqual(tuple(reopened.snapshots.iterdir()), ())
        committed = reopened.store(model, [1, 4, 9], boundary_kind="custom")
        self.assertEqual(reopened.receipts(), (committed,))

    def test_orphan_gc_leaves_unowned_hash_named_payload_and_seed_untouched(
        self,
    ) -> None:
        cache = SemanticStateAnchorCache(self.root / "cache")
        payload_prefix = "a" * 64
        payload_raw = b"foreign but content-addressed payload"
        payload_sha = hashlib.sha256(payload_raw).hexdigest()
        payload = cache.snapshots / f"{payload_prefix}.{payload_sha}.npz"
        payload.write_bytes(payload_raw)

        seed_prefix = "b" * 64
        hidden = torch.zeros((1, 1, 128), dtype=torch.float32)
        tensor_sha = cache._seed_tensor_sha256(hidden)
        seed_raw = save_safetensors(
            {"seed_hidden": hidden},
            metadata={
                "prefix_sha256": seed_prefix,
                "schema": SEMANTIC_ANCHOR_SEED_SCHEMA,
                "tensor_sha256": tensor_sha,
            },
        )
        seed_sha = hashlib.sha256(seed_raw).hexdigest()
        seed = cache.snapshots / f"{seed_prefix}.{seed_sha}.seed.safetensors"
        seed.write_bytes(seed_raw)

        receipt = cache.gc_orphans()
        self.assertEqual(receipt.deleted_manifest_names, ())
        self.assertEqual(receipt.deleted_payload_names, ())
        self.assertEqual(receipt.reclaimed_bytes, 0)
        self.assertEqual(
            {path.name for path in cache.snapshots.iterdir()},
            {payload.name, seed.name},
        )


if __name__ == "__main__":
    unittest.main()
