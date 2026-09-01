from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from safetensors.torch import save_file
import torch

from immer.knowledge import Streamer
from immer.runtimes.qwen3_8.model import StreamedQwen38
from immer.runtimes.qwen3_8.pager import Qwen38WeightPager
from immer.runtimes.qwen3_8.semantic_state_cache import (
    SemanticStateAnchorCache,
    SemanticStateCacheConflict,
    SemanticStateCacheError,
    semantic_label_sha256,
    token_prefix_sha256,
)

from test_qwen3_8_model import _tiny_config, _tiny_weights


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


class _CrashAfterNativeSave:
    def __init__(self, model: StreamedQwen38) -> None:
        self.model = model

    def save_state(self, *args, **kwargs):
        self.model.save_state(*args, **kwargs)
        raise RuntimeError("simulated crash after native snapshot")


class Qwen38SemanticStateCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix=".qwen-semantic-cache-test-", dir=Path.cwd()
        )
        self.root = Path(self.temporary.name)
        self.model_root = self.root / "model"
        self.model_root.mkdir()
        self.config = _tiny_config()
        save_file(_tiny_weights(self.config), self.model_root / "model.safetensors")
        self.resources: list[tuple[Qwen38WeightPager, Streamer]] = []

    def tearDown(self) -> None:
        for pager, source in reversed(self.resources):
            pager.close()
            source.close()
        self.temporary.cleanup()

    def _model(self, *, dtype: str = "float32") -> StreamedQwen38:
        source = Streamer.from_local(self.model_root, budget_mb=20, use_cache=False)
        pager = Qwen38WeightPager(
            source,
            device="cpu",
            compute_dtype=dtype,
            max_resident_bytes=2 * 1024**2,
        )
        self.resources.append((pager, source))
        return StreamedQwen38(
            self.config,
            pager,
            max_batch_size=1,
            max_seq_len=32,
        )

    def _charged(
        self,
        cache: SemanticStateAnchorCache,
        tokens: list[int],
        *,
        boundary: str = "turn",
    ):
        model = self._model()
        hidden, _ = model.prefill([tokens])
        return cache.store(
            model,
            tokens,
            boundary_kind=boundary,
            seed_hidden=hidden[:, -1:],
            semantic_label_sha256=semantic_label_sha256(f"fixture:{len(tokens)}"),
        )

    def test_prefix_hash_is_canonical_and_index_never_stores_tokens_or_text(
        self,
    ) -> None:
        expected = hashlib.sha256(
            b'{"schema":"immer.qwen3.8-token-prefix/v1","token_ids":[1,4,9]}'
        ).hexdigest()
        self.assertEqual(token_prefix_sha256([1, 4, 9]), expected)
        self.assertEqual(token_prefix_sha256((1, 4, 9)), expected)
        self.assertNotEqual(token_prefix_sha256([1, 4]), expected)
        with self.assertRaises(ValueError):
            token_prefix_sha256([True])

        cache = SemanticStateAnchorCache(self.root / "cache")
        receipt = self._charged(cache, [1, 4, 9], boundary="tool-output")
        self.assertEqual(receipt.boundary_kind, "tool-output")
        index_text = cache.index_path.read_text(encoding="utf-8")
        self.assertNotIn("token_ids", index_text)
        self.assertNotIn("fixture:", index_text)
        self.assertNotIn("[1,4,9]", index_text)
        self.assertIn(receipt.prefix_sha256, index_text)

    def test_deepest_exact_prefix_restores_bit_exact_continuation(self) -> None:
        cache = SemanticStateAnchorCache(self.root / "cache")
        short = self._charged(cache, [1, 4], boundary="turn")
        deep = self._charged(cache, [1, 4, 9], boundary="thinking")

        uninterrupted = self._model()
        uninterrupted.prefill([[1, 4, 9]])
        expected_hidden, _ = uninterrupted.decode([[7]])

        restored = self._model()
        hit = cache.restore_deepest(restored, [1, 4, 9, 7])
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.anchor.prefix_sha256, deep.prefix_sha256)
        self.assertEqual(hit.anchor.prefix_length, 3)
        self.assertEqual(hit.anchor.hit_count, 1)
        self.assertEqual(hit.anchor.last_access_sequence, deep.last_access_sequence + 1)
        self.assertFalse(hit.exact_prefix)
        self.assertIsNone(hit.seed_hidden)
        actual_hidden, _ = restored.decode([[7]])
        self.assertTrue(torch.equal(actual_hidden, expected_hidden))
        self.assertEqual(restored.next_position, uninterrupted.next_position)
        self.assertEqual(restored.state_bytes, uninterrupted.state_bytes)

        expected_state = uninterrupted.save_state(self.root / "expected.json")
        actual_state = restored.save_state(self.root / "actual.json")
        self.assertEqual(
            actual_state["payload_sha256"], expected_state["payload_sha256"]
        )

        short_target = self._model()
        short_hit = cache.restore_deepest(short_target, [1, 4, 8])
        self.assertIsNotNone(short_hit)
        assert short_hit is not None
        self.assertEqual(short_hit.anchor.prefix_sha256, short.prefix_sha256)
        self.assertEqual(short_target.next_position, 2)

    def test_exact_prefix_restores_seed_hidden_for_head_without_token_replay(
        self,
    ) -> None:
        cache = SemanticStateAnchorCache(self.root / "cache")
        source = self._model()
        prefix = [1, 4, 9]
        hidden, _ = source.prefill([prefix])
        expected_values, expected_ids = source.pager.topk_logits(
            hidden[:, -1],
            k=1,
            name=source.output_head_name,
        )
        anchor = cache.store(
            source,
            prefix,
            boundary_kind="turn",
            seed_hidden=hidden[:, -1:],
        )

        target = self._model()
        restored = cache.restore_deepest(target, prefix)
        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertTrue(restored.exact_prefix)
        self.assertEqual(restored.anchor.prefix_sha256, anchor.prefix_sha256)
        self.assertIsNotNone(restored.seed_hidden)
        self.assertEqual(target.next_position, len(prefix))
        assert restored.seed_hidden is not None
        self.assertTrue(torch.equal(restored.seed_hidden, hidden[:, -1:]))
        actual_values, actual_ids = target.pager.topk_logits(
            restored.seed_hidden[:, -1],
            k=1,
            name=target.output_head_name,
        )
        self.assertTrue(torch.equal(actual_ids, expected_ids))
        self.assertTrue(torch.equal(actual_values, expected_values))
        # The head scan consumes no token and therefore cannot advance or
        # positionally duplicate the prompt's final token.
        self.assertEqual(target.next_position, len(prefix))

    def test_miss_does_not_mutate_index_clock_or_model(self) -> None:
        cache = SemanticStateAnchorCache(self.root / "cache")
        self._charged(cache, [1, 4])
        before = cache.index_path.read_bytes()
        target = self._model()
        self.assertIsNone(cache.restore_deepest(target, [2, 4, 9]))
        self.assertEqual(target.next_position, 0)
        self.assertEqual(cache.index_path.read_bytes(), before)

    def test_native_identity_mismatch_evicts_stale_anchor_as_a_miss(self) -> None:
        cache = SemanticStateAnchorCache(self.root / "cache")
        anchor = self._charged(cache, [1, 4, 9])
        incompatible = self._model(dtype="bfloat16")
        self.assertIsNone(cache.restore_deepest(incompatible, [1, 4, 9, 7]))
        self.assertEqual(incompatible.next_position, 0)
        self.assertEqual(cache.receipts(), ())
        self.assertFalse(
            (cache.snapshots / anchor.snapshot_manifest_name).exists()
        )
        self.assertFalse((cache.snapshots / anchor.snapshot_payload_name).exists())

    def test_manifest_payload_index_tamper_and_symlinks_fail_closed(self) -> None:
        cache = SemanticStateAnchorCache(self.root / "cache")
        anchor = self._charged(cache, [1, 4, 9])
        manifest = cache.snapshots / anchor.snapshot_manifest_name
        manifest.write_bytes(manifest.read_bytes() + b" ")
        with self.assertRaisesRegex(
            SemanticStateCacheError, "byte count mismatch|exceeds its byte limit"
        ):
            cache.lookup_deepest([1, 4, 9])

        symlink_root = self.root / "cache-link"
        os.symlink(cache.root, symlink_root)
        with self.assertRaisesRegex(SemanticStateCacheError, "non-symlink"):
            SemanticStateAnchorCache(symlink_root)

        other = self.root / "other"
        other.mkdir()
        replacement = cache.root / "index-link.json"
        os.symlink(other, replacement)
        cache.index_path.unlink()
        os.replace(replacement, cache.index_path)
        with self.assertRaises(SemanticStateCacheError):
            SemanticStateAnchorCache(cache.root)

    def test_sealed_path_escape_in_index_is_rejected(self) -> None:
        cache = SemanticStateAnchorCache(self.root / "cache")
        self._charged(cache, [1, 4])
        document = json.loads(cache.index_path.read_text(encoding="utf-8"))
        anchor = document["body"]["anchors"][0]
        anchor["snapshot_manifest_name"] = "../escape.json"
        receipt_body = dict(anchor)
        receipt_body.pop("receipt_sha256")
        anchor["receipt_sha256"] = hashlib.sha256(
            _canonical_json(receipt_body)
        ).hexdigest()
        document["body_sha256"] = hashlib.sha256(
            _canonical_json(document["body"])
        ).hexdigest()
        cache.index_path.write_bytes(_canonical_json(document) + b"\n")
        with self.assertRaisesRegex(SemanticStateCacheError, "prefix-bound"):
            SemanticStateAnchorCache(cache.root)

    def test_payload_and_seed_hidden_tamper_are_detected_before_hit_commit(
        self,
    ) -> None:
        payload_cache = SemanticStateAnchorCache(self.root / "payload-cache")
        payload_anchor = self._charged(payload_cache, [1, 4, 9])
        payload = payload_cache.snapshots / payload_anchor.snapshot_payload_name
        with payload.open("r+b") as stream:
            stream.seek(payload.stat().st_size // 2)
            original = stream.read(1)
            stream.seek(-1, os.SEEK_CUR)
            stream.write(bytes([original[0] ^ 0x01]))
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
            stream.write(bytes([original[0] ^ 0x01]))
        with self.assertRaisesRegex(SemanticStateCacheError, "seed hidden SHA-256"):
            seed_cache.lookup_deepest([1, 4, 9])
        self.assertEqual(seed_cache.receipts()[0].hit_count, 0)

    def test_corrupt_unused_seed_does_not_block_suffix_restore(self) -> None:
        cache = SemanticStateAnchorCache(self.root / "cache")
        anchor = self._charged(cache, [1, 4, 9])
        assert anchor.seed_hidden_name is not None
        seed = cache.snapshots / anchor.seed_hidden_name
        with seed.open("r+b") as stream:
            stream.seek(seed.stat().st_size // 2)
            original = stream.read(1)
            stream.seek(-1, os.SEEK_CUR)
            stream.write(bytes([original[0] ^ 0x01]))

        uninterrupted = self._model()
        uninterrupted.prefill([[1, 4, 9]])
        expected, _ = uninterrupted.decode([[7]])
        suffix_target = self._model()
        restored = cache.restore_deepest(suffix_target, [1, 4, 9, 7])
        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertFalse(restored.exact_prefix)
        self.assertIsNone(restored.seed_hidden)
        actual, _ = suffix_target.decode([[7]])
        self.assertTrue(torch.equal(actual, expected))

        exact_target = self._model()
        with self.assertRaisesRegex(SemanticStateCacheError, "seed hidden SHA-256"):
            cache.restore_deepest(exact_target, [1, 4, 9])
        self.assertEqual(exact_target.next_position, 0)
        self.assertEqual(exact_target.state_bytes, 0)
        # Only the valid suffix restore reached the durable hit commit.
        self.assertEqual(cache.receipts()[0].hit_count, 1)

    def test_alternate_same_identity_snapshot_receipt_resets_before_hit(self) -> None:
        cache = SemanticStateAnchorCache(self.root / "cache")
        anchor = self._charged(cache, [1, 4, 9])

        alternate = self._model()
        alternate.prefill([[2, 3, 8]])
        alternate_path = self.root / "alternate.json"
        alternate_receipt = alternate.save_state(alternate_path)
        self.assertNotEqual(
            alternate_receipt["manifest_body_sha256"],
            anchor.snapshot_body_sha256,
        )

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
        self.assertEqual(target.state_bytes, 0)
        self.assertEqual(cache.receipts()[0].hit_count, 0)

    def test_hit_commit_failure_resets_successfully_loaded_state(self) -> None:
        cache = SemanticStateAnchorCache(self.root / "cache")
        self._charged(cache, [1, 4, 9])
        target = self._model()
        with mock.patch.object(
            cache, "_commit_hit", side_effect=RuntimeError("hit commit failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "hit commit failed"):
                cache.restore_deepest(target, [1, 4, 9, 7])
        self.assertEqual(target.next_position, 0)
        self.assertEqual(target.state_bytes, 0)
        self.assertEqual(cache.receipts()[0].hit_count, 0)

    def test_budget_uses_durable_hit_recency_for_lru_eviction(self) -> None:
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
        assert touched is not None
        self.assertEqual(touched.prefix_sha256, first.prefix_sha256)
        third = self._charged(cache, [1, 4, 9])

        retained = {row.prefix_sha256 for row in cache.receipts()}
        self.assertEqual(retained, {first.prefix_sha256, third.prefix_sha256})
        self.assertNotIn(
            second.snapshot_manifest_name,
            {path.name for path in cache.snapshots.iterdir()},
        )
        self.assertLessEqual(cache.total_bytes, budget)

    def test_reopen_preserves_receipts_and_logical_lru_counters(self) -> None:
        root = self.root / "cache"
        cache = SemanticStateAnchorCache(root)
        stored = self._charged(cache, [1, 4], boundary="tool-call")
        hit = cache.lookup_deepest([1, 4, 9])
        self.assertIsNotNone(hit)
        reopened = SemanticStateAnchorCache(root)
        self.assertEqual(reopened.receipts(), (hit,))
        self.assertGreater(hit.last_access_sequence, stored.last_access_sequence)

    def test_snapshot_first_crash_leaves_invisible_orphan_for_explicit_gc(self) -> None:
        root = self.root / "cache"
        cache = SemanticStateAnchorCache(root)
        model = self._model()
        model.prefill([[1, 4, 9]])
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

        reopened = SemanticStateAnchorCache(root)
        gc = reopened.gc_orphans()
        self.assertEqual(len(gc.deleted_manifest_names), 1)
        self.assertEqual(len(gc.deleted_payload_names), 1)
        self.assertGreater(gc.reclaimed_bytes, 0)
        self.assertEqual(tuple(reopened.snapshots.iterdir()), ())
        committed = reopened.store(model, [1, 4, 9], boundary_kind="custom")
        self.assertEqual(reopened.receipts(), (committed,))


if __name__ == "__main__":
    unittest.main()
