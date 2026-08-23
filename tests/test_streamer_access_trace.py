from __future__ import annotations

import json
import struct
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from immer.knowledge.access_trace import (
    AccessTrace,
    AccessTraceIdentityError,
    AccessTraceIntegrityError,
    AccessTraceRecorder,
    replay_access_trace,
)
from immer.knowledge.streamer import Streamer


def _write_shard(path: Path, payload: bytes, tensor_name: str) -> int:
    header = {
        tensor_name: {
            "dtype": "U8",
            "shape": [len(payload)],
            "data_offsets": [0, len(payload)],
        }
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return 8 + len(encoded)


def _fixture(root: Path) -> tuple[int, int]:
    root.mkdir()
    first = _write_shard(root / "a.safetensors", bytes(range(64)), "a")
    second = _write_shard(root / "b.safetensors", bytes(range(64, 128)), "b")
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 128},
                "weight_map": {"a": "a.safetensors", "b": "b.safetensors"},
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return first, second


class StreamerAccessTraceTests(unittest.TestCase):
    def test_scalar_and_batch_record_logical_leaves_including_cache_hits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "source"
            first, _second = _fixture(root)
            recorder = AccessTraceRecorder()
            source = Streamer.from_local(
                root,
                cache_dir=base / "cache",
                access_observer=recorder,
            )

            with recorder.scope(prompt_digest="abc", layer=7, attempt=1):
                self.assertEqual(
                    source.raw_bytes("a.safetensors", first, 4),
                    bytes(range(4)),
                )
                self.assertEqual(
                    source.raw_bytes("a.safetensors", first, 4),
                    bytes(range(4)),
                )
                result = source.raw_bytes_many(
                    "a.safetensors",
                    [(first + 4, 2), (first + 6, 2), (first + 4, 2)],
                    resident_limit_bytes=4,
                )
            self.assertEqual(
                [bytes(part) for part in result.parts],
                [b"\x04\x05", b"\x06\x07", b"\x04\x05"],
            )

            trace = recorder.snapshot()
            self.assertEqual(len(trace.operations), 3)
            cold, cached, batch = trace.operations
            self.assertEqual(cold.cache_hits, 0)
            self.assertGreater(cold.source_bytes, 0)
            self.assertEqual(cached.cache_hits, 1)
            self.assertEqual(cached.source_bytes, 0)
            self.assertEqual(
                [(leaf.offset, leaf.length) for leaf in batch.leaves],
                [(first + 4, 2), (first + 6, 2), (first + 4, 2)],
            )
            self.assertEqual(
                dict(batch.tags),
                {"attempt": 1, "layer": 7, "prompt_digest": "abc"},
            )
            self.assertEqual(source.metrics()["access_observer_events"], 3)
            self.assertEqual(source.metrics()["access_observer_leaves"], 5)

    def test_observer_errors_and_drops_never_change_read_result(self) -> None:
        class BrokenObserver:
            def observe(self, _operation):
                raise RuntimeError("observer broke")

        class DroppingObserver:
            def observe(self, _operation):
                return False

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "source"
            first, _second = _fixture(root)
            source = Streamer.from_local(root, cache_dir=base / "cache")
            source.set_access_observer(BrokenObserver())
            self.assertEqual(
                source.raw_bytes("a.safetensors", first, 3), b"\x00\x01\x02"
            )
            self.assertEqual(source.metrics()["access_observer_errors"], 1)

            source.set_access_observer(DroppingObserver())
            self.assertEqual(
                source.raw_bytes("a.safetensors", first, 3), b"\x00\x01\x02"
            )
            metrics = source.metrics()
            self.assertEqual(metrics["access_observer_drops"], 1)
            self.assertEqual(metrics["access_observer_events"], 0)

    def test_observer_does_not_swallow_process_control_exceptions(self) -> None:
        class StoppingObserver:
            def observe(self, _operation):
                raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "source"
            first, _second = _fixture(root)
            source = Streamer.from_local(root, cache_dir=base / "cache")
            source.set_access_observer(StoppingObserver())
            with self.assertRaises(KeyboardInterrupt):
                source.raw_bytes("a.safetensors", first, 1)

    def test_identity_mismatch_fails_before_payload_io(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "source"
            first, _second = _fixture(root)
            recorder = AccessTraceRecorder()
            source = Streamer.from_local(
                root,
                cache_dir=base / "record-cache",
                access_observer=recorder,
            )
            source.raw_bytes("a.safetensors", first, 4)
            trace = recorder.snapshot()

            replay = Streamer.from_local(
                root,
                revision="different",
                cache_dir=base / "replay-cache",
            )
            with mock.patch.object(
                replay, "raw_bytes_many", wraps=replay.raw_bytes_many
            ) as payload_reads:
                with self.assertRaises(AccessTraceIdentityError):
                    replay_access_trace(replay, trace)
            payload_reads.assert_not_called()

            matching = Streamer.from_local(root, cache_dir=base / "matching-cache")
            matching.inventory()
            wrong_fingerprint = "0" * 64
            if wrong_fingerprint == trace.inventory_fingerprint:
                wrong_fingerprint = "1" * 64
            mismatched_trace = AccessTrace(
                repo_id=trace.repo_id,
                revision=trace.revision,
                inventory_fingerprint=wrong_fingerprint,
                operations=tuple(
                    replace(operation, inventory_fingerprint=wrong_fingerprint)
                    for operation in trace.operations
                ),
            )
            with mock.patch.object(
                matching, "raw_bytes_many", wraps=matching.raw_bytes_many
            ) as payload_reads:
                with self.assertRaises(AccessTraceIdentityError):
                    replay_access_trace(matching, mismatched_trace)
            payload_reads.assert_not_called()

    def test_replay_deduplicates_bounds_and_never_mixes_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "source"
            first, second = _fixture(root)
            recorder = AccessTraceRecorder()
            source = Streamer.from_local(
                root,
                cache_dir=base / "record-cache",
                access_observer=recorder,
            )
            source.raw_bytes_many(
                "a.safetensors",
                [(first, 4), (first, 4), (first + 8, 4)],
                resident_limit_bytes=8,
            )
            source.raw_bytes("b.safetensors", second, 4)
            trace = recorder.snapshot()

            replay = Streamer.from_local(root, cache_dir=base / "replay-cache")
            replay.inventory()
            with mock.patch.object(
                replay, "raw_bytes_many", wraps=replay.raw_bytes_many
            ) as reads:
                receipt = replay_access_trace(
                    replay,
                    trace,
                    max_leaves=2,
                    max_bytes=8,
                )

            self.assertEqual(receipt.unique_leaves, 3)
            self.assertEqual(receipt.duplicate_leaves, 1)
            self.assertEqual(receipt.selected_leaves, 2)
            self.assertEqual(receipt.limit_skipped_leaves, 1)
            self.assertEqual(receipt.warmed_leaves, 2)
            self.assertEqual(reads.call_count, 1)
            self.assertEqual(reads.call_args.args[0], "a.safetensors")
            self.assertEqual(reads.call_args.kwargs["max_gap_bytes"], 0)

    def test_default_replay_consumes_full_trace_in_configurable_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "source"
            first, second = _fixture(root)
            recorder = AccessTraceRecorder()
            source = Streamer.from_local(
                root,
                cache_dir=base / "record-cache",
                access_observer=recorder,
            )
            source.raw_bytes_many(
                "a.safetensors",
                [(first, 4), (first + 8, 4), (first + 16, 4)],
                resident_limit_bytes=12,
            )
            source.raw_bytes("b.safetensors", second, 4)
            trace = recorder.snapshot()

            replay = Streamer.from_local(root, cache_dir=base / "replay-cache")
            receipt = replay_access_trace(
                replay,
                trace,
                window_max_leaves=2,
                window_max_bytes=8,
            )

            self.assertEqual(receipt.unique_leaves, 4)
            self.assertEqual(receipt.selected_leaves, 4)
            self.assertEqual(receipt.limit_skipped_leaves, 0)
            self.assertEqual(receipt.warmed_leaves, 4)
            self.assertEqual(receipt.windows, 3)
            self.assertEqual(receipt.largest_window_leaves, 2)
            self.assertEqual(receipt.largest_window_bytes, 8)

    def test_recorder_is_complete_by_default_and_limits_are_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "source"
            first, _second = _fixture(root)
            complete = AccessTraceRecorder()
            source = Streamer.from_local(
                root,
                cache_dir=base / "complete-cache",
                access_observer=complete,
            )
            for offset in range(10):
                source.raw_bytes("a.safetensors", first + offset, 1)
            self.assertEqual(complete.metrics()["operations"], 10)
            self.assertEqual(complete.metrics()["dropped_capacity"], 0)

            limited = AccessTraceRecorder(max_operations=1)
            source.set_access_observer(limited)
            source.raw_bytes("a.safetensors", first, 1)
            source.raw_bytes("a.safetensors", first + 1, 1)
            self.assertEqual(limited.metrics()["operations"], 1)
            self.assertEqual(limited.metrics()["dropped_capacity"], 1)

    def test_exact_replay_warms_scalar_leaf_and_budget_decline_is_explicit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "source"
            first, _second = _fixture(root)
            recorder = AccessTraceRecorder()
            source = Streamer.from_local(
                root,
                cache_dir=base / "record-cache",
                access_observer=recorder,
            )
            expected = source.raw_bytes("a.safetensors", first, 16)
            trace = recorder.snapshot()

            replay = Streamer.from_local(root, cache_dir=base / "replay-cache")
            receipt = replay_access_trace(replay, trace)
            self.assertEqual(receipt.warmed_leaves, 1)
            moved = replay.bytes_moved()
            self.assertEqual(replay.raw_bytes("a.safetensors", first, 16), expected)
            self.assertEqual(replay.bytes_moved(), moved)

            tight = Streamer.from_local(
                root,
                budget_mb=1,
                cache_dir=base / "tight-cache",
            )
            tight.inventory()
            tight.budget.limit = tight.budget.total
            declined = replay_access_trace(tight, trace)
            self.assertTrue(declined.budget_declined)
            self.assertEqual(declined.warmed_leaves, 0)
            self.assertEqual(declined.budget_declined_leaves, 1)

    def test_trace_round_trip_rejects_tamper_and_noncanonical_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "source"
            first, _second = _fixture(root)
            recorder = AccessTraceRecorder()
            source = Streamer.from_local(
                root,
                cache_dir=base / "cache",
                access_observer=recorder,
            )
            source.raw_bytes("a.safetensors", first, 4)
            trace = recorder.snapshot()

            self.assertEqual(AccessTrace.from_bytes(trace.to_bytes()), trace)
            tampered = trace.to_document()
            tampered["operations"][0]["leaves"][0]["length"] = 3
            with self.assertRaisesRegex(AccessTraceIntegrityError, "SHA-256"):
                AccessTrace.from_document(tampered)
            pretty = json.dumps(trace.to_document(), indent=2).encode("utf-8")
            with self.assertRaisesRegex(AccessTraceIntegrityError, "not canonical"):
                AccessTrace.from_bytes(pretty)

    def test_scope_rejects_unbounded_or_nonfinite_tags(self) -> None:
        recorder = AccessTraceRecorder()
        with self.assertRaises(AccessTraceIntegrityError):
            with recorder.scope(value=float("nan")):
                pass
        with self.assertRaises(AccessTraceIntegrityError):
            with recorder.scope(value="x" * 513):
                pass


if __name__ == "__main__":
    unittest.main()
