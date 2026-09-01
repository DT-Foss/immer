from __future__ import annotations

from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

import torch

from immer.runtimes.qwen3_8.layer_mlp_crystal import (
    Layer63MlpResidualCrystalIdentity,
)
from immer.runtimes.qwen3_8.layer_mlp_o1 import Layer63MlpO1Accumulator
from immer.runtimes.qwen3_8.layer_mlp_o1_runtime import (
    Layer63MlpO1AsyncWorker,
    Layer63MlpO1Batch,
    Layer63MlpO1RequestBuffer,
)
from immer.runtimes.qwen3_8.layer_transition_crystal import (
    LayerTransitionProjectionIdentity,
)


def _identity() -> Layer63MlpResidualCrystalIdentity:
    return Layer63MlpResidualCrystalIdentity(
        model_sha256="1" * 64,
        q4_sha256="2" * 64,
        graph_revision_sha256="3" * 64,
        atlas_revision_sha256="4" * 64,
        projection=LayerTransitionProjectionIdentity(
            hidden_dim=8,
            sketch_dim=3,
            seed_sha256="5" * 64,
        ),
    )


def _batch(rows: int, *, seed: int = 7) -> Layer63MlpO1Batch:
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(rows, 8, generator=generator).to(torch.bfloat16)
    feature = torch.randn(rows, 8, generator=generator).to(torch.bfloat16)
    target = (base.float() + feature.float() * 0.1).to(torch.bfloat16)
    return Layer63MlpO1Batch(base, feature, target)


class LayerMlpO1RuntimeTests(unittest.TestCase):
    def test_request_buffer_collects_only_small_cpu_bf16_rows(self) -> None:
        buffer = Layer63MlpO1RequestBuffer(8)
        rows = _batch(2)
        for index in range(2):
            buffer(
                rows.attention_residual[index : index + 1].reshape(1, 1, 8),
                rows.mlp_input[index : index + 1].reshape(1, 1, 8),
                rows.layer_output[index : index + 1].reshape(1, 1, 8),
            )
        batch = buffer.batch()
        self.assertIsNotNone(batch)
        assert batch is not None
        self.assertEqual(batch.rows, 2)
        self.assertTrue(torch.equal(batch.attention_residual, rows.attention_residual))
        with self.assertRaisesRegex(ValueError, "exact CPU BF16 K1"):
            buffer(
                torch.zeros((1, 2, 8), dtype=torch.bfloat16),
                torch.zeros((1, 1, 8), dtype=torch.bfloat16),
                torch.zeros((1, 1, 8), dtype=torch.bfloat16),
            )

    def test_async_settle_flush_and_close_publish_one_atomic_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            accumulator = Layer63MlpO1Accumulator(
                Path(temporary) / "o1.json",
                _identity(),
                packed_weight_bytes_avoided=123,
            )
            worker = Layer63MlpO1AsyncWorker(accumulator, queue_capacity=2)
            self.assertTrue(worker.submit(_batch(5)))
            self.assertTrue(worker.flush(timeout=5))
            metrics = worker.metrics()
            self.assertEqual(metrics["queued_batches"], 1)
            self.assertEqual(metrics["settled_batches"], 1)
            self.assertEqual(metrics["settled_rows"], 5)
            self.assertEqual(metrics["accumulated_rows"], 5)
            self.assertEqual(metrics["generation"], 1)
            self.assertEqual(metrics["pending_batches"], 0)
            worker.close()
            worker.close()
            self.assertTrue(worker.metrics()["closed"])

    def test_queue_bound_drops_without_blocking_submitter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            accumulator = Layer63MlpO1Accumulator(
                Path(temporary) / "o1.json",
                _identity(),
                packed_weight_bytes_avoided=123,
            )
            started = threading.Event()
            release = threading.Event()
            original = accumulator.observe

            def blocked(*args):
                started.set()
                self.assertTrue(release.wait(5))
                return original(*args)

            with mock.patch.object(accumulator, "observe", side_effect=blocked):
                worker = Layer63MlpO1AsyncWorker(accumulator, queue_capacity=1)
                self.assertTrue(worker.submit(_batch(1, seed=1)))
                self.assertTrue(started.wait(5))
                self.assertTrue(worker.submit(_batch(1, seed=2)))
                self.assertFalse(worker.submit(_batch(1, seed=3)))
                metrics = worker.metrics()
                self.assertEqual(metrics["dropped_batches"], 1)
                self.assertEqual(metrics["dropped_rows"], 1)
                release.set()
                self.assertTrue(worker.flush(timeout=5))
                worker.close()

    def test_worker_error_is_counted_and_does_not_kill_later_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            accumulator = Layer63MlpO1Accumulator(
                Path(temporary) / "o1.json",
                _identity(),
                packed_weight_bytes_avoided=123,
            )
            original = accumulator.observe
            calls = 0

            def fail_once(*args):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise RuntimeError("synthetic")
                return original(*args)

            with mock.patch.object(accumulator, "observe", side_effect=fail_once):
                worker = Layer63MlpO1AsyncWorker(accumulator, queue_capacity=2)
                self.assertTrue(worker.submit(_batch(1, seed=4)))
                self.assertTrue(worker.submit(_batch(4, seed=5)))
                self.assertTrue(worker.flush(timeout=5))
                metrics = worker.metrics()
                self.assertEqual(metrics["error_batches"], 1)
                self.assertEqual(metrics["settled_batches"], 1)
                self.assertEqual(metrics["accumulated_rows"], 4)
                self.assertEqual(metrics["last_error_type"], "builtins.RuntimeError")
                worker.close()


if __name__ == "__main__":
    unittest.main()
