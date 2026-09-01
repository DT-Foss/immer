from __future__ import annotations

from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

import torch

from immer.runtimes.qwen3_8.layer_mlp_crystal import (
    Layer63MlpResidualCrystalIdentity,
    LayerMlpResidualCrystalIdentity,
)
from immer.runtimes.qwen3_8.layer_mlp_o1 import (
    Layer63MlpO1Accumulator,
    LayerMlpO1Accumulator,
)
from immer.runtimes.qwen3_8.layer_mlp_o1_runtime import (
    LAYER_MLP_O1_POOL_METRICS_SCHEMA,
    Layer63MlpO1AsyncWorker,
    Layer63MlpO1Batch,
    Layer63MlpO1RequestBuffer,
    LayerMlpO1AsyncPool,
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


def _generic_identity(layer_index: int) -> LayerMlpResidualCrystalIdentity:
    return LayerMlpResidualCrystalIdentity(
        model_sha256="1" * 64,
        q4_sha256="2" * 64,
        graph_revision_sha256="3" * 64,
        atlas_revision_sha256="4" * 64,
        projection=LayerTransitionProjectionIdentity(
            hidden_dim=8,
            sketch_dim=3,
            seed_sha256="5" * 64,
        ),
        layer_index=layer_index,
    )


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


class LayerMlpO1AsyncPoolTests(unittest.TestCase):
    def _generic_accumulator(
        self,
        root: str,
        layer_index: int,
    ) -> LayerMlpO1Accumulator:
        return LayerMlpO1Accumulator(
            Path(root) / f"layer-{layer_index}.json",
            _generic_identity(layer_index),
            packed_weight_bytes_avoided=123,
        )

    def test_mixed_legacy_and_generic_layers_select_least_charged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            generic = self._generic_accumulator(temporary, 2)
            legacy = Layer63MlpO1Accumulator(
                Path(temporary) / "layer-63.json",
                _identity(),
                packed_weight_bytes_avoided=123,
            )
            generic.observe(
                _batch(3, seed=20).attention_residual,
                _batch(3, seed=20).mlp_input,
                _batch(3, seed=20).layer_output,
            )
            legacy_rows = _batch(1, seed=21)
            legacy.observe(
                legacy_rows.attention_residual,
                legacy_rows.mlp_input,
                legacy_rows.layer_output,
            )
            pool = LayerMlpO1AsyncPool({63: legacy, 2: generic})
            try:
                selected = pool.select_request_buffer()
                layer_index, buffer = selected
                self.assertEqual(layer_index, 63)
                self.assertIs(buffer, selected.buffer)
                self.assertEqual(buffer.hidden_dim, 8)
                self.assertEqual(pool.layer_indices, (2, 63))
                self.assertTrue(pool._thread.daemon)
            finally:
                pool.close()

    def test_selection_counts_pending_rows_and_breaks_ties_by_layer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = self._generic_accumulator(temporary, 1)
            second = self._generic_accumulator(temporary, 2)
            started = threading.Event()
            release = threading.Event()
            original = first.observe

            def blocked(*args):
                started.set()
                self.assertTrue(release.wait(5))
                return original(*args)

            with mock.patch.object(first, "observe", side_effect=blocked):
                pool = LayerMlpO1AsyncPool({2: second, 1: first}, queue_capacity=2)
                try:
                    self.assertEqual(pool.select_request_buffer().layer_index, 1)
                    self.assertTrue(pool.submit(1, _batch(3, seed=30)))
                    self.assertTrue(started.wait(5))
                    self.assertEqual(pool.select_request_buffer().layer_index, 2)
                    release.set()
                    self.assertTrue(pool.flush(timeout=5))
                    self.assertEqual(pool.select_request_buffer().layer_index, 2)
                finally:
                    release.set()
                    pool.close()

    def test_one_global_queue_drops_and_reports_per_layer_and_global_metrics(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = self._generic_accumulator(temporary, 4)
            second = self._generic_accumulator(temporary, 5)
            started = threading.Event()
            release = threading.Event()
            original = first.observe

            def blocked(*args):
                started.set()
                self.assertTrue(release.wait(5))
                return original(*args)

            with mock.patch.object(first, "observe", side_effect=blocked):
                pool = LayerMlpO1AsyncPool({4: first, 5: second}, queue_capacity=1)
                try:
                    self.assertTrue(pool.submit(4, _batch(2, seed=40)))
                    self.assertTrue(started.wait(5))
                    self.assertTrue(pool.submit(5, _batch(3, seed=41)))
                    self.assertFalse(pool.submit(4, _batch(4, seed=42)))
                    pending = pool.metrics()
                    self.assertEqual(
                        pending["schema"], LAYER_MLP_O1_POOL_METRICS_SCHEMA
                    )
                    self.assertEqual(pending["global"]["pending_batches"], 2)
                    self.assertEqual(pending["global"]["pending_rows"], 5)
                    self.assertEqual(pending["global"]["dropped_batches"], 1)
                    self.assertEqual(pending["layers"]["4"]["dropped_rows"], 4)
                    self.assertEqual(pending["layers"]["5"]["dropped_rows"], 0)
                    release.set()
                    self.assertTrue(pool.flush(timeout=5))
                    metrics = pool.metrics()
                    self.assertEqual(metrics["global"]["settled_batches"], 2)
                    self.assertEqual(metrics["global"]["settled_rows"], 5)
                    self.assertEqual(metrics["global"]["accumulated_rows"], 5)
                    self.assertEqual(metrics["global"]["generation"], 2)
                    self.assertEqual(metrics["layers"]["4"]["generation"], 1)
                    self.assertEqual(metrics["layers"]["5"]["generation"], 1)
                    self.assertEqual(metrics["global"]["pending_batches"], 0)
                finally:
                    release.set()
                    pool.close()

    def test_layer_error_does_not_kill_shared_worker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = self._generic_accumulator(temporary, 7)
            second = self._generic_accumulator(temporary, 8)
            with mock.patch.object(
                first,
                "observe",
                side_effect=RuntimeError("synthetic"),
            ):
                pool = LayerMlpO1AsyncPool({7: first, 8: second}, queue_capacity=2)
                try:
                    self.assertTrue(pool.submit(7, _batch(1, seed=50)))
                    self.assertTrue(pool.submit(8, _batch(4, seed=51)))
                    self.assertTrue(pool.flush(timeout=5))
                    metrics = pool.metrics()
                    self.assertEqual(metrics["global"]["error_batches"], 1)
                    self.assertEqual(metrics["layers"]["7"]["error_rows"], 1)
                    self.assertEqual(
                        metrics["layers"]["7"]["last_error_type"],
                        "builtins.RuntimeError",
                    )
                    self.assertEqual(metrics["layers"]["8"]["settled_rows"], 4)
                    self.assertEqual(metrics["global"]["feature_rank"], 3)
                    self.assertTrue(metrics["layers"]["8"]["ready"])
                    self.assertFalse(metrics["global"]["ready"])
                finally:
                    pool.close()

    def test_close_flushes_and_rejects_new_request_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            accumulator = self._generic_accumulator(temporary, 9)
            started = threading.Event()
            release = threading.Event()
            closed = threading.Event()
            original = accumulator.observe

            def blocked(*args):
                started.set()
                self.assertTrue(release.wait(5))
                return original(*args)

            with mock.patch.object(accumulator, "observe", side_effect=blocked):
                pool = LayerMlpO1AsyncPool({9: accumulator}, queue_capacity=1)
                self.assertTrue(pool.submit(9, _batch(2, seed=60)))
                self.assertTrue(started.wait(5))

                def close_pool() -> None:
                    pool.close()
                    closed.set()

                closer = threading.Thread(target=close_pool)
                closer.start()
                self.assertFalse(closed.wait(0.05))
                release.set()
                self.assertTrue(closed.wait(5))
                closer.join()
                pool.close()
                self.assertTrue(pool.metrics()["closed"])
                self.assertEqual(pool.metrics()["global"]["settled_rows"], 2)
                self.assertFalse(pool.submit(9, _batch(1, seed=61)))
                self.assertEqual(pool.metrics()["global"]["dropped_rows"], 1)
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    pool.select_request_buffer()

    def test_mapping_and_batch_dimensions_are_authenticated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            accumulator = self._generic_accumulator(temporary, 3)
            with self.assertRaisesRegex(ValueError, "identity layer_index"):
                LayerMlpO1AsyncPool({2: accumulator})
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                LayerMlpO1AsyncPool({})
            pool = LayerMlpO1AsyncPool({3: accumulator})
            try:
                wrong = Layer63MlpO1Batch(
                    torch.zeros((1, 7), dtype=torch.bfloat16),
                    torch.zeros((1, 7), dtype=torch.bfloat16),
                    torch.zeros((1, 7), dtype=torch.bfloat16),
                )
                with self.assertRaisesRegex(ValueError, "hidden dimension"):
                    pool.submit(3, wrong)
                with self.assertRaisesRegex(ValueError, "not configured"):
                    pool.submit(2, _batch(1))
            finally:
                pool.close()


if __name__ == "__main__":
    unittest.main()
