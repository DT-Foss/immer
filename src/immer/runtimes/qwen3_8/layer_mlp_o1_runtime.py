"""Passive asynchronous O1 charging for the exact layer-63 MLP seam."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from queue import Empty, Full, Queue
import threading
import time

import torch

from .layer_mlp_crystal import LayerMlpCrystalIntegrityError
from .layer_mlp_o1 import (
    Layer63MlpO1Accumulator,
    Layer63MlpO1Snapshot,
    LayerMlpO1Accumulator,
)


LAYER_MLP_O1_RUNTIME_METRICS_SCHEMA = "immer.qwen3.8-layer63-mlp-o1-runtime/v1"
LAYER_MLP_O1_POOL_METRICS_SCHEMA = "immer.qwen3.8-layer-mlp-o1-runtime/v2"


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class Layer63MlpO1Batch:
    """One request-owned transient batch of exact BF16 K1 rows."""

    attention_residual: torch.Tensor
    mlp_input: torch.Tensor
    layer_output: torch.Tensor

    def __post_init__(self) -> None:
        tensors = (
            self.attention_residual,
            self.mlp_input,
            self.layer_output,
        )
        if any(
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != torch.bfloat16
            or value.layout != torch.strided
            or value.ndim != 2
            or not value.is_contiguous()
            for value in tensors
        ):
            raise ValueError("O1 request batch must contain contiguous CPU BF16 rows")
        shape = tuple(self.attention_residual.shape)
        if not shape or shape[0] < 1 or shape[1] < 1:
            raise ValueError("O1 request batch must not be empty")
        if any(tuple(value.shape) != shape for value in tensors[1:]):
            raise ValueError("O1 request batch tensor shapes differ")

    @property
    def rows(self) -> int:
        return int(self.attention_residual.shape[0])


class Layer63MlpO1RequestBuffer:
    """Collect one successful generation's already-detached K1 rows in RAM."""

    def __init__(self, hidden_dim: int) -> None:
        self.hidden_dim = _positive_int(hidden_dim, "hidden_dim")
        self._attention_residual: list[torch.Tensor] = []
        self._mlp_input: list[torch.Tensor] = []
        self._layer_output: list[torch.Tensor] = []

    def _row(self, value: torch.Tensor, field: str) -> torch.Tensor:
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != torch.bfloat16
            or value.layout != torch.strided
            or tuple(value.shape) != (1, 1, self.hidden_dim)
            or not value.is_contiguous()
        ):
            raise ValueError(f"{field} is not one exact CPU BF16 K1 row")
        return value.reshape(1, self.hidden_dim)

    def __call__(
        self,
        attention_residual: torch.Tensor,
        mlp_input: torch.Tensor,
        layer_output: torch.Tensor,
    ) -> None:
        base = self._row(attention_residual, "attention_residual")
        feature = self._row(mlp_input, "mlp_input")
        target = self._row(layer_output, "layer_output")
        self._attention_residual.append(base)
        self._mlp_input.append(feature)
        self._layer_output.append(target)

    @property
    def rows(self) -> int:
        return len(self._attention_residual)

    def batch(self) -> Layer63MlpO1Batch | None:
        if not self._attention_residual:
            return None
        return Layer63MlpO1Batch(
            attention_residual=torch.cat(self._attention_residual, dim=0),
            mlp_input=torch.cat(self._mlp_input, dim=0),
            layer_output=torch.cat(self._layer_output, dim=0),
        )


class Layer63MlpO1AsyncWorker:
    """Own one accumulator and settle bounded request batches off the hot path."""

    _STOP = object()

    def __init__(
        self,
        accumulator: Layer63MlpO1Accumulator,
        *,
        queue_capacity: int = 8,
    ) -> None:
        if not isinstance(accumulator, Layer63MlpO1Accumulator):
            raise TypeError("accumulator must be a Layer63MlpO1Accumulator")
        self.accumulator = accumulator
        self.queue_capacity = _positive_int(queue_capacity, "queue_capacity")
        self._queue: Queue[Layer63MlpO1Batch | object] = Queue(
            maxsize=self.queue_capacity
        )
        self._condition = threading.Condition()
        self._queued_batches = 0
        self._queued_rows = 0
        self._settled_batches = 0
        self._settled_rows = 0
        self._dropped_batches = 0
        self._dropped_rows = 0
        self._error_batches = 0
        self._error_rows = 0
        self._pending_batches = 0
        self._pending_rows = 0
        self._last_error_type: str | None = None
        self._snapshot: Layer63MlpO1Snapshot | None = None
        try:
            self._snapshot = accumulator.snapshot()
        except LayerMlpCrystalIntegrityError as exc:
            # A newly configured O1 state has no generation until its first row.
            if str(exc) != "MLP O1 state has no observations":
                raise
            self._snapshot = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="immer-layer63-mlp-o1",
            daemon=True,
        )
        self._thread.start()

    def request_buffer(self) -> Layer63MlpO1RequestBuffer:
        return Layer63MlpO1RequestBuffer(self.accumulator.identity.hidden_dim)

    def submit(self, batch: Layer63MlpO1Batch | None) -> bool:
        """Enqueue without waiting; a full queue drops the complete request batch."""

        if batch is None:
            return False
        if not isinstance(batch, Layer63MlpO1Batch):
            raise TypeError("batch must be a Layer63MlpO1Batch or None")
        with self._condition:
            if self._closed:
                self._dropped_batches += 1
                self._dropped_rows += batch.rows
                return False
            try:
                self._queue.put_nowait(batch)
            except Full:
                self._dropped_batches += 1
                self._dropped_rows += batch.rows
                return False
            self._queued_batches += 1
            self._queued_rows += batch.rows
            self._pending_batches += 1
            self._pending_rows += batch.rows
            self._condition.notify_all()
            return True

    def _run(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.25)
            except Empty:
                continue
            try:
                if item is self._STOP:
                    return
                assert isinstance(item, Layer63MlpO1Batch)
                try:
                    snapshot = self.accumulator.observe(
                        item.attention_residual,
                        item.mlp_input,
                        item.layer_output,
                    )
                except Exception as exc:
                    with self._condition:
                        self._error_batches += 1
                        self._error_rows += item.rows
                        self._last_error_type = (
                            f"{type(exc).__module__}.{type(exc).__qualname__}"
                        )
                else:
                    with self._condition:
                        self._snapshot = snapshot
                        self._settled_batches += 1
                        self._settled_rows += item.rows
                finally:
                    with self._condition:
                        self._pending_batches -= 1
                        self._pending_rows -= item.rows
                        self._condition.notify_all()
            finally:
                self._queue.task_done()

    def flush(self, timeout: float | None = None) -> bool:
        """Wait for all accepted batches; inference never calls this method."""

        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout < 0
        ):
            raise ValueError("timeout must be non-negative or None")
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._condition:
            while self._pending_batches:
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def metrics(self) -> dict[str, object]:
        with self._condition:
            snapshot = self._snapshot
            return {
                "accumulated_rows": (
                    0 if snapshot is None else snapshot.accumulated_rows
                ),
                "closed": self._closed,
                "dropped_batches": self._dropped_batches,
                "dropped_rows": self._dropped_rows,
                "error_batches": self._error_batches,
                "error_rows": self._error_rows,
                "feature_rank": 0 if snapshot is None else snapshot.feature_rank,
                "generation": 0 if snapshot is None else snapshot.generation,
                "identity_sha256": self.accumulator.identity.identity_sha256,
                "last_error_type": self._last_error_type,
                "pending_batches": self._pending_batches,
                "pending_rows": self._pending_rows,
                "queue_capacity": self.queue_capacity,
                "queued_batches": self._queued_batches,
                "queued_rows": self._queued_rows,
                "ready": False if snapshot is None else snapshot.ready,
                "schema": LAYER_MLP_O1_RUNTIME_METRICS_SCHEMA,
                "settled_batches": self._settled_batches,
                "settled_rows": self._settled_rows,
                "sketch_dim": self.accumulator.identity.sketch_dim,
            }

    def close(self) -> None:
        if self._closed:
            return
        with self._condition:
            self._closed = True
        self.flush()
        self._queue.put(self._STOP)
        self._thread.join()


@dataclass(frozen=True, slots=True)
class LayerMlpO1SelectedBuffer:
    """One request's deterministically selected layer and transient buffer."""

    layer_index: int
    buffer: Layer63MlpO1RequestBuffer

    def __iter__(self):
        yield self.layer_index
        yield self.buffer


@dataclass(slots=True)
class _LayerMlpO1RuntimeState:
    accumulator: Layer63MlpO1Accumulator
    snapshot: Layer63MlpO1Snapshot | None
    queued_batches: int = 0
    queued_rows: int = 0
    settled_batches: int = 0
    settled_rows: int = 0
    dropped_batches: int = 0
    dropped_rows: int = 0
    error_batches: int = 0
    error_rows: int = 0
    pending_batches: int = 0
    pending_rows: int = 0
    last_error_type: str | None = None


@dataclass(frozen=True, slots=True)
class _LayerMlpO1QueuedBatch:
    layer_index: int
    batch: Layer63MlpO1Batch


class LayerMlpO1AsyncPool:
    """Settle exact MLP observations for many layers on one bounded worker."""

    _STOP = object()

    def __init__(
        self,
        accumulators: Mapping[
            int,
            Layer63MlpO1Accumulator | LayerMlpO1Accumulator,
        ],
        *,
        queue_capacity: int = 8,
    ) -> None:
        if not isinstance(accumulators, Mapping):
            raise TypeError("accumulators must be a layer-index mapping")
        if not accumulators:
            raise ValueError("accumulators must not be empty")
        self.queue_capacity = _positive_int(queue_capacity, "queue_capacity")
        states: dict[int, _LayerMlpO1RuntimeState] = {}
        for raw_layer, accumulator in accumulators.items():
            if (
                isinstance(raw_layer, bool)
                or not isinstance(raw_layer, int)
                or raw_layer < 0
            ):
                raise ValueError(
                    "accumulator layer indices must be non-negative integers"
                )
            if not isinstance(
                accumulator,
                (Layer63MlpO1Accumulator, LayerMlpO1Accumulator),
            ):
                raise TypeError("accumulator values must be MLP O1 accumulators")
            identity_layer = accumulator.identity.layer_index
            if raw_layer != identity_layer:
                raise ValueError(
                    "accumulator mapping key differs from its identity layer_index"
                )
            try:
                snapshot = accumulator.snapshot()
            except LayerMlpCrystalIntegrityError as exc:
                if str(exc) != "MLP O1 state has no observations":
                    raise
                snapshot = None
            states[raw_layer] = _LayerMlpO1RuntimeState(
                accumulator=accumulator,
                snapshot=snapshot,
            )
        self._layers = dict(sorted(states.items()))
        self._queue: Queue[_LayerMlpO1QueuedBatch | object] = Queue(
            maxsize=self.queue_capacity
        )
        self._condition = threading.Condition()
        self._close_lock = threading.Lock()
        self._last_error_type: str | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="immer-layer-mlp-o1-pool",
            daemon=True,
        )
        self._thread.start()

    @property
    def layer_indices(self) -> tuple[int, ...]:
        return tuple(self._layers)

    def select_request_buffer(self) -> LayerMlpO1SelectedBuffer:
        """Select the least-charged layer without touching persistent state."""

        with self._condition:
            if self._closed:
                raise RuntimeError("MLP O1 pool is closed")
            layer_index, state = min(
                self._layers.items(),
                key=lambda item: (
                    (
                        0
                        if item[1].snapshot is None
                        else item[1].snapshot.accumulated_rows
                    )
                    + item[1].pending_rows,
                    item[0],
                ),
            )
            hidden_dim = state.accumulator.identity.hidden_dim
        return LayerMlpO1SelectedBuffer(
            layer_index=layer_index,
            buffer=Layer63MlpO1RequestBuffer(hidden_dim),
        )

    def submit(self, layer_index: int, batch: Layer63MlpO1Batch | None) -> bool:
        """Enqueue one layer batch without waiting for compute or persistence."""

        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or layer_index not in self._layers
        ):
            raise ValueError("layer_index is not configured in the MLP O1 pool")
        if batch is None:
            return False
        if not isinstance(batch, Layer63MlpO1Batch):
            raise TypeError("batch must be a Layer63MlpO1Batch or None")
        with self._condition:
            state = self._layers[layer_index]
            if (
                batch.attention_residual.shape[1]
                != state.accumulator.identity.hidden_dim
            ):
                raise ValueError("batch hidden dimension differs from layer identity")
            if self._closed:
                state.dropped_batches += 1
                state.dropped_rows += batch.rows
                return False
            try:
                self._queue.put_nowait(
                    _LayerMlpO1QueuedBatch(layer_index=layer_index, batch=batch)
                )
            except Full:
                state.dropped_batches += 1
                state.dropped_rows += batch.rows
                return False
            state.queued_batches += 1
            state.queued_rows += batch.rows
            state.pending_batches += 1
            state.pending_rows += batch.rows
            self._condition.notify_all()
            return True

    def _run(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.25)
            except Empty:
                continue
            try:
                if item is self._STOP:
                    return
                assert isinstance(item, _LayerMlpO1QueuedBatch)
                state = self._layers[item.layer_index]
                snapshot: Layer63MlpO1Snapshot | None = None
                error: Exception | None = None
                try:
                    snapshot = state.accumulator.observe(
                        item.batch.attention_residual,
                        item.batch.mlp_input,
                        item.batch.layer_output,
                    )
                except Exception as exc:
                    error = exc
                with self._condition:
                    if error is None:
                        assert snapshot is not None
                        state.snapshot = snapshot
                        state.settled_batches += 1
                        state.settled_rows += item.batch.rows
                    else:
                        error_type = (
                            f"{type(error).__module__}.{type(error).__qualname__}"
                        )
                        state.error_batches += 1
                        state.error_rows += item.batch.rows
                        state.last_error_type = error_type
                        self._last_error_type = error_type
                    state.pending_batches -= 1
                    state.pending_rows -= item.batch.rows
                    self._condition.notify_all()
            finally:
                self._queue.task_done()

    def flush(self, timeout: float | None = None) -> bool:
        """Wait for all accepted layer batches; inference never calls this."""

        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout < 0
        ):
            raise ValueError("timeout must be non-negative or None")
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._condition:
            while any(state.pending_batches for state in self._layers.values()):
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    @staticmethod
    def _layer_metrics(
        layer_index: int,
        state: _LayerMlpO1RuntimeState,
    ) -> dict[str, object]:
        snapshot = state.snapshot
        return {
            "accumulated_rows": 0 if snapshot is None else snapshot.accumulated_rows,
            "dropped_batches": state.dropped_batches,
            "dropped_rows": state.dropped_rows,
            "error_batches": state.error_batches,
            "error_rows": state.error_rows,
            "feature_rank": 0 if snapshot is None else snapshot.feature_rank,
            "generation": 0 if snapshot is None else snapshot.generation,
            "identity_sha256": state.accumulator.identity.identity_sha256,
            "last_error_type": state.last_error_type,
            "layer_index": layer_index,
            "pending_batches": state.pending_batches,
            "pending_rows": state.pending_rows,
            "queued_batches": state.queued_batches,
            "queued_rows": state.queued_rows,
            "ready": False if snapshot is None else snapshot.ready,
            "settled_batches": state.settled_batches,
            "settled_rows": state.settled_rows,
            "sketch_dim": state.accumulator.identity.sketch_dim,
        }

    def metrics(self) -> dict[str, object]:
        with self._condition:
            layers = {
                str(layer_index): self._layer_metrics(layer_index, state)
                for layer_index, state in self._layers.items()
            }
            values = tuple(layers.values())
            global_metrics = {
                "accumulated_rows": sum(
                    int(layer["accumulated_rows"]) for layer in values
                ),
                "dropped_batches": sum(
                    int(layer["dropped_batches"]) for layer in values
                ),
                "dropped_rows": sum(int(layer["dropped_rows"]) for layer in values),
                "error_batches": sum(int(layer["error_batches"]) for layer in values),
                "error_rows": sum(int(layer["error_rows"]) for layer in values),
                "feature_rank": sum(int(layer["feature_rank"]) for layer in values),
                "generation": sum(int(layer["generation"]) for layer in values),
                "last_error_type": self._last_error_type,
                "pending_batches": sum(
                    int(layer["pending_batches"]) for layer in values
                ),
                "pending_rows": sum(int(layer["pending_rows"]) for layer in values),
                "queued_batches": sum(int(layer["queued_batches"]) for layer in values),
                "queued_rows": sum(int(layer["queued_rows"]) for layer in values),
                "ready": all(bool(layer["ready"]) for layer in values),
                "ready_layers": sum(bool(layer["ready"]) for layer in values),
                "settled_batches": sum(
                    int(layer["settled_batches"]) for layer in values
                ),
                "settled_rows": sum(int(layer["settled_rows"]) for layer in values),
            }
            return {
                "closed": self._closed,
                "global": global_metrics,
                "layer_count": len(layers),
                "layers": layers,
                "queue_capacity": self.queue_capacity,
                "schema": LAYER_MLP_O1_POOL_METRICS_SCHEMA,
            }

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            with self._condition:
                self._closed = True
            self.flush()
            self._queue.put(self._STOP)
            self._thread.join()


__all__ = [
    "LAYER_MLP_O1_RUNTIME_METRICS_SCHEMA",
    "LAYER_MLP_O1_POOL_METRICS_SCHEMA",
    "Layer63MlpO1AsyncWorker",
    "Layer63MlpO1Batch",
    "Layer63MlpO1RequestBuffer",
    "LayerMlpO1AsyncPool",
    "LayerMlpO1SelectedBuffer",
]
