"""One-weight-at-a-time BF16 paging for the streamed Qwen3.8 text decoder."""

from __future__ import annotations

import gc
import sys
import threading
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from ..deepseek_v4.causal_weights import CausalTensorReader
from .config import validate_source_identity


class Qwen38PagerError(RuntimeError):
    """A checkpoint tensor cannot be materialized within the pager contract."""


@dataclass(slots=True)
class PagerMetrics:
    tensor_reads: int = 0
    row_reads: int = 0
    linear_calls: int = 0
    embedding_rows: int = 0
    head_rows: int = 0
    logical_weight_bytes: int = 0
    materialized_tensor_bytes: int = 0
    peak_planned_resident_bytes: int = 0
    materialized_weight_releases: int = 0
    release_boundaries: int = 0
    mps_cache_purges: int = 0


@dataclass(frozen=True, slots=True)
class _TensorLayout:
    name: str
    shape: tuple[int, ...]
    shard: str
    absolute: int
    payload_bytes: int

    @property
    def numel(self) -> int:
        value = 1
        for dimension in self.shape:
            value *= dimension
        return value


class Qwen38WeightPager:
    """Execute exact BF16 safetensors ranges without retaining model weights.

    The source owns its verified disk cache and transport budget.  The pager
    owns no weight cache: a matrix is range-read, moved to the resolved compute
    dtype/device, consumed, and released before the next matrix.  A lock keeps
    concurrent callers from violating the single-weight resident bound.
    """

    DEFAULT_MAX_RESIDENT_BYTES = 384 * 1024**2
    DEFAULT_HEAD_BLOCK_ROWS = 2048
    WEIGHT_CACHE_POLICY = "one-shot-bf16-exact-range/v1"

    def __init__(
        self,
        source: Any,
        *,
        device: str = "auto",
        compute_dtype: str = "auto",
        max_resident_bytes: int = DEFAULT_MAX_RESIDENT_BYTES,
        close_source: bool = False,
        require_source_identity: bool = False,
        causal_tensor_reader: CausalTensorReader | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - neural extra
            raise Qwen38PagerError("Qwen3.8 execution requires torch") from exc
        if sys.byteorder != "little":  # safetensors scalar encoding is little-endian
            raise Qwen38PagerError("Qwen3.8 BF16 paging requires a little-endian host")
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        if device not in {"cpu", "mps"}:
            raise ValueError("device must be 'auto', 'cpu', or 'mps'")
        if device == "mps" and not torch.backends.mps.is_available():
            raise Qwen38PagerError("MPS was requested but is unavailable")
        if compute_dtype == "auto":
            compute_dtype = "bfloat16"
        try:
            dtype = getattr(torch, compute_dtype)
        except AttributeError as exc:
            raise ValueError(
                f"unsupported torch compute dtype: {compute_dtype}"
            ) from exc
        if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError("compute_dtype must be float16, bfloat16, or float32")
        if (
            isinstance(max_resident_bytes, bool)
            or not isinstance(max_resident_bytes, int)
            or max_resident_bytes <= 0
        ):
            raise ValueError("max_resident_bytes must be a positive integer")
        if not isinstance(close_source, bool):
            raise ValueError("close_source must be a boolean")
        if not isinstance(require_source_identity, bool):
            raise ValueError("require_source_identity must be a boolean")
        if causal_tensor_reader is not None:
            if not isinstance(causal_tensor_reader, CausalTensorReader):
                raise TypeError("causal_tensor_reader must be a CausalTensorReader")
            if causal_tensor_reader.source is not source:
                raise ValueError("causal tensor reader must own this exact source")

        self.torch = torch
        self.source = source
        self.device = torch.device(device)
        self.compute_dtype = dtype
        self.max_resident_bytes = max_resident_bytes
        self.close_source = close_source
        self.causal_tensor_reader = causal_tensor_reader
        self.source_identity = validate_source_identity(
            getattr(source, "repo_id", None),
            getattr(source, "revision", None),
            require_identity=require_source_identity,
        )
        self._stats = PagerMetrics()
        self._lock = threading.RLock()
        self._closed = False

    @property
    def resolved_device(self) -> str:
        return str(self.device)

    @property
    def resolved_dtype(self) -> str:
        return str(self.compute_dtype).removeprefix("torch.")

    @staticmethod
    def _weight_name(prefix: str) -> str:
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("weight name must be a non-empty string")
        return prefix if prefix.endswith(".weight") else f"{prefix}.weight"

    @staticmethod
    def _consecutive_runs(ids: Iterable[int]) -> tuple[tuple[int, int], ...]:
        values = sorted(set(ids))
        if not values:
            return ()
        runs: list[tuple[int, int]] = []
        start = previous = values[0]
        for value in values[1:]:
            if value != previous + 1:
                runs.append((start, previous + 1))
                start = value
            previous = value
        runs.append((start, previous + 1))
        return tuple(runs)

    def _ensure_open(self) -> None:
        if self._closed:
            raise Qwen38PagerError("Qwen3.8 pager is closed")

    def _dtype_bytes(self, dtype: Any) -> int:
        return int(self.torch.empty((), dtype=dtype).element_size())

    def _preflight_resident(
        self,
        *,
        payload_bytes: int,
        numel: int,
        target_dtype: Any,
        label: str,
    ) -> int:
        planned = payload_bytes + numel * self._dtype_bytes(target_dtype)
        if planned > self.max_resident_bytes:
            raise Qwen38PagerError(
                f"{label} needs {planned} resident bytes (BF16 payload plus "
                f"{str(target_dtype).removeprefix('torch.')} tensor), limit is "
                f"{self.max_resident_bytes}"
            )
        self._stats.peak_planned_resident_bytes = max(
            self._stats.peak_planned_resident_bytes, planned
        )
        return planned

    def _layout(self, name: str) -> _TensorLayout:
        if self.causal_tensor_reader is not None:
            plan = self.causal_tensor_reader.resolve_tensor_plan(name)
            meta = {
                "data_start": 0,
                "dtype": plan.dtype,
                "offset_in_shard": (plan.absolute_offset, plan.absolute_end),
                "shape": plan.shape,
                "shard": plan.shard,
            }
        else:
            try:
                meta = self.source.find(name)
            except KeyError:
                raise
            except Exception as exc:
                raise Qwen38PagerError(f"cannot resolve tensor {name!r}") from exc
        dtype = str(meta.get("dtype", "")).upper()
        if dtype != "BF16":
            raise Qwen38PagerError(
                f"Qwen3.8 tensor {name!r} must be BF16, got {dtype or 'missing'}"
            )
        raw_shape = meta.get("shape")
        if not isinstance(raw_shape, (list, tuple)) or not raw_shape:
            raise Qwen38PagerError(f"tensor {name!r} has no valid shape")
        shape: list[int] = []
        for dimension in raw_shape:
            if (
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension <= 0
            ):
                raise Qwen38PagerError(f"tensor {name!r} has an invalid shape")
            shape.append(dimension)
        offsets = meta.get("offset_in_shard")
        if not isinstance(offsets, (list, tuple)) or len(offsets) != 2:
            raise Qwen38PagerError(f"tensor {name!r} has invalid shard offsets")
        begin, end = offsets
        if (
            isinstance(begin, bool)
            or isinstance(end, bool)
            or not isinstance(begin, int)
            or not isinstance(end, int)
            or begin < 0
            or end <= begin
        ):
            raise Qwen38PagerError(f"tensor {name!r} has invalid shard offsets")
        shard = meta.get("shard")
        data_start = meta.get("data_start")
        if not isinstance(shard, str) or not shard:
            raise Qwen38PagerError(f"tensor {name!r} has no shard identity")
        if (
            isinstance(data_start, bool)
            or not isinstance(data_start, int)
            or data_start < 0
        ):
            raise Qwen38PagerError(f"tensor {name!r} has invalid data_start")
        numel = 1
        for dimension in shape:
            numel *= dimension
        payload_bytes = end - begin
        if payload_bytes != numel * 2:
            raise Qwen38PagerError(
                f"tensor {name!r} shape requires {numel * 2} BF16 bytes, "
                f"offsets contain {payload_bytes}"
            )
        return _TensorLayout(
            name=name,
            shape=tuple(shape),
            shard=shard,
            absolute=data_start + begin,
            payload_bytes=payload_bytes,
        )

    def _decode_bf16(
        self,
        raw: bytes,
        *,
        shape: tuple[int, ...],
        dtype: Any,
        device: Any,
        name: str,
    ) -> Any:
        numel = 1
        for dimension in shape:
            numel *= dimension
        if len(raw) != numel * 2:
            raise Qwen38PagerError(
                f"short BF16 payload for {name!r}: {len(raw)}/{numel * 2} bytes"
            )
        # The read-only warning is harmless here: the view is never returned or
        # mutated.  It is immediately copied to owned CPU memory or moved to the
        # compute device before the source payload is released.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given buffer is not writable",
                category=UserWarning,
            )
            storage = self.torch.frombuffer(raw, dtype=self.torch.bfloat16)
        storage = storage.reshape(shape)
        target_device = self.torch.device(device)
        if target_device.type == "cpu" and dtype == self.torch.bfloat16:
            result = storage.clone()
        else:
            result = storage.to(device=target_device, dtype=dtype)
        result.requires_grad_(False)
        return result

    def _read_tensor(
        self,
        name: str,
        *,
        dtype: Any,
        device: Any,
    ) -> Any:
        layout = self._layout(name)
        self._preflight_resident(
            payload_bytes=layout.payload_bytes,
            numel=layout.numel,
            target_dtype=dtype,
            label=name,
        )
        if self.causal_tensor_reader is None:
            raw = self.source.raw_bytes(
                layout.shard, layout.absolute, layout.payload_bytes
            )
        else:
            receipt = self.causal_tensor_reader.read_tensor_range(
                name,
                length=layout.payload_bytes,
            )
            if (
                receipt.plan.shard != layout.shard
                or receipt.plan.absolute_offset != layout.absolute
                or receipt.length != layout.payload_bytes
            ):
                raise Qwen38PagerError("causal tensor receipt disagrees with layout")
            raw = receipt.part
        result = self._decode_bf16(
            raw,
            shape=layout.shape,
            dtype=dtype,
            device=device,
            name=name,
        )
        self._stats.tensor_reads += 1
        self._stats.logical_weight_bytes += layout.payload_bytes
        self._stats.materialized_tensor_bytes += result.numel() * result.element_size()
        return result

    def _read_rows(
        self,
        name: str,
        start_row: int,
        n_rows: int,
        *,
        dtype: Any,
        device: Any,
    ) -> Any:
        layout = self._layout(name)
        if len(layout.shape) != 2:
            raise Qwen38PagerError(f"row paging requires a 2D tensor: {name!r}")
        total_rows, columns = layout.shape
        if (
            isinstance(start_row, bool)
            or not isinstance(start_row, int)
            or isinstance(n_rows, bool)
            or not isinstance(n_rows, int)
            or start_row < 0
            or n_rows <= 0
            or start_row > total_rows - n_rows
        ):
            raise IndexError(
                f"rows [{start_row}, {start_row + n_rows}) outside [0, {total_rows})"
            )
        row_bytes = columns * 2
        payload_bytes = n_rows * row_bytes
        numel = n_rows * columns
        self._preflight_resident(
            payload_bytes=payload_bytes,
            numel=numel,
            target_dtype=dtype,
            label=f"{name}[{start_row}:{start_row + n_rows}]",
        )
        relative_offset = start_row * row_bytes
        if self.causal_tensor_reader is None:
            raw = self.source.raw_bytes(
                layout.shard,
                layout.absolute + relative_offset,
                payload_bytes,
            )
        else:
            receipt = self.causal_tensor_reader.read_tensor_range(
                name,
                relative_offset=relative_offset,
                length=payload_bytes,
            )
            if (
                receipt.plan.shard != layout.shard
                or receipt.plan.absolute_offset != layout.absolute
                or receipt.relative_offset != relative_offset
                or receipt.length != payload_bytes
            ):
                raise Qwen38PagerError(
                    "causal tensor row receipt disagrees with layout"
                )
            raw = receipt.part
        result = self._decode_bf16(
            raw,
            shape=(n_rows, columns),
            dtype=dtype,
            device=device,
            name=name,
        )
        self._stats.row_reads += 1
        self._stats.logical_weight_bytes += payload_bytes
        self._stats.materialized_tensor_bytes += result.numel() * result.element_size()
        return result

    def tensor_torch(
        self,
        name: str,
        *,
        dtype: Any | None = None,
        device: str | Any | None = None,
    ) -> Any:
        """Read one BF16 tensor with exact range and resident preflight."""

        with self._lock:
            self._ensure_open()
            target_dtype = self.compute_dtype if dtype is None else dtype
            if target_dtype not in {
                self.torch.float16,
                self.torch.bfloat16,
                self.torch.float32,
            }:
                raise ValueError("dtype must be float16, bfloat16, or float32")
            target_device = self.device if device is None else self.torch.device(device)
            return self._read_tensor(name, dtype=target_dtype, device=target_device)

    def linear(
        self,
        x: Any,
        name: str,
        *,
        output_dtype: Any | None = None,
    ) -> Any:
        """Apply one bias-free checkpoint matrix and release it before return."""

        with self._lock:
            self._ensure_open()
            if not isinstance(x, self.torch.Tensor):
                x = self.torch.as_tensor(x)
            weight_name = self._weight_name(name)
            compute_x = x.to(device=self.device, dtype=self.compute_dtype)
            weight = self._read_tensor(
                weight_name, dtype=self.compute_dtype, device=self.device
            )
            if weight.ndim != 2:
                del weight
                raise Qwen38PagerError(f"linear weight {weight_name!r} must be 2D")
            if compute_x.shape[-1] != weight.shape[1]:
                dimensions = tuple(weight.shape)
                del weight
                raise Qwen38PagerError(
                    f"linear input width {compute_x.shape[-1]} disagrees with "
                    f"{weight_name}{dimensions}"
                )
            try:
                result = self.torch.nn.functional.linear(compute_x, weight)
                if output_dtype is not None:
                    result = result.to(dtype=output_dtype)
                self._stats.linear_calls += 1
                return result
            finally:
                del weight
                self._stats.materialized_weight_releases += 1

    def _validate_ids(self, name: str, token_ids: Iterable[int]) -> tuple[int, ...]:
        layout = self._layout(name)
        if len(layout.shape) != 2:
            raise Qwen38PagerError(f"token-row tensor {name!r} must be 2D")
        try:
            ids = tuple(int(value) for value in token_ids)
        except (TypeError, ValueError) as exc:
            raise ValueError("token IDs must be an iterable of integers") from exc
        vocab = layout.shape[0]
        if any(value < 0 or value >= vocab for value in ids):
            raise IndexError(f"token ID outside [0, {vocab})")
        return ids

    def _selected_rows(self, name: str, token_ids: tuple[int, ...]) -> Any:
        if not token_ids:
            columns = self._layout(name).shape[1]
            return self.torch.empty(
                (0, columns), device=self.device, dtype=self.compute_dtype
            )
        layout = self._layout(name)
        columns = layout.shape[1]
        runs = self._consecutive_runs(token_ids)
        unique_target_bytes = (
            len(set(token_ids)) * columns * self._dtype_bytes(self.compute_dtype)
        )
        largest_payload = max(stop - start for start, stop in runs) * columns * 2
        output_bytes = len(token_ids) * columns * self._dtype_bytes(self.compute_dtype)
        planned = unique_target_bytes + max(largest_payload, output_bytes)
        if planned > self.max_resident_bytes:
            raise Qwen38PagerError(
                f"selected rows from {name!r} need {planned} resident bytes, "
                f"limit is {self.max_resident_bytes}"
            )
        self._stats.peak_planned_resident_bytes = max(
            self._stats.peak_planned_resident_bytes, planned
        )
        by_id: dict[int, Any] = {}
        for start, stop in runs:
            rows = self._read_rows(
                name,
                start,
                stop - start,
                dtype=self.compute_dtype,
                device=self.device,
            )
            for offset, row in enumerate(rows.unbind(0)):
                by_id[start + offset] = row
        return self.torch.stack([by_id[value] for value in token_ids], dim=0)

    def embedding(
        self,
        token_ids: Iterable[int],
        *,
        name: str = "model.language_model.embed_tokens.weight",
    ) -> Any:
        """Read only requested embedding rows, preserving ID order and repeats."""

        with self._lock:
            self._ensure_open()
            ids = self._validate_ids(name, token_ids)
            rows = self._selected_rows(name, ids)
            self._stats.embedding_rows += len(ids)
            return rows

    def candidate_logits(
        self,
        hidden: Any,
        token_ids: Iterable[int],
        *,
        name: str = "lm_head.weight",
    ) -> Any:
        """Score an explicit token set without materializing the full LM head."""

        with self._lock:
            self._ensure_open()
            ids = self._validate_ids(name, token_ids)
            if not ids:
                raise ValueError("candidate token IDs must not be empty")
            if not isinstance(hidden, self.torch.Tensor):
                hidden = self.torch.as_tensor(hidden)
            compute_hidden = hidden.to(device=self.device, dtype=self.compute_dtype)
            rows = self._selected_rows(name, ids)
            if compute_hidden.shape[-1] != rows.shape[1]:
                del rows
                raise Qwen38PagerError("hidden width disagrees with LM head")
            try:
                return self.torch.nn.functional.linear(compute_hidden, rows)
            finally:
                del rows
                self._stats.head_rows += len(ids)
                self._stats.materialized_weight_releases += 1

    def topk_logits(
        self,
        hidden: Any,
        *,
        k: int = 1,
        name: str = "lm_head.weight",
        block_rows: int = DEFAULT_HEAD_BLOCK_ROWS,
        progress: Callable[[dict[str, int]], None] | None = None,
    ) -> tuple[Any, Any]:
        """Compute exact global top-k logits via bounded contiguous head blocks."""

        with self._lock:
            self._ensure_open()
            layout = self._layout(name)
            if len(layout.shape) != 2:
                raise Qwen38PagerError(f"LM head {name!r} must be 2D")
            vocab, columns = layout.shape
            if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= vocab:
                raise ValueError(f"k must be an integer in [1, {vocab}]")
            if (
                isinstance(block_rows, bool)
                or not isinstance(block_rows, int)
                or block_rows <= 0
            ):
                raise ValueError("block_rows must be a positive integer")
            if progress is not None and not callable(progress):
                raise TypeError("progress must be callable or None")
            if not isinstance(hidden, self.torch.Tensor):
                hidden = self.torch.as_tensor(hidden)
            compute_hidden = hidden.to(device=self.device, dtype=self.compute_dtype)
            if compute_hidden.shape[-1] != columns:
                raise Qwen38PagerError("hidden width disagrees with LM head")
            leading_shape = tuple(compute_hidden.shape[:-1])
            flat = compute_hidden.reshape(-1, columns)
            best_values = None
            best_ids = None
            for start in range(0, vocab, block_rows):
                count = min(block_rows, vocab - start)
                rows = self._read_rows(
                    name,
                    start,
                    count,
                    dtype=self.compute_dtype,
                    device=self.device,
                )
                logits = self.torch.nn.functional.linear(flat, rows)
                local_k = min(k, count)
                token_ids = self.torch.arange(
                    start,
                    start + count,
                    device=logits.device,
                    dtype=self.torch.long,
                ).expand_as(logits)
                values, indices = self._stable_topk(logits, token_ids, local_k)
                del logits
                del rows
                self._stats.head_rows += count
                self._stats.materialized_weight_releases += 1
                if best_values is None:
                    best_values, best_ids = values, indices
                    if progress is not None:
                        progress(
                            {
                                "start_row": start,
                                "rows": count,
                                "rows_done": start + count,
                                "vocab_rows": vocab,
                            }
                        )
                    continue
                merged_values = self.torch.cat((best_values, values), dim=-1)
                merged_ids = self.torch.cat((best_ids, indices), dim=-1)
                best_values, best_ids = self._stable_topk(merged_values, merged_ids, k)
                if progress is not None:
                    progress(
                        {
                            "start_row": start,
                            "rows": count,
                            "rows_done": start + count,
                            "vocab_rows": vocab,
                        }
                    )
            assert best_values is not None and best_ids is not None
            return (
                best_values.reshape(*leading_shape, k),
                best_ids.reshape(*leading_shape, k),
            )

    def _stable_topk(self, values: Any, ids: Any, k: int) -> tuple[Any, Any]:
        """Sort by descending logit and then ascending token ID on exact ties."""

        if values.shape != ids.shape or values.ndim < 1:
            raise ValueError("stable top-k values and IDs must share a shape")
        if not 0 < k <= values.shape[-1]:
            raise ValueError("stable top-k k is outside the final dimension")
        id_order = self.torch.argsort(ids, dim=-1, stable=True)
        ordered_ids = self.torch.gather(ids, -1, id_order)
        ordered_values = self.torch.gather(values, -1, id_order)
        value_order = self.torch.argsort(
            ordered_values, dim=-1, descending=True, stable=True
        )[..., :k]
        return (
            self.torch.gather(ordered_values, -1, value_order),
            self.torch.gather(ordered_ids, -1, value_order),
        )

    def _release_locked(self) -> None:
        gc.collect()
        self._stats.release_boundaries += 1
        if self.device.type == "mps":
            empty_cache = getattr(getattr(self.torch, "mps", None), "empty_cache", None)
            if callable(empty_cache):
                empty_cache()
                self._stats.mps_cache_purges += 1

    def release(self) -> None:
        """Drop allocator caches at an explicit decoder-layer boundary."""

        with self._lock:
            self._ensure_open()
            self._release_locked()

    def close(self) -> None:
        """Wait for any active call, release allocator state, and close if owned."""

        with self._lock:
            if self._closed:
                return
            self._release_locked()
            if self.close_source:
                close = getattr(self.source, "close", None)
                if callable(close):
                    close()
            self._closed = True

    def metrics(self) -> dict[str, Any]:
        source_metrics_method = getattr(self.source, "metrics", None)
        source_metrics = (
            dict(source_metrics_method()) if callable(source_metrics_method) else {}
        )
        causal_metrics = (
            {}
            if self.causal_tensor_reader is None
            else {
                f"causal_tensor_{key}": value
                for key, value in self.causal_tensor_reader.metrics().items()
            }
        )
        with self._lock:
            return {
                **source_metrics,
                **asdict(self._stats),
                **causal_metrics,
                "causal_tensor_reader_attached": (
                    self.causal_tensor_reader is not None
                ),
                "device": self.resolved_device,
                "compute_dtype": self.resolved_dtype,
                "max_resident_bytes": self.max_resident_bytes,
                "weight_cache_policy": self.WEIGHT_CACHE_POLICY,
                "source_identity": self.source_identity,
                "closed": self._closed,
            }

    def __enter__(self) -> "Qwen38WeightPager":
        self._ensure_open()
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()
