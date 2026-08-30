"""One-weight-at-a-time paging for the streamed Qwen3.5 text decoder."""

from __future__ import annotations

import gc
import os
from pathlib import Path
import sys
import threading
import warnings
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Iterable

from ..deepseek_v4.causal_weights import CausalTensorReader
from .config import validate_source_identity

if TYPE_CHECKING:
    from .q4 import Q4Bank


class Qwen38PagerError(RuntimeError):
    """A checkpoint tensor cannot be materialized within the pager contract."""


def _linux_current_rss_bytes() -> int | None:
    statm = Path("/proc/self/statm")
    try:
        fields = statm.read_text(encoding="ascii").split()
        resident_pages = int(fields[1])
        page_bytes = int(os.sysconf("SC_PAGE_SIZE"))
        if resident_pages >= 0 and page_bytes > 0:
            return resident_pages * page_bytes
    except (IndexError, OSError, ValueError):
        return None
    return None


@lru_cache(maxsize=1)
def _darwin_rss_reader() -> Callable[[], int | None] | None:
    """Bind Darwin's current-RSS proc_pidinfo call once per process."""

    if sys.platform != "darwin":
        return None
    try:
        import ctypes

        class ProcTaskInfo(ctypes.Structure):
            _fields_ = (
                ("pti_virtual_size", ctypes.c_uint64),
                ("pti_resident_size", ctypes.c_uint64),
                ("pti_total_user", ctypes.c_uint64),
                ("pti_total_system", ctypes.c_uint64),
                ("pti_threads_user", ctypes.c_uint64),
                ("pti_threads_system", ctypes.c_uint64),
                ("pti_policy", ctypes.c_int32),
                ("pti_faults", ctypes.c_int32),
                ("pti_pageins", ctypes.c_int32),
                ("pti_cow_faults", ctypes.c_int32),
                ("pti_messages_sent", ctypes.c_int32),
                ("pti_messages_received", ctypes.c_int32),
                ("pti_syscalls_mach", ctypes.c_int32),
                ("pti_syscalls_unix", ctypes.c_int32),
                ("pti_csw", ctypes.c_int32),
                ("pti_threadnum", ctypes.c_int32),
                ("pti_numrunning", ctypes.c_int32),
                ("pti_priority", ctypes.c_int32),
            )

        proc_pidinfo = ctypes.CDLL(
            "/usr/lib/libproc.dylib",
            use_errno=True,
        ).proc_pidinfo
        proc_pidinfo.argtypes = (
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        )
        proc_pidinfo.restype = ctypes.c_int
        expected = ctypes.sizeof(ProcTaskInfo)
    except (AttributeError, OSError):
        return None

    def read() -> int | None:
        info = ProcTaskInfo()
        received = proc_pidinfo(
            os.getpid(),
            4,  # PROC_PIDTASKINFO
            0,
            ctypes.byref(info),
            expected,
        )
        rss = int(info.pti_resident_size)
        return rss if received == expected and rss >= 0 else None

    return read


def _darwin_current_rss_bytes() -> int | None:
    reader = _darwin_rss_reader()
    return None if reader is None else reader()


def _process_rss_bytes() -> int | None:
    """Return current RSS only; historical peak counters are never accepted."""

    if sys.platform == "darwin":
        return _darwin_current_rss_bytes()
    if sys.platform.startswith("linux"):
        return _linux_current_rss_bytes()
    return None


@dataclass(slots=True)
class PagerMetrics:
    tensor_reads: int = 0
    row_reads: int = 0
    linear_calls: int = 0
    packed_linear_calls: int = 0
    packed_linear_rows: int = 0
    grouped_linear_calls: int = 0
    grouped_linear_matrices: int = 0
    embedding_rows: int = 0
    head_rows: int = 0
    logical_weight_bytes: int = 0
    materialized_tensor_bytes: int = 0
    zero_copy_tensor_reads: int = 0
    zero_copy_bytes_avoided: int = 0
    direct_tensor_fills: int = 0
    direct_tensor_fill_bytes: int = 0
    peak_planned_resident_bytes: int = 0
    materialized_weight_releases: int = 0
    release_boundaries: int = 0
    gc_policy_boundaries: int = 0
    mps_cache_purges: int = 0
    gc_collections: int = 0
    gc_collections_skipped: int = 0
    gc_collections_forced: int = 0
    gc_collections_pressure: int = 0
    gc_collections_interval: int = 0
    gc_collections_fail_closed: int = 0
    gc_q4_mmap_skips: int = 0
    gc_objects_collected: int = 0
    gc_rss_measurement_failures: int = 0
    gc_last_observed_rss_bytes: int = 0
    gc_peak_observed_rss_bytes: int = 0


@dataclass(frozen=True, slots=True)
class _TensorLayout:
    name: str
    shape: tuple[int, ...]
    dtype: str
    item_bytes: int
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
    """Execute exact safetensors ranges without retaining model weights.

    The source owns its verified disk cache and transport budget.  The pager
    owns no weight cache: a matrix is range-read, moved to the resolved compute
    dtype/device, consumed, and released before the next matrix.  A lock keeps
    concurrent callers from violating the single-weight resident bound.
    """

    DEFAULT_MAX_RESIDENT_BYTES = 384 * 1024**2
    DEFAULT_HEAD_BLOCK_ROWS = 2048
    DEFAULT_GC_INTERVAL_BOUNDARIES = 256
    DEFAULT_GC_RSS_HEADROOM_BYTES = 1024**3
    WEIGHT_CACHE_POLICY = "one-shot-qwen35-direct-fill/v4"
    HEAD_SCORE_POLICY = "cpu-bf16-explicit-fp32-accumulate-rne/v1"
    LEGACY_HEAD_SCORE_POLICY = "backend-bf16-linear/v1"
    Q4_WEIGHT_CACHE_POLICY = "causal-mmap-q4_0-q8_0/v1"

    def __init__(
        self,
        source: Any,
        *,
        device: str = "auto",
        compute_dtype: str = "auto",
        max_resident_bytes: int = DEFAULT_MAX_RESIDENT_BYTES,
        gc_interval_boundaries: int = DEFAULT_GC_INTERVAL_BOUNDARIES,
        gc_rss_limit_bytes: int | None = None,
        close_source: bool = False,
        require_source_identity: bool = False,
        causal_tensor_reader: CausalTensorReader | None = None,
        exact_head_index: Any | None = None,
        q4_bank: Q4Bank | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - neural extra
            raise Qwen38PagerError("Qwen3.8 execution requires torch") from exc
        if sys.byteorder != "little":  # safetensors scalar encoding is little-endian
            raise Qwen38PagerError("Qwen3.5 paging requires a little-endian host")
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
        if (
            isinstance(gc_interval_boundaries, bool)
            or not isinstance(gc_interval_boundaries, int)
            or gc_interval_boundaries <= 0
        ):
            raise ValueError("gc_interval_boundaries must be a positive integer")
        if gc_rss_limit_bytes is not None and (
            isinstance(gc_rss_limit_bytes, bool)
            or not isinstance(gc_rss_limit_bytes, int)
            or gc_rss_limit_bytes <= 0
        ):
            raise ValueError("gc_rss_limit_bytes must be a positive integer or None")
        if not isinstance(require_source_identity, bool):
            raise ValueError("require_source_identity must be a boolean")
        if causal_tensor_reader is not None:
            if not isinstance(causal_tensor_reader, CausalTensorReader):
                raise TypeError("causal_tensor_reader must be a CausalTensorReader")
            if causal_tensor_reader.source is not source:
                raise ValueError("causal tensor reader must own this exact source")
        if exact_head_index is not None and not callable(
            getattr(exact_head_index, "topk_logits", None)
        ):
            raise TypeError("exact_head_index must expose topk_logits() or be None")
        if q4_bank is not None:
            if torch.device(device).type != "cpu":
                raise ValueError("Q4 execution requires the CPU device")
            if exact_head_index is not None:
                raise ValueError("Q4 execution and the exact BF16 head are exclusive")
            if not all(
                callable(getattr(q4_bank, method, None))
                for method in (
                    "has",
                    "linear",
                    "linear_group",
                    "rows",
                    "metrics",
                    "close",
                )
            ):
                raise TypeError("q4_bank does not expose the Q4 execution contract")

        self.torch = torch
        self.source = source
        self.device = torch.device(device)
        self.compute_dtype = dtype
        self.max_resident_bytes = max_resident_bytes
        self.gc_interval_boundaries = gc_interval_boundaries
        initial_rss = _process_rss_bytes() if gc_rss_limit_bytes is None else None
        if gc_rss_limit_bytes is None:
            rss_headroom = max(
                self.DEFAULT_GC_RSS_HEADROOM_BYTES,
                2 * max_resident_bytes,
            )
            gc_rss_limit_bytes = (
                rss_headroom if initial_rss is None else initial_rss + rss_headroom
            )
        self.gc_rss_limit_bytes = gc_rss_limit_bytes
        self.close_source = close_source
        self.causal_tensor_reader = causal_tensor_reader
        self.exact_head_index = exact_head_index
        self.q4_bank = q4_bank
        if q4_bank is not None:
            self.WEIGHT_CACHE_POLICY = self.Q4_WEIGHT_CACHE_POLICY
        self.source_identity = validate_source_identity(
            getattr(source, "repo_id", None),
            getattr(source, "revision", None),
            require_identity=require_source_identity,
        )
        self._stats = PagerMetrics()
        if initial_rss is not None:
            self._stats.gc_last_observed_rss_bytes = initial_rss
            self._stats.gc_peak_observed_rss_bytes = initial_rss
        self._lock = threading.RLock()
        self._closed = False

    def _source_access_scope(self, *, tensor: str, read_kind: str) -> Any:
        scope = getattr(self.source, "access_scope", None)
        if not callable(scope):
            return nullcontext()
        return scope(tensor=tensor, read_kind=read_kind)

    def attach_exact_head_index(self, index: Any | None) -> None:
        """Attach or detach one exact, target-bound LM-head accelerator."""

        with self._lock:
            self._ensure_open()
            if index is not None and not callable(getattr(index, "topk_logits", None)):
                raise TypeError("exact head index must expose topk_logits()")
            if index is not None and self.q4_bank is not None:
                raise ValueError("Q4 execution and the exact BF16 head are exclusive")
            self.exact_head_index = index

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
        target_materialized: bool = True,
    ) -> int:
        target_bytes = (
            numel * self._dtype_bytes(target_dtype) if target_materialized else 0
        )
        planned = payload_bytes + target_bytes
        if planned > self.max_resident_bytes:
            raise Qwen38PagerError(
                f"{label} needs {planned} resident bytes (source payload plus "
                f"{str(target_dtype).removeprefix('torch.')} tensor), limit is "
                f"{self.max_resident_bytes}"
            )
        self._stats.peak_planned_resident_bytes = max(
            self._stats.peak_planned_resident_bytes, planned
        )
        return planned

    def _preflight_q4_linear(
        self,
        layout: _TensorLayout,
        *,
        input_rows: int,
        output_dtype: Any | None,
        label: str,
        extra_bytes: int = 0,
    ) -> int:
        if input_rows <= 0 or len(layout.shape) != 2:
            raise Qwen38PagerError("Q4 linear preflight received an invalid shape")
        output_rows, input_columns = layout.shape
        final_dtype = self.compute_dtype if output_dtype is None else output_dtype
        final_item_bytes = self._dtype_bytes(final_dtype)
        input_float_bytes = input_rows * input_columns * 4
        input_q8_bytes = input_rows * (input_columns // 32) * 34
        output_float_bytes = input_rows * output_rows * 4
        output_final_bytes = input_rows * output_rows * final_item_bytes
        planned = (
            input_float_bytes
            + input_q8_bytes
            + output_float_bytes
            + output_final_bytes
            + extra_bytes
        )
        if planned > self.max_resident_bytes:
            raise Qwen38PagerError(
                f"{label} needs {planned} transient Q4 bytes, limit is "
                f"{self.max_resident_bytes}"
            )
        self._stats.peak_planned_resident_bytes = max(
            self._stats.peak_planned_resident_bytes,
            planned,
        )
        return planned

    def _preflight_q4_rows(
        self,
        layout: _TensorLayout,
        token_ids: tuple[int, ...],
    ) -> int:
        unique_rows = len(set(token_ids))
        columns = layout.shape[1]
        decoded_bytes = unique_rows * columns * 4
        unique_final_bytes = (
            unique_rows * columns * self._dtype_bytes(self.compute_dtype)
        )
        restored_bytes = (
            0
            if unique_rows == len(token_ids)
            else len(token_ids) * (columns * self._dtype_bytes(self.compute_dtype) + 8)
        )
        planned = decoded_bytes + unique_final_bytes + restored_bytes
        if planned > self.max_resident_bytes:
            raise Qwen38PagerError(
                f"selected Q4 rows from {layout.name!r} need {planned} transient "
                f"bytes, limit is {self.max_resident_bytes}"
            )
        self._stats.peak_planned_resident_bytes = max(
            self._stats.peak_planned_resident_bytes,
            planned,
        )
        return planned

    def _preflight_q4_linear_rows(
        self,
        layout: _TensorLayout,
        *,
        input_rows: int,
        selected_rows: int,
        output_dtype: Any | None,
        label: str,
        extra_bytes: int = 0,
    ) -> int:
        if input_rows <= 0 or selected_rows <= 0 or len(layout.shape) != 2:
            raise Qwen38PagerError(
                "Q4 selected-row linear preflight received an invalid shape"
            )
        if selected_rows > layout.shape[0]:
            raise Qwen38PagerError("Q4 selected-row count exceeds the matrix")
        final_dtype = self.compute_dtype if output_dtype is None else output_dtype
        input_columns = layout.shape[1]
        planned = (
            input_rows * input_columns * 4
            + input_rows * (input_columns // 32) * 34
            + input_rows * selected_rows * 4
            + input_rows * selected_rows * self._dtype_bytes(final_dtype)
            + extra_bytes
        )
        if planned > self.max_resident_bytes:
            raise Qwen38PagerError(
                f"{label} needs {planned} transient selected-row Q4 bytes, "
                f"limit is {self.max_resident_bytes}"
            )
        self._stats.peak_planned_resident_bytes = max(
            self._stats.peak_planned_resident_bytes,
            planned,
        )
        return planned

    def _preflight_q4_group(
        self,
        layouts: tuple[_TensorLayout, ...],
        *,
        input_rows: int,
        output_dtype: Any | None,
        label: str,
    ) -> int:
        if not layouts or input_rows <= 0:
            raise Qwen38PagerError("Q4 group preflight received an invalid shape")
        input_columns = layouts[0].shape[1]
        if any(
            len(layout.shape) != 2 or layout.shape[1] != input_columns
            for layout in layouts
        ):
            raise Qwen38PagerError("Q4 grouped matrix widths differ")
        final_dtype = self.compute_dtype if output_dtype is None else output_dtype
        input_float_bytes = input_rows * input_columns * 4
        input_q8_bytes = input_rows * (input_columns // 32) * 34
        output_rows = sum(layout.shape[0] for layout in layouts)
        output_float_bytes = input_rows * output_rows * 4
        output_final_bytes = input_rows * output_rows * self._dtype_bytes(final_dtype)
        planned = (
            input_float_bytes + input_q8_bytes + output_float_bytes + output_final_bytes
        )
        if planned > self.max_resident_bytes:
            raise Qwen38PagerError(
                f"{label} needs {planned} transient grouped Q4 bytes, limit is "
                f"{self.max_resident_bytes}"
            )
        self._stats.peak_planned_resident_bytes = max(
            self._stats.peak_planned_resident_bytes,
            planned,
        )
        return planned

    def _preflight_q4_mlp(
        self,
        layouts: tuple[_TensorLayout, _TensorLayout, _TensorLayout],
        *,
        input_rows: int,
    ) -> int:
        gate, up, down = layouts
        if (
            input_rows <= 0
            or gate.shape != up.shape
            or len(gate.shape) != 2
            or len(down.shape) != 2
            or down.shape[1] != gate.shape[0]
        ):
            raise Qwen38PagerError("Q4 full MLP preflight shape is invalid")
        hidden = gate.shape[1]
        intermediate = gate.shape[0]
        output = down.shape[0]
        planned = input_rows * (
            hidden * 4
            + (hidden // 32) * 34
            + intermediate * 4
            + (intermediate // 32) * 34
            + output * (4 + self._dtype_bytes(self.compute_dtype))
        )
        if planned > self.max_resident_bytes:
            raise Qwen38PagerError(
                f"Q4 full MLP needs {planned} transient bytes, limit is "
                f"{self.max_resident_bytes}"
            )
        self._stats.peak_planned_resident_bytes = max(
            self._stats.peak_planned_resident_bytes,
            planned,
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
        item_bytes = {"BF16": 2, "F32": 4}.get(dtype)
        if item_bytes is None:
            raise Qwen38PagerError(
                f"Qwen3.5 tensor {name!r} must be BF16 or F32, got {dtype or 'missing'}"
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
        if payload_bytes != numel * item_bytes:
            raise Qwen38PagerError(
                f"tensor {name!r} shape requires {numel * item_bytes} "
                f"{dtype} bytes, "
                f"offsets contain {payload_bytes}"
            )
        return _TensorLayout(
            name=name,
            shape=tuple(shape),
            dtype=dtype,
            item_bytes=item_bytes,
            shard=shard,
            absolute=data_start + begin,
            payload_bytes=payload_bytes,
        )

    def _decode_tensor(
        self,
        raw: bytes,
        *,
        shape: tuple[int, ...],
        source_dtype: str,
        dtype: Any,
        device: Any,
        name: str,
        zero_copy_cpu: bool = False,
    ) -> Any:
        numel = 1
        for dimension in shape:
            numel *= dimension
        item_bytes = {"BF16": 2, "F32": 4}.get(source_dtype)
        if item_bytes is None:  # pragma: no cover - guarded by _layout.
            raise Qwen38PagerError(
                f"unsupported source dtype for {name!r}: {source_dtype}"
            )
        expected_bytes = numel * item_bytes
        if len(raw) != expected_bytes:
            raise Qwen38PagerError(
                f"short {source_dtype} payload for {name!r}: "
                f"{len(raw)}/{expected_bytes} bytes"
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
            storage = self.torch.frombuffer(
                raw,
                dtype=(
                    self.torch.bfloat16
                    if source_dtype == "BF16"
                    else self.torch.float32
                ),
            )
        storage = storage.reshape(shape)
        target_device = self.torch.device(device)
        if zero_copy_cpu and target_device.type == "cpu" and dtype == storage.dtype:
            result = storage
            self._stats.zero_copy_tensor_reads += 1
            self._stats.zero_copy_bytes_avoided += expected_bytes
        elif target_device.type == "cpu" and dtype == storage.dtype:
            result = storage.clone()
        else:
            result = storage.to(device=target_device, dtype=dtype)
        result.requires_grad_(False)
        return result

    def _direct_fill_supported(
        self,
        *,
        source_dtype: Any,
        dtype: Any,
        device: Any,
    ) -> bool:
        return (
            self.torch.device(device).type == "cpu"
            and dtype == source_dtype
            and getattr(self.source, "raw_bytes_into_available", False) is True
            and callable(getattr(self.source, "raw_bytes_into", None))
        )

    def _direct_fill_existing(
        self,
        layout: _TensorLayout,
        *,
        relative_offset: int,
        result: Any,
    ) -> bool:
        source_dtype = (
            self.torch.bfloat16 if layout.dtype == "BF16" else self.torch.float32
        )
        if not self._direct_fill_supported(
            source_dtype=source_dtype,
            dtype=result.dtype,
            device=result.device,
        ):
            return False
        if not result.is_contiguous():
            raise Qwen38PagerError("direct tensor target must be contiguous")
        byte_view = memoryview(result.view(self.torch.uint8).numpy()).cast("B")
        if self.causal_tensor_reader is None:
            receipt = self.source.raw_bytes_into(
                layout.shard,
                layout.absolute + relative_offset,
                byte_view,
            )
            try:
                length = receipt.length
            except AttributeError as exc:
                raise Qwen38PagerError(
                    "direct tensor source returned an invalid receipt"
                ) from exc
        else:
            receipt = self.causal_tensor_reader.read_tensor_into(
                layout.name,
                byte_view,
                relative_offset=relative_offset,
            )
            if (
                receipt.plan.shard != layout.shard
                or receipt.plan.absolute_offset != layout.absolute
                or receipt.relative_offset != relative_offset
            ):
                raise Qwen38PagerError(
                    "causal direct-fill receipt disagrees with layout"
                )
            length = receipt.length
        if length != byte_view.nbytes:
            raise Qwen38PagerError("direct tensor source returned a short receipt")
        self._stats.direct_tensor_fills += 1
        self._stats.direct_tensor_fill_bytes += byte_view.nbytes
        return True

    def _direct_fill_tensor(
        self,
        layout: _TensorLayout,
        *,
        relative_offset: int,
        shape: tuple[int, ...],
        dtype: Any,
        device: Any,
    ) -> Any | None:
        source_dtype = (
            self.torch.bfloat16 if layout.dtype == "BF16" else self.torch.float32
        )
        if not self._direct_fill_supported(
            source_dtype=source_dtype,
            dtype=dtype,
            device=device,
        ):
            return None
        result = self.torch.empty(shape, device=device, dtype=dtype)
        if not self._direct_fill_existing(
            layout,
            relative_offset=relative_offset,
            result=result,
        ):  # pragma: no cover - identical support check above.
            return None
        return result

    def _read_tensor(
        self,
        name: str,
        *,
        dtype: Any,
        device: Any,
        zero_copy_cpu: bool = False,
    ) -> Any:
        layout = self._layout(name)
        source_dtype = (
            self.torch.bfloat16 if layout.dtype == "BF16" else self.torch.float32
        )
        zero_copy = (
            zero_copy_cpu
            and self.torch.device(device).type == "cpu"
            and dtype == source_dtype
        )
        self._preflight_resident(
            payload_bytes=layout.payload_bytes,
            numel=layout.numel,
            target_dtype=dtype,
            label=name,
            target_materialized=not zero_copy,
        )
        result = self._direct_fill_tensor(
            layout,
            relative_offset=0,
            shape=layout.shape,
            dtype=dtype,
            device=device,
        )
        if result is None:
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
                    raise Qwen38PagerError(
                        "causal tensor receipt disagrees with layout"
                    )
                raw = receipt.part
            result = self._decode_tensor(
                raw,
                shape=layout.shape,
                source_dtype=layout.dtype,
                dtype=dtype,
                device=device,
                name=name,
                zero_copy_cpu=zero_copy,
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
        row_bytes = columns * layout.item_bytes
        payload_bytes = n_rows * row_bytes
        numel = n_rows * columns
        source_dtype = (
            self.torch.bfloat16 if layout.dtype == "BF16" else self.torch.float32
        )
        direct_fill = self._direct_fill_supported(
            source_dtype=source_dtype,
            dtype=dtype,
            device=device,
        )
        self._preflight_resident(
            payload_bytes=payload_bytes,
            numel=numel,
            target_dtype=dtype,
            label=f"{name}[{start_row}:{start_row + n_rows}]",
            target_materialized=not direct_fill,
        )
        relative_offset = start_row * row_bytes
        with self._source_access_scope(tensor=name, read_kind="rows"):
            result = self._direct_fill_tensor(
                layout,
                relative_offset=relative_offset,
                shape=(n_rows, columns),
                dtype=dtype,
                device=device,
            )
            if result is None:
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
                result = self._decode_tensor(
                    raw,
                    shape=(n_rows, columns),
                    source_dtype=layout.dtype,
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
        zero_copy_cpu: bool = False,
    ) -> Any:
        """Read one BF16/F32 tensor with exact range and resident preflight."""

        with self._lock:
            self._ensure_open()
            if not isinstance(zero_copy_cpu, bool):
                raise TypeError("zero_copy_cpu must be boolean")
            target_dtype = self.compute_dtype if dtype is None else dtype
            if target_dtype not in {
                self.torch.float16,
                self.torch.bfloat16,
                self.torch.float32,
            }:
                raise ValueError("dtype must be float16, bfloat16, or float32")
            target_device = self.device if device is None else self.torch.device(device)
            with self._source_access_scope(tensor=name, read_kind="tensor"):
                return self._read_tensor(
                    name,
                    dtype=target_dtype,
                    device=target_device,
                    zero_copy_cpu=zero_copy_cpu,
                )

    def tensor_rows(self, name: str, row_ids: Iterable[int]) -> Any:
        """Read arbitrary 2D rows through the active authenticated range plane.

        Row order and repeats are preserved. Consecutive addresses are
        coalesced, and the resident preflight covers the complete selected-row
        result rather than each source range independently.
        """

        with self._lock:
            self._ensure_open()
            ids = self._validate_ids(name, row_ids)
            with self._source_access_scope(tensor=name, read_kind="selected-rows"):
                return self._selected_rows(name, ids)

    def linear(
        self,
        x: Any,
        name: str,
        *,
        output_dtype: Any | None = None,
        weight_observer: Callable[[Any, Any], None] | None = None,
    ) -> Any:
        """Apply one bias-free checkpoint matrix and release it before return."""

        with self._lock:
            self._ensure_open()
            if weight_observer is not None and not callable(weight_observer):
                raise TypeError("weight_observer must be callable or None")
            if not isinstance(x, self.torch.Tensor):
                x = self.torch.as_tensor(x)
            weight_name = self._weight_name(name)
            compute_x = x.to(device=self.device, dtype=self.compute_dtype)
            q4_bank = self.q4_bank
            if q4_bank is not None and q4_bank.has(weight_name):
                layout = self._layout(weight_name)
                if len(layout.shape) != 2:
                    raise Qwen38PagerError(f"linear weight {weight_name!r} must be 2D")
                if compute_x.shape[-1] != layout.shape[1]:
                    raise Qwen38PagerError(
                        f"linear input width {compute_x.shape[-1]} disagrees with "
                        f"{weight_name}{layout.shape}"
                    )
                if weight_observer is not None:
                    raise Qwen38PagerError(
                        "Q4 execution cannot expose a materialized floating weight"
                    )
                self._preflight_q4_linear(
                    layout,
                    input_rows=compute_x.numel() // compute_x.shape[-1],
                    output_dtype=output_dtype,
                    label=weight_name,
                )
                result = q4_bank.linear(
                    compute_x,
                    weight_name,
                    output_dtype=output_dtype,
                )
                self._stats.linear_calls += 1
                return result
            with self._source_access_scope(tensor=weight_name, read_kind="linear"):
                weight = self._read_tensor(
                    weight_name,
                    dtype=self.compute_dtype,
                    device=self.device,
                    zero_copy_cpu=True,
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
                if weight_observer is not None:
                    weight_observer(weight, result)
                self._stats.linear_calls += 1
                return result
            finally:
                del weight
                self._stats.materialized_weight_releases += 1

    def linear_group(
        self,
        x: Any,
        names: Iterable[str],
        *,
        output_dtype: Any | None = None,
    ) -> tuple[Any, ...]:
        """Apply multiple same-input matrices with one packed activation pass."""

        with self._lock:
            self._ensure_open()
            try:
                prefixes = tuple(names)
            except TypeError as exc:
                raise TypeError("linear group names must be iterable") from exc
            if len(prefixes) < 2:
                raise ValueError("linear group requires at least two matrices")
            weight_names = tuple(self._weight_name(name) for name in prefixes)
            if len(set(weight_names)) != len(weight_names):
                raise ValueError("linear group names must be distinct")
            if not isinstance(x, self.torch.Tensor):
                x = self.torch.as_tensor(x)
            compute_x = x.to(device=self.device, dtype=self.compute_dtype)
            q4_bank = self.q4_bank
            if q4_bank is None or not all(q4_bank.has(name) for name in weight_names):
                return tuple(
                    self.linear(
                        compute_x,
                        name,
                        output_dtype=output_dtype,
                    )
                    for name in weight_names
                )
            layouts = tuple(self._layout(name) for name in weight_names)
            input_columns = layouts[0].shape[1]
            if compute_x.shape[-1] != input_columns or any(
                layout.shape[1] != input_columns for layout in layouts
            ):
                raise Qwen38PagerError("grouped linear input widths disagree")
            self._preflight_q4_group(
                layouts,
                input_rows=compute_x.numel() // compute_x.shape[-1],
                output_dtype=output_dtype,
                label=" + ".join(weight_names),
            )
            results = q4_bank.linear_group(
                compute_x,
                weight_names,
                output_dtype=output_dtype,
            )
            self._stats.linear_calls += len(weight_names)
            self._stats.grouped_linear_calls += 1
            self._stats.grouped_linear_matrices += len(weight_names)
            return results

    def linear_group_many(
        self,
        inputs: Iterable[Any],
        names: Iterable[str],
        *,
        output_dtype: Any | None = None,
        packed: bool = False,
    ) -> tuple[tuple[Any, ...], ...]:
        """Apply a same-input matrix group to independent token rows once."""

        with self._lock:
            self._ensure_open()
            try:
                values = tuple(inputs)
                prefixes = tuple(names)
            except TypeError as exc:
                raise TypeError("grouped inputs and names must be iterable") from exc
            if len(values) < 2:
                raise ValueError("linear_group_many requires at least two inputs")
            if len(prefixes) < 2:
                raise ValueError("linear_group_many requires at least two matrices")
            if not isinstance(packed, bool):
                raise TypeError("packed must be boolean")
            weight_names = tuple(self._weight_name(name) for name in prefixes)
            if len(set(weight_names)) != len(weight_names):
                raise ValueError("linear group names must be distinct")
            q4_bank = self.q4_bank
            if q4_bank is None or not all(q4_bank.has(name) for name in weight_names):
                return tuple(
                    self.linear_many(
                        values,
                        name,
                        output_dtype=output_dtype,
                        packed=packed,
                    )
                    for name in weight_names
                )

            layouts = tuple(self._layout(name) for name in weight_names)
            input_width = layouts[0].shape[1]
            if any(
                len(layout.shape) != 2 or layout.shape[1] != input_width
                for layout in layouts
            ):
                raise Qwen38PagerError("grouped linear input widths disagree")
            compute_inputs = []
            for index, value in enumerate(values):
                if not isinstance(value, self.torch.Tensor):
                    try:
                        value = self.torch.as_tensor(value)
                    except (TypeError, ValueError) as exc:
                        raise TypeError(
                            f"linear input {index} cannot be converted to a tensor"
                        ) from exc
                if value.ndim < 1 or value.shape[-1] != input_width:
                    raise Qwen38PagerError(
                        f"linear input {index} disagrees with grouped weights"
                    )
                compute_inputs.append(
                    value.to(device=self.device, dtype=self.compute_dtype)
                )
            shapes = tuple(tuple(value.shape) for value in compute_inputs)
            counts = tuple(
                value.numel() // value.shape[-1] for value in compute_inputs
            )
            combined = self.torch.cat(
                tuple(value.reshape(-1, input_width) for value in compute_inputs),
                dim=0,
            )
            grouped = self.linear_group(
                combined,
                weight_names,
                output_dtype=output_dtype,
            )
            if not packed:
                self._stats.linear_calls += (
                    (len(compute_inputs) - 1) * len(weight_names)
                )
            outputs = []
            for result, layout in zip(grouped, layouts, strict=True):
                rows = []
                offset = 0
                for shape, count in zip(shapes, counts, strict=True):
                    rows.append(
                        result[offset : offset + count].reshape(
                            (*shape[:-1], layout.shape[0])
                        )
                    )
                    offset += count
                if offset != result.shape[0]:
                    raise Qwen38PagerError("grouped linear lost an input row")
                outputs.append(tuple(rows))
            return tuple(outputs)

    def mlp(
        self,
        x: Any,
        names: Iterable[str],
        *,
        activation_page_topk: int | None = None,
    ) -> Any:
        """Execute packed BF16 SwiGLU and optionally return ranked/total energy."""

        with self._lock:
            self._ensure_open()
            prefixes = tuple(names)
            if len(prefixes) != 3:
                raise ValueError("MLP requires Gate, Up, and Down names")
            weight_names = tuple(self._weight_name(name) for name in prefixes)
            q4_bank = self.q4_bank
            if (
                q4_bank is None
                or not callable(getattr(q4_bank, "mlp", None))
                or not all(q4_bank.has(name) for name in weight_names)
                or self.compute_dtype != self.torch.bfloat16
            ):
                raise Qwen38PagerError("native Q4 full MLP is unavailable")
            if not isinstance(x, self.torch.Tensor):
                x = self.torch.as_tensor(x)
            compute = x.to(device=self.device, dtype=self.compute_dtype)
            layouts = tuple(self._layout(name) for name in weight_names)
            self._preflight_q4_mlp(
                layouts,  # type: ignore[arg-type]
                input_rows=compute.numel() // compute.shape[-1],
            )
            result = q4_bank.mlp(
                compute,
                weight_names,
                output_dtype=self.compute_dtype,
                activation_page_topk=activation_page_topk,
            )
            self._stats.linear_calls += 3
            self._stats.grouped_linear_calls += 1
            self._stats.grouped_linear_matrices += 3
            return result

    def mlp_selected_pages(
        self,
        x: Any,
        names: Iterable[str],
        page_ids: Any,
    ) -> Any:
        """Execute only explicit 64-neuron pages of a packed BF16 SwiGLU MLP."""

        with self._lock:
            self._ensure_open()
            prefixes = tuple(names)
            if len(prefixes) != 3:
                raise ValueError("MLP requires Gate, Up, and Down names")
            weight_names = tuple(self._weight_name(name) for name in prefixes)
            q4_bank = self.q4_bank
            if (
                q4_bank is None
                or not callable(getattr(q4_bank, "mlp_selected_pages", None))
                or not all(q4_bank.has(name) for name in weight_names)
                or self.compute_dtype != self.torch.bfloat16
            ):
                raise Qwen38PagerError("native Q4 selected-page MLP is unavailable")
            if not isinstance(x, self.torch.Tensor):
                x = self.torch.as_tensor(x)
            compute = x.to(device=self.device, dtype=self.compute_dtype)
            layouts = tuple(self._layout(name) for name in weight_names)
            self._preflight_q4_mlp(
                layouts,  # type: ignore[arg-type]
                input_rows=compute.numel() // compute.shape[-1],
            )
            result = q4_bank.mlp_selected_pages(
                compute,
                weight_names,
                page_ids,
                output_dtype=self.compute_dtype,
            )
            self._stats.linear_calls += 3
            self._stats.grouped_linear_calls += 1
            self._stats.grouped_linear_matrices += 3
            return result

    def linear_many(
        self,
        inputs: Iterable[Any],
        name: str,
        *,
        output_dtype: Any | None = None,
        packed: bool = False,
        weight_observer: Callable[[Any, tuple[Any, ...]], None] | None = None,
    ) -> tuple[Any, ...]:
        """Apply one matrix to multiple inputs with sequential or packed GEMMs.

        Every input is converted and shape-checked before the checkpoint range
        is read.  The matrix is then materialized exactly once and passed to a
        separate ``F.linear`` invocation for each input by default. ``packed``
        concatenates their rows into one physical GEMM for the explicit fast
        runtime; output shapes and ordering remain unchanged.
        """

        with self._lock:
            self._ensure_open()
            try:
                values = tuple(inputs)
            except TypeError as exc:
                raise TypeError("inputs must be an iterable") from exc
            if len(values) < 2:
                raise ValueError("linear_many requires at least two inputs")
            if not isinstance(packed, bool):
                raise TypeError("packed must be boolean")
            if weight_observer is not None and not callable(weight_observer):
                raise TypeError("weight_observer must be callable or None")

            weight_name = self._weight_name(name)
            layout = self._layout(weight_name)
            if len(layout.shape) != 2:
                raise Qwen38PagerError(f"linear weight {weight_name!r} must be 2D")
            input_width = layout.shape[1]
            compute_inputs: list[Any] = []
            for index, value in enumerate(values):
                if not isinstance(value, self.torch.Tensor):
                    try:
                        value = self.torch.as_tensor(value)
                    except (TypeError, ValueError) as exc:
                        raise TypeError(
                            f"linear input {index} cannot be converted to a tensor"
                        ) from exc
                if value.ndim < 1:
                    raise Qwen38PagerError(
                        f"linear input {index} must have at least one dimension"
                    )
                compute_x = value.to(
                    device=self.device,
                    dtype=self.compute_dtype,
                )
                if compute_x.shape[-1] != input_width:
                    raise Qwen38PagerError(
                        f"linear input {index} width {compute_x.shape[-1]} "
                        f"disagrees with {weight_name}{layout.shape}"
                    )
                compute_inputs.append(compute_x)

            if output_dtype is not None:
                # Validate the conversion before a source byte is read.  This
                # accepts exactly the dtype forms understood by Tensor.to.
                self.torch.empty((), device=self.device).to(dtype=output_dtype)

            q4_bank = self.q4_bank
            if q4_bank is not None and q4_bank.has(weight_name):
                if weight_observer is not None:
                    raise Qwen38PagerError(
                        "Q4 execution cannot expose a materialized floating weight"
                    )
                shapes = [tuple(value.shape) for value in compute_inputs]
                counts = [value.numel() // value.shape[-1] for value in compute_inputs]
                self._preflight_q4_linear(
                    layout,
                    input_rows=sum(counts),
                    output_dtype=output_dtype,
                    label=weight_name,
                )
                combined = self.torch.cat(
                    [value.reshape(-1, input_width) for value in compute_inputs],
                    dim=0,
                )
                packed_result = q4_bank.linear(
                    combined,
                    weight_name,
                    output_dtype=output_dtype,
                )
                results = []
                offset = 0
                for shape, count in zip(shapes, counts, strict=True):
                    results.append(
                        packed_result[offset : offset + count].reshape(
                            (*shape[:-1], layout.shape[0])
                        )
                    )
                    offset += count
                self._stats.linear_calls += 1 if packed else len(compute_inputs)
                if packed:
                    self._stats.packed_linear_calls += 1
                    self._stats.packed_linear_rows += sum(counts)
                return tuple(results)

            with self._source_access_scope(
                tensor=weight_name,
                read_kind="linear-many",
            ):
                weight = self._read_tensor(
                    weight_name,
                    dtype=self.compute_dtype,
                    device=self.device,
                    zero_copy_cpu=True,
                )
            try:
                if packed:
                    shapes = [tuple(value.shape) for value in compute_inputs]
                    counts = [
                        value.numel() // value.shape[-1] for value in compute_inputs
                    ]
                    combined = self.torch.cat(
                        [value.reshape(-1, input_width) for value in compute_inputs],
                        dim=0,
                    )
                    packed_result = self.torch.nn.functional.linear(combined, weight)
                    if output_dtype is not None:
                        packed_result = packed_result.to(dtype=output_dtype)
                    results = []
                    offset = 0
                    for shape, count in zip(shapes, counts, strict=True):
                        results.append(
                            packed_result[offset : offset + count].reshape(
                                (*shape[:-1], layout.shape[0])
                            )
                        )
                        offset += count
                    self._stats.linear_calls += 1
                    self._stats.packed_linear_calls += 1
                    self._stats.packed_linear_rows += sum(counts)
                    final = tuple(results)
                    if weight_observer is not None:
                        weight_observer(weight, final)
                    return final
                results: list[Any] = []
                for compute_x in compute_inputs:
                    result = self.torch.nn.functional.linear(compute_x, weight)
                    if output_dtype is not None:
                        result = result.to(dtype=output_dtype)
                    results.append(result)
                    self._stats.linear_calls += 1
                final = tuple(results)
                if weight_observer is not None:
                    weight_observer(weight, final)
                return final
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
        q4_bank = self.q4_bank
        if q4_bank is not None and q4_bank.has(name):
            self._preflight_q4_rows(self._layout(name), token_ids)
            return q4_bank.rows(name, token_ids, dtype=self.compute_dtype)
        if not token_ids:
            columns = self._layout(name).shape[1]
            return self.torch.empty(
                (0, columns), device=self.device, dtype=self.compute_dtype
            )
        layout = self._layout(name)
        columns = layout.shape[1]
        runs = self._consecutive_runs(token_ids)
        unique_ids = tuple(
            value for start, stop in runs for value in range(start, stop)
        )
        source_dtype = (
            self.torch.bfloat16 if layout.dtype == "BF16" else self.torch.float32
        )
        direct_fill = self._direct_fill_supported(
            source_dtype=source_dtype,
            dtype=self.compute_dtype,
            device=self.device,
        )
        unique_target_bytes = (
            len(unique_ids) * columns * self._dtype_bytes(self.compute_dtype)
        )
        largest_payload = (
            max(stop - start for start, stop in runs) * columns * layout.item_bytes
        )
        output_bytes = len(token_ids) * columns * self._dtype_bytes(self.compute_dtype)
        ordered_unique = token_ids == unique_ids
        if direct_fill:
            restore_bytes = (
                0
                if ordered_unique
                else len(token_ids) * self._dtype_bytes(self.torch.long)
            )
            planned = unique_target_bytes + (
                0 if ordered_unique else output_bytes + restore_bytes
            )
        else:
            planned = unique_target_bytes + max(largest_payload, output_bytes)
        if planned > self.max_resident_bytes:
            raise Qwen38PagerError(
                f"selected rows from {name!r} need {planned} resident bytes, "
                f"limit is {self.max_resident_bytes}"
            )
        self._stats.peak_planned_resident_bytes = max(
            self._stats.peak_planned_resident_bytes, planned
        )
        if direct_fill:
            unique_rows = self.torch.empty(
                (len(unique_ids), columns),
                device=self.device,
                dtype=self.compute_dtype,
            )
            unique_offset = 0
            row_bytes = columns * layout.item_bytes
            for start, stop in runs:
                count = stop - start
                target = unique_rows[unique_offset : unique_offset + count]
                if not self._direct_fill_existing(
                    layout,
                    relative_offset=start * row_bytes,
                    result=target,
                ):  # pragma: no cover - direct_fill was checked above.
                    raise Qwen38PagerError("direct selected-row fill disappeared")
                payload_bytes = count * row_bytes
                self._stats.row_reads += 1
                self._stats.logical_weight_bytes += payload_bytes
                self._stats.materialized_tensor_bytes += payload_bytes
                unique_offset += count
            if ordered_unique:
                return unique_rows
            by_id = {value: index for index, value in enumerate(unique_ids)}
            restore = self.torch.tensor(
                [by_id[value] for value in token_ids],
                device=self.device,
                dtype=self.torch.long,
            )
            return unique_rows.index_select(0, restore)
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
            with self._source_access_scope(tensor=name, read_kind="embedding"):
                rows = self._selected_rows(name, ids)
            if self.q4_bank is not None and self.q4_bank.has(name):
                self.q4_bank.record_embedding(len(ids))
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
            columns = self._layout(name).shape[1]
            if compute_hidden.shape[-1] != columns:
                raise Qwen38PagerError("hidden width disagrees with LM head")
            self._preflight_head_score(
                query_rows=compute_hidden.numel() // columns,
                head_rows=len(ids),
                columns=columns,
            )
            with self._source_access_scope(tensor=name, read_kind="candidate-head"):
                rows = self._selected_rows(name, ids)
            if rows.shape[1] != columns:
                del rows
                raise Qwen38PagerError("hidden width disagrees with LM head")
            try:
                if self.q4_bank is not None and self.q4_bank.has(name):
                    self.q4_bank.record_candidates(len(ids))
                return self._score_head_rows(compute_hidden, rows)
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
        """Compute exact global top-k through a certificate or the full scan."""

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
            q4_bank = self.q4_bank
            if q4_bank is not None and q4_bank.has(name):
                # Scan the exact packed head in bounded row intervals. The
                # native selected-row kernel uses the same Q8 input quantizer
                # and dot product as the former all-vocabulary call, while each
                # consumed mmap interval can leave RSS immediately.
                native_topk = getattr(q4_bank, "topk", None)
                if (
                    callable(native_topk)
                    and self.compute_dtype == self.torch.bfloat16
                    and k <= 256
                ):
                    active_rows = min(max(block_rows, 8192), vocab)
                    self._preflight_q4_linear_rows(
                        layout,
                        input_rows=compute_hidden.numel() // columns,
                        selected_rows=active_rows,
                        output_dtype=self.compute_dtype,
                        label=f"{name} head",
                        extra_bytes=active_rows * 16,
                    )
                    values, token_ids = native_topk(
                        compute_hidden,
                        name,
                        k=k,
                        block_rows=block_rows,
                        output_dtype=self.compute_dtype,
                    )
                    self._stats.head_rows += vocab
                    if progress is not None:
                        progress(
                            {
                                "start_row": 0,
                                "rows": vocab,
                                "rows_done": vocab,
                                "vocab_rows": vocab,
                            }
                        )
                elif all(
                    callable(getattr(q4_bank, method, None))
                    for method in ("linear_rows", "discard_rows")
                ):
                    active_rows = min(block_rows, vocab)
                    self._preflight_q4_linear_rows(
                        layout,
                        input_rows=compute_hidden.numel() // columns,
                        selected_rows=active_rows,
                        output_dtype=self.compute_dtype,
                        label=f"{name} head",
                        extra_bytes=active_rows * 48,
                    )
                    values, token_ids = self._topk_logits_q4_rows_locked(
                        compute_hidden,
                        k=k,
                        name=name,
                        block_rows=block_rows,
                        progress=progress,
                    )
                else:  # narrow third-party test doubles retain the old ABI.
                    self._preflight_q4_linear(
                        layout,
                        input_rows=compute_hidden.numel() // columns,
                        output_dtype=self.compute_dtype,
                        label=f"{name} head",
                        extra_bytes=vocab * 48,
                    )
                    logits = q4_bank.linear(
                        compute_hidden,
                        name,
                        output_dtype=self.compute_dtype,
                    )
                    all_ids = self.torch.arange(
                        vocab,
                        device=self.device,
                        dtype=self.torch.long,
                    ).expand_as(logits)
                    values, token_ids = self._stable_topk(logits, all_ids, k)
                    del logits
                    self._stats.head_rows += vocab
                q4_bank.record_head()
                return values, token_ids
            self._preflight_head_score(
                query_rows=compute_hidden.numel() // columns,
                head_rows=min(block_rows, vocab),
                columns=columns,
            )
            index = self.exact_head_index
            if index is not None and progress is None:
                indexed = index.topk_logits(
                    self,
                    compute_hidden,
                    k=k,
                    name=name,
                    block_rows=block_rows,
                )
                if indexed is not None:
                    try:
                        values, token_ids = indexed
                    except (TypeError, ValueError) as exc:
                        raise Qwen38PagerError(
                            "exact head index returned an invalid result"
                        ) from exc
                    expected = (*compute_hidden.shape[:-1], k)
                    if (
                        not isinstance(values, self.torch.Tensor)
                        or not isinstance(token_ids, self.torch.Tensor)
                        or tuple(values.shape) != expected
                        or tuple(token_ids.shape) != expected
                        or values.device != self.device
                        or token_ids.device != self.device
                        or values.dtype != self.compute_dtype
                        or token_ids.dtype != self.torch.long
                    ):
                        raise Qwen38PagerError(
                            "exact head index result differs from the pager ABI"
                        )
                    return values, token_ids
            return self._topk_logits_full_locked(
                compute_hidden,
                k=k,
                name=name,
                block_rows=block_rows,
                progress=progress,
            )

    def _topk_logits_q4_rows_locked(
        self,
        compute_hidden: Any,
        *,
        k: int,
        name: str,
        block_rows: int,
        progress: Callable[[dict[str, int]], None] | None = None,
    ) -> tuple[Any, Any]:
        """Run one exact packed global top-k without a model-sized head spike."""

        q4_bank = self.q4_bank
        if q4_bank is None:  # pragma: no cover - private call contract.
            raise AssertionError("packed head scan requires a Q4 bank")
        layout = self._layout(name)
        vocab, columns = layout.shape
        leading_shape = tuple(compute_hidden.shape[:-1])
        flat = compute_hidden.reshape(-1, columns)
        best_values = None
        best_ids = None
        for start in range(0, vocab, block_rows):
            count = min(block_rows, vocab - start)
            row_ids = tuple(range(start, start + count))
            try:
                logits = q4_bank.linear_rows(
                    flat,
                    name,
                    row_ids,
                    output_dtype=self.compute_dtype,
                )
                token_ids = self.torch.arange(
                    start,
                    start + count,
                    device=logits.device,
                    dtype=self.torch.long,
                ).expand_as(logits)
                values, indices = self._stable_topk(
                    logits,
                    token_ids,
                    min(k, count),
                )
            finally:
                q4_bank.discard_rows(name, start, count)
            del logits
            self._stats.head_rows += count
            if best_values is None:
                best_values, best_ids = values, indices
            else:
                merged_values = self.torch.cat((best_values, values), dim=-1)
                merged_ids = self.torch.cat((best_ids, indices), dim=-1)
                best_values, best_ids = self._stable_topk(
                    merged_values,
                    merged_ids,
                    min(k, merged_values.shape[-1]),
                )
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

    def _topk_logits_full_locked(
        self,
        compute_hidden: Any,
        *,
        k: int,
        name: str,
        block_rows: int,
        progress: Callable[[dict[str, int]], None] | None = None,
    ) -> tuple[Any, Any]:
        """Run the canonical full scan; caller owns the pager lock and ABI checks."""

        layout = self._layout(name)
        vocab, columns = layout.shape
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
            logits = self._score_head_rows(flat, rows)
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
            else:
                merged_values = self.torch.cat((best_values, values), dim=-1)
                merged_ids = self.torch.cat((best_ids, indices), dim=-1)
                merge_k = min(k, merged_values.shape[-1])
                best_values, best_ids = self._stable_topk(
                    merged_values, merged_ids, merge_k
                )
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

    def _score_head_rows(self, hidden: Any, rows: Any) -> Any:
        """Canonical CPU head scorer: FP32 accumulate, BF16 output RNE."""

        if (
            self.exact_head_index is not None
            and self.device.type == "cpu"
            and self.compute_dtype == self.torch.bfloat16
            and hidden.dtype == self.torch.bfloat16
            and rows.dtype == self.torch.bfloat16
        ):
            return self.torch.nn.functional.linear(hidden.float(), rows.float()).to(
                self.torch.bfloat16
            )
        return self.torch.nn.functional.linear(hidden, rows)

    def _preflight_head_score(
        self,
        *,
        query_rows: int,
        head_rows: int,
        columns: int,
    ) -> int:
        """Bound the explicit FP32 scorer's simultaneous CPU tensors."""

        if (
            self.exact_head_index is None
            or self.device.type != "cpu"
            or self.compute_dtype != self.torch.bfloat16
        ):
            return 0
        planned = (
            head_rows * columns * (2 + 4)
            + query_rows * columns * 4
            + query_rows * head_rows * (4 + 2)
        )
        if planned > self.max_resident_bytes:
            raise Qwen38PagerError(
                f"canonical head scorer needs {planned} resident bytes, "
                f"limit is {self.max_resident_bytes}"
            )
        self._stats.peak_planned_resident_bytes = max(
            self._stats.peak_planned_resident_bytes,
            planned,
        )
        return planned

    def _collect_locked(self, kind: str) -> None:
        collected = gc.collect()
        self._stats.gc_collections += 1
        self._stats.gc_objects_collected += max(0, int(collected))
        if kind == "forced":
            self._stats.gc_collections_forced += 1
        elif kind == "pressure":
            self._stats.gc_collections_pressure += 1
        elif kind == "interval":
            self._stats.gc_collections_interval += 1
        elif kind == "fail_closed":
            self._stats.gc_collections_fail_closed += 1
        else:  # pragma: no cover - private call contract.
            raise AssertionError(f"unknown GC collection kind: {kind}")

    def _release_locked(self, *, force_gc: bool) -> None:
        self._stats.release_boundaries += 1
        if force_gc:
            if self.q4_bank is not None:
                release_touched = getattr(self.q4_bank, "release_touched", None)
                if callable(release_touched):
                    release_touched(force_prefetch=True)
            self._collect_locked("forced")
        elif self.q4_bank is not None:
            # Every model layer is already an execution boundary. Drop the
            # read-only packed pages touched by that layer so 64 layers never
            # accumulate into one model-sized RSS. The mappings and arithmetic
            # remain unchanged; subsequent layers/tokens fault the same local
            # bytes back through the OS page cache.
            release_touched = getattr(self.q4_bank, "release_touched", None)
            if callable(release_touched):
                release_touched()
            self._stats.gc_policy_boundaries += 1
            rss = _process_rss_bytes()
            if rss is None:
                self._stats.gc_rss_measurement_failures += 1
            else:
                self._stats.gc_last_observed_rss_bytes = rss
                self._stats.gc_peak_observed_rss_bytes = max(
                    self._stats.gc_peak_observed_rss_bytes,
                    rss,
                )
            self._stats.gc_collections_skipped += 1
            self._stats.gc_q4_mmap_skips += 1
        else:
            self._stats.gc_policy_boundaries += 1
            rss = _process_rss_bytes()
            if rss is None:
                self._stats.gc_rss_measurement_failures += 1
                self._collect_locked("fail_closed")
            else:
                self._stats.gc_last_observed_rss_bytes = rss
                self._stats.gc_peak_observed_rss_bytes = max(
                    self._stats.gc_peak_observed_rss_bytes,
                    rss,
                )
                if rss >= self.gc_rss_limit_bytes:
                    self._collect_locked("pressure")
                elif (
                    self._stats.gc_policy_boundaries % self.gc_interval_boundaries == 0
                ):
                    self._collect_locked("interval")
                else:
                    self._stats.gc_collections_skipped += 1
        if self.device.type == "mps":
            empty_cache = getattr(getattr(self.torch, "mps", None), "empty_cache", None)
            if callable(empty_cache):
                empty_cache()
                self._stats.mps_cache_purges += 1

    def release(self, *, force_gc: bool = False) -> None:
        """Release one boundary and collect only under the configured policy."""

        with self._lock:
            self._ensure_open()
            if not isinstance(force_gc, bool):
                raise ValueError("force_gc must be a boolean")
            self._release_locked(force_gc=force_gc)

    def close(self) -> None:
        """Wait for any active call, release allocator state, and close if owned."""

        with self._lock:
            if self._closed:
                return
            self._release_locked(force_gc=True)
            if self.close_source:
                close = getattr(self.source, "close", None)
                if callable(close):
                    close()
            if self.q4_bank is not None:
                self.q4_bank.close()
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
        exact_head = self.exact_head_index
        exact_head_metrics_method = getattr(exact_head, "metrics", None)
        exact_head_metrics = (
            {
                f"exact_head_{key}": value
                for key, value in dict(exact_head_metrics_method()).items()
            }
            if callable(exact_head_metrics_method)
            else {}
        )
        q4_bank = self.q4_bank
        q4_metrics = (
            {f"q4_{key}": value for key, value in q4_bank.metrics().items()}
            if q4_bank is not None
            else {}
        )
        with self._lock:
            return {
                **source_metrics,
                **asdict(self._stats),
                **causal_metrics,
                **exact_head_metrics,
                **q4_metrics,
                "causal_tensor_reader_attached": (
                    self.causal_tensor_reader is not None
                ),
                "exact_head_index_attached": exact_head is not None,
                "q4_bank_attached": q4_bank is not None,
                "device": self.resolved_device,
                "compute_dtype": self.resolved_dtype,
                "max_resident_bytes": self.max_resident_bytes,
                "gc_interval_boundaries": self.gc_interval_boundaries,
                "gc_rss_limit_bytes": self.gc_rss_limit_bytes,
                "weight_cache_policy": self.WEIGHT_CACHE_POLICY,
                "head_score_policy": (
                    self.HEAD_SCORE_POLICY
                    if exact_head is not None
                    else self.LEGACY_HEAD_SCORE_POLICY
                ),
                "source_identity": self.source_identity,
                "closed": self._closed,
            }

    def __enter__(self) -> "Qwen38WeightPager":
        self._ensure_open()
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()
