"""Bounded weight materialization for the streamed DeepSeek-V4 decoder."""

from __future__ import annotations

import gc
import math
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from contextvars import Context, copy_context
from dataclasses import asdict, dataclass
from collections.abc import Callable, Iterable, Mapping
import re
import threading
import time
from typing import Any, Protocol

import numpy as np

from ...knowledge.streamer import TensorEncodingError, TensorSource
from .quantization import (
    dequantize_fp8_e4m3,
    quantize_fp8_e4m3_parts,
    unpack_fp4_e2m1,
)


class DeepSeekPagerError(RuntimeError):
    """A checkpoint tensor cannot be executed by the bounded pager."""


@dataclass(slots=True)
class PagerMetrics:
    linear_calls: int = 0
    expert_calls: int = 0
    coalesced_expert_calls: int = 0
    expert_source_ranges: int = 0
    embedding_rows: int = 0
    head_rows: int = 0
    head_logical_leaves: int = 0
    head_transport_batches: int = 0
    head_transport_envelopes: int = 0
    head_transport_source_bytes: int = 0
    head_planned_range_calls_avoided: int = 0
    head_transport_fallbacks: int = 0
    head_transport_fallback_leaves: int = 0
    logical_weight_bytes: int = 0
    materialized_float_bytes: int = 0
    materialized_scale_bytes: int = 0
    peak_single_weight_bytes: int = 0
    materialized_weight_releases: int = 0
    release_boundaries: int = 0
    mps_cache_purges: int = 0
    block_scaled_linear_calls: int = 0
    fp8_k128_tiles: int = 0
    fp4_k32_tiles: int = 0
    expert_prefetch_submitted: int = 0
    expert_prefetch_consumed: int = 0
    expert_prefetch_failures: int = 0
    expert_prefetch_sync_fallbacks: int = 0
    expert_prefetch_payload_bytes: int = 0
    expert_prefetch_peak_bytes: int = 0
    expert_prefetch_wait_ns: int = 0
    expert_prefetch_ready_before_consume: int = 0
    expert_prefetch_cancelled: int = 0
    expert_prefetch_max_outstanding: int = 0
    expert_prefetch_batches_submitted: int = 0
    expert_reservoir_submitted: int = 0
    expert_reservoir_prediction_hits: int = 0
    expert_reservoir_usable_hits: int = 0
    expert_reservoir_misses: int = 0
    expert_reservoir_wasted: int = 0
    expert_reservoir_cancelled: int = 0
    expert_reservoir_failures: int = 0
    expert_reservoir_ready_hits: int = 0
    expert_reservoir_wait_ns: int = 0
    expert_reservoir_payload_bytes: int = 0
    expert_reservoir_hit_payload_bytes: int = 0
    expert_reservoir_wasted_payload_bytes: int = 0
    expert_reservoir_source_requests: int = 0
    expert_reservoir_source_bytes: int = 0
    expert_transport_envelopes: int = 0
    expert_transport_source_bytes: int = 0
    expert_range_requests_avoided: int = 0
    expert_range_gap_bytes: int = 0
    causal_expert_plan_resolves: int = 0
    causal_expert_plan_hits: int = 0
    causal_expert_plan_misses: int = 0
    causal_expert_plan_fallbacks: int = 0
    causal_expert_plan_invalid: int = 0


@dataclass(frozen=True, slots=True)
class ExpertTensorLayout:
    """One tensor leaf inside an exact expert source range.

    ``absolute_offset`` addresses the shard file, not the safetensors data
    section.  The leaf occupies the half-open interval
    ``[absolute_offset, absolute_end)``; ``range_offset`` addresses the same
    leaf relative to its enclosing :class:`ExpertSourceRange`.
    """

    name: str
    dtype: str
    shape: tuple[int, ...]
    absolute_offset: int
    length: int
    range_offset: int

    @property
    def absolute_end(self) -> int:
        return self.absolute_offset + self.length


@dataclass(frozen=True, slots=True)
class ExpertSourceRange:
    """One contiguous, exact half-open shard range for an official expert."""

    shard: str
    absolute_offset: int
    length: int
    tensors: tuple[ExpertTensorLayout, ...]

    @property
    def absolute_end(self) -> int:
        return self.absolute_offset + self.length


@dataclass(frozen=True, slots=True)
class OfficialExpertRangePlan:
    """Immutable metadata-only byte plan for one official routed expert."""

    base: str
    layer: int
    expert_id: int
    ranges: tuple[ExpertSourceRange, ...]
    payload_bytes: int


class CausalExpertPlanResolver(Protocol):
    """Duck contract for a causal expert-address plane.

    The protocol deliberately lives beside the pager plan types.  A concrete
    causal graph reader may import those types without the pager importing the
    graph module back and creating a runtime cycle.  Missing bindings use
    ``KeyError``; integrity/layout failures must use another exception and are
    always fail-closed by the pager.
    """

    def resolve_expert_plans(
        self,
        layer: int,
        expert_ids: Iterable[int],
    ) -> tuple[OfficialExpertRangePlan, ...]: ...


@dataclass(frozen=True, slots=True)
class _CoalescedTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    logical_bytes: int
    payload: memoryview


@dataclass(frozen=True, slots=True)
class _CoalescedExpert:
    tensors: dict[str, _CoalescedTensor]
    source_ranges: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class _CoalescedExpertBatch:
    experts: dict[str, _CoalescedExpert]
    resident_bytes: int
    source_requests: int
    source_bytes: int


@dataclass(frozen=True, slots=True)
class _ExpertReadPlan:
    base: str
    layouts: tuple[tuple[str, int, int, tuple[dict[str, Any], ...]], ...]
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class _ExpertBatchPlan:
    experts: tuple[_ExpertReadPlan, ...]
    payload_bytes: int
    range_requests_avoided: int


@dataclass(frozen=True, slots=True)
class _HeadRowLeaf:
    start_row: int
    count: int
    absolute: int
    length: int


@dataclass(slots=True)
class _ExpertPrefetchBatch:
    plan: _ExpertBatchPlan
    future: Future[_CoalescedExpertBatch] | None
    remaining: int
    submitted: bool = False
    retired: bool = False
    metrics_accounted: bool = False


@dataclass(slots=True)
class _ExpertPrefetch:
    base: str
    plan: _ExpertReadPlan
    batch: _ExpertPrefetchBatch
    payload_bytes: int
    index: int
    window: _ExpertPrefetchWindow | None = None
    consumed: bool = False


@dataclass(slots=True)
class _ExpertPrefetchWindow:
    bases: tuple[str, ...]
    tickets: tuple[_ExpertPrefetch, ...]
    batches: tuple[_ExpertPrefetchBatch, ...]
    executor: ThreadPoolExecutor | None
    submission_context: Context
    next_submit: int = 0
    next_consume: int = 0
    closed: bool = False
    compatibility_single: bool = False


@dataclass(slots=True)
class _ExpertPayload:
    base: str
    expert: _CoalescedExpert | None
    payload_bytes: int
    window: _ExpertPrefetchWindow
    batch: _ExpertPrefetchBatch
    consumed: bool = False


@dataclass(slots=True)
class _ExpertReservoirTicket:
    base: str
    plan: _ExpertReadPlan
    future: Future[_CoalescedExpertBatch]
    index: int
    accounted: bool = False


@dataclass(slots=True)
class _ExpertReservoir:
    bases: tuple[str, ...]
    tickets: tuple[_ExpertReservoirTicket, ...]
    executor: ThreadPoolExecutor
    payload_bytes: int
    closed: bool = False


@dataclass(slots=True)
class _ExpertReservoirPayload:
    owner: object
    base: str
    expert: _CoalescedExpert | None
    payload_bytes: int
    consumed: bool = False


@dataclass(frozen=True, slots=True)
class _MaterializedWeight:
    """One decoded checkpoint matrix without prematurely applying MX scales."""

    values: np.ndarray
    scales: np.ndarray | None
    storage_dtype: str
    logical_bytes: int


@dataclass(frozen=True, slots=True)
class _QuantizedActivation:
    """Decoded E4M3 values and the separate K128 activation scales."""

    values: np.ndarray
    scales: np.ndarray


class DeepSeekWeightPager:
    """Materialize one matrix at a time and immediately release it.

    ``TensorSource`` owns provenance, transfer budgets, and the verified disk
    cache.  This class owns only short-lived decoded matrices and compute.  It
    therefore cannot accidentally retain a complete frontier checkpoint.
    """

    QUANTIZED_ACCUMULATION_POLICY = "mx-block-scaled-fp32/v1"
    EXPERT_PREFETCH_POLICY = "exact-router-window-q3-a2/v2"
    EXPERT_PREFETCH_TRANSPORT_POLICY = "streamer-exact-range/v1"
    EXPERT_PREFETCH_ADJACENT_POLICY = "exact-router-window-q3-a2-adjacent-pairs/v3"
    EXPERT_PREFETCH_ADJACENT_TRANSPORT_POLICY = (
        "streamer-exact-leaf-adjacent-envelope/v2"
    )
    EXPERT_PREFETCH_WORKERS = 2
    EXPERT_PREFETCH_ACTIVE_READ_LIMIT = 2
    EXPERT_PREFETCH_MAX_OUTSTANDING = 3
    EXPERT_PREFETCH_MAX_EXPERTS = 3
    EXPERT_PREFETCH_RESIDENT_LIMIT_BYTES = 48 * 1024**2
    EXPERT_PREFETCH_PAYLOAD_LIMIT_BYTES = 14 * 1024**2
    EXPERT_RESERVOIR_POLICY = "markov-micro-window-exact-fallback/v1"
    EXPERT_RESERVOIR_WORKERS = 4
    EXPERT_RESERVOIR_DEFAULT_BUDGET_BYTES = 256 * 1024**2
    EXPERT_RANGE_COALESCE_DEFAULT_MAX_EXPERTS = 1
    EXPERT_RANGE_COALESCE_MAX_EXPERTS = 2
    EXPERT_RANGE_COALESCE_MAX_GAP_BYTES = 0
    HEAD_TRANSPORT_POLICY = "exact-head-leaf-adjacent-envelope/v1"
    HEAD_TRANSPORT_DEFAULT_RANGE_BATCH_BLOCKS = 1
    HEAD_TRANSPORT_MAX_RANGE_BATCH_BLOCKS = 8
    HEAD_TRANSPORT_RESIDENT_LIMIT_BYTES = 64 * 1024**2
    HEAD_TRANSPORT_MAX_GAP_BYTES = 0
    _OFFICIAL_EXPERT_BASE = re.compile(
        r"^layers\.(0|[1-9][0-9]*)\.ffn\.experts\.(0|[1-9][0-9]*)$"
    )

    def __init__(
        self,
        source: TensorSource,
        *,
        device: str = "auto",
        compute_dtype: str = "auto",
        simulate_activation_quantization: bool = True,
        expert_prefetch: bool = True,
        expert_range_coalesce_max_experts: int | None = None,
        expert_reservoir_budget_bytes: int | None = None,
        expert_reservoir_workers: int | None = None,
        causal_weight_reader: CausalExpertPlanResolver | None = None,
        causal_missing_fallback: bool = False,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - neural extra
            raise DeepSeekPagerError("DeepSeek-V4 execution requires torch") from exc
        self.torch = torch
        self.source = source
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        if device not in {"cpu", "mps"}:
            raise ValueError("device must be 'auto', 'cpu', or 'mps'")
        if device == "mps" and not torch.backends.mps.is_available():
            raise DeepSeekPagerError("MPS was requested but is unavailable")
        self.device = torch.device(device)
        if compute_dtype == "auto":
            # DeepSeek V4's published main decoder uses BF16 activations.
            # Keep this invariant on the CPU reference path as well: silently
            # promoting only CPU execution to FP32 changes routing boundaries
            # and makes CPU/MPS parity evidence incomparable.
            compute_dtype = "bfloat16"
        try:
            dtype = getattr(torch, compute_dtype)
        except AttributeError as exc:
            raise ValueError(
                f"unsupported torch compute dtype: {compute_dtype}"
            ) from exc
        if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError("compute_dtype must be float16, bfloat16, or float32")
        self.compute_dtype = dtype
        self.simulate_activation_quantization = bool(simulate_activation_quantization)
        self.expert_prefetch_enabled = bool(expert_prefetch)
        if expert_range_coalesce_max_experts is None:
            expert_range_coalesce_max_experts = (
                self.EXPERT_RANGE_COALESCE_DEFAULT_MAX_EXPERTS
            )
        if (
            isinstance(expert_range_coalesce_max_experts, bool)
            or not isinstance(expert_range_coalesce_max_experts, int)
            or not 1
            <= expert_range_coalesce_max_experts
            <= self.EXPERT_RANGE_COALESCE_MAX_EXPERTS
        ):
            raise ValueError(
                "expert_range_coalesce_max_experts must be an integer in [1, 2]"
            )
        self.expert_range_coalesce_max_experts = expert_range_coalesce_max_experts
        if expert_reservoir_budget_bytes is None:
            expert_reservoir_budget_bytes = self.EXPERT_RESERVOIR_DEFAULT_BUDGET_BYTES
        if (
            isinstance(expert_reservoir_budget_bytes, bool)
            or not isinstance(expert_reservoir_budget_bytes, int)
            or expert_reservoir_budget_bytes < 1
        ):
            raise ValueError("expert_reservoir_budget_bytes must be positive")
        if expert_reservoir_workers is None:
            expert_reservoir_workers = self.EXPERT_RESERVOIR_WORKERS
        if (
            isinstance(expert_reservoir_workers, bool)
            or not isinstance(expert_reservoir_workers, int)
            or expert_reservoir_workers < 1
        ):
            raise ValueError("expert_reservoir_workers must be positive")
        self.expert_reservoir_budget_bytes = expert_reservoir_budget_bytes
        self.expert_reservoir_workers = expert_reservoir_workers
        self._stats = PagerMetrics()
        self._prefetch_lock = threading.Lock()
        self._prefetch_executor: ThreadPoolExecutor | None = None
        self._active_prefetch_window: _ExpertPrefetchWindow | None = None
        self._active_prefetch_payload: _ExpertPayload | None = None
        self._draining_prefetch: set[Future[_CoalescedExpertBatch]] = set()
        self._draining_executor: ThreadPoolExecutor | None = None
        self._active_expert_reservoir: _ExpertReservoir | None = None
        self._retired_expert_reservoirs: list[_ExpertReservoir] = []
        self._expert_reservoir_executor: ThreadPoolExecutor | None = None
        self._expert_reservoir_owner = object()
        self._causal_weight_reader: CausalExpertPlanResolver | None = None
        self._causal_missing_fallback = False
        self.attach_causal_weight_reader(
            causal_weight_reader,
            fallback_on_missing=causal_missing_fallback,
        )

    def attach_causal_weight_reader(
        self,
        reader: CausalExpertPlanResolver | None,
        *,
        fallback_on_missing: bool = False,
    ) -> None:
        """Attach or detach the routed-expert causal address plane.

        Once attached, canonical routed experts are resolved exclusively from
        stored absolute ranges and tensor offsets.  A missing binding fails
        closed unless ``fallback_on_missing`` is explicitly enabled; malformed
        or conflicting bindings never fall back to tensor-name discovery.
        Shared/dense experts are outside this routed-address contract and keep
        their normal checkpoint metadata path.
        """

        if not isinstance(fallback_on_missing, bool):
            raise ValueError("fallback_on_missing must be a boolean")
        if reader is None:
            if fallback_on_missing:
                raise ValueError("missing fallback requires an attached reader")
        else:
            if not callable(getattr(reader, "resolve_expert_plans", None)):
                raise TypeError(
                    "causal weight reader must implement resolve_expert_plans"
                )
            layout = getattr(reader, "layout", None)
            expected_fingerprint = getattr(layout, "layout_fingerprint", None)
            reader_source = getattr(reader, "source", None)
            if expected_fingerprint and reader_source is not self.source:
                source_metrics = getattr(self.source, "metrics", None)
                snapshot = source_metrics() if callable(source_metrics) else None
                observed_fingerprint = (
                    snapshot.get("inventory_source_fingerprint")
                    if isinstance(snapshot, Mapping)
                    else None
                )
                if observed_fingerprint != expected_fingerprint:
                    raise DeepSeekPagerError(
                        "causal weight reader layout does not match the pager source"
                    )
        with self._prefetch_lock:
            if (
                self._active_prefetch_window is not None
                or self._active_prefetch_payload is not None
                or self._draining_prefetch
                or self._draining_executor is not None
                or self._active_expert_reservoir is not None
                or self._retired_expert_reservoirs
            ):
                raise DeepSeekPagerError(
                    "cannot replace the causal weight reader during expert prefetch"
                )
            self._causal_weight_reader = reader
            self._causal_missing_fallback = fallback_on_missing

    @property
    def expert_prefetch_policy(self) -> str:
        if not self.expert_prefetch_enabled:
            return "disabled"
        if self.expert_range_coalesce_max_experts >= 2:
            return self.EXPERT_PREFETCH_ADJACENT_POLICY
        return self.EXPERT_PREFETCH_POLICY

    @property
    def expert_prefetch_transport_policy(self) -> str:
        if not self.expert_prefetch_enabled:
            return "disabled"
        if self.expert_range_coalesce_max_experts >= 2:
            return self.EXPERT_PREFETCH_ADJACENT_TRANSPORT_POLICY
        return self.EXPERT_PREFETCH_TRANSPORT_POLICY

    @staticmethod
    def _weight_name(prefix: str) -> str:
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("weight prefix must be a non-empty string")
        return prefix if prefix.endswith(".weight") else f"{prefix}.weight"

    @staticmethod
    def _payload_bytes(meta: dict[str, Any]) -> int:
        begin, end = meta["offset_in_shard"]
        return int(end) - int(begin)

    @staticmethod
    def _consecutive_runs(ids: Iterable[int]) -> list[tuple[int, int]]:
        unique = sorted(set(int(value) for value in ids))
        if not unique:
            return []
        runs: list[tuple[int, int]] = []
        run_start = previous = unique[0]
        for value in unique[1:]:
            if value != previous + 1:
                runs.append((run_start, previous + 1))
                run_start = value
            previous = value
        runs.append((run_start, previous + 1))
        return runs

    @staticmethod
    def _decode_coalesced_tensor(tensor: _CoalescedTensor) -> np.ndarray:
        """Decode one safetensors payload already covered by a larger range."""

        dtype = tensor.dtype
        expected = int(np.prod(tensor.shape, dtype=np.int64))
        if len(tensor.payload) != tensor.logical_bytes:
            raise DeepSeekPagerError(
                f"short coalesced payload for {tensor.name}: "
                f"{len(tensor.payload)}/{tensor.logical_bytes} bytes"
            )
        if dtype == "I8":
            values = np.frombuffer(tensor.payload, dtype=np.int8)
        elif dtype in {"F8_E4M3", "F8_E4M3FN"}:
            bits = np.frombuffer(tensor.payload, dtype=np.uint8)
            invalid = np.flatnonzero((bits & np.uint8(0x7F)) == np.uint8(0x7F))
            if invalid.size:
                index = int(invalid[0])
                raise TensorEncodingError(
                    f"{tensor.name}: Reservierte/nicht-endliche {dtype}-Kodierung "
                    f"0x{int(bits[index]):02X} bei Element {index}"
                )
            exponent = ((bits >> 3) & 0x0F).astype(np.int16)
            mantissa = (bits & 0x07).astype(np.float32)
            values = np.empty(bits.shape, dtype=np.float32)
            subnormal = exponent == 0
            values[subnormal] = np.ldexp(mantissa[subnormal], -9)
            normal = ~subnormal
            values[normal] = np.ldexp(
                np.float32(1.0) + mantissa[normal] * np.float32(0.125),
                exponent[normal] - 7,
            )
            values = np.copysign(
                values,
                np.where(bits & 0x80, np.float32(-1.0), np.float32(1.0)),
            )
        elif dtype == "F8_E8M0":
            bits = np.frombuffer(tensor.payload, dtype=np.uint8)
            invalid = np.flatnonzero(bits == np.uint8(0xFF))
            if invalid.size:
                index = int(invalid[0])
                raise TensorEncodingError(
                    f"{tensor.name}: Reservierte/nicht-endliche {dtype}-Kodierung "
                    f"0xFF bei Element {index}"
                )
            values = np.ldexp(
                np.ones(bits.shape, dtype=np.float32),
                bits.astype(np.int16) - 127,
            )
        elif dtype == "BF16":
            words = np.frombuffer(tensor.payload, dtype="<u2").astype(np.uint32)
            words <<= 16
            values = words.view(np.float32)
        elif dtype == "F16":
            values = np.frombuffer(tensor.payload, dtype="<f2").copy()
        elif dtype == "F32":
            values = np.frombuffer(tensor.payload, dtype="<f4").copy()
        else:  # guarded by _coalesced_expert below
            raise DeepSeekPagerError(
                f"unsupported coalesced dtype {dtype!r} at {tensor.name}"
            )
        if values.size != expected:
            raise DeepSeekPagerError(
                f"coalesced shape mismatch for {tensor.name}: "
                f"{values.size} values for {tensor.shape}"
            )
        return values.reshape(tensor.shape)

    @staticmethod
    def _head_row_layout(
        meta: dict[str, Any],
        *,
        block_rows: int,
    ) -> tuple[str, str, int, tuple[_HeadRowLeaf, ...]] | None:
        """Plan exact LM-head leaves, or decline before transport I/O.

        Every planned leaf is byte-identical to the range ``TensorSource.rows``
        requests for the same unchanged compute block. Only plain floating
        checkpoint rows are eligible; quantized heads require scale-aware
        decoding and remain on the scalar source path.
        """

        itemsize_by_dtype = {"BF16": 2, "F16": 2, "F32": 4}
        try:
            shape = tuple(int(value) for value in meta["shape"])
            if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
                return None
            dtype = str(meta["dtype"]).upper()
            itemsize = itemsize_by_dtype.get(dtype)
            if itemsize is None:
                return None
            shard = str(meta["shard"])
            data_start = int(meta["data_start"])
            offsets = meta["offset_in_shard"]
            if (
                not shard
                or data_start < 8
                or not isinstance(offsets, (list, tuple))
                or len(offsets) != 2
            ):
                return None
            begin, end = (int(value) for value in offsets)
            if begin < 0 or end <= begin:
                return None
            vocab, columns = shape
            row_bytes = columns * itemsize
            if end - begin != vocab * row_bytes:
                return None
        except (KeyError, TypeError, ValueError, OverflowError):
            return None

        leaves = tuple(
            _HeadRowLeaf(
                start_row=start,
                count=min(block_rows, vocab - start),
                absolute=data_start + begin + start * row_bytes,
                length=min(block_rows, vocab - start) * row_bytes,
            )
            for start in range(0, vocab, block_rows)
        )
        return shard, dtype, columns, leaves

    def _read_head_leaf_batch(
        self,
        *,
        name: str,
        shard: str,
        dtype: str,
        columns: int,
        leaves: tuple[_HeadRowLeaf, ...],
    ) -> tuple[_CoalescedTensor, ...]:
        """Fetch exact row leaves with a bounded, fail-closed receipt."""

        raw_bytes_many = getattr(self.source, "raw_bytes_many", None)
        if not callable(raw_bytes_many):  # guarded by topk_logits planning
            raise DeepSeekPagerError("head multi-range capability disappeared")
        requested = tuple((leaf.absolute, leaf.length) for leaf in leaves)
        requested_bytes = sum(leaf.length for leaf in leaves)
        if requested_bytes > self.HEAD_TRANSPORT_RESIDENT_LIMIT_BYTES:
            raise DeepSeekPagerError(
                "head multi-range plan exceeds the raw resident limit"
            )
        result = raw_bytes_many(
            shard,
            requested,
            resident_limit_bytes=self.HEAD_TRANSPORT_RESIDENT_LIMIT_BYTES,
            max_gap_bytes=self.HEAD_TRANSPORT_MAX_GAP_BYTES,
        )
        try:
            parts = tuple(result.parts)
            resident_bytes = int(result.resident_bytes)
            source_requests = int(result.source_requests)
            source_bytes = int(result.source_bytes)
        except (AttributeError, TypeError, ValueError) as exc:
            raise DeepSeekPagerError(
                f"invalid head multi-range receipt for {shard}"
            ) from exc
        if len(parts) != len(leaves):
            raise DeepSeekPagerError(
                f"head multi-range part count mismatch for {shard}: "
                f"{len(parts)}/{len(leaves)}"
            )
        if (
            resident_bytes != requested_bytes
            or resident_bytes > self.HEAD_TRANSPORT_RESIDENT_LIMIT_BYTES
            or not 0 <= source_requests <= len(leaves)
            or not 0 <= source_bytes <= requested_bytes
        ):
            raise DeepSeekPagerError(
                f"invalid head multi-range receipt for {shard}: resident="
                f"{resident_bytes}/{requested_bytes}, requests={source_requests}, "
                f"source_bytes={source_bytes}"
            )

        tensors: list[_CoalescedTensor] = []
        for leaf, part in zip(leaves, parts, strict=True):
            view = part if isinstance(part, memoryview) else memoryview(part)
            if not view.readonly or len(view) != leaf.length:
                raise DeepSeekPagerError(
                    f"invalid readonly head leaf for {name}[{leaf.start_row}:"
                    f"{leaf.start_row + leaf.count}]: {len(view)}/{leaf.length} bytes"
                )
            tensors.append(
                _CoalescedTensor(
                    name=(f"{name}[{leaf.start_row}:{leaf.start_row + leaf.count}]"),
                    dtype=dtype,
                    shape=(leaf.count, columns),
                    logical_bytes=leaf.length,
                    payload=view,
                )
            )
        self._stats.head_transport_batches += 1
        self._stats.head_transport_envelopes += source_requests
        self._stats.head_transport_source_bytes += source_bytes
        # This is caller-level planning, not a physical-I/O claim: warm leaves
        # also avoid scalar API calls while the receipt correctly reports zero
        # source envelopes/bytes.
        self._stats.head_planned_range_calls_avoided += max(0, len(leaves) - 1)
        return tuple(tensors)

    @staticmethod
    def _coalesced_group_layout(
        metas: list[dict[str, Any]],
    ) -> tuple[str, int, int, list[dict[str, Any]]] | None:
        """Return one exact adjacent source interval, or decline safely."""

        if not metas:
            return None
        try:
            shard = str(metas[0]["shard"])
            data_start = int(metas[0]["data_start"])
            ordered = sorted(metas, key=lambda meta: int(meta["offset_in_shard"][0]))
            if not shard or data_start < 8:
                return None
            for meta in ordered:
                offsets = meta["offset_in_shard"]
                if (
                    str(meta["shard"]) != shard
                    or int(meta["data_start"]) != data_start
                    or not isinstance(offsets, list)
                    or len(offsets) != 2
                    or int(offsets[1]) <= int(offsets[0])
                ):
                    return None
            if any(
                int(left["offset_in_shard"][1]) != int(right["offset_in_shard"][0])
                for left, right in zip(ordered, ordered[1:], strict=False)
            ):
                return None
            begin = int(ordered[0]["offset_in_shard"][0])
            end = int(ordered[-1]["offset_in_shard"][1])
        except (KeyError, TypeError, ValueError, IndexError):
            return None
        return shard, data_start + begin, end - begin, ordered

    def _expert_plan(
        self,
        base: str,
        *,
        require_payload_reader: bool = True,
    ) -> _ExpertReadPlan | None:
        """Resolve an official expert's two ranges on the main thread.

        Published V4 shards place ``w1/w2/w3`` next to one another and do the
        same for their scales.  Synthetic sources and repacked checkpoints are
        allowed to use any layout; those decline before any range is read.
        Metadata-only callers may disable the payload-capability check while
        retaining exactly the same layout validation.
        """

        raw_bytes = getattr(self.source, "raw_bytes", None)
        if require_payload_reader and not callable(raw_bytes):
            return None
        match = self._OFFICIAL_EXPERT_BASE.fullmatch(base)
        reader = self._causal_weight_reader
        if reader is not None and match is not None:
            layer, expert_id = int(match.group(1)), int(match.group(2))
            self._stats.causal_expert_plan_resolves += 1
            try:
                resolved = reader.resolve_expert_plans(layer, (expert_id,))
            except KeyError as exc:
                self._stats.causal_expert_plan_misses += 1
                if self._causal_missing_fallback:
                    self._stats.causal_expert_plan_fallbacks += 1
                    return self._source_expert_plan(base)
                raise DeepSeekPagerError(
                    f"causal weight binding is missing for {base}"
                ) from exc
            except Exception as exc:
                self._stats.causal_expert_plan_invalid += 1
                raise DeepSeekPagerError(
                    f"causal weight resolution failed for {base}"
                ) from exc
            if not isinstance(resolved, tuple):
                self._stats.causal_expert_plan_invalid += 1
                raise DeepSeekPagerError(
                    f"causal weight resolver returned a non-tuple for {base}"
                )
            if not resolved:
                self._stats.causal_expert_plan_misses += 1
                if self._causal_missing_fallback:
                    self._stats.causal_expert_plan_fallbacks += 1
                    return self._source_expert_plan(base)
                raise DeepSeekPagerError(f"causal weight binding is missing for {base}")
            if len(resolved) != 1:
                self._stats.causal_expert_plan_invalid += 1
                raise DeepSeekPagerError(
                    f"causal weight resolver returned {len(resolved)} plans for {base}"
                )
            try:
                plan = self._private_expert_plan(
                    resolved[0],
                    base=base,
                    layer=layer,
                    expert_id=expert_id,
                )
            except DeepSeekPagerError:
                self._stats.causal_expert_plan_invalid += 1
                raise
            self._stats.causal_expert_plan_hits += 1
            return plan
        return self._source_expert_plan(base)

    def _source_expert_plan(self, base: str) -> _ExpertReadPlan | None:
        """Discover one expert through safetensors metadata when no rail applies."""

        weight_names = [f"{base}.{role}.weight" for role in ("w1", "w2", "w3")]
        try:
            weight_metas = [self.source.find(name) for name in weight_names]
        except (KeyError, TypeError, ValueError):
            return None
        weight_dtypes = {str(meta.get("dtype", "")).upper() for meta in weight_metas}
        if len(weight_dtypes) != 1 or not weight_dtypes <= {
            "I8",
            "F8_E4M3",
            "F8_E4M3FN",
        }:
            return None
        scale_names = [name.removesuffix("weight") + "scale" for name in weight_names]
        try:
            scale_metas = [self.source.find(name) for name in scale_names]
        except (KeyError, TypeError, ValueError):
            return None
        if any(str(meta.get("dtype", "")).upper() != "F8_E8M0" for meta in scale_metas):
            return None
        layouts = [
            self._coalesced_group_layout(scale_metas),
            self._coalesced_group_layout(weight_metas),
        ]
        if any(layout is None for layout in layouts):
            return None

        frozen_layouts = tuple(
            (
                layout[0],
                layout[1],
                layout[2],
                tuple(dict(meta) for meta in layout[3]),
            )
            for layout in layouts
            if layout is not None
        )
        payload_bytes = sum(layout[2] for layout in frozen_layouts)
        return _ExpertReadPlan(
            base=base,
            layouts=frozen_layouts,
            payload_bytes=payload_bytes,
        )

    @staticmethod
    def _private_expert_plan(
        public: OfficialExpertRangePlan,
        *,
        base: str,
        layer: int,
        expert_id: int,
    ) -> _ExpertReadPlan:
        """Validate and lower one stored causal plan without metadata lookup."""

        if not isinstance(public, OfficialExpertRangePlan):
            raise DeepSeekPagerError(
                f"causal weight resolver returned an unsupported plan for {base}"
            )
        if (public.base, public.layer, public.expert_id) != (base, layer, expert_id):
            raise DeepSeekPagerError(
                f"causal weight plan coordinates do not match {base}"
            )
        if not isinstance(public.ranges, tuple) or len(public.ranges) != 2:
            raise DeepSeekPagerError(
                f"causal weight plan for {base} must contain scale and weight ranges"
            )

        expected_groups = (
            tuple(f"{base}.{role}.scale" for role in ("w1", "w2", "w3")),
            tuple(f"{base}.{role}.weight" for role in ("w1", "w2", "w3")),
        )
        layouts: list[tuple[str, int, int, tuple[dict[str, Any], ...]]] = []
        observed_weight_dtypes: set[str] = set()
        payload_bytes = 0
        for range_index, (source_range, expected_names) in enumerate(
            zip(public.ranges, expected_groups, strict=True)
        ):
            if not isinstance(source_range, ExpertSourceRange):
                raise DeepSeekPagerError(
                    f"causal weight range {range_index} for {base} is invalid"
                )
            shard = source_range.shard
            absolute = source_range.absolute_offset
            length = source_range.length
            if not isinstance(shard, str) or not shard:
                raise DeepSeekPagerError(
                    f"causal weight range {range_index} for {base} has no shard"
                )
            if (
                isinstance(absolute, bool)
                or not isinstance(absolute, int)
                or absolute < 8
                or isinstance(length, bool)
                or not isinstance(length, int)
                or length <= 0
            ):
                raise DeepSeekPagerError(
                    f"causal weight range {range_index} for {base} is invalid"
                )
            tensors = source_range.tensors
            if (
                not isinstance(tensors, tuple)
                or any(not isinstance(tensor, ExpertTensorLayout) for tensor in tensors)
                or len({tensor.name for tensor in tensors}) != len(expected_names)
                or {tensor.name for tensor in tensors} != set(expected_names)
            ):
                raise DeepSeekPagerError(
                    f"causal weight tensor membership is invalid for {base} "
                    f"range {range_index}"
                )

            metas: list[dict[str, Any]] = []
            cursor = 0
            # Tensor tuple order is physical offset order, not role order.  A
            # local repack may place w2 before w1 while preserving the same
            # names, exact offsets, and decoded computation.
            for tensor in tensors:
                tensor_length = tensor.length
                range_offset = tensor.range_offset
                tensor_absolute = tensor.absolute_offset
                shape = tensor.shape
                dtype = tensor.dtype
                if (
                    isinstance(tensor_length, bool)
                    or not isinstance(tensor_length, int)
                    or tensor_length <= 0
                    or isinstance(range_offset, bool)
                    or not isinstance(range_offset, int)
                    or range_offset != cursor
                    or isinstance(tensor_absolute, bool)
                    or not isinstance(tensor_absolute, int)
                    or tensor_absolute != absolute + range_offset
                ):
                    raise DeepSeekPagerError(
                        f"causal weight tensor offsets are invalid for {tensor.name}"
                    )
                if (
                    not isinstance(shape, tuple)
                    or not shape
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value <= 0
                        for value in shape
                    )
                ):
                    raise DeepSeekPagerError(
                        f"causal weight tensor shape is invalid for {tensor.name}"
                    )
                if not isinstance(dtype, str):
                    raise DeepSeekPagerError(
                        f"causal weight tensor dtype is invalid for {tensor.name}"
                    )
                dtype = dtype.upper()
                if range_index == 0:
                    if dtype != "F8_E8M0":
                        raise DeepSeekPagerError(
                            f"causal weight scale dtype is invalid for {tensor.name}"
                        )
                else:
                    observed_weight_dtypes.add(dtype)
                    if dtype not in {"I8", "F8_E4M3", "F8_E4M3FN"}:
                        raise DeepSeekPagerError(
                            f"causal weight dtype is invalid for {tensor.name}"
                        )
                expected_length = math.prod(shape)
                if tensor_length != expected_length:
                    raise DeepSeekPagerError(
                        f"causal weight byte length is invalid for {tensor.name}"
                    )
                end = range_offset + tensor_length
                if end > length:
                    raise DeepSeekPagerError(
                        f"causal weight tensor exceeds its range for {tensor.name}"
                    )
                metas.append(
                    {
                        "name": tensor.name,
                        "dtype": dtype,
                        "shape": list(shape),
                        "shard": shard,
                        # These synthetic metadata coordinates reproduce the
                        # exact stored absolutes; no safetensors header lookup
                        # participates in lowering or decoding.
                        "data_start": absolute,
                        "offset_in_shard": [range_offset, end],
                    }
                )
                cursor = end
            if cursor != length:
                raise DeepSeekPagerError(
                    f"causal weight tensors do not exactly cover range {range_index} "
                    f"for {base}"
                )
            payload_bytes += length
            layouts.append((shard, absolute, length, tuple(metas)))

        if len(observed_weight_dtypes) != 1:
            raise DeepSeekPagerError(f"causal weight encodings disagree within {base}")
        if public.payload_bytes != payload_bytes:
            raise DeepSeekPagerError(
                f"causal weight payload total is invalid for {base}"
            )
        return _ExpertReadPlan(
            base=base,
            layouts=tuple(layouts),
            payload_bytes=payload_bytes,
        )

    @classmethod
    def _parse_official_expert_base(cls, base: str) -> tuple[int, int]:
        if not isinstance(base, str):
            raise ValueError("expert bases must be strings")
        match = cls._OFFICIAL_EXPERT_BASE.fullmatch(base)
        if match is None:
            raise ValueError(
                "expert base must be canonical 'layers.<layer>.ffn.experts.<expert_id>'"
            )
        return int(match.group(1)), int(match.group(2))

    @staticmethod
    def _validate_expert_coordinate(value: Any, *, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        return value

    @staticmethod
    def _public_expert_plan(
        plan: _ExpertReadPlan,
        *,
        layer: int,
        expert_id: int,
    ) -> OfficialExpertRangePlan:
        ranges: list[ExpertSourceRange] = []
        for shard, absolute, length, metas in plan.layouts:
            tensors: list[ExpertTensorLayout] = []
            for meta in metas:
                begin, end = (int(value) for value in meta["offset_in_shard"])
                tensor_absolute = int(meta["data_start"]) + begin
                tensors.append(
                    ExpertTensorLayout(
                        name=str(meta["name"]),
                        dtype=str(meta["dtype"]).upper(),
                        shape=tuple(int(value) for value in meta["shape"]),
                        absolute_offset=tensor_absolute,
                        length=end - begin,
                        range_offset=tensor_absolute - absolute,
                    )
                )
            ranges.append(
                ExpertSourceRange(
                    shard=shard,
                    absolute_offset=absolute,
                    length=length,
                    tensors=tuple(tensors),
                )
            )
        return OfficialExpertRangePlan(
            base=plan.base,
            layer=layer,
            expert_id=expert_id,
            ranges=tuple(ranges),
            payload_bytes=plan.payload_bytes,
        )

    def plan_expert_ranges(
        self,
        expert_bases_or_layer: Iterable[str] | int,
        expert_ids: Iterable[int] | None = None,
    ) -> tuple[OfficialExpertRangePlan, ...]:
        """Project exact official-expert layouts without reading payload bytes.

        Pass either an iterable of canonical expert bases or ``(layer,
        expert_ids)`` as the two arguments.  Inputs are materialized once,
        deduplicated in first-seen order, and have no fanout, top-k, cache, or
        executor-window limit.  A base iterable spanning decoder layers is
        rejected because one plan set represents one router decision.

        This method calls ``TensorSource.find`` for checkpoint metadata only;
        it never calls ``raw_bytes``, ``raw_bytes_many``, tensor decoding, or
        prefetch APIs.  On a cold ``Streamer``, ``find`` may load or prepare the
        inventory, so latency-sensitive callers should prepare the inventory
        before entering the model hot path.
        """

        coordinates: list[tuple[str, int, int]] = []
        if expert_ids is None:
            if isinstance(expert_bases_or_layer, (str, bytes)) or isinstance(
                expert_bases_or_layer, int
            ):
                raise ValueError(
                    "pass an iterable of expert bases, or layer plus expert_ids"
                )
            seen_bases: set[str] = set()
            for base in expert_bases_or_layer:
                layer, expert_id = self._parse_official_expert_base(base)
                if base in seen_bases:
                    continue
                seen_bases.add(base)
                coordinates.append((base, layer, expert_id))
            layers = {layer for _, layer, _ in coordinates}
            if len(layers) > 1:
                raise ValueError("expert bases must all belong to the same layer")
        else:
            layer = self._validate_expert_coordinate(
                expert_bases_or_layer,
                name="layer",
            )
            if isinstance(expert_ids, (str, bytes)):
                raise ValueError("expert_ids must be an iterable of integers")
            seen_ids: set[int] = set()
            for raw_expert_id in expert_ids:
                expert_id = self._validate_expert_coordinate(
                    raw_expert_id,
                    name="expert_id",
                )
                if expert_id in seen_ids:
                    continue
                seen_ids.add(expert_id)
                coordinates.append(
                    (
                        f"layers.{layer}.ffn.experts.{expert_id}",
                        layer,
                        expert_id,
                    )
                )

        public: list[OfficialExpertRangePlan] = []
        for base, layer, expert_id in coordinates:
            plan = self._expert_plan(base, require_payload_reader=False)
            if plan is None:
                raise DeepSeekPagerError(
                    f"cannot form an exact metadata range plan for {base}; "
                    "the expert is missing or its official adjacent "
                    "weight/scale layout is unavailable"
                )
            public.append(
                self._public_expert_plan(
                    plan,
                    layer=layer,
                    expert_id=expert_id,
                )
            )
        return tuple(public)

    def _read_expert_plan(self, plan: _ExpertReadPlan) -> _CoalescedExpert:
        """Read one pre-resolved plan without any metadata/source discovery."""

        payloads: list[memoryview] = []
        for layout in plan.layouts:
            shard, absolute, length, _ = layout
            payload = self.source.raw_bytes(shard, absolute, length)
            if len(payload) != length:
                raise DeepSeekPagerError(
                    f"short coalesced source range for {plan.base}: "
                    f"{len(payload)}/{length} bytes"
                )
            payloads.append(memoryview(payload))
        return self._expert_from_layout_payloads(plan, tuple(payloads))

    @staticmethod
    def _expert_from_layout_payloads(
        plan: _ExpertReadPlan,
        payloads: tuple[memoryview, ...],
    ) -> _CoalescedExpert:
        if len(payloads) != len(plan.layouts):
            raise DeepSeekPagerError(
                f"coalesced layout count mismatch for {plan.base}: "
                f"{len(payloads)}/{len(plan.layouts)}"
            )
        tensors: dict[str, _CoalescedTensor] = {}
        for layout, group_view in zip(plan.layouts, payloads, strict=True):
            _, _, length, ordered = layout
            if len(group_view) != length:
                raise DeepSeekPagerError(
                    f"short coalesced source range for {plan.base}: "
                    f"{len(group_view)}/{length} bytes"
                )
            group_begin = int(ordered[0]["offset_in_shard"][0])
            for meta in ordered:
                begin, end = (int(value) for value in meta["offset_in_shard"])
                name = str(meta["name"])
                tensors[name] = _CoalescedTensor(
                    name=name,
                    dtype=str(meta["dtype"]).upper(),
                    shape=tuple(int(value) for value in meta["shape"]),
                    logical_bytes=end - begin,
                    payload=group_view[begin - group_begin : end - group_begin],
                )
        return _CoalescedExpert(
            tensors=tensors,
            source_ranges=len(plan.layouts),
            payload_bytes=plan.payload_bytes,
        )

    @staticmethod
    def _plans_are_exactly_adjacent(
        left: _ExpertReadPlan, right: _ExpertReadPlan
    ) -> bool:
        if len(left.layouts) != len(right.layouts):
            return False
        return all(
            left_layout[0] == right_layout[0]
            and left_layout[1] + left_layout[2] == right_layout[1]
            for left_layout, right_layout in zip(
                left.layouts, right.layouts, strict=True
            )
        )

    def _batch_expert_plans(
        self, plans: tuple[_ExpertReadPlan, ...]
    ) -> tuple[_ExpertPrefetchBatch, ...]:
        raw_bytes_many = getattr(self.source, "raw_bytes_many", None)
        batches: list[_ExpertPrefetchBatch] = []
        capacity = min(
            self.EXPERT_PREFETCH_MAX_OUTSTANDING,
            self.EXPERT_PREFETCH_MAX_EXPERTS,
        )
        for chunk_start in range(0, len(plans), capacity):
            chunk = plans[chunk_start : chunk_start + capacity]
            index = 0
            while index < len(chunk):
                width = 1
                if (
                    callable(raw_bytes_many)
                    and self.expert_range_coalesce_max_experts >= 2
                    and index + 1 < len(chunk)
                    and self._plans_are_exactly_adjacent(chunk[index], chunk[index + 1])
                ):
                    width = 2
                experts = chunk[index : index + width]
                requests_avoided = len(experts[0].layouts) if len(experts) == 2 else 0
                plan = _ExpertBatchPlan(
                    experts=experts,
                    payload_bytes=sum(expert.payload_bytes for expert in experts),
                    range_requests_avoided=requests_avoided,
                )
                batches.append(
                    _ExpertPrefetchBatch(
                        plan=plan,
                        future=None,
                        remaining=len(experts),
                    )
                )
                index += width
        return tuple(batches)

    def _read_expert_batch(self, plan: _ExpertBatchPlan) -> _CoalescedExpertBatch:
        """Read one or two planned experts through exact leaf-cache keys."""

        raw_bytes_many = getattr(self.source, "raw_bytes_many", None)
        if not callable(raw_bytes_many):
            experts = {
                expert.base: self._read_expert_plan(expert) for expert in plan.experts
            }
            return _CoalescedExpertBatch(
                experts=experts,
                resident_bytes=plan.payload_bytes,
                source_requests=sum(len(expert.layouts) for expert in plan.experts),
                source_bytes=plan.payload_bytes,
            )

        indexed: dict[
            str,
            list[
                tuple[
                    int,
                    int,
                    tuple[str, int, int, tuple[dict[str, Any], ...]],
                ]
            ],
        ] = {}
        for expert_index, expert in enumerate(plan.experts):
            for layout_index, layout in enumerate(expert.layouts):
                indexed.setdefault(layout[0], []).append(
                    (expert_index, layout_index, layout)
                )

        parts_by_layout: dict[tuple[int, int], memoryview] = {}
        resident_bytes = 0
        source_requests = 0
        source_bytes = 0
        for shard, entries in indexed.items():
            requested = tuple((layout[1], layout[2]) for _, _, layout in entries)
            result = raw_bytes_many(
                shard,
                requested,
                resident_limit_bytes=sum(length for _, length in requested),
                max_gap_bytes=self.EXPERT_RANGE_COALESCE_MAX_GAP_BYTES,
            )
            parts = tuple(result.parts)
            if len(parts) != len(entries):
                raise DeepSeekPagerError(
                    f"multi-range part count mismatch for {shard}: "
                    f"{len(parts)}/{len(entries)}"
                )
            batch_resident = int(result.resident_bytes)
            batch_requests = int(result.source_requests)
            batch_source_bytes = int(result.source_bytes)
            requested_bytes = sum(length for _, length in requested)
            if (
                batch_resident != requested_bytes
                or batch_requests < 0
                or batch_source_bytes < 0
            ):
                raise DeepSeekPagerError(
                    f"invalid multi-range receipt for {shard}: resident="
                    f"{batch_resident}, requests={batch_requests}, "
                    f"source_bytes={batch_source_bytes}"
                )
            resident_bytes += batch_resident
            source_requests += batch_requests
            source_bytes += batch_source_bytes
            for (expert_index, layout_index, layout), part in zip(
                entries, parts, strict=True
            ):
                view = part if isinstance(part, memoryview) else memoryview(part)
                if not view.readonly or len(view) != layout[2]:
                    raise DeepSeekPagerError(
                        f"invalid readonly multi-range part for {shard}: "
                        f"{len(view)}/{layout[2]} bytes"
                    )
                parts_by_layout[(expert_index, layout_index)] = view

        if resident_bytes != plan.payload_bytes:
            raise DeepSeekPagerError(
                "multi-range resident bytes do not match the exact expert payload"
            )
        experts: dict[str, _CoalescedExpert] = {}
        for expert_index, expert in enumerate(plan.experts):
            payloads = tuple(
                parts_by_layout[(expert_index, layout_index)]
                for layout_index in range(len(expert.layouts))
            )
            experts[expert.base] = self._expert_from_layout_payloads(expert, payloads)
        return _CoalescedExpertBatch(
            experts=experts,
            resident_bytes=resident_bytes,
            source_requests=source_requests,
            source_bytes=source_bytes,
        )

    def _coalesced_expert(self, base: str) -> _CoalescedExpert | None:
        """Synchronously fetch an adjacent expert or decline safely."""

        plan = self._expert_plan(base)
        return None if plan is None else self._read_expert_plan(plan)

    def _window_resident_bytes_locked(self, window: _ExpertPrefetchWindow) -> int:
        return sum(
            batch.plan.payload_bytes
            for batch in window.batches
            if batch.submitted and not batch.retired
        )

    def _submit_window_reads_locked(self, window: _ExpertPrefetchWindow) -> None:
        if window is not self._active_prefetch_window or window.closed:
            raise DeepSeekPagerError("expert prefetch window is stale or closed")
        executor = window.executor
        if executor is None:
            raise DeepSeekPagerError("expert prefetch window has no executor")
        while window.next_submit < len(window.batches):
            outstanding = sum(
                batch.remaining
                for batch in window.batches
                if batch.submitted and not batch.retired
            )
            batch = window.batches[window.next_submit]
            expert_capacity = min(
                self.EXPERT_PREFETCH_MAX_OUTSTANDING,
                self.EXPERT_PREFETCH_MAX_EXPERTS,
            )
            if outstanding + batch.remaining > expert_capacity:
                break
            resident_after = (
                self._window_resident_bytes_locked(window) + batch.plan.payload_bytes
            )
            if resident_after > self.EXPERT_PREFETCH_RESIDENT_LIMIT_BYTES:
                break
            # ThreadPoolExecutor deliberately starts workers with an empty
            # context. Capture each submission independently so request-local
            # access-trace tags follow both initially queued and later sliding
            # window reads without serializing or disabling prefetch.
            context = window.submission_context.copy()
            batch.future = executor.submit(
                context.run,
                self._read_expert_batch,
                batch.plan,
            )
            batch.submitted = True
            window.next_submit += 1
            self._stats.expert_prefetch_submitted += batch.remaining
            self._stats.expert_prefetch_batches_submitted += 1
            self._stats.expert_prefetch_payload_bytes += batch.plan.payload_bytes
            outstanding += batch.remaining
            self._stats.expert_prefetch_max_outstanding = max(
                self._stats.expert_prefetch_max_outstanding,
                outstanding,
            )
            self._stats.expert_prefetch_peak_bytes = max(
                self._stats.expert_prefetch_peak_bytes,
                resident_after,
            )

    def prefetch_expert_window(
        self, ordered_bases: Iterable[str]
    ) -> _ExpertPrefetchWindow | None:
        """Plan every routed expert, then open one exact bounded I/O window."""

        bases = tuple(ordered_bases)
        if not bases or any(not isinstance(base, str) or not base for base in bases):
            raise ValueError("expert bases must be a non-empty sequence of strings")
        if len(set(bases)) != len(bases):
            raise ValueError("expert prefetch window bases must be unique")
        if not self.expert_prefetch_enabled:
            return None
        with self._prefetch_lock:
            if self._draining_prefetch or self._draining_executor is not None:
                self._stats.expert_prefetch_sync_fallbacks += 1
                return None
            if self._active_prefetch_window is not None:
                raise DeepSeekPagerError("expert prefetch window is already active")

        # Discovery is deliberately complete before the executor exists: an
        # invalid final expert therefore causes zero speculative source reads.
        plans: list[_ExpertReadPlan] = []
        for base in bases:
            plan = self._expert_plan(base)
            if plan is None:
                self._stats.expert_prefetch_sync_fallbacks += 1
                return None
            plans.append(plan)
        if any(
            plan.payload_bytes > self.EXPERT_PREFETCH_PAYLOAD_LIMIT_BYTES
            for plan in plans
        ) or any(
            sum(
                plan.payload_bytes
                for plan in plans[start : start + self.EXPERT_PREFETCH_MAX_EXPERTS]
            )
            > self.EXPERT_PREFETCH_RESIDENT_LIMIT_BYTES
            for start in range(len(plans))
        ):
            self._stats.expert_prefetch_sync_fallbacks += 1
            return None

        frozen_plans = tuple(plans)
        batches = self._batch_expert_plans(frozen_plans)
        executor = ThreadPoolExecutor(
            max_workers=self.EXPERT_PREFETCH_WORKERS,
            thread_name_prefix="immer-v4-expert-prefetch",
        )
        window = _ExpertPrefetchWindow(
            bases=bases,
            tickets=(),
            batches=batches,
            executor=executor,
            submission_context=copy_context(),
        )
        batch_by_base = {
            expert.base: batch for batch in batches for expert in batch.plan.experts
        }
        tickets = tuple(
            _ExpertPrefetch(
                base=plan.base,
                plan=plan,
                batch=batch_by_base[plan.base],
                payload_bytes=plan.payload_bytes,
                index=index,
                window=window,
            )
            for index, plan in enumerate(frozen_plans)
        )
        window.tickets = tickets
        declined = False
        try:
            with self._prefetch_lock:
                if self._draining_prefetch or self._draining_executor is not None:
                    self._stats.expert_prefetch_sync_fallbacks += 1
                    declined = True
                elif self._active_prefetch_window is not None:
                    raise DeepSeekPagerError("expert prefetch window is already active")
                else:
                    self._prefetch_executor = executor
                    self._active_prefetch_window = window
                    self._submit_window_reads_locked(window)
        except BaseException:
            with self._prefetch_lock:
                if self._active_prefetch_window is window:
                    self._active_prefetch_window = None
                if self._prefetch_executor is executor:
                    self._prefetch_executor = None
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        if declined:
            executor.shutdown(wait=True, cancel_futures=True)
            return None
        return window

    def consume_expert_window(
        self, window: _ExpertPrefetchWindow, expected_base: str
    ) -> _ExpertPayload:
        """Consume the next planned expert without changing compute order."""

        with self._prefetch_lock:
            if window is not self._active_prefetch_window or window.closed:
                raise DeepSeekPagerError("expert prefetch window is stale or closed")
            if self._active_prefetch_payload is not None:
                raise DeepSeekPagerError(
                    "previous expert prefetch payload is still active"
                )
            if window.next_consume >= len(window.tickets):
                raise DeepSeekPagerError("expert prefetch window is already consumed")
            ticket = window.tickets[window.next_consume]
            if ticket.base != expected_base:
                raise DeepSeekPagerError(
                    "expert prefetch window consume is out of order"
                )
            batch = ticket.batch
            if ticket.consumed or batch.retired or batch.future is None:
                raise DeepSeekPagerError("expert prefetch ticket is stale or consumed")
            future = batch.future
            ready = future.done()
        started = time.perf_counter_ns()
        try:
            batch_result = future.result()
            expert = batch_result.experts[ticket.base]
        except BaseException as exc:
            self._stats.expert_prefetch_failures += 1
            self.close_expert_window(window, cancel=True)
            if isinstance(exc, KeyError):
                raise DeepSeekPagerError(
                    f"prefetch batch omitted planned expert {ticket.base}"
                ) from exc
            raise
        waited = time.perf_counter_ns() - started
        payload = _ExpertPayload(
            base=ticket.base,
            expert=expert,
            payload_bytes=ticket.payload_bytes,
            window=window,
            batch=batch,
        )
        try:
            with self._prefetch_lock:
                if window is not self._active_prefetch_window or window.closed:
                    raise DeepSeekPagerError(
                        "expert prefetch window changed during consume"
                    )
                if ticket is not window.tickets[window.next_consume]:
                    raise DeepSeekPagerError(
                        "expert prefetch ticket changed during consume"
                    )
                if batch.future is not future or batch.retired:
                    raise DeepSeekPagerError(
                        "expert prefetch batch changed during consume"
                    )
                ticket.consumed = True
                window.next_consume += 1
                self._active_prefetch_payload = payload
                self._stats.expert_prefetch_consumed += 1
                self._stats.expert_prefetch_wait_ns += waited
                self._account_expert_batch_result_locked(batch, batch_result)
                if ready:
                    self._stats.expert_prefetch_ready_before_consume += 1
                self._submit_window_reads_locked(window)
        except BaseException:
            self.close_expert_window(window, cancel=True)
            raise
        return payload

    def _account_expert_batch_result_locked(
        self,
        batch: _ExpertPrefetchBatch,
        result: _CoalescedExpertBatch,
    ) -> None:
        if batch.metrics_accounted:
            return
        batch.metrics_accounted = True
        self._stats.expert_transport_envelopes += result.source_requests
        self._stats.expert_transport_source_bytes += result.source_bytes
        self._stats.expert_range_requests_avoided += batch.plan.range_requests_avoided

    def close_expert_window(
        self, window: _ExpertPrefetchWindow, *, cancel: bool = False
    ) -> None:
        """Close a completed window or detach at most two failed I/O reads."""

        if cancel:
            self._cancel_expert_window(window)
            return
        with self._prefetch_lock:
            if window is not self._active_prefetch_window or window.closed:
                raise DeepSeekPagerError("expert prefetch window is stale or closed")
            if self._active_prefetch_payload is not None:
                raise DeepSeekPagerError(
                    "cannot close expert prefetch window with an active payload"
                )
            if window.next_consume != len(window.tickets) or any(
                not batch.retired or batch.future is not None or batch.remaining
                for batch in window.batches
            ):
                raise DeepSeekPagerError(
                    "cannot close expert prefetch window before ordered consumption"
                )
            window.closed = True
            executor = window.executor
            window.executor = None
            self._active_prefetch_window = None
            if self._prefetch_executor is executor:
                self._prefetch_executor = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    def _cancel_expert_window(self, window: _ExpertPrefetchWindow) -> None:
        done: list[tuple[Future[_CoalescedExpertBatch], _ExpertPrefetchBatch]] = []
        draining: list[tuple[Future[_CoalescedExpertBatch], _ExpertPrefetchBatch]] = []
        with self._prefetch_lock:
            if window is not self._active_prefetch_window or window.closed:
                raise DeepSeekPagerError("expert prefetch window is stale or closed")
            window.closed = True
            for batch in window.batches:
                future = batch.future
                if future is None:
                    batch.remaining = 0
                    batch.retired = True
                    continue
                future.cancel()
                batch.future = None
                batch.remaining = 0
                batch.retired = True
                if future.done():
                    done.append((future, batch))
                else:
                    draining.append((future, batch))
            for ticket in window.tickets:
                if not ticket.consumed:
                    ticket.consumed = True
                    self._stats.expert_prefetch_cancelled += 1
            payload = self._active_prefetch_payload
            if payload is not None and payload.window is window:
                payload.expert = None
                payload.consumed = True
                self._active_prefetch_payload = None
                self._stats.expert_prefetch_cancelled += 1
            executor = window.executor
            window.executor = None
            self._active_prefetch_window = None
            if self._prefetch_executor is executor:
                self._prefetch_executor = None
            if len(draining) > self.EXPERT_PREFETCH_ACTIVE_READ_LIMIT:
                raise DeepSeekPagerError("expert prefetch drain bound was exceeded")
            self._draining_prefetch.update(future for future, _batch in draining)
            if draining:
                self._draining_executor = executor
        if executor is not None:
            executor.shutdown(wait=not draining, cancel_futures=True)
        for future, batch in done:
            self._finish_cancelled_prefetch(future, batch, detached=False)
        for future, batch in draining:
            future.add_done_callback(
                lambda completed, owned_batch=batch: self._finish_cancelled_prefetch(
                    completed, owned_batch, detached=True
                )
            )

    def _finish_cancelled_prefetch(
        self,
        future: Future[_CoalescedExpertBatch],
        batch: _ExpertPrefetchBatch,
        *,
        detached: bool,
    ) -> None:
        """Retire one of at most two detached exact source reads."""

        try:
            result = future.result()
        except BaseException:
            result = None
        with self._prefetch_lock:
            if result is not None:
                self._account_expert_batch_result_locked(batch, result)
            if detached:
                self._draining_prefetch.discard(future)
            # Keep ownership of the shut-down executor after its final future
            # completes.  A layer release can reap it once that is immediate;
            # terminal close must join it before the TensorSource may close.

    @staticmethod
    def _reservoir_batch_plan(plan: _ExpertReadPlan) -> _ExpertBatchPlan:
        return _ExpertBatchPlan(
            experts=(plan,),
            payload_bytes=plan.payload_bytes,
            range_requests_avoided=0,
        )

    def _account_reservoir_result(
        self,
        ticket: _ExpertReservoirTicket,
        result: _CoalescedExpertBatch,
    ) -> None:
        with self._prefetch_lock:
            if ticket.accounted:
                return
            ticket.accounted = True
            self._stats.expert_reservoir_source_requests += result.source_requests
            self._stats.expert_reservoir_source_bytes += result.source_bytes

    def _retire_reservoir(self, reservoir: _ExpertReservoir) -> None:
        if all(ticket.future.done() for ticket in reservoir.tickets):
            for ticket in reservoir.tickets:
                try:
                    result = ticket.future.result()
                except BaseException:
                    continue
                self._account_reservoir_result(ticket, result)
            return
        with self._prefetch_lock:
            self._retired_expert_reservoirs.append(reservoir)

    def _reap_retired_reservoirs(self, *, wait: bool) -> None:
        with self._prefetch_lock:
            reservoirs = tuple(self._retired_expert_reservoirs)
        for reservoir in reservoirs:
            if not wait and any(
                not ticket.future.done() for ticket in reservoir.tickets
            ):
                continue
            for ticket in reservoir.tickets:
                try:
                    result = ticket.future.result()
                except BaseException:
                    continue
                self._account_reservoir_result(ticket, result)
            with self._prefetch_lock:
                if reservoir in self._retired_expert_reservoirs:
                    self._retired_expert_reservoirs.remove(reservoir)

    def prefetch_expert_reservoir(
        self, ordered_bases: Iterable[str]
    ) -> _ExpertReservoir | None:
        """Start ranked next-layer reads under a byte budget, not an expert cap."""

        bases = tuple(ordered_bases)
        if not bases or any(not isinstance(base, str) or not base for base in bases):
            raise ValueError("reservoir bases must be a non-empty sequence of strings")
        if len(set(bases)) != len(bases):
            raise ValueError("reservoir bases must be unique")
        with self._prefetch_lock:
            if self._active_expert_reservoir is not None:
                raise DeepSeekPagerError("an expert reservoir is already active")
        self._reap_retired_reservoirs(wait=False)

        plans: list[_ExpertReadPlan] = []
        payload_bytes = 0
        for base in bases:
            plan = self._expert_plan(base)
            if plan is None:
                continue
            candidate_bytes = payload_bytes + plan.payload_bytes
            if candidate_bytes > self.expert_reservoir_budget_bytes:
                break
            plans.append(plan)
            payload_bytes = candidate_bytes
        if not plans:
            return None

        with self._prefetch_lock:
            executor = self._expert_reservoir_executor
            if executor is None:
                executor = ThreadPoolExecutor(
                    max_workers=self.expert_reservoir_workers,
                    thread_name_prefix="immer-v4-expert-reservoir",
                )
                self._expert_reservoir_executor = executor
        context = copy_context()
        tickets: list[_ExpertReservoirTicket] = []
        try:
            for index, plan in enumerate(plans):
                future = executor.submit(
                    context.copy().run,
                    self._read_expert_batch,
                    self._reservoir_batch_plan(plan),
                )
                tickets.append(
                    _ExpertReservoirTicket(
                        base=plan.base,
                        plan=plan,
                        future=future,
                        index=index,
                    )
                )
        except BaseException:
            for ticket in tickets:
                ticket.future.cancel()
            if tickets:
                self._retire_reservoir(
                    _ExpertReservoir(
                        bases=tuple(ticket.base for ticket in tickets),
                        tickets=tuple(tickets),
                        executor=executor,
                        payload_bytes=sum(
                            ticket.plan.payload_bytes for ticket in tickets
                        ),
                        closed=True,
                    )
                )
            raise
        reservoir = _ExpertReservoir(
            bases=tuple(plan.base for plan in plans),
            tickets=tuple(tickets),
            executor=executor,
            payload_bytes=payload_bytes,
        )
        with self._prefetch_lock:
            if self._active_expert_reservoir is not None:
                for ticket in reservoir.tickets:
                    ticket.future.cancel()
                reservoir.closed = True
                self._retired_expert_reservoirs.append(reservoir)
                raise DeepSeekPagerError("an expert reservoir is already active")
            self._active_expert_reservoir = reservoir
            self._stats.expert_reservoir_submitted += len(tickets)
            self._stats.expert_reservoir_payload_bytes += payload_bytes
        return reservoir

    def bind_expert_reservoir(
        self,
        reservoir: _ExpertReservoir,
        exact_bases: Iterable[str],
    ) -> dict[str, _ExpertReservoirPayload]:
        """Bind predictions to exact demand and return only verified payload hits."""

        bases = tuple(exact_bases)
        if not bases or any(not isinstance(base, str) or not base for base in bases):
            raise ValueError("exact bases must be a non-empty sequence of strings")
        if len(set(bases)) != len(bases):
            raise ValueError("exact bases must be unique")
        with self._prefetch_lock:
            if reservoir is not self._active_expert_reservoir or reservoir.closed:
                raise DeepSeekPagerError("expert reservoir is stale or closed")
            reservoir.closed = True
            self._active_expert_reservoir = None

        exact = set(bases)
        predicted = {ticket.base for ticket in reservoir.tickets}
        predicted_hits = exact & predicted
        with self._prefetch_lock:
            self._stats.expert_reservoir_prediction_hits += len(predicted_hits)
            self._stats.expert_reservoir_misses += len(exact - predicted)

        payloads: dict[str, _ExpertReservoirPayload] = {}
        for ticket in reservoir.tickets:
            if ticket.base not in exact:
                cancelled = ticket.future.cancel()
                with self._prefetch_lock:
                    self._stats.expert_reservoir_wasted += 1
                    self._stats.expert_reservoir_wasted_payload_bytes += (
                        ticket.plan.payload_bytes
                    )
                    self._stats.expert_reservoir_cancelled += int(cancelled)
                continue
            ready = ticket.future.done()
            started = time.perf_counter_ns()
            try:
                result = ticket.future.result()
                expert = result.experts[ticket.base]
            except BaseException:
                with self._prefetch_lock:
                    self._stats.expert_reservoir_failures += 1
                    self._stats.expert_reservoir_misses += 1
                continue
            waited = time.perf_counter_ns() - started
            self._account_reservoir_result(ticket, result)
            payloads[ticket.base] = _ExpertReservoirPayload(
                owner=self._expert_reservoir_owner,
                base=ticket.base,
                expert=expert,
                payload_bytes=ticket.plan.payload_bytes,
            )
            with self._prefetch_lock:
                self._stats.expert_reservoir_usable_hits += 1
                self._stats.expert_reservoir_wait_ns += waited
                self._stats.expert_reservoir_hit_payload_bytes += (
                    ticket.plan.payload_bytes
                )
                self._stats.expert_reservoir_ready_hits += int(ready)

        self._retire_reservoir(reservoir)
        return payloads

    def cancel_expert_reservoir(self, reservoir: _ExpertReservoir) -> None:
        """Cancel a prediction plan that cannot be aligned to the next block."""

        with self._prefetch_lock:
            if reservoir is not self._active_expert_reservoir or reservoir.closed:
                raise DeepSeekPagerError("expert reservoir is stale or closed")
            reservoir.closed = True
            self._active_expert_reservoir = None
        cancelled = 0
        for ticket in reservoir.tickets:
            cancelled += int(ticket.future.cancel())
        with self._prefetch_lock:
            self._stats.expert_reservoir_cancelled += cancelled
        self._retire_reservoir(reservoir)

    def _claim_reservoir_payload(
        self,
        payload: _ExpertReservoirPayload,
        expected_base: str,
    ) -> _CoalescedExpert:
        with self._prefetch_lock:
            if (
                payload.owner is not self._expert_reservoir_owner
                or payload.consumed
                or payload.base != expected_base
            ):
                raise DeepSeekPagerError(
                    "expert reservoir payload is stale or mismatched"
                )
            expert = payload.expert
            if expert is None:
                raise DeepSeekPagerError("expert reservoir payload is empty")
            payload.consumed = True
            return expert

    def _release_reservoir_payload(self, payload: _ExpertReservoirPayload) -> None:
        with self._prefetch_lock:
            if (
                payload.owner is not self._expert_reservoir_owner
                or not payload.consumed
                or payload.expert is None
            ):
                raise DeepSeekPagerError("expert reservoir payload release is invalid")
            payload.expert = None

    def discard_expert_reservoir_payload(
        self, payload: _ExpertReservoirPayload
    ) -> None:
        with self._prefetch_lock:
            if payload.owner is not self._expert_reservoir_owner or payload.consumed:
                raise DeepSeekPagerError("expert reservoir payload is already consumed")
            payload.consumed = True
            payload.expert = None

    # Compatibility adapter for the standalone transport smoke and external
    # callers written against exact-router-one-ahead/v1.
    def prefetch_expert(self, base: str) -> _ExpertPrefetch | None:
        window = self.prefetch_expert_window((base,))
        if window is None:
            return None
        window.compatibility_single = True
        return window.tickets[0]

    def consume_expert_prefetch(
        self, ticket: _ExpertPrefetch, expected_base: str
    ) -> _ExpertPayload:
        window = ticket.window
        if window is None or window.tickets[0] is not ticket or ticket.consumed:
            raise DeepSeekPagerError("expert prefetch ticket is stale or consumed")
        payload = self.consume_expert_window(window, expected_base)
        with self._prefetch_lock:
            if window is not self._active_prefetch_window or window.closed:
                raise DeepSeekPagerError("expert prefetch window is stale or closed")
            window.closed = True
            executor = window.executor
            window.executor = None
            self._active_prefetch_window = None
            if self._prefetch_executor is executor:
                self._prefetch_executor = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        return payload

    def cancel_expert_prefetch(self, ticket: _ExpertPrefetch) -> None:
        window = ticket.window
        if window is None or window.tickets[0] is not ticket or ticket.consumed:
            raise DeepSeekPagerError("expert prefetch ticket is stale or consumed")
        self.close_expert_window(window, cancel=True)

    def _claim_expert_payload(
        self, payload: _ExpertPayload, expected_base: str
    ) -> _CoalescedExpert:
        with self._prefetch_lock:
            if payload is not self._active_prefetch_payload or payload.consumed:
                raise DeepSeekPagerError("expert prefetch payload is stale or consumed")
            if payload.base != expected_base:
                raise DeepSeekPagerError("expert prefetch payload base mismatch")
            expert = payload.expert
            if expert is None:
                raise DeepSeekPagerError("expert prefetch payload is empty")
            payload.consumed = True
        return expert

    def _retire_expert_payload_locked(self, payload: _ExpertPayload) -> None:
        batch = payload.batch
        if batch.retired or batch.remaining <= 0:
            raise DeepSeekPagerError("expert prefetch batch release is invalid")
        batch.remaining -= 1
        if batch.remaining == 0:
            batch.future = None
            batch.retired = True
        window = payload.window
        if window is self._active_prefetch_window and not window.closed:
            self._submit_window_reads_locked(window)

    def discard_expert_payload(self, payload: _ExpertPayload) -> None:
        """Release an unclaimed payload when planning the next read fails."""

        with self._prefetch_lock:
            if payload is not self._active_prefetch_payload or payload.consumed:
                raise DeepSeekPagerError("expert prefetch payload is stale or consumed")
            payload.consumed = True
            payload.expert = None
            self._active_prefetch_payload = None
            self._retire_expert_payload_locked(payload)
            self._stats.expert_prefetch_cancelled += 1
            close_compatibility = (
                payload.window.compatibility_single and not payload.window.closed
            )
        if close_compatibility:
            self.close_expert_window(payload.window)

    def _release_expert_payload(self, payload: _ExpertPayload) -> None:
        with self._prefetch_lock:
            if payload is not self._active_prefetch_payload or not payload.consumed:
                raise DeepSeekPagerError("expert prefetch payload release is invalid")
            payload.expert = None
            self._active_prefetch_payload = None
            self._retire_expert_payload_locked(payload)
            close_compatibility = (
                payload.window.compatibility_single and not payload.window.closed
            )
        if close_compatibility:
            self.close_expert_window(payload.window)

    def _priority_scope(self, name: str, priority: int | None = None) -> Any:
        callback = getattr(self.source, "cache_priority", None)
        if not callable(callback):
            return nullcontext()
        if priority is None:
            priority = 0 if ".ffn.experts." in name else 1
        return callback(priority)

    @staticmethod
    def _decoded_float_array(value: Any, name: str) -> np.ndarray:
        array = np.asarray(value)
        if array.dtype.kind != "f":
            raise DeepSeekPagerError(
                f"decoded quantized tensor {name} must contain floating-point values"
            )
        result = np.ascontiguousarray(array, dtype=np.float32)
        if not np.isfinite(result).all():
            raise DeepSeekPagerError(f"decoded quantized tensor {name} is non-finite")
        return result

    @classmethod
    def _encoded_weight_parts(
        cls,
        raw: Any,
        scale: Any,
        *,
        storage_dtype: str,
        weight_name: str,
        logical_bytes: int,
    ) -> _MaterializedWeight:
        scales = cls._decoded_float_array(
            scale, weight_name.removesuffix("weight") + "scale"
        )
        if scales.ndim != 2 or np.any(scales <= 0):
            raise DeepSeekPagerError(
                f"MX scale for {weight_name} must be a positive 2D matrix"
            )
        if storage_dtype in {"F8_E4M3", "F8_E4M3FN"}:
            values = cls._decoded_float_array(raw, weight_name)
            if values.ndim != 2:
                raise DeepSeekPagerError(f"MXFP8 weight {weight_name} must be 2D")
            expected = (
                (values.shape[0] + 127) // 128,
                (values.shape[1] + 127) // 128,
            )
        elif storage_dtype == "I8":
            try:
                values = unpack_fp4_e2m1(raw)
            except (TypeError, ValueError) as exc:
                raise DeepSeekPagerError(
                    f"invalid packed MXFP4 weight {weight_name}: {exc}"
                ) from exc
            if values.ndim != 2:
                raise DeepSeekPagerError(f"MXFP4 weight {weight_name} must be 2D")
            expected = (values.shape[0], (values.shape[1] + 31) // 32)
        else:  # pragma: no cover - callers guard the encoding
            raise DeepSeekPagerError(
                f"unsupported encoded weight dtype {storage_dtype!r} at {weight_name}"
            )
        if scales.shape != expected:
            raise DeepSeekPagerError(
                f"MX scale shape {scales.shape} for {weight_name} does not match {expected}"
            )
        return _MaterializedWeight(
            values=np.ascontiguousarray(values, dtype=np.float32),
            scales=scales,
            storage_dtype=storage_dtype,
            logical_bytes=logical_bytes,
        )

    def _materialize_weight(self, prefix: str) -> _MaterializedWeight:
        weight_name = self._weight_name(prefix)
        meta = self.source.find(weight_name)
        storage_dtype = str(meta["dtype"]).upper()
        logical_bytes = self._payload_bytes(meta)
        with self._priority_scope(weight_name):
            raw = self.source.tensor(weight_name)
            if storage_dtype in {"F8_E4M3", "F8_E4M3FN"}:
                scale_name = weight_name.removesuffix("weight") + "scale"
                scale = self.source.tensor(scale_name)
                logical_bytes += self._payload_bytes(self.source.find(scale_name))
                weight = self._encoded_weight_parts(
                    raw,
                    scale,
                    storage_dtype=storage_dtype,
                    weight_name=weight_name,
                    logical_bytes=logical_bytes,
                )
            elif storage_dtype == "I8":
                scale_name = weight_name.removesuffix("weight") + "scale"
                scale = self.source.tensor(scale_name)
                logical_bytes += self._payload_bytes(self.source.find(scale_name))
                weight = self._encoded_weight_parts(
                    raw,
                    scale,
                    storage_dtype=storage_dtype,
                    weight_name=weight_name,
                    logical_bytes=logical_bytes,
                )
            elif storage_dtype in {"BF16", "F16", "F32"}:
                values = np.ascontiguousarray(raw, dtype=np.float32)
                if values.ndim != 2:
                    raise DeepSeekPagerError(f"linear weight {weight_name} must be 2D")
                weight = _MaterializedWeight(
                    values=values,
                    scales=None,
                    storage_dtype=storage_dtype,
                    logical_bytes=logical_bytes,
                )
            else:
                raise DeepSeekPagerError(
                    f"unsupported linear weight dtype {storage_dtype!r} at {weight_name}"
                )
        self._record_materialization(weight)
        return weight

    def _materialize_coalesced_weight(
        self, payload: _CoalescedExpert, prefix: str
    ) -> _MaterializedWeight:
        weight_name = self._weight_name(prefix)
        scale_name = weight_name.removesuffix("weight") + "scale"
        try:
            encoded_weight = payload.tensors[weight_name]
            encoded_scale = payload.tensors[scale_name]
        except KeyError as exc:  # pragma: no cover - layout builder owns this invariant
            raise DeepSeekPagerError(
                f"coalesced expert is missing {exc.args[0]!r}"
            ) from exc
        raw = self._decode_coalesced_tensor(encoded_weight)
        scale = self._decode_coalesced_tensor(encoded_scale)
        storage_dtype = encoded_weight.dtype
        logical_bytes = encoded_weight.logical_bytes + encoded_scale.logical_bytes
        weight = self._encoded_weight_parts(
            raw,
            scale,
            storage_dtype=storage_dtype,
            weight_name=weight_name,
            logical_bytes=logical_bytes,
        )
        self._record_materialization(weight)
        return weight

    def _record_materialization(self, weight: _MaterializedWeight) -> None:
        materialized = int(weight.values.nbytes)
        self._stats.logical_weight_bytes += weight.logical_bytes
        self._stats.materialized_float_bytes += materialized
        if weight.scales is not None:
            self._stats.materialized_scale_bytes += int(weight.scales.nbytes)
        self._stats.peak_single_weight_bytes = max(
            self._stats.peak_single_weight_bytes, materialized
        )

    @staticmethod
    def _dequantize_materialized(weight: _MaterializedWeight) -> np.ndarray:
        if weight.scales is None:
            return weight.values
        if weight.storage_dtype in {"F8_E4M3", "F8_E4M3FN"}:
            return dequantize_fp8_e4m3(weight.values, weight.scales)
        if weight.storage_dtype == "I8":
            full_scale = np.repeat(weight.scales, 32, axis=1)[
                :, : weight.values.shape[1]
            ]
            return np.ascontiguousarray(weight.values * full_scale, dtype=np.float32)
        raise DeepSeekPagerError(
            f"cannot dequantize unsupported weight dtype {weight.storage_dtype!r}"
        )

    def _prepare_linear_input(
        self, x: Any, *, quantized: bool, compute_dtype: Any
    ) -> Any:
        torch = self.torch
        if quantized:
            cpu_x = np.ascontiguousarray(x.detach().to("cpu", torch.float32).numpy())
            values, scales = quantize_fp8_e4m3_parts(cpu_x, block_size=128)
            return _QuantizedActivation(values=values, scales=scales)
        return x.to(self.device, dtype=compute_dtype)

    def _block_scaled_linear(
        self,
        activation: _QuantizedActivation,
        weight: _MaterializedWeight,
        *,
        result_dtype: Any,
        result_device: Any,
        row_start: int = 0,
        row_stop: int | None = None,
    ) -> Any:
        """Execute the published MXFP8/MXFP4 reduction order exactly.

        Raw E4M3/E2M1 values are multiplied first, each K-tile result is
        scaled, and those scaled tiles are accumulated in FP32.  Scaling a
        fully dequantized matrix in one BF16 GEMM is numerically different and
        changes V4 logits after enough decoder layers.
        """

        torch = self.torch
        if weight.scales is None or weight.storage_dtype not in {
            "F8_E4M3",
            "F8_E4M3FN",
            "I8",
        }:
            raise DeepSeekPagerError("block-scaled GEMM requires an MX weight")
        if (
            activation.values.ndim < 1
            or activation.values.shape[-1] != weight.values.shape[1]
        ):
            raise DeepSeekPagerError(
                "quantized activation K dimension does not match checkpoint weight"
            )
        k = int(weight.values.shape[1])
        if k % 128:
            raise DeepSeekPagerError(
                f"official MX activation quantizer requires K divisible by 128, got {k}"
            )
        n = int(weight.values.shape[0])
        if row_stop is None:
            row_stop = n
        if not 0 <= row_start < row_stop <= n:
            raise ValueError("weight row interval is outside the materialized matrix")

        leading_shape = tuple(int(value) for value in activation.values.shape[:-1])
        qx = torch.from_numpy(activation.values.reshape(-1, k)).to(
            self.device, dtype=torch.float32
        )
        sx = torch.from_numpy(activation.scales.reshape(-1, k // 128)).to(
            self.device, dtype=torch.float32
        )
        qw = torch.from_numpy(
            np.ascontiguousarray(weight.values[row_start:row_stop])
        ).to(self.device, dtype=torch.float32)
        if weight.storage_dtype in {"F8_E4M3", "F8_E4M3FN"}:
            row_groups = np.arange(row_start, row_stop, dtype=np.int64) // 128
            selected_scales = np.ascontiguousarray(weight.scales[row_groups])
            block_k = 128
        else:
            selected_scales = np.ascontiguousarray(weight.scales[row_start:row_stop])
            block_k = 32
        sw = torch.from_numpy(selected_scales).to(self.device, dtype=torch.float32)
        accumulator = torch.zeros(
            (qx.shape[0], row_stop - row_start),
            device=self.device,
            dtype=torch.float32,
        )
        try:
            for k_start in range(0, k, block_k):
                k_stop = k_start + block_k
                k_block = k_start // block_k
                partial = torch.nn.functional.linear(
                    qx[:, k_start:k_stop], qw[:, k_start:k_stop]
                )
                activation_block = k_start // 128
                combined_scale = sx[:, activation_block, None] * sw[None, :, k_block]
                accumulator.add_(partial * combined_scale)
                if block_k == 128:
                    self._stats.fp8_k128_tiles += 1
                else:
                    self._stats.fp4_k32_tiles += 1
            result = accumulator.reshape(*leading_shape, row_stop - row_start)
            result = result.to(dtype=result_dtype)
            if result_device != self.device:
                result = result.to(result_device)
            self._stats.block_scaled_linear_calls += 1
            return result
        finally:
            del accumulator
            del qw
            del sw
            del qx
            del sx

    def _linear_materialized(
        self,
        compute_x: Any,
        weight: _MaterializedWeight,
        *,
        quantized: bool,
        result_dtype: Any,
        result_device: Any,
        compute_dtype: Any,
    ) -> Any:
        if quantized:
            if not isinstance(compute_x, _QuantizedActivation):
                raise DeepSeekPagerError(
                    "exact quantized linear requires separate activation values/scales"
                )
            result = self._block_scaled_linear(
                compute_x,
                weight,
                result_dtype=result_dtype,
                result_device=result_device,
            )
            self._stats.linear_calls += 1
            return result

        torch = self.torch
        dequantized = self._dequantize_materialized(weight)
        compute_weight = torch.from_numpy(dequantized).to(
            self.device, dtype=compute_dtype
        )
        try:
            result = torch.nn.functional.linear(compute_x, compute_weight)
            result = result.to(dtype=result_dtype)
            if result_device != self.device:
                result = result.to(result_device)
            self._stats.linear_calls += 1
            return result
        finally:
            del compute_weight
            del dequantized

    def _release_materialized_weight(self) -> None:
        """Record an operation-local matrix release without flushing MPS.

        Each caller deletes its decoded NumPy matrix and each compute helper
        deletes the device tensor before reaching this hook.  PyTorch's MPS
        caching allocator can therefore reuse those blocks within a decoder
        layer.  Flushing here would turn every projection into a device-wide
        synchronization; :meth:`release` owns that work at the explicit layer
        boundary instead.
        """

        self._stats.materialized_weight_releases += 1

    def _uses_mps_allocator(self) -> bool:
        return self.device.type == "mps"

    def _purge_mps_cache(self) -> bool:
        """Purge the MPS allocator when supported, returning whether it ran."""

        if not self._uses_mps_allocator():
            return False
        mps = getattr(self.torch, "mps", None)
        empty_cache = getattr(mps, "empty_cache", None)
        if not callable(empty_cache):
            return False
        empty_cache()
        self._stats.mps_cache_purges += 1
        return True

    def linear(
        self,
        x: Any,
        prefix: str,
        *,
        activation_quantization: bool | None = None,
        output_dtype: Any | None = None,
        compute_dtype: Any | None = None,
    ) -> Any:
        """Run one official checkpoint matrix and release it before returning."""

        torch = self.torch
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        weight = self._materialize_weight(prefix)
        should_quantize = (
            self.simulate_activation_quantization
            if activation_quantization is None
            else bool(activation_quantization)
        ) and weight.storage_dtype in {"F8_E4M3", "F8_E4M3FN", "I8"}
        original_dtype = x.dtype
        active_dtype = compute_dtype or self.compute_dtype
        if active_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError(
                "compute_dtype must be a torch floating-point compute dtype"
            )
        compute_x = self._prepare_linear_input(
            x, quantized=should_quantize, compute_dtype=active_dtype
        )
        try:
            return self._linear_materialized(
                compute_x,
                weight,
                quantized=should_quantize,
                result_dtype=output_dtype or original_dtype,
                result_device=x.device,
                compute_dtype=active_dtype,
            )
        finally:
            del weight
            self._release_materialized_weight()

    def expert(
        self,
        x: Any,
        base: str,
        *,
        route_weight: Any | None = None,
        swiglu_limit: float = 0.0,
        prefetched_payload: _ExpertPayload | _ExpertReservoirPayload | None = None,
    ) -> Any:
        """Evaluate one SwiGLU expert with two source reads when possible.

        The official immutable shard layout stores the three encoded weights
        in one adjacent interval and the three scales in another.  They are
        fetched together but decoded and moved to the compute device strictly
        in execution order: ``w1``, ``w3``, then ``w2``.  The two up
        projections consume the very same activation-QDQ tensor.
        """

        torch = self.torch
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if not isinstance(base, str) or not base:
            raise ValueError("expert base must be a non-empty string")
        if not np.isfinite(float(swiglu_limit)) or float(swiglu_limit) < 0:
            raise ValueError("swiglu_limit must be finite and non-negative")

        claimed_payload = prefetched_payload
        if isinstance(claimed_payload, _ExpertPayload):
            payload = self._claim_expert_payload(claimed_payload, base)
        elif isinstance(claimed_payload, _ExpertReservoirPayload):
            payload = self._claim_reservoir_payload(claimed_payload, base)
        elif claimed_payload is None:
            payload = self._coalesced_expert(base)
        else:
            raise TypeError("prefetched_payload is not owned by this pager")
        try:
            result = self._expert_with_payload(
                x,
                base,
                payload=payload,
                route_weight=route_weight,
                swiglu_limit=swiglu_limit,
            )
        finally:
            if isinstance(claimed_payload, _ExpertPayload):
                # The local expert view may pin a two-expert envelope. Drop it
                # before retiring the batch so a refill cannot exceed the
                # resident-byte proof while this frame still owns the buffer.
                del payload
                self._release_expert_payload(claimed_payload)
            elif isinstance(claimed_payload, _ExpertReservoirPayload):
                del payload
                self._release_reservoir_payload(claimed_payload)
        return result

    def _expert_with_payload(
        self,
        x: Any,
        base: str,
        *,
        payload: _CoalescedExpert | None,
        route_weight: Any | None,
        swiglu_limit: float,
    ) -> Any:
        torch = self.torch
        if payload is None:
            materialize = self._materialize_weight
        else:

            def materialize(prefix: str) -> _MaterializedWeight:
                return self._materialize_coalesced_weight(payload, prefix)

            self._stats.coalesced_expert_calls += 1
            self._stats.expert_source_ranges += payload.source_ranges

        quantized_dtypes = {"F8_E4M3", "F8_E4M3FN", "I8"}
        original_dtype = x.dtype

        gate_weight = materialize(f"{base}.w1")
        gate_quantized = (
            self.simulate_activation_quantization
            and gate_weight.storage_dtype in quantized_dtypes
        )
        shared_input = self._prepare_linear_input(
            x,
            quantized=gate_quantized,
            compute_dtype=self.compute_dtype,
        )
        try:
            gate = self._linear_materialized(
                shared_input,
                gate_weight,
                quantized=gate_quantized,
                result_dtype=original_dtype,
                result_device=x.device,
                compute_dtype=self.compute_dtype,
            ).float()
        finally:
            del gate_weight
            self._release_materialized_weight()

        up_weight = materialize(f"{base}.w3")
        up_quantized = (
            self.simulate_activation_quantization
            and up_weight.storage_dtype in quantized_dtypes
        )
        up_input = (
            shared_input
            if up_quantized == gate_quantized
            else self._prepare_linear_input(
                x,
                quantized=up_quantized,
                compute_dtype=self.compute_dtype,
            )
        )
        try:
            up = self._linear_materialized(
                up_input,
                up_weight,
                quantized=up_quantized,
                result_dtype=original_dtype,
                result_device=x.device,
                compute_dtype=self.compute_dtype,
            ).float()
        finally:
            del up_weight
            self._release_materialized_weight()

        limit = float(swiglu_limit)
        if limit > 0:
            up = torch.clamp(up, min=-limit, max=limit)
            gate = torch.clamp(gate, max=limit)
        hidden = torch.nn.functional.silu(gate) * up
        if route_weight is not None:
            hidden = hidden * route_weight
        down_input = hidden.to(original_dtype)

        down_weight = materialize(f"{base}.w2")
        down_quantized = (
            self.simulate_activation_quantization
            and down_weight.storage_dtype in quantized_dtypes
        )
        compute_down = self._prepare_linear_input(
            down_input,
            quantized=down_quantized,
            compute_dtype=self.compute_dtype,
        )
        try:
            result = self._linear_materialized(
                compute_down,
                down_weight,
                quantized=down_quantized,
                result_dtype=original_dtype,
                result_device=x.device,
                compute_dtype=self.compute_dtype,
            )
        finally:
            del down_weight
            self._release_materialized_weight()
        self._stats.expert_calls += 1
        return result

    def grouped_linear(
        self,
        x: Any,
        prefix: str,
        *,
        groups: int,
        activation_quantization: bool = False,
        output_dtype: Any | None = None,
    ) -> Any:
        """Apply one independently parameterized matrix per group.

        DeepSeek V4 stores ``wo_a`` as ``[groups * out, in]`` while its input
        is ``[..., groups, in]``.  Treating that tensor as a dense linear would
        mix groups and silently implement a different architecture.
        """

        torch = self.torch
        if isinstance(groups, bool) or not isinstance(groups, int) or groups <= 0:
            raise ValueError("groups must be a positive integer")
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if x.ndim < 2 or x.shape[-2] != groups:
            raise ValueError(f"grouped input must end in [{groups}, in_features]")
        weight = self._materialize_weight(prefix)
        try:
            if weight.values.shape[0] % groups or weight.values.shape[1] != x.shape[-1]:
                raise DeepSeekPagerError(
                    f"grouped weight {weight.values.shape} is incompatible with "
                    f"input {tuple(x.shape)}"
                )
            should_quantize = bool(
                activation_quantization
            ) and weight.storage_dtype in {
                "F8_E4M3",
                "F8_E4M3FN",
                "I8",
            }
            original_dtype = x.dtype
            out_per_group = weight.values.shape[0] // groups
            if should_quantize:
                outputs = []
                for group in range(groups):
                    prepared = self._prepare_linear_input(
                        x[..., group, :],
                        quantized=True,
                        compute_dtype=self.compute_dtype,
                    )
                    outputs.append(
                        self._block_scaled_linear(
                            prepared,
                            weight,
                            row_start=group * out_per_group,
                            row_stop=(group + 1) * out_per_group,
                            result_dtype=output_dtype or original_dtype,
                            result_device=x.device,
                        )
                    )
                self._stats.linear_calls += 1
                return torch.stack(outputs, dim=-2)

            compute_x = x.to(self.device, dtype=self.compute_dtype)
            dequantized = self._dequantize_materialized(weight)
            compute_weight = torch.from_numpy(
                dequantized.reshape(groups, out_per_group, dequantized.shape[1])
            ).to(self.device, dtype=self.compute_dtype)
            try:
                result = torch.einsum("...gi,goi->...go", compute_x, compute_weight)
                result = result.to(dtype=output_dtype or original_dtype)
                if x.device != self.device:
                    result = result.to(x.device)
                self._stats.linear_calls += 1
                return result
            finally:
                del compute_weight
                del dequantized
        finally:
            del weight
            self._release_materialized_weight()

    def tensor_torch(
        self,
        name: str,
        *,
        dtype: Any | None = None,
        device: str | Any | None = None,
    ) -> Any:
        """Load one small control tensor (norm/router/HC), never a quantized matrix."""

        torch = self.torch
        meta = self.source.find(name)
        storage_dtype = str(meta["dtype"]).upper()
        if storage_dtype in {"F8_E4M3", "F8_E4M3FN", "F8_E8M0"}:
            raise DeepSeekPagerError(
                f"tensor_torch() refuses unscaled float8 tensor {name}"
            )
        with self._priority_scope(name, 1):
            array = self.source.tensor(name)
        result = torch.from_numpy(np.ascontiguousarray(array))
        target_device = self.device if device is None else torch.device(device)
        if dtype is not None or target_device.type != "cpu":
            result = result.to(device=target_device, dtype=dtype or result.dtype)
        result.requires_grad_(False)
        return result

    def embedding(self, token_ids: Iterable[int], *, name: str = "embed.weight") -> Any:
        """Read only requested embedding rows, coalescing consecutive token IDs."""

        torch = self.torch
        ids = [int(token_id) for token_id in token_ids]
        meta = self.source.find(name)
        if len(meta["shape"]) != 2:
            raise DeepSeekPagerError(f"embedding tensor {name} must be 2D")
        vocab = int(meta["shape"][0])
        if any(token_id < 0 or token_id >= vocab for token_id in ids):
            raise IndexError(f"embedding token outside [0, {vocab})")
        if not ids:
            return torch.empty((0, int(meta["shape"][1])), device=self.device)

        runs = self._consecutive_runs(ids)
        by_id: dict[int, np.ndarray] = {}
        with self._priority_scope(name, 1):
            for start, stop in runs:
                rows = self.source.rows(name, start_row=start, n_rows=stop - start)
                for offset, row in enumerate(rows):
                    by_id[start + offset] = row
        array = np.stack([by_id[token_id] for token_id in ids])
        self._stats.embedding_rows += len(ids)
        return torch.from_numpy(np.ascontiguousarray(array)).to(
            self.device, dtype=self.compute_dtype
        )

    def candidate_logits(
        self,
        hidden: Any,
        token_ids: Iterable[int],
        *,
        name: str = "head.weight",
    ) -> Any:
        """Score an explicit token set without materializing the 1 GiB LM head."""

        torch = self.torch
        ids = [int(token_id) for token_id in token_ids]
        if not ids:
            raise ValueError("candidate token IDs must not be empty")
        meta = self.source.find(name)
        if len(meta["shape"]) != 2:
            raise DeepSeekPagerError(f"head tensor {name} must be 2D")
        vocab = int(meta["shape"][0])
        if any(token_id < 0 or token_id >= vocab for token_id in ids):
            raise IndexError(f"candidate token outside [0, {vocab})")
        by_id: dict[int, np.ndarray] = {}
        with self._priority_scope(name, 1):
            for start, stop in self._consecutive_runs(ids):
                rows = self.source.rows(name, start_row=start, n_rows=stop - start)
                for offset, row in enumerate(rows):
                    by_id[start + offset] = row
        rows = [by_id[token_id] for token_id in ids]
        # ParallelHead stores checkpoint BF16 rows as FP32 parameters and
        # explicitly evaluates ``F.linear(x.float(), weight)``.
        weight = torch.from_numpy(np.ascontiguousarray(rows)).to(
            self.device, dtype=torch.float32
        )
        x = hidden.to(self.device, dtype=torch.float32)
        self._stats.head_rows += len(ids)
        return torch.nn.functional.linear(x, weight)

    def _stable_topk(self, values: Any, ids: Any, k: int) -> tuple[Any, Any]:
        """Lexicographic top-k: larger logit first, lower token ID on ties."""

        torch = self.torch
        if values.shape != ids.shape or values.ndim < 1:
            raise ValueError("stable top-k values/ids must have the same shape")
        if not 0 < k <= values.shape[-1]:
            raise ValueError("stable top-k k is outside the final dimension")
        id_order = torch.argsort(ids, dim=-1, stable=True)
        ordered_ids = torch.gather(ids, -1, id_order)
        ordered_values = torch.gather(values, -1, id_order)
        value_order = torch.argsort(
            ordered_values, dim=-1, descending=True, stable=True
        )[..., :k]
        return (
            torch.gather(ordered_values, -1, value_order),
            torch.gather(ordered_ids, -1, value_order),
        )

    def topk_logits(
        self,
        hidden: Any,
        *,
        k: int = 1,
        block_rows: int = 1024,
        transport_range_batch_blocks: int = 1,
        name: str = "head.weight",
        progress: Callable[[dict[str, int]], None] | None = None,
        instrument_block_observer: Callable[[int, Any], None] | None = None,
    ) -> tuple[Any, Any]:
        """Scan the vocabulary head in bounded row blocks and return global top-k.

        ``transport_range_batch_blocks`` is transport-only: values above one
        batch exact adjacent source leaves without changing ``block_rows``,
        decode order, compute order, or top-k reduction order. The observer is
        an instrumentation hook and receives a detached clone of each original
        compute block's logits after that block has been reduced.
        """

        torch = self.torch
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        if (
            isinstance(block_rows, bool)
            or not isinstance(block_rows, int)
            or block_rows <= 0
        ):
            raise ValueError("block_rows must be a positive integer")
        if (
            isinstance(transport_range_batch_blocks, bool)
            or not isinstance(transport_range_batch_blocks, int)
            or not self.HEAD_TRANSPORT_DEFAULT_RANGE_BATCH_BLOCKS
            <= transport_range_batch_blocks
            <= self.HEAD_TRANSPORT_MAX_RANGE_BATCH_BLOCKS
        ):
            raise ValueError(
                "transport_range_batch_blocks must be an integer in [1, 8]"
            )
        if instrument_block_observer is not None and not callable(
            instrument_block_observer
        ):
            raise ValueError("instrument_block_observer must be callable or None")
        meta = self.source.find(name)
        shape = tuple(int(value) for value in meta["shape"])
        if len(shape) != 2:
            raise DeepSeekPagerError(f"head tensor {name} must be 2D")
        vocab = shape[0]
        if k > vocab:
            raise ValueError("k exceeds vocabulary size")
        # Match the published ParallelHead, which promotes both activation and
        # BF16 checkpoint rows to FP32 before computing logits.
        x = hidden.to(self.device, dtype=torch.float32)
        best_values = None
        best_ids = None

        def consume_block(start: int, count: int, rows: np.ndarray) -> None:
            nonlocal best_values, best_ids
            weight = torch.from_numpy(np.ascontiguousarray(rows)).to(
                self.device, dtype=torch.float32
            )
            try:
                logits = torch.nn.functional.linear(x, weight)
                local_k = min(k, count)
                token_ids = torch.arange(
                    start, start + count, dtype=torch.long, device=logits.device
                ).expand_as(logits)
                values, indices = self._stable_topk(logits, token_ids, local_k)
                if best_values is None:
                    best_values, best_ids = values, indices
                else:
                    joined_values = torch.cat((best_values, values), dim=-1)
                    joined_ids = torch.cat((best_ids, indices), dim=-1)
                    best_values, best_ids = self._stable_topk(
                        joined_values, joined_ids, k
                    )
                self._stats.head_rows += count
                if instrument_block_observer is not None:
                    instrument_block_observer(start, logits.detach().clone())
                if progress is not None:
                    progress(
                        {
                            "start_row": start,
                            "rows": count,
                            "rows_done": start + count,
                            "vocab_rows": vocab,
                        }
                    )
            finally:
                del weight

        logical_leaves = tuple(
            (start, min(block_rows, vocab - start))
            for start in range(0, vocab, block_rows)
        )
        self._stats.head_logical_leaves += len(logical_leaves)
        raw_bytes_many = getattr(self.source, "raw_bytes_many", None)
        head_layout = None
        if transport_range_batch_blocks > 1 and callable(raw_bytes_many):
            head_layout = self._head_row_layout(meta, block_rows=block_rows)

        if transport_range_batch_blocks == 1:
            for start, count in logical_leaves:
                with self._priority_scope(name, 1):
                    rows = self.source.rows(name, start_row=start, n_rows=count)
                consume_block(start, count, rows)
        elif head_layout is None:
            self._stats.head_transport_fallbacks += 1
            self._stats.head_transport_fallback_leaves += len(logical_leaves)
            for start, count in logical_leaves:
                with self._priority_scope(name, 1):
                    rows = self.source.rows(name, start_row=start, n_rows=count)
                consume_block(start, count, rows)
        else:
            shard, dtype, columns, leaves = head_layout
            index = 0
            while index < len(leaves):
                leaf = leaves[index]
                if leaf.length > self.HEAD_TRANSPORT_RESIDENT_LIMIT_BYTES:
                    self._stats.head_transport_fallbacks += 1
                    self._stats.head_transport_fallback_leaves += 1
                    with self._priority_scope(name, 1):
                        rows = self.source.rows(
                            name,
                            start_row=leaf.start_row,
                            n_rows=leaf.count,
                        )
                    consume_block(leaf.start_row, leaf.count, rows)
                    index += 1
                    continue

                stop = index
                resident_bytes = 0
                while (
                    stop < len(leaves) and stop - index < transport_range_batch_blocks
                ):
                    candidate_bytes = resident_bytes + leaves[stop].length
                    if candidate_bytes > self.HEAD_TRANSPORT_RESIDENT_LIMIT_BYTES:
                        break
                    resident_bytes = candidate_bytes
                    stop += 1
                batch = leaves[index:stop]
                with self._priority_scope(name, 1):
                    tensors = self._read_head_leaf_batch(
                        name=name,
                        shard=shard,
                        dtype=dtype,
                        columns=columns,
                        leaves=batch,
                    )
                for original, tensor in zip(batch, tensors, strict=True):
                    rows = self._decode_coalesced_tensor(tensor)
                    consume_block(original.start_row, original.count, rows)
                index = stop
        assert best_values is not None and best_ids is not None
        return best_values, best_ids

    def release(self) -> None:
        """Drop allocator caches at an explicit decoder-layer boundary."""

        with self._prefetch_lock:
            if self._active_prefetch_payload is not None:
                raise DeepSeekPagerError(
                    "cannot release pager with an active expert prefetch payload"
                )
            if self._active_prefetch_window is not None:
                raise DeepSeekPagerError(
                    "cannot release pager with a live expert prefetch window"
                )
            executor = self._prefetch_executor
            self._prefetch_executor = None
            # Failed reads deliberately drain in the background so one bad
            # speculative fetch does not stall the current layer.  Reap its
            # executor here only when every future has already retired.
            draining_executor = (
                None if self._draining_prefetch else self._draining_executor
            )
            if draining_executor is not None:
                self._draining_executor = None
        executors = {owned for owned in (executor, draining_executor) if owned}
        for owned in executors:
            owned.shutdown(wait=True, cancel_futures=True)
        self._reap_retired_reservoirs(wait=False)
        gc.collect()
        self._stats.release_boundaries += 1
        self._purge_mps_cache()

    def close(self) -> None:
        """Terminally drain prefetch I/O before closing the tensor source.

        Unlike :meth:`release`, this method may wait for a cancelled HTTP read.
        Every future and its executor are joined outside ``_prefetch_lock`` so
        source teardown cannot race a worker or deadlock its completion callback.
        """

        with self._prefetch_lock:
            window = self._active_prefetch_window
        if window is not None:
            self._cancel_expert_window(window)
        with self._prefetch_lock:
            reservoir = self._active_expert_reservoir
        if reservoir is not None:
            self.cancel_expert_reservoir(reservoir)

        while True:
            with self._prefetch_lock:
                futures = tuple(self._draining_prefetch)
                executor = self._draining_executor
                if not futures and executor is None:
                    break
            for future in futures:
                try:
                    future.result()
                except BaseException:
                    pass
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            with self._prefetch_lock:
                for future in futures:
                    if future.done():
                        self._draining_prefetch.discard(future)
                if self._draining_executor is executor:
                    self._draining_executor = None

        self._reap_retired_reservoirs(wait=True)
        with self._prefetch_lock:
            reservoir_executor = self._expert_reservoir_executor
            self._expert_reservoir_executor = None
        if reservoir_executor is not None:
            reservoir_executor.shutdown(wait=True, cancel_futures=True)
        self.release()

    def metrics(self) -> dict[str, Any]:
        source_metrics = self.source.metrics()
        with self._prefetch_lock:
            draining = bool(self._draining_prefetch)
            reservoir_active = self._active_expert_reservoir is not None
            reservoir_retired = len(self._retired_expert_reservoirs)
        return {
            **asdict(self._stats),
            "device": str(self.device),
            "compute_dtype": str(self.compute_dtype).removeprefix("torch."),
            "quantized_accumulation_policy": self.QUANTIZED_ACCUMULATION_POLICY,
            "head_transport_policy": self.HEAD_TRANSPORT_POLICY,
            "head_transport_default_range_batch_blocks": (
                self.HEAD_TRANSPORT_DEFAULT_RANGE_BATCH_BLOCKS
            ),
            "head_transport_max_range_batch_blocks": (
                self.HEAD_TRANSPORT_MAX_RANGE_BATCH_BLOCKS
            ),
            "head_transport_resident_limit_bytes": (
                self.HEAD_TRANSPORT_RESIDENT_LIMIT_BYTES
            ),
            "head_transport_max_gap_bytes": self.HEAD_TRANSPORT_MAX_GAP_BYTES,
            "expert_prefetch_policy": self.expert_prefetch_policy,
            "expert_prefetch_transport_policy": (self.expert_prefetch_transport_policy),
            "expert_prefetch_payload_limit_bytes": (
                self.EXPERT_PREFETCH_PAYLOAD_LIMIT_BYTES
            ),
            "expert_prefetch_workers": self.EXPERT_PREFETCH_WORKERS,
            "expert_prefetch_active_read_limit": (
                self.EXPERT_PREFETCH_ACTIVE_READ_LIMIT
            ),
            "expert_prefetch_max_outstanding_limit": (
                self.EXPERT_PREFETCH_MAX_OUTSTANDING
            ),
            "expert_prefetch_max_experts": self.EXPERT_PREFETCH_MAX_EXPERTS,
            "expert_prefetch_resident_limit_bytes": (
                self.EXPERT_PREFETCH_RESIDENT_LIMIT_BYTES
            ),
            "expert_range_coalesce_max_experts": (
                self.expert_range_coalesce_max_experts
            ),
            "expert_range_coalesce_max_gap_bytes": (
                self.EXPERT_RANGE_COALESCE_MAX_GAP_BYTES
            ),
            "expert_prefetch_draining": draining,
            "expert_reservoir_policy": self.EXPERT_RESERVOIR_POLICY,
            "expert_reservoir_budget_bytes": self.expert_reservoir_budget_bytes,
            "expert_reservoir_workers": self.expert_reservoir_workers,
            "expert_reservoir_active": reservoir_active,
            "expert_reservoir_retired": reservoir_retired,
            "causal_weight_reader_attached": self._causal_weight_reader is not None,
            "causal_missing_fallback": self._causal_missing_fallback,
            "source": source_metrics,
        }


__all__ = [
    "CausalExpertPlanResolver",
    "DeepSeekPagerError",
    "DeepSeekWeightPager",
    "ExpertSourceRange",
    "ExpertTensorLayout",
    "OfficialExpertRangePlan",
    "PagerMetrics",
]
