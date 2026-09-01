"""Residual-certified product-quantized rail for exact Qwen LM-head search."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import platform
import stat
import tempfile
import threading
from typing import Any, Mapping, cast

import numpy as np
import torch
from safetensors.torch import load, save_file

from ..ooe.identity import canonical_json_bytes, require_sha256


EXACT_HEAD_SCHEMA = "immer.qwen3.8-exact-head-pq/v1"
EXACT_HEAD_MANIFEST_SCHEMA = "immer.qwen3.8-exact-head-manifest/v1"
EXACT_HEAD_SCORE_ABI = "cpu-bf16-explicit-fp32-accumulate-rne/v1"
Q4_EXACT_HEAD_SCORE_ABI = "cpu-q8_0xq8_0-native-f32-topk-bf16-rne/v1"
_ROW_BOUND_PROBE_PAGES = 64
_MANIFEST_NAME = "manifest.json"
_PAYLOAD_NAME = "index.safetensors"
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_PAYLOAD_KEYS = frozenset(
    {
        "codebooks",
        "codes",
        "residual_radii",
        "row_norms",
        "node_presence",
        "node_max_residual",
        "node_max_norm",
        "node_min_token",
        "node_start_page",
        "node_page_count",
        "node_child_start",
        "node_child_count",
    }
)


class ExactHeadError(RuntimeError):
    """The exact-head artifact or its runtime certificate is invalid."""


class ExactHeadNotApplicable(ExactHeadError):
    """A valid index cannot accelerate this scorer/request ABI."""


def _backend_sha256() -> str:
    cpu_capability = getattr(getattr(torch.backends, "cpu", None), "get_cpu_capability", None)
    identity = {
        "cpu_capability": (
            cpu_capability() if callable(cpu_capability) else "unknown"
        ),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "machine": platform.machine(),
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        "score_abi": EXACT_HEAD_SCORE_ABI,
        "torch_version": str(torch.__version__),
    }
    return _sha256_bytes(canonical_json_bytes(identity))


def _q4_backend_sha256(q4_bank: Any) -> str:
    identity = getattr(q4_bank, "identity", None)
    if not isinstance(identity, Mapping):
        raise ExactHeadError("Q4 exact-head backend identity is unavailable")
    return _sha256_bytes(
        canonical_json_bytes(
            {
                "host_backend_sha256": _backend_sha256(),
                "q4_bank": dict(identity),
                "score_abi": Q4_EXACT_HEAD_SCORE_ABI,
            }
        )
    )


@dataclass(frozen=True, slots=True)
class ExactHeadConfig:
    subspace_width: int = 32
    codebook_size: int = 256
    page_rows: int = 64
    fanout: int = 16
    kmeans_iterations: int = 4
    assignment_chunk_rows: int = 4096
    max_query_rows: int = 16

    def __post_init__(self) -> None:
        for field in (
            "subspace_width",
            "codebook_size",
            "page_rows",
            "fanout",
            "kmeans_iterations",
            "assignment_chunk_rows",
            "max_query_rows",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.codebook_size > 256:
            raise ValueError("codebook_size must fit uint8")
        if self.fanout < 2:
            raise ValueError("fanout must be at least two")
        if self.max_query_rows > 16:
            raise ValueError("max_query_rows must not exceed the rolling window")

    def to_record(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_record(cls, value: object) -> "ExactHeadConfig":
        expected = {field.name for field in fields(cls)}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ExactHeadError("exact-head config record is invalid")
        try:
            return cls(**{key: value[key] for key in expected})
        except (TypeError, ValueError) as exc:
            raise ExactHeadError("exact-head config values are invalid") from exc


@dataclass(frozen=True, slots=True)
class ExactHeadBinding:
    repo_id: str
    revision: str
    inventory_fingerprint: str
    tensor_name: str
    tensor_sha256: str
    vocab_size: int
    hidden_size: int
    tensor_dtype: str = "BF16"
    score_abi: str = EXACT_HEAD_SCORE_ABI
    torch_version: str = torch.__version__
    backend_sha256: str = ""

    def __post_init__(self) -> None:
        for field in ("repo_id", "revision", "tensor_name", "tensor_dtype", "score_abi", "torch_version"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must be non-empty")
        object.__setattr__(
            self,
            "inventory_fingerprint",
            require_sha256(self.inventory_fingerprint, field="inventory_fingerprint"),
        )
        object.__setattr__(
            self,
            "tensor_sha256",
            require_sha256(self.tensor_sha256, field="tensor_sha256"),
        )
        backend = self.backend_sha256 or _backend_sha256()
        object.__setattr__(
            self,
            "backend_sha256",
            require_sha256(backend, field="backend_sha256"),
        )
        if (self.tensor_dtype, self.score_abi) not in {
            ("BF16", EXACT_HEAD_SCORE_ABI),
            ("Q8_0", Q4_EXACT_HEAD_SCORE_ABI),
        }:
            raise ValueError("exact-head binding scorer ABI is unsupported")
        for field in ("vocab_size", "hidden_size"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{field} must be at least two")

    def to_record(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_record(cls, value: object) -> "ExactHeadBinding":
        expected = {field.name for field in fields(cls)}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ExactHeadError("exact-head binding record is invalid")
        try:
            return cls(**{key: value[key] for key in expected})
        except (TypeError, ValueError) as exc:
            raise ExactHeadError("exact-head binding values are invalid") from exc


@dataclass(frozen=True, slots=True)
class ExactHeadReceipt:
    manifest_sha256: str
    payload_sha256: str
    payload_bytes: int
    tensor_sha256: str
    index_bytes: int

    def to_record(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ExactHeadMetrics:
    calls: int = 0
    applicable_calls: int = 0
    fallback_calls: int = 0
    pages_scored: int = 0
    pages_pruned: int = 0
    rows_scored: int = 0
    rows_pruned: int = 0
    bound_nodes: int = 0
    row_bound_rows: int = 0
    selected_row_reads: int = 0
    selected_rows_scored: int = 0
    full_leaf_fallbacks: int = 0
    selected_row_cost_fallbacks: int = 0
    row_bound_probe_pages: int = 0
    row_bound_disabled_calls: int = 0
    selected_row_logical_bytes: int = 0
    row_certificate_logical_bytes_avoided: int = 0
    logical_head_bytes_avoided: int = 0
    packed_rows_scored: int = 0
    packed_rows_avoided: int = 0
    packed_weight_bytes_avoided: int = 0
    last_fallback_reason: str = ""


@dataclass(frozen=True, slots=True)
class _QueryBoundTables:
    scores: np.ndarray
    absolute: np.ndarray
    hnorm: np.ndarray


def _sha256_bytes(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def _tensor_raw_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _bf16_has_subnormal(tensor: torch.Tensor) -> bool:
    if tensor.dtype != torch.bfloat16:
        raise TypeError("subnormal inspection requires BF16 storage")
    bits = tensor.detach().contiguous().view(torch.uint16).cpu().numpy()
    exponent = bits & np.uint16(0x7F80)
    mantissa = bits & np.uint16(0x007F)
    return bool(((exponent == 0) & (mantissa != 0)).any())


def _ceil_float32(values: np.ndarray) -> np.ndarray:
    """Smallest representable FP32 value not below a float64 upper bound."""

    upper = np.asarray(values, dtype=np.float64)
    with np.errstate(over="ignore"):
        result = np.asarray(upper, dtype=np.float32)
    below = np.asarray(result, dtype=np.float64) < upper
    result[below] = np.nextafter(
        result[below], np.float32(np.inf), dtype=np.float32
    )
    return result


def _norm_upper(values: np.ndarray) -> np.ndarray:
    """Outward float64 L2 reduction followed by outward FP32 storage."""

    data = np.asarray(values, dtype=np.float64)
    width = data.shape[-1]
    u64 = 2.0**-53
    gamma = (width + 2) * u64 / (1.0 - (width + 2) * u64)
    raw = np.sum(data * data, axis=-1, dtype=np.float64)
    squared = np.nextafter(
        raw * (1.0 + u64) / (1.0 - gamma) + width * 2.0**-1074,
        np.inf,
    )
    return _ceil_float32(np.nextafter(np.sqrt(squared), np.inf))


def _strict_json(data: bytes) -> Mapping[str, object]:
    def pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in rows:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ExactHeadError("exact-head manifest is invalid JSON") from exc
    if not isinstance(value, Mapping) or canonical_json_bytes(value) != data:
        raise ExactHeadError("exact-head manifest is not canonical JSON")
    return value


def _stable_read(path: Path, maximum: int) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | int(getattr(os, "O_CLOEXEC", 0))
            | int(getattr(os, "O_NOFOLLOW", 0)),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise ExactHeadError(f"invalid bounded exact-head file: {path}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(4 * 1024 * 1024, remaining))
            if not chunk:
                raise ExactHeadError(f"short exact-head file: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ExactHeadError(f"exact-head file changed while read: {path}")
        return b"".join(chunks)
    except ExactHeadError:
        raise
    except OSError as exc:
        raise ExactHeadError(f"cannot read exact-head file: {path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _stable_read_at(directory_fd: int, name: str, maximum: int) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY
            | int(getattr(os, "O_CLOEXEC", 0))
            | int(getattr(os, "O_NOFOLLOW", 0)),
            dir_fd=directory_fd,
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise ExactHeadError(f"invalid bounded exact-head file: {name}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(4 * 1024 * 1024, remaining))
            if not chunk:
                raise ExactHeadError(f"short exact-head file: {name}")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ExactHeadError(f"exact-head file changed while read: {name}")
        return b"".join(chunks)
    except ExactHeadError:
        raise
    except OSError as exc:
        raise ExactHeadError(f"cannot read exact-head file: {name}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


class ExactHeadIndex:
    """Immutable PQ certificate plus exact canonical-page scorer."""

    def __init__(
        self,
        *,
        config: ExactHeadConfig,
        binding: ExactHeadBinding,
        tensors: Mapping[str, torch.Tensor],
        manifest_sha256: str = "0" * 64,
        payload_sha256: str = "0" * 64,
        payload_bytes: int = 0,
    ) -> None:
        self.config = config
        self.binding = binding
        self._tensors = {key: value.detach().cpu().contiguous() for key, value in tensors.items()}
        self._validate_tensors()
        self.leaf_count = math.ceil(binding.vocab_size / config.page_rows)
        self.root_index = len(self._tensors["node_min_token"]) - 1
        self.receipt = ExactHeadReceipt(
            manifest_sha256=require_sha256(manifest_sha256, field="manifest_sha256"),
            payload_sha256=require_sha256(payload_sha256, field="payload_sha256"),
            payload_bytes=int(payload_bytes),
            tensor_sha256=binding.tensor_sha256,
            index_bytes=sum(value.numel() * value.element_size() for value in self._tensors.values()),
        )
        self._metrics = ExactHeadMetrics()
        self._metrics_lock = threading.Lock()
        self._runtime_lock = threading.RLock()
        self._closed = False
        self._q4_zero_prune_disabled = False

    @property
    def supports_q4(self) -> bool:
        return self.binding.score_abi == Q4_EXACT_HEAD_SCORE_ABI

    @classmethod
    def build(
        cls,
        head: torch.Tensor,
        *,
        binding: ExactHeadBinding,
        config: ExactHeadConfig = ExactHeadConfig(),
    ) -> "ExactHeadIndex":
        if (
            not isinstance(head, torch.Tensor)
            or head.device.type != "cpu"
            or head.dtype != torch.bfloat16
            or head.ndim != 2
            or tuple(head.shape) != (binding.vocab_size, binding.hidden_size)
            or not bool(torch.isfinite(head).all())
        ):
            raise ValueError("head must be the finite bound CPU BF16 matrix")
        if _tensor_raw_sha256(head) != binding.tensor_sha256:
            raise ValueError("head bytes differ from the binding")
        if _bf16_has_subnormal(head):
            raise ExactHeadError("exact-head scorer forbids BF16 subnormal weights")
        source = head.float().numpy()
        codebooks = cls._fit_codebooks(source, config=config)
        codes, residual, norms = cls._encode_source(
            source,
            codebooks=codebooks,
            config=config,
        )
        tensors = cls._tree_tensors(
            codebooks=codebooks,
            codes=codes,
            residual=residual,
            norms=norms,
            config=config,
        )
        tensors.update(
            {
                "codebooks": torch.from_numpy(codebooks),
                "codes": torch.from_numpy(codes),
                "residual_radii": torch.from_numpy(residual),
                "row_norms": torch.from_numpy(norms),
            }
        )
        return cls(config=config, binding=binding, tensors=tensors)

    @staticmethod
    def _fit_codebooks(
        source: np.ndarray,
        *,
        config: ExactHeadConfig,
    ) -> np.ndarray:
        if source.ndim != 2 or not np.isfinite(source).all():
            raise ValueError("PQ source must be a finite matrix")
        vocab, width = source.shape
        k = min(config.codebook_size, vocab)
        subspaces = math.ceil(width / config.subspace_width)
        codebooks = np.zeros(
            (subspaces, k, config.subspace_width), dtype=np.float32
        )
        for subspace in range(subspaces):
            begin = subspace * config.subspace_width
            end = min(width, begin + config.subspace_width)
            data = np.asarray(source[:, begin:end], dtype=np.float32)
            initial = (np.arange(k, dtype=np.int64) * vocab) // k
            centroids = data[initial].copy()
            assignments = np.zeros(vocab, dtype=np.int64)
            for _iteration in range(config.kmeans_iterations):
                for start in range(0, vocab, config.assignment_chunk_rows):
                    stop = min(vocab, start + config.assignment_chunk_rows)
                    chunk = data[start:stop]
                    chunk64 = np.asarray(chunk, dtype=np.float64)
                    centroids64 = np.asarray(centroids, dtype=np.float64)
                    distances = (
                        np.sum(chunk64 * chunk64, axis=1, dtype=np.float64)[:, None]
                        + np.sum(
                            centroids64 * centroids64,
                            axis=1,
                            dtype=np.float64,
                        )[None, :]
                        - 2.0 * chunk64 @ centroids64.T
                    )
                    assignments[start:stop] = np.argmin(distances, axis=1)
                sums = np.zeros_like(centroids, dtype=np.float64)
                counts = np.bincount(assignments, minlength=k)
                np.add.at(sums, assignments, np.asarray(data, dtype=np.float64))
                occupied = counts > 0
                next_centroids = np.asarray(centroids, dtype=np.float64)
                next_centroids[occupied] = (
                    sums[occupied] / counts[occupied, None]
                )
                next_centroids[~occupied] = centroids[~occupied]
                centroids = np.asarray(next_centroids, dtype=np.float32)
            codebooks[subspace, :, : end - begin] = centroids
        return codebooks

    @staticmethod
    def _encode_source(
        source: np.ndarray,
        *,
        codebooks: np.ndarray,
        config: ExactHeadConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vocab, width = source.shape
        subspaces, k, _padded = codebooks.shape
        codes = np.zeros((vocab, subspaces), dtype=np.uint8)
        residual_squared = np.zeros(vocab, dtype=np.float64)
        residual_absolute = np.zeros(vocab, dtype=np.float64)
        for subspace in range(subspaces):
            begin = subspace * config.subspace_width
            end = min(width, begin + config.subspace_width)
            data = np.asarray(source[:, begin:end], dtype=np.float32)
            centroids = codebooks[subspace, :, : end - begin]
            for start in range(0, vocab, config.assignment_chunk_rows):
                stop = min(vocab, start + config.assignment_chunk_rows)
                chunk = data[start:stop]
                chunk64 = np.asarray(chunk, dtype=np.float64)
                centroids64 = np.asarray(centroids, dtype=np.float64)
                distances = (
                    np.sum(chunk64 * chunk64, axis=1, dtype=np.float64)[:, None]
                    + np.sum(
                        centroids64 * centroids64,
                        axis=1,
                        dtype=np.float64,
                    )[
                        None, :
                    ]
                    - 2.0
                    * chunk64
                    @ centroids64.T
                )
                codes[start:stop, subspace] = np.argmin(
                    distances, axis=1
                ).astype(np.uint8)
            reconstructed = codebooks[subspace, codes[:, subspace], : end - begin]
            delta = np.asarray(source[:, begin:end], dtype=np.float64) - np.asarray(
                reconstructed, dtype=np.float64
            )
            local_squared = np.sum(delta * delta, axis=1, dtype=np.float64)
            local_width = end - begin
            local_gamma = (local_width + 2) * 2.0**-53 / (
                1.0 - (local_width + 2) * 2.0**-53
            )
            local_upper = np.nextafter(
                local_squared
                * (1.0 + 2.0**-53)
                / (1.0 - local_gamma)
                + local_width * 2.0**-1074,
                np.inf,
            )
            residual_squared += local_upper
            residual_absolute += np.abs(local_upper)
        u64 = 2.0**-53
        gamma = (subspaces + 1) * u64 / (1.0 - (subspaces + 1) * u64)
        residual_squared = np.nextafter(
            residual_squared
            + gamma * residual_absolute
            + width * 2.0**-1074,
            np.inf,
        )
        residual = _ceil_float32(np.nextafter(np.sqrt(residual_squared), np.inf))
        norms = _norm_upper(source)
        return codes, residual, norms

    @staticmethod
    def _fixed_encoding_stats(
        source: np.ndarray,
        *,
        codebooks: np.ndarray,
        codes: np.ndarray,
        config: ExactHeadConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Recompute certified residuals after the physical head is quantized."""

        if source.ndim != 2 or not np.isfinite(source).all():
            raise ValueError("fixed PQ source must be a finite matrix")
        vocab, width = source.shape
        if codes.shape != (vocab, codebooks.shape[0]):
            raise ValueError("fixed PQ codes do not match the source rows")
        residual_squared = np.zeros(vocab, dtype=np.float64)
        residual_absolute = np.zeros(vocab, dtype=np.float64)
        for subspace in range(codebooks.shape[0]):
            begin = subspace * config.subspace_width
            end = min(width, begin + config.subspace_width)
            reconstructed = codebooks[
                subspace,
                codes[:, subspace],
                : end - begin,
            ]
            delta = np.asarray(source[:, begin:end], dtype=np.float64) - np.asarray(
                reconstructed,
                dtype=np.float64,
            )
            local_squared = np.sum(delta * delta, axis=1, dtype=np.float64)
            local_width = end - begin
            local_gamma = (local_width + 2) * 2.0**-53 / (
                1.0 - (local_width + 2) * 2.0**-53
            )
            local_upper = np.nextafter(
                local_squared
                * (1.0 + 2.0**-53)
                / (1.0 - local_gamma)
                + local_width * 2.0**-1074,
                np.inf,
            )
            residual_squared += local_upper
            residual_absolute += np.abs(local_upper)
        u64 = 2.0**-53
        gamma = (codebooks.shape[0] + 1) * u64 / (
            1.0 - (codebooks.shape[0] + 1) * u64
        )
        residual_squared = np.nextafter(
            residual_squared
            + gamma * residual_absolute
            + width * 2.0**-1074,
            np.inf,
        )
        residual = _ceil_float32(np.nextafter(np.sqrt(residual_squared), np.inf))
        return residual, _norm_upper(source)

    @classmethod
    def build_from_pager(
        cls,
        pager: Any,
        *,
        name: str = "lm_head.weight",
        config: ExactHeadConfig = ExactHeadConfig(),
        sample_rows: int = 4096,
    ) -> "ExactHeadIndex":
        """Build from local head ranges only; no model forward is executed."""

        if (
            isinstance(sample_rows, bool)
            or not isinstance(sample_rows, int)
            or sample_rows < 1
        ):
            raise ValueError("sample_rows must be a positive integer")
        layout = pager._layout(name)
        if layout.dtype != "BF16" or len(layout.shape) != 2:
            raise ExactHeadError("exact-head builder requires a BF16 matrix")
        vocab, width = layout.shape
        source_before = pager.source.metrics()
        contract_reader = getattr(pager.source, "reader", None)
        upstream = getattr(contract_reader, "upstream", None)
        source_identity_method = getattr(upstream, "source_identity", None)
        shard_before = (
            dict(source_identity_method(layout.shard))
            if callable(source_identity_method)
            else None
        )
        sample_count = min(vocab, max(config.codebook_size, sample_rows))
        sample_ids = tuple(
            sorted(
                {
                    min(vocab - 1, (index * vocab) // sample_count)
                    for index in range(sample_count)
                }
            )
        )
        sample = pager.tensor_rows(name, sample_ids).float().cpu().numpy()
        codebooks = cls._fit_codebooks(sample, config=config)
        subspaces = codebooks.shape[0]
        codes = np.zeros((vocab, subspaces), dtype=np.uint8)
        residual = np.zeros(vocab, dtype=np.float32)
        norms = np.zeros(vocab, dtype=np.float32)
        digest = hashlib.sha256()
        for start in range(0, vocab, config.assignment_chunk_rows):
            count = min(config.assignment_chunk_rows, vocab - start)
            rows = pager._read_rows(
                name,
                start,
                count,
                dtype=torch.bfloat16,
                device=torch.device("cpu"),
            )
            if _bf16_has_subnormal(rows):
                raise ExactHeadError(
                    "exact-head scorer forbids BF16 subnormal weights"
                )
            digest.update(
                rows.detach().contiguous().view(torch.uint8).numpy().tobytes()
            )
            encoded, radii, row_norms = cls._encode_source(
                rows.float().numpy(),
                codebooks=codebooks,
                config=config,
            )
            codes[start : start + count] = encoded
            residual[start : start + count] = radii
            norms[start : start + count] = row_norms
            del rows
        source_metrics = pager.source.metrics()
        for key in ("repo_id", "revision", "inventory_source_fingerprint"):
            if source_metrics.get(key) != source_before.get(key):
                raise ExactHeadError("head source identity changed during build")
        if shard_before is not None and dict(source_identity_method(layout.shard)) != shard_before:
            raise ExactHeadError("head shard identity changed during build")
        try:
            binding = ExactHeadBinding(
                repo_id=source_metrics["repo_id"],
                revision=source_metrics["revision"],
                inventory_fingerprint=source_metrics[
                    "inventory_source_fingerprint"
                ],
                tensor_name=name,
                tensor_sha256=digest.hexdigest(),
                vocab_size=vocab,
                hidden_size=width,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ExactHeadError(
                "head source lacks an immutable build identity"
            ) from exc
        tensors = cls._tree_tensors(
            codebooks=codebooks,
            codes=codes,
            residual=residual,
            norms=norms,
            config=config,
        )
        tensors.update(
            {
                "codebooks": torch.from_numpy(codebooks),
                "codes": torch.from_numpy(codes),
                "residual_radii": torch.from_numpy(residual),
                "row_norms": torch.from_numpy(norms),
            }
        )
        return cls(config=config, binding=binding, tensors=tensors)

    @classmethod
    def rebind_q4_head(
        cls,
        source: "ExactHeadIndex",
        pager: Any,
        *,
        name: str = "lm_head.weight",
    ) -> "ExactHeadIndex":
        """Re-certify an existing PQ partition against the packed Q8 head."""

        if not isinstance(source, ExactHeadIndex) or source._closed:
            raise TypeError("source must be an open ExactHeadIndex")
        if (
            source.binding.score_abi != EXACT_HEAD_SCORE_ABI
            or source.binding.tensor_name != name
        ):
            raise ExactHeadError("Q4 rebinding requires a BF16 source index")
        q4_bank = getattr(pager, "q4_bank", None)
        if q4_bank is None or not bool(getattr(q4_bank, "has", lambda _name: False)(name)):
            raise ExactHeadError("Q4 rebinding requires the packed output head")
        try:
            entry = q4_bank.entries[name]
        except (AttributeError, KeyError) as exc:
            raise ExactHeadError("Q4 output-head entry is unavailable") from exc
        if (
            entry.format != "q8_0"
            or tuple(entry.shape)
            != (source.binding.vocab_size, source.binding.hidden_size)
        ):
            raise ExactHeadError("Q4 exact-head requires the bound Q8_0 matrix")
        metrics = pager.source.metrics()
        expected_source = {
            "repo_id": source.binding.repo_id,
            "revision": source.binding.revision,
            "inventory_source_fingerprint": (
                source.binding.inventory_fingerprint
            ),
        }
        if any(metrics.get(field) != value for field, value in expected_source.items()):
            raise ExactHeadError("Q4 target source differs from the BF16 index")

        config = source.config
        codebooks = np.asarray(
            source._tensors["codebooks"].numpy(),
            dtype=np.float32,
        ).copy()
        codes = np.asarray(source._tensors["codes"].numpy(), dtype=np.uint8).copy()
        residual = np.empty(source.binding.vocab_size, dtype=np.float32)
        norms = np.empty(source.binding.vocab_size, dtype=np.float32)
        for start in range(0, source.binding.vocab_size, config.assignment_chunk_rows):
            count = min(
                config.assignment_chunk_rows,
                source.binding.vocab_size - start,
            )
            row_ids = tuple(range(start, start + count))
            rows = q4_bank.rows(name, row_ids, dtype=torch.float32)
            try:
                radii, row_norms = cls._fixed_encoding_stats(
                    rows.numpy(),
                    codebooks=codebooks,
                    codes=codes[start : start + count],
                    config=config,
                )
                residual[start : start + count] = radii
                norms[start : start + count] = row_norms
            finally:
                del rows
                discard = getattr(q4_bank, "discard_rows", None)
                if callable(discard):
                    discard(name, start, count)
        binding = ExactHeadBinding(
            repo_id=source.binding.repo_id,
            revision=source.binding.revision,
            inventory_fingerprint=source.binding.inventory_fingerprint,
            tensor_name=name,
            tensor_sha256=entry.payload_sha256,
            vocab_size=source.binding.vocab_size,
            hidden_size=source.binding.hidden_size,
            tensor_dtype="Q8_0",
            score_abi=Q4_EXACT_HEAD_SCORE_ABI,
            torch_version=torch.__version__,
            backend_sha256=_q4_backend_sha256(q4_bank),
        )
        tensors = cls._tree_tensors(
            codebooks=codebooks,
            codes=codes,
            residual=residual,
            norms=norms,
            config=config,
        )
        tensors.update(
            {
                "codebooks": torch.from_numpy(codebooks),
                "codes": torch.from_numpy(codes),
                "residual_radii": torch.from_numpy(residual),
                "row_norms": torch.from_numpy(norms),
            }
        )
        return cls(config=config, binding=binding, tensors=tensors)

    @staticmethod
    def _tree_tensors(
        *,
        codebooks: np.ndarray,
        codes: np.ndarray,
        residual: np.ndarray,
        norms: np.ndarray,
        config: ExactHeadConfig,
    ) -> dict[str, torch.Tensor]:
        vocab, subspaces = codes.shape
        k = codebooks.shape[1]
        packed_bytes = math.ceil(k / 8)
        presence_rows: list[np.ndarray] = []
        max_residual: list[float] = []
        max_norm: list[float] = []
        min_token: list[int] = []
        start_page: list[int] = []
        page_count: list[int] = []
        child_start: list[int] = []
        child_count: list[int] = []
        leaves: list[int] = []
        pages = math.ceil(vocab / config.page_rows)
        for page in range(pages):
            begin = page * config.page_rows
            end = min(vocab, begin + config.page_rows)
            mask = np.zeros((subspaces, packed_bytes), dtype=np.uint8)
            for subspace in range(subspaces):
                present = np.zeros(k, dtype=np.uint8)
                present[np.unique(codes[begin:end, subspace])] = 1
                mask[subspace] = np.packbits(present, bitorder="little")
            leaves.append(len(presence_rows))
            presence_rows.append(mask)
            max_residual.append(float(np.max(residual[begin:end])))
            max_norm.append(float(np.max(norms[begin:end])))
            min_token.append(begin)
            start_page.append(page)
            page_count.append(1)
            child_start.append(-1)
            child_count.append(0)
        level = leaves
        while len(level) > 1:
            following: list[int] = []
            for offset in range(0, len(level), config.fanout):
                children = level[offset : offset + config.fanout]
                index = len(presence_rows)
                following.append(index)
                presence_rows.append(
                    np.bitwise_or.reduce([presence_rows[child] for child in children])
                )
                max_residual.append(max(max_residual[child] for child in children))
                max_norm.append(max(max_norm[child] for child in children))
                min_token.append(min(min_token[child] for child in children))
                start_page.append(min(start_page[child] for child in children))
                page_count.append(sum(page_count[child] for child in children))
                child_start.append(children[0])
                child_count.append(len(children))
            level = following
        return {
            "node_presence": torch.from_numpy(np.stack(presence_rows)),
            "node_max_residual": torch.from_numpy(
                _ceil_float32(np.asarray(max_residual))
            ),
            "node_max_norm": torch.from_numpy(
                _ceil_float32(np.asarray(max_norm))
            ),
            "node_min_token": torch.tensor(min_token, dtype=torch.int64),
            "node_start_page": torch.tensor(start_page, dtype=torch.int64),
            "node_page_count": torch.tensor(page_count, dtype=torch.int64),
            "node_child_start": torch.tensor(child_start, dtype=torch.int64),
            "node_child_count": torch.tensor(child_count, dtype=torch.int64),
        }

    def _validate_tensors(self) -> None:
        if set(self._tensors) != _PAYLOAD_KEYS:
            raise ExactHeadError("exact-head payload tensor inventory is invalid")
        codebooks = self._tensors["codebooks"]
        codes = self._tensors["codes"]
        residual = self._tensors["residual_radii"]
        norms = self._tensors["row_norms"]
        subspaces = math.ceil(self.binding.hidden_size / self.config.subspace_width)
        k = min(self.config.codebook_size, self.binding.vocab_size)
        if (
            codebooks.dtype != torch.float32
            or tuple(codebooks.shape)
            != (subspaces, k, self.config.subspace_width)
            or codes.dtype != torch.uint8
            or tuple(codes.shape) != (self.binding.vocab_size, subspaces)
            or residual.dtype != torch.float32
            or tuple(residual.shape) != (self.binding.vocab_size,)
            or norms.dtype != torch.float32
            or tuple(norms.shape) != (self.binding.vocab_size,)
            or not bool(torch.isfinite(codebooks).all())
            or bool(torch.isnan(residual).any())
            or bool(torch.isnan(norms).any())
            or bool((residual < 0).any())
            or bool((norms < 0).any())
            or int(codes.max()) >= k
        ):
            raise ExactHeadError("exact-head PQ tensors are invalid")
        nodes = len(self._tensors["node_min_token"])
        packed = math.ceil(k / 8)
        if (
            self._tensors["node_presence"].dtype != torch.uint8
            or tuple(self._tensors["node_presence"].shape)
            != (nodes, subspaces, packed)
        ):
            raise ExactHeadError("exact-head node presence tensor is invalid")
        for key in ("node_max_residual", "node_max_norm"):
            if (
                self._tensors[key].dtype != torch.float32
                or tuple(self._tensors[key].shape) != (nodes,)
                or bool(torch.isnan(self._tensors[key]).any())
                or bool((self._tensors[key] < 0).any())
            ):
                raise ExactHeadError(f"{key} is invalid")
        for key in ("node_min_token", "node_start_page", "node_page_count", "node_child_start", "node_child_count"):
            if self._tensors[key].dtype != torch.int64 or tuple(self._tensors[key].shape) != (nodes,):
                raise ExactHeadError(f"{key} is invalid")
        leaves = math.ceil(self.binding.vocab_size / self.config.page_rows)
        if nodes < leaves or int(self._tensors["node_page_count"][-1]) != leaves:
            raise ExactHeadError("exact-head tree does not cover every page")
        presence = self._tensors["node_presence"].numpy()
        unpacked_presence = np.unpackbits(
            presence, axis=-1, bitorder="little"
        )
        if bool(unpacked_presence[..., k:].any()):
            raise ExactHeadError("exact-head node mask has nonzero unused bits")
        residual_values = residual.numpy()
        norm_values = norms.numpy()
        code_values = codes.numpy()
        for page in range(leaves):
            begin = page * self.config.page_rows
            end = min(self.binding.vocab_size, begin + self.config.page_rows)
            for subspace in range(subspaces):
                expected = np.zeros(k, dtype=np.uint8)
                expected[np.unique(code_values[begin:end, subspace])] = 1
                packed_expected = np.packbits(expected, bitorder="little")
                if not np.array_equal(presence[page, subspace], packed_expected):
                    raise ExactHeadError("exact-head leaf code mask is invalid")
            if (
                float(self._tensors["node_max_residual"][page])
                < float(np.max(residual_values[begin:end]))
                or float(self._tensors["node_max_norm"][page])
                < float(np.max(norm_values[begin:end]))
                or int(self._tensors["node_min_token"][page]) != begin
                or int(self._tensors["node_start_page"][page]) != page
                or int(self._tensors["node_page_count"][page]) != 1
                or int(self._tensors["node_child_start"][page]) != -1
                or int(self._tensors["node_child_count"][page]) != 0
            ):
                raise ExactHeadError("exact-head leaf aggregate is invalid")
        parent_counts = [0] * nodes
        for node in range(leaves, nodes):
            first = int(self._tensors["node_child_start"][node])
            count = int(self._tensors["node_child_count"][node])
            if (
                first < 0
                or not 1 <= count <= self.config.fanout
                or first + count > node
            ):
                raise ExactHeadError("exact-head internal child range is invalid")
            children = tuple(range(first, first + count))
            cursor = int(self._tensors["node_start_page"][children[0]])
            interval_start = cursor
            for child in children:
                if int(self._tensors["node_start_page"][child]) != cursor:
                    raise ExactHeadError(
                        "exact-head child page intervals overlap or have gaps"
                    )
                cursor += int(self._tensors["node_page_count"][child])
                parent_counts[child] += 1
            if (
                not np.array_equal(
                    presence[node],
                    np.bitwise_or.reduce(presence[list(children)], axis=0),
                )
                or float(self._tensors["node_max_residual"][node])
                < max(
                    float(self._tensors["node_max_residual"][child])
                    for child in children
                )
                or float(self._tensors["node_max_norm"][node])
                < max(
                    float(self._tensors["node_max_norm"][child])
                    for child in children
                )
                or int(self._tensors["node_min_token"][node])
                != min(
                    int(self._tensors["node_min_token"][child])
                    for child in children
                )
                or int(self._tensors["node_start_page"][node])
                != interval_start
                or int(self._tensors["node_page_count"][node])
                != cursor - interval_start
            ):
                raise ExactHeadError("exact-head internal aggregate is invalid")
        root = nodes - 1
        if (
            int(self._tensors["node_start_page"][root]) != 0
            or int(self._tensors["node_page_count"][root]) != leaves
            or parent_counts[root] != 0
            or any(parent_counts[node] != 1 for node in range(root))
        ):
            raise ExactHeadError("exact-head tree parenthood is invalid")

    def save(self, root: str | Path) -> ExactHeadReceipt:
        target = Path(root).expanduser().absolute()
        target.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            raise ExactHeadError("exact-head root must not be a symlink")
        if any(target.iterdir()):
            raise ExactHeadError("exact-head artifact destination is not empty")
        payload = target / _PAYLOAD_NAME
        manifest = target / _MANIFEST_NAME
        descriptor, temporary = tempfile.mkstemp(prefix=".exact-head-", suffix=".safetensors", dir=target)
        os.close(descriptor)
        try:
            save_file(self._tensors, temporary)
            with open(temporary, "rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, payload)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        payload_data = _stable_read(payload, 8 * 1024**3)
        payload_sha = _sha256_bytes(payload_data)
        body = {
            "binding": self.binding.to_record(),
            "config": self.config.to_record(),
            "index_bytes": sum(value.numel() * value.element_size() for value in self._tensors.values()),
            "payload_bytes": len(payload_data),
            "payload_name": _PAYLOAD_NAME,
            "payload_sha256": payload_sha,
            "payload_tensors": sorted(_PAYLOAD_KEYS),
            "schema": EXACT_HEAD_SCHEMA,
        }
        envelope = {
            "body": body,
            "body_sha256": _sha256_bytes(canonical_json_bytes(body)),
            "schema": EXACT_HEAD_MANIFEST_SCHEMA,
        }
        manifest_data = canonical_json_bytes(envelope)
        descriptor, manifest_tmp = tempfile.mkstemp(prefix=".exact-head-", suffix=".json", dir=target)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(manifest_data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(manifest_tmp, manifest)
        finally:
            if os.path.exists(manifest_tmp):
                os.unlink(manifest_tmp)
        directory_fd = os.open(
            target,
            os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0)),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        self.receipt = ExactHeadReceipt(
            manifest_sha256=_sha256_bytes(manifest_data),
            payload_sha256=payload_sha,
            payload_bytes=len(payload_data),
            tensor_sha256=self.binding.tensor_sha256,
            index_bytes=cast(int, body["index_bytes"]),
        )
        return self.receipt

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        expected_binding: ExactHeadBinding | None = None,
        max_payload_bytes: int = 256 * 1024**2,
    ) -> "ExactHeadIndex":
        if (
            isinstance(max_payload_bytes, bool)
            or not isinstance(max_payload_bytes, int)
            or max_payload_bytes < 1
        ):
            raise ValueError("max_payload_bytes must be a positive integer")
        base = Path(root).expanduser().absolute()
        try:
            metadata = base.lstat()
        except OSError as exc:
            raise ExactHeadError("exact-head root is missing") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ExactHeadError("exact-head root must be a plain directory")
        root_fd: int | None = None
        try:
            root_fd = os.open(
                base,
                os.O_RDONLY
                | int(getattr(os, "O_CLOEXEC", 0))
                | int(getattr(os, "O_DIRECTORY", 0))
                | int(getattr(os, "O_NOFOLLOW", 0)),
            )
            opened = os.fstat(root_fd)
            if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                raise ExactHeadError("exact-head root changed while opened")
            manifest_data = _stable_read_at(
                root_fd, _MANIFEST_NAME, _MAX_MANIFEST_BYTES
            )
            payload_data = _stable_read_at(
                root_fd, _PAYLOAD_NAME, max_payload_bytes
            )
        except ExactHeadError:
            raise
        except OSError as exc:
            raise ExactHeadError("cannot open exact-head root") from exc
        finally:
            if root_fd is not None:
                os.close(root_fd)
        envelope = _strict_json(manifest_data)
        body = envelope.get("body")
        expected_body = {
            "binding",
            "config",
            "index_bytes",
            "payload_bytes",
            "payload_name",
            "payload_sha256",
            "payload_tensors",
            "schema",
        }
        if (
            set(envelope) != {"body", "body_sha256", "schema"}
            or envelope.get("schema") != EXACT_HEAD_MANIFEST_SCHEMA
            or not isinstance(body, Mapping)
            or set(body) != expected_body
            or envelope.get("body_sha256") != _sha256_bytes(canonical_json_bytes(body))
            or body.get("schema") != EXACT_HEAD_SCHEMA
            or body.get("payload_name") != _PAYLOAD_NAME
            or body.get("payload_tensors") != sorted(_PAYLOAD_KEYS)
        ):
            raise ExactHeadError("exact-head manifest envelope is invalid")
        config = ExactHeadConfig.from_record(body.get("config"))
        binding = ExactHeadBinding.from_record(body.get("binding"))
        if expected_binding is not None and binding != expected_binding:
            raise ExactHeadError("exact-head binding differs from the mounted target")
        if (
            body.get("payload_bytes") != len(payload_data)
            or body.get("payload_sha256") != _sha256_bytes(payload_data)
        ):
            raise ExactHeadError("exact-head payload hash changed")
        tensors = load(payload_data)
        index = cls(
            config=config,
            binding=binding,
            tensors=tensors,
            manifest_sha256=_sha256_bytes(manifest_data),
            payload_sha256=_sha256_bytes(payload_data),
            payload_bytes=len(payload_data),
        )
        if body.get("index_bytes") != index.receipt.index_bytes:
            raise ExactHeadError("exact-head index byte count changed")
        return index

    def metrics(self) -> dict[str, object]:
        with self._metrics_lock:
            return {**asdict(self._metrics), "manifest_sha256": self.receipt.manifest_sha256}

    def close(self) -> None:
        with self._runtime_lock:
            self._closed = True
            self._tensors.clear()

    def _fallback(self, reason: str) -> None:
        with self._metrics_lock:
            self._metrics.fallback_calls += 1
            self._metrics.last_fallback_reason = reason

    def validate_mount(
        self,
        pager: Any,
        *,
        name: str,
        block_rows: int,
    ) -> None:
        if self._closed:
            raise ExactHeadError("exact-head index is closed")
        if (
            str(pager.device) != "cpu"
            or pager.compute_dtype != torch.bfloat16
            or self.binding.torch_version != torch.__version__
        ):
            raise ExactHeadNotApplicable(
                "exact-head score ABI differs from the pager"
            )
        if name != self.binding.tensor_name or (
            not self.supports_q4 and block_rows != self.config.page_rows
        ):
            raise ExactHeadNotApplicable(
                "exact-head head/page ABI differs from the pager"
            )
        layout = pager._layout(name)
        if layout.dtype != "BF16" or layout.shape != (
            self.binding.vocab_size,
            self.binding.hidden_size,
        ):
            raise ExactHeadError("exact-head layout differs from the pager")
        if self.supports_q4:
            q4_bank = getattr(pager, "q4_bank", None)
            try:
                entry = q4_bank.entries[name]
            except (AttributeError, KeyError) as exc:
                raise ExactHeadNotApplicable(
                    "Q4 exact-head packed matrix is unavailable"
                ) from exc
            if (
                entry.format != "q8_0"
                or tuple(entry.shape) != tuple(layout.shape)
                or entry.payload_sha256 != self.binding.tensor_sha256
                or self.binding.tensor_dtype != "Q8_0"
                or self.binding.backend_sha256 != _q4_backend_sha256(q4_bank)
            ):
                raise ExactHeadNotApplicable(
                    "Q4 exact-head score ABI differs from the packed target"
                )
        elif (
            self.binding.score_abi != EXACT_HEAD_SCORE_ABI
            or getattr(pager, "HEAD_SCORE_POLICY", None) != self.binding.score_abi
            or self.binding.backend_sha256 != _backend_sha256()
        ):
            raise ExactHeadNotApplicable(
                "exact-head score ABI differs from the pager"
            )
        metrics = pager.source.metrics()
        if (
            metrics.get("repo_id") != self.binding.repo_id
            or metrics.get("revision") != self.binding.revision
            or metrics.get("inventory_source_fingerprint")
            != self.binding.inventory_fingerprint
        ):
            raise ExactHeadError("exact-head source identity differs from the pager")
        find = getattr(pager.source, "find", None)
        if callable(find) and not self.supports_q4:
            metadata = find(name)
            raw_sha256 = metadata.get("raw_sha256") if isinstance(metadata, Mapping) else None
            if raw_sha256 is not None and raw_sha256 != self.binding.tensor_sha256:
                raise ExactHeadError("exact-head tensor digest differs from the pager")

    def _applicable(self, pager: Any, hidden: torch.Tensor, *, k: int, name: str, block_rows: int) -> tuple[bool, str]:
        try:
            self.validate_mount(pager, name=name, block_rows=block_rows)
        except ExactHeadError as exc:
            return False, str(exc)
        flat_rows = hidden.numel() // hidden.shape[-1]
        if flat_rows < 1 or flat_rows > self.config.max_query_rows or hidden.requires_grad:
            return False, "query-shape"
        if not bool(torch.isfinite(hidden).all()) or k < 1:
            return False, "query-values"
        if _bf16_has_subnormal(hidden):
            return False, "query-subnormal"
        if self.supports_q4 and self._q4_zero_prune_disabled:
            return False, "q4-zero-prune-disabled"
        return True, ""

    def _query_bound_tables(self, hidden: torch.Tensor) -> _QueryBoundTables:
        flat = np.asarray(hidden.detach().cpu().float().numpy(), dtype=np.float64).reshape(
            -1, self.binding.hidden_size
        )
        codebooks = np.asarray(self._tensors["codebooks"].numpy(), dtype=np.float64)
        queries = len(flat)
        subspaces, k, _width = codebooks.shape
        scores = np.empty((queries, subspaces, k), dtype=np.float64)
        absolute = np.empty_like(scores)
        hnorm = np.asarray(_norm_upper(flat), dtype=np.float64)
        for subspace in range(subspaces):
            begin = subspace * self.config.subspace_width
            end = min(self.binding.hidden_size, begin + self.config.subspace_width)
            h = flat[:, begin:end]
            c = codebooks[subspace, :, : end - begin]
            raw_scores = h @ c.T
            raw_absolute = np.abs(h) @ np.abs(c).T
            u64 = 2.0**-53
            gamma64 = (end - begin) * u64 / (1.0 - (end - begin) * u64)
            scores[:, subspace] = np.nextafter(
                raw_scores + gamma64 * raw_absolute,
                np.inf,
            )
            absolute[:, subspace] = np.nextafter(
                raw_absolute / (1.0 - gamma64),
                np.inf,
            )
        return _QueryBoundTables(
            scores=scores,
            absolute=absolute,
            hnorm=hnorm,
        )

    def _finish_caps(
        self,
        caps: np.ndarray,
        absolute: np.ndarray,
        *,
        hnorm: np.ndarray,
        residual: np.ndarray,
    ) -> np.ndarray:
        subspaces = self._tensors["codebooks"].shape[0]
        u64 = 2.0**-53
        gamma_subspaces = subspaces * u64 / (1.0 - subspaces * u64)
        caps = np.nextafter(caps + gamma_subspaces * absolute, np.inf)
        absolute = np.nextafter(
            absolute / (1.0 - gamma_subspaces),
            np.inf,
        )
        residual_term = np.nextafter(
            hnorm[:, None] * residual[None, :],
            np.inf,
        )
        caps = np.nextafter(caps + residual_term, np.inf)
        absolute = np.nextafter(absolute + residual_term, np.inf)
        dimension = self.binding.hidden_size
        operations = 2 * dimension + 4
        u32 = 2.0**-24
        gamma = operations * u32 / (1.0 - operations * u32)
        beta = operations * 2.0**-126 / (1.0 - operations * u32)
        accumulator = np.nextafter(caps + gamma * absolute + beta, np.inf)
        accumulator = np.nextafter(
            accumulator
            + 2.0**-7 * (absolute + gamma * absolute + beta)
            + 2.0**-133,
            np.inf,
        )
        overflow = (
            absolute >= np.finfo(np.float32).max
        ) | (accumulator > float(torch.finfo(torch.bfloat16).max))
        accumulator[overflow] = np.inf
        return accumulator

    def _node_caps_from_tables(self, tables: _QueryBoundTables) -> np.ndarray:
        presence = np.asarray(self._tensors["node_presence"].numpy())
        residual = np.asarray(
            self._tensors["node_max_residual"].numpy(),
            dtype=np.float64,
        )
        k = tables.scores.shape[2]
        node_count = len(presence)
        caps = np.zeros((len(tables.hnorm), node_count), dtype=np.float64)
        absolute = np.zeros_like(caps)
        for subspace in range(tables.scores.shape[1]):
            unpacked = np.unpackbits(
                presence[:, subspace], axis=-1, bitorder="little"
            )[:, :k].astype(bool)
            for node in range(node_count):
                codes = unpacked[node]
                caps[:, node] += np.max(
                    tables.scores[:, subspace, codes], axis=1
                )
                absolute[:, node] += np.max(
                    tables.absolute[:, subspace, codes], axis=1
                )
        return self._finish_caps(
            caps,
            absolute,
            hnorm=tables.hnorm,
            residual=residual,
        )

    def _row_caps_from_tables(
        self,
        tables: _QueryBoundTables,
        row_ids: np.ndarray,
    ) -> np.ndarray:
        codes = np.asarray(self._tensors["codes"].numpy()[row_ids], dtype=np.int64)
        residual = np.asarray(
            self._tensors["residual_radii"].numpy()[row_ids],
            dtype=np.float64,
        )
        caps = np.zeros((len(tables.hnorm), len(row_ids)), dtype=np.float64)
        absolute = np.zeros_like(caps)
        for subspace in range(tables.scores.shape[1]):
            selected = codes[:, subspace]
            caps += tables.scores[:, subspace, selected]
            absolute += tables.absolute[:, subspace, selected]
        return self._finish_caps(
            caps,
            absolute,
            hnorm=tables.hnorm,
            residual=residual,
        )

    def node_caps(self, hidden: torch.Tensor) -> np.ndarray:
        return self._node_caps_from_tables(self._query_bound_tables(hidden))

    def row_caps(
        self,
        hidden: torch.Tensor,
        row_ids: Any,
    ) -> np.ndarray:
        try:
            ids = tuple(int(value) for value in row_ids)
        except (TypeError, ValueError) as exc:
            raise ValueError("row_ids must be an iterable of token IDs") from exc
        if not ids:
            return np.empty((hidden.numel() // hidden.shape[-1], 0), dtype=np.float64)
        if any(value < 0 or value >= self.binding.vocab_size for value in ids):
            raise IndexError("row ID is outside the bound vocabulary")
        return self._row_caps_from_tables(
            self._query_bound_tables(hidden),
            np.asarray(ids, dtype=np.int64),
        )

    def leaf_caps(self, hidden: torch.Tensor) -> np.ndarray:
        return self.node_caps(hidden)[:, : self.leaf_count]

    @staticmethod
    def prunable(cap: float, threshold_value: float, page_min_token: int, threshold_token: int) -> bool:
        return cap < threshold_value or (
            cap == threshold_value and page_min_token >= threshold_token
        )

    @staticmethod
    def _selected_read_is_economic(
        selected_ids: tuple[int, ...],
        page_rows: int,
    ) -> bool:
        """Use scattered transport only for a material, low-run byte cut."""

        if not selected_ids or len(selected_ids) * 2 > page_rows:
            return False
        runs = 1 + sum(
            following != previous + 1
            for previous, following in zip(selected_ids, selected_ids[1:])
        )
        return runs <= 4

    def topk_logits(
        self,
        pager: Any,
        hidden: torch.Tensor,
        *,
        k: int,
        name: str,
        block_rows: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        with self._runtime_lock:
            with self._metrics_lock:
                self._metrics.calls += 1
            applicable, reason = self._applicable(
                pager, hidden, k=k, name=name, block_rows=block_rows
            )
            if not applicable:
                self._fallback(reason)
                return None
            with self._metrics_lock:
                self._metrics.applicable_calls += 1
            leading = tuple(hidden.shape[:-1])
            flat = hidden.reshape(-1, self.binding.hidden_size)
            q4_bank = getattr(pager, "q4_bank", None) if self.supports_q4 else None
            bound_hidden = (
                q4_bank.quantized_input(flat, dtype=torch.float32)
                if q4_bank is not None
                else flat
            )
            bound_tables = self._query_bound_tables(bound_hidden)
            caps = self._node_caps_from_tables(bound_tables)
            child_start = self._tensors["node_child_start"]
            child_count = self._tensors["node_child_count"]
            start_page = self._tensors["node_start_page"]
            node_pages = self._tensors["node_page_count"]
            node_min = self._tensors["node_min_token"]
            frontier: list[tuple[float, int]] = [
                (-float(np.max(caps[:, self.root_index])), self.root_index)
            ]
            best_values: torch.Tensor | None = None
            best_ids: torch.Tensor | None = None
            pages_scored = pages_pruned = rows_scored = rows_pruned = 0
            bound_nodes = 0
            row_bound_rows = 0
            selected_row_reads = 0
            selected_rows_scored = 0
            full_leaf_fallbacks = 0
            selected_row_cost_fallbacks = 0
            row_bound_probe_pages = 0
            row_bound_saving_pages = 0
            row_bounds_enabled = True
            row_bounds_disabled = False
            row_certificate_rows_pruned = 0
            while frontier:
                _priority, node = heapq.heappop(frontier)
                bound_nodes += 1
                if (
                    row_bounds_enabled
                    and best_values is not None
                    and best_values.shape[-1] == k
                ):
                    skip = all(
                        self.prunable(
                            float(caps[row, node]),
                            float(best_values[row, -1]),
                            int(node_min[node]),
                            int(best_ids[row, -1]),
                        )
                        for row in range(len(flat))
                    )
                    if skip:
                        pages = int(node_pages[node])
                        page = int(start_page[node])
                        begin = page * self.config.page_rows
                        end = min(
                            self.binding.vocab_size,
                            (page + pages) * self.config.page_rows,
                        )
                        pages_pruned += pages
                        rows_pruned += end - begin
                        continue
                children = int(child_count[node])
                if children:
                    first = int(child_start[node])
                    for child in range(first, first + children):
                        heapq.heappush(
                            frontier,
                            (-float(np.max(caps[:, child])), child),
                        )
                    continue
                page = int(start_page[node])
                start = page * self.config.page_rows
                count = min(
                    self.config.page_rows, self.binding.vocab_size - start
                )
                selected_ids = tuple(range(start, start + count))
                if best_values is not None and best_values.shape[-1] == k:
                    page_ids = np.arange(start, start + count, dtype=np.int64)
                    per_row_caps = self._row_caps_from_tables(
                        bound_tables,
                        page_ids,
                    )
                    row_bound_rows += count
                    row_bound_probe_pages += 1
                    selected_ids = tuple(
                        int(token_id)
                        for column, token_id in enumerate(page_ids)
                        if not all(
                            self.prunable(
                                float(per_row_caps[row, column]),
                                float(best_values[row, -1]),
                                int(token_id),
                                int(best_ids[row, -1]),
                            )
                            for row in range(len(flat))
                        )
                    )
                    certified = count - len(selected_ids)
                    rows_pruned += certified
                    row_certificate_rows_pruned += certified
                    if not selected_ids:
                        row_bound_saving_pages += 1
                        pages_pruned += 1
                        continue

                rows = None
                if q4_bank is not None:
                    use_selected = len(selected_ids) < count
                    if use_selected:
                        row_bound_saving_pages += 1
                        selected_row_reads += 1
                        selected_rows_scored += len(selected_ids)
                    else:
                        full_leaf_fallbacks += 1
                    preflight = getattr(pager, "_preflight_q4_linear_rows", None)
                    if not callable(preflight):
                        raise ExactHeadError(
                            "Q4 exact-head scorer lacks selected-row preflight"
                        )
                    preflight(
                        pager._layout(name),
                        input_rows=len(flat),
                        selected_rows=len(selected_ids),
                        output_dtype=pager.compute_dtype,
                        label=f"{name} exact-head survivors",
                    )
                    try:
                        logits = q4_bank.linear_rows(
                            flat,
                            name,
                            selected_ids,
                            output_dtype=pager.compute_dtype,
                        )
                    finally:
                        q4_bank.discard_rows(name, start, count)
                else:
                    selected_reader = getattr(pager, "_selected_rows", None)
                    score_preflight = getattr(pager, "_preflight_head_score", None)
                    use_selected = (
                        len(selected_ids) < count
                        and callable(selected_reader)
                        and callable(score_preflight)
                        and self._selected_read_is_economic(selected_ids, count)
                    )
                    if use_selected:
                        row_bound_saving_pages += 1
                        score_preflight(
                            query_rows=len(flat),
                            head_rows=count,
                            columns=self.binding.hidden_size,
                        )
                        rows = selected_reader(name, selected_ids)
                        selected_offsets = torch.tensor(
                            [token_id - start for token_id in selected_ids],
                            device=pager.device,
                            dtype=torch.long,
                        )
                        score_rows = torch.zeros(
                            (count, self.binding.hidden_size),
                            device=pager.device,
                            dtype=pager.compute_dtype,
                        )
                        score_rows.index_copy_(0, selected_offsets, rows)
                        del rows
                        page_logits = pager._score_head_rows(flat, score_rows)
                        logits = page_logits.index_select(-1, selected_offsets)
                        del page_logits, score_rows, selected_offsets
                        selected_row_reads += 1
                        selected_rows_scored += len(selected_ids)
                    else:
                        if len(selected_ids) < count:
                            selected_row_cost_fallbacks += 1
                            rows_pruned -= count - len(selected_ids)
                            row_certificate_rows_pruned -= count - len(selected_ids)
                            selected_ids = tuple(range(start, start + count))
                        full_leaf_fallbacks += 1
                        rows = pager._read_rows(
                            name,
                            start,
                            count,
                            dtype=pager.compute_dtype,
                            device=pager.device,
                        )
                        logits = pager._score_head_rows(flat, rows)
                if (
                    row_bounds_enabled
                    and row_bound_probe_pages >= _ROW_BOUND_PROBE_PAGES
                    and row_bound_saving_pages == 0
                ):
                    if (
                        q4_bank is not None
                        and pages_pruned == 0
                        and rows_pruned == 0
                    ):
                        self._q4_zero_prune_disabled = True
                        with self._metrics_lock:
                            self._metrics.pages_scored += pages_scored
                            self._metrics.rows_scored += rows_scored
                            self._metrics.bound_nodes += bound_nodes
                            self._metrics.row_bound_rows += row_bound_rows
                            self._metrics.full_leaf_fallbacks += (
                                full_leaf_fallbacks
                            )
                            self._metrics.row_bound_probe_pages += (
                                row_bound_probe_pages
                            )
                            self._metrics.row_bound_disabled_calls += 1
                            self._metrics.packed_rows_scored += rows_scored
                        self._fallback("q4-zero-prune-probe")
                        return None
                    row_bounds_enabled = False
                    row_bounds_disabled = True
                token_ids = torch.tensor(
                    selected_ids,
                    device=pager.device,
                    dtype=torch.long,
                ).expand_as(logits)
                values, ids = pager._stable_topk(
                    logits, token_ids, min(k, len(selected_ids))
                )
                del logits
                if q4_bank is None and not use_selected:
                    del rows
                pager._stats.head_rows += len(selected_ids)
                pager._stats.materialized_weight_releases += 1
                pages_scored += 1
                rows_scored += len(selected_ids)
                if best_values is None:
                    best_values, best_ids = values, ids
                else:
                    merge_k = min(
                        k, best_values.shape[-1] + values.shape[-1]
                    )
                    best_values, best_ids = pager._stable_topk(
                        torch.cat((best_values, values), dim=-1),
                        torch.cat((best_ids, ids), dim=-1),
                        merge_k,
                    )
            assert best_values is not None and best_ids is not None
            with self._metrics_lock:
                self._metrics.pages_scored += pages_scored
                self._metrics.pages_pruned += pages_pruned
                self._metrics.rows_scored += rows_scored
                self._metrics.rows_pruned += rows_pruned
                self._metrics.bound_nodes += bound_nodes
                self._metrics.row_bound_rows += row_bound_rows
                self._metrics.selected_row_reads += selected_row_reads
                self._metrics.selected_rows_scored += selected_rows_scored
                self._metrics.full_leaf_fallbacks += full_leaf_fallbacks
                self._metrics.selected_row_cost_fallbacks += (
                    selected_row_cost_fallbacks
                )
                self._metrics.row_bound_probe_pages += row_bound_probe_pages
                self._metrics.row_bound_disabled_calls += int(row_bounds_disabled)
                logical_row_bytes = (
                    q4_bank.entries[name].row_bytes
                    if q4_bank is not None
                    else self.binding.hidden_size * 2
                )
                selected_bytes = selected_rows_scored * logical_row_bytes
                row_avoided_bytes = (
                    row_certificate_rows_pruned * logical_row_bytes
                )
                self._metrics.selected_row_logical_bytes += selected_bytes
                self._metrics.row_certificate_logical_bytes_avoided += (
                    row_avoided_bytes
                )
                self._metrics.logical_head_bytes_avoided += (
                    rows_pruned * logical_row_bytes
                )
                if q4_bank is not None:
                    self._metrics.packed_rows_scored += rows_scored
                    self._metrics.packed_rows_avoided += rows_pruned
                    self._metrics.packed_weight_bytes_avoided += (
                        rows_pruned * logical_row_bytes
                    )
                self._metrics.last_fallback_reason = ""
            return (
                (
                    best_values.to(dtype=pager.compute_dtype)
                    if q4_bank is not None
                    else best_values
                ).reshape(*leading, k),
                best_ids.reshape(*leading, k),
            )


__all__ = [
    "EXACT_HEAD_MANIFEST_SCHEMA",
    "EXACT_HEAD_SCHEMA",
    "EXACT_HEAD_SCORE_ABI",
    "Q4_EXACT_HEAD_SCORE_ABI",
    "ExactHeadBinding",
    "ExactHeadConfig",
    "ExactHeadError",
    "ExactHeadIndex",
    "ExactHeadMetrics",
    "ExactHeadNotApplicable",
    "ExactHeadReceipt",
]
