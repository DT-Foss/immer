"""Stateful, portable DeepSeek-V4 attention cache machinery.

The public checkpoint's attention module combines a circular local window with
learned KV compression.  The official implementation stores that state inside
CUDA ``nn.Module`` buffers; this module separates it from weight ownership so
the range-streamed runtime can keep only activations resident.  Projection,
normalization, and QAT are callbacks, which keeps streamed weights in the pager
and makes the state itself usable on both CPU and Apple MPS.

Only two native call shapes exist in the published forward: an arbitrary
position-zero prefill and contiguous one-token decode.  ``DeepSeekV4Compressor``
also accepts a non-zero multi-token chunk as a convenience and evaluates it as
successive native decode steps, preserving the exact state transition.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Mapping

import torch

from .kernels import (
    apply_rotary_emb,
    compressed_indices,
    sparse_attention,
    window_indices,
)
from .snapshot import SnapshotTensor


LinearCallback = Callable[[torch.Tensor, str], torch.Tensor]
RMSCallback = Callable[[torch.Tensor, str], torch.Tensor]
QATCallback = Callable[[torch.Tensor, str], torch.Tensor]
QATMode = Literal["compressed-kv", "indexer"]


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _position(value: int, name: str = "start_pos") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _floating_3d(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point torch tensor")
    if value.ndim != 3:
        raise ValueError(f"{name} must have [batch, sequence, feature] shape")
    if value.shape[0] <= 0 or value.shape[1] <= 0 or value.shape[2] <= 0:
        raise ValueError(f"{name} dimensions must be non-empty")
    return value


def _identity_qat(value: torch.Tensor, _mode: str) -> torch.Tensor:
    return value


def _snapshot_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"snapshot {name} must be an object")
    return value


def _snapshot_int(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"snapshot {name} is invalid")
    if maximum is not None and value > maximum:
        raise ValueError(f"snapshot {name} exceeds its bound")
    return value


def _snapshot_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"snapshot {name} must be boolean")
    return value


def _snapshot_tensor(
    tensors: Mapping[str, torch.Tensor],
    name: Any,
    *,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(name, str) or name not in tensors:
        raise ValueError("snapshot tensor reference is missing")
    value = tensors[name]
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"snapshot tensor {name!r} has an incompatible shape")
    if not value.is_floating_point():
        raise ValueError(f"snapshot tensor {name!r} must be floating-point")
    # The transactional model loader gives every role a unique, already
    # verified tensor.  Taking ownership without cloning avoids a second full
    # device copy while the previous request remains available for rollback.
    return value.detach()


class CompressedKVCache:
    """Lazily growing compressed cache with an explicit hard context bound."""

    def __init__(self, max_batch_size: int, max_entries: int, head_dim: int) -> None:
        self.max_batch_size = _positive_int(max_batch_size, "max_batch_size")
        if isinstance(max_entries, bool) or not isinstance(max_entries, int):
            raise TypeError("max_entries must be an integer")
        if max_entries < 0:
            raise ValueError("max_entries must be non-negative")
        self.max_entries = max_entries
        self.head_dim = _positive_int(head_dim, "head_dim")
        self._storage: torch.Tensor | None = None
        self._batch_size = 0
        self._length = 0

    @property
    def length(self) -> int:
        return self._length

    @property
    def capacity(self) -> int:
        return 0 if self._storage is None else self._storage.shape[1]

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def nbytes(self) -> int:
        if self._storage is None:
            return 0
        return self._storage.numel() * self._storage.element_size()

    def _ensure(self, reference: torch.Tensor, required: int) -> None:
        if required > self.max_entries:
            raise ValueError(
                f"compressed cache requires {required} entries, limit is {self.max_entries}"
            )
        batch = reference.shape[0]
        if batch > self.max_batch_size:
            raise ValueError(
                f"batch size {batch} exceeds configured maximum {self.max_batch_size}"
            )
        if self._storage is not None:
            if batch != self._batch_size:
                if self._length:
                    raise ValueError("batch size cannot change while cache is active")
                self._storage = None
                self._batch_size = 0
            if self._storage is not None and (
                self._storage.device != reference.device
                or self._storage.dtype != reference.dtype
            ):
                if self._length:
                    raise ValueError("cache device/dtype cannot change while active")
                self._storage = None
                self._batch_size = 0
            if self._storage is not None and required <= self.capacity:
                return
        elif required == 0:
            self._batch_size = batch
            return

        old_capacity = self.capacity
        capacity = min(
            self.max_entries,
            max(required, 1 if old_capacity == 0 else old_capacity * 2),
        )
        storage = reference.new_zeros((batch, capacity, self.head_dim))
        if self._storage is not None and self._length:
            storage[:, : self._length].copy_(self._storage[:, : self._length])
        self._storage = storage
        self._batch_size = batch

    def write(self, start: int, values: torch.Tensor) -> None:
        _position(start, "start")
        values = _floating_3d(values, "values")
        values = values.detach()
        if values.shape[-1] != self.head_dim:
            raise ValueError("compressed value feature size does not match head_dim")
        end = start + values.shape[1]
        if start > self._length:
            raise ValueError("compressed cache writes must be contiguous or overwrite")
        self._ensure(values, end)
        if values.shape[1]:
            assert self._storage is not None
            self._storage[:, start:end].copy_(values)
        self._length = max(self._length, end)

    def active(self) -> torch.Tensor:
        if self._storage is None:
            raise RuntimeError("compressed cache has no tensor until its first write")
        return self._storage[:, : self._length]

    def active_like(self, reference: torch.Tensor) -> torch.Tensor:
        """Return active storage, including a correctly typed empty cache."""

        if self._storage is None:
            return reference.new_empty((reference.shape[0], 0, self.head_dim))
        return self.active()

    def reset(self, *, release: bool = False) -> None:
        self._length = 0
        if release:
            self._storage = None
            self._batch_size = 0
        elif self._storage is not None:
            self._storage.zero_()

    def _snapshot_state(
        self,
        prefix: str,
        tensors: dict[str, SnapshotTensor],
    ) -> dict[str, Any]:
        if self._length < 0 or self._length > self.max_entries:
            raise ValueError("compressed cache length is inconsistent")
        capacity = self.capacity
        if self._length > capacity:
            raise ValueError("compressed cache length exceeds allocated capacity")
        if self._storage is None:
            if self._batch_size != 0 or capacity != 0 or self._length != 0:
                raise ValueError("unallocated compressed cache has mutable state")
            tensor_name = None
        else:
            if not 0 < self._batch_size <= self.max_batch_size:
                raise ValueError("compressed cache batch size is inconsistent")
            expected = (self._batch_size, capacity, self.head_dim)
            if tuple(self._storage.shape) != expected:
                raise ValueError("compressed cache storage shape is inconsistent")
            tensor_name = f"{prefix}.storage"
            tensors[tensor_name] = SnapshotTensor(self._storage)
        return {
            "kind": "compressed-kv-cache",
            "max_batch_size": self.max_batch_size,
            "max_entries": self.max_entries,
            "head_dim": self.head_dim,
            "batch_size": self._batch_size,
            "length": self._length,
            "capacity": capacity,
            "storage": tensor_name,
        }

    def _restore_snapshot(
        self,
        metadata: Any,
        tensors: Mapping[str, torch.Tensor],
    ) -> None:
        raw = _snapshot_mapping(metadata, "compressed cache")
        expected = {
            "kind": "compressed-kv-cache",
            "max_batch_size": self.max_batch_size,
            "max_entries": self.max_entries,
            "head_dim": self.head_dim,
        }
        if any(raw.get(key) != value for key, value in expected.items()):
            raise ValueError("compressed cache structure does not match the runtime")
        batch = _snapshot_int(
            raw.get("batch_size"),
            "compressed cache batch_size",
            maximum=self.max_batch_size,
        )
        length = _snapshot_int(
            raw.get("length"),
            "compressed cache length",
            maximum=self.max_entries,
        )
        capacity = _snapshot_int(
            raw.get("capacity"),
            "compressed cache capacity",
            maximum=self.max_entries,
        )
        if length > capacity:
            raise ValueError("compressed cache length exceeds capacity")
        storage_name = raw.get("storage")
        if storage_name is None:
            if batch or length or capacity:
                raise ValueError("compressed cache metadata omits required storage")
            storage = None
        else:
            if batch == 0 or capacity == 0:
                raise ValueError("compressed cache storage has empty dimensions")
            storage = _snapshot_tensor(
                tensors,
                storage_name,
                shape=(batch, capacity, self.head_dim),
            )
        self._storage = storage
        self._batch_size = batch
        self._length = length


class CircularKVCache:
    """Fixed-size physical KV ring used by the official decode indices."""

    def __init__(self, max_batch_size: int, window_size: int, head_dim: int) -> None:
        self.max_batch_size = _positive_int(max_batch_size, "max_batch_size")
        self.window_size = _positive_int(window_size, "window_size")
        self.head_dim = _positive_int(head_dim, "head_dim")
        self._storage: torch.Tensor | None = None
        self._batch_size = 0
        self._next_position = 0
        self._length = 0

    @property
    def next_position(self) -> int:
        return self._next_position

    @property
    def length(self) -> int:
        return self._length

    @property
    def nbytes(self) -> int:
        if self._storage is None:
            return 0
        return self._storage.numel() * self._storage.element_size()

    def _ensure(self, reference: torch.Tensor) -> torch.Tensor:
        batch = reference.shape[0]
        if batch > self.max_batch_size:
            raise ValueError(
                f"batch size {batch} exceeds configured maximum {self.max_batch_size}"
            )
        if self._storage is None:
            self._storage = reference.new_zeros(
                (batch, self.window_size, self.head_dim)
            )
            self._batch_size = batch
        elif (
            batch != self._batch_size
            or self._storage.device != reference.device
            or self._storage.dtype != reference.dtype
        ):
            if self._next_position:
                raise ValueError("cache batch/device/dtype cannot change while active")
            self._storage = reference.new_zeros(
                (batch, self.window_size, self.head_dim)
            )
            self._batch_size = batch
        return self._storage

    def prefill(self, values: torch.Tensor) -> None:
        values = _floating_3d(values, "values")
        values = values.detach()
        if values.shape[-1] != self.head_dim:
            raise ValueError("KV feature size does not match head_dim")
        if self._next_position != 0:
            raise ValueError("prefill requires an empty cache; call reset first")
        storage = self._ensure(values)
        storage.zero_()
        seqlen = values.shape[1]
        if seqlen <= self.window_size:
            storage[:, :seqlen].copy_(values)
        else:
            cutoff = seqlen % self.window_size
            latest = values[:, -self.window_size :]
            first, second = latest.split((self.window_size - cutoff, cutoff), dim=1)
            storage[:, cutoff:].copy_(first)
            if cutoff:
                storage[:, :cutoff].copy_(second)
        self._next_position = seqlen
        self._length = min(seqlen, self.window_size)

    def append(self, values: torch.Tensor, start_pos: int) -> None:
        values = _floating_3d(values, "values")
        values = values.detach()
        _position(start_pos)
        if values.shape[1] != 1:
            raise ValueError("circular decode appends exactly one token")
        if values.shape[-1] != self.head_dim:
            raise ValueError("KV feature size does not match head_dim")
        if start_pos != self._next_position:
            raise ValueError(
                f"non-contiguous decode: expected {self._next_position}, got {start_pos}"
            )
        storage = self._ensure(values)
        storage[:, start_pos % self.window_size].copy_(values[:, 0])
        self._next_position += 1
        self._length = min(self._length + 1, self.window_size)

    def physical(self) -> torch.Tensor:
        if self._storage is None:
            raise RuntimeError("KV cache has not been initialized")
        return self._storage

    def ordered(self) -> torch.Tensor:
        """Return the active local context from oldest to newest."""

        storage = self.physical()
        if self._length < self.window_size:
            return storage[:, : self._length]
        oldest = self._next_position % self.window_size
        return torch.cat((storage[:, oldest:], storage[:, :oldest]), dim=1)

    def reset(self, *, release: bool = False) -> None:
        self._next_position = 0
        self._length = 0
        if release:
            self._storage = None
            self._batch_size = 0
        elif self._storage is not None:
            self._storage.zero_()

    def _snapshot_state(
        self,
        prefix: str,
        tensors: dict[str, SnapshotTensor],
    ) -> dict[str, Any]:
        expected_length = min(self._next_position, self.window_size)
        if self._length != expected_length:
            raise ValueError("circular cache cursor/length is inconsistent")
        if self._storage is None:
            if self._batch_size != 0 or self._next_position or self._length:
                raise ValueError("unallocated circular cache has mutable state")
            tensor_name = None
        else:
            if not 0 < self._batch_size <= self.max_batch_size:
                raise ValueError("circular cache batch size is inconsistent")
            expected = (self._batch_size, self.window_size, self.head_dim)
            if tuple(self._storage.shape) != expected:
                raise ValueError("circular cache storage shape is inconsistent")
            tensor_name = f"{prefix}.storage"
            tensors[tensor_name] = SnapshotTensor(self._storage)
        return {
            "kind": "circular-kv-cache",
            "max_batch_size": self.max_batch_size,
            "window_size": self.window_size,
            "head_dim": self.head_dim,
            "batch_size": self._batch_size,
            "next_position": self._next_position,
            "length": self._length,
            "storage": tensor_name,
        }

    def _restore_snapshot(
        self,
        metadata: Any,
        tensors: Mapping[str, torch.Tensor],
    ) -> None:
        raw = _snapshot_mapping(metadata, "circular cache")
        expected = {
            "kind": "circular-kv-cache",
            "max_batch_size": self.max_batch_size,
            "window_size": self.window_size,
            "head_dim": self.head_dim,
        }
        if any(raw.get(key) != value for key, value in expected.items()):
            raise ValueError("circular cache structure does not match the runtime")
        batch = _snapshot_int(
            raw.get("batch_size"),
            "circular cache batch_size",
            maximum=self.max_batch_size,
        )
        next_position = _snapshot_int(
            raw.get("next_position"), "circular cache next_position"
        )
        length = _snapshot_int(
            raw.get("length"),
            "circular cache length",
            maximum=self.window_size,
        )
        if length != min(next_position, self.window_size):
            raise ValueError("circular cache cursor/length is inconsistent")
        storage_name = raw.get("storage")
        if storage_name is None:
            if batch or next_position or length:
                raise ValueError("circular cache metadata omits required storage")
            storage = None
        else:
            if batch == 0:
                raise ValueError("circular cache storage has an empty batch")
            storage = _snapshot_tensor(
                tensors,
                storage_name,
                shape=(batch, self.window_size, self.head_dim),
            )
        self._storage = storage
        self._batch_size = batch
        self._next_position = next_position
        self._length = length


class DeepSeekV4Compressor:
    """Exact learned gated-pooling state transition from the official model."""

    def __init__(
        self,
        *,
        prefix: str,
        compress_ratio: int,
        head_dim: int,
        rope_head_dim: int,
        max_batch_size: int,
        max_seq_len: int,
        ape: torch.Tensor,
        freqs_cis: torch.Tensor,
        linear: LinearCallback,
        rms: RMSCallback,
        qat: QATCallback | None = None,
        rotate: bool = False,
    ) -> None:
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("prefix must be a non-empty string")
        self.prefix = prefix
        self.compress_ratio = _positive_int(compress_ratio, "compress_ratio")
        self.head_dim = _positive_int(head_dim, "head_dim")
        self.rope_head_dim = _positive_int(rope_head_dim, "rope_head_dim")
        if self.rope_head_dim > self.head_dim or self.rope_head_dim % 2:
            raise ValueError("rope_head_dim must be even and no larger than head_dim")
        self.max_batch_size = _positive_int(max_batch_size, "max_batch_size")
        self.max_seq_len = _positive_int(max_seq_len, "max_seq_len")
        self.overlap = self.compress_ratio == 4
        self.rotate = bool(rotate)
        self.linear = linear
        self.rms = rms
        self.qat = _identity_qat if qat is None else qat
        if not callable(linear) or not callable(rms) or not callable(self.qat):
            raise TypeError("linear, rms, and qat must be callable")

        coff = 2 if self.overlap else 1
        if not isinstance(ape, torch.Tensor) or not ape.is_floating_point():
            raise TypeError("ape must be a floating-point torch tensor")
        if tuple(ape.shape) != (self.compress_ratio, coff * self.head_dim):
            raise ValueError(
                f"ape shape must be {(self.compress_ratio, coff * self.head_dim)}"
            )
        if not isinstance(freqs_cis, torch.Tensor):
            raise TypeError("freqs_cis must be a torch tensor")
        if freqs_cis.shape[0] < self.max_seq_len:
            raise ValueError("freqs_cis is shorter than max_seq_len")
        if freqs_cis.shape[1] != self.rope_head_dim // 2:
            raise ValueError("freqs_cis rotary dimension does not match rope_head_dim")
        self.ape = ape.detach()
        self.freqs_cis = freqs_cis.detach()
        self.cache = CompressedKVCache(
            self.max_batch_size,
            self.max_seq_len // self.compress_ratio,
            self.head_dim,
        )
        self._kv_state: torch.Tensor | None = None
        self._score_state: torch.Tensor | None = None
        self._batch_size = 0
        self._next_position = 0

    @property
    def next_position(self) -> int:
        return self._next_position

    @property
    def state_nbytes(self) -> int:
        total = self.cache.nbytes
        if self._kv_state is not None:
            total += self._kv_state.numel() * self._kv_state.element_size()
        if self._score_state is not None:
            total += self._score_state.numel() * self._score_state.element_size()
        return total

    def _ensure_state(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        if batch > self.max_batch_size:
            raise ValueError(
                f"batch size {batch} exceeds configured maximum {self.max_batch_size}"
            )
        coff = 2 if self.overlap else 1
        shape = (
            batch,
            coff * self.compress_ratio,
            coff * self.head_dim,
        )
        if self._kv_state is None:
            self._kv_state = torch.zeros(shape, device=x.device, dtype=torch.float32)
            self._score_state = torch.full(
                shape, float("-inf"), device=x.device, dtype=torch.float32
            )
            self._batch_size = batch
        elif batch != self._batch_size or self._kv_state.device != x.device:
            if self._next_position or self.cache.length:
                raise ValueError("state batch/device cannot change while active")
            self._kv_state = torch.zeros(shape, device=x.device, dtype=torch.float32)
            self._score_state = torch.full(
                shape, float("-inf"), device=x.device, dtype=torch.float32
            )
            self._batch_size = batch
        assert self._score_state is not None
        return self._kv_state, self._score_state

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.linear(x.float(), f"{self.prefix}.wkv")
        scores = self.linear(x.float(), f"{self.prefix}.wgate")
        if not isinstance(values, torch.Tensor) or not isinstance(scores, torch.Tensor):
            raise TypeError("linear callback must return torch tensors")
        coff = 2 if self.overlap else 1
        shape = (*x.shape[:2], coff * self.head_dim)
        if tuple(values.shape) != shape or tuple(scores.shape) != shape:
            raise ValueError(
                f"compressor projections must have shape {shape}, got "
                f"{tuple(values.shape)} and {tuple(scores.shape)}"
            )
        return values.float(), scores.float()

    def _finish(
        self,
        values: torch.Tensor,
        *,
        source_dtype: torch.dtype,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        values = self.rms(values.to(dtype=source_dtype), f"{self.prefix}.norm.weight")
        if not isinstance(values, torch.Tensor) or tuple(values.shape[-1:]) != (
            self.head_dim,
        ):
            raise ValueError("rms callback returned an incompatible compressor tensor")
        freqs = self.freqs_cis.index_select(
            0, positions.to(device=self.freqs_cis.device, dtype=torch.long)
        )
        rope = values[..., -self.rope_head_dim :]
        apply_rotary_emb(rope, freqs)
        mode: QATMode = "indexer" if self.rotate else "compressed-kv"
        quantized = self.qat(values, mode)
        if not isinstance(quantized, torch.Tensor) or quantized.shape != values.shape:
            raise ValueError("qat callback must return a tensor with unchanged shape")
        return quantized

    def _prefill(self, x: torch.Tensor) -> torch.Tensor | None:
        # Position-zero is a new native request, mirroring fresh module buffers.
        if self._next_position or self.cache.length:
            raise ValueError(
                "position-zero prefill requires reset() on non-empty state"
            )
        kv_state, score_state = self._ensure_state(x)
        values, scores = self._project(x)
        batch, seqlen = x.shape[:2]
        ratio = self.compress_ratio
        dim = self.head_dim
        remainder = seqlen % ratio
        cutoff = seqlen - remainder
        offset = ratio if self.overlap else 0

        if self.overlap and cutoff >= ratio:
            kv_state[:batch, :ratio].copy_(values[:, cutoff - ratio : cutoff].detach())
            score_state[:batch, :ratio].copy_(
                scores[:, cutoff - ratio : cutoff].detach()
                + self.ape.to(device=scores.device, dtype=scores.dtype)
            )
        if remainder:
            kv_state[:batch, offset : offset + remainder].copy_(
                values[:, cutoff:].detach()
            )
            score_state[:batch, offset : offset + remainder].copy_(
                scores[:, cutoff:].detach()
                + self.ape[:remainder].to(device=scores.device, dtype=scores.dtype)
            )

        self._next_position = seqlen
        if cutoff == 0:
            return None
        values = values[:, :cutoff].unflatten(1, (-1, ratio))
        scores = scores[:, :cutoff].unflatten(1, (-1, ratio))
        scores = scores + self.ape.to(device=scores.device, dtype=scores.dtype)
        if self.overlap:
            blocks = cutoff // ratio
            overlap_values = values.new_zeros((batch, blocks, 2 * ratio, dim))
            overlap_scores = scores.new_full(
                (batch, blocks, 2 * ratio, dim), float("-inf")
            )
            overlap_values[:, :, ratio:] = values[..., dim:]
            overlap_scores[:, :, ratio:] = scores[..., dim:]
            if blocks > 1:
                overlap_values[:, 1:, :ratio] = values[:, :-1, :, :dim]
                overlap_scores[:, 1:, :ratio] = scores[:, :-1, :, :dim]
            values, scores = overlap_values, overlap_scores
        pooled = (values * scores.softmax(dim=2)).sum(dim=2)
        positions = torch.arange(0, cutoff, ratio, device=x.device, dtype=torch.long)
        pooled = self._finish(pooled, source_dtype=x.dtype, positions=positions)
        self.cache.write(0, pooled)
        return pooled

    def _decode_one(self, x: torch.Tensor, start_pos: int) -> torch.Tensor | None:
        if x.shape[1] != 1:
            raise ValueError("native compressor decode requires one token")
        if start_pos != self._next_position:
            raise ValueError(
                f"non-contiguous compressor decode: expected {self._next_position}, "
                f"got {start_pos}"
            )
        kv_state, score_state = self._ensure_state(x)
        values, scores = self._project(x)
        ratio = self.compress_ratio
        dim = self.head_dim
        batch = x.shape[0]
        slot = start_pos % ratio
        scores = scores + self.ape[slot].to(device=scores.device, dtype=scores.dtype)
        complete = (start_pos + 1) % ratio == 0

        if self.overlap:
            kv_state[:batch, ratio + slot].copy_(values[:, 0].detach())
            score_state[:batch, ratio + slot].copy_(scores[:, 0].detach())
            if complete:
                merged_values = torch.cat(
                    (kv_state[:batch, :ratio, :dim], kv_state[:batch, ratio:, dim:]),
                    dim=1,
                )
                merged_scores = torch.cat(
                    (
                        score_state[:batch, :ratio, :dim],
                        score_state[:batch, ratio:, dim:],
                    ),
                    dim=1,
                )
                values = (merged_values * merged_scores.softmax(dim=1)).sum(
                    dim=1, keepdim=True
                )
                kv_state[:batch, :ratio].copy_(kv_state[:batch, ratio:])
                score_state[:batch, :ratio].copy_(score_state[:batch, ratio:])
        else:
            kv_state[:batch, slot].copy_(values[:, 0].detach())
            score_state[:batch, slot].copy_(scores[:, 0].detach())
            if complete:
                values = (kv_state[:batch] * score_state[:batch].softmax(dim=1)).sum(
                    dim=1, keepdim=True
                )

        self._next_position += 1
        if not complete:
            return None
        block_position = start_pos + 1 - ratio
        values = self._finish(
            values,
            source_dtype=x.dtype,
            positions=torch.tensor([block_position], device=x.device),
        )
        self.cache.write(start_pos // ratio, values)
        return values

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor | None:
        """Compress a native prefill/decode input and update the bounded cache."""

        x = _floating_3d(x, "x")
        _position(start_pos)
        end_pos = start_pos + x.shape[1]
        if end_pos > self.max_seq_len:
            raise ValueError(
                f"context end {end_pos} exceeds configured maximum {self.max_seq_len}"
            )
        if start_pos == 0:
            return self._prefill(x)
        if x.shape[1] == 1:
            return self._decode_one(x, start_pos)

        # Exact convenience path for chunked prefill after the initial chunk.
        outputs: list[torch.Tensor] = []
        for index in range(x.shape[1]):
            result = self._decode_one(x[:, index : index + 1], start_pos + index)
            if result is not None:
                outputs.append(result)
        if not outputs:
            return None
        return torch.cat(outputs, dim=1)

    __call__ = forward

    def reset(self, *, release: bool = False) -> None:
        self._next_position = 0
        self.cache.reset(release=release)
        if release:
            self._kv_state = None
            self._score_state = None
            self._batch_size = 0
        else:
            if self._kv_state is not None:
                self._kv_state.zero_()
            if self._score_state is not None:
                self._score_state.fill_(float("-inf"))

    def _snapshot_state(
        self,
        prefix: str,
        tensors: dict[str, SnapshotTensor],
    ) -> dict[str, Any]:
        if not 0 <= self._next_position <= self.max_seq_len:
            raise ValueError("compressor cursor exceeds its context bound")
        expected_cache_length = self._next_position // self.compress_ratio
        if self.cache.length != expected_cache_length:
            raise ValueError("compressor cursor/cache length is inconsistent")
        if (self._kv_state is None) != (self._score_state is None):
            raise ValueError("compressor overlap state is only partially allocated")
        if self._kv_state is None:
            if self._batch_size != 0:
                raise ValueError("unallocated compressor has a batch size")
            kv_name = score_name = None
        else:
            if not 0 < self._batch_size <= self.max_batch_size:
                raise ValueError("compressor batch size is inconsistent")
            coff = 2 if self.overlap else 1
            expected = (
                self._batch_size,
                coff * self.compress_ratio,
                coff * self.head_dim,
            )
            if (
                tuple(self._kv_state.shape) != expected
                or tuple(self._score_state.shape) != expected
            ):
                raise ValueError("compressor overlap state shape is inconsistent")
            kv_name = f"{prefix}.kv_state"
            score_name = f"{prefix}.score_state"
            tensors[kv_name] = SnapshotTensor(self._kv_state)
            tensors[score_name] = SnapshotTensor(
                self._score_state, finite_policy="finite_or_neg_inf"
            )
        return {
            "kind": "deepseek-v4-compressor",
            "prefix": self.prefix,
            "compress_ratio": self.compress_ratio,
            "head_dim": self.head_dim,
            "rope_head_dim": self.rope_head_dim,
            "max_batch_size": self.max_batch_size,
            "max_seq_len": self.max_seq_len,
            "overlap": self.overlap,
            "rotate": self.rotate,
            "batch_size": self._batch_size,
            "next_position": self._next_position,
            "kv_state": kv_name,
            "score_state": score_name,
            "cache": self.cache._snapshot_state(f"{prefix}.cache", tensors),
        }

    def _restore_snapshot(
        self,
        metadata: Any,
        tensors: Mapping[str, torch.Tensor],
    ) -> None:
        raw = _snapshot_mapping(metadata, "compressor")
        expected = {
            "kind": "deepseek-v4-compressor",
            "prefix": self.prefix,
            "compress_ratio": self.compress_ratio,
            "head_dim": self.head_dim,
            "rope_head_dim": self.rope_head_dim,
            "max_batch_size": self.max_batch_size,
            "max_seq_len": self.max_seq_len,
            "overlap": self.overlap,
            "rotate": self.rotate,
        }
        if any(raw.get(key) != value for key, value in expected.items()):
            raise ValueError("compressor structure does not match the runtime")
        batch = _snapshot_int(
            raw.get("batch_size"),
            "compressor batch_size",
            maximum=self.max_batch_size,
        )
        next_position = _snapshot_int(
            raw.get("next_position"),
            "compressor next_position",
            maximum=self.max_seq_len,
        )
        kv_name = raw.get("kv_state")
        score_name = raw.get("score_state")
        if (kv_name is None) != (score_name is None):
            raise ValueError("compressor overlap state references are inconsistent")
        if kv_name is None:
            if batch:
                raise ValueError("compressor metadata omits allocated overlap state")
            kv_state = score_state = None
        else:
            if batch == 0:
                raise ValueError("compressor overlap state has an empty batch")
            coff = 2 if self.overlap else 1
            shape = (
                batch,
                coff * self.compress_ratio,
                coff * self.head_dim,
            )
            kv_state = _snapshot_tensor(tensors, kv_name, shape=shape)
            score_state = _snapshot_tensor(tensors, score_name, shape=shape)
        self.cache._restore_snapshot(raw.get("cache"), tensors)
        expected_cache_length = next_position // self.compress_ratio
        if self.cache.length != expected_cache_length:
            raise ValueError("compressor cursor/cache length is inconsistent")
        if self.cache.batch_size not in {0, batch}:
            raise ValueError("compressor/cache batch sizes are inconsistent")
        self._kv_state = kv_state
        self._score_state = score_state
        self._batch_size = batch
        self._next_position = next_position


class DeepSeekV4Indexer:
    """Official ratio-4 learned compressed-position selector."""

    def __init__(
        self,
        *,
        prefix: str,
        compressor: DeepSeekV4Compressor,
        n_heads: int,
        head_dim: int,
        rope_head_dim: int,
        index_topk: int,
        freqs_cis: torch.Tensor,
        linear: LinearCallback,
        qat: QATCallback | None = None,
    ) -> None:
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("prefix must be a non-empty string")
        if compressor.compress_ratio != 4 or not compressor.rotate:
            raise ValueError("indexer requires a rotating ratio-4 compressor")
        self.prefix = prefix
        self.compressor = compressor
        self.n_heads = _positive_int(n_heads, "n_heads")
        self.head_dim = _positive_int(head_dim, "head_dim")
        self.rope_head_dim = _positive_int(rope_head_dim, "rope_head_dim")
        if self.head_dim != compressor.head_dim:
            raise ValueError("indexer and compressor head dimensions must match")
        if self.rope_head_dim > self.head_dim or self.rope_head_dim % 2:
            raise ValueError("invalid indexer rope_head_dim")
        self.index_topk = _positive_int(index_topk, "index_topk")
        if not isinstance(freqs_cis, torch.Tensor):
            raise TypeError("freqs_cis must be a torch tensor")
        if freqs_cis.shape[0] < compressor.max_seq_len:
            raise ValueError("freqs_cis is shorter than the compressor context")
        if freqs_cis.shape[1] != self.rope_head_dim // 2:
            raise ValueError("freqs_cis rotary dimension does not match rope_head_dim")
        self.freqs_cis = freqs_cis.detach()
        self.linear = linear
        self.qat = _identity_qat if qat is None else qat
        if not callable(linear) or not callable(self.qat):
            raise TypeError("linear and qat must be callable")

    @property
    def next_position(self) -> int:
        return self.compressor.next_position

    @property
    def cache(self) -> CompressedKVCache:
        return self.compressor.cache

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: int,
        offset: int,
    ) -> torch.Tensor:
        x = _floating_3d(x, "x")
        qr = _floating_3d(qr, "qr")
        _position(start_pos)
        _position(offset, "offset")
        if x.shape[:2] != qr.shape[:2]:
            raise ValueError("x and qr batch/sequence dimensions must match")
        if start_pos > 0 and x.shape[1] != 1:
            raise ValueError("native indexer decode requires one token")
        end_pos = start_pos + x.shape[1]
        if end_pos > self.compressor.max_seq_len:
            raise ValueError("indexer context exceeds compressor maximum")

        q = self.linear(qr, f"{self.prefix}.wq_b")
        expected = (*x.shape[:2], self.n_heads * self.head_dim)
        if not isinstance(q, torch.Tensor) or tuple(q.shape) != expected:
            raise ValueError(f"index query projection must have shape {expected}")
        q = q.unflatten(-1, (self.n_heads, self.head_dim))
        freqs = self.freqs_cis[start_pos:end_pos]
        apply_rotary_emb(q[..., -self.rope_head_dim :], freqs)
        q = self.qat(q, "indexer")
        if not isinstance(q, torch.Tensor) or tuple(q.shape) != (
            *x.shape[:2],
            self.n_heads,
            self.head_dim,
        ):
            raise ValueError("qat callback returned an incompatible index query")

        self.compressor.forward(x, start_pos)
        n_blocks = end_pos // self.compressor.compress_ratio
        kv = self.cache.active_like(x.new_empty((x.shape[0], 0, self.head_dim)))
        if kv.shape[1] != n_blocks:
            raise RuntimeError(
                f"indexer cache has {kv.shape[1]} blocks, expected {n_blocks}"
            )
        weights = self.linear(x, f"{self.prefix}.weights_proj")
        if not isinstance(weights, torch.Tensor) or tuple(weights.shape) != (
            *x.shape[:2],
            self.n_heads,
        ):
            raise ValueError("index weight projection has incompatible shape")
        weights = weights * (self.head_dim**-0.5 * self.n_heads**-0.5)

        if n_blocks == 0:
            return torch.empty((*x.shape[:2], 0), device=x.device, dtype=torch.int32)
        scores = torch.einsum("bshd,btd->bsht", q, kv.to(q.dtype))
        scores = (scores.relu() * weights.unsqueeze(-1)).sum(dim=2)
        if start_pos == 0:
            blocks = torch.arange(n_blocks, device=x.device).repeat(x.shape[1], 1)
            causal = torch.arange(1, x.shape[1] + 1, device=x.device).unsqueeze(1)
            causal = causal // self.compressor.compress_ratio
            scores = scores.masked_fill(blocks >= causal, float("-inf"))
        topk = scores.topk(min(self.index_topk, n_blocks), dim=-1).indices
        if start_pos == 0:
            complete = (
                torch.arange(1, x.shape[1] + 1, device=x.device).unsqueeze(1)
                // self.compressor.compress_ratio
            )
            topk = torch.where(topk >= complete, -1, topk + offset)
        else:
            topk = topk + offset
        return topk.to(dtype=torch.int32)

    __call__ = forward

    def reset(self, *, release: bool = False) -> None:
        self.compressor.reset(release=release)


@dataclass(frozen=True, slots=True)
class AttentionAssembly:
    """Native sparse-attention operands after cache mutation."""

    kv: torch.Tensor
    indices: torch.Tensor
    start_pos: int
    local_cache_entries: int
    compressed_cache_entries: int

    def attend(
        self,
        q: torch.Tensor,
        attn_sink: torch.Tensor,
        scale: float | None = None,
    ) -> torch.Tensor:
        return sparse_attention(q, self.kv, attn_sink, self.indices, scale)


class NativeAttentionState:
    """Assemble official local/compressed KV operands for prefill and decode."""

    def __init__(
        self,
        *,
        max_batch_size: int,
        max_seq_len: int,
        window_size: int,
        head_dim: int,
        compress_ratio: int = 0,
        compressor: DeepSeekV4Compressor | None = None,
        indexer: DeepSeekV4Indexer | None = None,
    ) -> None:
        self.max_batch_size = _positive_int(max_batch_size, "max_batch_size")
        self.max_seq_len = _positive_int(max_seq_len, "max_seq_len")
        self.window_size = _positive_int(window_size, "window_size")
        self.head_dim = _positive_int(head_dim, "head_dim")
        if isinstance(compress_ratio, bool) or not isinstance(compress_ratio, int):
            raise TypeError("compress_ratio must be an integer")
        if compress_ratio < 0:
            raise ValueError("compress_ratio must be non-negative")
        self.compress_ratio = compress_ratio
        self.compressor = compressor
        self.indexer = indexer
        if compress_ratio == 0:
            if compressor is not None or indexer is not None:
                raise ValueError("uncompressed attention cannot own compressor/indexer")
        else:
            if compressor is None or compressor.compress_ratio != compress_ratio:
                raise ValueError("compressed attention requires a matching compressor")
            if compressor.head_dim != head_dim:
                raise ValueError("attention/compressor head dimensions must match")
            if compressor.max_batch_size < max_batch_size:
                raise ValueError(
                    "compressor max_batch_size is smaller than attention state"
                )
            if compressor.max_seq_len < max_seq_len:
                raise ValueError(
                    "compressor max_seq_len is smaller than attention state"
                )
            if compress_ratio == 4 and indexer is None:
                raise ValueError("ratio-4 attention requires the learned indexer")
            if compress_ratio != 4 and indexer is not None:
                raise ValueError("only ratio-4 attention has a learned indexer")
            if indexer is not None and (
                indexer.compressor.max_batch_size < max_batch_size
                or indexer.compressor.max_seq_len < max_seq_len
            ):
                raise ValueError(
                    "indexer compressor limits are smaller than attention state"
                )
        self.local = CircularKVCache(max_batch_size, window_size, head_dim)

    @property
    def next_position(self) -> int:
        return self.local.next_position

    @property
    def state_nbytes(self) -> int:
        total = self.local.nbytes
        if self.compressor is not None:
            total += self.compressor.state_nbytes
        if self.indexer is not None:
            total += self.indexer.compressor.state_nbytes
        return total

    def assemble(
        self,
        kv: torch.Tensor,
        *,
        start_pos: int,
        x: torch.Tensor | None = None,
        qr: torch.Tensor | None = None,
    ) -> AttentionAssembly:
        kv = _floating_3d(kv, "kv")
        _position(start_pos)
        if kv.shape[-1] != self.head_dim:
            raise ValueError("KV feature size does not match attention head_dim")
        end_pos = start_pos + kv.shape[1]
        if end_pos > self.max_seq_len:
            raise ValueError("attention context exceeds configured maximum")

        # Validate all caller-owned operands before mutating either cache.
        branch_x: torch.Tensor | None = None
        branch_qr: torch.Tensor | None = None
        if self.compress_ratio:
            if x is None:
                raise ValueError("compressed attention requires branch input x")
            branch_x = _floating_3d(x, "x")
            if branch_x.shape[:2] != kv.shape[:2]:
                raise ValueError("x and kv batch/sequence dimensions must match")
            if self.indexer is not None:
                if qr is None:
                    raise ValueError(
                        "ratio-4 attention requires normalized q-lora input qr"
                    )
                branch_qr = _floating_3d(qr, "qr")
                if branch_qr.shape[:2] != kv.shape[:2]:
                    raise ValueError("qr and kv batch/sequence dimensions must match")
        batch, seqlen = kv.shape[:2]
        if start_pos == 0:
            if self.next_position != 0:
                raise ValueError("position-zero prefill requires reset()")
        else:
            if seqlen != 1:
                raise ValueError("native attention decode requires one token")
            if start_pos != self.next_position:
                raise ValueError(
                    f"non-contiguous attention decode: expected "
                    f"{self.next_position}, got {start_pos}"
                )

        local_indices = window_indices(
            self.window_size,
            batch,
            seqlen,
            start_pos,
            device=kv.device,
        )
        compressed_count = 0
        compressed_topk: torch.Tensor | None = None
        if self.compress_ratio:
            assert branch_x is not None
            offset = seqlen if start_pos == 0 else self.window_size
            if self.indexer is not None:
                assert branch_qr is not None
                compressed_topk = self.indexer.forward(
                    branch_x, branch_qr, start_pos, offset
                )
            else:
                compressed_topk = compressed_indices(
                    self.compress_ratio,
                    batch,
                    seqlen,
                    start_pos,
                    offset,
                    device=kv.device,
                )

        # Match the published mutation order: learned indexer first, local ring
        # second, and the main compressed cache immediately before attention.
        if start_pos == 0:
            self.local.prefill(kv)
            attention_kv = kv
            local_entries = seqlen
        else:
            self.local.append(kv, start_pos)
            attention_kv = self.local.physical()
            local_entries = self.window_size

        if self.compress_ratio:
            assert branch_x is not None
            assert compressed_topk is not None
            assert self.compressor is not None
            compressed = self.compressor.forward(branch_x, start_pos)
            active = self.compressor.cache.active_like(kv)
            compressed_count = active.shape[1]
            if start_pos == 0:
                # The native prefill attends only to blocks produced by this prefill.
                if compressed is not None:
                    active = compressed
                else:
                    active = kv.new_empty((batch, 0, self.head_dim))
            active = active.to(device=kv.device, dtype=kv.dtype)
            attention_kv = torch.cat((attention_kv, active), dim=1)
            local_indices = torch.cat((local_indices, compressed_topk), dim=-1)

        return AttentionAssembly(
            kv=attention_kv,
            indices=local_indices,
            start_pos=start_pos,
            local_cache_entries=local_entries,
            compressed_cache_entries=compressed_count,
        )

    __call__ = assemble

    def reset(self, *, release: bool = False) -> None:
        self.local.reset(release=release)
        if self.compressor is not None:
            self.compressor.reset(release=release)
        if self.indexer is not None:
            self.indexer.reset(release=release)

    def _snapshot_state(
        self,
        prefix: str,
        tensors: dict[str, SnapshotTensor],
    ) -> dict[str, Any]:
        position = self.next_position
        if position > self.max_seq_len:
            raise ValueError("attention cursor exceeds its context bound")
        compressor = None
        if self.compressor is not None:
            if self.compressor.next_position != position:
                raise ValueError("attention/main compressor cursors are inconsistent")
            compressor = self.compressor._snapshot_state(
                f"{prefix}.compressor", tensors
            )
        indexer = None
        if self.indexer is not None:
            if self.indexer.next_position != position:
                raise ValueError("attention/indexer cursors are inconsistent")
            indexer = self.indexer.compressor._snapshot_state(
                f"{prefix}.indexer_compressor", tensors
            )
        return {
            "kind": "native-attention-state",
            "max_batch_size": self.max_batch_size,
            "max_seq_len": self.max_seq_len,
            "window_size": self.window_size,
            "head_dim": self.head_dim,
            "compress_ratio": self.compress_ratio,
            "next_position": position,
            "local": self.local._snapshot_state(f"{prefix}.local", tensors),
            "compressor": compressor,
            "indexer_compressor": indexer,
        }

    def _restore_snapshot(
        self,
        metadata: Any,
        tensors: Mapping[str, torch.Tensor],
    ) -> None:
        raw = _snapshot_mapping(metadata, "native attention state")
        expected = {
            "kind": "native-attention-state",
            "max_batch_size": self.max_batch_size,
            "max_seq_len": self.max_seq_len,
            "window_size": self.window_size,
            "head_dim": self.head_dim,
            "compress_ratio": self.compress_ratio,
        }
        if any(raw.get(key) != value for key, value in expected.items()):
            raise ValueError("attention-state structure does not match the runtime")
        position = _snapshot_int(
            raw.get("next_position"),
            "attention next_position",
            maximum=self.max_seq_len,
        )
        self.local._restore_snapshot(raw.get("local"), tensors)
        if self.local.next_position != position:
            raise ValueError("attention/local cache cursors are inconsistent")
        compressor_meta = raw.get("compressor")
        if self.compressor is None:
            if compressor_meta is not None:
                raise ValueError("snapshot has an unexpected main compressor")
        else:
            if compressor_meta is None:
                raise ValueError("snapshot omits the main compressor")
            self.compressor._restore_snapshot(compressor_meta, tensors)
            if self.compressor.next_position != position:
                raise ValueError("attention/main compressor cursors are inconsistent")
        indexer_meta = raw.get("indexer_compressor")
        if self.indexer is None:
            if indexer_meta is not None:
                raise ValueError("snapshot has an unexpected indexer compressor")
        else:
            if indexer_meta is None:
                raise ValueError("snapshot omits the indexer compressor")
            self.indexer.compressor._restore_snapshot(indexer_meta, tensors)
            if self.indexer.next_position != position:
                raise ValueError("attention/indexer cursors are inconsistent")


__all__ = [
    "AttentionAssembly",
    "CircularKVCache",
    "CompressedKVCache",
    "DeepSeekV4Compressor",
    "DeepSeekV4Indexer",
    "LinearCallback",
    "NativeAttentionState",
    "QATCallback",
    "RMSCallback",
]
