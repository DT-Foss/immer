"""Persistent head-routed DeltaNet output projection on a packed Q4 bank."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
import threading
from typing import Mapping, Sequence

import numpy as np
import torch

from ..ooe.identity import canonical_json_bytes
from ..ooe.math_core import sinkhorn_project
from .q4 import Q4Bank, Q4_BLOCK_SIZE


PACKED_DELTA_HEAD_ROUTER_SCHEMA = "immer.qwen3.8-packed-delta-head-router/v1"
PACKED_DELTA_HEAD_STATE_SCHEMA = "immer.qwen3.8-packed-delta-head-markov/v1"

_ROUTE_POLICY = "mean-square+sinkhorn-first-order/v1"
_STATE_MODE = "delta-head-out-proj"
_TRANSITION_PRIOR = 1e-3
_MARKOV_WEIGHT = 0.10
_OVERLAP_DECAY = 0.80
_COUNT_RENORMALIZE_AT = 1 << 52
_SINKHORN_REFRESH_INTERVAL = 8


class PackedDeltaHeadRouterError(RuntimeError):
    """The packed DeltaNet head route violates its execution contract."""


class PackedDeltaHeadRouterStateError(PackedDeltaHeadRouterError):
    """Persistent packed DeltaNet routing state is corrupt or incompatible."""


@dataclass(slots=True)
class _LayerState:
    counts: np.ndarray
    previous: tuple[int, ...] = ()
    overlap_ema: float = 0.0
    transitions: int = 0

    def clone(self) -> _LayerState:
        return _LayerState(
            counts=self.counts.copy(),
            previous=self.previous,
            overlap_ema=self.overlap_ema,
            transitions=self.transitions,
        )


class PackedDeltaHeadRouter:
    """Select DeltaNet value heads and project only their packed Q4 columns.

    ``mixed`` is the already gated-and-normalized DeltaNet output.  Routing is
    therefore outside the exact convolution and recurrent-state computation;
    omitted heads are equivalent to zeroing only their input columns in the
    final ``out_proj`` matrix.
    """

    def __init__(
        self,
        bank: Q4Bank,
        *,
        active_layers: Sequence[int],
        state_path: str | Path | None = None,
        value_heads: int = 48,
        head_dim: int = 128,
        max_selected_heads: int = 32,
    ) -> None:
        if not callable(getattr(bank, "linear_selected_blocks", None)):
            raise TypeError("bank must provide linear_selected_blocks")
        bank_identity = getattr(bank, "identity", None)
        if not isinstance(bank_identity, Mapping):
            raise TypeError("bank identity must be a mapping")
        manifest_sha256 = bank_identity.get("manifest_sha256")
        if not isinstance(manifest_sha256, str) or not manifest_sha256:
            raise ValueError("bank identity has no Q4 manifest digest")

        layers = self._layers(active_layers)
        heads = self._positive_int(value_heads, "value_heads")
        dimension = self._positive_int(head_dim, "head_dim")
        maximum = self._positive_int(max_selected_heads, "max_selected_heads")
        if dimension % Q4_BLOCK_SIZE:
            raise ValueError("head_dim must be divisible by the Q4 block size")
        if maximum > heads:
            raise ValueError("max_selected_heads cannot exceed value_heads")

        actions = tuple(sorted({min(width, maximum) for width in (24, 32, 40)}))
        self.bank = bank
        self.value_heads = heads
        self.head_dim = dimension
        self.max_selected_heads = maximum
        self.blocks_per_head = dimension // Q4_BLOCK_SIZE
        self.width_actions = actions
        self._active_layers = frozenset(layers)
        self._identity = {
            "active_layers": list(layers),
            "head_dim": dimension,
            "max_selected_heads": maximum,
            "mode": _STATE_MODE,
            "q4_manifest_sha256": manifest_sha256,
            "value_heads": heads,
        }
        self.requested_state_path = (
            None
            if state_path is None
            else Path(state_path).expanduser().absolute()
        )
        self.state_path = self._namespaced_path(self.requested_state_path)
        self._states = {
            layer: _LayerState(
                counts=np.zeros((heads, heads), dtype=np.int64),
            )
            for layer in layers
        }
        self._counters = {
            "calls": 0,
            "full_equivalent_bytes": 0,
            "logical_bytes_saved": 0,
            "rows": 0,
            "selected_blocks": 0,
            "selected_heads": 0,
            "sinkhorn_projections": 0,
            "transitions": 0,
        }
        self._width_counts = {width: 0 for width in actions}
        self._sinkhorn_cache: dict[int, np.ndarray] = {}
        self._lock = threading.RLock()
        self._dirty = False
        self._closed = False
        self._load()

    @staticmethod
    def _positive_int(value: int, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 1:
            raise ValueError(f"{name} must be positive")
        return value

    @staticmethod
    def _layers(values: Sequence[int]) -> tuple[int, ...]:
        if isinstance(values, (str, bytes)):
            raise TypeError("active_layers must be a sequence of integers")
        try:
            rows = tuple(values)
        except TypeError as exc:
            raise TypeError("active_layers must be a sequence of integers") from exc
        if not rows:
            raise ValueError("active_layers must not be empty")
        if any(
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or layer < 0
            for layer in rows
        ):
            raise ValueError("active_layers contains an invalid layer")
        if len(set(rows)) != len(rows):
            raise ValueError("active_layers contains a duplicate")
        return tuple(sorted(rows))

    @property
    def active_layers(self) -> tuple[int, ...]:
        return tuple(sorted(self._active_layers))

    def supports_layer(self, layer: int) -> bool:
        return (
            not isinstance(layer, bool)
            and isinstance(layer, int)
            and layer in self._active_layers
        )

    def _namespaced_path(self, requested: Path | None) -> Path | None:
        if requested is None:
            return None
        namespace = hashlib.sha256(
            canonical_json_bytes(self._identity)
        ).hexdigest()[:16]
        suffix = requested.suffix or ".json"
        return requested.with_name(
            f"{requested.stem}.delta-head-{namespace}{suffix}"
        )

    def snapshot_identity(
        self, *, transport_neutral: bool = False
    ) -> dict[str, object]:
        identity: dict[str, object] = {
            "head_dim": self.head_dim,
            "layers": list(self.active_layers),
            "max_selected_heads": self.max_selected_heads,
            "q4_manifest_sha256": self._identity["q4_manifest_sha256"],
            "route_policy": _ROUTE_POLICY,
            "schema": PACKED_DELTA_HEAD_ROUTER_SCHEMA,
            "state_persistent": self.state_path is not None,
            "value_heads": self.value_heads,
            "width_actions": list(self.width_actions),
        }
        if not transport_neutral:
            root = getattr(self.bank, "root", None)
            if root is not None:
                identity["bank_root"] = str(root)
            if self.state_path is not None:
                identity["state_path"] = str(self.state_path)
        return identity

    @staticmethod
    def _entry_value(entry: object, field: str) -> object:
        if isinstance(entry, Mapping):
            try:
                return entry[field]
            except KeyError:
                raise PackedDeltaHeadRouterError(
                    f"Q4 entry has no {field}"
                ) from None
        try:
            return getattr(entry, field)
        except AttributeError:
            raise PackedDeltaHeadRouterError(f"Q4 entry has no {field}") from None

    def _entry(self, name: str) -> tuple[int, int, int]:
        if not isinstance(name, str) or not name:
            raise ValueError("Q4 tensor name must be non-empty")
        entries = getattr(self.bank, "entries", None)
        if not isinstance(entries, Mapping):
            raise PackedDeltaHeadRouterError("Q4 bank has no tensor inventory")
        try:
            entry = entries[name]
        except KeyError:
            raise KeyError(name) from None
        raw_shape = self._entry_value(entry, "shape")
        try:
            shape = tuple(raw_shape)  # type: ignore[arg-type]
        except TypeError as exc:
            raise PackedDeltaHeadRouterError("Q4 entry shape is invalid") from exc
        if (
            len(shape) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in shape
            )
            or shape[1] != self.value_heads * self.head_dim
        ):
            raise PackedDeltaHeadRouterError(
                "DeltaNet out_proj shape differs from the head router"
            )
        payload_bytes = self._entry_value(entry, "payload_bytes")
        if (
            isinstance(payload_bytes, bool)
            or not isinstance(payload_bytes, int)
            or payload_bytes < 1
        ):
            raise PackedDeltaHeadRouterError("Q4 entry payload size is invalid")
        return shape[0], shape[1], payload_bytes

    def _validate_mixed(self, mixed: object) -> torch.Tensor:
        if not isinstance(mixed, torch.Tensor):
            raise TypeError("mixed must be a torch.Tensor")
        if (
            mixed.ndim != 3
            or mixed.shape[0] < 1
            or mixed.shape[1] < 1
            or mixed.shape[2] != self.value_heads * self.head_dim
        ):
            raise ValueError("mixed differs from [B, S, value_heads * head_dim]")
        if mixed.device.type != "cpu":
            raise ValueError("packed DeltaNet head routing is CPU-only")
        if not mixed.is_floating_point():
            raise TypeError("mixed must use a floating-point dtype")
        if not bool(torch.isfinite(mixed).all()):
            raise ValueError("mixed must contain only finite values")
        return mixed

    def _choose_width(self, state: _LayerState) -> int:
        if state.transitions < 1:
            return self.width_actions[-1]
        if state.overlap_ema >= 0.38:
            return self.width_actions[0]
        if state.overlap_ema >= 0.22:
            return self.width_actions[min(1, len(self.width_actions) - 1)]
        return self.width_actions[-1]

    def _normalized_energy(self, energy: np.ndarray) -> np.ndarray:
        peak = float(np.max(energy, initial=0.0))
        if peak <= np.finfo(np.float64).tiny:
            return np.full(self.value_heads, 1.0 / self.value_heads)
        scaled = energy / peak
        total = float(math.fsum(float(value) for value in scaled))
        return np.ascontiguousarray(scaled / total, dtype=np.float64)

    def _select(
        self,
        energy: np.ndarray,
        *,
        layer: int,
        width: int,
        state: _LayerState,
    ) -> tuple[tuple[int, ...], bool]:
        kernel = self._sinkhorn_cache.get(layer)
        refresh = kernel is None or (
            state.transitions > 0
            and state.transitions % _SINKHORN_REFRESH_INTERVAL == 0
        )
        if refresh:
            transition_weights = state.counts.astype(np.float64)
            transition_weights += _TRANSITION_PRIOR
            kernel = sinkhorn_project(
                transition_weights,
                rounds=3,
                max_nodes=self.value_heads,
            )
            self._sinkhorn_cache[layer] = kernel
        assert kernel is not None
        live = self._normalized_energy(energy)
        if state.previous:
            previous = np.asarray(state.previous, dtype=np.int64)
            prior = kernel[previous].mean(axis=0)
            prior_total = float(prior.sum())
            if prior_total > 0.0:
                prior = prior / prior_total
            score = (1.0 - _MARKOV_WEIGHT) * live + _MARKOV_WEIGHT * prior
        else:
            score = live
        ordered = sorted(
            range(self.value_heads),
            key=lambda head: (-float(score[head]), head),
        )
        return tuple(ordered[:width]), refresh

    @staticmethod
    def _overlap(left: tuple[int, ...], right: tuple[int, ...]) -> float:
        before = set(left)
        after = set(right)
        return len(before & after) / len(before | after)

    def _observe(self, state: _LayerState, selected: tuple[int, ...]) -> None:
        previous = state.previous
        if previous:
            overlap = self._overlap(previous, selected)
            if state.transitions == 0:
                state.overlap_ema = overlap
            else:
                state.overlap_ema = (
                    _OVERLAP_DECAY * state.overlap_ema
                    + (1.0 - _OVERLAP_DECAY) * overlap
                )
            if int(state.counts.max(initial=0)) >= _COUNT_RENORMALIZE_AT:
                state.counts //= 2
            state.counts[np.ix_(previous, selected)] += 1
            state.transitions += 1
        state.previous = selected

    def _routes(
        self,
        flat: torch.Tensor,
        *,
        layer: int,
        width: int,
        state: _LayerState,
    ) -> tuple[tuple[tuple[int, ...], ...], int]:
        rows = flat.detach().to(dtype=torch.float64).reshape(
            flat.shape[0], self.value_heads, self.head_dim
        )
        scale = rows.abs().amax(dim=(1, 2), keepdim=True)
        scale = torch.where(scale > 0.0, scale, torch.ones_like(scale))
        energies = (rows / scale).square().mean(dim=2).numpy()
        selected: list[tuple[int, ...]] = []
        sinkhorn_refreshes = 0
        for energy in energies:
            route, refreshed = self._select(
                energy,
                layer=layer,
                width=width,
                state=state,
            )
            sinkhorn_refreshes += int(refreshed)
            self._observe(state, route)
            selected.append(route)
        return tuple(selected), sinkhorn_refreshes

    def _block_ids(self, routes: tuple[tuple[int, ...], ...]) -> torch.Tensor:
        return torch.tensor(
            [
                [
                    head * self.blocks_per_head + offset
                    for head in route
                    for offset in range(self.blocks_per_head)
                ]
                for route in routes
            ],
            dtype=torch.int64,
        )

    @staticmethod
    def _gather_blocks(flat: torch.Tensor, block_ids: torch.Tensor) -> torch.Tensor:
        all_blocks = flat.reshape(flat.shape[0], -1, Q4_BLOCK_SIZE)
        gather_ids = block_ids.unsqueeze(-1).expand(-1, -1, Q4_BLOCK_SIZE)
        return torch.gather(all_blocks, 1, gather_ids).contiguous()

    def _project(
        self,
        mixed: tuple[torch.Tensor, ...],
        name: str,
        *,
        layer: int,
    ) -> tuple[torch.Tensor, ...]:
        if not self.supports_layer(layer):
            raise KeyError(f"no packed DeltaNet head route for layer {layer}")
        if not mixed:
            raise ValueError("project_many requires at least one mixed tensor")
        tensors = tuple(self._validate_mixed(value) for value in mixed)
        dtypes = {value.dtype for value in tensors}
        if len(dtypes) != 1:
            raise ValueError("project_many mixed tensors must share one dtype")
        output_columns, _, payload_bytes = self._entry(name)
        shapes = tuple(tuple(value.shape) for value in tensors)
        row_counts = tuple(value.numel() // value.shape[-1] for value in tensors)
        flat = torch.cat(
            tuple(value.detach().reshape(rows, -1) for value, rows in zip(
                tensors, row_counts, strict=True
            )),
            dim=0,
        ).contiguous()

        with self._lock:
            if self._closed:
                raise PackedDeltaHeadRouterError("packed DeltaNet head router is closed")
            working = self._states[layer].clone()
            width = self._choose_width(working)
            routes, sinkhorn_refreshes = self._routes(
                flat,
                layer=layer,
                width=width,
                state=working,
            )
            block_ids = self._block_ids(routes)
            block_values = self._gather_blocks(flat, block_ids)
            output = self.bank.linear_selected_blocks(
                block_values,
                block_ids,
                name,
                output_dtype=flat.dtype,
            )
            if (
                not isinstance(output, torch.Tensor)
                or output.device.type != "cpu"
                or output.shape != (flat.shape[0], output_columns)
            ):
                raise PackedDeltaHeadRouterError(
                    "Q4 selected-block projection returned an invalid output"
                )
            output = output.to(dtype=flat.dtype)

            total_rows = flat.shape[0]
            selected_bytes = (
                total_rows * payload_bytes * width // self.value_heads
            )
            full_bytes = total_rows * payload_bytes
            transition_delta = working.transitions - self._states[layer].transitions
            self._states[layer] = working
            self._counters["calls"] += 1
            self._counters["rows"] += total_rows
            self._counters["full_equivalent_bytes"] += full_bytes
            self._counters["logical_bytes_saved"] += max(
                0, full_bytes - selected_bytes
            )
            self._counters["selected_heads"] += total_rows * width
            self._counters["selected_blocks"] += (
                total_rows * width * self.blocks_per_head
            )
            self._counters["sinkhorn_projections"] += sinkhorn_refreshes
            self._counters["transitions"] += transition_delta
            self._width_counts[width] += 1
            self._dirty = True

            chunks = output.split(row_counts, dim=0)
            return tuple(
                chunk.reshape(*shape[:-1], output_columns).to(dtype=value.dtype)
                for chunk, shape, value in zip(
                    chunks, shapes, tensors, strict=True
                )
            )

    def project(
        self,
        mixed: torch.Tensor,
        name: str,
        *,
        layer: int,
    ) -> torch.Tensor:
        """Route and project one ``[B, S, H*D]`` DeltaNet output tensor."""

        return self._project((mixed,), name, layer=layer)[0]

    def project_many(
        self,
        mixed: tuple[torch.Tensor, ...],
        name: str,
        *,
        layer: int,
    ) -> tuple[torch.Tensor, ...]:
        """Route independent DeltaNet tensors with one packed Q4 projection."""

        if not isinstance(mixed, tuple):
            raise TypeError("project_many mixed values must be a tuple")
        return self._project(mixed, name, layer=layer)

    def _state_document(self) -> dict[str, object]:
        return {
            "counters": dict(self._counters),
            "identity": dict(self._identity),
            "layers": {
                str(layer): {
                    "counts": state.counts.tolist(),
                    "overlap_ema": state.overlap_ema,
                    "previous": list(state.previous),
                    "transitions": state.transitions,
                }
                for layer, state in sorted(self._states.items())
            },
            "schema": PACKED_DELTA_HEAD_STATE_SCHEMA,
            "width_counts": {
                str(width): count
                for width, count in sorted(self._width_counts.items())
            },
        }

    def _load(self) -> None:
        path = self.state_path
        if path is None or (not path.exists() and not path.is_symlink()):
            return
        try:
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                raise PackedDeltaHeadRouterStateError(
                    "packed DeltaNet routing state is not a regular file"
                )
            raw = path.read_bytes()
            document = json.loads(raw.decode("utf-8"))
        except PackedDeltaHeadRouterStateError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PackedDeltaHeadRouterStateError(
                "packed DeltaNet routing state is unreadable"
            ) from exc
        if (
            not isinstance(document, dict)
            or set(document)
            != {"counters", "identity", "layers", "schema", "width_counts"}
            or document.get("schema") != PACKED_DELTA_HEAD_STATE_SCHEMA
            or document.get("identity") != self._identity
            or canonical_json_bytes(document) != raw
        ):
            raise PackedDeltaHeadRouterStateError(
                "packed DeltaNet routing state identity changed"
            )
        layers = document["layers"]
        if not isinstance(layers, dict) or set(layers) != {
            str(layer) for layer in self.active_layers
        }:
            raise PackedDeltaHeadRouterStateError(
                "packed DeltaNet routing layers are invalid"
            )

        restored: dict[int, _LayerState] = {}
        for layer in self.active_layers:
            row = layers[str(layer)]
            if not isinstance(row, dict) or set(row) != {
                "counts",
                "overlap_ema",
                "previous",
                "transitions",
            }:
                raise PackedDeltaHeadRouterStateError(
                    "packed DeltaNet routing layer state is invalid"
                )
            counts = row["counts"]
            if (
                not isinstance(counts, list)
                or len(counts) != self.value_heads
                or any(
                    not isinstance(values, list)
                    or len(values) != self.value_heads
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                        or value > np.iinfo(np.int64).max
                        for value in values
                    )
                    for values in counts
                )
            ):
                raise PackedDeltaHeadRouterStateError(
                    "packed DeltaNet transition counts are invalid"
                )
            previous = row["previous"]
            overlap_ema = row["overlap_ema"]
            transitions = row["transitions"]
            if (
                not isinstance(previous, list)
                or len(previous) > self.max_selected_heads
                or any(
                    isinstance(head, bool)
                    or not isinstance(head, int)
                    or not 0 <= head < self.value_heads
                    for head in previous
                )
                or len(set(previous)) != len(previous)
                or isinstance(overlap_ema, bool)
                or not isinstance(overlap_ema, (int, float))
                or not math.isfinite(float(overlap_ema))
                or not 0.0 <= float(overlap_ema) <= 1.0
                or isinstance(transitions, bool)
                or not isinstance(transitions, int)
                or transitions < 0
            ):
                raise PackedDeltaHeadRouterStateError(
                    "packed DeltaNet routing layer values are invalid"
                )
            restored[layer] = _LayerState(
                counts=np.asarray(counts, dtype=np.int64),
                previous=tuple(previous),
                overlap_ema=float(overlap_ema),
                transitions=transitions,
            )

        counters = document["counters"]
        if (
            not isinstance(counters, dict)
            or set(counters) != set(self._counters)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in counters.values()
            )
        ):
            raise PackedDeltaHeadRouterStateError(
                "packed DeltaNet routing counters are invalid"
            )
        width_counts = document["width_counts"]
        if (
            not isinstance(width_counts, dict)
            or set(width_counts) != {str(width) for width in self.width_actions}
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in width_counts.values()
            )
        ):
            raise PackedDeltaHeadRouterStateError(
                "packed DeltaNet width counters are invalid"
            )
        if counters["transitions"] != sum(
            state.transitions for state in restored.values()
        ):
            raise PackedDeltaHeadRouterStateError(
                "packed DeltaNet transition totals disagree"
            )
        self._states = restored
        self._counters = {key: int(value) for key, value in counters.items()}
        self._width_counts = {
            width: int(width_counts[str(width)]) for width in self.width_actions
        }

    def _flush_locked(self) -> None:
        path = self.state_path
        if path is None or not self._dirty:
            return
        data = canonical_json_bytes(self._state_document())
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
        descriptor: int | None = None
        try:
            descriptor = os.open(
                temporary,
                os.O_CREAT
                | os.O_EXCL
                | os.O_WRONLY
                | int(getattr(os, "O_CLOEXEC", 0))
                | int(getattr(os, "O_NOFOLLOW", 0)),
                0o600,
            )
            view = memoryview(data)
            offset = 0
            while offset < len(view):
                written = os.write(descriptor, view[offset:])
                if written <= 0:
                    raise OSError("short packed DeltaNet routing state write")
                offset += written
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.replace(temporary, path)
            self._dirty = False
        finally:
            if descriptor is not None:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)

    def metrics(self) -> dict[str, int]:
        """Return cumulative counters after durably flushing controller state."""

        with self._lock:
            self._flush_locked()
            return {
                **self._counters,
                **{
                    f"width_{width}": count
                    for width, count in sorted(self._width_counts.items())
                },
            }

    def close(self) -> None:
        """Flush routing state without closing the shared Q4 bank."""

        with self._lock:
            if self._closed:
                return
            self._flush_locked()
            self._closed = True

    def __enter__(self) -> PackedDeltaHeadRouter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "PACKED_DELTA_HEAD_ROUTER_SCHEMA",
    "PACKED_DELTA_HEAD_STATE_SCHEMA",
    "PackedDeltaHeadRouter",
    "PackedDeltaHeadRouterError",
    "PackedDeltaHeadRouterStateError",
]
