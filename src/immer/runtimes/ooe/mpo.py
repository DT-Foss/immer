"""Bounded tensor-train/MPO storage for action-conditioned world models.

The action axis is represented by a leading tensor-train core and every
factored state coordinate by a genuine matrix-product-operator core with
``(left_rank, input_dimension, output_dimension, right_rank)`` layout.  A
factorization is selected only when it meets the requested reconstruction
tolerance *and* is smaller than the dense tensor.  Otherwise the exact dense
tensor is retained and the failed low-rank attempt remains visible in the
receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import json
import math
import re
import struct
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .identity import canonical_json_bytes, require_sha256
from .math_core import array_sha256


FloatArray = NDArray[np.float64]

MAX_MPO_ACTIONS = 256
MAX_MPO_STATES = 65_536
MAX_MPO_SITES = 16
MAX_MPO_RANK = 1_024
MAX_MPO_PAYLOAD_BYTES = 256 * 1024 * 1024
MAX_MPO_WORK_BYTES = 768 * 1024 * 1024

_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}")


class MPOIntegrityError(ValueError):
    """Raised when an MPO payload does not satisfy its canonical receipt."""


def _integer(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        upper = "" if maximum is None else f" and at most {maximum}"
        raise ValueError(f"{name} must be at least {minimum}{upper}")
    return result


def _identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical identifier")
    return value


def _hash_items(
    value: Mapping[str, str] | Iterable[tuple[str, str]],
    *,
    name: str,
) -> tuple[tuple[str, str], ...]:
    items = value.items() if isinstance(value, Mapping) else value
    result = tuple(
        sorted(
            (
                _identifier(key, name=f"{name} name"),
                require_sha256(digest, field=f"{name}[{key!r}]"),
            )
            for key, digest in items
        )
    )
    if not result or len({key for key, _ in result}) != len(result):
        raise ValueError(f"{name} must be non-empty and uniquely named")
    return result


def _float_from_hex(value: object, *, name: str) -> float:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a hexadecimal float")
    try:
        result = float.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a hexadecimal float") from exc
    if not math.isfinite(result) or result < 0.0 or result.hex() != value:
        raise ValueError(f"{name} must be canonical, finite and non-negative")
    return result


def _body_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(value))).hexdigest()


def _decode_canonical(data: bytes) -> Any:
    if not isinstance(data, bytes) or len(data) > 2 * MAX_MPO_PAYLOAD_BYTES:
        raise MPOIntegrityError("MPO payload is not bounded immutable bytes")
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MPOIntegrityError("MPO payload is not valid JSON") from exc
    if canonical_json_bytes(value) != data:
        raise MPOIntegrityError("MPO payload JSON is not canonical")
    return value


def _state_shape(value: Sequence[int], *, states: int) -> tuple[int, ...]:
    shape = tuple(
        _integer(
            dimension,
            name="state dimension",
            minimum=2,
            maximum=MAX_MPO_STATES,
        )
        for dimension in value
    )
    if not 1 <= len(shape) <= MAX_MPO_SITES:
        raise ValueError(f"state_shape length must lie in [1, {MAX_MPO_SITES}]")
    if math.prod(shape) != states:
        raise ValueError("state_shape product must equal the transition state count")
    return shape


def _action_tensor(value: ArrayLike) -> FloatArray:
    raw = np.asarray(value)
    if raw.ndim != 3 or raw.shape[0] < 1 or raw.shape[1] < 2:
        raise ValueError("transition tensor must have shape (actions, states, states)")
    if raw.shape[1] != raw.shape[2]:
        raise ValueError("transition matrices must be square")
    actions, states, _ = (int(item) for item in raw.shape)
    if actions > MAX_MPO_ACTIONS or states > MAX_MPO_STATES:
        raise ValueError("transition tensor exceeds action or state bounds")
    byte_count = raw.size * np.dtype(np.float64).itemsize
    if byte_count > MAX_MPO_PAYLOAD_BYTES:
        raise ValueError("transition tensor exceeds MAX_MPO_PAYLOAD_BYTES")
    try:
        tensor = np.asarray(raw, dtype=np.float64, order="C")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("transition tensor must contain real numbers") from exc
    if not np.isfinite(tensor).all() or np.any(tensor < 0.0):
        raise ValueError("transition tensor must be finite and non-negative")
    if not np.allclose(tensor.sum(axis=2), 1.0, atol=1e-12, rtol=1e-12):
        raise ValueError("every action transition must be row-stochastic")
    return tensor


def _little_f8_bytes(value: FloatArray) -> bytes:
    return np.asarray(value, dtype="<f8", order="C").tobytes(order="C")


def _array_from_bytes(data: bytes, shape: Sequence[int]) -> FloatArray:
    dimensions = tuple(int(value) for value in shape)
    required = math.prod(dimensions) * np.dtype("<f8").itemsize
    if required > MAX_MPO_PAYLOAD_BYTES or len(data) != required:
        raise MPOIntegrityError("MPO numeric payload length does not match its shape")
    return np.frombuffer(data, dtype="<f8").reshape(dimensions).copy()


def _factor_payload_sha256(
    *,
    action_core_shape: tuple[int, int, int] | None,
    action_core_bytes: bytes | None,
    site_core_shapes: tuple[tuple[int, int, int, int], ...],
    site_core_bytes: tuple[bytes, ...],
    dense_fallback_bytes: bytes | None,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"immer-ooe-mpo-factor-payload/v1\0")
    if dense_fallback_bytes is not None:
        digest.update(b"dense\0")
        digest.update(dense_fallback_bytes)
        return digest.hexdigest()
    if action_core_shape is None or action_core_bytes is None:
        raise MPOIntegrityError("compressed MPO is missing its action core")
    digest.update(b"mpo\0")
    for dimension in action_core_shape:
        digest.update(struct.pack("<Q", dimension))
    digest.update(action_core_bytes)
    for shape, data in zip(site_core_shapes, site_core_bytes, strict=True):
        for dimension in shape:
            digest.update(struct.pack("<Q", dimension))
        digest.update(data)
    return digest.hexdigest()


def _physical_tensor(tensor: FloatArray, shape: tuple[int, ...]) -> FloatArray:
    sites = len(shape)
    expanded = tensor.reshape((tensor.shape[0], *shape, *shape))
    axes = [0]
    for index in range(sites):
        axes.extend((1 + index, 1 + sites + index))
    interleaved = expanded.transpose(axes)
    return np.ascontiguousarray(
        interleaved.reshape((tensor.shape[0], *(value * value for value in shape)))
    )


def _canonicalize_svd_signs(u: FloatArray, vh: FloatArray) -> None:
    for column in range(u.shape[1]):
        pivot = int(np.argmax(np.abs(u[:, column])))
        if u[pivot, column] < 0.0:
            u[:, column] *= -1.0
            vh[column, :] *= -1.0


def _tt_svd(
    physical: FloatArray,
    *,
    maximum_rank: int,
    relative_tolerance: float,
    max_work_bytes: int,
) -> tuple[tuple[FloatArray, ...], tuple[int, ...]]:
    modes = tuple(int(value) for value in physical.shape)
    if len(modes) < 2:
        raise ValueError("MPO needs an action mode and at least one state site")
    norm = float(np.linalg.norm(physical))
    absolute_budget = relative_tolerance * norm
    cut_budget = absolute_budget / math.sqrt(len(modes) - 1)
    unfolding = physical.copy()
    left_rank = 1
    cores: list[FloatArray] = []
    bond_ranks = [1]
    for index, mode in enumerate(modes[:-1]):
        matrix = unfolding.reshape(left_rank * mode, -1)
        short = min(matrix.shape)
        estimated = (
            matrix.size + matrix.shape[0] * short + short + short * matrix.shape[1]
        ) * np.dtype(np.float64).itemsize
        if estimated > max_work_bytes:
            raise ValueError(
                "TT-SVD working set exceeds the configured max_work_bytes"
            )
        u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        _canonicalize_svd_signs(u, vh)
        possible = singular.size
        if relative_tolerance == 0.0:
            required_rank = possible
        else:
            tail_squared = np.concatenate(
                (np.cumsum(np.square(singular[::-1]))[::-1], np.zeros(1))
            )
            required_rank = possible
            for candidate in range(1, possible + 1):
                if math.sqrt(float(tail_squared[candidate])) <= cut_budget:
                    required_rank = candidate
                    break
        rank = min(required_rank, maximum_rank)
        core = np.ascontiguousarray(
            u[:, :rank].reshape(left_rank, mode, rank), dtype=np.float64
        )
        cores.append(core)
        unfolding = np.ascontiguousarray(
            singular[:rank, None] * vh[:rank, :], dtype=np.float64
        )
        remaining = modes[index + 1 :]
        unfolding = unfolding.reshape((rank, *remaining))
        left_rank = rank
        bond_ranks.append(rank)
    cores.append(
        np.ascontiguousarray(unfolding.reshape(left_rank, modes[-1], 1))
    )
    bond_ranks.append(1)
    return tuple(cores), tuple(bond_ranks)


def _reconstruct_tt(cores: Sequence[FloatArray]) -> FloatArray:
    current = cores[0]
    for core in cores[1:]:
        current = np.tensordot(current, core, axes=([-1], [0]))
    return np.ascontiguousarray(np.squeeze(current, axis=(0, -1)), dtype=np.float64)


def _physical_to_action_tensor(
    physical: FloatArray,
    *,
    action_count: int,
    state_shape: tuple[int, ...],
) -> FloatArray:
    interleaved_shape: list[int] = [action_count]
    for dimension in state_shape:
        interleaved_shape.extend((dimension, dimension))
    interleaved = physical.reshape(tuple(interleaved_shape))
    sites = len(state_shape)
    axes = [0]
    axes.extend(1 + 2 * index for index in range(sites))
    axes.extend(2 + 2 * index for index in range(sites))
    expanded = interleaved.transpose(axes)
    states = math.prod(state_shape)
    return np.ascontiguousarray(expanded.reshape(action_count, states, states))


@dataclass(frozen=True, slots=True)
class MPOReceipt:
    action_ids: tuple[str, ...]
    state_shape: tuple[int, ...]
    source_world_model_sha256: str
    graph_revision_sha256: str
    verifier_hashes: tuple[tuple[str, str], ...]
    source_tensor_sha256: str
    mode: str
    selection_reason: str
    requested_max_rank: int
    attempted_bond_ranks: tuple[int, ...]
    relative_tolerance_hex: str
    attempted_absolute_error_hex: str
    attempted_relative_error_hex: str
    attempted_mpo_met_tolerance: bool
    reconstruction_absolute_error_hex: str
    reconstruction_relative_error_hex: str
    reconstruction_sha256: str
    factor_payload_sha256: str
    dense_numeric_bytes: int
    stored_numeric_bytes: int
    compression_ratio_hex: str

    FORMAT = "immer-ooe-action-conditioned-mpo-receipt/v1"

    def __post_init__(self) -> None:
        actions = tuple(_identifier(value, name="action_id") for value in self.action_ids)
        if not 1 <= len(actions) <= MAX_MPO_ACTIONS or len(set(actions)) != len(actions):
            raise ValueError("action_ids must be bounded and unique")
        states = math.prod(self.state_shape)
        shape = _state_shape(self.state_shape, states=states)
        if states > MAX_MPO_STATES:
            raise ValueError("state_shape exceeds MAX_MPO_STATES")
        if self.mode not in {"mpo", "dense_exact_fallback"}:
            raise ValueError("unsupported MPO receipt mode")
        if self.selection_reason not in {
            "tolerance_met_and_compact",
            "rank_budget_exceeded",
            "mpo_not_compact",
        }:
            raise ValueError("unsupported MPO selection reason")
        rank = _integer(
            self.requested_max_rank,
            name="requested_max_rank",
            minimum=1,
            maximum=MAX_MPO_RANK,
        )
        ranks = tuple(
            _integer(value, name="bond rank", minimum=1, maximum=MAX_MPO_RANK)
            for value in self.attempted_bond_ranks
        )
        if len(ranks) != len(shape) + 2 or ranks[0] != 1 or ranks[-1] != 1:
            raise ValueError("attempted_bond_ranks do not match the MPO topology")
        if any(value > rank for value in ranks[1:-1]):
            raise ValueError("attempted bond rank exceeds requested_max_rank")
        if not isinstance(self.attempted_mpo_met_tolerance, bool):
            raise TypeError("attempted_mpo_met_tolerance must be a boolean")
        for field in (
            "relative_tolerance_hex",
            "attempted_absolute_error_hex",
            "attempted_relative_error_hex",
            "reconstruction_absolute_error_hex",
            "reconstruction_relative_error_hex",
            "compression_ratio_hex",
        ):
            _float_from_hex(getattr(self, field), name=field)
        dense_bytes = _integer(
            self.dense_numeric_bytes,
            name="dense_numeric_bytes",
            minimum=1,
            maximum=MAX_MPO_PAYLOAD_BYTES,
        )
        stored_bytes = _integer(
            self.stored_numeric_bytes,
            name="stored_numeric_bytes",
            minimum=1,
            maximum=MAX_MPO_PAYLOAD_BYTES,
        )
        expected_dense_bytes = (
            len(actions) * states * states * np.dtype(np.float64).itemsize
        )
        if dense_bytes != expected_dense_bytes:
            raise ValueError("dense byte count does not match receipt dimensions")
        ratio = _float_from_hex(self.compression_ratio_hex, name="compression_ratio_hex")
        if not math.isclose(
            ratio,
            stored_bytes / dense_bytes,
            rel_tol=1e-15,
            abs_tol=0.0,
        ):
            raise ValueError("compression ratio does not match payload sizes")
        if self.mode == "mpo":
            if self.selection_reason != "tolerance_met_and_compact":
                raise ValueError("compressed MPO must record its successful selection")
            if not self.attempted_mpo_met_tolerance or stored_bytes >= dense_bytes:
                raise ValueError("compressed MPO must meet tolerance and be compact")
            if _float_from_hex(
                self.reconstruction_relative_error_hex,
                name="reconstruction_relative_error_hex",
            ) > _float_from_hex(
                self.relative_tolerance_hex,
                name="relative_tolerance_hex",
            ):
                raise ValueError("compressed MPO exceeds its reconstruction tolerance")
        else:
            if stored_bytes != dense_bytes:
                raise ValueError("exact fallback must retain the complete dense tensor")
            if self.attempted_mpo_met_tolerance != (
                self.selection_reason == "mpo_not_compact"
            ):
                raise ValueError("fallback reason contradicts attempted MPO result")
            if _float_from_hex(
                self.reconstruction_absolute_error_hex,
                name="reconstruction_absolute_error_hex",
            ) != 0.0:
                raise ValueError("exact fallback must have zero reconstruction error")
            if _float_from_hex(
                self.reconstruction_relative_error_hex,
                name="reconstruction_relative_error_hex",
            ) != 0.0:
                raise ValueError("exact fallback must have zero relative error")
        object.__setattr__(self, "action_ids", actions)
        object.__setattr__(self, "state_shape", shape)
        object.__setattr__(self, "requested_max_rank", rank)
        object.__setattr__(self, "attempted_bond_ranks", ranks)
        object.__setattr__(
            self,
            "source_world_model_sha256",
            require_sha256(
                self.source_world_model_sha256,
                field="source_world_model_sha256",
            ),
        )
        object.__setattr__(
            self,
            "graph_revision_sha256",
            require_sha256(
                self.graph_revision_sha256,
                field="graph_revision_sha256",
            ),
        )
        object.__setattr__(
            self,
            "verifier_hashes",
            _hash_items(self.verifier_hashes, name="verifier_hashes"),
        )
        for field in (
            "source_tensor_sha256",
            "reconstruction_sha256",
            "factor_payload_sha256",
        ):
            object.__setattr__(
                self,
                field,
                require_sha256(getattr(self, field), field=field),
            )

    @property
    def action_count(self) -> int:
        return len(self.action_ids)

    @property
    def state_count(self) -> int:
        return math.prod(self.state_shape)

    @property
    def exact_fallback(self) -> bool:
        return self.mode == "dense_exact_fallback"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_ids": list(self.action_ids),
            "attempted_absolute_error_hex": self.attempted_absolute_error_hex,
            "attempted_bond_ranks": list(self.attempted_bond_ranks),
            "attempted_mpo_met_tolerance": self.attempted_mpo_met_tolerance,
            "attempted_relative_error_hex": self.attempted_relative_error_hex,
            "compression_ratio_hex": self.compression_ratio_hex,
            "dense_numeric_bytes": self.dense_numeric_bytes,
            "factor_payload_sha256": self.factor_payload_sha256,
            "format": self.FORMAT,
            "graph_revision_sha256": self.graph_revision_sha256,
            "mode": self.mode,
            "reconstruction_absolute_error_hex": self.reconstruction_absolute_error_hex,
            "reconstruction_relative_error_hex": self.reconstruction_relative_error_hex,
            "reconstruction_sha256": self.reconstruction_sha256,
            "relative_tolerance_hex": self.relative_tolerance_hex,
            "requested_max_rank": self.requested_max_rank,
            "selection_reason": self.selection_reason,
            "source_tensor_sha256": self.source_tensor_sha256,
            "source_world_model_sha256": self.source_world_model_sha256,
            "state_shape": list(self.state_shape),
            "stored_numeric_bytes": self.stored_numeric_bytes,
            "verifier_hashes": dict(self.verifier_hashes),
        }

    @classmethod
    def from_dict(cls, value: object) -> "MPOReceipt":
        if not isinstance(value, dict) or value.get("format") != cls.FORMAT:
            raise MPOIntegrityError("unsupported MPO receipt")
        expected = {
            "action_ids",
            "attempted_absolute_error_hex",
            "attempted_bond_ranks",
            "attempted_mpo_met_tolerance",
            "attempted_relative_error_hex",
            "compression_ratio_hex",
            "dense_numeric_bytes",
            "factor_payload_sha256",
            "format",
            "graph_revision_sha256",
            "mode",
            "reconstruction_absolute_error_hex",
            "reconstruction_relative_error_hex",
            "reconstruction_sha256",
            "relative_tolerance_hex",
            "requested_max_rank",
            "selection_reason",
            "source_tensor_sha256",
            "source_world_model_sha256",
            "state_shape",
            "stored_numeric_bytes",
            "verifier_hashes",
        }
        if set(value) != expected or not isinstance(value.get("verifier_hashes"), dict):
            raise MPOIntegrityError("MPO receipt fields do not match schema")
        try:
            return cls(
                action_ids=tuple(value["action_ids"]),
                state_shape=tuple(value["state_shape"]),
                source_world_model_sha256=value["source_world_model_sha256"],
                graph_revision_sha256=value["graph_revision_sha256"],
                verifier_hashes=_hash_items(
                    value["verifier_hashes"], name="verifier_hashes"
                ),
                source_tensor_sha256=value["source_tensor_sha256"],
                mode=value["mode"],
                selection_reason=value["selection_reason"],
                requested_max_rank=value["requested_max_rank"],
                attempted_bond_ranks=tuple(value["attempted_bond_ranks"]),
                relative_tolerance_hex=value["relative_tolerance_hex"],
                attempted_absolute_error_hex=value["attempted_absolute_error_hex"],
                attempted_relative_error_hex=value["attempted_relative_error_hex"],
                attempted_mpo_met_tolerance=value["attempted_mpo_met_tolerance"],
                reconstruction_absolute_error_hex=value[
                    "reconstruction_absolute_error_hex"
                ],
                reconstruction_relative_error_hex=value[
                    "reconstruction_relative_error_hex"
                ],
                reconstruction_sha256=value["reconstruction_sha256"],
                factor_payload_sha256=value["factor_payload_sha256"],
                dense_numeric_bytes=value["dense_numeric_bytes"],
                stored_numeric_bytes=value["stored_numeric_bytes"],
                compression_ratio_hex=value["compression_ratio_hex"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MPOIntegrityError("invalid MPO receipt") from exc

    @property
    def sha256(self) -> str:
        return _body_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class ActionConditionedMPO:
    """Immutable selected MPO or explicit exact dense fallback."""

    receipt: MPOReceipt
    action_core_shape: tuple[int, int, int] | None
    action_core_bytes: bytes | None
    site_core_shapes: tuple[tuple[int, int, int, int], ...]
    site_core_bytes: tuple[bytes, ...]
    dense_fallback_bytes: bytes | None
    _source_verified: bool

    FORMAT = "immer-ooe-action-conditioned-mpo/v1"
    ENVELOPE_FORMAT = "immer-ooe-action-conditioned-mpo-envelope/v1"

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, MPOReceipt):
            raise TypeError("receipt must be an MPOReceipt")
        if not isinstance(self._source_verified, bool):
            raise TypeError("_source_verified must be a boolean")
        if self.receipt.mode == "mpo":
            if self.action_core_shape is None or self.action_core_bytes is None:
                raise MPOIntegrityError("compressed MPO has no action core")
            if self.dense_fallback_bytes is not None:
                raise MPOIntegrityError("compressed MPO must not carry dense fallback")
            action_shape = tuple(
                _integer(value, name="action core dimension", minimum=1)
                for value in self.action_core_shape
            )
            if len(action_shape) != 3 or action_shape[:2] != (
                1,
                self.receipt.action_count,
            ):
                raise MPOIntegrityError("action core shape does not match receipt")
            _array_from_bytes(self.action_core_bytes, action_shape)
            if len(self.site_core_shapes) != len(self.receipt.state_shape):
                raise MPOIntegrityError("MPO site-core count does not match state_shape")
            if len(self.site_core_bytes) != len(self.site_core_shapes):
                raise MPOIntegrityError("MPO site-core byte count mismatch")
            left_rank = action_shape[-1]
            stored = len(self.action_core_bytes)
            for index, (raw_shape, data) in enumerate(
                zip(self.site_core_shapes, self.site_core_bytes, strict=True)
            ):
                shape = tuple(
                    _integer(value, name="site core dimension", minimum=1)
                    for value in raw_shape
                )
                dimension = self.receipt.state_shape[index]
                if len(shape) != 4 or shape[:3] != (
                    left_rank,
                    dimension,
                    dimension,
                ):
                    raise MPOIntegrityError("MPO site-core topology mismatch")
                _array_from_bytes(data, shape)
                left_rank = shape[-1]
                stored += len(data)
            if left_rank != 1:
                raise MPOIntegrityError("last MPO bond rank must equal one")
            if stored != self.receipt.stored_numeric_bytes:
                raise MPOIntegrityError("MPO stored byte count mismatch")
        else:
            if any(
                value
                for value in (
                    self.action_core_shape,
                    self.action_core_bytes,
                    self.site_core_shapes,
                    self.site_core_bytes,
                )
            ):
                raise MPOIntegrityError("exact fallback must not retain rejected cores")
            if self.dense_fallback_bytes is None:
                raise MPOIntegrityError("exact fallback is missing its dense tensor")
            _array_from_bytes(
                self.dense_fallback_bytes,
                (
                    self.receipt.action_count,
                    self.receipt.state_count,
                    self.receipt.state_count,
                ),
            )
            if len(self.dense_fallback_bytes) != self.receipt.stored_numeric_bytes:
                raise MPOIntegrityError("dense fallback byte count mismatch")
        payload_sha = _factor_payload_sha256(
            action_core_shape=self.action_core_shape,
            action_core_bytes=self.action_core_bytes,
            site_core_shapes=self.site_core_shapes,
            site_core_bytes=self.site_core_bytes,
            dense_fallback_bytes=self.dense_fallback_bytes,
        )
        if payload_sha != self.receipt.factor_payload_sha256:
            raise MPOIntegrityError("MPO factor payload hash mismatch")
        reconstruction = self._reconstruct_unverified()
        if array_sha256(reconstruction) != self.receipt.reconstruction_sha256:
            raise MPOIntegrityError("MPO reconstruction hash mismatch")
        if self.receipt.exact_fallback:
            if array_sha256(reconstruction) != self.receipt.source_tensor_sha256:
                raise MPOIntegrityError("dense fallback is not the exact source tensor")

    @property
    def compressed(self) -> bool:
        return self.receipt.mode == "mpo"

    @property
    def source_verified(self) -> bool:
        return self._source_verified

    def _require_source_verified(self) -> None:
        if not self._source_verified:
            raise MPOIntegrityError(
                "structural-only MPO cannot execute before exact source verification"
            )

    def restore_cores(self) -> tuple[FloatArray, ...]:
        if not self.compressed:
            return ()
        assert self.action_core_shape is not None
        assert self.action_core_bytes is not None
        action = _array_from_bytes(self.action_core_bytes, self.action_core_shape)
        sites = tuple(
            _array_from_bytes(data, shape)
            for shape, data in zip(
                self.site_core_shapes,
                self.site_core_bytes,
                strict=True,
            )
        )
        return (action, *sites)

    def _reconstruct_unverified(self) -> FloatArray:
        if not self.compressed:
            assert self.dense_fallback_bytes is not None
            return _array_from_bytes(
                self.dense_fallback_bytes,
                (
                    self.receipt.action_count,
                    self.receipt.state_count,
                    self.receipt.state_count,
                ),
            )
        cores = self.restore_cores()
        physical_cores = [cores[0]]
        physical_cores.extend(
            core.reshape(core.shape[0], core.shape[1] * core.shape[2], core.shape[3])
            for core in cores[1:]
        )
        physical = _reconstruct_tt(physical_cores)
        return _physical_to_action_tensor(
            physical,
            action_count=self.receipt.action_count,
            state_shape=self.receipt.state_shape,
        )

    def reconstruct(self) -> FloatArray:
        self._require_source_verified()
        return self._reconstruct_unverified()

    def verify_exact_source(
        self,
        source_tensor: ArrayLike | None = None,
        *,
        source_resolver: Callable[[MPOReceipt], ArrayLike] | None = None,
    ) -> "ActionConditionedMPO":
        """Bind a structural payload to its exact external world-model tensor.

        Compressed payloads cannot prove their own approximation error because
        they intentionally do not carry the dense source.  Verification must
        therefore resolve that source externally, then recompute both its hash
        and the actual reconstruction error before execution is enabled.
        """

        if (source_tensor is None) == (source_resolver is None):
            raise ValueError(
                "provide exactly one of source_tensor or source_resolver"
            )
        resolved = (
            source_resolver(self.receipt)
            if source_resolver is not None
            else source_tensor
        )
        source = _action_tensor(resolved)
        expected_shape = (
            self.receipt.action_count,
            self.receipt.state_count,
            self.receipt.state_count,
        )
        if source.shape != expected_shape:
            raise MPOIntegrityError("external source tensor shape mismatch")
        if array_sha256(source) != self.receipt.source_tensor_sha256:
            raise MPOIntegrityError("external source tensor hash mismatch")
        reconstruction = self._reconstruct_unverified()
        absolute_error = float(np.linalg.norm(reconstruction - source))
        source_norm = float(np.linalg.norm(source))
        relative_error = absolute_error / source_norm if source_norm else 0.0
        tolerance = float.fromhex(self.receipt.relative_tolerance_hex)
        if relative_error > tolerance:
            raise MPOIntegrityError(
                "actual MPO reconstruction error exceeds receipt tolerance"
            )
        claimed_absolute = float.fromhex(
            self.receipt.reconstruction_absolute_error_hex
        )
        claimed_relative = float.fromhex(
            self.receipt.reconstruction_relative_error_hex
        )
        absolute_slack = 64.0 * np.finfo(np.float64).eps * max(1.0, source_norm)
        if not math.isclose(
            absolute_error,
            claimed_absolute,
            rel_tol=1e-12,
            abs_tol=absolute_slack,
        ) or not math.isclose(
            relative_error,
            claimed_relative,
            rel_tol=1e-12,
            abs_tol=64.0 * np.finfo(np.float64).eps,
        ):
            raise MPOIntegrityError(
                "actual MPO reconstruction error contradicts receipt"
            )
        return type(self)(
            receipt=self.receipt,
            action_core_shape=self.action_core_shape,
            action_core_bytes=self.action_core_bytes,
            site_core_shapes=self.site_core_shapes,
            site_core_bytes=self.site_core_bytes,
            dense_fallback_bytes=self.dense_fallback_bytes,
            _source_verified=True,
        )

    def transition_kernel(self, action: int | str) -> FloatArray:
        self._require_source_verified()
        if isinstance(action, str):
            try:
                index = self.receipt.action_ids.index(action)
            except ValueError as exc:
                raise KeyError(f"unknown MPO action: {action}") from exc
        else:
            index = _integer(
                action,
                name="action index",
                maximum=self.receipt.action_count - 1,
            )
        states = self.receipt.state_count
        if not self.compressed:
            assert self.dense_fallback_bytes is not None
            matrix_values = states * states
            offset = index * matrix_values * np.dtype("<f8").itemsize
            return np.frombuffer(
                self.dense_fallback_bytes,
                dtype="<f8",
                count=matrix_values,
                offset=offset,
            ).reshape(states, states).copy()

        cores = self.restore_cores()
        # Select one action before touching any state core.  The live tensor
        # therefore starts at one bond vector, not an A×S×S reconstruction.
        action_vector = np.ascontiguousarray(cores[0][0, index, :])
        selected: FloatArray = action_vector
        for core in cores[1:]:
            selected = np.tensordot(selected, core, axes=([-1], [0]))
        interleaved = np.squeeze(selected, axis=-1)
        sites = len(self.receipt.state_shape)
        axes = [2 * site for site in range(sites)]
        axes.extend(2 * site + 1 for site in range(sites))
        return np.ascontiguousarray(
            interleaved.transpose(axes).reshape(states, states),
            dtype=np.float64,
        )

    def as_action_kernels(self) -> dict[str, FloatArray]:
        """Expose execution-valid kernels through the option planner's API.

        TT-SVD reconstruction can place values a few ulps below zero and move
        row mass by the same scale.  The receipt continues to describe the raw
        reconstruction; this boundary view removes only numerical residue and
        restores exact stochastic rows before a planning consumer sees it.
        """

        residue_bound = max(
            1e-14,
            4.0 * float.fromhex(self.receipt.relative_tolerance_hex),
        )
        result: dict[str, FloatArray] = {}
        for action in self.receipt.action_ids:
            kernel = self.transition_kernel(action)
            if float(np.min(kernel)) < -residue_bound:
                raise MPOIntegrityError(
                    "MPO action contraction is materially non-stochastic"
                )
            row_error = float(np.max(np.abs(kernel.sum(axis=1) - 1.0)))
            if row_error > residue_bound:
                raise MPOIntegrityError("MPO action contraction lost material row mass")
            kernel = np.maximum(kernel, 0.0)
            kernel /= kernel.sum(axis=1, keepdims=True)
            result[action] = np.ascontiguousarray(kernel)
        return result

    def _factor_dict(self) -> dict[str, Any]:
        if not self.compressed:
            assert self.dense_fallback_bytes is not None
            return {
                "data_base64": base64.b64encode(self.dense_fallback_bytes).decode(
                    "ascii"
                ),
                "storage": "dense-float64-le",
            }
        assert self.action_core_shape is not None
        assert self.action_core_bytes is not None
        return {
            "action_core": {
                "data_base64": base64.b64encode(self.action_core_bytes).decode(
                    "ascii"
                ),
                "shape": list(self.action_core_shape),
            },
            "site_cores": [
                {
                    "data_base64": base64.b64encode(data).decode("ascii"),
                    "shape": list(shape),
                }
                for shape, data in zip(
                    self.site_core_shapes,
                    self.site_core_bytes,
                    strict=True,
                )
            ],
            "storage": "mpo-cores-float64-le",
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "factors": self._factor_dict(),
            "format": self.FORMAT,
            "receipt": self.receipt.to_dict(),
        }

    def to_bytes(self) -> bytes:
        body = self.to_dict()
        return canonical_json_bytes(
            {
                "body": body,
                "body_sha256": _body_sha256(body),
                "format": self.ENVELOPE_FORMAT,
            }
        )

    @classmethod
    def _parse_payload(cls, data: bytes) -> "ActionConditionedMPO":
        envelope = _decode_canonical(data)
        if (
            not isinstance(envelope, dict)
            or set(envelope) != {"body", "body_sha256", "format"}
            or envelope.get("format") != cls.ENVELOPE_FORMAT
            or not isinstance(envelope.get("body"), dict)
        ):
            raise MPOIntegrityError("invalid MPO envelope")
        body = envelope["body"]
        if _body_sha256(body) != envelope.get("body_sha256"):
            raise MPOIntegrityError("MPO envelope hash mismatch")
        if set(body) != {"factors", "format", "receipt"} or body.get(
            "format"
        ) != cls.FORMAT:
            raise MPOIntegrityError("MPO body does not match schema")
        receipt = MPOReceipt.from_dict(body.get("receipt"))
        factors = body.get("factors")
        if not isinstance(factors, dict):
            raise MPOIntegrityError("MPO factors must be an object")
        try:
            if receipt.mode == "dense_exact_fallback":
                if set(factors) != {"data_base64", "storage"} or factors.get(
                    "storage"
                ) != "dense-float64-le":
                    raise MPOIntegrityError("invalid dense fallback descriptor")
                dense = base64.b64decode(factors["data_base64"], validate=True)
                result = cls(
                    receipt=receipt,
                    action_core_shape=None,
                    action_core_bytes=None,
                    site_core_shapes=(),
                    site_core_bytes=(),
                    dense_fallback_bytes=dense,
                    _source_verified=False,
                )
            else:
                if set(factors) != {"action_core", "site_cores", "storage"} or factors.get(
                    "storage"
                ) != "mpo-cores-float64-le":
                    raise MPOIntegrityError("invalid MPO core descriptor")
                action = factors["action_core"]
                sites = factors["site_cores"]
                if (
                    not isinstance(action, dict)
                    or set(action) != {"data_base64", "shape"}
                    or not isinstance(sites, list)
                ):
                    raise MPOIntegrityError("invalid MPO core schema")
                if any(
                    not isinstance(site, dict)
                    or set(site) != {"data_base64", "shape"}
                    for site in sites
                ):
                    raise MPOIntegrityError("invalid MPO site-core schema")
                result = cls(
                    receipt=receipt,
                    action_core_shape=tuple(action["shape"]),
                    action_core_bytes=base64.b64decode(
                        action["data_base64"], validate=True
                    ),
                    site_core_shapes=tuple(tuple(site["shape"]) for site in sites),
                    site_core_bytes=tuple(
                        base64.b64decode(site["data_base64"], validate=True)
                        for site in sites
                    ),
                    dense_fallback_bytes=None,
                    _source_verified=False,
                )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, MPOIntegrityError):
                raise
            raise MPOIntegrityError("invalid MPO factor encoding") from exc
        if result.to_dict() != body:
            raise MPOIntegrityError("MPO payload failed canonical roundtrip")
        return result

    @classmethod
    def parse_structure_only(cls, data: bytes) -> "ActionConditionedMPO":
        """Parse hashes and core topology without granting execution rights."""

        return cls._parse_payload(data)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        source_tensor: ArrayLike | None = None,
        source_resolver: Callable[[MPOReceipt], ArrayLike] | None = None,
    ) -> "ActionConditionedMPO":
        """Load an executable payload, verifying compressed data externally."""

        result = cls._parse_payload(data)
        if result.compressed:
            if source_tensor is None and source_resolver is None:
                raise MPOIntegrityError(
                    "compressed MPO load requires an exact external source resolver"
                )
            return result.verify_exact_source(
                source_tensor,
                source_resolver=source_resolver,
            )
        if source_tensor is not None or source_resolver is not None:
            return result.verify_exact_source(
                source_tensor,
                source_resolver=source_resolver,
            )
        # A dense fallback already carries the complete source, whose exact
        # canonical hash was checked in __post_init__.
        return cls(
            receipt=result.receipt,
            action_core_shape=None,
            action_core_bytes=None,
            site_core_shapes=(),
            site_core_bytes=(),
            dense_fallback_bytes=result.dense_fallback_bytes,
            _source_verified=True,
        )


def factorize_action_transitions(
    transition_tensor: ArrayLike,
    *,
    action_ids: Sequence[str],
    state_shape: Sequence[int],
    source_world_model_sha256: str,
    graph_revision_sha256: str,
    verifier_hashes: Mapping[str, str],
    max_rank: int,
    relative_tolerance: float = 1e-10,
    max_work_bytes: int = MAX_MPO_WORK_BYTES,
) -> ActionConditionedMPO:
    """TT-SVD an action-conditioned model or return an explicit exact fallback."""

    tensor = _action_tensor(transition_tensor)
    actions = tuple(_identifier(value, name="action_id") for value in action_ids)
    if len(actions) != tensor.shape[0] or len(set(actions)) != len(actions):
        raise ValueError("action_ids must uniquely label every tensor action")
    shape = _state_shape(state_shape, states=int(tensor.shape[1]))
    rank_limit = _integer(
        max_rank,
        name="max_rank",
        minimum=1,
        maximum=MAX_MPO_RANK,
    )
    tolerance = float(relative_tolerance)
    if not math.isfinite(tolerance) or not 0.0 <= tolerance <= 1.0:
        raise ValueError("relative_tolerance must lie in [0, 1]")
    work_limit = _integer(
        max_work_bytes,
        name="max_work_bytes",
        minimum=tensor.nbytes,
        maximum=8 * MAX_MPO_WORK_BYTES,
    )
    model_hash = require_sha256(
        source_world_model_sha256,
        field="source_world_model_sha256",
    )
    graph_hash = require_sha256(
        graph_revision_sha256,
        field="graph_revision_sha256",
    )
    verifiers = _hash_items(verifier_hashes, name="verifier_hashes")
    physical = _physical_tensor(tensor, shape)
    tt_cores, bond_ranks = _tt_svd(
        physical,
        maximum_rank=rank_limit,
        relative_tolerance=tolerance,
        max_work_bytes=work_limit,
    )
    attempted_physical = _reconstruct_tt(tt_cores)
    attempted = _physical_to_action_tensor(
        attempted_physical,
        action_count=len(actions),
        state_shape=shape,
    )
    absolute_error = float(np.linalg.norm(attempted - tensor))
    source_norm = float(np.linalg.norm(tensor))
    relative_error = absolute_error / source_norm if source_norm else 0.0
    met_tolerance = relative_error <= tolerance
    action_core = tt_cores[0]
    site_cores = tuple(
        core.reshape(
            core.shape[0],
            dimension,
            dimension,
            core.shape[2],
        )
        for core, dimension in zip(tt_cores[1:], shape, strict=True)
    )
    action_core_bytes = _little_f8_bytes(action_core)
    site_core_bytes = tuple(_little_f8_bytes(core) for core in site_cores)
    mpo_bytes = len(action_core_bytes) + sum(map(len, site_core_bytes))
    dense_bytes = tensor.nbytes
    if met_tolerance and mpo_bytes < dense_bytes:
        mode = "mpo"
        reason = "tolerance_met_and_compact"
        selected_action_shape: tuple[int, int, int] | None = tuple(action_core.shape)
        selected_action_bytes: bytes | None = action_core_bytes
        selected_site_shapes = tuple(tuple(core.shape) for core in site_cores)
        selected_site_bytes = site_core_bytes
        dense_fallback = None
        selected_reconstruction = attempted
        reconstruction_absolute_error = absolute_error
        reconstruction_relative_error = relative_error
        stored_bytes = mpo_bytes
    else:
        mode = "dense_exact_fallback"
        reason = "rank_budget_exceeded" if not met_tolerance else "mpo_not_compact"
        selected_action_shape = None
        selected_action_bytes = None
        selected_site_shapes = ()
        selected_site_bytes = ()
        dense_fallback = _little_f8_bytes(tensor)
        selected_reconstruction = tensor
        reconstruction_absolute_error = 0.0
        reconstruction_relative_error = 0.0
        stored_bytes = dense_bytes
    payload_sha = _factor_payload_sha256(
        action_core_shape=selected_action_shape,
        action_core_bytes=selected_action_bytes,
        site_core_shapes=selected_site_shapes,
        site_core_bytes=selected_site_bytes,
        dense_fallback_bytes=dense_fallback,
    )
    receipt = MPOReceipt(
        action_ids=actions,
        state_shape=shape,
        source_world_model_sha256=model_hash,
        graph_revision_sha256=graph_hash,
        verifier_hashes=verifiers,
        source_tensor_sha256=array_sha256(tensor),
        mode=mode,
        selection_reason=reason,
        requested_max_rank=rank_limit,
        attempted_bond_ranks=bond_ranks,
        relative_tolerance_hex=tolerance.hex(),
        attempted_absolute_error_hex=absolute_error.hex(),
        attempted_relative_error_hex=relative_error.hex(),
        attempted_mpo_met_tolerance=met_tolerance,
        reconstruction_absolute_error_hex=reconstruction_absolute_error.hex(),
        reconstruction_relative_error_hex=reconstruction_relative_error.hex(),
        reconstruction_sha256=array_sha256(selected_reconstruction),
        factor_payload_sha256=payload_sha,
        dense_numeric_bytes=dense_bytes,
        stored_numeric_bytes=stored_bytes,
        compression_ratio_hex=(stored_bytes / dense_bytes).hex(),
    )
    return ActionConditionedMPO(
        receipt=receipt,
        action_core_shape=selected_action_shape,
        action_core_bytes=selected_action_bytes,
        site_core_shapes=selected_site_shapes,
        site_core_bytes=selected_site_bytes,
        dense_fallback_bytes=dense_fallback,
        _source_verified=True,
    )


__all__ = [
    "ActionConditionedMPO",
    "MAX_MPO_ACTIONS",
    "MAX_MPO_PAYLOAD_BYTES",
    "MAX_MPO_RANK",
    "MAX_MPO_SITES",
    "MAX_MPO_STATES",
    "MAX_MPO_WORK_BYTES",
    "MPOIntegrityError",
    "MPOReceipt",
    "factorize_action_transitions",
]
