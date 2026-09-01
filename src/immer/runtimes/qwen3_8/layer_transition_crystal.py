"""Quotient-space transition crystals for Qwen3.8 layer 63.

The crystal replaces one K1 layer transition with a small affine operator in a
deterministic quotient space.  For an exact BF16 input row ``h`` it computes::

    z       = h P
    z_next  = z A + b
    h_next  = h + (z_next - z) P^+

``P`` is a transport-neutral Rademacher projection derived solely from a
sealed projection identity.  It is never sampled from ambient RNG state and
is never stored as an opaque device object.  A crystal is usable only inside
its finite sketch-space coverage ball and below the caller's explicit error
budget.

Artifacts and banks use canonical, hash-sealed JSON with exact little-endian
float64 tensor payloads.  Loading rejects duplicate keys, non-canonical JSON,
identity drift, tensor tampering, symlinks, unstable files, and partial state.
No pickle or host path participates in an identity.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import tempfile
import threading
from typing import Any, Iterator

import torch


LAYER_TRANSITION_PROJECTION_SCHEMA = "immer.qwen3.8-layer-transition-projection/v1"
LAYER_TRANSITION_IDENTITY_SCHEMA = "immer.qwen3.8-layer-transition-crystal-identity/v1"
LAYER_TRANSITION_COVERAGE_SCHEMA = "immer.qwen3.8-layer-transition-crystal-coverage/v1"
LAYER_TRANSITION_CRYSTAL_SCHEMA = "immer.qwen3.8-layer-transition-crystal/v1"
LAYER_TRANSITION_CRYSTAL_ENVELOPE_SCHEMA = (
    "immer.qwen3.8-layer-transition-crystal-envelope/v1"
)
LAYER_TRANSITION_BANK_SCHEMA = "immer.qwen3.8-layer-transition-crystal-bank/v1"
LAYER_TRANSITION_BANK_ENVELOPE_SCHEMA = (
    "immer.qwen3.8-layer-transition-crystal-bank-envelope/v1"
)
LAYER_TRANSITION_TENSOR_SCHEMA = "immer.qwen3.8-layer-transition-exact-f64-tensor/v1"
LAYER_TRANSITION_PROJECTION_ABI = (
    "immer.qwen3.8/cartography-rademacher-torch-seed64-f64-hidden-normalised/v1"
)
LAYER_TRANSITION_LIFT_ABI = (
    "immer.qwen3.8/row-affine-quotient-moore-penrose-residual-lift/v1"
)

TARGET_LAYER_INDEX = 63

_MAX_COUNTER = (1 << 63) - 1
_MAX_HIDDEN_DIM = 1 << 16
_MAX_SKETCH_DIM = 4096
_MAX_CRYSTALS = 1 << 16
_MAX_TENSOR_ELEMENTS = 1 << 24
_MAX_TENSOR_BYTES = _MAX_TENSOR_ELEMENTS * 8
_MAX_STATE_BYTES = 512 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024
_HEX = frozenset("0123456789abcdef")


class LayerTransitionCrystalError(RuntimeError):
    """A layer-transition crystal input or state is invalid."""


class LayerTransitionCrystalIntegrityError(LayerTransitionCrystalError):
    """A crystal or bank is malformed, unstable, or hash-inconsistent."""


class LayerTransitionCrystalIdentityError(LayerTransitionCrystalError):
    """A crystal belongs to another immutable model/graph identity."""


class LayerTransitionCrystalCapacityError(LayerTransitionCrystalError):
    """A bounded crystal bank has no room for another unique crystal."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LayerTransitionCrystalIntegrityError(
            "layer-transition data is not canonical JSON"
        ) from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_document(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value))


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and not (set(value) - _HEX)


def _digest(value: object, field: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _uint(
    value: object,
    *,
    field: str,
    positive: bool = False,
    maximum: int = _MAX_COUNTER,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < int(positive)
        or value > maximum
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must be a bounded {qualifier} integer")
    return value


def _finite_non_negative(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return result


def _bounded_add(left: int, right: int) -> int:
    return min(_MAX_COUNTER, left + right)


def _json_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LayerTransitionCrystalIntegrityError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _parse_canonical_json(value: bytes, *, kind: str) -> Mapping[str, Any]:
    if not isinstance(value, bytes):
        raise TypeError(f"{kind} bytes must be bytes")
    if len(value) > _MAX_STATE_BYTES:
        raise LayerTransitionCrystalIntegrityError(f"{kind} exceeds its byte limit")
    try:
        document = json.loads(value, object_pairs_hook=_json_no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LayerTransitionCrystalIntegrityError(f"{kind} is not valid JSON") from exc
    if not isinstance(document, Mapping):
        raise LayerTransitionCrystalIntegrityError(f"{kind} is not an object")
    if _canonical_json(document) != value:
        raise LayerTransitionCrystalIntegrityError(f"{kind} is not canonical JSON")
    return document


def _sealed_document(body: Mapping[str, Any], schema: str) -> bytes:
    return _canonical_json(
        {
            "body": body,
            "body_sha256": _sha256_document(body),
            "schema": schema,
        }
    )


def _unseal_document(value: bytes, *, schema: str, kind: str) -> Mapping[str, Any]:
    document = _parse_canonical_json(value, kind=kind)
    if set(document) != {"body", "body_sha256", "schema"}:
        raise LayerTransitionCrystalIntegrityError(
            f"{kind} envelope fields are invalid"
        )
    if document["schema"] != schema:
        raise LayerTransitionCrystalIntegrityError(f"{kind} schema is invalid")
    body = document["body"]
    if not isinstance(body, Mapping):
        raise LayerTransitionCrystalIntegrityError(f"{kind} body is invalid")
    if not _is_sha256(document["body_sha256"]):
        raise LayerTransitionCrystalIntegrityError(f"{kind} body digest is invalid")
    if document["body_sha256"] != _sha256_document(body):
        raise LayerTransitionCrystalIntegrityError(f"{kind} body SHA-256 mismatch")
    return body


def _shape(value: object, *, field: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{field} must be a shape sequence")
    result = tuple(value)
    if not 1 <= len(result) <= 2:
        raise ValueError(f"{field} must have rank one or two")
    elements = 1
    for index, dimension in enumerate(result):
        _uint(
            dimension,
            field=f"{field}[{index}]",
            positive=True,
            maximum=_MAX_HIDDEN_DIM,
        )
        elements *= dimension
        if elements > _MAX_TENSOR_ELEMENTS:
            raise ValueError(f"{field} exceeds the tensor element limit")
    return result


def _f64_bytes(value: torch.Tensor, *, field: str) -> tuple[tuple[int, ...], bytes]:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{field} must be a torch.Tensor")
    shape = _shape(tuple(value.shape), field=f"{field} shape")
    cpu = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(cpu).all().item()):
        raise ValueError(f"{field} must contain only finite values")
    flat = cpu.reshape(-1).tolist()
    raw = struct.pack(f"<{len(flat)}d", *flat)
    if len(raw) > _MAX_TENSOR_BYTES:
        raise ValueError(f"{field} exceeds the tensor byte limit")
    return shape, raw


def _tensor_digest(shape: tuple[int, ...], raw: bytes) -> str:
    header = {
        "dtype": "float64",
        "schema": LAYER_TRANSITION_TENSOR_SCHEMA,
        "shape": list(shape),
    }
    return _sha256_bytes(_canonical_json(header) + b"\0" + raw)


def _tensor_record(value: torch.Tensor, *, field: str) -> dict[str, object]:
    shape, raw = _f64_bytes(value, field=field)
    return {
        "data": base64.b64encode(raw).decode("ascii"),
        "data_sha256": _tensor_digest(shape, raw),
        "dtype": "float64",
        "schema": LAYER_TRANSITION_TENSOR_SCHEMA,
        "shape": list(shape),
    }


def _tensor_from_record(value: object, *, field: str) -> torch.Tensor:
    if not isinstance(value, Mapping) or set(value) != {
        "data",
        "data_sha256",
        "dtype",
        "schema",
        "shape",
    }:
        raise LayerTransitionCrystalIntegrityError(f"{field} tensor fields are invalid")
    if value["schema"] != LAYER_TRANSITION_TENSOR_SCHEMA or value["dtype"] != "float64":
        raise LayerTransitionCrystalIntegrityError(f"{field} tensor ABI is invalid")
    try:
        shape = _shape(value["shape"], field=f"{field} shape")
        if not isinstance(value["data"], str):
            raise TypeError("tensor data is not text")
        raw = base64.b64decode(value["data"], validate=True)
    except (TypeError, ValueError) as exc:
        raise LayerTransitionCrystalIntegrityError(
            f"{field} tensor payload is invalid"
        ) from exc
    elements = math.prod(shape)
    if len(raw) != elements * 8 or len(raw) > _MAX_TENSOR_BYTES:
        raise LayerTransitionCrystalIntegrityError(
            f"{field} tensor byte width is invalid"
        )
    if not _is_sha256(value["data_sha256"]) or value["data_sha256"] != _tensor_digest(
        shape, raw
    ):
        raise LayerTransitionCrystalIntegrityError(f"{field} tensor SHA-256 mismatch")
    unpacked = struct.unpack(f"<{elements}d", raw)
    tensor = torch.tensor(unpacked, dtype=torch.float64).reshape(shape).contiguous()
    if not bool(torch.isfinite(tensor).all().item()):
        raise LayerTransitionCrystalIntegrityError(
            f"{field} tensor contains a non-finite value"
        )
    return tensor


@dataclass(frozen=True, slots=True)
class LayerTransitionProjectionIdentity:
    """Transport-neutral identity of the deterministic quotient projection."""

    hidden_dim: int
    sketch_dim: int
    seed_sha256: str
    abi: str = LAYER_TRANSITION_PROJECTION_ABI

    def __post_init__(self) -> None:
        _uint(
            self.hidden_dim,
            field="hidden_dim",
            positive=True,
            maximum=_MAX_HIDDEN_DIM,
        )
        _uint(
            self.sketch_dim,
            field="sketch_dim",
            positive=True,
            maximum=_MAX_SKETCH_DIM,
        )
        if self.sketch_dim > self.hidden_dim:
            raise ValueError("sketch_dim cannot exceed hidden_dim")
        if self.hidden_dim * self.sketch_dim > _MAX_TENSOR_ELEMENTS:
            raise ValueError("projection exceeds the tensor element limit")
        _digest(self.seed_sha256, "seed_sha256")
        if self.abi != LAYER_TRANSITION_PROJECTION_ABI:
            raise ValueError("projection ABI is not implemented")

    def _base_record(self) -> dict[str, object]:
        return {
            "abi": self.abi,
            "hidden_dim": self.hidden_dim,
            "schema": LAYER_TRANSITION_PROJECTION_SCHEMA,
            "seed_sha256": self.seed_sha256,
            "sketch_dim": self.sketch_dim,
        }

    @property
    def projection_sha256(self) -> str:
        return _projection_sha256_cached(
            self.hidden_dim,
            self.sketch_dim,
            self.seed_sha256,
            self.abi,
        )

    @property
    def identity_sha256(self) -> str:
        return self.projection_sha256

    def matrix(self) -> torch.Tensor:
        """Return a caller-owned CPU float64 copy of ``P``."""

        return _projection_matrix_cached(
            self.hidden_dim,
            self.sketch_dim,
            self.seed_sha256,
            self.abi,
        ).clone()

    def pseudoinverse(self) -> torch.Tensor:
        """Return a caller-owned CPU float64 Moore-Penrose inverse of ``P``."""

        return _projection_pseudoinverse_cached(
            self.hidden_dim,
            self.sketch_dim,
            self.seed_sha256,
            self.abi,
        ).clone()

    def to_record(self) -> dict[str, object]:
        return self._base_record() | {
            "projection_sha256": self.projection_sha256,
        }

    @classmethod
    def from_record(cls, value: object) -> "LayerTransitionProjectionIdentity":
        if not isinstance(value, Mapping) or set(value) != {
            "abi",
            "hidden_dim",
            "projection_sha256",
            "schema",
            "seed_sha256",
            "sketch_dim",
        }:
            raise LayerTransitionCrystalIntegrityError(
                "projection identity fields are invalid"
            )
        if value["schema"] != LAYER_TRANSITION_PROJECTION_SCHEMA:
            raise LayerTransitionCrystalIntegrityError(
                "projection identity schema is invalid"
            )
        try:
            identity = cls(
                hidden_dim=value["hidden_dim"],
                sketch_dim=value["sketch_dim"],
                seed_sha256=value["seed_sha256"],
                abi=value["abi"],
            )
        except (TypeError, ValueError) as exc:
            raise LayerTransitionCrystalIntegrityError(
                "projection identity values are invalid"
            ) from exc
        if value["projection_sha256"] != identity.projection_sha256:
            raise LayerTransitionCrystalIntegrityError(
                "projection identity SHA-256 mismatch"
            )
        return identity


@lru_cache(maxsize=16)
def _projection_matrix_cached(
    hidden_dim: int,
    sketch_dim: int,
    seed_sha256: str,
    abi: str,
) -> torch.Tensor:
    if abi != LAYER_TRANSITION_PROJECTION_ABI:
        raise ValueError("projection ABI is not implemented")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed_sha256[:16], 16))
    bits = torch.randint(
        0,
        2,
        (hidden_dim, sketch_dim),
        dtype=torch.int8,
        generator=generator,
    )
    scale = 1.0 / math.sqrt(hidden_dim)
    signs = bits.to(dtype=torch.float64).mul_(2.0).sub_(1.0).mul_(scale)
    matrix = signs.contiguous()
    # A quotient lift requires full column rank.  A deterministic seed that
    # creates a rank-deficient matrix is invalid rather than silently changed.
    if int(torch.linalg.matrix_rank(matrix).item()) != sketch_dim:
        raise ValueError("Rademacher projection is rank deficient")
    return matrix


@lru_cache(maxsize=16)
def _projection_sha256_cached(
    hidden_dim: int,
    sketch_dim: int,
    seed_sha256: str,
    abi: str,
) -> str:
    matrix = _projection_matrix_cached(hidden_dim, sketch_dim, seed_sha256, abi)
    shape, raw = _f64_bytes(matrix, field="projection")
    record = {
        "abi": abi,
        "hidden_dim": hidden_dim,
        "schema": LAYER_TRANSITION_PROJECTION_SCHEMA,
        "seed_sha256": seed_sha256,
        "sketch_dim": sketch_dim,
    }
    return _sha256_bytes(
        _canonical_json(record)
        + b"\0"
        + _canonical_json({"shape": list(shape), "dtype": "float64"})
        + b"\0"
        + raw
    )


@lru_cache(maxsize=16)
def _projection_pseudoinverse_cached(
    hidden_dim: int,
    sketch_dim: int,
    seed_sha256: str,
    abi: str,
) -> torch.Tensor:
    projection = _projection_matrix_cached(
        hidden_dim,
        sketch_dim,
        seed_sha256,
        abi,
    )
    # P has full column rank, so P^+ = (P^T P)^-1 P^T.  solve() avoids an
    # unnecessary full HxS SVD while remaining the Moore-Penrose inverse.
    gram = projection.T @ projection
    pinv = torch.linalg.solve(gram, projection.T).contiguous()
    identity = pinv @ projection
    tolerance = 256.0 * torch.finfo(torch.float64).eps * max(hidden_dim, sketch_dim)
    if not torch.allclose(
        identity,
        torch.eye(sketch_dim, dtype=torch.float64),
        rtol=tolerance,
        atol=tolerance,
    ):
        raise ValueError("Rademacher projection pseudoinverse is unstable")
    return pinv


@dataclass(frozen=True, slots=True)
class LayerTransitionCrystalIdentity:
    """Model/Q4/graph/atlas/projection identity without transport details."""

    model_sha256: str
    q4_sha256: str
    graph_revision_sha256: str
    atlas_revision_sha256: str
    projection: LayerTransitionProjectionIdentity
    layer_index: int = TARGET_LAYER_INDEX
    lift_abi: str = LAYER_TRANSITION_LIFT_ABI

    def __post_init__(self) -> None:
        for field in (
            "model_sha256",
            "q4_sha256",
            "graph_revision_sha256",
            "atlas_revision_sha256",
        ):
            _digest(getattr(self, field), field)
        if not isinstance(self.projection, LayerTransitionProjectionIdentity):
            raise TypeError("projection must be a LayerTransitionProjectionIdentity")
        if self.layer_index != TARGET_LAYER_INDEX:
            raise ValueError(f"only layer {TARGET_LAYER_INDEX} is supported")
        if self.lift_abi != LAYER_TRANSITION_LIFT_ABI:
            raise ValueError("quotient-lift ABI is not implemented")

    @property
    def hidden_dim(self) -> int:
        return self.projection.hidden_dim

    @property
    def sketch_dim(self) -> int:
        return self.projection.sketch_dim

    @property
    def identity_sha256(self) -> str:
        return _sha256_document(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "atlas_revision_sha256": self.atlas_revision_sha256,
            "graph_revision_sha256": self.graph_revision_sha256,
            "layer_index": self.layer_index,
            "lift_abi": self.lift_abi,
            "model_sha256": self.model_sha256,
            "projection": self.projection.to_record(),
            "q4_sha256": self.q4_sha256,
            "schema": LAYER_TRANSITION_IDENTITY_SCHEMA,
        }

    @classmethod
    def from_record(cls, value: object) -> "LayerTransitionCrystalIdentity":
        if not isinstance(value, Mapping) or set(value) != {
            "atlas_revision_sha256",
            "graph_revision_sha256",
            "layer_index",
            "lift_abi",
            "model_sha256",
            "projection",
            "q4_sha256",
            "schema",
        }:
            raise LayerTransitionCrystalIntegrityError(
                "layer-transition identity fields are invalid"
            )
        if value["schema"] != LAYER_TRANSITION_IDENTITY_SCHEMA:
            raise LayerTransitionCrystalIntegrityError(
                "layer-transition identity schema is invalid"
            )
        try:
            return cls(
                model_sha256=value["model_sha256"],
                q4_sha256=value["q4_sha256"],
                graph_revision_sha256=value["graph_revision_sha256"],
                atlas_revision_sha256=value["atlas_revision_sha256"],
                projection=LayerTransitionProjectionIdentity.from_record(
                    value["projection"]
                ),
                layer_index=value["layer_index"],
                lift_abi=value["lift_abi"],
            )
        except (TypeError, ValueError) as exc:
            raise LayerTransitionCrystalIntegrityError(
                "layer-transition identity values are invalid"
            ) from exc


@dataclass(frozen=True, slots=True)
class LayerTransitionCoverage:
    """Finite applicability ball and conservative observed error envelope."""

    center: torch.Tensor
    sketch_radius: float
    error_radius: float
    sample_count: int
    max_observed_error: float

    def __post_init__(self) -> None:
        shape, _raw = _f64_bytes(self.center, field="coverage center")
        if len(shape) != 1:
            raise ValueError("coverage center must be a vector")
        center = self.center.detach().to(device="cpu", dtype=torch.float64).clone()
        object.__setattr__(self, "center", center.contiguous())
        radius = _finite_non_negative(self.sketch_radius, "sketch_radius")
        error = _finite_non_negative(self.error_radius, "error_radius")
        observed = _finite_non_negative(
            self.max_observed_error,
            "max_observed_error",
        )
        _uint(self.sample_count, field="sample_count", positive=True)
        if error < observed:
            raise ValueError("error_radius cannot be below max_observed_error")
        object.__setattr__(self, "sketch_radius", radius)
        object.__setattr__(self, "error_radius", error)
        object.__setattr__(self, "max_observed_error", observed)

    @property
    def sketch_dim(self) -> int:
        return int(self.center.numel())

    def to_record(self) -> dict[str, object]:
        return {
            "center": _tensor_record(self.center, field="coverage center"),
            "error_radius": self.error_radius,
            "max_observed_error": self.max_observed_error,
            "sample_count": self.sample_count,
            "schema": LAYER_TRANSITION_COVERAGE_SCHEMA,
            "sketch_radius": self.sketch_radius,
        }

    @classmethod
    def from_record(cls, value: object) -> "LayerTransitionCoverage":
        if not isinstance(value, Mapping) or set(value) != {
            "center",
            "error_radius",
            "max_observed_error",
            "sample_count",
            "schema",
            "sketch_radius",
        }:
            raise LayerTransitionCrystalIntegrityError(
                "layer-transition coverage fields are invalid"
            )
        if value["schema"] != LAYER_TRANSITION_COVERAGE_SCHEMA:
            raise LayerTransitionCrystalIntegrityError(
                "layer-transition coverage schema is invalid"
            )
        try:
            return cls(
                center=_tensor_from_record(value["center"], field="coverage center"),
                sketch_radius=value["sketch_radius"],
                error_radius=value["error_radius"],
                sample_count=value["sample_count"],
                max_observed_error=value["max_observed_error"],
            )
        except (TypeError, ValueError) as exc:
            raise LayerTransitionCrystalIntegrityError(
                "layer-transition coverage values are invalid"
            ) from exc


def _hidden_batch(
    value: torch.Tensor,
    *,
    hidden_dim: int,
    field: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{field} must be a torch.Tensor")
    if value.dtype != torch.bfloat16:
        raise TypeError(f"{field} must have dtype torch.bfloat16")
    if value.layout != torch.strided:
        raise TypeError(f"{field} must be a strided tensor")
    if value.ndim == 3:
        if value.shape[1:] != (1, hidden_dim):
            raise ValueError(f"{field} must have shape [N, 1, {hidden_dim}]")
        result = (
            value.detach()
            .to(device="cpu", dtype=torch.float64)
            .reshape(value.shape[0], hidden_dim)
        )
    elif value.ndim == 2:
        if value.shape[1] != hidden_dim:
            raise ValueError(f"{field} must have shape [N, {hidden_dim}]")
        result = value.detach().to(device="cpu", dtype=torch.float64)
    else:
        raise ValueError(f"{field} must be a rank-two or rank-three batch")
    if value.shape[0] < 1:
        raise ValueError(f"{field} must contain at least one row")
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError(f"{field} contains a non-finite value")
    return result.contiguous()


def _k1_hidden(
    value: torch.Tensor, *, hidden_dim: int
) -> tuple[torch.Tensor, torch.device]:
    if not isinstance(value, torch.Tensor):
        raise TypeError("hidden must be a torch.Tensor")
    if value.dtype != torch.bfloat16:
        raise TypeError("hidden must have dtype torch.bfloat16")
    if value.layout != torch.strided:
        raise TypeError("hidden must be a strided tensor")
    if tuple(value.shape) != (1, 1, hidden_dim):
        raise ValueError(f"hidden must have exact K1 shape [1, 1, {hidden_dim}]")
    cpu = value.detach().to(device="cpu", dtype=torch.float64).reshape(hidden_dim)
    if not bool(torch.isfinite(cpu).all().item()):
        raise ValueError("hidden contains a non-finite value")
    return cpu.contiguous(), value.device


class LayerTransitionCrystal:
    """One immutable affine layer-63 quotient transition artifact."""

    __slots__ = (
        "identity",
        "logical_weight_bytes_replaced",
        "source_compute_crystal_sha256",
        "source_compute_edge_sha256",
        "_operator",
        "_bias",
        "_body_sha256",
        "_coverage",
    )

    def __init__(
        self,
        *,
        identity: LayerTransitionCrystalIdentity,
        operator: torch.Tensor,
        bias: torch.Tensor,
        coverage: LayerTransitionCoverage,
        logical_weight_bytes_replaced: int,
        source_compute_crystal_sha256: str | None = None,
        source_compute_edge_sha256: str | None = None,
    ) -> None:
        if not isinstance(identity, LayerTransitionCrystalIdentity):
            raise TypeError("identity must be a LayerTransitionCrystalIdentity")
        if not isinstance(coverage, LayerTransitionCoverage):
            raise TypeError("coverage must be a LayerTransitionCoverage")
        sketch_dim = identity.sketch_dim
        operator = operator.detach().to(device="cpu", dtype=torch.float64).clone()
        bias = bias.detach().to(device="cpu", dtype=torch.float64).clone()
        if tuple(operator.shape) != (sketch_dim, sketch_dim):
            raise ValueError(f"operator must have shape [{sketch_dim}, {sketch_dim}]")
        if tuple(bias.shape) != (sketch_dim,):
            raise ValueError(f"bias must have shape [{sketch_dim}]")
        if coverage.sketch_dim != sketch_dim:
            raise ValueError("coverage center dimension differs from projection")
        if not bool(torch.isfinite(operator).all().item()) or not bool(
            torch.isfinite(bias).all().item()
        ):
            raise ValueError("operator and bias must contain only finite values")
        logical = _uint(
            logical_weight_bytes_replaced,
            field="logical_weight_bytes_replaced",
        )
        self.identity = identity
        self._coverage = LayerTransitionCoverage(
            center=coverage.center,
            sketch_radius=coverage.sketch_radius,
            error_radius=coverage.error_radius,
            sample_count=coverage.sample_count,
            max_observed_error=coverage.max_observed_error,
        )
        self.logical_weight_bytes_replaced = logical
        if source_compute_crystal_sha256 is not None:
            source_compute_crystal_sha256 = _digest(
                source_compute_crystal_sha256,
                "source_compute_crystal_sha256",
            )
        self.source_compute_crystal_sha256 = source_compute_crystal_sha256
        if source_compute_edge_sha256 is not None:
            source_compute_edge_sha256 = _digest(
                source_compute_edge_sha256,
                "source_compute_edge_sha256",
            )
        if (source_compute_crystal_sha256 is None) != (
            source_compute_edge_sha256 is None
        ):
            raise ValueError(
                "source ComputeCrystal and edge SHA-256 must be supplied together"
            )
        self.source_compute_edge_sha256 = source_compute_edge_sha256
        self._operator = operator.contiguous()
        self._bias = bias.contiguous()
        self._body_sha256 = _sha256_document(self.to_record())

    @property
    def crystal_sha256(self) -> str:
        return self._body_sha256

    @property
    def content_sha256(self) -> str:
        return self.crystal_sha256

    @property
    def operator(self) -> torch.Tensor:
        return self._operator.clone()

    @property
    def bias(self) -> torch.Tensor:
        return self._bias.clone()

    @property
    def coverage(self) -> LayerTransitionCoverage:
        """Return a caller-owned copy of the immutable coverage envelope."""

        return LayerTransitionCoverage(
            center=self._coverage.center,
            sketch_radius=self._coverage.sketch_radius,
            error_radius=self._coverage.error_radius,
            sample_count=self._coverage.sample_count,
            max_observed_error=self._coverage.max_observed_error,
        )

    def to_record(self) -> dict[str, object]:
        return {
            "bias": _tensor_record(self._bias, field="bias"),
            "coverage": self._coverage.to_record(),
            "identity": self.identity.to_record(),
            "identity_sha256": self.identity.identity_sha256,
            "logical_weight_bytes_replaced": self.logical_weight_bytes_replaced,
            "operator": _tensor_record(self._operator, field="operator"),
            "schema": LAYER_TRANSITION_CRYSTAL_SCHEMA,
            "source_compute_crystal_sha256": self.source_compute_crystal_sha256,
            "source_compute_edge_sha256": self.source_compute_edge_sha256,
        }

    def to_bytes(self) -> bytes:
        return _sealed_document(
            self.to_record(),
            LAYER_TRANSITION_CRYSTAL_ENVELOPE_SCHEMA,
        )

    @classmethod
    def from_record(cls, value: object) -> "LayerTransitionCrystal":
        if not isinstance(value, Mapping) or set(value) != {
            "bias",
            "coverage",
            "identity",
            "identity_sha256",
            "logical_weight_bytes_replaced",
            "operator",
            "schema",
            "source_compute_crystal_sha256",
            "source_compute_edge_sha256",
        }:
            raise LayerTransitionCrystalIntegrityError(
                "layer-transition crystal fields are invalid"
            )
        if value["schema"] != LAYER_TRANSITION_CRYSTAL_SCHEMA:
            raise LayerTransitionCrystalIntegrityError(
                "layer-transition crystal schema is invalid"
            )
        try:
            identity = LayerTransitionCrystalIdentity.from_record(value["identity"])
            if value["identity_sha256"] != identity.identity_sha256:
                raise ValueError("crystal identity SHA-256 mismatch")
            return cls(
                identity=identity,
                operator=_tensor_from_record(value["operator"], field="operator"),
                bias=_tensor_from_record(value["bias"], field="bias"),
                coverage=LayerTransitionCoverage.from_record(value["coverage"]),
                logical_weight_bytes_replaced=value["logical_weight_bytes_replaced"],
                source_compute_crystal_sha256=value["source_compute_crystal_sha256"],
                source_compute_edge_sha256=value["source_compute_edge_sha256"],
            )
        except (TypeError, ValueError) as exc:
            raise LayerTransitionCrystalIntegrityError(
                "layer-transition crystal values are invalid"
            ) from exc

    @classmethod
    def from_bytes(cls, value: bytes) -> "LayerTransitionCrystal":
        body = _unseal_document(
            value,
            schema=LAYER_TRANSITION_CRYSTAL_ENVELOPE_SCHEMA,
            kind="layer-transition crystal",
        )
        return cls.from_record(body)

    @classmethod
    def fit(
        cls,
        *,
        identity: LayerTransitionCrystalIdentity,
        source_hidden: torch.Tensor,
        target_hidden: torch.Tensor,
        logical_weight_bytes_replaced: int,
        ridge: float = 1e-8,
        coverage_guard: float = 0.0,
        error_guard: float = 0.0,
        source_compute_crystal_sha256: str | None = None,
        source_compute_edge_sha256: str | None = None,
    ) -> "LayerTransitionCrystal":
        """Fit one affine quotient operator and derive conservative fields.

        ``error_radius`` is the maximum BF16 replacement residual observed on
        the fit rows plus ``error_guard``.  ``sketch_radius`` is the maximum
        source-sketch distance from the fit centroid plus ``coverage_guard``.
        Both values are rounded upward to the next representable float.
        """

        if not isinstance(identity, LayerTransitionCrystalIdentity):
            raise TypeError("identity must be a LayerTransitionCrystalIdentity")
        ridge_value = _finite_non_negative(ridge, "ridge")
        coverage_guard_value = _finite_non_negative(
            coverage_guard,
            "coverage_guard",
        )
        error_guard_value = _finite_non_negative(error_guard, "error_guard")
        source = _hidden_batch(
            source_hidden,
            hidden_dim=identity.hidden_dim,
            field="source_hidden",
        )
        target = _hidden_batch(
            target_hidden,
            hidden_dim=identity.hidden_dim,
            field="target_hidden",
        )
        if source.shape != target.shape:
            raise ValueError("source_hidden and target_hidden shapes differ")
        projection = _projection_matrix_cached(
            identity.hidden_dim,
            identity.sketch_dim,
            identity.projection.seed_sha256,
            identity.projection.abi,
        )
        pinv = _projection_pseudoinverse_cached(
            identity.hidden_dim,
            identity.sketch_dim,
            identity.projection.seed_sha256,
            identity.projection.abi,
        )
        source_sketch = source @ projection
        target_sketch = target @ projection
        ones = torch.ones((source.shape[0], 1), dtype=torch.float64)
        design = torch.cat((source_sketch, ones), dim=1)
        gram = design.T @ design
        regulariser = torch.eye(gram.shape[0], dtype=torch.float64) * ridge_value
        regulariser[-1, -1] = 0.0
        try:
            coefficients = torch.linalg.solve(
                gram + regulariser,
                design.T @ target_sketch,
            )
        except torch.linalg.LinAlgError:
            coefficients = torch.linalg.pinv(gram + regulariser) @ (
                design.T @ target_sketch
            )
        operator = coefficients[:-1].contiguous()
        bias = coefficients[-1].contiguous()
        predicted_sketch = source_sketch @ operator + bias
        lifted = source + (predicted_sketch - source_sketch) @ pinv
        lifted_bf16 = lifted.to(dtype=torch.bfloat16).to(dtype=torch.float64)
        residuals = torch.linalg.vector_norm(lifted_bf16 - target, dim=1)
        max_observed = float(residuals.max().item())
        center = source_sketch.mean(dim=0)
        distances = torch.linalg.vector_norm(source_sketch - center, dim=1)
        sketch_radius = math.nextafter(
            float(distances.max().item()) + coverage_guard_value,
            math.inf,
        )
        error_radius = math.nextafter(
            max_observed + error_guard_value,
            math.inf,
        )
        return cls(
            identity=identity,
            operator=operator,
            bias=bias,
            coverage=LayerTransitionCoverage(
                center=center,
                sketch_radius=sketch_radius,
                error_radius=error_radius,
                sample_count=int(source.shape[0]),
                max_observed_error=max_observed,
            ),
            logical_weight_bytes_replaced=logical_weight_bytes_replaced,
            source_compute_crystal_sha256=source_compute_crystal_sha256,
            source_compute_edge_sha256=source_compute_edge_sha256,
        )

    @classmethod
    def calibrate_affine(
        cls,
        *,
        identity: LayerTransitionCrystalIdentity,
        operator: torch.Tensor,
        bias: torch.Tensor,
        source_hidden: torch.Tensor,
        target_hidden: torch.Tensor,
        logical_weight_bytes_replaced: int,
        source_compute_crystal_sha256: str | None = None,
        source_compute_edge_sha256: str | None = None,
        coverage_guard: float = 0.0,
        error_guard: float = 0.0,
    ) -> "LayerTransitionCrystal":
        """Calibrate one fixed O1 affine operator on exact Q4 hidden rows."""

        if not isinstance(identity, LayerTransitionCrystalIdentity):
            raise TypeError("identity must be a LayerTransitionCrystalIdentity")
        coverage_guard_value = _finite_non_negative(
            coverage_guard,
            "coverage_guard",
        )
        error_guard_value = _finite_non_negative(error_guard, "error_guard")
        source = _hidden_batch(
            source_hidden,
            hidden_dim=identity.hidden_dim,
            field="source_hidden",
        )
        target = _hidden_batch(
            target_hidden,
            hidden_dim=identity.hidden_dim,
            field="target_hidden",
        )
        if source.shape != target.shape:
            raise ValueError("source_hidden and target_hidden shapes differ")
        weight = operator.detach().to(device="cpu", dtype=torch.float64).clone()
        offset = bias.detach().to(device="cpu", dtype=torch.float64).clone()
        if tuple(weight.shape) != (identity.sketch_dim, identity.sketch_dim):
            raise ValueError("operator shape differs from the quotient projection")
        if tuple(offset.shape) != (identity.sketch_dim,):
            raise ValueError("bias shape differs from the quotient projection")
        if not bool(torch.isfinite(weight).all().item()) or not bool(
            torch.isfinite(offset).all().item()
        ):
            raise ValueError("operator and bias must contain only finite values")
        projection = _projection_matrix_cached(
            identity.hidden_dim,
            identity.sketch_dim,
            identity.projection.seed_sha256,
            identity.projection.abi,
        )
        pinv = _projection_pseudoinverse_cached(
            identity.hidden_dim,
            identity.sketch_dim,
            identity.projection.seed_sha256,
            identity.projection.abi,
        )
        source_sketch = source @ projection
        predicted_sketch = source_sketch @ weight + offset
        lifted = source + (predicted_sketch - source_sketch) @ pinv
        lifted_bf16 = lifted.to(dtype=torch.bfloat16).to(dtype=torch.float64)
        residuals = torch.linalg.vector_norm(lifted_bf16 - target, dim=1)
        max_observed = float(residuals.max().item())
        center = source_sketch.mean(dim=0)
        distances = torch.linalg.vector_norm(source_sketch - center, dim=1)
        return cls(
            identity=identity,
            operator=weight,
            bias=offset,
            coverage=LayerTransitionCoverage(
                center=center,
                sketch_radius=math.nextafter(
                    float(distances.max().item()) + coverage_guard_value,
                    math.inf,
                ),
                error_radius=math.nextafter(
                    max_observed + error_guard_value,
                    math.inf,
                ),
                sample_count=int(source.shape[0]),
                max_observed_error=max_observed,
            ),
            logical_weight_bytes_replaced=logical_weight_bytes_replaced,
            source_compute_crystal_sha256=source_compute_crystal_sha256,
            source_compute_edge_sha256=source_compute_edge_sha256,
        )

    def coverage_distance(self, hidden: torch.Tensor) -> float:
        vector, _device = _k1_hidden(hidden, hidden_dim=self.identity.hidden_dim)
        projection = _projection_matrix_cached(
            self.identity.hidden_dim,
            self.identity.sketch_dim,
            self.identity.projection.seed_sha256,
            self.identity.projection.abi,
        )
        sketch = vector @ projection
        return float(torch.linalg.vector_norm(sketch - self._coverage.center).item())

    def eligible(self, hidden: torch.Tensor, *, max_error_radius: float) -> bool:
        allowed = _finite_non_negative(max_error_radius, "max_error_radius")
        distance = self.coverage_distance(hidden)
        return (
            distance <= self._coverage.sketch_radius
            and self._coverage.error_radius <= allowed
        )

    def apply(
        self,
        hidden: torch.Tensor,
        *,
        max_error_radius: float,
    ) -> "LayerTransitionReplacement | None":
        allowed = _finite_non_negative(max_error_radius, "max_error_radius")
        vector, device = _k1_hidden(hidden, hidden_dim=self.identity.hidden_dim)
        projection = _projection_matrix_cached(
            self.identity.hidden_dim,
            self.identity.sketch_dim,
            self.identity.projection.seed_sha256,
            self.identity.projection.abi,
        )
        pinv = _projection_pseudoinverse_cached(
            self.identity.hidden_dim,
            self.identity.sketch_dim,
            self.identity.projection.seed_sha256,
            self.identity.projection.abi,
        )
        sketch = vector @ projection
        distance = float(
            torch.linalg.vector_norm(sketch - self._coverage.center).item()
        )
        if (
            distance > self._coverage.sketch_radius
            or self._coverage.error_radius > allowed
        ):
            return None
        predicted_sketch = sketch @ self._operator + self._bias
        lifted = vector + (predicted_sketch - sketch) @ pinv
        if not bool(torch.isfinite(lifted).all().item()):
            raise LayerTransitionCrystalIntegrityError(
                "quotient lift produced a non-finite hidden row"
            )
        output = lifted.reshape(1, 1, self.identity.hidden_dim).to(
            device=device,
            dtype=torch.bfloat16,
        )
        if not bool(torch.isfinite(output).all().item()):
            raise LayerTransitionCrystalIntegrityError(
                "BF16 quotient lift produced a non-finite hidden row"
            )
        return LayerTransitionReplacement(
            output=output,
            crystal_sha256=self.crystal_sha256,
            coverage_distance=distance,
            coverage_radius=self._coverage.sketch_radius,
            error_radius=self._coverage.error_radius,
            logical_weight_bytes_replaced=self.logical_weight_bytes_replaced,
        )


@dataclass(frozen=True, slots=True)
class LayerTransitionReplacement:
    """One successful physical layer-transition replacement."""

    output: torch.Tensor
    crystal_sha256: str
    coverage_distance: float
    coverage_radius: float
    error_radius: float
    logical_weight_bytes_replaced: int

    def __post_init__(self) -> None:
        _digest(self.crystal_sha256, "crystal_sha256")
        if not isinstance(self.output, torch.Tensor):
            raise TypeError("replacement output must be a torch.Tensor")
        if self.output.dtype != torch.bfloat16 or self.output.ndim != 3:
            raise ValueError("replacement output must be a rank-three BF16 tensor")
        _finite_non_negative(self.coverage_distance, "coverage_distance")
        _finite_non_negative(self.coverage_radius, "coverage_radius")
        _finite_non_negative(self.error_radius, "error_radius")
        _uint(
            self.logical_weight_bytes_replaced,
            field="logical_weight_bytes_replaced",
        )

    @property
    def hidden(self) -> torch.Tensor:
        return self.output


@dataclass(frozen=True, slots=True)
class LayerTransitionCrystalMetrics:
    identity_sha256: str
    crystal_count: int
    attempts: int
    replacements: int
    fallbacks: int
    logical_weight_bytes_replaced: int
    output_bytes_emitted: int

    def __post_init__(self) -> None:
        _digest(self.identity_sha256, "identity_sha256")
        for field in (
            "crystal_count",
            "attempts",
            "replacements",
            "fallbacks",
            "logical_weight_bytes_replaced",
            "output_bytes_emitted",
        ):
            _uint(getattr(self, field), field=field)
        if self.replacements + self.fallbacks != self.attempts:
            raise ValueError("attempt metrics do not settle exactly once")

    @property
    def bytes_replaced(self) -> int:
        return self.logical_weight_bytes_replaced

    def to_dict(self) -> dict[str, object]:
        return {
            "attempts": self.attempts,
            "bytes_replaced": self.logical_weight_bytes_replaced,
            "crystal_count": self.crystal_count,
            "fallbacks": self.fallbacks,
            "identity_sha256": self.identity_sha256,
            "logical_weight_bytes_replaced": self.logical_weight_bytes_replaced,
            "output_bytes_emitted": self.output_bytes_emitted,
            "replacements": self.replacements,
        }


def _stable_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _stable_regular_bytes(
    path: Path,
) -> tuple[bytes, tuple[int, int, int, int, int]]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(
            os,
            "O_NOFOLLOW",
            0,
        )
    )
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise LayerTransitionCrystalIntegrityError(
            "layer-transition bank cannot be opened"
        ) from exc
    try:
        before = os.fstat(descriptor)
        linked_before = os.lstat(path)
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(linked_before.st_mode)
            or not _same_inode(before, linked_before)
        ):
            raise LayerTransitionCrystalIntegrityError(
                "layer-transition bank must be a stable regular file"
            )
        if before.st_size < 0 or before.st_size > _MAX_STATE_BYTES:
            raise LayerTransitionCrystalIntegrityError(
                "layer-transition bank exceeds its byte limit"
            )
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, _READ_CHUNK_BYTES))
            if not chunk:
                raise LayerTransitionCrystalIntegrityError(
                    "layer-transition bank was truncated"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise LayerTransitionCrystalIntegrityError(
                "layer-transition bank grew while reading"
            )
        after = os.fstat(descriptor)
        linked_after = os.lstat(path)
        if (
            _stable_signature(before) != _stable_signature(after)
            or _stable_signature(linked_before) != _stable_signature(linked_after)
            or not _same_inode(after, linked_after)
        ):
            raise LayerTransitionCrystalIntegrityError(
                "layer-transition bank changed while reading"
            )
        return b"".join(chunks), _stable_signature(after)
    except OSError as exc:
        raise LayerTransitionCrystalIntegrityError(
            "layer-transition bank cannot be read safely"
        ) from exc
    finally:
        os.close(descriptor)


_THREAD_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, threading.RLock] = {}


def _thread_lock(path: Path) -> threading.RLock:
    key = os.path.abspath(os.fspath(path))
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.RLock())


def _validate_parent(path: Path) -> tuple[Path, os.stat_result]:
    parent = path.parent
    try:
        value = os.lstat(parent)
    except OSError as exc:
        raise LayerTransitionCrystalIntegrityError(
            "layer-transition bank parent is unavailable"
        ) from exc
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
        raise LayerTransitionCrystalIntegrityError(
            "layer-transition bank parent must be a directory"
        )
    return parent, value


@contextmanager
def _exclusive_state_lock(path: Path) -> Iterator[None]:
    lock = _thread_lock(path)
    with lock:
        parent, parent_before = _validate_parent(path)
        lock_path = parent / f".{path.name}.lock"
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            raise LayerTransitionCrystalIntegrityError(
                "layer-transition bank lock is unavailable"
            ) from exc
        try:
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            opened = os.fstat(descriptor)
            linked = os.lstat(lock_path)
            if (
                not stat.S_ISREG(opened.st_mode)
                or not stat.S_ISREG(linked.st_mode)
                or not _same_inode(opened, linked)
            ):
                raise LayerTransitionCrystalIntegrityError(
                    "layer-transition bank lock is not a stable regular file"
                )
            yield
            parent_after = os.lstat(parent)
            linked_after = os.lstat(lock_path)
            if not _same_inode(opened, linked_after) or not _same_inode(
                parent_before, parent_after
            ):
                raise LayerTransitionCrystalIntegrityError(
                    "layer-transition bank lock changed during transaction"
                )
        except OSError as exc:
            raise LayerTransitionCrystalIntegrityError(
                "layer-transition bank lock failed"
            ) from exc
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def _atomic_write(path: Path, data: bytes) -> None:
    parent, parent_before = _validate_parent(path)
    try:
        current = os.lstat(path)
    except FileNotFoundError:
        current = None
    except OSError as exc:
        raise LayerTransitionCrystalIntegrityError(
            "layer-transition bank destination is unavailable"
        ) from exc
    if current is not None and (
        stat.S_ISLNK(current.st_mode) or not stat.S_ISREG(current.st_mode)
    ):
        raise LayerTransitionCrystalIntegrityError(
            "layer-transition bank destination must be a regular file"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("zero-byte bank write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        if not _same_inode(parent_before, os.lstat(parent)):
            raise LayerTransitionCrystalIntegrityError(
                "layer-transition bank parent changed before publication"
            )
        try:
            destination = os.lstat(path)
        except FileNotFoundError:
            destination = None
        if destination is not None and (
            stat.S_ISLNK(destination.st_mode) or not stat.S_ISREG(destination.st_mode)
        ):
            raise LayerTransitionCrystalIntegrityError(
                "layer-transition bank destination changed type"
            )
        os.replace(temporary, path)
        directory = os.open(
            parent,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        published = os.lstat(path)
        if not stat.S_ISREG(published.st_mode):
            raise LayerTransitionCrystalIntegrityError(
                "published layer-transition bank is not regular"
            )
    except LayerTransitionCrystalError:
        raise
    except OSError as exc:
        raise LayerTransitionCrystalIntegrityError(
            "layer-transition bank could not be published atomically"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@dataclass(frozen=True, slots=True)
class _LayerTransitionBankState:
    identity: LayerTransitionCrystalIdentity
    max_crystals: int
    generation: int = 0
    publications: int = 0
    crystals: tuple[LayerTransitionCrystal, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.identity, LayerTransitionCrystalIdentity):
            raise TypeError("bank identity is invalid")
        _uint(
            self.max_crystals,
            field="max_crystals",
            positive=True,
            maximum=_MAX_CRYSTALS,
        )
        _uint(self.generation, field="generation")
        _uint(self.publications, field="publications")
        crystals = tuple(self.crystals)
        if len(crystals) > self.max_crystals:
            raise ValueError("bank exceeds max_crystals")
        if tuple(sorted(crystals, key=lambda item: item.crystal_sha256)) != crystals:
            raise ValueError("bank crystals are not canonically ordered")
        if len({item.crystal_sha256 for item in crystals}) != len(crystals):
            raise ValueError("bank contains duplicate crystals")
        if any(
            item.identity.identity_sha256 != self.identity.identity_sha256
            for item in crystals
        ):
            raise ValueError("bank contains a foreign crystal identity")
        object.__setattr__(self, "crystals", crystals)

    def to_record(self) -> dict[str, object]:
        return {
            "crystals": [crystal.to_record() for crystal in self.crystals],
            "generation": self.generation,
            "identity": self.identity.to_record(),
            "identity_sha256": self.identity.identity_sha256,
            "max_crystals": self.max_crystals,
            "publications": self.publications,
            "schema": LAYER_TRANSITION_BANK_SCHEMA,
        }

    def to_bytes(self) -> bytes:
        encoded = _sealed_document(
            self.to_record(),
            LAYER_TRANSITION_BANK_ENVELOPE_SCHEMA,
        )
        if len(encoded) > _MAX_STATE_BYTES:
            raise LayerTransitionCrystalIntegrityError(
                "layer-transition bank exceeds its byte limit"
            )
        return encoded

    @classmethod
    def from_bytes(cls, value: bytes) -> "_LayerTransitionBankState":
        body = _unseal_document(
            value,
            schema=LAYER_TRANSITION_BANK_ENVELOPE_SCHEMA,
            kind="layer-transition bank",
        )
        if set(body) != {
            "crystals",
            "generation",
            "identity",
            "identity_sha256",
            "max_crystals",
            "publications",
            "schema",
        }:
            raise LayerTransitionCrystalIntegrityError(
                "layer-transition bank fields are invalid"
            )
        if body["schema"] != LAYER_TRANSITION_BANK_SCHEMA:
            raise LayerTransitionCrystalIntegrityError(
                "layer-transition bank schema is invalid"
            )
        try:
            identity = LayerTransitionCrystalIdentity.from_record(body["identity"])
            if body["identity_sha256"] != identity.identity_sha256:
                raise ValueError("bank identity SHA-256 mismatch")
            if isinstance(body["crystals"], (str, bytes, bytearray)) or not isinstance(
                body["crystals"], Sequence
            ):
                raise TypeError("bank crystals are not a sequence")
            return cls(
                identity=identity,
                max_crystals=body["max_crystals"],
                generation=body["generation"],
                publications=body["publications"],
                crystals=tuple(
                    LayerTransitionCrystal.from_record(item)
                    for item in body["crystals"]
                ),
            )
        except (TypeError, ValueError) as exc:
            raise LayerTransitionCrystalIntegrityError(
                "layer-transition bank values are invalid"
            ) from exc


class LayerTransitionCrystalBank:
    """Persistent bounded layer-63 crystal bank with runtime economics."""

    def __init__(
        self,
        state_path: str | os.PathLike[str],
        identity: LayerTransitionCrystalIdentity,
        *,
        max_crystals: int = 64,
    ) -> None:
        if not isinstance(identity, LayerTransitionCrystalIdentity):
            raise TypeError("identity must be a LayerTransitionCrystalIdentity")
        self.state_path = Path(state_path)
        if not self.state_path.name:
            raise ValueError("state_path must name a file")
        _uint(
            max_crystals,
            field="max_crystals",
            positive=True,
            maximum=_MAX_CRYSTALS,
        )
        self.identity = identity
        self.max_crystals = max_crystals
        self._lock = _thread_lock(self.state_path)
        self._file_signature: tuple[int, int, int, int, int] | None = None
        self._had_persistent_state = False
        self._state = _LayerTransitionBankState(
            identity=identity,
            max_crystals=max_crystals,
        )
        self._attempts = 0
        self._replacements = 0
        self._fallbacks = 0
        self._logical_weight_bytes_replaced = 0
        self._output_bytes_emitted = 0
        with self._lock:
            self._reload(required=False)

    @classmethod
    def load(
        cls,
        state_path: str | os.PathLike[str],
    ) -> "LayerTransitionCrystalBank":
        """Open an existing bank using the identity stored in its sealed state."""

        path = Path(state_path)
        raw, signature = _stable_regular_bytes(path)
        state = _LayerTransitionBankState.from_bytes(raw)
        bank = cls.__new__(cls)
        bank.state_path = path
        bank.identity = state.identity
        bank.max_crystals = state.max_crystals
        bank._lock = _thread_lock(path)
        bank._file_signature = signature
        bank._had_persistent_state = True
        bank._state = state
        bank._attempts = 0
        bank._replacements = 0
        bank._fallbacks = 0
        bank._logical_weight_bytes_replaced = 0
        bank._output_bytes_emitted = 0
        return bank

    def _validate_state(self, state: _LayerTransitionBankState) -> None:
        if state.identity.identity_sha256 != self.identity.identity_sha256:
            raise LayerTransitionCrystalIdentityError(
                "layer-transition bank belongs to a different identity"
            )
        if state.max_crystals != self.max_crystals:
            raise LayerTransitionCrystalIdentityError(
                "layer-transition bank max_crystals differs from persistent state"
            )

    def _reload(self, *, required: bool) -> None:
        try:
            raw, signature = _stable_regular_bytes(self.state_path)
        except FileNotFoundError:
            if required or self._had_persistent_state:
                raise LayerTransitionCrystalIntegrityError(
                    "layer-transition bank disappeared"
                )
            self._file_signature = None
            self._state = _LayerTransitionBankState(
                identity=self.identity,
                max_crystals=self.max_crystals,
            )
            return
        state = _LayerTransitionBankState.from_bytes(raw)
        self._validate_state(state)
        self._state = state
        self._file_signature = signature
        self._had_persistent_state = True

    def _refresh_if_changed(self) -> None:
        try:
            linked = os.lstat(self.state_path)
        except FileNotFoundError:
            if self._had_persistent_state:
                raise LayerTransitionCrystalIntegrityError(
                    "layer-transition bank disappeared"
                )
            return
        if stat.S_ISLNK(linked.st_mode) or not stat.S_ISREG(linked.st_mode):
            raise LayerTransitionCrystalIntegrityError(
                "layer-transition bank must be a regular file"
            )
        if _stable_signature(linked) == self._file_signature:
            return
        self._reload(required=True)

    def publish(self, crystal: LayerTransitionCrystal) -> str:
        """Publish one immutable crystal in one locked atomic replacement."""

        if not isinstance(crystal, LayerTransitionCrystal):
            raise TypeError("crystal must be a LayerTransitionCrystal")
        if crystal.identity.identity_sha256 != self.identity.identity_sha256:
            raise LayerTransitionCrystalIdentityError(
                "cannot publish a crystal from another identity"
            )
        with _exclusive_state_lock(self.state_path):
            self._reload(required=self._had_persistent_state)
            if any(
                item.crystal_sha256 == crystal.crystal_sha256
                for item in self._state.crystals
            ):
                return crystal.crystal_sha256
            if len(self._state.crystals) >= self.max_crystals:
                raise LayerTransitionCrystalCapacityError(
                    "layer-transition bank is full"
                )
            next_state = _LayerTransitionBankState(
                identity=self.identity,
                max_crystals=self.max_crystals,
                generation=_bounded_add(self._state.generation, 1),
                publications=_bounded_add(self._state.publications, 1),
                crystals=tuple(
                    sorted(
                        (*self._state.crystals, crystal),
                        key=lambda item: item.crystal_sha256,
                    )
                ),
            )
            _atomic_write(self.state_path, next_state.to_bytes())
            self._state = next_state
            published = os.lstat(self.state_path)
            self._file_signature = _stable_signature(published)
            self._had_persistent_state = True
        return crystal.crystal_sha256

    def replace(
        self,
        hidden: torch.Tensor,
        *,
        max_error_radius: float,
    ) -> LayerTransitionReplacement | None:
        """Try the tightest eligible crystal and settle one runtime attempt."""

        replacements = self.replace_many(
            (hidden,),
            max_error_radius=max_error_radius,
        )
        return None if replacements is None else replacements[0]

    def replace_many(
        self,
        hidden: Sequence[torch.Tensor],
        *,
        max_error_radius: float,
    ) -> tuple[LayerTransitionReplacement, ...] | None:
        """Replace one K1-K16 wave atomically or settle the entire wave as fallback."""

        allowed = _finite_non_negative(max_error_radius, "max_error_radius")
        if isinstance(hidden, (str, bytes, bytearray)):
            raise TypeError("hidden rows must be a tensor sequence")
        try:
            rows = tuple(hidden)
        except TypeError as exc:
            raise TypeError("hidden rows must be a tensor sequence") from exc
        if not 1 <= len(rows) <= 16:
            raise ValueError("hidden rows must contain one to sixteen K1 rows")
        # Validate before charging an attempt: malformed caller input is a
        # contract violation, not a runtime fallback.
        for row in rows:
            _k1_hidden(row, hidden_dim=self.identity.hidden_dim)
        with self._lock:
            self._refresh_if_changed()
            selected: list[LayerTransitionCrystal] = []
            for row in rows:
                ranked: list[tuple[float, float, str, LayerTransitionCrystal]] = []
                for crystal in self._state.crystals:
                    distance = crystal.coverage_distance(row)
                    if (
                        distance <= crystal._coverage.sketch_radius
                        and crystal._coverage.error_radius <= allowed
                    ):
                        ranked.append(
                            (
                                crystal._coverage.error_radius,
                                distance,
                                crystal.crystal_sha256,
                                crystal,
                            )
                        )
                if not ranked:
                    self._attempts = _bounded_add(self._attempts, len(rows))
                    self._fallbacks = _bounded_add(self._fallbacks, len(rows))
                    return None
                selected.append(min(ranked)[-1])
            self._attempts = _bounded_add(self._attempts, len(rows))
            try:
                replacements = tuple(
                    crystal.apply(row, max_error_radius=allowed)
                    for crystal, row in zip(selected, rows, strict=True)
                )
            except Exception:
                self._fallbacks = _bounded_add(self._fallbacks, len(rows))
                raise
            if any(replacement is None for replacement in replacements):
                self._fallbacks = _bounded_add(self._fallbacks, len(rows))
                raise LayerTransitionCrystalIntegrityError(
                    "eligible crystal rejected the same hidden row"
                )
            settled = tuple(
                replacement for replacement in replacements if replacement is not None
            )
            self._replacements = _bounded_add(self._replacements, len(rows))
            self._logical_weight_bytes_replaced = _bounded_add(
                self._logical_weight_bytes_replaced,
                sum(row.logical_weight_bytes_replaced for row in settled),
            )
            self._output_bytes_emitted = _bounded_add(
                self._output_bytes_emitted,
                sum(row.output.numel() * row.output.element_size() for row in settled),
            )
            return settled

    def try_replace(
        self,
        hidden: torch.Tensor,
        *,
        max_error_radius: float,
    ) -> LayerTransitionReplacement | None:
        return self.replace(hidden, max_error_radius=max_error_radius)

    def metrics(self) -> LayerTransitionCrystalMetrics:
        with self._lock:
            self._refresh_if_changed()
            return LayerTransitionCrystalMetrics(
                identity_sha256=self.identity.identity_sha256,
                crystal_count=len(self._state.crystals),
                attempts=self._attempts,
                replacements=self._replacements,
                fallbacks=self._fallbacks,
                logical_weight_bytes_replaced=(self._logical_weight_bytes_replaced),
                output_bytes_emitted=self._output_bytes_emitted,
            )

    @property
    def crystals(self) -> tuple[LayerTransitionCrystal, ...]:
        with self._lock:
            self._refresh_if_changed()
            return tuple(self._state.crystals)


__all__ = [
    "LAYER_TRANSITION_LIFT_ABI",
    "LAYER_TRANSITION_PROJECTION_ABI",
    "LayerTransitionCoverage",
    "LayerTransitionCrystal",
    "LayerTransitionCrystalBank",
    "LayerTransitionCrystalCapacityError",
    "LayerTransitionCrystalError",
    "LayerTransitionCrystalIdentity",
    "LayerTransitionCrystalIdentityError",
    "LayerTransitionCrystalIntegrityError",
    "LayerTransitionCrystalMetrics",
    "LayerTransitionProjectionIdentity",
    "LayerTransitionReplacement",
    "TARGET_LAYER_INDEX",
]
