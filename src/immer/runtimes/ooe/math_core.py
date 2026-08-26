"""Numerically bounded Foss Markov-kernel primitives.

The formulas in this module are the production form of the original
``foss-markov-kernel-poc`` math core.  Every dense operation is explicitly
bounded, canonical hashes include shape and dtype, and public diagnostics are
JSON-serializable without a NumPy-aware encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]

DEFAULT_MAX_NODES = 4_096
DEFAULT_MAX_DENSE_BYTES = 256 * 1024 * 1024


def _positive_int(value: int, *, name: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    lower = 0 if allow_zero else 1
    if value < lower:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _positive_float(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _dense_budget(
    shape: tuple[int, ...],
    *,
    max_bytes: int,
    arrays: int = 1,
) -> int:
    limit = _positive_int(max_bytes, name="max_bytes")
    count = math.prod(shape)
    required = count * np.dtype(np.float64).itemsize * arrays
    if required > limit:
        raise ValueError(
            f"dense working set requires {required} bytes, exceeding {limit}"
        )
    return required


def as_float_matrix(
    matrix: ArrayLike,
    *,
    name: str = "matrix",
    square: bool = False,
    nonnegative: bool = False,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
    working_arrays: int = 1,
) -> FloatArray:
    """Return a finite, C-contiguous float64 matrix under a dense-memory cap."""

    node_limit = _positive_int(max_nodes, name="max_nodes")
    raw = np.asarray(matrix)
    if raw.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 matrix")
    rows, columns = (int(raw.shape[0]), int(raw.shape[1]))
    if rows < 1 or columns < 1:
        raise ValueError(f"{name} must not be empty")
    if square and rows != columns:
        raise ValueError(f"{name} must be square")
    if max(rows, columns) > node_limit:
        raise ValueError(
            f"{name} dimension {max(rows, columns)} exceeds max_nodes={node_limit}"
        )
    _dense_budget(
        (rows, columns),
        max_bytes=max_bytes,
        arrays=_positive_int(working_arrays, name="working_arrays"),
    )
    try:
        value = np.asarray(raw, dtype=np.float64, order="C")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numbers") from exc
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    if nonnegative and np.any(value < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return value


def _as_square(
    matrix: ArrayLike,
    *,
    nonnegative: bool = False,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
    working_arrays: int = 1,
) -> FloatArray:
    return as_float_matrix(
        matrix,
        square=True,
        nonnegative=nonnegative,
        max_nodes=max_nodes,
        max_bytes=max_bytes,
        working_arrays=working_arrays,
    )


def array_sha256(
    array: ArrayLike,
    *,
    max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
) -> str:
    """Hash a numeric array canonically as little-endian float64 plus shape."""

    raw = np.asarray(array)
    if raw.ndim < 1:
        raise ValueError("array must have at least one dimension")
    _dense_budget(tuple(int(value) for value in raw.shape), max_bytes=max_bytes)
    try:
        value = np.asarray(raw, dtype="<f8", order="C")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("array must contain real numbers") from exc
    if not np.isfinite(value).all():
        raise ValueError("array must contain only finite values")
    digest = hashlib.sha256()
    digest.update(b"immer-ooe-f64-array/v1\0")
    digest.update(struct.pack("<Q", value.ndim))
    for dimension in value.shape:
        digest.update(struct.pack("<Q", int(dimension)))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def normalize_rows(
    matrix: ArrayLike,
    *,
    eps: float = 1e-15,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
) -> FloatArray:
    """Normalize non-negative rows; empty rows become uniform distributions."""

    threshold = _positive_float(eps, name="eps")
    value = as_float_matrix(
        matrix,
        nonnegative=True,
        max_nodes=max_nodes,
        max_bytes=max_bytes,
        working_arrays=3,
    ).copy()
    totals = value.sum(axis=1, keepdims=True)
    empty = totals[:, 0] <= threshold
    if np.any(empty):
        value[empty] = 1.0 / value.shape[1]
        totals = value.sum(axis=1, keepdims=True)
    return np.ascontiguousarray(value / totals, dtype=np.float64)


def sinkhorn_project(
    matrix: ArrayLike,
    *,
    rounds: int = 50,
    eps: float = 1e-15,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
) -> FloatArray:
    """Project a non-negative square matrix toward the Birkhoff polytope."""

    iterations = _positive_int(rounds, name="rounds")
    floor = _positive_float(eps, name="eps")
    value = _as_square(
        matrix,
        nonnegative=True,
        max_nodes=max_nodes,
        max_bytes=max_bytes,
        working_arrays=4,
    ).copy()
    np.maximum(value, floor, out=value)
    for _ in range(iterations):
        value /= value.sum(axis=1, keepdims=True)
        value /= value.sum(axis=0, keepdims=True)
    value /= value.sum(axis=1, keepdims=True)
    if not np.isfinite(value).all():
        raise ArithmeticError("Sinkhorn projection produced a non-finite value")
    return np.ascontiguousarray(value)


def sinkhorn_rg_cool(
    matrix: ArrayLike,
    *,
    alpha: float,
    rounds: int = 3,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
) -> FloatArray:
    """Mix with the uniform fixed point, then apply short Sinkhorn flow."""

    mixing = float(alpha)
    if not math.isfinite(mixing) or not 0.0 <= mixing <= 1.0:
        raise ValueError("alpha must be finite and lie in [0, 1]")
    iterations = _positive_int(rounds, name="rounds")
    value = _as_square(
        matrix,
        nonnegative=True,
        max_nodes=max_nodes,
        max_bytes=max_bytes,
        working_arrays=6,
    )
    size = value.shape[0]
    base = sinkhorn_project(
        value,
        rounds=max(10, iterations),
        max_nodes=max_nodes,
        max_bytes=max_bytes,
    )
    mixed = (1.0 - mixing) * base + mixing / size
    return sinkhorn_project(
        mixed,
        rounds=iterations,
        max_nodes=max_nodes,
        max_bytes=max_bytes,
    )


def mobius_compose(left: float, right: float, *, eps: float = 1e-12) -> float:
    """Foss/Mobius composition ``f(a,b)=(a+b)/(1+ab)``."""

    floor = _positive_float(eps, name="eps")
    if floor >= 1.0:
        raise ValueError("eps must be smaller than one")
    a = float(left)
    b = float(right)
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("couplings must be finite")
    a = float(np.clip(a, -1.0 + floor, 1.0 - floor))
    b = float(np.clip(b, -1.0 + floor, 1.0 - floor))
    return float((a + b) / (1.0 + a * b))


@dataclass(slots=True)
class RapidityLedger:
    """Additive ``xi=atanh(lambda)`` ledger for Möbius composition."""

    xi: float = 0.0
    limit: float = 20.0

    def __post_init__(self) -> None:
        self.xi = float(self.xi)
        self.limit = _positive_float(self.limit, name="limit")
        if not math.isfinite(self.xi) or abs(self.xi) > self.limit:
            raise ValueError("xi must be finite and within the ledger limit")

    def add(self, coupling: float) -> float:
        value = float(coupling)
        if not math.isfinite(value):
            raise ValueError("coupling must be finite")
        clipped = float(np.clip(value, -1.0 + 1e-12, 1.0 - 1e-12))
        self.xi = float(np.clip(self.xi + math.atanh(clipped), -self.limit, self.limit))
        return self.value

    def extend(self, couplings: ArrayLike) -> float:
        values = np.asarray(couplings, dtype=np.float64)
        if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
            raise ValueError("couplings must be a non-empty finite vector")
        for coupling in values:
            self.add(float(coupling))
        return self.value

    @property
    def value(self) -> float:
        return math.tanh(self.xi)

    def to_dict(self) -> dict[str, float | str]:
        return {
            "schema": "immer-ooe-rapidity-ledger/v1",
            "xi": float(self.xi),
            "limit": float(self.limit),
            "coupling": float(self.value),
        }


def asymmetry_parameter(
    matrix: ArrayLike,
    *,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
) -> float:
    value = _as_square(
        matrix,
        max_nodes=max_nodes,
        max_bytes=max_bytes,
        working_arrays=4,
    )
    symmetric = 0.5 * (value + value.T)
    antisymmetric = 0.5 * (value - value.T)
    asymmetry = float(np.linalg.norm(antisymmetric, ord="fro"))
    symmetry = float(np.linalg.norm(symmetric, ord="fro"))
    return 0.0 if asymmetry + symmetry == 0.0 else asymmetry / (asymmetry + symmetry)


def spectral_gap(
    matrix: ArrayLike,
    *,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
) -> float:
    value = _as_square(
        matrix,
        max_nodes=max_nodes,
        max_bytes=max_bytes,
        working_arrays=4,
    )
    eigenvalues = np.linalg.eigvals(value)
    perron = int(np.argmin(np.abs(eigenvalues - 1.0)))
    remainder = np.delete(eigenvalues, perron)
    if remainder.size == 0:
        return 1.0
    return float(max(0.0, 1.0 - float(np.max(np.abs(remainder)))))


def tv_contraction(
    matrix: ArrayLike,
    *,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
) -> float:
    """Total-variation/Dobrushin contraction coefficient."""

    value = normalize_rows(
        _as_square(
            matrix,
            nonnegative=True,
            max_nodes=max_nodes,
            max_bytes=max_bytes,
            working_arrays=4,
        ),
        max_nodes=max_nodes,
        max_bytes=max_bytes,
    )
    maximum = 0.0
    for index in range(value.shape[0]):
        distances = 0.5 * np.abs(value[index] - value).sum(axis=1)
        maximum = max(maximum, float(distances.max()))
    return maximum


def normalized_entropy(
    matrix: ArrayLike,
    *,
    eps: float = 1e-15,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
) -> float:
    floor = _positive_float(eps, name="eps")
    value = normalize_rows(
        matrix,
        eps=floor,
        max_nodes=max_nodes,
        max_bytes=max_bytes,
    )
    if value.shape[1] == 1:
        return 0.0
    entropy = -np.sum(value * np.log(np.maximum(value, floor)), axis=1)
    return float(np.mean(entropy) / math.log(value.shape[1]))


def ginibre_s2(
    matrix: ArrayLike,
    *,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
) -> float | None:
    """Small-matrix nearest-neighbour spectral diagnostic, not a universality test."""

    value = _as_square(
        matrix,
        max_nodes=max_nodes,
        max_bytes=max_bytes,
        working_arrays=4,
    )
    eigenvalues = np.linalg.eigvals(value)
    if eigenvalues.size < 4:
        return None
    perron = int(np.argmin(np.abs(eigenvalues - 1.0)))
    points = np.delete(eigenvalues, perron)
    distances: list[float] = []
    for index, point in enumerate(points):
        others = np.delete(points, index)
        distances.append(float(np.min(np.abs(point - others))))
    mean = float(np.mean(distances))
    if mean <= 1e-15:
        return None
    normalized = np.asarray(distances, dtype=np.float64) / mean
    return float(np.mean(normalized**2))


def jensen_gap(couplings: ArrayLike) -> float:
    """Convex Lorentz-factor Jensen gap used as a concentration diagnostic."""

    raw = np.asarray(couplings)
    if raw.ndim != 1 or raw.size < 1:
        raise ValueError("couplings must be a non-empty vector")
    try:
        values = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("couplings must contain real numbers") from exc
    if not np.isfinite(values).all():
        raise ValueError("couplings must be finite")
    values = np.clip(values, -0.999999, 0.999999)
    gamma = 1.0 / np.sqrt(1.0 - values**2)
    mean_gamma = float(gamma.mean())
    mean_value = float(values.mean())
    gamma_mean = 1.0 / math.sqrt(1.0 - mean_value**2)
    return mean_gamma - gamma_mean


@dataclass(frozen=True, slots=True)
class KernelDiagnostics:
    """Canonical scalar receipt for a finite square kernel."""

    size: int
    matrix_sha256: str
    row_sum_max_error: float
    column_sum_max_error: float
    minimum: float
    maximum: float
    asymmetry: float
    spectral_gap: float
    tv_contraction: float
    normalized_entropy: float
    ginibre_s2: float | None

    @classmethod
    def from_matrix(
        cls,
        matrix: ArrayLike,
        *,
        max_nodes: int = DEFAULT_MAX_NODES,
        max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
    ) -> "KernelDiagnostics":
        value = _as_square(
            matrix,
            nonnegative=True,
            max_nodes=max_nodes,
            max_bytes=max_bytes,
            working_arrays=8,
        )
        return cls(
            size=int(value.shape[0]),
            matrix_sha256=array_sha256(value),
            row_sum_max_error=float(np.max(np.abs(value.sum(axis=1) - 1.0))),
            column_sum_max_error=float(np.max(np.abs(value.sum(axis=0) - 1.0))),
            minimum=float(value.min()),
            maximum=float(value.max()),
            asymmetry=asymmetry_parameter(
                value, max_nodes=max_nodes, max_bytes=max_bytes
            ),
            spectral_gap=spectral_gap(value, max_nodes=max_nodes, max_bytes=max_bytes),
            tv_contraction=tv_contraction(
                value, max_nodes=max_nodes, max_bytes=max_bytes
            ),
            normalized_entropy=normalized_entropy(
                value, max_nodes=max_nodes, max_bytes=max_bytes
            ),
            ginibre_s2=ginibre_s2(value, max_nodes=max_nodes, max_bytes=max_bytes),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "immer-ooe-kernel-diagnostics/v1",
            "size": self.size,
            "matrix_sha256": self.matrix_sha256,
            "row_sum_max_error": self.row_sum_max_error,
            "column_sum_max_error": self.column_sum_max_error,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "asymmetry": self.asymmetry,
            "spectral_gap": self.spectral_gap,
            "tv_contraction": self.tv_contraction,
            "normalized_entropy": self.normalized_entropy,
            "ginibre_s2": self.ginibre_s2,
        }


def kernel_diagnostics(
    matrix: ArrayLike,
    *,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
) -> dict[str, Any]:
    """Return canonical JSON-ready diagnostics for ``matrix``."""

    return KernelDiagnostics.from_matrix(
        matrix,
        max_nodes=max_nodes,
        max_bytes=max_bytes,
    ).to_dict()


__all__ = [
    "DEFAULT_MAX_DENSE_BYTES",
    "DEFAULT_MAX_NODES",
    "FloatArray",
    "KernelDiagnostics",
    "RapidityLedger",
    "array_sha256",
    "as_float_matrix",
    "asymmetry_parameter",
    "ginibre_s2",
    "jensen_gap",
    "kernel_diagnostics",
    "mobius_compose",
    "normalize_rows",
    "normalized_entropy",
    "sinkhorn_project",
    "sinkhorn_rg_cool",
    "spectral_gap",
    "tv_contraction",
]
