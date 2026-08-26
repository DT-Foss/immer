"""Deterministic operator search over the Birkhoff polytope.

The module stores a permutation mixture as ``K`` permutations and ``K``
weights, never as a factorial permutation basis.  Dense matrices only exist
while a kernel is constructed, verified, or scored.  All persistent search
state is content addressed and encoded as canonical JSON.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .consensus import fiedler_eigenspace
from .identity import canonical_json_bytes, require_sha256
from .math_core import array_sha256, as_float_matrix


FloatArray = NDArray[np.float64]

PERMUTATION_MIXTURE_SCHEMA = "immer-ooe-permutation-mixture/v1"
BIRKHOFF_DECOMPOSITION_SCHEMA = "immer-ooe-birkhoff-decomposition/v1"
BIRKHOFF_RECEIPT_SCHEMA = "immer-ooe-birkhoff-reconstruction/v1"
THOMPSON_STATE_SCHEMA = "immer-ooe-contextual-thompson/v1"
DIAGONAL_STEP_SCHEMA = "immer-ooe-diagonal-success-step/v1"
MAP_ELITES_SCHEMA = "immer-ooe-behavioral-map-elites/v1"
SPECTRAL_FILTER_SCHEMA = "immer-ooe-matched-spectral-filter/v1"
OPERATOR_GENOME_SCHEMA = "immer-ooe-operator-genome/v1"

MAX_BVN_DIMENSION = 256
MAX_BVN_COMPONENTS = (MAX_BVN_DIMENSION - 1) ** 2 + 1
MAX_PERMUTATION_ATOMS = 65_536
MAX_ARCHIVE_CELLS = 1_000_000
MAX_ELITES_PER_CELL = 64
MAX_OBJECTIVES = 32
MAX_VECTOR_DIMENSION = 65_536
MAX_CONTEXTS = 65_536
MAX_MUTATION_ARMS = 1_024
MAX_SERIALIZED_BYTES = 256 * 1024 * 1024

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}")


class BvNSearchIntegrityError(ValueError):
    """A persisted operator-search object failed authentication."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha_uniform_index(domain: bytes, bound: int) -> int:
    """Map a content-addressed domain to ``[0, bound)`` without modulo bias."""

    upper = _bounded_int(
        bound,
        name="sampling bound",
        minimum=1,
        maximum=MAX_ARCHIVE_CELLS,
    )
    modulus = 1 << 64
    limit = modulus - (modulus % upper)
    counter = 0
    while True:
        digest = hashlib.sha256(
            domain + counter.to_bytes(8, "little", signed=False)
        ).digest()
        value = int.from_bytes(digest[:8], "little")
        if value < limit:
            return value % upper
        counter += 1


def _bounded_int(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must lie in [{minimum}, {maximum}]")
    return value


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return 0.0 if result == 0.0 else result


def _positive(value: object, *, name: str, allow_zero: bool = False) -> float:
    result = _finite(value, name=name)
    if result < 0.0 or (result == 0.0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} is not a canonical identifier")
    return value


def _float_tuple(
    values: Iterable[object],
    *,
    name: str,
    minimum_length: int = 1,
    maximum_length: int = MAX_VECTOR_DIMENSION,
) -> tuple[float, ...]:
    result = tuple(_finite(value, name=name) for value in values)
    if not minimum_length <= len(result) <= maximum_length:
        raise ValueError(
            f"{name} length must lie in [{minimum_length}, {maximum_length}]"
        )
    return result


def _hex_floats(values: Sequence[float]) -> list[str]:
    return [float(value).hex() for value in values]


def _parse_hex_floats(
    values: object,
    *,
    name: str,
    minimum_length: int = 1,
    maximum_length: int = MAX_VECTOR_DIMENSION,
) -> tuple[float, ...]:
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a list")
    if not minimum_length <= len(values) <= maximum_length:
        raise ValueError(f"{name} has an invalid length")
    try:
        return _float_tuple(
            (float.fromhex(value) for value in values if isinstance(value, str)),
            name=name,
            minimum_length=minimum_length,
            maximum_length=maximum_length,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains an invalid hexadecimal float") from exc


def _strict_document(data: bytes, *, schema: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(data, bytes) or len(data) > MAX_SERIALIZED_BYTES:
        raise BvNSearchIntegrityError("payload must be bounded immutable bytes")
    try:
        root = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BvNSearchIntegrityError("payload is not JSON") from exc
    if (
        not isinstance(root, dict)
        or set(root) != {"body", "body_sha256", "schema"}
        or root.get("schema") != schema
        or not isinstance(root.get("body"), dict)
        or canonical_json_bytes(root) != data
    ):
        raise BvNSearchIntegrityError("payload is not canonical")
    body = root["body"]
    try:
        body_hash = require_sha256(root["body_sha256"], field="body_sha256")
    except (TypeError, ValueError) as exc:
        raise BvNSearchIntegrityError("payload body hash is invalid") from exc
    if _digest(body) != body_hash:
        raise BvNSearchIntegrityError("payload body hash mismatch")
    return root, body


def _encode_document(schema: str, body: Mapping[str, object]) -> bytes:
    document = {
        "body": dict(body),
        "body_sha256": _digest(body),
        "schema": schema,
    }
    encoded = canonical_json_bytes(document)
    if len(encoded) > MAX_SERIALIZED_BYTES:
        raise ValueError("serialized search state exceeds its byte bound")
    return encoded


@dataclass(frozen=True, slots=True, order=True)
class PermutationAtom:
    """One permutation vertex stored as a length-``n`` image tuple."""

    permutation: tuple[int, ...]

    def __post_init__(self) -> None:
        raw_permutation = tuple(self.permutation)
        size = _bounded_int(
            len(raw_permutation),
            name="permutation dimension",
            minimum=1,
            maximum=MAX_BVN_DIMENSION,
        )
        permutation: list[int] = []
        for value in raw_permutation:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError("permutation image must be an integer")
            image = int(value)
            _bounded_int(
                image,
                name="permutation image",
                minimum=0,
                maximum=size - 1,
            )
            permutation.append(image)
        if len(set(permutation)) != size:
            raise ValueError("permutation images must be a bijection")
        object.__setattr__(self, "permutation", tuple(permutation))

    @property
    def size(self) -> int:
        return len(self.permutation)

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {"permutation": list(self.permutation)}

    @classmethod
    def from_dict(cls, value: object) -> "PermutationAtom":
        if not isinstance(value, dict) or set(value) != {"permutation"}:
            raise ValueError("invalid permutation atom")
        raw = value["permutation"]
        if not isinstance(raw, list):
            raise ValueError("permutation must be a list")
        return cls(tuple(raw))

    def dense(self) -> FloatArray:
        result = np.zeros((self.size, self.size), dtype=np.float64)
        result[np.arange(self.size), np.asarray(self.permutation)] = 1.0
        return result

    def apply(self, value: ArrayLike) -> FloatArray:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim < 1 or array.shape[-1] != self.size:
            raise ValueError("value trailing dimension must match the permutation")
        if not np.isfinite(array).all():
            raise ValueError("value must contain only finite numbers")
        images = np.asarray(self.permutation, dtype=np.int64)
        inverse = np.empty(self.size, dtype=np.int64)
        inverse[images] = np.arange(self.size, dtype=np.int64)
        return np.ascontiguousarray(
            np.take(array, inverse, axis=-1), dtype=np.float64
        )


def _validate_atoms(atoms: Sequence[PermutationAtom]) -> tuple[PermutationAtom, ...]:
    result = tuple(atoms)
    if not result:
        raise ValueError("at least one permutation atom is required")
    if any(not isinstance(atom, PermutationAtom) for atom in result):
        raise TypeError("atoms must be PermutationAtom values")
    size = result[0].size
    if any(atom.size != size for atom in result):
        raise ValueError("all permutation atoms must have the same dimension")
    if len(set(result)) != len(result):
        raise ValueError("permutation atoms must be unique")
    if len(result) > MAX_PERMUTATION_ATOMS:
        raise ValueError("permutation mixture exceeds its atom bound")
    return result


@dataclass(frozen=True, slots=True)
class PermutationMixture:
    """A convex permutation mixture with ``O(Kn)`` persistent storage."""

    atoms: tuple[PermutationAtom, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        atoms = _validate_atoms(self.atoms)
        weights = _float_tuple(
            self.weights,
            name="mixture weight",
            maximum_length=len(atoms),
        )
        if len(weights) != len(atoms):
            raise ValueError("weights and permutation atoms must have equal length")
        if any(weight <= 0.0 for weight in weights):
            raise ValueError("mixture weights must be strictly positive")
        if not math.isclose(math.fsum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("mixture weights must sum to one")
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "weights", weights)

    @property
    def size(self) -> int:
        return self.atoms[0].size

    @property
    def component_count(self) -> int:
        return len(self.atoms)

    def reconstruct(self) -> FloatArray:
        result = np.zeros((self.size, self.size), dtype=np.float64)
        contributions: dict[tuple[int, int], list[float]] = defaultdict(list)
        for atom, weight in zip(self.atoms, self.weights, strict=True):
            for row, column in enumerate(atom.permutation):
                contributions[(row, column)].append(weight)
        for (row, column), values in contributions.items():
            result[row, column] = math.fsum(values)
        return np.ascontiguousarray(result)

    @property
    def kernel_sha256(self) -> str:
        return array_sha256(self.reconstruct())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def _body(self) -> dict[str, object]:
        return {
            "atoms": [atom.to_dict() for atom in self.atoms],
            "component_count": self.component_count,
            "dimension": self.size,
            "kernel_sha256": self.kernel_sha256,
            "weights_hex": _hex_floats(self.weights),
        }

    def to_bytes(self) -> bytes:
        return _encode_document(PERMUTATION_MIXTURE_SCHEMA, self._body())

    @classmethod
    def from_bytes(cls, data: bytes) -> "PermutationMixture":
        _, body = _strict_document(data, schema=PERMUTATION_MIXTURE_SCHEMA)
        expected = {
            "atoms",
            "component_count",
            "dimension",
            "kernel_sha256",
            "weights_hex",
        }
        if set(body) != expected or not isinstance(body["atoms"], list):
            raise BvNSearchIntegrityError("invalid permutation-mixture body")
        try:
            atoms = tuple(PermutationAtom.from_dict(value) for value in body["atoms"])
            weights = _parse_hex_floats(
                body["weights_hex"],
                name="weights",
                maximum_length=MAX_PERMUTATION_ATOMS,
            )
            result = cls(atoms=atoms, weights=weights)
            require_sha256(body["kernel_sha256"], field="kernel_sha256")
        except (TypeError, ValueError) as exc:
            raise BvNSearchIntegrityError("invalid permutation mixture") from exc
        if result._body() != body:
            raise BvNSearchIntegrityError("permutation-mixture derived fields changed")
        return result


def canonical_theta(theta: ArrayLike, *, atom_count: int) -> FloatArray:
    """Fix the softmax gauge by setting the first logit to exactly zero."""

    count = _bounded_int(
        atom_count,
        name="atom_count",
        minimum=1,
        maximum=MAX_PERMUTATION_ATOMS,
    )
    raw = np.asarray(theta)
    if raw.ndim != 1 or raw.shape != (count,):
        raise ValueError(f"theta must have exact shape ({count},)")
    try:
        result = np.asarray(raw, dtype=np.float64).copy()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("theta must contain real numbers") from exc
    if not np.isfinite(result).all():
        raise ValueError("theta must contain only finite numbers")
    anchor = float(result[0])
    result -= anchor
    if not np.isfinite(result).all():
        raise ValueError("theta gauge subtraction overflowed float64")
    result[0] = 0.0
    result[result == 0.0] = 0.0
    return np.ascontiguousarray(result)


def mixture_from_theta(
    theta: ArrayLike,
    atoms: Sequence[PermutationAtom],
) -> PermutationMixture:
    vertices = _validate_atoms(atoms)
    gauge = canonical_theta(theta, atom_count=len(vertices))
    shifted = gauge - float(np.max(gauge))
    exponentials = np.exp(shifted)
    denominator = math.fsum(float(value) for value in exponentials)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ArithmeticError("softmax normalization failed")
    weighted_atoms = tuple(
        (atom, float(value) / denominator)
        for atom, value in zip(vertices, exponentials, strict=True)
        if value > 0.0
    )
    if not weighted_atoms:
        raise ArithmeticError("softmax lost every permutation atom")
    retained_atoms = tuple(atom for atom, _weight in weighted_atoms)
    weights = tuple(weight for _atom, weight in weighted_atoms)
    correction = 1.0 - math.fsum(weights)
    if correction != 0.0:
        mutable = list(weights)
        pivot = int(np.argmax(mutable))
        mutable[pivot] += correction
        weights = tuple(mutable)
    return PermutationMixture(retained_atoms, weights)


def theta_to_doubly_stochastic(
    theta: ArrayLike,
    atoms: Sequence[PermutationAtom],
) -> FloatArray:
    return mixture_from_theta(theta, atoms).reconstruct()


@dataclass(frozen=True, slots=True)
class OperatorGenome:
    """Immutable BvN search genome with an ordered composition lineage."""

    atoms: tuple[PermutationAtom, ...]
    theta: tuple[float, ...]
    map_family: str
    composition_parent_sha256s: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        atoms = _validate_atoms(self.atoms)
        gauge = canonical_theta(self.theta, atom_count=len(atoms))
        family = _identifier(self.map_family, name="map_family")
        parents = tuple(
            require_sha256(value, field="composition_parent_sha256")
            for value in self.composition_parent_sha256s
        )
        if len(parents) > MAX_PERMUTATION_ATOMS:
            raise ValueError("composition lineage exceeds its parent bound")
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "theta", tuple(float(value) for value in gauge))
        object.__setattr__(self, "map_family", family)
        object.__setattr__(self, "composition_parent_sha256s", parents)

    @property
    def mixture(self) -> PermutationMixture:
        return mixture_from_theta(self.theta, self.atoms)

    def reconstruct(self) -> FloatArray:
        return self.mixture.reconstruct()

    @property
    def kernel_sha256(self) -> str:
        return self.mixture.kernel_sha256

    def _body(self) -> dict[str, object]:
        return {
            "atoms": [atom.to_dict() for atom in self.atoms],
            "composition_parent_sha256s": list(self.composition_parent_sha256s),
            "dimension": self.atoms[0].size,
            "kernel_sha256": self.kernel_sha256,
            "map_family": self.map_family,
            "theta_hex": _hex_floats(self.theta),
        }

    def to_bytes(self) -> bytes:
        return _encode_document(OPERATOR_GENOME_SCHEMA, self._body())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "OperatorGenome":
        _, body = _strict_document(data, schema=OPERATOR_GENOME_SCHEMA)
        expected = {
            "atoms",
            "composition_parent_sha256s",
            "dimension",
            "kernel_sha256",
            "map_family",
            "theta_hex",
        }
        if (
            set(body) != expected
            or not isinstance(body["atoms"], list)
            or not isinstance(body["composition_parent_sha256s"], list)
        ):
            raise BvNSearchIntegrityError("invalid operator-genome body")
        try:
            result = cls(
                atoms=tuple(PermutationAtom.from_dict(value) for value in body["atoms"]),
                theta=_parse_hex_floats(
                    body["theta_hex"],
                    name="theta",
                    maximum_length=MAX_PERMUTATION_ATOMS,
                ),
                map_family=body["map_family"],
                composition_parent_sha256s=tuple(
                    body["composition_parent_sha256s"]
                ),
            )
            require_sha256(body["kernel_sha256"], field="kernel_sha256")
        except (TypeError, ValueError) as exc:
            raise BvNSearchIntegrityError("invalid operator genome") from exc
        if result._body() != body:
            raise BvNSearchIntegrityError("operator-genome derived fields changed")
        return result


# Candidate-code spelling, now strict rather than pad/truncate permissive.
ds_from_theta = theta_to_doubly_stochastic


def _validate_doubly_stochastic(
    kernel: ArrayLike,
    *,
    tolerance: float,
) -> FloatArray:
    threshold = _positive(tolerance, name="tolerance")
    value = as_float_matrix(
        kernel,
        name="kernel",
        square=True,
        nonnegative=True,
        max_nodes=MAX_BVN_DIMENSION,
        working_arrays=4,
    )
    size = int(value.shape[0])
    _bounded_int(
        size,
        name="kernel dimension",
        minimum=1,
        maximum=MAX_BVN_DIMENSION,
    )
    if not np.allclose(value.sum(axis=1), 1.0, atol=threshold, rtol=0.0):
        raise ValueError("kernel rows must sum to one")
    if not np.allclose(value.sum(axis=0), 1.0, atol=threshold, rtol=0.0):
        raise ValueError("kernel columns must sum to one")
    return np.ascontiguousarray(value, dtype=np.float64)


def _positive_support_perfect_matching(residual: FloatArray) -> tuple[int, ...]:
    """Find a deterministic perfect matching using alternating-path BFS."""

    size = residual.shape[0]
    row_to_column = np.full(size, -1, dtype=np.int64)
    column_to_row = np.full(size, -1, dtype=np.int64)
    support = tuple(np.flatnonzero(residual[row] > 0.0) for row in range(size))
    if any(columns.size == 0 for columns in support):
        raise ArithmeticError("positive support has an empty row")

    for root in range(size):
        row_parent = np.full(size, -2, dtype=np.int64)
        column_parent = np.full(size, -1, dtype=np.int64)
        queue = [root]
        row_parent[root] = -1
        head = 0
        free_column = -1
        while head < len(queue) and free_column < 0:
            row = queue[head]
            head += 1
            for raw_column in support[int(row)]:
                column = int(raw_column)
                if column_parent[column] >= 0:
                    continue
                column_parent[column] = row
                matched_row = int(column_to_row[column])
                if matched_row < 0:
                    free_column = column
                    break
                if row_parent[matched_row] == -2:
                    row_parent[matched_row] = column
                    queue.append(matched_row)
        if free_column < 0:
            raise ArithmeticError("positive support has no perfect matching")

        column = free_column
        while column >= 0:
            row = int(column_parent[column])
            previous_column = int(row_to_column[row])
            row_to_column[row] = column
            column_to_row[column] = row
            column = previous_column

    if np.any(row_to_column < 0) or len(set(row_to_column.tolist())) != size:
        raise ArithmeticError("perfect-matching construction failed")
    return tuple(int(value) for value in row_to_column)


@dataclass(frozen=True, slots=True)
class BirkhoffReconstructionReceipt:
    decomposition_sha256: str
    source_kernel_sha256: str
    reconstructed_kernel_sha256: str
    dimension: int
    component_count: int
    max_abs_error: float
    exact_bitwise: bool
    tolerance: float

    def __post_init__(self) -> None:
        for field in (
            "decomposition_sha256",
            "source_kernel_sha256",
            "reconstructed_kernel_sha256",
        ):
            object.__setattr__(
                self,
                field,
                require_sha256(getattr(self, field), field=field),
            )
        _bounded_int(
            self.dimension,
            name="dimension",
            minimum=1,
            maximum=MAX_BVN_DIMENSION,
        )
        _bounded_int(
            self.component_count,
            name="component_count",
            minimum=1,
            maximum=MAX_BVN_COMPONENTS,
        )
        object.__setattr__(
            self,
            "max_abs_error",
            _positive(self.max_abs_error, name="max_abs_error", allow_zero=True),
        )
        object.__setattr__(
            self,
            "tolerance",
            _positive(self.tolerance, name="tolerance"),
        )
        if not isinstance(self.exact_bitwise, bool):
            raise TypeError("exact_bitwise must be a boolean")

    @property
    def accepted(self) -> bool:
        return self.max_abs_error <= self.tolerance

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "component_count": self.component_count,
            "decomposition_sha256": self.decomposition_sha256,
            "dimension": self.dimension,
            "exact_bitwise": self.exact_bitwise,
            "max_abs_error_hex": self.max_abs_error.hex(),
            "reconstructed_kernel_sha256": self.reconstructed_kernel_sha256,
            "schema": BIRKHOFF_RECEIPT_SCHEMA,
            "source_kernel_sha256": self.source_kernel_sha256,
            "tolerance_hex": self.tolerance.hex(),
        }

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict())

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_bytes(cls, data: bytes) -> "BirkhoffReconstructionReceipt":
        if not isinstance(data, bytes) or len(data) > 64 * 1024:
            raise BvNSearchIntegrityError("receipt must be bounded immutable bytes")
        try:
            root = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BvNSearchIntegrityError("receipt is not JSON") from exc
        expected = {
            "accepted",
            "component_count",
            "decomposition_sha256",
            "dimension",
            "exact_bitwise",
            "max_abs_error_hex",
            "reconstructed_kernel_sha256",
            "schema",
            "source_kernel_sha256",
            "tolerance_hex",
        }
        if (
            not isinstance(root, dict)
            or set(root) != expected
            or root.get("schema") != BIRKHOFF_RECEIPT_SCHEMA
            or canonical_json_bytes(root) != data
        ):
            raise BvNSearchIntegrityError("receipt is not canonical")
        try:
            result = cls(
                decomposition_sha256=root["decomposition_sha256"],
                source_kernel_sha256=root["source_kernel_sha256"],
                reconstructed_kernel_sha256=root["reconstructed_kernel_sha256"],
                dimension=root["dimension"],
                component_count=root["component_count"],
                max_abs_error=float.fromhex(root["max_abs_error_hex"]),
                exact_bitwise=root["exact_bitwise"],
                tolerance=float.fromhex(root["tolerance_hex"]),
            )
        except (TypeError, ValueError) as exc:
            raise BvNSearchIntegrityError("invalid reconstruction receipt") from exc
        if result.to_dict() != root:
            raise BvNSearchIntegrityError("receipt derived fields changed")
        return result


@dataclass(frozen=True, slots=True)
class BirkhoffDecomposition:
    """Authenticated constructive decomposition of one source kernel."""

    mixture: PermutationMixture
    source_kernel_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.mixture, PermutationMixture):
            raise TypeError("mixture must be a PermutationMixture")
        object.__setattr__(
            self,
            "source_kernel_sha256",
            require_sha256(self.source_kernel_sha256, field="source_kernel_sha256"),
        )
        if self.mixture.component_count > (self.mixture.size - 1) ** 2 + 1:
            raise ValueError("decomposition exceeds the Birkhoff dimension bound")

    @property
    def atoms(self) -> tuple[PermutationAtom, ...]:
        return self.mixture.atoms

    @property
    def weights(self) -> tuple[float, ...]:
        return self.mixture.weights

    @property
    def size(self) -> int:
        return self.mixture.size

    @property
    def component_count(self) -> int:
        return self.mixture.component_count

    def reconstruct(self) -> FloatArray:
        return self.mixture.reconstruct()

    def _body(self) -> dict[str, object]:
        return {
            "atoms": [atom.to_dict() for atom in self.atoms],
            "component_count": self.component_count,
            "dimension": self.size,
            "reconstructed_kernel_sha256": self.mixture.kernel_sha256,
            "source_kernel_sha256": self.source_kernel_sha256,
            "weights_hex": _hex_floats(self.weights),
        }

    def to_bytes(self) -> bytes:
        return _encode_document(BIRKHOFF_DECOMPOSITION_SCHEMA, self._body())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def verify_source(
        self,
        kernel: ArrayLike,
        *,
        tolerance: float = 1e-12,
    ) -> BirkhoffReconstructionReceipt:
        source = _validate_doubly_stochastic(kernel, tolerance=tolerance)
        source_hash = array_sha256(source)
        if source_hash != self.source_kernel_sha256:
            raise BvNSearchIntegrityError("source kernel identity mismatch")
        reconstructed = self.reconstruct()
        error = float(np.max(np.abs(source - reconstructed)))
        receipt = BirkhoffReconstructionReceipt(
            decomposition_sha256=self.sha256,
            source_kernel_sha256=source_hash,
            reconstructed_kernel_sha256=array_sha256(reconstructed),
            dimension=self.size,
            component_count=self.component_count,
            max_abs_error=error,
            exact_bitwise=bool(np.array_equal(source, reconstructed)),
            tolerance=float(tolerance),
        )
        if not receipt.accepted:
            raise BvNSearchIntegrityError("decomposition does not reconstruct its source")
        return receipt

    @classmethod
    def from_bytes(cls, data: bytes) -> "BirkhoffDecomposition":
        _, body = _strict_document(data, schema=BIRKHOFF_DECOMPOSITION_SCHEMA)
        expected = {
            "atoms",
            "component_count",
            "dimension",
            "reconstructed_kernel_sha256",
            "source_kernel_sha256",
            "weights_hex",
        }
        if set(body) != expected or not isinstance(body["atoms"], list):
            raise BvNSearchIntegrityError("invalid Birkhoff-decomposition body")
        try:
            atoms = tuple(PermutationAtom.from_dict(value) for value in body["atoms"])
            weights = _parse_hex_floats(
                body["weights_hex"],
                name="weights",
                maximum_length=MAX_BVN_COMPONENTS,
            )
            result = cls(
                mixture=PermutationMixture(atoms, weights),
                source_kernel_sha256=body["source_kernel_sha256"],
            )
            require_sha256(
                body["reconstructed_kernel_sha256"],
                field="reconstructed_kernel_sha256",
            )
        except (TypeError, ValueError) as exc:
            raise BvNSearchIntegrityError("invalid Birkhoff decomposition") from exc
        if result._body() != body:
            raise BvNSearchIntegrityError("Birkhoff derived fields changed")
        return result


def birkhoff_von_neumann(
    kernel: ArrayLike,
    *,
    tolerance: float = 1e-12,
) -> BirkhoffDecomposition:
    """Construct a deterministic positive-support Birkhoff decomposition."""

    source = _validate_doubly_stochastic(kernel, tolerance=tolerance)
    residual = source.copy()
    size = residual.shape[0]
    bound = (size - 1) ** 2 + 1
    atoms: list[PermutationAtom] = []
    weights: list[float] = []
    threshold = float(tolerance)

    while float(np.max(residual)) > threshold:
        if len(atoms) >= bound:
            raise ArithmeticError("constructive decomposition exceeded its exact bound")
        permutation = _positive_support_perfect_matching(residual)
        atom = PermutationAtom(permutation)
        coefficient = min(residual[row, column] for row, column in enumerate(permutation))
        if not math.isfinite(coefficient) or coefficient <= 0.0:
            raise ArithmeticError("decomposition produced a non-positive coefficient")
        atoms.append(atom)
        weights.append(float(coefficient))
        for row, column in enumerate(permutation):
            residual[row, column] -= coefficient
        if np.any(residual < -threshold):
            raise ArithmeticError("decomposition produced a negative residual")
        residual[np.abs(residual) <= threshold] = 0.0

    if np.max(np.abs(residual)) > threshold:
        raise ArithmeticError("decomposition left a non-zero residual")
    if not atoms:
        raise ArithmeticError("decomposition produced no permutation atoms")

    # Matching choices are unique after each positive support edge disappears,
    # but combining is both defensive and preserves the O(Kn) representation.
    combined: dict[PermutationAtom, list[float]] = defaultdict(list)
    for atom, weight in zip(atoms, weights, strict=True):
        combined[atom].append(weight)
    ordered_atoms = tuple(sorted(combined))
    ordered_weights = [math.fsum(combined[atom]) for atom in ordered_atoms]
    correction = 1.0 - math.fsum(ordered_weights)
    if abs(correction) > threshold:
        raise ArithmeticError("decomposition weights do not sum to one")
    pivot = int(np.argmax(ordered_weights))
    ordered_weights[pivot] += correction
    mixture = PermutationMixture(ordered_atoms, tuple(ordered_weights))
    result = BirkhoffDecomposition(
        mixture=mixture,
        source_kernel_sha256=array_sha256(source),
    )
    result.verify_source(source, tolerance=tolerance)
    return result


# Short spelling retained for mathematical call sites.
bvn_decompose = birkhoff_von_neumann


@dataclass(frozen=True, slots=True, order=True)
class ThompsonPosterior:
    context: str
    arm: str
    successes: int = 0
    failures: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "context", _identifier(self.context, name="context"))
        object.__setattr__(self, "arm", _identifier(self.arm, name="arm"))
        _bounded_int(
            self.successes,
            name="successes",
            minimum=0,
            maximum=(1 << 63) - 1,
        )
        _bounded_int(
            self.failures,
            name="failures",
            minimum=0,
            maximum=(1 << 63) - 1,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "context": self.context,
            "failures": self.failures,
            "successes": self.successes,
        }


@dataclass(frozen=True, slots=True)
class ThompsonChoice:
    arm: str
    context: str
    decision_index: int
    parent_state_sha256: str
    scores: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm", _identifier(self.arm, name="arm"))
        object.__setattr__(self, "context", _identifier(self.context, name="context"))
        require_sha256(self.parent_state_sha256, field="parent_state_sha256")
        _bounded_int(
            self.decision_index,
            name="decision_index",
            minimum=0,
            maximum=(1 << 63) - 1,
        )
        normalized = tuple(
            (_identifier(arm, name="score arm"), _finite(score, name="score"))
            for arm, score in self.scores
        )
        if not normalized or tuple(arm for arm, _ in normalized) != tuple(
            sorted(arm for arm, _ in normalized)
        ):
            raise ValueError("choice scores must contain canonical sorted arms")
        if self.arm not in {arm for arm, _ in normalized}:
            raise ValueError("chosen arm is absent from the score receipt")
        object.__setattr__(self, "scores", normalized)

    @property
    def sha256(self) -> str:
        return _digest(
            {
                "arm": self.arm,
                "context": self.context,
                "decision_index": self.decision_index,
                "parent_state_sha256": self.parent_state_sha256,
                "scores": [
                    {"arm": arm, "score_hex": score.hex()}
                    for arm, score in self.scores
                ],
            }
        )


@dataclass(frozen=True, slots=True)
class ContextualThompsonMutation:
    """Immutable contextual beta-bandit state for mutation-family choice."""

    arms: tuple[str, ...]
    seed_sha256: str
    decision_index: int = 0
    posteriors: tuple[ThompsonPosterior, ...] = ()

    def __post_init__(self) -> None:
        arms = tuple(_identifier(arm, name="arm") for arm in self.arms)
        if not 1 <= len(arms) <= MAX_MUTATION_ARMS or len(set(arms)) != len(arms):
            raise ValueError("mutation arms must be a bounded unique sequence")
        if arms != tuple(sorted(arms)):
            raise ValueError("mutation arms must use canonical lexical order")
        object.__setattr__(self, "arms", arms)
        object.__setattr__(
            self,
            "seed_sha256",
            require_sha256(self.seed_sha256, field="seed_sha256"),
        )
        _bounded_int(
            self.decision_index,
            name="decision_index",
            minimum=0,
            maximum=(1 << 63) - 1,
        )
        posteriors = tuple(self.posteriors)
        if len(posteriors) > MAX_CONTEXTS * len(arms):
            raise ValueError("posterior table exceeds its bound")
        if any(not isinstance(item, ThompsonPosterior) for item in posteriors):
            raise TypeError("posteriors must be ThompsonPosterior values")
        posterior_keys = tuple((item.context, item.arm) for item in posteriors)
        if (
            posteriors != tuple(sorted(posteriors))
            or len(set(posterior_keys)) != len(posteriors)
        ):
            raise ValueError("posteriors must be unique and canonically sorted")
        if any(item.arm not in arms for item in posteriors):
            raise ValueError("posterior refers to an unknown mutation arm")
        object.__setattr__(self, "posteriors", posteriors)

    def _body(self) -> dict[str, object]:
        return {
            "arms": list(self.arms),
            "decision_index": self.decision_index,
            "posteriors": [item.to_dict() for item in self.posteriors],
            "seed_sha256": self.seed_sha256,
        }

    def to_bytes(self) -> bytes:
        return _encode_document(THOMPSON_STATE_SCHEMA, self._body())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def posterior(self, context: str, arm: str) -> ThompsonPosterior:
        wanted_context = _identifier(context, name="context")
        wanted_arm = _identifier(arm, name="arm")
        if wanted_arm not in self.arms:
            raise ValueError("unknown mutation arm")
        for item in self.posteriors:
            if item.context == wanted_context and item.arm == wanted_arm:
                return item
        return ThompsonPosterior(wanted_context, wanted_arm)

    def choose(self, context: str) -> tuple[ThompsonChoice, "ContextualThompsonMutation"]:
        wanted_context = _identifier(context, name="context")
        scores: list[tuple[str, float]] = []
        for arm in self.arms:
            posterior = self.posterior(wanted_context, arm)
            entropy = hashlib.sha256(
                canonical_json_bytes(
                    {
                        "arm": arm,
                        "context": wanted_context,
                        "decision_index": self.decision_index,
                        "seed_sha256": self.seed_sha256,
                        "state_sha256": self.sha256,
                    }
                )
            ).digest()
            generator = np.random.Generator(
                np.random.PCG64(int.from_bytes(entropy[:16], "little"))
            )
            score = float(
                generator.beta(1.0 + posterior.successes, 1.0 + posterior.failures)
            )
            scores.append((arm, score))
        chosen = min(scores, key=lambda item: (-item[1], item[0]))[0]
        receipt = ThompsonChoice(
            arm=chosen,
            context=wanted_context,
            decision_index=self.decision_index,
            parent_state_sha256=self.sha256,
            scores=tuple(scores),
        )
        return receipt, replace(self, decision_index=self.decision_index + 1)

    def observe(self, context: str, arm: str, *, success: bool) -> "ContextualThompsonMutation":
        if not isinstance(success, bool):
            raise TypeError("success must be a boolean")
        wanted = self.posterior(context, arm)
        updated = replace(
            wanted,
            successes=wanted.successes + int(success),
            failures=wanted.failures + int(not success),
        )
        table = {
            (item.context, item.arm): item
            for item in self.posteriors
        }
        table[(updated.context, updated.arm)] = updated
        return replace(self, posteriors=tuple(sorted(table.values())))

    @classmethod
    def from_bytes(cls, data: bytes) -> "ContextualThompsonMutation":
        _, body = _strict_document(data, schema=THOMPSON_STATE_SCHEMA)
        if set(body) != {"arms", "decision_index", "posteriors", "seed_sha256"}:
            raise BvNSearchIntegrityError("invalid Thompson state body")
        if not isinstance(body["arms"], list) or not isinstance(
            body["posteriors"], list
        ):
            raise BvNSearchIntegrityError("invalid Thompson state vectors")
        try:
            posteriors = tuple(
                ThompsonPosterior(
                    context=item["context"],
                    arm=item["arm"],
                    successes=item["successes"],
                    failures=item["failures"],
                )
                for item in body["posteriors"]
                if isinstance(item, dict)
                and set(item) == {"arm", "context", "failures", "successes"}
            )
            if len(posteriors) != len(body["posteriors"]):
                raise ValueError("invalid posterior entry")
            result = cls(
                arms=tuple(body["arms"]),
                seed_sha256=body["seed_sha256"],
                decision_index=body["decision_index"],
                posteriors=posteriors,
            )
        except (TypeError, ValueError) as exc:
            raise BvNSearchIntegrityError("invalid Thompson state") from exc
        if result._body() != body:
            raise BvNSearchIntegrityError("Thompson derived fields changed")
        return result


@dataclass(frozen=True, slots=True)
class DiagonalStepProposal:
    parent_state_sha256: str
    generation: int
    context_sha256: str
    standardized_step: tuple[float, ...]
    candidate: tuple[float, ...]

    def __post_init__(self) -> None:
        for field in ("parent_state_sha256", "context_sha256"):
            object.__setattr__(
                self,
                field,
                require_sha256(getattr(self, field), field=field),
            )
        _bounded_int(
            self.generation,
            name="generation",
            minimum=0,
            maximum=(1 << 63) - 1,
        )
        step = _float_tuple(self.standardized_step, name="standardized_step")
        candidate = _float_tuple(self.candidate, name="candidate")
        if len(step) != len(candidate):
            raise ValueError("proposal vectors must have equal length")
        object.__setattr__(self, "standardized_step", step)
        object.__setattr__(self, "candidate", candidate)

    @property
    def sha256(self) -> str:
        return _digest(
            {
                "candidate_hex": _hex_floats(self.candidate),
                "context_sha256": self.context_sha256,
                "generation": self.generation,
                "parent_state_sha256": self.parent_state_sha256,
                "standardized_step_hex": _hex_floats(self.standardized_step),
            }
        )


@dataclass(frozen=True, slots=True)
class DiagonalSuccessStepAdapter:
    """A diagonal success-rule step adapter; this is not full CMA-ES."""

    mean: tuple[float, ...]
    log_steps: tuple[float, ...]
    seed_sha256: str
    generation: int = 0
    target_success: float = 0.2
    learning_rate: float = 0.15
    min_log_step: float = -20.0
    max_log_step: float = 8.0

    def __post_init__(self) -> None:
        mean = _float_tuple(self.mean, name="mean")
        log_steps = _float_tuple(self.log_steps, name="log_steps")
        if len(mean) != len(log_steps):
            raise ValueError("mean and log_steps must have equal length")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "log_steps", log_steps)
        object.__setattr__(
            self,
            "seed_sha256",
            require_sha256(self.seed_sha256, field="seed_sha256"),
        )
        _bounded_int(
            self.generation,
            name="generation",
            minimum=0,
            maximum=(1 << 63) - 1,
        )
        target = _finite(self.target_success, name="target_success")
        rate = _positive(self.learning_rate, name="learning_rate")
        lower = _finite(self.min_log_step, name="min_log_step")
        upper = _finite(self.max_log_step, name="max_log_step")
        if not 0.0 < target < 1.0:
            raise ValueError("target_success must lie strictly inside (0, 1)")
        if lower >= upper:
            raise ValueError("min_log_step must be below max_log_step")
        if any(not lower <= value <= upper for value in log_steps):
            raise ValueError("log_steps lie outside the configured bounds")
        object.__setattr__(self, "target_success", target)
        object.__setattr__(self, "learning_rate", rate)
        object.__setattr__(self, "min_log_step", lower)
        object.__setattr__(self, "max_log_step", upper)

    @property
    def dimension(self) -> int:
        return len(self.mean)

    def _body(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "generation": self.generation,
            "learning_rate_hex": self.learning_rate.hex(),
            "log_steps_hex": _hex_floats(self.log_steps),
            "max_log_step_hex": self.max_log_step.hex(),
            "mean_hex": _hex_floats(self.mean),
            "min_log_step_hex": self.min_log_step.hex(),
            "seed_sha256": self.seed_sha256,
            "target_success_hex": self.target_success.hex(),
        }

    def to_bytes(self) -> bytes:
        return _encode_document(DIAGONAL_STEP_SCHEMA, self._body())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def propose(self, *, context_sha256: str) -> DiagonalStepProposal:
        context = require_sha256(context_sha256, field="context_sha256")
        entropy = hashlib.sha256(
            canonical_json_bytes(
                {
                    "context_sha256": context,
                    "generation": self.generation,
                    "seed_sha256": self.seed_sha256,
                    "state_sha256": self.sha256,
                }
            )
        ).digest()
        generator = np.random.Generator(
            np.random.PCG64(int.from_bytes(entropy[:16], "little"))
        )
        step = generator.standard_normal(self.dimension)
        candidate = np.asarray(self.mean) + np.exp(np.asarray(self.log_steps)) * step
        return DiagonalStepProposal(
            parent_state_sha256=self.sha256,
            generation=self.generation,
            context_sha256=context,
            standardized_step=tuple(float(value) for value in step),
            candidate=tuple(float(value) for value in candidate),
        )

    def update(
        self,
        proposal: DiagonalStepProposal,
        *,
        success: bool,
    ) -> "DiagonalSuccessStepAdapter":
        if not isinstance(proposal, DiagonalStepProposal):
            raise TypeError("proposal must be a DiagonalStepProposal")
        if not isinstance(success, bool):
            raise TypeError("success must be a boolean")
        if (
            proposal.parent_state_sha256 != self.sha256
            or proposal.generation != self.generation
        ):
            raise BvNSearchIntegrityError("proposal does not belong to this state")
        expected = np.asarray(self.mean) + np.exp(np.asarray(self.log_steps)) * np.asarray(
            proposal.standardized_step
        )
        if not np.array_equal(expected, np.asarray(proposal.candidate)):
            raise BvNSearchIntegrityError("proposal candidate was altered")
        signal = (1.0 if success else 0.0) - self.target_success
        excitation = 0.5 + np.minimum(
            np.abs(np.asarray(proposal.standardized_step)), 3.0
        ) / 6.0
        log_steps = np.clip(
            np.asarray(self.log_steps) + self.learning_rate * signal * excitation,
            self.min_log_step,
            self.max_log_step,
        )
        return replace(
            self,
            mean=proposal.candidate if success else self.mean,
            log_steps=tuple(float(value) for value in log_steps),
            generation=self.generation + 1,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "DiagonalSuccessStepAdapter":
        _, body = _strict_document(data, schema=DIAGONAL_STEP_SCHEMA)
        expected = {
            "dimension",
            "generation",
            "learning_rate_hex",
            "log_steps_hex",
            "max_log_step_hex",
            "mean_hex",
            "min_log_step_hex",
            "seed_sha256",
            "target_success_hex",
        }
        if set(body) != expected:
            raise BvNSearchIntegrityError("invalid diagonal-step body")
        try:
            result = cls(
                mean=_parse_hex_floats(body["mean_hex"], name="mean"),
                log_steps=_parse_hex_floats(body["log_steps_hex"], name="log_steps"),
                seed_sha256=body["seed_sha256"],
                generation=body["generation"],
                target_success=float.fromhex(body["target_success_hex"]),
                learning_rate=float.fromhex(body["learning_rate_hex"]),
                min_log_step=float.fromhex(body["min_log_step_hex"]),
                max_log_step=float.fromhex(body["max_log_step_hex"]),
            )
        except (TypeError, ValueError) as exc:
            raise BvNSearchIntegrityError("invalid diagonal-step state") from exc
        if result._body() != body:
            raise BvNSearchIntegrityError("diagonal-step derived fields changed")
        return result


@dataclass(frozen=True, slots=True)
class BehavioralElite:
    """Content-addressed candidate plus discrete behavior and maximized objectives."""

    candidate_sha256: str
    descriptor: tuple[int, ...]
    objectives: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_sha256",
            require_sha256(self.candidate_sha256, field="candidate_sha256"),
        )
        raw_descriptor = tuple(self.descriptor)
        if not raw_descriptor or len(raw_descriptor) > 32:
            raise ValueError("behavior descriptor has an invalid dimension")
        descriptor: list[int] = []
        for coordinate in raw_descriptor:
            if isinstance(coordinate, bool) or not isinstance(
                coordinate, (int, np.integer)
            ):
                raise TypeError("descriptor coordinate must be an integer")
            normalized = int(coordinate)
            _bounded_int(
                normalized,
                name="descriptor coordinate",
                minimum=0,
                maximum=MAX_ARCHIVE_CELLS - 1,
            )
            descriptor.append(normalized)
        objectives = _float_tuple(
            self.objectives,
            name="objective",
            maximum_length=MAX_OBJECTIVES,
        )
        object.__setattr__(self, "descriptor", tuple(descriptor))
        object.__setattr__(self, "objectives", objectives)

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_sha256": self.candidate_sha256,
            "descriptor": list(self.descriptor),
            "objectives_hex": _hex_floats(self.objectives),
        }


def _dominates(left: BehavioralElite, right: BehavioralElite) -> bool:
    return all(a >= b for a, b in zip(left.objectives, right.objectives, strict=True)) and any(
        a > b for a, b in zip(left.objectives, right.objectives, strict=True)
    )


def _elite_priority(elite: BehavioralElite) -> tuple[object, ...]:
    # Objectives are maximized.  SHA-256 supplies the final canonical tie.
    return (*(-value for value in elite.objectives), elite.candidate_sha256)


class BehavioralMAPElites:
    """Discrete behavior archive with deterministic bounded Pareto cells."""

    def __init__(
        self,
        bin_counts: Sequence[int],
        *,
        objective_count: int,
        max_elites_per_cell: int = 1,
    ) -> None:
        raw_bins = tuple(bin_counts)
        if not raw_bins or len(raw_bins) > 32:
            raise ValueError("bin_counts has an invalid dimension")
        bins_list: list[int] = []
        for count in raw_bins:
            if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
                raise TypeError("bin count must be an integer")
            normalized = int(count)
            _bounded_int(
                normalized,
                name="bin count",
                minimum=1,
                maximum=MAX_ARCHIVE_CELLS,
            )
            bins_list.append(normalized)
        bins = tuple(bins_list)
        denominator = math.prod(bins)
        if denominator > MAX_ARCHIVE_CELLS:
            raise ValueError("MAP-Elites grid exceeds its cell bound")
        self.bin_counts = bins
        self.objective_count = _bounded_int(
            objective_count,
            name="objective_count",
            minimum=1,
            maximum=MAX_OBJECTIVES,
        )
        self.max_elites_per_cell = _bounded_int(
            max_elites_per_cell,
            name="max_elites_per_cell",
            minimum=1,
            maximum=MAX_ELITES_PER_CELL,
        )
        self._cells: dict[tuple[int, ...], tuple[BehavioralElite, ...]] = {}

    @property
    def coverage_denominator(self) -> int:
        return math.prod(self.bin_counts)

    @property
    def occupied_cells(self) -> int:
        return len(self._cells)

    @property
    def coverage(self) -> float:
        return self.occupied_cells / self.coverage_denominator

    @property
    def elite_count(self) -> int:
        return sum(len(values) for values in self._cells.values())

    def _validate_elite(self, elite: BehavioralElite) -> None:
        if not isinstance(elite, BehavioralElite):
            raise TypeError("elite must be a BehavioralElite")
        if len(elite.descriptor) != len(self.bin_counts):
            raise ValueError("elite descriptor dimension mismatch")
        if any(
            not 0 <= coordinate < count
            for coordinate, count in zip(
                elite.descriptor, self.bin_counts, strict=True
            )
        ):
            raise ValueError("elite descriptor lies outside the archive grid")
        if len(elite.objectives) != self.objective_count:
            raise ValueError("elite objective dimension mismatch")

    def add(self, elite: BehavioralElite) -> bool:
        """Insert an elite and return whether the canonical cell changed."""

        self._validate_elite(elite)
        current = self._cells.get(elite.descriptor, ())
        previous = current

        # Equal objective vectors represent one Pareto point; canonical SHA wins.
        equal = [item for item in current if item.objectives == elite.objectives]
        if equal:
            winner = min((*equal, elite), key=lambda item: item.candidate_sha256)
            current = tuple(item for item in current if item.objectives != elite.objectives)
            current = (*current, winner)
        elif any(_dominates(item, elite) for item in current):
            return False
        else:
            current = tuple(item for item in current if not _dominates(elite, item))
            current = (*current, elite)

        canonical = tuple(sorted(current, key=_elite_priority))[
            : self.max_elites_per_cell
        ]
        if canonical:
            self._cells[elite.descriptor] = canonical
        else:
            self._cells.pop(elite.descriptor, None)
        return canonical != previous

    def cell(self, descriptor: Sequence[int]) -> tuple[BehavioralElite, ...]:
        key = tuple(descriptor)
        if len(key) != len(self.bin_counts):
            raise ValueError("descriptor dimension mismatch")
        return self._cells.get(key, ())

    def sample_cell_uniform(
        self,
        *,
        seed_sha256: str,
        nonce: int,
    ) -> BehavioralElite:
        """Sample occupied cells uniformly, then sample within the chosen cell."""

        seed = require_sha256(seed_sha256, field="seed_sha256")
        index = _bounded_int(
            nonce,
            name="nonce",
            minimum=0,
            maximum=(1 << 63) - 1,
        )
        if not self._cells:
            raise LookupError("cannot sample an empty MAP-Elites archive")
        cells = tuple(sorted(self._cells))
        entropy = canonical_json_bytes(
            {
                "archive_sha256": self.sha256,
                "nonce": index,
                "seed_sha256": seed,
            }
        )
        cell_index = _sha_uniform_index(b"map-elites-cell\0" + entropy, len(cells))
        cell = self._cells[cells[cell_index]]
        elite_index = _sha_uniform_index(b"map-elites-elite\0" + entropy, len(cell))
        return cell[elite_index]

    def _body(self) -> dict[str, object]:
        cells = [
            {
                "descriptor": list(descriptor),
                "elites": [elite.to_dict() for elite in self._cells[descriptor]],
            }
            for descriptor in sorted(self._cells)
        ]
        return {
            "bin_counts": list(self.bin_counts),
            "cells": cells,
            "coverage_denominator": self.coverage_denominator,
            "coverage_hex": self.coverage.hex(),
            "elite_count": self.elite_count,
            "max_elites_per_cell": self.max_elites_per_cell,
            "objective_count": self.objective_count,
            "occupied_cells": self.occupied_cells,
        }

    def to_bytes(self) -> bytes:
        return _encode_document(MAP_ELITES_SCHEMA, self._body())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "BehavioralMAPElites":
        _, body = _strict_document(data, schema=MAP_ELITES_SCHEMA)
        expected = {
            "bin_counts",
            "cells",
            "coverage_denominator",
            "coverage_hex",
            "elite_count",
            "max_elites_per_cell",
            "objective_count",
            "occupied_cells",
        }
        if (
            set(body) != expected
            or not isinstance(body["bin_counts"], list)
            or not isinstance(body["cells"], list)
        ):
            raise BvNSearchIntegrityError("invalid MAP-Elites body")
        try:
            archive = cls(
                body["bin_counts"],
                objective_count=body["objective_count"],
                max_elites_per_cell=body["max_elites_per_cell"],
            )
            seen: set[tuple[int, ...]] = set()
            for cell in body["cells"]:
                if not isinstance(cell, dict) or set(cell) != {
                    "descriptor",
                    "elites",
                }:
                    raise ValueError("invalid archive cell")
                descriptor = tuple(cell["descriptor"])
                if descriptor in seen or not isinstance(cell["elites"], list):
                    raise ValueError("duplicate or invalid archive cell")
                seen.add(descriptor)
                parsed: list[BehavioralElite] = []
                for item in cell["elites"]:
                    if not isinstance(item, dict) or set(item) != {
                        "candidate_sha256",
                        "descriptor",
                        "objectives_hex",
                    }:
                        raise ValueError("invalid elite entry")
                    elite = BehavioralElite(
                        candidate_sha256=item["candidate_sha256"],
                        descriptor=tuple(item["descriptor"]),
                        objectives=_parse_hex_floats(
                            item["objectives_hex"],
                            name="objectives",
                            maximum_length=MAX_OBJECTIVES,
                        ),
                    )
                    archive._validate_elite(elite)
                    if elite.descriptor != descriptor:
                        raise ValueError("elite descriptor disagrees with its cell")
                    parsed.append(elite)
                canonical = tuple(sorted(parsed, key=_elite_priority))
                if not canonical or len(canonical) > archive.max_elites_per_cell:
                    raise ValueError("archive cell has an invalid elite count")
                for left_index, left in enumerate(canonical):
                    if any(
                        _dominates(right, left)
                        for right_index, right in enumerate(canonical)
                        if right_index != left_index
                    ):
                        raise ValueError("archive cell is not a Pareto set")
                if len({elite.objectives for elite in canonical}) != len(canonical):
                    raise ValueError("archive cell contains duplicate Pareto points")
                archive._cells[descriptor] = canonical
        except (TypeError, ValueError) as exc:
            raise BvNSearchIntegrityError("invalid MAP-Elites archive") from exc
        if archive._body() != body:
            raise BvNSearchIntegrityError("MAP-Elites derived fields changed")
        return archive


# Conventional spelling for call sites that use the paper's name.
MAPElitesArchive = BehavioralMAPElites


@dataclass(frozen=True, slots=True)
class FiedlerNoveltyReceipt:
    adjacency_sha256: str
    candidate_kernel_sha256: str
    reference_kernel_sha256s: tuple[str, ...]
    fiedler_eigenvalue: float
    eigenspace_rank: int
    projected_signature_sha256: str
    novelty: float

    def __post_init__(self) -> None:
        for field in (
            "adjacency_sha256",
            "candidate_kernel_sha256",
            "projected_signature_sha256",
        ):
            object.__setattr__(
                self,
                field,
                require_sha256(getattr(self, field), field=field),
            )
        references = tuple(
            require_sha256(value, field="reference_kernel_sha256")
            for value in self.reference_kernel_sha256s
        )
        object.__setattr__(self, "reference_kernel_sha256s", references)
        object.__setattr__(
            self,
            "fiedler_eigenvalue",
            _positive(self.fiedler_eigenvalue, name="fiedler_eigenvalue"),
        )
        _bounded_int(
            self.eigenspace_rank,
            name="eigenspace_rank",
            minimum=1,
            maximum=MAX_BVN_DIMENSION,
        )
        object.__setattr__(
            self,
            "novelty",
            _positive(self.novelty, name="novelty", allow_zero=True),
        )

    @property
    def sha256(self) -> str:
        return _digest(
            {
                "adjacency_sha256": self.adjacency_sha256,
                "candidate_kernel_sha256": self.candidate_kernel_sha256,
                "eigenspace_rank": self.eigenspace_rank,
                "fiedler_eigenvalue_hex": self.fiedler_eigenvalue.hex(),
                "novelty_hex": self.novelty.hex(),
                "projected_signature_sha256": self.projected_signature_sha256,
                "reference_kernel_sha256s": list(self.reference_kernel_sha256s),
            }
        )


@dataclass(frozen=True, slots=True)
class FiedlerEdgeNoveltyReceipt:
    """Basis-invariant novelty of one missing or existing graph connection."""

    adjacency_sha256: str
    projector_sha256: str
    left_index: int
    right_index: int
    fiedler_eigenvalue: float
    eigenspace_rank: int
    novelty: float
    direct_edge: bool

    def __post_init__(self) -> None:
        for field in ("adjacency_sha256", "projector_sha256"):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        left = _bounded_int(
            self.left_index,
            name="left_index",
            minimum=0,
            maximum=MAX_BVN_DIMENSION - 1,
        )
        right = _bounded_int(
            self.right_index,
            name="right_index",
            minimum=0,
            maximum=MAX_BVN_DIMENSION - 1,
        )
        if left >= right:
            raise ValueError("Fiedler edge endpoints must be canonical and distinct")
        object.__setattr__(self, "left_index", left)
        object.__setattr__(self, "right_index", right)
        object.__setattr__(
            self,
            "fiedler_eigenvalue",
            _positive(self.fiedler_eigenvalue, name="fiedler_eigenvalue"),
        )
        _bounded_int(
            self.eigenspace_rank,
            name="eigenspace_rank",
            minimum=1,
            maximum=MAX_BVN_DIMENSION,
        )
        novelty = _positive(self.novelty, name="novelty", allow_zero=True)
        if novelty > 1.0 + 1e-12:
            raise ValueError("projector edge novelty must lie in [0, 1]")
        object.__setattr__(self, "novelty", min(1.0, novelty))
        if not isinstance(self.direct_edge, bool):
            raise TypeError("direct_edge must be boolean")

    def priority(
        self,
        confidence: float,
        *,
        direct_edge_penalty: float = 0.3,
    ) -> float:
        certainty = _positive(confidence, name="confidence", allow_zero=True)
        penalty = _positive(
            direct_edge_penalty,
            name="direct_edge_penalty",
            allow_zero=True,
        )
        if certainty > 1.0 or penalty > 1.0:
            raise ValueError("confidence and direct-edge penalty must lie in [0, 1]")
        return certainty * self.novelty * (penalty if self.direct_edge else 1.0)

    @property
    def sha256(self) -> str:
        return _digest(
            {
                "adjacency_sha256": self.adjacency_sha256,
                "direct_edge": self.direct_edge,
                "eigenspace_rank": self.eigenspace_rank,
                "fiedler_eigenvalue_hex": self.fiedler_eigenvalue.hex(),
                "left_index": self.left_index,
                "novelty_hex": self.novelty.hex(),
                "projector_sha256": self.projector_sha256,
                "right_index": self.right_index,
            }
        )


def fiedler_edge_novelty(
    adjacency: ArrayLike,
    left_index: int,
    right_index: int,
) -> FiedlerEdgeNoveltyReceipt:
    """Compute David Foss's Fiedler-gap proposal in projector form.

    ``sqrt((e_i-e_j)^T Π (e_i-e_j) / 2)`` is invariant to sign, rotation,
    and multiplicity of the Fiedler eigenspace.  For a one-dimensional space it
    reduces to ``abs(v2[i]-v2[j]) / sqrt(2)`` under unit-norm eigenvectors.
    """

    graph = as_float_matrix(
        adjacency,
        name="adjacency",
        square=True,
        nonnegative=True,
        max_nodes=MAX_BVN_DIMENSION,
        working_arrays=5,
    )
    size = int(graph.shape[0])
    left = _bounded_int(
        left_index,
        name="left_index",
        minimum=0,
        maximum=size - 1,
    )
    right = _bounded_int(
        right_index,
        name="right_index",
        minimum=0,
        maximum=size - 1,
    )
    if left == right:
        raise ValueError("Fiedler edge endpoints must be distinct")
    left, right = min(left, right), max(left, right)
    eigenvalue, basis = fiedler_eigenspace(
        graph,
        max_nodes=MAX_BVN_DIMENSION,
    )
    projector = np.ascontiguousarray(basis @ basis.T)
    difference = np.zeros(size, dtype=np.float64)
    difference[left] = 1.0
    difference[right] = -1.0
    quadratic = float(difference @ projector @ difference)
    if quadratic < -1e-12:
        raise ArithmeticError("Fiedler projector produced negative distance")
    novelty = math.sqrt(max(0.0, quadratic) / 2.0)
    return FiedlerEdgeNoveltyReceipt(
        adjacency_sha256=array_sha256(graph),
        projector_sha256=array_sha256(projector),
        left_index=left,
        right_index=right,
        fiedler_eigenvalue=eigenvalue,
        eigenspace_rank=int(basis.shape[1]),
        novelty=novelty,
        direct_edge=bool(graph[left, right] > 0.0),
    )


def fiedler_projector_novelty(
    adjacency: ArrayLike,
    candidate_kernel: ArrayLike,
    reference_kernels: Sequence[ArrayLike] = (),
) -> FiedlerNoveltyReceipt:
    """Measure novelty after orthogonal projection onto the Fiedler eigenspace."""

    graph = as_float_matrix(
        adjacency,
        name="adjacency",
        square=True,
        nonnegative=True,
        max_nodes=MAX_BVN_DIMENSION,
        working_arrays=4,
    )
    eigenvalue, basis = fiedler_eigenspace(
        graph,
        max_nodes=MAX_BVN_DIMENSION,
    )
    candidate = as_float_matrix(
        candidate_kernel,
        name="candidate_kernel",
        square=True,
        max_nodes=MAX_BVN_DIMENSION,
        working_arrays=4,
    )
    if candidate.shape != graph.shape:
        raise ValueError("candidate kernel and adjacency dimensions differ")
    signature = np.ascontiguousarray(basis.T @ candidate @ basis)
    references: list[FloatArray] = []
    reference_hashes: list[str] = []
    for ordinal, reference_kernel in enumerate(reference_kernels):
        reference = as_float_matrix(
            reference_kernel,
            name=f"reference_kernel[{ordinal}]",
            square=True,
            max_nodes=MAX_BVN_DIMENSION,
            working_arrays=4,
        )
        if reference.shape != candidate.shape:
            raise ValueError("reference kernel dimension mismatch")
        references.append(np.ascontiguousarray(basis.T @ reference @ basis))
        reference_hashes.append(array_sha256(reference))
    if references:
        novelty = min(float(np.linalg.norm(signature - item, ord="fro")) for item in references)
    else:
        novelty = float(np.linalg.norm(signature, ord="fro"))
    return FiedlerNoveltyReceipt(
        adjacency_sha256=array_sha256(graph),
        candidate_kernel_sha256=array_sha256(candidate),
        reference_kernel_sha256s=tuple(reference_hashes),
        fiedler_eigenvalue=eigenvalue,
        eigenspace_rank=int(basis.shape[1]),
        projected_signature_sha256=array_sha256(signature),
        novelty=novelty,
    )


SPECTRAL_FEATURE_NAMES = (
    "subdominant_radius",
    "spectral_gap",
    "nontrivial_abs_mean",
    "nontrivial_abs_std",
    "complex_fraction",
    "uniform_frobenius",
)


def spectral_proposal_features(kernel: ArrayLike) -> tuple[float, ...]:
    value = _validate_doubly_stochastic(kernel, tolerance=1e-10)
    size = value.shape[0]
    eigenvalues = np.linalg.eigvals(value)
    stationary = int(np.argmin(np.abs(eigenvalues - 1.0)))
    nontrivial = np.delete(eigenvalues, stationary)
    if nontrivial.size:
        magnitudes = np.abs(nontrivial)
        radius = float(np.max(magnitudes))
        mean = float(np.mean(magnitudes))
        standard_deviation = float(np.std(magnitudes))
        complex_fraction = float(np.mean(np.abs(np.imag(nontrivial)) > 1e-10))
    else:
        radius = mean = standard_deviation = complex_fraction = 0.0
    uniform = np.full_like(value, 1.0 / size)
    return (
        radius,
        1.0 - radius,
        mean,
        standard_deviation,
        complex_fraction,
        float(np.linalg.norm(value - uniform, ord="fro")),
    )


def _empirical_parameters(
    features: Sequence[Sequence[float]],
    *,
    quantile: float,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], float]:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] != len(
        SPECTRAL_FEATURE_NAMES
    ):
        raise ValueError("matched null requires at least three complete feature rows")
    if not np.isfinite(matrix).all():
        raise ValueError("null features must be finite")
    center_array = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - center_array), axis=0)
    scale_array = np.maximum(1.4826 * mad, 1e-12)
    scores_array = np.linalg.norm(
        (matrix - center_array) / scale_array,
        axis=1,
    ) / math.sqrt(matrix.shape[1])
    q = _finite(quantile, name="quantile")
    if not 0.0 <= q <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    ordered = np.sort(scores_array)
    quantile_index = max(0, math.ceil(q * len(ordered)) - 1)
    return (
        tuple(float(value) for value in center_array),
        tuple(float(value) for value in scale_array),
        tuple(float(value) for value in scores_array),
        float(ordered[quantile_index]),
    )


@dataclass(frozen=True, slots=True)
class SpectralProposalDecision:
    filter_sha256: str
    kernel_sha256: str
    features: tuple[float, ...]
    anomaly_score: float
    threshold: float
    accepted: bool

    def __post_init__(self) -> None:
        for field in ("filter_sha256", "kernel_sha256"):
            object.__setattr__(
                self,
                field,
                require_sha256(getattr(self, field), field=field),
            )
        features = _float_tuple(
            self.features,
            name="features",
            minimum_length=len(SPECTRAL_FEATURE_NAMES),
            maximum_length=len(SPECTRAL_FEATURE_NAMES),
        )
        object.__setattr__(self, "features", features)
        object.__setattr__(
            self,
            "anomaly_score",
            _positive(self.anomaly_score, name="anomaly_score", allow_zero=True),
        )
        object.__setattr__(
            self,
            "threshold",
            _positive(self.threshold, name="threshold", allow_zero=True),
        )
        if not isinstance(self.accepted, bool) or self.accepted != (
            self.anomaly_score > self.threshold
        ):
            raise ValueError("accepted does not match the calibrated threshold")

    @property
    def sha256(self) -> str:
        return _digest(
            {
                "accepted": self.accepted,
                "anomaly_score_hex": self.anomaly_score.hex(),
                "features_hex": _hex_floats(self.features),
                "filter_sha256": self.filter_sha256,
                "kernel_sha256": self.kernel_sha256,
                "threshold_hex": self.threshold.hex(),
            }
        )


@dataclass(frozen=True, slots=True)
class MatchedSpectralProposalFilter:
    """Deterministic robust spectral prefilter calibrated on a matched null."""

    dimension: int
    null_kernel_sha256s: tuple[str, ...]
    null_features: tuple[tuple[float, ...], ...]
    quantile: float
    center: tuple[float, ...]
    scale: tuple[float, ...]
    null_scores: tuple[float, ...]
    threshold: float

    def __post_init__(self) -> None:
        _bounded_int(
            self.dimension,
            name="dimension",
            minimum=1,
            maximum=MAX_BVN_DIMENSION,
        )
        hashes = tuple(
            require_sha256(value, field="null_kernel_sha256")
            for value in self.null_kernel_sha256s
        )
        if len(hashes) < 3 or len(set(hashes)) != len(hashes):
            raise ValueError("matched null requires three distinct kernel identities")
        features = tuple(
            _float_tuple(
                row,
                name="null feature",
                minimum_length=len(SPECTRAL_FEATURE_NAMES),
                maximum_length=len(SPECTRAL_FEATURE_NAMES),
            )
            for row in self.null_features
        )
        if len(features) != len(hashes):
            raise ValueError("null hashes and feature rows must have equal length")
        center, scale, scores, threshold = _empirical_parameters(
            features,
            quantile=self.quantile,
        )
        supplied_center = _float_tuple(
            self.center,
            name="center",
            minimum_length=len(SPECTRAL_FEATURE_NAMES),
            maximum_length=len(SPECTRAL_FEATURE_NAMES),
        )
        supplied_scale = _float_tuple(
            self.scale,
            name="scale",
            minimum_length=len(SPECTRAL_FEATURE_NAMES),
            maximum_length=len(SPECTRAL_FEATURE_NAMES),
        )
        supplied_scores = _float_tuple(
            self.null_scores,
            name="null_scores",
            minimum_length=len(hashes),
            maximum_length=len(hashes),
        )
        supplied_threshold = _positive(
            self.threshold,
            name="threshold",
            allow_zero=True,
        )
        if (
            supplied_center != center
            or supplied_scale != scale
            or supplied_scores != scores
            or supplied_threshold != threshold
        ):
            raise ValueError("spectral calibration derived fields disagree with null")
        object.__setattr__(self, "null_kernel_sha256s", hashes)
        object.__setattr__(self, "null_features", features)
        object.__setattr__(self, "quantile", _finite(self.quantile, name="quantile"))
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "null_scores", scores)
        object.__setattr__(self, "threshold", threshold)

    @classmethod
    def calibrate(
        cls,
        null_kernels: Sequence[ArrayLike],
        *,
        quantile: float = 0.95,
    ) -> "MatchedSpectralProposalFilter":
        if len(null_kernels) < 3:
            raise ValueError("matched null requires at least three kernels")
        matrices = tuple(
            _validate_doubly_stochastic(kernel, tolerance=1e-10)
            for kernel in null_kernels
        )
        dimension = matrices[0].shape[0]
        if any(matrix.shape != (dimension, dimension) for matrix in matrices):
            raise ValueError("matched-null kernel dimensions differ")
        hashes = tuple(array_sha256(matrix) for matrix in matrices)
        if len(set(hashes)) != len(hashes):
            raise ValueError("matched-null kernels must have distinct identities")
        features = tuple(spectral_proposal_features(matrix) for matrix in matrices)
        center, scale, scores, threshold = _empirical_parameters(
            features,
            quantile=quantile,
        )
        return cls(
            dimension=dimension,
            null_kernel_sha256s=hashes,
            null_features=features,
            quantile=float(quantile),
            center=center,
            scale=scale,
            null_scores=scores,
            threshold=threshold,
        )

    def evaluate(self, kernel: ArrayLike) -> SpectralProposalDecision:
        value = _validate_doubly_stochastic(kernel, tolerance=1e-10)
        if value.shape != (self.dimension, self.dimension):
            raise ValueError("proposal kernel dimension differs from matched null")
        features = spectral_proposal_features(value)
        anomaly = float(
            np.linalg.norm(
                (np.asarray(features) - np.asarray(self.center)) / np.asarray(self.scale)
            )
            / math.sqrt(len(features))
        )
        return SpectralProposalDecision(
            filter_sha256=self.sha256,
            kernel_sha256=array_sha256(value),
            features=features,
            anomaly_score=anomaly,
            threshold=self.threshold,
            accepted=anomaly > self.threshold,
        )

    def _body(self) -> dict[str, object]:
        return {
            "center_hex": _hex_floats(self.center),
            "dimension": self.dimension,
            "feature_names": list(SPECTRAL_FEATURE_NAMES),
            "null_features_hex": [_hex_floats(row) for row in self.null_features],
            "null_kernel_sha256s": list(self.null_kernel_sha256s),
            "null_scores_hex": _hex_floats(self.null_scores),
            "quantile_hex": self.quantile.hex(),
            "scale_hex": _hex_floats(self.scale),
            "threshold_hex": self.threshold.hex(),
        }

    def to_bytes(self) -> bytes:
        return _encode_document(SPECTRAL_FILTER_SCHEMA, self._body())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "MatchedSpectralProposalFilter":
        _, body = _strict_document(data, schema=SPECTRAL_FILTER_SCHEMA)
        expected = {
            "center_hex",
            "dimension",
            "feature_names",
            "null_features_hex",
            "null_kernel_sha256s",
            "null_scores_hex",
            "quantile_hex",
            "scale_hex",
            "threshold_hex",
        }
        if set(body) != expected or body["feature_names"] != list(
            SPECTRAL_FEATURE_NAMES
        ):
            raise BvNSearchIntegrityError("invalid spectral-filter body")
        if not isinstance(body["null_features_hex"], list) or not isinstance(
            body["null_kernel_sha256s"], list
        ):
            raise BvNSearchIntegrityError("invalid spectral null table")
        try:
            result = cls(
                dimension=body["dimension"],
                null_kernel_sha256s=tuple(body["null_kernel_sha256s"]),
                null_features=tuple(
                    _parse_hex_floats(
                        row,
                        name="null feature",
                        minimum_length=len(SPECTRAL_FEATURE_NAMES),
                        maximum_length=len(SPECTRAL_FEATURE_NAMES),
                    )
                    for row in body["null_features_hex"]
                ),
                quantile=float.fromhex(body["quantile_hex"]),
                center=_parse_hex_floats(
                    body["center_hex"],
                    name="center",
                    minimum_length=len(SPECTRAL_FEATURE_NAMES),
                    maximum_length=len(SPECTRAL_FEATURE_NAMES),
                ),
                scale=_parse_hex_floats(
                    body["scale_hex"],
                    name="scale",
                    minimum_length=len(SPECTRAL_FEATURE_NAMES),
                    maximum_length=len(SPECTRAL_FEATURE_NAMES),
                ),
                null_scores=_parse_hex_floats(
                    body["null_scores_hex"],
                    name="null_scores",
                    minimum_length=len(body["null_kernel_sha256s"]),
                    maximum_length=len(body["null_kernel_sha256s"]),
                ),
                threshold=float.fromhex(body["threshold_hex"]),
            )
        except (TypeError, ValueError) as exc:
            raise BvNSearchIntegrityError("invalid spectral filter") from exc
        if result._body() != body:
            raise BvNSearchIntegrityError("spectral-filter derived fields changed")
        return result


__all__ = [
    "BIRKHOFF_DECOMPOSITION_SCHEMA",
    "BIRKHOFF_RECEIPT_SCHEMA",
    "BvNSearchIntegrityError",
    "BehavioralElite",
    "BehavioralMAPElites",
    "BirkhoffDecomposition",
    "BirkhoffReconstructionReceipt",
    "ContextualThompsonMutation",
    "DiagonalStepProposal",
    "DiagonalSuccessStepAdapter",
    "FiedlerNoveltyReceipt",
    "FiedlerEdgeNoveltyReceipt",
    "MAPElitesArchive",
    "MAX_BVN_COMPONENTS",
    "MAX_BVN_DIMENSION",
    "MAX_PERMUTATION_ATOMS",
    "MatchedSpectralProposalFilter",
    "OPERATOR_GENOME_SCHEMA",
    "OperatorGenome",
    "PermutationAtom",
    "PermutationMixture",
    "SPECTRAL_FEATURE_NAMES",
    "SpectralProposalDecision",
    "ThompsonChoice",
    "ThompsonPosterior",
    "birkhoff_von_neumann",
    "bvn_decompose",
    "canonical_theta",
    "ds_from_theta",
    "fiedler_projector_novelty",
    "fiedler_edge_novelty",
    "mixture_from_theta",
    "spectral_proposal_features",
    "theta_to_doubly_stochastic",
]
