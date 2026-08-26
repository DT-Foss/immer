"""Generic, compositional storage and discharge of completed numerical work.

Compute crystals are deliberately independent of models, prompts, tasks, and
evaluators.  A crystal is a content-addressed safe numerical operator with a
typed tensor ABI and explicit work accounting.  Programs compose crystal
addresses; the VM restores the exact operators and applies them to values that
need not exist when the crystals were created.

Only fixed numerical operators are executable.  This module never evaluates
source text, imports payload-selected code, or calls arbitrary Python hooks.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from .crystal import CrystalStore, CrystalStoreError, ManifestConflictError
from .identity import canonical_json_bytes, require_sha256

COMPUTE_CRYSTAL_SCHEMA = "immer-ooe-compute-crystal/v1"
COMPUTE_PROGRAM_SCHEMA = "immer-ooe-compute-program/v1"
COMPUTE_CHARGE_SCHEMA = "immer-ooe-compute-charge/v1"
COMPUTE_RECEIPT_SCHEMA = "immer-ooe-compute-discharge/v1"
COMPUTE_BANK_MANIFEST_SCHEMA = "immer-ooe-compute-bank-manifest/v1"
COMPUTE_BANK_MANIFEST_STATE = "ooe-compute-crystal-bank-manifest/v1"
COMPUTE_BANK_MANIFEST_COMMIT_SCHEMA = "immer-ooe-compute-bank-commit/v1"
FUSION_WORK_PROVENANCE_SCHEMA = "immer-ooe-fusion-work-provenance/v1"
FUSION_WORK_PROVENANCE_EXTENSION = "immer.fusion-work-provenance"

AFFINE_FLOAT64 = "affine.float64/v1"
PERMUTATION = "permutation/v1"
LOOKUP_FLOAT64 = "lookup.float64/v1"
MARKOV_FLOAT64 = "markov.float64/v1"
CAUSAL_MIX_FLOAT64 = "causal-mix.float64/v1"
OPERATOR_KINDS = frozenset(
    (
        AFFINE_FLOAT64,
        PERMUTATION,
        LOOKUP_FLOAT64,
        MARKOV_FLOAT64,
        CAUSAL_MIX_FLOAT64,
    )
)

MAX_CRYSTAL_BYTES = 32 * 1024 * 1024
MAX_PROGRAM_BYTES = 4 * 1024 * 1024
MAX_CHARGE_BYTES = 8 * 1024 * 1024
MAX_RECEIPT_BYTES = 64 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_NUMERIC_PAYLOAD_BYTES = 24 * 1024 * 1024
MAX_EXTENSION_BYTES = 256 * 1024
MAX_TENSOR_RANK = 16
MAX_TENSOR_DIMENSION = 1_000_000
MAX_PROGRAM_STEPS = 65_536
MAX_PARENTS = 65_536
MAX_MANIFEST_HISTORY_STATES = 100_000
MAX_WORK_UNITS = (1 << 63) - 1

_FLOAT64 = "float64"
_INT64 = "int64"
_DTYPES = frozenset((_FLOAT64, _INT64))
_CRYSTAL_OBJECT_PREFIX = "ooe-compute-crystal-object/v1:"
_PROGRAM_OBJECT_PREFIX = "ooe-compute-program-object/v1:"
_CHARGE_OBJECT_PREFIX = "ooe-compute-charge-object/v1:"
_MANIFEST_HISTORY_PREFIX = "ooe-compute-crystal-bank-history/v1:"
_MANIFEST_COMMIT_PREFIX = "ooe-compute-crystal-bank-commit/v1:"
_MANIFEST_STATE_NAME = COMPUTE_BANK_MANIFEST_STATE
_BANK_LOCK_NAME = ".compute-crystals.lock"
_STATE_NAME_RE = re.compile(
    rb'^\{"format":"immer-ooe-controller-state/v1","generation":[1-9][0-9]*,"name":"([^"\\]*)","payload_base64":"'
)


class ComputeCrystalError(RuntimeError):
    """Base error for the generic stored-compute substrate."""


class ComputeCrystalIntegrityError(ComputeCrystalError):
    """An artifact, lineage, state object, or manifest failed authentication."""


class ComputeCrystalABIError(ComputeCrystalError):
    """A tensor or program edge violates its typed numerical ABI."""


class ComputeCrystalConflictError(ComputeCrystalError):
    """A manifest compare-and-swap precondition is stale."""


class ComputeCrystalMissError(ComputeCrystalError):
    """A requested content-addressed artifact is not published."""


class ComputeCrystalFusionError(ComputeCrystalError):
    """A chain cannot be exactly partially evaluated into one operator."""


def _digest_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_json(data: bytes, *, label: str, maximum: int) -> object:
    if not isinstance(data, bytes):
        raise TypeError(f"{label} must be immutable bytes")
    if len(data) > maximum:
        raise ComputeCrystalIntegrityError(f"{label} exceeds its hard byte limit")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ComputeCrystalIntegrityError(
            f"{label} is invalid canonical JSON"
        ) from exc
    if canonical_json_bytes(value) != data:
        raise ComputeCrystalIntegrityError(f"{label} is not canonical JSON")
    return value


def _bounded_uint(value: object, *, field: str, positive: bool = False) -> int:
    lower = 1 if positive else 0
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < lower
        or value > MAX_WORK_UNITS
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must be a bounded {qualifier} integer")
    return value


def _checked_work(value: int, *, field: str) -> int:
    return _bounded_uint(value, field=field)


def _checked_add(left: int, right: int, *, field: str) -> int:
    value = left + right
    return _checked_work(value, field=field)


def _checked_multiply(left: int, right: int, *, field: str) -> int:
    value = left * right
    return _checked_work(value, field=field)


def _shape_tuple(value: object, *, field: str, allow_empty: bool) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a shape sequence")
    if len(value) > MAX_TENSOR_RANK:
        raise ValueError(f"{field} exceeds the tensor-rank bound")
    if not allow_empty and not value:
        raise ValueError(f"{field} must not be empty")
    result: list[int] = []
    element_count = 1
    for dimension in value:
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or not 1 <= dimension <= MAX_TENSOR_DIMENSION
        ):
            raise ValueError(f"{field} has an invalid dimension")
        element_count *= dimension
        if element_count * 8 > MAX_NUMERIC_PAYLOAD_BYTES:
            raise ValueError(f"{field} exceeds the numeric payload byte bound")
        result.append(dimension)
    return tuple(result)


def _canonical_float64(value: object, *, field: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim > MAX_TENSOR_RANK:
        raise ValueError(f"{field} exceeds the tensor-rank bound")
    if array.size * 8 > MAX_NUMERIC_PAYLOAD_BYTES:
        raise ValueError(f"{field} exceeds the numeric payload byte bound")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must contain only finite float64 values")
    result = np.array(array, dtype="<f8", order="C", copy=True)
    result[result == 0.0] = 0.0
    return cast(NDArray[np.float64], result)


def _canonical_int64(value: object, *, field: str) -> NDArray[np.int64]:
    array = np.asarray(value)
    if array.dtype.kind not in "iu":
        raise ValueError(f"{field} must contain integer values")
    if array.ndim > MAX_TENSOR_RANK:
        raise ValueError(f"{field} exceeds the tensor-rank bound")
    if array.size * 8 > MAX_NUMERIC_PAYLOAD_BYTES:
        raise ValueError(f"{field} exceeds the numeric payload byte bound")
    if array.size and (
        np.any(array < np.iinfo(np.int64).min) or np.any(array > np.iinfo(np.int64).max)
    ):
        raise ValueError(f"{field} contains an out-of-range int64 value")
    return cast(NDArray[np.int64], np.array(array, dtype="<i8", order="C", copy=True))


def _array_document(array: NDArray[Any], *, dtype: str) -> dict[str, object]:
    if dtype == _FLOAT64:
        value = _canonical_float64(array, field="numeric payload")
        storage = "float64-le"
    elif dtype == _INT64:
        value = _canonical_int64(array, field="numeric payload")
        storage = "int64-le"
    else:
        raise ValueError("unsupported numeric payload dtype")
    return {
        "data_base64": base64.b64encode(value.tobytes(order="C")).decode("ascii"),
        "shape": list(value.shape),
        "storage": storage,
    }


def _decode_array(
    value: object,
    *,
    dtype: Literal["float64", "int64"],
    field: str,
    allow_scalar: bool,
) -> NDArray[Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "data_base64",
        "shape",
        "storage",
    }:
        raise ValueError(f"{field} has an invalid numeric descriptor")
    expected_storage = "float64-le" if dtype == _FLOAT64 else "int64-le"
    if value.get("storage") != expected_storage:
        raise ValueError(f"{field} has an unsupported storage dtype")
    shape = _shape_tuple(
        value.get("shape"), field=f"{field}.shape", allow_empty=allow_scalar
    )
    element_count = math.prod(shape)
    decoded_length = element_count * 8
    encoded = value.get("data_base64")
    expected_encoded_length = 4 * ((decoded_length + 2) // 3)
    if (
        not isinstance(encoded, str)
        or not encoded.isascii()
        or len(encoded) != expected_encoded_length
    ):
        raise ValueError(f"{field} has an invalid encoded byte length")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} has invalid base64 data") from exc
    if len(raw) != decoded_length or base64.b64encode(raw).decode("ascii") != encoded:
        raise ValueError(f"{field} has non-canonical numeric bytes")
    numpy_dtype = "<f8" if dtype == _FLOAT64 else "<i8"
    array = np.frombuffer(raw, dtype=numpy_dtype).reshape(shape)
    if dtype == _FLOAT64:
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{field} contains a non-finite float64 value")
        if np.any((array == 0.0) & np.signbit(array)):
            raise ValueError(f"{field} contains non-canonical negative zero")
    return array


def _canonical_extensions(value: Mapping[str, object] | None) -> bytes:
    document: dict[str, object] = {} if value is None else dict(value)
    for key in document:
        if (
            not isinstance(key, str)
            or not key
            or key != key.strip()
            or "\x00" in key
            or len(key.encode("utf-8")) > 256
        ):
            raise ValueError("extension names must be canonical non-empty text")
    data = canonical_json_bytes(document)
    if len(data) > MAX_EXTENSION_BYTES:
        raise ValueError("extensions exceed their hard byte limit")
    decoded = _strict_json(
        data, label="compute-crystal extensions", maximum=MAX_EXTENSION_BYTES
    )
    if not isinstance(decoded, dict):
        raise ValueError("extensions must be a JSON object")
    return data


@dataclass(frozen=True, slots=True)
class NumericalABI:
    """A tensor dtype plus a fixed trailing shape and arbitrary leading axes."""

    dtype: str
    trailing_shape: tuple[int, ...]

    FORMAT = "immer-numerical-abi/v1"

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPES:
            raise ValueError("ABI dtype must be float64 or int64")
        if not isinstance(self.trailing_shape, tuple):
            raise TypeError("ABI trailing_shape must be an immutable tuple")
        object.__setattr__(
            self,
            "trailing_shape",
            _shape_tuple(
                self.trailing_shape,
                field="ABI trailing_shape",
                allow_empty=True,
            ),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "dtype": self.dtype,
            "format": self.FORMAT,
            "trailing_shape": list(self.trailing_shape),
        }

    @classmethod
    def from_record(cls, value: object) -> "NumericalABI":
        if (
            not isinstance(value, Mapping)
            or set(value) != {"dtype", "format", "trailing_shape"}
            or value.get("format") != cls.FORMAT
        ):
            raise ValueError("invalid numerical ABI")
        trailing = value.get("trailing_shape")
        if not isinstance(trailing, list):
            raise ValueError("ABI trailing_shape must be a list")
        return cls(dtype=cast(str, value.get("dtype")), trailing_shape=tuple(trailing))

    @property
    def sha256(self) -> str:
        return _digest_json(self.to_record())

    def validate(self, value: object, *, field: str = "tensor") -> NDArray[Any]:
        if type(value) is not np.ndarray:
            raise ComputeCrystalABIError(
                f"{field} must be an exact numpy.ndarray; convert trusted values explicitly"
            )
        array = cast(NDArray[Any], value)
        expected = np.dtype(np.float64 if self.dtype == _FLOAT64 else np.int64)
        if array.dtype != expected:
            raise ComputeCrystalABIError(
                f"{field} dtype is {array.dtype}, expected {self.dtype}"
            )
        suffix_rank = len(self.trailing_shape)
        if suffix_rank > array.ndim or (
            suffix_rank and tuple(array.shape[-suffix_rank:]) != self.trailing_shape
        ):
            raise ComputeCrystalABIError(
                f"{field} shape {array.shape} does not end in {self.trailing_shape}"
            )
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise ComputeCrystalABIError(f"{field} contains non-finite values")
        return array

    def application_count(self, value: object) -> int:
        array = self.validate(value)
        suffix_rank = len(self.trailing_shape)
        leading = array.shape if suffix_rank == 0 else array.shape[:-suffix_rank]
        return math.prod(leading) if leading else 1


def _operator_contract(
    operator_kind: str, payload: object
) -> tuple[NumericalABI, NumericalABI, int]:
    if not isinstance(payload, Mapping):
        raise ValueError("operator payload must be a JSON object")

    if operator_kind == AFFINE_FLOAT64:
        if set(payload) != {"bias", "matrix"}:
            raise ValueError("affine payload fields are invalid")
        matrix = _decode_array(
            payload["matrix"],
            dtype=_FLOAT64,
            field="affine.matrix",
            allow_scalar=False,
        )
        bias = _decode_array(
            payload["bias"],
            dtype=_FLOAT64,
            field="affine.bias",
            allow_scalar=False,
        )
        if matrix.ndim != 2 or bias.ndim != 1 or bias.shape != (matrix.shape[0],):
            raise ValueError("affine matrix/bias shapes are incompatible")
        output_dimension, input_dimension = matrix.shape
        work = _checked_work(
            2 * int(input_dimension) * int(output_dimension),
            field="affine discharge work",
        )
        return (
            NumericalABI(_FLOAT64, (int(input_dimension),)),
            NumericalABI(_FLOAT64, (int(output_dimension),)),
            work,
        )

    if operator_kind == PERMUTATION:
        if set(payload) != {"dtype", "indices"}:
            raise ValueError("permutation payload fields are invalid")
        dtype = payload.get("dtype")
        if dtype not in _DTYPES:
            raise ValueError("permutation dtype must be float64 or int64")
        indices = _decode_array(
            payload["indices"],
            dtype=_INT64,
            field="permutation.indices",
            allow_scalar=False,
        )
        if indices.ndim != 1 or indices.size < 1:
            raise ValueError("permutation indices must be a non-empty vector")
        expected = np.arange(indices.size, dtype=np.int64)
        if not np.array_equal(np.sort(indices), expected):
            raise ValueError("permutation indices must contain each position once")
        abi = NumericalABI(cast(str, dtype), (int(indices.size),))
        return abi, abi, int(indices.size)

    if operator_kind == LOOKUP_FLOAT64:
        if set(payload) != {"table"}:
            raise ValueError("lookup payload fields are invalid")
        table = _decode_array(
            payload["table"],
            dtype=_FLOAT64,
            field="lookup.table",
            allow_scalar=False,
        )
        if table.ndim < 1 or table.shape[0] < 1:
            raise ValueError("lookup table must have at least one row")
        value_shape = tuple(int(value) for value in table.shape[1:])
        work = max(1, math.prod(value_shape))
        return (
            NumericalABI(_INT64, ()),
            NumericalABI(_FLOAT64, value_shape),
            _checked_work(work, field="lookup discharge work"),
        )

    if operator_kind == MARKOV_FLOAT64:
        if set(payload) != {"kernel"}:
            raise ValueError("Markov payload fields are invalid")
        kernel = _decode_array(
            payload["kernel"],
            dtype=_FLOAT64,
            field="markov.kernel",
            allow_scalar=False,
        )
        if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
            raise ValueError("Markov kernel must be a square matrix")
        if np.any(kernel < 0.0):
            raise ValueError("Markov kernel must be non-negative")
        if not np.allclose(kernel.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("Markov kernel rows must sum to one")
        dimension = int(kernel.shape[0])
        work = _checked_work(
            dimension * (2 * dimension - 1),
            field="Markov discharge work",
        )
        abi = NumericalABI(_FLOAT64, (dimension,))
        return abi, abi, work

    if operator_kind == CAUSAL_MIX_FLOAT64:
        if set(payload) != {"kernel"}:
            raise ValueError("causal-mix payload fields are invalid")
        kernel = _decode_array(
            payload["kernel"],
            dtype=_FLOAT64,
            field="causal-mix.kernel",
            allow_scalar=False,
        )
        if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
            raise ValueError("causal-mix kernel must be a square matrix")
        if np.any(kernel < 0.0):
            raise ValueError("causal-mix kernel must be non-negative")
        dimension = int(kernel.shape[0])
        if any(np.any(kernel[row, row + 1 :] != 0.0) for row in range(dimension)):
            raise ValueError("causal-mix kernel must be exactly causal")
        if not np.allclose(kernel.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("causal-mix kernel rows must sum to one")
        work = _checked_work(
            dimension * (2 * dimension - 1),
            field="causal-mix discharge work",
        )
        abi = NumericalABI(_FLOAT64, (dimension,))
        return abi, abi, work

    raise ValueError("unsupported compute-crystal operator kind")


def _operator_arrays(crystal: "ComputeCrystal") -> tuple[NDArray[Any], ...]:
    payload = crystal.payload
    if crystal.operator_kind == AFFINE_FLOAT64:
        return (
            _decode_array(
                payload["matrix"],
                dtype=_FLOAT64,
                field="affine.matrix",
                allow_scalar=False,
            ),
            _decode_array(
                payload["bias"],
                dtype=_FLOAT64,
                field="affine.bias",
                allow_scalar=False,
            ),
        )
    if crystal.operator_kind == PERMUTATION:
        return (
            _decode_array(
                payload["indices"],
                dtype=_INT64,
                field="permutation.indices",
                allow_scalar=False,
            ),
        )
    if crystal.operator_kind == LOOKUP_FLOAT64:
        return (
            _decode_array(
                payload["table"],
                dtype=_FLOAT64,
                field="lookup.table",
                allow_scalar=False,
            ),
        )
    if crystal.operator_kind == MARKOV_FLOAT64:
        return (
            _decode_array(
                payload["kernel"],
                dtype=_FLOAT64,
                field="markov.kernel",
                allow_scalar=False,
            ),
        )
    if crystal.operator_kind == CAUSAL_MIX_FLOAT64:
        return (
            _decode_array(
                payload["kernel"],
                dtype=_FLOAT64,
                field="causal-mix.kernel",
                allow_scalar=False,
            ),
        )
    raise AssertionError("validated operator kind changed")


@dataclass(frozen=True, slots=True)
class ComputeCrystal:
    """One canonical, task-agnostic, executable stored-compute artifact."""

    operator_kind: str
    input_abi: NumericalABI
    output_abi: NumericalABI
    payload_json: bytes
    parent_sha256s: tuple[str, ...]
    discharge_work_units: int
    extensions_json: bytes = b"{}"

    FORMAT = COMPUTE_CRYSTAL_SCHEMA

    def __post_init__(self) -> None:
        if self.operator_kind not in OPERATOR_KINDS:
            raise ValueError("unsupported compute-crystal operator kind")
        if not isinstance(self.input_abi, NumericalABI) or not isinstance(
            self.output_abi, NumericalABI
        ):
            raise TypeError("input_abi and output_abi must be NumericalABI values")
        payload = _strict_json(
            self.payload_json,
            label="compute-crystal operator payload",
            maximum=MAX_NUMERIC_PAYLOAD_BYTES,
        )
        expected_input, expected_output, expected_work = _operator_contract(
            self.operator_kind, payload
        )
        if self.input_abi != expected_input or self.output_abi != expected_output:
            raise ValueError("operator payload and numerical ABI disagree")
        discharge = _bounded_uint(
            self.discharge_work_units,
            field="discharge_work_units",
            positive=True,
        )
        if discharge != expected_work:
            raise ValueError("discharge work does not match the numerical operator")
        if not isinstance(self.parent_sha256s, tuple):
            raise TypeError("parent_sha256s must be an immutable tuple")
        if len(self.parent_sha256s) > MAX_PARENTS:
            raise ValueError("compute crystal has too many parents")
        parents = tuple(
            require_sha256(digest, field="parent_sha256s")
            for digest in self.parent_sha256s
        )
        if parents and len(parents) < 2:
            raise ValueError("a fused crystal must bind at least two parents")
        extensions = _strict_json(
            self.extensions_json,
            label="compute-crystal extensions",
            maximum=MAX_EXTENSION_BYTES,
        )
        if not isinstance(extensions, dict):
            raise ValueError("compute-crystal extensions must be a JSON object")
        if _canonical_extensions(extensions) != self.extensions_json:
            raise ValueError("compute-crystal extensions are not canonical")
        if not parents and FUSION_WORK_PROVENANCE_EXTENSION in extensions:
            raise ValueError("a primitive crystal cannot carry fusion-work provenance")
        object.__setattr__(self, "parent_sha256s", parents)
        object.__setattr__(self, "discharge_work_units", discharge)

    @classmethod
    def _from_operator(
        cls,
        *,
        operator_kind: str,
        payload: Mapping[str, object],
        parents: Sequence["ComputeCrystal"] = (),
        extensions: Mapping[str, object] | None = None,
    ) -> "ComputeCrystal":
        payload_json = canonical_json_bytes(dict(payload))
        input_abi, output_abi, discharge = _operator_contract(
            operator_kind, dict(payload)
        )
        parent_tuple = tuple(parents)
        if parent_tuple:
            if len(parent_tuple) < 2 or any(
                not isinstance(parent, ComputeCrystal) for parent in parent_tuple
            ):
                raise TypeError("fused parents must be at least two ComputeCrystals")
        return cls(
            operator_kind=operator_kind,
            input_abi=input_abi,
            output_abi=output_abi,
            payload_json=payload_json,
            parent_sha256s=tuple(parent.sha256 for parent in parent_tuple),
            discharge_work_units=discharge,
            extensions_json=_canonical_extensions(extensions),
        )

    @classmethod
    def affine(
        cls,
        matrix: object,
        bias: object | None = None,
        *,
        extensions: Mapping[str, object] | None = None,
    ) -> "ComputeCrystal":
        weight = _canonical_float64(matrix, field="affine matrix")
        if weight.ndim != 2 or min(weight.shape) < 1:
            raise ValueError("affine matrix must be a non-empty rank-2 array")
        offset = (
            np.zeros(weight.shape[0], dtype=np.float64)
            if bias is None
            else _canonical_float64(bias, field="affine bias")
        )
        if offset.shape != (weight.shape[0],):
            raise ValueError("affine bias must match the output dimension")
        return cls._from_operator(
            operator_kind=AFFINE_FLOAT64,
            payload={
                "bias": _array_document(offset, dtype=_FLOAT64),
                "matrix": _array_document(weight, dtype=_FLOAT64),
            },
            extensions=extensions,
        )

    @classmethod
    def permutation(
        cls,
        indices: object,
        *,
        dtype: Literal["float64", "int64"] = _FLOAT64,
        extensions: Mapping[str, object] | None = None,
    ) -> "ComputeCrystal":
        order = _canonical_int64(indices, field="permutation indices")
        return cls._from_operator(
            operator_kind=PERMUTATION,
            payload={
                "dtype": dtype,
                "indices": _array_document(order, dtype=_INT64),
            },
            extensions=extensions,
        )

    @classmethod
    def lookup(
        cls,
        table: object,
        *,
        extensions: Mapping[str, object] | None = None,
    ) -> "ComputeCrystal":
        values = _canonical_float64(table, field="lookup table")
        if values.ndim < 1 or values.shape[0] < 1:
            raise ValueError("lookup table must contain at least one row")
        return cls._from_operator(
            operator_kind=LOOKUP_FLOAT64,
            payload={"table": _array_document(values, dtype=_FLOAT64)},
            extensions=extensions,
        )

    @classmethod
    def markov(
        cls,
        kernel: object,
        *,
        extensions: Mapping[str, object] | None = None,
    ) -> "ComputeCrystal":
        transition = _canonical_float64(kernel, field="Markov kernel")
        return cls._from_operator(
            operator_kind=MARKOV_FLOAT64,
            payload={"kernel": _array_document(transition, dtype=_FLOAT64)},
            extensions=extensions,
        )

    @classmethod
    def causal_mix(
        cls,
        kernel: object,
        *,
        extensions: Mapping[str, object] | None = None,
    ) -> "ComputeCrystal":
        transition = _canonical_float64(kernel, field="causal-mix kernel")
        return cls._from_operator(
            operator_kind=CAUSAL_MIX_FLOAT64,
            payload={"kernel": _array_document(transition, dtype=_FLOAT64)},
            extensions=extensions,
        )

    @property
    def payload(self) -> dict[str, object]:
        value = json.loads(self.payload_json)
        if not isinstance(value, dict):
            raise AssertionError("validated operator payload changed type")
        return value

    @property
    def extensions(self) -> dict[str, object]:
        value = json.loads(self.extensions_json)
        if not isinstance(value, dict):
            raise AssertionError("validated extensions changed type")
        return value

    @property
    def is_fused(self) -> bool:
        return bool(self.parent_sha256s)

    def as_record(self) -> dict[str, object]:
        return {
            "abi": {
                "input": self.input_abi.to_record(),
                "output": self.output_abi.to_record(),
            },
            "discharge_work_units": self.discharge_work_units,
            "extensions": self.extensions,
            "operator_kind": self.operator_kind,
            "parents": list(self.parent_sha256s),
            "payload": self.payload,
        }

    def to_document(self) -> dict[str, object]:
        body = self.as_record()
        return {
            "body": body,
            "body_sha256": _digest_json(body),
            "schema": self.FORMAT,
        }

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_document())
        if len(data) > MAX_CRYSTAL_BYTES:
            raise ValueError("compute crystal exceeds its hard byte limit")
        return data

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "ComputeCrystal":
        value = _strict_json(data, label="compute crystal", maximum=MAX_CRYSTAL_BYTES)
        if (
            not isinstance(value, Mapping)
            or set(value) != {"body", "body_sha256", "schema"}
            or value.get("schema") != cls.FORMAT
        ):
            raise ComputeCrystalIntegrityError("compute-crystal envelope is invalid")
        body = value.get("body")
        expected = {
            "abi",
            "discharge_work_units",
            "extensions",
            "operator_kind",
            "parents",
            "payload",
        }
        if not isinstance(body, Mapping) or set(body) != expected:
            raise ComputeCrystalIntegrityError("compute-crystal body is invalid")
        try:
            claimed = require_sha256(value.get("body_sha256"), field="body_sha256")
        except ValueError as exc:
            raise ComputeCrystalIntegrityError(
                "compute-crystal body SHA-256 is invalid"
            ) from exc
        if claimed != _digest_json(body):
            raise ComputeCrystalIntegrityError("compute-crystal body SHA-256 mismatch")
        abi = body.get("abi")
        parents = body.get("parents")
        if not isinstance(abi, Mapping) or set(abi) != {"input", "output"}:
            raise ComputeCrystalIntegrityError("compute-crystal ABI record is invalid")
        if not isinstance(parents, list):
            raise ComputeCrystalIntegrityError("compute-crystal parents must be a list")
        extensions = body.get("extensions")
        payload = body.get("payload")
        if not isinstance(extensions, Mapping) or not isinstance(payload, Mapping):
            raise ComputeCrystalIntegrityError(
                "compute-crystal payload or extensions are invalid"
            )
        try:
            crystal = cls(
                operator_kind=cast(str, body.get("operator_kind")),
                input_abi=NumericalABI.from_record(abi.get("input")),
                output_abi=NumericalABI.from_record(abi.get("output")),
                payload_json=canonical_json_bytes(dict(payload)),
                parent_sha256s=tuple(parents),
                discharge_work_units=cast(int, body.get("discharge_work_units")),
                extensions_json=canonical_json_bytes(dict(extensions)),
            )
        except ComputeCrystalIntegrityError:
            raise
        except (TypeError, ValueError, OverflowError) as exc:
            raise ComputeCrystalIntegrityError(
                "compute-crystal validation failed"
            ) from exc
        if crystal.to_bytes() != data:
            raise ComputeCrystalIntegrityError(
                "compute crystal failed canonical reconstruction"
            )
        return crystal

    def apply(self, value: object) -> NDArray[Any]:
        array = self.input_abi.validate(value, field="compute-crystal input")
        operator = _operator_arrays(self)
        if self.operator_kind == AFFINE_FLOAT64:
            matrix, bias = operator
            output = np.einsum("...i,oi->...o", array, matrix, optimize=False) + bias
        elif self.operator_kind == PERMUTATION:
            (indices,) = operator
            output = np.take(array, indices, axis=-1)
        elif self.operator_kind == LOOKUP_FLOAT64:
            (table,) = operator
            indices = cast(NDArray[np.int64], array)
            if np.any(indices < 0) or np.any(indices >= table.shape[0]):
                raise ComputeCrystalABIError("lookup index lies outside the table")
            output = table[indices]
        elif self.operator_kind == MARKOV_FLOAT64:
            (kernel,) = operator
            output = np.einsum("...i,ij->...j", array, kernel, optimize=False)
        elif self.operator_kind == CAUSAL_MIX_FLOAT64:
            (kernel,) = operator
            output = np.einsum("...k,qk->...q", array, kernel, optimize=False)
        else:
            raise AssertionError("validated operator kind changed")
        result = np.asarray(output)
        self.output_abi.validate(result, field="compute-crystal output")
        return result


def _validate_chain(crystals: Sequence[ComputeCrystal]) -> tuple[ComputeCrystal, ...]:
    chain = tuple(crystals)
    if not chain:
        raise ComputeCrystalABIError("a compute program needs at least one crystal")
    if len(chain) > MAX_PROGRAM_STEPS:
        raise ComputeCrystalABIError("compute program exceeds its step bound")
    for index, crystal in enumerate(chain):
        if not isinstance(crystal, ComputeCrystal):
            raise TypeError("compute program steps must be ComputeCrystals")
        if index and chain[index - 1].output_abi != crystal.input_abi:
            raise ComputeCrystalABIError(
                f"program ABI mismatch between steps {index - 1} and {index}"
            )
    return chain


@dataclass(frozen=True, slots=True)
class ComputeProgram:
    """A canonical ordered composition of content-addressed compute crystals."""

    crystal_sha256s: tuple[str, ...]
    input_abi: NumericalABI
    output_abi: NumericalABI

    FORMAT = COMPUTE_PROGRAM_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.crystal_sha256s, tuple):
            raise TypeError("crystal_sha256s must be an immutable tuple")
        if not 1 <= len(self.crystal_sha256s) <= MAX_PROGRAM_STEPS:
            raise ValueError("compute program must have a bounded non-empty step list")
        object.__setattr__(
            self,
            "crystal_sha256s",
            tuple(
                require_sha256(digest, field="crystal_sha256s")
                for digest in self.crystal_sha256s
            ),
        )
        if not isinstance(self.input_abi, NumericalABI) or not isinstance(
            self.output_abi, NumericalABI
        ):
            raise TypeError("program ABIs must be NumericalABI values")

    @classmethod
    def compose(cls, crystals: Sequence[ComputeCrystal]) -> "ComputeProgram":
        chain = _validate_chain(crystals)
        return cls(
            crystal_sha256s=tuple(crystal.sha256 for crystal in chain),
            input_abi=chain[0].input_abi,
            output_abi=chain[-1].output_abi,
        )

    def validate_crystals(
        self, crystals: Sequence[ComputeCrystal]
    ) -> tuple[ComputeCrystal, ...]:
        chain = _validate_chain(crystals)
        if tuple(crystal.sha256 for crystal in chain) != self.crystal_sha256s:
            raise ComputeCrystalIntegrityError(
                "program step addresses do not match the restored crystals"
            )
        if (
            self.input_abi != chain[0].input_abi
            or self.output_abi != chain[-1].output_abi
        ):
            raise ComputeCrystalIntegrityError(
                "program envelope ABIs do not match its crystal chain"
            )
        return chain

    def as_record(self) -> dict[str, object]:
        return {
            "crystals": list(self.crystal_sha256s),
            "input_abi": self.input_abi.to_record(),
            "output_abi": self.output_abi.to_record(),
        }

    def to_document(self) -> dict[str, object]:
        body = self.as_record()
        return {
            "body": body,
            "body_sha256": _digest_json(body),
            "schema": self.FORMAT,
        }

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_document())
        if len(data) > MAX_PROGRAM_BYTES:
            raise ValueError("compute program exceeds its hard byte limit")
        return data

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "ComputeProgram":
        value = _strict_json(data, label="compute program", maximum=MAX_PROGRAM_BYTES)
        if (
            not isinstance(value, Mapping)
            or set(value) != {"body", "body_sha256", "schema"}
            or value.get("schema") != cls.FORMAT
        ):
            raise ComputeCrystalIntegrityError("compute-program envelope is invalid")
        body = value.get("body")
        if not isinstance(body, Mapping) or set(body) != {
            "crystals",
            "input_abi",
            "output_abi",
        }:
            raise ComputeCrystalIntegrityError("compute-program body is invalid")
        try:
            claimed = require_sha256(value.get("body_sha256"), field="body_sha256")
        except ValueError as exc:
            raise ComputeCrystalIntegrityError(
                "compute-program body SHA-256 is invalid"
            ) from exc
        if claimed != _digest_json(body):
            raise ComputeCrystalIntegrityError("compute-program body SHA-256 mismatch")
        crystals = body.get("crystals")
        if not isinstance(crystals, list):
            raise ComputeCrystalIntegrityError("program crystals must be a list")
        try:
            program = cls(
                crystal_sha256s=tuple(crystals),
                input_abi=NumericalABI.from_record(body.get("input_abi")),
                output_abi=NumericalABI.from_record(body.get("output_abi")),
            )
        except (TypeError, ValueError) as exc:
            raise ComputeCrystalIntegrityError(
                "compute-program validation failed"
            ) from exc
        if program.to_bytes() != data:
            raise ComputeCrystalIntegrityError(
                "compute program failed canonical reconstruction"
            )
        return program


def _fusion_work_record(
    chain: Sequence[ComputeCrystal],
    *,
    live_work_units: int,
) -> dict[str, object]:
    sources = _validate_chain(chain)
    source_work_units = tuple(
        _provenance_source_work_units(crystal) for crystal in sources
    )
    equivalent = 0
    for work in source_work_units:
        equivalent = _checked_add(
            equivalent,
            work,
            field="fusion equivalent source work",
        )
    live = _bounded_uint(
        live_work_units,
        field="fusion live discharge work",
        positive=True,
    )
    return {
        "equivalent_unfused_source_work": equivalent,
        "format": FUSION_WORK_PROVENANCE_SCHEMA,
        "historical_work_released": max(0, equivalent - live),
        "live_discharge_work": live,
        "source_crystal_sha256s": [crystal.sha256 for crystal in sources],
        "source_equivalent_work_units": list(source_work_units),
    }


def fusion_work_provenance(crystal: ComputeCrystal) -> dict[str, object] | None:
    """Return sealed transitive fusion-work metadata when it is present.

    Legacy fused crystals without this optional extension retain their original
    immediate-parent accounting.  Fusion and bank-lineage reconstruction derive
    this record again from the direct parents before a charge can be published.
    """

    if not isinstance(crystal, ComputeCrystal):
        raise TypeError("crystal must be a ComputeCrystal")
    record = crystal.extensions.get(FUSION_WORK_PROVENANCE_EXTENSION)
    if record is None:
        return None
    if not crystal.parent_sha256s:
        raise ComputeCrystalIntegrityError(
            "a primitive crystal cannot claim fusion-work provenance"
        )
    if not isinstance(record, Mapping):
        raise ComputeCrystalIntegrityError(
            "fusion-work provenance must be a JSON object"
        )
    expected_fields = {
        "equivalent_unfused_source_work",
        "format",
        "historical_work_released",
        "live_discharge_work",
        "source_crystal_sha256s",
        "source_equivalent_work_units",
    }
    if set(record) != expected_fields or record.get("format") != (
        FUSION_WORK_PROVENANCE_SCHEMA
    ):
        raise ComputeCrystalIntegrityError("fusion-work provenance fields are invalid")
    sources = record.get("source_crystal_sha256s")
    work_units = record.get("source_equivalent_work_units")
    if not isinstance(sources, list) or not isinstance(work_units, list):
        raise ComputeCrystalIntegrityError(
            "fusion-work provenance source inventory is invalid"
        )
    try:
        normalized_sources = tuple(
            require_sha256(value, field="fusion source crystal") for value in sources
        )
        normalized_work = tuple(
            _bounded_uint(value, field="fusion source work", positive=True)
            for value in work_units
        )
        equivalent = _bounded_uint(
            record.get("equivalent_unfused_source_work"),
            field="fusion equivalent source work",
            positive=True,
        )
        live = _bounded_uint(
            record.get("live_discharge_work"),
            field="fusion live discharge work",
            positive=True,
        )
        released = _bounded_uint(
            record.get("historical_work_released"),
            field="fusion historical work released",
        )
    except (TypeError, ValueError) as exc:
        raise ComputeCrystalIntegrityError(
            "fusion-work provenance values are invalid"
        ) from exc
    if (
        normalized_sources != crystal.parent_sha256s
        or len(normalized_work) != len(normalized_sources)
        or equivalent != sum(normalized_work)
        or live != crystal.discharge_work_units
        or released != max(0, equivalent - live)
    ):
        raise ComputeCrystalIntegrityError(
            "fusion-work provenance disagrees with the crystal"
        )
    return dict(record)


def equivalent_unfused_work_units(crystal: ComputeCrystal) -> int:
    """Return the sealed source-work total represented by one crystal."""

    record = fusion_work_provenance(crystal)
    if record is None:
        return crystal.discharge_work_units
    return cast(int, record["equivalent_unfused_source_work"])


def _provenance_source_work_units(crystal: ComputeCrystal) -> int:
    if crystal.is_fused and fusion_work_provenance(crystal) is None:
        raise ComputeCrystalFusionError(
            "legacy fused source lacks transitive work provenance"
        )
    return equivalent_unfused_work_units(crystal)


def _validate_fusion_work_provenance(
    result: ComputeCrystal,
    chain: Sequence[ComputeCrystal],
) -> None:
    record = fusion_work_provenance(result)
    if record is None:
        return
    expected = _fusion_work_record(
        chain,
        live_work_units=result.discharge_work_units,
    )
    if record != expected:
        raise ComputeCrystalIntegrityError(
            "fusion-work provenance disagrees with its direct source chain"
        )


def fuse_affine_chain(
    crystals: Sequence[ComputeCrystal],
    *,
    extensions: Mapping[str, object] | None = None,
) -> ComputeCrystal:
    chain = _validate_chain(crystals)
    if len(chain) < 2 or any(
        crystal.operator_kind != AFFINE_FLOAT64 for crystal in chain
    ):
        raise ComputeCrystalFusionError(
            "affine fusion needs two or more affine crystals"
        )
    first_matrix, first_bias = _operator_arrays(chain[0])
    matrix = np.array(first_matrix, dtype=np.float64, copy=True)
    bias = np.array(first_bias, dtype=np.float64, copy=True)
    for crystal in chain[1:]:
        next_matrix, next_bias = _operator_arrays(crystal)
        bias = np.einsum("oi,i->o", next_matrix, bias, optimize=False) + next_bias
        matrix = np.einsum("oi,ij->oj", next_matrix, matrix, optimize=False)
    matrix[matrix == 0.0] = 0.0
    bias[bias == 0.0] = 0.0
    payload = {
        "bias": _array_document(bias, dtype=_FLOAT64),
        "matrix": _array_document(matrix, dtype=_FLOAT64),
    }
    result = ComputeCrystal._from_operator(
        operator_kind=AFFINE_FLOAT64,
        payload=payload,
        parents=chain,
        extensions=extensions,
    )
    _validate_fusion_work_provenance(result, chain)
    return result


def fuse_permutation_chain(
    crystals: Sequence[ComputeCrystal],
    *,
    extensions: Mapping[str, object] | None = None,
) -> ComputeCrystal:
    chain = _validate_chain(crystals)
    if len(chain) < 2 or any(crystal.operator_kind != PERMUTATION for crystal in chain):
        raise ComputeCrystalFusionError(
            "permutation fusion needs two or more permutation crystals"
        )
    (first,) = _operator_arrays(chain[0])
    indices = np.array(first, dtype=np.int64, copy=True)
    for crystal in chain[1:]:
        (following,) = _operator_arrays(crystal)
        indices = indices[following]
    payload = {
        "dtype": chain[0].input_abi.dtype,
        "indices": _array_document(indices, dtype=_INT64),
    }
    result = ComputeCrystal._from_operator(
        operator_kind=PERMUTATION,
        payload=payload,
        parents=chain,
        extensions=extensions,
    )
    _validate_fusion_work_provenance(result, chain)
    return result


def fuse_markov_chain(
    crystals: Sequence[ComputeCrystal],
    *,
    extensions: Mapping[str, object] | None = None,
) -> ComputeCrystal:
    chain = _validate_chain(crystals)
    if len(chain) < 2 or any(
        crystal.operator_kind != MARKOV_FLOAT64 for crystal in chain
    ):
        raise ComputeCrystalFusionError("Markov fusion needs two or more kernels")
    (first,) = _operator_arrays(chain[0])
    kernel = np.array(first, dtype=np.float64, copy=True)
    for crystal in chain[1:]:
        (following,) = _operator_arrays(crystal)
        kernel = np.einsum("ij,jk->ik", kernel, following, optimize=False)
    kernel[kernel == 0.0] = 0.0
    payload = {"kernel": _array_document(kernel, dtype=_FLOAT64)}
    result = ComputeCrystal._from_operator(
        operator_kind=MARKOV_FLOAT64,
        payload=payload,
        parents=chain,
        extensions=extensions,
    )
    _validate_fusion_work_provenance(result, chain)
    return result


def fuse_causal_mix_chain(
    crystals: Sequence[ComputeCrystal],
    *,
    extensions: Mapping[str, object] | None = None,
) -> ComputeCrystal:
    chain = _validate_chain(crystals)
    if len(chain) < 2 or any(
        crystal.operator_kind != CAUSAL_MIX_FLOAT64 for crystal in chain
    ):
        raise ComputeCrystalFusionError(
            "causal-mix fusion needs two or more causal kernels"
        )
    (first,) = _operator_arrays(chain[0])
    kernel = np.array(first, dtype=np.float64, copy=True)
    for crystal in chain[1:]:
        (following,) = _operator_arrays(crystal)
        kernel = np.einsum("qj,jk->qk", following, kernel, optimize=False)
        for row in range(kernel.shape[0]):
            kernel[row, row + 1 :] = 0.0
    kernel[kernel == 0.0] = 0.0
    payload = {"kernel": _array_document(kernel, dtype=_FLOAT64)}
    result = ComputeCrystal._from_operator(
        operator_kind=CAUSAL_MIX_FLOAT64,
        payload=payload,
        parents=chain,
        extensions=extensions,
    )
    _validate_fusion_work_provenance(result, chain)
    return result


def fuse_compatible_chain(
    crystals: Sequence[ComputeCrystal],
    *,
    extensions: Mapping[str, object] | None = None,
) -> ComputeCrystal:
    chain = _validate_chain(crystals)
    kinds = {crystal.operator_kind for crystal in chain}
    if kinds == {AFFINE_FLOAT64}:
        return fuse_affine_chain(chain, extensions=extensions)
    if kinds == {PERMUTATION}:
        return fuse_permutation_chain(chain, extensions=extensions)
    if kinds == {MARKOV_FLOAT64}:
        return fuse_markov_chain(chain, extensions=extensions)
    if kinds == {CAUSAL_MIX_FLOAT64}:
        return fuse_causal_mix_chain(chain, extensions=extensions)
    raise ComputeCrystalFusionError(
        "only homogeneous affine, permutation, Markov, or causal-mix chains are fusible"
    )


def fuse_compatible_chain_with_provenance(
    crystals: Sequence[ComputeCrystal],
    *,
    extensions: Mapping[str, object] | None = None,
) -> ComputeCrystal:
    """Fuse a chain and bind its complete transitive work representation.

    This opt-in path preserves byte compatibility for existing crystals.  New
    recursive compilers use it so a compact DAG does not lose the amount of
    primitive work represented by already-fused child nodes.
    """

    chain = _validate_chain(crystals)
    document = {} if extensions is None else dict(extensions)
    if FUSION_WORK_PROVENANCE_EXTENSION in document:
        raise ValueError(
            f"{FUSION_WORK_PROVENANCE_EXTENSION!r} is a reserved extension"
        )
    preliminary = fuse_compatible_chain(chain, extensions=document)
    document[FUSION_WORK_PROVENANCE_EXTENSION] = _fusion_work_record(
        chain,
        live_work_units=preliminary.discharge_work_units,
    )
    return fuse_compatible_chain(chain, extensions=document)


@dataclass(frozen=True, slots=True)
class ComputeChargeReceipt:
    """Authenticated accounting basis for discharging one fused crystal.

    The receipt never claims that its source program is a minimal circuit.  It
    names the exact manifest-published program whose live work is replaced by
    the fused crystal, plus the verifier and verification-receipt identities
    that authorized that charge basis.
    """

    source_program_sha256: str
    source_crystal_sha256s: tuple[str, ...]
    fused_crystal_sha256: str
    source_work_units: int
    live_work_units: int
    charge_verifier_sha256: str
    verification_receipt_sha256: str

    FORMAT = COMPUTE_CHARGE_SCHEMA

    def __post_init__(self) -> None:
        for field_name in (
            "source_program_sha256",
            "fused_crystal_sha256",
            "charge_verifier_sha256",
            "verification_receipt_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )
        if not isinstance(self.source_crystal_sha256s, tuple):
            raise TypeError("source_crystal_sha256s must be an immutable tuple")
        if not 2 <= len(self.source_crystal_sha256s) <= MAX_PROGRAM_STEPS:
            raise ValueError("a charge basis needs a bounded multi-crystal source")
        object.__setattr__(
            self,
            "source_crystal_sha256s",
            tuple(
                require_sha256(digest, field="source_crystal_sha256s")
                for digest in self.source_crystal_sha256s
            ),
        )
        object.__setattr__(
            self,
            "source_work_units",
            _bounded_uint(
                self.source_work_units,
                field="source_work_units",
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "live_work_units",
            _bounded_uint(
                self.live_work_units,
                field="live_work_units",
                positive=True,
            ),
        )

    @classmethod
    def create(
        cls,
        *,
        source_program: ComputeProgram,
        source_crystals: Sequence[ComputeCrystal],
        fused_crystal: ComputeCrystal,
        charge_verifier_sha256: str,
        verification_receipt_sha256: str,
    ) -> "ComputeChargeReceipt":
        if not isinstance(source_program, ComputeProgram):
            raise TypeError("source_program must be a ComputeProgram")
        if not isinstance(fused_crystal, ComputeCrystal):
            raise TypeError("fused_crystal must be a ComputeCrystal")
        chain = source_program.validate_crystals(source_crystals)
        if fused_crystal.parent_sha256s != source_program.crystal_sha256s:
            raise ComputeCrystalIntegrityError(
                "fused crystal parents do not equal the charged source program"
            )
        if (
            fused_crystal.input_abi != source_program.input_abi
            or fused_crystal.output_abi != source_program.output_abi
        ):
            raise ComputeCrystalIntegrityError(
                "fused crystal ABI does not equal the charged source program"
            )
        expected = fuse_compatible_chain(
            chain,
            extensions=fused_crystal.extensions,
        )
        if expected.to_bytes() != fused_crystal.to_bytes():
            raise ComputeCrystalIntegrityError(
                "fused crystal is not the exact charged source composition"
            )
        source_work = 0
        for crystal in chain:
            source_work = _checked_add(
                source_work,
                _provenance_source_work_units(crystal),
                field="charged source work",
            )
        return cls(
            source_program_sha256=source_program.sha256,
            source_crystal_sha256s=source_program.crystal_sha256s,
            fused_crystal_sha256=fused_crystal.sha256,
            source_work_units=source_work,
            live_work_units=fused_crystal.discharge_work_units,
            charge_verifier_sha256=charge_verifier_sha256,
            verification_receipt_sha256=verification_receipt_sha256,
        )

    @property
    def historical_work_released(self) -> int:
        return max(0, self.source_work_units - self.live_work_units)

    def validate_basis(
        self,
        *,
        source_program: ComputeProgram,
        source_crystals: Sequence[ComputeCrystal],
        fused_crystal: ComputeCrystal,
    ) -> None:
        expected = self.create(
            source_program=source_program,
            source_crystals=source_crystals,
            fused_crystal=fused_crystal,
            charge_verifier_sha256=self.charge_verifier_sha256,
            verification_receipt_sha256=self.verification_receipt_sha256,
        )
        if expected.to_bytes() != self.to_bytes():
            raise ComputeCrystalIntegrityError(
                "compute charge disagrees with its published source basis"
            )

    def as_record(self) -> dict[str, object]:
        return {
            "charge_verifier_sha256": self.charge_verifier_sha256,
            "fused_crystal_sha256": self.fused_crystal_sha256,
            "live_work_units": self.live_work_units,
            "source_crystal_sha256s": list(self.source_crystal_sha256s),
            "source_program_sha256": self.source_program_sha256,
            "source_work_units": self.source_work_units,
            "verification_receipt_sha256": self.verification_receipt_sha256,
        }

    def to_document(self) -> dict[str, object]:
        body = self.as_record()
        return {
            "body": body,
            "body_sha256": _digest_json(body),
            "schema": self.FORMAT,
        }

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_document())
        if len(data) > MAX_CHARGE_BYTES:
            raise ValueError("compute charge exceeds its hard byte limit")
        return data

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "ComputeChargeReceipt":
        value = _strict_json(data, label="compute charge", maximum=MAX_CHARGE_BYTES)
        if (
            not isinstance(value, Mapping)
            or set(value) != {"body", "body_sha256", "schema"}
            or value.get("schema") != cls.FORMAT
        ):
            raise ComputeCrystalIntegrityError("compute-charge envelope is invalid")
        body = value.get("body")
        expected_fields = {
            "charge_verifier_sha256",
            "fused_crystal_sha256",
            "live_work_units",
            "source_crystal_sha256s",
            "source_program_sha256",
            "source_work_units",
            "verification_receipt_sha256",
        }
        if not isinstance(body, Mapping) or set(body) != expected_fields:
            raise ComputeCrystalIntegrityError("compute-charge body is invalid")
        try:
            claimed = require_sha256(value.get("body_sha256"), field="body_sha256")
        except ValueError as exc:
            raise ComputeCrystalIntegrityError(
                "compute-charge body SHA-256 is invalid"
            ) from exc
        if claimed != _digest_json(body):
            raise ComputeCrystalIntegrityError("compute-charge body SHA-256 mismatch")
        source_crystals = body.get("source_crystal_sha256s")
        if not isinstance(source_crystals, list):
            raise ComputeCrystalIntegrityError(
                "compute-charge source crystal chain must be a list"
            )
        try:
            receipt = cls(
                source_program_sha256=cast(str, body.get("source_program_sha256")),
                source_crystal_sha256s=tuple(source_crystals),
                fused_crystal_sha256=cast(str, body.get("fused_crystal_sha256")),
                source_work_units=cast(int, body.get("source_work_units")),
                live_work_units=cast(int, body.get("live_work_units")),
                charge_verifier_sha256=cast(str, body.get("charge_verifier_sha256")),
                verification_receipt_sha256=cast(
                    str, body.get("verification_receipt_sha256")
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ComputeCrystalIntegrityError(
                "compute-charge validation failed"
            ) from exc
        if receipt.to_bytes() != data:
            raise ComputeCrystalIntegrityError(
                "compute charge failed canonical reconstruction"
            )
        return receipt


def tensor_sha256(value: object, abi: NumericalABI) -> str:
    array = abi.validate(value)
    if abi.dtype == _FLOAT64:
        canonical = np.asarray(array, dtype="<f8", order="C").tobytes(order="C")
    else:
        canonical = np.asarray(array, dtype="<i8", order="C").tobytes(order="C")
    return _digest_json(
        {
            "data_sha256": hashlib.sha256(canonical).hexdigest(),
            "dtype": abi.dtype,
            "shape": list(array.shape),
        }
    )


@dataclass(frozen=True, slots=True)
class ComputeExecutionReceipt:
    program_sha256: str
    input_sha256: str
    output_sha256: str
    charge_basis_sha256: str | None
    executed_operator_count: int
    equivalent_unfused_source_work: int
    live_discharge_work: int
    historical_work_released: int

    FORMAT = COMPUTE_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for field_name in ("program_sha256", "input_sha256", "output_sha256"):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )
        charge_basis = self.charge_basis_sha256
        if charge_basis is not None:
            charge_basis = require_sha256(charge_basis, field="charge_basis_sha256")
        count = _bounded_uint(
            self.executed_operator_count,
            field="executed_operator_count",
            positive=True,
        )
        source = _bounded_uint(
            self.equivalent_unfused_source_work,
            field="equivalent_unfused_source_work",
        )
        live = _bounded_uint(
            self.live_discharge_work,
            field="live_discharge_work",
        )
        released = _bounded_uint(
            self.historical_work_released,
            field="historical_work_released",
        )
        if released != max(0, source - live):
            raise ValueError("historical work released must equal max(source-live, 0)")
        if charge_basis is None and (source != live or released != 0):
            raise ValueError(
                "unbased compute receipts cannot claim historical work release"
            )
        object.__setattr__(self, "charge_basis_sha256", charge_basis)
        object.__setattr__(self, "executed_operator_count", count)
        object.__setattr__(self, "equivalent_unfused_source_work", source)
        object.__setattr__(self, "live_discharge_work", live)
        object.__setattr__(self, "historical_work_released", released)

    def as_record(self) -> dict[str, object]:
        return {
            "charge_basis_sha256": self.charge_basis_sha256,
            "equivalent_unfused_source_work": self.equivalent_unfused_source_work,
            "executed_operator_count": self.executed_operator_count,
            "historical_work_released": self.historical_work_released,
            "input_sha256": self.input_sha256,
            "live_discharge_work": self.live_discharge_work,
            "output_sha256": self.output_sha256,
            "program_sha256": self.program_sha256,
        }

    def to_document(self) -> dict[str, object]:
        body = self.as_record()
        return {
            "body": body,
            "body_sha256": _digest_json(body),
            "schema": self.FORMAT,
        }

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_document())
        if len(data) > MAX_RECEIPT_BYTES:
            raise ValueError("compute receipt exceeds its hard byte limit")
        return data

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "ComputeExecutionReceipt":
        value = _strict_json(data, label="compute receipt", maximum=MAX_RECEIPT_BYTES)
        if (
            not isinstance(value, Mapping)
            or set(value) != {"body", "body_sha256", "schema"}
            or value.get("schema") != cls.FORMAT
        ):
            raise ComputeCrystalIntegrityError("compute receipt envelope is invalid")
        body = value.get("body")
        expected = {
            "charge_basis_sha256",
            "equivalent_unfused_source_work",
            "executed_operator_count",
            "historical_work_released",
            "input_sha256",
            "live_discharge_work",
            "output_sha256",
            "program_sha256",
        }
        if not isinstance(body, Mapping) or set(body) != expected:
            raise ComputeCrystalIntegrityError("compute receipt body is invalid")
        try:
            claimed = require_sha256(value.get("body_sha256"), field="body_sha256")
        except ValueError as exc:
            raise ComputeCrystalIntegrityError(
                "compute receipt body SHA-256 is invalid"
            ) from exc
        if claimed != _digest_json(body):
            raise ComputeCrystalIntegrityError("compute receipt body SHA-256 mismatch")
        try:
            receipt = cls(**dict(body))
        except (TypeError, ValueError) as exc:
            raise ComputeCrystalIntegrityError(
                "compute receipt validation failed"
            ) from exc
        if receipt.to_bytes() != data:
            raise ComputeCrystalIntegrityError(
                "compute receipt failed canonical reconstruction"
            )
        return receipt


@dataclass(frozen=True, slots=True)
class ComputeExecution:
    output: NDArray[Any]
    receipt: ComputeExecutionReceipt


@dataclass(frozen=True, slots=True)
class ComputeBankManifest:
    generation: int
    crystal_sha256s: tuple[str, ...]
    program_sha256s: tuple[str, ...]
    charge_sha256s: tuple[str, ...]
    previous_manifest_sha256: str | None

    FORMAT = COMPUTE_BANK_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        generation = _bounded_uint(self.generation, field="manifest generation")
        crystals = tuple(
            sorted(
                {
                    require_sha256(digest, field="manifest crystal digest")
                    for digest in self.crystal_sha256s
                }
            )
        )
        programs = tuple(
            sorted(
                {
                    require_sha256(digest, field="manifest program digest")
                    for digest in self.program_sha256s
                }
            )
        )
        charges = tuple(
            sorted(
                {
                    require_sha256(digest, field="manifest charge digest")
                    for digest in self.charge_sha256s
                }
            )
        )
        if (
            crystals != self.crystal_sha256s
            or programs != self.program_sha256s
            or charges != self.charge_sha256s
        ):
            raise ValueError("manifest content addresses must be sorted and unique")
        previous = self.previous_manifest_sha256
        if previous is not None:
            previous = require_sha256(previous, field="previous_manifest_sha256")
        if generation == 0 and (
            crystals or programs or charges or previous is not None
        ):
            raise ValueError("empty manifest generation must contain no history")
        if generation > 0 and previous is None:
            raise ValueError("non-empty manifest generation must bind its predecessor")
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "previous_manifest_sha256", previous)

    @classmethod
    def empty(cls) -> "ComputeBankManifest":
        return cls(0, (), (), (), None)

    def as_record(self) -> dict[str, object]:
        return {
            "charges": list(self.charge_sha256s),
            "crystals": list(self.crystal_sha256s),
            "generation": self.generation,
            "previous_manifest_sha256": self.previous_manifest_sha256,
            "programs": list(self.program_sha256s),
        }

    def to_document(self) -> dict[str, object]:
        body = self.as_record()
        return {
            "body": body,
            "body_sha256": _digest_json(body),
            "schema": self.FORMAT,
        }

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_document())
        if len(data) > MAX_MANIFEST_BYTES:
            raise ValueError("compute-bank manifest exceeds its hard byte limit")
        return data

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "ComputeBankManifest":
        value = _strict_json(
            data,
            label="compute-bank manifest",
            maximum=MAX_MANIFEST_BYTES,
        )
        if (
            not isinstance(value, Mapping)
            or set(value) != {"body", "body_sha256", "schema"}
            or value.get("schema") != cls.FORMAT
        ):
            raise ComputeCrystalIntegrityError("compute-bank manifest is invalid")
        body = value.get("body")
        if not isinstance(body, Mapping) or set(body) != {
            "charges",
            "crystals",
            "generation",
            "previous_manifest_sha256",
            "programs",
        }:
            raise ComputeCrystalIntegrityError("compute-bank manifest body is invalid")
        try:
            claimed = require_sha256(value.get("body_sha256"), field="body_sha256")
        except ValueError as exc:
            raise ComputeCrystalIntegrityError(
                "compute-bank manifest body SHA-256 is invalid"
            ) from exc
        if claimed != _digest_json(body):
            raise ComputeCrystalIntegrityError("compute-bank manifest SHA-256 mismatch")
        crystals = body.get("crystals")
        programs = body.get("programs")
        charges = body.get("charges")
        if (
            not isinstance(crystals, list)
            or not isinstance(programs, list)
            or not isinstance(charges, list)
        ):
            raise ComputeCrystalIntegrityError("manifest inventories must be lists")
        try:
            manifest = cls(
                generation=cast(int, body.get("generation")),
                crystal_sha256s=tuple(crystals),
                program_sha256s=tuple(programs),
                charge_sha256s=tuple(charges),
                previous_manifest_sha256=cast(
                    str | None, body.get("previous_manifest_sha256")
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ComputeCrystalIntegrityError(
                "compute-bank manifest validation failed"
            ) from exc
        if manifest.to_bytes() != data:
            raise ComputeCrystalIntegrityError(
                "compute-bank manifest failed canonical reconstruction"
            )
        return manifest


def _manifest_commit_bytes(manifest_sha256: str) -> bytes:
    address = require_sha256(manifest_sha256, field="manifest_sha256")
    body = {"manifest_sha256": address}
    return canonical_json_bytes(
        {
            "body": body,
            "body_sha256": _digest_json(body),
            "schema": COMPUTE_BANK_MANIFEST_COMMIT_SCHEMA,
        }
    )


def _decode_manifest_commit(data: bytes) -> str:
    value = _strict_json(
        data,
        label="compute-bank manifest commit",
        maximum=4096,
    )
    if (
        not isinstance(value, Mapping)
        or set(value) != {"body", "body_sha256", "schema"}
        or value.get("schema") != COMPUTE_BANK_MANIFEST_COMMIT_SCHEMA
    ):
        raise ComputeCrystalIntegrityError("compute-bank commit marker is invalid")
    body = value.get("body")
    if not isinstance(body, Mapping) or set(body) != {"manifest_sha256"}:
        raise ComputeCrystalIntegrityError("compute-bank commit-marker body is invalid")
    try:
        claimed = require_sha256(value.get("body_sha256"), field="body_sha256")
        manifest_sha256 = require_sha256(
            body.get("manifest_sha256"), field="manifest_sha256"
        )
    except ValueError as exc:
        raise ComputeCrystalIntegrityError(
            "compute-bank commit-marker digest is invalid"
        ) from exc
    if claimed != _digest_json(body):
        raise ComputeCrystalIntegrityError(
            "compute-bank commit-marker SHA-256 mismatch"
        )
    if _manifest_commit_bytes(manifest_sha256) != data:
        raise ComputeCrystalIntegrityError(
            "compute-bank commit marker failed canonical reconstruction"
        )
    return manifest_sha256


def _validate_manifest_extension(
    previous: ComputeBankManifest, current: ComputeBankManifest
) -> None:
    if current.generation != previous.generation + 1:
        raise ComputeCrystalIntegrityError(
            "compute-bank history generation is not monotonic"
        )
    if current.previous_manifest_sha256 != previous.sha256:
        raise ComputeCrystalIntegrityError(
            "compute-bank history predecessor hash is invalid"
        )
    previous_crystals = set(previous.crystal_sha256s)
    current_crystals = set(current.crystal_sha256s)
    previous_programs = set(previous.program_sha256s)
    current_programs = set(current.program_sha256s)
    previous_charges = set(previous.charge_sha256s)
    current_charges = set(current.charge_sha256s)
    if (
        not previous_crystals <= current_crystals
        or not previous_programs <= current_programs
        or not previous_charges <= current_charges
    ):
        raise ComputeCrystalIntegrityError(
            "compute-bank history inventories are not append-only"
        )
    previous_size = (
        len(previous_crystals) + len(previous_programs) + len(previous_charges)
    )
    current_size = len(current_crystals) + len(current_programs) + len(current_charges)
    if current_size != previous_size + 1 or current.generation != current_size:
        raise ComputeCrystalIntegrityError(
            "compute-bank history must append exactly one content address"
        )


def _validate_committed_manifest_history(
    histories: Mapping[str, ComputeBankManifest], commits: set[str]
) -> ComputeBankManifest:
    if (
        len(histories) > MAX_MANIFEST_HISTORY_STATES
        or len(commits) > MAX_MANIFEST_HISTORY_STATES
    ):
        raise ComputeCrystalIntegrityError(
            "compute-bank manifest history exceeds its hard state bound"
        )
    by_generation: dict[int, tuple[str, ComputeBankManifest]] = {}
    for digest in commits:
        manifest = histories.get(digest)
        if manifest is None:
            raise ComputeCrystalIntegrityError(
                "committed manifest is missing its immutable history object"
            )
        if manifest.sha256 != digest:
            raise ComputeCrystalIntegrityError(
                "manifest history address does not match its bytes"
            )
        if manifest.generation < 1:
            raise ComputeCrystalIntegrityError(
                "the deterministic empty manifest must not be committed"
            )
        existing = by_generation.get(manifest.generation)
        if existing is not None and existing[0] != digest:
            raise ComputeCrystalIntegrityError(
                "compute-bank manifest history contains a committed fork"
            )
        by_generation[manifest.generation] = (digest, manifest)

    previous = ComputeBankManifest.empty()
    if not by_generation:
        return previous
    maximum = max(by_generation)
    if set(by_generation) != set(range(1, maximum + 1)):
        raise ComputeCrystalIntegrityError(
            "compute-bank committed manifest history contains a generation gap"
        )
    for generation in range(1, maximum + 1):
        _digest, current = by_generation[generation]
        _validate_manifest_extension(previous, current)
        previous = current
    return previous


@dataclass(frozen=True, slots=True)
class ComputeBankPublication:
    artifact_kind: Literal["charge", "crystal", "program"]
    payload_sha256: str
    generation: int
    manifest_sha256: str
    current_anchor_sha256: str
    object_created: bool
    manifest_changed: bool


class ComputeCrystalBank:
    """Atomic append-only storage for crystals and programs over CrystalStore.

    The internal history detects partial rollback, forks, gaps, and inventory
    replacement.  No self-contained store can distinguish a complete,
    internally consistent replacement from the original store.  Persist the
    ``current_anchor_sha256`` returned by each publication in an external trust
    domain and supply it as ``trusted_manifest_sha256`` (or through
    ``trusted_head_resolver``) when reopening a bank to detect full replacement.
    """

    def __init__(
        self,
        store: CrystalStore | str | os.PathLike[str],
        *,
        manifest_retry_limit: int = 8,
        trusted_manifest_sha256: str | None = None,
        trusted_head_resolver: Callable[[], str] | None = None,
    ) -> None:
        if (
            isinstance(manifest_retry_limit, bool)
            or not isinstance(manifest_retry_limit, int)
            or not 1 <= manifest_retry_limit <= 1024
        ):
            raise ValueError("manifest_retry_limit must lie in [1, 1024]")
        self.store = store if isinstance(store, CrystalStore) else CrystalStore(store)
        self.root = Path(self.store.root)
        self.manifest_retry_limit = manifest_retry_limit
        self.trusted_manifest_sha256 = (
            None
            if trusted_manifest_sha256 is None
            else require_sha256(
                trusted_manifest_sha256,
                field="trusted_manifest_sha256",
            )
        )
        if trusted_head_resolver is not None and not callable(trusted_head_resolver):
            raise TypeError("trusted_head_resolver must be callable")
        self.trusted_head_resolver = trusted_head_resolver

    def _trusted_anchors(self) -> tuple[str, ...]:
        anchors: list[str] = []
        if self.trusted_manifest_sha256 is not None:
            anchors.append(self.trusted_manifest_sha256)
        if self.trusted_head_resolver is not None:
            try:
                resolved = self.trusted_head_resolver()
            except Exception as exc:
                raise ComputeCrystalIntegrityError(
                    "trusted manifest-head resolver failed"
                ) from exc
            try:
                anchors.append(
                    require_sha256(resolved, field="trusted resolved manifest head")
                )
            except ValueError as exc:
                raise ComputeCrystalIntegrityError(
                    "trusted manifest-head resolver returned an invalid anchor"
                ) from exc
        return tuple(dict.fromkeys(anchors))

    def _assert_trusted_anchors(
        self,
        head: ComputeBankManifest,
        histories: Mapping[str, ComputeBankManifest],
        commits: set[str],
    ) -> None:
        empty_sha256 = ComputeBankManifest.empty().sha256
        for anchor in self._trusted_anchors():
            if anchor == empty_sha256:
                continue
            ancestor = histories.get(anchor)
            if anchor not in commits or ancestor is None:
                raise ComputeCrystalIntegrityError(
                    "committed manifest head does not descend from the trusted anchor"
                )
            if ancestor.generation > head.generation:
                raise ComputeCrystalIntegrityError(
                    "committed manifest head predates the trusted anchor"
                )

    @contextmanager
    def _locked(self) -> Iterator[None]:
        path = self.root / _BANK_LOCK_NAME
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as exc:
            raise ComputeCrystalIntegrityError(
                "cannot open compute-crystal bank lock"
            ) from exc
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise ComputeCrystalIntegrityError(
                    "compute-crystal bank lock is not a regular file"
                )
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            linked = path.lstat()
            if (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino):
                raise ComputeCrystalIntegrityError(
                    "compute-crystal bank lock changed while acquiring it"
                )
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    @staticmethod
    def crystal_state_name(payload_sha256: str) -> str:
        return _CRYSTAL_OBJECT_PREFIX + require_sha256(
            payload_sha256, field="payload_sha256"
        )

    @staticmethod
    def program_state_name(payload_sha256: str) -> str:
        return _PROGRAM_OBJECT_PREFIX + require_sha256(
            payload_sha256, field="payload_sha256"
        )

    @staticmethod
    def charge_state_name(payload_sha256: str) -> str:
        return _CHARGE_OBJECT_PREFIX + require_sha256(
            payload_sha256, field="payload_sha256"
        )

    @staticmethod
    def manifest_history_state_name(manifest_sha256: str) -> str:
        return _MANIFEST_HISTORY_PREFIX + require_sha256(
            manifest_sha256, field="manifest_sha256"
        )

    @staticmethod
    def manifest_commit_state_name(manifest_sha256: str) -> str:
        return _MANIFEST_COMMIT_PREFIX + require_sha256(
            manifest_sha256, field="manifest_sha256"
        )

    def _restore_state(self, name: str) -> bytes:
        try:
            return self.store.restore_state(name)
        except KeyError:
            raise
        except CrystalStoreError as exc:
            raise ComputeCrystalIntegrityError(
                "compute-crystal state failed integrity"
            ) from exc

    def _bank_history_state_names_unlocked(self) -> tuple[str, ...]:
        names: list[str] = []
        root_fd = os.open(self.root, self.store._directory_flags())
        try:
            state_fd = os.open("state", self.store._directory_flags(), dir_fd=root_fd)
            try:
                for filename in sorted(os.listdir(state_fd)):
                    if not filename.endswith(".state"):
                        continue
                    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                    try:
                        descriptor = os.open(filename, flags, dir_fd=state_fd)
                    except OSError:
                        continue
                    try:
                        before = os.fstat(descriptor)
                        if not stat.S_ISREG(before.st_mode):
                            continue
                        prefix = os.read(descriptor, 4096)
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
                            raise ComputeCrystalIntegrityError(
                                "compute-bank state changed during history scan"
                            )
                    finally:
                        os.close(descriptor)
                    match = _STATE_NAME_RE.match(prefix)
                    if match is None:
                        continue
                    try:
                        name = match.group(1).decode("ascii")
                    except UnicodeDecodeError:
                        continue
                    if not name.startswith(
                        (_MANIFEST_HISTORY_PREFIX, _MANIFEST_COMMIT_PREFIX)
                    ):
                        continue
                    expected_filename = self.store._state_filename(name)
                    if filename != expected_filename:
                        raise ComputeCrystalIntegrityError(
                            "compute-bank history state filename is not name-bound"
                        )
                    names.append(name)
            finally:
                os.close(state_fd)
        finally:
            os.close(root_fd)
        if len(names) > 2 * MAX_MANIFEST_HISTORY_STATES:
            raise ComputeCrystalIntegrityError(
                "compute-bank history scan exceeds its hard state bound"
            )
        return tuple(names)

    def _history_inventory_unlocked(
        self,
    ) -> tuple[dict[str, ComputeBankManifest], set[str]]:
        histories: dict[str, ComputeBankManifest] = {}
        commits: set[str] = set()
        for name in self._bank_history_state_names_unlocked():
            payload = self._restore_state(name)
            if name.startswith(_MANIFEST_HISTORY_PREFIX):
                suffix = name[len(_MANIFEST_HISTORY_PREFIX) :]
                try:
                    address = require_sha256(suffix, field="manifest history address")
                except ValueError as exc:
                    raise ComputeCrystalIntegrityError(
                        "compute-bank history state name is invalid"
                    ) from exc
                manifest = ComputeBankManifest.from_bytes(payload)
                if manifest.sha256 != address:
                    raise ComputeCrystalIntegrityError(
                        "compute-bank history state is stored under another address"
                    )
                histories[address] = manifest
            else:
                suffix = name[len(_MANIFEST_COMMIT_PREFIX) :]
                try:
                    address = require_sha256(suffix, field="manifest commit address")
                except ValueError as exc:
                    raise ComputeCrystalIntegrityError(
                        "compute-bank commit state name is invalid"
                    ) from exc
                if _decode_manifest_commit(payload) != address:
                    raise ComputeCrystalIntegrityError(
                        "compute-bank commit marker is stored under another address"
                    )
                commits.add(address)
        return histories, commits

    def _manifest_unlocked(self) -> ComputeBankManifest:
        try:
            data = self._restore_state(_MANIFEST_STATE_NAME)
        except KeyError:
            head = None
        else:
            head = ComputeBankManifest.from_bytes(data)

        histories, commits = self._history_inventory_unlocked()
        latest = _validate_committed_manifest_history(histories, commits)
        if head is None:
            if latest.generation != 0:
                raise ComputeCrystalIntegrityError(
                    "compute-bank manifest head was deleted or rolled back"
                )
            empty = ComputeBankManifest.empty()
            self._assert_trusted_anchors(empty, histories, commits)
            return empty
        if head.generation == 0:
            raise ComputeCrystalIntegrityError(
                "the deterministic empty manifest must not be persisted as a head"
            )
        historical = histories.get(head.sha256)
        if historical is None or historical.to_bytes() != head.to_bytes():
            raise ComputeCrystalIntegrityError(
                "compute-bank head lacks its exact immutable history object"
            )
        if head.sha256 not in commits:
            _validate_manifest_extension(latest, head)
            marker = _manifest_commit_bytes(head.sha256)
            self._publish_content_unlocked(
                name=self.manifest_commit_state_name(head.sha256),
                data=marker,
                digest=hashlib.sha256(marker).hexdigest(),
            )
            commits.add(head.sha256)
            latest = _validate_committed_manifest_history(histories, commits)
        if latest.sha256 != head.sha256 or latest.generation != head.generation:
            raise ComputeCrystalIntegrityError(
                "compute-bank manifest head is a validly resealed rollback"
            )
        self._assert_trusted_anchors(head, histories, commits)
        return head

    def manifest(self) -> ComputeBankManifest:
        with self._locked():
            return self._manifest_unlocked()

    def current_anchor_sha256(self) -> str:
        """Return the head digest that callers persist outside this store."""

        return self.manifest().sha256

    def assert_descends_from(self, manifest_sha256: str) -> str:
        """Verify one external manifest anchor against the current history."""

        anchor = require_sha256(manifest_sha256, field="manifest_sha256")
        with self._locked():
            head = self._manifest_unlocked()
            if anchor == ComputeBankManifest.empty().sha256:
                return head.sha256
            histories, commits = self._history_inventory_unlocked()
            ancestor = histories.get(anchor)
            if (
                ancestor is None
                or anchor not in commits
                or ancestor.generation > head.generation
            ):
                raise ComputeCrystalIntegrityError(
                    "compute bank does not descend from the supplied manifest anchor"
                )
            return head.sha256

    @staticmethod
    def _check_expected_generation(
        manifest: ComputeBankManifest, expected_generation: int | None
    ) -> None:
        if expected_generation is None:
            return
        expected = _bounded_uint(expected_generation, field="expected_generation")
        if manifest.generation != expected:
            raise ComputeCrystalConflictError(
                f"compute-bank generation is {manifest.generation}, expected {expected}"
            )

    def _publish_content_unlocked(self, *, name: str, data: bytes, digest: str) -> bool:
        try:
            existing = self._restore_state(name)
        except KeyError:
            existing = None
        if existing is not None:
            if existing != data or hashlib.sha256(existing).hexdigest() != digest:
                raise ComputeCrystalIntegrityError(
                    "content-addressed compute object contains different bytes"
                )
            return False
        try:
            publication = self.store.publish_state(name, data)
        except ManifestConflictError as exc:
            raise ComputeCrystalConflictError(
                "content-addressed object publication conflicted"
            ) from exc
        except CrystalStoreError as exc:
            raise ComputeCrystalIntegrityError(
                "content-addressed object publication failed integrity"
            ) from exc
        restored = self._restore_state(name)
        if restored != data or hashlib.sha256(restored).hexdigest() != digest:
            raise ComputeCrystalIntegrityError(
                "published compute object failed immediate verification"
            )
        return publication.changed

    def _commit_manifest_unlocked(
        self, current: ComputeBankManifest, updated: ComputeBankManifest
    ) -> None:
        history = updated.to_bytes()
        self._publish_content_unlocked(
            name=self.manifest_history_state_name(updated.sha256),
            data=history,
            digest=updated.sha256,
        )
        expected = None if current.generation == 0 else current.sha256
        try:
            self.store.publish_state(
                _MANIFEST_STATE_NAME,
                history,
                expected_sha256=expected,
            )
        except ManifestConflictError as exc:
            raise ComputeCrystalConflictError(
                "compute-bank manifest compare-and-swap conflicted"
            ) from exc
        except CrystalStoreError as exc:
            raise ComputeCrystalIntegrityError(
                "compute-bank manifest publication failed integrity"
            ) from exc
        marker = _manifest_commit_bytes(updated.sha256)
        self._publish_content_unlocked(
            name=self.manifest_commit_state_name(updated.sha256),
            data=marker,
            digest=hashlib.sha256(marker).hexdigest(),
        )

    def _restore_crystal_object_unlocked(self, digest: str) -> ComputeCrystal:
        address = require_sha256(digest, field="crystal_sha256")
        try:
            data = self._restore_state(self.crystal_state_name(address))
        except KeyError as exc:
            raise ComputeCrystalMissError(
                f"missing compute crystal: {address}"
            ) from exc
        if hashlib.sha256(data).hexdigest() != address:
            raise ComputeCrystalIntegrityError(
                "compute-crystal content address does not match its bytes"
            )
        crystal = ComputeCrystal.from_bytes(data)
        if crystal.sha256 != address:
            raise ComputeCrystalIntegrityError(
                "compute crystal failed semantic content rehash"
            )
        return crystal

    def _restore_crystal_recursive_unlocked(
        self,
        digest: str,
        manifest: ComputeBankManifest,
        *,
        memo: dict[str, ComputeCrystal],
        visiting: set[str],
    ) -> ComputeCrystal:
        address = require_sha256(digest, field="crystal_sha256")
        if address not in manifest.crystal_sha256s:
            raise ComputeCrystalMissError(
                f"compute crystal is not published by the manifest: {address}"
            )
        if address in memo:
            return memo[address]
        if address in visiting:
            raise ComputeCrystalIntegrityError(
                "compute-crystal lineage contains a cycle"
            )
        visiting.add(address)
        try:
            crystal = self._restore_crystal_object_unlocked(address)
            if crystal.parent_sha256s:
                parents = tuple(
                    self._restore_crystal_recursive_unlocked(
                        parent,
                        manifest,
                        memo=memo,
                        visiting=visiting,
                    )
                    for parent in crystal.parent_sha256s
                )
                try:
                    expected = fuse_compatible_chain(
                        parents,
                        extensions=crystal.extensions,
                    )
                except ComputeCrystalError as exc:
                    raise ComputeCrystalIntegrityError(
                        "compute-crystal lineage is not a valid exact fusion"
                    ) from exc
                if expected.to_bytes() != crystal.to_bytes():
                    raise ComputeCrystalIntegrityError(
                        "fused crystal disagrees with its authenticated parent chain"
                    )
            memo[address] = crystal
            return crystal
        finally:
            visiting.remove(address)

    def _publish_crystal_once(
        self,
        crystal: ComputeCrystal,
        *,
        expected_generation: int | None = None,
    ) -> ComputeBankPublication:
        if not isinstance(crystal, ComputeCrystal):
            raise TypeError("crystal must be a ComputeCrystal")
        data = crystal.to_bytes()
        digest = crystal.sha256
        with self._locked():
            current = self._manifest_unlocked()
            self._check_expected_generation(current, expected_generation)
            if crystal.parent_sha256s:
                memo: dict[str, ComputeCrystal] = {}
                parents = tuple(
                    self._restore_crystal_recursive_unlocked(
                        parent,
                        current,
                        memo=memo,
                        visiting=set(),
                    )
                    for parent in crystal.parent_sha256s
                )
                expected = fuse_compatible_chain(
                    parents,
                    extensions=crystal.extensions,
                )
                if expected.to_bytes() != data:
                    raise ComputeCrystalIntegrityError(
                        "fused crystal does not match its published parents"
                    )
            object_created = self._publish_content_unlocked(
                name=self.crystal_state_name(digest),
                data=data,
                digest=digest,
            )
            if digest in current.crystal_sha256s:
                return ComputeBankPublication(
                    artifact_kind="crystal",
                    payload_sha256=digest,
                    generation=current.generation,
                    manifest_sha256=current.sha256,
                    current_anchor_sha256=current.sha256,
                    object_created=object_created,
                    manifest_changed=False,
                )
            updated = ComputeBankManifest(
                generation=current.generation + 1,
                crystal_sha256s=tuple(sorted((*current.crystal_sha256s, digest))),
                program_sha256s=current.program_sha256s,
                charge_sha256s=current.charge_sha256s,
                previous_manifest_sha256=current.sha256,
            )
            self._commit_manifest_unlocked(current, updated)
            return ComputeBankPublication(
                artifact_kind="crystal",
                payload_sha256=digest,
                generation=updated.generation,
                manifest_sha256=updated.sha256,
                current_anchor_sha256=updated.sha256,
                object_created=object_created,
                manifest_changed=True,
            )

    def publish_crystal(
        self,
        crystal: ComputeCrystal,
        *,
        expected_generation: int | None = None,
    ) -> ComputeBankPublication:
        """Publish one crystal, retrying transient manifest CAS races."""

        for attempt in range(self.manifest_retry_limit):
            try:
                return self._publish_crystal_once(
                    crystal,
                    expected_generation=expected_generation,
                )
            except ComputeCrystalConflictError:
                if (
                    expected_generation is not None
                    or attempt + 1 == self.manifest_retry_limit
                ):
                    raise
        raise AssertionError("bounded manifest retry loop did not terminate")

    def restore_crystal(self, payload_sha256: str) -> ComputeCrystal:
        address = require_sha256(payload_sha256, field="payload_sha256")
        with self._locked():
            manifest = self._manifest_unlocked()
            return self._restore_crystal_recursive_unlocked(
                address,
                manifest,
                memo={},
                visiting=set(),
            )

    def _restore_program_object_unlocked(self, digest: str) -> ComputeProgram:
        address = require_sha256(digest, field="program_sha256")
        try:
            data = self._restore_state(self.program_state_name(address))
        except KeyError as exc:
            raise ComputeCrystalMissError(
                f"missing compute program: {address}"
            ) from exc
        if hashlib.sha256(data).hexdigest() != address:
            raise ComputeCrystalIntegrityError(
                "compute-program content address does not match its bytes"
            )
        program = ComputeProgram.from_bytes(data)
        if program.sha256 != address:
            raise ComputeCrystalIntegrityError(
                "compute program failed semantic content rehash"
            )
        return program

    def _resolve_program_unlocked(
        self,
        program: ComputeProgram,
        manifest: ComputeBankManifest,
    ) -> tuple[ComputeCrystal, ...]:
        memo: dict[str, ComputeCrystal] = {}
        crystals = tuple(
            self._restore_crystal_recursive_unlocked(
                digest,
                manifest,
                memo=memo,
                visiting=set(),
            )
            for digest in program.crystal_sha256s
        )
        return program.validate_crystals(crystals)

    def _publish_program_once(
        self,
        program: ComputeProgram,
        *,
        expected_generation: int | None = None,
    ) -> ComputeBankPublication:
        if not isinstance(program, ComputeProgram):
            raise TypeError("program must be a ComputeProgram")
        data = program.to_bytes()
        digest = program.sha256
        with self._locked():
            current = self._manifest_unlocked()
            self._check_expected_generation(current, expected_generation)
            self._resolve_program_unlocked(program, current)
            object_created = self._publish_content_unlocked(
                name=self.program_state_name(digest),
                data=data,
                digest=digest,
            )
            if digest in current.program_sha256s:
                return ComputeBankPublication(
                    artifact_kind="program",
                    payload_sha256=digest,
                    generation=current.generation,
                    manifest_sha256=current.sha256,
                    current_anchor_sha256=current.sha256,
                    object_created=object_created,
                    manifest_changed=False,
                )
            updated = ComputeBankManifest(
                generation=current.generation + 1,
                crystal_sha256s=current.crystal_sha256s,
                program_sha256s=tuple(sorted((*current.program_sha256s, digest))),
                charge_sha256s=current.charge_sha256s,
                previous_manifest_sha256=current.sha256,
            )
            self._commit_manifest_unlocked(current, updated)
            return ComputeBankPublication(
                artifact_kind="program",
                payload_sha256=digest,
                generation=updated.generation,
                manifest_sha256=updated.sha256,
                current_anchor_sha256=updated.sha256,
                object_created=object_created,
                manifest_changed=True,
            )

    def publish_program(
        self,
        program: ComputeProgram,
        *,
        expected_generation: int | None = None,
    ) -> ComputeBankPublication:
        """Publish one program, retrying transient manifest CAS races."""

        for attempt in range(self.manifest_retry_limit):
            try:
                return self._publish_program_once(
                    program,
                    expected_generation=expected_generation,
                )
            except ComputeCrystalConflictError:
                if (
                    expected_generation is not None
                    or attempt + 1 == self.manifest_retry_limit
                ):
                    raise
        raise AssertionError("bounded manifest retry loop did not terminate")

    def restore_program(self, payload_sha256: str) -> ComputeProgram:
        address = require_sha256(payload_sha256, field="payload_sha256")
        with self._locked():
            manifest = self._manifest_unlocked()
            if address not in manifest.program_sha256s:
                raise ComputeCrystalMissError(
                    f"compute program is not published by the manifest: {address}"
                )
            program = self._restore_program_object_unlocked(address)
            self._resolve_program_unlocked(program, manifest)
            return program

    def _restore_charge_object_unlocked(self, digest: str) -> ComputeChargeReceipt:
        address = require_sha256(digest, field="charge_sha256")
        try:
            data = self._restore_state(self.charge_state_name(address))
        except KeyError as exc:
            raise ComputeCrystalMissError(f"missing compute charge: {address}") from exc
        if hashlib.sha256(data).hexdigest() != address:
            raise ComputeCrystalIntegrityError(
                "compute-charge content address does not match its bytes"
            )
        charge = ComputeChargeReceipt.from_bytes(data)
        if charge.sha256 != address:
            raise ComputeCrystalIntegrityError(
                "compute charge failed semantic content rehash"
            )
        return charge

    def _resolve_charge_unlocked(
        self,
        charge: ComputeChargeReceipt,
        manifest: ComputeBankManifest,
    ) -> tuple[ComputeProgram, tuple[ComputeCrystal, ...], ComputeCrystal]:
        if charge.source_program_sha256 not in manifest.program_sha256s:
            raise ComputeCrystalMissError(
                "compute charge references an unpublished source program"
            )
        source_program = self._restore_program_object_unlocked(
            charge.source_program_sha256
        )
        source_crystals = self._resolve_program_unlocked(source_program, manifest)
        fused_crystal = self._restore_crystal_recursive_unlocked(
            charge.fused_crystal_sha256,
            manifest,
            memo={},
            visiting=set(),
        )
        charge.validate_basis(
            source_program=source_program,
            source_crystals=source_crystals,
            fused_crystal=fused_crystal,
        )
        return source_program, source_crystals, fused_crystal

    def _publish_charge_once(
        self,
        charge: ComputeChargeReceipt,
        *,
        expected_generation: int | None = None,
    ) -> ComputeBankPublication:
        if not isinstance(charge, ComputeChargeReceipt):
            raise TypeError("charge must be a ComputeChargeReceipt")
        data = charge.to_bytes()
        digest = charge.sha256
        with self._locked():
            current = self._manifest_unlocked()
            self._check_expected_generation(current, expected_generation)
            self._resolve_charge_unlocked(charge, current)
            object_created = self._publish_content_unlocked(
                name=self.charge_state_name(digest),
                data=data,
                digest=digest,
            )
            if digest in current.charge_sha256s:
                return ComputeBankPublication(
                    artifact_kind="charge",
                    payload_sha256=digest,
                    generation=current.generation,
                    manifest_sha256=current.sha256,
                    current_anchor_sha256=current.sha256,
                    object_created=object_created,
                    manifest_changed=False,
                )
            updated = ComputeBankManifest(
                generation=current.generation + 1,
                crystal_sha256s=current.crystal_sha256s,
                program_sha256s=current.program_sha256s,
                charge_sha256s=tuple(sorted((*current.charge_sha256s, digest))),
                previous_manifest_sha256=current.sha256,
            )
            self._commit_manifest_unlocked(current, updated)
            return ComputeBankPublication(
                artifact_kind="charge",
                payload_sha256=digest,
                generation=updated.generation,
                manifest_sha256=updated.sha256,
                current_anchor_sha256=updated.sha256,
                object_created=object_created,
                manifest_changed=True,
            )

    def publish_charge(
        self,
        charge: ComputeChargeReceipt,
        *,
        expected_generation: int | None = None,
    ) -> ComputeBankPublication:
        """Publish a verified charge basis with manifest CAS retry."""

        for attempt in range(self.manifest_retry_limit):
            try:
                return self._publish_charge_once(
                    charge,
                    expected_generation=expected_generation,
                )
            except ComputeCrystalConflictError:
                if (
                    expected_generation is not None
                    or attempt + 1 == self.manifest_retry_limit
                ):
                    raise
        raise AssertionError("bounded manifest retry loop did not terminate")

    def restore_charge(self, payload_sha256: str) -> ComputeChargeReceipt:
        address = require_sha256(payload_sha256, field="payload_sha256")
        with self._locked():
            manifest = self._manifest_unlocked()
            if address not in manifest.charge_sha256s:
                raise ComputeCrystalMissError(
                    f"compute charge is not published by the manifest: {address}"
                )
            charge = self._restore_charge_object_unlocked(address)
            self._resolve_charge_unlocked(charge, manifest)
            return charge

    def resolve_program(
        self, program: ComputeProgram | str
    ) -> tuple[ComputeProgram, tuple[ComputeCrystal, ...]]:
        with self._locked():
            manifest = self._manifest_unlocked()
            if isinstance(program, str):
                address = require_sha256(program, field="program_sha256")
            elif isinstance(program, ComputeProgram):
                address = program.sha256
            else:
                raise TypeError("program must be a ComputeProgram or SHA-256 address")
            if address not in manifest.program_sha256s:
                raise ComputeCrystalMissError(
                    f"compute program is not published by the manifest: {address}"
                )
            resolved = self._restore_program_object_unlocked(address)
            if (
                isinstance(program, ComputeProgram)
                and resolved.to_bytes() != program.to_bytes()
            ):
                raise ComputeCrystalIntegrityError(
                    "in-memory program disagrees with its published content address"
                )
            crystals = self._resolve_program_unlocked(resolved, manifest)
            return resolved, crystals

    def resolve_program_with_charge(
        self,
        program: ComputeProgram | str,
        charge_sha256: str | None,
    ) -> tuple[
        ComputeProgram,
        tuple[ComputeCrystal, ...],
        ComputeChargeReceipt | None,
    ]:
        with self._locked():
            manifest = self._manifest_unlocked()
            if isinstance(program, str):
                address = require_sha256(program, field="program_sha256")
            elif isinstance(program, ComputeProgram):
                address = program.sha256
            else:
                raise TypeError("program must be a ComputeProgram or SHA-256 address")
            if address not in manifest.program_sha256s:
                raise ComputeCrystalMissError(
                    f"compute program is not published by the manifest: {address}"
                )
            resolved = self._restore_program_object_unlocked(address)
            if (
                isinstance(program, ComputeProgram)
                and resolved.to_bytes() != program.to_bytes()
            ):
                raise ComputeCrystalIntegrityError(
                    "in-memory program disagrees with its published content address"
                )
            crystals = self._resolve_program_unlocked(resolved, manifest)
            if charge_sha256 is None:
                return resolved, crystals, None
            charge_address = require_sha256(charge_sha256, field="charge_basis_sha256")
            if charge_address not in manifest.charge_sha256s:
                raise ComputeCrystalMissError(
                    "compute charge is not published by the manifest"
                )
            charge = self._restore_charge_object_unlocked(charge_address)
            self._resolve_charge_unlocked(charge, manifest)
            if len(crystals) != 1 or crystals[0].sha256 != charge.fused_crystal_sha256:
                raise ComputeCrystalIntegrityError(
                    "compute charge does not apply to this discharge program"
                )
            return resolved, crystals, charge


class ComputeCrystalVM:
    """Restore and discharge safe numerical compute programs."""

    def __init__(self, bank: ComputeCrystalBank) -> None:
        if not isinstance(bank, ComputeCrystalBank):
            raise TypeError("bank must be a ComputeCrystalBank")
        self.bank = bank

    def execute(
        self,
        program: ComputeProgram | str,
        value: object,
        *,
        charge_basis_sha256: str | None = None,
    ) -> ComputeExecution:
        resolved, crystals, charge = self.bank.resolve_program_with_charge(
            program,
            charge_basis_sha256,
        )
        current = resolved.input_abi.validate(value, field="compute-program input")
        input_digest = tensor_sha256(current, resolved.input_abi)
        live_work = 0
        charged_applications = (
            crystals[0].input_abi.application_count(current)
            if charge is not None
            else None
        )
        for crystal in crystals:
            applications = crystal.input_abi.application_count(current)
            live_work = _checked_add(
                live_work,
                _checked_multiply(
                    crystal.discharge_work_units,
                    applications,
                    field="live discharge work",
                ),
                field="live discharge work",
            )
            current = crystal.apply(current)
        if charge is None:
            source_work = live_work
        else:
            if charged_applications is None:
                raise AssertionError("validated charge lost its application count")
            source_work = _checked_multiply(
                charge.source_work_units,
                charged_applications,
                field="charged equivalent source work",
            )
            charged_live_work = _checked_multiply(
                charge.live_work_units,
                charged_applications,
                field="charged live discharge work",
            )
            if charged_live_work != live_work:
                raise ComputeCrystalIntegrityError(
                    "compute charge live work disagrees with actual discharge"
                )
        resolved.output_abi.validate(current, field="compute-program output")
        receipt = ComputeExecutionReceipt(
            program_sha256=resolved.sha256,
            input_sha256=input_digest,
            output_sha256=tensor_sha256(current, resolved.output_abi),
            charge_basis_sha256=None if charge is None else charge.sha256,
            executed_operator_count=len(crystals),
            equivalent_unfused_source_work=source_work,
            live_discharge_work=live_work,
            historical_work_released=max(0, source_work - live_work),
        )
        return ComputeExecution(output=current, receipt=receipt)

    discharge = execute
