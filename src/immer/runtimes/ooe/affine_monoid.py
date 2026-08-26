"""Canonical exact guarded affine monoids over integer and modular state.

The executable core is deliberately small: an action carries an exact pair
``(A, b)`` and maps ``x`` to ``A x + b``.  Integer coordinates use unbounded
Python integers subject to an explicit bit bound; modular coordinates reduce
each output row in its declared quotient ring.  A guard-stage trace preserves
the domain of partial actions when atoms are fused.  Consequently fusion uses
the ordinary affine law while stack underflow, DFA phases, and parser grammar
remain exact instead of being silently widened.

The production constructors at the bottom of the module build useful finite
machines from that core.  Fingerprints intentionally stop at a collision-prone
candidate result; only :func:`verify_fingerprint_equality` emits an exact
equality receipt.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
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
from typing import Any, Literal

from .crystal import CrystalStore, CrystalStoreError, StatePublication
from .identity import canonical_json_bytes, require_sha256


STATE_SCHEMA = "immer-ooe-affine-state/v1"
ACTION_SCHEMA = "immer-ooe-affine-action/v1"
PROGRAM_SCHEMA = "immer-ooe-affine-program/v1"
STATE_RECEIPT_SCHEMA = "immer-ooe-affine-state-receipt/v1"
PROGRAM_RECEIPT_SCHEMA = "immer-ooe-affine-program-receipt/v1"
EXECUTION_RECEIPT_SCHEMA = "immer-ooe-affine-execution-receipt/v1"
COMPOSITION_RECEIPT_SCHEMA = "immer-ooe-affine-composition-receipt/v1"
FINGERPRINT_VERIFICATION_SCHEMA = "immer-ooe-fingerprint-verification/v1"
EXECUTION_BUNDLE_SCHEMA = "immer-ooe-affine-execution-bundle/v1"

MAX_DIMENSION = 512
MAX_ACTIONS = 65_536
MAX_GUARDS = 4_096
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_LOGICAL_CAPACITY = 1_000_000
MAX_BIT_BOUND = 1_000_000
MAX_WORK_UNITS = (1 << 63) - 1

_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.:+-]{0,127}$")


class AffineMonoidError(ValueError):
    """Base error for malformed or inapplicable affine artifacts."""


class AffineIntegrityError(AffineMonoidError):
    """A canonical artifact or receipt failed authentication."""


class AffineDomainError(AffineMonoidError):
    """A state or action violates its declared exact domain."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _integer(
    value: object,
    *,
    field_name: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AffineIntegrityError(f"{field_name} must be an integer")
    if minimum is not None and value < minimum:
        raise AffineIntegrityError(f"{field_name} is below its minimum")
    if maximum is not None and value > maximum:
        raise AffineIntegrityError(f"{field_name} exceeds its maximum")
    return value


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise AffineIntegrityError(f"{field_name} is not a canonical name")
    return value


def _strict_decode(data: bytes) -> object:
    if not isinstance(data, bytes) or len(data) > MAX_ARTIFACT_BYTES:
        raise AffineIntegrityError("artifact must be bounded immutable bytes")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
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
        raise AffineIntegrityError("artifact is not strict JSON") from exc
    if canonical_json_bytes(value) != data:
        raise AffineIntegrityError("artifact is not canonical JSON")
    return value


def _seal(schema: str, body: Mapping[str, Any]) -> dict[str, Any]:
    exact = dict(body)
    return {"body": exact, "body_sha256": _digest(exact), "schema": schema}


def _open(value: object, *, schema: str) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != {"body", "body_sha256", "schema"}
        or value.get("schema") != schema
        or not isinstance(value.get("body"), dict)
    ):
        raise AffineIntegrityError(f"unsupported {schema} envelope")
    expected = require_sha256(value["body_sha256"], field="body_sha256")
    body = dict(value["body"])
    if _digest(body) != expected:
        raise AffineIntegrityError("artifact body SHA-256 mismatch")
    return body


def _keys(value: Mapping[str, Any], expected: set[str], *, field_name: str) -> None:
    if set(value) != expected:
        raise AffineIntegrityError(f"{field_name} has unknown or missing fields")


def _matrix(
    value: object, *, dimension: int, field_name: str
) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != dimension:
        raise AffineIntegrityError(f"{field_name} must be a square matrix")
    rows: list[tuple[int, ...]] = []
    for row_index, raw_row in enumerate(value):
        if not isinstance(raw_row, (list, tuple)) or len(raw_row) != dimension:
            raise AffineIntegrityError(f"{field_name}[{row_index}] has the wrong width")
        rows.append(
            tuple(
                _integer(item, field_name=f"{field_name}[{row_index}]")
                for item in raw_row
            )
        )
    return tuple(rows)


def _vector(value: object, *, dimension: int, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != dimension:
        raise AffineIntegrityError(f"{field_name} has the wrong dimension")
    return tuple(_integer(item, field_name=field_name) for item in value)


def _identity(dimension: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(1 if row == column else 0 for column in range(dimension))
        for row in range(dimension)
    )


@dataclass(frozen=True, slots=True)
class RingSpec:
    """One coordinate ring: exact integers or ``Z/modulus Z``."""

    kind: Literal["integer", "modular"] = "integer"
    modulus: int | None = None

    def __post_init__(self) -> None:
        if self.kind == "integer":
            if self.modulus is not None:
                raise AffineDomainError("integer rings do not have a modulus")
        elif self.kind == "modular":
            modulus = _integer(
                self.modulus,
                field_name="modulus",
                minimum=2,
                maximum=(1 << 63) - 1,
            )
            object.__setattr__(self, "modulus", modulus)
        else:
            raise AffineDomainError("ring kind must be integer or modular")

    def normalize(self, value: int) -> int:
        return value if self.kind == "integer" else value % int(self.modulus)

    def to_record(self) -> dict[str, object]:
        return {"kind": self.kind, "modulus": self.modulus}

    @classmethod
    def from_record(cls, value: object) -> RingSpec:
        if not isinstance(value, dict):
            raise AffineIntegrityError("ring must be an object")
        _keys(value, {"kind", "modulus"}, field_name="ring")
        return cls(kind=value["kind"], modulus=value["modulus"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class FieldBound:
    """Optional semantic interval for one state coordinate."""

    minimum: int | None = None
    maximum: int | None = None

    def __post_init__(self) -> None:
        if self.minimum is not None:
            _integer(self.minimum, field_name="bound.minimum")
        if self.maximum is not None:
            _integer(self.maximum, field_name="bound.maximum")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise AffineDomainError("field bound is empty")

    def contains(self, value: int) -> bool:
        return (self.minimum is None or value >= self.minimum) and (
            self.maximum is None or value <= self.maximum
        )

    def to_record(self) -> dict[str, int | None]:
        return {"maximum": self.maximum, "minimum": self.minimum}

    @classmethod
    def from_record(cls, value: object) -> FieldBound:
        if not isinstance(value, dict):
            raise AffineIntegrityError("field bound must be an object")
        _keys(value, {"maximum", "minimum"}, field_name="field bound")
        return cls(minimum=value["minimum"], maximum=value["maximum"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class VectorStateSchema:
    """A named vector ABI with per-coordinate rings and hard bounds."""

    name: str
    fields: tuple[str, ...]
    rings: tuple[RingSpec, ...]
    bounds: tuple[FieldBound, ...]
    initial: tuple[int, ...]
    dead: tuple[int, ...]
    logical_capacity: int
    max_abs_bits: int
    dead_index: int
    phase_index: int | None = None
    phases: tuple[tuple[int, str], ...] = ()
    nonzero_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, field_name="schema.name"))
        fields = tuple(_name(item, field_name="schema field") for item in self.fields)
        if not fields or len(fields) > MAX_DIMENSION or len(set(fields)) != len(fields):
            raise AffineDomainError("schema fields must be unique and bounded")
        object.__setattr__(self, "fields", fields)
        dimension = len(fields)
        rings = tuple(self.rings)
        bounds = tuple(self.bounds)
        object.__setattr__(self, "rings", rings)
        object.__setattr__(self, "bounds", bounds)
        if len(rings) != dimension or len(bounds) != dimension:
            raise AffineDomainError("rings and bounds must match the state dimension")
        if not all(isinstance(item, RingSpec) for item in rings):
            raise AffineDomainError("every coordinate needs a ring")
        if not all(isinstance(item, FieldBound) for item in bounds):
            raise AffineDomainError("every coordinate needs a bound")
        object.__setattr__(
            self,
            "logical_capacity",
            _integer(
                self.logical_capacity,
                field_name="logical_capacity",
                minimum=1,
                maximum=MAX_LOGICAL_CAPACITY,
            ),
        )
        object.__setattr__(
            self,
            "max_abs_bits",
            _integer(
                self.max_abs_bits,
                field_name="max_abs_bits",
                minimum=1,
                maximum=MAX_BIT_BOUND,
            ),
        )
        object.__setattr__(
            self,
            "dead_index",
            _integer(
                self.dead_index,
                field_name="dead_index",
                minimum=0,
                maximum=dimension - 1,
            ),
        )
        if self.phase_index is not None:
            object.__setattr__(
                self,
                "phase_index",
                _integer(
                    self.phase_index,
                    field_name="phase_index",
                    minimum=0,
                    maximum=dimension - 1,
                ),
            )
        phases = tuple(
            (
                _integer(code, field_name="phase code"),
                _name(label, field_name="phase label"),
            )
            for code, label in self.phases
        )
        if len({code for code, _ in phases}) != len(phases) or len(
            {label for _, label in phases}
        ) != len(phases):
            raise AffineDomainError("phase codes and labels must be unique")
        if phases and self.phase_index is None:
            raise AffineDomainError("phase labels require a phase coordinate")
        object.__setattr__(self, "phases", phases)
        nonzero_indices = tuple(
            _integer(
                index,
                field_name="nonzero index",
                minimum=0,
                maximum=dimension - 1,
            )
            for index in self.nonzero_indices
        )
        if len(set(nonzero_indices)) != len(nonzero_indices):
            raise AffineDomainError("nonzero coordinate indices must be unique")
        object.__setattr__(self, "nonzero_indices", nonzero_indices)
        initial = self._normalize_candidate(self.initial, field_name="initial")
        dead = self._normalize_candidate(self.dead, field_name="dead")
        if initial[self.dead_index] != 0 or dead[self.dead_index] != 1:
            raise AffineDomainError("initial/dead marker must be exactly 0/1")
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "dead", dead)

    @property
    def dimension(self) -> int:
        return len(self.fields)

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())

    def field_index(self, name: str) -> int:
        try:
            return self.fields.index(name)
        except ValueError as exc:
            raise AffineDomainError(f"unknown state field: {name}") from exc

    def _normalize_candidate(
        self, value: Sequence[int], *, field_name: str
    ) -> tuple[int, ...]:
        candidate = _vector(value, dimension=self.dimension, field_name=field_name)
        result: list[int] = []
        for index, raw in enumerate(candidate):
            normalized = self.rings[index].normalize(raw)
            if normalized.bit_length() > self.max_abs_bits:
                raise AffineDomainError(f"{field_name} exceeds the bit bound")
            if not self.bounds[index].contains(normalized):
                raise AffineDomainError(f"{field_name} violates field bounds")
            result.append(normalized)
        if self.phases and result[self.dead_index] == 0:
            phase = result[int(self.phase_index)]
            if phase not in {code for code, _ in self.phases}:
                raise AffineDomainError(f"{field_name} has an undeclared DFA phase")
        if result[self.dead_index] == 0 and any(
            result[index] == 0 for index in self.nonzero_indices
        ):
            raise AffineDomainError(f"{field_name} violates a nonzero coordinate")
        return tuple(result)

    def state(self, values: Sequence[int] | None = None) -> AffineState:
        exact = (
            self.initial
            if values is None
            else self._normalize_candidate(values, field_name="state")
        )
        return AffineState(schema_sha256=self.sha256, values=exact)

    def dead_state(self) -> AffineState:
        return AffineState(schema_sha256=self.sha256, values=self.dead)

    def is_dead(self, state: AffineState) -> bool:
        self.validate_state(state)
        return state.values[self.dead_index] == 1

    def validate_state(self, state: AffineState) -> None:
        if state.schema_sha256 != self.sha256:
            raise AffineDomainError("state schema SHA-256 mismatch")
        normalized = self._normalize_candidate(state.values, field_name="state")
        if normalized != state.values:
            raise AffineDomainError("state is not in canonical ring representation")

    def to_record(self) -> dict[str, object]:
        return {
            "bounds": [item.to_record() for item in self.bounds],
            "dead": list(self.dead),
            "dead_index": self.dead_index,
            "fields": list(self.fields),
            "initial": list(self.initial),
            "logical_capacity": self.logical_capacity,
            "max_abs_bits": self.max_abs_bits,
            "name": self.name,
            "nonzero_indices": list(self.nonzero_indices),
            "phase_index": self.phase_index,
            "phases": [[code, label] for code, label in self.phases],
            "rings": [item.to_record() for item in self.rings],
        }

    @classmethod
    def from_record(cls, value: object) -> VectorStateSchema:
        if not isinstance(value, dict):
            raise AffineIntegrityError("state schema must be an object")
        _keys(
            value,
            {
                "bounds",
                "dead",
                "dead_index",
                "fields",
                "initial",
                "logical_capacity",
                "max_abs_bits",
                "name",
                "nonzero_indices",
                "phase_index",
                "phases",
                "rings",
            },
            field_name="state schema",
        )
        raw_fields = value["fields"]
        raw_rings = value["rings"]
        raw_bounds = value["bounds"]
        raw_phases = value["phases"]
        if not all(
            isinstance(item, list)
            for item in (
                raw_fields,
                raw_rings,
                raw_bounds,
                value["initial"],
                value["dead"],
                raw_phases,
                value["nonzero_indices"],
            )
        ):
            raise AffineIntegrityError("state schema sequences must be arrays")
        if not all(isinstance(item, list) and len(item) == 2 for item in raw_phases):
            raise AffineIntegrityError("phases must be code-label pairs")
        return cls(
            name=value["name"],  # type: ignore[arg-type]
            fields=tuple(raw_fields),  # type: ignore[arg-type]
            rings=tuple(RingSpec.from_record(item) for item in raw_rings),  # type: ignore[arg-type]
            bounds=tuple(FieldBound.from_record(item) for item in raw_bounds),  # type: ignore[arg-type]
            initial=tuple(value["initial"]),  # type: ignore[arg-type]
            dead=tuple(value["dead"]),  # type: ignore[arg-type]
            logical_capacity=value["logical_capacity"],  # type: ignore[arg-type]
            max_abs_bits=value["max_abs_bits"],  # type: ignore[arg-type]
            dead_index=value["dead_index"],  # type: ignore[arg-type]
            phase_index=value["phase_index"],  # type: ignore[arg-type]
            phases=tuple((item[0], item[1]) for item in raw_phases),
            nonzero_indices=tuple(value["nonzero_indices"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class AffineState:
    """An immutable canonical vector state."""

    schema_sha256: str
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_sha256",
            require_sha256(self.schema_sha256, field="schema_sha256"),
        )
        object.__setattr__(
            self,
            "values",
            tuple(_integer(item, field_name="state value") for item in self.values),
        )

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {"schema_sha256": self.schema_sha256, "values": list(self.values)}

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(_seal(STATE_SCHEMA, self.to_record()))

    @classmethod
    def from_bytes(cls, data: bytes, *, schema: VectorStateSchema) -> AffineState:
        body = _open(_strict_decode(data), schema=STATE_SCHEMA)
        _keys(body, {"schema_sha256", "values"}, field_name="state")
        if not isinstance(body["values"], list):
            raise AffineIntegrityError("state values must be an array")
        state = cls(
            schema_sha256=body["schema_sha256"],  # type: ignore[arg-type]
            values=tuple(body["values"]),  # type: ignore[arg-type]
        )
        schema.validate_state(state)
        return state


GuardKind = Literal[
    "eq",
    "ne",
    "range",
    "eq_index",
    "all_zero",
    "not_all_zero",
    "stack_layout",
]


@dataclass(frozen=True, slots=True)
class Guard:
    """A serializable exact predicate over a pre-action state."""

    kind: GuardKind
    index: int | None = None
    value: int | None = None
    other_index: int | None = None
    minimum: int | None = None
    maximum: int | None = None
    indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        allowed = {
            "eq",
            "ne",
            "range",
            "eq_index",
            "all_zero",
            "not_all_zero",
            "stack_layout",
        }
        if self.kind not in allowed:
            raise AffineDomainError("unsupported guard kind")
        for attr in ("index", "value", "other_index", "minimum", "maximum"):
            item = getattr(self, attr)
            if item is not None:
                _integer(item, field_name=f"guard.{attr}")
        object.__setattr__(
            self,
            "indices",
            tuple(
                _integer(item, field_name="guard index", minimum=0)
                for item in self.indices
            ),
        )
        if self.kind in {"eq", "ne"}:
            valid = self.index is not None and self.value is not None
        elif self.kind == "range":
            valid = (
                self.index is not None
                and self.minimum is not None
                and self.maximum is not None
                and self.minimum <= self.maximum
            )
        elif self.kind == "eq_index":
            valid = self.index is not None and self.other_index is not None
        elif self.kind in {"all_zero", "not_all_zero"}:
            valid = bool(self.indices)
        else:
            valid = self.index is not None and bool(self.indices)
        if not valid:
            raise AffineDomainError("guard parameters do not match its kind")

    @classmethod
    def equal(cls, index: int, value: int) -> Guard:
        return cls("eq", index=index, value=value)

    @classmethod
    def unequal(cls, index: int, value: int) -> Guard:
        return cls("ne", index=index, value=value)

    @classmethod
    def between(cls, index: int, minimum: int, maximum: int) -> Guard:
        return cls("range", index=index, minimum=minimum, maximum=maximum)

    @classmethod
    def same(cls, first: int, second: int) -> Guard:
        return cls("eq_index", index=first, other_index=second)

    @classmethod
    def stack_layout(cls, depth_index: int, slot_indices: Sequence[int]) -> Guard:
        """Require occupied nonzero slots followed by canonical zero slack."""

        return cls("stack_layout", index=depth_index, indices=tuple(slot_indices))

    def validate_dimension(self, dimension: int) -> None:
        for index in (self.index, self.other_index, *self.indices):
            if index is not None and not 0 <= index < dimension:
                raise AffineDomainError("guard index exceeds the state dimension")

    def evaluate(self, values: tuple[int, ...]) -> bool:
        if self.kind == "eq":
            return values[int(self.index)] == self.value
        if self.kind == "ne":
            return values[int(self.index)] != self.value
        if self.kind == "range":
            return int(self.minimum) <= values[int(self.index)] <= int(self.maximum)
        if self.kind == "eq_index":
            return values[int(self.index)] == values[int(self.other_index)]
        if self.kind == "all_zero":
            return all(values[index] == 0 for index in self.indices)
        if self.kind == "not_all_zero":
            return any(values[index] != 0 for index in self.indices)
        depth = values[int(self.index)]
        return 0 <= depth <= len(self.indices) and all(
            (values[index] != 0) if offset < depth else (values[index] == 0)
            for offset, index in enumerate(self.indices)
        )

    def to_record(self) -> dict[str, object]:
        return {
            "index": self.index,
            "indices": list(self.indices),
            "kind": self.kind,
            "maximum": self.maximum,
            "minimum": self.minimum,
            "other_index": self.other_index,
            "value": self.value,
        }

    @classmethod
    def from_record(cls, value: object) -> Guard:
        if not isinstance(value, dict):
            raise AffineIntegrityError("guard must be an object")
        _keys(
            value,
            {"index", "indices", "kind", "maximum", "minimum", "other_index", "value"},
            field_name="guard",
        )
        if not isinstance(value["indices"], list):
            raise AffineIntegrityError("guard indices must be an array")
        return cls(
            kind=value["kind"],  # type: ignore[arg-type]
            index=value["index"],  # type: ignore[arg-type]
            value=value["value"],  # type: ignore[arg-type]
            other_index=value["other_index"],  # type: ignore[arg-type]
            minimum=value["minimum"],  # type: ignore[arg-type]
            maximum=value["maximum"],  # type: ignore[arg-type]
            indices=tuple(value["indices"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class AffineStage:
    """One guarded, unfused affine stage retained in an atom's domain trace."""

    matrix: tuple[tuple[int, ...], ...]
    bias: tuple[int, ...]
    guards: tuple[Guard, ...] = ()

    def __post_init__(self) -> None:
        dimension = len(self.matrix)
        if not 1 <= dimension <= MAX_DIMENSION:
            raise AffineDomainError("affine stage has an invalid dimension")
        object.__setattr__(
            self,
            "matrix",
            _matrix(self.matrix, dimension=dimension, field_name="stage matrix"),
        )
        object.__setattr__(
            self,
            "bias",
            _vector(self.bias, dimension=dimension, field_name="stage bias"),
        )
        guards = tuple(self.guards)
        if not all(isinstance(item, Guard) for item in guards):
            raise AffineDomainError("stage guards must be Guard objects")
        object.__setattr__(self, "guards", guards)

    def to_record(self) -> dict[str, object]:
        return {
            "bias": list(self.bias),
            "guards": [guard.to_record() for guard in self.guards],
            "matrix": [list(row) for row in self.matrix],
        }

    @classmethod
    def from_record(cls, value: object, *, dimension: int) -> AffineStage:
        if not isinstance(value, dict):
            raise AffineIntegrityError("affine stage must be an object")
        _keys(value, {"bias", "guards", "matrix"}, field_name="affine stage")
        if not isinstance(value["guards"], list):
            raise AffineIntegrityError("stage guards must be an array")
        guards = tuple(Guard.from_record(item) for item in value["guards"])  # type: ignore[arg-type]
        return cls(
            matrix=_matrix(
                value["matrix"], dimension=dimension, field_name="stage matrix"
            ),
            bias=_vector(value["bias"], dimension=dimension, field_name="stage bias"),
            guards=guards,
        )


def _compatible_map(source: RingSpec, target: RingSpec) -> bool:
    if source.kind == "integer":
        return True
    return target.kind == "modular" and int(source.modulus) % int(target.modulus) == 0


def _validate_stage(stage: AffineStage, schema: VectorStateSchema) -> None:
    dimension = schema.dimension
    matrix = _matrix(stage.matrix, dimension=dimension, field_name="matrix")
    _vector(stage.bias, dimension=dimension, field_name="bias")
    if len(stage.guards) > MAX_GUARDS:
        raise AffineDomainError("action has too many guards")
    for guard in stage.guards:
        guard.validate_dimension(dimension)
    for row in range(dimension):
        for column in range(dimension):
            if matrix[row][column] and not _compatible_map(
                schema.rings[column], schema.rings[row]
            ):
                raise AffineDomainError(
                    "matrix crosses incompatible quotient-ring coordinates"
                )


def _apply_pair(
    schema: VectorStateSchema,
    matrix: tuple[tuple[int, ...], ...],
    bias: tuple[int, ...],
    values: tuple[int, ...],
) -> tuple[int, ...]:
    result: list[int] = []
    for row, ring in enumerate(schema.rings):
        total = bias[row]
        for column, coefficient in enumerate(matrix[row]):
            total += coefficient * values[column]
        result.append(ring.normalize(total))
    return tuple(result)


def _compose_pair(
    schema: VectorStateSchema,
    first_matrix: tuple[tuple[int, ...], ...],
    first_bias: tuple[int, ...],
    second_matrix: tuple[tuple[int, ...], ...],
    second_bias: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Return ``second(first(x))`` as ``(A2 A1, A2 b1 + b2)``."""

    dimension = schema.dimension
    matrix_rows: list[tuple[int, ...]] = []
    bias: list[int] = []
    for row, ring in enumerate(schema.rings):
        new_row = []
        for column in range(dimension):
            coefficient = sum(
                second_matrix[row][middle] * first_matrix[middle][column]
                for middle in range(dimension)
            )
            new_row.append(ring.normalize(coefficient))
        matrix_rows.append(tuple(new_row))
        offset = second_bias[row] + sum(
            second_matrix[row][middle] * first_bias[middle]
            for middle in range(dimension)
        )
        bias.append(ring.normalize(offset))
    return tuple(matrix_rows), tuple(bias)


@dataclass(frozen=True, slots=True)
class AffineActionAtom:
    """An immutable action atom and its exact partial-domain trace."""

    name: str
    schema_sha256: str
    matrix: tuple[tuple[int, ...], ...]
    bias: tuple[int, ...]
    stages: tuple[AffineStage, ...]
    work_units: int = 1
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, field_name="action.name"))
        object.__setattr__(
            self,
            "schema_sha256",
            require_sha256(self.schema_sha256, field="schema_sha256"),
        )
        dimension = len(self.matrix)
        if not 1 <= dimension <= MAX_DIMENSION:
            raise AffineDomainError("action has an invalid dimension")
        object.__setattr__(
            self,
            "matrix",
            _matrix(self.matrix, dimension=dimension, field_name="matrix"),
        )
        object.__setattr__(
            self,
            "bias",
            _vector(self.bias, dimension=dimension, field_name="bias"),
        )
        stages = tuple(self.stages)
        if not all(isinstance(item, AffineStage) for item in stages):
            raise AffineDomainError("action stages must be AffineStage objects")
        object.__setattr__(self, "stages", stages)
        if not stages or len(stages) > MAX_ACTIONS:
            raise AffineDomainError("an action needs a bounded non-empty stage trace")
        object.__setattr__(
            self,
            "work_units",
            _integer(
                self.work_units,
                field_name="work_units",
                minimum=0,
                maximum=MAX_WORK_UNITS,
            ),
        )
        metadata = tuple(
            sorted(
                (
                    _name(key, field_name="metadata key"),
                    _name(value, field_name="metadata value"),
                )
                for key, value in self.metadata
            )
        )
        if len({key for key, _ in metadata}) != len(metadata):
            raise AffineDomainError("metadata keys must be unique")
        object.__setattr__(self, "metadata", metadata)

    @classmethod
    def create(
        cls,
        schema: VectorStateSchema,
        *,
        name: str,
        matrix: Sequence[Sequence[int]],
        bias: Sequence[int],
        guards: Sequence[Guard] = (),
        work_units: int = 1,
        metadata: Mapping[str, str] | None = None,
    ) -> AffineActionAtom:
        raw_matrix = _matrix(matrix, dimension=schema.dimension, field_name="matrix")
        raw_bias = _vector(bias, dimension=schema.dimension, field_name="bias")
        exact_matrix = tuple(
            tuple(schema.rings[row].normalize(item) for item in raw_row)
            for row, raw_row in enumerate(raw_matrix)
        )
        exact_bias = tuple(
            schema.rings[row].normalize(item) for row, item in enumerate(raw_bias)
        )
        stage = AffineStage(exact_matrix, exact_bias, tuple(guards))
        atom = cls(
            name=name,
            schema_sha256=schema.sha256,
            matrix=exact_matrix,
            bias=exact_bias,
            stages=(stage,),
            work_units=work_units,
            metadata=tuple((metadata or {}).items()),
        )
        atom.validate(schema)
        return atom

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())

    def validate(self, schema: VectorStateSchema) -> None:
        if self.schema_sha256 != schema.sha256:
            raise AffineDomainError("action schema SHA-256 mismatch")
        for stage in self.stages:
            _validate_stage(stage, schema)
            for row in stage.matrix:
                if any(abs(value).bit_length() > schema.max_abs_bits for value in row):
                    raise AffineDomainError("matrix coefficient exceeds the bit bound")
            if any(
                abs(value).bit_length() > schema.max_abs_bits for value in stage.bias
            ):
                raise AffineDomainError("bias coefficient exceeds the bit bound")
        expected_matrix = _identity(schema.dimension)
        expected_bias = (0,) * schema.dimension
        for stage in self.stages:
            expected_matrix, expected_bias = _compose_pair(
                schema,
                expected_matrix,
                expected_bias,
                stage.matrix,
                stage.bias,
            )
        canonical_matrix = tuple(
            tuple(schema.rings[row].normalize(value) for value in raw_row)
            for row, raw_row in enumerate(self.matrix)
        )
        canonical_bias = tuple(
            schema.rings[row].normalize(value) for row, value in enumerate(self.bias)
        )
        for row in canonical_matrix:
            if any(abs(value).bit_length() > schema.max_abs_bits for value in row):
                raise AffineDomainError(
                    "fused matrix coefficient exceeds the bit bound"
                )
        if any(
            abs(value).bit_length() > schema.max_abs_bits for value in canonical_bias
        ):
            raise AffineDomainError("fused bias coefficient exceeds the bit bound")
        if expected_matrix != canonical_matrix or expected_bias != canonical_bias:
            raise AffineIntegrityError("fused (A,b) disagrees with its stage trace")
        if canonical_matrix != self.matrix or canonical_bias != self.bias:
            raise AffineIntegrityError(
                "action coefficients are not canonical in their rings"
            )

    def apply(self, schema: VectorStateSchema, state: AffineState) -> AffineState:
        self.validate(schema)
        schema.validate_state(state)
        if schema.is_dead(state):
            return schema.dead_state()
        current = state.values
        for stage in self.stages:
            if not all(guard.evaluate(current) for guard in stage.guards):
                return schema.dead_state()
            candidate = _apply_pair(schema, stage.matrix, stage.bias, current)
            try:
                current = schema.state(candidate).values
            except AffineDomainError:
                return schema.dead_state()
        fused = _apply_pair(schema, self.matrix, self.bias, state.values)
        if fused != current:
            raise AffineIntegrityError("fused execution disagrees with stage execution")
        return schema.state(current)

    def to_record(self) -> dict[str, object]:
        return {
            "bias": list(self.bias),
            "matrix": [list(row) for row in self.matrix],
            "metadata": [[key, value] for key, value in self.metadata],
            "name": self.name,
            "schema_sha256": self.schema_sha256,
            "stages": [stage.to_record() for stage in self.stages],
            "work_units": self.work_units,
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(_seal(ACTION_SCHEMA, self.to_record()))

    @classmethod
    def from_bytes(cls, data: bytes, *, schema: VectorStateSchema) -> AffineActionAtom:
        return cls.from_record(
            _open(_strict_decode(data), schema=ACTION_SCHEMA), schema=schema
        )

    @classmethod
    def from_record(
        cls, value: object, *, schema: VectorStateSchema
    ) -> AffineActionAtom:
        if not isinstance(value, dict):
            raise AffineIntegrityError("action must be an object")
        _keys(
            value,
            {
                "bias",
                "matrix",
                "metadata",
                "name",
                "schema_sha256",
                "stages",
                "work_units",
            },
            field_name="action",
        )
        metadata = value["metadata"]
        if not isinstance(metadata, list) or not all(
            isinstance(item, list) and len(item) == 2 for item in metadata
        ):
            raise AffineIntegrityError("action metadata must be key-value pairs")
        if not isinstance(value["stages"], list):
            raise AffineIntegrityError("action stages must be an array")
        atom = cls(
            name=value["name"],  # type: ignore[arg-type]
            schema_sha256=value["schema_sha256"],  # type: ignore[arg-type]
            matrix=_matrix(
                value["matrix"], dimension=schema.dimension, field_name="matrix"
            ),
            bias=_vector(value["bias"], dimension=schema.dimension, field_name="bias"),
            stages=tuple(
                AffineStage.from_record(item, dimension=schema.dimension)
                for item in value["stages"]  # type: ignore[union-attr]
            ),
            work_units=value["work_units"],  # type: ignore[arg-type]
            metadata=tuple((item[0], item[1]) for item in metadata),
        )
        atom.validate(schema)
        return atom


@dataclass(frozen=True, slots=True)
class CompositionReceipt:
    schema_sha256: str
    first_action_sha256: str
    second_action_sha256: str
    composed_action_sha256: str
    matrix_sha256: str
    bias_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "schema_sha256",
            "first_action_sha256",
            "second_action_sha256",
            "composed_action_sha256",
            "matrix_sha256",
            "bias_sha256",
        ):
            object.__setattr__(
                self, name, require_sha256(getattr(self, name), field=name)
            )

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())

    def to_record(self) -> dict[str, str]:
        return {
            "bias_sha256": self.bias_sha256,
            "composed_action_sha256": self.composed_action_sha256,
            "first_action_sha256": self.first_action_sha256,
            "matrix_sha256": self.matrix_sha256,
            "schema_sha256": self.schema_sha256,
            "second_action_sha256": self.second_action_sha256,
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(_seal(COMPOSITION_RECEIPT_SCHEMA, self.to_record()))

    @classmethod
    def from_bytes(cls, data: bytes) -> CompositionReceipt:
        body = _open(_strict_decode(data), schema=COMPOSITION_RECEIPT_SCHEMA)
        _keys(body, set(cls.__dataclass_fields__), field_name="composition receipt")
        return cls(**body)  # type: ignore[arg-type]


def compose_actions(
    schema: VectorStateSchema,
    first: AffineActionAtom,
    second: AffineActionAtom,
    *,
    name: str | None = None,
) -> tuple[AffineActionAtom, CompositionReceipt]:
    """Fuse ``second(first(x))`` while retaining every partial-action guard."""

    first.validate(schema)
    second.validate(schema)
    matrix, bias = _compose_pair(
        schema, first.matrix, first.bias, second.matrix, second.bias
    )
    work_units = first.work_units + second.work_units
    if work_units > MAX_WORK_UNITS:
        raise AffineDomainError("composed work exceeds its integer bound")
    composed_name = name or f"{first.name}+{second.name}"
    if len(composed_name) > 128:
        composed_name = f"Compose:{_digest([first.sha256, second.sha256])[:32]}"
    composed = AffineActionAtom(
        name=composed_name,
        schema_sha256=schema.sha256,
        matrix=matrix,
        bias=bias,
        stages=first.stages + second.stages,
        work_units=work_units,
        metadata=(("family", "composed"),),
    )
    composed.validate(schema)
    receipt = CompositionReceipt(
        schema_sha256=schema.sha256,
        first_action_sha256=first.sha256,
        second_action_sha256=second.sha256,
        composed_action_sha256=composed.sha256,
        matrix_sha256=_digest([list(row) for row in matrix]),
        bias_sha256=_digest(list(bias)),
    )
    return composed, receipt


@dataclass(frozen=True, slots=True)
class AffineProgram:
    """A self-contained schema-pinned sequence of affine action atoms."""

    schema: VectorStateSchema
    actions: tuple[AffineActionAtom, ...]
    name: str = "AffineProgram"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, field_name="program.name"))
        actions = tuple(self.actions)
        if not all(isinstance(item, AffineActionAtom) for item in actions):
            raise AffineDomainError("program actions must be affine atoms")
        object.__setattr__(self, "actions", actions)
        if len(actions) > MAX_ACTIONS:
            raise AffineDomainError("program exceeds the action bound")
        for action in actions:
            action.validate(self.schema)

    @property
    def sha256(self) -> str:
        return _digest(self.to_body())

    def to_body(self) -> dict[str, object]:
        return {
            "actions": [action.to_record() for action in self.actions],
            "name": self.name,
            "schema": self.schema.to_record(),
            "schema_sha256": self.schema.sha256,
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(_seal(PROGRAM_SCHEMA, self.to_body()))

    @classmethod
    def from_bytes(cls, data: bytes) -> AffineProgram:
        body = _open(_strict_decode(data), schema=PROGRAM_SCHEMA)
        _keys(
            body, {"actions", "name", "schema", "schema_sha256"}, field_name="program"
        )
        schema = VectorStateSchema.from_record(body["schema"])
        if (
            require_sha256(body["schema_sha256"], field="schema_sha256")
            != schema.sha256
        ):
            raise AffineIntegrityError("embedded state schema SHA-256 mismatch")
        if not isinstance(body["actions"], list):
            raise AffineIntegrityError("program actions must be a list")
        return cls(
            schema=schema,
            actions=tuple(
                AffineActionAtom.from_record(item, schema=schema)
                for item in body["actions"]
            ),
            name=body["name"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class StateReceipt:
    schema_sha256: str
    state_sha256: str
    role: Literal["initial", "final", "checkpoint"]
    values_sha256: str

    def __post_init__(self) -> None:
        for name in ("schema_sha256", "state_sha256", "values_sha256"):
            object.__setattr__(
                self, name, require_sha256(getattr(self, name), field=name)
            )
        if self.role not in {"initial", "final", "checkpoint"}:
            raise AffineIntegrityError("unknown state receipt role")

    @classmethod
    def issue(
        cls, state: AffineState, *, role: Literal["initial", "final", "checkpoint"]
    ) -> StateReceipt:
        return cls(state.schema_sha256, state.sha256, role, _digest(list(state.values)))

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())

    def to_record(self) -> dict[str, str]:
        return {
            "role": self.role,
            "schema_sha256": self.schema_sha256,
            "state_sha256": self.state_sha256,
            "values_sha256": self.values_sha256,
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(_seal(STATE_RECEIPT_SCHEMA, self.to_record()))

    @classmethod
    def from_bytes(cls, data: bytes) -> StateReceipt:
        body = _open(_strict_decode(data), schema=STATE_RECEIPT_SCHEMA)
        _keys(body, set(cls.__dataclass_fields__), field_name="state receipt")
        return cls(**body)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ProgramReceipt:
    schema_sha256: str
    program_sha256: str
    action_sha256s: tuple[str, ...]
    step_count: int
    work_units: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_sha256",
            require_sha256(self.schema_sha256, field="schema_sha256"),
        )
        object.__setattr__(
            self,
            "program_sha256",
            require_sha256(self.program_sha256, field="program_sha256"),
        )
        object.__setattr__(
            self,
            "action_sha256s",
            tuple(
                require_sha256(item, field="action_sha256")
                for item in self.action_sha256s
            ),
        )
        object.__setattr__(
            self,
            "step_count",
            _integer(
                self.step_count,
                field_name="step_count",
                minimum=0,
                maximum=MAX_ACTIONS,
            ),
        )
        if self.step_count != len(self.action_sha256s):
            raise AffineIntegrityError("program receipt step count mismatch")
        object.__setattr__(
            self,
            "work_units",
            _integer(
                self.work_units,
                field_name="work_units",
                minimum=0,
                maximum=MAX_WORK_UNITS,
            ),
        )

    @classmethod
    def issue(cls, program: AffineProgram) -> ProgramReceipt:
        work = sum(action.work_units for action in program.actions)
        if work > MAX_WORK_UNITS:
            raise AffineDomainError("program work exceeds its bound")
        return cls(
            schema_sha256=program.schema.sha256,
            program_sha256=program.sha256,
            action_sha256s=tuple(action.sha256 for action in program.actions),
            step_count=len(program.actions),
            work_units=work,
        )

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "action_sha256s": list(self.action_sha256s),
            "program_sha256": self.program_sha256,
            "schema_sha256": self.schema_sha256,
            "step_count": self.step_count,
            "work_units": self.work_units,
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(_seal(PROGRAM_RECEIPT_SCHEMA, self.to_record()))

    @classmethod
    def from_bytes(cls, data: bytes) -> ProgramReceipt:
        body = _open(_strict_decode(data), schema=PROGRAM_RECEIPT_SCHEMA)
        _keys(body, set(cls.__dataclass_fields__), field_name="program receipt")
        return cls(
            schema_sha256=body["schema_sha256"],  # type: ignore[arg-type]
            program_sha256=body["program_sha256"],  # type: ignore[arg-type]
            action_sha256s=tuple(body["action_sha256s"]),  # type: ignore[arg-type]
            step_count=body["step_count"],  # type: ignore[arg-type]
            work_units=body["work_units"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ExecutionReceipt:
    schema_sha256: str
    program_sha256: str
    initial_state_sha256: str
    final_state_sha256: str
    transition_state_sha256s: tuple[str, ...]
    action_sha256s: tuple[str, ...]
    verifier_sha256s: tuple[str, ...]
    dead_at_step: int | None
    work_units: int

    def __post_init__(self) -> None:
        for name in (
            "schema_sha256",
            "program_sha256",
            "initial_state_sha256",
            "final_state_sha256",
        ):
            object.__setattr__(
                self, name, require_sha256(getattr(self, name), field=name)
            )
        for name in ("transition_state_sha256s", "action_sha256s", "verifier_sha256s"):
            object.__setattr__(
                self,
                name,
                tuple(require_sha256(item, field=name) for item in getattr(self, name)),
            )
        if len(self.action_sha256s) > MAX_ACTIONS:
            raise AffineIntegrityError("execution receipt exceeds the action bound")
        if len(self.transition_state_sha256s) != len(self.action_sha256s):
            raise AffineIntegrityError("execution transition/action count mismatch")
        if self.dead_at_step is not None:
            _integer(
                self.dead_at_step,
                field_name="dead_at_step",
                minimum=0,
                maximum=max(0, len(self.action_sha256s) - 1),
            )
        object.__setattr__(
            self,
            "work_units",
            _integer(
                self.work_units,
                field_name="work_units",
                minimum=0,
                maximum=MAX_WORK_UNITS,
            ),
        )

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "action_sha256s": list(self.action_sha256s),
            "dead_at_step": self.dead_at_step,
            "final_state_sha256": self.final_state_sha256,
            "initial_state_sha256": self.initial_state_sha256,
            "program_sha256": self.program_sha256,
            "schema_sha256": self.schema_sha256,
            "transition_state_sha256s": list(self.transition_state_sha256s),
            "verifier_sha256s": list(self.verifier_sha256s),
            "work_units": self.work_units,
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(_seal(EXECUTION_RECEIPT_SCHEMA, self.to_record()))

    @classmethod
    def from_bytes(cls, data: bytes) -> ExecutionReceipt:
        body = _open(_strict_decode(data), schema=EXECUTION_RECEIPT_SCHEMA)
        _keys(body, set(cls.__dataclass_fields__), field_name="execution receipt")
        return cls(
            schema_sha256=body["schema_sha256"],  # type: ignore[arg-type]
            program_sha256=body["program_sha256"],  # type: ignore[arg-type]
            initial_state_sha256=body["initial_state_sha256"],  # type: ignore[arg-type]
            final_state_sha256=body["final_state_sha256"],  # type: ignore[arg-type]
            transition_state_sha256s=tuple(body["transition_state_sha256s"]),  # type: ignore[arg-type]
            action_sha256s=tuple(body["action_sha256s"]),  # type: ignore[arg-type]
            verifier_sha256s=tuple(body["verifier_sha256s"]),  # type: ignore[arg-type]
            dead_at_step=body["dead_at_step"],  # type: ignore[arg-type]
            work_units=body["work_units"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    state: AffineState
    state_receipt: StateReceipt
    program_receipt: ProgramReceipt
    execution_receipt: ExecutionReceipt


class AffineMonoidRuntime:
    """Deterministically execute a program on any schema-compatible state."""

    @staticmethod
    def execute(
        program: AffineProgram,
        *,
        initial_state: AffineState | None = None,
        verifier_sha256s: Sequence[str] = (),
    ) -> ExecutionResult:
        schema = program.schema
        state = schema.state() if initial_state is None else initial_state
        schema.validate_state(state)
        initial = state
        transitions: list[str] = []
        dead_at: int | None = None
        work = 0
        for index, action in enumerate(program.actions):
            state = action.apply(schema, state)
            transitions.append(state.sha256)
            work += action.work_units
            if work > MAX_WORK_UNITS:
                raise AffineDomainError("execution work exceeds its bound")
            if dead_at is None and schema.is_dead(state):
                dead_at = index
        verifiers = tuple(
            require_sha256(item, field="verifier_sha256") for item in verifier_sha256s
        )
        program_receipt = ProgramReceipt.issue(program)
        state_receipt = StateReceipt.issue(state, role="final")
        execution = ExecutionReceipt(
            schema_sha256=schema.sha256,
            program_sha256=program.sha256,
            initial_state_sha256=initial.sha256,
            final_state_sha256=state.sha256,
            transition_state_sha256s=tuple(transitions),
            action_sha256s=tuple(action.sha256 for action in program.actions),
            verifier_sha256s=verifiers,
            dead_at_step=dead_at,
            work_units=work,
        )
        return ExecutionResult(state, state_receipt, program_receipt, execution)


@dataclass(frozen=True, slots=True)
class ExecutionBundle:
    """One self-contained, replay-verified execution payload.

    The program, both endpoint states, and every receipt are sealed together.
    Restoration replays the program from the embedded initial state and
    requires byte-identical receipts, so a collection of individually valid
    but mutually unrelated artifacts cannot be spliced into a bundle.
    """

    program: AffineProgram
    initial_state: AffineState
    final_state: AffineState
    state_receipt: StateReceipt
    program_receipt: ProgramReceipt
    execution_receipt: ExecutionReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.program, AffineProgram):
            raise TypeError("program must be an AffineProgram")
        for name in ("initial_state", "final_state"):
            if not isinstance(getattr(self, name), AffineState):
                raise TypeError(f"{name} must be an AffineState")
        if not isinstance(self.state_receipt, StateReceipt):
            raise TypeError("state_receipt must be a StateReceipt")
        if not isinstance(self.program_receipt, ProgramReceipt):
            raise TypeError("program_receipt must be a ProgramReceipt")
        if not isinstance(self.execution_receipt, ExecutionReceipt):
            raise TypeError("execution_receipt must be an ExecutionReceipt")
        self.program.schema.validate_state(self.initial_state)
        self.program.schema.validate_state(self.final_state)
        replay = AffineMonoidRuntime.execute(
            self.program,
            initial_state=self.initial_state,
            verifier_sha256s=self.execution_receipt.verifier_sha256s,
        )
        if replay.state != self.final_state:
            raise AffineIntegrityError("execution bundle final state failed replay")
        if replay.state_receipt != self.state_receipt:
            raise AffineIntegrityError("execution bundle state receipt failed replay")
        if replay.program_receipt != self.program_receipt:
            raise AffineIntegrityError("execution bundle program receipt failed replay")
        if replay.execution_receipt != self.execution_receipt:
            raise AffineIntegrityError(
                "execution bundle execution receipt failed replay"
            )

    @classmethod
    def capture(
        cls,
        program: AffineProgram,
        result: ExecutionResult,
        *,
        initial_state: AffineState | None = None,
    ) -> ExecutionBundle:
        if not isinstance(result, ExecutionResult):
            raise TypeError("result must be an ExecutionResult")
        initial = program.schema.state() if initial_state is None else initial_state
        return cls(
            program=program,
            initial_state=initial,
            final_state=result.state,
            state_receipt=result.state_receipt,
            program_receipt=result.program_receipt,
            execution_receipt=result.execution_receipt,
        )

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "execution_receipt": self.execution_receipt.to_record(),
            "final_state": self.final_state.to_record(),
            "initial_state": self.initial_state.to_record(),
            "program": self.program.to_body(),
            "program_receipt": self.program_receipt.to_record(),
            "state_receipt": self.state_receipt.to_record(),
        }

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(_seal(EXECUTION_BUNDLE_SCHEMA, self.to_record()))
        if len(data) > MAX_ARTIFACT_BYTES:
            raise AffineDomainError("execution bundle exceeds its hard byte limit")
        return data

    @classmethod
    def from_bytes(cls, data: bytes) -> ExecutionBundle:
        body = _open(_strict_decode(data), schema=EXECUTION_BUNDLE_SCHEMA)
        _keys(
            body,
            {
                "execution_receipt",
                "final_state",
                "initial_state",
                "program",
                "program_receipt",
                "state_receipt",
            },
            field_name="execution bundle",
        )
        for name in body:
            if not isinstance(body[name], dict):
                raise AffineIntegrityError(f"execution bundle {name} must be an object")
        program = AffineProgram.from_bytes(
            canonical_json_bytes(_seal(PROGRAM_SCHEMA, body["program"]))
        )
        initial = AffineState.from_bytes(
            canonical_json_bytes(_seal(STATE_SCHEMA, body["initial_state"])),
            schema=program.schema,
        )
        final = AffineState.from_bytes(
            canonical_json_bytes(_seal(STATE_SCHEMA, body["final_state"])),
            schema=program.schema,
        )
        state_receipt = StateReceipt.from_bytes(
            canonical_json_bytes(_seal(STATE_RECEIPT_SCHEMA, body["state_receipt"]))
        )
        program_receipt = ProgramReceipt.from_bytes(
            canonical_json_bytes(_seal(PROGRAM_RECEIPT_SCHEMA, body["program_receipt"]))
        )
        execution_receipt = ExecutionReceipt.from_bytes(
            canonical_json_bytes(
                _seal(EXECUTION_RECEIPT_SCHEMA, body["execution_receipt"])
            )
        )
        return cls(
            program=program,
            initial_state=initial,
            final_state=final,
            state_receipt=state_receipt,
            program_receipt=program_receipt,
            execution_receipt=execution_receipt,
        )


@dataclass(frozen=True, slots=True)
class AffineArtifactPublication:
    artifact_kind: Literal["program", "execution_bundle"]
    artifact_sha256: str
    payload_sha256: str
    generation: int
    changed: bool

    def __post_init__(self) -> None:
        if self.artifact_kind not in {"program", "execution_bundle"}:
            raise AffineIntegrityError("unsupported affine artifact kind")
        object.__setattr__(
            self,
            "artifact_sha256",
            require_sha256(self.artifact_sha256, field="artifact_sha256"),
        )
        object.__setattr__(
            self,
            "payload_sha256",
            require_sha256(self.payload_sha256, field="payload_sha256"),
        )
        object.__setattr__(
            self,
            "generation",
            _integer(self.generation, field_name="generation", minimum=1),
        )
        if not isinstance(self.changed, bool):
            raise AffineIntegrityError("publication changed flag must be boolean")


class AffineMonoidBank:
    """Crash-safe content-addressed programs and single-payload bundles."""

    _PROGRAM_PREFIX = "ooe-affine-program-object/v1:"
    _BUNDLE_PREFIX = "ooe-affine-execution-bundle-object/v1:"
    _LOCK_NAME = ".affine-monoid.lock"

    def __init__(self, store: CrystalStore | str | os.PathLike[str]) -> None:
        self.store = store if isinstance(store, CrystalStore) else CrystalStore(store)
        self.root = Path(self.store.root)

    @staticmethod
    def program_state_name(program_sha256: str) -> str:
        return f"{AffineMonoidBank._PROGRAM_PREFIX}{require_sha256(program_sha256, field='program_sha256')}"

    @staticmethod
    def bundle_state_name(bundle_sha256: str) -> str:
        return f"{AffineMonoidBank._BUNDLE_PREFIX}{require_sha256(bundle_sha256, field='bundle_sha256')}"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        path = self.root / self._LOCK_NAME
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as exc:
            raise AffineIntegrityError("cannot open affine-monoid bank lock") from exc
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise AffineIntegrityError("affine-monoid bank lock is not regular")
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            linked = path.lstat()
            if (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino):
                raise AffineIntegrityError("affine-monoid bank lock changed")
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _restore_raw(self, name: str) -> bytes:
        try:
            return self.store.restore_state(name)
        except KeyError:
            raise
        except CrystalStoreError as exc:
            raise AffineIntegrityError(
                "affine artifact failed store integrity"
            ) from exc

    def _publish(
        self,
        *,
        kind: Literal["program", "execution_bundle"],
        artifact_sha256: str,
        state_name: str,
        payload: bytes,
    ) -> AffineArtifactPublication:
        expected_payload_sha256 = hashlib.sha256(payload).hexdigest()
        with self._locked():
            try:
                existing = self._restore_raw(state_name)
            except KeyError:
                existing = None
            if existing is not None and existing != payload:
                raise AffineIntegrityError(
                    "content-addressed affine artifact was rebound"
                )
            try:
                publication: StatePublication = self.store.publish_state(
                    state_name, payload
                )
            except CrystalStoreError as exc:
                raise AffineIntegrityError(
                    "affine artifact publication failed store integrity"
                ) from exc
            if publication.payload_sha256 != expected_payload_sha256:
                raise AffineIntegrityError("affine publication payload digest mismatch")
            return AffineArtifactPublication(
                artifact_kind=kind,
                artifact_sha256=artifact_sha256,
                payload_sha256=publication.payload_sha256,
                generation=publication.generation,
                changed=publication.changed,
            )

    def publish_program(self, program: AffineProgram) -> AffineArtifactPublication:
        if not isinstance(program, AffineProgram):
            raise TypeError("program must be an AffineProgram")
        return self._publish(
            kind="program",
            artifact_sha256=program.sha256,
            state_name=self.program_state_name(program.sha256),
            payload=program.to_bytes(),
        )

    def restore_program(self, program_sha256: str) -> AffineProgram:
        digest = require_sha256(program_sha256, field="program_sha256")
        with self._locked():
            try:
                data = self._restore_raw(self.program_state_name(digest))
            except KeyError as exc:
                raise KeyError(f"unknown affine program: {digest}") from exc
            program = AffineProgram.from_bytes(data)
            if program.sha256 != digest:
                raise AffineIntegrityError(
                    "program content address does not match its payload"
                )
            return program

    def publish_bundle(self, bundle: ExecutionBundle) -> AffineArtifactPublication:
        if not isinstance(bundle, ExecutionBundle):
            raise TypeError("bundle must be an ExecutionBundle")
        return self._publish(
            kind="execution_bundle",
            artifact_sha256=bundle.sha256,
            state_name=self.bundle_state_name(bundle.sha256),
            payload=bundle.to_bytes(),
        )

    def restore_bundle(self, bundle_sha256: str) -> ExecutionBundle:
        digest = require_sha256(bundle_sha256, field="bundle_sha256")
        with self._locked():
            try:
                data = self._restore_raw(self.bundle_state_name(digest))
            except KeyError as exc:
                raise KeyError(f"unknown affine execution bundle: {digest}") from exc
            bundle = ExecutionBundle.from_bytes(data)
            if bundle.sha256 != digest:
                raise AffineIntegrityError(
                    "execution-bundle content address does not match its payload"
                )
            return bundle

    def capture_and_publish(
        self,
        program: AffineProgram,
        result: ExecutionResult,
        *,
        initial_state: AffineState | None = None,
    ) -> tuple[ExecutionBundle, AffineArtifactPublication]:
        bundle = ExecutionBundle.capture(program, result, initial_state=initial_state)
        return bundle, self.publish_bundle(bundle)


def _blank_matrix(dimension: int) -> list[list[int]]:
    return [[0] * dimension for _ in range(dimension)]


def _mutable_identity(dimension: int) -> list[list[int]]:
    return [list(row) for row in _identity(dimension)]


def _bounds(*items: tuple[int | None, int | None]) -> tuple[FieldBound, ...]:
    return tuple(FieldBound(minimum, maximum) for minimum, maximum in items)


@dataclass(frozen=True, slots=True)
class StackMachine:
    """Explicit bounded LIFO stack; slot zero is the top."""

    schema: VectorStateSchema
    actions: tuple[AffineActionAtom, ...]
    symbols: tuple[int, ...]

    def action(self, name: str) -> AffineActionAtom:
        for action in self.actions:
            if action.name == name:
                return action
        raise AffineDomainError(f"unknown stack action: {name}")

    def push(self, symbol: int) -> AffineActionAtom:
        return self.action(f"push:{symbol}")

    def top(self, state: AffineState) -> int:
        self.schema.validate_state(state)
        if self.schema.is_dead(state) or state.values[-2] == 0:
            raise AffineDomainError("empty or dead stack has no top")
        return state.values[0]


def build_stack_machine(*, capacity: int, symbols: Sequence[int]) -> StackMachine:
    capacity = _integer(capacity, field_name="capacity", minimum=1, maximum=256)
    exact_symbols = tuple(
        sorted(
            set(
                _integer(
                    item,
                    field_name="symbol",
                    minimum=1,
                    maximum=(1 << 63) - 1,
                )
                for item in symbols
            )
        )
    )
    if not exact_symbols:
        raise AffineDomainError("stack alphabet must not be empty")
    maximum_symbol = max(exact_symbols)
    dimension = capacity + 2
    depth = capacity
    dead = capacity + 1
    schema = VectorStateSchema(
        name=f"Stack{capacity}",
        fields=tuple(f"slot{index}" for index in range(capacity)) + ("depth", "dead"),
        rings=(RingSpec(),) * dimension,
        bounds=_bounds(
            *((0, maximum_symbol) for _ in range(capacity)),
            (0, capacity),
            (0, 1),
        ),
        initial=(0,) * dimension,
        dead=(0,) * (dimension - 1) + (1,),
        logical_capacity=capacity,
        max_abs_bits=max(2, maximum_symbol.bit_length(), capacity.bit_length()),
        dead_index=dead,
    )
    actions: list[AffineActionAtom] = []
    layout = Guard.stack_layout(depth, tuple(range(capacity)))
    for symbol in exact_symbols:
        matrix = _blank_matrix(dimension)
        for slot in range(1, capacity):
            matrix[slot][slot - 1] = 1
        matrix[depth][depth] = 1
        matrix[dead][dead] = 1
        bias = [0] * dimension
        bias[0] = symbol
        bias[depth] = 1
        actions.append(
            AffineActionAtom.create(
                schema,
                name=f"push:{symbol}",
                matrix=matrix,
                bias=bias,
                guards=(
                    Guard.equal(dead, 0),
                    layout,
                    Guard.between(depth, 0, capacity - 1),
                ),
                metadata={"family": "stack", "operation": "push"},
            )
        )
    matrix = _blank_matrix(dimension)
    for slot in range(capacity - 1):
        matrix[slot][slot + 1] = 1
    matrix[depth][depth] = 1
    matrix[dead][dead] = 1
    bias = [0] * dimension
    bias[depth] = -1
    actions.append(
        AffineActionAtom.create(
            schema,
            name="pop",
            matrix=matrix,
            bias=bias,
            guards=(
                Guard.equal(dead, 0),
                layout,
                Guard.between(depth, 1, capacity),
            ),
            metadata={"family": "stack", "operation": "pop"},
        )
    )
    actions.append(
        AffineActionAtom.create(
            schema,
            name="peek",
            matrix=_identity(dimension),
            bias=(0,) * dimension,
            guards=(
                Guard.equal(dead, 0),
                layout,
                Guard.between(depth, 1, capacity),
            ),
            metadata={"family": "stack", "operation": "peek"},
        )
    )
    return StackMachine(schema, tuple(actions), exact_symbols)


@dataclass(frozen=True, slots=True)
class AnBnCnMachine:
    schema: VectorStateSchema
    actions: tuple[AffineActionAtom, ...]

    def action(self, name: str) -> AffineActionAtom:
        for action in self.actions:
            if action.name == name:
                return action
        raise AffineDomainError(f"unknown a^n b^n c^n action: {name}")

    def accepts(self, state: AffineState) -> bool:
        self.schema.validate_state(state)
        return not self.schema.is_dead(state) and state.values[2] == 3

    def run(self, text: str) -> ExecutionResult:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        actions = tuple(
            self.action(character if character in "abc" else "invalid")
            for character in text
        ) + (self.action("finish"),)
        return AffineMonoidRuntime.execute(
            AffineProgram(self.schema, actions, "AnBnCnRun")
        )


def build_anbncn_machine(*, max_count: int) -> AnBnCnMachine:
    max_count = _integer(
        max_count, field_name="max_count", minimum=1, maximum=MAX_LOGICAL_CAPACITY
    )
    # x=#a-#b, y=#b-#c, phase: 0=a, 1=b, 2=c, 3=accepted.
    schema = VectorStateSchema(
        name="AnBnCn",
        fields=("a_minus_b", "b_minus_c", "phase", "dead"),
        rings=(RingSpec(),) * 4,
        bounds=_bounds((0, max_count), (0, max_count), (0, 3), (0, 1)),
        initial=(0, 0, 0, 0),
        dead=(0, 0, 0, 1),
        logical_capacity=max_count,
        max_abs_bits=max(2, max_count.bit_length()),
        dead_index=3,
        phase_index=2,
        phases=((0, "A"), (1, "B"), (2, "C"), (3, "ACCEPT")),
    )
    identity = _mutable_identity(4)
    actions: list[AffineActionAtom] = []
    bias = [1, 0, 0, 0]
    actions.append(
        AffineActionAtom.create(
            schema,
            name="a",
            matrix=identity,
            bias=bias,
            guards=(
                Guard.equal(3, 0),
                Guard.equal(2, 0),
                Guard.between(0, 0, max_count - 1),
            ),
            metadata={"family": "anbncn", "operation": "a"},
        )
    )
    matrix = _mutable_identity(4)
    matrix[2] = [0, 0, 0, 0]
    bias = [-1, 1, 1, 0]
    actions.append(
        AffineActionAtom.create(
            schema,
            name="b",
            matrix=matrix,
            bias=bias,
            guards=(
                Guard.equal(3, 0),
                Guard.between(2, 0, 1),
                Guard.between(0, 1, max_count),
            ),
            metadata={"family": "anbncn", "operation": "b"},
        )
    )
    matrix = _mutable_identity(4)
    matrix[2] = [0, 0, 0, 0]
    bias = [0, -1, 2, 0]
    actions.append(
        AffineActionAtom.create(
            schema,
            name="c",
            matrix=matrix,
            bias=bias,
            guards=(
                Guard.equal(3, 0),
                Guard.between(2, 1, 2),
                Guard.equal(0, 0),
                Guard.between(1, 1, max_count),
            ),
            metadata={"family": "anbncn", "operation": "c"},
        )
    )
    matrix = _mutable_identity(4)
    matrix[2] = [0, 0, 0, 0]
    bias = [0, 0, 3, 0]
    actions.append(
        AffineActionAtom.create(
            schema,
            name="finish",
            matrix=matrix,
            bias=bias,
            guards=(
                Guard.equal(3, 0),
                Guard.equal(2, 2),
                Guard.equal(0, 0),
                Guard.equal(1, 0),
            ),
            metadata={"family": "anbncn", "operation": "finish"},
        )
    )
    actions.append(
        AffineActionAtom.create(
            schema,
            name="invalid",
            matrix=_identity(4),
            bias=(0,) * 4,
            guards=(Guard.equal(3, 2),),
            metadata={"family": "anbncn", "operation": "invalid"},
        )
    )
    return AnBnCnMachine(schema, tuple(actions))


@dataclass(frozen=True, slots=True)
class FingerprintMachine:
    schema: VectorStateSchema
    base: int
    moduli: tuple[int, int]
    max_length: int

    def _symbol(
        self, value: int, *, side: Literal["left", "right"]
    ) -> AffineActionAtom:
        value = _integer(value, field_name="symbol", minimum=0, maximum=255)
        dimension = self.schema.dimension
        matrix = _mutable_identity(dimension)
        bias = [0] * dimension
        if side == "left":
            hash_indices = (0, 1)
            length_index = 4
            phase_guard = Guard.equal(6, 0)
            name = f"L:{value}"
        else:
            hash_indices = (2, 3)
            length_index = 5
            phase_guard = Guard.equal(6, 1)
            name = f"R:{value}"
        for index in hash_indices:
            matrix[index][index] = self.base
            bias[index] = value + 1
        bias[length_index] = 1
        return AffineActionAtom.create(
            self.schema,
            name=name,
            matrix=matrix,
            bias=bias,
            guards=(
                Guard.equal(8, 0),
                phase_guard,
                Guard.between(length_index, 0, self.max_length - 1),
            ),
            metadata={"family": "fingerprint", "operation": side},
        )

    def left(self, value: int) -> AffineActionAtom:
        return self._symbol(value, side="left")

    def right(self, value: int) -> AffineActionAtom:
        return self._symbol(value, side="right")

    def separator(self) -> AffineActionAtom:
        matrix = _mutable_identity(self.schema.dimension)
        matrix[6] = [0] * self.schema.dimension
        bias = [0] * self.schema.dimension
        bias[6] = 1
        return AffineActionAtom.create(
            self.schema,
            name="SEP",
            matrix=matrix,
            bias=bias,
            guards=(
                Guard.equal(8, 0),
                Guard.equal(6, 0),
                Guard.between(4, 1, self.max_length),
            ),
            metadata={"family": "fingerprint", "operation": "separator"},
        )

    def finish(self) -> AffineActionAtom:
        matrix = _mutable_identity(self.schema.dimension)
        matrix[6] = [0] * self.schema.dimension
        matrix[7] = [0] * self.schema.dimension
        bias = [0] * self.schema.dimension
        bias[6] = 2
        bias[7] = 1
        return AffineActionAtom.create(
            self.schema,
            name="FPFINISH",
            matrix=matrix,
            bias=bias,
            guards=(
                Guard.equal(8, 0),
                Guard.equal(6, 1),
                Guard.between(5, 1, self.max_length),
                Guard.same(0, 2),
                Guard.same(1, 3),
                Guard.same(4, 5),
            ),
            metadata={"family": "fingerprint", "operation": "candidate"},
        )

    def candidate_equal(self, state: AffineState) -> bool:
        self.schema.validate_state(state)
        return not self.schema.is_dead(state) and state.values[6:8] == (2, 1)

    def compare(self, left: bytes, right: bytes) -> ExecutionResult:
        if not isinstance(left, bytes) or not isinstance(right, bytes):
            raise TypeError("fingerprint inputs must be immutable bytes")
        actions = (
            tuple(self.left(value) for value in left)
            + (self.separator(),)
            + tuple(self.right(value) for value in right)
            + (self.finish(),)
        )
        return AffineMonoidRuntime.execute(
            AffineProgram(self.schema, actions, "FingerprintRun")
        )


def build_fingerprint_machine(
    *,
    max_length: int,
    base: int = 257,
    moduli: tuple[int, int] = (1_000_000_007, 1_000_000_009),
) -> FingerprintMachine:
    max_length = _integer(
        max_length, field_name="max_length", minimum=1, maximum=MAX_LOGICAL_CAPACITY
    )
    base = _integer(base, field_name="base", minimum=2, maximum=(1 << 31) - 1)
    first = _integer(
        moduli[0], field_name="first modulus", minimum=2, maximum=(1 << 63) - 1
    )
    second = _integer(
        moduli[1], field_name="second modulus", minimum=2, maximum=(1 << 63) - 1
    )
    if first == second or math.gcd(base, first) != 1 or math.gcd(base, second) != 1:
        raise AffineDomainError(
            "fingerprint moduli must be distinct and coprime to the base"
        )
    fields = (
        "left_h1",
        "left_h2",
        "right_h1",
        "right_h2",
        "left_length",
        "right_length",
        "phase",
        "candidate",
        "dead",
    )
    rings = (
        RingSpec("modular", first),
        RingSpec("modular", second),
        RingSpec("modular", first),
        RingSpec("modular", second),
        RingSpec(),
        RingSpec(),
        RingSpec(),
        RingSpec(),
        RingSpec(),
    )
    schema = VectorStateSchema(
        name="DualModFingerprint",
        fields=fields,
        rings=rings,
        bounds=_bounds(
            (0, first - 1),
            (0, second - 1),
            (0, first - 1),
            (0, second - 1),
            (0, max_length),
            (0, max_length),
            (0, 2),
            (0, 1),
            (0, 1),
        ),
        initial=(0,) * 9,
        dead=(0,) * 8 + (1,),
        logical_capacity=max_length,
        max_abs_bits=max(
            first.bit_length(), second.bit_length(), max_length.bit_length()
        ),
        dead_index=8,
        phase_index=6,
        phases=((0, "LEFT"), (1, "RIGHT"), (2, "CANDIDATE")),
    )
    return FingerprintMachine(schema, base, (first, second), max_length)


@dataclass(frozen=True, slots=True)
class FingerprintVerificationReceipt:
    execution_receipt_sha256: str
    left_sha256: str
    right_sha256: str
    verifier_sha256: str
    modular_candidate: bool
    exact_equal: bool

    def __post_init__(self) -> None:
        for name in (
            "execution_receipt_sha256",
            "left_sha256",
            "right_sha256",
            "verifier_sha256",
        ):
            object.__setattr__(
                self, name, require_sha256(getattr(self, name), field=name)
            )
        if not isinstance(self.modular_candidate, bool) or not isinstance(
            self.exact_equal, bool
        ):
            raise AffineIntegrityError("fingerprint verdicts must be booleans")
        if self.exact_equal and not self.modular_candidate:
            raise AffineIntegrityError(
                "exact equality cannot bypass the modular candidate path"
            )

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "exact_equal": self.exact_equal,
            "execution_receipt_sha256": self.execution_receipt_sha256,
            "left_sha256": self.left_sha256,
            "modular_candidate": self.modular_candidate,
            "right_sha256": self.right_sha256,
            "verifier_sha256": self.verifier_sha256,
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(
            _seal(FINGERPRINT_VERIFICATION_SCHEMA, self.to_record())
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> FingerprintVerificationReceipt:
        body = _open(_strict_decode(data), schema=FINGERPRINT_VERIFICATION_SCHEMA)
        _keys(
            body, set(cls.__dataclass_fields__), field_name="fingerprint verification"
        )
        return cls(**body)  # type: ignore[arg-type]


def verify_fingerprint_equality(
    machine: FingerprintMachine,
    result: ExecutionResult,
    *,
    left: bytes,
    right: bytes,
    verifier_sha256: str,
) -> FingerprintVerificationReceipt:
    """Cross the modular collision boundary with an exact byte comparison."""

    if not isinstance(left, bytes) or not isinstance(right, bytes):
        raise TypeError("fingerprint inputs must be immutable bytes")
    candidate = machine.candidate_equal(result.state)
    expected = machine.compare(left, right)
    if expected.execution_receipt.sha256 != result.execution_receipt.sha256:
        raise AffineIntegrityError(
            "fingerprint evidence does not match the supplied inputs"
        )
    return FingerprintVerificationReceipt(
        execution_receipt_sha256=result.execution_receipt.sha256,
        left_sha256=hashlib.sha256(left).hexdigest(),
        right_sha256=hashlib.sha256(right).hexdigest(),
        verifier_sha256=require_sha256(verifier_sha256, field="verifier_sha256"),
        modular_candidate=candidate,
        exact_equal=candidate and left == right,
    )


@dataclass(frozen=True, slots=True)
class DecimalHornerMachine:
    """Strict ``[+-]?(0|[1-9][0-9]*)`` parser using exact Horner updates."""

    schema: VectorStateSchema
    max_digits: int
    actions: tuple[AffineActionAtom, ...]

    def action(self, name: str) -> AffineActionAtom:
        for action in self.actions:
            if action.name == name:
                return action
        raise AffineDomainError(f"unknown decimal action: {name}")

    def parse(self, text: str) -> ExecutionResult:
        if not isinstance(text, str):
            raise TypeError("decimal input must be a string")
        actions: list[AffineActionAtom] = []
        offset = 0
        if text.startswith("+"):
            actions.append(self.action("plus"))
            offset = 1
        elif text.startswith("-"):
            actions.append(self.action("minus"))
            offset = 1
        for digit_offset, character in enumerate(text[offset:]):
            if character < "0" or character > "9":
                # A grammar error is represented by an explicit invalid action.
                actions.append(self.action("invalid"))
            elif character == "0":
                actions.append(
                    self.action(
                        "digit:0:first" if digit_offset == 0 else "digit:0:next"
                    )
                )
            else:
                actions.append(self.action(f"digit:{character}"))
        actions.append(self.action("finish"))
        return AffineMonoidRuntime.execute(
            AffineProgram(self.schema, tuple(actions), "DecimalHornerRun")
        )

    def value(self, state: AffineState) -> int:
        self.schema.validate_state(state)
        if self.schema.is_dead(state) or state.values[3] != 4:
            raise AffineDomainError("decimal state is not accepted")
        return state.values[0] * state.values[2]


def build_decimal_horner_machine(*, max_digits: int = 128) -> DecimalHornerMachine:
    max_digits = _integer(
        max_digits, field_name="max_digits", minimum=1, maximum=10_000
    )
    maximum = 10**max_digits - 1
    bit_bound = max(2, maximum.bit_length())
    # magnitude, digits, sign, phase, dead. phase 0=start, 1=sign, 2=digits,
    # 3=sole zero, 4=accepted.
    schema = VectorStateSchema(
        name=f"DecimalHorner{max_digits}",
        fields=("magnitude", "digits", "sign", "phase", "dead"),
        rings=(RingSpec(),) * 5,
        bounds=_bounds((0, maximum), (0, max_digits), (-1, 1), (0, 4), (0, 1)),
        initial=(0, 0, 1, 0, 0),
        dead=(0, 0, 0, 0, 1),
        logical_capacity=max_digits,
        max_abs_bits=bit_bound,
        dead_index=4,
        phase_index=3,
        phases=((0, "START"), (1, "SIGNED"), (2, "DIGITS"), (3, "ZERO"), (4, "ACCEPT")),
    )
    actions: list[AffineActionAtom] = []
    for name, sign in (("plus", 1), ("minus", -1)):
        matrix = _mutable_identity(5)
        matrix[2] = [0] * 5
        matrix[3] = [0] * 5
        bias = [0, 0, sign, 1, 0]
        actions.append(
            AffineActionAtom.create(
                schema,
                name=name,
                matrix=matrix,
                bias=bias,
                guards=(Guard.equal(4, 0), Guard.equal(3, 0)),
                metadata={"family": "decimal", "operation": "sign"},
            )
        )
    # Zero needs two guarded atoms: as the first digit it closes the numeral;
    # after a non-zero digit it is an ordinary Horner step.  Both remain exact
    # affine actions, and neither can be substituted for the other.
    matrix = _mutable_identity(5)
    matrix[0][0] = 10
    matrix[3] = [0] * 5
    bias = [0, 1, 0, 3, 0]
    actions.append(
        AffineActionAtom.create(
            schema,
            name="digit:0:first",
            matrix=matrix,
            bias=bias,
            guards=(Guard.equal(4, 0), Guard.between(3, 0, 1), Guard.equal(1, 0)),
            metadata={"family": "decimal", "operation": "digit"},
        )
    )
    matrix = _mutable_identity(5)
    matrix[0][0] = 10
    matrix[3] = [0] * 5
    bias = [0, 1, 0, 2, 0]
    actions.append(
        AffineActionAtom.create(
            schema,
            name="digit:0:next",
            matrix=matrix,
            bias=bias,
            guards=(
                Guard.equal(4, 0),
                Guard.equal(3, 2),
                Guard.between(1, 1, max_digits),
            ),
            metadata={"family": "decimal", "operation": "digit"},
        )
    )
    for digit in range(1, 10):
        matrix = _mutable_identity(5)
        matrix[0][0] = 10
        matrix[3] = [0] * 5
        bias = [digit, 1, 0, 2, 0]
        actions.append(
            AffineActionAtom.create(
                schema,
                name=f"digit:{digit}",
                matrix=matrix,
                bias=bias,
                guards=(
                    Guard.equal(4, 0),
                    Guard.between(3, 0, 2),
                    Guard.between(1, 0, max_digits - 1),
                ),
                metadata={"family": "decimal", "operation": "digit"},
            )
        )
    matrix = _mutable_identity(5)
    matrix[3] = [0] * 5
    bias = [0, 0, 0, 4, 0]
    actions.append(
        AffineActionAtom.create(
            schema,
            name="finish",
            matrix=matrix,
            bias=bias,
            guards=(
                Guard.equal(4, 0),
                Guard.between(3, 2, 3),
                Guard.between(1, 1, max_digits),
            ),
            metadata={"family": "decimal", "operation": "finish"},
        )
    )
    # Always-failing action converts foreign grammar symbols to the dead state.
    actions.append(
        AffineActionAtom.create(
            schema,
            name="invalid",
            matrix=_identity(5),
            bias=(0,) * 5,
            guards=(Guard.equal(4, 2),),
            metadata={"family": "decimal", "operation": "invalid"},
        )
    )
    return DecimalHornerMachine(schema, max_digits, tuple(actions))


GroupFamily = Literal["additive", "multiplicative", "cyclic"]


@dataclass(frozen=True, slots=True)
class GroupMapBridge:
    """Exact affine bridge for additive, multiplicative, and cyclic maps."""

    family: GroupFamily
    schema: VectorStateSchema

    def action(self, element: int) -> AffineActionAtom:
        element = _integer(element, field_name="group element")
        if self.family == "multiplicative" and element == 0:
            raise AffineDomainError("zero is not a multiplicative group element")
        if abs(element).bit_length() > self.schema.max_abs_bits:
            raise AffineDomainError("group element exceeds the declared bit bound")
        if self.family == "additive":
            coefficient, offset = 1, element
        elif self.family == "multiplicative":
            coefficient, offset = element, 0
        else:
            coefficient, offset = 1, element
        matrix = ((coefficient, 0), (0, 1))
        bias = (offset, 0)
        action_name = f"{self.family}:{element}"
        if len(action_name) > 128:
            action_name = f"{self.family}:{_digest(element)[:32]}"
        return AffineActionAtom.create(
            self.schema,
            name=action_name,
            matrix=matrix,
            bias=bias,
            guards=(Guard.equal(1, 0),),
            metadata={"family": self.family, "operation": "group_map"},
        )

    def value(self, state: AffineState) -> int:
        self.schema.validate_state(state)
        if self.schema.is_dead(state):
            raise AffineDomainError("dead group state has no value")
        return state.values[0]


def build_group_map_bridge(
    family: GroupFamily,
    *,
    initial: int | None = None,
    modulus: int | None = None,
    max_abs_bits: int = 4096,
) -> GroupMapBridge:
    if family not in {"additive", "multiplicative", "cyclic"}:
        raise AffineDomainError("unsupported group-map family")
    max_abs_bits = _integer(
        max_abs_bits, field_name="max_abs_bits", minimum=2, maximum=MAX_BIT_BOUND
    )
    if family == "cyclic":
        modulus = _integer(
            modulus, field_name="modulus", minimum=2, maximum=(1 << 63) - 1
        )
        ring = RingSpec("modular", modulus)
        bound = FieldBound(0, modulus - 1)
        start = (
            0
            if initial is None
            else ring.normalize(_integer(initial, field_name="initial"))
        )
    else:
        if modulus is not None:
            raise AffineDomainError("only cyclic bridges accept a modulus")
        ring = RingSpec()
        limit = (1 << max_abs_bits) - 1
        bound = FieldBound(-limit, limit)
        default = 1 if family == "multiplicative" else 0
        start = default if initial is None else _integer(initial, field_name="initial")
        if family == "multiplicative" and start == 0:
            raise AffineDomainError("zero is not a multiplicative group state")
    schema = VectorStateSchema(
        name=f"{family.capitalize()}Bridge",
        fields=("value", "dead"),
        rings=(ring, RingSpec()),
        bounds=(bound, FieldBound(0, 1)),
        initial=(start, 0),
        dead=(0, 1),
        logical_capacity=MAX_ACTIONS,
        max_abs_bits=max_abs_bits,
        dead_index=1,
        nonzero_indices=(0,) if family == "multiplicative" else (),
    )
    return GroupMapBridge(family, schema)


__all__ = [
    "AffineActionAtom",
    "AffineDomainError",
    "AffineIntegrityError",
    "AffineArtifactPublication",
    "AffineMonoidBank",
    "AffineMonoidRuntime",
    "AffineProgram",
    "AffineStage",
    "AffineState",
    "AnBnCnMachine",
    "CompositionReceipt",
    "DecimalHornerMachine",
    "ExecutionReceipt",
    "ExecutionResult",
    "ExecutionBundle",
    "FieldBound",
    "FingerprintMachine",
    "FingerprintVerificationReceipt",
    "GroupMapBridge",
    "Guard",
    "ProgramReceipt",
    "RingSpec",
    "StackMachine",
    "StateReceipt",
    "VectorStateSchema",
    "build_anbncn_machine",
    "build_decimal_horner_machine",
    "build_fingerprint_machine",
    "build_group_map_bridge",
    "build_stack_machine",
    "compose_actions",
    "verify_fingerprint_equality",
]
