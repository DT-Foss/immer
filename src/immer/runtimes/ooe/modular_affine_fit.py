"""Receipt-bound exact affine identification over a prime modular field.

The fitter learns one additive action

``y = A x + b  (mod p)``

from authenticated training observations and verifies it against a disjoint
calibration set.  It does not introduce another execution engine: the learned
operator is emitted as the existing :class:`AffineActionAtom`, with the
existing :class:`RingSpec` describing every learned coordinate.

``VectorStateSchema`` reserves one coordinate as its absorbing dead marker.
Observations contain the remaining ``d`` live coordinates in schema order;
the fitter embeds the learned ``d``-dimensional action into the full schema and
preserves the dead marker structurally.  Training rows are ordered by their
content address before exact Gauss-Jordan elimination, so input iteration order
cannot change the selected identifying rows or the resulting receipt.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import base64
import hashlib
import json
from typing import Any, Literal, cast

from .affine_monoid import AffineActionAtom, RingSpec, VectorStateSchema
from .identity import canonical_json_bytes, require_sha256

MODULAR_AFFINE_OBSERVATION_SCHEMA = "immer-ooe-modular-affine-observation/v1"
MODULAR_AFFINE_FIT_RECEIPT_SCHEMA = "immer-ooe-modular-affine-fit-receipt/v1"
MODULAR_AFFINE_FIT_ALGORITHM = "ExactModularAffineV1"
MODULAR_AFFINE_FIT_ALGORITHM_SHA256 = hashlib.sha256(
    canonical_json_bytes(
        {
            "equation": "y=A*x+b(mod-p)",
            "field": "explicit-prime-modulus",
            "row_order": "observation-sha256-ascending",
            "solver": "exact-deterministic-gauss-jordan/v1",
            "verification": "all-train-and-held-out-calibration-rows",
        }
    )
).hexdigest()

MAX_MODULUS = (1 << 63) - 1
MAX_FIT_DIMENSION = 511
MAX_OBSERVATIONS_PER_SPLIT = 65_536
MAX_OBSERVATION_BYTES = 1024 * 1024
MAX_FIT_RECEIPT_BYTES = 64 * 1024 * 1024
MAX_ELIMINATION_WORK = 1_000_000_000

ObservationSplit = Literal["train", "calibration"]


class ModularAffineFitError(ValueError):
    """Base error for an invalid exact modular-affine fit request."""


class ModularAffineIntegrityError(ModularAffineFitError):
    """An observation, identity, action, or receipt binding was modified."""


class ModularAffineIdentificationError(ModularAffineFitError):
    """Training evidence does not identify one exact affine operator."""


class ModularAffineCalibrationError(ModularAffineFitError):
    """The identified operator fails a held-out calibration observation."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _uint(
    value: object,
    *,
    field: str,
    minimum: int = 0,
    maximum: int = (1 << 63) - 1,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ModularAffineFitError(
            f"{field} must be an integer in [{minimum}, {maximum}]"
        )
    return value


def _is_prime(value: int) -> bool:
    """Deterministic primality test for the complete unsigned 64-bit range."""

    if value < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    if value in small_primes:
        return True
    if any(value % prime == 0 for prime in small_primes):
        return False

    odd = value - 1
    power = 0
    while odd % 2 == 0:
        power += 1
        odd //= 2
    # This base set is deterministic for n < 2**64.
    for raw_base in (2, 325, 9_375, 28_178, 450_775, 9_780_504, 1_795_265_022):
        base = raw_base % value
        if base == 0:
            continue
        witness = pow(base, odd, value)
        if witness in (1, value - 1):
            continue
        for _ in range(power - 1):
            witness = (witness * witness) % value
            if witness == value - 1:
                break
        else:
            return False
    return True


def _prime_modulus(value: object) -> int:
    modulus = _uint(value, field="modulus", minimum=2, maximum=MAX_MODULUS)
    if not _is_prime(modulus):
        raise ModularAffineFitError("modulus must be prime")
    return modulus


def _coordinate_vector(
    value: object,
    *,
    field: str,
    modulus: int,
    dimension: int | None = None,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ModularAffineFitError(f"{field} must be a bounded integer sequence")
    result = tuple(value)
    if not 1 <= len(result) <= MAX_FIT_DIMENSION:
        raise ModularAffineFitError(f"{field} has an invalid dimension")
    if dimension is not None and len(result) != dimension:
        raise ModularAffineFitError(f"{field} has the wrong dimension")
    exact: list[int] = []
    for index, item in enumerate(result):
        exact.append(
            _uint(
                item,
                field=f"{field}[{index}]",
                maximum=modulus - 1,
            )
        )
    return tuple(exact)


def _hash_sequence(
    values: Sequence[str],
    *,
    field: str,
    minimum: int,
    maximum: int,
    sorted_unique: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ModularAffineIntegrityError(f"{field} must be a bounded sequence")
    try:
        result = tuple(require_sha256(item, field=field) for item in values)
    except (TypeError, ValueError) as exc:
        raise ModularAffineIntegrityError(
            f"{field} contains an invalid SHA-256"
        ) from exc
    if not minimum <= len(result) <= maximum:
        raise ModularAffineIntegrityError(
            f"{field} must contain {minimum}..{maximum} entries"
        )
    if len(set(result)) != len(result):
        raise ModularAffineIntegrityError(f"{field} must be unique")
    if sorted_unique and result != tuple(sorted(result)):
        raise ModularAffineIntegrityError(f"{field} must be sorted")
    return result


def _sealed(schema: str, body: Mapping[str, Any]) -> dict[str, Any]:
    normalized = cast(dict[str, Any], json.loads(canonical_json_bytes(dict(body))))
    return {"body": normalized, "body_sha256": _digest(normalized), "schema": schema}


def _strict_body(
    data: bytes,
    *,
    schema: str,
    fields: frozenset[str],
    label: str,
    maximum: int,
) -> dict[str, Any]:
    if not isinstance(data, bytes):
        raise TypeError(f"{label} must be immutable bytes")
    if not data or len(data) > maximum:
        raise ModularAffineIntegrityError(f"{label} exceeds its byte bound")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        document = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ModularAffineIntegrityError(f"{label} is not strict JSON") from exc
    if canonical_json_bytes(document) != data:
        raise ModularAffineIntegrityError(f"{label} is not canonical JSON")
    if (
        not isinstance(document, Mapping)
        or set(document) != {"body", "body_sha256", "schema"}
        or document.get("schema") != schema
        or not isinstance(document.get("body"), Mapping)
    ):
        raise ModularAffineIntegrityError(f"{label} envelope is invalid")
    body = cast(Mapping[str, Any], document["body"])
    try:
        claimed = require_sha256(
            cast(str, document["body_sha256"]), field=f"{label}.body_sha256"
        )
    except (TypeError, ValueError) as exc:
        raise ModularAffineIntegrityError(f"{label} seal is invalid") from exc
    if claimed != _digest(body):
        raise ModularAffineIntegrityError(f"{label} SHA-256 mismatch")
    if set(body) != fields:
        raise ModularAffineIntegrityError(f"{label} has unknown or missing fields")
    return cast(dict[str, Any], json.loads(canonical_json_bytes(body)))


@dataclass(frozen=True, slots=True)
class ModularAffineObservation:
    """One content-addressed live-state transition used by the exact fitter.

    ``input_values`` and ``output_values`` omit the schema's dead coordinate.
    The source receipt, execution-contract identity, schema, split, temporal
    position, modulus, and both vectors all contribute to :attr:`sha256`.
    """

    split: ObservationSplit | str
    temporal_index: int
    input_values: tuple[int, ...]
    output_values: tuple[int, ...]
    modulus: int
    schema_sha256: str
    execution_identity_sha256: str
    source_receipt_sha256: str

    def __post_init__(self) -> None:
        if self.split not in ("train", "calibration"):
            raise ModularAffineFitError("split must be train or calibration")
        object.__setattr__(
            self, "temporal_index", _uint(self.temporal_index, field="temporal_index")
        )
        modulus = _prime_modulus(self.modulus)
        object.__setattr__(self, "modulus", modulus)
        inputs = _coordinate_vector(
            self.input_values, field="input_values", modulus=modulus
        )
        outputs = _coordinate_vector(
            self.output_values,
            field="output_values",
            modulus=modulus,
            dimension=len(inputs),
        )
        object.__setattr__(self, "input_values", inputs)
        object.__setattr__(self, "output_values", outputs)
        for field in (
            "schema_sha256",
            "execution_identity_sha256",
            "source_receipt_sha256",
        ):
            try:
                value = require_sha256(getattr(self, field), field=field)
            except (TypeError, ValueError) as exc:
                raise ModularAffineIntegrityError(f"{field} is invalid") from exc
            object.__setattr__(self, field, value)

    @property
    def dimension(self) -> int:
        return len(self.input_values)

    def to_record(self) -> dict[str, object]:
        return {
            "execution_identity_sha256": self.execution_identity_sha256,
            "input_values": list(self.input_values),
            "modulus": self.modulus,
            "output_values": list(self.output_values),
            "schema_sha256": self.schema_sha256,
            "source_receipt_sha256": self.source_receipt_sha256,
            "split": self.split,
            "temporal_index": self.temporal_index,
        }

    @property
    def sha256(self) -> str:
        return _digest(
            {"schema": MODULAR_AFFINE_OBSERVATION_SCHEMA, **self.to_record()}
        )

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(
            _sealed(MODULAR_AFFINE_OBSERVATION_SCHEMA, self.to_record())
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> ModularAffineObservation:
        body = _strict_body(
            data,
            schema=MODULAR_AFFINE_OBSERVATION_SCHEMA,
            fields=frozenset(
                {
                    "execution_identity_sha256",
                    "input_values",
                    "modulus",
                    "output_values",
                    "schema_sha256",
                    "source_receipt_sha256",
                    "split",
                    "temporal_index",
                }
            ),
            label="modular-affine observation",
            maximum=MAX_OBSERVATION_BYTES,
        )
        return cls(
            split=body["split"],
            temporal_index=body["temporal_index"],
            input_values=tuple(body["input_values"]),
            output_values=tuple(body["output_values"]),
            modulus=body["modulus"],
            schema_sha256=body["schema_sha256"],
            execution_identity_sha256=body["execution_identity_sha256"],
            source_receipt_sha256=body["source_receipt_sha256"],
        )


def _validate_schema(
    schema: VectorStateSchema, *, modulus: int, dimension: int
) -> tuple[int, ...]:
    if not isinstance(schema, VectorStateSchema):
        raise TypeError("schema must be a VectorStateSchema")
    if schema.dimension != dimension + 1:
        raise ModularAffineFitError(
            "observation dimension must equal schema dimension minus its dead marker"
        )
    active = tuple(
        index for index in range(schema.dimension) if index != schema.dead_index
    )
    expected = RingSpec("modular", modulus)
    if any(schema.rings[index] != expected for index in active):
        raise ModularAffineFitError(
            "every learned schema coordinate must use the explicit modular RingSpec"
        )
    return active


def _live_state_values(
    schema: VectorStateSchema, active_values: Sequence[int], active: Sequence[int]
) -> tuple[int, ...]:
    full = [0] * schema.dimension
    for index, value in zip(active, active_values, strict=True):
        full[index] = value
    full[schema.dead_index] = 0
    try:
        state = schema.state(tuple(full))
        if schema.is_dead(state):
            raise ModularAffineFitError("observation describes a dead state")
    except ModularAffineFitError:
        raise
    except ValueError as exc:
        raise ModularAffineFitError(
            "observation violates the declared affine state schema"
        ) from exc
    return state.values


def _canonical_observations(
    values: Sequence[ModularAffineObservation],
    *,
    split: ObservationSplit,
    schema: VectorStateSchema,
    modulus: int,
    execution_identity_sha256: str,
    dimension: int,
    active: Sequence[int],
) -> tuple[ModularAffineObservation, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ModularAffineFitError(f"{split} observations must be a bounded sequence")
    try:
        observations = tuple(values)
    except TypeError as exc:
        raise ModularAffineFitError(
            f"{split} observations must be a bounded sequence"
        ) from exc
    minimum = dimension + 1 if split == "train" else 1
    if not minimum <= len(observations) <= MAX_OBSERVATIONS_PER_SPLIT:
        raise ModularAffineIdentificationError(
            f"{split} observations must contain {minimum}.."
            f"{MAX_OBSERVATIONS_PER_SPLIT} rows"
        )
    for row in observations:
        if not isinstance(row, ModularAffineObservation):
            raise TypeError("every observation must be ModularAffineObservation")
        if row.split != split:
            raise ModularAffineIntegrityError(
                f"{split} observation carries the wrong split binding"
            )
        if row.modulus != modulus:
            raise ModularAffineIntegrityError("observation modulus identity mismatch")
        if row.schema_sha256 != schema.sha256:
            raise ModularAffineIntegrityError("observation schema identity mismatch")
        if row.execution_identity_sha256 != execution_identity_sha256:
            raise ModularAffineIntegrityError(
                "observation execution-contract identity mismatch"
            )
        if row.dimension != dimension:
            raise ModularAffineIntegrityError("observation dimension mismatch")
        _live_state_values(schema, row.input_values, active)
        _live_state_values(schema, row.output_values, active)
    ordered = tuple(sorted(observations, key=lambda row: row.sha256))
    if len({row.sha256 for row in ordered}) != len(ordered):
        raise ModularAffineIntegrityError(f"{split} observations must be unique")
    return ordered


def _solve_exact(
    observations: Sequence[ModularAffineObservation], *, modulus: int
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
    tuple[str, ...],
    int,
]:
    """Solve a full-column-rank modular affine system by exact RREF."""

    dimension = observations[0].dimension
    parameters = dimension + 1
    estimated_work = len(observations) * parameters * (parameters + dimension)
    if estimated_work > MAX_ELIMINATION_WORK:
        raise ModularAffineFitError("exact elimination exceeds its work bound")

    rows: list[list[int]] = []
    row_sha256s: list[str] = []
    for observation in observations:
        rows.append([*observation.input_values, 1, *observation.output_values])
        row_sha256s.append(observation.sha256)

    pivot_row = 0
    pivot_sha256s: list[str] = []
    for column in range(parameters):
        selected = next(
            (
                row
                for row in range(pivot_row, len(rows))
                if rows[row][column] % modulus != 0
            ),
            None,
        )
        if selected is None:
            break
        if selected != pivot_row:
            rows[pivot_row], rows[selected] = rows[selected], rows[pivot_row]
            row_sha256s[pivot_row], row_sha256s[selected] = (
                row_sha256s[selected],
                row_sha256s[pivot_row],
            )
        inverse = pow(rows[pivot_row][column] % modulus, -1, modulus)
        rows[pivot_row] = [(value * inverse) % modulus for value in rows[pivot_row]]
        for row_index in range(len(rows)):
            if row_index == pivot_row:
                continue
            factor = rows[row_index][column] % modulus
            if factor:
                rows[row_index] = [
                    (left - factor * right) % modulus
                    for left, right in zip(
                        rows[row_index], rows[pivot_row], strict=True
                    )
                ]
        pivot_sha256s.append(row_sha256s[pivot_row])
        pivot_row += 1

    rank = pivot_row
    if rank != parameters:
        raise ModularAffineIdentificationError(
            f"training design rank {rank} does not identify {parameters} affine parameters"
        )
    for row in rows[rank:]:
        if not any(row[:parameters]) and any(row[parameters:]):
            raise ModularAffineIdentificationError(
                "training observations are inconsistent over the prime field"
            )

    coefficients = tuple(
        tuple(rows[parameter][parameters + output] for output in range(dimension))
        for parameter in range(parameters)
    )
    matrix = tuple(
        tuple(
            coefficients[input_index][output_index] for input_index in range(dimension)
        )
        for output_index in range(dimension)
    )
    bias = tuple(coefficients[dimension][output] for output in range(dimension))
    return matrix, bias, tuple(pivot_sha256s), rank


def _predict(
    matrix: Sequence[Sequence[int]],
    bias: Sequence[int],
    inputs: Sequence[int],
    *,
    modulus: int,
) -> tuple[int, ...]:
    return tuple(
        (
            offset
            + sum(
                coefficient * value
                for coefficient, value in zip(row, inputs, strict=True)
            )
        )
        % modulus
        for row, offset in zip(matrix, bias, strict=True)
    )


def _embed_action(
    schema: VectorStateSchema,
    *,
    active: Sequence[int],
    name: str,
    matrix: Sequence[Sequence[int]],
    bias: Sequence[int],
    execution_identity_sha256: str,
    calibration_verifier_sha256: str,
) -> AffineActionAtom:
    full_matrix = [[0] * schema.dimension for _ in range(schema.dimension)]
    full_bias = [0] * schema.dimension
    for output_index, schema_output in enumerate(active):
        for input_index, schema_input in enumerate(active):
            full_matrix[schema_output][schema_input] = matrix[output_index][input_index]
        full_bias[schema_output] = bias[output_index]
    full_matrix[schema.dead_index][schema.dead_index] = 1
    return AffineActionAtom.create(
        schema,
        name=name,
        matrix=full_matrix,
        bias=full_bias,
        work_units=len(active) * (len(active) + 1),
        metadata={
            "fit_algorithm": MODULAR_AFFINE_FIT_ALGORITHM,
            "fit_calibration": f"sha256-{calibration_verifier_sha256}",
            "fit_identity": f"sha256-{execution_identity_sha256}",
        },
    )


@dataclass(frozen=True, slots=True)
class ModularAffineFitReceipt:
    """Sealed evidence and exact existing-action payload for one accepted fit."""

    modulus: int
    dimension: int
    schema_sha256: str
    execution_identity_sha256: str
    calibration_verifier_sha256: str
    algorithm_sha256: str
    train_observation_sha256s: tuple[str, ...]
    train_source_receipt_sha256s: tuple[str, ...]
    calibration_observation_sha256s: tuple[str, ...]
    calibration_source_receipt_sha256s: tuple[str, ...]
    pivot_observation_sha256s: tuple[str, ...]
    rank: int
    action_sha256: str
    action_bytes_sha256: str
    action_bytes: bytes

    def __post_init__(self) -> None:
        modulus = _prime_modulus(self.modulus)
        object.__setattr__(self, "modulus", modulus)
        dimension = _uint(
            self.dimension,
            field="dimension",
            minimum=1,
            maximum=MAX_FIT_DIMENSION,
        )
        object.__setattr__(self, "dimension", dimension)
        for field in (
            "schema_sha256",
            "execution_identity_sha256",
            "calibration_verifier_sha256",
            "algorithm_sha256",
            "action_sha256",
            "action_bytes_sha256",
        ):
            try:
                value = require_sha256(getattr(self, field), field=field)
            except (TypeError, ValueError) as exc:
                raise ModularAffineIntegrityError(f"{field} is invalid") from exc
            object.__setattr__(self, field, value)
        if self.algorithm_sha256 != MODULAR_AFFINE_FIT_ALGORITHM_SHA256:
            raise ModularAffineIntegrityError("fit algorithm identity mismatch")

        train = _hash_sequence(
            self.train_observation_sha256s,
            field="train_observation_sha256s",
            minimum=dimension + 1,
            maximum=MAX_OBSERVATIONS_PER_SPLIT,
            sorted_unique=True,
        )
        train_sources = _hash_sequence(
            self.train_source_receipt_sha256s,
            field="train_source_receipt_sha256s",
            minimum=len(train),
            maximum=len(train),
        )
        calibration = _hash_sequence(
            self.calibration_observation_sha256s,
            field="calibration_observation_sha256s",
            minimum=1,
            maximum=MAX_OBSERVATIONS_PER_SPLIT,
            sorted_unique=True,
        )
        calibration_sources = _hash_sequence(
            self.calibration_source_receipt_sha256s,
            field="calibration_source_receipt_sha256s",
            minimum=len(calibration),
            maximum=len(calibration),
        )
        pivots = _hash_sequence(
            self.pivot_observation_sha256s,
            field="pivot_observation_sha256s",
            minimum=dimension + 1,
            maximum=dimension + 1,
        )
        if not set(pivots) <= set(train):
            raise ModularAffineIntegrityError(
                "pivot rows are not training observations"
            )
        if set(train) & set(calibration):
            raise ModularAffineIntegrityError(
                "train and calibration observations overlap"
            )
        if set(train_sources) & set(calibration_sources):
            raise ModularAffineIntegrityError(
                "train and calibration source receipts overlap"
            )
        object.__setattr__(self, "train_observation_sha256s", train)
        object.__setattr__(self, "train_source_receipt_sha256s", train_sources)
        object.__setattr__(self, "calibration_observation_sha256s", calibration)
        object.__setattr__(
            self, "calibration_source_receipt_sha256s", calibration_sources
        )
        object.__setattr__(self, "pivot_observation_sha256s", pivots)
        rank = _uint(self.rank, field="rank", minimum=1, maximum=dimension + 1)
        if rank != dimension + 1:
            raise ModularAffineIntegrityError("fit receipt is not fully identified")
        object.__setattr__(self, "rank", rank)
        if not isinstance(self.action_bytes, bytes):
            raise TypeError("action_bytes must be immutable bytes")
        if not self.action_bytes or len(self.action_bytes) > MAX_FIT_RECEIPT_BYTES:
            raise ModularAffineIntegrityError("action payload exceeds its byte bound")
        if hashlib.sha256(self.action_bytes).hexdigest() != self.action_bytes_sha256:
            raise ModularAffineIntegrityError("action payload SHA-256 mismatch")

    @property
    def ring_spec(self) -> RingSpec:
        return RingSpec("modular", self.modulus)

    def to_record(self) -> dict[str, object]:
        return {
            "action_base64": base64.b64encode(self.action_bytes).decode("ascii"),
            "action_bytes_sha256": self.action_bytes_sha256,
            "action_sha256": self.action_sha256,
            "algorithm_sha256": self.algorithm_sha256,
            "calibration_observation_sha256s": list(
                self.calibration_observation_sha256s
            ),
            "calibration_source_receipt_sha256s": list(
                self.calibration_source_receipt_sha256s
            ),
            "calibration_verifier_sha256": self.calibration_verifier_sha256,
            "dimension": self.dimension,
            "execution_identity_sha256": self.execution_identity_sha256,
            "modulus": self.modulus,
            "pivot_observation_sha256s": list(self.pivot_observation_sha256s),
            "rank": self.rank,
            "schema_sha256": self.schema_sha256,
            "train_observation_sha256s": list(self.train_observation_sha256s),
            "train_source_receipt_sha256s": list(self.train_source_receipt_sha256s),
        }

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(
            _sealed(MODULAR_AFFINE_FIT_RECEIPT_SCHEMA, self.to_record())
        )
        if len(data) > MAX_FIT_RECEIPT_BYTES:
            raise ModularAffineIntegrityError("fit receipt exceeds its byte bound")
        return data

    def restore_action(self, schema: VectorStateSchema) -> AffineActionAtom:
        active = _validate_schema(
            schema, modulus=self.modulus, dimension=self.dimension
        )
        del active
        if schema.sha256 != self.schema_sha256:
            raise ModularAffineIntegrityError("fit receipt schema identity mismatch")
        try:
            action = AffineActionAtom.from_bytes(self.action_bytes, schema=schema)
        except ValueError as exc:
            raise ModularAffineIntegrityError(
                "fit receipt action payload is invalid"
            ) from exc
        if action.sha256 != self.action_sha256:
            raise ModularAffineIntegrityError("fit receipt action identity mismatch")
        metadata = dict(action.metadata)
        expected_metadata = {
            "fit_algorithm": MODULAR_AFFINE_FIT_ALGORITHM,
            "fit_calibration": f"sha256-{self.calibration_verifier_sha256}",
            "fit_identity": f"sha256-{self.execution_identity_sha256}",
        }
        if metadata != expected_metadata:
            raise ModularAffineIntegrityError("fit action metadata identity mismatch")
        return action

    @classmethod
    def from_bytes(
        cls, data: bytes, *, schema: VectorStateSchema
    ) -> ModularAffineFitReceipt:
        body = _strict_body(
            data,
            schema=MODULAR_AFFINE_FIT_RECEIPT_SCHEMA,
            fields=frozenset(
                {
                    "action_base64",
                    "action_bytes_sha256",
                    "action_sha256",
                    "algorithm_sha256",
                    "calibration_observation_sha256s",
                    "calibration_source_receipt_sha256s",
                    "calibration_verifier_sha256",
                    "dimension",
                    "execution_identity_sha256",
                    "modulus",
                    "pivot_observation_sha256s",
                    "rank",
                    "schema_sha256",
                    "train_observation_sha256s",
                    "train_source_receipt_sha256s",
                }
            ),
            label="modular-affine fit receipt",
            maximum=MAX_FIT_RECEIPT_BYTES,
        )
        encoded = body["action_base64"]
        if not isinstance(encoded, str) or not encoded.isascii():
            raise ModularAffineIntegrityError("action payload encoding is invalid")
        try:
            action_bytes = base64.b64decode(encoded, validate=True)
        except (TypeError, ValueError) as exc:
            raise ModularAffineIntegrityError(
                "action payload encoding is invalid"
            ) from exc
        if base64.b64encode(action_bytes).decode("ascii") != encoded:
            raise ModularAffineIntegrityError(
                "action payload encoding is not canonical"
            )
        receipt = cls(
            modulus=body["modulus"],
            dimension=body["dimension"],
            schema_sha256=body["schema_sha256"],
            execution_identity_sha256=body["execution_identity_sha256"],
            calibration_verifier_sha256=body["calibration_verifier_sha256"],
            algorithm_sha256=body["algorithm_sha256"],
            train_observation_sha256s=tuple(body["train_observation_sha256s"]),
            train_source_receipt_sha256s=tuple(body["train_source_receipt_sha256s"]),
            calibration_observation_sha256s=tuple(
                body["calibration_observation_sha256s"]
            ),
            calibration_source_receipt_sha256s=tuple(
                body["calibration_source_receipt_sha256s"]
            ),
            pivot_observation_sha256s=tuple(body["pivot_observation_sha256s"]),
            rank=body["rank"],
            action_sha256=body["action_sha256"],
            action_bytes_sha256=body["action_bytes_sha256"],
            action_bytes=action_bytes,
        )
        receipt.restore_action(schema)
        return receipt


@dataclass(frozen=True, slots=True)
class ModularAffineFit:
    """Accepted exact fit: existing action plus its sealed evidence receipt."""

    action: AffineActionAtom
    receipt: ModularAffineFitReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.action, AffineActionAtom):
            raise TypeError("action must be an AffineActionAtom")
        if not isinstance(self.receipt, ModularAffineFitReceipt):
            raise TypeError("receipt must be a ModularAffineFitReceipt")
        if self.action.sha256 != self.receipt.action_sha256:
            raise ModularAffineIntegrityError("fit action and receipt disagree")

    @property
    def ring_spec(self) -> RingSpec:
        return self.receipt.ring_spec


def _validated_inputs(
    schema: VectorStateSchema,
    *,
    modulus: int,
    execution_identity_sha256: str,
    train_observations: Sequence[ModularAffineObservation],
    calibration_observations: Sequence[ModularAffineObservation],
) -> tuple[
    tuple[int, ...],
    tuple[ModularAffineObservation, ...],
    tuple[ModularAffineObservation, ...],
]:
    modulus = _prime_modulus(modulus)
    try:
        identity = require_sha256(
            execution_identity_sha256, field="execution_identity_sha256"
        )
    except (TypeError, ValueError) as exc:
        raise ModularAffineIntegrityError(
            "execution_identity_sha256 is invalid"
        ) from exc
    if not train_observations:
        raise ModularAffineIdentificationError("training observations are empty")
    first = train_observations[0]
    if not isinstance(first, ModularAffineObservation):
        raise TypeError("every observation must be ModularAffineObservation")
    dimension = first.dimension
    active = _validate_schema(schema, modulus=modulus, dimension=dimension)
    train = _canonical_observations(
        train_observations,
        split="train",
        schema=schema,
        modulus=modulus,
        execution_identity_sha256=identity,
        dimension=dimension,
        active=active,
    )
    calibration = _canonical_observations(
        calibration_observations,
        split="calibration",
        schema=schema,
        modulus=modulus,
        execution_identity_sha256=identity,
        dimension=dimension,
        active=active,
    )
    if {row.sha256 for row in train} & {row.sha256 for row in calibration}:
        raise ModularAffineIntegrityError("train and calibration observations overlap")
    if {row.source_receipt_sha256 for row in train} & {
        row.source_receipt_sha256 for row in calibration
    }:
        raise ModularAffineIntegrityError(
            "train and calibration source receipts overlap"
        )
    return active, train, calibration


def fit_modular_affine_action(
    schema: VectorStateSchema,
    *,
    name: str,
    modulus: int,
    execution_identity_sha256: str,
    calibration_verifier_sha256: str,
    train_observations: Sequence[ModularAffineObservation],
    calibration_observations: Sequence[ModularAffineObservation],
) -> ModularAffineFit:
    """Identify, verify, and seal one exact affine action over ``GF(modulus)``."""

    prime = _prime_modulus(modulus)
    try:
        identity = require_sha256(
            execution_identity_sha256, field="execution_identity_sha256"
        )
        verifier = require_sha256(
            calibration_verifier_sha256, field="calibration_verifier_sha256"
        )
    except (TypeError, ValueError) as exc:
        raise ModularAffineIntegrityError("fit identity SHA-256 is invalid") from exc
    active, train, calibration = _validated_inputs(
        schema,
        modulus=prime,
        execution_identity_sha256=identity,
        train_observations=train_observations,
        calibration_observations=calibration_observations,
    )
    matrix, bias, pivots, rank = _solve_exact(train, modulus=prime)
    for row in train:
        if _predict(matrix, bias, row.input_values, modulus=prime) != row.output_values:
            raise ModularAffineIdentificationError(
                "training observations do not define one exact affine operator"
            )
    for row in calibration:
        if _predict(matrix, bias, row.input_values, modulus=prime) != row.output_values:
            raise ModularAffineCalibrationError(
                "identified action fails held-out calibration"
            )
    action = _embed_action(
        schema,
        active=active,
        name=name,
        matrix=matrix,
        bias=bias,
        execution_identity_sha256=identity,
        calibration_verifier_sha256=verifier,
    )
    action_bytes = action.to_bytes()
    receipt = ModularAffineFitReceipt(
        modulus=prime,
        dimension=len(active),
        schema_sha256=schema.sha256,
        execution_identity_sha256=identity,
        calibration_verifier_sha256=verifier,
        algorithm_sha256=MODULAR_AFFINE_FIT_ALGORITHM_SHA256,
        train_observation_sha256s=tuple(row.sha256 for row in train),
        train_source_receipt_sha256s=tuple(row.source_receipt_sha256 for row in train),
        calibration_observation_sha256s=tuple(row.sha256 for row in calibration),
        calibration_source_receipt_sha256s=tuple(
            row.source_receipt_sha256 for row in calibration
        ),
        pivot_observation_sha256s=pivots,
        rank=rank,
        action_sha256=action.sha256,
        action_bytes_sha256=hashlib.sha256(action_bytes).hexdigest(),
        action_bytes=action_bytes,
    )
    receipt.restore_action(schema)
    return ModularAffineFit(action=action, receipt=receipt)


def verify_modular_affine_fit(
    receipt: ModularAffineFitReceipt,
    schema: VectorStateSchema,
    *,
    train_observations: Sequence[ModularAffineObservation],
    calibration_observations: Sequence[ModularAffineObservation],
) -> str:
    """Replay identification and calibration, returning the accepted receipt SHA."""

    if not isinstance(receipt, ModularAffineFitReceipt):
        raise TypeError("receipt must be a ModularAffineFitReceipt")
    action = receipt.restore_action(schema)
    active, train, calibration = _validated_inputs(
        schema,
        modulus=receipt.modulus,
        execution_identity_sha256=receipt.execution_identity_sha256,
        train_observations=train_observations,
        calibration_observations=calibration_observations,
    )
    if tuple(row.sha256 for row in train) != receipt.train_observation_sha256s:
        raise ModularAffineIntegrityError("training observation binding mismatch")
    if tuple(row.source_receipt_sha256 for row in train) != (
        receipt.train_source_receipt_sha256s
    ):
        raise ModularAffineIntegrityError("training source-receipt binding mismatch")
    if tuple(row.sha256 for row in calibration) != (
        receipt.calibration_observation_sha256s
    ):
        raise ModularAffineIntegrityError("calibration observation binding mismatch")
    if tuple(row.source_receipt_sha256 for row in calibration) != (
        receipt.calibration_source_receipt_sha256s
    ):
        raise ModularAffineIntegrityError("calibration source-receipt binding mismatch")

    matrix, bias, pivots, rank = _solve_exact(train, modulus=receipt.modulus)
    if pivots != receipt.pivot_observation_sha256s or rank != receipt.rank:
        raise ModularAffineIntegrityError("deterministic identification trace mismatch")
    expected = _embed_action(
        schema,
        active=active,
        name=action.name,
        matrix=matrix,
        bias=bias,
        execution_identity_sha256=receipt.execution_identity_sha256,
        calibration_verifier_sha256=receipt.calibration_verifier_sha256,
    )
    if expected.to_bytes() != receipt.action_bytes or expected.sha256 != action.sha256:
        raise ModularAffineIntegrityError("replayed action payload mismatch")
    for row in train:
        if (
            _predict(matrix, bias, row.input_values, modulus=receipt.modulus)
            != row.output_values
        ):
            raise ModularAffineIdentificationError(
                "training observations no longer verify exactly"
            )
    for row in calibration:
        if (
            _predict(matrix, bias, row.input_values, modulus=receipt.modulus)
            != row.output_values
        ):
            raise ModularAffineCalibrationError(
                "held-out calibration no longer verifies exactly"
            )
    return receipt.sha256


__all__ = [
    "MAX_FIT_DIMENSION",
    "MAX_OBSERVATIONS_PER_SPLIT",
    "MODULAR_AFFINE_FIT_ALGORITHM_SHA256",
    "MODULAR_AFFINE_FIT_RECEIPT_SCHEMA",
    "MODULAR_AFFINE_OBSERVATION_SCHEMA",
    "ModularAffineCalibrationError",
    "ModularAffineFit",
    "ModularAffineFitError",
    "ModularAffineFitReceipt",
    "ModularAffineIdentificationError",
    "ModularAffineIntegrityError",
    "ModularAffineObservation",
    "fit_modular_affine_action",
    "verify_modular_affine_fit",
]
