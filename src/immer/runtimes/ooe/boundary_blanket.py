"""Receipt-bound exhaustive Markov-blanket discovery at Qwen layer boundaries.

The cartography runtime records five transitions for every prompt/layer macro.
Together those transitions expose six named boundary variables and the exact
layer output.  This module reconstructs that macro atomically, fits every one
of the 63 non-empty boundary subsets plus the fixed decoder residual identity
on an earlier training split, selects on a separate calibration split, and
evaluates a later holdout without refitting.

Token rows are useful numerical samples, but they are not independent pieces
of evidence.  Both fitting and scoring therefore give every prompt/layer macro
unit weight, irrespective of its token count.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import re
from typing import cast

import numpy as np
from numpy.typing import NDArray

from .identity import canonical_json_bytes, require_sha256
from .operator_harvester import (
    ContextualOperatorObservation,
    HarvesterState,
    OperatorHarvesterIntegrityError,
)


BOUNDARY_CANDIDATES = (
    "layer_input",
    "attention_input",
    "attention_output",
    "attention_residual",
    "mlp_input",
    "mlp_output",
)
BOUNDARY_TARGET = "layer_output"
BOUNDARY_SEMANTIC_SCHEMA = "immer-ooe-qwen-boundary-candidate-schema/v1"
BOUNDARY_CORPUS_SCHEMA = "immer-ooe-qwen-boundary-corpus/v1"
BOUNDARY_FIT_SCHEMA = "immer-ooe-qwen-boundary-blanket-fit/v4"
BOUNDARY_HOLDOUT_SCHEMA = "immer-ooe-qwen-boundary-blanket-holdout/v4"
BOUNDARY_LOQO_SCHEMA = "immer-ooe-qwen-boundary-blanket-loqo/v4"
BOUNDARY_NUMERIC_ABI = "float64-le-centered-svd-ridge-macro-weighted-nrmse/v2"
BOUNDARY_SELECTION_ALGORITHM = (
    "exhaustive-63/train-macro-weighted-ridge/calibration-nrmse/"
    "plus-fixed-residual-sum/smallest-within-absolute-tolerance-"
    "then-score-family-lexicographic/v2"
)
DEFAULT_RIDGE_GRID = (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)
DEFAULT_SELECTION_TOLERANCE = 1e-10
DEFAULT_NORMALIZATION_FLOOR = 1e-12
DEFAULT_MAX_CONDITION_NUMBER = 1e12
DEFAULT_MAX_WORKING_BYTES = 1024**3
MAX_MACROS = 65_536
MAX_FEATURE_DIMENSION = 4096
MAX_OUTPUT_DIMENSION = 4096
MAX_RECEIPT_BYTES = 64 * 1024 * 1024

FloatArray = NDArray[np.float64]
_LAYER_STATE = re.compile(r"qwen\.layer\.([0-9]+)\.(.+)-sketch\Z")


class BoundaryBlanketError(RuntimeError):
    """Base error for boundary-blanket fitting and verification."""


class BoundaryBlanketIntegrityError(BoundaryBlanketError):
    """Persisted evidence or a sealed result failed exact validation."""


class BoundaryBlanketConditionError(BoundaryBlanketError):
    """No configured ridge produced a finite, bounded-condition fit."""


class BoundaryBlanketCapacityError(BoundaryBlanketError):
    """A bounded fit would exceed its declared working-memory limit."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_json(data: bytes, *, schema: str) -> Mapping[str, object]:
    if not isinstance(data, bytes):
        raise TypeError("sealed receipt must be immutable bytes")
    if not data or len(data) > MAX_RECEIPT_BYTES:
        raise BoundaryBlanketIntegrityError("sealed receipt exceeds its byte bound")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise BoundaryBlanketIntegrityError(
            "sealed receipt is not strict JSON"
        ) from exc
    if (
        not isinstance(value, Mapping)
        or set(value) != {"body", "body_sha256", "schema"}
        or value.get("schema") != schema
        or not isinstance(value.get("body"), Mapping)
        or value.get("body_sha256") != _digest(value.get("body"))
    ):
        raise BoundaryBlanketIntegrityError("sealed receipt envelope is invalid")
    if canonical_json_bytes(value) != data:
        raise BoundaryBlanketIntegrityError("sealed receipt is not canonical")
    return cast(Mapping[str, object], value)


def _finite(value: object, *, field: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite numeric data")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"{field} must be finite numeric data")
    return 0.0 if result == 0.0 else result


def _positive_int(value: object, *, field: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= maximum
    ):
        raise ValueError(f"{field} must lie in [1, {maximum}]")
    return value


def _canonical_array(value: object, *, field: str) -> FloatArray:
    if type(value) is not np.ndarray or value.dtype != np.dtype(np.float64):
        raise TypeError(f"{field} must be an exact float64 numpy.ndarray")
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError(f"{field} must be a non-empty matrix")
    if not bool(np.isfinite(value).all()):
        raise ValueError(f"{field} contains non-finite values")
    result = np.array(value, dtype="<f8", order="C", copy=True)
    result[result == 0.0] = 0.0
    result.flags.writeable = False
    return cast(FloatArray, result)


def _array_sha256(value: FloatArray) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _array_record(value: FloatArray) -> dict[str, object]:
    array = _canonical_array(value, field="array")
    return {
        "data_base64": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
        "dtype": "float64-le",
        "sha256": _array_sha256(array),
        "shape": [int(row) for row in array.shape],
    }


def _array_from_record(value: object, *, field: str) -> FloatArray:
    if not isinstance(value, Mapping) or set(value) != {
        "data_base64",
        "dtype",
        "sha256",
        "shape",
    }:
        raise BoundaryBlanketIntegrityError(f"{field} array record is invalid")
    if value.get("dtype") != "float64-le":
        raise BoundaryBlanketIntegrityError(f"{field} array dtype is invalid")
    shape = value.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(
            isinstance(row, bool) or not isinstance(row, int) or row < 1
            for row in shape
        )
    ):
        raise BoundaryBlanketIntegrityError(f"{field} array shape is invalid")
    encoded = value.get("data_base64")
    expected_bytes = int(shape[0]) * int(shape[1]) * 8
    if (
        not isinstance(encoded, str)
        or not encoded.isascii()
        or len(encoded) > 4 * ((expected_bytes + 2) // 3)
    ):
        raise BoundaryBlanketIntegrityError(f"{field} array encoding is invalid")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (TypeError, ValueError) as exc:
        raise BoundaryBlanketIntegrityError(
            f"{field} array encoding is invalid"
        ) from exc
    if len(raw) != expected_bytes:
        raise BoundaryBlanketIntegrityError(f"{field} array byte length is invalid")
    claimed = require_sha256(value.get("sha256"), field=f"{field}.sha256")
    if hashlib.sha256(raw).hexdigest() != claimed:
        raise BoundaryBlanketIntegrityError(f"{field} array hash mismatch")
    result = np.frombuffer(raw, dtype="<f8").reshape((int(shape[0]), int(shape[1])))
    return _canonical_array(result, field=field)


def _subset_inventory() -> tuple[tuple[str, ...], ...]:
    return tuple(
        subset
        for size in range(1, len(BOUNDARY_CANDIDATES) + 1)
        for subset in itertools.combinations(BOUNDARY_CANDIDATES, size)
    )


ALL_BOUNDARY_SUBSETS = _subset_inventory()
FITTED_AFFINE_FAMILY = "fitted_affine"
FIXED_RESIDUAL_SUM_FAMILY = "fixed_attention_residual_plus_mlp_output"
FIXED_RESIDUAL_SUBSET = ("attention_residual", "mlp_output")
ALL_BOUNDARY_MODEL_KEYS = tuple(
    (FITTED_AFFINE_FAMILY, subset) for subset in ALL_BOUNDARY_SUBSETS
) + ((FIXED_RESIDUAL_SUM_FAMILY, FIXED_RESIDUAL_SUBSET),)
BOUNDARY_MODEL_INVENTORY = {
    "fitted_affine_subset_count": len(ALL_BOUNDARY_SUBSETS),
    "structural_families": [FIXED_RESIDUAL_SUM_FAMILY],
    "total_model_count": len(ALL_BOUNDARY_MODEL_KEYS),
}
BOUNDARY_CANDIDATE_SCHEMA_SHA256 = _digest(
    {
        "candidates": list(BOUNDARY_CANDIDATES),
        "target": BOUNDARY_TARGET,
        "semantics": {
            "attention_input": "RMSNorm(layer_input)",
            "attention_output": "attention(attention_input)",
            "attention_residual": "layer_input + attention_output",
            "layer_input": "decoder layer input",
            "mlp_input": "RMSNorm(attention_residual)",
            "mlp_output": "MLP(mlp_input)",
        },
        "schema": BOUNDARY_SEMANTIC_SCHEMA,
        "target_semantics": "attention_residual + mlp_output",
    }
)


@dataclass(frozen=True, slots=True)
class BoundaryMacro:
    """One atomic prompt/layer observation cluster."""

    chronology: int
    layer: int
    prompt_sha256: str
    measurement_sha256: str
    model_pin_sha256: str
    atlas_revision_sha256: str
    atlas_sequence: int
    atlas_event_sha256: str
    weight_rail_revision_sha256: str
    transition_receipt_sha256s: tuple[str, ...]
    observation_record_sha256s: tuple[str, ...]
    candidates: tuple[FloatArray, ...]
    target: FloatArray

    def __post_init__(self) -> None:
        _positive_int(self.chronology, field="chronology", maximum=2**63 - 1)
        if (
            isinstance(self.layer, bool)
            or not isinstance(self.layer, int)
            or self.layer < 0
        ):
            raise ValueError("layer must be a non-negative integer")
        for field in (
            "prompt_sha256",
            "measurement_sha256",
            "model_pin_sha256",
            "atlas_revision_sha256",
            "atlas_event_sha256",
            "weight_rail_revision_sha256",
        ):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        if self.atlas_sequence != self.chronology:
            raise ValueError("macro chronology must equal its Atlas sequence")
        transitions = tuple(
            require_sha256(row, field="transition_receipt_sha256s")
            for row in self.transition_receipt_sha256s
        )
        observations = tuple(
            require_sha256(row, field="observation_record_sha256s")
            for row in self.observation_record_sha256s
        )
        if len(transitions) != 5 or len(set(transitions)) != 5:
            raise ValueError("a boundary macro requires five transition receipts")
        if len(observations) != 5 or len(set(observations)) != 5:
            raise ValueError("a boundary macro requires five observation receipts")
        arrays = tuple(
            _canonical_array(row, field=f"candidate[{index}]")
            for index, row in enumerate(self.candidates)
        )
        if len(arrays) != len(BOUNDARY_CANDIDATES):
            raise ValueError("boundary macro candidate inventory is incomplete")
        target = _canonical_array(self.target, field="target")
        if any(row.shape != target.shape for row in arrays):
            raise ValueError("boundary macro projected shapes disagree")
        if target.shape[1] > MAX_OUTPUT_DIMENSION:
            raise ValueError("boundary macro output dimension exceeds its bound")
        object.__setattr__(self, "transition_receipt_sha256s", transitions)
        object.__setattr__(self, "observation_record_sha256s", observations)
        object.__setattr__(self, "candidates", arrays)
        object.__setattr__(self, "target", target)

    @property
    def key(self) -> tuple[int, str, int]:
        return self.chronology, self.measurement_sha256, self.layer

    @property
    def feature_dimension(self) -> int:
        return int(self.target.shape[1])

    @property
    def token_rows(self) -> int:
        return int(self.target.shape[0])

    def candidate(self, name: str) -> FloatArray:
        try:
            index = BOUNDARY_CANDIDATES.index(name)
        except ValueError as exc:
            raise KeyError(name) from exc
        return self.candidates[index]

    def evidence_record(self) -> dict[str, object]:
        return {
            "atlas_event_sha256": self.atlas_event_sha256,
            "atlas_revision_sha256": self.atlas_revision_sha256,
            "atlas_sequence": self.atlas_sequence,
            "candidate_sha256s": {
                name: _array_sha256(value)
                for name, value in zip(
                    BOUNDARY_CANDIDATES, self.candidates, strict=True
                )
            },
            "chronology": self.chronology,
            "layer": self.layer,
            "measurement_sha256": self.measurement_sha256,
            "model_pin_sha256": self.model_pin_sha256,
            "observation_record_sha256s": list(self.observation_record_sha256s),
            "prompt_sha256": self.prompt_sha256,
            "target_sha256": _array_sha256(self.target),
            "token_rows": self.token_rows,
            "transition_receipt_sha256s": list(self.transition_receipt_sha256s),
            "weight_rail_revision_sha256": self.weight_rail_revision_sha256,
        }


def _state_stage(receipt_state: str) -> tuple[int, str]:
    match = _LAYER_STATE.fullmatch(receipt_state)
    if match is None:
        raise BoundaryBlanketIntegrityError(
            f"unknown Qwen boundary state: {receipt_state}"
        )
    return int(match.group(1)), match.group(2)


_EXPECTED_TRANSITIONS = (
    ("pre-hidden", "post-hidden"),
    ("attention.input", "attention.output"),
    ("layer.input", "attention.residual"),
    ("mlp.input", "mlp.output"),
    ("attention.residual", "layer.output"),
)


def _same_array(left: FloatArray, right: FloatArray) -> bool:
    return left.shape == right.shape and bool(np.array_equal(left, right))


def _macro_from_observations(
    observations: Sequence[ContextualOperatorObservation],
) -> BoundaryMacro:
    rows = tuple(observations)
    if len(rows) != 5:
        raise BoundaryBlanketIntegrityError(
            "a prompt/layer macro must contain exactly five transitions"
        )
    if any(not isinstance(row, ContextualOperatorObservation) for row in rows):
        raise TypeError("macro observations have the wrong type")
    measurement_sha = rows[0].receipt.measurement_sha256
    if any(row.receipt.measurement_sha256 != measurement_sha for row in rows):
        raise BoundaryBlanketIntegrityError("macro crosses MeasurementReceipts")
    by_transition: dict[tuple[str, str], ContextualOperatorObservation] = {}
    layer: int | None = None
    for row in rows:
        source_layer, source = _state_stage(row.receipt.source_state)
        target_layer, target = _state_stage(row.receipt.target_state)
        if source_layer != target_layer:
            raise BoundaryBlanketIntegrityError("boundary transition crosses layers")
        if layer is None:
            layer = source_layer
        elif layer != source_layer:
            raise BoundaryBlanketIntegrityError("macro crosses decoder layers")
        key = (source, target)
        if key not in _EXPECTED_TRANSITIONS or key in by_transition:
            raise BoundaryBlanketIntegrityError(
                "macro has an unknown or duplicate boundary transition"
            )
        by_transition[key] = row
    if set(by_transition) != set(_EXPECTED_TRANSITIONS) or layer is None:
        raise BoundaryBlanketIntegrityError("macro boundary inventory is incomplete")
    if any(row.measurement.coordinate.layer != layer for row in rows):
        raise BoundaryBlanketIntegrityError(
            "boundary layer differs from its measurement coordinate"
        )
    first = rows[0]
    common = (
        first.receipt.model_pin_sha256,
        first.receipt.atlas_revision.sha256,
        first.receipt.atlas_revision.sequence,
        first.receipt.atlas_revision.event_sha256,
        first.measurement.weight_rail_revision.sha256,
        first.measurement.probe.prompt_signature,
    )
    for row in rows[1:]:
        candidate = (
            row.receipt.model_pin_sha256,
            row.receipt.atlas_revision.sha256,
            row.receipt.atlas_revision.sequence,
            row.receipt.atlas_revision.event_sha256,
            row.measurement.weight_rail_revision.sha256,
            row.measurement.probe.prompt_signature,
        )
        if candidate != common:
            raise BoundaryBlanketIntegrityError(
                "macro provenance differs across boundary transitions"
            )
    whole = by_transition[("pre-hidden", "post-hidden")]
    attention = by_transition[("attention.input", "attention.output")]
    attention_residual = by_transition[("layer.input", "attention.residual")]
    mlp = by_transition[("mlp.input", "mlp.output")]
    output = by_transition[("attention.residual", "layer.output")]
    if (
        not _same_array(whole.input_array, attention_residual.input_array)
        or not _same_array(whole.output_array, output.output_array)
        or not _same_array(attention_residual.output_array, output.input_array)
    ):
        raise BoundaryBlanketIntegrityError(
            "duplicated boundary projections are not byte-identical"
        )
    candidates = (
        whole.input_array,
        attention.input_array,
        attention.output_array,
        attention_residual.output_array,
        mlp.input_array,
        mlp.output_array,
    )
    if any(row.shape != whole.output_array.shape for row in candidates):
        raise BoundaryBlanketIntegrityError("macro boundary projection shapes disagree")
    ordered_rows = tuple(by_transition[key] for key in _EXPECTED_TRANSITIONS)
    return BoundaryMacro(
        chronology=first.receipt.atlas_revision.sequence,
        layer=layer,
        prompt_sha256=first.measurement.probe.prompt_signature,
        measurement_sha256=measurement_sha,
        model_pin_sha256=first.receipt.model_pin_sha256,
        atlas_revision_sha256=first.receipt.atlas_revision.sha256,
        atlas_sequence=first.receipt.atlas_revision.sequence,
        atlas_event_sha256=first.receipt.atlas_revision.event_sha256,
        weight_rail_revision_sha256=first.measurement.weight_rail_revision.sha256,
        transition_receipt_sha256s=tuple(row.receipt.sha256 for row in ordered_rows),
        observation_record_sha256s=tuple(
            _digest(row.to_record()) for row in ordered_rows
        ),
        candidates=cast(tuple[FloatArray, ...], candidates),
        target=whole.output_array,
    )


@dataclass(frozen=True, slots=True)
class BoundaryCorpus:
    """Authenticated macro view reconstructed from one HarvesterState."""

    harvester_identity_sha256: str
    harvester_state_sha256: str
    model_pin_sha256: str
    macros: tuple[BoundaryMacro, ...]

    def __post_init__(self) -> None:
        for field in (
            "harvester_identity_sha256",
            "harvester_state_sha256",
            "model_pin_sha256",
        ):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        macros = tuple(self.macros)
        if not macros or len(macros) > MAX_MACROS:
            raise ValueError("boundary corpus macro inventory is invalid")
        if any(not isinstance(row, BoundaryMacro) for row in macros):
            raise TypeError("boundary corpus contains an invalid macro")
        if tuple(sorted(macros, key=lambda row: row.key)) != macros:
            raise ValueError("boundary macros must be in strict chronology")
        if len({row.chronology for row in macros}) != len(macros):
            raise ValueError("boundary macro chronology must be unique")
        if any(row.model_pin_sha256 != self.model_pin_sha256 for row in macros):
            raise BoundaryBlanketIntegrityError("boundary corpus crosses model pins")
        dimensions = {row.feature_dimension for row in macros}
        if len(dimensions) != 1:
            raise BoundaryBlanketIntegrityError(
                "boundary corpus crosses projection dimensions"
            )
        object.__setattr__(self, "macros", macros)

    @classmethod
    def from_harvester_state(
        cls,
        state: HarvesterState,
        *,
        expected_harvester_identity_sha256: str | None = None,
        expected_model_pin_sha256: str | None = None,
        expected_latest_atlas_revision_sha256: str | None = None,
    ) -> "BoundaryCorpus":
        if not isinstance(state, HarvesterState):
            raise TypeError("state must be a HarvesterState")
        # Round-trip through the strict parser.  This rejects manually assembled
        # stale/tampered objects before any numerical work begins.
        try:
            authenticated = HarvesterState.from_bytes(state.to_bytes())
        except (OperatorHarvesterIntegrityError, TypeError, ValueError) as exc:
            raise BoundaryBlanketIntegrityError(
                "harvester state failed strict reconstruction"
            ) from exc
        if authenticated.to_bytes() != state.to_bytes():
            raise BoundaryBlanketIntegrityError(
                "harvester state reconstruction differs"
            )
        if expected_harvester_identity_sha256 is not None:
            expected_identity = require_sha256(
                expected_harvester_identity_sha256,
                field="expected_harvester_identity_sha256",
            )
            if state.identity_sha256 != expected_identity:
                raise BoundaryBlanketIntegrityError("harvester identity is stale")
        observations_by_measurement: dict[str, list[ContextualOperatorObservation]] = {}
        all_observation_sha256s: set[str] = set()
        for _group_sha, observations in state.groups:
            for observation in observations:
                receipt_sha = observation.receipt.sha256
                if receipt_sha in all_observation_sha256s:
                    raise BoundaryBlanketIntegrityError(
                        "harvester state repeats a transition receipt"
                    )
                all_observation_sha256s.add(receipt_sha)
                observations_by_measurement.setdefault(
                    observation.receipt.measurement_sha256, []
                ).append(observation)
        macros = tuple(
            sorted(
                (
                    _macro_from_observations(rows)
                    for rows in observations_by_measurement.values()
                ),
                key=lambda row: row.key,
            )
        )
        if not macros:
            raise BoundaryBlanketIntegrityError("harvester has no boundary macros")
        model_pins = {row.model_pin_sha256 for row in macros}
        if len(model_pins) != 1:
            raise BoundaryBlanketIntegrityError("harvester crosses model pins")
        model_pin = next(iter(model_pins))
        if expected_model_pin_sha256 is not None and model_pin != require_sha256(
            expected_model_pin_sha256, field="expected_model_pin_sha256"
        ):
            raise BoundaryBlanketIntegrityError("model pin is stale")
        if expected_latest_atlas_revision_sha256 is not None:
            expected_revision = require_sha256(
                expected_latest_atlas_revision_sha256,
                field="expected_latest_atlas_revision_sha256",
            )
            latest = max(macros, key=lambda row: row.chronology)
            if latest.atlas_revision_sha256 != expected_revision:
                raise BoundaryBlanketIntegrityError("latest Atlas revision is stale")
        return cls(
            harvester_identity_sha256=state.identity_sha256,
            harvester_state_sha256=state.sha256,
            model_pin_sha256=model_pin,
            macros=macros,
        )

    @property
    def feature_dimension(self) -> int:
        return self.macros[0].feature_dimension

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_schema_sha256": BOUNDARY_CANDIDATE_SCHEMA_SHA256,
            "harvester_identity_sha256": self.harvester_identity_sha256,
            "harvester_state_sha256": self.harvester_state_sha256,
            "macros": [row.evidence_record() for row in self.macros],
            "model_pin_sha256": self.model_pin_sha256,
            "schema": BOUNDARY_CORPUS_SCHEMA,
        }


@dataclass(frozen=True, slots=True)
class BoundaryFitConfig:
    selection_tolerance: float = DEFAULT_SELECTION_TOLERANCE
    ridge_grid: tuple[float, ...] = DEFAULT_RIDGE_GRID
    normalization_floor: float = DEFAULT_NORMALIZATION_FLOOR
    max_condition_number: float = DEFAULT_MAX_CONDITION_NUMBER
    max_working_bytes: int = DEFAULT_MAX_WORKING_BYTES

    def __post_init__(self) -> None:
        tolerance = _finite(
            self.selection_tolerance, field="selection_tolerance", nonnegative=True
        )
        floor = _finite(
            self.normalization_floor, field="normalization_floor", nonnegative=True
        )
        if floor <= 0.0:
            raise ValueError("normalization_floor must be positive")
        condition = _finite(
            self.max_condition_number,
            field="max_condition_number",
            nonnegative=True,
        )
        if condition < 1.0:
            raise ValueError("max_condition_number must be at least one")
        ridges = tuple(
            _finite(row, field="ridge_grid", nonnegative=True)
            for row in self.ridge_grid
        )
        if not ridges or tuple(sorted(set(ridges))) != ridges:
            raise ValueError("ridge_grid must be sorted, unique, and non-empty")
        max_bytes = _positive_int(
            self.max_working_bytes,
            field="max_working_bytes",
            maximum=2**63 - 1,
        )
        object.__setattr__(self, "selection_tolerance", tolerance)
        object.__setattr__(self, "ridge_grid", ridges)
        object.__setattr__(self, "normalization_floor", floor)
        object.__setattr__(self, "max_condition_number", condition)
        object.__setattr__(self, "max_working_bytes", max_bytes)

    def to_record(self) -> dict[str, object]:
        return {
            "max_condition_number": self.max_condition_number,
            "max_working_bytes": self.max_working_bytes,
            "normalization_floor": self.normalization_floor,
            "ridge_grid": list(self.ridge_grid),
            "selection_algorithm": BOUNDARY_SELECTION_ALGORITHM,
            "selection_tolerance": self.selection_tolerance,
        }

    @classmethod
    def from_record(cls, value: object) -> "BoundaryFitConfig":
        if not isinstance(value, Mapping) or set(value) != {
            "max_condition_number",
            "max_working_bytes",
            "normalization_floor",
            "ridge_grid",
            "selection_algorithm",
            "selection_tolerance",
        }:
            raise BoundaryBlanketIntegrityError("fit config is invalid")
        if value.get("selection_algorithm") != BOUNDARY_SELECTION_ALGORITHM:
            raise BoundaryBlanketIntegrityError("selection algorithm is unknown")
        ridges = value.get("ridge_grid")
        if not isinstance(ridges, list):
            raise BoundaryBlanketIntegrityError("ridge grid is invalid")
        try:
            return cls(
                selection_tolerance=cast(float, value.get("selection_tolerance")),
                ridge_grid=tuple(cast(Sequence[float], ridges)),
                normalization_floor=cast(float, value.get("normalization_floor")),
                max_condition_number=cast(float, value.get("max_condition_number")),
                max_working_bytes=cast(int, value.get("max_working_bytes")),
            )
        except (TypeError, ValueError) as exc:
            raise BoundaryBlanketIntegrityError("fit config validation failed") from exc


@dataclass(frozen=True, slots=True)
class RidgeDiagnostic:
    ridge: float
    rank: int
    condition_number: float | None
    admissible: bool
    calibration_score: float | None

    def __post_init__(self) -> None:
        _finite(self.ridge, field="ridge", nonnegative=True)
        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank < 0
        ):
            raise ValueError("rank must be a non-negative integer")
        if self.condition_number is not None:
            _finite(self.condition_number, field="condition_number", nonnegative=True)
        if not isinstance(self.admissible, bool):
            raise TypeError("admissible must be bool")
        if self.calibration_score is not None:
            _finite(
                self.calibration_score,
                field="calibration_score",
                nonnegative=True,
            )
        if self.admissible != (self.calibration_score is not None):
            raise ValueError("ridge diagnostic admissibility is inconsistent")

    def to_record(self) -> dict[str, object]:
        return {
            "admissible": self.admissible,
            "calibration_score": self.calibration_score,
            "condition_number": self.condition_number,
            "rank": self.rank,
            "ridge": self.ridge,
        }

    @classmethod
    def from_record(cls, value: object) -> "RidgeDiagnostic":
        if not isinstance(value, Mapping) or set(value) != {
            "admissible",
            "calibration_score",
            "condition_number",
            "rank",
            "ridge",
        }:
            raise BoundaryBlanketIntegrityError("ridge diagnostic is invalid")
        try:
            return cls(**dict(value))
        except (TypeError, ValueError) as exc:
            raise BoundaryBlanketIntegrityError(
                "ridge diagnostic validation failed"
            ) from exc


@dataclass(frozen=True, slots=True)
class BoundarySubsetModel:
    model_family: str
    subset: tuple[str, ...]
    coefficient: FloatArray
    ridge: float
    design_rank: int
    condition_number: float
    training_score: float
    calibration_score: float
    ridge_diagnostics: tuple[RidgeDiagnostic, ...]

    def __post_init__(self) -> None:
        if self.model_family not in (
            FITTED_AFFINE_FAMILY,
            FIXED_RESIDUAL_SUM_FAMILY,
        ):
            raise ValueError("model_family is unknown")
        subset = tuple(self.subset)
        if subset not in ALL_BOUNDARY_SUBSETS:
            raise ValueError("model subset is not a canonical non-empty subset")
        if (
            self.model_family == FIXED_RESIDUAL_SUM_FAMILY
            and subset != FIXED_RESIDUAL_SUBSET
        ):
            raise ValueError("fixed residual family requires its structural pair")
        coefficient = _canonical_array(self.coefficient, field="coefficient")
        if self.model_family == FIXED_RESIDUAL_SUM_FAMILY:
            dimension = coefficient.shape[1]
            expected = np.zeros((2 * dimension + 1, dimension), dtype=np.float64)
            expected[:dimension, :] = np.eye(dimension, dtype=np.float64)
            expected[dimension : 2 * dimension, :] = np.eye(dimension, dtype=np.float64)
            if not np.array_equal(coefficient, expected):
                raise ValueError(
                    "fixed residual coefficient is not identity-plus-identity"
                )
            if self.ridge != 0.0 or self.condition_number != 1.0:
                raise ValueError(
                    "fixed residual family cannot carry learned regularization"
                )
        for field in (
            "ridge",
            "condition_number",
            "training_score",
            "calibration_score",
        ):
            _finite(getattr(self, field), field=field, nonnegative=True)
        if (
            isinstance(self.design_rank, bool)
            or not isinstance(self.design_rank, int)
            or self.design_rank < 0
        ):
            raise ValueError("design_rank must be a non-negative integer")
        diagnostics = tuple(self.ridge_diagnostics)
        if not diagnostics or any(
            not isinstance(row, RidgeDiagnostic) for row in diagnostics
        ):
            raise ValueError("ridge diagnostics are incomplete")
        if tuple(row.ridge for row in diagnostics) != tuple(
            sorted({row.ridge for row in diagnostics})
        ):
            raise ValueError("ridge diagnostics must be sorted and unique")
        selected = [
            row
            for row in diagnostics
            if row.admissible
            and row.ridge == self.ridge
            and row.calibration_score == self.calibration_score
        ]
        if len(selected) != 1:
            raise ValueError("selected ridge is absent from its diagnostics")
        object.__setattr__(self, "subset", subset)
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "ridge_diagnostics", diagnostics)

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())

    @property
    def key(self) -> tuple[str, tuple[str, ...]]:
        return self.model_family, self.subset

    def to_record(self) -> dict[str, object]:
        return {
            "calibration_score": self.calibration_score,
            "coefficient": _array_record(self.coefficient),
            "condition_number": self.condition_number,
            "design_rank": self.design_rank,
            "model_family": self.model_family,
            "ridge": self.ridge,
            "ridge_diagnostics": [row.to_record() for row in self.ridge_diagnostics],
            "subset": list(self.subset),
            "training_score": self.training_score,
        }

    @classmethod
    def from_record(cls, value: object) -> "BoundarySubsetModel":
        if not isinstance(value, Mapping) or set(value) != {
            "calibration_score",
            "coefficient",
            "condition_number",
            "design_rank",
            "model_family",
            "ridge",
            "ridge_diagnostics",
            "subset",
            "training_score",
        }:
            raise BoundaryBlanketIntegrityError("subset model is invalid")
        subset = value.get("subset")
        diagnostics = value.get("ridge_diagnostics")
        if not isinstance(subset, list) or not isinstance(diagnostics, list):
            raise BoundaryBlanketIntegrityError("subset model inventory is invalid")
        try:
            return cls(
                model_family=cast(str, value.get("model_family")),
                subset=tuple(cast(Sequence[str], subset)),
                coefficient=_array_from_record(
                    value.get("coefficient"), field="coefficient"
                ),
                ridge=cast(float, value.get("ridge")),
                design_rank=cast(int, value.get("design_rank")),
                condition_number=cast(float, value.get("condition_number")),
                training_score=cast(float, value.get("training_score")),
                calibration_score=cast(float, value.get("calibration_score")),
                ridge_diagnostics=tuple(
                    RidgeDiagnostic.from_record(row) for row in diagnostics
                ),
            )
        except BoundaryBlanketIntegrityError:
            raise
        except (TypeError, ValueError) as exc:
            raise BoundaryBlanketIntegrityError(
                "subset model validation failed"
            ) from exc

    def predict(self, macro: BoundaryMacro) -> FloatArray:
        features = _features(macro, self.subset)
        if self.coefficient.shape != (
            features.shape[1] + 1,
            macro.feature_dimension,
        ):
            raise BoundaryBlanketIntegrityError(
                "model coefficient ABI differs from the macro"
            )
        result = features @ self.coefficient[:-1] + self.coefficient[-1]
        if not bool(np.isfinite(result).all()):
            raise BoundaryBlanketIntegrityError("model prediction is non-finite")
        return _canonical_array(result, field="prediction")


def _indices(values: Sequence[int], *, field: str, macro_count: int) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{field} must be a sequence of indices")
    result = tuple(values)
    if (
        not result
        or tuple(sorted(set(result))) != result
        or any(
            isinstance(row, bool)
            or not isinstance(row, int)
            or not 0 <= row < macro_count
            for row in result
        )
    ):
        raise ValueError(f"{field} must be sorted, unique, non-empty indices")
    return result


def _features(macro: BoundaryMacro, subset: Sequence[str]) -> FloatArray:
    arrays = tuple(macro.candidate(name) for name in subset)
    result = np.concatenate(arrays, axis=1)
    if not bool(np.isfinite(result).all()):
        raise BoundaryBlanketIntegrityError("feature matrix is non-finite")
    return cast(FloatArray, result)


def _macro_nrmse(prediction: FloatArray, target: FloatArray, *, floor: float) -> float:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    difference = prediction - target
    rmse = math.sqrt(float(np.mean(difference * difference)))
    scale = max(math.sqrt(float(np.mean(target * target))), floor)
    score = rmse / scale
    if not math.isfinite(score):
        raise BoundaryBlanketIntegrityError("normalized RMSE is non-finite")
    return 0.0 if score == 0.0 else score


def _score_model(
    model: BoundarySubsetModel | tuple[tuple[str, ...], FloatArray],
    macros: Sequence[BoundaryMacro],
    *,
    floor: float,
) -> float:
    scores = []
    if isinstance(model, BoundarySubsetModel):
        for macro in macros:
            scores.append(_macro_nrmse(model.predict(macro), macro.target, floor=floor))
    else:
        subset, coefficient = model
        for macro in macros:
            features = _features(macro, subset)
            prediction = features @ coefficient[:-1] + coefficient[-1]
            scores.append(_macro_nrmse(prediction, macro.target, floor=floor))
    result = math.fsum(scores) / len(scores)
    if not math.isfinite(result):
        raise BoundaryBlanketIntegrityError("macro score is non-finite")
    return 0.0 if result == 0.0 else result


def _weighted_centered_design(
    macros: Sequence[BoundaryMacro], subset: tuple[str, ...]
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Build a stable centered design while giving each macro total weight one."""

    feature_dimension = macros[0].feature_dimension * len(subset)
    output_dimension = macros[0].feature_dimension
    feature_mean = np.zeros((1, feature_dimension), dtype=np.float64)
    target_mean = np.zeros((1, output_dimension), dtype=np.float64)
    for macro in macros:
        features = _features(macro, subset)
        feature_mean += np.mean(features, axis=0, keepdims=True)
        target_mean += np.mean(macro.target, axis=0, keepdims=True)
    feature_mean /= float(len(macros))
    target_mean /= float(len(macros))
    total_rows = sum(row.token_rows for row in macros)
    design = np.empty((total_rows, feature_dimension), dtype=np.float64)
    target = np.empty((total_rows, output_dimension), dtype=np.float64)
    offset = 0
    for macro in macros:
        features = _features(macro, subset)
        row_count = macro.token_rows
        stop = offset + row_count
        scale = 1.0 / math.sqrt(float(row_count))
        design[offset:stop, :] = (features - feature_mean) * scale
        target[offset:stop, :] = (macro.target - target_mean) * scale
        offset = stop
    if not bool(np.isfinite(design).all()) or not bool(np.isfinite(target).all()):
        raise BoundaryBlanketConditionError("weighted centered design is non-finite")
    return (
        cast(FloatArray, design),
        cast(FloatArray, target),
        _canonical_array(feature_mean, field="feature_mean"),
        _canonical_array(target_mean, field="target_mean"),
    )


def _ridge_condition(
    singular: FloatArray,
    *,
    feature_count: int,
    macro_count: int,
    ridge: float,
) -> float:
    squared = np.zeros(feature_count, dtype=np.float64)
    squared[: singular.shape[0]] = singular * singular
    feature_singular = np.sqrt(squared + ridge)
    combined = np.concatenate(
        (feature_singular, np.array([math.sqrt(float(macro_count))]))
    )
    largest = float(np.max(combined))
    smallest = float(np.min(combined))
    tolerance = max(largest, 1.0) * np.finfo(np.float64).eps * len(combined)
    if smallest <= tolerance:
        return math.inf
    return largest / smallest


def _fit_subset(
    train: Sequence[BoundaryMacro],
    calibration: Sequence[BoundaryMacro],
    subset: tuple[str, ...],
    config: BoundaryFitConfig,
) -> BoundarySubsetModel:
    design, centered_target, feature_mean, target_mean = _weighted_centered_design(
        train, subset
    )
    try:
        left, singular, right_transpose = np.linalg.svd(design, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise BoundaryBlanketConditionError(
            f"SVD did not converge for subset {','.join(subset)}"
        ) from exc
    if not (
        bool(np.isfinite(left).all())
        and bool(np.isfinite(singular).all())
        and bool(np.isfinite(right_transpose).all())
    ):
        raise BoundaryBlanketConditionError("fit SVD produced non-finite values")
    feature_count = design.shape[1]
    largest = 0.0 if singular.size == 0 else float(singular[0])
    tolerance = max(largest, 1.0) * np.finfo(np.float64).eps * max(design.shape)
    feature_rank = int(np.count_nonzero(singular > tolerance))
    design_rank = feature_rank + 1
    projected_target = left.T @ centered_target
    diagnostics: list[RidgeDiagnostic] = []
    candidates: list[tuple[float, float, FloatArray, float]] = []
    for ridge in config.ridge_grid:
        condition = _ridge_condition(
            cast(FloatArray, singular),
            feature_count=feature_count,
            macro_count=len(train),
            ridge=ridge,
        )
        coefficient: FloatArray | None = None
        calibration_score: float | None = None
        admissible = (
            math.isfinite(condition) and condition <= config.max_condition_number
        )
        if admissible:
            denominator = singular * singular + ridge
            gains = np.divide(
                singular,
                denominator,
                out=np.zeros_like(singular),
                where=denominator > 0.0,
            )
            slope = right_transpose.T @ (gains[:, None] * projected_target)
            intercept = target_mean - feature_mean @ slope
            solved = np.concatenate((slope, intercept), axis=0)
            if bool(np.isfinite(solved).all()):
                coefficient = _canonical_array(solved, field="coefficient")
                calibration_score = _score_model(
                    (subset, coefficient),
                    calibration,
                    floor=config.normalization_floor,
                )
            else:
                admissible = False
        if not admissible:
            condition_value = None if not math.isfinite(condition) else condition
            diagnostics.append(
                RidgeDiagnostic(ridge, design_rank, condition_value, False, None)
            )
            continue
        assert coefficient is not None and calibration_score is not None
        diagnostics.append(
            RidgeDiagnostic(ridge, design_rank, condition, True, calibration_score)
        )
        candidates.append((calibration_score, ridge, coefficient, condition))
    if not candidates:
        raise BoundaryBlanketConditionError(
            f"no admissible ridge for subset {','.join(subset)}"
        )
    calibration_score, ridge, coefficient, condition = min(
        candidates, key=lambda row: (row[0], row[1])
    )
    training_score = _score_model(
        (subset, coefficient), train, floor=config.normalization_floor
    )
    return BoundarySubsetModel(
        model_family=FITTED_AFFINE_FAMILY,
        subset=subset,
        coefficient=coefficient,
        ridge=ridge,
        design_rank=design_rank,
        condition_number=condition,
        training_score=training_score,
        calibration_score=calibration_score,
        ridge_diagnostics=tuple(diagnostics),
    )


def _fixed_residual_sum_model(
    train: Sequence[BoundaryMacro],
    calibration: Sequence[BoundaryMacro],
    config: BoundaryFitConfig,
) -> BoundarySubsetModel:
    """Materialize the Qwen block identity with no learned coefficient."""

    dimension = train[0].feature_dimension
    coefficient = np.zeros((2 * dimension + 1, dimension), dtype=np.float64)
    coefficient[:dimension, :] = np.eye(dimension, dtype=np.float64)
    coefficient[dimension : 2 * dimension, :] = np.eye(dimension, dtype=np.float64)
    sealed = _canonical_array(coefficient, field="fixed_residual_coefficient")
    training_score = _score_model(
        (FIXED_RESIDUAL_SUBSET, sealed),
        train,
        floor=config.normalization_floor,
    )
    calibration_score = _score_model(
        (FIXED_RESIDUAL_SUBSET, sealed),
        calibration,
        floor=config.normalization_floor,
    )
    diagnostic = RidgeDiagnostic(
        ridge=0.0,
        rank=0,
        condition_number=1.0,
        admissible=True,
        calibration_score=calibration_score,
    )
    return BoundarySubsetModel(
        model_family=FIXED_RESIDUAL_SUM_FAMILY,
        subset=FIXED_RESIDUAL_SUBSET,
        coefficient=sealed,
        ridge=0.0,
        design_rank=0,
        condition_number=1.0,
        training_score=training_score,
        calibration_score=calibration_score,
        ridge_diagnostics=(diagnostic,),
    )


def _mean_target(macros: Sequence[BoundaryMacro]) -> FloatArray:
    dimension = macros[0].feature_dimension
    result = np.zeros((1, dimension), dtype=np.float64)
    for macro in macros:
        result += np.mean(macro.target, axis=0, keepdims=True)
    result /= float(len(macros))
    return _canonical_array(result, field="mean_target")


def _split_record(corpus: BoundaryCorpus, indices: Sequence[int]) -> dict[str, object]:
    macros = tuple(corpus.macros[index] for index in indices)
    return {
        "atlas_revision_sha256s": [row.atlas_revision_sha256 for row in macros],
        "atlas_sequences": [row.atlas_sequence for row in macros],
        "indices": list(indices),
        "layers": sorted({row.layer for row in macros}),
        "measurement_receipt_sha256s": [row.measurement_sha256 for row in macros],
        "observation_record_sha256s": [
            digest for row in macros for digest in row.observation_record_sha256s
        ],
        "prompt_sha256s": sorted({row.prompt_sha256 for row in macros}),
        "transition_receipt_sha256s": [
            digest for row in macros for digest in row.transition_receipt_sha256s
        ],
        "weight_rail_revision_sha256s": sorted(
            {row.weight_rail_revision_sha256 for row in macros}
        ),
    }


def _validate_split_record(value: object, *, label: str) -> Mapping[str, object]:
    expected = {
        "atlas_revision_sha256s",
        "atlas_sequences",
        "indices",
        "layers",
        "measurement_receipt_sha256s",
        "observation_record_sha256s",
        "prompt_sha256s",
        "transition_receipt_sha256s",
        "weight_rail_revision_sha256s",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise BoundaryBlanketIntegrityError(f"{label} split record is invalid")
    for field in expected:
        if not isinstance(value.get(field), list):
            raise BoundaryBlanketIntegrityError(f"{label}.{field} is invalid")
    return value


@dataclass(frozen=True, slots=True)
class BoundaryBlanketFit:
    corpus_sha256: str
    harvester_identity_sha256: str
    harvester_state_sha256: str
    model_pin_sha256: str
    candidate_schema_sha256: str
    feature_dimension: int
    config: BoundaryFitConfig
    train: Mapping[str, object]
    calibration: Mapping[str, object]
    mean_target: FloatArray
    models: tuple[BoundarySubsetModel, ...]
    selected_model_family: str
    selected_subset: tuple[str, ...]
    best_calibration_score: float
    best_fitted_calibration_score: float
    full_calibration_score: float

    def __post_init__(self) -> None:
        for field in (
            "corpus_sha256",
            "harvester_identity_sha256",
            "harvester_state_sha256",
            "model_pin_sha256",
            "candidate_schema_sha256",
        ):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        if self.candidate_schema_sha256 != BOUNDARY_CANDIDATE_SCHEMA_SHA256:
            raise BoundaryBlanketIntegrityError("candidate semantic schema is unknown")
        dimension = _positive_int(
            self.feature_dimension,
            field="feature_dimension",
            maximum=MAX_FEATURE_DIMENSION,
        )
        if not isinstance(self.config, BoundaryFitConfig):
            raise TypeError("config must be BoundaryFitConfig")
        train = _validate_split_record(self.train, label="train")
        calibration = _validate_split_record(self.calibration, label="calibration")
        train_indices = tuple(cast(Sequence[int], train.get("indices")))
        cal_indices = tuple(cast(Sequence[int], calibration.get("indices")))
        if set(train_indices) & set(cal_indices):
            raise BoundaryBlanketIntegrityError("fit splits overlap")
        if max(train_indices) >= min(cal_indices):
            raise BoundaryBlanketIntegrityError(
                "calibration must be later than training in corpus chronology"
            )
        mean = _canonical_array(self.mean_target, field="mean_target")
        if mean.shape != (1, dimension):
            raise ValueError("mean target has the wrong ABI")
        models = tuple(self.models)
        if (
            len(models) != len(ALL_BOUNDARY_MODEL_KEYS)
            or tuple(row.key for row in models) != ALL_BOUNDARY_MODEL_KEYS
        ):
            raise BoundaryBlanketIntegrityError(
                "fit must contain every fitted subset and structural family"
            )
        for model in models:
            expected_shape = (len(model.subset) * dimension + 1, dimension)
            if model.coefficient.shape != expected_shape:
                raise BoundaryBlanketIntegrityError(
                    "subset coefficient has the wrong numerical ABI"
                )
            if model.model_family == FITTED_AFFINE_FAMILY and (
                tuple(row.ridge for row in model.ridge_diagnostics)
                != self.config.ridge_grid
            ):
                raise BoundaryBlanketIntegrityError(
                    "subset ridge diagnostics differ from the fit config"
                )
            if model.model_family == FIXED_RESIDUAL_SUM_FAMILY and (
                len(model.ridge_diagnostics) != 1
                or model.ridge_diagnostics[0].ridge != 0.0
            ):
                raise BoundaryBlanketIntegrityError(
                    "fixed structural family has learned ridge diagnostics"
                )
        if self.selected_model_family not in (
            FITTED_AFFINE_FAMILY,
            FIXED_RESIDUAL_SUM_FAMILY,
        ):
            raise BoundaryBlanketIntegrityError("selected model family is unknown")
        selected = tuple(self.selected_subset)
        by_key = {row.key: row for row in models}
        selected_key = (self.selected_model_family, selected)
        if selected_key not in by_key:
            raise BoundaryBlanketIntegrityError("selected subset is absent")
        best = min(row.calibration_score for row in models)
        fitted_models = tuple(
            row for row in models if row.model_family == FITTED_AFFINE_FAMILY
        )
        best_fitted = min(row.calibration_score for row in fitted_models)
        full = by_key[(FITTED_AFFINE_FAMILY, BOUNDARY_CANDIDATES)].calibration_score
        _finite(
            self.best_calibration_score,
            field="best_calibration_score",
            nonnegative=True,
        )
        _finite(
            self.best_fitted_calibration_score,
            field="best_fitted_calibration_score",
            nonnegative=True,
        )
        _finite(
            self.full_calibration_score,
            field="full_calibration_score",
            nonnegative=True,
        )
        if (
            best != self.best_calibration_score
            or best_fitted != self.best_fitted_calibration_score
            or full != self.full_calibration_score
        ):
            raise BoundaryBlanketIntegrityError("fit summary scores are inconsistent")
        threshold = min(best, full) + self.config.selection_tolerance
        eligible = [row for row in models if row.calibration_score <= threshold]
        expected_selected = min(
            eligible,
            key=lambda row: (
                len(row.subset),
                row.calibration_score,
                row.model_family,
                row.subset,
            ),
        )
        if selected_key != expected_selected.key:
            raise BoundaryBlanketIntegrityError(
                "selected subset violates the algorithm"
            )
        object.__setattr__(self, "feature_dimension", dimension)
        object.__setattr__(self, "train", train)
        object.__setattr__(self, "calibration", calibration)
        object.__setattr__(self, "mean_target", mean)
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "selected_model_family", self.selected_model_family)
        object.__setattr__(self, "selected_subset", selected)

    @property
    def selected_model(self) -> BoundarySubsetModel:
        return next(
            row
            for row in self.models
            if row.key == (self.selected_model_family, self.selected_subset)
        )

    @property
    def full_model(self) -> BoundarySubsetModel:
        return next(
            row
            for row in self.models
            if row.key == (FITTED_AFFINE_FAMILY, BOUNDARY_CANDIDATES)
        )

    @property
    def best_fitted_model(self) -> BoundarySubsetModel:
        return min(
            (row for row in self.models if row.model_family == FITTED_AFFINE_FAMILY),
            key=lambda row: (row.calibration_score, len(row.subset), row.subset),
        )

    def _verify_splits_against_corpus(self, corpus: BoundaryCorpus) -> None:
        """Rebuild both split identities from the authenticated external corpus."""

        if not isinstance(corpus, BoundaryCorpus):
            raise TypeError("corpus must be a BoundaryCorpus")
        if (
            self.corpus_sha256 != corpus.sha256
            or self.harvester_state_sha256 != corpus.harvester_state_sha256
            or self.harvester_identity_sha256 != corpus.harvester_identity_sha256
            or self.model_pin_sha256 != corpus.model_pin_sha256
        ):
            raise BoundaryBlanketIntegrityError("fit belongs to another corpus state")
        train_indices = _indices(
            cast(Sequence[int], self.train.get("indices")),
            field="train.indices",
            macro_count=len(corpus.macros),
        )
        calibration_indices = _indices(
            cast(Sequence[int], self.calibration.get("indices")),
            field="calibration.indices",
            macro_count=len(corpus.macros),
        )
        if dict(self.train) != _split_record(corpus, train_indices):
            raise BoundaryBlanketIntegrityError(
                "train split differs from the authenticated corpus"
            )
        if dict(self.calibration) != _split_record(corpus, calibration_indices):
            raise BoundaryBlanketIntegrityError(
                "calibration split differs from the authenticated corpus"
            )

    def verify_against_corpus(self, corpus: BoundaryCorpus) -> None:
        """Recompute the complete fit and demand a byte-identical receipt."""

        self._verify_splits_against_corpus(corpus)
        train_indices = tuple(cast(Sequence[int], self.train.get("indices")))
        calibration_indices = tuple(
            cast(Sequence[int], self.calibration.get("indices"))
        )
        recomputed = fit_boundary_blanket(
            corpus,
            train_indices=train_indices,
            calibration_indices=calibration_indices,
            config=self.config,
        )
        if recomputed.to_bytes() != self.to_bytes():
            raise BoundaryBlanketIntegrityError(
                "fit differs from a complete authenticated recomputation"
            )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def to_document(self) -> dict[str, object]:
        body = {
            "best_calibration_score": self.best_calibration_score,
            "best_fitted_calibration_score": self.best_fitted_calibration_score,
            "calibration": dict(self.calibration),
            "candidate_schema_sha256": self.candidate_schema_sha256,
            "config": self.config.to_record(),
            "corpus_sha256": self.corpus_sha256,
            "feature_dimension": self.feature_dimension,
            "full_calibration_score": self.full_calibration_score,
            "harvester_identity_sha256": self.harvester_identity_sha256,
            "harvester_state_sha256": self.harvester_state_sha256,
            "mean_target": _array_record(self.mean_target),
            "model_pin_sha256": self.model_pin_sha256,
            "model_inventory": BOUNDARY_MODEL_INVENTORY,
            "models": [row.to_record() for row in self.models],
            "numeric_abi": BOUNDARY_NUMERIC_ABI,
            "selected_model_family": self.selected_model_family,
            "selected_subset": list(self.selected_subset),
            "selection_algorithm": BOUNDARY_SELECTION_ALGORITHM,
            "train": dict(self.train),
        }
        return {
            "body": body,
            "body_sha256": _digest(body),
            "schema": BOUNDARY_FIT_SCHEMA,
        }

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_document())
        if len(data) > MAX_RECEIPT_BYTES:
            raise BoundaryBlanketCapacityError("fit receipt exceeds its byte bound")
        return data

    @classmethod
    def from_bytes(cls, data: bytes, *, corpus: BoundaryCorpus) -> "BoundaryBlanketFit":
        envelope = _strict_json(data, schema=BOUNDARY_FIT_SCHEMA)
        body = envelope.get("body")
        expected = {
            "best_calibration_score",
            "best_fitted_calibration_score",
            "calibration",
            "candidate_schema_sha256",
            "config",
            "corpus_sha256",
            "feature_dimension",
            "full_calibration_score",
            "harvester_identity_sha256",
            "harvester_state_sha256",
            "mean_target",
            "model_pin_sha256",
            "model_inventory",
            "models",
            "numeric_abi",
            "selected_model_family",
            "selected_subset",
            "selection_algorithm",
            "train",
        }
        if not isinstance(body, Mapping) or set(body) != expected:
            raise BoundaryBlanketIntegrityError("fit body is invalid")
        if (
            body.get("numeric_abi") != BOUNDARY_NUMERIC_ABI
            or body.get("selection_algorithm") != BOUNDARY_SELECTION_ALGORITHM
            or body.get("model_inventory") != BOUNDARY_MODEL_INVENTORY
        ):
            raise BoundaryBlanketIntegrityError("fit numerical algorithm is unknown")
        models = body.get("models")
        selected = body.get("selected_subset")
        if not isinstance(models, list) or not isinstance(selected, list):
            raise BoundaryBlanketIntegrityError("fit inventories are invalid")
        try:
            result = cls(
                corpus_sha256=cast(str, body.get("corpus_sha256")),
                harvester_identity_sha256=cast(
                    str, body.get("harvester_identity_sha256")
                ),
                harvester_state_sha256=cast(str, body.get("harvester_state_sha256")),
                model_pin_sha256=cast(str, body.get("model_pin_sha256")),
                candidate_schema_sha256=cast(str, body.get("candidate_schema_sha256")),
                feature_dimension=cast(int, body.get("feature_dimension")),
                config=BoundaryFitConfig.from_record(body.get("config")),
                train=_validate_split_record(body.get("train"), label="train"),
                calibration=_validate_split_record(
                    body.get("calibration"), label="calibration"
                ),
                mean_target=_array_from_record(
                    body.get("mean_target"), field="mean_target"
                ),
                models=tuple(BoundarySubsetModel.from_record(row) for row in models),
                selected_model_family=cast(str, body.get("selected_model_family")),
                selected_subset=tuple(cast(Sequence[str], selected)),
                best_calibration_score=cast(float, body.get("best_calibration_score")),
                best_fitted_calibration_score=cast(
                    float, body.get("best_fitted_calibration_score")
                ),
                full_calibration_score=cast(float, body.get("full_calibration_score")),
            )
        except BoundaryBlanketIntegrityError:
            raise
        except (TypeError, ValueError) as exc:
            raise BoundaryBlanketIntegrityError("fit validation failed") from exc
        if result.to_bytes() != data:
            raise BoundaryBlanketIntegrityError("fit reconstruction differs")
        result.verify_against_corpus(corpus)
        return result


def _preflight_fit_capacity(corpus: BoundaryCorpus, config: BoundaryFitConfig) -> None:
    dimension = corpus.feature_dimension
    maximum_columns = len(BOUNDARY_CANDIDATES) * dimension + 1
    # Gram, regularized Gram, cross, coefficient, eigensolver copy, plus every
    # persisted coefficient.  This conservatively covers the largest centered
    # SVD design over the complete authenticated corpus, not only the fit split.
    total_rows = sum(row.token_rows for row in corpus.macros)
    working = 8 * (
        3 * total_rows * maximum_columns
        + 3 * total_rows * dimension
        + 3 * maximum_columns * maximum_columns
        + 3 * maximum_columns * dimension
    )
    persisted = (
        8
        * sum(
            (len(subset) * dimension + 1) * dimension for subset in ALL_BOUNDARY_SUBSETS
        )
        + 8 * (2 * dimension + 1) * dimension
    )
    required = working + persisted
    if required > config.max_working_bytes:
        raise BoundaryBlanketCapacityError(
            f"boundary fit requires {required} bytes, limit is {config.max_working_bytes}"
        )


def fit_boundary_blanket(
    corpus: BoundaryCorpus,
    *,
    train_indices: Sequence[int],
    calibration_indices: Sequence[int],
    config: BoundaryFitConfig | None = None,
) -> BoundaryBlanketFit:
    """Fit on training macros and select using calibration macros only."""

    if not isinstance(corpus, BoundaryCorpus):
        raise TypeError("corpus must be a BoundaryCorpus")
    selected_config = BoundaryFitConfig() if config is None else config
    if not isinstance(selected_config, BoundaryFitConfig):
        raise TypeError("config must be BoundaryFitConfig")
    train_ids = _indices(
        train_indices, field="train_indices", macro_count=len(corpus.macros)
    )
    calibration_ids = _indices(
        calibration_indices,
        field="calibration_indices",
        macro_count=len(corpus.macros),
    )
    if set(train_ids) & set(calibration_ids):
        raise ValueError("train and calibration indices overlap")
    if max(train_ids) >= min(calibration_ids):
        raise ValueError("calibration must follow training in chronology")
    _preflight_fit_capacity(corpus, selected_config)
    train = tuple(corpus.macros[index] for index in train_ids)
    calibration = tuple(corpus.macros[index] for index in calibration_ids)
    fitted_models = tuple(
        _fit_subset(train, calibration, subset, selected_config)
        for subset in ALL_BOUNDARY_SUBSETS
    )
    structural_model = _fixed_residual_sum_model(train, calibration, selected_config)
    models = fitted_models + (structural_model,)
    best = min(row.calibration_score for row in models)
    best_fitted = min(row.calibration_score for row in fitted_models)
    full = next(
        row.calibration_score
        for row in fitted_models
        if row.subset == BOUNDARY_CANDIDATES
    )
    threshold = min(best, full) + selected_config.selection_tolerance
    selected = min(
        (row for row in models if row.calibration_score <= threshold),
        key=lambda row: (
            len(row.subset),
            row.calibration_score,
            row.model_family,
            row.subset,
        ),
    )
    result = BoundaryBlanketFit(
        corpus_sha256=corpus.sha256,
        harvester_identity_sha256=corpus.harvester_identity_sha256,
        harvester_state_sha256=corpus.harvester_state_sha256,
        model_pin_sha256=corpus.model_pin_sha256,
        candidate_schema_sha256=BOUNDARY_CANDIDATE_SCHEMA_SHA256,
        feature_dimension=corpus.feature_dimension,
        config=selected_config,
        train=_split_record(corpus, train_ids),
        calibration=_split_record(corpus, calibration_ids),
        mean_target=_mean_target(train),
        models=models,
        selected_model_family=selected.model_family,
        selected_subset=selected.subset,
        best_calibration_score=best,
        best_fitted_calibration_score=best_fitted,
        full_calibration_score=full,
    )
    result._verify_splits_against_corpus(corpus)
    return result


def _baseline_score(
    name: str,
    macros: Sequence[BoundaryMacro],
    fit: BoundaryBlanketFit,
) -> float:
    scores = []
    for macro in macros:
        if name == "no_memory_mean":
            prediction = np.broadcast_to(fit.mean_target, macro.target.shape)
        elif name == "layer_input_identity":
            prediction = macro.candidate("layer_input")
        elif name == "attention_residual_plus_mlp_output":
            prediction = macro.candidate("attention_residual") + macro.candidate(
                "mlp_output"
            )
        else:  # pragma: no cover - internal closed inventory
            raise AssertionError(name)
        scores.append(
            _macro_nrmse(
                cast(FloatArray, prediction),
                macro.target,
                floor=fit.config.normalization_floor,
            )
        )
    return math.fsum(scores) / len(scores)


def _placebo_score(
    model: BoundarySubsetModel,
    macros: Sequence[BoundaryMacro],
    *,
    floor: float,
) -> float:
    """Score against a deterministic wrong flat-coordinate correspondence."""

    scores = []
    for macro in macros:
        prediction = model.predict(macro)
        flat = macro.target.reshape(-1)
        placebo = (
            np.roll(flat, 1).reshape(macro.target.shape)
            if flat.size > 1
            else -macro.target
        )
        scores.append(
            _macro_nrmse(
                prediction,
                cast(FloatArray, placebo),
                floor=floor,
            )
        )
    return math.fsum(scores) / len(scores)


def _metric_record(model: BoundarySubsetModel, score: float) -> dict[str, object]:
    return {
        "model_family": model.model_family,
        "model_sha256": model.sha256,
        "ridge": model.ridge,
        "score": score,
        "subset": list(model.subset),
    }


def _evaluate_boundary_blanket_holdout(
    fit: BoundaryBlanketFit,
    corpus: BoundaryCorpus,
    *,
    holdout_indices: Sequence[int],
    require_later: bool,
    holdout_kind: str,
    _verify_complete: bool = True,
) -> dict[str, object]:
    """Score one disjoint holdout without changing any fitted coefficient."""

    if not isinstance(fit, BoundaryBlanketFit):
        raise TypeError("fit must be a BoundaryBlanketFit")
    if not isinstance(corpus, BoundaryCorpus):
        raise TypeError("corpus must be a BoundaryCorpus")
    if not isinstance(require_later, bool):
        raise TypeError("require_later must be bool")
    if not isinstance(_verify_complete, bool):
        raise TypeError("_verify_complete must be bool")
    if holdout_kind not in ("temporal_chronological", "loqo_prompt"):
        raise ValueError("holdout_kind is unknown")
    fit.verify_against_corpus(corpus)
    holdout_ids = _indices(
        holdout_indices, field="holdout_indices", macro_count=len(corpus.macros)
    )
    fit_ids = set(cast(Sequence[int], fit.train.get("indices"))) | set(
        cast(Sequence[int], fit.calibration.get("indices"))
    )
    if fit_ids & set(holdout_ids):
        raise ValueError("holdout overlaps fit or calibration")
    latest_fit_chronology = max(corpus.macros[index].chronology for index in fit_ids)
    earliest_holdout_chronology = min(
        corpus.macros[index].chronology for index in holdout_ids
    )
    if require_later and latest_fit_chronology >= earliest_holdout_chronology:
        raise ValueError("holdout must follow training and calibration in chronology")
    macros = tuple(corpus.macros[index] for index in holdout_ids)
    scores = {
        model.key: _score_model(model, macros, floor=fit.config.normalization_floor)
        for model in fit.models
    }
    selected = fit.selected_model
    full = fit.full_model
    best_fitted = fit.best_fitted_model
    equal_size = tuple(
        model for model in fit.models if len(model.subset) == len(selected.subset)
    )
    equal_records = tuple(
        _metric_record(model, scores[model.key])
        for model in sorted(equal_size, key=lambda row: (row.subset, row.model_family))
    )
    ordered_equal = sorted(
        equal_size,
        key=lambda row: (scores[row.key], row.model_family, row.subset),
    )
    rank = ordered_equal.index(selected) + 1
    equal_scores = sorted(scores[row.key] for row in equal_size)
    midpoint = len(equal_scores) // 2
    median = (
        equal_scores[midpoint]
        if len(equal_scores) % 2
        else (equal_scores[midpoint - 1] + equal_scores[midpoint]) / 2.0
    )
    seen_indices = sorted(fit_ids)
    seen = tuple(corpus.macros[index] for index in seen_indices)
    seen_layers = sorted({row.layer for row in seen})
    holdout_layers = sorted({row.layer for row in macros})
    unseen_layers = sorted(set(holdout_layers) - set(seen_layers))
    seen_prompts = {row.prompt_sha256 for row in seen}
    holdout_prompts = {row.prompt_sha256 for row in macros}
    repeated_prompts = sorted(seen_prompts & holdout_prompts)
    topology_holdout = bool(unseen_layers) and holdout_prompts <= seen_prompts
    holdout_record = _split_record(corpus, holdout_ids)
    residual_score = _baseline_score("attention_residual_plus_mlp_output", macros, fit)
    selected_score = scores[selected.key]
    full_score = scores[full.key]
    best_fitted_score = scores[best_fitted.key]
    winner_name, winner_score = min(
        (
            ("fitted_full", full_score),
            ("best_fitted_affine", best_fitted_score),
            ("selected_model", selected_score),
            ("fixed_attention_residual_plus_mlp_output", residual_score),
        ),
        key=lambda row: (row[1], row[0]),
    )
    body = {
        "baselines": {
            "attention_residual_plus_mlp_output": residual_score,
            "layer_input_identity": _baseline_score(
                "layer_input_identity", macros, fit
            ),
            "no_memory_mean": _baseline_score("no_memory_mean", macros, fit),
            "placebo_algorithm": "target-flat-coordinate-rotate-one/v1",
            "full_target_permutation_placebo": _placebo_score(
                full, macros, floor=fit.config.normalization_floor
            ),
            "selected_target_permutation_placebo": _placebo_score(
                selected, macros, floor=fit.config.normalization_floor
            ),
        },
        "candidate_schema_sha256": fit.candidate_schema_sha256,
        "corpus_sha256": corpus.sha256,
        "best_fitted_affine": _metric_record(best_fitted, best_fitted_score),
        "equal_size_alternatives": list(equal_records),
        "equal_size_median_score": median,
        "fit_sha256": fit.sha256,
        "full": _metric_record(full, scores[full.key]),
        "harvester_state_sha256": corpus.harvester_state_sha256,
        "holdout": holdout_record,
        "holdout_kind": holdout_kind,
        "holdout_layer_indices": holdout_layers,
        "model_pin_sha256": corpus.model_pin_sha256,
        "numeric_abi": BOUNDARY_NUMERIC_ABI,
        "repeated_prompt_sha256s": repeated_prompts,
        "selected": _metric_record(selected, scores[selected.key]),
        "selected_equal_size_rank": rank,
        "selected_equal_size_total": len(equal_size),
        "topology_holdout": topology_holdout,
        "train_calibration_layer_indices": seen_layers,
        "unseen_layer_indices": unseen_layers,
        "winner": {"name": winner_name, "score": winner_score},
    }
    document = {
        "body": body,
        "body_sha256": _digest(body),
        "schema": BOUNDARY_HOLDOUT_SCHEMA,
    }
    if _verify_complete:
        verify_boundary_blanket_holdout(document, fit=fit, corpus=corpus)
    else:
        _strict_json(canonical_json_bytes(document), schema=BOUNDARY_HOLDOUT_SCHEMA)
    return document


def evaluate_boundary_blanket_holdout(
    fit: BoundaryBlanketFit,
    corpus: BoundaryCorpus,
    *,
    holdout_indices: Sequence[int],
) -> dict[str, object]:
    """Score a strictly later chronological holdout without refitting."""

    return _evaluate_boundary_blanket_holdout(
        fit,
        corpus,
        holdout_indices=holdout_indices,
        require_later=True,
        holdout_kind="temporal_chronological",
    )


def verify_boundary_blanket_holdout(
    document: Mapping[str, object],
    *,
    fit: BoundaryBlanketFit,
    corpus: BoundaryCorpus,
) -> None:
    """Rebuild the holdout identity and topology from authenticated evidence."""

    if not isinstance(document, Mapping):
        raise TypeError("document must be a mapping")
    if not isinstance(fit, BoundaryBlanketFit):
        raise TypeError("fit must be a BoundaryBlanketFit")
    if not isinstance(corpus, BoundaryCorpus):
        raise TypeError("corpus must be a BoundaryCorpus")
    fit.verify_against_corpus(corpus)
    envelope = _strict_json(
        canonical_json_bytes(document), schema=BOUNDARY_HOLDOUT_SCHEMA
    )
    body = envelope.get("body")
    expected_fields = {
        "baselines",
        "best_fitted_affine",
        "candidate_schema_sha256",
        "corpus_sha256",
        "equal_size_alternatives",
        "equal_size_median_score",
        "fit_sha256",
        "full",
        "harvester_state_sha256",
        "holdout",
        "holdout_kind",
        "holdout_layer_indices",
        "model_pin_sha256",
        "numeric_abi",
        "repeated_prompt_sha256s",
        "selected",
        "selected_equal_size_rank",
        "selected_equal_size_total",
        "topology_holdout",
        "train_calibration_layer_indices",
        "unseen_layer_indices",
        "winner",
    }
    if not isinstance(body, Mapping) or set(body) != expected_fields:
        raise BoundaryBlanketIntegrityError("holdout body is invalid")
    if (
        body.get("fit_sha256") != fit.sha256
        or body.get("corpus_sha256") != corpus.sha256
        or body.get("harvester_state_sha256") != corpus.harvester_state_sha256
        or body.get("model_pin_sha256") != corpus.model_pin_sha256
        or body.get("candidate_schema_sha256") != fit.candidate_schema_sha256
        or body.get("numeric_abi") != BOUNDARY_NUMERIC_ABI
    ):
        raise BoundaryBlanketIntegrityError("holdout provenance is stale")
    split = _validate_split_record(body.get("holdout"), label="holdout")
    holdout_indices = _indices(
        cast(Sequence[int], split.get("indices")),
        field="holdout.indices",
        macro_count=len(corpus.macros),
    )
    if dict(split) != _split_record(corpus, holdout_indices):
        raise BoundaryBlanketIntegrityError(
            "holdout split differs from the authenticated corpus"
        )
    fit_indices = set(cast(Sequence[int], fit.train.get("indices"))) | set(
        cast(Sequence[int], fit.calibration.get("indices"))
    )
    if fit_indices & set(holdout_indices):
        raise BoundaryBlanketIntegrityError("holdout overlaps fit evidence")
    kind = body.get("holdout_kind")
    if kind not in ("temporal_chronological", "loqo_prompt"):
        raise BoundaryBlanketIntegrityError("holdout kind is unknown")
    if kind == "temporal_chronological" and max(
        corpus.macros[index].chronology for index in fit_indices
    ) >= min(corpus.macros[index].chronology for index in holdout_indices):
        raise BoundaryBlanketIntegrityError("temporal holdout precedes fit evidence")
    fit_macros = tuple(corpus.macros[index] for index in sorted(fit_indices))
    holdout_macros = tuple(corpus.macros[index] for index in holdout_indices)
    fit_layers = sorted({row.layer for row in fit_macros})
    holdout_layers = sorted({row.layer for row in holdout_macros})
    unseen_layers = sorted(set(holdout_layers) - set(fit_layers))
    fit_prompts = {row.prompt_sha256 for row in fit_macros}
    holdout_prompts = {row.prompt_sha256 for row in holdout_macros}
    repeated_prompts = sorted(fit_prompts & holdout_prompts)
    topology = bool(unseen_layers) and holdout_prompts <= fit_prompts
    if (
        body.get("train_calibration_layer_indices") != fit_layers
        or body.get("holdout_layer_indices") != holdout_layers
        or body.get("unseen_layer_indices") != unseen_layers
        or body.get("repeated_prompt_sha256s") != repeated_prompts
        or body.get("topology_holdout") != topology
    ):
        raise BoundaryBlanketIntegrityError(
            "holdout topology differs from the authenticated corpus"
        )
    recomputed = _evaluate_boundary_blanket_holdout(
        fit,
        corpus,
        holdout_indices=holdout_indices,
        require_later=(kind == "temporal_chronological"),
        holdout_kind=cast(str, kind),
        _verify_complete=False,
    )
    if canonical_json_bytes(recomputed) != canonical_json_bytes(document):
        raise BoundaryBlanketIntegrityError(
            "holdout differs from a complete authenticated recomputation"
        )


def _build_boundary_blanket_loqo(
    corpus: BoundaryCorpus,
    *,
    config: BoundaryFitConfig | None = None,
) -> dict[str, object]:
    """Run a separate leave-one-query-out prompt generalization audit."""

    if not isinstance(corpus, BoundaryCorpus):
        raise TypeError("corpus must be a BoundaryCorpus")
    selected_config = BoundaryFitConfig() if config is None else config
    prompts = sorted({row.prompt_sha256 for row in corpus.macros})
    if len(prompts) < 3:
        raise ValueError("LOQO requires at least three distinct prompts")
    folds = []
    for prompt in prompts:
        holdout_ids = tuple(
            index
            for index, macro in enumerate(corpus.macros)
            if macro.prompt_sha256 == prompt
        )
        remaining = tuple(
            index
            for index, macro in enumerate(corpus.macros)
            if macro.prompt_sha256 != prompt
        )
        split = max(1, (3 * len(remaining)) // 4)
        if split >= len(remaining):
            split = len(remaining) - 1
        train_ids = remaining[:split]
        calibration_ids = remaining[split:]
        fit = fit_boundary_blanket(
            corpus,
            train_indices=train_ids,
            calibration_indices=calibration_ids,
            config=selected_config,
        )
        evaluation = _evaluate_boundary_blanket_holdout(
            fit,
            corpus,
            holdout_indices=holdout_ids,
            require_later=False,
            holdout_kind="loqo_prompt",
        )
        evaluation_body = cast(Mapping[str, object], evaluation["body"])
        folds.append(
            {
                "evaluation": evaluation,
                "fit_sha256": fit.sha256,
                "holdout_sha256": evaluation["body_sha256"],
                "left_out_prompt_sha256": prompt,
                "selected_model_family": fit.selected_model_family,
                "selected_score": cast(
                    Mapping[str, object], evaluation_body["selected"]
                )["score"],
                "selected_subset": list(fit.selected_subset),
            }
        )
    body = {
        "candidate_schema_sha256": BOUNDARY_CANDIDATE_SCHEMA_SHA256,
        "corpus_sha256": corpus.sha256,
        "folds": folds,
        "harvester_state_sha256": corpus.harvester_state_sha256,
        "model_pin_sha256": corpus.model_pin_sha256,
        "numeric_abi": BOUNDARY_NUMERIC_ABI,
        "prompt_count": len(prompts),
    }
    return {
        "body": body,
        "body_sha256": _digest(body),
        "schema": BOUNDARY_LOQO_SCHEMA,
    }


def verify_boundary_blanket_loqo(
    corpus: BoundaryCorpus,
    document: Mapping[str, object],
    config: BoundaryFitConfig | None = None,
) -> None:
    """Recompute every LOQO fold and require a byte-identical audit."""

    if not isinstance(corpus, BoundaryCorpus):
        raise TypeError("corpus must be a BoundaryCorpus")
    if not isinstance(document, Mapping):
        raise TypeError("document must be a mapping")
    selected_config = BoundaryFitConfig() if config is None else config
    if not isinstance(selected_config, BoundaryFitConfig):
        raise TypeError("config must be a BoundaryFitConfig")
    _strict_json(canonical_json_bytes(document), schema=BOUNDARY_LOQO_SCHEMA)
    recomputed = _build_boundary_blanket_loqo(corpus, config=selected_config)
    if canonical_json_bytes(recomputed) != canonical_json_bytes(document):
        raise BoundaryBlanketIntegrityError(
            "LOQO audit differs from a complete authenticated recomputation"
        )


def evaluate_boundary_blanket_loqo(
    corpus: BoundaryCorpus,
    *,
    config: BoundaryFitConfig | None = None,
) -> dict[str, object]:
    """Build and fully verify a leave-one-query-out prompt audit."""

    selected_config = BoundaryFitConfig() if config is None else config
    document = _build_boundary_blanket_loqo(corpus, config=selected_config)
    verify_boundary_blanket_loqo(corpus, document, selected_config)
    return document


__all__ = [
    "ALL_BOUNDARY_SUBSETS",
    "BOUNDARY_CANDIDATES",
    "BOUNDARY_CANDIDATE_SCHEMA_SHA256",
    "BOUNDARY_FIT_SCHEMA",
    "BOUNDARY_HOLDOUT_SCHEMA",
    "BOUNDARY_LOQO_SCHEMA",
    "BOUNDARY_NUMERIC_ABI",
    "BOUNDARY_SELECTION_ALGORITHM",
    "BOUNDARY_TARGET",
    "FITTED_AFFINE_FAMILY",
    "FIXED_RESIDUAL_SUBSET",
    "FIXED_RESIDUAL_SUM_FAMILY",
    "BoundaryBlanketCapacityError",
    "BoundaryBlanketConditionError",
    "BoundaryBlanketError",
    "BoundaryBlanketFit",
    "BoundaryBlanketIntegrityError",
    "BoundaryCorpus",
    "BoundaryFitConfig",
    "BoundaryMacro",
    "BoundarySubsetModel",
    "DEFAULT_RIDGE_GRID",
    "evaluate_boundary_blanket_holdout",
    "evaluate_boundary_blanket_loqo",
    "fit_boundary_blanket",
    "verify_boundary_blanket_holdout",
    "verify_boundary_blanket_loqo",
]
