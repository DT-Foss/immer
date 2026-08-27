"""Continuous, receipt-bound discovery of executable numerical operators.

``MeasurementReceipt.numeric_summaries`` are intentionally never converted back
into activations.  A contextual emitter must provide the actual numerical input
and output arrays and seal their hashes together with an already-active Atlas
measurement and an authenticated revision in that Atlas's append-only history.
The harvester runs each batch under one stable current head, fits on earlier
observations in emitter order, verifies on the latest held-out observation, and
only then publishes a ``ComputeCrystal`` and an ``OperatorEdge``.

The iterator is deliberately scheduler-free.  O1 or any other idle-time runtime
can call :meth:`ContinuousOperatorHarvester.step` with a bounded batch or consume
:meth:`ContinuousOperatorHarvester.iter_steps` without giving this module a
thread, timer, or model dependency.
"""

from __future__ import annotations

import base64
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
from typing import Any, Literal, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..qwen3_8.semantic_atlas import (
    GraphRevision,
    MeasurementReceipt,
    SemanticWeightAtlas,
)
from .compute_crystals import (
    AFFINE_FLOAT64,
    CAUSAL_MIX_FLOAT64,
    MARKOV_FLOAT64,
    PERMUTATION,
    ComputeBankPublication,
    ComputeCrystal,
    ComputeCrystalBank,
    NumericalABI,
    tensor_sha256,
)
from .compute_graph import (
    ComputeOperatorGraph,
    ComputeOperatorGraphConflictError,
    OperatorEdge,
)
from .crystal import ManifestConflictError, StatePublication
from .identity import canonical_json_bytes, require_sha256

CONTEXTUAL_TRANSITION_SCHEMA = "immer-ooe-contextual-transition/v1"
HARVESTER_STATE_SCHEMA = "immer-ooe-operator-harvester-state/v1"
CANDIDATE_EVIDENCE_SCHEMA = "immer-ooe-operator-candidate-evidence/v1"
HARVESTER_VERIFIER_SCHEMA = "immer-ooe-operator-harvester-verifier/v1"
HARVESTER_STATE_PREFIX = "ooe-operator-harvester/v1:"
QWEN_CONTEXT_EMITTER_SCHEMA = "immer-ooe-qwen-context-emitter/v1"

MAX_CURSOR_BYTES = 4096
MAX_STATE_NAME_BYTES = 1024
MAX_STATE_BYTES = 48 * 1024 * 1024
MAX_CONTEXT_ARRAY_BYTES = 8 * 1024 * 1024
MAX_DIMENSION = 4096
MAX_OBSERVATIONS_PER_BATCH = 4096
MAX_GROUPS = 4096
MAX_SAMPLES_PER_GROUP = 4096
MAX_RECENT_RECEIPTS = 65_536
MAX_GRAPH_CAS_RETRIES = 32
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CONTEXT_CURSOR = re.compile(r"atlas-([0-9]{20}):([0-9a-f]{64}):([0-9a-f]{64})\Z")


class OperatorHarvesterError(RuntimeError):
    """Base error for contextual operator harvesting."""


class OperatorHarvesterIntegrityError(OperatorHarvesterError):
    """A provider receipt, Atlas binding, or persisted state failed validation."""


class OperatorHarvesterConflictError(OperatorHarvesterError):
    """A stable Atlas snapshot or bounded graph/state CAS could not be obtained."""


class OperatorHarvesterCapacityError(OperatorHarvesterError):
    """A configured hard bound was reached before accepting more evidence."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


QWEN_CONTEXT_EMITTER_SHA256 = _digest(
    {
        "input": "qwen-cartography-contextual-hidden-transition/v1",
        "output": CONTEXTUAL_TRANSITION_SCHEMA,
        "schema": QWEN_CONTEXT_EMITTER_SCHEMA,
    }
)


def _text(value: object, *, field: str, maximum: int = 1024) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > maximum
    ):
        raise ValueError(f"{field} must be canonical non-empty text")
    return value


def _cursor(value: object, *, allow_none: bool) -> str | None:
    if value is None and allow_none:
        return None
    return _text(value, field="provider cursor", maximum=MAX_CURSOR_BYTES)


def _positive_int(value: object, *, field: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= maximum
    ):
        raise ValueError(f"{field} must lie in [1, {maximum}]")
    return value


def _finite_nonnegative(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return 0.0 if result == 0.0 else result


def _strict_json(data: bytes, *, label: str, maximum: int) -> object:
    if not isinstance(data, bytes):
        raise TypeError(f"{label} must be immutable bytes")
    if len(data) > maximum:
        raise OperatorHarvesterIntegrityError(f"{label} exceeds its byte bound")

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
        raise OperatorHarvesterIntegrityError(f"{label} is not strict JSON") from exc
    if canonical_json_bytes(value) != data:
        raise OperatorHarvesterIntegrityError(f"{label} is not canonical JSON")
    return value


def _canonical_array(value: object, abi: NumericalABI, *, field: str) -> NDArray[Any]:
    array = abi.validate(value, field=field)
    if array.ndim < 1:
        raise ValueError(f"{field} must contain a feature axis")
    if not 1 <= array.shape[-1] <= MAX_DIMENSION:
        raise ValueError(f"{field} feature dimension exceeds its bound")
    if array.nbytes > MAX_CONTEXT_ARRAY_BYTES:
        raise ValueError(f"{field} exceeds its byte bound")
    dtype = np.float64 if abi.dtype == "float64" else np.int64
    result = np.array(array, dtype=dtype, order="C", copy=True)
    if result.dtype == np.dtype(np.float64):
        result[result == 0.0] = 0.0
    result.flags.writeable = False
    return result


def _array_record(array: NDArray[Any], abi: NumericalABI) -> dict[str, object]:
    value = _canonical_array(array, abi, field="context array")
    storage = "float64-le" if abi.dtype == "float64" else "int64-le"
    little = np.asarray(value, dtype="<f8" if abi.dtype == "float64" else "<i8")
    return {
        "data_base64": base64.b64encode(little.tobytes(order="C")).decode("ascii"),
        "shape": list(value.shape),
        "storage": storage,
    }


def _array_from_record(value: object, abi: NumericalABI, *, field: str) -> NDArray[Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "data_base64",
        "shape",
        "storage",
    }:
        raise OperatorHarvesterIntegrityError(f"{field} descriptor is invalid")
    storage = "float64-le" if abi.dtype == "float64" else "int64-le"
    if value.get("storage") != storage:
        raise OperatorHarvesterIntegrityError(f"{field} storage is invalid")
    shape = value.get("shape")
    if (
        not isinstance(shape, list)
        or not shape
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1
            for item in shape
        )
    ):
        raise OperatorHarvesterIntegrityError(f"{field} shape is invalid")
    count = math.prod(shape)
    if count * 8 > MAX_CONTEXT_ARRAY_BYTES:
        raise OperatorHarvesterIntegrityError(f"{field} exceeds its byte bound")
    encoded = value.get("data_base64")
    if not isinstance(encoded, str) or not encoded.isascii():
        raise OperatorHarvesterIntegrityError(f"{field} encoding is invalid")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (TypeError, ValueError) as exc:
        raise OperatorHarvesterIntegrityError(f"{field} encoding is invalid") from exc
    if len(raw) != count * 8 or base64.b64encode(raw).decode("ascii") != encoded:
        raise OperatorHarvesterIntegrityError(f"{field} bytes are not canonical")
    dtype = "<f8" if abi.dtype == "float64" else "<i8"
    decoded = np.frombuffer(raw, dtype=dtype).reshape(tuple(shape))
    result = np.array(decoded, dtype=np.float64 if abi.dtype == "float64" else np.int64)
    return _canonical_array(result, abi, field=field)


def _measurement_statistics_record(
    measurement: MeasurementReceipt,
) -> dict[str, object]:
    """Expose authenticated scalars as scalars, never as reconstructed tensors."""

    return {
        "numeric_summaries": [row.as_record() for row in measurement.numeric_summaries],
        "placebo_effects": [row.as_record() for row in measurement.placebo_effects],
    }


def _measurement_statistics_schema_record(
    measurement: MeasurementReceipt,
) -> dict[str, object]:
    return {
        "numeric_summary_metrics": [
            row.metric for row in measurement.numeric_summaries
        ],
        "placebo_effect_metrics": [row.metric for row in measurement.placebo_effects],
    }


def _runtime_family_record(measurement: MeasurementReceipt) -> dict[str, object]:
    runtime = measurement.runtime
    return {
        "code_revision": runtime.code_revision,
        "dependency_manifest_sha256": runtime.dependency_manifest_sha256,
        "model_pin_sha256": measurement.model_pin.sha256,
        "platform_sha256": runtime.platform_sha256,
        "source_manifest_sha256": runtime.source_manifest_sha256,
        "weight_rail_revision_sha256": measurement.weight_rail_revision.sha256,
    }


@dataclass(frozen=True, slots=True)
class ContextualTransitionReceipt:
    """Sealed identity for contextual arrays emitted from one Atlas measurement."""

    measurement_sha256: str
    atlas_revision: GraphRevision
    model_pin_sha256: str
    runtime_sha256: str
    runtime_family_sha256: str
    intervention_sha256: str
    intervention_mode: str
    coordinate_sha256: str
    emitter_sha256: str
    feature_schema_sha256: str
    action_schema_sha256: str
    statistics_schema_sha256: str
    statistics_sha256: str
    source_state: str
    target_state: str
    granularity: Literal["operator", "segment"] | str
    segment_start: int | None
    segment_end: int | None
    input_abi: NumericalABI
    output_abi: NumericalABI
    input_sha256: str
    output_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "measurement_sha256",
            "model_pin_sha256",
            "runtime_sha256",
            "runtime_family_sha256",
            "intervention_sha256",
            "coordinate_sha256",
            "emitter_sha256",
            "feature_schema_sha256",
            "action_schema_sha256",
            "statistics_schema_sha256",
            "statistics_sha256",
            "input_sha256",
            "output_sha256",
        ):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        if not isinstance(self.atlas_revision, GraphRevision):
            raise TypeError("atlas_revision must be a GraphRevision")
        if not isinstance(self.input_abi, NumericalABI) or not isinstance(
            self.output_abi, NumericalABI
        ):
            raise TypeError("input_abi and output_abi must be NumericalABI values")
        if self.input_abi.dtype != "float64" or self.output_abi.dtype != "float64":
            raise ValueError("operator harvesting currently requires float64 context")
        if (
            len(self.input_abi.trailing_shape) != 1
            or len(self.output_abi.trailing_shape) != 1
        ):
            raise ValueError("operator harvesting requires vector ABIs")
        object.__setattr__(
            self, "source_state", _text(self.source_state, field="source_state")
        )
        object.__setattr__(
            self, "target_state", _text(self.target_state, field="target_state")
        )
        object.__setattr__(
            self,
            "intervention_mode",
            _text(self.intervention_mode, field="intervention_mode"),
        )
        if self.granularity not in ("operator", "segment"):
            raise ValueError("granularity must be operator or segment")
        if self.granularity == "operator":
            if self.segment_start is not None or self.segment_end is not None:
                raise ValueError("operator granularity cannot carry a segment range")
        else:
            if (
                isinstance(self.segment_start, bool)
                or isinstance(self.segment_end, bool)
                or not isinstance(self.segment_start, int)
                or not isinstance(self.segment_end, int)
                or not 0 <= self.segment_start < self.segment_end
            ):
                raise ValueError("segment granularity requires a valid half-open range")
            if (
                self.segment_end - self.segment_start
                != self.input_abi.trailing_shape[0]
            ):
                raise ValueError("segment range and input ABI dimension disagree")

    def as_record(self) -> dict[str, object]:
        return {
            "action_schema_sha256": self.action_schema_sha256,
            "atlas_revision": self.atlas_revision.to_document(),
            "coordinate_sha256": self.coordinate_sha256,
            "emitter_sha256": self.emitter_sha256,
            "feature_schema_sha256": self.feature_schema_sha256,
            "granularity": self.granularity,
            "input_abi": self.input_abi.to_record(),
            "input_sha256": self.input_sha256,
            "measurement_sha256": self.measurement_sha256,
            "model_pin_sha256": self.model_pin_sha256,
            "output_abi": self.output_abi.to_record(),
            "output_sha256": self.output_sha256,
            "runtime_sha256": self.runtime_sha256,
            "runtime_family_sha256": self.runtime_family_sha256,
            "intervention_sha256": self.intervention_sha256,
            "intervention_mode": self.intervention_mode,
            "segment_end": self.segment_end,
            "segment_start": self.segment_start,
            "source_state": self.source_state,
            "statistics_schema_sha256": self.statistics_schema_sha256,
            "statistics_sha256": self.statistics_sha256,
            "target_state": self.target_state,
        }

    def to_document(self) -> dict[str, object]:
        body = self.as_record()
        return {
            "schema": CONTEXTUAL_TRANSITION_SCHEMA,
            "body": body,
            "body_sha256": _digest(body),
        }

    @property
    def sha256(self) -> str:
        return _digest(self.to_document())

    @classmethod
    def from_document(cls, value: object) -> "ContextualTransitionReceipt":
        if (
            not isinstance(value, Mapping)
            or set(value) != {"schema", "body", "body_sha256"}
            or value.get("schema") != CONTEXTUAL_TRANSITION_SCHEMA
        ):
            raise OperatorHarvesterIntegrityError("context receipt envelope is invalid")
        body = value.get("body")
        expected = {
            "action_schema_sha256",
            "atlas_revision",
            "coordinate_sha256",
            "emitter_sha256",
            "feature_schema_sha256",
            "granularity",
            "input_abi",
            "input_sha256",
            "measurement_sha256",
            "model_pin_sha256",
            "output_abi",
            "output_sha256",
            "runtime_sha256",
            "runtime_family_sha256",
            "intervention_sha256",
            "intervention_mode",
            "segment_end",
            "segment_start",
            "source_state",
            "statistics_schema_sha256",
            "statistics_sha256",
            "target_state",
        }
        if not isinstance(body, Mapping) or set(body) != expected:
            raise OperatorHarvesterIntegrityError("context receipt body is invalid")
        try:
            claimed = require_sha256(value.get("body_sha256"), field="body_sha256")
            if claimed != _digest(body):
                raise OperatorHarvesterIntegrityError(
                    "context receipt body hash mismatch"
                )
            return cls(
                measurement_sha256=cast(str, body.get("measurement_sha256")),
                atlas_revision=GraphRevision.from_document(
                    cast(Mapping[str, Any], body.get("atlas_revision"))
                ),
                model_pin_sha256=cast(str, body.get("model_pin_sha256")),
                runtime_sha256=cast(str, body.get("runtime_sha256")),
                runtime_family_sha256=cast(str, body.get("runtime_family_sha256")),
                intervention_sha256=cast(str, body.get("intervention_sha256")),
                intervention_mode=cast(str, body.get("intervention_mode")),
                coordinate_sha256=cast(str, body.get("coordinate_sha256")),
                emitter_sha256=cast(str, body.get("emitter_sha256")),
                feature_schema_sha256=cast(str, body.get("feature_schema_sha256")),
                action_schema_sha256=cast(str, body.get("action_schema_sha256")),
                source_state=cast(str, body.get("source_state")),
                target_state=cast(str, body.get("target_state")),
                statistics_schema_sha256=cast(
                    str, body.get("statistics_schema_sha256")
                ),
                statistics_sha256=cast(str, body.get("statistics_sha256")),
                granularity=cast(str, body.get("granularity")),
                segment_start=cast(int | None, body.get("segment_start")),
                segment_end=cast(int | None, body.get("segment_end")),
                input_abi=NumericalABI.from_record(body.get("input_abi")),
                output_abi=NumericalABI.from_record(body.get("output_abi")),
                input_sha256=cast(str, body.get("input_sha256")),
                output_sha256=cast(str, body.get("output_sha256")),
            )
        except OperatorHarvesterIntegrityError:
            raise
        except (TypeError, ValueError) as exc:
            raise OperatorHarvesterIntegrityError(
                "context receipt validation failed"
            ) from exc


@dataclass(frozen=True, slots=True)
class ContextualOperatorObservation:
    """Actual contextual arrays plus their independently authenticated identities."""

    measurement: MeasurementReceipt
    receipt: ContextualTransitionReceipt
    input_array: NDArray[np.float64]
    output_array: NDArray[np.float64]

    def __post_init__(self) -> None:
        if not isinstance(self.measurement, MeasurementReceipt):
            raise TypeError("measurement must be a MeasurementReceipt")
        if not isinstance(self.receipt, ContextualTransitionReceipt):
            raise TypeError("receipt must be a ContextualTransitionReceipt")
        receipt = self.receipt
        if (
            receipt.measurement_sha256 != self.measurement.sha256
            or receipt.model_pin_sha256 != self.measurement.model_pin.sha256
            or receipt.runtime_sha256 != self.measurement.runtime.sha256
            or receipt.runtime_family_sha256
            != _digest(_runtime_family_record(self.measurement))
            or receipt.intervention_sha256 != self.measurement.intervention.sha256
            or receipt.intervention_mode != self.measurement.intervention.mode
            or receipt.coordinate_sha256 != self.measurement.coordinate.sha256
            or receipt.statistics_schema_sha256
            != _digest(_measurement_statistics_schema_record(self.measurement))
            or receipt.statistics_sha256
            != _digest(_measurement_statistics_record(self.measurement))
        ):
            raise OperatorHarvesterIntegrityError(
                "context receipt differs from its MeasurementReceipt"
            )
        input_array = cast(
            NDArray[np.float64],
            _canonical_array(self.input_array, receipt.input_abi, field="input_array"),
        )
        output_array = cast(
            NDArray[np.float64],
            _canonical_array(
                self.output_array, receipt.output_abi, field="output_array"
            ),
        )
        if input_array.shape[:-1] != output_array.shape[:-1]:
            raise ValueError("context input/output leading shapes differ")
        if tensor_sha256(input_array, receipt.input_abi) != receipt.input_sha256:
            raise OperatorHarvesterIntegrityError("context input hash mismatch")
        if tensor_sha256(output_array, receipt.output_abi) != receipt.output_sha256:
            raise OperatorHarvesterIntegrityError("context output hash mismatch")
        object.__setattr__(self, "input_array", input_array)
        object.__setattr__(self, "output_array", output_array)

    @classmethod
    def capture(
        cls,
        measurement: MeasurementReceipt,
        *,
        atlas_revision: GraphRevision,
        emitter_sha256: str,
        feature_schema_sha256: str,
        action_schema_sha256: str,
        source_state: str,
        target_state: str,
        input_array: NDArray[np.float64],
        output_array: NDArray[np.float64],
        granularity: Literal["operator", "segment"] = "operator",
        segment_start: int | None = None,
        segment_end: int | None = None,
    ) -> "ContextualOperatorObservation":
        if not isinstance(measurement, MeasurementReceipt):
            raise TypeError("measurement must be a MeasurementReceipt")
        if type(input_array) is not np.ndarray or type(output_array) is not np.ndarray:
            raise TypeError("context arrays must be exact numpy.ndarray values")
        if input_array.dtype != np.dtype(np.float64) or output_array.dtype != np.dtype(
            np.float64
        ):
            raise TypeError("context arrays must be float64")
        if input_array.ndim < 1 or output_array.ndim < 1:
            raise ValueError("context arrays require a feature axis")
        input_abi = NumericalABI("float64", (int(input_array.shape[-1]),))
        output_abi = NumericalABI("float64", (int(output_array.shape[-1]),))
        receipt = ContextualTransitionReceipt(
            measurement_sha256=measurement.sha256,
            atlas_revision=atlas_revision,
            model_pin_sha256=measurement.model_pin.sha256,
            runtime_sha256=measurement.runtime.sha256,
            runtime_family_sha256=_digest(_runtime_family_record(measurement)),
            intervention_sha256=measurement.intervention.sha256,
            intervention_mode=measurement.intervention.mode,
            coordinate_sha256=measurement.coordinate.sha256,
            emitter_sha256=emitter_sha256,
            feature_schema_sha256=feature_schema_sha256,
            action_schema_sha256=action_schema_sha256,
            statistics_schema_sha256=_digest(
                _measurement_statistics_schema_record(measurement)
            ),
            statistics_sha256=_digest(_measurement_statistics_record(measurement)),
            source_state=source_state,
            target_state=target_state,
            granularity=granularity,
            segment_start=segment_start,
            segment_end=segment_end,
            input_abi=input_abi,
            output_abi=output_abi,
            input_sha256=tensor_sha256(input_array, input_abi),
            output_sha256=tensor_sha256(output_array, output_abi),
        )
        return cls(measurement, receipt, input_array, output_array)

    def to_record(self) -> dict[str, object]:
        return {
            "input_array": _array_record(self.input_array, self.receipt.input_abi),
            "measurement": self.measurement.to_document(),
            "output_array": _array_record(self.output_array, self.receipt.output_abi),
            "receipt": self.receipt.to_document(),
        }

    @classmethod
    def from_record(cls, value: object) -> "ContextualOperatorObservation":
        if not isinstance(value, Mapping) or set(value) != {
            "input_array",
            "measurement",
            "output_array",
            "receipt",
        }:
            raise OperatorHarvesterIntegrityError("stored observation is invalid")
        try:
            receipt = ContextualTransitionReceipt.from_document(value.get("receipt"))
            return cls(
                measurement=MeasurementReceipt.from_document(
                    cast(Mapping[str, Any], value.get("measurement"))
                ),
                receipt=receipt,
                input_array=cast(
                    NDArray[np.float64],
                    _array_from_record(
                        value.get("input_array"), receipt.input_abi, field="input_array"
                    ),
                ),
                output_array=cast(
                    NDArray[np.float64],
                    _array_from_record(
                        value.get("output_array"),
                        receipt.output_abi,
                        field="output_array",
                    ),
                ),
            )
        except OperatorHarvesterIntegrityError:
            raise
        except (TypeError, ValueError) as exc:
            raise OperatorHarvesterIntegrityError(
                "stored observation validation failed"
            ) from exc


@dataclass(frozen=True, slots=True)
class ContextualObservationBatch:
    """One bounded cursor page returned by a contextual emitter."""

    next_cursor: str
    observations: tuple[ContextualOperatorObservation, ...]
    exhausted: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "next_cursor",
            _text(self.next_cursor, field="next_cursor", maximum=MAX_CURSOR_BYTES),
        )
        observations = tuple(self.observations)
        if len(observations) > MAX_OBSERVATIONS_PER_BATCH or any(
            not isinstance(row, ContextualOperatorObservation) for row in observations
        ):
            raise ValueError("observation batch is invalid")
        if not isinstance(self.exhausted, bool):
            raise TypeError("exhausted must be bool")
        if len({row.receipt.sha256 for row in observations}) != len(observations):
            raise ValueError("observation batch contains duplicate receipts")
        object.__setattr__(self, "observations", observations)


class SingleBatchContextualProvider:
    """One monotonic real-probe page for immediate crash-resumable harvesting."""

    emitter_sha256 = QWEN_CONTEXT_EMITTER_SHA256

    def __init__(
        self,
        *,
        cursor: str,
        observations: Sequence[ContextualOperatorObservation],
    ) -> None:
        self.cursor = cast(
            str,
            _cursor(cursor, allow_none=False),
        )
        if _CONTEXT_CURSOR.fullmatch(self.cursor) is None:
            raise ValueError("single contextual page cursor is malformed")
        rows = tuple(observations)
        if not rows or len(rows) > MAX_OBSERVATIONS_PER_BATCH:
            raise ValueError("single contextual page must be bounded and non-empty")
        if any(row.receipt.emitter_sha256 != self.emitter_sha256 for row in rows):
            raise ValueError("single contextual page contains another emitter")
        self.observations = rows

    def poll(
        self, *, after_cursor: str | None, limit: int
    ) -> ContextualObservationBatch:
        previous = _cursor(after_cursor, allow_none=True)
        requested = _positive_int(
            limit,
            field="provider limit",
            maximum=MAX_OBSERVATIONS_PER_BATCH,
        )
        if previous is not None:
            current_match = _CONTEXT_CURSOR.fullmatch(self.cursor)
            previous_match = _CONTEXT_CURSOR.fullmatch(previous)
            if current_match is None or previous_match is None:
                raise OperatorHarvesterIntegrityError(
                    "persisted contextual cursor belongs to another provider"
                )
            if previous == self.cursor:
                return ContextualObservationBatch(previous, (), True)
            previous_generation = int(previous_match.group(1))
            current_generation = int(current_match.group(1))
            if previous_generation >= current_generation:
                raise OperatorHarvesterIntegrityError(
                    "contextual cursor conflicts with the probe page order"
                )
        if len(self.observations) > requested:
            raise OperatorHarvesterCapacityError(
                "one probe emitted more contextual transitions than the step limit"
            )
        return ContextualObservationBatch(self.cursor, self.observations, True)


def contextual_observations_from_probe_result(
    result: object,
    *,
    atlas_revision: GraphRevision,
) -> tuple[ContextualOperatorObservation, ...]:
    """Convert real Qwen projected hidden transitions into harvester evidence.

    The exact Qwen types are imported lazily so the base OoE package keeps no
    mandatory Torch import.  ``CartographyProbeResult.verify`` authenticates
    every projected array against the sealed cartography layer record before
    this function creates numerical transition receipts.
    """

    if not isinstance(atlas_revision, GraphRevision):
        raise TypeError("atlas_revision must be a GraphRevision")
    from ..qwen3_8.cartography_probe import (
        CartographyProbeResult,
        ContextualBoundarySketch,
        ContextualHiddenTransition,
    )

    if not isinstance(result, CartographyProbeResult):
        raise TypeError("result must be a verified Qwen CartographyProbeResult")
    measurement = result.measurement
    transitions = result.contextual_hidden_transitions
    boundary_sketches = result.contextual_boundary_sketches
    result.verify()
    observations = []
    for transition in transitions:
        if not isinstance(transition, ContextualHiddenTransition):
            raise OperatorHarvesterIntegrityError(
                "Qwen contextual transition is malformed"
            )
        layer = transition.layer
        seed_sha256 = transition.projection_seed_sha256
        dimensions = transition.output_dimensions
        pre_array = transition.pre_array
        post_array = transition.post_array
        feature_schema = _digest(
            {
                "dimensions": dimensions,
                "projection_seed_sha256": require_sha256(
                    seed_sha256, field="projection_seed_sha256"
                ),
                "schema": "immer-ooe-qwen-hidden-sketch-feature/v1",
            }
        )
        action_schema = _digest(
            {
                "intervention_sha256": measurement.intervention.sha256,
                "layer": layer,
                "operator": "decoder-layer-pre-sketch-to-post-sketch",
                "schema": "immer-ooe-qwen-hidden-transition-action/v1",
            }
        )
        observations.append(
            ContextualOperatorObservation.capture(
                measurement,
                atlas_revision=atlas_revision,
                emitter_sha256=QWEN_CONTEXT_EMITTER_SHA256,
                feature_schema_sha256=feature_schema,
                action_schema_sha256=action_schema,
                source_state=f"qwen.layer.{layer}.pre-hidden-sketch",
                target_state=f"qwen.layer.{layer}.post-hidden-sketch",
                input_array=cast(NDArray[np.float64], pre_array),
                output_array=cast(NDArray[np.float64], post_array),
            )
        )
    transition_by_layer = {row.layer: row for row in transitions}
    boundary_by_key: dict[tuple[int, str], ContextualBoundarySketch] = {}
    for boundary in boundary_sketches:
        if not isinstance(boundary, ContextualBoundarySketch):
            raise OperatorHarvesterIntegrityError(
                "Qwen contextual boundary sketch is malformed"
            )
        boundary_by_key[(boundary.layer, boundary.stage)] = boundary
    boundary_pairs = (
        ("attention.input", "attention.output"),
        ("layer.input", "attention.residual"),
        ("mlp.input", "mlp.output"),
        ("attention.residual", "layer.output"),
    )
    for layer, whole in sorted(transition_by_layer.items()):
        for source_stage, target_stage in boundary_pairs:
            source = boundary_by_key.get((layer, source_stage))
            target = boundary_by_key.get((layer, target_stage))
            if (source_stage != "layer.input" and source is None) or (
                target_stage != "layer.output" and target is None
            ):
                raise OperatorHarvesterIntegrityError(
                    "Qwen contextual boundary pair is incomplete"
                )
            source_array = (
                whole.pre_array if source_stage == "layer.input" else source.array
            )
            target_array = (
                whole.post_array if target_stage == "layer.output" else target.array
            )
            source_seed = (
                whole.projection_seed_sha256
                if source_stage == "layer.input"
                else source.projection_seed_sha256
            )
            target_seed = (
                whole.projection_seed_sha256
                if target_stage == "layer.output"
                else target.projection_seed_sha256
            )
            if (
                source_array.shape != target_array.shape
                or source_seed != target_seed
                or source_seed != whole.projection_seed_sha256
            ):
                raise OperatorHarvesterIntegrityError(
                    "Qwen contextual boundary pair has incompatible projections"
                )
            feature_schema = _digest(
                {
                    "dimensions": whole.output_dimensions,
                    "projection_seed_sha256": source_seed,
                    "source_stage": source_stage,
                    "target_stage": target_stage,
                    "schema": "immer-ooe-qwen-boundary-sketch-feature/v1",
                }
            )
            action_schema = _digest(
                {
                    "intervention_sha256": measurement.intervention.sha256,
                    "layer": layer,
                    "operator": "decoder-sublayer-boundary-transition",
                    "source_stage": source_stage,
                    "target_stage": target_stage,
                    "schema": "immer-ooe-qwen-boundary-transition-action/v1",
                }
            )
            observations.append(
                ContextualOperatorObservation.capture(
                    measurement,
                    atlas_revision=atlas_revision,
                    emitter_sha256=QWEN_CONTEXT_EMITTER_SHA256,
                    feature_schema_sha256=feature_schema,
                    action_schema_sha256=action_schema,
                    source_state=f"qwen.layer.{layer}.{source_stage}-sketch",
                    target_state=f"qwen.layer.{layer}.{target_stage}-sketch",
                    input_array=cast(NDArray[np.float64], source_array),
                    output_array=cast(NDArray[np.float64], target_array),
                )
            )
    return tuple(observations)


def probe_result_context_cursor(
    measurement: MeasurementReceipt,
    atlas_revision: GraphRevision,
) -> str:
    """Return a lexical cursor ordered by the append-only Atlas generation."""

    if not isinstance(measurement, MeasurementReceipt):
        raise TypeError("measurement must be a MeasurementReceipt")
    if not isinstance(atlas_revision, GraphRevision):
        raise TypeError("atlas_revision must be a GraphRevision")
    return (
        f"atlas-{atlas_revision.sequence:020d}:"
        f"{atlas_revision.event_sha256}:{measurement.sha256}"
    )


@runtime_checkable
class ContextualOperatorProvider(Protocol):
    """Replayable cursor provider implemented by O1/contextual instrumentation."""

    emitter_sha256: str

    def poll(
        self, *, after_cursor: str | None, limit: int
    ) -> ContextualObservationBatch:
        """Return only records after ``after_cursor`` and a durable next cursor."""


@dataclass(frozen=True, slots=True)
class HarvesterConfig:
    minimum_observations: int = 3
    minimum_fit_rows: int = 4
    max_observations_per_step: int = 64
    max_samples_per_group: int = 256
    max_groups: int = 1024
    max_recent_receipts: int = 4096
    affine_absolute_tolerance: float = 1e-10
    affine_relative_tolerance: float = 1e-10
    markov_absolute_tolerance: float = 1e-10
    graph_cas_retries: int = 8

    def __post_init__(self) -> None:
        _positive_int(
            self.minimum_observations,
            field="minimum_observations",
            maximum=MAX_SAMPLES_PER_GROUP,
        )
        if self.minimum_observations < 2:
            raise ValueError("minimum_observations must reserve a holdout observation")
        _positive_int(
            self.minimum_fit_rows,
            field="minimum_fit_rows",
            maximum=1_000_000,
        )
        _positive_int(
            self.max_observations_per_step,
            field="max_observations_per_step",
            maximum=MAX_OBSERVATIONS_PER_BATCH,
        )
        _positive_int(
            self.max_samples_per_group,
            field="max_samples_per_group",
            maximum=MAX_SAMPLES_PER_GROUP,
        )
        if self.max_samples_per_group < self.minimum_observations:
            raise ValueError("sample window is smaller than minimum_observations")
        _positive_int(self.max_groups, field="max_groups", maximum=MAX_GROUPS)
        _positive_int(
            self.max_recent_receipts,
            field="max_recent_receipts",
            maximum=MAX_RECENT_RECEIPTS,
        )
        _positive_int(
            self.graph_cas_retries,
            field="graph_cas_retries",
            maximum=MAX_GRAPH_CAS_RETRIES,
        )
        _finite_nonnegative(
            self.affine_absolute_tolerance, field="affine_absolute_tolerance"
        )
        _finite_nonnegative(
            self.affine_relative_tolerance, field="affine_relative_tolerance"
        )
        _finite_nonnegative(
            self.markov_absolute_tolerance, field="markov_absolute_tolerance"
        )

    def as_record(self) -> dict[str, object]:
        return {
            "affine_absolute_tolerance": self.affine_absolute_tolerance,
            "affine_relative_tolerance": self.affine_relative_tolerance,
            "graph_cas_retries": self.graph_cas_retries,
            "markov_absolute_tolerance": self.markov_absolute_tolerance,
            "max_groups": self.max_groups,
            "max_observations_per_step": self.max_observations_per_step,
            "max_recent_receipts": self.max_recent_receipts,
            "max_samples_per_group": self.max_samples_per_group,
            "minimum_fit_rows": self.minimum_fit_rows,
            "minimum_observations": self.minimum_observations,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())


def harvester_identity_sha256(
    *,
    model_pin_sha256: str,
    config: HarvesterConfig,
    emitter_sha256: str = QWEN_CONTEXT_EMITTER_SHA256,
) -> str:
    """Return the persistent harvester identity without opening its provider."""

    if not isinstance(config, HarvesterConfig):
        raise TypeError("config must be a HarvesterConfig")
    return _digest(
        {
            "atlas_model_pin_sha256": require_sha256(
                model_pin_sha256, field="model_pin_sha256"
            ),
            "config_sha256": config.sha256,
            "emitter_sha256": require_sha256(emitter_sha256, field="emitter_sha256"),
            "schema": HARVESTER_STATE_SCHEMA,
        }
    )


def harvester_state_name(
    *,
    model_pin_sha256: str,
    config: HarvesterConfig,
    emitter_sha256: str = QWEN_CONTEXT_EMITTER_SHA256,
) -> str:
    identity = harvester_identity_sha256(
        model_pin_sha256=model_pin_sha256,
        config=config,
        emitter_sha256=emitter_sha256,
    )
    return f"{HARVESTER_STATE_PREFIX}{identity}"


def _group_record(receipt: ContextualTransitionReceipt) -> dict[str, object]:
    return {
        "action_schema_sha256": receipt.action_schema_sha256,
        "feature_schema_sha256": receipt.feature_schema_sha256,
        "granularity": receipt.granularity,
        "input_abi_sha256": receipt.input_abi.sha256,
        "intervention_mode": receipt.intervention_mode,
        "model_pin_sha256": receipt.model_pin_sha256,
        "output_abi_sha256": receipt.output_abi.sha256,
        "runtime_family_sha256": receipt.runtime_family_sha256,
        "segment_end": receipt.segment_end,
        "segment_start": receipt.segment_start,
        "source_state": receipt.source_state,
        "statistics_schema_sha256": receipt.statistics_schema_sha256,
        "target_state": receipt.target_state,
    }


def _group_sha256(receipt: ContextualTransitionReceipt) -> str:
    return _digest(_group_record(receipt))


@dataclass(frozen=True, slots=True)
class PromotionCheckpoint:
    group_sha256: str
    operator_kind: str
    crystal_sha256: str
    evidence_sha256: str
    edge_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "group_sha256",
            require_sha256(self.group_sha256, field="group_sha256"),
        )
        if self.operator_kind not in (
            AFFINE_FLOAT64,
            PERMUTATION,
            MARKOV_FLOAT64,
            CAUSAL_MIX_FLOAT64,
        ):
            raise ValueError("checkpoint operator kind is invalid")
        for field in ("crystal_sha256", "evidence_sha256", "edge_sha256"):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )

    @property
    def key(self) -> str:
        return f"{self.group_sha256}:{self.operator_kind}"

    def to_record(self) -> dict[str, object]:
        return {
            "crystal_sha256": self.crystal_sha256,
            "edge_sha256": self.edge_sha256,
            "evidence_sha256": self.evidence_sha256,
            "group_sha256": self.group_sha256,
            "operator_kind": self.operator_kind,
        }

    @classmethod
    def from_record(cls, value: object) -> "PromotionCheckpoint":
        if not isinstance(value, Mapping) or set(value) != {
            "crystal_sha256",
            "edge_sha256",
            "evidence_sha256",
            "group_sha256",
            "operator_kind",
        }:
            raise OperatorHarvesterIntegrityError("promotion checkpoint is invalid")
        try:
            return cls(**dict(value))
        except (TypeError, ValueError) as exc:
            raise OperatorHarvesterIntegrityError(
                "promotion checkpoint validation failed"
            ) from exc


@dataclass(frozen=True, slots=True)
class HarvesterState:
    identity_sha256: str
    provider_cursor: str | None
    groups: tuple[tuple[str, tuple[ContextualOperatorObservation, ...]], ...]
    promotions: tuple[PromotionCheckpoint, ...]
    recent_receipt_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "identity_sha256",
            require_sha256(self.identity_sha256, field="identity_sha256"),
        )
        object.__setattr__(
            self, "provider_cursor", _cursor(self.provider_cursor, allow_none=True)
        )
        groups = tuple(self.groups)
        if len(groups) > MAX_GROUPS:
            raise ValueError("harvester state has too many groups")
        prior_group = ""
        for group_sha, observations in groups:
            require_sha256(group_sha, field="group_sha256")
            if group_sha <= prior_group:
                raise ValueError("harvester groups must be sorted and unique")
            prior_group = group_sha
            rows = tuple(observations)
            if not rows or len(rows) > MAX_SAMPLES_PER_GROUP:
                raise ValueError("harvester group sample inventory is invalid")
            if any(not isinstance(row, ContextualOperatorObservation) for row in rows):
                raise TypeError("harvester group contains invalid observations")
            if any(_group_sha256(row.receipt) != group_sha for row in rows):
                raise ValueError("harvester group identity mismatch")
            if len({row.receipt.sha256 for row in rows}) != len(rows):
                raise ValueError("harvester group contains duplicate observations")
            if len({row.measurement.probe.prompt_signature for row in rows}) != len(
                rows
            ):
                raise ValueError("harvester group contains duplicate prompt contexts")
        promotions = tuple(self.promotions)
        if tuple(sorted(promotions, key=lambda row: row.key)) != promotions or len(
            {row.key for row in promotions}
        ) != len(promotions):
            raise ValueError("promotion checkpoints must be sorted and unique")
        recent = tuple(
            require_sha256(value, field="recent_receipt_sha256s")
            for value in self.recent_receipt_sha256s
        )
        if len(recent) > MAX_RECENT_RECEIPTS or len(set(recent)) != len(recent):
            raise ValueError("recent receipt inventory is invalid")
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "promotions", promotions)
        object.__setattr__(self, "recent_receipt_sha256s", recent)

    @classmethod
    def empty(cls, identity_sha256: str) -> "HarvesterState":
        return cls(identity_sha256, None, (), (), ())

    def to_document(self) -> dict[str, object]:
        body = {
            "groups": [
                {
                    "group_sha256": group_sha,
                    "observations": [row.to_record() for row in observations],
                }
                for group_sha, observations in self.groups
            ],
            "identity_sha256": self.identity_sha256,
            "promotions": [row.to_record() for row in self.promotions],
            "provider_cursor": self.provider_cursor,
            "recent_receipt_sha256s": list(self.recent_receipt_sha256s),
        }
        return {
            "schema": HARVESTER_STATE_SCHEMA,
            "body": body,
            "body_sha256": _digest(body),
        }

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_document())
        if len(data) > MAX_STATE_BYTES:
            raise OperatorHarvesterCapacityError(
                "harvester state exceeds its persistent byte bound"
            )
        return data

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "HarvesterState":
        value = _strict_json(
            data, label="operator harvester state", maximum=MAX_STATE_BYTES
        )
        if (
            not isinstance(value, Mapping)
            or set(value) != {"schema", "body", "body_sha256"}
            or value.get("schema") != HARVESTER_STATE_SCHEMA
        ):
            raise OperatorHarvesterIntegrityError("harvester state envelope is invalid")
        body = value.get("body")
        if not isinstance(body, Mapping) or set(body) != {
            "groups",
            "identity_sha256",
            "promotions",
            "provider_cursor",
            "recent_receipt_sha256s",
        }:
            raise OperatorHarvesterIntegrityError("harvester state body is invalid")
        if value.get("body_sha256") != _digest(body):
            raise OperatorHarvesterIntegrityError("harvester state hash mismatch")
        raw_groups = body.get("groups")
        raw_promotions = body.get("promotions")
        raw_recent = body.get("recent_receipt_sha256s")
        if (
            not isinstance(raw_groups, list)
            or not isinstance(raw_promotions, list)
            or not isinstance(raw_recent, list)
        ):
            raise OperatorHarvesterIntegrityError("harvester inventories are invalid")
        groups: list[tuple[str, tuple[ContextualOperatorObservation, ...]]] = []
        try:
            for group in raw_groups:
                if not isinstance(group, Mapping) or set(group) != {
                    "group_sha256",
                    "observations",
                }:
                    raise OperatorHarvesterIntegrityError(
                        "stored harvester group is invalid"
                    )
                observations = group.get("observations")
                if not isinstance(observations, list):
                    raise OperatorHarvesterIntegrityError(
                        "stored group observations are invalid"
                    )
                groups.append(
                    (
                        cast(str, group.get("group_sha256")),
                        tuple(
                            ContextualOperatorObservation.from_record(row)
                            for row in observations
                        ),
                    )
                )
            state = cls(
                identity_sha256=cast(str, body.get("identity_sha256")),
                provider_cursor=cast(str | None, body.get("provider_cursor")),
                groups=tuple(groups),
                promotions=tuple(
                    PromotionCheckpoint.from_record(row) for row in raw_promotions
                ),
                recent_receipt_sha256s=tuple(raw_recent),
            )
        except OperatorHarvesterIntegrityError:
            raise
        except (TypeError, ValueError) as exc:
            raise OperatorHarvesterIntegrityError(
                "harvester state validation failed"
            ) from exc
        if state.to_bytes() != data:
            raise OperatorHarvesterIntegrityError(
                "harvester state failed canonical reconstruction"
            )
        return state


@dataclass(frozen=True, slots=True)
class HarvestRejection:
    observation_receipt_sha256: str
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_receipt_sha256",
            require_sha256(
                self.observation_receipt_sha256,
                field="observation_receipt_sha256",
            ),
        )
        object.__setattr__(self, "reason", _text(self.reason, field="reason"))


@dataclass(frozen=True, slots=True)
class CandidateEvidenceStream:
    group_sha256: str
    operator_kind: str
    status: Literal["pending", "rejected", "promoted", "verified-existing"] | str
    reason: str
    observation_receipt_sha256s: tuple[str, ...]
    fit_receipt_sha256s: tuple[str, ...]
    holdout_receipt_sha256: str | None
    verifier_sha256: str
    crystal_sha256: str | None = None
    evidence_sha256: str | None = None

    def __post_init__(self) -> None:
        require_sha256(self.group_sha256, field="group_sha256")
        require_sha256(self.verifier_sha256, field="verifier_sha256")
        if self.operator_kind not in (
            AFFINE_FLOAT64,
            PERMUTATION,
            MARKOV_FLOAT64,
            CAUSAL_MIX_FLOAT64,
        ):
            raise ValueError("candidate operator kind is invalid")
        if self.status not in ("pending", "rejected", "promoted", "verified-existing"):
            raise ValueError("candidate status is invalid")
        _text(self.reason, field="candidate reason")
        receipts = tuple(
            require_sha256(value, field="observation_receipt_sha256s")
            for value in self.observation_receipt_sha256s
        )
        fit = tuple(
            require_sha256(value, field="fit_receipt_sha256s")
            for value in self.fit_receipt_sha256s
        )
        if tuple(sorted(receipts)) != receipts or tuple(sorted(fit)) != fit:
            raise ValueError("candidate receipt identities must be sorted")
        if self.holdout_receipt_sha256 is not None:
            require_sha256(self.holdout_receipt_sha256, field="holdout_receipt_sha256")
        if self.crystal_sha256 is not None:
            require_sha256(self.crystal_sha256, field="crystal_sha256")
        if self.evidence_sha256 is not None:
            require_sha256(self.evidence_sha256, field="evidence_sha256")


@dataclass(frozen=True, slots=True)
class HarvestPromotion:
    candidate: CandidateEvidenceStream
    edge: OperatorEdge
    bank_publication: ComputeBankPublication
    graph_changed: bool
    graph_state_sha256: str


@dataclass(frozen=True, slots=True)
class HarvestStepResult:
    atlas_revision: GraphRevision
    cursor_before: str | None
    cursor_after: str
    exhausted: bool
    accepted_observations: int
    rejections: tuple[HarvestRejection, ...]
    candidates: tuple[CandidateEvidenceStream, ...]
    promotions: tuple[HarvestPromotion, ...]
    state_sha256: str
    state_publication: StatePublication
    graph_state_sha256: str


@dataclass(frozen=True, slots=True)
class _FittedCandidate:
    group_sha256: str
    operator_kind: str
    crystal: ComputeCrystal
    evidence_sha256: str
    verifier_sha256: str
    observation_receipt_sha256s: tuple[str, ...]
    fit_receipt_sha256s: tuple[str, ...]
    holdout_receipt_sha256: str
    source_state: str
    target_state: str


class ContinuousOperatorHarvester:
    """Bounded, crash-resumable O1/Atlas-to-ComputeGraph operator harvester."""

    def __init__(
        self,
        *,
        atlas: SemanticWeightAtlas,
        provider: ContextualOperatorProvider,
        graph: ComputeOperatorGraph,
        config: HarvesterConfig | None = None,
        state_name: str | None = None,
    ) -> None:
        if not isinstance(atlas, SemanticWeightAtlas):
            raise TypeError("atlas must be a SemanticWeightAtlas")
        if not isinstance(graph, ComputeOperatorGraph):
            raise TypeError("graph must be a ComputeOperatorGraph")
        if not isinstance(provider, ContextualOperatorProvider):
            raise TypeError("provider must implement ContextualOperatorProvider")
        emitter_sha256 = require_sha256(
            provider.emitter_sha256, field="provider.emitter_sha256"
        )
        selected_config = HarvesterConfig() if config is None else config
        if not isinstance(selected_config, HarvesterConfig):
            raise TypeError("config must be HarvesterConfig")
        self.atlas = atlas
        self.provider = provider
        self.graph = graph
        self.bank: ComputeCrystalBank = graph.bank
        self.config = selected_config
        self.emitter_sha256 = emitter_sha256
        self.identity_sha256 = harvester_identity_sha256(
            model_pin_sha256=atlas.model_pin.sha256,
            config=selected_config,
            emitter_sha256=emitter_sha256,
        )
        default_name = f"{HARVESTER_STATE_PREFIX}{self.identity_sha256}"
        self.state_name = _text(
            default_name if state_name is None else state_name,
            field="state_name",
            maximum=MAX_STATE_NAME_BYTES,
        )
        self.verifier_sha256 = _digest(
            {
                "affine_fit": "deterministic-lstsq-full-rank+held-out-allclose",
                "config_sha256": selected_config.sha256,
                "markov_fit": "deterministic-lstsq-stochastic-kernel+held-out-allclose",
                "permutation_fit": "unique-exact-column-bijection+held-out-array-equality",
                "scalar_summary_reconstruction": False,
                "schema": HARVESTER_VERIFIER_SCHEMA,
            }
        )
        lock_digest = hashlib.sha256(self.state_name.encode("utf-8")).hexdigest()
        self._lock_path = (
            Path(self.bank.root) / f".operator-harvester-{lock_digest}.lock"
        )

    @contextmanager
    def _locked(self) -> Iterator[None]:
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(self._lock_path, flags, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _load_state(self) -> tuple[HarvesterState, str | None]:
        try:
            data = self.bank.store.restore_state(self.state_name)
        except KeyError:
            return HarvesterState.empty(self.identity_sha256), None
        state = HarvesterState.from_bytes(data)
        if state.identity_sha256 != self.identity_sha256:
            raise OperatorHarvesterIntegrityError(
                "persisted harvester identity differs from this runtime"
            )
        return state, hashlib.sha256(data).hexdigest()

    def state(self) -> HarvesterState:
        with self._locked():
            return self._load_state()[0]

    def _observation_rejection(
        self,
        observation: ContextualOperatorObservation,
        *,
        recent: set[str],
        measurements_by_coordinate: dict[str, dict[str, MeasurementReceipt]],
        measurement_ids_by_group: Mapping[str, set[str]],
        prompt_signatures_by_group: Mapping[str, set[str]],
    ) -> str | None:
        receipt = observation.receipt
        if receipt.emitter_sha256 != self.emitter_sha256:
            return "emitter-identity-mismatch"
        # A continuous emitter naturally produces receipts at successive Atlas
        # heads.  Requiring every backlog item to equal the newest head would
        # discard all but the final measurement of a real O1 run.  Historical
        # membership plus active-byte equality authenticates the observation;
        # the outer step still pins one stable current head before and after
        # the complete batch.
        if not self.atlas.contains_revision(receipt.atlas_revision):
            return "atlas-revision-outside-history"
        if observation.measurement.observation_status == "invalidated":
            return "invalidated-measurement"
        if observation.measurement.intervention.mode == "placebo":
            return "placebo-measurement"
        if receipt.sha256 in recent:
            return "duplicate-context-receipt"
        group_sha = _group_sha256(receipt)
        if observation.measurement.sha256 in measurement_ids_by_group.get(
            group_sha, set()
        ):
            return "duplicate-measurement-in-group"
        if observation.measurement.probe.prompt_signature in (
            prompt_signatures_by_group.get(group_sha, set())
        ):
            return "duplicate-prompt-in-group"
        coordinate = receipt.coordinate_sha256
        active = measurements_by_coordinate.get(coordinate)
        if active is None:
            result = self.atlas.query_by_coordinate(observation.measurement.coordinate)
            active = {row.sha256: row for row in result.measurements}
            measurements_by_coordinate[coordinate] = active
        matched = active.get(observation.measurement.sha256)
        if (
            matched is None
            or matched.to_document() != observation.measurement.to_document()
        ):
            return "measurement-not-active-in-atlas"
        return None

    def _assert_stored_group_is_active(
        self,
        observations: Sequence[ContextualOperatorObservation],
        *,
        measurements_by_coordinate: dict[str, dict[str, MeasurementReceipt]],
    ) -> None:
        """Rebind persisted samples to this Atlas before they influence a new fit."""

        for observation in observations:
            receipt = observation.receipt
            if not self.atlas.contains_revision(receipt.atlas_revision):
                raise OperatorHarvesterIntegrityError(
                    "stored context names a revision outside this Atlas history"
                )
            active = measurements_by_coordinate.get(receipt.coordinate_sha256)
            if active is None:
                result = self.atlas.query_by_coordinate(
                    observation.measurement.coordinate
                )
                active = {row.sha256: row for row in result.measurements}
                measurements_by_coordinate[receipt.coordinate_sha256] = active
            matched = active.get(observation.measurement.sha256)
            if (
                matched is None
                or matched.to_document() != observation.measurement.to_document()
            ):
                raise OperatorHarvesterIntegrityError(
                    "stored contextual measurement is no longer active in the Atlas"
                )

    @staticmethod
    def _flatten(
        observations: Sequence[ContextualOperatorObservation],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        inputs = np.concatenate(
            [
                row.input_array.reshape(-1, row.input_array.shape[-1])
                for row in observations
            ],
            axis=0,
        )
        outputs = np.concatenate(
            [
                row.output_array.reshape(-1, row.output_array.shape[-1])
                for row in observations
            ],
            axis=0,
        )
        return (
            cast(NDArray[np.float64], np.asarray(inputs, dtype=np.float64)),
            cast(NDArray[np.float64], np.asarray(outputs, dtype=np.float64)),
        )

    def _extensions(
        self,
        *,
        group_sha256: str,
        operator_kind: str,
        observations: Sequence[ContextualOperatorObservation],
        fit_receipts: Sequence[str],
        holdout_receipt: str,
    ) -> dict[str, object]:
        first = observations[0].receipt
        return {
            "operator_harvester": {
                "action_schema_sha256": first.action_schema_sha256,
                "coordinate_sha256": first.coordinate_sha256,
                "coordinate_sha256s": sorted(
                    row.receipt.coordinate_sha256 for row in observations
                ),
                "feature_schema_sha256": first.feature_schema_sha256,
                "fit_receipt_sha256s": list(fit_receipts),
                "granularity": first.granularity,
                "group_sha256": group_sha256,
                "holdout_receipt_sha256": holdout_receipt,
                "measurement_sha256s": sorted(
                    row.measurement.sha256 for row in observations
                ),
                "model_pin_sha256": first.model_pin_sha256,
                "intervention_mode": first.intervention_mode,
                "intervention_sha256": first.intervention_sha256,
                "operator_kind": operator_kind,
                "runtime_sha256": first.runtime_sha256,
                "runtime_family_sha256": first.runtime_family_sha256,
                "runtime_sha256s": sorted(
                    row.receipt.runtime_sha256 for row in observations
                ),
                "schema": CANDIDATE_EVIDENCE_SCHEMA,
                "statistics_schema_sha256": first.statistics_schema_sha256,
                "statistics_sha256s": sorted(
                    row.receipt.statistics_sha256 for row in observations
                ),
                "verifier_sha256": self.verifier_sha256,
            }
        }

    def _fit_affine(
        self,
        group_sha: str,
        fit: Sequence[ContextualOperatorObservation],
        holdout: ContextualOperatorObservation,
        all_observations: Sequence[ContextualOperatorObservation],
    ) -> tuple[ComputeCrystal | None, str]:
        x_fit, y_fit = self._flatten(fit)
        required_rows = max(self.config.minimum_fit_rows, x_fit.shape[1] + 1)
        if x_fit.shape[0] < required_rows:
            return None, f"affine-needs-{required_rows}-fit-rows"
        augmented = np.concatenate(
            (x_fit, np.ones((x_fit.shape[0], 1), dtype=np.float64)), axis=1
        )
        coefficients, _residuals, rank, _singular = np.linalg.lstsq(
            augmented, y_fit, rcond=None
        )
        if int(rank) != augmented.shape[1]:
            return None, "affine-fit-design-not-full-rank"
        matrix = coefficients[:-1].T
        bias = coefficients[-1]
        receipts = tuple(sorted(row.receipt.sha256 for row in fit))
        extension = self._extensions(
            group_sha256=group_sha,
            operator_kind=AFFINE_FLOAT64,
            observations=all_observations,
            fit_receipts=receipts,
            holdout_receipt=holdout.receipt.sha256,
        )
        crystal = ComputeCrystal.affine(matrix, bias, extensions=extension)
        train_prediction = crystal.apply(x_fit)
        x_holdout, y_holdout = self._flatten((holdout,))
        holdout_prediction = crystal.apply(x_holdout)
        kwargs = {
            "rtol": self.config.affine_relative_tolerance,
            "atol": self.config.affine_absolute_tolerance,
        }
        if not np.allclose(train_prediction, y_fit, **kwargs):
            return None, "affine-train-residual-exceeds-tolerance"
        if not np.allclose(holdout_prediction, y_holdout, **kwargs):
            return None, "affine-holdout-residual-exceeds-tolerance"
        return crystal, "affine-held-out-verification-passed"

    def _fit_permutation(
        self,
        group_sha: str,
        fit: Sequence[ContextualOperatorObservation],
        holdout: ContextualOperatorObservation,
        all_observations: Sequence[ContextualOperatorObservation],
    ) -> tuple[ComputeCrystal | None, str]:
        x_fit, y_fit = self._flatten(fit)
        if x_fit.shape[1] != y_fit.shape[1]:
            return None, "permutation-requires-equal-dimensions"
        if x_fit.shape[0] < self.config.minimum_fit_rows:
            return None, f"permutation-needs-{self.config.minimum_fit_rows}-fit-rows"
        indices: list[int] = []
        for output_index in range(y_fit.shape[1]):
            matches = [
                input_index
                for input_index in range(x_fit.shape[1])
                if np.array_equal(y_fit[:, output_index], x_fit[:, input_index])
            ]
            if len(matches) != 1:
                return None, "permutation-column-match-is-not-unique"
            indices.append(matches[0])
        if len(set(indices)) != len(indices):
            return None, "permutation-column-map-is-not-bijective"
        receipts = tuple(sorted(row.receipt.sha256 for row in fit))
        extension = self._extensions(
            group_sha256=group_sha,
            operator_kind=PERMUTATION,
            observations=all_observations,
            fit_receipts=receipts,
            holdout_receipt=holdout.receipt.sha256,
        )
        crystal = ComputeCrystal.permutation(indices, extensions=extension)
        x_holdout, y_holdout = self._flatten((holdout,))
        if not np.array_equal(crystal.apply(x_fit), y_fit):
            return None, "permutation-train-verification-failed"
        if not np.array_equal(crystal.apply(x_holdout), y_holdout):
            return None, "permutation-holdout-verification-failed"
        return crystal, "exact-permutation-held-out-verification-passed"

    def _fit_markov(
        self,
        group_sha: str,
        fit: Sequence[ContextualOperatorObservation],
        holdout: ContextualOperatorObservation,
        all_observations: Sequence[ContextualOperatorObservation],
    ) -> tuple[ComputeCrystal | None, str]:
        x_fit, y_fit = self._flatten(fit)
        if x_fit.shape[1] != y_fit.shape[1]:
            return None, "markov-requires-equal-dimensions"
        dimension = x_fit.shape[1]
        required_rows = max(self.config.minimum_fit_rows, dimension)
        if x_fit.shape[0] < required_rows:
            return None, f"markov-needs-{required_rows}-fit-rows"
        tolerance = self.config.markov_absolute_tolerance
        if (
            np.any(x_fit < -tolerance)
            or np.any(y_fit < -tolerance)
            or not np.allclose(x_fit.sum(axis=1), 1.0, rtol=0.0, atol=tolerance)
            or not np.allclose(y_fit.sum(axis=1), 1.0, rtol=0.0, atol=tolerance)
        ):
            return None, "markov-context-is-not-row-stochastic"
        kernel, _residuals, rank, _singular = np.linalg.lstsq(x_fit, y_fit, rcond=None)
        if int(rank) != dimension:
            return None, "markov-fit-design-not-full-rank"
        if np.any(kernel < -tolerance):
            return None, "markov-kernel-has-negative-mass"
        canonical_kernel = np.maximum(kernel, 0.0)
        row_sums = canonical_kernel.sum(axis=1)
        if np.any(row_sums <= 0.0) or not np.allclose(
            row_sums, 1.0, rtol=0.0, atol=tolerance
        ):
            return None, "markov-kernel-rows-do-not-sum-to-one"
        canonical_kernel = canonical_kernel / row_sums[:, None]
        receipts = tuple(sorted(row.receipt.sha256 for row in fit))
        extension = self._extensions(
            group_sha256=group_sha,
            operator_kind=MARKOV_FLOAT64,
            observations=all_observations,
            fit_receipts=receipts,
            holdout_receipt=holdout.receipt.sha256,
        )
        crystal = ComputeCrystal.markov(canonical_kernel, extensions=extension)
        x_holdout, y_holdout = self._flatten((holdout,))
        if not np.allclose(crystal.apply(x_fit), y_fit, rtol=0.0, atol=tolerance):
            return None, "markov-train-residual-exceeds-tolerance"
        if not np.allclose(
            crystal.apply(x_holdout), y_holdout, rtol=0.0, atol=tolerance
        ):
            return None, "markov-holdout-residual-exceeds-tolerance"
        return crystal, "markov-held-out-verification-passed"

    def _candidate(
        self,
        group_sha: str,
        observations: Sequence[ContextualOperatorObservation],
        operator_kind: str,
    ) -> tuple[CandidateEvidenceStream, _FittedCandidate | None]:
        # HarvesterState preserves the provider cursor order.  Using digest
        # order here would silently turn the temporal holdout into a random
        # hash split and could validate a candidate on an older sample while
        # training on newer evidence.
        ordered = tuple(observations)
        # Inventory order is canonical; temporal role is carried explicitly by
        # fit_receipt_sha256s and holdout_receipt_sha256.
        receipt_shas = tuple(sorted(row.receipt.sha256 for row in ordered))
        if len(ordered) < self.config.minimum_observations:
            return (
                CandidateEvidenceStream(
                    group_sha,
                    operator_kind,
                    "pending",
                    f"needs-{self.config.minimum_observations}-observations",
                    receipt_shas,
                    (),
                    None,
                    self.verifier_sha256,
                ),
                None,
            )
        fit = ordered[:-1]
        holdout = ordered[-1]
        if operator_kind == AFFINE_FLOAT64:
            crystal, reason = self._fit_affine(group_sha, fit, holdout, ordered)
        elif operator_kind == PERMUTATION:
            crystal, reason = self._fit_permutation(group_sha, fit, holdout, ordered)
        elif operator_kind == MARKOV_FLOAT64:
            crystal, reason = self._fit_markov(group_sha, fit, holdout, ordered)
        else:
            raise AssertionError("candidate operator kind changed")
        fit_shas = tuple(sorted(row.receipt.sha256 for row in fit))
        if crystal is None:
            return (
                CandidateEvidenceStream(
                    group_sha,
                    operator_kind,
                    "rejected",
                    reason,
                    receipt_shas,
                    fit_shas,
                    holdout.receipt.sha256,
                    self.verifier_sha256,
                ),
                None,
            )
        evidence = _digest(
            {
                "crystal_sha256": crystal.sha256,
                "fit_receipt_sha256s": list(fit_shas),
                "group_sha256": group_sha,
                "holdout_receipt_sha256": holdout.receipt.sha256,
                "observation_receipt_sha256s": list(receipt_shas),
                "operator_kind": operator_kind,
                "schema": CANDIDATE_EVIDENCE_SCHEMA,
                "verifier_sha256": self.verifier_sha256,
            }
        )
        stream = CandidateEvidenceStream(
            group_sha,
            operator_kind,
            "promoted",
            reason,
            receipt_shas,
            fit_shas,
            holdout.receipt.sha256,
            self.verifier_sha256,
            crystal.sha256,
            evidence,
        )
        first = ordered[0].receipt
        fitted = _FittedCandidate(
            group_sha,
            operator_kind,
            crystal,
            evidence,
            self.verifier_sha256,
            receipt_shas,
            fit_shas,
            holdout.receipt.sha256,
            first.source_state,
            first.target_state,
        )
        return stream, fitted

    def _append_edges_cas(self, edges: Sequence[OperatorEdge]) -> tuple[str, bool]:
        additions = tuple(edges)
        if not additions:
            state = self.graph.state()
            return state.sha256, False
        for _attempt in range(self.config.graph_cas_retries):
            state = self.graph.state()
            try:
                updated, changed = self.graph.append_edges(
                    additions,
                    expected_generation=state.generation,
                    expected_state_sha256=state.sha256,
                )
            except ComputeOperatorGraphConflictError:
                continue
            return updated.sha256, changed
        raise OperatorHarvesterConflictError(
            "operator graph exhausted the harvester CAS retry bound"
        )

    def step(self, *, limit: int | None = None) -> HarvestStepResult:
        """Consume one bounded provider page and atomically persist its cursor."""

        selected_limit = (
            self.config.max_observations_per_step if limit is None else limit
        )
        _positive_int(
            selected_limit,
            field="step limit",
            maximum=self.config.max_observations_per_step,
        )
        with self._locked():
            state, prior_state_sha = self._load_state()
            self.atlas.verify_or_raise()
            atlas_revision = self.atlas.revision()
            batch = self.provider.poll(
                after_cursor=state.provider_cursor, limit=selected_limit
            )
            if not isinstance(batch, ContextualObservationBatch):
                raise TypeError("provider.poll must return ContextualObservationBatch")
            if len(batch.observations) > selected_limit:
                raise OperatorHarvesterCapacityError(
                    "provider returned more observations than requested"
                )
            if batch.observations and batch.next_cursor == state.provider_cursor:
                raise OperatorHarvesterIntegrityError(
                    "a non-empty provider page did not advance its cursor"
                )

            groups = {group_sha: list(rows) for group_sha, rows in state.groups}
            promotions_by_key = {row.key: row for row in state.promotions}
            recent_order = list(state.recent_receipt_sha256s)
            recent = set(recent_order)
            measurement_ids_by_group: dict[str, set[str]] = {
                group_sha: {row.measurement.sha256 for row in rows}
                for group_sha, rows in groups.items()
            }
            prompt_signatures_by_group: dict[str, set[str]] = {
                group_sha: {row.measurement.probe.prompt_signature for row in rows}
                for group_sha, rows in groups.items()
            }
            measurements_by_coordinate: dict[str, dict[str, MeasurementReceipt]] = {}
            accepted = 0
            rejections: list[HarvestRejection] = []
            affected_groups: set[str] = set()
            for observation in batch.observations:
                reason = self._observation_rejection(
                    observation,
                    recent=recent,
                    measurements_by_coordinate=measurements_by_coordinate,
                    measurement_ids_by_group=measurement_ids_by_group,
                    prompt_signatures_by_group=prompt_signatures_by_group,
                )
                if reason is not None:
                    rejections.append(
                        HarvestRejection(observation.receipt.sha256, reason)
                    )
                    continue
                group_sha = _group_sha256(observation.receipt)
                if group_sha not in groups and len(groups) >= self.config.max_groups:
                    rejections.append(
                        HarvestRejection(
                            observation.receipt.sha256, "group-capacity-reached"
                        )
                    )
                    continue
                rows = groups.setdefault(group_sha, [])
                rows.append(observation)
                if len(rows) > self.config.max_samples_per_group:
                    del rows[: len(rows) - self.config.max_samples_per_group]
                measurement_ids_by_group.setdefault(group_sha, set()).add(
                    observation.measurement.sha256
                )
                prompt_signatures_by_group.setdefault(group_sha, set()).add(
                    observation.measurement.probe.prompt_signature
                )
                recent.add(observation.receipt.sha256)
                recent_order.append(observation.receipt.sha256)
                if len(recent_order) > self.config.max_recent_receipts:
                    removed = recent_order.pop(0)
                    recent.discard(removed)
                affected_groups.add(group_sha)
                accepted += 1

            if self.atlas.revision() != atlas_revision:
                raise OperatorHarvesterConflictError(
                    "semantic Atlas changed during contextual authentication"
                )

            candidate_streams: list[CandidateEvidenceStream] = []
            fitted_candidates: list[_FittedCandidate] = []
            for group_sha in sorted(affected_groups):
                observations = tuple(groups[group_sha])
                self._assert_stored_group_is_active(
                    observations,
                    measurements_by_coordinate=measurements_by_coordinate,
                )
                for operator_kind in (AFFINE_FLOAT64, PERMUTATION, MARKOV_FLOAT64):
                    stream, fitted = self._candidate(
                        group_sha, observations, operator_kind
                    )
                    if fitted is not None:
                        checkpoint = promotions_by_key.get(
                            f"{group_sha}:{operator_kind}"
                        )
                        if checkpoint is not None and (
                            checkpoint.crystal_sha256 == fitted.crystal.sha256
                        ):
                            stream = CandidateEvidenceStream(
                                stream.group_sha256,
                                stream.operator_kind,
                                "verified-existing",
                                "held-out-verification-matches-published-crystal",
                                stream.observation_receipt_sha256s,
                                stream.fit_receipt_sha256s,
                                stream.holdout_receipt_sha256,
                                stream.verifier_sha256,
                                stream.crystal_sha256,
                                stream.evidence_sha256,
                            )
                        else:
                            fitted_candidates.append(fitted)
                    candidate_streams.append(stream)

            if self.atlas.revision() != atlas_revision:
                raise OperatorHarvesterConflictError(
                    "semantic Atlas changed during persisted-evidence validation"
                )

            publications: dict[str, ComputeBankPublication] = {}
            edges: list[OperatorEdge] = []
            fitted_by_edge: dict[str, _FittedCandidate] = {}
            stream_by_key = {
                f"{row.group_sha256}:{row.operator_kind}": row
                for row in candidate_streams
            }
            for fitted in fitted_candidates:
                publication = self.bank.publish_crystal(fitted.crystal)
                publications[fitted.crystal.sha256] = publication
                edge = OperatorEdge(
                    source_state=fitted.source_state,
                    target_state=fitted.target_state,
                    crystal_sha256=fitted.crystal.sha256,
                    verifier_sha256=fitted.verifier_sha256,
                    evidence_sha256=fitted.evidence_sha256,
                    weight=float(len(fitted.observation_receipt_sha256s)),
                )
                edges.append(edge)
                fitted_by_edge[edge.sha256] = fitted
            graph_state_sha, graph_changed = self._append_edges_cas(edges)

            harvest_promotions: list[HarvestPromotion] = []
            for edge in edges:
                fitted = fitted_by_edge[edge.sha256]
                checkpoint = PromotionCheckpoint(
                    fitted.group_sha256,
                    fitted.operator_kind,
                    fitted.crystal.sha256,
                    fitted.evidence_sha256,
                    edge.sha256,
                )
                promotions_by_key[checkpoint.key] = checkpoint
                stream = stream_by_key[checkpoint.key]
                harvest_promotions.append(
                    HarvestPromotion(
                        candidate=stream,
                        edge=edge,
                        bank_publication=publications[fitted.crystal.sha256],
                        graph_changed=graph_changed,
                        graph_state_sha256=graph_state_sha,
                    )
                )

            updated_state = HarvesterState(
                identity_sha256=self.identity_sha256,
                provider_cursor=batch.next_cursor,
                groups=tuple(
                    (group_sha, tuple(groups[group_sha]))
                    for group_sha in sorted(groups)
                ),
                promotions=tuple(
                    sorted(promotions_by_key.values(), key=lambda row: row.key)
                ),
                recent_receipt_sha256s=tuple(recent_order),
            )
            data = updated_state.to_bytes()
            try:
                state_publication = self.bank.store.publish_state(
                    self.state_name,
                    data,
                    expected_sha256=prior_state_sha,
                )
            except ManifestConflictError as exc:
                raise OperatorHarvesterConflictError(
                    "harvester state compare-and-swap conflicted"
                ) from exc
            if self.bank.store.restore_state(self.state_name) != data:
                raise OperatorHarvesterIntegrityError(
                    "harvester state failed immediate restoration"
                )
            return HarvestStepResult(
                atlas_revision=atlas_revision,
                cursor_before=state.provider_cursor,
                cursor_after=batch.next_cursor,
                exhausted=batch.exhausted,
                accepted_observations=accepted,
                rejections=tuple(rejections),
                candidates=tuple(candidate_streams),
                promotions=tuple(harvest_promotions),
                state_sha256=updated_state.sha256,
                state_publication=state_publication,
                graph_state_sha256=graph_state_sha,
            )

    def iter_steps(
        self,
        *,
        limit: int | None = None,
        max_steps: int | None = None,
        stop_on_idle: bool = True,
    ) -> Iterator[HarvestStepResult]:
        """Yield bounded resumable steps; the external runtime owns scheduling."""

        if max_steps is not None:
            _positive_int(max_steps, field="max_steps", maximum=(1 << 31) - 1)
        if not isinstance(stop_on_idle, bool):
            raise TypeError("stop_on_idle must be bool")
        completed = 0
        while max_steps is None or completed < max_steps:
            result = self.step(limit=limit)
            yield result
            completed += 1
            if result.exhausted or (
                stop_on_idle
                and result.accepted_observations == 0
                and not result.rejections
            ):
                return


__all__ = [
    "CANDIDATE_EVIDENCE_SCHEMA",
    "CONTEXTUAL_TRANSITION_SCHEMA",
    "HARVESTER_STATE_SCHEMA",
    "CandidateEvidenceStream",
    "ContextualObservationBatch",
    "ContextualOperatorObservation",
    "ContextualOperatorProvider",
    "ContextualTransitionReceipt",
    "ContinuousOperatorHarvester",
    "HarvestPromotion",
    "HarvestRejection",
    "HarvestStepResult",
    "HarvesterConfig",
    "HarvesterState",
    "OperatorHarvesterCapacityError",
    "OperatorHarvesterConflictError",
    "OperatorHarvesterError",
    "OperatorHarvesterIntegrityError",
    "PromotionCheckpoint",
    "QWEN_CONTEXT_EMITTER_SCHEMA",
    "QWEN_CONTEXT_EMITTER_SHA256",
    "SingleBatchContextualProvider",
    "contextual_observations_from_probe_result",
    "probe_result_context_cursor",
    "harvester_identity_sha256",
    "harvester_state_name",
]
