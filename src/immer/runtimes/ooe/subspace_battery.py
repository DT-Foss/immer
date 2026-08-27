"""Exact collision limits for joint Qwen gate/up subspace batteries.

This module evaluates cache keys formed from the *live* Qwen MLP boundary:
``RMSNorm(attention_residual) -> (gate_proj, up_proj)``.  It never reconstructs
that distribution from embeddings and never substitutes a Softmax attention
path for IMMER's Prefix-Sinkhorn runtime.  Basis selection and quantizer scales
use training groups only.  Calibration and holdout remain chronological atomic
groups, while cache hits are counted at the actual row/query granularity.

The current cartography HarvesterState contains a projected ``mlp.input`` sketch
but not the exact joint gate/up projections.  Its adapter therefore emits a
sealed instrumentation request and fails closed.  The generic evaluator is
ready for the required causal-runtime receipt without fabricating a live claim.
"""

from __future__ import annotations

import base64
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from typing import cast

import numpy as np
from numpy.typing import NDArray

from .identity import canonical_json_bytes, require_sha256
from .operator_harvester import HarvesterState, OperatorHarvesterIntegrityError


SUBSPACE_GROUP_SCHEMA = "immer-ooe-qwen-joint-subspace-group/v1"
SUBSPACE_CORPUS_SCHEMA = "immer-ooe-qwen-joint-subspace-corpus/v1"
SUBSPACE_FIT_SCHEMA = "immer-ooe-qwen-joint-subspace-fit/v1"
SUBSPACE_HOLDOUT_SCHEMA = "immer-ooe-qwen-joint-subspace-holdout/v1"
SUBSPACE_CONTENT_ADDRESS_SCHEMA = "immer-ooe-qwen-joint-subspace-content-address/v1"
SUBSPACE_INSTRUMENT_REQUEST_SCHEMA = (
    "immer-ooe-qwen-joint-subspace-instrument-request/v1"
)
SUBSPACE_CONTEXT_KIND = "qwen.mlp.input.rmsnorm-post-attention/v1"
SUBSPACE_ATTENTION_KIND = "immer.prefix-sinkhorn-causal/v1"
SUBSPACE_NUMERIC_ABI = "float64-le/train-only-energy-scale/rint-quant/v1"
SUBSPACE_SELECTION_ALGORITHM = (
    "joint-gate-up/top-k-train-group-energy/symmetric-channel-scale/"
    "chronological-cache-replay/zero-wrong-collision-promotion/v1"
)
SUBSPACE_EXACT_VERIFIER_SHA256 = hashlib.sha256(
    b"immer:subspace-battery:exact-output-sha256-replay/v1"
).hexdigest()

DEFAULT_K_VALUES = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_QUANT_BITS = (2, 4, 8, 12, 16)
DEFAULT_SCALE_FLOOR = 1e-12
DEFAULT_RANDOM_SEED_SHA256 = hashlib.sha256(
    b"immer:subspace-battery:random-control/v1"
).hexdigest()
DEFAULT_MAX_WORKING_BYTES = 1024**3
MAX_GROUPS = 65_536
MAX_ROWS_PER_GROUP = 65_536
MAX_HIDDEN_DIMENSION = 65_536
MAX_INTERMEDIATE_DIMENSION = 131_072
MAX_SOURCE_RECEIPTS_PER_GROUP = 64
MAX_RECEIPT_BYTES = 128 * 1024 * 1024

FloatArray = NDArray[np.float64]


class SubspaceBatteryError(RuntimeError):
    """Base error for exact Qwen subspace-battery evaluation."""


class SubspaceBatteryIntegrityError(SubspaceBatteryError):
    """A corpus, model, result, or source receipt failed exact replay."""


class SubspaceBatteryCapacityError(SubspaceBatteryError):
    """A requested sweep exceeds its declared resource bound."""


class SubspaceInstrumentationRequired(SubspaceBatteryError):
    """The source has sketches but lacks exact joint gate/up evidence."""

    def __init__(self, request: "SubspaceInstrumentationRequest") -> None:
        if not isinstance(request, SubspaceInstrumentationRequest):
            raise TypeError("request must be a SubspaceInstrumentationRequest")
        self.request = request
        super().__init__(
            "exact RMSNormed context plus gate_proj/up_proj instrumentation is required"
        )


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_json(data: bytes, *, schema: str, label: str) -> Mapping[str, object]:
    if not isinstance(data, bytes):
        raise TypeError(f"{label} must be immutable bytes")
    if not data or len(data) > MAX_RECEIPT_BYTES:
        raise SubspaceBatteryIntegrityError(f"{label} exceeds its byte bound")

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
        raise SubspaceBatteryIntegrityError(f"{label} is not strict JSON") from exc
    if (
        not isinstance(value, Mapping)
        or set(value) != {"body", "body_sha256", "schema"}
        or value.get("schema") != schema
        or not isinstance(value.get("body"), Mapping)
        or value.get("body_sha256") != _digest(value.get("body"))
        or canonical_json_bytes(value) != data
    ):
        raise SubspaceBatteryIntegrityError(f"{label} envelope is invalid")
    return cast(Mapping[str, object], value)


def _sealed(schema: str, body: Mapping[str, object]) -> dict[str, object]:
    normalized = json.loads(canonical_json_bytes(dict(body)))
    return {"body": normalized, "body_sha256": _digest(normalized), "schema": schema}


def _uint(value: object, *, field: str, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= 2**63 - 1
    ):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{field} must be a {qualifier}integer")
    return value


def _finite(value: object, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite numeric data")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{field} must be finite numeric data")
    return 0.0 if result == 0.0 else result


def _hashes(
    values: Sequence[str],
    *,
    field: str,
    sorted_unique: bool = False,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{field} must be a sequence")
    result = tuple(require_sha256(row, field=field) for row in values)
    if not allow_empty and not result:
        raise ValueError(f"{field} must be non-empty")
    if sorted_unique and result != tuple(sorted(set(result))):
        raise ValueError(f"{field} must be sorted and unique")
    return result


def _canonical_array(
    value: object,
    *,
    field: str,
    dimensions: int | None = None,
) -> FloatArray:
    if type(value) is not np.ndarray or value.dtype != np.dtype(np.float64):
        raise TypeError(f"{field} must be an exact float64 numpy.ndarray")
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError(f"{field} must be a non-empty matrix")
    if dimensions is not None and value.shape[1] != dimensions:
        raise ValueError(f"{field} has the wrong feature dimension")
    if not bool(np.isfinite(value).all()):
        raise ValueError(f"{field} contains non-finite values")
    result = np.array(value, dtype="<f8", order="C", copy=True)
    result[result == 0.0] = 0.0
    result.flags.writeable = False
    return cast(FloatArray, result)


def _array_record(value: FloatArray) -> dict[str, object]:
    array = _canonical_array(value, field="array")
    raw = array.tobytes(order="C")
    return {
        "data_base64": base64.b64encode(raw).decode("ascii"),
        "dtype": "float64-le",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "shape": [int(row) for row in array.shape],
    }


def _array_address_record(value: FloatArray) -> dict[str, object]:
    array = _canonical_array(value, field="array")
    raw = array.tobytes(order="C")
    return {
        "dtype": "float64-le",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "shape": [int(row) for row in array.shape],
    }


def graph_revision_sha256(sequence: int, event_sha256: str) -> str:
    """Return the exact SemanticWeightAtlas GraphRevision address."""

    return _digest(
        {
            "event_sha256": require_sha256(event_sha256, field="event_sha256"),
            "sequence": _uint(sequence, field="sequence"),
        }
    )


def projection_evidence_sha256(
    *,
    model_pin_sha256: str,
    graph_revision_sha256: str,
    layer: int,
    source_receipt_sha256s: Sequence[str],
    projection_verifier_sha256: str,
    context_states: FloatArray,
    gate_projection: FloatArray,
    up_projection: FloatArray,
) -> str:
    """Bind exact contextual states and both Qwen projection outputs."""

    return _digest(
        {
            "attention_kind": SUBSPACE_ATTENTION_KIND,
            "context_kind": SUBSPACE_CONTEXT_KIND,
            "context_states_sha256": cast(
                str, _array_address_record(context_states)["sha256"]
            ),
            "gate_projection_sha256": cast(
                str, _array_address_record(gate_projection)["sha256"]
            ),
            "graph_revision_sha256": require_sha256(
                graph_revision_sha256, field="graph_revision_sha256"
            ),
            "layer": _uint(layer, field="layer"),
            "model_pin_sha256": require_sha256(
                model_pin_sha256, field="model_pin_sha256"
            ),
            "projection_verifier_sha256": require_sha256(
                projection_verifier_sha256, field="projection_verifier_sha256"
            ),
            "schema": "immer-ooe-qwen-context-gate-up-projection-evidence/v1",
            "source_receipt_sha256s": list(
                _hashes(
                    source_receipt_sha256s,
                    field="source_receipt_sha256s",
                    sorted_unique=True,
                )
            ),
            "up_projection_sha256": cast(
                str, _array_address_record(up_projection)["sha256"]
            ),
        }
    )


def output_evidence_sha256(
    *,
    output_payload_sha256s: Sequence[str],
    output_verifier_sha256: str,
    source_receipt_sha256s: Sequence[str],
) -> str:
    """Bind every exact output payload to its external verifier evidence."""

    return _digest(
        {
            "output_payload_sha256s": list(
                _hashes(output_payload_sha256s, field="output_payload_sha256s")
            ),
            "output_verifier_sha256": require_sha256(
                output_verifier_sha256, field="output_verifier_sha256"
            ),
            "schema": "immer-ooe-qwen-subspace-output-evidence/v1",
            "source_receipt_sha256s": list(
                _hashes(
                    source_receipt_sha256s,
                    field="source_receipt_sha256s",
                    sorted_unique=True,
                )
            ),
        }
    )


def _array_from_record(value: object, *, field: str) -> FloatArray:
    if not isinstance(value, Mapping) or set(value) != {
        "data_base64",
        "dtype",
        "sha256",
        "shape",
    }:
        raise SubspaceBatteryIntegrityError(f"{field} array record is invalid")
    if value.get("dtype") != "float64-le":
        raise SubspaceBatteryIntegrityError(f"{field} dtype is invalid")
    shape = value.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(
            isinstance(row, bool) or not isinstance(row, int) or row < 1
            for row in shape
        )
    ):
        raise SubspaceBatteryIntegrityError(f"{field} shape is invalid")
    size = int(shape[0]) * int(shape[1]) * 8
    encoded = value.get("data_base64")
    if (
        not isinstance(encoded, str)
        or not encoded.isascii()
        or len(encoded) > 4 * ((size + 2) // 3)
    ):
        raise SubspaceBatteryIntegrityError(f"{field} encoding is invalid")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (TypeError, ValueError) as exc:
        raise SubspaceBatteryIntegrityError(f"{field} encoding is invalid") from exc
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != require_sha256(
        value.get("sha256"), field=f"{field}.sha256"
    ):
        raise SubspaceBatteryIntegrityError(f"{field} byte hash is invalid")
    result = np.frombuffer(raw, dtype="<f8").reshape((int(shape[0]), int(shape[1])))
    return _canonical_array(result, field=field)


@dataclass(frozen=True, slots=True)
class SubspaceObservationGroup:
    """One atomic chronological prompt/layer group from causal Qwen execution."""

    logical_time: int
    group_sha256: str
    model_pin_sha256: str
    graph_revision_sha256: str
    graph_sequence: int
    graph_event_sha256: str
    layer: int
    prompt_sha256: str
    source_receipt_sha256s: tuple[str, ...]
    projection_verifier_sha256: str
    projection_evidence_sha256: str
    output_verifier_sha256: str
    output_evidence_sha256: str
    context_states: FloatArray
    gate_projection: FloatArray
    up_projection: FloatArray
    output_payload_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        logical_time = _uint(self.logical_time, field="logical_time", positive=True)
        sequence = _uint(self.graph_sequence, field="graph_sequence")
        if sequence != logical_time:
            raise ValueError("group logical time must equal its graph sequence")
        layer = _uint(self.layer, field="layer")
        for field in (
            "group_sha256",
            "model_pin_sha256",
            "graph_revision_sha256",
            "graph_event_sha256",
            "prompt_sha256",
            "projection_verifier_sha256",
            "projection_evidence_sha256",
            "output_verifier_sha256",
            "output_evidence_sha256",
        ):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        if self.graph_revision_sha256 != graph_revision_sha256(
            sequence, self.graph_event_sha256
        ):
            raise SubspaceBatteryIntegrityError(
                "group graph revision differs from sequence/event evidence"
            )
        sources = _hashes(
            self.source_receipt_sha256s,
            field="source_receipt_sha256s",
            sorted_unique=True,
        )
        if len(sources) > MAX_SOURCE_RECEIPTS_PER_GROUP:
            raise ValueError("group has too many source receipts")
        context = _canonical_array(self.context_states, field="context_states")
        gate = _canonical_array(self.gate_projection, field="gate_projection")
        up = _canonical_array(self.up_projection, field="up_projection")
        if (
            context.shape[0] != gate.shape[0]
            or gate.shape != up.shape
            or context.shape[0] > MAX_ROWS_PER_GROUP
            or context.shape[1] > MAX_HIDDEN_DIMENSION
            or gate.shape[1] > MAX_INTERMEDIATE_DIMENSION
        ):
            raise ValueError("group contextual/projection shapes are incompatible")
        outputs = _hashes(self.output_payload_sha256s, field="output_payload_sha256s")
        if len(outputs) != context.shape[0]:
            raise ValueError("group output inventory differs from its row count")
        if self.projection_evidence_sha256 != projection_evidence_sha256(
            model_pin_sha256=self.model_pin_sha256,
            graph_revision_sha256=self.graph_revision_sha256,
            layer=layer,
            source_receipt_sha256s=sources,
            projection_verifier_sha256=self.projection_verifier_sha256,
            context_states=context,
            gate_projection=gate,
            up_projection=up,
        ):
            raise SubspaceBatteryIntegrityError(
                "group projection evidence differs from contextual gate/up arrays"
            )
        if self.output_evidence_sha256 != output_evidence_sha256(
            output_payload_sha256s=outputs,
            output_verifier_sha256=self.output_verifier_sha256,
            source_receipt_sha256s=sources,
        ):
            raise SubspaceBatteryIntegrityError(
                "group output evidence differs from exact payload inventory"
            )
        object.__setattr__(self, "logical_time", logical_time)
        object.__setattr__(self, "graph_sequence", sequence)
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "source_receipt_sha256s", sources)
        object.__setattr__(self, "context_states", context)
        object.__setattr__(self, "gate_projection", gate)
        object.__setattr__(self, "up_projection", up)
        object.__setattr__(self, "output_payload_sha256s", outputs)

    @property
    def row_count(self) -> int:
        return int(self.context_states.shape[0])

    @property
    def hidden_dimension(self) -> int:
        return int(self.context_states.shape[1])

    @property
    def intermediate_dimension(self) -> int:
        return int(self.gate_projection.shape[1])

    @property
    def sha256(self) -> str:
        return _digest(
            {
                "attention_kind": SUBSPACE_ATTENTION_KIND,
                "content_address_schema": SUBSPACE_CONTENT_ADDRESS_SCHEMA,
                "context_kind": SUBSPACE_CONTEXT_KIND,
                "context_states": _array_address_record(self.context_states),
                "gate_projection": _array_address_record(self.gate_projection),
                "graph_event_sha256": self.graph_event_sha256,
                "graph_revision_sha256": self.graph_revision_sha256,
                "graph_sequence": self.graph_sequence,
                "group_sha256": self.group_sha256,
                "layer": self.layer,
                "logical_time": self.logical_time,
                "model_pin_sha256": self.model_pin_sha256,
                "output_evidence_sha256": self.output_evidence_sha256,
                "output_payload_sha256s": list(self.output_payload_sha256s),
                "output_verifier_sha256": self.output_verifier_sha256,
                "projection_evidence_sha256": self.projection_evidence_sha256,
                "projection_verifier_sha256": self.projection_verifier_sha256,
                "prompt_sha256": self.prompt_sha256,
                "source_receipt_sha256s": list(self.source_receipt_sha256s),
                "up_projection": _array_address_record(self.up_projection),
            }
        )

    def to_record(self) -> dict[str, object]:
        return {
            "attention_kind": SUBSPACE_ATTENTION_KIND,
            "context_kind": SUBSPACE_CONTEXT_KIND,
            "context_states": _array_record(self.context_states),
            "gate_projection": _array_record(self.gate_projection),
            "graph_event_sha256": self.graph_event_sha256,
            "graph_revision_sha256": self.graph_revision_sha256,
            "graph_sequence": self.graph_sequence,
            "group_sha256": self.group_sha256,
            "layer": self.layer,
            "logical_time": self.logical_time,
            "model_pin_sha256": self.model_pin_sha256,
            "output_evidence_sha256": self.output_evidence_sha256,
            "output_payload_sha256s": list(self.output_payload_sha256s),
            "output_verifier_sha256": self.output_verifier_sha256,
            "projection_evidence_sha256": self.projection_evidence_sha256,
            "projection_verifier_sha256": self.projection_verifier_sha256,
            "prompt_sha256": self.prompt_sha256,
            "source_receipt_sha256s": list(self.source_receipt_sha256s),
            "up_projection": _array_record(self.up_projection),
        }

    @classmethod
    def from_record(cls, value: object) -> "SubspaceObservationGroup":
        expected = {
            "attention_kind",
            "context_kind",
            "context_states",
            "gate_projection",
            "graph_event_sha256",
            "graph_revision_sha256",
            "graph_sequence",
            "group_sha256",
            "layer",
            "logical_time",
            "model_pin_sha256",
            "output_evidence_sha256",
            "output_payload_sha256s",
            "output_verifier_sha256",
            "projection_evidence_sha256",
            "projection_verifier_sha256",
            "prompt_sha256",
            "source_receipt_sha256s",
            "up_projection",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or value.get("attention_kind") != SUBSPACE_ATTENTION_KIND
            or value.get("context_kind") != SUBSPACE_CONTEXT_KIND
            or not isinstance(value.get("source_receipt_sha256s"), list)
            or not isinstance(value.get("output_payload_sha256s"), list)
        ):
            raise SubspaceBatteryIntegrityError("subspace group record is invalid")
        try:
            return cls(
                logical_time=cast(int, value.get("logical_time")),
                group_sha256=cast(str, value.get("group_sha256")),
                model_pin_sha256=cast(str, value.get("model_pin_sha256")),
                graph_revision_sha256=cast(str, value.get("graph_revision_sha256")),
                graph_sequence=cast(int, value.get("graph_sequence")),
                graph_event_sha256=cast(str, value.get("graph_event_sha256")),
                layer=cast(int, value.get("layer")),
                prompt_sha256=cast(str, value.get("prompt_sha256")),
                source_receipt_sha256s=tuple(
                    cast(list[str], value.get("source_receipt_sha256s"))
                ),
                projection_verifier_sha256=cast(
                    str, value.get("projection_verifier_sha256")
                ),
                projection_evidence_sha256=cast(
                    str, value.get("projection_evidence_sha256")
                ),
                output_verifier_sha256=cast(str, value.get("output_verifier_sha256")),
                output_evidence_sha256=cast(str, value.get("output_evidence_sha256")),
                context_states=_array_from_record(
                    value.get("context_states"), field="context_states"
                ),
                gate_projection=_array_from_record(
                    value.get("gate_projection"), field="gate_projection"
                ),
                up_projection=_array_from_record(
                    value.get("up_projection"), field="up_projection"
                ),
                output_payload_sha256s=tuple(
                    cast(list[str], value.get("output_payload_sha256s"))
                ),
            )
        except SubspaceBatteryIntegrityError:
            raise
        except (TypeError, ValueError) as exc:
            raise SubspaceBatteryIntegrityError(
                "subspace group validation failed"
            ) from exc


@dataclass(frozen=True, slots=True)
class SubspaceCorpus:
    """Sealed joint gate/up evidence from one model and projection ABI."""

    model_pin_sha256: str
    groups: tuple[SubspaceObservationGroup, ...]

    def __post_init__(self) -> None:
        model_pin = require_sha256(self.model_pin_sha256, field="model_pin_sha256")
        groups = tuple(self.groups)
        if not groups or len(groups) > MAX_GROUPS:
            raise ValueError("subspace corpus group inventory is invalid")
        if any(not isinstance(row, SubspaceObservationGroup) for row in groups):
            raise TypeError("subspace corpus contains an invalid group")
        if (
            tuple(sorted(groups, key=lambda row: (row.logical_time, row.group_sha256)))
            != groups
        ):
            raise ValueError("subspace groups must follow strict chronology")
        if len({row.logical_time for row in groups}) != len(groups):
            raise ValueError("subspace group chronology is duplicated")
        if len({row.group_sha256 for row in groups}) != len(groups):
            raise ValueError("subspace group identity is duplicated")
        if any(row.model_pin_sha256 != model_pin for row in groups):
            raise SubspaceBatteryIntegrityError("subspace corpus crosses model pins")
        if (
            len({row.hidden_dimension for row in groups}) != 1
            or len({row.intermediate_dimension for row in groups}) != 1
        ):
            raise SubspaceBatteryIntegrityError("subspace corpus crosses tensor ABIs")
        object.__setattr__(self, "model_pin_sha256", model_pin)
        object.__setattr__(self, "groups", groups)

    @property
    def hidden_dimension(self) -> int:
        return self.groups[0].hidden_dimension

    @property
    def intermediate_dimension(self) -> int:
        return self.groups[0].intermediate_dimension

    @property
    def sha256(self) -> str:
        return _digest(
            {
                "content_address_schema": SUBSPACE_CONTENT_ADDRESS_SCHEMA,
                "graph_revision_sha256s": [
                    row.graph_revision_sha256 for row in self.groups
                ],
                "group_receipt_sha256s": [row.sha256 for row in self.groups],
                "group_sha256s": [row.group_sha256 for row in self.groups],
                "model_pin_sha256": self.model_pin_sha256,
            }
        )

    def to_document(self) -> dict[str, object]:
        return _sealed(
            SUBSPACE_CORPUS_SCHEMA,
            {
                "attention_kind": SUBSPACE_ATTENTION_KIND,
                "context_kind": SUBSPACE_CONTEXT_KIND,
                "graph_revision_sha256s": [
                    row.graph_revision_sha256 for row in self.groups
                ],
                "group_sha256s": [row.group_sha256 for row in self.groups],
                "groups": [row.to_record() for row in self.groups],
                "model_pin_sha256": self.model_pin_sha256,
                "output_verifier_sha256s": sorted(
                    {row.output_verifier_sha256 for row in self.groups}
                ),
                "projection_verifier_sha256s": sorted(
                    {row.projection_verifier_sha256 for row in self.groups}
                ),
                "source_receipt_sha256s": sorted(
                    {
                        receipt
                        for row in self.groups
                        for receipt in row.source_receipt_sha256s
                    }
                ),
            },
        )

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_document())
        if len(data) > MAX_RECEIPT_BYTES:
            raise SubspaceBatteryCapacityError("subspace corpus exceeds byte bound")
        return data

    @classmethod
    def from_bytes(cls, data: bytes) -> "SubspaceCorpus":
        envelope = _strict_json(data, schema=SUBSPACE_CORPUS_SCHEMA, label="corpus")
        body = envelope.get("body")
        expected = {
            "attention_kind",
            "context_kind",
            "graph_revision_sha256s",
            "group_sha256s",
            "groups",
            "model_pin_sha256",
            "output_verifier_sha256s",
            "projection_verifier_sha256s",
            "source_receipt_sha256s",
        }
        if (
            not isinstance(body, Mapping)
            or set(body) != expected
            or body.get("attention_kind") != SUBSPACE_ATTENTION_KIND
            or body.get("context_kind") != SUBSPACE_CONTEXT_KIND
            or not isinstance(body.get("groups"), list)
        ):
            raise SubspaceBatteryIntegrityError("subspace corpus body is invalid")
        try:
            groups = tuple(
                SubspaceObservationGroup.from_record(row)
                for row in cast(list[object], body.get("groups"))
            )
            result = cls(
                model_pin_sha256=cast(str, body.get("model_pin_sha256")),
                groups=groups,
            )
            pins = {
                "graph_revision_sha256s": [row.graph_revision_sha256 for row in groups],
                "group_sha256s": [row.group_sha256 for row in groups],
                "output_verifier_sha256s": sorted(
                    {row.output_verifier_sha256 for row in groups}
                ),
                "projection_verifier_sha256s": sorted(
                    {row.projection_verifier_sha256 for row in groups}
                ),
                "source_receipt_sha256s": sorted(
                    {
                        receipt
                        for row in groups
                        for receipt in row.source_receipt_sha256s
                    }
                ),
            }
            if any(body.get(field) != actual for field, actual in pins.items()):
                raise SubspaceBatteryIntegrityError("subspace corpus pin changed")
        except SubspaceBatteryIntegrityError:
            raise
        except (TypeError, ValueError) as exc:
            raise SubspaceBatteryIntegrityError(
                "subspace corpus validation failed"
            ) from exc
        if result.to_bytes() != data:
            raise SubspaceBatteryIntegrityError(
                "subspace corpus reconstruction changed"
            )
        return result


@dataclass(frozen=True, slots=True)
class SubspaceSweepConfig:
    k_values: tuple[int, ...] = DEFAULT_K_VALUES
    quant_bits: tuple[int, ...] = DEFAULT_QUANT_BITS
    scale_floor: float = DEFAULT_SCALE_FLOOR
    random_seed_sha256: str = DEFAULT_RANDOM_SEED_SHA256
    max_working_bytes: int = DEFAULT_MAX_WORKING_BYTES

    def __post_init__(self) -> None:
        k_values = tuple(self.k_values)
        bits = tuple(self.quant_bits)
        if (
            not k_values
            or k_values != tuple(sorted(set(k_values)))
            or any(
                isinstance(row, bool) or not isinstance(row, int) or row < 1
                for row in k_values
            )
        ):
            raise ValueError("k_values must be sorted unique positive integers")
        if (
            not bits
            or bits != tuple(sorted(set(bits)))
            or any(
                isinstance(row, bool) or not isinstance(row, int) or not 2 <= row <= 16
                for row in bits
            )
        ):
            raise ValueError("quant_bits must be sorted unique values in [2, 16]")
        floor = _finite(self.scale_floor, field="scale_floor", positive=True)
        seed = require_sha256(self.random_seed_sha256, field="random_seed_sha256")
        maximum = _uint(
            self.max_working_bytes, field="max_working_bytes", positive=True
        )
        object.__setattr__(self, "k_values", k_values)
        object.__setattr__(self, "quant_bits", bits)
        object.__setattr__(self, "scale_floor", floor)
        object.__setattr__(self, "random_seed_sha256", seed)
        object.__setattr__(self, "max_working_bytes", maximum)

    def to_record(self) -> dict[str, object]:
        return {
            "k_values": list(self.k_values),
            "max_working_bytes": self.max_working_bytes,
            "numeric_abi": SUBSPACE_NUMERIC_ABI,
            "quant_bits": list(self.quant_bits),
            "random_seed_sha256": self.random_seed_sha256,
            "scale_floor": self.scale_floor,
            "selection_algorithm": SUBSPACE_SELECTION_ALGORITHM,
        }

    @classmethod
    def from_record(cls, value: object) -> "SubspaceSweepConfig":
        if not isinstance(value, Mapping) or set(value) != {
            "k_values",
            "max_working_bytes",
            "numeric_abi",
            "quant_bits",
            "random_seed_sha256",
            "scale_floor",
            "selection_algorithm",
        }:
            raise SubspaceBatteryIntegrityError("subspace config is invalid")
        if (
            value.get("numeric_abi") != SUBSPACE_NUMERIC_ABI
            or value.get("selection_algorithm") != SUBSPACE_SELECTION_ALGORITHM
            or not isinstance(value.get("k_values"), list)
            or not isinstance(value.get("quant_bits"), list)
        ):
            raise SubspaceBatteryIntegrityError("subspace config ABI is unknown")
        try:
            return cls(
                k_values=tuple(cast(list[int], value.get("k_values"))),
                quant_bits=tuple(cast(list[int], value.get("quant_bits"))),
                scale_floor=cast(float, value.get("scale_floor")),
                random_seed_sha256=cast(str, value.get("random_seed_sha256")),
                max_working_bytes=cast(int, value.get("max_working_bytes")),
            )
        except (TypeError, ValueError) as exc:
            raise SubspaceBatteryIntegrityError(
                "subspace config validation failed"
            ) from exc


@dataclass(frozen=True, slots=True)
class CollisionMetrics:
    query_rows: int
    candidate_hits: int
    exact_verified_hits: int
    wrong_collisions: int
    misses: int
    prior_wrong_collision_rows: int
    stored_keys: int
    stored_bytes: int

    def __post_init__(self) -> None:
        for field in (
            "query_rows",
            "candidate_hits",
            "exact_verified_hits",
            "wrong_collisions",
            "misses",
            "prior_wrong_collision_rows",
            "stored_keys",
            "stored_bytes",
        ):
            object.__setattr__(self, field, _uint(getattr(self, field), field=field))
        if (
            self.candidate_hits != self.exact_verified_hits + self.wrong_collisions
            or self.query_rows != self.candidate_hits + self.misses
        ):
            raise ValueError("collision metric partition is inconsistent")

    @property
    def any_wrong_collision(self) -> bool:
        return bool(self.prior_wrong_collision_rows or self.wrong_collisions)

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_hits": self.candidate_hits,
            "exact_verified_hits": self.exact_verified_hits,
            "misses": self.misses,
            "prior_wrong_collision_rows": self.prior_wrong_collision_rows,
            "query_rows": self.query_rows,
            "stored_bytes": self.stored_bytes,
            "stored_keys": self.stored_keys,
            "wrong_collisions": self.wrong_collisions,
        }

    @classmethod
    def from_record(cls, value: object) -> "CollisionMetrics":
        if not isinstance(value, Mapping) or set(value) != {
            "candidate_hits",
            "exact_verified_hits",
            "misses",
            "prior_wrong_collision_rows",
            "query_rows",
            "stored_bytes",
            "stored_keys",
            "wrong_collisions",
        }:
            raise SubspaceBatteryIntegrityError("collision metrics are invalid")
        try:
            return cls(**dict(value))
        except (TypeError, ValueError) as exc:
            raise SubspaceBatteryIntegrityError(
                "collision metric validation failed"
            ) from exc


_MODEL_FAMILIES = ("candidate", "full", "marginal", "random")


@dataclass(frozen=True, slots=True)
class SubspaceKeyModel:
    family: str
    k: int
    quant_bits: int
    basis_indices: tuple[int, ...]
    scale: FloatArray | None
    quantizer_sha256: str
    train_wrong_collision_rows: int
    calibration_metrics: CollisionMetrics
    calibration_safe: bool

    def __post_init__(self) -> None:
        if self.family not in _MODEL_FAMILIES:
            raise ValueError("subspace model family is invalid")
        k = _uint(self.k, field="k")
        bits = _uint(self.quant_bits, field="quant_bits")
        basis = tuple(self.basis_indices)
        if self.family == "marginal":
            if k != 0 or bits != 0 or basis or self.scale is not None:
                raise ValueError("marginal control cannot carry a subspace key")
        else:
            if (
                k < 1
                or not 2 <= bits <= 16
                or len(basis) != k
                or len(set(basis)) != k
                or any(
                    isinstance(row, bool) or not isinstance(row, int) or row < 0
                    for row in basis
                )
            ):
                raise ValueError("subspace key basis is invalid")
            scale = _canonical_array(self.scale, field="scale")
            if scale.shape != (1, 2 * k) or np.any(scale <= 0.0):
                raise ValueError("subspace key scale is invalid")
            object.__setattr__(self, "scale", scale)
        quantizer = require_sha256(self.quantizer_sha256, field="quantizer_sha256")
        if quantizer != _quantizer_sha256(
            bits, cast(FloatArray | None, self.scale), family=self.family
        ):
            raise SubspaceBatteryIntegrityError(
                "subspace quantizer hash differs from bits/scale ABI"
            )
        train_wrong = _uint(
            self.train_wrong_collision_rows, field="train_wrong_collision_rows"
        )
        if not isinstance(self.calibration_metrics, CollisionMetrics):
            raise TypeError("calibration_metrics must be CollisionMetrics")
        expected_safe = not (train_wrong or self.calibration_metrics.wrong_collisions)
        if (
            not isinstance(self.calibration_safe, bool)
            or self.calibration_safe != expected_safe
        ):
            raise ValueError("calibration safety verdict is inconsistent")
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "quant_bits", bits)
        object.__setattr__(self, "basis_indices", basis)
        object.__setattr__(self, "quantizer_sha256", quantizer)
        object.__setattr__(self, "train_wrong_collision_rows", train_wrong)

    @property
    def key(self) -> tuple[str, int, int]:
        return self.family, self.k, self.quant_bits

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "basis_indices": list(self.basis_indices),
            "calibration_metrics": self.calibration_metrics.to_record(),
            "calibration_safe": self.calibration_safe,
            "family": self.family,
            "k": self.k,
            "quant_bits": self.quant_bits,
            "quantizer_sha256": self.quantizer_sha256,
            "scale": None if self.scale is None else _array_record(self.scale),
            "train_wrong_collision_rows": self.train_wrong_collision_rows,
        }

    @classmethod
    def from_record(cls, value: object) -> "SubspaceKeyModel":
        if (
            not isinstance(value, Mapping)
            or set(value)
            != {
                "basis_indices",
                "calibration_metrics",
                "calibration_safe",
                "family",
                "k",
                "quant_bits",
                "quantizer_sha256",
                "scale",
                "train_wrong_collision_rows",
            }
            or not isinstance(value.get("basis_indices"), list)
        ):
            raise SubspaceBatteryIntegrityError("subspace key model is invalid")
        try:
            return cls(
                family=cast(str, value.get("family")),
                k=cast(int, value.get("k")),
                quant_bits=cast(int, value.get("quant_bits")),
                basis_indices=tuple(cast(list[int], value.get("basis_indices"))),
                scale=(
                    None
                    if value.get("scale") is None
                    else _array_from_record(value.get("scale"), field="scale")
                ),
                quantizer_sha256=cast(str, value.get("quantizer_sha256")),
                train_wrong_collision_rows=cast(
                    int, value.get("train_wrong_collision_rows")
                ),
                calibration_metrics=CollisionMetrics.from_record(
                    value.get("calibration_metrics")
                ),
                calibration_safe=cast(bool, value.get("calibration_safe")),
            )
        except SubspaceBatteryIntegrityError:
            raise
        except (TypeError, ValueError) as exc:
            raise SubspaceBatteryIntegrityError(
                "subspace key model validation failed"
            ) from exc


def _indices(values: Sequence[int], *, field: str, count: int) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{field} must be a sequence")
    result = tuple(values)
    if (
        not result
        or result != tuple(sorted(set(result)))
        or any(
            isinstance(row, bool) or not isinstance(row, int) or not 0 <= row < count
            for row in result
        )
    ):
        raise ValueError(f"{field} must be sorted unique corpus indices")
    return result


def _split_record(corpus: SubspaceCorpus, indices: Sequence[int]) -> dict[str, object]:
    groups = tuple(corpus.groups[index] for index in indices)
    return {
        "graph_revision_sha256s": [row.graph_revision_sha256 for row in groups],
        "group_receipt_sha256s": [row.sha256 for row in groups],
        "group_sha256s": [row.group_sha256 for row in groups],
        "indices": list(indices),
        "layers": sorted({row.layer for row in groups}),
        "logical_times": [row.logical_time for row in groups],
        "output_payload_sha256s": [
            value for row in groups for value in row.output_payload_sha256s
        ],
        "prompt_sha256s": sorted({row.prompt_sha256 for row in groups}),
        "row_count": sum(row.row_count for row in groups),
        "source_receipt_sha256s": [
            value for row in groups for value in row.source_receipt_sha256s
        ],
    }


def _validate_split(value: object, *, label: str) -> Mapping[str, object]:
    expected = {
        "graph_revision_sha256s",
        "group_receipt_sha256s",
        "group_sha256s",
        "indices",
        "layers",
        "logical_times",
        "output_payload_sha256s",
        "prompt_sha256s",
        "row_count",
        "source_receipt_sha256s",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise SubspaceBatteryIntegrityError(f"{label} split is invalid")
    if any(
        not isinstance(value.get(field), list)
        for field in expected
        if field != "row_count"
    ):
        raise SubspaceBatteryIntegrityError(f"{label} split inventory is invalid")
    return value


def _groups(
    corpus: SubspaceCorpus, indices: Sequence[int]
) -> tuple[SubspaceObservationGroup, ...]:
    return tuple(corpus.groups[index] for index in indices)


def _joint_energy(groups: Sequence[SubspaceObservationGroup]) -> FloatArray:
    dimension = groups[0].intermediate_dimension
    energy = np.zeros(dimension, dtype=np.float64)
    for group in groups:
        energy += np.mean(
            group.gate_projection * group.gate_projection
            + group.up_projection * group.up_projection,
            axis=0,
        )
    energy /= float(len(groups))
    if not bool(np.isfinite(energy).all()):
        raise SubspaceBatteryIntegrityError("train-only joint energy is non-finite")
    return cast(FloatArray, energy)


def _top_basis(energy: FloatArray, k: int) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(int(energy.shape[0])),
            key=lambda index: (-float(energy[index]), index),
        )[:k]
    )


def _random_basis(seed_sha256: str, dimension: int, k: int) -> tuple[int, ...]:
    ranked = sorted(
        range(dimension),
        key=lambda index: (
            hashlib.sha256(
                f"{seed_sha256}:{dimension}:{k}:{index}".encode("ascii")
            ).digest(),
            index,
        ),
    )
    return tuple(ranked[:k])


def _joint_rows(group: SubspaceObservationGroup, basis: Sequence[int]) -> FloatArray:
    indices = np.asarray(tuple(basis), dtype=np.int64)
    result = np.concatenate(
        (group.gate_projection[:, indices], group.up_projection[:, indices]), axis=1
    )
    if not bool(np.isfinite(result).all()):
        raise SubspaceBatteryIntegrityError("joint gate/up features are non-finite")
    return cast(FloatArray, result)


def _train_scale(
    groups: Sequence[SubspaceObservationGroup],
    basis: Sequence[int],
    *,
    floor: float,
) -> FloatArray:
    maximum = np.zeros(2 * len(tuple(basis)), dtype=np.float64)
    for group in groups:
        maximum = np.maximum(maximum, np.max(np.abs(_joint_rows(group, basis)), axis=0))
    maximum = np.maximum(maximum, floor)
    return _canonical_array(maximum.reshape(1, -1), field="train_scale")


def _quantizer_sha256(bits: int, scale: FloatArray | None, *, family: str) -> str:
    return _digest(
        {
            "bits": bits,
            "family": family,
            "joint_order": ["gate_proj", "up_proj"],
            "numeric_abi": SUBSPACE_NUMERIC_ABI,
            "rounding": "numpy-rint-ties-to-even",
            "scale_sha256": (
                None if scale is None else cast(str, _array_record(scale)["sha256"])
            ),
            "storage": "little-endian-int8-or-int16",
        }
    )


def _key_bytes(
    group: SubspaceObservationGroup,
    basis: Sequence[int],
    scale: FloatArray,
    bits: int,
) -> tuple[bytes, ...]:
    values = _joint_rows(group, basis)
    qmax = (1 << (bits - 1)) - 1
    normalized = values / scale
    quantized = np.clip(np.rint(normalized * qmax), -qmax, qmax)
    dtype = np.dtype("<i1") if bits <= 8 else np.dtype("<i2")
    encoded = np.asarray(quantized, dtype=dtype, order="C")
    return tuple(encoded[index].tobytes(order="C") for index in range(encoded.shape[0]))


def _row_stream(
    groups: Sequence[SubspaceObservationGroup],
) -> tuple[tuple[SubspaceObservationGroup, int, str], ...]:
    return tuple(
        (group, row, output)
        for group in groups
        for row, output in enumerate(group.output_payload_sha256s)
    )


@dataclass(slots=True)
class _CacheState:
    outputs: dict[bytes, Counter[str]]
    key_width: int
    marginal_output: str | None = None

    @property
    def ambiguous_rows(self) -> int:
        return sum(
            sum(counts.values()) for counts in self.outputs.values() if len(counts) > 1
        )

    @property
    def stored_bytes(self) -> int:
        return sum(len(key) + 32 * len(counts) for key, counts in self.outputs.items())


def _build_cache(
    model: SubspaceKeyModel | tuple[str, tuple[int, ...], int, FloatArray | None],
    train: Sequence[SubspaceObservationGroup],
) -> _CacheState:
    if isinstance(model, SubspaceKeyModel):
        family, basis, bits, scale = (
            model.family,
            model.basis_indices,
            model.quant_bits,
            model.scale,
        )
    else:
        family, basis, bits, scale = model
    if family == "marginal":
        all_outputs = Counter(
            output for group in train for output in group.output_payload_sha256s
        )
        selected = min(
            output
            for output, count in all_outputs.items()
            if count == max(all_outputs.values())
        )
        return _CacheState(
            {b"": Counter({selected: all_outputs[selected]})}, 0, selected
        )
    assert scale is not None
    cache: dict[bytes, Counter[str]] = {}
    for group in train:
        keys = _key_bytes(group, basis, scale, bits)
        for key, output in zip(keys, group.output_payload_sha256s, strict=True):
            cache.setdefault(key, Counter())[output] += 1
    key_width = len(next(iter(cache))) if cache else 0
    return _CacheState(cache, key_width)


def _simulate_queries(
    cache: _CacheState,
    family: str,
    basis: Sequence[int],
    bits: int,
    scale: FloatArray | None,
    groups: Sequence[SubspaceObservationGroup],
    *,
    update: bool,
    basis_bytes: int,
    scale_bytes: int,
) -> CollisionMetrics:
    candidate_hits = exact = wrong = misses = 0
    prior_wrong = cache.ambiguous_rows
    for group in groups:
        keys = (
            (b"",) * group.row_count
            if family == "marginal"
            else _key_bytes(group, basis, cast(FloatArray, scale), bits)
        )
        for key, output in zip(keys, group.output_payload_sha256s, strict=True):
            counts = cache.outputs.get(key)
            if counts is None:
                misses += 1
                if update:
                    cache.outputs[key] = Counter({output: 1})
                continue
            candidate_hits += 1
            if len(counts) == 1 and output in counts:
                exact += 1
                if update:
                    counts[output] += 1
            else:
                wrong += 1
                if update and family != "marginal":
                    counts[output] += 1
    return CollisionMetrics(
        query_rows=sum(group.row_count for group in groups),
        candidate_hits=candidate_hits,
        exact_verified_hits=exact,
        wrong_collisions=wrong,
        misses=misses,
        prior_wrong_collision_rows=prior_wrong,
        stored_keys=len(cache.outputs),
        stored_bytes=cache.stored_bytes + basis_bytes + scale_bytes,
    )


def _model_spec(
    family: str,
    k: int,
    bits: int,
    basis: tuple[int, ...],
    scale: FloatArray | None,
    train: Sequence[SubspaceObservationGroup],
    calibration: Sequence[SubspaceObservationGroup],
) -> SubspaceKeyModel:
    cache = _build_cache((family, basis, bits, scale), train)
    train_wrong = cache.ambiguous_rows
    metrics = _simulate_queries(
        cache,
        family,
        basis,
        bits,
        scale,
        calibration,
        update=True,
        basis_bytes=4 * len(basis),
        scale_bytes=0 if scale is None else int(scale.nbytes),
    )
    return SubspaceKeyModel(
        family=family,
        k=k,
        quant_bits=bits,
        basis_indices=basis,
        scale=scale,
        quantizer_sha256=_quantizer_sha256(bits, scale, family=family),
        train_wrong_collision_rows=train_wrong,
        calibration_metrics=metrics,
        calibration_safe=not (train_wrong or metrics.wrong_collisions),
    )


def _expected_model_keys(
    config: SubspaceSweepConfig, intermediate_dimension: int
) -> tuple[tuple[str, int, int], ...]:
    k_values = tuple(k for k in config.k_values if k <= intermediate_dimension)
    return (
        tuple(
            ("candidate", k, bits)
            for k, bits in itertools.product(k_values, config.quant_bits)
        )
        + tuple(
            ("random", k, bits)
            for k, bits in itertools.product(k_values, config.quant_bits)
        )
        + tuple(("full", intermediate_dimension, bits) for bits in config.quant_bits)
        + (("marginal", 0, 0),)
    )


@dataclass(frozen=True, slots=True)
class SubspaceBatteryFit:
    corpus_sha256: str
    model_pin_sha256: str
    graph_revision_sha256s: tuple[str, ...]
    source_receipt_sha256s: tuple[str, ...]
    projection_verifier_sha256s: tuple[str, ...]
    output_verifier_sha256s: tuple[str, ...]
    config: SubspaceSweepConfig
    train: Mapping[str, object]
    calibration: Mapping[str, object]
    models: tuple[SubspaceKeyModel, ...]

    def __post_init__(self) -> None:
        for field in ("corpus_sha256", "model_pin_sha256"):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        graph_revisions = _hashes(
            self.graph_revision_sha256s,
            field="graph_revision_sha256s",
            sorted_unique=True,
        )
        source_receipts = _hashes(
            self.source_receipt_sha256s,
            field="source_receipt_sha256s",
            sorted_unique=True,
        )
        projection_verifiers = _hashes(
            self.projection_verifier_sha256s,
            field="projection_verifier_sha256s",
            sorted_unique=True,
        )
        output_verifiers = _hashes(
            self.output_verifier_sha256s,
            field="output_verifier_sha256s",
            sorted_unique=True,
        )
        if not isinstance(self.config, SubspaceSweepConfig):
            raise TypeError("config must be a SubspaceSweepConfig")
        train = _validate_split(self.train, label="train")
        calibration = _validate_split(self.calibration, label="calibration")
        train_indices = tuple(cast(list[int], train.get("indices")))
        calibration_indices = tuple(cast(list[int], calibration.get("indices")))
        if set(train_indices) & set(calibration_indices) or max(train_indices) >= min(
            calibration_indices
        ):
            raise SubspaceBatteryIntegrityError(
                "subspace train/calibration chronology is invalid"
            )
        models = tuple(self.models)
        if any(not isinstance(row, SubspaceKeyModel) for row in models):
            raise TypeError("fit contains an invalid subspace model")
        if len({row.key for row in models}) != len(models):
            raise ValueError("subspace fit model inventory is duplicated")
        object.__setattr__(self, "graph_revision_sha256s", graph_revisions)
        object.__setattr__(self, "source_receipt_sha256s", source_receipts)
        object.__setattr__(self, "projection_verifier_sha256s", projection_verifiers)
        object.__setattr__(self, "output_verifier_sha256s", output_verifiers)
        object.__setattr__(self, "train", train)
        object.__setattr__(self, "calibration", calibration)
        object.__setattr__(self, "models", models)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def to_document(self) -> dict[str, object]:
        return _sealed(
            SUBSPACE_FIT_SCHEMA,
            {
                "attention_kind": SUBSPACE_ATTENTION_KIND,
                "calibration": dict(self.calibration),
                "config": self.config.to_record(),
                "corpus_sha256": self.corpus_sha256,
                "evaluator_verifier_sha256": SUBSPACE_EXACT_VERIFIER_SHA256,
                "graph_revision_sha256s": list(self.graph_revision_sha256s),
                "model_inventory": [list(row.key) for row in self.models],
                "model_pin_sha256": self.model_pin_sha256,
                "models": [row.to_record() for row in self.models],
                "numeric_abi": SUBSPACE_NUMERIC_ABI,
                "output_verifier_sha256s": list(self.output_verifier_sha256s),
                "projection_verifier_sha256s": list(self.projection_verifier_sha256s),
                "source_receipt_sha256s": list(self.source_receipt_sha256s),
                "train": dict(self.train),
            },
        )

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_document())
        if len(data) > MAX_RECEIPT_BYTES:
            raise SubspaceBatteryCapacityError("subspace fit exceeds byte bound")
        return data

    def _verify_split(self, corpus: SubspaceCorpus) -> None:
        if not isinstance(corpus, SubspaceCorpus):
            raise TypeError("corpus must be a SubspaceCorpus")
        if (
            self.corpus_sha256 != corpus.sha256
            or self.model_pin_sha256 != corpus.model_pin_sha256
        ):
            raise SubspaceBatteryIntegrityError(
                "subspace fit belongs to another corpus"
            )
        train_indices = _indices(
            cast(list[int], self.train.get("indices")),
            field="train.indices",
            count=len(corpus.groups),
        )
        calibration_indices = _indices(
            cast(list[int], self.calibration.get("indices")),
            field="calibration.indices",
            count=len(corpus.groups),
        )
        if dict(self.train) != _split_record(corpus, train_indices) or dict(
            self.calibration
        ) != _split_record(corpus, calibration_indices):
            raise SubspaceBatteryIntegrityError(
                "subspace fit split differs from authenticated corpus"
            )

    def verify_against(self, corpus: SubspaceCorpus) -> None:
        self._verify_split(corpus)
        recomputed = fit_subspace_battery(
            corpus,
            train_group_indices=cast(list[int], self.train.get("indices")),
            calibration_group_indices=cast(list[int], self.calibration.get("indices")),
            config=self.config,
        )
        if recomputed.to_bytes() != self.to_bytes():
            raise SubspaceBatteryIntegrityError(
                "subspace fit differs from complete recomputation"
            )

    @classmethod
    def from_bytes(cls, data: bytes, *, corpus: SubspaceCorpus) -> "SubspaceBatteryFit":
        envelope = _strict_json(data, schema=SUBSPACE_FIT_SCHEMA, label="fit")
        body = envelope.get("body")
        expected = {
            "attention_kind",
            "calibration",
            "config",
            "corpus_sha256",
            "evaluator_verifier_sha256",
            "graph_revision_sha256s",
            "model_inventory",
            "model_pin_sha256",
            "models",
            "numeric_abi",
            "output_verifier_sha256s",
            "projection_verifier_sha256s",
            "source_receipt_sha256s",
            "train",
        }
        if (
            not isinstance(body, Mapping)
            or set(body) != expected
            or body.get("attention_kind") != SUBSPACE_ATTENTION_KIND
            or body.get("numeric_abi") != SUBSPACE_NUMERIC_ABI
            or body.get("evaluator_verifier_sha256") != SUBSPACE_EXACT_VERIFIER_SHA256
            or not isinstance(body.get("models"), list)
            or not isinstance(body.get("model_inventory"), list)
        ):
            raise SubspaceBatteryIntegrityError("subspace fit body is invalid")
        sequence_fields = (
            "graph_revision_sha256s",
            "output_verifier_sha256s",
            "projection_verifier_sha256s",
            "source_receipt_sha256s",
        )
        if any(not isinstance(body.get(field), list) for field in sequence_fields):
            raise SubspaceBatteryIntegrityError("subspace fit pins are invalid")
        try:
            models = tuple(
                SubspaceKeyModel.from_record(row)
                for row in cast(list[object], body.get("models"))
            )
            result = cls(
                corpus_sha256=cast(str, body.get("corpus_sha256")),
                model_pin_sha256=cast(str, body.get("model_pin_sha256")),
                graph_revision_sha256s=tuple(
                    cast(list[str], body.get("graph_revision_sha256s"))
                ),
                source_receipt_sha256s=tuple(
                    cast(list[str], body.get("source_receipt_sha256s"))
                ),
                projection_verifier_sha256s=tuple(
                    cast(list[str], body.get("projection_verifier_sha256s"))
                ),
                output_verifier_sha256s=tuple(
                    cast(list[str], body.get("output_verifier_sha256s"))
                ),
                config=SubspaceSweepConfig.from_record(body.get("config")),
                train=_validate_split(body.get("train"), label="train"),
                calibration=_validate_split(
                    body.get("calibration"), label="calibration"
                ),
                models=models,
            )
            if body.get("model_inventory") != [list(row.key) for row in models]:
                raise SubspaceBatteryIntegrityError("subspace model inventory changed")
        except SubspaceBatteryIntegrityError:
            raise
        except (TypeError, ValueError) as exc:
            raise SubspaceBatteryIntegrityError(
                "subspace fit validation failed"
            ) from exc
        if result.to_bytes() != data:
            raise SubspaceBatteryIntegrityError("subspace fit reconstruction changed")
        result.verify_against(corpus)
        return result


def _preflight(corpus: SubspaceCorpus, config: SubspaceSweepConfig) -> None:
    total_rows = sum(row.row_count for row in corpus.groups)
    dimension = corpus.intermediate_dimension
    model_count = len(_expected_model_keys(config, dimension))
    required = 8 * total_rows * (corpus.hidden_dimension + 2 * dimension)
    required += model_count * (16 * dimension + total_rows * 8)
    if required > config.max_working_bytes:
        raise SubspaceBatteryCapacityError(
            f"subspace sweep requires {required} bytes, limit is {config.max_working_bytes}"
        )


def fit_subspace_battery(
    corpus: SubspaceCorpus,
    *,
    train_group_indices: Sequence[int],
    calibration_group_indices: Sequence[int],
    config: SubspaceSweepConfig | None = None,
) -> SubspaceBatteryFit:
    """Fit train-only bases/scales and sweep calibration collision limits."""

    if not isinstance(corpus, SubspaceCorpus):
        raise TypeError("corpus must be a SubspaceCorpus")
    selected_config = SubspaceSweepConfig() if config is None else config
    if not isinstance(selected_config, SubspaceSweepConfig):
        raise TypeError("config must be a SubspaceSweepConfig")
    train_indices = _indices(
        train_group_indices, field="train_group_indices", count=len(corpus.groups)
    )
    calibration_indices = _indices(
        calibration_group_indices,
        field="calibration_group_indices",
        count=len(corpus.groups),
    )
    if set(train_indices) & set(calibration_indices) or max(train_indices) >= min(
        calibration_indices
    ):
        raise ValueError("calibration groups must follow disjoint training groups")
    dimension = corpus.intermediate_dimension
    k_values = tuple(k for k in selected_config.k_values if k <= dimension)
    if not k_values:
        raise ValueError("no configured k fits the Qwen intermediate dimension")
    _preflight(corpus, selected_config)
    train = _groups(corpus, train_indices)
    calibration = _groups(corpus, calibration_indices)
    energy = _joint_energy(train)
    models: list[SubspaceKeyModel] = []
    for family in ("candidate", "random"):
        for k, bits in itertools.product(k_values, selected_config.quant_bits):
            basis = (
                _top_basis(energy, k)
                if family == "candidate"
                else _random_basis(selected_config.random_seed_sha256, dimension, k)
            )
            scale = _train_scale(train, basis, floor=selected_config.scale_floor)
            models.append(
                _model_spec(family, k, bits, basis, scale, train, calibration)
            )
    full_basis = tuple(range(dimension))
    full_scale = _train_scale(train, full_basis, floor=selected_config.scale_floor)
    for bits in selected_config.quant_bits:
        models.append(
            _model_spec(
                "full",
                dimension,
                bits,
                full_basis,
                full_scale,
                train,
                calibration,
            )
        )
    models.append(_model_spec("marginal", 0, 0, (), None, train, calibration))
    expected_keys = _expected_model_keys(selected_config, dimension)
    if tuple(row.key for row in models) != expected_keys:
        raise AssertionError("subspace model construction order changed")
    graph_revisions = tuple(
        sorted({row.graph_revision_sha256 for row in corpus.groups})
    )
    source_receipts = tuple(
        sorted(
            {receipt for row in corpus.groups for receipt in row.source_receipt_sha256s}
        )
    )
    result = SubspaceBatteryFit(
        corpus_sha256=corpus.sha256,
        model_pin_sha256=corpus.model_pin_sha256,
        graph_revision_sha256s=graph_revisions,
        source_receipt_sha256s=source_receipts,
        projection_verifier_sha256s=tuple(
            sorted({row.projection_verifier_sha256 for row in corpus.groups})
        ),
        output_verifier_sha256s=tuple(
            sorted({row.output_verifier_sha256 for row in corpus.groups})
        ),
        config=selected_config,
        train=_split_record(corpus, train_indices),
        calibration=_split_record(corpus, calibration_indices),
        models=tuple(models),
    )
    result._verify_split(corpus)
    return result


@dataclass(frozen=True, slots=True)
class HoldoutModelResult:
    model_sha256: str
    family: str
    k: int
    quant_bits: int
    metrics: CollisionMetrics
    promoted: bool
    promotion_reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_sha256",
            require_sha256(self.model_sha256, field="model_sha256"),
        )
        if self.family not in _MODEL_FAMILIES:
            raise ValueError("holdout model family is invalid")
        _uint(self.k, field="k")
        _uint(self.quant_bits, field="quant_bits")
        if not isinstance(self.metrics, CollisionMetrics):
            raise TypeError("metrics must be CollisionMetrics")
        if not isinstance(self.promoted, bool):
            raise TypeError("promoted must be bool")
        expected_reason = (
            "promoted-zero-wrong-collision"
            if self.promoted
            else (
                "control-not-promotable"
                if self.family != "candidate"
                else "wrong-collision-or-no-verified-hit"
            )
        )
        if self.promotion_reason != expected_reason:
            raise ValueError("holdout promotion reason is inconsistent")

    @property
    def key(self) -> tuple[str, int, int]:
        return self.family, self.k, self.quant_bits

    def to_record(self) -> dict[str, object]:
        return {
            "family": self.family,
            "k": self.k,
            "metrics": self.metrics.to_record(),
            "model_sha256": self.model_sha256,
            "promoted": self.promoted,
            "promotion_reason": self.promotion_reason,
            "quant_bits": self.quant_bits,
        }

    @classmethod
    def from_record(cls, value: object) -> "HoldoutModelResult":
        if not isinstance(value, Mapping) or set(value) != {
            "family",
            "k",
            "metrics",
            "model_sha256",
            "promoted",
            "promotion_reason",
            "quant_bits",
        }:
            raise SubspaceBatteryIntegrityError("holdout model result is invalid")
        try:
            return cls(
                model_sha256=cast(str, value.get("model_sha256")),
                family=cast(str, value.get("family")),
                k=cast(int, value.get("k")),
                quant_bits=cast(int, value.get("quant_bits")),
                metrics=CollisionMetrics.from_record(value.get("metrics")),
                promoted=cast(bool, value.get("promoted")),
                promotion_reason=cast(str, value.get("promotion_reason")),
            )
        except SubspaceBatteryIntegrityError:
            raise
        except (TypeError, ValueError) as exc:
            raise SubspaceBatteryIntegrityError(
                "holdout model result validation failed"
            ) from exc


@dataclass(frozen=True, slots=True)
class SubspaceBatteryEvaluation:
    corpus_sha256: str
    fit_sha256: str
    model_pin_sha256: str
    holdout: Mapping[str, object]
    results: tuple[HoldoutModelResult, ...]
    promoted_model_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in ("corpus_sha256", "fit_sha256", "model_pin_sha256"):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        holdout = _validate_split(self.holdout, label="holdout")
        results = tuple(self.results)
        if not results or any(
            not isinstance(row, HoldoutModelResult) for row in results
        ):
            raise ValueError("holdout result inventory is invalid")
        if len({row.key for row in results}) != len(results):
            raise ValueError("holdout result inventory is duplicated")
        promoted = _hashes(
            self.promoted_model_sha256s,
            field="promoted_model_sha256s",
            sorted_unique=True,
            allow_empty=True,
        )
        expected_promoted = tuple(
            sorted(row.model_sha256 for row in results if row.promoted)
        )
        if promoted != expected_promoted:
            raise ValueError("promoted model inventory is inconsistent")
        object.__setattr__(self, "holdout", holdout)
        object.__setattr__(self, "results", results)
        object.__setattr__(self, "promoted_model_sha256s", promoted)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def to_document(self) -> dict[str, object]:
        return _sealed(
            SUBSPACE_HOLDOUT_SCHEMA,
            {
                "attention_kind": SUBSPACE_ATTENTION_KIND,
                "corpus_sha256": self.corpus_sha256,
                "evaluator_verifier_sha256": SUBSPACE_EXACT_VERIFIER_SHA256,
                "fit_sha256": self.fit_sha256,
                "holdout": dict(self.holdout),
                "model_pin_sha256": self.model_pin_sha256,
                "numeric_abi": SUBSPACE_NUMERIC_ABI,
                "promoted_model_sha256s": list(self.promoted_model_sha256s),
                "promotion_policy": (
                    "candidate-only/train+calibration+holdout-zero-wrong/"
                    "calibration-and-holdout-exact-hit/v1"
                ),
                "results": [row.to_record() for row in self.results],
            },
        )

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_document())
        if len(data) > MAX_RECEIPT_BYTES:
            raise SubspaceBatteryCapacityError("subspace holdout exceeds byte bound")
        return data

    def verify_against(self, fit: SubspaceBatteryFit, corpus: SubspaceCorpus) -> None:
        if not isinstance(fit, SubspaceBatteryFit):
            raise TypeError("fit must be a SubspaceBatteryFit")
        fit.verify_against(corpus)
        if (
            self.corpus_sha256 != corpus.sha256
            or self.fit_sha256 != fit.sha256
            or self.model_pin_sha256 != corpus.model_pin_sha256
        ):
            raise SubspaceBatteryIntegrityError(
                "subspace holdout belongs to another fit or corpus"
            )
        indices = _indices(
            cast(list[int], self.holdout.get("indices")),
            field="holdout.indices",
            count=len(corpus.groups),
        )
        if dict(self.holdout) != _split_record(corpus, indices):
            raise SubspaceBatteryIntegrityError(
                "subspace holdout split differs from authenticated corpus"
            )
        recomputed = _evaluate_subspace_battery(
            fit,
            corpus,
            holdout_group_indices=indices,
        )
        if recomputed.to_bytes() != self.to_bytes():
            raise SubspaceBatteryIntegrityError(
                "subspace holdout differs from complete recomputation"
            )

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        fit: SubspaceBatteryFit,
        corpus: SubspaceCorpus,
    ) -> "SubspaceBatteryEvaluation":
        envelope = _strict_json(data, schema=SUBSPACE_HOLDOUT_SCHEMA, label="holdout")
        body = envelope.get("body")
        expected = {
            "attention_kind",
            "corpus_sha256",
            "evaluator_verifier_sha256",
            "fit_sha256",
            "holdout",
            "model_pin_sha256",
            "numeric_abi",
            "promoted_model_sha256s",
            "promotion_policy",
            "results",
        }
        if (
            not isinstance(body, Mapping)
            or set(body) != expected
            or body.get("attention_kind") != SUBSPACE_ATTENTION_KIND
            or body.get("numeric_abi") != SUBSPACE_NUMERIC_ABI
            or body.get("evaluator_verifier_sha256") != SUBSPACE_EXACT_VERIFIER_SHA256
            or body.get("promotion_policy")
            != (
                "candidate-only/train+calibration+holdout-zero-wrong/"
                "calibration-and-holdout-exact-hit/v1"
            )
            or not isinstance(body.get("promoted_model_sha256s"), list)
            or not isinstance(body.get("results"), list)
        ):
            raise SubspaceBatteryIntegrityError("subspace holdout body is invalid")
        try:
            result = cls(
                corpus_sha256=cast(str, body.get("corpus_sha256")),
                fit_sha256=cast(str, body.get("fit_sha256")),
                model_pin_sha256=cast(str, body.get("model_pin_sha256")),
                holdout=_validate_split(body.get("holdout"), label="holdout"),
                results=tuple(
                    HoldoutModelResult.from_record(row)
                    for row in cast(list[object], body.get("results"))
                ),
                promoted_model_sha256s=tuple(
                    cast(list[str], body.get("promoted_model_sha256s"))
                ),
            )
        except SubspaceBatteryIntegrityError:
            raise
        except (TypeError, ValueError) as exc:
            raise SubspaceBatteryIntegrityError(
                "subspace holdout validation failed"
            ) from exc
        if result.to_bytes() != data:
            raise SubspaceBatteryIntegrityError(
                "subspace holdout reconstruction changed"
            )
        result.verify_against(fit, corpus)
        return result


def _replay_to_holdout_cache(
    model: SubspaceKeyModel,
    train: Sequence[SubspaceObservationGroup],
    calibration: Sequence[SubspaceObservationGroup],
) -> _CacheState:
    cache = _build_cache(model, train)
    _simulate_queries(
        cache,
        model.family,
        model.basis_indices,
        model.quant_bits,
        model.scale,
        calibration,
        update=True,
        basis_bytes=4 * len(model.basis_indices),
        scale_bytes=0 if model.scale is None else int(model.scale.nbytes),
    )
    return cache


def _evaluate_subspace_battery(
    fit: SubspaceBatteryFit,
    corpus: SubspaceCorpus,
    *,
    holdout_group_indices: Sequence[int],
) -> SubspaceBatteryEvaluation:
    """Evaluate later groups and promote only zero-wrong candidate keys."""

    if not isinstance(fit, SubspaceBatteryFit):
        raise TypeError("fit must be a SubspaceBatteryFit")
    if not isinstance(corpus, SubspaceCorpus):
        raise TypeError("corpus must be a SubspaceCorpus")
    fit._verify_split(corpus)
    holdout_indices = _indices(
        holdout_group_indices,
        field="holdout_group_indices",
        count=len(corpus.groups),
    )
    fit_indices = set(cast(list[int], fit.train.get("indices"))) | set(
        cast(list[int], fit.calibration.get("indices"))
    )
    if fit_indices & set(holdout_indices):
        raise ValueError("subspace holdout overlaps train or calibration")
    if max(corpus.groups[index].logical_time for index in fit_indices) >= min(
        corpus.groups[index].logical_time for index in holdout_indices
    ):
        raise ValueError("subspace holdout must follow train and calibration")
    train = _groups(corpus, cast(list[int], fit.train.get("indices")))
    calibration = _groups(corpus, cast(list[int], fit.calibration.get("indices")))
    holdout = _groups(corpus, holdout_indices)
    results = []
    for model in fit.models:
        cache = _replay_to_holdout_cache(model, train, calibration)
        metrics = _simulate_queries(
            cache,
            model.family,
            model.basis_indices,
            model.quant_bits,
            model.scale,
            holdout,
            update=True,
            basis_bytes=4 * len(model.basis_indices),
            scale_bytes=0 if model.scale is None else int(model.scale.nbytes),
        )
        promoted = bool(
            model.family == "candidate"
            and model.calibration_safe
            and model.calibration_metrics.exact_verified_hits > 0
            and not metrics.prior_wrong_collision_rows
            and metrics.wrong_collisions == 0
            and metrics.exact_verified_hits > 0
        )
        reason = (
            "promoted-zero-wrong-collision"
            if promoted
            else (
                "control-not-promotable"
                if model.family != "candidate"
                else "wrong-collision-or-no-verified-hit"
            )
        )
        results.append(
            HoldoutModelResult(
                model_sha256=model.sha256,
                family=model.family,
                k=model.k,
                quant_bits=model.quant_bits,
                metrics=metrics,
                promoted=promoted,
                promotion_reason=reason,
            )
        )
    result = SubspaceBatteryEvaluation(
        corpus_sha256=corpus.sha256,
        fit_sha256=fit.sha256,
        model_pin_sha256=corpus.model_pin_sha256,
        holdout=_split_record(corpus, holdout_indices),
        results=tuple(results),
        promoted_model_sha256s=tuple(
            sorted(row.model_sha256 for row in results if row.promoted)
        ),
    )
    return result


def evaluate_subspace_battery(
    fit: SubspaceBatteryFit,
    corpus: SubspaceCorpus,
    *,
    holdout_group_indices: Sequence[int],
) -> SubspaceBatteryEvaluation:
    """Build and fully recompute-verify the later collision holdout."""

    result = _evaluate_subspace_battery(
        fit, corpus, holdout_group_indices=holdout_group_indices
    )
    result.verify_against(fit, corpus)
    return result


@dataclass(frozen=True, slots=True)
class SubspaceInstrumentationRequest:
    harvester_state_sha256: str
    harvester_identity_sha256: str
    model_pin_sha256s: tuple[str, ...]
    graph_revision_sha256s: tuple[str, ...]
    available_transition_receipt_sha256s: tuple[str, ...]
    missing_evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in ("harvester_state_sha256", "harvester_identity_sha256"):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        for field in (
            "model_pin_sha256s",
            "graph_revision_sha256s",
            "available_transition_receipt_sha256s",
        ):
            object.__setattr__(
                self,
                field,
                _hashes(
                    getattr(self, field),
                    field=field,
                    sorted_unique=True,
                    allow_empty=True,
                ),
            )
        missing = tuple(self.missing_evidence)
        expected = (
            "exact-rmsnorm-post-attention-context-float64",
            "joint-gate-proj-output-float64",
            "joint-up-proj-output-float64",
            "exact-output-payload-sha256-per-row",
            "context-to-gate-up-projection-verifier",
        )
        if missing != expected:
            raise ValueError("instrumentation request missing-evidence ABI changed")
        object.__setattr__(self, "missing_evidence", missing)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def to_document(self) -> dict[str, object]:
        return _sealed(
            SUBSPACE_INSTRUMENT_REQUEST_SCHEMA,
            {
                "attention_kind": SUBSPACE_ATTENTION_KIND,
                "available_transition_receipt_sha256s": list(
                    self.available_transition_receipt_sha256s
                ),
                "context_kind": SUBSPACE_CONTEXT_KIND,
                "graph_revision_sha256s": list(self.graph_revision_sha256s),
                "harvester_identity_sha256": self.harvester_identity_sha256,
                "harvester_state_sha256": self.harvester_state_sha256,
                "missing_evidence": list(self.missing_evidence),
                "model_pin_sha256s": list(self.model_pin_sha256s),
                "required_observer_stages": [
                    "mlp.input",
                    "mlp.gate",
                    "mlp.up",
                    "mlp.output",
                ],
                "static_embedding_means_allowed": False,
                "softmax_attention_allowed": False,
            },
        )

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_document())

    @classmethod
    def from_bytes(cls, data: bytes) -> "SubspaceInstrumentationRequest":
        envelope = _strict_json(
            data,
            schema=SUBSPACE_INSTRUMENT_REQUEST_SCHEMA,
            label="instrument request",
        )
        body = envelope.get("body")
        expected = {
            "attention_kind",
            "available_transition_receipt_sha256s",
            "context_kind",
            "graph_revision_sha256s",
            "harvester_identity_sha256",
            "harvester_state_sha256",
            "missing_evidence",
            "model_pin_sha256s",
            "required_observer_stages",
            "static_embedding_means_allowed",
            "softmax_attention_allowed",
        }
        if (
            not isinstance(body, Mapping)
            or set(body) != expected
            or body.get("attention_kind") != SUBSPACE_ATTENTION_KIND
            or body.get("context_kind") != SUBSPACE_CONTEXT_KIND
            or body.get("required_observer_stages")
            != ["mlp.input", "mlp.gate", "mlp.up", "mlp.output"]
            or body.get("static_embedding_means_allowed") is not False
            or body.get("softmax_attention_allowed") is not False
        ):
            raise SubspaceBatteryIntegrityError(
                "instrumentation request body is invalid"
            )
        for field in (
            "available_transition_receipt_sha256s",
            "graph_revision_sha256s",
            "missing_evidence",
            "model_pin_sha256s",
        ):
            if not isinstance(body.get(field), list):
                raise SubspaceBatteryIntegrityError(
                    "instrumentation request inventory is invalid"
                )
        try:
            result = cls(
                harvester_state_sha256=cast(str, body.get("harvester_state_sha256")),
                harvester_identity_sha256=cast(
                    str, body.get("harvester_identity_sha256")
                ),
                model_pin_sha256s=tuple(cast(list[str], body.get("model_pin_sha256s"))),
                graph_revision_sha256s=tuple(
                    cast(list[str], body.get("graph_revision_sha256s"))
                ),
                available_transition_receipt_sha256s=tuple(
                    cast(
                        list[str],
                        body.get("available_transition_receipt_sha256s"),
                    )
                ),
                missing_evidence=tuple(cast(list[str], body.get("missing_evidence"))),
            )
        except (TypeError, ValueError) as exc:
            raise SubspaceBatteryIntegrityError(
                "instrumentation request validation failed"
            ) from exc
        if result.to_bytes() != data:
            raise SubspaceBatteryIntegrityError(
                "instrumentation request reconstruction changed"
            )
        return result


def inspect_harvester_subspace_evidence(
    state: HarvesterState,
) -> SubspaceInstrumentationRequest:
    """Authenticate the current sketch state and describe exact missing evidence."""

    if not isinstance(state, HarvesterState):
        raise TypeError("state must be a HarvesterState")
    try:
        authenticated = HarvesterState.from_bytes(state.to_bytes())
    except (OperatorHarvesterIntegrityError, TypeError, ValueError) as exc:
        raise SubspaceBatteryIntegrityError(
            "harvester state failed exact reconstruction"
        ) from exc
    if authenticated.to_bytes() != state.to_bytes():
        raise SubspaceBatteryIntegrityError("harvester state reconstruction differs")
    observations = tuple(
        observation for _group, rows in state.groups for observation in rows
    )
    model_pins = tuple(sorted({row.receipt.model_pin_sha256 for row in observations}))
    revisions = tuple(
        sorted({row.receipt.atlas_revision.sha256 for row in observations})
    )
    available = tuple(
        sorted(
            row.receipt.sha256
            for row in observations
            if ".mlp.input-sketch" in row.receipt.source_state
            or ".mlp.output-sketch" in row.receipt.target_state
        )
    )
    return SubspaceInstrumentationRequest(
        harvester_state_sha256=state.sha256,
        harvester_identity_sha256=state.identity_sha256,
        model_pin_sha256s=model_pins,
        graph_revision_sha256s=revisions,
        available_transition_receipt_sha256s=available,
        missing_evidence=(
            "exact-rmsnorm-post-attention-context-float64",
            "joint-gate-proj-output-float64",
            "joint-up-proj-output-float64",
            "exact-output-payload-sha256-per-row",
            "context-to-gate-up-projection-verifier",
        ),
    )


def subspace_corpus_from_harvester_state(state: HarvesterState) -> SubspaceCorpus:
    """Fail closed: projected boundary sketches cannot become joint Qwen keys."""

    request = inspect_harvester_subspace_evidence(state)
    raise SubspaceInstrumentationRequired(request)


__all__ = [
    "DEFAULT_K_VALUES",
    "DEFAULT_QUANT_BITS",
    "SUBSPACE_ATTENTION_KIND",
    "SUBSPACE_CONTEXT_KIND",
    "SUBSPACE_CONTENT_ADDRESS_SCHEMA",
    "SUBSPACE_CORPUS_SCHEMA",
    "SUBSPACE_EXACT_VERIFIER_SHA256",
    "SUBSPACE_FIT_SCHEMA",
    "SUBSPACE_HOLDOUT_SCHEMA",
    "SUBSPACE_INSTRUMENT_REQUEST_SCHEMA",
    "SUBSPACE_NUMERIC_ABI",
    "CollisionMetrics",
    "HoldoutModelResult",
    "SubspaceBatteryCapacityError",
    "SubspaceBatteryError",
    "SubspaceBatteryEvaluation",
    "SubspaceBatteryFit",
    "SubspaceBatteryIntegrityError",
    "SubspaceCorpus",
    "SubspaceInstrumentationRequired",
    "SubspaceInstrumentationRequest",
    "SubspaceKeyModel",
    "SubspaceObservationGroup",
    "SubspaceSweepConfig",
    "evaluate_subspace_battery",
    "fit_subspace_battery",
    "graph_revision_sha256",
    "inspect_harvester_subspace_evidence",
    "output_evidence_sha256",
    "projection_evidence_sha256",
    "subspace_corpus_from_harvester_state",
]
