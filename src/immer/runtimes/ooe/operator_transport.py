"""Analysis-only transport diagnostics for causal Prefix-Sinkhorn operators.

For source and target attention operators ``A`` and ``B`` this module fits the
intertwiner ``C`` in ``C A = B C``.  It compares separately fitted head-pair
maps with a global map, identity, and a deterministic random orthogonal
placebo.  Every fit is trained and evaluated on disjoint prompt/position
groups and is replayed from the sealed corpus during deserialization.

The current Qwen semantic atlas does *not* persist per-head attention
operators.  Its adapter therefore returns a sealed missing-measurement receipt
and refuses to manufacture matrices from summaries or from archive Softmax.
The required production seam is the routed probability tensor emitted by
``Qwen38NativeHeadCrsa.route``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
import base64
import binascii
import fcntl
import hashlib
import json
import math
import os
import stat
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
import torch

from immer.attention.crsa.operators import AttentionSpec, prefix_log
from immer.runtimes.qwen3_8.semantic_atlas import (
    GraphRevision,
    MeasurementReceipt,
)

from .crystal import CrystalStore, CrystalStoreError, CrystalTamperError
from .identity import canonical_json_bytes, require_sha256


QWEN_PREFIX_SINKHORN_ATTENTION_MODE = "native-prefix-sinkhorn"
OPERATOR_TRANSPORT_OBSERVATION_SCHEMA = "immer-ooe-operator-transport-observation/v1"
OPERATOR_TRANSPORT_CORPUS_SCHEMA = "immer-ooe-operator-transport-corpus/v1"
OPERATOR_TRANSPORT_SPLIT_SCHEMA = "immer-ooe-operator-transport-split/v1"
OPERATOR_TRANSPORT_CONFIG_SCHEMA = "immer-ooe-operator-transport-config/v1"
OPERATOR_TRANSPORT_KERNEL_SCHEMA = "immer-ooe-operator-transport-kernel/v1"
OPERATOR_TRANSPORT_METRICS_SCHEMA = "immer-ooe-operator-transport-metrics/v1"
OPERATOR_TRANSPORT_FIT_SCHEMA = "immer-ooe-operator-transport-fit/v1"
OPERATOR_TRANSPORT_HOLDOUT_SCHEMA = "immer-ooe-operator-transport-holdout/v1"
QWEN_OPERATOR_TRANSPORT_AVAILABILITY_SCHEMA = (
    "immer-ooe-qwen-operator-transport-availability/v1"
)
QWEN_PREFIX_SINKHORN_CAPTURE_SCHEMA = "immer-ooe-prefix-sinkhorn-capture/v1"
QWEN_PREFIX_SINKHORN_CAPTURE_INVENTORY_SCHEMA = (
    "immer-ooe-prefix-sinkhorn-capture-inventory/v1"
)

MAX_OPERATOR_DIMENSION = 32
MAX_OPERATOR_OBSERVATIONS = 100_000
MAX_OPERATOR_TRANSPORT_BYTES = 256 * 1024 * 1024
MAX_PREFIX_SINKHORN_CAPTURE_BYTES = 1024 * 1024
_QWEN_OPERATOR_SEAM = (
    "Qwen38NativeHeadCrsa.route:streaming_prefix_log.routed"
    "[batch,selected_head,query,key]-before-native-blend"
)
_MISSING_QWEN_MEASUREMENT = "per-head-prefix-sinkhorn-operator-matrix"


class OperatorTransportError(ValueError):
    """Base error for operator-transport diagnostics."""


class OperatorTransportIntegrityError(OperatorTransportError):
    """A seal, evidence binding, coefficient, or metric changed."""


class OperatorTransportLeakageError(OperatorTransportError):
    """Prompt/position groups overlap across temporal partitions."""


class OperatorTransportFitError(OperatorTransportError):
    """The intertwiner is rank deficient, ill-conditioned, or unidentified."""


class OperatorTransportMissingMeasurementError(OperatorTransportError):
    """Qwen artifacts lack the required native per-head operator matrices."""


class OperatorTransportCaptureConflictError(OperatorTransportIntegrityError):
    """One measurement/capture key was already bound to different bytes."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _seal(schema: str, body: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(body)
    return {"schema": schema, "body": normalized, "body_sha256": _digest(normalized)}


def _strict_json(data: bytes, *, label: str) -> object:
    if not isinstance(data, bytes) or not data:
        raise OperatorTransportIntegrityError(f"{label} must be immutable bytes")
    if len(data) > MAX_OPERATOR_TRANSPORT_BYTES:
        raise OperatorTransportIntegrityError(f"{label} exceeds its byte bound")

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
        raise OperatorTransportIntegrityError(f"{label} is not strict JSON") from exc
    if canonical_json_bytes(value) != data:
        raise OperatorTransportIntegrityError(f"{label} is not canonical JSON")
    return value


def _unseal(value: object, *, schema: str, label: str) -> Mapping[str, object]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema", "body", "body_sha256"}
        or value.get("schema") != schema
        or not isinstance(value.get("body"), Mapping)
    ):
        raise OperatorTransportIntegrityError(f"invalid {label} envelope")
    body = cast(Mapping[str, object], value.get("body"))
    try:
        claimed = require_sha256(value.get("body_sha256"), field="body_sha256")
    except ValueError as exc:
        raise OperatorTransportIntegrityError(f"invalid {label} body hash") from exc
    if claimed != _digest(body):
        raise OperatorTransportIntegrityError(f"{label} body hash mismatch")
    return body


def _text(value: object, *, field: str, maximum: int = 1024) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > maximum
    ):
        raise OperatorTransportError(f"{field} must be canonical non-empty text")
    return value


def _uint(value: object, *, field: str, positive: bool = False) -> int:
    lower = 1 if positive else 0
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not lower <= value <= (1 << 63) - 1
    ):
        qualifier = "positive" if positive else "non-negative"
        raise OperatorTransportError(f"{field} must be a bounded {qualifier} integer")
    return value


def _finite(
    value: object,
    *,
    field: str,
    nonnegative: bool = False,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OperatorTransportError(f"{field} must be finite numeric data")
    result = float(value)
    if not math.isfinite(result):
        raise OperatorTransportError(f"{field} must be finite")
    if (nonnegative and result < 0.0) or (positive and result <= 0.0):
        raise OperatorTransportError(f"{field} violates its sign bound")
    return 0.0 if result == 0.0 else result


def _fraction(value: Fraction, *, field: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError(f"{field} must be exact int/Fraction data")
    result = Fraction(value)
    if not 0 < result < 1:
        raise OperatorTransportError(f"{field} must lie strictly inside (0, 1)")
    return result


def _matrix(value: object, *, field: str) -> NDArray[np.float64]:
    if type(value) is not np.ndarray:
        raise TypeError(f"{field} must be an exact numpy.ndarray")
    array = cast(NDArray[Any], value)
    if array.dtype != np.dtype(np.float64):
        raise TypeError(f"{field} must use float64")
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise OperatorTransportError(f"{field} must be square")
    dimension = int(array.shape[0])
    if not 3 <= dimension <= MAX_OPERATOR_DIMENSION:
        raise OperatorTransportError(
            f"{field} dimension must lie in [3, {MAX_OPERATOR_DIMENSION}]"
        )
    if not np.isfinite(array).all():
        raise OperatorTransportError(f"{field} must be finite")
    result = np.array(array, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _operator(value: object, *, field: str) -> NDArray[np.float64]:
    matrix = _matrix(value, field=field)
    # Native Qwen routes are computed in float32 when the checkpoint runs in
    # BF16. Preserve those exact promoted float32 values in float64 storage;
    # only the row-sum validation uses a float32-scale tolerance.
    if float(np.max(np.abs(np.triu(matrix, 1)))) != 0.0:
        raise OperatorTransportError(f"{field} is not causal lower-triangular")
    if float(np.min(matrix)) < 0.0:
        raise OperatorTransportError(f"{field} contains negative probability mass")
    if float(np.max(np.abs(matrix.sum(axis=1) - 1.0))) > 2.0**-18:
        raise OperatorTransportError(f"{field} rows do not sum to one")
    result = np.array(matrix, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _matrix_record(matrix: NDArray[np.float64]) -> dict[str, object]:
    value = _matrix(matrix, field="matrix")
    little = np.asarray(value, dtype="<f8", order="C")
    raw = little.tobytes(order="C")
    return {
        "dtype": "float64-le",
        "shape": list(value.shape),
        "data_base64": base64.b64encode(raw).decode("ascii"),
        "data_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _matrix_from_record(value: object, *, field: str) -> NDArray[np.float64]:
    if not isinstance(value, Mapping) or set(value) != {
        "dtype",
        "shape",
        "data_base64",
        "data_sha256",
    }:
        raise OperatorTransportIntegrityError(f"invalid {field} matrix record")
    if value.get("dtype") != "float64-le":
        raise OperatorTransportIntegrityError(f"invalid {field} matrix dtype")
    shape = value.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in shape)
        or shape[0] != shape[1]
        or not 3 <= shape[0] <= MAX_OPERATOR_DIMENSION
    ):
        raise OperatorTransportIntegrityError(f"invalid {field} matrix shape")
    encoded = value.get("data_base64")
    if not isinstance(encoded, str):
        raise OperatorTransportIntegrityError(f"invalid {field} matrix payload")
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError, binascii.Error) as exc:
        raise OperatorTransportIntegrityError(
            f"invalid {field} matrix encoding"
        ) from exc
    try:
        claimed = require_sha256(value.get("data_sha256"), field="data_sha256")
    except ValueError as exc:
        raise OperatorTransportIntegrityError(
            f"invalid {field} matrix digest"
        ) from exc
    if claimed != hashlib.sha256(raw).hexdigest() or len(raw) != shape[0] * shape[1] * 8:
        raise OperatorTransportIntegrityError(f"{field} matrix digest/length mismatch")
    result = np.frombuffer(raw, dtype="<f8").astype(np.float64, copy=True)
    result = result.reshape((shape[0], shape[1]))
    return _matrix(result, field=field)


def causal_prefix_sinkhorn_operator(
    logits: NDArray[np.float64],
    *,
    alpha: float = 1.0,
    diagonal_debit: float = 0.0,
) -> NDArray[np.float64]:
    """Evaluate IMMER's causal Prefix-Sinkhorn operator for one head."""

    matrix = _matrix(logits, field="logits")
    alpha_value = _finite(alpha, field="alpha", nonnegative=True)
    debit = _finite(
        diagonal_debit, field="diagonal_debit", nonnegative=True
    )
    tensor = torch.from_numpy(np.array(matrix, copy=True)).reshape(
        1, 1, matrix.shape[0], matrix.shape[1]
    )
    with torch.no_grad():
        routed = prefix_log(
            tensor,
            AttentionSpec(
                kind="prefix_log",
                alpha=alpha_value,
                diagonal_debit=debit,
            ),
        )[0, 0]
    result = np.array(routed.cpu().numpy(), dtype=np.float64, order="C", copy=True)
    return _operator(result, field="prefix_sinkhorn_operator")


def operator_transport_evidence_sha256(
    *,
    temporal_index: int,
    model_pin_sha256: str,
    graph_revision: GraphRevision,
    attention_mode: str,
    prompt_sha256: str,
    position_group_sha256: str,
    source_head: int,
    target_head: int,
    source_measurement_sha256: str,
    target_measurement_sha256: str,
    source_operator: NDArray[np.float64],
    target_operator: NDArray[np.float64],
) -> str:
    """Derive the evidence address from every observation authority and byte."""

    time = _uint(temporal_index, field="temporal_index", positive=True)
    if not isinstance(graph_revision, GraphRevision):
        raise TypeError("graph_revision must be GraphRevision")
    mode = _text(attention_mode, field="attention_mode", maximum=64)
    if mode != QWEN_PREFIX_SINKHORN_ATTENTION_MODE:
        raise OperatorTransportError(
            "operator evidence accepts only native Prefix-Sinkhorn attention"
        )
    source = _operator(source_operator, field="source_operator")
    target = _operator(target_operator, field="target_operator")
    if source.shape != target.shape:
        raise OperatorTransportError("operator evidence dimensions differ")
    return _digest(
        {
            "schema": "immer-ooe-operator-transport-evidence/v1",
            "temporal_index": time,
            "model_pin_sha256": require_sha256(
                model_pin_sha256, field="model_pin_sha256"
            ),
            "graph_revision": graph_revision.to_document(),
            "attention_mode": mode,
            "prompt_sha256": require_sha256(
                prompt_sha256, field="prompt_sha256"
            ),
            "position_group_sha256": require_sha256(
                position_group_sha256, field="position_group_sha256"
            ),
            "source_head": _uint(source_head, field="source_head"),
            "target_head": _uint(target_head, field="target_head"),
            "source_measurement_sha256": require_sha256(
                source_measurement_sha256, field="source_measurement_sha256"
            ),
            "target_measurement_sha256": require_sha256(
                target_measurement_sha256, field="target_measurement_sha256"
            ),
            "source_operator_sha256": cast(
                str, _matrix_record(source)["data_sha256"]
            ),
            "target_operator_sha256": cast(
                str, _matrix_record(target)["data_sha256"]
            ),
        }
    )


@dataclass(frozen=True, slots=True)
class OperatorTransportObservation:
    temporal_index: int
    model_pin_sha256: str
    graph_revision: GraphRevision
    attention_mode: str
    prompt_sha256: str
    position_group_sha256: str
    source_head: int
    target_head: int
    source_measurement_sha256: str
    target_measurement_sha256: str
    evidence_sha256: str
    source_operator: NDArray[np.float64]
    target_operator: NDArray[np.float64]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "temporal_index",
            _uint(self.temporal_index, field="temporal_index", positive=True),
        )
        if not isinstance(self.graph_revision, GraphRevision):
            raise TypeError("graph_revision must be a GraphRevision")
        for field in (
            "model_pin_sha256",
            "prompt_sha256",
            "position_group_sha256",
            "source_measurement_sha256",
            "target_measurement_sha256",
            "evidence_sha256",
        ):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        mode = _text(self.attention_mode, field="attention_mode", maximum=64)
        if mode != QWEN_PREFIX_SINKHORN_ATTENTION_MODE:
            raise OperatorTransportError(
                "operator transport accepts only native Prefix-Sinkhorn attention"
            )
        source_head = _uint(self.source_head, field="source_head")
        target_head = _uint(self.target_head, field="target_head")
        source = _operator(self.source_operator, field="source_operator")
        target = _operator(self.target_operator, field="target_operator")
        if source.shape != target.shape:
            raise OperatorTransportError("source and target operator shapes differ")
        expected_evidence = operator_transport_evidence_sha256(
            temporal_index=self.temporal_index,
            model_pin_sha256=self.model_pin_sha256,
            graph_revision=self.graph_revision,
            attention_mode=mode,
            prompt_sha256=self.prompt_sha256,
            position_group_sha256=self.position_group_sha256,
            source_head=source_head,
            target_head=target_head,
            source_measurement_sha256=self.source_measurement_sha256,
            target_measurement_sha256=self.target_measurement_sha256,
            source_operator=source,
            target_operator=target,
        )
        if self.evidence_sha256 != expected_evidence:
            raise OperatorTransportIntegrityError(
                "observation evidence differs from its authorities and operators"
            )
        object.__setattr__(self, "attention_mode", mode)
        object.__setattr__(self, "source_head", source_head)
        object.__setattr__(self, "target_head", target_head)
        object.__setattr__(self, "source_operator", source)
        object.__setattr__(self, "target_operator", target)

    @property
    def split_group_sha256(self) -> str:
        return _digest(
            {
                "schema": "immer-ooe-operator-transport-group/v1",
                "prompt_sha256": self.prompt_sha256,
                "position_group_sha256": self.position_group_sha256,
            }
        )

    def to_dict(self) -> dict[str, object]:
        return _seal(
            OPERATOR_TRANSPORT_OBSERVATION_SCHEMA,
            {
                "temporal_index": self.temporal_index,
                "model_pin_sha256": self.model_pin_sha256,
                "graph_revision": self.graph_revision.to_document(),
                "attention_mode": self.attention_mode,
                "prompt_sha256": self.prompt_sha256,
                "position_group_sha256": self.position_group_sha256,
                "split_group_sha256": self.split_group_sha256,
                "source_head": self.source_head,
                "target_head": self.target_head,
                "source_measurement_sha256": self.source_measurement_sha256,
                "target_measurement_sha256": self.target_measurement_sha256,
                "evidence_sha256": self.evidence_sha256,
                "source_operator": _matrix_record(self.source_operator),
                "target_operator": _matrix_record(self.target_operator),
            },
        )

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> "OperatorTransportObservation":
        body = _unseal(
            value,
            schema=OPERATOR_TRANSPORT_OBSERVATION_SCHEMA,
            label="operator transport observation",
        )
        expected = {
            "temporal_index",
            "model_pin_sha256",
            "graph_revision",
            "attention_mode",
            "prompt_sha256",
            "position_group_sha256",
            "split_group_sha256",
            "source_head",
            "target_head",
            "source_measurement_sha256",
            "target_measurement_sha256",
            "evidence_sha256",
            "source_operator",
            "target_operator",
        }
        if set(body) != expected:
            raise OperatorTransportIntegrityError("invalid observation body")
        try:
            result = cls(
                temporal_index=cast(int, body.get("temporal_index")),
                model_pin_sha256=cast(str, body.get("model_pin_sha256")),
                graph_revision=GraphRevision.from_document(
                    cast(Mapping[str, Any], body.get("graph_revision"))
                ),
                attention_mode=cast(str, body.get("attention_mode")),
                prompt_sha256=cast(str, body.get("prompt_sha256")),
                position_group_sha256=cast(
                    str, body.get("position_group_sha256")
                ),
                source_head=cast(int, body.get("source_head")),
                target_head=cast(int, body.get("target_head")),
                source_measurement_sha256=cast(
                    str, body.get("source_measurement_sha256")
                ),
                target_measurement_sha256=cast(
                    str, body.get("target_measurement_sha256")
                ),
                evidence_sha256=cast(str, body.get("evidence_sha256")),
                source_operator=_matrix_from_record(
                    body.get("source_operator"), field="source_operator"
                ),
                target_operator=_matrix_from_record(
                    body.get("target_operator"), field="target_operator"
                ),
            )
        except (TypeError, ValueError) as exc:
            raise OperatorTransportIntegrityError(
                "operator transport observation failed validation"
            ) from exc
        if body.get("split_group_sha256") != result.split_group_sha256:
            raise OperatorTransportIntegrityError("observation group identity changed")
        if result.to_dict() != dict(value):
            raise OperatorTransportIntegrityError(
                "observation changed during reconstruction"
            )
        return result


@dataclass(frozen=True, slots=True)
class OperatorTransportCorpus:
    model_pin_sha256: str
    graph_revision: GraphRevision
    attention_mode: str
    observations: tuple[OperatorTransportObservation, ...]
    observation_sha256s: tuple[str, ...]
    evidence_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        model_pin = require_sha256(self.model_pin_sha256, field="model_pin_sha256")
        if not isinstance(self.graph_revision, GraphRevision):
            raise TypeError("graph_revision must be a GraphRevision")
        mode = _text(self.attention_mode, field="attention_mode", maximum=64)
        if mode != QWEN_PREFIX_SINKHORN_ATTENTION_MODE:
            raise OperatorTransportError("corpus attention mode is not Prefix-Sinkhorn")
        observations = tuple(self.observations)
        if not 1 <= len(observations) <= MAX_OPERATOR_OBSERVATIONS or any(
            not isinstance(row, OperatorTransportObservation) for row in observations
        ):
            raise OperatorTransportError("corpus observation inventory is invalid")
        observations = tuple(
            sorted(
                observations,
                key=lambda row: (
                    row.temporal_index,
                    row.split_group_sha256,
                    row.source_head,
                    row.target_head,
                    row.sha256,
                ),
            )
        )
        dimension = observations[0].source_operator.shape
        for row in observations:
            if (
                row.model_pin_sha256 != model_pin
                or row.graph_revision != self.graph_revision
                or row.attention_mode != mode
                or row.source_operator.shape != dimension
            ):
                raise OperatorTransportIntegrityError(
                    "corpus observation pin or operator dimension changed"
                )
        hashes = tuple(row.sha256 for row in observations)
        evidence = tuple(row.evidence_sha256 for row in observations)
        if tuple(self.observation_sha256s) != hashes:
            raise OperatorTransportIntegrityError("observation hash inventory changed")
        if tuple(self.evidence_sha256s) != evidence:
            raise OperatorTransportIntegrityError("evidence hash inventory changed")
        if len(set(hashes)) != len(hashes) or len(set(evidence)) != len(evidence):
            raise OperatorTransportIntegrityError("corpus evidence is duplicated")
        group_times: dict[str, int] = {}
        for row in observations:
            prior = group_times.setdefault(row.split_group_sha256, row.temporal_index)
            if prior != row.temporal_index:
                raise OperatorTransportIntegrityError(
                    "one prompt/position group spans temporal indices"
                )
        if len(set(group_times.values())) != len(group_times):
            raise OperatorTransportLeakageError(
                "two prompt/position groups share a temporal index"
            )
        object.__setattr__(self, "model_pin_sha256", model_pin)
        object.__setattr__(self, "attention_mode", mode)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "observation_sha256s", hashes)
        object.__setattr__(self, "evidence_sha256s", evidence)

    @property
    def operator_dimension(self) -> int:
        return int(self.observations[0].source_operator.shape[0])

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return _seal(
            OPERATOR_TRANSPORT_CORPUS_SCHEMA,
            {
                "model_pin_sha256": self.model_pin_sha256,
                "graph_revision": self.graph_revision.to_document(),
                "attention_mode": self.attention_mode,
                "operator_dimension": self.operator_dimension,
                "observations": [row.to_dict() for row in self.observations],
                "observation_sha256s": list(self.observation_sha256s),
                "evidence_sha256s": list(self.evidence_sha256s),
            },
        )

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_dict())
        if len(data) > MAX_OPERATOR_TRANSPORT_BYTES:
            raise OperatorTransportError("operator transport corpus exceeds byte bound")
        return data

    @classmethod
    def from_bytes(cls, data: bytes) -> "OperatorTransportCorpus":
        value = _strict_json(data, label="operator transport corpus")
        body = _unseal(
            value, schema=OPERATOR_TRANSPORT_CORPUS_SCHEMA, label="corpus"
        )
        expected = {
            "model_pin_sha256",
            "graph_revision",
            "attention_mode",
            "operator_dimension",
            "observations",
            "observation_sha256s",
            "evidence_sha256s",
        }
        if set(body) != expected or any(
            not isinstance(body.get(field), list)
            for field in ("observations", "observation_sha256s", "evidence_sha256s")
        ):
            raise OperatorTransportIntegrityError("invalid corpus body")
        try:
            result = cls(
                model_pin_sha256=cast(str, body.get("model_pin_sha256")),
                graph_revision=GraphRevision.from_document(
                    cast(Mapping[str, Any], body.get("graph_revision"))
                ),
                attention_mode=cast(str, body.get("attention_mode")),
                observations=tuple(
                    OperatorTransportObservation.from_dict(row)
                    for row in cast(list[object], body.get("observations"))
                ),
                observation_sha256s=tuple(
                    cast(list[str], body.get("observation_sha256s"))
                ),
                evidence_sha256s=tuple(
                    cast(list[str], body.get("evidence_sha256s"))
                ),
            )
        except (TypeError, ValueError) as exc:
            raise OperatorTransportIntegrityError("corpus failed validation") from exc
        if body.get("operator_dimension") != result.operator_dimension:
            raise OperatorTransportIntegrityError("corpus dimension changed")
        if result.to_bytes() != data:
            raise OperatorTransportIntegrityError("corpus replay changed")
        return result


@dataclass(frozen=True, slots=True)
class OperatorTransportSplit:
    corpus_sha256: str
    train_group_sha256s: tuple[str, ...]
    calibration_group_sha256s: tuple[str, ...]
    holdout_group_sha256s: tuple[str, ...]
    train_observation_sha256s: tuple[str, ...]
    calibration_observation_sha256s: tuple[str, ...]
    holdout_observation_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "corpus_sha256",
            require_sha256(self.corpus_sha256, field="corpus_sha256"),
        )
        sets: list[set[str]] = []
        for field in (
            "train_group_sha256s",
            "calibration_group_sha256s",
            "holdout_group_sha256s",
            "train_observation_sha256s",
            "calibration_observation_sha256s",
            "holdout_observation_sha256s",
        ):
            values = tuple(
                require_sha256(value, field=field) for value in getattr(self, field)
            )
            if not values or len(set(values)) != len(values):
                raise OperatorTransportLeakageError(f"{field} is empty or duplicated")
            object.__setattr__(self, field, values)
            sets.append(set(values))
        for collection in (sets[:3], sets[3:]):
            if any(
                left & right
                for index, left in enumerate(collection)
                for right in collection[index + 1 :]
            ):
                raise OperatorTransportLeakageError(
                    "transport split groups or observations overlap"
                )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return _seal(
            OPERATOR_TRANSPORT_SPLIT_SCHEMA,
            {
                "corpus_sha256": self.corpus_sha256,
                "train_group_sha256s": list(self.train_group_sha256s),
                "calibration_group_sha256s": list(
                    self.calibration_group_sha256s
                ),
                "holdout_group_sha256s": list(self.holdout_group_sha256s),
                "train_observation_sha256s": list(
                    self.train_observation_sha256s
                ),
                "calibration_observation_sha256s": list(
                    self.calibration_observation_sha256s
                ),
                "holdout_observation_sha256s": list(
                    self.holdout_observation_sha256s
                ),
            },
        )

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_bytes(cls, data: bytes) -> "OperatorTransportSplit":
        value = _strict_json(data, label="operator transport split")
        body = _unseal(value, schema=OPERATOR_TRANSPORT_SPLIT_SCHEMA, label="split")
        expected = {
            "corpus_sha256",
            "train_group_sha256s",
            "calibration_group_sha256s",
            "holdout_group_sha256s",
            "train_observation_sha256s",
            "calibration_observation_sha256s",
            "holdout_observation_sha256s",
        }
        if set(body) != expected or any(
            not isinstance(body.get(field), list) for field in expected - {"corpus_sha256"}
        ):
            raise OperatorTransportIntegrityError("invalid transport split body")
        try:
            result = cls(
                corpus_sha256=cast(str, body.get("corpus_sha256")),
                train_group_sha256s=tuple(
                    cast(list[str], body.get("train_group_sha256s"))
                ),
                calibration_group_sha256s=tuple(
                    cast(list[str], body.get("calibration_group_sha256s"))
                ),
                holdout_group_sha256s=tuple(
                    cast(list[str], body.get("holdout_group_sha256s"))
                ),
                train_observation_sha256s=tuple(
                    cast(list[str], body.get("train_observation_sha256s"))
                ),
                calibration_observation_sha256s=tuple(
                    cast(list[str], body.get("calibration_observation_sha256s"))
                ),
                holdout_observation_sha256s=tuple(
                    cast(list[str], body.get("holdout_observation_sha256s"))
                ),
            )
        except (TypeError, ValueError) as exc:
            raise OperatorTransportIntegrityError("split failed validation") from exc
        if result.to_bytes() != data:
            raise OperatorTransportIntegrityError("split replay changed")
        return result


def chronological_operator_transport_split(
    corpus: OperatorTransportCorpus,
    *,
    train_fraction: Fraction = Fraction(3, 5),
    calibration_fraction: Fraction = Fraction(1, 5),
) -> OperatorTransportSplit:
    if not isinstance(corpus, OperatorTransportCorpus):
        raise TypeError("corpus must be an OperatorTransportCorpus")
    train_ratio = _fraction(train_fraction, field="train_fraction")
    calibration_ratio = _fraction(
        calibration_fraction, field="calibration_fraction"
    )
    if train_ratio + calibration_ratio >= 1:
        raise OperatorTransportError("split fractions must leave a holdout")
    group_rows: dict[str, list[OperatorTransportObservation]] = defaultdict(list)
    for row in corpus.observations:
        group_rows[row.split_group_sha256].append(row)
    groups = sorted(
        group_rows,
        key=lambda group: (group_rows[group][0].temporal_index, group),
    )
    if len(groups) < 3:
        raise OperatorTransportLeakageError(
            "at least three prompt/position groups are required"
        )
    count = len(groups)
    train_count = max(
        1,
        min(count - 2, count * train_ratio.numerator // train_ratio.denominator),
    )
    calibration_count = max(
        1,
        min(
            count - train_count - 1,
            count * calibration_ratio.numerator // calibration_ratio.denominator,
        ),
    )
    train = tuple(groups[:train_count])
    calibration = tuple(groups[train_count : train_count + calibration_count])
    holdout = tuple(groups[train_count + calibration_count :])
    partitions = (set(train), set(calibration), set(holdout))

    def inventory(groups_set: set[str]) -> tuple[str, ...]:
        return tuple(
            row.sha256
            for row in corpus.observations
            if row.split_group_sha256 in groups_set
        )

    result = OperatorTransportSplit(
        corpus_sha256=corpus.sha256,
        train_group_sha256s=train,
        calibration_group_sha256s=calibration,
        holdout_group_sha256s=holdout,
        train_observation_sha256s=inventory(partitions[0]),
        calibration_observation_sha256s=inventory(partitions[1]),
        holdout_observation_sha256s=inventory(partitions[2]),
    )
    all_hashes = set().union(
        set(result.train_observation_sha256s),
        set(result.calibration_observation_sha256s),
        set(result.holdout_observation_sha256s),
    )
    if all_hashes != set(corpus.observation_sha256s):
        raise OperatorTransportIntegrityError("split does not exactly cover corpus")
    times = tuple(
        tuple(
            row.temporal_index
            for row in corpus.observations
            if row.split_group_sha256 in groups_set
        )
        for groups_set in partitions
    )
    if not (max(times[0]) < min(times[1]) and max(times[1]) < min(times[2])):
        raise OperatorTransportLeakageError("transport split is not strictly temporal")
    return result


@dataclass(frozen=True, slots=True)
class OperatorTransportConfig:
    additive_residual: bool = False
    maximum_system_condition: float = 1.0e12
    maximum_transport_condition: float = 1.0e8
    minimum_nullspace_gap: float = 1.001
    rank_tolerance_multiplier: float = 64.0
    random_seed_sha256: str = hashlib.sha256(
        b"immer-ooe-operator-transport-random/v1"
    ).hexdigest()

    def __post_init__(self) -> None:
        if not isinstance(self.additive_residual, bool):
            raise TypeError("additive_residual must be bool")
        for field in (
            "maximum_system_condition",
            "maximum_transport_condition",
            "minimum_nullspace_gap",
            "rank_tolerance_multiplier",
        ):
            object.__setattr__(
                self,
                field,
                _finite(getattr(self, field), field=field, positive=True),
            )
        if self.minimum_nullspace_gap < 1.0:
            raise OperatorTransportError("minimum_nullspace_gap must be at least one")
        object.__setattr__(
            self,
            "random_seed_sha256",
            require_sha256(self.random_seed_sha256, field="random_seed_sha256"),
        )

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": OPERATOR_TRANSPORT_CONFIG_SCHEMA,
            "additive_residual": self.additive_residual,
            "maximum_system_condition": self.maximum_system_condition,
            "maximum_transport_condition": self.maximum_transport_condition,
            "minimum_nullspace_gap": self.minimum_nullspace_gap,
            "rank_tolerance_multiplier": self.rank_tolerance_multiplier,
            "random_seed_sha256": self.random_seed_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "OperatorTransportConfig":
        expected = {
            "schema",
            "additive_residual",
            "maximum_system_condition",
            "maximum_transport_condition",
            "minimum_nullspace_gap",
            "rank_tolerance_multiplier",
            "random_seed_sha256",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or value.get("schema") != OPERATOR_TRANSPORT_CONFIG_SCHEMA
        ):
            raise OperatorTransportIntegrityError("invalid transport config")
        try:
            return cls(
                additive_residual=cast(bool, value.get("additive_residual")),
                maximum_system_condition=cast(
                    float, value.get("maximum_system_condition")
                ),
                maximum_transport_condition=cast(
                    float, value.get("maximum_transport_condition")
                ),
                minimum_nullspace_gap=cast(
                    float, value.get("minimum_nullspace_gap")
                ),
                rank_tolerance_multiplier=cast(
                    float, value.get("rank_tolerance_multiplier")
                ),
                random_seed_sha256=cast(str, value.get("random_seed_sha256")),
            )
        except (TypeError, ValueError) as exc:
            raise OperatorTransportIntegrityError(
                "transport config failed validation"
            ) from exc


@dataclass(frozen=True, slots=True)
class OperatorTransportKernel:
    name: str
    source_head: int | None
    target_head: int | None
    coefficient: NDArray[np.float64]
    additive_residual: NDArray[np.float64]
    system_smallest_singular: float
    nullspace_gap: float
    system_condition: float
    transport_condition: float
    training_observation_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        name = _text(self.name, field="kernel.name", maximum=32)
        if name not in {"per_head", "global", "identity", "random"}:
            raise OperatorTransportError("unknown transport kernel name")
        if (self.source_head is None) != (self.target_head is None):
            raise OperatorTransportError("kernel head pair is incomplete")
        source_head = self.source_head
        target_head = self.target_head
        if source_head is not None and target_head is not None:
            source_head = _uint(source_head, field="source_head")
            target_head = _uint(target_head, field="target_head")
        if name == "per_head" and source_head is None:
            raise OperatorTransportError("per-head kernel requires a head pair")
        if name != "per_head" and source_head is not None:
            raise OperatorTransportError("non-per-head kernel cannot carry heads")
        coefficient = _matrix(self.coefficient, field="coefficient")
        residual = _matrix(self.additive_residual, field="additive_residual")
        if coefficient.shape != residual.shape:
            raise OperatorTransportError("kernel coefficient/residual shapes differ")
        for field in (
            "system_smallest_singular",
            "nullspace_gap",
            "system_condition",
            "transport_condition",
        ):
            object.__setattr__(
                self,
                field,
                _finite(getattr(self, field), field=field, nonnegative=True),
            )
        hashes = tuple(
            require_sha256(value, field="training_observation_sha256s")
            for value in self.training_observation_sha256s
        )
        if not hashes or len(set(hashes)) != len(hashes):
            raise OperatorTransportIntegrityError("kernel training inventory is invalid")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "source_head", source_head)
        object.__setattr__(self, "target_head", target_head)
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "additive_residual", residual)
        object.__setattr__(self, "training_observation_sha256s", hashes)

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return _seal(
            OPERATOR_TRANSPORT_KERNEL_SCHEMA,
            {
                "name": self.name,
                "source_head": self.source_head,
                "target_head": self.target_head,
                "coefficient": _matrix_record(self.coefficient),
                "additive_residual": _matrix_record(self.additive_residual),
                "system_smallest_singular": self.system_smallest_singular,
                "nullspace_gap": self.nullspace_gap,
                "system_condition": self.system_condition,
                "transport_condition": self.transport_condition,
                "training_observation_sha256s": list(
                    self.training_observation_sha256s
                ),
            },
        )

    @classmethod
    def from_dict(cls, value: object) -> "OperatorTransportKernel":
        body = _unseal(
            value, schema=OPERATOR_TRANSPORT_KERNEL_SCHEMA, label="kernel"
        )
        expected = {
            "name",
            "source_head",
            "target_head",
            "coefficient",
            "additive_residual",
            "system_smallest_singular",
            "nullspace_gap",
            "system_condition",
            "transport_condition",
            "training_observation_sha256s",
        }
        if set(body) != expected or not isinstance(
            body.get("training_observation_sha256s"), list
        ):
            raise OperatorTransportIntegrityError("invalid kernel body")
        try:
            return cls(
                name=cast(str, body.get("name")),
                source_head=cast(int | None, body.get("source_head")),
                target_head=cast(int | None, body.get("target_head")),
                coefficient=_matrix_from_record(
                    body.get("coefficient"), field="coefficient"
                ),
                additive_residual=_matrix_from_record(
                    body.get("additive_residual"), field="additive_residual"
                ),
                system_smallest_singular=cast(
                    float, body.get("system_smallest_singular")
                ),
                nullspace_gap=cast(float, body.get("nullspace_gap")),
                system_condition=cast(float, body.get("system_condition")),
                transport_condition=cast(float, body.get("transport_condition")),
                training_observation_sha256s=tuple(
                    cast(list[str], body.get("training_observation_sha256s"))
                ),
            )
        except (TypeError, ValueError) as exc:
            raise OperatorTransportIntegrityError("kernel failed validation") from exc


@dataclass(frozen=True, slots=True)
class OperatorTransportMetrics:
    model_name: str
    observation_count: int
    group_count: int
    mean_relative_residual: float
    root_mean_square_relative_residual: float
    maximum_relative_residual: float
    evidence_sha256: str

    def __post_init__(self) -> None:
        name = _text(self.model_name, field="model_name", maximum=32)
        if name not in {"per_head", "global", "identity", "random"}:
            raise OperatorTransportError("unknown metrics model")
        observations = _uint(
            self.observation_count, field="observation_count", positive=True
        )
        groups = _uint(self.group_count, field="group_count", positive=True)
        if groups > observations:
            raise OperatorTransportError("metric groups exceed observations")
        for field in (
            "mean_relative_residual",
            "root_mean_square_relative_residual",
            "maximum_relative_residual",
        ):
            object.__setattr__(
                self,
                field,
                _finite(getattr(self, field), field=field, nonnegative=True),
            )
        object.__setattr__(
            self,
            "evidence_sha256",
            require_sha256(self.evidence_sha256, field="evidence_sha256"),
        )
        object.__setattr__(self, "model_name", name)
        object.__setattr__(self, "observation_count", observations)
        object.__setattr__(self, "group_count", groups)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": OPERATOR_TRANSPORT_METRICS_SCHEMA,
            "model_name": self.model_name,
            "observation_count": self.observation_count,
            "group_count": self.group_count,
            "mean_relative_residual": self.mean_relative_residual,
            "root_mean_square_relative_residual": (
                self.root_mean_square_relative_residual
            ),
            "maximum_relative_residual": self.maximum_relative_residual,
            "evidence_sha256": self.evidence_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "OperatorTransportMetrics":
        expected = {
            "schema",
            "model_name",
            "observation_count",
            "group_count",
            "mean_relative_residual",
            "root_mean_square_relative_residual",
            "maximum_relative_residual",
            "evidence_sha256",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or value.get("schema") != OPERATOR_TRANSPORT_METRICS_SCHEMA
        ):
            raise OperatorTransportIntegrityError("invalid transport metrics")
        try:
            return cls(
                model_name=cast(str, value.get("model_name")),
                observation_count=cast(int, value.get("observation_count")),
                group_count=cast(int, value.get("group_count")),
                mean_relative_residual=cast(
                    float, value.get("mean_relative_residual")
                ),
                root_mean_square_relative_residual=cast(
                    float, value.get("root_mean_square_relative_residual")
                ),
                maximum_relative_residual=cast(
                    float, value.get("maximum_relative_residual")
                ),
                evidence_sha256=cast(str, value.get("evidence_sha256")),
            )
        except (TypeError, ValueError) as exc:
            raise OperatorTransportIntegrityError("metrics failed validation") from exc


def _partition_observations(
    corpus: OperatorTransportCorpus,
    split: OperatorTransportSplit,
) -> tuple[
    tuple[OperatorTransportObservation, ...],
    tuple[OperatorTransportObservation, ...],
    tuple[OperatorTransportObservation, ...],
]:
    if split.corpus_sha256 != corpus.sha256:
        raise OperatorTransportIntegrityError("split names another corpus")
    by_sha = {row.sha256: row for row in corpus.observations}
    result: list[tuple[OperatorTransportObservation, ...]] = []
    group_inventories = (
        split.train_group_sha256s,
        split.calibration_group_sha256s,
        split.holdout_group_sha256s,
    )
    observation_inventories = (
        split.train_observation_sha256s,
        split.calibration_observation_sha256s,
        split.holdout_observation_sha256s,
    )
    for groups, observations in zip(
        group_inventories, observation_inventories, strict=True
    ):
        try:
            rows = tuple(by_sha[address] for address in observations)
        except KeyError as exc:
            raise OperatorTransportIntegrityError(
                "split observation is absent from corpus"
            ) from exc
        if tuple(dict.fromkeys(row.split_group_sha256 for row in rows)) != groups:
            raise OperatorTransportIntegrityError("split group inventory changed")
        result.append(rows)
    if set().union(*(set(values) for values in observation_inventories)) != set(
        corpus.observation_sha256s
    ):
        raise OperatorTransportIntegrityError("split does not cover corpus exactly")
    return cast(
        tuple[
            tuple[OperatorTransportObservation, ...],
            tuple[OperatorTransportObservation, ...],
            tuple[OperatorTransportObservation, ...],
        ],
        tuple(result),
    )


def _design_matrix(
    observations: Sequence[OperatorTransportObservation],
) -> NDArray[np.float64]:
    dimension = observations[0].source_operator.shape[0]
    identity = np.eye(dimension, dtype=np.float64)
    rows = [
        np.kron(row.source_operator.T, identity)
        - np.kron(identity, row.target_operator)
        for row in observations
    ]
    return np.asarray(np.vstack(rows), dtype=np.float64)


def _residual_mean(
    coefficient: NDArray[np.float64],
    observations: Sequence[OperatorTransportObservation],
) -> NDArray[np.float64]:
    result = np.mean(
        np.stack(
            [
                row.target_operator @ coefficient
                - coefficient @ row.source_operator
                for row in observations
            ]
        ),
        axis=0,
    )
    return np.asarray(result, dtype=np.float64)


def _fit_intertwiner(
    observations: Sequence[OperatorTransportObservation],
    *,
    config: OperatorTransportConfig,
    name: str,
    source_head: int | None,
    target_head: int | None,
) -> OperatorTransportKernel:
    if not observations:
        raise OperatorTransportFitError("transport fit has no observations")
    design = _design_matrix(observations)
    dimension = observations[0].source_operator.shape[0]
    # Prefix-Sinkhorn matrices all share the causal absorbing first row, so an
    # unconstrained homogeneous Sylvester system has a structural extra
    # nullspace.  The transport itself is therefore parameterized as a causal
    # row-mass-preserving map: C starts at I; every free lower entry adds mass
    # there and debits the diagonal in the same row.  This removes the zero and
    # scale solutions without injecting labels or Softmax operators.
    identity_coefficient = np.eye(dimension, dtype=np.float64)
    base = identity_coefficient.reshape(-1, order="F")
    parameter_columns: list[NDArray[np.float64]] = []
    for row in range(1, dimension):
        for column in range(row):
            # All causal row-stochastic attention operators share the absorbing
            # first-token projector, leaving one exact transport gauge.  Fix
            # C[1,0]=0 canonically; the remaining representative has identical
            # intertwining action and becomes fully identifiable.
            if row == 1 and column == 0:
                continue
            vector = np.zeros(dimension * dimension, dtype=np.float64)
            vector[row + column * dimension] = 1.0
            vector[row + row * dimension] = -1.0
            parameter_columns.append(vector)
    parameterization = np.column_stack(parameter_columns)
    # NumPy/Accelerate's tiny GEMV path can surface stale floating-point flags
    # left by Torch's log-domain Prefix-Sinkhorn kernel as spurious overflow or
    # divide warnings despite finite inputs and outputs.  Explicit einsum keeps
    # the warning-fatal diagnostic deterministic.
    constrained_design = np.einsum(
        "ij,jk->ik", design, parameterization, optimize=False
    )
    if config.additive_residual:
        residual_design = np.tile(parameterization, (len(observations), 1))
        fitted_design = np.column_stack(
            (constrained_design, residual_design)
        )
    else:
        fitted_design = constrained_design
    target = -np.einsum("ij,j->i", design, base, optimize=False)
    columns = fitted_design.shape[1]
    if fitted_design.shape[0] < columns:
        raise OperatorTransportFitError(
            "transport system has fewer equations than identifiable coefficients"
        )
    singular = np.linalg.svd(fitted_design, compute_uv=False)
    if singular.size != columns:
        raise OperatorTransportFitError("transport parameterization is incomplete")
    maximum = float(singular[0])
    tolerance = (
        np.finfo(np.float64).eps
        * max(fitted_design.shape)
        * max(1.0, maximum)
        * config.rank_tolerance_multiplier
    )
    smallest = float(singular[-1])
    if smallest <= tolerance:
        raise OperatorTransportFitError(
            "transport system is rank deficient in its causal mass-preserving class"
        )
    system_condition = maximum / smallest
    if system_condition > config.maximum_system_condition:
        raise OperatorTransportFitError("transport system is ill-conditioned")
    gap = smallest / tolerance
    if gap < config.minimum_nullspace_gap:
        raise OperatorTransportFitError("transport identification is not separated")
    parameters, _residuals, fitted_rank, _singular = np.linalg.lstsq(
        fitted_design, target, rcond=None
    )
    if fitted_rank != columns:
        raise OperatorTransportFitError("transport least-squares rank changed")
    coefficient_parameter_count = parameterization.shape[1]
    coefficient = (
        base + parameterization @ parameters[:coefficient_parameter_count]
    ).reshape(
        (dimension, dimension), order="F"
    )
    if (
        float(np.max(np.abs(np.triu(coefficient, 1)))) > 2.0**-40
        or float(np.max(np.abs(coefficient.sum(axis=1) - 1.0))) > 2.0**-40
    ):
        raise OperatorTransportFitError(
            "transport coefficient left its causal mass-preserving class"
        )
    transport_condition = float(np.linalg.cond(coefficient))
    if (
        not math.isfinite(transport_condition)
        or transport_condition > config.maximum_transport_condition
    ):
        raise OperatorTransportFitError("transport coefficient is singular or ill-conditioned")
    if config.additive_residual:
        residual = (
            parameterization
            @ parameters[coefficient_parameter_count:]
        ).reshape((dimension, dimension), order="F")
    else:
        residual = np.zeros_like(coefficient)
    return OperatorTransportKernel(
        name=name,
        source_head=source_head,
        target_head=target_head,
        coefficient=np.asarray(coefficient, dtype=np.float64),
        additive_residual=np.asarray(residual, dtype=np.float64),
        system_smallest_singular=smallest,
        nullspace_gap=gap,
        system_condition=system_condition,
        transport_condition=transport_condition,
        training_observation_sha256s=tuple(row.sha256 for row in observations),
    )


def _baseline_kernel(
    observations: Sequence[OperatorTransportObservation],
    *,
    config: OperatorTransportConfig,
    name: Literal["identity", "random"],
) -> OperatorTransportKernel:
    dimension = observations[0].source_operator.shape[0]
    if name == "identity":
        coefficient = np.eye(dimension, dtype=np.float64)
    else:
        seed = int(config.random_seed_sha256[:16], 16)
        generator = np.random.default_rng(seed)
        coefficient = np.zeros((dimension, dimension), dtype=np.float64)
        for row in range(dimension):
            coefficient[row, : row + 1] = generator.dirichlet(
                np.ones(row + 1, dtype=np.float64)
            )
    residual = (
        _residual_mean(coefficient, observations)
        if config.additive_residual
        else np.zeros_like(coefficient)
    )
    return OperatorTransportKernel(
        name=name,
        source_head=None,
        target_head=None,
        coefficient=np.asarray(coefficient, dtype=np.float64),
        additive_residual=np.asarray(residual, dtype=np.float64),
        system_smallest_singular=0.0,
        nullspace_gap=0.0,
        system_condition=1.0,
        transport_condition=float(np.linalg.cond(coefficient)),
        training_observation_sha256s=tuple(row.sha256 for row in observations),
    )


def _relative_residual(
    row: OperatorTransportObservation,
    kernel: OperatorTransportKernel,
) -> float:
    left = kernel.coefficient @ row.source_operator + kernel.additive_residual
    right = row.target_operator @ kernel.coefficient
    numerator = float(np.linalg.norm(left - right, ord="fro"))
    denominator = max(
        np.finfo(np.float64).tiny,
        float(np.linalg.norm(left, ord="fro"))
        + float(np.linalg.norm(right, ord="fro")),
    )
    return numerator / denominator


def _evaluate(
    model_name: str,
    observations: Sequence[OperatorTransportObservation],
    *,
    per_head: Mapping[tuple[int, int], OperatorTransportKernel],
    global_kernel: OperatorTransportKernel,
    identity_kernel: OperatorTransportKernel,
    random_kernel: OperatorTransportKernel,
) -> OperatorTransportMetrics:
    kernels = {
        "global": global_kernel,
        "identity": identity_kernel,
        "random": random_kernel,
    }
    by_group: dict[str, list[float]] = defaultdict(list)
    evidence: list[dict[str, object]] = []
    for row in observations:
        if model_name == "per_head":
            try:
                kernel = per_head[(row.source_head, row.target_head)]
            except KeyError as exc:
                raise OperatorTransportIntegrityError(
                    "evaluation lacks a fitted head-pair kernel"
                ) from exc
        else:
            kernel = kernels[model_name]
        value = _relative_residual(row, kernel)
        by_group[row.split_group_sha256].append(value)
        evidence.append(
            {
                "observation_sha256": row.sha256,
                "kernel_sha256": kernel.sha256,
                "relative_residual": value,
            }
        )
    group_values = tuple(
        sum(values) / len(values) for _, values in sorted(by_group.items())
    )
    mean = sum(group_values) / len(group_values)
    rms = math.sqrt(sum(value * value for value in group_values) / len(group_values))
    return OperatorTransportMetrics(
        model_name=model_name,
        observation_count=len(observations),
        group_count=len(group_values),
        mean_relative_residual=mean,
        root_mean_square_relative_residual=rms,
        maximum_relative_residual=max(group_values),
        evidence_sha256=_digest(
            {
                "schema": "immer-ooe-operator-transport-metric-evidence/v1",
                "model_name": model_name,
                "rows": evidence,
            }
        ),
    )


def _metrics_inventory(
    observations: Sequence[OperatorTransportObservation],
    *,
    per_head: Mapping[tuple[int, int], OperatorTransportKernel],
    global_kernel: OperatorTransportKernel,
    identity_kernel: OperatorTransportKernel,
    random_kernel: OperatorTransportKernel,
) -> tuple[OperatorTransportMetrics, ...]:
    return tuple(
        _evaluate(
            name,
            observations,
            per_head=per_head,
            global_kernel=global_kernel,
            identity_kernel=identity_kernel,
            random_kernel=random_kernel,
        )
        for name in ("per_head", "global", "identity", "random")
    )


@dataclass(frozen=True, slots=True)
class OperatorTransportFitReceipt:
    corpus: OperatorTransportCorpus
    split: OperatorTransportSplit
    config: OperatorTransportConfig
    per_head_kernels: tuple[OperatorTransportKernel, ...]
    global_kernel: OperatorTransportKernel
    identity_kernel: OperatorTransportKernel
    random_kernel: OperatorTransportKernel
    train_metrics: tuple[OperatorTransportMetrics, ...]
    calibration_metrics: tuple[OperatorTransportMetrics, ...]
    best_calibration_model: str

    def __post_init__(self) -> None:
        if not isinstance(self.corpus, OperatorTransportCorpus):
            raise TypeError("corpus must be OperatorTransportCorpus")
        if not isinstance(self.split, OperatorTransportSplit):
            raise TypeError("split must be OperatorTransportSplit")
        if not isinstance(self.config, OperatorTransportConfig):
            raise TypeError("config must be OperatorTransportConfig")
        _partition_observations(self.corpus, self.split)
        per_head = tuple(
            sorted(
                self.per_head_kernels,
                key=lambda row: (cast(int, row.source_head), cast(int, row.target_head)),
            )
        )
        if not per_head or any(row.name != "per_head" for row in per_head):
            raise OperatorTransportIntegrityError("per-head kernel inventory is invalid")
        pairs = tuple((row.source_head, row.target_head) for row in per_head)
        if len(set(pairs)) != len(pairs):
            raise OperatorTransportIntegrityError("per-head kernel pair is duplicated")
        for kernel, name in (
            (self.global_kernel, "global"),
            (self.identity_kernel, "identity"),
            (self.random_kernel, "random"),
        ):
            if not isinstance(kernel, OperatorTransportKernel) or kernel.name != name:
                raise OperatorTransportIntegrityError(f"{name} kernel role changed")
        train_metrics = tuple(self.train_metrics)
        calibration_metrics = tuple(self.calibration_metrics)
        expected_names = ("per_head", "global", "identity", "random")
        if (
            tuple(row.model_name for row in train_metrics) != expected_names
            or tuple(row.model_name for row in calibration_metrics) != expected_names
        ):
            raise OperatorTransportIntegrityError("metric model inventory changed")
        best = min(
            calibration_metrics,
            key=lambda row: (row.mean_relative_residual, row.model_name),
        ).model_name
        if self.best_calibration_model != best:
            raise OperatorTransportIntegrityError("best calibration model changed")
        object.__setattr__(self, "per_head_kernels", per_head)
        object.__setattr__(self, "train_metrics", train_metrics)
        object.__setattr__(self, "calibration_metrics", calibration_metrics)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return _seal(
            OPERATOR_TRANSPORT_FIT_SCHEMA,
            {
                "corpus": self.corpus.to_dict(),
                "corpus_sha256": self.corpus.sha256,
                "split": self.split.to_dict(),
                "split_sha256": self.split.sha256,
                "config": self.config.to_dict(),
                "config_sha256": self.config.sha256,
                "model_pin_sha256": self.corpus.model_pin_sha256,
                "graph_revision": self.corpus.graph_revision.to_document(),
                "attention_mode": self.corpus.attention_mode,
                "source_evidence_sha256s": list(self.corpus.evidence_sha256s),
                "per_head_kernels": [row.to_dict() for row in self.per_head_kernels],
                "global_kernel": self.global_kernel.to_dict(),
                "identity_kernel": self.identity_kernel.to_dict(),
                "random_kernel": self.random_kernel.to_dict(),
                "train_metrics": [row.to_dict() for row in self.train_metrics],
                "calibration_metrics": [
                    row.to_dict() for row in self.calibration_metrics
                ],
                "best_calibration_model": self.best_calibration_model,
            },
        )

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_dict())
        if len(data) > MAX_OPERATOR_TRANSPORT_BYTES:
            raise OperatorTransportError("transport fit exceeds byte bound")
        return data

    def verify_or_raise(self) -> bool:
        expected = fit_operator_transport(self.corpus, self.split, config=self.config)
        if expected.to_dict() != self.to_dict():
            raise OperatorTransportIntegrityError(
                "operator transport fit does not recompute exactly"
            )
        return True

    @classmethod
    def from_bytes(cls, data: bytes) -> "OperatorTransportFitReceipt":
        value = _strict_json(data, label="operator transport fit")
        body = _unseal(value, schema=OPERATOR_TRANSPORT_FIT_SCHEMA, label="fit")
        expected = {
            "corpus",
            "corpus_sha256",
            "split",
            "split_sha256",
            "config",
            "config_sha256",
            "model_pin_sha256",
            "graph_revision",
            "attention_mode",
            "source_evidence_sha256s",
            "per_head_kernels",
            "global_kernel",
            "identity_kernel",
            "random_kernel",
            "train_metrics",
            "calibration_metrics",
            "best_calibration_model",
        }
        list_fields = {
            "source_evidence_sha256s",
            "per_head_kernels",
            "train_metrics",
            "calibration_metrics",
        }
        if set(body) != expected or any(
            not isinstance(body.get(field), list) for field in list_fields
        ):
            raise OperatorTransportIntegrityError("invalid transport fit body")
        try:
            corpus = OperatorTransportCorpus.from_bytes(
                canonical_json_bytes(body.get("corpus"))
            )
            split = OperatorTransportSplit.from_bytes(
                canonical_json_bytes(body.get("split"))
            )
            config = OperatorTransportConfig.from_dict(body.get("config"))
            result = cls(
                corpus=corpus,
                split=split,
                config=config,
                per_head_kernels=tuple(
                    OperatorTransportKernel.from_dict(row)
                    for row in cast(list[object], body.get("per_head_kernels"))
                ),
                global_kernel=OperatorTransportKernel.from_dict(
                    body.get("global_kernel")
                ),
                identity_kernel=OperatorTransportKernel.from_dict(
                    body.get("identity_kernel")
                ),
                random_kernel=OperatorTransportKernel.from_dict(
                    body.get("random_kernel")
                ),
                train_metrics=tuple(
                    OperatorTransportMetrics.from_dict(row)
                    for row in cast(list[object], body.get("train_metrics"))
                ),
                calibration_metrics=tuple(
                    OperatorTransportMetrics.from_dict(row)
                    for row in cast(list[object], body.get("calibration_metrics"))
                ),
                best_calibration_model=cast(
                    str, body.get("best_calibration_model")
                ),
            )
            pins = {
                "corpus_sha256": corpus.sha256,
                "split_sha256": split.sha256,
                "config_sha256": config.sha256,
                "model_pin_sha256": corpus.model_pin_sha256,
                "graph_revision": corpus.graph_revision.to_document(),
                "attention_mode": corpus.attention_mode,
                "source_evidence_sha256s": list(corpus.evidence_sha256s),
            }
            if any(body.get(field) != actual for field, actual in pins.items()):
                raise OperatorTransportIntegrityError("transport fit pin changed")
        except OperatorTransportIntegrityError:
            raise
        except (TypeError, ValueError) as exc:
            raise OperatorTransportIntegrityError("transport fit failed validation") from exc
        if result.to_bytes() != data:
            raise OperatorTransportIntegrityError("transport fit replay changed")
        result.verify_or_raise()
        return result


def fit_operator_transport(
    corpus: OperatorTransportCorpus,
    split: OperatorTransportSplit,
    *,
    config: OperatorTransportConfig | None = None,
) -> OperatorTransportFitReceipt:
    if not isinstance(corpus, OperatorTransportCorpus):
        raise TypeError("corpus must be OperatorTransportCorpus")
    if not isinstance(split, OperatorTransportSplit):
        raise TypeError("split must be OperatorTransportSplit")
    active_config = config or OperatorTransportConfig()
    if not isinstance(active_config, OperatorTransportConfig):
        raise TypeError("config must be OperatorTransportConfig")
    train, calibration, _holdout = _partition_observations(corpus, split)
    by_pair: dict[
        tuple[int, int], list[OperatorTransportObservation]
    ] = defaultdict(list)
    for row in train:
        by_pair[(row.source_head, row.target_head)].append(row)
    pairs_in_all = {
        (row.source_head, row.target_head) for row in corpus.observations
    }
    if set(by_pair) != pairs_in_all:
        raise OperatorTransportLeakageError(
            "training split does not cover every held-out head pair"
        )
    per_head = tuple(
        _fit_intertwiner(
            rows,
            config=active_config,
            name="per_head",
            source_head=pair[0],
            target_head=pair[1],
        )
        for pair, rows in sorted(by_pair.items())
    )
    global_kernel = _fit_intertwiner(
        train,
        config=active_config,
        name="global",
        source_head=None,
        target_head=None,
    )
    identity_kernel = _baseline_kernel(
        train, config=active_config, name="identity"
    )
    random_kernel = _baseline_kernel(train, config=active_config, name="random")
    per_head_map = {
        (cast(int, row.source_head), cast(int, row.target_head)): row
        for row in per_head
    }
    train_metrics = _metrics_inventory(
        train,
        per_head=per_head_map,
        global_kernel=global_kernel,
        identity_kernel=identity_kernel,
        random_kernel=random_kernel,
    )
    calibration_metrics = _metrics_inventory(
        calibration,
        per_head=per_head_map,
        global_kernel=global_kernel,
        identity_kernel=identity_kernel,
        random_kernel=random_kernel,
    )
    best = min(
        calibration_metrics,
        key=lambda row: (row.mean_relative_residual, row.model_name),
    ).model_name
    return OperatorTransportFitReceipt(
        corpus=corpus,
        split=split,
        config=active_config,
        per_head_kernels=per_head,
        global_kernel=global_kernel,
        identity_kernel=identity_kernel,
        random_kernel=random_kernel,
        train_metrics=train_metrics,
        calibration_metrics=calibration_metrics,
        best_calibration_model=best,
    )


@dataclass(frozen=True, slots=True)
class OperatorTransportHoldoutReceipt:
    fit: OperatorTransportFitReceipt
    holdout_metrics: tuple[OperatorTransportMetrics, ...]
    best_holdout_model: str

    def __post_init__(self) -> None:
        if not isinstance(self.fit, OperatorTransportFitReceipt):
            raise TypeError("fit must be OperatorTransportFitReceipt")
        metrics = tuple(self.holdout_metrics)
        names = ("per_head", "global", "identity", "random")
        if tuple(row.model_name for row in metrics) != names:
            raise OperatorTransportIntegrityError("holdout metric inventory changed")
        best = min(
            metrics, key=lambda row: (row.mean_relative_residual, row.model_name)
        ).model_name
        if self.best_holdout_model != best:
            raise OperatorTransportIntegrityError("best holdout model changed")
        object.__setattr__(self, "holdout_metrics", metrics)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return _seal(
            OPERATOR_TRANSPORT_HOLDOUT_SCHEMA,
            {
                "fit": self.fit.to_dict(),
                "fit_sha256": self.fit.sha256,
                "holdout_observation_sha256s": list(
                    self.fit.split.holdout_observation_sha256s
                ),
                "holdout_metrics": [row.to_dict() for row in self.holdout_metrics],
                "best_holdout_model": self.best_holdout_model,
            },
        )

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_dict())
        if len(data) > MAX_OPERATOR_TRANSPORT_BYTES:
            raise OperatorTransportError("holdout receipt exceeds byte bound")
        return data

    def verify_or_raise(self) -> bool:
        self.fit.verify_or_raise()
        expected = evaluate_operator_transport_holdout(self.fit)
        if expected.to_dict() != self.to_dict():
            raise OperatorTransportIntegrityError(
                "transport holdout does not recompute exactly"
            )
        return True

    @classmethod
    def from_bytes(cls, data: bytes) -> "OperatorTransportHoldoutReceipt":
        value = _strict_json(data, label="operator transport holdout")
        body = _unseal(
            value, schema=OPERATOR_TRANSPORT_HOLDOUT_SCHEMA, label="holdout"
        )
        expected = {
            "fit",
            "fit_sha256",
            "holdout_observation_sha256s",
            "holdout_metrics",
            "best_holdout_model",
        }
        if (
            set(body) != expected
            or not isinstance(body.get("holdout_observation_sha256s"), list)
            or not isinstance(body.get("holdout_metrics"), list)
        ):
            raise OperatorTransportIntegrityError("invalid holdout body")
        try:
            fit = OperatorTransportFitReceipt.from_bytes(
                canonical_json_bytes(body.get("fit"))
            )
            result = cls(
                fit=fit,
                holdout_metrics=tuple(
                    OperatorTransportMetrics.from_dict(row)
                    for row in cast(list[object], body.get("holdout_metrics"))
                ),
                best_holdout_model=cast(str, body.get("best_holdout_model")),
            )
            if (
                body.get("fit_sha256") != fit.sha256
                or body.get("holdout_observation_sha256s")
                != list(fit.split.holdout_observation_sha256s)
            ):
                raise OperatorTransportIntegrityError("holdout pin changed")
        except OperatorTransportIntegrityError:
            raise
        except (TypeError, ValueError) as exc:
            raise OperatorTransportIntegrityError("holdout failed validation") from exc
        if result.to_bytes() != data:
            raise OperatorTransportIntegrityError("holdout replay changed")
        result.verify_or_raise()
        return result


def evaluate_operator_transport_holdout(
    fit: OperatorTransportFitReceipt,
) -> OperatorTransportHoldoutReceipt:
    if not isinstance(fit, OperatorTransportFitReceipt):
        raise TypeError("fit must be OperatorTransportFitReceipt")
    _train, _calibration, holdout = _partition_observations(fit.corpus, fit.split)
    per_head = {
        (cast(int, row.source_head), cast(int, row.target_head)): row
        for row in fit.per_head_kernels
    }
    metrics = _metrics_inventory(
        holdout,
        per_head=per_head,
        global_kernel=fit.global_kernel,
        identity_kernel=fit.identity_kernel,
        random_kernel=fit.random_kernel,
    )
    best = min(
        metrics, key=lambda row: (row.mean_relative_residual, row.model_name)
    ).model_name
    return OperatorTransportHoldoutReceipt(fit, metrics, best)


def _capture_operator_sha256(operator: NDArray[np.float64]) -> str:
    record = _matrix_record(_operator(operator, field="capture operator"))
    return cast(str, record["data_sha256"])


def _capture_key_sha256(
    measurement_sha256: str,
    capture_spec_sha256: str,
) -> str:
    return _digest(
        {
            "schema": "immer-ooe-prefix-sinkhorn-capture-key/v1",
            "measurement_sha256": require_sha256(
                measurement_sha256, field="measurement_sha256"
            ),
            "capture_spec_sha256": require_sha256(
                capture_spec_sha256, field="capture_spec_sha256"
            ),
        }
    )


@dataclass(frozen=True, slots=True)
class QwenPrefixSinkhornCaptureReceipt:
    """One bounded native Prefix-Sinkhorn operator sidecar."""

    measurement_sha256: str
    model_pin_sha256: str
    atlas_revision: GraphRevision
    probe_identity_sha256: str
    prompt_sha256: str
    intervention_sha256: str
    source_evidence_sha256: str
    capture_spec_sha256: str
    attention_spec_sha256: str
    attention_mode: str
    layer: int
    head_indices: tuple[int, ...]
    query_positions: tuple[int, ...]
    key_positions: tuple[int, ...]
    operator_sha256s: tuple[str, ...]
    operators: tuple[NDArray[np.float64], ...]

    def __post_init__(self) -> None:
        for field in (
            "measurement_sha256",
            "model_pin_sha256",
            "probe_identity_sha256",
            "prompt_sha256",
            "intervention_sha256",
            "source_evidence_sha256",
            "capture_spec_sha256",
            "attention_spec_sha256",
        ):
            object.__setattr__(
                self,
                field,
                require_sha256(getattr(self, field), field=field),
            )
        if not isinstance(self.atlas_revision, GraphRevision):
            raise TypeError("atlas_revision must be GraphRevision")
        if self.attention_mode != QWEN_PREFIX_SINKHORN_ATTENTION_MODE:
            raise OperatorTransportError("capture attention mode is not Prefix-Sinkhorn")
        if isinstance(self.layer, bool) or self.layer != 27:
            raise OperatorTransportError("native Prefix-Sinkhorn capture requires layer 27")
        try:
            heads = tuple(self.head_indices)
            queries = tuple(self.query_positions)
            keys = tuple(self.key_positions)
            claimed = tuple(self.operator_sha256s)
            raw_operators = tuple(self.operators)
        except TypeError as exc:
            raise TypeError("capture inventories must be sequences") from exc
        if heads != (2, 8, 14, 20):
            raise OperatorTransportError(
                "capture heads must be the native query heads (2, 8, 14, 20)"
            )
        dimension = len(queries)
        if not 3 <= dimension <= MAX_OPERATOR_DIMENSION:
            raise OperatorTransportError("capture position count lies outside [3, 32]")
        expected_positions = tuple(range(dimension))
        if queries != expected_positions or keys != expected_positions:
            raise OperatorTransportError(
                "capture query/key positions must be one complete absolute prefix"
            )
        if len(raw_operators) != len(heads) or len(claimed) != len(heads):
            raise OperatorTransportError("capture operator/head inventory differs")
        operators = tuple(
            _operator(value, field=f"operators[{index}]")
            for index, value in enumerate(raw_operators)
        )
        if any(operator.shape != (dimension, dimension) for operator in operators):
            raise OperatorTransportError("capture operator dimensions differ")
        hashes = tuple(
            require_sha256(value, field="operator_sha256s") for value in claimed
        )
        expected_hashes = tuple(_capture_operator_sha256(value) for value in operators)
        if hashes != expected_hashes:
            raise OperatorTransportIntegrityError(
                "capture operator hashes differ from their exact bytes"
            )
        object.__setattr__(self, "head_indices", heads)
        object.__setattr__(self, "query_positions", queries)
        object.__setattr__(self, "key_positions", keys)
        object.__setattr__(self, "operator_sha256s", hashes)
        object.__setattr__(self, "operators", operators)

    @classmethod
    def create(
        cls,
        measurement: MeasurementReceipt,
        *,
        atlas_revision: GraphRevision,
        capture_spec_sha256: str,
        attention_spec_sha256: str,
        operators: Sequence[NDArray[np.float64]],
        head_indices: Sequence[int] = (2, 8, 14, 20),
        layer: int = 27,
    ) -> "QwenPrefixSinkhornCaptureReceipt":
        if not isinstance(measurement, MeasurementReceipt):
            raise TypeError("measurement must be MeasurementReceipt")
        if measurement.intervention.mode != "native":
            raise OperatorTransportError("operator capture requires a native measurement")
        matrices = tuple(operators)
        if not matrices:
            raise OperatorTransportError("operator capture is empty")
        first = _operator(matrices[0], field="operators[0]")
        dimension = int(first.shape[0])
        positions = tuple(range(dimension))
        return cls(
            measurement_sha256=measurement.sha256,
            model_pin_sha256=measurement.model_pin.sha256,
            atlas_revision=atlas_revision,
            probe_identity_sha256=measurement.probe.sha256,
            prompt_sha256=measurement.probe.prompt_signature,
            intervention_sha256=measurement.intervention.sha256,
            source_evidence_sha256=measurement.evidence_sha256,
            capture_spec_sha256=capture_spec_sha256,
            attention_spec_sha256=attention_spec_sha256,
            attention_mode=QWEN_PREFIX_SINKHORN_ATTENTION_MODE,
            layer=layer,
            head_indices=tuple(head_indices),
            query_positions=positions,
            key_positions=positions,
            operator_sha256s=tuple(_capture_operator_sha256(row) for row in matrices),
            operators=matrices,
        )

    @property
    def capture_key_sha256(self) -> str:
        return _capture_key_sha256(
            self.measurement_sha256,
            self.capture_spec_sha256,
        )

    def to_dict(self) -> dict[str, object]:
        body = {
            "measurement_sha256": self.measurement_sha256,
            "model_pin_sha256": self.model_pin_sha256,
            "atlas_revision": self.atlas_revision.to_document(),
            "probe_identity_sha256": self.probe_identity_sha256,
            "prompt_sha256": self.prompt_sha256,
            "intervention_sha256": self.intervention_sha256,
            "source_evidence_sha256": self.source_evidence_sha256,
            "capture_spec_sha256": self.capture_spec_sha256,
            "capture_key_sha256": self.capture_key_sha256,
            "attention_spec_sha256": self.attention_spec_sha256,
            "attention_mode": self.attention_mode,
            "layer": self.layer,
            "head_indices": list(self.head_indices),
            "query_positions": list(self.query_positions),
            "key_positions": list(self.key_positions),
            "operator_sha256s": list(self.operator_sha256s),
            "operators": [_matrix_record(row) for row in self.operators],
        }
        return _seal(QWEN_PREFIX_SINKHORN_CAPTURE_SCHEMA, body)

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_dict())
        if len(data) > MAX_PREFIX_SINKHORN_CAPTURE_BYTES:
            raise OperatorTransportError("Prefix-Sinkhorn capture exceeds byte bound")
        return data

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def verify_against_measurement(
        self,
        measurement: MeasurementReceipt,
        *,
        atlas_revision: GraphRevision,
    ) -> bool:
        if not isinstance(measurement, MeasurementReceipt):
            raise TypeError("measurement must be MeasurementReceipt")
        if not isinstance(atlas_revision, GraphRevision):
            raise TypeError("atlas_revision must be GraphRevision")
        if (
            measurement.sha256 != self.measurement_sha256
            or measurement.model_pin.sha256 != self.model_pin_sha256
            or measurement.probe.sha256 != self.probe_identity_sha256
            or measurement.probe.prompt_signature != self.prompt_sha256
            or measurement.intervention.sha256 != self.intervention_sha256
            or measurement.intervention.mode != "native"
            or measurement.evidence_sha256 != self.source_evidence_sha256
            or atlas_revision != self.atlas_revision
        ):
            raise OperatorTransportIntegrityError(
                "capture differs from its Atlas measurement authority"
            )
        return True

    @classmethod
    def from_bytes(cls, data: bytes) -> "QwenPrefixSinkhornCaptureReceipt":
        if not isinstance(data, bytes) or len(data) > MAX_PREFIX_SINKHORN_CAPTURE_BYTES:
            raise OperatorTransportIntegrityError("invalid Prefix-Sinkhorn capture bytes")
        value = _strict_json(data, label="Prefix-Sinkhorn capture")
        body = _unseal(
            value,
            schema=QWEN_PREFIX_SINKHORN_CAPTURE_SCHEMA,
            label="Prefix-Sinkhorn capture",
        )
        expected = {
            "measurement_sha256",
            "model_pin_sha256",
            "atlas_revision",
            "probe_identity_sha256",
            "prompt_sha256",
            "intervention_sha256",
            "source_evidence_sha256",
            "capture_spec_sha256",
            "capture_key_sha256",
            "attention_spec_sha256",
            "attention_mode",
            "layer",
            "head_indices",
            "query_positions",
            "key_positions",
            "operator_sha256s",
            "operators",
        }
        if set(body) != expected:
            raise OperatorTransportIntegrityError("invalid Prefix-Sinkhorn capture body")
        try:
            operators_value = body.get("operators")
            if not isinstance(operators_value, list):
                raise TypeError("operators must be a list")
            result = cls(
                measurement_sha256=cast(str, body.get("measurement_sha256")),
                model_pin_sha256=cast(str, body.get("model_pin_sha256")),
                atlas_revision=GraphRevision.from_document(body.get("atlas_revision")),
                probe_identity_sha256=cast(str, body.get("probe_identity_sha256")),
                prompt_sha256=cast(str, body.get("prompt_sha256")),
                intervention_sha256=cast(str, body.get("intervention_sha256")),
                source_evidence_sha256=cast(str, body.get("source_evidence_sha256")),
                capture_spec_sha256=cast(str, body.get("capture_spec_sha256")),
                attention_spec_sha256=cast(str, body.get("attention_spec_sha256")),
                attention_mode=cast(str, body.get("attention_mode")),
                layer=cast(int, body.get("layer")),
                head_indices=tuple(cast(list[int], body.get("head_indices"))),
                query_positions=tuple(cast(list[int], body.get("query_positions"))),
                key_positions=tuple(cast(list[int], body.get("key_positions"))),
                operator_sha256s=tuple(
                    cast(list[str], body.get("operator_sha256s"))
                ),
                operators=tuple(
                    _matrix_from_record(row, field=f"operators[{index}]")
                    for index, row in enumerate(operators_value)
                ),
            )
        except OperatorTransportIntegrityError:
            raise
        except (TypeError, ValueError) as exc:
            raise OperatorTransportIntegrityError(
                "Prefix-Sinkhorn capture failed validation"
            ) from exc
        if body.get("capture_key_sha256") != result.capture_key_sha256:
            raise OperatorTransportIntegrityError("capture key changed")
        if result.to_bytes() != data:
            raise OperatorTransportIntegrityError("capture reconstruction changed")
        return result


@dataclass(frozen=True, slots=True)
class QwenPrefixSinkhornCaptureAudit:
    receipt_count: int
    receipt_sha256s: tuple[str, ...]
    orphan_state_filenames: tuple[str, ...]
    inventory_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.receipt_count, bool)
            or not isinstance(self.receipt_count, int)
            or self.receipt_count < 0
        ):
            raise ValueError("receipt_count must be non-negative")
        receipts = tuple(
            require_sha256(value, field="receipt_sha256s")
            for value in self.receipt_sha256s
        )
        if receipts != tuple(sorted(set(receipts))) or len(receipts) != self.receipt_count:
            raise ValueError("audit receipt inventory is invalid")
        orphans = tuple(self.orphan_state_filenames)
        if orphans != tuple(sorted(set(orphans))) or any(
            not isinstance(value, str) or not value.endswith(".state")
            for value in orphans
        ):
            raise ValueError("audit orphan inventory is invalid")
        object.__setattr__(self, "receipt_sha256s", receipts)
        object.__setattr__(self, "orphan_state_filenames", orphans)
        object.__setattr__(
            self,
            "inventory_sha256",
            require_sha256(self.inventory_sha256, field="inventory_sha256"),
        )


class QwenPrefixSinkhornCaptureBank:
    """Crash-safe, no-replace measurement/capture sidecar bank."""

    _INVENTORY_STATE = "qwen-prefix-sinkhorn-capture-inventory/v1"
    _CAPTURE_STATE_PREFIX = "qwen-prefix-sinkhorn-capture/v1:"
    _LOCK_NAME = ".prefix-sinkhorn-capture.lock"

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.store = CrystalStore(
            root,
            max_state_bytes=MAX_PREFIX_SINKHORN_CAPTURE_BYTES,
        )
        self.root = self.store.root

    @contextmanager
    def _locked(self) -> Iterator[None]:
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.root / self._LOCK_NAME, flags, 0o600)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise OperatorTransportIntegrityError("capture bank lock is not regular")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    @staticmethod
    def _capture_state_name(key_sha256: str) -> str:
        return QwenPrefixSinkhornCaptureBank._CAPTURE_STATE_PREFIX + require_sha256(
            key_sha256, field="capture_key_sha256"
        )

    @staticmethod
    def _state_filename(name: str) -> str:
        return hashlib.sha256(name.encode("utf-8")).hexdigest() + ".state"

    def _inventory_bytes(self, entries: Mapping[str, str]) -> bytes:
        normalized = {
            require_sha256(key, field="capture inventory key"): require_sha256(
                value, field="capture inventory receipt"
            )
            for key, value in entries.items()
        }
        body = {
            "entries": [
                {"capture_key_sha256": key, "receipt_sha256": normalized[key]}
                for key in sorted(normalized)
            ]
        }
        return canonical_json_bytes(
            _seal(QWEN_PREFIX_SINKHORN_CAPTURE_INVENTORY_SCHEMA, body)
        )

    def _read_inventory(self) -> tuple[dict[str, str], bytes | None]:
        try:
            data = self.store.restore_state(self._INVENTORY_STATE)
        except KeyError:
            return {}, None
        value = _strict_json(data, label="capture inventory")
        body = _unseal(
            value,
            schema=QWEN_PREFIX_SINKHORN_CAPTURE_INVENTORY_SCHEMA,
            label="capture inventory",
        )
        if set(body) != {"entries"} or not isinstance(body.get("entries"), list):
            raise OperatorTransportIntegrityError("invalid capture inventory")
        entries: dict[str, str] = {}
        for row in cast(list[object], body.get("entries")):
            if not isinstance(row, Mapping) or set(row) != {
                "capture_key_sha256",
                "receipt_sha256",
            }:
                raise OperatorTransportIntegrityError("invalid capture inventory row")
            key = require_sha256(
                row.get("capture_key_sha256"), field="capture_key_sha256"
            )
            receipt = require_sha256(row.get("receipt_sha256"), field="receipt_sha256")
            if key in entries:
                raise OperatorTransportIntegrityError("duplicate capture inventory key")
            entries[key] = receipt
        if data != self._inventory_bytes(entries):
            raise OperatorTransportIntegrityError("capture inventory is not canonical")
        return entries, data

    def publish(
        self,
        receipt: QwenPrefixSinkhornCaptureReceipt,
    ) -> QwenPrefixSinkhornCaptureReceipt:
        if not isinstance(receipt, QwenPrefixSinkhornCaptureReceipt):
            raise TypeError("receipt must be QwenPrefixSinkhornCaptureReceipt")
        data = receipt.to_bytes()
        key = receipt.capture_key_sha256
        state_name = self._capture_state_name(key)
        try:
            with self._locked():
                entries, inventory_data = self._read_inventory()
                try:
                    current = self.store.restore_state(state_name)
                except KeyError:
                    current = None
                if current is not None and current != data:
                    raise OperatorTransportCaptureConflictError(
                        "capture key is already bound to different bytes"
                    )
                if current is None:
                    self.store.publish_state(state_name, data)
                prior = entries.get(key)
                if prior is not None and prior != receipt.sha256:
                    raise OperatorTransportCaptureConflictError(
                        "capture inventory is already bound to a different receipt"
                    )
                entries[key] = receipt.sha256
                next_inventory = self._inventory_bytes(entries)
                if next_inventory != inventory_data:
                    self.store.publish_state(
                        self._INVENTORY_STATE,
                        next_inventory,
                        expected_sha256=(
                            None
                            if inventory_data is None
                            else hashlib.sha256(inventory_data).hexdigest()
                        ),
                    )
        except OperatorTransportError:
            raise
        except (CrystalStoreError, CrystalTamperError, OSError, ValueError) as exc:
            raise OperatorTransportIntegrityError("capture publication failed") from exc
        return receipt

    def restore(
        self,
        measurement_sha256: str,
        capture_spec_sha256: str,
    ) -> QwenPrefixSinkhornCaptureReceipt:
        key = _capture_key_sha256(measurement_sha256, capture_spec_sha256)
        try:
            with self._locked():
                entries, _inventory = self._read_inventory()
                expected = entries.get(key)
                if expected is None:
                    raise KeyError(f"unknown Prefix-Sinkhorn capture: {key}")
                data = self.store.restore_state(self._capture_state_name(key))
        except KeyError:
            raise
        except OperatorTransportError:
            raise
        except (CrystalStoreError, CrystalTamperError, OSError, ValueError) as exc:
            raise OperatorTransportIntegrityError("capture restore failed") from exc
        receipt = QwenPrefixSinkhornCaptureReceipt.from_bytes(data)
        if receipt.sha256 != expected or receipt.capture_key_sha256 != key:
            raise OperatorTransportIntegrityError("capture differs from inventory")
        return receipt

    def receipts(self) -> tuple[QwenPrefixSinkhornCaptureReceipt, ...]:
        """Restore the complete authenticated inventory in temporal order."""

        try:
            with self._locked():
                entries, _inventory = self._read_inventory()
                rows = []
                for key, expected in sorted(entries.items()):
                    data = self.store.restore_state(self._capture_state_name(key))
                    receipt = QwenPrefixSinkhornCaptureReceipt.from_bytes(data)
                    if receipt.capture_key_sha256 != key or receipt.sha256 != expected:
                        raise OperatorTransportIntegrityError(
                            "capture inventory differs during complete restore"
                        )
                    rows.append(receipt)
        except OperatorTransportError:
            raise
        except (CrystalStoreError, CrystalTamperError, OSError, ValueError) as exc:
            raise OperatorTransportIntegrityError(
                "complete capture restore failed"
            ) from exc
        return tuple(
            sorted(
                rows,
                key=lambda row: (
                    row.atlas_revision.sequence,
                    row.prompt_sha256,
                    row.capture_key_sha256,
                ),
            )
        )

    def audit(self) -> QwenPrefixSinkhornCaptureAudit:
        try:
            with self._locked():
                entries, inventory_data = self._read_inventory()
                if inventory_data is None:
                    inventory_data = self._inventory_bytes({})
                receipts: list[str] = []
                expected_files = {
                    self._state_filename(self._INVENTORY_STATE)
                } if entries else set()
                for key, expected in sorted(entries.items()):
                    state_name = self._capture_state_name(key)
                    data = self.store.restore_state(state_name)
                    receipt = QwenPrefixSinkhornCaptureReceipt.from_bytes(data)
                    if receipt.capture_key_sha256 != key or receipt.sha256 != expected:
                        raise OperatorTransportIntegrityError(
                            "capture audit found an inventory mismatch"
                        )
                    receipts.append(receipt.sha256)
                    expected_files.add(self._state_filename(state_name))
                state_root = self.root / "state"
                actual_files: set[str] = set()
                for path in state_root.iterdir():
                    metadata = path.lstat()
                    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(
                        metadata.st_mode
                    ):
                        raise OperatorTransportIntegrityError(
                            "capture state inventory contains a non-regular entry"
                        )
                    actual_files.add(path.name)
        except OperatorTransportError:
            raise
        except (CrystalStoreError, CrystalTamperError, OSError, ValueError) as exc:
            raise OperatorTransportIntegrityError("capture audit failed") from exc
        return QwenPrefixSinkhornCaptureAudit(
            receipt_count=len(receipts),
            receipt_sha256s=tuple(sorted(receipts)),
            orphan_state_filenames=tuple(sorted(actual_files - expected_files)),
            inventory_sha256=hashlib.sha256(inventory_data).hexdigest(),
        )


@dataclass(frozen=True, slots=True)
class QwenOperatorTransportAvailability:
    model_pin_sha256: str
    graph_revision: GraphRevision
    attention_mode: str
    measurement_sha256s: tuple[str, ...]
    available: bool
    missing_measurements: tuple[str, ...]
    required_runtime_seam: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_pin_sha256",
            require_sha256(self.model_pin_sha256, field="model_pin_sha256"),
        )
        if not isinstance(self.graph_revision, GraphRevision):
            raise TypeError("graph_revision must be GraphRevision")
        mode = _text(self.attention_mode, field="attention_mode", maximum=64)
        if mode != QWEN_PREFIX_SINKHORN_ATTENTION_MODE:
            raise OperatorTransportError("Qwen adapter requires Prefix-Sinkhorn mode")
        measurements = tuple(
            require_sha256(value, field="measurement_sha256s")
            for value in self.measurement_sha256s
        )
        if len(set(measurements)) != len(measurements):
            raise OperatorTransportIntegrityError("measurement inventory is duplicated")
        if not isinstance(self.available, bool):
            raise TypeError("available must be bool")
        missing = tuple(
            _text(value, field="missing_measurements", maximum=128)
            for value in self.missing_measurements
        )
        if self.available or missing != (_MISSING_QWEN_MEASUREMENT,):
            raise OperatorTransportIntegrityError(
                "current Qwen adapter must report the exact missing matrix"
            )
        seam = _text(self.required_runtime_seam, field="required_runtime_seam")
        if seam != _QWEN_OPERATOR_SEAM:
            raise OperatorTransportIntegrityError("Qwen runtime seam changed")
        object.__setattr__(self, "attention_mode", mode)
        object.__setattr__(self, "measurement_sha256s", measurements)
        object.__setattr__(self, "missing_measurements", missing)
        object.__setattr__(self, "required_runtime_seam", seam)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return _seal(
            QWEN_OPERATOR_TRANSPORT_AVAILABILITY_SCHEMA,
            {
                "model_pin_sha256": self.model_pin_sha256,
                "graph_revision": self.graph_revision.to_document(),
                "attention_mode": self.attention_mode,
                "measurement_sha256s": list(self.measurement_sha256s),
                "available": self.available,
                "missing_measurements": list(self.missing_measurements),
                "required_runtime_seam": self.required_runtime_seam,
            },
        )

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_bytes(cls, data: bytes) -> "QwenOperatorTransportAvailability":
        value = _strict_json(data, label="Qwen operator transport availability")
        body = _unseal(
            value,
            schema=QWEN_OPERATOR_TRANSPORT_AVAILABILITY_SCHEMA,
            label="Qwen availability",
        )
        expected = {
            "model_pin_sha256",
            "graph_revision",
            "attention_mode",
            "measurement_sha256s",
            "available",
            "missing_measurements",
            "required_runtime_seam",
        }
        if set(body) != expected or any(
            not isinstance(body.get(field), list)
            for field in ("measurement_sha256s", "missing_measurements")
        ):
            raise OperatorTransportIntegrityError("invalid Qwen availability body")
        try:
            result = cls(
                model_pin_sha256=cast(str, body.get("model_pin_sha256")),
                graph_revision=GraphRevision.from_document(
                    cast(Mapping[str, Any], body.get("graph_revision"))
                ),
                attention_mode=cast(str, body.get("attention_mode")),
                measurement_sha256s=tuple(
                    cast(list[str], body.get("measurement_sha256s"))
                ),
                available=cast(bool, body.get("available")),
                missing_measurements=tuple(
                    cast(list[str], body.get("missing_measurements"))
                ),
                required_runtime_seam=cast(
                    str, body.get("required_runtime_seam")
                ),
            )
        except (TypeError, ValueError) as exc:
            raise OperatorTransportIntegrityError(
                "Qwen availability failed validation"
            ) from exc
        if result.to_bytes() != data:
            raise OperatorTransportIntegrityError("Qwen availability replay changed")
        return result


def inspect_qwen_operator_transport_availability(
    measurements: Sequence[MeasurementReceipt],
    *,
    model_pin_sha256: str,
    graph_revision: GraphRevision,
) -> QwenOperatorTransportAvailability:
    if isinstance(measurements, (str, bytes)):
        raise TypeError("measurements must be a sequence")
    rows = tuple(measurements)
    if any(not isinstance(row, MeasurementReceipt) for row in rows):
        raise TypeError("measurements must contain MeasurementReceipt values")
    model_pin = require_sha256(model_pin_sha256, field="model_pin_sha256")
    if not isinstance(graph_revision, GraphRevision):
        raise TypeError("graph_revision must be GraphRevision")
    if any(row.model_pin.sha256 != model_pin for row in rows):
        raise OperatorTransportIntegrityError("Qwen measurements changed model pin")
    return QwenOperatorTransportAvailability(
        model_pin_sha256=model_pin,
        graph_revision=graph_revision,
        attention_mode=QWEN_PREFIX_SINKHORN_ATTENTION_MODE,
        measurement_sha256s=tuple(row.sha256 for row in rows),
        available=False,
        missing_measurements=(_MISSING_QWEN_MEASUREMENT,),
        required_runtime_seam=_QWEN_OPERATOR_SEAM,
    )


def qwen_operator_transport_corpus_from_atlas(
    measurements: Sequence[MeasurementReceipt],
    *,
    model_pin_sha256: str,
    graph_revision: GraphRevision,
) -> OperatorTransportCorpus:
    availability = inspect_qwen_operator_transport_availability(
        measurements,
        model_pin_sha256=model_pin_sha256,
        graph_revision=graph_revision,
    )
    raise OperatorTransportMissingMeasurementError(
        f"{availability.missing_measurements[0]} is absent; capture "
        f"{availability.required_runtime_seam} instead of reconstructing archive Softmax"
    )


def qwen_operator_transport_corpus_from_captures(
    captures: Sequence[QwenPrefixSinkhornCaptureReceipt],
    *,
    graph_revision: GraphRevision,
    head_pairs: Sequence[tuple[int, int]] = ((2, 8), (14, 20)),
) -> OperatorTransportCorpus:
    """Build real head-pair observations from authenticated native sidecars."""

    if isinstance(captures, (str, bytes)):
        raise TypeError("captures must be a sequence")
    rows = tuple(captures)
    if not rows or any(
        not isinstance(row, QwenPrefixSinkhornCaptureReceipt) for row in rows
    ):
        raise TypeError("captures must contain Prefix-Sinkhorn capture receipts")
    if not isinstance(graph_revision, GraphRevision):
        raise TypeError("graph_revision must be GraphRevision")
    try:
        pairs = tuple((int(source), int(target)) for source, target in head_pairs)
    except (TypeError, ValueError) as exc:
        raise TypeError("head_pairs must contain integer pairs") from exc
    if (
        not pairs
        or len(set(pairs)) != len(pairs)
        or any(source == target for source, target in pairs)
    ):
        raise OperatorTransportError("head_pairs must be unique non-identity pairs")
    ordered = tuple(
        sorted(
            rows,
            key=lambda row: (
                row.atlas_revision.sequence,
                row.prompt_sha256,
                row.capture_key_sha256,
            ),
        )
    )
    model_pin = ordered[0].model_pin_sha256
    dimension = len(ordered[0].query_positions)
    attention_spec = ordered[0].attention_spec_sha256
    prompts: set[str] = set()
    captures_seen: set[str] = set()
    observations: list[OperatorTransportObservation] = []
    for temporal_index, capture in enumerate(ordered, start=1):
        if (
            capture.model_pin_sha256 != model_pin
            or len(capture.query_positions) != dimension
            or capture.attention_spec_sha256 != attention_spec
            or capture.attention_mode != QWEN_PREFIX_SINKHORN_ATTENTION_MODE
        ):
            raise OperatorTransportIntegrityError(
                "capture corpus model, dimension, spec, or attention mode changed"
            )
        if capture.capture_key_sha256 in captures_seen:
            raise OperatorTransportIntegrityError("capture corpus contains a duplicate")
        if capture.prompt_sha256 in prompts:
            raise OperatorTransportLeakageError(
                "capture corpus repeats one prompt across temporal groups"
            )
        captures_seen.add(capture.capture_key_sha256)
        prompts.add(capture.prompt_sha256)
        by_head = dict(zip(capture.head_indices, capture.operators, strict=True))
        position_group = _digest(
            {
                "schema": "immer-ooe-prefix-sinkhorn-position-group/v1",
                "attention_spec_sha256": capture.attention_spec_sha256,
                "capture_receipt_sha256": capture.sha256,
                "capture_spec_sha256": capture.capture_spec_sha256,
                "query_positions": list(capture.query_positions),
                "key_positions": list(capture.key_positions),
            }
        )
        for source_head, target_head in pairs:
            try:
                source = by_head[source_head]
                target = by_head[target_head]
            except KeyError as exc:
                raise OperatorTransportIntegrityError(
                    "head pair is absent from a capture receipt"
                ) from exc
            evidence = operator_transport_evidence_sha256(
                temporal_index=temporal_index,
                model_pin_sha256=model_pin,
                graph_revision=graph_revision,
                attention_mode=QWEN_PREFIX_SINKHORN_ATTENTION_MODE,
                prompt_sha256=capture.prompt_sha256,
                position_group_sha256=position_group,
                source_head=source_head,
                target_head=target_head,
                source_measurement_sha256=capture.measurement_sha256,
                target_measurement_sha256=capture.measurement_sha256,
                source_operator=source,
                target_operator=target,
            )
            observations.append(
                OperatorTransportObservation(
                    temporal_index=temporal_index,
                    model_pin_sha256=model_pin,
                    graph_revision=graph_revision,
                    attention_mode=QWEN_PREFIX_SINKHORN_ATTENTION_MODE,
                    prompt_sha256=capture.prompt_sha256,
                    position_group_sha256=position_group,
                    source_head=source_head,
                    target_head=target_head,
                    source_measurement_sha256=capture.measurement_sha256,
                    target_measurement_sha256=capture.measurement_sha256,
                    evidence_sha256=evidence,
                    source_operator=source,
                    target_operator=target,
                )
            )
    normalized = tuple(
        sorted(
            observations,
            key=lambda row: (
                row.temporal_index,
                row.split_group_sha256,
                row.source_head,
                row.target_head,
                row.sha256,
            ),
        )
    )
    return OperatorTransportCorpus(
        model_pin_sha256=model_pin,
        graph_revision=graph_revision,
        attention_mode=QWEN_PREFIX_SINKHORN_ATTENTION_MODE,
        observations=normalized,
        observation_sha256s=tuple(row.sha256 for row in normalized),
        evidence_sha256s=tuple(row.evidence_sha256 for row in normalized),
    )


__all__ = [
    "MAX_OPERATOR_DIMENSION",
    "MAX_OPERATOR_OBSERVATIONS",
    "MAX_OPERATOR_TRANSPORT_BYTES",
    "MAX_PREFIX_SINKHORN_CAPTURE_BYTES",
    "OPERATOR_TRANSPORT_CORPUS_SCHEMA",
    "OPERATOR_TRANSPORT_FIT_SCHEMA",
    "OPERATOR_TRANSPORT_HOLDOUT_SCHEMA",
    "OPERATOR_TRANSPORT_OBSERVATION_SCHEMA",
    "OPERATOR_TRANSPORT_SPLIT_SCHEMA",
    "QWEN_PREFIX_SINKHORN_CAPTURE_INVENTORY_SCHEMA",
    "QWEN_PREFIX_SINKHORN_CAPTURE_SCHEMA",
    "OperatorTransportCaptureConflictError",
    "OperatorTransportConfig",
    "OperatorTransportCorpus",
    "OperatorTransportError",
    "OperatorTransportFitError",
    "OperatorTransportFitReceipt",
    "OperatorTransportHoldoutReceipt",
    "OperatorTransportIntegrityError",
    "OperatorTransportKernel",
    "OperatorTransportLeakageError",
    "OperatorTransportMetrics",
    "OperatorTransportMissingMeasurementError",
    "OperatorTransportObservation",
    "OperatorTransportSplit",
    "QWEN_PREFIX_SINKHORN_ATTENTION_MODE",
    "QwenOperatorTransportAvailability",
    "QwenPrefixSinkhornCaptureAudit",
    "QwenPrefixSinkhornCaptureBank",
    "QwenPrefixSinkhornCaptureReceipt",
    "causal_prefix_sinkhorn_operator",
    "chronological_operator_transport_split",
    "evaluate_operator_transport_holdout",
    "fit_operator_transport",
    "inspect_qwen_operator_transport_availability",
    "operator_transport_evidence_sha256",
    "qwen_operator_transport_corpus_from_atlas",
    "qwen_operator_transport_corpus_from_captures",
]
