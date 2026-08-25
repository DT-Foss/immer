"""Authenticated semantic coordinates over immutable Qwen weight ranges.

The causal weight graph already answers *where* a tensor lives.  This module
adds append-only observations about what an exact tensor range did for an
exact prompt and intervention.  It is deliberately a control-plane sidecar:
no method owns a tensor source or exposes a weight-write operation.

Observations and verified labels are different record types.  A semantic
label is promoted only through an explicit placebo/effect gate and a declared
replica quorum.  Conflicting observations and promotions remain addressable;
the atlas never replaces an earlier record with a newer opinion.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import PurePosixPath
import re
from typing import Any, Literal

from immer.knowledge.livecausal import LiveGraph

from ..deepseek_v4.causal_weights import TensorRangePlan


MODEL_PIN_SCHEMA = "immer.qwen3.8-semantic-atlas-model-pin/v1"
WEIGHT_COORDINATE_SCHEMA = "immer.qwen3.8-semantic-atlas-coordinate/v1"
PROBE_IDENTITY_SCHEMA = "immer.qwen3.8-semantic-atlas-probe/v1"
INTERVENTION_IDENTITY_SCHEMA = "immer.qwen3.8-semantic-atlas-intervention/v1"
RUNTIME_PROVENANCE_SCHEMA = "immer.qwen3.8-semantic-atlas-runtime/v1"
GRAPH_REVISION_SCHEMA = "immer.qwen3.8-semantic-atlas-graph-revision/v1"
MEASUREMENT_RECEIPT_SCHEMA = "immer.qwen3.8-semantic-atlas-measurement/v1"
REPLICA_RECEIPT_SCHEMA = "immer.qwen3.8-semantic-atlas-replica/v1"
EVIDENCE_POLICY_SCHEMA = "immer.qwen3.8-semantic-atlas-policy/v1"
LABEL_PROMOTION_SCHEMA = "immer.qwen3.8-semantic-atlas-label-promotion/v1"
COVERAGE_MATRIX_SCHEMA = "immer.qwen3.8-semantic-atlas-coverage/v1"
CONSENSUS_SNAPSHOT_SCHEMA = "immer.qwen3.8-semantic-atlas-consensus/v1"
ATLAS_EDGE_SCHEMA = "immer.qwen3.8-semantic-atlas-edge/v1"

_PREFIX = "semantic-atlas:v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CODE_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_DTYPE = re.compile(r"[A-Z][A-Z0-9_]*\Z")
_LAYER = re.compile(r"(?:^|\.)layers\.(0|[1-9][0-9]*)(?:\.|$)")
_UINT64_MAX = (1 << 64) - 1
_DTYPE_BYTES = {
    "BF16": 2,
    "BOOL": 1,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "U8": 1,
    "U16": 2,
    "U32": 4,
    "U64": 8,
}
_INTERVENTION_MODES = frozenset(
    ("passive", "off", "native", "ablation", "patch", "placebo")
)
_OBSERVATION_STATUSES = frozenset(("recorded", "eligible", "invalidated"))
_REPLICA_VERDICTS = frozenset(("support", "oppose", "inconclusive"))
_EFFECT_DIRECTIONS = frozenset(("positive", "negative", "absolute"))


class SemanticAtlasError(ValueError):
    """Base class for invalid semantic-atlas input or state."""


class SemanticAtlasIntegrityError(SemanticAtlasError):
    """Stored semantic records do not satisfy their canonical identity."""


class SemanticAtlasStaleCheckpointError(SemanticAtlasIntegrityError):
    """The atlas belongs to a different immutable Qwen checkpoint."""


class SemanticAtlasConflictError(SemanticAtlasIntegrityError):
    """One content identity resolves to incompatible immutable records."""


class SemanticAtlasPromotionError(SemanticAtlasError):
    """Evidence does not satisfy the explicitly selected promotion policy."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SemanticAtlasError("value is not canonical JSON") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _seal(schema: str, body: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(_canonical(dict(body)))
    return {"body": normalized, "schema": schema, "sha256": _digest(normalized)}


def _unseal(
    document: Mapping[str, Any],
    *,
    schema: str,
    fields: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(document, Mapping) or set(document) != {
        "body",
        "schema",
        "sha256",
    }:
        raise SemanticAtlasIntegrityError("sealed document shape is invalid")
    if document.get("schema") != schema:
        raise SemanticAtlasIntegrityError("sealed document schema is invalid")
    body = document.get("body")
    if not isinstance(body, Mapping) or set(body) != fields:
        raise SemanticAtlasIntegrityError("sealed document body is invalid")
    claimed = document.get("sha256")
    if not isinstance(claimed, str) or _SHA256.fullmatch(claimed) is None:
        raise SemanticAtlasIntegrityError("sealed document SHA-256 is invalid")
    if claimed != _digest(body):
        raise SemanticAtlasIntegrityError("sealed document SHA-256 mismatch")
    return json.loads(_canonical(dict(body)))


def _text(value: object, label: str, *, maximum: int = 2048) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value) > maximum
    ):
        raise SemanticAtlasError(f"{label} is not canonical text")
    return value


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SemanticAtlasError(f"{label} must be a lowercase SHA-256")
    return value


def _uint(value: object, label: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= _UINT64_MAX
    ):
        qualifier = "positive " if positive else ""
        raise SemanticAtlasError(f"{label} must be a {qualifier}uint64")
    return value


def _finite(value: object, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SemanticAtlasError(f"{label} must be finite numeric data")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise SemanticAtlasError(f"{label} must be finite numeric data")
    return 0.0 if result == 0.0 else result


def _safe_shard(value: object) -> str:
    shard = _text(value, "shard", maximum=1024)
    if "\\" in shard:
        raise SemanticAtlasError("shard must use POSIX separators")
    path = PurePosixPath(shard)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise SemanticAtlasError("shard must be a safe relative path")
    return shard


def _label_key(label: str, *, verified: bool) -> str:
    value = _text(label, "semantic_label", maximum=512)
    kind = "verified-label" if verified else "observed-label"
    return f"{_PREFIX}:{kind}:{_digest({'semantic_label': value})}"


def _record_key(kind: str, sha256: str) -> str:
    return f"{_PREFIX}:record:{kind}:{_sha(sha256, 'record_sha256')}"


@dataclass(frozen=True, slots=True)
class ModelPin:
    """Logical model, physical bundle, and executing code revision."""

    repo_id: str
    revision: str
    bundle_fingerprint: str
    bundle_manifest_sha256: str
    code_revision: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "repo_id", _text(self.repo_id, "repo_id"))
        object.__setattr__(self, "revision", _text(self.revision, "revision"))
        object.__setattr__(
            self,
            "bundle_fingerprint",
            _sha(self.bundle_fingerprint, "bundle_fingerprint"),
        )
        object.__setattr__(
            self,
            "bundle_manifest_sha256",
            _sha(self.bundle_manifest_sha256, "bundle_manifest_sha256"),
        )
        if (
            not isinstance(self.code_revision, str)
            or _CODE_REVISION.fullmatch(self.code_revision) is None
        ):
            raise SemanticAtlasError("code_revision must be a full Git/SHA revision")

    def as_record(self) -> dict[str, str]:
        return {
            "bundle_fingerprint": self.bundle_fingerprint,
            "bundle_manifest_sha256": self.bundle_manifest_sha256,
            "code_revision": self.code_revision,
            "repo_id": self.repo_id,
            "revision": self.revision,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())

    def to_document(self) -> dict[str, Any]:
        return _seal(MODEL_PIN_SCHEMA, self.as_record())

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> ModelPin:
        body = _unseal(
            document,
            schema=MODEL_PIN_SCHEMA,
            fields=frozenset(
                (
                    "bundle_fingerprint",
                    "bundle_manifest_sha256",
                    "code_revision",
                    "repo_id",
                    "revision",
                )
            ),
        )
        return cls(**body)


@dataclass(frozen=True, slots=True)
class WeightCoordinate:
    """One tensor plan plus one exact selected half-open byte range."""

    layer: int
    module: str
    tensor: str
    dtype: str
    shape: tuple[int, ...]
    shard: str
    tensor_absolute_offset: int
    tensor_length: int
    range_absolute_offset: int
    range_length: int
    head_index: int | None = None
    row_start: int | None = None
    row_end: int | None = None

    def __post_init__(self) -> None:
        layer = _uint(self.layer, "layer")
        module = _text(self.module, "module")
        tensor = _text(self.tensor, "tensor")
        if tensor != module and not tensor.startswith(f"{module}."):
            raise SemanticAtlasError("module is not the tensor's exact module prefix")
        matches = tuple(int(match) for match in _LAYER.findall(tensor))
        if len(matches) != 1 or matches[0] != layer:
            raise SemanticAtlasError("tensor name does not identify coordinate layer")
        dtype = _text(self.dtype, "dtype", maximum=32)
        if _DTYPE.fullmatch(dtype) is None or dtype not in _DTYPE_BYTES:
            raise SemanticAtlasError("dtype is unsupported for exact byte validation")
        try:
            shape = tuple(self.shape)
        except TypeError as exc:
            raise SemanticAtlasError("shape must be an integer sequence") from exc
        if not shape:
            raise SemanticAtlasError("shape must not be empty")
        normalized_shape = tuple(
            _uint(value, "shape dimension", positive=True) for value in shape
        )
        shard = _safe_shard(self.shard)
        tensor_offset = _uint(self.tensor_absolute_offset, "tensor_absolute_offset")
        tensor_length = _uint(self.tensor_length, "tensor_length", positive=True)
        range_offset = _uint(self.range_absolute_offset, "range_absolute_offset")
        range_length = _uint(self.range_length, "range_length", positive=True)
        element_count = math.prod(normalized_shape)
        if element_count * _DTYPE_BYTES[dtype] != tensor_length:
            raise SemanticAtlasError("tensor length disagrees with dtype and shape")
        tensor_end = tensor_offset + tensor_length
        range_end = range_offset + range_length
        if tensor_end > _UINT64_MAX or range_end > _UINT64_MAX:
            raise SemanticAtlasError("coordinate byte range overflows uint64")
        if not tensor_offset <= range_offset < range_end <= tensor_end:
            raise SemanticAtlasError("selected byte range is outside tensor plan")
        head = self.head_index
        if head is not None:
            head = _uint(head, "head_index")
        if (self.row_start is None) != (self.row_end is None):
            raise SemanticAtlasError("row_start and row_end must be supplied together")
        row_start = self.row_start
        row_end = self.row_end
        if row_start is not None and row_end is not None:
            row_start = _uint(row_start, "row_start")
            row_end = _uint(row_end, "row_end", positive=True)
            if not row_start < row_end <= normalized_shape[0]:
                raise SemanticAtlasError("row range is outside tensor shape")
            row_bytes = math.prod(normalized_shape[1:]) * _DTYPE_BYTES[dtype]
            expected_offset = tensor_offset + row_start * row_bytes
            expected_length = (row_end - row_start) * row_bytes
            if (range_offset, range_length) != (expected_offset, expected_length):
                raise SemanticAtlasError("row range and selected byte range disagree")
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "module", module)
        object.__setattr__(self, "tensor", tensor)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "shape", normalized_shape)
        object.__setattr__(self, "shard", shard)
        object.__setattr__(self, "tensor_absolute_offset", tensor_offset)
        object.__setattr__(self, "tensor_length", tensor_length)
        object.__setattr__(self, "range_absolute_offset", range_offset)
        object.__setattr__(self, "range_length", range_length)
        object.__setattr__(self, "head_index", head)
        object.__setattr__(self, "row_start", row_start)
        object.__setattr__(self, "row_end", row_end)

    @classmethod
    def from_plan(
        cls,
        plan: TensorRangePlan,
        *,
        layer: int,
        module: str,
        head_index: int | None = None,
        row_start: int | None = None,
        row_end: int | None = None,
        relative_byte_offset: int = 0,
        byte_length: int | None = None,
    ) -> WeightCoordinate:
        if not isinstance(plan, TensorRangePlan):
            raise TypeError("plan must be a TensorRangePlan")
        relative = _uint(relative_byte_offset, "relative_byte_offset")
        if row_start is not None or row_end is not None:
            if relative != 0 or byte_length is not None:
                raise SemanticAtlasError(
                    "row coordinates cannot also specify an arbitrary byte range"
                )
            if row_start is None or row_end is None:
                raise SemanticAtlasError("row range must be complete")
            dtype_bytes = _DTYPE_BYTES.get(plan.dtype)
            if dtype_bytes is None or not plan.shape:
                raise SemanticAtlasError("plan cannot be mapped to exact rows")
            row_bytes = math.prod(plan.shape[1:]) * dtype_bytes
            relative = row_start * row_bytes
            byte_length = (row_end - row_start) * row_bytes
        length = plan.length - relative if byte_length is None else byte_length
        return cls(
            layer=layer,
            module=module,
            tensor=plan.name,
            dtype=plan.dtype,
            shape=tuple(plan.shape),
            shard=plan.shard,
            tensor_absolute_offset=plan.absolute_offset,
            tensor_length=plan.length,
            range_absolute_offset=plan.absolute_offset + relative,
            range_length=length,
            head_index=head_index,
            row_start=row_start,
            row_end=row_end,
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "head_index": self.head_index,
            "layer": self.layer,
            "module": self.module,
            "range_absolute_offset": self.range_absolute_offset,
            "range_length": self.range_length,
            "row_end": self.row_end,
            "row_start": self.row_start,
            "shape": list(self.shape),
            "shard": self.shard,
            "tensor": self.tensor,
            "tensor_absolute_offset": self.tensor_absolute_offset,
            "tensor_length": self.tensor_length,
        }

    @property
    def tensor_plan_sha256(self) -> str:
        return _digest(
            {
                "absolute_offset": self.tensor_absolute_offset,
                "dtype": self.dtype,
                "length": self.tensor_length,
                "name": self.tensor,
                "shape": list(self.shape),
                "shard": self.shard,
            }
        )

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())

    def matches_plan(self, plan: TensorRangePlan) -> bool:
        return isinstance(plan, TensorRangePlan) and (
            self.tensor,
            self.dtype,
            self.shape,
            self.shard,
            self.tensor_absolute_offset,
            self.tensor_length,
        ) == (
            plan.name,
            plan.dtype,
            tuple(plan.shape),
            plan.shard,
            plan.absolute_offset,
            plan.length,
        )

    def to_document(self) -> dict[str, Any]:
        return _seal(WEIGHT_COORDINATE_SCHEMA, self.as_record())

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> WeightCoordinate:
        body = _unseal(
            document,
            schema=WEIGHT_COORDINATE_SCHEMA,
            fields=frozenset(
                (
                    "dtype",
                    "head_index",
                    "layer",
                    "module",
                    "range_absolute_offset",
                    "range_length",
                    "row_end",
                    "row_start",
                    "shape",
                    "shard",
                    "tensor",
                    "tensor_absolute_offset",
                    "tensor_length",
                )
            ),
        )
        body["shape"] = tuple(body["shape"])
        return cls(**body)


@dataclass(frozen=True, slots=True)
class ProbeIdentity:
    """Hashes of the exact question, tokens, family, and label source."""

    question_sha256: str
    token_sha256: str
    family_sha256: str
    label_source_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "question_sha256",
            "token_sha256",
            "family_sha256",
            "label_source_sha256",
        ):
            object.__setattr__(self, field, _sha(getattr(self, field), field))

    def as_record(self) -> dict[str, str]:
        return {
            "family_sha256": self.family_sha256,
            "label_source_sha256": self.label_source_sha256,
            "question_sha256": self.question_sha256,
            "token_sha256": self.token_sha256,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())

    @property
    def prompt_signature(self) -> str:
        return _digest(
            {
                "question_sha256": self.question_sha256,
                "token_sha256": self.token_sha256,
            }
        )

    def to_document(self) -> dict[str, Any]:
        return _seal(PROBE_IDENTITY_SCHEMA, self.as_record())

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> ProbeIdentity:
        body = _unseal(
            document,
            schema=PROBE_IDENTITY_SCHEMA,
            fields=frozenset(
                (
                    "family_sha256",
                    "label_source_sha256",
                    "question_sha256",
                    "token_sha256",
                )
            ),
        )
        return cls(**body)


@dataclass(frozen=True, slots=True)
class InterventionIdentity:
    """Exact passive/native/ablation/patch/placebo experiment identity."""

    mode: Literal["passive", "off", "native", "ablation", "patch", "placebo"] | str
    configuration_sha256: str

    def __post_init__(self) -> None:
        mode = _text(self.mode, "intervention mode", maximum=32)
        if mode not in _INTERVENTION_MODES:
            raise SemanticAtlasError("intervention mode is invalid")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(
            self,
            "configuration_sha256",
            _sha(self.configuration_sha256, "configuration_sha256"),
        )

    def as_record(self) -> dict[str, str]:
        return {
            "configuration_sha256": self.configuration_sha256,
            "mode": str(self.mode),
        }

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())

    def to_document(self) -> dict[str, Any]:
        return _seal(INTERVENTION_IDENTITY_SCHEMA, self.as_record())

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> InterventionIdentity:
        body = _unseal(
            document,
            schema=INTERVENTION_IDENTITY_SCHEMA,
            fields=frozenset(("configuration_sha256", "mode")),
        )
        return cls(**body)


@dataclass(frozen=True, slots=True)
class RuntimeProvenance:
    """Hashes sufficient to reproduce the executing runtime environment."""

    code_revision: str
    source_manifest_sha256: str
    dependency_manifest_sha256: str
    runtime_configuration_sha256: str
    platform_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.code_revision, str)
            or _CODE_REVISION.fullmatch(self.code_revision) is None
        ):
            raise SemanticAtlasError("runtime code_revision is invalid")
        for field in (
            "source_manifest_sha256",
            "dependency_manifest_sha256",
            "runtime_configuration_sha256",
            "platform_sha256",
        ):
            object.__setattr__(self, field, _sha(getattr(self, field), field))

    def as_record(self) -> dict[str, str]:
        return {
            "code_revision": self.code_revision,
            "dependency_manifest_sha256": self.dependency_manifest_sha256,
            "platform_sha256": self.platform_sha256,
            "runtime_configuration_sha256": self.runtime_configuration_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())

    def to_document(self) -> dict[str, Any]:
        return _seal(RUNTIME_PROVENANCE_SCHEMA, self.as_record())

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> RuntimeProvenance:
        body = _unseal(
            document,
            schema=RUNTIME_PROVENANCE_SCHEMA,
            fields=frozenset(
                (
                    "code_revision",
                    "dependency_manifest_sha256",
                    "platform_sha256",
                    "runtime_configuration_sha256",
                    "source_manifest_sha256",
                )
            ),
        )
        return cls(**body)


@dataclass(frozen=True, slots=True)
class GraphRevision:
    """One exact append-only LiveGraph journal head."""

    sequence: int
    event_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequence", _uint(self.sequence, "graph sequence"))
        object.__setattr__(
            self,
            "event_sha256",
            _sha(self.event_sha256, "graph event_sha256"),
        )

    @classmethod
    def from_live_revision(cls, revision: Sequence[object]) -> GraphRevision:
        if isinstance(revision, (str, bytes)) or len(revision) != 2:
            raise SemanticAtlasError("LiveGraph revision must be [sequence, SHA-256]")
        return cls(sequence=revision[0], event_sha256=revision[1])  # type: ignore[arg-type]

    def as_record(self) -> dict[str, Any]:
        return {"event_sha256": self.event_sha256, "sequence": self.sequence}

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())

    def to_document(self) -> dict[str, Any]:
        return _seal(GRAPH_REVISION_SCHEMA, self.as_record())

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> GraphRevision:
        body = _unseal(
            document,
            schema=GRAPH_REVISION_SCHEMA,
            fields=frozenset(("event_sha256", "sequence")),
        )
        return cls(**body)


@dataclass(frozen=True, slots=True)
class NumericSummary:
    """Mergeable sufficient statistics; activations themselves are not stored."""

    metric: str
    count: int
    total: float
    total_squares: float
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        metric = _text(self.metric, "summary metric", maximum=256)
        count = _uint(self.count, "summary count", positive=True)
        total = _finite(self.total, "summary total")
        total_squares = _finite(
            self.total_squares, "summary total_squares", nonnegative=True
        )
        minimum = _finite(self.minimum, "summary minimum")
        maximum = _finite(self.maximum, "summary maximum")
        if minimum > maximum:
            raise SemanticAtlasError("summary minimum exceeds maximum")
        tolerance = 1e-10 * max(
            1.0, abs(total), abs(minimum * count), abs(maximum * count)
        )
        if total < minimum * count - tolerance or total > maximum * count + tolerance:
            raise SemanticAtlasError("summary total is outside its extrema")
        cauchy_tolerance = 1e-10 * max(1.0, total * total, count * total_squares)
        if total * total > count * total_squares + cauchy_tolerance:
            raise SemanticAtlasError("summary total_squares violates Cauchy bound")
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "total", total)
        object.__setattr__(self, "total_squares", total_squares)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    @property
    def mean(self) -> float:
        return self.total / self.count

    def as_record(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "maximum": self.maximum,
            "metric": self.metric,
            "minimum": self.minimum,
            "total": self.total,
            "total_squares": self.total_squares,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> NumericSummary:
        if not isinstance(record, Mapping) or set(record) != {
            "count",
            "maximum",
            "metric",
            "minimum",
            "total",
            "total_squares",
        }:
            raise SemanticAtlasIntegrityError("numeric summary schema is invalid")
        return cls(**dict(record))


@dataclass(frozen=True, slots=True)
class PlaceboEffect:
    """One observed-minus-placebo effect bound to the placebo receipt."""

    metric: str
    placebo_measurement_sha256: str
    observed_mean: float
    placebo_mean: float
    delta: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "metric", _text(self.metric, "effect metric", maximum=256)
        )
        object.__setattr__(
            self,
            "placebo_measurement_sha256",
            _sha(self.placebo_measurement_sha256, "placebo_measurement_sha256"),
        )
        observed = _finite(self.observed_mean, "observed_mean")
        placebo = _finite(self.placebo_mean, "placebo_mean")
        delta = _finite(self.delta, "effect delta")
        expected = observed - placebo
        if not math.isclose(delta, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise SemanticAtlasError("effect delta is not observed_mean - placebo_mean")
        object.__setattr__(self, "observed_mean", observed)
        object.__setattr__(self, "placebo_mean", placebo)
        object.__setattr__(self, "delta", delta)

    def as_record(self) -> dict[str, Any]:
        return {
            "delta": self.delta,
            "metric": self.metric,
            "observed_mean": self.observed_mean,
            "placebo_mean": self.placebo_mean,
            "placebo_measurement_sha256": self.placebo_measurement_sha256,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> PlaceboEffect:
        if not isinstance(record, Mapping) or set(record) != {
            "delta",
            "metric",
            "observed_mean",
            "placebo_mean",
            "placebo_measurement_sha256",
        }:
            raise SemanticAtlasIntegrityError("placebo effect schema is invalid")
        return cls(**dict(record))


@dataclass(frozen=True, slots=True)
class MeasurementReceipt:
    """Immutable observation; it is not by itself a verified semantic label."""

    model_pin: ModelPin
    coordinate: WeightCoordinate
    probe: ProbeIdentity
    intervention: InterventionIdentity
    observation_status: Literal["recorded", "eligible", "invalidated"] | str
    observed_semantic_label: str | None
    hidden_sha256: str
    activation_sha256: str
    logits_sha256: str
    state_sha256: str
    access_trace_sha256: str
    evidence_sha256: str
    weight_rail_revision: GraphRevision
    atlas_head_revision: GraphRevision
    numeric_summaries: tuple[NumericSummary, ...]
    placebo_effects: tuple[PlaceboEffect, ...]
    runtime: RuntimeProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.model_pin, ModelPin):
            raise TypeError("model_pin must be a ModelPin")
        if not isinstance(self.coordinate, WeightCoordinate):
            raise TypeError("coordinate must be a WeightCoordinate")
        if not isinstance(self.probe, ProbeIdentity):
            raise TypeError("probe must be a ProbeIdentity")
        if not isinstance(self.intervention, InterventionIdentity):
            raise TypeError("intervention must be an InterventionIdentity")
        if not isinstance(self.runtime, RuntimeProvenance):
            raise TypeError("runtime must be RuntimeProvenance")
        if not isinstance(self.weight_rail_revision, GraphRevision):
            raise TypeError("weight_rail_revision must be a GraphRevision")
        if not isinstance(self.atlas_head_revision, GraphRevision):
            raise TypeError("atlas_head_revision must be a GraphRevision")
        if self.runtime.code_revision != self.model_pin.code_revision:
            raise SemanticAtlasError("runtime and model pin code revisions differ")
        status = _text(self.observation_status, "observation_status", maximum=32)
        if status not in _OBSERVATION_STATUSES:
            raise SemanticAtlasError("observation_status is invalid")
        label = self.observed_semantic_label
        if label is not None:
            label = _text(label, "observed_semantic_label", maximum=512)
        if status == "eligible" and label is None:
            raise SemanticAtlasError("eligible observation requires an observed label")
        for field in (
            "hidden_sha256",
            "activation_sha256",
            "logits_sha256",
            "state_sha256",
            "access_trace_sha256",
            "evidence_sha256",
        ):
            object.__setattr__(self, field, _sha(getattr(self, field), field))
        try:
            summaries = tuple(self.numeric_summaries)
            effects = tuple(self.placebo_effects)
        except TypeError as exc:
            raise SemanticAtlasError(
                "measurement summaries/effects must be sequences"
            ) from exc
        if not summaries or any(
            not isinstance(row, NumericSummary) for row in summaries
        ):
            raise SemanticAtlasError("measurement requires numeric summaries")
        if any(not isinstance(row, PlaceboEffect) for row in effects):
            raise SemanticAtlasError("placebo_effects are invalid")
        summaries = tuple(sorted(summaries, key=lambda row: row.metric))
        effects = tuple(
            sorted(
                effects, key=lambda row: (row.metric, row.placebo_measurement_sha256)
            )
        )
        if len({row.metric for row in summaries}) != len(summaries):
            raise SemanticAtlasError("numeric summary metric is duplicated")
        if len({row.metric for row in effects}) != len(effects):
            raise SemanticAtlasError("placebo effect metric is duplicated")
        by_metric = {row.metric: row for row in summaries}
        for effect in effects:
            summary = by_metric.get(effect.metric)
            if summary is None or not math.isclose(
                summary.mean,
                effect.observed_mean,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise SemanticAtlasError("effect is not bound to its observed summary")
        if self.intervention.mode == "placebo" and effects:
            raise SemanticAtlasError("a placebo observation cannot contain effects")
        object.__setattr__(self, "observation_status", status)
        object.__setattr__(self, "observed_semantic_label", label)
        object.__setattr__(self, "numeric_summaries", summaries)
        object.__setattr__(self, "placebo_effects", effects)

    def as_record(self) -> dict[str, Any]:
        return {
            "access_trace_sha256": self.access_trace_sha256,
            "activation_sha256": self.activation_sha256,
            "atlas_head_revision": self.atlas_head_revision.to_document(),
            "coordinate": self.coordinate.to_document(),
            "evidence_sha256": self.evidence_sha256,
            "hidden_sha256": self.hidden_sha256,
            "intervention": self.intervention.to_document(),
            "logits_sha256": self.logits_sha256,
            "model_pin": self.model_pin.to_document(),
            "numeric_summaries": [row.as_record() for row in self.numeric_summaries],
            "observation_status": str(self.observation_status),
            "observed_semantic_label": self.observed_semantic_label,
            "placebo_effects": [row.as_record() for row in self.placebo_effects],
            "probe": self.probe.to_document(),
            "runtime": self.runtime.to_document(),
            "state_sha256": self.state_sha256,
            "tensor_plan_sha256": self.coordinate.tensor_plan_sha256,
            "weight_rail_revision": self.weight_rail_revision.to_document(),
        }

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())

    def to_document(self) -> dict[str, Any]:
        return _seal(MEASUREMENT_RECEIPT_SCHEMA, self.as_record())

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> MeasurementReceipt:
        body = _unseal(
            document,
            schema=MEASUREMENT_RECEIPT_SCHEMA,
            fields=frozenset(
                (
                    "access_trace_sha256",
                    "activation_sha256",
                    "atlas_head_revision",
                    "coordinate",
                    "evidence_sha256",
                    "hidden_sha256",
                    "intervention",
                    "logits_sha256",
                    "model_pin",
                    "numeric_summaries",
                    "observation_status",
                    "observed_semantic_label",
                    "placebo_effects",
                    "probe",
                    "runtime",
                    "state_sha256",
                    "tensor_plan_sha256",
                    "weight_rail_revision",
                )
            ),
        )
        coordinate = WeightCoordinate.from_document(body.pop("coordinate"))
        tensor_plan_sha256 = body.pop("tensor_plan_sha256")
        if tensor_plan_sha256 != coordinate.tensor_plan_sha256:
            raise SemanticAtlasIntegrityError("measurement tensor plan SHA mismatch")
        body["model_pin"] = ModelPin.from_document(body["model_pin"])
        body["coordinate"] = coordinate
        body["probe"] = ProbeIdentity.from_document(body["probe"])
        body["intervention"] = InterventionIdentity.from_document(body["intervention"])
        body["runtime"] = RuntimeProvenance.from_document(body["runtime"])
        body["weight_rail_revision"] = GraphRevision.from_document(
            body["weight_rail_revision"]
        )
        body["atlas_head_revision"] = GraphRevision.from_document(
            body["atlas_head_revision"]
        )
        body["numeric_summaries"] = tuple(
            NumericSummary.from_record(row) for row in body["numeric_summaries"]
        )
        body["placebo_effects"] = tuple(
            PlaceboEffect.from_record(row) for row in body["placebo_effects"]
        )
        return cls(**body)


@dataclass(frozen=True, slots=True)
class ReplicaReceipt:
    """One declared replica vote with an explicit, non-security reputation weight."""

    model_pin: ModelPin
    coordinate: WeightCoordinate
    measurement_sha256: str
    probe_sha256: str
    intervention_sha256: str
    replica_id_sha256: str
    replica_measurement_sha256: str
    verdict: Literal["support", "oppose", "inconclusive"] | str
    semantic_label: str | None
    reputation_weight: float
    evidence_sha256: str
    runtime: RuntimeProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.model_pin, ModelPin):
            raise TypeError("model_pin must be a ModelPin")
        if not isinstance(self.coordinate, WeightCoordinate):
            raise TypeError("coordinate must be a WeightCoordinate")
        if not isinstance(self.runtime, RuntimeProvenance):
            raise TypeError("runtime must be RuntimeProvenance")
        if self.runtime.code_revision != self.model_pin.code_revision:
            raise SemanticAtlasError("replica runtime code revision differs")
        for field in (
            "measurement_sha256",
            "probe_sha256",
            "intervention_sha256",
            "replica_id_sha256",
            "replica_measurement_sha256",
            "evidence_sha256",
        ):
            object.__setattr__(self, field, _sha(getattr(self, field), field))
        verdict = _text(self.verdict, "replica verdict", maximum=32)
        if verdict not in _REPLICA_VERDICTS:
            raise SemanticAtlasError("replica verdict is invalid")
        label = self.semantic_label
        if label is not None:
            label = _text(label, "replica semantic_label", maximum=512)
        if verdict == "support" and label is None:
            raise SemanticAtlasError("support vote requires a semantic label")
        if verdict == "inconclusive" and label is not None:
            raise SemanticAtlasError("inconclusive vote cannot assert a label")
        weight = _finite(self.reputation_weight, "reputation_weight", nonnegative=True)
        if weight > 1.0:
            raise SemanticAtlasError("reputation_weight must be normalized to [0, 1]")
        object.__setattr__(self, "verdict", verdict)
        object.__setattr__(self, "semantic_label", label)
        object.__setattr__(self, "reputation_weight", weight)

    def as_record(self) -> dict[str, Any]:
        return {
            "coordinate": self.coordinate.to_document(),
            "evidence_sha256": self.evidence_sha256,
            "intervention_sha256": self.intervention_sha256,
            "measurement_sha256": self.measurement_sha256,
            "model_pin": self.model_pin.to_document(),
            "probe_sha256": self.probe_sha256,
            "replica_id_sha256": self.replica_id_sha256,
            "replica_measurement_sha256": self.replica_measurement_sha256,
            "reputation_weight": self.reputation_weight,
            "runtime": self.runtime.to_document(),
            "security_claim": "none",
            "semantic_label": self.semantic_label,
            "tensor_plan_sha256": self.coordinate.tensor_plan_sha256,
            "verdict": str(self.verdict),
        }

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())

    def to_document(self) -> dict[str, Any]:
        return _seal(REPLICA_RECEIPT_SCHEMA, self.as_record())

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> ReplicaReceipt:
        body = _unseal(
            document,
            schema=REPLICA_RECEIPT_SCHEMA,
            fields=frozenset(
                (
                    "coordinate",
                    "evidence_sha256",
                    "intervention_sha256",
                    "measurement_sha256",
                    "model_pin",
                    "probe_sha256",
                    "replica_id_sha256",
                    "replica_measurement_sha256",
                    "reputation_weight",
                    "runtime",
                    "security_claim",
                    "semantic_label",
                    "tensor_plan_sha256",
                    "verdict",
                )
            ),
        )
        if body.pop("security_claim") != "none":
            raise SemanticAtlasIntegrityError(
                "replica makes an unsupported security claim"
            )
        coordinate = WeightCoordinate.from_document(body.pop("coordinate"))
        if body.pop("tensor_plan_sha256") != coordinate.tensor_plan_sha256:
            raise SemanticAtlasIntegrityError("replica tensor plan SHA mismatch")
        body["model_pin"] = ModelPin.from_document(body["model_pin"])
        body["coordinate"] = coordinate
        body["runtime"] = RuntimeProvenance.from_document(body["runtime"])
        return cls(**body)


@dataclass(frozen=True, slots=True)
class EvidencePolicy:
    """Explicit promotion thresholds; this is not a Byzantine-security claim."""

    name: str
    minimum_replica_count: int
    minimum_support_count: int
    minimum_support_weight: float
    effect_metric: str
    minimum_effect: float
    effect_direction: Literal["positive", "negative", "absolute"] | str = "absolute"
    require_placebo: bool = True
    require_strict_count_majority: bool = True
    require_strict_weight_majority: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "policy name", maximum=256))
        replica_count = _uint(
            self.minimum_replica_count, "minimum_replica_count", positive=True
        )
        support_count = _uint(
            self.minimum_support_count, "minimum_support_count", positive=True
        )
        if support_count > replica_count:
            raise SemanticAtlasError("minimum_support_count exceeds replica quorum")
        support_weight = _finite(
            self.minimum_support_weight, "minimum_support_weight", nonnegative=True
        )
        effect_metric = _text(self.effect_metric, "effect_metric", maximum=256)
        minimum_effect = _finite(
            self.minimum_effect, "minimum_effect", nonnegative=True
        )
        direction = _text(self.effect_direction, "effect_direction", maximum=32)
        if direction not in _EFFECT_DIRECTIONS:
            raise SemanticAtlasError("effect_direction is invalid")
        for field in (
            "require_placebo",
            "require_strict_count_majority",
            "require_strict_weight_majority",
        ):
            if not isinstance(getattr(self, field), bool):
                raise SemanticAtlasError(f"{field} must be a boolean")
        object.__setattr__(self, "minimum_replica_count", replica_count)
        object.__setattr__(self, "minimum_support_count", support_count)
        object.__setattr__(self, "minimum_support_weight", support_weight)
        object.__setattr__(self, "effect_metric", effect_metric)
        object.__setattr__(self, "minimum_effect", minimum_effect)
        object.__setattr__(self, "effect_direction", direction)

    def as_record(self) -> dict[str, Any]:
        return {
            "effect_direction": str(self.effect_direction),
            "effect_metric": self.effect_metric,
            "minimum_effect": self.minimum_effect,
            "minimum_replica_count": self.minimum_replica_count,
            "minimum_support_count": self.minimum_support_count,
            "minimum_support_weight": self.minimum_support_weight,
            "name": self.name,
            "require_placebo": self.require_placebo,
            "require_strict_count_majority": self.require_strict_count_majority,
            "require_strict_weight_majority": self.require_strict_weight_majority,
            "security_claim": "none",
        }

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())

    def to_document(self) -> dict[str, Any]:
        return _seal(EVIDENCE_POLICY_SCHEMA, self.as_record())

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> EvidencePolicy:
        body = _unseal(
            document,
            schema=EVIDENCE_POLICY_SCHEMA,
            fields=frozenset(
                (
                    "effect_direction",
                    "effect_metric",
                    "minimum_effect",
                    "minimum_replica_count",
                    "minimum_support_count",
                    "minimum_support_weight",
                    "name",
                    "require_placebo",
                    "require_strict_count_majority",
                    "require_strict_weight_majority",
                    "security_claim",
                )
            ),
        )
        if body.pop("security_claim") != "none":
            raise SemanticAtlasIntegrityError(
                "policy makes an unsupported security claim"
            )
        return cls(**body)


@dataclass(frozen=True, slots=True)
class SemanticLabelPromotion:
    """A verified label receipt produced from one explicit evidence policy."""

    model_pin: ModelPin
    coordinate: WeightCoordinate
    probe: ProbeIdentity
    measurement_sha256: str
    semantic_label: str
    policy: EvidencePolicy
    support_replica_sha256s: tuple[str, ...]
    oppose_replica_sha256s: tuple[str, ...]
    inconclusive_replica_sha256s: tuple[str, ...]
    support_count: int
    oppose_count: int
    support_weight: float
    oppose_weight: float
    placebo_effect_sha256: str
    runtime: RuntimeProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.model_pin, ModelPin):
            raise TypeError("model_pin must be a ModelPin")
        if not isinstance(self.coordinate, WeightCoordinate):
            raise TypeError("coordinate must be a WeightCoordinate")
        if not isinstance(self.probe, ProbeIdentity):
            raise TypeError("probe must be a ProbeIdentity")
        if not isinstance(self.policy, EvidencePolicy):
            raise TypeError("policy must be an EvidencePolicy")
        if not isinstance(self.runtime, RuntimeProvenance):
            raise TypeError("runtime must be RuntimeProvenance")
        if self.runtime.code_revision != self.model_pin.code_revision:
            raise SemanticAtlasError("promotion runtime code revision differs")
        object.__setattr__(
            self,
            "measurement_sha256",
            _sha(self.measurement_sha256, "measurement_sha256"),
        )
        object.__setattr__(
            self,
            "semantic_label",
            _text(self.semantic_label, "semantic_label", maximum=512),
        )
        sets: list[tuple[str, ...]] = []
        for field in (
            "support_replica_sha256s",
            "oppose_replica_sha256s",
            "inconclusive_replica_sha256s",
        ):
            try:
                values = tuple(
                    sorted(_sha(value, field) for value in getattr(self, field))
                )
            except TypeError as exc:
                raise SemanticAtlasError(f"{field} must be a sequence") from exc
            if len(set(values)) != len(values):
                raise SemanticAtlasError(f"{field} contains duplicates")
            object.__setattr__(self, field, values)
            sets.append(values)
        if (
            set(sets[0]) & set(sets[1])
            or set(sets[0]) & set(sets[2])
            or set(sets[1]) & set(sets[2])
        ):
            raise SemanticAtlasError("promotion replica vote sets overlap")
        support_count = _uint(self.support_count, "support_count")
        oppose_count = _uint(self.oppose_count, "oppose_count")
        if support_count != len(sets[0]) or oppose_count != len(sets[1]):
            raise SemanticAtlasError("promotion vote counts disagree with receipts")
        support_weight = _finite(
            self.support_weight, "support_weight", nonnegative=True
        )
        oppose_weight = _finite(self.oppose_weight, "oppose_weight", nonnegative=True)
        object.__setattr__(
            self,
            "placebo_effect_sha256",
            _sha(self.placebo_effect_sha256, "placebo_effect_sha256"),
        )
        object.__setattr__(self, "support_count", support_count)
        object.__setattr__(self, "oppose_count", oppose_count)
        object.__setattr__(self, "support_weight", support_weight)
        object.__setattr__(self, "oppose_weight", oppose_weight)

    def as_record(self) -> dict[str, Any]:
        return {
            "coordinate": self.coordinate.to_document(),
            "inconclusive_replica_sha256s": list(self.inconclusive_replica_sha256s),
            "measurement_sha256": self.measurement_sha256,
            "model_pin": self.model_pin.to_document(),
            "oppose_count": self.oppose_count,
            "oppose_replica_sha256s": list(self.oppose_replica_sha256s),
            "oppose_weight": self.oppose_weight,
            "placebo_effect_sha256": self.placebo_effect_sha256,
            "policy": self.policy.to_document(),
            "probe": self.probe.to_document(),
            "runtime": self.runtime.to_document(),
            "security_claim": "none",
            "semantic_label": self.semantic_label,
            "support_count": self.support_count,
            "support_replica_sha256s": list(self.support_replica_sha256s),
            "support_weight": self.support_weight,
            "tensor_plan_sha256": self.coordinate.tensor_plan_sha256,
            "verification_status": "verified",
        }

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())

    def to_document(self) -> dict[str, Any]:
        return _seal(LABEL_PROMOTION_SCHEMA, self.as_record())

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> SemanticLabelPromotion:
        body = _unseal(
            document,
            schema=LABEL_PROMOTION_SCHEMA,
            fields=frozenset(
                (
                    "coordinate",
                    "inconclusive_replica_sha256s",
                    "measurement_sha256",
                    "model_pin",
                    "oppose_count",
                    "oppose_replica_sha256s",
                    "oppose_weight",
                    "placebo_effect_sha256",
                    "policy",
                    "probe",
                    "runtime",
                    "security_claim",
                    "semantic_label",
                    "support_count",
                    "support_replica_sha256s",
                    "support_weight",
                    "tensor_plan_sha256",
                    "verification_status",
                )
            ),
        )
        if (
            body.pop("security_claim") != "none"
            or body.pop("verification_status") != "verified"
        ):
            raise SemanticAtlasIntegrityError(
                "promotion status/security marker is invalid"
            )
        coordinate = WeightCoordinate.from_document(body.pop("coordinate"))
        if body.pop("tensor_plan_sha256") != coordinate.tensor_plan_sha256:
            raise SemanticAtlasIntegrityError("promotion tensor plan SHA mismatch")
        body["model_pin"] = ModelPin.from_document(body["model_pin"])
        body["coordinate"] = coordinate
        body["probe"] = ProbeIdentity.from_document(body["probe"])
        body["policy"] = EvidencePolicy.from_document(body["policy"])
        body["runtime"] = RuntimeProvenance.from_document(body["runtime"])
        for field in (
            "support_replica_sha256s",
            "oppose_replica_sha256s",
            "inconclusive_replica_sha256s",
        ):
            body[field] = tuple(body[field])
        return cls(**body)


AtlasDocument = MeasurementReceipt | ReplicaReceipt | SemanticLabelPromotion


@dataclass(frozen=True, slots=True)
class AtlasAppendReceipt:
    record_kind: str
    record_sha256: str
    segment_sha256: str
    appended: bool


@dataclass(frozen=True, slots=True)
class AtlasQueryResult:
    measurements: tuple[MeasurementReceipt, ...]
    replicas: tuple[ReplicaReceipt, ...]
    promotions: tuple[SemanticLabelPromotion, ...]


class SemanticWeightAtlas:
    """Append-only semantic labels mounted over one exact Qwen checkpoint."""

    def __init__(
        self,
        graph: LiveGraph | str | os.PathLike[str],
        *,
        model_pin: ModelPin,
        tensor_plans: Iterable[TensorRangePlan] | Mapping[str, TensorRangePlan],
    ) -> None:
        if not isinstance(model_pin, ModelPin):
            raise TypeError("model_pin must be a ModelPin")
        self.graph = graph if isinstance(graph, LiveGraph) else LiveGraph(graph)
        self.model_pin = model_pin
        values = (
            tensor_plans.values() if isinstance(tensor_plans, Mapping) else tensor_plans
        )
        try:
            plans = tuple(values)
        except TypeError as exc:
            raise SemanticAtlasError("tensor_plans must be iterable") from exc
        if not plans or any(not isinstance(plan, TensorRangePlan) for plan in plans):
            raise SemanticAtlasError("tensor_plans must contain TensorRangePlan values")
        by_name: dict[str, TensorRangePlan] = {}
        for plan in plans:
            prior = by_name.get(plan.name)
            if prior is not None and prior != plan:
                raise SemanticAtlasConflictError("tensor plan name is ambiguous")
            by_name[plan.name] = plan
        self._tensor_plans = by_name
        # LiveGraph authenticates its journal and segment set at mount.  This
        # pass validates atlas schemas/relations through its cached offsets;
        # explicit ``verify_or_raise`` remains the force-rehash audit path.
        self._collect_documents()

    def _validate_pin_coordinate(
        self, model_pin: ModelPin, coordinate: WeightCoordinate
    ) -> None:
        if model_pin != self.model_pin:
            raise SemanticAtlasStaleCheckpointError(
                "semantic receipt belongs to a stale/different Qwen checkpoint"
            )
        plan = self._tensor_plans.get(coordinate.tensor)
        if plan is None or not coordinate.matches_plan(plan):
            raise SemanticAtlasIntegrityError(
                "semantic coordinate does not match the authenticated tensor plan"
            )

    @staticmethod
    def _kind(document: AtlasDocument) -> str:
        if isinstance(document, MeasurementReceipt):
            return "measurement"
        if isinstance(document, ReplicaReceipt):
            return "replica"
        if isinstance(document, SemanticLabelPromotion):
            return "promotion"
        raise TypeError("unsupported atlas document")

    def _index_keys(self, document: AtlasDocument) -> dict[str, str]:
        kind = self._kind(document)
        keys = {
            "all": f"{_PREFIX}:all:{kind}:{self.model_pin.sha256}",
            "coordinate": (f"{_PREFIX}:coordinate:{kind}:{document.coordinate.sha256}"),
        }
        if isinstance(document, MeasurementReceipt):
            keys["prompt"] = (
                f"{_PREFIX}:prompt:measurement:{document.probe.prompt_signature}"
            )
            if document.observed_semantic_label is not None:
                keys["observed_label"] = _label_key(
                    document.observed_semantic_label, verified=False
                )
        elif isinstance(document, ReplicaReceipt):
            keys["measurement"] = (
                f"{_PREFIX}:measurement-replicas:{document.measurement_sha256}"
            )
        else:
            keys["prompt"] = (
                f"{_PREFIX}:prompt:promotion:{document.probe.prompt_signature}"
            )
            keys["measurement"] = (
                f"{_PREFIX}:measurement-promotions:{document.measurement_sha256}"
            )
            keys["verified_label"] = _label_key(document.semantic_label, verified=True)
        return keys

    def _edge_records(self, document: AtlasDocument) -> tuple[dict[str, Any], ...]:
        kind = self._kind(document)
        sealed = document.to_document()
        receipt_sha = sealed["sha256"]
        receipt_key = _record_key(kind, receipt_sha)
        common = {
            "coordinate_sha256": document.coordinate.sha256,
            "document_sha256": receipt_sha,
            "model_pin_sha256": document.model_pin.sha256,
            "record_kind": kind,
            "schema": ATLAS_EDGE_SCHEMA,
        }
        primary = {
            **common,
            "document": sealed,
            "edge_kind": "primary",
            "outcome_key": f"{_PREFIX}:document:{receipt_sha}",
            "trigger_key": receipt_key,
        }
        indexes = [
            {
                **common,
                "edge_kind": "index",
                "index_kind": index_kind,
                "outcome_key": receipt_key,
                "trigger_key": trigger,
            }
            for index_kind, trigger in sorted(self._index_keys(document).items())
        ]
        return (primary, *indexes)

    def _decode_document(self, document: Mapping[str, Any]) -> AtlasDocument:
        schema = document.get("schema") if isinstance(document, Mapping) else None
        if schema == MEASUREMENT_RECEIPT_SCHEMA:
            value: AtlasDocument = MeasurementReceipt.from_document(document)
        elif schema == REPLICA_RECEIPT_SCHEMA:
            value = ReplicaReceipt.from_document(document)
        elif schema == LABEL_PROMOTION_SCHEMA:
            value = SemanticLabelPromotion.from_document(document)
        else:
            raise SemanticAtlasIntegrityError("atlas document schema is unknown")
        self._validate_pin_coordinate(value.model_pin, value.coordinate)
        return value

    def _validate_edge(
        self,
        record: Mapping[str, Any],
        *,
        primaries: Mapping[str, AtlasDocument] | None = None,
    ) -> AtlasDocument | None:
        if record.get("schema") != ATLAS_EDGE_SCHEMA:
            return None
        edge_kind = record.get("edge_kind")
        common = {
            "coordinate_sha256",
            "document_sha256",
            "edge_kind",
            "model_pin_sha256",
            "outcome_key",
            "record_kind",
            "schema",
            "trigger_key",
        }
        expected = common | ({"document"} if edge_kind == "primary" else {"index_kind"})
        if edge_kind not in ("primary", "index") or set(record) != expected:
            raise SemanticAtlasIntegrityError("atlas edge schema is invalid")
        kind = record.get("record_kind")
        if kind not in ("measurement", "replica", "promotion"):
            raise SemanticAtlasIntegrityError("atlas edge record kind is invalid")
        receipt_sha = _sha(record.get("document_sha256"), "document_sha256")
        if record.get("model_pin_sha256") != self.model_pin.sha256:
            raise SemanticAtlasStaleCheckpointError(
                "atlas edge checkpoint pin is stale"
            )
        if edge_kind == "primary":
            value = self._decode_document(record["document"])
            if (
                self._kind(value) != kind
                or value.sha256 != receipt_sha
                or record.get("coordinate_sha256") != value.coordinate.sha256
                or record.get("trigger_key") != _record_key(kind, receipt_sha)
                or record.get("outcome_key") != f"{_PREFIX}:document:{receipt_sha}"
            ):
                raise SemanticAtlasIntegrityError(
                    "atlas primary edge identity mismatch"
                )
            return value
        if primaries is None:
            return None
        receipt_key = _record_key(kind, receipt_sha)
        value = primaries.get(receipt_key)
        if value is None:
            raise SemanticAtlasIntegrityError(
                "atlas index references a missing primary"
            )
        index_kind = record.get("index_kind")
        expected_keys = self._index_keys(value)
        if (
            index_kind not in expected_keys
            or record.get("trigger_key") != expected_keys[index_kind]
            or record.get("outcome_key") != receipt_key
            or record.get("coordinate_sha256") != value.coordinate.sha256
        ):
            raise SemanticAtlasIntegrityError("atlas index edge identity mismatch")
        return value

    def _collect_documents(self) -> dict[str, AtlasDocument]:
        primaries: dict[str, AtlasDocument] = {}
        for segment_sha in self.graph.store.segments():
            records = tuple(
                record
                for _sha_value, _index, record in self.graph.store.iter_records(
                    segment_sha
                )
            )
            atlas_records = tuple(
                record
                for record in records
                if record.get("schema") == ATLAS_EDGE_SCHEMA
            )
            if not atlas_records:
                continue
            if len(atlas_records) != len(records):
                raise SemanticAtlasIntegrityError(
                    "atlas segment mixes semantic and foreign record families"
                )
            primary_records = tuple(
                record
                for record in atlas_records
                if record.get("edge_kind") == "primary"
            )
            if len(primary_records) != 1:
                raise SemanticAtlasIntegrityError(
                    "atlas segment must contain exactly one primary document"
                )
            value = self._validate_edge(primary_records[0])
            assert value is not None
            expected_records = self._edge_records(value)
            if atlas_records != expected_records:
                raise SemanticAtlasIntegrityError(
                    "atlas segment is not one atomic primary/index record set"
                )
            key = _record_key(self._kind(value), value.sha256)
            prior = primaries.get(key)
            if prior is not None and prior.to_document() != value.to_document():
                raise SemanticAtlasConflictError("atlas content identity collision")
            primaries[key] = value
            for record in atlas_records[1:]:
                self._validate_edge(record, primaries=primaries)
        self._validate_relations(primaries)
        return primaries

    @staticmethod
    def _measurement_map(
        primaries: Mapping[str, AtlasDocument],
    ) -> dict[str, MeasurementReceipt]:
        return {
            value.sha256: value
            for value in primaries.values()
            if isinstance(value, MeasurementReceipt)
        }

    def _validate_measurement_effects_from(
        self,
        receipt: MeasurementReceipt,
        measurements: Mapping[str, MeasurementReceipt],
    ) -> None:
        summaries = {row.metric: row for row in receipt.numeric_summaries}
        for effect in receipt.placebo_effects:
            placebo = measurements.get(effect.placebo_measurement_sha256)
            if placebo is None:
                raise SemanticAtlasIntegrityError(
                    "measurement references an inactive placebo receipt"
                )
            if (
                placebo.model_pin != receipt.model_pin
                or placebo.coordinate != receipt.coordinate
                or placebo.probe != receipt.probe
                or placebo.intervention.mode != "placebo"
            ):
                raise SemanticAtlasIntegrityError(
                    "measurement placebo identity differs from observation"
                )
            placebo_summaries = {row.metric: row for row in placebo.numeric_summaries}
            if effect.metric not in summaries or effect.metric not in placebo_summaries:
                raise SemanticAtlasIntegrityError(
                    "measurement effect metric is absent from observation/placebo"
                )
            if not math.isclose(
                summaries[effect.metric].mean,
                effect.observed_mean,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ) or not math.isclose(
                placebo_summaries[effect.metric].mean,
                effect.placebo_mean,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise SemanticAtlasIntegrityError(
                    "measurement effect means are not authenticated summaries"
                )

    @staticmethod
    def _validate_replica_target(
        replica: ReplicaReceipt,
        target: MeasurementReceipt,
    ) -> None:
        if (
            replica.model_pin != target.model_pin
            or replica.coordinate != target.coordinate
            or replica.probe_sha256 != target.probe.sha256
            or replica.intervention_sha256 != target.intervention.sha256
        ):
            raise SemanticAtlasIntegrityError(
                "replica identity differs from target measurement"
            )
        if (
            replica.verdict == "support"
            and replica.semantic_label != target.observed_semantic_label
        ):
            raise SemanticAtlasIntegrityError(
                "replica support vote differs from observed label"
            )
        if (
            replica.verdict == "oppose"
            and replica.semantic_label == target.observed_semantic_label
        ):
            raise SemanticAtlasIntegrityError(
                "replica opposition repeats the observed label"
            )

    def _validate_promotion_snapshot(
        self,
        promotion: SemanticLabelPromotion,
        *,
        measurements: Mapping[str, MeasurementReceipt],
        replicas: Mapping[str, ReplicaReceipt],
    ) -> None:
        measurement = measurements.get(promotion.measurement_sha256)
        if measurement is None:
            raise SemanticAtlasIntegrityError(
                "promotion references an inactive measurement"
            )
        if (
            measurement.observation_status != "eligible"
            or measurement.observed_semantic_label != promotion.semantic_label
            or measurement.model_pin != promotion.model_pin
            or measurement.coordinate != promotion.coordinate
            or measurement.probe != promotion.probe
            or measurement.runtime != promotion.runtime
        ):
            raise SemanticAtlasIntegrityError(
                "promotion differs from its eligible observation"
            )
        policy = promotion.policy
        effect = next(
            (
                row
                for row in measurement.placebo_effects
                if row.sha256 == promotion.placebo_effect_sha256
            ),
            None,
        )
        if effect is None or not self._effect_passes(effect, policy):
            raise SemanticAtlasIntegrityError(
                "promotion placebo/effect evidence does not pass policy"
            )
        vote_groups = (
            ("support", promotion.support_replica_sha256s),
            ("oppose", promotion.oppose_replica_sha256s),
            ("inconclusive", promotion.inconclusive_replica_sha256s),
        )
        selected: list[ReplicaReceipt] = []
        groups: dict[str, tuple[ReplicaReceipt, ...]] = {}
        for verdict, receipt_shas in vote_groups:
            rows: list[ReplicaReceipt] = []
            for receipt_sha in receipt_shas:
                replica = replicas.get(receipt_sha)
                if replica is None or replica.measurement_sha256 != measurement.sha256:
                    raise SemanticAtlasIntegrityError(
                        "promotion references an inactive/wrong replica"
                    )
                self._validate_replica_target(replica, measurement)
                if replica.verdict != verdict:
                    raise SemanticAtlasIntegrityError(
                        "promotion categorizes a replica under the wrong verdict"
                    )
                rows.append(replica)
                selected.append(replica)
            groups[verdict] = tuple(rows)
        if len({row.replica_id_sha256 for row in selected}) != len(selected):
            raise SemanticAtlasIntegrityError(
                "promotion counts one replica identity more than once"
            )
        support = groups["support"]
        oppose = groups["oppose"]
        support_weight = math.fsum(row.reputation_weight for row in support)
        oppose_weight = math.fsum(row.reputation_weight for row in oppose)
        if (
            len(selected) < policy.minimum_replica_count
            or len(support) < policy.minimum_support_count
            or support_weight < policy.minimum_support_weight
            or promotion.support_count != len(support)
            or promotion.oppose_count != len(oppose)
            or not math.isclose(
                promotion.support_weight,
                support_weight,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                promotion.oppose_weight,
                oppose_weight,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise SemanticAtlasIntegrityError(
                "promotion quorum/reputation snapshot does not satisfy policy"
            )
        if policy.require_strict_count_majority and len(support) <= len(oppose):
            raise SemanticAtlasIntegrityError(
                "promotion snapshot lacks strict count majority"
            )
        if policy.require_strict_weight_majority and support_weight <= oppose_weight:
            raise SemanticAtlasIntegrityError(
                "promotion snapshot lacks strict weighted majority"
            )

    def _validate_relations(self, primaries: Mapping[str, AtlasDocument]) -> None:
        measurements = self._measurement_map(primaries)
        replicas = {
            value.sha256: value
            for value in primaries.values()
            if isinstance(value, ReplicaReceipt)
        }
        promotions = tuple(
            value
            for value in primaries.values()
            if isinstance(value, SemanticLabelPromotion)
        )
        for measurement in measurements.values():
            self._validate_measurement_effects_from(measurement, measurements)
        for replica in replicas.values():
            target = measurements.get(replica.measurement_sha256)
            if target is None:
                raise SemanticAtlasIntegrityError(
                    "replica references an inactive measurement"
                )
            self._validate_replica_target(replica, target)
        for promotion in promotions:
            self._validate_promotion_snapshot(
                promotion,
                measurements=measurements,
                replicas=replicas,
            )

    def verify_or_raise(self) -> bool:
        if not self.graph.store.verify(include_tombstones=True):
            raise SemanticAtlasIntegrityError("LiveGraph segment verification failed")
        self._collect_documents()
        return True

    def verify(self) -> bool:
        try:
            self.verify_or_raise()
        except (SemanticAtlasError, RuntimeError):
            return False
        return True

    def revision(self) -> GraphRevision:
        """Return the mutable semantic-atlas head, separate from weight rails."""

        return GraphRevision.from_live_revision(self.graph.store.revision())

    def drop_segments(self, shas: Iterable[str]) -> tuple[str, ...]:
        """Deactivate segments without orphaning active semantic evidence."""

        try:
            requested = tuple(dict.fromkeys(shas))
        except TypeError as exc:
            raise SemanticAtlasError("segment SHAs must be iterable") from exc
        active = set(self.graph.store.segments())
        selected = active.intersection(requested)
        if not selected:
            return self.graph.drop_segments(requested)
        remaining: dict[str, AtlasDocument] = {}
        for segment_sha in sorted(active - selected):
            for _sha_value, _index, record in self.graph.store.iter_records(
                segment_sha
            ):
                if (
                    record.get("schema") == ATLAS_EDGE_SCHEMA
                    and record.get("edge_kind") == "primary"
                ):
                    value = self._validate_edge(record)
                    assert value is not None
                    remaining[_record_key(self._kind(value), value.sha256)] = value
        # Reject before changing the manifest.  Callers can drop the complete
        # dependent segment set in one operation and keep tombstones reusable.
        self._validate_relations(remaining)
        return self.graph.drop_segments(requested)

    def _append(self, document: AtlasDocument) -> AtlasAppendReceipt:
        self._validate_pin_coordinate(document.model_pin, document.coordinate)
        before = set(self.graph.store.segments())
        expected_records = self._edge_records(document)
        segment_sha = self.graph.append_segment(expected_records)
        actual_records = tuple(
            record
            for _sha_value, _index, record in self.graph.store.iter_records(segment_sha)
        )
        if actual_records != expected_records:
            raise SemanticAtlasIntegrityError(
                "LiveGraph append did not return the atomic atlas segment"
            )
        return AtlasAppendReceipt(
            record_kind=self._kind(document),
            record_sha256=document.sha256,
            segment_sha256=segment_sha,
            appended=segment_sha not in before,
        )

    def _primary(self, kind: str, sha256: str) -> AtlasDocument:
        key = _record_key(kind, sha256)
        edges = self.graph.query_base(key)
        values: dict[str, AtlasDocument] = {}
        for edge in edges:
            for record in self.graph.resolve_derivation(edge["derivation"]):
                value = self._validate_edge(record)
                if value is None:
                    continue
                values[value.sha256] = value
        if not values:
            raise SemanticAtlasError(f"{kind} receipt is not active: {sha256}")
        if set(values) != {sha256}:
            raise SemanticAtlasConflictError("record key resolved a content collision")
        return values[sha256]

    def _query_index(self, key: str) -> tuple[AtlasDocument, ...]:
        documents: dict[tuple[str, str], AtlasDocument] = {}
        for edge in self.graph.query_base(key):
            for record in self.graph.resolve_derivation(edge["derivation"]):
                if record.get("schema") != ATLAS_EDGE_SCHEMA:
                    continue
                if record.get("edge_kind") != "index":
                    raise SemanticAtlasIntegrityError(
                        "query index returned a primary edge"
                    )
                kind = record.get("record_kind")
                sha256 = _sha(record.get("document_sha256"), "document_sha256")
                value = self._primary(str(kind), sha256)
                self._validate_edge(
                    record,
                    primaries={_record_key(str(kind), sha256): value},
                )
                documents[(str(kind), sha256)] = value
        return tuple(documents[key] for key in sorted(documents))

    def _validate_effect_references(self, receipt: MeasurementReceipt) -> None:
        summaries = {row.metric: row for row in receipt.numeric_summaries}
        for effect in receipt.placebo_effects:
            placebo_value = self._primary(
                "measurement", effect.placebo_measurement_sha256
            )
            if not isinstance(placebo_value, MeasurementReceipt):
                raise SemanticAtlasIntegrityError(
                    "placebo reference is not a measurement"
                )
            placebo = placebo_value
            if (
                placebo.model_pin != receipt.model_pin
                or placebo.coordinate != receipt.coordinate
                or placebo.probe != receipt.probe
                or placebo.intervention.mode != "placebo"
            ):
                raise SemanticAtlasError(
                    "effect placebo identity does not match observation"
                )
            placebo_summaries = {row.metric: row for row in placebo.numeric_summaries}
            if effect.metric not in summaries or effect.metric not in placebo_summaries:
                raise SemanticAtlasError(
                    "effect metric is absent from observation/placebo"
                )
            if not math.isclose(
                placebo_summaries[effect.metric].mean,
                effect.placebo_mean,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise SemanticAtlasError("effect placebo mean is unauthenticated")

    def append_measurement(self, receipt: MeasurementReceipt) -> AtlasAppendReceipt:
        if not isinstance(receipt, MeasurementReceipt):
            raise TypeError("receipt must be a MeasurementReceipt")
        self._validate_pin_coordinate(receipt.model_pin, receipt.coordinate)
        self._validate_effect_references(receipt)
        return self._append(receipt)

    def append_replica(self, receipt: ReplicaReceipt) -> AtlasAppendReceipt:
        if not isinstance(receipt, ReplicaReceipt):
            raise TypeError("receipt must be a ReplicaReceipt")
        self._validate_pin_coordinate(receipt.model_pin, receipt.coordinate)
        target_value = self._primary("measurement", receipt.measurement_sha256)
        if not isinstance(target_value, MeasurementReceipt):
            raise SemanticAtlasIntegrityError("replica target is not a measurement")
        target = target_value
        if (
            receipt.model_pin != target.model_pin
            or receipt.coordinate != target.coordinate
            or receipt.probe_sha256 != target.probe.sha256
            or receipt.intervention_sha256 != target.intervention.sha256
        ):
            raise SemanticAtlasError("replica identity differs from target measurement")
        if (
            receipt.verdict == "support"
            and receipt.semantic_label != target.observed_semantic_label
        ):
            raise SemanticAtlasError("support vote does not name the observed label")
        if (
            receipt.verdict == "oppose"
            and receipt.semantic_label == target.observed_semantic_label
        ):
            raise SemanticAtlasError("oppose vote repeats the observed label")
        return self._append(receipt)

    def _replicas_for_measurement(
        self, measurement_sha256: str
    ) -> tuple[ReplicaReceipt, ...]:
        key = f"{_PREFIX}:measurement-replicas:{_sha(measurement_sha256, 'measurement_sha256')}"
        values = self._query_index(key)
        replicas = tuple(value for value in values if isinstance(value, ReplicaReceipt))
        return tuple(sorted(replicas, key=lambda row: row.sha256))

    @staticmethod
    def _effect_passes(effect: PlaceboEffect, policy: EvidencePolicy) -> bool:
        if effect.metric != policy.effect_metric:
            return False
        if policy.effect_direction == "positive":
            value = effect.delta
        elif policy.effect_direction == "negative":
            value = -effect.delta
        else:
            value = abs(effect.delta)
        return value >= policy.minimum_effect

    def promote_label(
        self,
        measurement_sha256: str,
        *,
        policy: EvidencePolicy,
    ) -> AtlasAppendReceipt:
        """Promote one observed label only after effect and quorum gates pass."""

        if not isinstance(policy, EvidencePolicy):
            raise TypeError("policy must be an EvidencePolicy")
        value = self._primary("measurement", measurement_sha256)
        if not isinstance(value, MeasurementReceipt):
            raise SemanticAtlasIntegrityError("promotion target is not a measurement")
        measurement = value
        if measurement.observation_status != "eligible":
            raise SemanticAtlasPromotionError("observation is not promotion-eligible")
        label = measurement.observed_semantic_label
        if label is None:  # protected by MeasurementReceipt, kept fail-closed.
            raise SemanticAtlasPromotionError("observation has no semantic label")
        effects = [
            effect
            for effect in measurement.placebo_effects
            if self._effect_passes(effect, policy)
        ]
        if policy.require_placebo and len(effects) != 1:
            raise SemanticAtlasPromotionError("placebo/effect gate did not pass")
        if not effects:
            raise SemanticAtlasPromotionError("effect threshold did not pass")
        effect = effects[0]
        self._validate_effect_references(measurement)

        replicas = self._replicas_for_measurement(measurement.sha256)
        by_replica: dict[str, ReplicaReceipt] = {}
        for replica in replicas:
            prior = by_replica.get(replica.replica_id_sha256)
            if prior is not None and prior.sha256 != replica.sha256:
                raise SemanticAtlasPromotionError(
                    "one replica identity submitted conflicting receipts"
                )
            by_replica[replica.replica_id_sha256] = replica
        distinct = tuple(by_replica.values())
        if len(distinct) < policy.minimum_replica_count:
            raise SemanticAtlasPromotionError("replica quorum is incomplete")
        support = tuple(
            row
            for row in distinct
            if row.verdict == "support" and row.semantic_label == label
        )
        oppose = tuple(row for row in distinct if row.verdict == "oppose")
        inconclusive = tuple(row for row in distinct if row.verdict == "inconclusive")
        support_weight = math.fsum(row.reputation_weight for row in support)
        oppose_weight = math.fsum(row.reputation_weight for row in oppose)
        if len(support) < policy.minimum_support_count:
            raise SemanticAtlasPromotionError("support count is below policy")
        if support_weight < policy.minimum_support_weight:
            raise SemanticAtlasPromotionError(
                "support reputation weight is below policy"
            )
        if policy.require_strict_count_majority and len(support) <= len(oppose):
            raise SemanticAtlasPromotionError("support lacks a strict count majority")
        if policy.require_strict_weight_majority and support_weight <= oppose_weight:
            raise SemanticAtlasPromotionError(
                "support lacks a strict weighted majority"
            )

        promotion = SemanticLabelPromotion(
            model_pin=measurement.model_pin,
            coordinate=measurement.coordinate,
            probe=measurement.probe,
            measurement_sha256=measurement.sha256,
            semantic_label=label,
            policy=policy,
            support_replica_sha256s=tuple(row.sha256 for row in support),
            oppose_replica_sha256s=tuple(row.sha256 for row in oppose),
            inconclusive_replica_sha256s=tuple(row.sha256 for row in inconclusive),
            support_count=len(support),
            oppose_count=len(oppose),
            support_weight=support_weight,
            oppose_weight=oppose_weight,
            placebo_effect_sha256=effect.sha256,
            runtime=measurement.runtime,
        )
        return self._append(promotion)

    def _query_result(
        self,
        measurements: Sequence[MeasurementReceipt],
        promotions: Sequence[SemanticLabelPromotion],
        replicas: Sequence[ReplicaReceipt] = (),
    ) -> AtlasQueryResult:
        by_replica = {row.sha256: row for row in replicas}
        for measurement in measurements:
            for replica in self._replicas_for_measurement(measurement.sha256):
                by_replica[replica.sha256] = replica
        return AtlasQueryResult(
            measurements=tuple(sorted(set(measurements), key=lambda row: row.sha256)),
            replicas=tuple(sorted(by_replica.values(), key=lambda row: row.sha256)),
            promotions=tuple(sorted(set(promotions), key=lambda row: row.sha256)),
        )

    def query_by_prompt_signature(self, prompt_signature: str) -> AtlasQueryResult:
        signature = _sha(prompt_signature, "prompt_signature")
        measurements = tuple(
            value
            for value in self._query_index(f"{_PREFIX}:prompt:measurement:{signature}")
            if isinstance(value, MeasurementReceipt)
        )
        promotions = tuple(
            value
            for value in self._query_index(f"{_PREFIX}:prompt:promotion:{signature}")
            if isinstance(value, SemanticLabelPromotion)
        )
        return self._query_result(measurements, promotions)

    def query_by_semantic_label(self, semantic_label: str) -> AtlasQueryResult:
        label = _text(semantic_label, "semantic_label", maximum=512)
        measurements = tuple(
            value
            for value in self._query_index(_label_key(label, verified=False))
            if isinstance(value, MeasurementReceipt)
        )
        promotions = tuple(
            value
            for value in self._query_index(_label_key(label, verified=True))
            if isinstance(value, SemanticLabelPromotion)
        )
        return self._query_result(measurements, promotions)

    def query_by_coordinate(
        self, coordinate: WeightCoordinate | str
    ) -> AtlasQueryResult:
        if isinstance(coordinate, WeightCoordinate):
            self._validate_pin_coordinate(self.model_pin, coordinate)
            coordinate_sha = coordinate.sha256
        else:
            coordinate_sha = _sha(coordinate, "coordinate_sha256")
        collections = {
            kind: self._query_index(f"{_PREFIX}:coordinate:{kind}:{coordinate_sha}")
            for kind in ("measurement", "replica", "promotion")
        }
        return self._query_result(
            tuple(
                value
                for value in collections["measurement"]
                if isinstance(value, MeasurementReceipt)
            ),
            tuple(
                value
                for value in collections["promotion"]
                if isinstance(value, SemanticLabelPromotion)
            ),
            tuple(
                value
                for value in collections["replica"]
                if isinstance(value, ReplicaReceipt)
            ),
        )

    def coverage_matrix(self) -> dict[str, Any]:
        """Return a deterministic coordinate-by-label observation/verification matrix."""

        documents = tuple(self._collect_documents().values())
        measurements = tuple(
            value for value in documents if isinstance(value, MeasurementReceipt)
        )
        promotions = tuple(
            value for value in documents if isinstance(value, SemanticLabelPromotion)
        )
        labels = tuple(
            sorted(
                {
                    row.observed_semantic_label
                    for row in measurements
                    if row.observed_semantic_label is not None
                }
                | {row.semantic_label for row in promotions}
            )
        )
        coordinates: dict[str, WeightCoordinate] = {}
        for row in (*measurements, *promotions):
            coordinates[row.coordinate.sha256] = row.coordinate
        row_keys = tuple(sorted(coordinates))
        observed = [[0 for _label in labels] for _coordinate in row_keys]
        verified = [[0 for _label in labels] for _coordinate in row_keys]
        row_index = {value: index for index, value in enumerate(row_keys)}
        label_index = {value: index for index, value in enumerate(labels)}
        for row in measurements:
            if row.observed_semantic_label is not None:
                observed[row_index[row.coordinate.sha256]][
                    label_index[row.observed_semantic_label]
                ] += 1
        for row in promotions:
            verified[row_index[row.coordinate.sha256]][
                label_index[row.semantic_label]
            ] += 1
        body = {
            "columns": list(labels),
            "measurement_count": len(measurements),
            "model_pin_sha256": self.model_pin.sha256,
            "observed_counts": observed,
            "promotion_count": len(promotions),
            "rows": [
                {
                    "coordinate": coordinates[key].to_document(),
                    "coordinate_sha256": key,
                }
                for key in row_keys
            ],
            "security_claim": "none",
            "verified_counts": verified,
        }
        return _seal(COVERAGE_MATRIX_SCHEMA, body)

    def consensus_by_coordinate(
        self, coordinate: WeightCoordinate | str
    ) -> dict[str, Any]:
        """Return every verified claim; conflicts stay visible and unordered."""

        result = self.query_by_coordinate(coordinate)
        if isinstance(coordinate, WeightCoordinate):
            coordinate_sha = coordinate.sha256
        else:
            coordinate_sha = _sha(coordinate, "coordinate_sha256")
        grouped: dict[str, list[SemanticLabelPromotion]] = {}
        for promotion in result.promotions:
            grouped.setdefault(promotion.semantic_label, []).append(promotion)
        claims = []
        for label, promotions in sorted(grouped.items()):
            claims.append(
                {
                    "promotion_sha256s": sorted(row.sha256 for row in promotions),
                    "semantic_label": label,
                    "snapshot_count": len(promotions),
                    "support_count_total": sum(row.support_count for row in promotions),
                    "support_weight_total": math.fsum(
                        row.support_weight for row in promotions
                    ),
                }
            )
        return _seal(
            CONSENSUS_SNAPSHOT_SCHEMA,
            {
                "claims": claims,
                "conflict": len(claims) > 1,
                "coordinate_sha256": coordinate_sha,
                "model_pin_sha256": self.model_pin.sha256,
                "security_claim": "none",
            },
        )


__all__ = [
    "ATLAS_EDGE_SCHEMA",
    "AtlasAppendReceipt",
    "AtlasQueryResult",
    "COVERAGE_MATRIX_SCHEMA",
    "CONSENSUS_SNAPSHOT_SCHEMA",
    "EVIDENCE_POLICY_SCHEMA",
    "EvidencePolicy",
    "GRAPH_REVISION_SCHEMA",
    "GraphRevision",
    "INTERVENTION_IDENTITY_SCHEMA",
    "InterventionIdentity",
    "LABEL_PROMOTION_SCHEMA",
    "MEASUREMENT_RECEIPT_SCHEMA",
    "MODEL_PIN_SCHEMA",
    "MeasurementReceipt",
    "ModelPin",
    "NumericSummary",
    "PROBE_IDENTITY_SCHEMA",
    "PlaceboEffect",
    "ProbeIdentity",
    "REPLICA_RECEIPT_SCHEMA",
    "RUNTIME_PROVENANCE_SCHEMA",
    "ReplicaReceipt",
    "RuntimeProvenance",
    "SemanticAtlasConflictError",
    "SemanticAtlasError",
    "SemanticAtlasIntegrityError",
    "SemanticAtlasPromotionError",
    "SemanticAtlasStaleCheckpointError",
    "SemanticLabelPromotion",
    "SemanticWeightAtlas",
    "WEIGHT_COORDINATE_SCHEMA",
    "WeightCoordinate",
]
