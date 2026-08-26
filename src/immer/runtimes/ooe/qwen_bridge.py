"""Authenticated Qwen/O1 measurements as bounded OoE feature receipts.

The bridge contains no text embedding shortcut.  It derives a fixed-size
numeric sketch from the sufficient statistics already emitted by the exact
Qwen cartography probe.  Every sketch remains bound to the immutable model,
``.causal`` weight coordinate, both graph heads, probe, verifier evidence,
and the O1 surprise/learning-progress signal that selected the measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from immer.runtimes.qwen3_8.semantic_atlas import (
    GraphRevision,
    MeasurementReceipt,
    ProbeIdentity,
)

from .identity import OoeSiteIdentity, canonical_json_bytes, require_sha256


OOE_ACTIONS = (
    "restore_anchor",
    "execute_fertig",
    "mount_organ",
    "probe_coordinate",
    "qwen_fallback",
)
OoeAction = Literal[
    "restore_anchor",
    "execute_fertig",
    "mount_organ",
    "probe_coordinate",
    "qwen_fallback",
]

FEATURE_RECEIPT_SCHEMA = "immer-ooe-qwen-feature-receipt/v1"
FEATURE_HASH_SCHEMA = "immer-ooe-qwen-numeric-feature-hash/v1"
ACTION_SCHEMA = {
    "actions": list(OOE_ACTIONS),
    "schema": "immer-ooe-runtime-action-alphabet/v1",
}
ACTION_SCHEMA_SHA256 = hashlib.sha256(canonical_json_bytes(ACTION_SCHEMA)).hexdigest()
DEFAULT_FEATURE_DIMENSIONS = 64
MAX_FEATURE_DIMENSIONS = 1024
MAX_EVIDENCE_HASHES = 256
_UINT64_MAX = (1 << 64) - 1


class QwenOoeBridgeError(ValueError):
    """A measurement cannot be represented by the exact OoE bridge."""


class QwenOoeBridgeIntegrityError(QwenOoeBridgeError):
    """A sealed feature receipt or one of its bindings was modified."""


class QwenOoeBridgeStaleError(QwenOoeBridgeIntegrityError):
    """A feature receipt belongs to another model or graph head."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _finite_nonnegative(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QwenOoeBridgeError(f"{field} must be finite non-negative numeric data")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise QwenOoeBridgeError(f"{field} must be finite non-negative numeric data")
    return 0.0 if result == 0.0 else result


def _uint64(value: object, *, field: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= _UINT64_MAX
    ):
        raise QwenOoeBridgeError(f"{field} must be a uint64")
    return value


def _dimensions(value: object) -> int:
    result = _uint64(value, field="feature_dimensions")
    if not 8 <= result <= MAX_FEATURE_DIMENSIONS:
        raise QwenOoeBridgeError(
            f"feature_dimensions must lie in [8, {MAX_FEATURE_DIMENSIONS}]"
        )
    return result


def _hashes(
    values: Sequence[str],
    *,
    field: str,
    required: str | None = None,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise QwenOoeBridgeError(f"{field} must be a SHA-256 sequence")
    try:
        normalized = tuple(
            sorted({require_sha256(value, field=field) for value in values})
        )
    except TypeError as exc:
        raise QwenOoeBridgeError(f"{field} must be a SHA-256 sequence") from exc
    if not normalized or len(normalized) > MAX_EVIDENCE_HASHES:
        raise QwenOoeBridgeError(
            f"{field} must contain 1..{MAX_EVIDENCE_HASHES} unique hashes"
        )
    if required is not None and required not in normalized:
        raise QwenOoeBridgeError(f"{field} does not bind the required evidence")
    return normalized


def validate_action(action: str) -> OoeAction:
    if action not in OOE_ACTIONS:
        raise QwenOoeBridgeError(f"unsupported OoE runtime action: {action!r}")
    return action  # type: ignore[return-value]


def feature_schema(dimensions: int = DEFAULT_FEATURE_DIMENSIONS) -> dict[str, Any]:
    size = _dimensions(dimensions)
    return {
        "dimensions": size,
        "input": "MeasurementReceipt.numeric_summaries+placebo_effects+O1",
        "metric_identity": "sha256-only",
        "normalization": "signed-x/(1+abs(x));count=log1p(x)/(1+log1p(x))",
        "reduction": "signed-feature-hash-then-tanh",
        "schema": FEATURE_HASH_SCHEMA,
    }


def feature_schema_sha256(dimensions: int = DEFAULT_FEATURE_DIMENSIONS) -> str:
    return _digest(feature_schema(dimensions))


def _bounded(value: float) -> float:
    if not math.isfinite(value):
        raise QwenOoeBridgeError("measurement summary contains non-finite data")
    return float(value / (1.0 + abs(value)))


def _bounded_count(value: int) -> float:
    transformed = math.log1p(value)
    return transformed / (1.0 + transformed)


def _feature_slot(
    *,
    dimensions: int,
    domain: str,
    metric_sha256: str,
    statistic: str,
) -> tuple[int, float]:
    seed = canonical_json_bytes(
        {
            "domain": domain,
            "metric_sha256": metric_sha256,
            "schema": FEATURE_HASH_SCHEMA,
            "statistic": statistic,
        }
    )
    raw = hashlib.sha256(seed).digest()
    return int.from_bytes(raw[:8], "little") % dimensions, (-1.0 if raw[8] & 1 else 1.0)


def measurement_numeric_sketch(
    measurement: MeasurementReceipt,
    *,
    o1_surprise: float,
    o1_learning_progress: float,
    dimensions: int = DEFAULT_FEATURE_DIMENSIONS,
) -> tuple[float, ...]:
    """Hash only bounded numeric sufficient statistics into a fixed vector."""

    if not isinstance(measurement, MeasurementReceipt):
        raise TypeError("measurement must be a MeasurementReceipt")
    size = _dimensions(dimensions)
    surprise = _finite_nonnegative(o1_surprise, field="o1_surprise")
    progress = _finite_nonnegative(
        o1_learning_progress,
        field="o1_learning_progress",
    )
    sketch = np.zeros(size, dtype=np.float64)

    def add(domain: str, metric_sha256: str, statistic: str, value: float) -> None:
        index, sign = _feature_slot(
            dimensions=size,
            domain=domain,
            metric_sha256=metric_sha256,
            statistic=statistic,
        )
        sketch[index] += sign * value

    for summary in measurement.numeric_summaries:
        metric_sha256 = hashlib.sha256(summary.metric.encode("utf-8")).hexdigest()
        variance = max(
            0.0,
            summary.total_squares / summary.count - summary.mean * summary.mean,
        )
        values = (
            ("count", _bounded_count(summary.count)),
            ("mean", _bounded(summary.mean)),
            ("stddev", _bounded(math.sqrt(variance))),
            ("minimum", _bounded(summary.minimum)),
            ("maximum", _bounded(summary.maximum)),
        )
        for statistic, value in values:
            add("summary", metric_sha256, statistic, value)

    for effect in measurement.placebo_effects:
        metric_sha256 = hashlib.sha256(effect.metric.encode("utf-8")).hexdigest()
        for statistic, value in (
            ("observed_mean", effect.observed_mean),
            ("placebo_mean", effect.placebo_mean),
            ("delta", effect.delta),
        ):
            add("placebo-effect", metric_sha256, statistic, _bounded(value))

    o1_metric = hashlib.sha256(b"immer:o1-cartography-signal/v1").hexdigest()
    add("o1", o1_metric, "surprise", surprise / (1.0 + surprise))
    add("o1", o1_metric, "learning_progress", progress / (1.0 + progress))

    # Preserve collision information while making every coordinate bounded.
    sketch = np.tanh(sketch)
    return tuple(0.0 if value == 0.0 else float(value) for value in sketch)


@dataclass(frozen=True, slots=True)
class QwenOoeFeatureReceipt:
    """Canonical, hash-sealed bridge from one real Qwen measurement to OoE."""

    temporal_index: int
    measurement_sha256: str
    model_pin_sha256: str
    weight_coordinate_sha256: str
    weight_graph_revision: GraphRevision
    atlas_graph_revision: GraphRevision
    probe: ProbeIdentity
    feature_schema_sha256: str
    action_schema_sha256: str
    verifier_sha256s: tuple[str, ...]
    evidence_sha256s: tuple[str, ...]
    o1_surprise: float
    o1_learning_progress: float
    feature_sketch: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "temporal_index",
            _uint64(self.temporal_index, field="temporal_index"),
        )
        for field in (
            "measurement_sha256",
            "model_pin_sha256",
            "weight_coordinate_sha256",
            "feature_schema_sha256",
            "action_schema_sha256",
        ):
            object.__setattr__(
                self,
                field,
                require_sha256(getattr(self, field), field=field),
            )
        if not isinstance(self.weight_graph_revision, GraphRevision):
            raise TypeError("weight_graph_revision must be a GraphRevision")
        if not isinstance(self.atlas_graph_revision, GraphRevision):
            raise TypeError("atlas_graph_revision must be a GraphRevision")
        if not isinstance(self.probe, ProbeIdentity):
            raise TypeError("probe must be a ProbeIdentity")
        object.__setattr__(
            self,
            "verifier_sha256s",
            _hashes(self.verifier_sha256s, field="verifier_sha256s"),
        )
        object.__setattr__(
            self,
            "evidence_sha256s",
            _hashes(self.evidence_sha256s, field="evidence_sha256s"),
        )
        object.__setattr__(
            self,
            "o1_surprise",
            _finite_nonnegative(self.o1_surprise, field="o1_surprise"),
        )
        object.__setattr__(
            self,
            "o1_learning_progress",
            _finite_nonnegative(
                self.o1_learning_progress,
                field="o1_learning_progress",
            ),
        )
        try:
            sketch = tuple(float(value) for value in self.feature_sketch)
        except (TypeError, ValueError, OverflowError) as exc:
            raise QwenOoeBridgeError("feature_sketch must be numeric") from exc
        size = _dimensions(len(sketch))
        if any(not math.isfinite(value) or abs(value) > 1.0 for value in sketch):
            raise QwenOoeBridgeError("feature_sketch must contain bounded finite data")
        if self.feature_schema_sha256 != feature_schema_sha256(size):
            raise QwenOoeBridgeIntegrityError(
                "feature schema does not match sketch size"
            )
        if self.action_schema_sha256 != ACTION_SCHEMA_SHA256:
            raise QwenOoeBridgeIntegrityError(
                "action schema is not the runtime alphabet"
            )
        object.__setattr__(
            self,
            "feature_sketch",
            tuple(0.0 if value == 0.0 else value for value in sketch),
        )

    @classmethod
    def from_measurement(
        cls,
        measurement: MeasurementReceipt,
        *,
        temporal_index: int,
        verifier_sha256s: Sequence[str],
        evidence_sha256s: Sequence[str] = (),
        o1_surprise: float,
        o1_learning_progress: float,
        dimensions: int = DEFAULT_FEATURE_DIMENSIONS,
    ) -> "QwenOoeFeatureReceipt":
        if not isinstance(measurement, MeasurementReceipt):
            raise TypeError("measurement must be a MeasurementReceipt")
        evidence = tuple(evidence_sha256s) + (
            measurement.evidence_sha256,
            measurement.access_trace_sha256,
        )
        return cls(
            temporal_index=temporal_index,
            measurement_sha256=measurement.sha256,
            model_pin_sha256=measurement.model_pin.sha256,
            weight_coordinate_sha256=measurement.coordinate.sha256,
            weight_graph_revision=measurement.weight_rail_revision,
            atlas_graph_revision=measurement.atlas_head_revision,
            probe=measurement.probe,
            feature_schema_sha256=feature_schema_sha256(dimensions),
            action_schema_sha256=ACTION_SCHEMA_SHA256,
            verifier_sha256s=tuple(verifier_sha256s),
            evidence_sha256s=evidence,
            o1_surprise=o1_surprise,
            o1_learning_progress=o1_learning_progress,
            feature_sketch=measurement_numeric_sketch(
                measurement,
                o1_surprise=o1_surprise,
                o1_learning_progress=o1_learning_progress,
                dimensions=dimensions,
            ),
        )

    @property
    def feature_dimensions(self) -> int:
        return len(self.feature_sketch)

    @property
    def weight_graph_revision_sha256(self) -> str:
        return self.weight_graph_revision.sha256

    @property
    def atlas_graph_revision_sha256(self) -> str:
        return self.atlas_graph_revision.sha256

    @property
    def site_identity(self) -> OoeSiteIdentity:
        return OoeSiteIdentity(
            model_pin_sha256=self.model_pin_sha256,
            weight_coordinate_sha256=self.weight_coordinate_sha256,
            graph_revision_sha256=self.weight_graph_revision.sha256,
            feature_schema_sha256=self.feature_schema_sha256,
            action_schema_sha256=self.action_schema_sha256,
        )

    @property
    def sketch_array(self) -> NDArray[np.float64]:
        value = np.asarray(self.feature_sketch, dtype=np.float64)
        value.setflags(write=False)
        return value

    def as_record(self) -> dict[str, Any]:
        return {
            "action_schema_sha256": self.action_schema_sha256,
            "atlas_graph_revision": self.atlas_graph_revision.to_document(),
            "atlas_graph_revision_sha256": self.atlas_graph_revision.sha256,
            "evidence_sha256s": list(self.evidence_sha256s),
            "feature_dimensions": self.feature_dimensions,
            "feature_schema_sha256": self.feature_schema_sha256,
            "feature_sketch": list(self.feature_sketch),
            "measurement_sha256": self.measurement_sha256,
            "model_pin_sha256": self.model_pin_sha256,
            "o1_learning_progress": self.o1_learning_progress,
            "o1_surprise": self.o1_surprise,
            "probe": self.probe.to_document(),
            "probe_sha256": self.probe.sha256,
            "site_identity": self.site_identity.to_dict(),
            "site_identity_sha256": self.site_identity.sha256,
            "temporal_index": self.temporal_index,
            "verifier_sha256s": list(self.verifier_sha256s),
            "weight_coordinate_sha256": self.weight_coordinate_sha256,
            "weight_graph_revision": self.weight_graph_revision.to_document(),
            "weight_graph_revision_sha256": self.weight_graph_revision.sha256,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())

    def to_document(self) -> dict[str, Any]:
        body = self.as_record()
        return {
            "body": body,
            "schema": FEATURE_RECEIPT_SCHEMA,
            "sha256": _digest(body),
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "QwenOoeFeatureReceipt":
        if not isinstance(document, Mapping) or set(document) != {
            "body",
            "schema",
            "sha256",
        }:
            raise QwenOoeBridgeIntegrityError("feature receipt envelope is invalid")
        if document.get("schema") != FEATURE_RECEIPT_SCHEMA:
            raise QwenOoeBridgeIntegrityError("feature receipt schema is invalid")
        body = document.get("body")
        expected = {
            "action_schema_sha256",
            "atlas_graph_revision",
            "atlas_graph_revision_sha256",
            "evidence_sha256s",
            "feature_dimensions",
            "feature_schema_sha256",
            "feature_sketch",
            "measurement_sha256",
            "model_pin_sha256",
            "o1_learning_progress",
            "o1_surprise",
            "probe",
            "probe_sha256",
            "site_identity",
            "site_identity_sha256",
            "temporal_index",
            "verifier_sha256s",
            "weight_coordinate_sha256",
            "weight_graph_revision",
            "weight_graph_revision_sha256",
        }
        if not isinstance(body, Mapping) or set(body) != expected:
            raise QwenOoeBridgeIntegrityError("feature receipt body is invalid")
        claimed = require_sha256(document.get("sha256"), field="sha256")
        if claimed != _digest(body):
            raise QwenOoeBridgeIntegrityError("feature receipt SHA-256 mismatch")
        normalized = json.loads(canonical_json_bytes(body))
        probe = ProbeIdentity.from_document(normalized["probe"])
        weight_revision = GraphRevision.from_document(
            normalized["weight_graph_revision"]
        )
        atlas_revision = GraphRevision.from_document(normalized["atlas_graph_revision"])
        site_identity = OoeSiteIdentity.from_dict(normalized["site_identity"])
        if normalized["feature_dimensions"] != len(normalized["feature_sketch"]):
            raise QwenOoeBridgeIntegrityError("feature dimension binding mismatch")
        if normalized["probe_sha256"] != probe.sha256:
            raise QwenOoeBridgeIntegrityError("probe binding mismatch")
        if normalized["weight_graph_revision_sha256"] != weight_revision.sha256:
            raise QwenOoeBridgeIntegrityError("weight graph revision binding mismatch")
        if normalized["atlas_graph_revision_sha256"] != atlas_revision.sha256:
            raise QwenOoeBridgeIntegrityError("atlas graph revision binding mismatch")
        receipt = cls(
            temporal_index=normalized["temporal_index"],
            measurement_sha256=normalized["measurement_sha256"],
            model_pin_sha256=normalized["model_pin_sha256"],
            weight_coordinate_sha256=normalized["weight_coordinate_sha256"],
            weight_graph_revision=weight_revision,
            atlas_graph_revision=atlas_revision,
            probe=probe,
            feature_schema_sha256=normalized["feature_schema_sha256"],
            action_schema_sha256=normalized["action_schema_sha256"],
            verifier_sha256s=tuple(normalized["verifier_sha256s"]),
            evidence_sha256s=tuple(normalized["evidence_sha256s"]),
            o1_surprise=normalized["o1_surprise"],
            o1_learning_progress=normalized["o1_learning_progress"],
            feature_sketch=tuple(normalized["feature_sketch"]),
        )
        if site_identity != receipt.site_identity:
            raise QwenOoeBridgeIntegrityError("site identity binding mismatch")
        if normalized["site_identity_sha256"] != receipt.site_identity.sha256:
            raise QwenOoeBridgeIntegrityError("site identity SHA-256 mismatch")
        if receipt.sha256 != claimed:
            raise QwenOoeBridgeIntegrityError("feature receipt reconstruction mismatch")
        return receipt

    def validate_measurement(self, measurement: MeasurementReceipt) -> None:
        if not isinstance(measurement, MeasurementReceipt):
            raise TypeError("measurement must be a MeasurementReceipt")
        expected = QwenOoeFeatureReceipt.from_measurement(
            measurement,
            temporal_index=self.temporal_index,
            verifier_sha256s=self.verifier_sha256s,
            evidence_sha256s=tuple(
                value
                for value in self.evidence_sha256s
                if value
                not in (measurement.evidence_sha256, measurement.access_trace_sha256)
            ),
            o1_surprise=self.o1_surprise,
            o1_learning_progress=self.o1_learning_progress,
            dimensions=self.feature_dimensions,
        )
        if expected != self:
            raise QwenOoeBridgeIntegrityError(
                "feature receipt does not match MeasurementReceipt"
            )

    def assert_current(
        self,
        *,
        model_pin_sha256: str,
        weight_graph_revision_sha256: str,
        atlas_graph_revision_sha256: str,
    ) -> None:
        expected = (
            require_sha256(model_pin_sha256, field="model_pin_sha256"),
            require_sha256(
                weight_graph_revision_sha256,
                field="weight_graph_revision_sha256",
            ),
            require_sha256(
                atlas_graph_revision_sha256,
                field="atlas_graph_revision_sha256",
            ),
        )
        actual = (
            self.model_pin_sha256,
            self.weight_graph_revision.sha256,
            self.atlas_graph_revision.sha256,
        )
        if actual != expected:
            raise QwenOoeBridgeStaleError("Qwen/OoE model or graph revision is stale")


__all__ = [
    "ACTION_SCHEMA",
    "ACTION_SCHEMA_SHA256",
    "DEFAULT_FEATURE_DIMENSIONS",
    "FEATURE_HASH_SCHEMA",
    "FEATURE_RECEIPT_SCHEMA",
    "MAX_FEATURE_DIMENSIONS",
    "OOE_ACTIONS",
    "OoeAction",
    "QwenOoeBridgeError",
    "QwenOoeBridgeIntegrityError",
    "QwenOoeBridgeStaleError",
    "QwenOoeFeatureReceipt",
    "feature_schema",
    "feature_schema_sha256",
    "measurement_numeric_sketch",
    "validate_action",
]
