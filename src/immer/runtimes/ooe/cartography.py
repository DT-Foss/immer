"""Direct O1/Atlas to OoE learning and Crystal-promotion bridge.

The cartography loop already owns the expensive Qwen probe and the authenticated
``MeasurementReceipt``.  This module turns that completed work into Markov-OoE
teacher transitions without another model forward, then promotes every site
whose explicitly recorded source coverage is executable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import threading
from typing import Any, Protocol

from immer.runtimes.qwen3_8.semantic_atlas import (
    AtlasQueryResult,
    GraphRevision,
    MeasurementReceipt,
)

from .controller import ActionExecution, OoeController, VerifiedTeacherTransition
from .crystal import CrystalPublication
from .identity import canonical_json_bytes, require_sha256
from .qwen_bridge import OoeAction, QwenOoeFeatureReceipt, validate_action


ATLAS_MEASUREMENT_VERIFIER_SCHEMA = "immer-ooe-atlas-measurement-verifier/v1"
ATLAS_PROBE_EXECUTOR_SCHEMA = "immer-ooe-atlas-probe-executor/v1"
ATLAS_PROBE_QUALITY_SCHEMA = "immer-ooe-atlas-probe-quality/v1"
CARTOGRAPHY_LEARNING_SCHEMA = "immer-ooe-cartography-learning/v1"


class _Atlas(Protocol):
    def verify_or_raise(self) -> None: ...

    def revision(self) -> GraphRevision: ...

    def query_by_prompt_signature(self, prompt_signature: str) -> AtlasQueryResult: ...


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def authenticated_measurement_verifier_sha256(
    atlas: _Atlas,
    measurement: MeasurementReceipt,
) -> str:
    """Force-authenticate Atlas state and bind one active measurement to its head."""

    if not isinstance(measurement, MeasurementReceipt):
        raise TypeError("measurement must be a MeasurementReceipt")
    verify = getattr(atlas, "verify_or_raise", None)
    query = getattr(atlas, "query_by_prompt_signature", None)
    revision = getattr(atlas, "revision", None)
    if not callable(verify) or not callable(query) or not callable(revision):
        raise TypeError("atlas must provide verify/query/revision")
    head_before = revision()
    if not isinstance(head_before, GraphRevision):
        raise TypeError("atlas revision must be a GraphRevision")
    verify()
    if revision() != head_before:
        raise RuntimeError("Atlas head changed while its journal was authenticated")
    result = query(measurement.probe.prompt_signature)
    if not isinstance(result, AtlasQueryResult):
        raise TypeError("atlas query must return AtlasQueryResult")
    active = {
        row.sha256: row
        for row in result.measurements
        if row.model_pin == measurement.model_pin
        and row.coordinate == measurement.coordinate
        and row.probe == measurement.probe
    }
    if active.get(measurement.sha256) != measurement:
        raise ValueError("measurement is not active in the authenticated Atlas")
    head = revision()
    if head != head_before:
        raise RuntimeError("Atlas head changed during the pinned measurement query")
    return _digest(
        {
            "active_measurement_sha256": measurement.sha256,
            "atlas_head_revision": head.to_document(),
            "measurement_atlas_revision": measurement.atlas_head_revision.to_document(),
            "model_pin_sha256": measurement.model_pin.sha256,
            "probe_sha256": measurement.probe.sha256,
            "schema": ATLAS_MEASUREMENT_VERIFIER_SCHEMA,
            "weight_graph_revision": measurement.weight_rail_revision.to_document(),
        }
    )


def atlas_probe_quality_sha256(
    receipt: QwenOoeFeatureReceipt,
    measurement: MeasurementReceipt,
    verifier_sha256: str,
) -> str:
    """Bind a zero-forward Atlas discharge to its exact feature and proof."""

    if not isinstance(receipt, QwenOoeFeatureReceipt):
        raise TypeError("receipt must be a QwenOoeFeatureReceipt")
    if not isinstance(measurement, MeasurementReceipt):
        raise TypeError("measurement must be a MeasurementReceipt")
    verifier = require_sha256(verifier_sha256, field="verifier_sha256")
    receipt.validate_measurement(measurement)
    if verifier not in receipt.verifier_sha256s:
        raise ValueError("Atlas verifier is not feature-bound")
    return _digest(
        {
            "feature_receipt_sha256": receipt.sha256,
            "measurement_sha256": measurement.sha256,
            "model_pin_sha256": measurement.model_pin.sha256,
            "schema": ATLAS_PROBE_QUALITY_SCHEMA,
            "verifier_sha256": verifier,
        }
    )


class AtlasProbeExecutor:
    """Discharge an already authenticated Atlas measurement with zero Qwen probes."""

    def __init__(self, atlas: _Atlas) -> None:
        for name in ("verify_or_raise", "revision", "query_by_prompt_signature"):
            if not callable(getattr(atlas, name, None)):
                raise TypeError("atlas must provide verify/query/revision")
        self.atlas = atlas

    def _measurement(self, receipt: QwenOoeFeatureReceipt) -> MeasurementReceipt:
        result = self.atlas.query_by_prompt_signature(receipt.probe.prompt_signature)
        if not isinstance(result, AtlasQueryResult):
            raise TypeError("atlas query must return AtlasQueryResult")
        matches = {
            measurement.sha256: measurement
            for measurement in result.measurements
            if isinstance(measurement, MeasurementReceipt)
            and measurement.model_pin.sha256 == receipt.model_pin_sha256
            and measurement.coordinate.sha256 == receipt.weight_coordinate_sha256
            and measurement.probe == receipt.probe
        }
        measurement = matches.get(receipt.measurement_sha256)
        if measurement is None:
            raise ValueError("feature measurement is not active in Atlas")
        receipt.validate_measurement(measurement)
        return measurement

    def __call__(self, receipt: QwenOoeFeatureReceipt) -> ActionExecution:
        if not isinstance(receipt, QwenOoeFeatureReceipt):
            raise TypeError("receipt must be a QwenOoeFeatureReceipt")
        measurement = self._measurement(receipt)
        verifier = authenticated_measurement_verifier_sha256(self.atlas, measurement)
        if verifier not in receipt.verifier_sha256s:
            raise ValueError("current Atlas proof is not feature-bound")
        quality = atlas_probe_quality_sha256(receipt, measurement, verifier)
        executor_sha256 = _digest(
            {
                "action": "probe_coordinate",
                "model_pin_sha256": measurement.model_pin.sha256,
                "schema": ATLAS_PROBE_EXECUTOR_SCHEMA,
                "weight_graph_revision_sha256": (
                    measurement.weight_rail_revision.sha256
                ),
            }
        )
        return ActionExecution(
            feature_receipt_sha256=receipt.sha256,
            action="probe_coordinate",
            executor_sha256=executor_sha256,
            verifier_sha256=verifier,
            evidence_sha256=measurement.evidence_sha256,
            quality_sha256=quality,
            result=measurement.to_document(),
            quality_verified=True,
            qwen_forwards=0,
            # One schedule row is one authenticated, complete cartography probe.
            # The cohort verifier re-derives this unit from the immutable
            # scheduler instead of trusting persisted execution counters.
            teacher_baseline_qwen_forwards=1,
        )


def verify_atlas_probe_execution(
    atlas: _Atlas,
    receipt: QwenOoeFeatureReceipt,
    execution: ActionExecution,
) -> bool:
    """Final exact verifier for a zero-forward Atlas probe discharge."""

    if not isinstance(receipt, QwenOoeFeatureReceipt) or not isinstance(
        execution, ActionExecution
    ):
        return False
    execution.assert_bound(receipt, "probe_coordinate")
    if not execution.quality_verified or execution.qwen_forwards != 0:
        return False
    try:
        measurement = MeasurementReceipt.from_document(execution.result)
        receipt.validate_measurement(measurement)
    except (TypeError, ValueError):
        return False
    verifier = authenticated_measurement_verifier_sha256(atlas, measurement)
    if execution.verifier_sha256 != verifier:
        return False
    return execution.quality_sha256 == atlas_probe_quality_sha256(
        receipt,
        measurement,
        verifier,
    )


@dataclass(frozen=True, slots=True)
class CartographyLearningReceipt:
    """One exact O1/Atlas observation ingested as a Markov teacher transition."""

    temporal_index: int
    measurement_sha256: str
    feature_receipt_sha256: str
    transition_sha256: str
    site_identity_sha256: str
    source_action: OoeAction | str
    target_action: OoeAction | str
    verifier_sha256: str
    evidence_sha256: str
    quality_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.temporal_index, bool)
            or not isinstance(self.temporal_index, int)
            or self.temporal_index < 0
        ):
            raise ValueError("temporal_index must be a non-negative integer")
        for field in (
            "measurement_sha256",
            "feature_receipt_sha256",
            "transition_sha256",
            "site_identity_sha256",
            "verifier_sha256",
            "evidence_sha256",
            "quality_sha256",
        ):
            object.__setattr__(
                self,
                field,
                require_sha256(getattr(self, field), field=field),
            )
        object.__setattr__(self, "source_action", validate_action(self.source_action))
        object.__setattr__(self, "target_action", validate_action(self.target_action))

    def as_record(self) -> dict[str, Any]:
        return {
            "evidence_sha256": self.evidence_sha256,
            "feature_receipt_sha256": self.feature_receipt_sha256,
            "measurement_sha256": self.measurement_sha256,
            "quality_sha256": self.quality_sha256,
            "schema": CARTOGRAPHY_LEARNING_SCHEMA,
            "site_identity_sha256": self.site_identity_sha256,
            "source_action": self.source_action,
            "target_action": self.target_action,
            "temporal_index": self.temporal_index,
            "transition_sha256": self.transition_sha256,
            "verifier_sha256": self.verifier_sha256,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())


class OoeCartographyBridge:
    """Consume completed Qwen/O1 measurements and maintain executable Crystals."""

    def __init__(self, controller: OoeController) -> None:
        if not isinstance(controller, OoeController):
            raise TypeError("controller must be an OoeController")
        self.controller = controller
        self._lock = threading.RLock()

    def ingest(
        self,
        measurement: MeasurementReceipt,
        *,
        source_action: OoeAction | str,
        target_action: OoeAction | str,
        verifier_sha256s: Sequence[str],
        o1_surprise: float,
        o1_learning_progress: float,
        evidence_sha256s: Sequence[str] = (),
        quality_sha256: str | None = None,
        weight: float = 1.0,
    ) -> CartographyLearningReceipt:
        """Ingest one already verified probe; this performs no Qwen forward."""

        if not isinstance(measurement, MeasurementReceipt):
            raise TypeError("measurement must be a MeasurementReceipt")
        source = validate_action(source_action)
        target = validate_action(target_action)
        verifiers = tuple(
            sorted(
                {
                    require_sha256(value, field="verifier_sha256s")
                    for value in verifier_sha256s
                }
            )
        )
        if not verifiers:
            raise ValueError("verifier_sha256s must not be empty")
        with self._lock:
            temporal_index = self.controller.last_temporal_index + 1
            feature = QwenOoeFeatureReceipt.from_measurement(
                measurement,
                temporal_index=temporal_index,
                verifier_sha256s=verifiers,
                evidence_sha256s=tuple(evidence_sha256s),
                o1_surprise=o1_surprise,
                o1_learning_progress=o1_learning_progress,
            )
            feature.validate_measurement(measurement)
            evidence = measurement.evidence_sha256
            quality = (
                _digest(
                    {
                        "feature_receipt_sha256": feature.sha256,
                        "measurement_sha256": measurement.sha256,
                        "schema": CARTOGRAPHY_LEARNING_SCHEMA,
                        "source_action": source,
                        "target_action": target,
                        "verifier_sha256": verifiers[0],
                    }
                )
                if quality_sha256 is None
                else require_sha256(quality_sha256, field="quality_sha256")
            )
            transition = VerifiedTeacherTransition(
                feature_receipt_sha256=feature.sha256,
                site_identity_sha256=feature.site_identity.sha256,
                source_action=source,
                target_action=target,
                verifier_sha256=verifiers[0],
                evidence_sha256=evidence,
                quality_sha256=quality,
                verified_quality=True,
                weight=weight,
            )
            self.controller.ingest_teacher(feature, transition)
            return CartographyLearningReceipt(
                temporal_index=temporal_index,
                measurement_sha256=measurement.sha256,
                feature_receipt_sha256=feature.sha256,
                transition_sha256=transition.sha256,
                site_identity_sha256=feature.site_identity.sha256,
                source_action=source,
                target_action=target,
                verifier_sha256=verifiers[0],
                evidence_sha256=evidence,
                quality_sha256=quality,
            )

    def ingest_authenticated(
        self,
        atlas: _Atlas,
        measurement: MeasurementReceipt,
        *,
        source_action: OoeAction | str,
        target_action: OoeAction | str,
        o1_surprise: float,
        o1_learning_progress: float,
        evidence_sha256s: Sequence[str] = (),
        weight: float = 1.0,
    ) -> CartographyLearningReceipt:
        """Authenticate the active Atlas record, then ingest its transition."""

        verifier = authenticated_measurement_verifier_sha256(atlas, measurement)
        return self.ingest(
            measurement,
            source_action=source_action,
            target_action=target_action,
            verifier_sha256s=(verifier,),
            evidence_sha256s=evidence_sha256s,
            o1_surprise=o1_surprise,
            o1_learning_progress=o1_learning_progress,
            weight=weight,
        )

    def promote_ready(self) -> tuple[CrystalPublication, ...]:
        """Promote all explicitly covered sites against one shared calibration."""

        with self._lock:
            ready: list[tuple[str, str, tuple[str, ...]]] = []
            for site_sha256 in self.controller.site_identity_sha256s:
                coverage = self.controller.coverage_receipt(site_sha256)
                covered = sum(
                    count >= self.controller.config.min_coverage_per_source
                    for count in coverage.per_source
                )
                if covered >= self.controller.config.min_promoted_sources:
                    ready.append(
                        (
                            site_sha256,
                            coverage.sha256,
                            coverage.verifier_sha256s,
                        )
                    )
            publications: list[CrystalPublication] = []
            for site_sha256, coverage_sha256, verifiers in ready:
                generation = self.controller.crystal_store.manifest().generation
                publication = self.controller.promote(
                    site_sha256,
                    coverage_sha256=coverage_sha256,
                    verifier_sha256s=verifiers,
                    expected_store_generation=generation,
                )
                publications.append(publication)
            return tuple(publications)


__all__ = [
    "ATLAS_MEASUREMENT_VERIFIER_SCHEMA",
    "ATLAS_PROBE_EXECUTOR_SCHEMA",
    "ATLAS_PROBE_QUALITY_SCHEMA",
    "AtlasProbeExecutor",
    "CARTOGRAPHY_LEARNING_SCHEMA",
    "CartographyLearningReceipt",
    "OoeCartographyBridge",
    "atlas_probe_quality_sha256",
    "authenticated_measurement_verifier_sha256",
    "verify_atlas_probe_execution",
]
