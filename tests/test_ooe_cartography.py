from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from immer.runtimes.deepseek_v4.causal_weights import TensorRangePlan
from immer.runtimes.ooe.cartography import (
    AtlasProbeExecutor,
    OoeCartographyBridge,
    authenticated_measurement_verifier_sha256,
    verify_atlas_probe_execution,
)
from immer.runtimes.ooe.controller import (
    ControllerConfig,
    OoeController,
    OoeControllerStaleError,
)
from immer.runtimes.ooe.crystal import CrystalStore
from immer.runtimes.ooe.qwen_bridge import QwenOoeFeatureReceipt
from immer.runtimes.qwen3_8.semantic_atlas import (
    AtlasQueryResult,
    GraphRevision,
    InterventionIdentity,
    MeasurementReceipt,
    ModelPin,
    NumericSummary,
    ProbeIdentity,
    RuntimeProvenance,
    WeightCoordinate,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


_PLAN = TensorRangePlan(
    name="model.layers.18.mlp.gate_proj.weight",
    dtype="BF16",
    shape=(8, 2),
    shard="model-00007-of-00018.safetensors",
    absolute_offset=4096,
    length=32,
)
_PIN = ModelPin(
    repo_id="Qwen/Qwen3.8-27B",
    revision="0123456789abcdef",
    bundle_fingerprint=_hash("bundle"),
    bundle_manifest_sha256=_hash("manifest"),
    code_revision="a" * 40,
)
_RUNTIME = RuntimeProvenance(
    code_revision="a" * 40,
    source_manifest_sha256=_hash("source"),
    dependency_manifest_sha256=_hash("deps"),
    runtime_configuration_sha256=_hash("config"),
    platform_sha256=_hash("platform"),
)
_WEIGHT_REVISION = GraphRevision(12, _hash("weight-12"))
_ATLAS_REVISION = GraphRevision(20, _hash("atlas-20"))


def _summary(metric: str, values: tuple[float, ...]) -> NumericSummary:
    return NumericSummary(
        metric=metric,
        count=len(values),
        total=sum(values),
        total_squares=sum(value * value for value in values),
        minimum=min(values),
        maximum=max(values),
    )


def _measurement(sample: int, atlas_revision: GraphRevision) -> MeasurementReceipt:
    coordinate = WeightCoordinate.from_plan(
        _PLAN,
        layer=18,
        module="model.layers.18.mlp.gate_proj",
        row_start=0,
        row_end=4,
    )
    signal = 2.0 + sample / 10.0
    return MeasurementReceipt(
        model_pin=_PIN,
        coordinate=coordinate,
        probe=ProbeIdentity(
            question_sha256=_hash(f"question-{sample}"),
            token_sha256=_hash(f"tokens-{sample}"),
            family_sha256=_hash("family"),
            label_source_sha256=_hash("labels"),
        ),
        intervention=InterventionIdentity(
            mode="native",
            configuration_sha256=_hash(f"native-{sample}"),
        ),
        observation_status="recorded",
        observed_semantic_label=None,
        hidden_sha256=_hash(f"hidden-{sample}"),
        activation_sha256=_hash(f"activation-{sample}"),
        logits_sha256=_hash(f"logits-{sample}"),
        state_sha256=_hash(f"state-{sample}"),
        access_trace_sha256=_hash(f"trace-{sample}"),
        evidence_sha256=_hash(f"evidence-{sample}"),
        weight_rail_revision=_WEIGHT_REVISION,
        atlas_head_revision=atlas_revision,
        numeric_summaries=(
            _summary("activation", (signal - 0.1, signal + 0.1)),
            _summary("hidden_rms", (signal + 0.5, signal + 0.75)),
        ),
        placebo_effects=(),
        runtime=_RUNTIME,
    )


class _Atlas:
    def __init__(
        self,
        measurements: tuple[MeasurementReceipt, ...],
        head: GraphRevision,
    ) -> None:
        self.measurements = measurements
        self.head = head
        self.verifications = 0

    def verify_or_raise(self) -> None:
        self.verifications += 1

    def revision(self) -> GraphRevision:
        return self.head

    def query_by_prompt_signature(self, prompt_signature: str) -> AtlasQueryResult:
        return AtlasQueryResult(
            measurements=tuple(
                row
                for row in self.measurements
                if row.probe.prompt_signature == prompt_signature
            ),
            replicas=(),
            promotions=(),
        )


class _MovingAtlas(_Atlas):
    def query_by_prompt_signature(self, prompt_signature: str) -> AtlasQueryResult:
        result = super().query_by_prompt_signature(prompt_signature)
        self.head = GraphRevision(self.head.sequence + 1, _hash("concurrent-append"))
        return result


class OoeCartographyBridgeTests(unittest.TestCase):
    def _controller(self, root: Path, *, action_executors=None) -> OoeController:
        return OoeController(
            model_pin_sha256=_PIN.sha256,
            weight_graph_revision_sha256=_WEIGHT_REVISION.sha256,
            atlas_graph_revision=_ATLAS_REVISION,
            crystal_store=CrystalStore(root),
            action_executors=action_executors,
            config=ControllerConfig(
                replicas=4,
                replica_fanout=4,
                min_coverage_per_source=1,
                min_promoted_sources=1,
                router_radius=1.0,
                router_min_margin=0.0,
                token_min_confidence=1e-9,
                consensus_tolerance=1e-7,
                consensus_max_rounds=4096,
                reservoir_size=8,
            ),
        )

    def test_authenticated_measurement_becomes_partial_executable_crystal(self) -> None:
        measurement = _measurement(0, _ATLAS_REVISION)
        atlas = _Atlas((measurement,), GraphRevision(21, _hash("atlas-21")))
        with tempfile.TemporaryDirectory() as tmp:
            controller = self._controller(Path(tmp))
            bridge = OoeCartographyBridge(controller)
            learning = bridge.ingest_authenticated(
                atlas,
                measurement,
                source_action="qwen_fallback",
                target_action="probe_coordinate",
                o1_surprise=0.25,
                o1_learning_progress=1.5,
            )

            self.assertEqual(atlas.verifications, 1)
            self.assertEqual(controller.last_temporal_index, 0)
            self.assertEqual(learning.measurement_sha256, measurement.sha256)
            coverage = controller.coverage_receipt(learning.site_identity_sha256)
            self.assertEqual(coverage.per_source[-1], 1)
            self.assertEqual(coverage.per_target[3], 1)
            publications = bridge.promote_ready()
            self.assertEqual(len(publications), 1)

            feature = QwenOoeFeatureReceipt.from_measurement(
                measurement,
                temporal_index=1,
                verifier_sha256s=(learning.verifier_sha256,),
                o1_surprise=0.25,
                o1_learning_progress=1.5,
            )
            warm = controller.decide(feature, "qwen_fallback", stream_id="warm")
            self.assertEqual(warm.origin, "crystal")
            self.assertEqual(warm.action, "probe_coordinate")
            uncovered = controller.decide(feature, "restore_anchor", stream_id="other")
            self.assertEqual(uncovered.origin, "abstention")
            self.assertEqual(uncovered.reason, "uncovered-source-action")

    def test_forward_atlas_append_is_accepted_and_rollback_is_rejected(self) -> None:
        first = _measurement(0, _ATLAS_REVISION)
        forward_revision = GraphRevision(21, _hash("atlas-21"))
        second = _measurement(1, forward_revision)
        with tempfile.TemporaryDirectory() as tmp:
            controller = self._controller(Path(tmp))
            bridge = OoeCartographyBridge(controller)
            bridge.ingest_authenticated(
                _Atlas((first,), forward_revision),
                first,
                source_action="qwen_fallback",
                target_action="probe_coordinate",
                o1_surprise=0.1,
                o1_learning_progress=0.2,
            )
            bridge.ingest_authenticated(
                _Atlas((second,), GraphRevision(22, _hash("atlas-22"))),
                second,
                source_action="probe_coordinate",
                target_action="probe_coordinate",
                o1_surprise=0.2,
                o1_learning_progress=0.3,
            )
            self.assertEqual(controller.atlas_graph_revision, forward_revision)

            rollback = _measurement(2, GraphRevision(19, _hash("atlas-19")))
            with self.assertRaises(OoeControllerStaleError):
                bridge.ingest_authenticated(
                    _Atlas((rollback,), GraphRevision(23, _hash("atlas-23"))),
                    rollback,
                    source_action="probe_coordinate",
                    target_action="probe_coordinate",
                    o1_surprise=0.3,
                    o1_learning_progress=0.4,
                )

    def test_verifier_rejects_measurement_absent_from_atlas(self) -> None:
        measurement = _measurement(0, _ATLAS_REVISION)
        atlas = _Atlas((), GraphRevision(21, _hash("atlas-21")))
        with self.assertRaisesRegex(ValueError, "not active"):
            authenticated_measurement_verifier_sha256(atlas, measurement)

    def test_verifier_rejects_concurrent_atlas_head_change(self) -> None:
        measurement = _measurement(0, _ATLAS_REVISION)
        atlas = _MovingAtlas((measurement,), GraphRevision(21, _hash("atlas-21")))
        with self.assertRaisesRegex(RuntimeError, "changed during"):
            authenticated_measurement_verifier_sha256(atlas, measurement)

    def test_promoted_probe_action_discharge_saves_real_teacher_probe(self) -> None:
        measurement = _measurement(0, _ATLAS_REVISION)
        atlas = _Atlas((measurement,), GraphRevision(21, _hash("atlas-21")))
        executor = AtlasProbeExecutor(atlas)
        with tempfile.TemporaryDirectory() as tmp:
            controller = self._controller(
                Path(tmp),
                action_executors={"probe_coordinate": executor},
            )
            bridge = OoeCartographyBridge(controller)
            learning = bridge.ingest_authenticated(
                atlas,
                measurement,
                source_action="qwen_fallback",
                target_action="probe_coordinate",
                o1_surprise=0.25,
                o1_learning_progress=1.5,
            )
            bridge.promote_ready()
            warm = QwenOoeFeatureReceipt.from_measurement(
                measurement,
                temporal_index=1,
                verifier_sha256s=(learning.verifier_sha256,),
                o1_surprise=0.25,
                o1_learning_progress=1.5,
            )
            decision = controller.try_warm(
                warm,
                "qwen_fallback",
                quality_verifier=lambda receipt, execution: (
                    verify_atlas_probe_execution(atlas, receipt, execution)
                ),
                stream_id="real-probe-discharge",
            )

            self.assertEqual(decision.origin, "crystal")
            self.assertEqual(decision.action, "probe_coordinate")
            self.assertTrue(decision.quality_verified)
            self.assertEqual(controller.metrics.saved_qwen_forwards, 0)
            accounting = controller.commit_warm(decision)
            self.assertEqual(accounting.saved_qwen_forwards, 1)
            self.assertEqual(controller.metrics.saved_qwen_forwards, 1)
            self.assertEqual(controller.metrics.executed_qwen_forwards, 0)


if __name__ == "__main__":
    unittest.main()
