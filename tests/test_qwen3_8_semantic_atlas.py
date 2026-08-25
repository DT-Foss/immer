from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from immer.knowledge.livecausal import LiveCausalIntegrityError, LiveGraph
from immer.runtimes.qwen3_8 import (
    EvidencePolicy,
    GraphRevision,
    InterventionIdentity,
    MeasurementReceipt,
    ModelPin,
    NumericSummary,
    PlaceboEffect,
    ProbeIdentity,
    ReplicaReceipt,
    RuntimeProvenance,
    SemanticAtlasError,
    SemanticAtlasIntegrityError,
    SemanticAtlasPromotionError,
    SemanticAtlasStaleCheckpointError,
    SemanticWeightAtlas,
    TensorRangePlan,
    WeightCoordinate,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


_CODE_REVISION = "1" * 40
_PLAN = TensorRangePlan(
    name="model.layers.2.mlp.gate_proj.weight",
    dtype="BF16",
    shape=(4, 2),
    shard="model-00001-of-00018.safetensors",
    absolute_offset=100,
    length=16,
)
_PIN = ModelPin(
    repo_id="Qwen/Qwen3.8-27B",
    revision="0123456789abcdef",
    bundle_fingerprint=_hash("bundle"),
    bundle_manifest_sha256=_hash("bundle-manifest"),
    code_revision=_CODE_REVISION,
)
_COORDINATE = WeightCoordinate.from_plan(
    _PLAN,
    layer=2,
    module="model.layers.2.mlp.gate_proj",
    head_index=3,
    row_start=1,
    row_end=3,
)
_PROBE = ProbeIdentity(
    question_sha256=_hash("question"),
    token_sha256=_hash("tokens"),
    family_sha256=_hash("family"),
    label_source_sha256=_hash("gold-free-source"),
)
_RUNTIME = RuntimeProvenance(
    code_revision=_CODE_REVISION,
    source_manifest_sha256=_hash("sources"),
    dependency_manifest_sha256=_hash("dependencies"),
    runtime_configuration_sha256=_hash("runtime-config"),
    platform_sha256=_hash("platform"),
)
_WEIGHT_RAIL_REVISION = GraphRevision(sequence=77, event_sha256=_hash("rail-head"))
_ATLAS_CAPTURE_REVISION = GraphRevision(sequence=0, event_sha256="0" * 64)


def _summary(metric: str, value: float) -> NumericSummary:
    return NumericSummary(
        metric=metric,
        count=2,
        total=2.0 * value,
        total_squares=2.0 * value * value,
        minimum=value,
        maximum=value,
    )


def _measurement(
    *,
    mode: str,
    status: str,
    label: str | None,
    value: float,
    effect: PlaceboEffect | None = None,
    suffix: str = "base",
) -> MeasurementReceipt:
    return MeasurementReceipt(
        model_pin=_PIN,
        coordinate=_COORDINATE,
        probe=_PROBE,
        intervention=InterventionIdentity(
            mode=mode,
            configuration_sha256=_hash(f"intervention-{mode}-{suffix}"),
        ),
        observation_status=status,
        observed_semantic_label=label,
        hidden_sha256=_hash(f"hidden-{suffix}"),
        activation_sha256=_hash(f"activation-{suffix}"),
        logits_sha256=_hash(f"logits-{suffix}"),
        state_sha256=_hash(f"state-{suffix}"),
        access_trace_sha256=_hash(f"access-{suffix}"),
        evidence_sha256=_hash(f"evidence-{suffix}"),
        weight_rail_revision=_WEIGHT_RAIL_REVISION,
        atlas_head_revision=_ATLAS_CAPTURE_REVISION,
        numeric_summaries=(_summary("activation_delta", value),),
        placebo_effects=() if effect is None else (effect,),
        runtime=_RUNTIME,
    )


def _replica(
    measurement: MeasurementReceipt,
    *,
    index: int,
    verdict: str,
    semantic_label: str | None,
    weight: float,
) -> ReplicaReceipt:
    return ReplicaReceipt(
        model_pin=_PIN,
        coordinate=_COORDINATE,
        measurement_sha256=measurement.sha256,
        probe_sha256=measurement.probe.sha256,
        intervention_sha256=measurement.intervention.sha256,
        replica_id_sha256=_hash(f"replica-{index}"),
        replica_measurement_sha256=_hash(f"replica-measurement-{index}"),
        verdict=verdict,
        semantic_label=semantic_label,
        reputation_weight=weight,
        evidence_sha256=_hash(f"replica-evidence-{index}"),
        runtime=_RUNTIME,
    )


_POLICY = EvidencePolicy(
    name="three-replica-positive-effect",
    minimum_replica_count=3,
    minimum_support_count=2,
    minimum_support_weight=1.2,
    effect_metric="activation_delta",
    minimum_effect=0.75,
    effect_direction="positive",
)


class SemanticAtlasSchemaTests(unittest.TestCase):
    def test_canonical_round_trip_and_tamper_rejection(self) -> None:
        placebo = _measurement(
            mode="placebo", status="recorded", label=None, value=1.0, suffix="placebo"
        )
        effect = PlaceboEffect(
            metric="activation_delta",
            placebo_measurement_sha256=placebo.sha256,
            observed_mean=2.0,
            placebo_mean=1.0,
            delta=1.0,
        )
        measurement = _measurement(
            mode="native",
            status="eligible",
            label="arithmetic/addition",
            value=2.0,
            effect=effect,
            suffix="native",
        )
        self.assertEqual(
            MeasurementReceipt.from_document(measurement.to_document()), measurement
        )
        self.assertEqual(
            WeightCoordinate.from_document(_COORDINATE.to_document()), _COORDINATE
        )

        tampered = measurement.to_document()
        tampered["body"]["hidden_sha256"] = _hash("tampered")
        with self.assertRaises(SemanticAtlasIntegrityError):
            MeasurementReceipt.from_document(tampered)

        with self.assertRaises(SemanticAtlasError):
            WeightCoordinate.from_plan(
                _PLAN,
                layer=2,
                module="model.layers.2.mlp.gate_proj",
                row_start=1,
                row_end=3,
                relative_byte_offset=1,
            )


class SemanticWeightAtlasTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix=".semantic-atlas-test-", dir=Path.cwd()
        )
        self.root = Path(self.temporary.name)
        self.atlas = SemanticWeightAtlas(
            self.root, model_pin=_PIN, tensor_plans=(_PLAN,)
        )
        self.assertEqual(self.atlas.revision(), _ATLAS_CAPTURE_REVISION)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _append_candidate(
        self,
        *,
        label: str = "arithmetic/addition",
        suffix: str = "candidate",
        delta: float = 1.0,
    ) -> tuple[MeasurementReceipt, MeasurementReceipt]:
        placebo = _measurement(
            mode="placebo",
            status="recorded",
            label=None,
            value=1.0,
            suffix=f"placebo-{suffix}",
        )
        self.atlas.append_measurement(placebo)
        effect = PlaceboEffect(
            metric="activation_delta",
            placebo_measurement_sha256=placebo.sha256,
            observed_mean=1.0 + delta,
            placebo_mean=1.0,
            delta=delta,
        )
        candidate = _measurement(
            mode="native" if suffix == "candidate" else "patch",
            status="eligible",
            label=label,
            value=1.0 + delta,
            effect=effect,
            suffix=suffix,
        )
        self.atlas.append_measurement(candidate)
        return placebo, candidate

    def _append_quorum(
        self, measurement: MeasurementReceipt, *, offset: int = 0
    ) -> tuple[ReplicaReceipt, ...]:
        rows = (
            _replica(
                measurement,
                index=offset,
                verdict="support",
                semantic_label=measurement.observed_semantic_label,
                weight=0.7,
            ),
            _replica(
                measurement,
                index=offset + 1,
                verdict="support",
                semantic_label=measurement.observed_semantic_label,
                weight=0.6,
            ),
            _replica(
                measurement,
                index=offset + 2,
                verdict="oppose",
                semantic_label="different/family",
                weight=0.1,
            ),
        )
        for row in rows:
            self.atlas.append_replica(row)
        return rows

    def test_append_query_reopen_idempotency_coverage_and_drop(self) -> None:
        placebo = _measurement(
            mode="placebo", status="recorded", label=None, value=1.0, suffix="plain"
        )
        first = self.atlas.append_measurement(placebo)
        replay = self.atlas.append_measurement(placebo)
        self.assertTrue(first.appended)
        self.assertFalse(replay.appended)
        self.assertEqual(first.segment_sha256, replay.segment_sha256)

        by_prompt = self.atlas.query_by_prompt_signature(_PROBE.prompt_signature)
        self.assertEqual(by_prompt.measurements, (placebo,))
        by_coordinate = self.atlas.query_by_coordinate(_COORDINATE)
        self.assertEqual(by_coordinate.measurements, (placebo,))
        coverage = self.atlas.coverage_matrix()
        self.assertEqual(coverage["body"]["measurement_count"], 1)
        self.assertEqual(
            coverage["body"]["rows"][0]["coordinate_sha256"], _COORDINATE.sha256
        )

        reopened = SemanticWeightAtlas(self.root, model_pin=_PIN, tensor_plans=(_PLAN,))
        self.assertEqual(
            reopened.query_by_coordinate(_COORDINATE).measurements, (placebo,)
        )
        self.assertTrue(reopened.verify())
        self.assertEqual(
            reopened.drop_segments((first.segment_sha256,)), (first.segment_sha256,)
        )
        self.assertEqual(reopened.query_by_coordinate(_COORDINATE).measurements, ())
        self.assertIn(first.segment_sha256, reopened.graph.store.tombstones())

    def test_steady_append_authenticates_delta_without_full_store_rehash(self) -> None:
        receipt = _measurement(
            mode="passive",
            status="recorded",
            label="routing",
            value=1.0,
            suffix="stream-delta",
        )
        with mock.patch.object(
            self.atlas.graph.store,
            "verify",
            side_effect=AssertionError(
                "steady append forced a full store verification"
            ),
        ):
            appended = self.atlas.append_measurement(receipt)
        self.assertTrue(appended.appended)
        self.assertEqual(
            self.atlas.query_by_coordinate(_COORDINATE).measurements, (receipt,)
        )

    def test_stale_checkpoint_and_wrong_tensor_ranges_fail_closed(self) -> None:
        receipt = _measurement(
            mode="passive", status="recorded", label="routing", value=1.0
        )
        self.atlas.append_measurement(receipt)
        stale_pin = replace(_PIN, revision="fedcba9876543210")
        with self.assertRaises(SemanticAtlasStaleCheckpointError):
            SemanticWeightAtlas(
                self.root,
                model_pin=stale_pin,
                tensor_plans=(_PLAN,),
            )

        stale_plan = replace(_PLAN, absolute_offset=_PLAN.absolute_offset + 32)
        wrong_coordinate = WeightCoordinate.from_plan(
            stale_plan,
            layer=2,
            module="model.layers.2.mlp.gate_proj",
        )
        with self.assertRaises(SemanticAtlasIntegrityError):
            self.atlas.append_measurement(replace(receipt, coordinate=wrong_coordinate))

    def test_drop_rejects_orphaned_effects_but_accepts_dependency_set(self) -> None:
        placebo, _candidate = self._append_candidate()
        placebo_segment = next(
            segment
            for segment in self.atlas.graph.store.segments()
            if any(
                record.get("document_sha256") == placebo.sha256
                for _sha_value, _index, record in self.atlas.graph.store.iter_records(
                    segment
                )
            )
        )
        with self.assertRaises(SemanticAtlasIntegrityError):
            self.atlas.drop_segments((placebo_segment,))
        self.assertIn(placebo_segment, self.atlas.graph.store.segments())

        active = self.atlas.graph.store.segments()
        self.assertEqual(set(self.atlas.drop_segments(active)), set(active))
        self.assertEqual(self.atlas.query_by_coordinate(_COORDINATE).measurements, ())

    def test_placebo_effect_and_explicit_quorum_gates(self) -> None:
        _placebo, candidate = self._append_candidate()
        with self.assertRaises(SemanticAtlasPromotionError):
            self.atlas.promote_label(candidate.sha256, policy=_POLICY)

        replicas = self._append_quorum(candidate)
        promoted = self.atlas.promote_label(candidate.sha256, policy=_POLICY)
        result = self.atlas.query_by_semantic_label("arithmetic/addition")
        self.assertEqual(result.measurements, (candidate,))
        self.assertEqual(len(result.promotions), 1)
        promotion = result.promotions[0]
        self.assertEqual(promotion.support_count, 2)
        self.assertAlmostEqual(promotion.support_weight, 1.3)
        self.assertEqual(
            set(promotion.support_replica_sha256s),
            {replicas[0].sha256, replicas[1].sha256},
        )
        self.assertEqual(promotion.to_document()["body"]["security_claim"], "none")
        self.assertTrue(promoted.appended)

        _weak_placebo, weak = self._append_candidate(suffix="weak", delta=0.2)
        self._append_quorum(weak, offset=10)
        with self.assertRaises(SemanticAtlasPromotionError):
            self.atlas.promote_label(weak.sha256, policy=_POLICY)

    def test_label_conflicts_are_preserved_never_overwritten(self) -> None:
        _first_placebo, first = self._append_candidate()
        self._append_quorum(first)
        self.atlas.promote_label(first.sha256, policy=_POLICY)

        _second_placebo, second = self._append_candidate(
            label="calendar/rate", suffix="calendar"
        )
        self._append_quorum(second, offset=20)
        self.atlas.promote_label(second.sha256, policy=_POLICY)

        by_coordinate = self.atlas.query_by_coordinate(_COORDINATE)
        self.assertEqual(
            {row.semantic_label for row in by_coordinate.promotions},
            {"arithmetic/addition", "calendar/rate"},
        )
        coverage = self.atlas.coverage_matrix()["body"]
        self.assertEqual(coverage["columns"], ["arithmetic/addition", "calendar/rate"])
        self.assertEqual(coverage["verified_counts"], [[1, 1]])
        consensus = self.atlas.consensus_by_coordinate(_COORDINATE)["body"]
        self.assertTrue(consensus["conflict"])
        self.assertEqual(
            [row["semantic_label"] for row in consensus["claims"]],
            ["arithmetic/addition", "calendar/rate"],
        )
        self.assertEqual(consensus["security_claim"], "none")

    def test_record_key_collision_and_segment_tamper_are_rejected(self) -> None:
        first = _measurement(
            mode="passive", status="recorded", label="routing", value=1.0, suffix="one"
        )
        second = _measurement(
            mode="off", status="recorded", label="routing", value=1.0, suffix="two"
        )
        self.atlas.append_measurement(first)
        forged = list(self.atlas._edge_records(second))[0]
        forged["document_sha256"] = first.sha256
        forged["trigger_key"] = forged["trigger_key"].replace(
            second.sha256, first.sha256
        )
        self.atlas.graph.append_segment((forged,))
        with self.assertRaises(SemanticAtlasIntegrityError):
            self.atlas.verify_or_raise()

        with tempfile.TemporaryDirectory(
            prefix=".semantic-atlas-tamper-", dir=Path.cwd()
        ) as temporary:
            atlas = SemanticWeightAtlas(
                temporary, model_pin=_PIN, tensor_plans=(_PLAN,)
            )
            appended = atlas.append_measurement(first)
            path = atlas.graph.store.segment_path(appended.segment_sha256)
            data = bytearray(path.read_bytes())
            marker = _hash("hidden-one").encode("ascii")
            offset = data.index(marker)
            data[offset] = ord("0") if data[offset] != ord("0") else ord("1")
            path.write_bytes(data)
            self.assertFalse(atlas.verify())
            with self.assertRaises(LiveCausalIntegrityError):
                LiveGraph(temporary)

    def test_replica_identity_equivocation_is_not_counted_as_security(self) -> None:
        _placebo, candidate = self._append_candidate()
        replicas = self._append_quorum(candidate)
        conflicting = replace(
            replicas[0],
            replica_measurement_sha256=_hash("conflicting-replica-output"),
            evidence_sha256=_hash("conflicting-replica-evidence"),
        )
        self.atlas.append_replica(conflicting)
        with self.assertRaises(SemanticAtlasPromotionError):
            self.atlas.promote_label(candidate.sha256, policy=_POLICY)


if __name__ == "__main__":
    unittest.main()
