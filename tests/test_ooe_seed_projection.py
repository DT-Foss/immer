from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import unittest

import numpy as np

from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.seed_projection import (
    GroupAwareZeroFailureRiskReceipt,
    SEED_TRAINING_MANIFEST_SCHEMA,
    SeedProjectionError,
    SeedProjectionIntegrityError,
    SeedProjectionSplitError,
    SeedTrainingBatch,
    SeedTrainingProjection,
    SeedVectorProjection,
)
from immer.runtimes.qwen3_8.semantic_atlas import (
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


def _vectors(*, feature: tuple[float, ...] = (0.25, -0.5), quality: float = 1.0) -> SeedVectorProjection:
    return SeedVectorProjection(
        feature_schema_sha256=_hash("feature-schema/v1"),
        feature=feature,
        action_schema_sha256=_hash("action-schema/v1"),
        action=(0.0, 1.0),
        consequence_schema_sha256=_hash("consequence-schema/v1"),
        consequence=(0.75,),
        quality_schema_sha256=_hash("quality-schema/v1"),
        quality=(quality,),
        work_cost_schema_sha256=_hash("work-cost-schema/v1"),
        work_cost=(0.125, 0.5),
    )


def _projection(
    *,
    group: str,
    sequence: int = 0,
    split: str = "train",
    prompt_generation: int = 1,
    generation: int = 0,
    quality: float = 1.0,
    feature: tuple[float, ...] = (0.25, -0.5),
) -> SeedTrainingProjection:
    return SeedTrainingProjection(
        source_kind="measurement",
        source_receipt_sha256s=(_hash(f"source:{group}:{sequence}"),),
        model_pin_sha256=_hash("model"),
        code_identity_sha256=_hash("code"),
        site_identity_sha256s=(_hash(f"site:{group}"),),
        graph_revision_sha256s=(_hash("weight-graph"), _hash("atlas-graph")),
        authority_sha256s=(_hash("authority"),),
        verifier_sha256s=(_hash("verifier"),),
        split=split,
        generation=generation,
        prompt_generation=prompt_generation,
        group_id_sha256=_hash(f"group:{group}"),
        sequence_index=sequence,
        vectors=_vectors(feature=feature, quality=quality),
    )


def _measurement() -> MeasurementReceipt:
    code = "d" * 40
    pin = ModelPin(
        repo_id="Qwen/Qwen3.8-27B",
        revision="0123456789abcdef",
        bundle_fingerprint=_hash("bundle"),
        bundle_manifest_sha256=_hash("bundle-manifest"),
        code_revision=code,
    )
    runtime = RuntimeProvenance(
        code_revision=code,
        source_manifest_sha256=_hash("source-manifest"),
        dependency_manifest_sha256=_hash("dependencies"),
        runtime_configuration_sha256=_hash("runtime-config"),
        platform_sha256=_hash("platform"),
    )
    coordinate = WeightCoordinate(
        layer=18,
        module="model.layers.18.mlp.gate_proj",
        tensor="model.layers.18.mlp.gate_proj.weight",
        dtype="BF16",
        shape=(2, 2),
        shard="model-00007-of-00018.safetensors",
        tensor_absolute_offset=4096,
        tensor_length=8,
        range_absolute_offset=4096,
        range_length=8,
        row_start=0,
        row_end=2,
    )
    return MeasurementReceipt(
        model_pin=pin,
        coordinate=coordinate,
        probe=ProbeIdentity(
            question_sha256=_hash("private-question"),
            token_sha256=_hash("tokens"),
            family_sha256=_hash("family"),
            label_source_sha256=_hash("label-source"),
        ),
        intervention=InterventionIdentity(
            mode="native",
            configuration_sha256=_hash("intervention"),
        ),
        observation_status="recorded",
        observed_semantic_label=None,
        hidden_sha256=_hash("hidden"),
        activation_sha256=_hash("activation"),
        logits_sha256=_hash("logits"),
        state_sha256=_hash("state"),
        access_trace_sha256=_hash("access"),
        evidence_sha256=_hash("evidence"),
        weight_rail_revision=GraphRevision(4, _hash("weight-event")),
        atlas_head_revision=GraphRevision(9, _hash("atlas-event")),
        numeric_summaries=(
            NumericSummary(
                metric="activation-rms",
                count=2,
                total=3.0,
                total_squares=5.0,
                minimum=1.0,
                maximum=2.0,
            ),
        ),
        placebo_effects=(),
        runtime=runtime,
    )


class SeedProjectionTests(unittest.TestCase):
    def test_vectors_are_finite_bounded_and_fixed_schema(self) -> None:
        self.assertEqual(_vectors().feature, (0.25, -0.5))
        with self.assertRaisesRegex(SeedProjectionError, "finite values"):
            _vectors(feature=(math.nan, 0.0))
        with self.assertRaisesRegex(SeedProjectionError, "finite values"):
            _vectors(feature=(1.0001, 0.0))
        with self.assertRaises(ValueError):
            replace(_vectors(), feature_schema_sha256="not-a-hash")

    def test_projection_is_canonical_replayable_and_prompt_free(self) -> None:
        row = _projection(group="alpha")
        data = row.to_bytes()
        self.assertEqual(SeedTrainingProjection.from_bytes(data), row)
        self.assertEqual(hashlib.sha256(data).hexdigest(), row.sha256)
        self.assertNotIn(b"private-question", data)

        tampered = json.loads(data)
        tampered["body"]["vectors"]["quality"]["values"] = [-1.0]
        with self.assertRaisesRegex(SeedProjectionIntegrityError, "hash mismatch"):
            SeedTrainingProjection.from_bytes(canonical_json_bytes(tampered))

        noncanonical = json.dumps(json.loads(data), indent=2).encode("utf-8")
        with self.assertRaisesRegex(SeedProjectionIntegrityError, "canonical"):
            SeedTrainingProjection.from_bytes(noncanonical)

    def test_measurement_adapter_retains_native_identities_and_uses_callback(self) -> None:
        measurement = _measurement()
        seen: list[MeasurementReceipt] = []

        def project(value: MeasurementReceipt) -> SeedVectorProjection:
            seen.append(value)
            return _vectors(feature=(0.1, 0.2))

        row = SeedTrainingProjection.from_measurement(
            measurement,
            split="holdout",
            generation=12,
            prompt_generation=7,
            group_id_sha256=_hash("measurement-group"),
            sequence_index=0,
            vector_projector=project,
            verifier_sha256s=(_hash("measurement-verifier"),),
            authority_sha256s=(_hash("extra-authority"),),
        )
        self.assertEqual(seen, [measurement])
        self.assertEqual(row.source_receipt_sha256s, (measurement.sha256,))
        self.assertEqual(row.model_pin_sha256, measurement.model_pin.sha256)
        self.assertEqual(row.code_identity_sha256, measurement.runtime.sha256)
        self.assertEqual(row.site_identity_sha256s, (measurement.coordinate.sha256,))
        self.assertEqual(
            set(row.graph_revision_sha256s),
            {
                measurement.weight_rail_revision.sha256,
                measurement.atlas_head_revision.sha256,
            },
        )
        self.assertIn(measurement.probe.sha256, row.authority_sha256s)
        self.assertIn(measurement.intervention.sha256, row.authority_sha256s)
        self.assertEqual(row.feature_vector, (0.1, 0.2))

    def test_batch_stable_order_manifest_and_btf_feature_view(self) -> None:
        rows = (
            _projection(group="beta", sequence=1, generation=13),
            _projection(group="alpha", sequence=0, generation=10),
            _projection(group="beta", sequence=0, generation=12),
            _projection(group="alpha", sequence=1, generation=11),
        )
        batch = SeedTrainingBatch(tuple(reversed(rows)))
        replay = SeedTrainingBatch.from_bytes(batch.to_bytes())
        self.assertEqual(replay, batch)
        self.assertEqual(
            [row.sequence_index for row in batch.projections],
            [0, 1, 0, 1],
        )

        tensor, mask, groups = batch.feature_tensor("train")
        self.assertEqual(tensor.shape, (2, 2, 2))
        self.assertEqual(mask.shape, (2, 2))
        self.assertTrue(mask.all())
        self.assertFalse(tensor.flags.writeable)
        self.assertFalse(mask.flags.writeable)
        self.assertEqual(groups, tuple(sorted(groups)))
        np.testing.assert_array_equal(tensor[:, :, 0], np.full((2, 2), 0.25))
        action_tensor, action_mask, action_groups = batch.vector_tensor(
            "train", "action"
        )
        self.assertEqual(action_tensor.shape, (2, 2, 2))
        np.testing.assert_array_equal(action_mask, mask)
        self.assertEqual(action_groups, groups)
        np.testing.assert_array_equal(action_tensor[:, :, 1], np.ones((2, 2)))
        with self.assertRaisesRegex(SeedProjectionError, "unknown"):
            batch.vector_tensor("train", "made-up")

        manifest = batch.to_seed_v3_manifest()
        self.assertEqual(manifest["schema"], SEED_TRAINING_MANIFEST_SCHEMA)
        body = manifest["body"]
        self.assertEqual(body["feature_dimensions"], 2)
        self.assertEqual(body["feature_schema_sha256"], _hash("feature-schema/v1"))
        self.assertEqual(body["padded_feature_shapes"]["train"], [2, 2, 2])
        self.assertEqual(body["axis_order"], ["group", "sequence", "feature"])
        self.assertEqual(body["batch_sha256"], batch.sha256)
        self.assertEqual(
            hashlib.sha256(batch.manifest_bytes()).hexdigest(),
            batch.manifest_sha256,
        )

    def test_batch_rejects_split_leakage_shape_drift_and_broken_sequences(self) -> None:
        shared_group = _hash("group:shared")
        train = replace(_projection(group="train"), group_id_sha256=shared_group)
        holdout = replace(
            _projection(group="holdout", split="holdout", prompt_generation=2),
            group_id_sha256=shared_group,
        )
        with self.assertRaisesRegex(SeedProjectionSplitError, "group crosses"):
            SeedTrainingBatch((train, holdout))

        calibration = _projection(
            group="calibration",
            split="calibration",
            prompt_generation=1,
        )
        with self.assertRaisesRegex(SeedProjectionSplitError, "prompt generation"):
            SeedTrainingBatch((train, calibration))

        drift = _projection(group="drift", feature=(0.25, -0.5, 0.75))
        with self.assertRaisesRegex(SeedProjectionError, "fixed vector"):
            SeedTrainingBatch((train, drift))

        broken = _projection(group="broken", sequence=1)
        with self.assertRaisesRegex(SeedProjectionError, "contiguous"):
            SeedTrainingBatch((broken,))

    def test_group_risk_counts_groups_not_correlated_rows(self) -> None:
        rows = tuple(
            _projection(
                group=f"g{group}",
                sequence=sequence,
                prompt_generation=5,
                generation=group * 10 + sequence,
            )
            for group in range(3)
            for sequence in range(5)
        )
        batch = SeedTrainingBatch(rows)
        groups = tuple(sorted({row.group_id_sha256 for row in rows}))
        risk = GroupAwareZeroFailureRiskReceipt.evaluate(
            batch,
            admitted_group_sha256s=groups,
            failure_predicate=lambda row: row.vectors.quality[0] < 0.0,
            failure_policy_sha256=_hash("quality-negative-is-failure/v1"),
            confidence=0.95,
            maximum_risk=0.7,
        )
        self.assertEqual(risk.admitted_group_count, 3)
        self.assertEqual(risk.admitted_row_count, 15)
        self.assertEqual(risk.worst_group_failures, 0)
        self.assertAlmostEqual(risk.exact_upper_bound, 1.0 - 0.05 ** (1.0 / 3.0))
        self.assertTrue(risk.promotion_admissible)
        self.assertEqual(
            GroupAwareZeroFailureRiskReceipt.from_bytes(risk.to_bytes()),
            risk,
        )

        failed_rows = list(rows)
        failed_rows[4] = replace(failed_rows[4], vectors=_vectors(quality=-1.0))
        failed_batch = SeedTrainingBatch(tuple(failed_rows))
        with self.assertRaisesRegex(SeedProjectionError, "rejects"):
            GroupAwareZeroFailureRiskReceipt.evaluate(
                failed_batch,
                admitted_group_sha256s=groups,
                failure_predicate=lambda row: row.vectors.quality[0] < 0.0,
                failure_policy_sha256=_hash("quality-negative-is-failure/v1"),
            )

    def test_risk_replay_detects_tamper(self) -> None:
        row = _projection(group="only")
        batch = SeedTrainingBatch((row,))
        risk = GroupAwareZeroFailureRiskReceipt.evaluate(
            batch,
            admitted_group_sha256s=(row.group_id_sha256,),
            failure_predicate=lambda _row: False,
            failure_policy_sha256=_hash("never-fail-test-policy"),
            maximum_risk=1.0,
        )
        document = json.loads(risk.to_bytes())
        document["body"]["admitted_row_count"] = 2
        with self.assertRaisesRegex(SeedProjectionIntegrityError, "hash mismatch"):
            GroupAwareZeroFailureRiskReceipt.from_bytes(
                canonical_json_bytes(document)
            )


if __name__ == "__main__":
    unittest.main()
