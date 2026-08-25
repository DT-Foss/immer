from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from immer.cognition.fertig.atlas_verifier import (
    FertigAtlasReplicaConflictError,
    FertigAtlasVerificationError,
    append_and_promote_fertig_label,
    fertig_label_evidence_sha256,
    verify_fertig_measurement,
)
from immer.runtimes.deepseek_v4.causal_weights import TensorRangePlan
from immer.runtimes.qwen3_8.cartography_probe import (
    CARTOGRAPHY_EVIDENCE_SCHEMA,
    ProbeCoordinateSpec,
    ProbeSpec,
    prompt_token_sha256,
)
from immer.runtimes.qwen3_8.native_crsa import Qwen38NativeHeadCrsa
from immer.runtimes.qwen3_8.semantic_atlas import (
    EvidencePolicy,
    GraphRevision,
    InterventionIdentity,
    MeasurementReceipt,
    ModelPin,
    NumericSummary,
    PlaceboEffect,
    RuntimeProvenance,
    SemanticWeightAtlas,
    WeightCoordinate,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _seal_evidence(probe_spec: ProbeSpec) -> dict[str, object]:
    body = {
        "probe_spec": json.loads(
            json.dumps(
                probe_spec.as_record(),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        ),
        "synthetic_fixture": "fertig-atlas-verifier",
    }
    encoded = json.dumps(
        body,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "body": body,
        "schema": CARTOGRAPHY_EVIDENCE_SCHEMA,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


_QUESTION = (
    "Peter wants to make different sized ice cubes with 32 ounces of water. "
    "He can make giant cubes that use 4 ounces per cube, medium cubes that "
    "use 2 ounces, and small cubes that use 1/2 an ounce. If he makes 3 giant "
    "cubes, 7 medium cubes, and 8 small cubes, how many ounces of water does "
    "he have left?"
)
_FORMULA_ONLY_QUESTION = (
    "Maggie's oven is malfunctioning. When she sets it to 450 the actual "
    "temperature is 468. If it's off by the same percentage for any recipe, "
    "what temperature should she set it at if her recipe calls for 520 degrees?"
)
_LABEL = "arithmetic/resource-balance"
_CODE_REVISION = "a" * 40
_PLAN = TensorRangePlan(
    name="model.layers.27.mlp.gate_proj.weight",
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
    layer=27,
    module="model.layers.27.mlp.gate_proj",
)
_RUNTIME = RuntimeProvenance(
    code_revision=_CODE_REVISION,
    source_manifest_sha256=_hash("source-manifest"),
    dependency_manifest_sha256=_hash("dependency-manifest"),
    runtime_configuration_sha256=_hash("runtime-configuration"),
    platform_sha256=_hash("platform"),
)
_WEIGHT_RAIL = GraphRevision(sequence=8, event_sha256=_hash("weight-rail"))
_ATLAS_HEAD = GraphRevision(sequence=0, event_sha256="0" * 64)
_POLICY = EvidencePolicy(
    name="fertig-single-verifier-positive-effect",
    minimum_replica_count=1,
    minimum_support_count=1,
    minimum_support_weight=1.0,
    effect_metric="activation_delta",
    minimum_effect=0.5,
    effect_direction="positive",
)


def _summary(value: float) -> NumericSummary:
    return NumericSummary(
        metric="activation_delta",
        count=2,
        total=2.0 * value,
        total_squares=2.0 * value * value,
        minimum=value,
        maximum=value,
    )


def _measurement_pair() -> tuple[
    ProbeSpec,
    dict[str, object],
    MeasurementReceipt,
    MeasurementReceipt,
]:
    expected = fertig_label_evidence_sha256(_QUESTION, _LABEL)
    prompt_tokens = (101, 202, 303)
    probe_spec = ProbeSpec(
        prompt_token_ids=prompt_tokens,
        prompt_sha256=prompt_token_sha256(prompt_tokens),
        start_layer=0,
        stop_layer=28,
        coordinate=ProbeCoordinateSpec(
            layer=27,
            module="model.layers.27.mlp.gate_proj",
            tensor="model.layers.27.mlp.gate_proj.weight",
        ),
        intervention_mode="native",
        code_revision=_CODE_REVISION,
        question_sha256=_hash(_QUESTION),
        family_sha256=_hash("resource-balance-family"),
        label_source_sha256=_hash("sealed-fertig-label-source"),
        semantic_label=_LABEL,
        label_evidence_sha256=expected,
        native_head_crsa=Qwen38NativeHeadCrsa(alpha=0.01),
    )
    assert probe_spec.as_record()["label_evidence_sha256"] == expected
    evidence_document = _seal_evidence(probe_spec)
    probe = probe_spec.probe_identity
    placebo = MeasurementReceipt(
        model_pin=_PIN,
        coordinate=_COORDINATE,
        probe=probe,
        intervention=InterventionIdentity(
            mode="placebo",
            configuration_sha256=_hash("placebo-configuration"),
        ),
        observation_status="recorded",
        observed_semantic_label=None,
        hidden_sha256=_hash("placebo-hidden"),
        activation_sha256=_hash("placebo-activation"),
        logits_sha256=_hash("placebo-logits"),
        state_sha256=_hash("placebo-state"),
        access_trace_sha256=_hash("placebo-access"),
        evidence_sha256=_hash("placebo-evidence"),
        weight_rail_revision=_WEIGHT_RAIL,
        atlas_head_revision=_ATLAS_HEAD,
        numeric_summaries=(_summary(1.0),),
        placebo_effects=(),
        runtime=_RUNTIME,
    )
    effect = PlaceboEffect(
        metric="activation_delta",
        placebo_measurement_sha256=placebo.sha256,
        observed_mean=2.0,
        placebo_mean=1.0,
        delta=1.0,
    )
    measurement = MeasurementReceipt(
        model_pin=_PIN,
        coordinate=_COORDINATE,
        probe=probe,
        intervention=InterventionIdentity(
            mode="native",
            configuration_sha256=_hash("native-configuration"),
        ),
        observation_status="eligible",
        observed_semantic_label=_LABEL,
        hidden_sha256=_hash("native-hidden"),
        activation_sha256=_hash("native-activation"),
        logits_sha256=_hash("native-logits"),
        state_sha256=_hash("native-state"),
        access_trace_sha256=_hash("native-access"),
        evidence_sha256=str(evidence_document["sha256"]),
        weight_rail_revision=_WEIGHT_RAIL,
        atlas_head_revision=_ATLAS_HEAD,
        numeric_summaries=(_summary(2.0),),
        placebo_effects=(effect,),
        runtime=_RUNTIME,
    )
    return probe_spec, evidence_document, placebo, measurement


class FertigAtlasVerifierTests(unittest.TestCase):
    def test_question_proof_and_label_bindings_fail_closed(self) -> None:
        probe_spec, evidence, _placebo, measurement = _measurement_pair()

        with self.assertRaisesRegex(
            FertigAtlasVerificationError, "differs from the sealed ProbeSpec"
        ):
            verify_fertig_measurement(
                f"{_QUESTION} ",
                probe_spec,
                measurement,
                evidence_document=evidence,
            )

        wrong_proof = replace(
            probe_spec,
            label_evidence_sha256=_hash("wrong-proof"),
        )
        wrong_proof_evidence = _seal_evidence(wrong_proof)
        with self.assertRaisesRegex(
            FertigAtlasVerificationError, "differs from ProbeSpec evidence"
        ):
            verify_fertig_measurement(
                _QUESTION,
                wrong_proof,
                replace(
                    measurement,
                    probe=wrong_proof.probe_identity,
                    evidence_sha256=str(wrong_proof_evidence["sha256"]),
                ),
                evidence_document=wrong_proof_evidence,
            )

        wrong_label = replace(
            probe_spec,
            semantic_label="arithmetic/incorrect-label",
        )
        wrong_label_evidence = _seal_evidence(wrong_label)
        with self.assertRaisesRegex(
            FertigAtlasVerificationError, "differs from ProbeSpec evidence"
        ):
            verify_fertig_measurement(
                _QUESTION,
                wrong_label,
                replace(
                    measurement,
                    observed_semantic_label="arithmetic/incorrect-label",
                    probe=wrong_label.probe_identity,
                    evidence_sha256=str(wrong_label_evidence["sha256"]),
                ),
                evidence_document=wrong_label_evidence,
            )
        with self.assertRaisesRegex(FertigAtlasVerificationError, "not eligible"):
            verify_fertig_measurement(
                _QUESTION,
                probe_spec,
                replace(measurement, observation_status="recorded"),
                evidence_document=evidence,
            )

        numeric_label = replace(probe_spec, semantic_label="2")
        with self.assertRaisesRegex(FertigAtlasVerificationError, "raw numeric answer"):
            verify_fertig_measurement(
                _QUESTION,
                numeric_label,
                replace(
                    measurement,
                    observed_semantic_label="2",
                    probe=numeric_label.probe_identity,
                ),
                evidence_document=evidence,
            )

        tampered_probe = replace(
            measurement.probe,
            token_sha256=_hash("tampered-token-identity"),
        )
        with self.assertRaisesRegex(
            FertigAtlasVerificationError, "ProbeIdentity differs"
        ):
            verify_fertig_measurement(
                _QUESTION,
                probe_spec,
                replace(measurement, probe=tampered_probe),
                evidence_document=evidence,
            )

    def test_full_probe_spec_and_cartography_evidence_are_bound(self) -> None:
        probe_spec, evidence, _placebo, measurement = _measurement_pair()
        changed_native = replace(
            probe_spec,
            native_head_crsa=Qwen38NativeHeadCrsa(alpha=0.02),
        )
        changed_window = replace(probe_spec, start_layer=26)

        for changed in (changed_native, changed_window):
            with self.subTest(spec=changed.as_record()):
                with self.assertRaisesRegex(
                    FertigAtlasVerificationError,
                    "evidence probe_spec differs",
                ):
                    verify_fertig_measurement(
                        _QUESTION,
                        changed,
                        measurement,
                        evidence_document=evidence,
                    )

        tampered = json.loads(json.dumps(evidence))
        tampered["body"]["probe_spec"]["start_layer"] = 26
        with self.assertRaisesRegex(
            FertigAtlasVerificationError,
            "evidence document SHA-256 mismatch",
        ):
            verify_fertig_measurement(
                _QUESTION,
                probe_spec,
                measurement,
                evidence_document=tampered,
            )

        resealed_changed_native = _seal_evidence(changed_native)
        with self.assertRaisesRegex(
            FertigAtlasVerificationError,
            "evidence SHA-256 differs from the measurement",
        ):
            verify_fertig_measurement(
                _QUESTION,
                changed_native,
                measurement,
                evidence_document=resealed_changed_native,
            )

    def test_formula_only_certificate_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(FertigAtlasVerificationError, "unsupported FERTIG"):
            fertig_label_evidence_sha256(_FORMULA_ONLY_QUESTION, _LABEL)

    def test_one_three_surface_replica_promotes_and_replays_idempotently(self) -> None:
        probe_spec, evidence, placebo, measurement = _measurement_pair()
        verification = verify_fertig_measurement(
            _QUESTION,
            probe_spec,
            measurement,
            evidence_document=evidence,
        )

        self.assertEqual(
            len(
                {
                    verification.label_evidence_sha256,
                    verification.candidate_verification_sha256,
                    verification.structural_audit_sha256,
                }
            ),
            3,
        )
        replica = verification.replica
        self.assertEqual(replica.verdict, "support")
        self.assertEqual(replica.semantic_label, _LABEL)
        self.assertEqual(replica.reputation_weight, 1.0)
        self.assertEqual(replica.probe_sha256, probe_spec.probe_identity.sha256)
        document = replica.to_document()
        self.assertEqual(document["body"]["security_claim"], "none")
        serialized = json.dumps(document, sort_keys=True)
        self.assertNotIn(_QUESTION, serialized)
        self.assertNotIn('"answer"', serialized)

        with tempfile.TemporaryDirectory(
            prefix=".fertig-atlas-verifier-", dir=Path.cwd()
        ) as temporary:
            atlas = SemanticWeightAtlas(
                temporary,
                model_pin=_PIN,
                tensor_plans=(_PLAN,),
            )
            atlas.append_measurement(placebo)
            atlas.append_measurement(measurement)
            first = append_and_promote_fertig_label(
                atlas,
                verification,
                policy=_POLICY,
            )
            self.assertTrue(first.replica_append.appended)
            self.assertTrue(first.promotion_append.appended)

            result = atlas.query_by_semantic_label(_LABEL)
            self.assertEqual(result.measurements, (measurement,))
            self.assertEqual(len(result.replicas), 1)
            self.assertEqual(len(result.promotions), 1)
            promotion = result.promotions[0]
            self.assertEqual(promotion.support_count, 1)
            self.assertAlmostEqual(promotion.support_weight, 1.0)
            self.assertEqual(promotion.to_document()["body"]["security_claim"], "none")

            replay = append_and_promote_fertig_label(
                atlas,
                verification,
                policy=_POLICY,
            )
            self.assertFalse(replay.replica_append.appended)
            self.assertFalse(replay.promotion_append.appended)
            self.assertTrue(atlas.verify())

    def test_conflicting_existing_verifier_identity_fails_before_mutation(self) -> None:
        probe_spec, evidence, placebo, measurement = _measurement_pair()
        verification = verify_fertig_measurement(
            _QUESTION,
            probe_spec,
            measurement,
            evidence_document=evidence,
        )
        changed_verifier_output = replace(
            verification.replica,
            replica_measurement_sha256=_hash("changed-verifier-measurement"),
            evidence_sha256=_hash("changed-verifier-evidence"),
        )
        with self.assertRaisesRegex(
            FertigAtlasVerificationError,
            "provenance/surface binding is invalid",
        ):
            replace(verification, replica=changed_verifier_output)

        with tempfile.TemporaryDirectory(
            prefix=".fertig-atlas-conflict-", dir=Path.cwd()
        ) as temporary:
            atlas = SemanticWeightAtlas(
                temporary,
                model_pin=_PIN,
                tensor_plans=(_PLAN,),
            )
            atlas.append_measurement(placebo)
            atlas.append_measurement(measurement)
            atlas.append_replica(changed_verifier_output)
            revision_before = atlas.revision()

            with self.assertRaisesRegex(
                FertigAtlasReplicaConflictError,
                "already submitted conflicting evidence",
            ):
                append_and_promote_fertig_label(
                    atlas,
                    verification,
                    policy=_POLICY,
                )

            self.assertEqual(atlas.revision(), revision_before)
            result = atlas.query_by_semantic_label(_LABEL)
            self.assertEqual(result.replicas, (changed_verifier_output,))
            self.assertEqual(result.promotions, ())


if __name__ == "__main__":
    unittest.main()
