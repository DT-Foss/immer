from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib
import unittest

from immer.runtimes.deepseek_v4.causal_weights import TensorRangePlan
from immer.runtimes.ooe.controller import ActionExecution, VerifiedTeacherTransition
from immer.runtimes.ooe.execution_learning import (
    ATLAS_MEASUREMENT_VERIFIER_SHA256,
    ActionAuthorityReceipt,
    ControllerActionTraceReceipt,
    ExecutionLearningReceipt,
    ExecutionQualityReceipt,
    ZERO_SHA256,
)
from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.predictive_quotient import (
    CENSORED_STATE_SHA256,
    PREDICTIVE_OUTCOME_SCHEMA_SHA256,
    TERMINAL_STATE_SHA256,
    PredictiveQuotient,
    PredictiveQuotientCoverageError,
    PredictiveQuotientIntegrityError,
    PredictiveState,
    PredictiveTransition,
    PredictiveValidationReceipt,
    VerifiedExecutionOutcome,
    build_predictive_quotient,
    derive_execution_transitions,
    derive_local_agent_transitions,
    validate_predictive_quotient,
    verify_predictive_quotient,
    verify_predictive_validation,
)
from immer.runtimes.ooe.qwen_bridge import QwenOoeFeatureReceipt
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


def _outcome(action: str) -> VerifiedExecutionOutcome:
    return VerifiedExecutionOutcome(
        action=action,
        executor_sha256=_hash(f"executor:{action}"),
        action_verifier_sha256=_hash(f"action-verifier:{action}"),
        quality_verifier_sha256=_hash("quality-verifier"),
        qwen_forwards=0,
        teacher_baseline_qwen_forwards=1,
    )


def _state(name: str, action: str) -> PredictiveState:
    return PredictiveState(_hash(f"site:{name}"), action)


def _transition(
    name: str,
    state: PredictiveState,
    outcome: VerifiedExecutionOutcome,
    successor: PredictiveState | None,
    *,
    temporal: int,
) -> PredictiveTransition:
    return PredictiveTransition(
        state=state,
        outcome=outcome,
        successor_state=successor,
        terminal=successor is None,
        censored=False,
        temporal_index=temporal,
        learning_receipt_sha256=_hash(f"learning:{name}"),
        trace_sha256=_hash(f"trace:{name}"),
        feature_receipt_sha256=_hash(f"feature:{name}"),
        execution_sha256=_hash(f"execution:{name}"),
        quality_sha256=_hash(f"quality:{name}"),
        teacher_transition_sha256=_hash(f"teacher:{name}"),
        action_authority_sha256=_hash(f"authority:{name}"),
    )


def _merge_system(*, negative: bool = False, suffix: str = "train"):
    a = _state("a", "qwen_fallback")
    b = _state("b", "qwen_fallback")
    c = _state("c", "probe_coordinate")
    d = _state("d", "probe_coordinate")
    probe = _outcome("probe_coordinate")
    mount = _outcome("mount_organ")
    final_d = _outcome("execute_fertig") if negative else mount
    return (
        _transition(f"{suffix}:a", a, probe, c, temporal=0),
        _transition(f"{suffix}:b", b, probe, d, temporal=1),
        _transition(f"{suffix}:c", c, mount, None, temporal=2),
        _transition(f"{suffix}:d", d, final_d, None, temporal=3),
    )


class PredictiveQuotientTests(unittest.TestCase):
    def test_empirical_joint_ratios_are_exact_and_drive_merge(self) -> None:
        a = _state("ratio-a", "qwen_fallback")
        b = _state("ratio-b", "qwen_fallback")
        probe = _outcome("probe_coordinate")
        mount = _outcome("mount_organ")
        transitions = tuple(
            _transition(name, state, outcome, None, temporal=index)
            for index, (name, state, outcome) in enumerate(
                (
                    ("ratio:a:p0", a, probe),
                    ("ratio:a:p1", a, probe),
                    ("ratio:a:m0", a, mount),
                    ("ratio:b:p0", b, probe),
                    ("ratio:b:p1", b, probe),
                    ("ratio:b:p2", b, probe),
                    ("ratio:b:p3", b, probe),
                    ("ratio:b:m0", b, mount),
                    ("ratio:b:m1", b, mount),
                )
            )
        )
        quotient = build_predictive_quotient(transitions)

        self.assertEqual(quotient.class_count, 1)
        self.assertEqual(quotient.compression, Fraction(2))
        probabilities = {
            row.outcome_sha256: row.fraction
            for row in quotient.classes[0].joint_distribution
        }
        self.assertEqual(probabilities[probe.sha256], Fraction(2, 3))
        self.assertEqual(probabilities[mount.sha256], Fraction(1, 3))

        changed = transitions[:-1] + (
            _transition("ratio:b:p4", b, probe, None, temporal=99),
        )
        self.assertEqual(build_predictive_quotient(changed).class_count, 2)

    def test_exact_recursive_merge_terminal_and_input_order_invariance(self) -> None:
        transitions = _merge_system()
        quotient = build_predictive_quotient(transitions)
        shuffled = build_predictive_quotient(tuple(reversed(transitions)))

        self.assertEqual(quotient.state_count, 4)
        self.assertEqual(quotient.class_count, 2)
        self.assertEqual(quotient.compression, Fraction(2))
        self.assertEqual(quotient.to_document(), shuffled.to_document())
        self.assertEqual(quotient.sha256, shuffled.sha256)
        self.assertTrue(verify_predictive_quotient(quotient))
        self.assertEqual(
            PredictiveQuotient.from_document(quotient.to_document()), quotient
        )

        mapping = quotient.state_to_class
        a, b, c, d = (row.state for row in transitions)
        self.assertEqual(mapping[a], mapping[b])
        self.assertEqual(mapping[c], mapping[d])
        self.assertNotEqual(mapping[a], mapping[c])
        self.assertEqual(sum(1 for row in quotient.transitions if row.terminal), 2)
        terminal_probabilities = (
            probability
            for predictive_class in quotient.classes
            for probability in predictive_class.joint_distribution
            if probability.terminal
        )
        self.assertTrue(
            all(
                row.successor_class_sha256 == TERMINAL_STATE_SHA256
                for row in terminal_probabilities
            )
        )
        self.assertEqual(
            quotient.outcome_schema_sha256, PREDICTIVE_OUTCOME_SCHEMA_SHA256
        )

    def test_different_outcome_and_recursive_successor_do_not_merge(self) -> None:
        quotient = build_predictive_quotient(_merge_system(negative=True))
        mapping = quotient.state_to_class
        states = [row.state for row in _merge_system(negative=True)]

        self.assertNotEqual(mapping[states[2]], mapping[states[3]])
        self.assertNotEqual(mapping[states[0]], mapping[states[1]])
        self.assertEqual(quotient.class_count, 4)
        self.assertEqual(quotient.compression, Fraction(1))

    def test_heldout_exact_tv_compression_replay_and_tamper(self) -> None:
        quotient = build_predictive_quotient(_merge_system())
        heldout = _merge_system(suffix="heldout")
        validation = validate_predictive_quotient(quotient, heldout)

        self.assertEqual(validation.max_total_variation, Fraction())
        self.assertEqual(validation.compression, Fraction(2))
        self.assertEqual(
            validate_predictive_quotient(quotient, tuple(reversed(heldout))),
            validation,
        )
        with self.assertRaises(PredictiveQuotientCoverageError):
            validate_predictive_quotient(quotient, (heldout[0],))
        censored_validation = validate_predictive_quotient(
            quotient,
            (heldout[0],),
            require_full_state_coverage=False,
        )
        self.assertEqual(censored_validation.max_total_variation, Fraction())
        self.assertEqual(censored_validation.state_coverage, Fraction(1, 4))
        self.assertEqual(censored_validation.class_coverage, Fraction(1, 2))
        self.assertEqual(
            PredictiveValidationReceipt.from_document(
                censored_validation.to_document()
            ),
            censored_validation,
        )
        self.assertTrue(verify_predictive_validation(quotient, heldout, validation))
        self.assertEqual(
            PredictiveValidationReceipt.from_document(validation.to_document()),
            validation,
        )
        changed = list(heldout)
        changed[2] = replace(changed[2], outcome=_outcome("execute_fertig"))
        changed_validation = validate_predictive_quotient(quotient, tuple(changed))
        self.assertEqual(changed_validation.max_total_variation, Fraction(1))

        tampered = quotient.to_document()
        tampered["body"]["assignments"][0]["class_sha256"] = _hash("forged-class")
        tampered["sha256"] = hashlib.sha256(
            canonical_json_bytes(tampered["body"])
        ).hexdigest()
        with self.assertRaises(PredictiveQuotientIntegrityError):
            PredictiveQuotient.from_document(tampered)

        malformed_transition = heldout[0].to_document()
        malformed_transition["body"]["successor_state"] = None
        malformed_transition["body"]["successor_state_sha256"] = _hash(
            "missing-successor"
        )
        malformed_transition["sha256"] = hashlib.sha256(
            canonical_json_bytes(malformed_transition["body"])
        ).hexdigest()
        with self.assertRaises(PredictiveQuotientIntegrityError):
            PredictiveTransition.from_document(malformed_transition)


_CODE = "e" * 40
_PLAN = TensorRangePlan(
    name="model.layers.18.mlp.gate_proj.weight",
    dtype="BF16",
    shape=(16, 4),
    shard="model-00007-of-00018.safetensors",
    absolute_offset=8192,
    length=128,
)
_PIN = ModelPin(
    repo_id="Qwen/Qwen3.8-27B",
    revision="0123456789abcdef",
    bundle_fingerprint=_hash("bundle"),
    bundle_manifest_sha256=_hash("bundle-manifest"),
    code_revision=_CODE,
)
_RUNTIME = RuntimeProvenance(
    code_revision=_CODE,
    source_manifest_sha256=_hash("source"),
    dependency_manifest_sha256=_hash("dependencies"),
    runtime_configuration_sha256=_hash("configuration"),
    platform_sha256=_hash("platform"),
)
_WEIGHT_GRAPH = GraphRevision(11, _hash("weight-graph"))
_ATLAS_GRAPH = GraphRevision(23, _hash("atlas-graph"))
_ATLAS_HEAD_SHA256 = _hash("authenticated-atlas-head")


def _atlas_authentication(measurement: MeasurementReceipt) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "atlas_head_sha256": _ATLAS_HEAD_SHA256,
                "atlas_model_pin_sha256": measurement.model_pin.sha256,
                "measurement_atlas_revision_sha256": (
                    measurement.atlas_head_revision.sha256
                ),
                "measurement_sha256": measurement.sha256,
                "schema": "immer-ooe-live-atlas-authentication/v1",
                "verifier_sha256": ATLAS_MEASUREMENT_VERIFIER_SHA256,
            }
        )
    ).hexdigest()


def _real_learning_receipt(
    *,
    index: int,
    site: int,
    source_action: str,
    target_action: str,
) -> ExecutionLearningReceipt:
    coordinate = WeightCoordinate.from_plan(
        _PLAN,
        layer=18,
        module="model.layers.18.mlp.gate_proj",
        row_start=site * 2,
        row_end=site * 2 + 2,
    )
    measurement = MeasurementReceipt(
        model_pin=_PIN,
        coordinate=coordinate,
        probe=ProbeIdentity(
            question_sha256=_hash(f"question:{index}"),
            token_sha256=_hash(f"tokens:{index}"),
            family_sha256=_hash("family"),
            label_source_sha256=_hash("label-source"),
        ),
        intervention=InterventionIdentity(
            mode="native", configuration_sha256=_hash(f"native:{index}")
        ),
        observation_status="recorded",
        observed_semantic_label=None,
        hidden_sha256=_hash(f"hidden:{index}"),
        activation_sha256=_hash(f"activation:{index}"),
        logits_sha256=_hash(f"logits:{index}"),
        state_sha256=_hash(f"state:{index}"),
        access_trace_sha256=_hash(f"access:{index}"),
        evidence_sha256=_hash(f"evidence:{index}"),
        weight_rail_revision=_WEIGHT_GRAPH,
        atlas_head_revision=_ATLAS_GRAPH,
        numeric_summaries=(
            NumericSummary(
                metric="activation-rms",
                count=2,
                total=2.5 + 2 * index,
                total_squares=3.25 + 5 * index + 2 * index * index,
                minimum=1.0 + index,
                maximum=1.5 + index,
            ),
        ),
        placebo_effects=(),
        runtime=_RUNTIME,
    )
    authority = ActionAuthorityReceipt(
        action=target_action,
        model_pin_sha256=_PIN.sha256,
        weight_graph_revision_sha256=_WEIGHT_GRAPH.sha256,
        executor_sha256=_hash(f"executor:{target_action}"),
        action_verifier_name=f"action:{target_action}",
        action_verifier_sha256=_hash(f"action-verifier:{target_action}"),
        quality_verifier_name="exact-quality",
        quality_verifier_sha256=_hash("quality-verifier"),
        evidence_sha256=_hash(f"authority-evidence:{target_action}"),
    )
    atlas_authentication = _atlas_authentication(measurement)
    feature = QwenOoeFeatureReceipt.from_measurement(
        measurement,
        temporal_index=index,
        verifier_sha256s=(
            ATLAS_MEASUREMENT_VERIFIER_SHA256,
            atlas_authentication,
            authority.sha256,
            authority.action_verifier_sha256,
            authority.quality_verifier_sha256,
        ),
        evidence_sha256s=(
            atlas_authentication,
            authority.sha256,
            authority.evidence_sha256,
        ),
        o1_surprise=0.25,
        o1_learning_progress=0.75,
    )
    execution = ActionExecution(
        feature_receipt_sha256=feature.sha256,
        action=target_action,
        executor_sha256=authority.executor_sha256,
        verifier_sha256=authority.action_verifier_sha256,
        evidence_sha256=authority.evidence_sha256,
        quality_sha256=_hash(f"execution-quality:{index}"),
        result={"action": target_action, "verified": True},
        quality_verified=True,
        qwen_forwards=0,
        teacher_baseline_qwen_forwards=1,
    )
    quality = ExecutionQualityReceipt(
        feature_receipt_sha256=feature.sha256,
        execution_sha256=execution.sha256,
        execution_quality_sha256=execution.quality_sha256,
        verifier_name=authority.quality_verifier_name,
        verifier_sha256=authority.quality_verifier_sha256,
        verified=True,
    )
    transition = VerifiedTeacherTransition(
        feature_receipt_sha256=feature.sha256,
        site_identity_sha256=feature.site_identity.sha256,
        source_action=source_action,
        target_action=execution.action,
        verifier_sha256=authority.quality_verifier_sha256,
        evidence_sha256=authority.evidence_sha256,
        quality_sha256=quality.sha256,
    )
    return ExecutionLearningReceipt(
        measurement=measurement,
        action_authority=authority,
        feature=feature,
        execution=execution,
        quality=quality,
        transition=transition,
        atlas_head_sha256=_ATLAS_HEAD_SHA256,
        atlas_authentication_sha256=atlas_authentication,
    )


class RealExecutionLearningIntegrationTests(unittest.TestCase):
    def test_receipt_trace_chain_derives_state_action_transitions(self) -> None:
        receipts = (
            _real_learning_receipt(
                index=0,
                site=0,
                source_action="qwen_fallback",
                target_action="probe_coordinate",
            ),
            _real_learning_receipt(
                index=1,
                site=1,
                source_action="probe_coordinate",
                target_action="mount_organ",
            ),
        )
        first_after = _hash("controller:1")
        traces = (
            ControllerActionTraceReceipt(
                transaction_sha256=_hash("transaction:0"),
                ordinal=0,
                temporal_index=0,
                previous_trace_sha256=ZERO_SHA256,
                previous_stream_head_sha256=ZERO_SHA256,
                measurement_sha256=receipts[0].measurement.sha256,
                learning_receipt_sha256=receipts[0].sha256,
                source_action="qwen_fallback",
                target_action="probe_coordinate",
                controller_snapshot_before_sha256=_hash("controller:0"),
                controller_snapshot_after_sha256=first_after,
            ),
            ControllerActionTraceReceipt(
                transaction_sha256=_hash("transaction:1"),
                ordinal=1,
                temporal_index=1,
                previous_trace_sha256=ZERO_SHA256,
                previous_stream_head_sha256=ZERO_SHA256,
                measurement_sha256=receipts[1].measurement.sha256,
                learning_receipt_sha256=receipts[1].sha256,
                source_action="probe_coordinate",
                target_action="mount_organ",
                controller_snapshot_before_sha256=first_after,
                controller_snapshot_after_sha256=_hash("controller:2"),
            ),
        )
        # The second trace must carry the first trace hash; construct it only
        # after the first receipt is sealed.
        traces = (
            traces[0],
            replace(
                traces[1],
                previous_trace_sha256=traces[0].sha256,
                previous_stream_head_sha256=traces[0].sha256,
            ),
        )
        transitions = derive_execution_transitions(receipts, traces, terminal=True)

        self.assertEqual(len(transitions), 2)
        self.assertEqual(
            transitions[0].state.site_identity_sha256,
            receipts[0].feature.site_identity.sha256,
        )
        self.assertEqual(transitions[0].state.source_action, "qwen_fallback")
        self.assertEqual(transitions[0].outcome.action, "probe_coordinate")
        self.assertEqual(
            transitions[0].successor_state.site_identity_sha256,
            receipts[1].feature.site_identity.sha256,
        )
        self.assertTrue(transitions[1].terminal)
        self.assertIsNone(transitions[1].successor_state)
        quotient = build_predictive_quotient(transitions)
        self.assertTrue(quotient.verify_or_raise())
        censored = derive_execution_transitions(receipts, traces, terminal=False)
        self.assertEqual(len(censored), 2)
        self.assertFalse(censored[0].terminal)
        self.assertTrue(censored[1].censored)
        self.assertEqual(
            censored[1].as_record()["successor_state_sha256"],
            CENSORED_STATE_SHA256,
        )
        one_step_censored = derive_execution_transitions(
            receipts[:1], traces[:1], terminal=False
        )
        self.assertEqual(len(one_step_censored), 1)
        self.assertTrue(one_step_censored[0].censored)
        censored_quotient = build_predictive_quotient(one_step_censored)
        censored_probability = censored_quotient.classes[0].joint_distribution[0]
        self.assertTrue(censored_probability.censored)
        self.assertFalse(censored_probability.terminal)
        self.assertEqual(
            censored_probability.successor_class_sha256,
            CENSORED_STATE_SHA256,
        )
        local = derive_local_agent_transitions(receipts, traces)
        self.assertEqual(
            local[0].successor_state,
            PredictiveState(
                receipts[0].feature.site_identity.sha256,
                receipts[0].execution.action,
            ),
        )
        self.assertNotEqual(
            local[0].successor_state.site_identity_sha256,
            transitions[0].successor_state.site_identity_sha256,
        )
        self.assertEqual(
            local[1].successor_state,
            PredictiveState(
                receipts[1].feature.site_identity.sha256,
                receipts[1].execution.action,
            ),
        )
        self.assertFalse(local[1].terminal)
        self.assertFalse(local[1].censored)

        broken = (
            traces[0],
            replace(
                traces[1],
                previous_trace_sha256=ZERO_SHA256,
                previous_stream_head_sha256=ZERO_SHA256,
            ),
        )
        with self.assertRaises(PredictiveQuotientIntegrityError):
            derive_execution_transitions(receipts, broken, terminal=True)


if __name__ == "__main__":
    unittest.main()
