from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest

from immer.runtimes.deepseek_v4.causal_weights import TensorRangePlan
from immer.runtimes.ooe.controller import (
    CONTROLLER_STATE_NAME,
    ActionExecution,
    ControllerConfig,
    OoeController,
    OoeControllerAmbiguityError,
    OoeControllerIntegrityError,
    OoeControllerStaleError,
    OoePromotionError,
    VerifiedTeacherTransition,
)
from immer.runtimes.ooe.crystal import CrystalStore, CrystalTamperError
from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.qwen_bridge import (
    ACTION_SCHEMA_SHA256,
    OOE_ACTIONS,
    QwenOoeBridgeIntegrityError,
    QwenOoeFeatureReceipt,
)
from immer.runtimes.qwen3_8.semantic_atlas import (
    GraphRevision,
    InterventionIdentity,
    MeasurementReceipt,
    ModelPin,
    NumericSummary,
    PlaceboEffect,
    ProbeIdentity,
    RuntimeProvenance,
    WeightCoordinate,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


_CODE_REVISION = "c" * 40
_PLAN = TensorRangePlan(
    name="model.layers.18.mlp.gate_proj.weight",
    dtype="BF16",
    shape=(16, 2),
    shard="model-00007-of-00018.safetensors",
    absolute_offset=8192,
    length=64,
)
_PIN = ModelPin(
    repo_id="Qwen/Qwen3.8-27B",
    revision="0123456789abcdef",
    bundle_fingerprint=_hash("qwen-bundle"),
    bundle_manifest_sha256=_hash("qwen-bundle-manifest"),
    code_revision=_CODE_REVISION,
)
_RUNTIME = RuntimeProvenance(
    code_revision=_CODE_REVISION,
    source_manifest_sha256=_hash("runtime-source"),
    dependency_manifest_sha256=_hash("runtime-dependencies"),
    runtime_configuration_sha256=_hash("runtime-configuration"),
    platform_sha256=_hash("runtime-platform"),
)
_WEIGHT_GRAPH = GraphRevision(42, _hash("weight-graph-42"))
_ATLAS_GRAPH = GraphRevision(93, _hash("atlas-graph-93"))
_VERIFIER = _hash("fertig-verifier-v1")
_EXECUTOR = _hash("ooe-action-executor-v1")


def _coordinate(site: int) -> WeightCoordinate:
    start = site * 4
    return WeightCoordinate.from_plan(
        _PLAN,
        layer=18,
        module="model.layers.18.mlp.gate_proj",
        row_start=start,
        row_end=start + 4,
    )


def _summary(metric: str, values: tuple[float, ...]) -> NumericSummary:
    return NumericSummary(
        metric=metric,
        count=len(values),
        total=sum(values),
        total_squares=sum(value * value for value in values),
        minimum=min(values),
        maximum=max(values),
    )


def _measurement(
    *,
    site: int,
    sample: int,
    signal: float,
    weight_graph: GraphRevision = _WEIGHT_GRAPH,
    atlas_graph: GraphRevision = _ATLAS_GRAPH,
) -> MeasurementReceipt:
    summaries = (
        _summary("activation_delta", (signal - 0.25, signal + 0.25)),
        _summary("hidden_rms", (abs(signal) + 0.5, abs(signal) + 1.0)),
    )
    placebo_mean = signal - 0.5
    effect = PlaceboEffect(
        metric="activation_delta",
        placebo_measurement_sha256=_hash(f"placebo-{site}-{sample}"),
        observed_mean=signal,
        placebo_mean=placebo_mean,
        delta=0.5,
    )
    return MeasurementReceipt(
        model_pin=_PIN,
        coordinate=_coordinate(site),
        probe=ProbeIdentity(
            question_sha256=_hash(f"question-{site}-{sample}"),
            token_sha256=_hash(f"tokens-{site}-{sample}"),
            family_sha256=_hash(f"family-{site}"),
            label_source_sha256=_hash("label-source"),
        ),
        intervention=InterventionIdentity(
            mode="native",
            configuration_sha256=_hash(f"native-{site}-{sample}"),
        ),
        observation_status="recorded",
        observed_semantic_label=None,
        hidden_sha256=_hash(f"hidden-{site}-{sample}"),
        activation_sha256=_hash(f"activation-{site}-{sample}"),
        logits_sha256=_hash(f"logits-{site}-{sample}"),
        state_sha256=_hash(f"state-{site}-{sample}"),
        access_trace_sha256=_hash(f"access-{site}-{sample}"),
        evidence_sha256=_hash(f"evidence-{site}-{sample}"),
        weight_rail_revision=weight_graph,
        atlas_head_revision=atlas_graph,
        numeric_summaries=summaries,
        placebo_effects=(effect,),
        runtime=_RUNTIME,
    )


def _feature(
    temporal_index: int,
    *,
    site: int = 0,
    source: int = 0,
    signal: float | None = None,
    weight_graph: GraphRevision = _WEIGHT_GRAPH,
    atlas_graph: GraphRevision = _ATLAS_GRAPH,
) -> tuple[QwenOoeFeatureReceipt, MeasurementReceipt]:
    measurement = _measurement(
        site=site,
        sample=source,
        signal=(2.0 + source * 0.1 if signal is None else signal),
        weight_graph=weight_graph,
        atlas_graph=atlas_graph,
    )
    receipt = QwenOoeFeatureReceipt.from_measurement(
        measurement,
        temporal_index=temporal_index,
        verifier_sha256s=(_VERIFIER,),
        evidence_sha256s=(_hash(f"supplement-{site}-{source}"),),
        o1_surprise=0.25 + source * 0.01,
        o1_learning_progress=1.0 + source * 0.02,
    )
    return receipt, measurement


def _transition(
    receipt: QwenOoeFeatureReceipt,
    source: str,
    target: str,
) -> VerifiedTeacherTransition:
    return VerifiedTeacherTransition(
        feature_receipt_sha256=receipt.sha256,
        site_identity_sha256=receipt.site_identity.sha256,
        source_action=source,
        target_action=target,
        verifier_sha256=_VERIFIER,
        evidence_sha256=receipt.evidence_sha256s[0],
        quality_sha256=_hash(f"quality:{receipt.sha256}:{source}:{target}"),
    )


def _execution(
    receipt: QwenOoeFeatureReceipt,
    action: str,
    *,
    qwen_forwards: int = 0,
    quality_verified: bool = True,
) -> ActionExecution:
    return ActionExecution(
        feature_receipt_sha256=receipt.sha256,
        action=action,
        executor_sha256=_EXECUTOR,
        verifier_sha256=_VERIFIER,
        evidence_sha256=receipt.evidence_sha256s[0],
        quality_sha256=_hash(f"execution-quality:{receipt.sha256}:{action}"),
        result={"action": action, "feature_receipt_sha256": receipt.sha256},
        quality_verified=quality_verified,
        qwen_forwards=qwen_forwards,
        teacher_baseline_qwen_forwards=1,
    )


class QwenOoeFeatureReceiptTests(unittest.TestCase):
    def test_exact_bindings_numeric_only_and_tamper_rejection(self) -> None:
        receipt, measurement = _feature(7, source=2)

        self.assertEqual(receipt.measurement_sha256, measurement.sha256)
        self.assertEqual(receipt.model_pin_sha256, measurement.model_pin.sha256)
        self.assertEqual(
            receipt.weight_coordinate_sha256, measurement.coordinate.sha256
        )
        self.assertEqual(
            receipt.weight_graph_revision_sha256,
            measurement.weight_rail_revision.sha256,
        )
        self.assertEqual(
            receipt.atlas_graph_revision_sha256,
            measurement.atlas_head_revision.sha256,
        )
        self.assertEqual(receipt.probe, measurement.probe)
        self.assertEqual(receipt.action_schema_sha256, ACTION_SCHEMA_SHA256)
        self.assertEqual(len(receipt.feature_sketch), 64)
        self.assertTrue(all(-1.0 <= value <= 1.0 for value in receipt.feature_sketch))
        receipt.validate_measurement(measurement)

        encoded = receipt.to_document()
        canonical = str(encoded)
        self.assertNotIn("activation_delta", canonical)
        self.assertNotIn("hidden_rms", canonical)
        self.assertEqual(QwenOoeFeatureReceipt.from_document(encoded), receipt)

        tampered = receipt.to_document()
        tampered["body"]["feature_sketch"][0] += 0.01
        with self.assertRaisesRegex(QwenOoeBridgeIntegrityError, "SHA-256 mismatch"):
            QwenOoeFeatureReceipt.from_document(tampered)

    def test_sketch_is_deterministic_and_measurement_bound(self) -> None:
        first, measurement = _feature(11, source=1)
        second = QwenOoeFeatureReceipt.from_measurement(
            measurement,
            temporal_index=11,
            verifier_sha256s=(_VERIFIER,),
            evidence_sha256s=(_hash("supplement-0-1"),),
            o1_surprise=0.26,
            o1_learning_progress=1.02,
        )
        self.assertEqual(first, second)
        changed = replace(measurement, hidden_sha256=_hash("changed-hidden"))
        with self.assertRaisesRegex(QwenOoeBridgeIntegrityError, "does not match"):
            first.validate_measurement(changed)


class OoeControllerTests(unittest.TestCase):
    def _controller(
        self,
        root: Path,
        *,
        executor_qwen_forwards: int = 0,
        executor_actions: tuple[str, ...] = OOE_ACTIONS,
        executor_quality: bool = True,
        min_promoted_sources: int = 1,
        atlas_revision_verifier=None,
    ) -> OoeController:
        executors = {
            action: (
                lambda receipt, action=action: _execution(
                    receipt,
                    action,
                    qwen_forwards=executor_qwen_forwards,
                    quality_verified=executor_quality,
                )
            )
            for action in executor_actions
        }
        return OoeController(
            model_pin_sha256=_PIN.sha256,
            weight_graph_revision_sha256=_WEIGHT_GRAPH.sha256,
            atlas_graph_revision=_ATLAS_GRAPH,
            crystal_store=CrystalStore(root),
            config=ControllerConfig(
                replicas=12,
                replica_fanout=4,
                min_coverage_per_source=1,
                min_promoted_sources=min_promoted_sources,
                router_radius=0.75,
                router_min_margin=0.0,
                token_min_confidence=0.02,
                consensus_tolerance=2e-5,
                consensus_max_rounds=4096,
                reservoir_size=16,
            ),
            action_executors=executors,
            atlas_revision_verifier=atlas_revision_verifier,
        )

    @staticmethod
    def _target(source_index: int, *, shift: int = 1) -> str:
        # Keep qwen_fallback in the source-state coverage while teaching the
        # Crystal four locally executable actions.  Predicting qwen_fallback
        # correctly still consumes a Qwen forward and therefore cannot count
        # as a saved forward in the warm proof.
        return OOE_ACTIONS[(source_index + shift) % (len(OOE_ACTIONS) - 1)]

    def test_action_execution_is_result_bound_and_canonical(self) -> None:
        receipt, _ = _feature(0)
        execution = _execution(receipt, OOE_ACTIONS[1])
        execution.assert_bound(receipt, OOE_ACTIONS[1])
        self.assertEqual(
            ActionExecution.from_document(execution.to_document()), execution
        )
        self.assertEqual(execution.saved_qwen_forwards, 1)

        tampered = execution.to_document()
        tampered["body"]["result"]["action"] = OOE_ACTIONS[2]
        with self.assertRaisesRegex(OoeControllerIntegrityError, "SHA-256 mismatch"):
            ActionExecution.from_document(tampered)

    def test_warm_accounting_commits_or_rejects_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = self._controller(Path(tmp))
            _, temporal = self._train_site(
                controller,
                temporal_start=0,
                site=0,
                shift=1,
                signal=2.0,
            )
            first, _ = _feature(temporal, source=0, signal=2.0)
            pending = controller.try_warm(
                first,
                OOE_ACTIONS[0],
                quality_verifier=lambda _feature, execution: execution.quality_verified,
            )
            self.assertEqual(pending.origin, "crystal")
            self.assertIsNotNone(pending.warm_transaction_sha256)
            self.assertEqual(controller.metrics.saved_qwen_forwards, 0)
            self.assertEqual(controller.metrics.crystal_executions, 0)

            committed = controller.commit_warm(pending)
            self.assertEqual(committed.disposition, "committed")
            self.assertEqual(committed.saved_qwen_forwards, 1)
            self.assertEqual(controller.commit_warm(pending), committed)
            self.assertEqual(controller.metrics.saved_qwen_forwards, 1)
            self.assertEqual(controller.metrics.crystal_executions, 1)
            with self.assertRaisesRegex(
                OoeControllerIntegrityError, "cannot reject a committed"
            ):
                controller.reject_warm(pending)

            second, _ = _feature(temporal + 1, source=1, signal=2.01)
            rejected_decision = controller.try_warm(
                second,
                OOE_ACTIONS[1],
                quality_verifier=lambda _feature, execution: execution.quality_verified,
            )
            rejected = controller.reject_warm(rejected_decision)
            self.assertEqual(rejected.disposition, "rejected")
            self.assertEqual(controller.reject_warm(rejected_decision), rejected)
            self.assertEqual(controller.metrics.saved_qwen_forwards, 1)
            self.assertEqual(controller.metrics.quality_failures, 1)
            with self.assertRaisesRegex(
                OoeControllerIntegrityError, "cannot commit a rejected"
            ):
                controller.commit_warm(rejected_decision)

            controller.save_snapshot()
            restored = OoeController.restore(
                crystal_store=controller.crystal_store,
                action_executors=controller._executors,
            )
            self.assertEqual(restored.commit_warm(pending), committed)
            self.assertEqual(restored.reject_warm(rejected_decision), rejected)
            self.assertEqual(restored.snapshot_bytes(), controller.snapshot_bytes())

    def test_prompt_feature_lookup_is_ordered_and_site_ambiguity_is_explicit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = self._controller(Path(tmp))
            first, _ = _feature(0, site=0, source=0, signal=2.0)
            controller.ingest_teacher(
                first,
                _transition(first, OOE_ACTIONS[0], OOE_ACTIONS[1]),
            )
            shared_question = first.probe.question_sha256

            second_measurement = _measurement(
                site=1,
                sample=0,
                signal=-20.0,
            )
            second_measurement = replace(
                second_measurement,
                probe=ProbeIdentity(
                    question_sha256=shared_question,
                    token_sha256=_hash("second-site-token"),
                    family_sha256=_hash("second-site-family"),
                    label_source_sha256=_hash("label-source"),
                ),
            )
            second = QwenOoeFeatureReceipt.from_measurement(
                second_measurement,
                temporal_index=1,
                verifier_sha256s=(_VERIFIER,),
                o1_surprise=0.5,
                o1_learning_progress=1.5,
            )
            controller.ingest_teacher(
                second,
                _transition(second, OOE_ACTIONS[0], OOE_ACTIONS[2]),
            )

            matches = controller.feature_receipts_for_prompt(shared_question)
            self.assertEqual(matches, (first, second))
            self.assertEqual(
                controller.latest_feature_receipt_for_prompt(
                    shared_question,
                    first.probe.token_sha256,
                ),
                first,
            )
            self.assertIsNone(
                controller.latest_feature_receipt_for_prompt(_hash("unknown-prompt"))
            )
            with self.assertRaisesRegex(
                OoeControllerAmbiguityError, "multiple OoE weight sites"
            ):
                controller.latest_feature_receipt_for_prompt(shared_question)

    def _train_site(
        self,
        controller: OoeController,
        *,
        temporal_start: int,
        site: int,
        shift: int,
        signal: float,
    ) -> tuple[str, int]:
        site_sha256 = ""
        for source_index, source in enumerate(OOE_ACTIONS):
            receipt, _ = _feature(
                temporal_start + source_index,
                site=site,
                source=source_index,
                signal=signal + source_index * 0.01,
            )
            target = self._target(source_index, shift=shift)

            def teacher(
                feature: QwenOoeFeatureReceipt,
                actual_source: str,
                *,
                expected=target,
            ) -> VerifiedTeacherTransition:
                return _transition(feature, actual_source, expected)

            decision = controller.resolve(
                receipt,
                source,
                teacher=teacher,
                quality_verifier=lambda _feature, execution: execution.quality_verified,
                stream_id=f"cold-{site}-{source_index}",
            )
            self.assertEqual(decision.origin, "teacher")
            self.assertEqual(decision.action, target)
            site_sha256 = receipt.site_identity.sha256
        coverage = controller.coverage_receipt(site_sha256)
        publication = controller.promote(
            site_sha256,
            coverage_sha256=coverage.sha256,
            verifier_sha256s=coverage.verifier_sha256s,
        )
        self.assertEqual(
            publication.payload_sha256, controller._sites[site_sha256].crystal_sha256
        )
        self.assertEqual(
            controller.crystal_store.restore(
                publication.payload_sha256
            ).consensus_receipt["topology"],
            "ps-lifted",
        )
        lift = controller.crystal_store.restore(
            publication.payload_sha256
        ).consensus_receipt["adaptive_lift"]
        self.assertEqual(
            lift["formula"],
            "clip(0.85-0.05*log(lambda2),floor,ceiling)",
        )
        return site_sha256, temporal_start + len(OOE_ACTIONS)

    def test_cold_teacher_then_warm_crystal_saves_qwen_forwards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = self._controller(Path(tmp))
            site_sha256, next_temporal = self._train_site(
                controller,
                temporal_start=0,
                site=0,
                shift=1,
                signal=2.0,
            )
            self.assertEqual(controller.metrics.teacher_calls, len(OOE_ACTIONS))

            warm_actions = []
            for source_index, source in enumerate(OOE_ACTIONS):
                receipt, _ = _feature(
                    next_temporal + source_index,
                    site=0,
                    source=source_index,
                    signal=2.0 + source_index * 0.01,
                )
                expected = self._target(source_index)

                def forbidden_teacher(
                    _feature: QwenOoeFeatureReceipt, _source: str
                ) -> VerifiedTeacherTransition:
                    raise AssertionError("warm Crystal called Qwen teacher")

                decision = controller.resolve(
                    receipt,
                    source,
                    teacher=forbidden_teacher,
                    quality_verifier=lambda _feature, execution, expected=expected: (
                        execution.action == expected and execution.quality_verified
                    ),
                    stream_id=f"warm-{source_index}",
                )
                self.assertEqual(decision.origin, "crystal")
                self.assertTrue(decision.quality_verified)
                warm_actions.append(decision.action)

            self.assertEqual(
                warm_actions,
                [self._target(index) for index in range(len(OOE_ACTIONS))],
            )
            self.assertEqual(controller.metrics.teacher_calls, len(OOE_ACTIONS))
            self.assertEqual(controller.metrics.saved_qwen_forwards, len(OOE_ACTIONS))
            self.assertEqual(controller.metrics.quality_failures, 0)
            self.assertEqual(controller.metrics.verified_results, 2 * len(OOE_ACTIONS))
            self.assertEqual(site_sha256, controller.site_identity_sha256s[0])

            publication = controller.save_snapshot()
            restored = OoeController.restore(
                crystal_store=controller.crystal_store,
                expected_model_pin_sha256=_PIN.sha256,
                expected_weight_graph_revision_sha256=_WEIGHT_GRAPH.sha256,
                expected_atlas_graph_revision=_ATLAS_GRAPH,
                action_executors=controller._executors,
            )
            self.assertEqual(restored.snapshot_bytes(), controller.snapshot_bytes())
            self.assertEqual(
                hashlib.sha256(controller.snapshot_bytes()).hexdigest(),
                publication.payload_sha256,
            )

    def test_partial_promotion_executes_covered_source_and_abstains_elsewhere(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = self._controller(Path(tmp))
            receipt, _ = _feature(0)
            controller.ingest_teacher(
                receipt,
                _transition(receipt, OOE_ACTIONS[0], OOE_ACTIONS[1]),
            )
            coverage = controller.coverage_receipt(receipt.site_identity.sha256)
            with self.assertRaisesRegex(OoePromotionError, "verifier"):
                controller.promote(
                    receipt.site_identity.sha256,
                    coverage_sha256=coverage.sha256,
                    verifier_sha256s=(_hash("wrong-verifier"),),
                )

            # A cold partial state remains exact and uncalibrated on restore.
            self.assertIsNone(controller.router.calibration_sha256)
            controller.save_snapshot()
            restored = OoeController.restore(
                crystal_store=controller.crystal_store,
                action_executors=controller._executors,
            )
            self.assertIsNone(restored.router.calibration_sha256)
            self.assertEqual(restored.snapshot_bytes(), controller.snapshot_bytes())

            controller.promote(
                receipt.site_identity.sha256,
                coverage_sha256=coverage.sha256,
                verifier_sha256s=coverage.verifier_sha256s,
            )
            covered = controller.decide(receipt, OOE_ACTIONS[0])
            uncovered = controller.decide(receipt, OOE_ACTIONS[1])
            self.assertEqual(covered.origin, "crystal")
            self.assertEqual(covered.action, OOE_ACTIONS[1])
            self.assertEqual(uncovered.origin, "abstention")
            self.assertEqual(uncovered.reason, "uncovered-source-action")

            strict = self._controller(
                Path(tmp) / "strict",
                min_promoted_sources=len(OOE_ACTIONS),
            )
            strict_receipt, _ = _feature(0)
            strict.ingest_teacher(
                strict_receipt,
                _transition(strict_receipt, OOE_ACTIONS[0], OOE_ACTIONS[1]),
            )
            strict_coverage = strict.coverage_receipt(
                strict_receipt.site_identity.sha256
            )
            with self.assertRaisesRegex(OoePromotionError, "promotable sources"):
                strict.promote(
                    strict_receipt.site_identity.sha256,
                    coverage_sha256=strict_coverage.sha256,
                    verifier_sha256s=strict_coverage.verifier_sha256s,
                )

    def test_failed_execution_quality_repairs_through_teacher_without_savings(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = self._controller(Path(tmp), executor_quality=False)
            site_sha256, next_temporal = self._train_site(
                controller,
                temporal_start=0,
                site=0,
                shift=1,
                signal=2.0,
            )
            receipt, _ = _feature(next_temporal, source=0, signal=2.0)

            def teacher(
                feature: QwenOoeFeatureReceipt, source: str
            ) -> VerifiedTeacherTransition:
                return _transition(feature, source, OOE_ACTIONS[1])

            decision = controller.resolve(
                receipt,
                OOE_ACTIONS[0],
                teacher=teacher,
                quality_verifier=lambda _feature, _execution: True,
            )
            self.assertEqual(decision.origin, "teacher")
            self.assertEqual(decision.reason, "execution-quality-repair")
            self.assertEqual(len(controller._sites[site_sha256].history), 6)
            self.assertEqual(controller.metrics.saved_qwen_forwards, 0)
            self.assertEqual(controller.metrics.teacher_calls, 6)
            self.assertEqual(controller.metrics.quality_failures, 1)

    def test_partial_executor_and_missing_executor_never_claim_savings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cold = self._controller(Path(tmp) / "cold", executor_actions=())
            cold_receipt, _ = _feature(0)
            cold_miss = cold.try_warm(
                cold_receipt,
                OOE_ACTIONS[0],
                quality_verifier=lambda _feature, _execution: True,
            )
            self.assertEqual(cold_miss.origin, "abstention")
            self.assertEqual(cold_miss.reason, "untrained")
            self.assertEqual(cold.metrics.teacher_calls, 0)
            self.assertEqual(cold.site_identity_sha256s, ())

            partial = self._controller(
                Path(tmp) / "partial",
                executor_qwen_forwards=1,
            )
            _, temporal = self._train_site(
                partial,
                temporal_start=0,
                site=0,
                shift=1,
                signal=2.0,
            )
            partial_actions = []
            for offset, source_index in enumerate((2, 3)):
                receipt, _ = _feature(
                    temporal + offset,
                    source=source_index,
                    signal=2.0 + source_index * 0.01,
                )
                decision = partial.resolve(
                    receipt,
                    OOE_ACTIONS[source_index],
                    teacher=lambda _feature, _source: (_ for _ in ()).throw(
                        AssertionError("verified partial executor called teacher")
                    ),
                    quality_verifier=lambda _feature, execution: (
                        execution.quality_verified
                    ),
                )
                self.assertEqual(decision.origin, "crystal")
                partial_actions.append(decision.action)
            self.assertEqual(
                partial_actions,
                ["probe_coordinate", "restore_anchor"],
            )
            self.assertEqual(partial.metrics.executed_qwen_forwards, 2)
            self.assertEqual(partial.metrics.saved_qwen_forwards, 0)

            missing = self._controller(
                Path(tmp) / "missing",
                executor_actions=(),
            )
            _, temporal = self._train_site(
                missing,
                temporal_start=0,
                site=0,
                shift=1,
                signal=2.0,
            )
            receipt, _ = _feature(temporal, source=0, signal=2.0)
            expected = self._target(0)
            warm_miss = missing.try_warm(
                receipt,
                OOE_ACTIONS[0],
                quality_verifier=lambda _feature, _execution: True,
            )
            self.assertEqual(warm_miss.origin, "abstention")
            self.assertEqual(warm_miss.reason, "missing-action-executor")
            self.assertEqual(missing.metrics.teacher_calls, len(OOE_ACTIONS))
            self.assertEqual(
                len(missing._sites[receipt.site_identity.sha256].history),
                len(OOE_ACTIONS),
            )
            fallback = missing.resolve(
                receipt,
                OOE_ACTIONS[0],
                teacher=lambda feature, source: _transition(feature, source, expected),
                quality_verifier=lambda _feature, _execution: True,
            )
            self.assertEqual(fallback.origin, "teacher")
            self.assertEqual(fallback.reason, "missing-action-executor")
            self.assertEqual(missing.metrics.saved_qwen_forwards, 0)
            self.assertEqual(missing.metrics.executed_qwen_forwards, 0)

    def test_live_atlas_accepts_authenticated_history_and_rejects_unknown_or_fork(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            accepted_head = GraphRevision(94, _hash("atlas-graph-94"))
            valid_atlas_events = {
                _ATLAS_GRAPH.sequence: _ATLAS_GRAPH.event_sha256,
                accepted_head.sequence: accepted_head.event_sha256,
            }
            controller = self._controller(
                Path(tmp),
                atlas_revision_verifier=lambda revision: (
                    valid_atlas_events.get(revision.sequence)
                    == revision.event_sha256
                ),
            )
            _, temporal = self._train_site(
                controller,
                temporal_start=0,
                site=0,
                shift=1,
                signal=2.0,
            )
            forward, _ = _feature(
                temporal,
                site=0,
                source=0,
                signal=2.0,
                atlas_graph=accepted_head,
            )
            self.assertEqual(
                controller.decide(forward, OOE_ACTIONS[0]).origin,
                "crystal",
            )
            self.assertEqual(controller.atlas_graph_revision, accepted_head)

            historical, _ = _feature(
                temporal + 1,
                site=0,
                source=0,
                signal=2.0,
                atlas_graph=_ATLAS_GRAPH,
            )
            self.assertEqual(
                controller.decide(historical, OOE_ACTIONS[0]).origin,
                "crystal",
            )
            controller.ingest_teacher(
                historical,
                _transition(historical, OOE_ACTIONS[0], OOE_ACTIONS[1]),
            )
            self.assertEqual(controller.atlas_graph_revision, accepted_head)

            unknown_revision = GraphRevision(92, _hash("atlas-graph-92-unknown"))
            unknown, _ = _feature(
                temporal + 2,
                site=0,
                source=0,
                signal=2.0,
                atlas_graph=unknown_revision,
            )
            with self.assertRaisesRegex(
                OoeControllerIntegrityError,
                "rejected the historical head",
            ):
                controller.ingest_teacher(
                    unknown,
                    _transition(unknown, OOE_ACTIONS[0], OOE_ACTIONS[1]),
                )

            fork = GraphRevision(94, _hash("atlas-graph-94-fork"))
            forked, _ = _feature(
                temporal + 3,
                site=0,
                source=0,
                signal=2.0,
                atlas_graph=fork,
            )
            with self.assertRaisesRegex(OoeControllerStaleError, "fork"):
                controller.decide(forked, OOE_ACTIONS[0])

            controller.save_snapshot()
            with self.assertRaisesRegex(
                OoeControllerIntegrityError, "requires its atlas revision verifier"
            ):
                OoeController.restore(crystal_store=controller.crystal_store)
            restored = OoeController.restore(
                crystal_store=controller.crystal_store,
                action_executors=controller._executors,
                atlas_revision_verifier=controller._atlas_revision_verifier,
                expected_atlas_graph_revision=accepted_head,
            )
            self.assertEqual(restored.atlas_graph_revision, accepted_head)

    def test_snapshot_restores_historical_revision_older_than_initial_head(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            historical = GraphRevision(0, "0" * 64)
            initial = GraphRevision(3, _hash("atlas-graph-3"))
            forward = GraphRevision(5, _hash("atlas-graph-5"))
            members = {
                revision.sequence: revision.event_sha256
                for revision in (historical, initial, forward)
            }

            def verifier(revision: GraphRevision) -> bool:
                return members.get(revision.sequence) == revision.event_sha256

            store = CrystalStore(Path(tmp))
            controller = OoeController(
                model_pin_sha256=_PIN.sha256,
                weight_graph_revision_sha256=_WEIGHT_GRAPH.sha256,
                atlas_graph_revision=initial,
                crystal_store=store,
                config=ControllerConfig(
                    replicas=4,
                    replica_fanout=4,
                    min_coverage_per_source=1,
                    min_promoted_sources=1,
                    router_radius=1.0e6,
                    router_min_margin=0.0,
                    token_min_confidence=1e-9,
                    consensus_tolerance=1e-7,
                    consensus_max_rounds=4096,
                    reservoir_size=8,
                ),
                atlas_revision_verifier=verifier,
            )
            revisions = (initial, forward, historical)
            for temporal, revision in enumerate(revisions):
                receipt, _ = _feature(
                    temporal,
                    site=0,
                    source=temporal,
                    signal=2.0,
                    atlas_graph=revision,
                )
                controller.ingest_teacher(
                    receipt,
                    _transition(
                        receipt,
                        OOE_ACTIONS[temporal],
                        OOE_ACTIONS[(temporal + 1) % len(OOE_ACTIONS)],
                    ),
                )
            self.assertEqual(controller.atlas_graph_revision, forward)
            controller.save_snapshot()
            restored = OoeController.restore(
                crystal_store=store,
                atlas_revision_verifier=verifier,
                expected_atlas_graph_revision=forward,
            )
            self.assertEqual(restored.atlas_graph_revision, forward)
            snapshot = json.loads(restored.snapshot_bytes())
            self.assertEqual(
                [
                    row["body"]["sequence"]
                    for row in snapshot["body"]["atlas_seen_revisions"]
                ],
                [0, 3, 5],
            )

    def test_resealed_router_threshold_tamper_cannot_change_execution_gate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = self._controller(Path(tmp))
            self._train_site(
                controller,
                temporal_start=0,
                site=0,
                shift=1,
                signal=2.0,
            )
            publication = controller.save_snapshot()
            original = controller.crystal_store.restore_state(CONTROLLER_STATE_NAME)
            current_sha256 = publication.payload_sha256

            for field, value in (("radius", 0.5), ("min_margin", 0.5)):
                document = json.loads(original)
                document["body"]["router"][field] = value
                document["sha256"] = hashlib.sha256(
                    canonical_json_bytes(document["body"])
                ).hexdigest()
                tampered = canonical_json_bytes(document)
                tamper_publication = controller.crystal_store.publish_state(
                    CONTROLLER_STATE_NAME,
                    tampered,
                    expected_sha256=current_sha256,
                )
                with self.assertRaisesRegex(
                    OoeControllerIntegrityError, "thresholds cannot be reproduced"
                ):
                    OoeController.restore(
                        crystal_store=controller.crystal_store,
                        action_executors=controller._executors,
                    )
                restored_publication = controller.crystal_store.publish_state(
                    CONTROLLER_STATE_NAME,
                    original,
                    expected_sha256=tamper_publication.payload_sha256,
                )
                current_sha256 = restored_publication.payload_sha256

    def test_controller_lock_serializes_warm_ingest_and_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = self._controller(Path(tmp))
            site_sha256, temporal = self._train_site(
                controller,
                temporal_start=0,
                site=0,
                shift=1,
                signal=2.0,
            )
            warm, _ = _feature(temporal + 1, source=0, signal=2.0)
            learned, _ = _feature(temporal, source=0, signal=2.0)
            transition = _transition(
                learned,
                OOE_ACTIONS[0],
                OOE_ACTIONS[1],
            )
            barrier = threading.Barrier(3)
            failures = []

            def run_warm() -> None:
                try:
                    barrier.wait()
                    controller.try_warm(
                        warm,
                        OOE_ACTIONS[0],
                        quality_verifier=lambda _feature, execution: (
                            execution.quality_verified
                        ),
                        stream_id="concurrent-warm",
                    )
                except BaseException as exc:  # asserted in the parent thread
                    failures.append(exc)

            def run_ingest() -> None:
                try:
                    barrier.wait()
                    controller.ingest_teacher(learned, transition)
                except BaseException as exc:  # asserted in the parent thread
                    failures.append(exc)

            warm_thread = threading.Thread(target=run_warm)
            ingest_thread = threading.Thread(target=run_ingest)
            warm_thread.start()
            ingest_thread.start()
            barrier.wait()
            controller.save_snapshot()
            warm_thread.join()
            ingest_thread.join()
            self.assertEqual(failures, [])
            self.assertEqual(
                len(controller._sites[site_sha256].history),
                len(OOE_ACTIONS) + 1,
            )

            controller.save_snapshot()
            restored = OoeController.restore(
                crystal_store=controller.crystal_store,
                action_executors=controller._executors,
            )
            self.assertEqual(restored.snapshot_bytes(), controller.snapshot_bytes())

    def test_recoverable_restore_repairs_promotion_before_snapshot_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = self._controller(Path(tmp))
            receipt, _ = _feature(0)
            controller.ingest_teacher(
                receipt,
                _transition(receipt, OOE_ACTIONS[0], OOE_ACTIONS[1]),
            )
            old_state = controller.save_snapshot()
            old_manifest = controller.crystal_store.manifest()
            coverage = controller.coverage_receipt(receipt.site_identity.sha256)
            controller.promote(
                receipt.site_identity.sha256,
                coverage_sha256=coverage.sha256,
                verifier_sha256s=coverage.verifier_sha256s,
            )
            # Crash: promotion is durable, controller state is deliberately old.
            with self.assertRaisesRegex(
                OoeControllerIntegrityError, "manifest changed"
            ):
                OoeController.restore(crystal_store=controller.crystal_store)

            recovered, recovery = OoeController.restore_recoverable(
                crystal_store=controller.crystal_store,
                action_executors=controller._executors,
                expected_old_manifest_generation=old_manifest.generation,
                expected_old_manifest_sha256=old_manifest.sha256,
                expected_old_state_sha256=old_state.payload_sha256,
            )
            self.assertEqual(recovery.old_state_sha256, old_state.payload_sha256)
            self.assertEqual(
                recovery.old_manifest_generation,
                old_manifest.generation,
            )
            self.assertGreater(
                recovery.new_manifest_generation,
                recovery.old_manifest_generation,
            )
            self.assertEqual(
                recovery.recovered_site_sha256s,
                (receipt.site_identity.sha256,),
            )
            self.assertTrue(recovered.crystal_store.audit().clean)
            exact = OoeController.restore(
                crystal_store=recovered.crystal_store,
                action_executors=controller._executors,
            )
            self.assertEqual(exact.snapshot_bytes(), recovered.snapshot_bytes())
            with self.assertRaisesRegex(
                OoeControllerIntegrityError, "strictly forward"
            ):
                OoeController.restore_recoverable(
                    crystal_store=recovered.crystal_store,
                    action_executors=controller._executors,
                )

    def test_recovery_rejects_extra_valid_manifest_site_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = self._controller(Path(tmp))
            receipt, _ = _feature(0)
            controller.ingest_teacher(
                receipt,
                _transition(receipt, OOE_ACTIONS[0], OOE_ACTIONS[1]),
            )
            old_state = controller.save_snapshot()
            old_manifest = controller.crystal_store.manifest()
            coverage = controller.coverage_receipt(receipt.site_identity.sha256)
            publication = controller.promote(
                receipt.site_identity.sha256,
                coverage_sha256=coverage.sha256,
                verifier_sha256s=coverage.verifier_sha256s,
            )
            legitimate = controller.crystal_store.restore(publication.payload_sha256)
            malicious = replace(legitimate, name=_hash("unknown-valid-site"))
            controller.crystal_store.publish(malicious)
            state_before = controller.crystal_store.restore_state(CONTROLLER_STATE_NAME)

            with self.assertRaisesRegex(
                OoeControllerIntegrityError, "unknown site name"
            ):
                OoeController.restore_recoverable(
                    crystal_store=controller.crystal_store,
                    action_executors=controller._executors,
                    expected_old_manifest_generation=old_manifest.generation,
                    expected_old_manifest_sha256=old_manifest.sha256,
                    expected_old_state_sha256=old_state.payload_sha256,
                )
            self.assertEqual(
                controller.crystal_store.restore_state(CONTROLLER_STATE_NAME),
                state_before,
            )

    def test_recovery_finishes_a_partially_published_multi_site_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = self._controller(Path(tmp))
            first_site, temporal = self._train_site(
                controller,
                temporal_start=0,
                site=0,
                shift=1,
                signal=2.0,
            )
            second_site, temporal = self._train_site(
                controller,
                temporal_start=temporal,
                site=1,
                shift=2,
                signal=-20.0,
            )
            first_coverage = controller.coverage_receipt(first_site)
            controller.promote(
                first_site,
                coverage_sha256=first_coverage.sha256,
                verifier_sha256s=first_coverage.verifier_sha256s,
            )

            for site, signal in ((0, 2.5), (1, -19.5)):
                receipt, _ = _feature(
                    temporal,
                    site=site,
                    source=0,
                    signal=signal,
                )
                controller.ingest_teacher(
                    receipt,
                    _transition(receipt, OOE_ACTIONS[0], OOE_ACTIONS[1]),
                )
                temporal += 1
            prepared_state = controller.save_snapshot()
            prepared_manifest = controller.crystal_store.manifest()

            first_coverage = controller.coverage_receipt(first_site)
            controller.promote(
                first_site,
                coverage_sha256=first_coverage.sha256,
                verifier_sha256s=first_coverage.verifier_sha256s,
            )
            # Crash after only the first publication in a two-site batch.
            recovered, recovery = OoeController.restore_recoverable(
                crystal_store=controller.crystal_store,
                action_executors=controller._executors,
                expected_old_manifest_generation=prepared_manifest.generation,
                expected_old_manifest_sha256=prepared_manifest.sha256,
                expected_old_state_sha256=prepared_state.payload_sha256,
            )

            self.assertEqual(
                recovery.recovered_site_sha256s,
                tuple(sorted((first_site, second_site))),
            )
            restored = OoeController.restore(
                crystal_store=controller.crystal_store,
                action_executors=controller._executors,
            )
            self.assertEqual(restored.snapshot_bytes(), recovered.snapshot_bytes())
            self.assertTrue(recovered.crystal_store.audit().clean)

    def test_novelty_stale_shuffle_and_state_tamper_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            first_site, next_temporal = self._train_site(
                controller,
                temporal_start=0,
                site=0,
                shift=1,
                signal=2.0,
            )
            second_site, next_temporal = self._train_site(
                controller,
                temporal_start=next_temporal,
                site=1,
                shift=2,
                signal=-20.0,
            )
            # Adding the second attractor deliberately invalidates the first
            # Crystal's calibration binding.  Re-promotion reuses its learned
            # agents but binds the new two-site router calibration.
            first_coverage = controller.coverage_receipt(first_site)
            controller.promote(
                first_site,
                coverage_sha256=first_coverage.sha256,
                verifier_sha256s=first_coverage.verifier_sha256s,
            )
            shuffle = controller.shuffled_crystal_map(seed=17)
            self.assertEqual(shuffle[first_site], second_site)
            self.assertEqual(shuffle[second_site], first_site)

            receipt, _ = _feature(
                next_temporal,
                site=0,
                source=0,
                signal=2.0,
            )
            real = controller.decide(
                receipt,
                OOE_ACTIONS[0],
                stream_id="real",
            )
            placebo = controller.decide_placebo(
                receipt,
                OOE_ACTIONS[0],
                shuffled_sites=shuffle,
                mode="shuffled-crystal",
                stream_id="placebo",
            )
            self.assertEqual(real.action, self._target(0, shift=1))
            self.assertEqual(placebo.action, self._target(0, shift=2))
            self.assertNotEqual(real.action, placebo.action)
            self.assertEqual(placebo.reason, "shuffled-crystal-placebo")
            self.assertEqual(placebo.routed_site_identity_sha256, first_site)
            self.assertEqual(placebo.crystal_site_identity_sha256, second_site)
            site_placebo = controller.decide_placebo(
                receipt,
                OOE_ACTIONS[0],
                shuffled_sites=shuffle,
                mode="shuffled-site",
                stream_id="site-placebo",
            )
            self.assertEqual(site_placebo.action, placebo.action)
            self.assertEqual(site_placebo.reason, "shuffled-site-placebo")
            self.assertEqual(site_placebo.routed_site_identity_sha256, second_site)
            self.assertEqual(site_placebo.crystal_site_identity_sha256, second_site)

            novel, _ = _feature(
                next_temporal + 1,
                site=2,
                source=0,
                signal=40.0,
            )
            novelty = controller.decide(novel, OOE_ACTIONS[0], stream_id="novel")
            self.assertEqual(novelty.origin, "abstention")
            self.assertEqual(novelty.action, "qwen_fallback")

            stale_graph = GraphRevision(43, _hash("weight-graph-43"))
            stale, _ = _feature(
                next_temporal + 2,
                site=0,
                source=0,
                signal=2.0,
                weight_graph=stale_graph,
            )
            with self.assertRaises(OoeControllerStaleError):
                controller.decide(stale, OOE_ACTIONS[0])

            controller.save_snapshot()
            with self.assertRaises(OoeControllerStaleError):
                OoeController.restore(
                    crystal_store=controller.crystal_store,
                    expected_model_pin_sha256=_hash("another-model"),
                )

            state_path = next((root / "state").glob("*.state"))
            data = bytearray(state_path.read_bytes())
            data[len(data) // 2] ^= 1
            state_path.chmod(0o600)
            state_path.write_bytes(data)
            with self.assertRaises((CrystalTamperError, OoeControllerIntegrityError)):
                OoeController.restore(crystal_store=controller.crystal_store)


if __name__ == "__main__":
    unittest.main()
