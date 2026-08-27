from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from immer.cognition.fertig import FertigSolver
from immer.runtimes.deepseek_v4.causal_weights import TensorRangePlan
from immer.runtimes.ooe.controller import (
    ActionExecution,
    ControllerConfig,
    OoeController,
)
from immer.runtimes.ooe.crystal import CrystalStore
from immer.runtimes.ooe.execution_learning import (
    ActionAuthorityReceipt,
    ExecutionLearningBank,
    ExecutionLearningBridge,
    ExecutionLearningExecutionError,
    ExecutionLearningIntegrityError,
    ExecutionLearningReceipt,
    ExecutionLearningStaleError,
)
from immer.runtimes.ooe.fertig_executor import (
    FERTIG_EXECUTION_AUTHORITY_SHA256,
    FERTIG_EXECUTOR_SHA256,
    FERTIG_QUALITY_VERIFIER_SHA256,
    FertigExactExecutor,
)
from immer.runtimes.ooe.qwen_bridge import QwenOoeFeatureReceipt
from immer.runtimes.ooe.s3_executor import TransientPromptRegistry
from immer.runtimes.qwen3_8.semantic_atlas import (
    GraphRevision,
    InterventionIdentity,
    MeasurementReceipt,
    ModelPin,
    NumericSummary,
    ProbeIdentity,
    RuntimeProvenance,
    SemanticWeightAtlas,
    WeightCoordinate,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


_CODE = "d" * 40
_PLAN = TensorRangePlan(
    name="model.layers.18.mlp.gate_proj.weight",
    dtype="BF16",
    shape=(32, 4),
    shard="model-00007-of-00018.safetensors",
    absolute_offset=4096,
    length=256,
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
_WEIGHT_GRAPH = GraphRevision(17, _hash("weight-graph-17"))
_EXACT_FERTIG_QUESTION = (
    "A phone tree is used to contact families and relatives of Ali's deceased "
    "coworker. Ali decided to call 3 families. Then each family calls 3 other "
    "families, and so on. How many families will be notified during the fourth "
    "round of calls?"
)


class ExecutionLearningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix=".execution-learning-test-", dir=Path.cwd()
        )
        self.root = Path(self.temporary.name)
        self.atlas = SemanticWeightAtlas(
            self.root / "atlas", model_pin=_PIN, tensor_plans=(_PLAN,)
        )
        self.store = CrystalStore(self.root / "crystals")
        self.controller = OoeController(
            model_pin_sha256=_PIN.sha256,
            weight_graph_revision_sha256=_WEIGHT_GRAPH.sha256,
            atlas_graph_revision=self.atlas.revision(),
            crystal_store=self.store,
            config=ControllerConfig(replicas=4, replica_fanout=2),
            atlas_revision_verifier=self.atlas.contains_revision,
        )
        self.bank = ExecutionLearningBank(self.store)
        self.execution_calls: dict[str, int] = {}
        self.fail_actions: set[str] = set()
        self.quality_pass = True
        self.authorities: dict[str, ActionAuthorityReceipt] = {}
        self.executors = {}
        for action in ("probe_coordinate", "mount_organ", "execute_fertig"):
            authority = ActionAuthorityReceipt(
                action=action,
                model_pin_sha256=_PIN.sha256,
                weight_graph_revision_sha256=_WEIGHT_GRAPH.sha256,
                executor_sha256=_hash(f"executor:{action}"),
                action_verifier_name=f"bound-action:{action}",
                action_verifier_sha256=_hash(f"action-verifier:{action}"),
                quality_verifier_name="exact-result-quality",
                quality_verifier_sha256=_hash("quality-verifier"),
                evidence_sha256=_hash(f"authority-evidence:{action}"),
            )
            self.authorities[action] = authority

            def execute(feature, *, selected=action, auth=authority):
                self.execution_calls[selected] = (
                    self.execution_calls.get(selected, 0) + 1
                )
                if selected in self.fail_actions:
                    raise RuntimeError("injected executor failure")
                return ActionExecution(
                    feature_receipt_sha256=feature.sha256,
                    action=selected,
                    executor_sha256=auth.executor_sha256,
                    verifier_sha256=auth.action_verifier_sha256,
                    evidence_sha256=auth.evidence_sha256,
                    quality_sha256=_hash(f"quality:{feature.sha256}:{selected}"),
                    result={
                        "action": selected,
                        "feature_receipt_sha256": feature.sha256,
                    },
                    quality_verified=True,
                    qwen_forwards=0,
                    teacher_baseline_qwen_forwards=1,
                )

            self.executors[action] = (authority.executor_sha256, execute)

        def quality(feature, execution):
            return self.quality_pass and (
                execution.result["feature_receipt_sha256"] == feature.sha256
            )

        self.quality_verifiers = {
            "exact-result-quality": (_hash("quality-verifier"), quality)
        }
        self.bridge = self._bridge()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _bridge(self) -> ExecutionLearningBridge:
        return ExecutionLearningBridge(
            controller=self.controller,
            atlas=self.atlas,
            bank=self.bank,
            action_authorities=self.authorities,
            action_executors=self.executors,
            quality_verifiers=self.quality_verifiers,
        )

    def _measurement(
        self,
        sample: int,
        *,
        row: int = 0,
        question: str | None = None,
    ) -> MeasurementReceipt:
        coordinate = WeightCoordinate.from_plan(
            _PLAN,
            layer=18,
            module="model.layers.18.mlp.gate_proj",
            row_start=row,
            row_end=row + 2,
        )
        value = 1.0 + sample * 0.125
        receipt = MeasurementReceipt(
            model_pin=_PIN,
            coordinate=coordinate,
            probe=ProbeIdentity(
                question_sha256=(
                    _hash(f"question:{sample}")
                    if question is None
                    else _hash(question)
                ),
                token_sha256=_hash(f"tokens:{sample}"),
                family_sha256=_hash("family"),
                label_source_sha256=_hash("label-source"),
            ),
            intervention=InterventionIdentity(
                mode="native", configuration_sha256=_hash(f"native:{sample}")
            ),
            observation_status="recorded",
            observed_semantic_label=None,
            hidden_sha256=_hash(f"hidden:{sample}"),
            activation_sha256=_hash(f"activation:{sample}"),
            logits_sha256=_hash(f"logits:{sample}"),
            state_sha256=_hash(f"state:{sample}"),
            access_trace_sha256=_hash(f"access:{sample}"),
            evidence_sha256=_hash(f"evidence:{sample}"),
            weight_rail_revision=_WEIGHT_GRAPH,
            atlas_head_revision=self.atlas.revision(),
            numeric_summaries=(
                NumericSummary(
                    metric="activation-rms",
                    count=2,
                    total=value + value + 0.5,
                    total_squares=value * value + (value + 0.5) ** 2,
                    minimum=value,
                    maximum=value + 0.5,
                ),
            ),
            placebo_effects=(),
            runtime=_RUNTIME,
        )
        self.atlas.append_measurement(receipt)
        return receipt

    def _learn(self, measurement: MeasurementReceipt, action: str):
        return self.bridge.learn_from_execution(
            measurement,
            action=action,
            o1_surprise=0.25,
            o1_learning_progress=0.75,
        )

    def test_real_execution_is_only_target_and_authority_preserves_site(self) -> None:
        measurement = self._measurement(1)
        base = QwenOoeFeatureReceipt.from_measurement(
            measurement,
            temporal_index=0,
            verifier_sha256s=(_hash("base-verifier"),),
            o1_surprise=0.25,
            o1_learning_progress=0.75,
        )
        learned = self._learn(measurement, "probe_coordinate")
        assert learned is not None

        self.assertEqual(learned.execution.action, "probe_coordinate")
        self.assertEqual(learned.transition.target_action, learned.execution.action)
        self.assertEqual(learned.transition.source_action, "qwen_fallback")
        self.assertEqual(learned.feature.site_identity, base.site_identity)
        self.assertIn(learned.action_authority.sha256, learned.feature.verifier_sha256s)
        self.assertEqual(
            ExecutionLearningReceipt.from_document(learned.to_document()), learned
        )
        self.assertEqual(self.bridge.controller.last_temporal_index, 0)

        fake_transition = replace(
            learned.transition,
            target_action="mount_organ",
        )
        with self.assertRaises(ExecutionLearningIntegrityError):
            replace(learned, transition=fake_transition)

        tampered = learned.to_document()
        tampered["body"]["execution"]["body"]["action"] = "mount_organ"
        with self.assertRaises(ExecutionLearningIntegrityError):
            ExecutionLearningReceipt.from_document(tampered)

        # Exact replay remains active after unrelated append-only Atlas growth
        # and returns the persistent receipt without executing twice.
        self._measurement(101, row=6)
        self.assertEqual(self._learn(measurement, "probe_coordinate"), learned)
        self.assertEqual(self.execution_calls["probe_coordinate"], 1)
        with self.assertRaises(ExecutionLearningStaleError):
            self.bridge.learn_from_execution(
                measurement,
                action="probe_coordinate",
                source_action="mount_organ",
                o1_surprise=0.25,
                o1_learning_progress=0.75,
            )
        with self.assertRaises(ExecutionLearningStaleError):
            self.bridge.learn_from_execution(
                measurement,
                action="probe_coordinate",
                temporal_index=1,
                o1_surprise=0.25,
                o1_learning_progress=0.75,
            )

    def test_execution_and_quality_failures_are_controller_neutral(self) -> None:
        execution_failure = self._measurement(2)
        before_controller = self.controller.snapshot_bytes()
        before_bank = self.bank.head()
        self.fail_actions.add("execute_fertig")
        with self.assertRaises(ExecutionLearningExecutionError):
            self._learn(execution_failure, "execute_fertig")
        self.assertEqual(self.controller.snapshot_bytes(), before_controller)
        self.assertEqual(self.bank.head(), before_bank)

        quality_failure = self._measurement(3, row=2)
        self.quality_pass = False
        self.assertIsNone(self._learn(quality_failure, "mount_organ"))
        self.assertEqual(self.controller.snapshot_bytes(), before_controller)
        self.assertEqual(self.bank.head(), before_bank)

    def test_temporal_source_chain_stale_pin_and_forged_measurement_rejected(
        self,
    ) -> None:
        first = self._measurement(4)
        second = self._measurement(5, row=2)
        one = self._learn(first, "probe_coordinate")
        old_head_bytes = self.bank.head().to_bytes()
        two = self._learn(second, "mount_organ")
        assert one is not None and two is not None
        head = self.bank.head()
        self.assertEqual(head.generation, 2)
        self.assertEqual(head.traces[1].source_action, "probe_coordinate")
        self.assertEqual(head.traces[1].previous_trace_sha256, head.traces[0].sha256)
        self.assertEqual(
            head.traces[1].temporal_index, head.traces[0].temporal_index + 1
        )
        with self.assertRaisesRegex(
            ExecutionLearningIntegrityError,
            "discontinuous",
        ):
            replace(
                head,
                traces=(
                    head.traces[0],
                    replace(
                        head.traces[1],
                        controller_snapshot_before_sha256=_hash(
                            "forked-controller-endpoint"
                        ),
                    ),
                ),
            )

        third = self._measurement(6, row=4)
        with self.assertRaises(ExecutionLearningStaleError):
            self.bridge.learn_from_execution(
                third,
                action="probe_coordinate",
                source_action="qwen_fallback",
                o1_surprise=0.25,
                o1_learning_progress=0.75,
            )

        forged = replace(third, hidden_sha256=_hash("forged-hidden"))
        with self.assertRaises(ExecutionLearningIntegrityError):
            self._learn(forged, "probe_coordinate")

        stale = replace(
            self.authorities["probe_coordinate"],
            model_pin_sha256=_hash("stale-model"),
        )
        authorities = dict(self.authorities)
        authorities["probe_coordinate"] = stale
        with self.assertRaises(ExecutionLearningStaleError):
            ExecutionLearningBridge(
                controller=self.controller,
                atlas=self.atlas,
                bank=self.bank,
                action_authorities=authorities,
                action_executors=self.executors,
                quality_verifiers=self.quality_verifiers,
            )

        # Even deleting the newest commit marker cannot hide a validly re-sealed
        # pointer rollback: the monotonic pointer generation proves it.
        current_head = self.bank.head()
        newest_commit = self.bank._commit_name(current_head.sha256)
        (self.store.root / "state" / self.store._state_filename(newest_commit)).unlink()
        self.store.publish_state(
            "ooe-execution-learning-head/v1",
            old_head_bytes,
            expected_sha256=current_head.sha256,
        )
        with self.assertRaisesRegex(ExecutionLearningIntegrityError, "generation"):
            self.bank.head()

    def test_controller_cas_then_trace_crash_recovers_without_reexecution(self) -> None:
        measurement = self._measurement(7)
        original_publish = self.store.publish_state
        failed = False

        def crash_after_controller(name, payload, *, expected_sha256=None):
            nonlocal failed
            if (
                name == "ooe-execution-learning-head/v1"
                and expected_sha256 is not None
                and not failed
            ):
                failed = True
                raise RuntimeError("crash after controller CAS")
            return original_publish(name, payload, expected_sha256=expected_sha256)

        with patch.object(
            self.store, "publish_state", side_effect=crash_after_controller
        ):
            with self.assertRaisesRegex(RuntimeError, "crash after controller CAS"):
                self._learn(measurement, "probe_coordinate")

        self.assertEqual(self.bridge.controller.last_temporal_index, 0)
        self.assertEqual(self.bank.head().generation, 0)
        self.assertEqual(self.execution_calls["probe_coordinate"], 1)

        recovered = self._learn(measurement, "probe_coordinate")
        assert recovered is not None
        self.controller = self.bridge.controller
        self.assertEqual(self.controller.last_temporal_index, 0)
        self.assertEqual(self.bank.head().generation, 1)
        self.assertEqual(self.execution_calls["probe_coordinate"], 1)
        self.assertEqual(
            self.bank.head().traces[0].controller_snapshot_after_sha256,
            hashlib.sha256(self.store.restore_state("qwen-ooe-controller")).hexdigest(),
        )

    def test_missing_commit_marker_is_repaired_only_after_ancestry_audit(self) -> None:
        measurement = self._measurement(70)
        learned = self._learn(measurement, "probe_coordinate")
        assert learned is not None
        head = self.bank.head()
        commit_name = self.bank._commit_name(head.sha256)
        commit_path = self.store.root / "state" / self.store._state_filename(
            commit_name
        )
        commit_path.unlink()
        self.assertEqual(self.bank.head(), head)
        self.assertTrue(commit_path.is_file())

        commit_path.unlink()
        predecessor_name = self.bank._history_name(head.previous_state_sha256)
        predecessor_path = self.store.root / "state" / self.store._state_filename(
            predecessor_name
        )
        predecessor_path.unlink()
        with self.assertRaisesRegex(
            ExecutionLearningIntegrityError,
            "predecessor is missing",
        ):
            self.bank.head()
        self.assertFalse(commit_path.exists())

    def test_exact_fertig_certificate_is_the_controller_teacher(self) -> None:
        measurement = self._measurement(8, question=_EXACT_FERTIG_QUESTION)
        registry = TransientPromptRegistry()
        executor = FertigExactExecutor(FertigSolver(), registry)
        authority = ActionAuthorityReceipt(
            action="execute_fertig",
            model_pin_sha256=_PIN.sha256,
            weight_graph_revision_sha256=_WEIGHT_GRAPH.sha256,
            executor_sha256=FERTIG_EXECUTOR_SHA256,
            action_verifier_name="fertig-exact-certificate",
            action_verifier_sha256=FERTIG_EXECUTION_AUTHORITY_SHA256,
            quality_verifier_name="fertig-exact-replay",
            quality_verifier_sha256=FERTIG_QUALITY_VERIFIER_SHA256,
            evidence_sha256=FERTIG_EXECUTION_AUTHORITY_SHA256,
        )

        def execute(feature):
            registry.bind(feature, _EXACT_FERTIG_QUESTION)
            return executor(feature)

        bridge = ExecutionLearningBridge(
            controller=self.controller,
            atlas=self.atlas,
            bank=self.bank,
            action_authorities={"execute_fertig": authority},
            action_executors={
                "execute_fertig": (FERTIG_EXECUTOR_SHA256, execute),
            },
            quality_verifiers={
                "fertig-exact-replay": (
                    FERTIG_QUALITY_VERIFIER_SHA256,
                    executor.verify,
                ),
            },
        )
        learned = bridge.learn_from_execution(
            measurement,
            action="execute_fertig",
            o1_surprise=0.25,
            o1_learning_progress=0.75,
        )
        assert learned is not None
        self.assertEqual(learned.execution.result["certificate"]["answer"], "81")
        self.assertEqual(learned.transition.target_action, "execute_fertig")
        self.assertEqual(learned.transition.source_action, "qwen_fallback")
        self.assertNotIn(_EXACT_FERTIG_QUESTION, str(learned.to_document()))
        self.assertEqual(
            bridge.learn_from_execution(
                measurement,
                action="execute_fertig",
                o1_surprise=0.25,
                o1_learning_progress=0.75,
            ),
            learned,
        )


if __name__ == "__main__":
    unittest.main()
