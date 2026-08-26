from __future__ import annotations

import base64
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np

from immer.runtimes.ooe.consensus import barbell_adjacency, complete_adjacency
from immer.runtimes.ooe.crystal import CrystalStore, ManifestConflictError
from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.planning import ExecutionOutcome, FiniteHorizonPlanner
from immer.runtimes.ooe.world_model import (
    ActionConditionedWorldModel,
    RegimeChangeSignal,
    TransitionEvidence,
    WorldModelTamperError,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _evidence(
    source: str,
    action: str,
    target: str,
    *,
    name: str,
    weight: float = 1.0,
    epoch: int = 0,
) -> TransitionEvidence:
    return TransitionEvidence(
        source_state=source,
        action=action,
        target_state=target,
        weight=weight,
        verifier_sha256=_hash(f"verifier:{name}"),
        evidence_sha256=_hash(f"evidence:{name}"),
        epoch=epoch,
    )


class ActionConditionedWorldModelTests(unittest.TestCase):
    def test_weighted_evidence_has_exact_rows_and_separate_epistemic_counts(
        self,
    ) -> None:
        model = ActionConditionedWorldModel(
            ("a", "b", "c"),
            ("move", "wait"),
        )
        model.observe(_evidence("a", "move", "b", name="light", weight=1.0))
        model.observe(_evidence("a", "move", "c", name="heavy", weight=3.0))

        row = model.transition_row("a", "move")
        self.assertEqual(float(row.sum()), 1.0)
        np.testing.assert_array_equal(row, np.array([0.0, 0.25, 0.75]))
        self.assertEqual(model.row_evidence_mass("a", "move"), 4.0)
        self.assertEqual(model.row_event_count("a", "move"), 2)
        prediction = model.predict("a", "move")
        self.assertFalse(prediction.abstained)
        self.assertEqual(prediction.evidence_mass, 4.0)
        self.assertEqual(prediction.event_count, 2)

        kernel = model.action_kernel("move")
        self.assertTrue(np.all(kernel >= 0.0))
        for normalized in kernel:
            self.assertEqual(float(normalized.sum()), 1.0)
        np.testing.assert_allclose(kernel[1], np.full(3, 1.0 / 3.0))

    def test_novel_and_high_entropy_transitions_abstain_without_mutation(self) -> None:
        model = ActionConditionedWorldModel(
            ("a", "b", "c"),
            ("move",),
            max_normalized_entropy=0.5,
        )
        before = model.sha256
        novel = model.predict("a", "move")
        self.assertTrue(novel.abstained)
        self.assertEqual(novel.reason, "novel-transition")
        self.assertEqual(model.sha256, before)

        model.observe(_evidence("a", "move", "b", name="branch-b"))
        model.observe(_evidence("a", "move", "c", name="branch-c"))
        learned = model.sha256
        uncertain = model.predict("a", "move")
        self.assertTrue(uncertain.abstained)
        self.assertEqual(uncertain.reason, "transition-entropy")
        self.assertEqual(model.sha256, learned)

    def test_observe_many_is_atomic_and_provenance_deduplicates(self) -> None:
        model = ActionConditionedWorldModel(("a", "b"), ("move",))
        first = _evidence("a", "move", "b", name="unique")
        before = model.sha256
        with self.assertRaisesRegex(KeyError, "target state"):
            model.observe_many(
                (first, _evidence("b", "move", "missing", name="invalid"))
            )
        self.assertEqual(model.sha256, before)
        model.observe(first)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            model.observe(first)

    def test_explicit_regime_decay_adapts_and_is_receipted(self) -> None:
        model = ActionConditionedWorldModel(
            ("root", "old", "new"),
            ("choose",),
        )
        model.observe(
            _evidence("root", "choose", "old", name="old-regime", weight=20.0)
        )
        stale = model.clone()
        signal = RegimeChangeSignal(
            signal_sha256=_hash("change-point-17"),
            verifier_sha256=_hash("cusum-verifier"),
            strength=0.99,
            retention=0.01,
            epoch=17,
            name="operator-remap",
        )
        receipt = model.apply_regime_change(signal)
        model.observe(_evidence("root", "choose", "new", name="new-regime", weight=3.0))
        stale.observe(_evidence("root", "choose", "new", name="new-regime", weight=3.0))

        adapted = model.predict("root", "choose")
        unadapted = stale.predict("root", "choose")
        self.assertGreater(adapted.probabilities[2], 0.93)
        self.assertGreater(unadapted.probabilities[1], 0.86)
        self.assertAlmostEqual(receipt.after_evidence_mass, 0.2)
        self.assertEqual(receipt.revision, 1)
        self.assertNotEqual(receipt.before_model_sha256, receipt.after_model_sha256)
        json.dumps(receipt.to_dict(), allow_nan=False, sort_keys=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            model.apply_regime_change(signal)

    def test_dense_and_node_bounds_fail_before_allocation(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires"):
            ActionConditionedWorldModel(
                tuple(f"s{index}" for index in range(10)),
                ("a", "b"),
                max_bytes=1_000,
            )
        with self.assertRaisesRegex(ValueError, "states count"):
            ActionConditionedWorldModel(("a", "b", "c"), ("x",), max_states=2)

    def test_complete_payload_is_canonical_little_endian_and_exactly_restorable(
        self,
    ) -> None:
        model = ActionConditionedWorldModel(
            ("s0", "s1", "s2"),
            ("advance", "wait"),
        )
        model.observe(
            _evidence("s0", "advance", "s1", name="persist-edge-0", weight=3.0)
        )
        model.observe(
            _evidence("s1", "advance", "s2", name="persist-edge-1", weight=3.0)
        )
        signals = [
            RegimeChangeSignal(
                signal_sha256=_hash(f"persist-signal-{index}"),
                verifier_sha256=_hash(f"persist-signal-verifier-{index}"),
                strength=0.2 + 0.1 * index,
                retention=0.9,
                epoch=10 + index,
                name=f"regime-{index}",
            )
            for index in range(2)
        ]
        # Chronology is deliberately not digest-sorted: it is a lineage, not a
        # mathematical set.
        signals.sort(key=lambda signal: signal.signal_sha256, reverse=True)
        for signal in signals:
            model.apply_regime_change(signal)

        payload = model.to_bytes()
        document = json.loads(payload)
        self.assertEqual(document["tensors"]["counts"]["dtype"], "<f8")
        self.assertEqual(document["tensors"]["event_counts"]["dtype"], "<u8")
        self.assertEqual(document["tensors"]["shape"], [2, 3, 3])
        self.assertEqual(document["model_sha256"], model.sha256)
        self.assertEqual(
            document["regime"]["signal_hashes"],
            [signal.signal_sha256 for signal in signals],
        )

        restored = ActionConditionedWorldModel.from_bytes(payload)
        self.assertEqual(restored.sha256, model.sha256)
        self.assertEqual(restored.to_dict(), model.to_dict())
        self.assertEqual(restored.to_bytes(), payload)
        np.testing.assert_array_equal(restored.counts, model.counts)
        np.testing.assert_array_equal(restored.event_counts, model.event_counts)
        self.assertEqual(restored.evidence_hashes, model.evidence_hashes)
        self.assertEqual(restored.verifier_hashes, model.verifier_hashes)

    def test_payload_rejects_duplicate_nan_base64_shape_byte_and_hash_tamper(
        self,
    ) -> None:
        model = ActionConditionedWorldModel(("s0", "s1"), ("advance",))
        model.observe(_evidence("s0", "advance", "s1", name="sealed-edge"))
        payload = model.to_bytes()

        with self.assertRaisesRegex(WorldModelTamperError, "strict JSON"):
            ActionConditionedWorldModel.from_bytes(b'{"format":"x","format":"y"}')
        with self.assertRaisesRegex(WorldModelTamperError, "strict JSON"):
            ActionConditionedWorldModel.from_bytes(b'{"value":NaN}')
        with self.assertRaisesRegex(WorldModelTamperError, "canonical JSON"):
            ActionConditionedWorldModel.from_bytes(payload + b" ")
        with self.assertRaisesRegex(WorldModelTamperError, "byte bound"):
            ActionConditionedWorldModel.from_bytes(
                payload, max_payload_bytes=len(payload) - 1
            )

        shape_tamper = json.loads(payload)
        shape_tamper["tensors"]["shape"] = [1, 3, 3]
        with (
            mock.patch(
                "immer.runtimes.ooe.world_model.base64.b64decode",
                side_effect=AssertionError("shape must fail before decode"),
            ),
            self.assertRaisesRegex(WorldModelTamperError, "shape mismatch"),
        ):
            ActionConditionedWorldModel.from_bytes(canonical_json_bytes(shape_tamper))

        byte_tamper = json.loads(payload)
        byte_tamper["tensors"]["counts"]["byte_count"] += 8
        with (
            mock.patch(
                "immer.runtimes.ooe.world_model.base64.b64decode",
                side_effect=AssertionError("byte count must fail before decode"),
            ),
            self.assertRaisesRegex(WorldModelTamperError, "byte count"),
        ):
            ActionConditionedWorldModel.from_bytes(canonical_json_bytes(byte_tamper))

        base64_tamper = json.loads(payload)
        encoded = base64_tamper["tensors"]["counts"]["data_base64"]
        base64_tamper["tensors"]["counts"]["data_base64"] = "!" + encoded[1:]
        with self.assertRaisesRegex(WorldModelTamperError, "base64"):
            ActionConditionedWorldModel.from_bytes(canonical_json_bytes(base64_tamper))

        duplicate_provenance = json.loads(payload)
        duplicate_provenance["evidence_hashes"].append(
            duplicate_provenance["evidence_hashes"][0]
        )
        with self.assertRaisesRegex(WorldModelTamperError, "unique"):
            ActionConditionedWorldModel.from_bytes(
                canonical_json_bytes(duplicate_provenance)
            )

        nan_tamper = json.loads(payload)
        nan_counts = bytearray(
            base64.b64decode(nan_tamper["tensors"]["counts"]["data_base64"])
        )
        nan_counts[:8] = struct.pack("<d", float("nan"))
        nan_tamper["tensors"]["counts"]["data_base64"] = base64.b64encode(
            nan_counts
        ).decode("ascii")
        nan_tamper["tensors"]["counts"]["raw_sha256"] = hashlib.sha256(
            nan_counts
        ).hexdigest()
        with self.assertRaisesRegex(WorldModelTamperError, "non-finite"):
            ActionConditionedWorldModel.from_bytes(canonical_json_bytes(nan_tamper))

        semantic_tamper = json.loads(payload)
        changed_counts = bytearray(
            base64.b64decode(semantic_tamper["tensors"]["counts"]["data_base64"])
        )
        changed_counts[:8] = struct.pack("<d", 0.25)
        semantic_tamper["tensors"]["counts"]["data_base64"] = base64.b64encode(
            changed_counts
        ).decode("ascii")
        semantic_tamper["tensors"]["counts"]["raw_sha256"] = hashlib.sha256(
            changed_counts
        ).hexdigest()
        with self.assertRaisesRegex(WorldModelTamperError, "semantic hash mismatch"):
            ActionConditionedWorldModel.from_bytes(
                canonical_json_bytes(semantic_tamper)
            )

        event_tamper = json.loads(payload)
        changed_events = bytearray(
            base64.b64decode(event_tamper["tensors"]["event_counts"]["data_base64"])
        )
        changed_events[:8] = (1).to_bytes(8, "little")
        event_tamper["tensors"]["event_counts"]["data_base64"] = base64.b64encode(
            changed_events
        ).decode("ascii")
        event_tamper["tensors"]["event_counts"]["raw_sha256"] = hashlib.sha256(
            changed_events
        ).hexdigest()
        with self.assertRaisesRegex(WorldModelTamperError, "event counts"):
            ActionConditionedWorldModel.from_bytes(canonical_json_bytes(event_tamper))

    def test_crystal_state_publish_restore_is_atomic_idempotent_and_plan_exact(
        self,
    ) -> None:
        model = ActionConditionedWorldModel(("s0", "s1", "s2"), ("advance", "wait"))
        model.observe(_evidence("s0", "advance", "s1", name="store-edge-0", weight=2.0))
        model.observe(_evidence("s1", "advance", "s2", name="store-edge-1", weight=2.0))
        planner = FiniteHorizonPlanner(model, min_predicted_success=0.99)
        plan = planner.plan_goal("s0", "s2", horizon=2).plan
        assert plan is not None

        def execute(source: str, action: str, ordinal: int) -> ExecutionOutcome:
            return ExecutionOutcome(
                target_state=f"s{int(source[1:]) + 1}",
                verifier_sha256=_hash(f"persist-live-verifier:{ordinal}"),
                evidence_sha256=_hash(f"persist-live-evidence:{ordinal}"),
            )

        execution = planner.execute(plan, execute)
        payload = model.to_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            store = CrystalStore(
                Path(temporary) / "bank", max_state_bytes=len(payload) + 4096
            )
            first = store.publish_state("ooe-world-model", payload)
            self.assertTrue(first.changed)
            self.assertEqual(first.generation, 1)
            same = store.publish_state("ooe-world-model", payload)
            self.assertFalse(same.changed)
            self.assertEqual(same.generation, 1)
            self.assertEqual(same.payload_sha256, first.payload_sha256)

            restored_payload = store.restore_state("ooe-world-model")
            restored = ActionConditionedWorldModel.from_bytes(restored_payload)
            restored_planner = FiniteHorizonPlanner(
                restored, min_predicted_success=0.99
            )
            restored_plan = restored_planner.plan_goal("s0", "s2", horizon=2).plan
            assert restored_plan is not None
            self.assertEqual(restored_plan.to_dict(), plan.to_dict())
            self.assertEqual(restored_plan.sha256, plan.sha256)
            restored_execution = restored_planner.execute(restored_plan, execute)
            self.assertEqual(restored_execution.to_dict(), execution.to_dict())
            self.assertEqual(restored_execution.sha256, execution.sha256)

            updated = model.clone()
            updated.observe(_evidence("s0", "wait", "s0", name="store-new-edge"))
            updated_payload = updated.to_bytes()
            with self.assertRaises(ManifestConflictError):
                store.publish_state(
                    "ooe-world-model",
                    updated_payload,
                    expected_sha256=_hash("wrong-store-parent"),
                )
            self.assertEqual(store.restore_state("ooe-world-model"), payload)
            second = store.publish_state(
                "ooe-world-model",
                updated_payload,
                expected_sha256=first.payload_sha256,
            )
            self.assertTrue(second.changed)
            self.assertEqual(second.generation, 2)
            self.assertEqual(store.restore_state("ooe-world-model"), updated_payload)
            self.assertEqual(tuple((store.root / "staging").iterdir()), ())


class DistributedWorldPlanningTests(unittest.TestCase):
    @staticmethod
    def _chain_replicas() -> tuple[list[ActionConditionedWorldModel], tuple[str, ...]]:
        states = tuple(f"s{index}" for index in range(8))
        replicas = [
            ActionConditionedWorldModel(states, ("advance", "reset")) for _ in range(4)
        ]
        # These are strictly one-step teacher observations.  No teacher ever
        # sees or labels a multi-step path.
        for index in range(7):
            replicas[index % 4].observe(
                _evidence(
                    states[index],
                    "advance",
                    states[index + 1],
                    name=f"edge-{index}",
                    weight=2.0,
                )
            )
        return replicas, states

    def test_ps_lifted_fusion_composes_unseen_long_plan_and_execution_receipt(
        self,
    ) -> None:
        replicas, states = self._chain_replicas()
        graph = barbell_adjacency(2, 2)
        fused = ActionConditionedWorldModel.fuse_ps_lifted(
            replicas,
            graph,
            tolerance=1e-10,
            max_rounds=4096,
        )
        self.assertTrue(fused.receipt.consensus.converged)
        self.assertEqual(fused.receipt.consensus.lifted_nodes, 4)
        self.assertEqual(fused.receipt.topology, "ps-lifted-z2-world")
        self.assertEqual(fused.model.total_events, 7)
        self.assertAlmostEqual(fused.model.total_evidence_mass, 14.0, places=8)

        planner = FiniteHorizonPlanner(fused.model, min_predicted_success=0.99)
        decision = planner.plan_goal("s0", "s7", horizon=7)
        self.assertFalse(decision.abstained)
        self.assertIsNotNone(decision.plan)
        plan = decision.plan
        assert plan is not None
        self.assertEqual(plan.expected_actions, ("advance",) * 7)
        self.assertEqual(plan.expected_states, states)
        self.assertAlmostEqual(plan.predicted_success, 1.0, places=12)

        def execute(source: str, action: str, ordinal: int) -> ExecutionOutcome:
            self.assertEqual(action, "advance")
            target = states[states.index(source) + 1]
            return ExecutionOutcome(
                target_state=target,
                verifier_sha256=_hash(f"live-verifier:{ordinal}"),
                evidence_sha256=_hash(f"live-evidence:{ordinal}"),
            )

        before = fused.model.sha256
        execution = planner.execute(plan, execute)
        self.assertTrue(execution.success)
        self.assertEqual(execution.final_state, "s7")
        self.assertEqual(len(execution.steps), 7)
        self.assertEqual(len(execution.transition_hashes), 7)
        self.assertEqual(len(execution.state_hashes), 8)
        self.assertEqual(len(execution.verifier_hashes), 7)
        self.assertEqual(fused.model.sha256, before)
        json.dumps(execution.to_dict(), allow_nan=False, sort_keys=True)

    def test_unknown_downstream_transition_forces_plan_abstention(self) -> None:
        model = ActionConditionedWorldModel(("s0", "s1", "s2"), ("advance",))
        model.observe(_evidence("s0", "advance", "s1", name="only-edge"))
        planner = FiniteHorizonPlanner(model, min_predicted_success=0.1)
        decision = planner.plan_goal("s0", "s2", horizon=2)
        self.assertTrue(decision.abstained)
        self.assertIsNone(decision.plan)
        self.assertEqual(decision.reason, "unreachable-or-uncovered")

    def test_replica_content_and_topology_are_bound_to_fusion(self) -> None:
        replicas, _ = self._chain_replicas()
        local = FiniteHorizonPlanner(replicas[0], min_predicted_success=0.1)
        self.assertTrue(local.plan_goal("s0", "s7", horizon=7).abstained)

        barbell = ActionConditionedWorldModel.fuse_ps_lifted(
            replicas,
            barbell_adjacency(2, 2),
            tolerance=1e-9,
            max_rounds=4096,
            topology="barbell-ps-lifted",
        )
        complete = ActionConditionedWorldModel.fuse_ps_lifted(
            replicas,
            complete_adjacency(4),
            tolerance=1e-9,
            max_rounds=4096,
            topology="complete-ps-lifted",
        )
        self.assertNotEqual(
            barbell.receipt.adjacency_sha256, complete.receipt.adjacency_sha256
        )
        self.assertNotEqual(
            barbell.receipt.lifted_transition_sha256,
            complete.receipt.lifted_transition_sha256,
        )
        self.assertNotEqual(barbell.receipt.sha256, complete.receipt.sha256)
        self.assertTrue(barbell.receipt.adaptive_lift)
        self.assertTrue(complete.receipt.adaptive_lift)
        self.assertGreater(barbell.receipt.pc, complete.receipt.pc)
        self.assertEqual(len(barbell.receipt.lift_parameters_sha256), 64)
        np.testing.assert_allclose(
            barbell.model.counts, complete.model.counts, atol=1e-8
        )
        self.assertFalse(
            FiniteHorizonPlanner(barbell.model)
            .plan_goal("s0", "s7", horizon=7)
            .abstained
        )

        duplicate = replicas[0].clone()
        with self.assertRaisesRegex(ValueError, "duplicate transition evidence"):
            ActionConditionedWorldModel.fuse_ps_lifted(
                (replicas[0], duplicate), complete_adjacency(2)
            )

    def test_placebo_shuffled_action_kernels_collapse_under_real_execution(
        self,
    ) -> None:
        states = tuple(f"s{index}" for index in range(5))
        actions = ("advance", "reset")
        real = ActionConditionedWorldModel(states, actions)
        shuffled = ActionConditionedWorldModel(states, actions)
        for index, state in enumerate(states[:-1]):
            real.observe(
                _evidence(
                    state,
                    "advance",
                    states[index + 1],
                    name=f"real-advance-{index}",
                )
            )
            real.observe(_evidence(state, "reset", "s0", name=f"real-reset-{index}"))
            # Placebo swaps action-conditioned rows while keeping the same
            # coverage, event count and apparent deterministic confidence.
            shuffled.observe(
                _evidence(
                    state,
                    "reset",
                    states[index + 1],
                    name=f"shuffled-reset-{index}",
                )
            )
            shuffled.observe(
                _evidence(
                    state,
                    "advance",
                    "s0",
                    name=f"shuffled-advance-{index}",
                )
            )

        real_planner = FiniteHorizonPlanner(real, min_predicted_success=0.99)
        fake_planner = FiniteHorizonPlanner(shuffled, min_predicted_success=0.99)
        real_plan = real_planner.plan_goal("s0", "s4", horizon=4).plan
        fake_plan = fake_planner.plan_goal("s0", "s4", horizon=4).plan
        assert real_plan is not None and fake_plan is not None
        self.assertEqual(real_plan.expected_actions, ("advance",) * 4)
        self.assertEqual(fake_plan.expected_actions, ("reset",) * 4)

        def real_world(source: str, action: str, ordinal: int) -> ExecutionOutcome:
            target = (
                states[min(states.index(source) + 1, len(states) - 1)]
                if action == "advance"
                else "s0"
            )
            return ExecutionOutcome(
                target_state=target,
                verifier_sha256=_hash(f"truth-verifier:{ordinal}"),
                evidence_sha256=_hash(f"truth:{source}:{action}:{ordinal}"),
            )

        real_receipt = real_planner.execute(real_plan, real_world)
        placebo_receipt = fake_planner.execute(fake_plan, real_world)
        self.assertTrue(real_receipt.success)
        self.assertFalse(placebo_receipt.success)
        self.assertEqual(placebo_receipt.final_state, "s0")
        self.assertEqual(placebo_receipt.reason, "policy-abstained")

    def test_horizon_and_stale_plan_are_hard_gates(self) -> None:
        model = ActionConditionedWorldModel(("s0", "s1"), ("advance",))
        model.observe(_evidence("s0", "advance", "s1", name="first"))
        planner = FiniteHorizonPlanner(model, max_horizon=2)
        with self.assertRaisesRegex(ValueError, "horizon"):
            planner.plan_goal("s0", "s1", horizon=3)
        plan = planner.plan_goal("s0", "s1", horizon=1).plan
        assert plan is not None
        model.observe(_evidence("s0", "advance", "s1", name="second"))
        with self.assertRaisesRegex(ValueError, "another world-model revision"):
            planner.execute(
                plan,
                lambda source, action, ordinal: ExecutionOutcome(
                    target_state="s1",
                    verifier_sha256=_hash("v"),
                    evidence_sha256=_hash("e"),
                ),
            )

    def test_plan_freezes_epistemic_gates_and_rejects_gate_hash_tamper(self) -> None:
        model = ActionConditionedWorldModel(("s0", "s1"), ("advance",))
        model.observe(_evidence("s0", "advance", "s1", name="gate-edge"))
        planner = FiniteHorizonPlanner(
            model,
            min_evidence_mass=0.5,
            max_normalized_entropy=0.8,
            min_peak_probability=0.7,
            min_predicted_success=0.9,
        )
        plan = planner.plan_goal("s0", "s1", horizon=1).plan
        assert plan is not None

        def execute(source: str, action: str, ordinal: int) -> ExecutionOutcome:
            return ExecutionOutcome(
                target_state="s1",
                verifier_sha256=_hash(f"gate-verifier:{ordinal}"),
                evidence_sha256=_hash(f"gate-execution:{ordinal}"),
            )

        baseline = planner.execute(plan, execute)
        self.assertEqual(plan.min_evidence_mass, 0.5)
        self.assertEqual(plan.max_normalized_entropy, 0.8)
        self.assertEqual(plan.min_peak_probability, 0.7)
        self.assertEqual(plan.min_predicted_success, 0.9)
        self.assertEqual(baseline.gate_config_sha256, plan.gate_config_sha256)

        # Runtime policy knobs may change for future plans.  This already
        # issued plan keeps the exact epistemic contract it was derived under.
        planner.min_evidence_mass = 10_000.0
        planner.max_normalized_entropy = 0.0
        planner.min_peak_probability = 1.0
        planner.min_predicted_success = 1.0
        replay = planner.execute(plan, execute)
        self.assertEqual(replay.to_dict(), baseline.to_dict())
        self.assertEqual(replay.sha256, baseline.sha256)

        with self.assertRaisesRegex(ValueError, "planning gate hash mismatch"):
            replace(plan, gate_config_sha256=_hash("tampered-gate-contract"))
        with self.assertRaisesRegex(ValueError, "planning gate hash mismatch"):
            replace(plan, min_evidence_mass=0.75)
        bypassed_constructor = replace(plan)
        object.__setattr__(
            bypassed_constructor,
            "gate_config_sha256",
            _hash("post-construction-tamper"),
        )
        with self.assertRaisesRegex(ValueError, "planning gate hash mismatch"):
            planner.execute(bypassed_constructor, execute)


if __name__ == "__main__":
    unittest.main()
