from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from immer.runtimes.ooe.affine_monoid import (
    AffineProgram,
    build_decimal_horner_machine,
)
from immer.runtimes.ooe.algebra_agents import (
    AlgebraRouterBank,
    AlgebraRouterState,
    OperatorAlgebraCandidate,
)
from immer.runtimes.ooe.compute_crystals import (
    AFFINE_FLOAT64,
    CAUSAL_MIX_FLOAT64,
    ComputeCrystal,
    ComputeCrystalBank,
)
from immer.runtimes.ooe.compute_graph import ComputeOperatorGraph, OperatorEdge
from immer.runtimes.ooe.harvest_algebra_bridge import (
    ComputeAlgebraExecutionReceipt,
    HarvestAlgebraArtifactBank,
    HarvestAlgebraBridgeIntegrityError,
    HarvestAlgebraBridgeReceipt,
    HarvestedCandidateProfile,
    admit_harvest_promotion,
    execute_and_observe_compute_candidate,
    execute_selected_compute_candidate,
    execution_verifier_sha256_for_harvest,
    harvest_promotion_to_algebra_candidate,
    profile_harvested_candidate,
)
from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.operator_harvester import (
    CandidateEvidenceStream,
    HarvestPromotion,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class _ExactArrayVerifier:
    def __init__(self, expected: np.ndarray, *, accepted: bool = True) -> None:
        self.expected = expected
        self.accepted = accepted
        self.verifier_sha256 = _hash("compute-execution-verifier/v1")

    def verify(self, candidate, execution):
        matched = np.array_equal(execution.output, self.expected)
        accepted = bool(self.accepted and matched)
        return accepted, canonical_json_bytes(
            {
                "accepted": accepted,
                "candidate_sha256": candidate.sha256,
                "output_sha256": execution.receipt.output_sha256,
                "schema": "test-exact-array-verifier/v1",
            }
        )


class HarvestAlgebraBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bank = ComputeCrystalBank(self.root / "compute")
        self.graph = ComputeOperatorGraph(self.bank)
        self.crystal = ComputeCrystal.affine(
            np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64),
            np.array([1.0, 1.0], dtype=np.float64),
            extensions={
                "operator_harvester": {
                    "granularity": "operator",
                    "group_sha256": _hash("group"),
                }
            },
        )
        publication = self.bank.publish_crystal(self.crystal)
        self.discovery_verifier = _hash("heldout-discovery-verifier")
        self.evidence = _hash("heldout-fit-evidence")
        self.edge = OperatorEdge(
            source_state="qwen.hidden.pre",
            target_state="qwen.hidden.post",
            crystal_sha256=self.crystal.sha256,
            verifier_sha256=self.discovery_verifier,
            evidence_sha256=self.evidence,
            weight=4.0,
        )
        initial = self.graph.state()
        graph_state, changed = self.graph.append_edges(
            (self.edge,),
            expected_generation=initial.generation,
            expected_state_sha256=initial.sha256,
        )
        self.assertTrue(changed)
        observations = tuple(sorted(_hash(f"context-{index}") for index in range(4)))
        fit = tuple(sorted(observations[:3]))
        stream = CandidateEvidenceStream(
            group_sha256=_hash("group"),
            operator_kind=AFFINE_FLOAT64,
            status="promoted",
            reason="affine-held-out-verification-passed",
            observation_receipt_sha256s=observations,
            fit_receipt_sha256s=fit,
            holdout_receipt_sha256=observations[-1],
            verifier_sha256=self.discovery_verifier,
            crystal_sha256=self.crystal.sha256,
            evidence_sha256=self.evidence,
        )
        self.promotion = HarvestPromotion(
            candidate=stream,
            edge=self.edge,
            bank_publication=publication,
            graph_changed=True,
            graph_state_sha256=graph_state.sha256,
        )
        self.execution_verifier = _hash("compute-execution-verifier/v1")

    def _profile(self) -> HarvestedCandidateProfile:
        return profile_harvested_candidate(
            self.promotion,
            self.crystal,
            bin_counts=(3, 4),
            objective_count=2,
            execution_verifier_sha256=self.execution_verifier,
        )

    def _candidate(self):
        return harvest_promotion_to_algebra_candidate(
            self.promotion,
            graph=self.graph,
            profile=self._profile(),
        )

    def test_harvested_operator_routes_executes_and_updates_thompson(self) -> None:
        candidate, publication, bridge = self._candidate()
        self.assertEqual(candidate.program_runtime, "compute-crystal-vm")
        self.assertEqual(candidate.discovery_verifier_sha256, self.discovery_verifier)
        self.assertEqual(candidate.verifier_sha256, self.execution_verifier)
        self.assertEqual(publication.artifact_kind, "program")
        self.assertEqual(
            HarvestedCandidateProfile.from_bytes(self._profile().to_bytes()),
            self._profile(),
        )
        self.assertEqual(
            HarvestAlgebraBridgeReceipt.from_bytes(bridge.to_bytes()),
            bridge,
        )

        router = AlgebraRouterState.bootstrap(
            (candidate,),
            seed_sha256=_hash("harvest-router-seed"),
            bin_counts=(3, 4),
            objective_count=2,
        )
        selected, selection, advanced = router.choose("qwen.hidden.pre")
        value = np.array([[3.0, -2.0], [0.5, 9.0]], dtype=np.float64)
        expected = 2.0 * value + 1.0
        updated, update, execution = execute_and_observe_compute_candidate(
            advanced,
            selection,
            selected,
            bank=self.bank,
            value=value,
            verifier=_ExactArrayVerifier(expected),
        )
        np.testing.assert_array_equal(execution.output, expected)
        self.assertFalse(execution.output.flags.writeable)
        self.assertTrue(execution.outcome.success)
        self.assertEqual(update.posterior_successes, 1)
        self.assertEqual(update.posterior_failures, 0)
        self.assertEqual(updated.bandit.decision_index, 1)
        self.assertEqual(
            ComputeAlgebraExecutionReceipt.from_bytes(execution.receipt.to_bytes()),
            execution.receipt,
        )

    def test_convenience_admission_places_compute_program_in_existing_router(
        self,
    ) -> None:
        machine = build_decimal_horner_machine(max_digits=8)
        program = AffineProgram(
            machine.schema,
            (machine.action("digit:1"), machine.action("finish")),
            "BridgeBaseline",
        )
        baseline = OperatorAlgebraCandidate(
            family="decimal",
            program=program,
            verifier_sha256=_hash("decimal-verifier"),
            evidence_sha256s=(_hash("decimal-evidence"),),
            behavior_descriptor=(2, 0),
            objectives=(0.5, 0.5),
        )
        router = AlgebraRouterState.bootstrap(
            (baseline,),
            seed_sha256=_hash("mixed-router-seed"),
            bin_counts=(3, 4),
            objective_count=2,
        )
        admitted = admit_harvest_promotion(
            self.promotion,
            graph=self.graph,
            router=router,
            execution_verifier_sha256=self.execution_verifier,
        )
        self.assertTrue(admitted.admission_receipt.admitted)
        self.assertIn(
            admitted.candidate.sha256,
            {candidate.sha256 for candidate in admitted.router.candidates},
        )
        self.assertEqual(
            admitted.bridge_receipt.profile_sha256, admitted.profile.sha256
        )

    def test_profile_bridge_and_candidate_restore_as_one_catalog(self) -> None:
        profile = self._profile()
        candidate, _publication, bridge = harvest_promotion_to_algebra_candidate(
            self.promotion,
            graph=self.graph,
            profile=profile,
        )
        router = AlgebraRouterState.bootstrap(
            (candidate,),
            seed_sha256=_hash("persistent-router-seed"),
            bin_counts=(3, 4),
            objective_count=2,
        )
        artifacts = HarvestAlgebraArtifactBank(self.bank.store)
        published = artifacts.publish(
            candidate,
            profile,
            bridge,
            admission_router_sha256=router.sha256,
            admission_receipt_sha256=None,
            bootstrap=True,
        )
        AlgebraRouterBank(self.bank.store).publish("harvested", router)

        restored_router = AlgebraRouterBank(self.bank.store).restore("harvested")
        reopened = HarvestAlgebraArtifactBank(self.bank.store)
        self.assertEqual(reopened.audit_router(restored_router), (published.record,))
        self.assertEqual(
            reopened.restore_profile(published.record.profile_sha256), profile
        )
        self.assertEqual(
            reopened.restore_bridge(published.record.bridge_receipt_sha256), bridge
        )
        repeated = reopened.publish(
            candidate,
            profile,
            replace(
                bridge,
                program_publication_manifest_sha256=self.bank.manifest().sha256,
            ),
            admission_router_sha256=router.sha256,
            admission_receipt_sha256=None,
            bootstrap=True,
        )
        self.assertEqual(repeated.record, published.record)
        self.assertFalse(repeated.catalog.changed)

    def test_execution_verifier_identity_is_specific_to_family_contract(self) -> None:
        first = execution_verifier_sha256_for_harvest(self.promotion, self.crystal)
        other = replace(
            self.promotion,
            edge=replace(
                self.edge,
                source_state="another.hidden.pre",
                target_state="another.hidden.post",
            ),
        )
        second = execution_verifier_sha256_for_harvest(other, self.crystal)
        self.assertNotEqual(first, second)

    def test_dedicated_causal_mix_promotion_enters_compute_algebra(self) -> None:
        kernel = np.array(
            [[1.0, 0.0, 0.0], [0.6, 0.4, 0.0], [0.2, 0.3, 0.5]],
            dtype=np.float64,
        )
        crystal = ComputeCrystal.causal_mix(kernel)
        publication = self.bank.publish_crystal(crystal)
        verifier = _hash("causal-mix-discovery")
        evidence = _hash("causal-mix-evidence")
        edge = OperatorEdge(
            source_state="prefix.values.before",
            target_state="prefix.values.after",
            crystal_sha256=crystal.sha256,
            verifier_sha256=verifier,
            evidence_sha256=evidence,
            weight=3.0,
        )
        state = self.graph.state()
        graph_state, _changed = self.graph.append_edges(
            (edge,),
            expected_generation=state.generation,
            expected_state_sha256=state.sha256,
        )
        observations = tuple(sorted(_hash(f"causal-context-{i}") for i in range(3)))
        stream = CandidateEvidenceStream(
            group_sha256=_hash("causal-mix-group"),
            operator_kind=CAUSAL_MIX_FLOAT64,
            status="promoted",
            reason="causal-kernel-contract-passed",
            observation_receipt_sha256s=observations,
            fit_receipt_sha256s=tuple(sorted(observations[:2])),
            holdout_receipt_sha256=observations[-1],
            verifier_sha256=verifier,
            crystal_sha256=crystal.sha256,
            evidence_sha256=evidence,
        )
        promotion = HarvestPromotion(
            candidate=stream,
            edge=edge,
            bank_publication=publication,
            graph_changed=True,
            graph_state_sha256=graph_state.sha256,
        )
        profile = profile_harvested_candidate(
            promotion,
            crystal,
            bin_counts=(4, 4),
            objective_count=2,
            execution_verifier_sha256=execution_verifier_sha256_for_harvest(
                promotion, crystal
            ),
        )
        candidate, _program_publication, bridge = (
            harvest_promotion_to_algebra_candidate(
                promotion,
                graph=self.graph,
                profile=profile,
            )
        )
        self.assertEqual(candidate.program.crystal_sha256s, (crystal.sha256,))
        self.assertEqual(candidate.behavior_descriptor[0], 3)
        self.assertEqual(bridge.crystal_sha256, crystal.sha256)

    def test_negative_execution_is_real_router_feedback_not_a_crash(self) -> None:
        candidate, _publication, _bridge = self._candidate()
        router = AlgebraRouterState.bootstrap(
            (candidate,),
            seed_sha256=_hash("negative-router-seed"),
            bin_counts=(3, 4),
            objective_count=2,
        )
        selected, selection, advanced = router.choose("qwen.hidden.pre")
        value = np.array([[1.0, 2.0]], dtype=np.float64)
        updated, update, execution = execute_and_observe_compute_candidate(
            advanced,
            selection,
            selected,
            bank=self.bank,
            value=value,
            verifier=_ExactArrayVerifier(2.0 * value + 1.0, accepted=False),
        )
        self.assertFalse(execution.outcome.success)
        self.assertEqual(update.posterior_successes, 0)
        self.assertEqual(update.posterior_failures, 1)
        self.assertEqual(updated.bandit.decision_index, 1)

    def test_stale_graph_candidate_selection_and_verifier_splices_reject(self) -> None:
        candidate, _publication, _bridge = self._candidate()
        router = AlgebraRouterState.bootstrap(
            (candidate,),
            seed_sha256=_hash("splice-router-seed"),
            bin_counts=(3, 4),
            objective_count=2,
        )
        selected, selection, _advanced = router.choose("qwen.hidden.pre")
        other_candidate = replace(
            candidate,
            family="other-compute",
            evidence_sha256s=(_hash("other-candidate-evidence"),),
        )
        other_router = AlgebraRouterState.bootstrap(
            (other_candidate,),
            seed_sha256=_hash("other-router-seed"),
            bin_counts=(3, 4),
            objective_count=2,
        )
        _other_selected, other, _other_advanced = other_router.choose("qwen.hidden.pre")
        value = np.array([[1.0, 2.0]], dtype=np.float64)
        with self.assertRaisesRegex(
            HarvestAlgebraBridgeIntegrityError,
            "selection and compute candidate",
        ):
            execute_selected_compute_candidate(
                other,
                selected,
                bank=self.bank,
                value=value,
                verifier=_ExactArrayVerifier(2.0 * value + 1.0),
            )
        wrong_verifier = _ExactArrayVerifier(2.0 * value + 1.0)
        wrong_verifier.verifier_sha256 = _hash("wrong-verifier")
        with self.assertRaisesRegex(
            HarvestAlgebraBridgeIntegrityError,
            "execution verifier",
        ):
            execute_selected_compute_candidate(
                selection,
                selected,
                bank=self.bank,
                value=value,
                verifier=wrong_verifier,
            )

        extra = ComputeCrystal.permutation((1, 0))
        self.bank.publish_crystal(extra)
        extra_edge = OperatorEdge(
            source_state="other.pre",
            target_state="other.post",
            crystal_sha256=extra.sha256,
            verifier_sha256=_hash("other-discovery"),
            evidence_sha256=_hash("other-evidence"),
        )
        state = self.graph.state()
        self.graph.append_edges(
            (extra_edge,),
            expected_generation=state.generation,
            expected_state_sha256=state.sha256,
        )
        with self.assertRaisesRegex(
            HarvestAlgebraBridgeIntegrityError,
            "stale",
        ):
            self._candidate()

    def test_receipt_tamper_and_candidate_bank_mismatch_reject(self) -> None:
        candidate, _publication, bridge = self._candidate()
        document = json.loads(bridge.to_bytes())
        document["body"]["candidate_sha256"] = _hash("tampered-candidate")
        with self.assertRaises(HarvestAlgebraBridgeIntegrityError):
            HarvestAlgebraBridgeReceipt.from_bytes(canonical_json_bytes(document))

        mismatched_stream = replace(
            self.promotion.candidate,
            crystal_sha256=_hash("another-crystal"),
        )
        mismatched = replace(self.promotion, candidate=mismatched_stream)
        with self.assertRaisesRegex(
            HarvestAlgebraBridgeIntegrityError,
            "candidate, edge, and bank",
        ):
            harvest_promotion_to_algebra_candidate(
                mismatched,
                graph=self.graph,
                profile=self._profile(),
            )
        self.assertEqual(candidate.program.crystal_sha256s, (self.crystal.sha256,))


if __name__ == "__main__":
    unittest.main()
