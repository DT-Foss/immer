from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import tempfile
import unittest

from immer.runtimes.ooe.affine_monoid import (
    AffineMonoidRuntime,
    AffineProgram,
    build_decimal_horner_machine,
    build_fingerprint_machine,
    build_stack_machine,
)
from immer.runtimes.ooe.algebra_agents import (
    AlgebraAgentIntegrityError,
    AlgebraRouterBank,
    AlgebraRouterState,
    AlgebraSelectionReceipt,
    AlgebraUpdateReceipt,
    ComputeProgramBindingReceipt,
    ExactVerifierArtifact,
    HeterogeneousABIError,
    HeterogeneousEnsembleResult,
    HeterogeneousProgramEnsemble,
    OperatorAlgebraCandidate,
    ParallelLaneResult,
    ParallelProgramLane,
    VerifierBoundOutcome,
)
from immer.runtimes.ooe.compute_crystals import ComputeCrystal, ComputeProgram
from immer.runtimes.ooe.crystal import CrystalStore, ManifestConflictError
from immer.runtimes.ooe.identity import canonical_json_bytes


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _programs() -> dict[str, AffineProgram]:
    stack = build_stack_machine(capacity=4, symbols=(1, 2, 3))
    fingerprint = build_fingerprint_machine(max_length=8, base=17, moduli=(101, 103))
    decimal = build_decimal_horner_machine(max_digits=8)
    return {
        "stack": AffineProgram(
            stack.schema,
            (stack.push(1), stack.push(2), stack.action("peek")),
            "StackAgentProgram",
        ),
        "fingerprint": AffineProgram(
            fingerprint.schema,
            (
                fingerprint.left(ord("x")),
                fingerprint.separator(),
                fingerprint.right(ord("x")),
                fingerprint.finish(),
            ),
            "FingerprintAgentProgram",
        ),
        "decimal": AffineProgram(
            decimal.schema,
            (
                decimal.action("digit:1"),
                decimal.action("digit:2"),
                decimal.action("digit:3"),
                decimal.action("finish"),
            ),
            "DecimalAgentProgram",
        ),
    }


def _candidates() -> tuple[OperatorAlgebraCandidate, ...]:
    programs = _programs()
    descriptors = {
        "stack": (0, 0),
        "fingerprint": (1, 0),
        "decimal": (2, 0),
    }
    return tuple(
        OperatorAlgebraCandidate(
            family=family,
            program=programs[family],
            verifier_sha256=_hash(f"{family}-verifier"),
            evidence_sha256s=tuple(
                sorted((_hash(f"{family}-evidence-a"), _hash(f"{family}-evidence-b")))
            ),
            behavior_descriptor=descriptors[family],
            objectives=(1.0, 1.0),
        )
        for family in ("stack", "fingerprint", "decimal")
    )


def _compute_program() -> ComputeProgram:
    return ComputeProgram.compose(
        (
            ComputeCrystal.markov(((0.75, 0.25), (0.125, 0.875))),
            ComputeCrystal.markov(((0.9, 0.1), (0.2, 0.8))),
        )
    )


def _compute_candidate() -> OperatorAlgebraCandidate:
    return OperatorAlgebraCandidate(
        family="markov-kernel",
        program=_compute_program(),
        verifier_sha256=_hash("markov-execution-verifier"),
        discovery_verifier_sha256=_hash("markov-discovery-verifier"),
        evidence_sha256s=(_hash("markov-discovery-evidence"),),
        behavior_descriptor=(0,),
        objectives=(1.0,),
    )


def _router() -> AlgebraRouterState:
    return AlgebraRouterState.bootstrap(
        _candidates(),
        seed_sha256=_hash("algebra-router-seed"),
        bin_counts=(3, 2),
        objective_count=2,
        max_elites_per_cell=2,
    )


def _step(
    state: AlgebraRouterState,
    context: str,
    winner: str,
    *,
    nonce: int,
) -> tuple[AlgebraRouterState, str]:
    candidate, selection, advanced = state.choose(context)
    success = candidate.family == winner
    outcome = VerifierBoundOutcome.issue(
        selection,
        candidate,
        receipt_payload=canonical_json_bytes(
            {
                "candidate_sha256": candidate.sha256,
                "nonce": nonce,
                "success": success,
                "verifier_sha256": candidate.verifier_sha256,
            }
        ),
        success=success,
    )
    updated, receipt = advanced.observe(selection, outcome)
    if success:
        assert receipt.posterior_successes >= 1
    else:
        assert receipt.posterior_failures >= 1
    return updated, candidate.family


def _train(
    state: AlgebraRouterState,
    winners: dict[str, str],
    *,
    rounds: int,
) -> AlgebraRouterState:
    nonce = 0
    for _ in range(rounds):
        for context, winner in sorted(winners.items()):
            state, _ = _step(state, context, winner, nonce=nonce)
            nonce += 1
    return state


def _evaluate(
    state: AlgebraRouterState,
    winners: dict[str, str],
    *,
    rounds: int,
) -> tuple[int, AlgebraRouterState]:
    correct = 0
    for _ in range(rounds):
        for context, winner in sorted(winners.items()):
            candidate, _, state = state.choose(context)
            correct += candidate.family == winner
    return correct, state


class OperatorAlgebraCandidateTests(unittest.TestCase):
    def test_candidate_binds_program_schema_verifier_evidence_and_behavior(
        self,
    ) -> None:
        candidate = _candidates()[0]
        restored = OperatorAlgebraCandidate.from_bytes(candidate.to_bytes())
        self.assertEqual(restored, candidate)
        self.assertEqual(restored.program_sha256, candidate.program.sha256)
        self.assertEqual(restored.schema_sha256, candidate.program.schema.sha256)
        self.assertEqual(
            restored.program_receipt.program_sha256, candidate.program.sha256
        )

        tampered = json.loads(candidate.to_bytes())
        tampered["body"]["schema_contract_sha256"] = _hash("foreign-schema")
        tampered["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(tampered["body"])
        ).hexdigest()
        with self.assertRaisesRegex(AlgebraAgentIntegrityError, "derived"):
            OperatorAlgebraCandidate.from_bytes(canonical_json_bytes(tampered))

    def test_noncanonical_and_unhashed_tamper_fail_closed(self) -> None:
        candidate = _candidates()[1]
        document = json.loads(candidate.to_bytes())
        document["body"]["family"] = "decimal"
        with self.assertRaisesRegex(AlgebraAgentIntegrityError, "hash"):
            OperatorAlgebraCandidate.from_bytes(canonical_json_bytes(document))
        with self.assertRaises(AlgebraAgentIntegrityError):
            OperatorAlgebraCandidate.from_bytes(candidate.to_bytes() + b"\n")

    def test_affine_v1_hash_and_default_discovery_verifier_remain_exact(self) -> None:
        candidate = _candidates()[0]
        self.assertEqual(candidate.discovery_verifier_sha256, candidate.verifier_sha256)
        document = json.loads(candidate.to_bytes())
        self.assertEqual(document["schema"], "immer-ooe-operator-algebra-candidate/v2")
        self.assertEqual(document["body"]["program_runtime"], "affine-monoid")

        legacy_body = {
            "behavior_descriptor": list(candidate.behavior_descriptor),
            "evidence_sha256s": list(candidate.evidence_sha256s),
            "family": candidate.family,
            "objectives_hex": [value.hex() for value in candidate.objectives],
            "program_base64": document["body"]["program_base64"],
            "program_receipt_sha256": candidate.program_receipt.sha256,
            "program_sha256": candidate.program_sha256,
            "schema_sha256": candidate.schema_sha256,
            "verifier_sha256": candidate.verifier_sha256,
        }
        legacy_payload = canonical_json_bytes(
            {
                "body": legacy_body,
                "body_sha256": hashlib.sha256(
                    canonical_json_bytes(legacy_body)
                ).hexdigest(),
                "schema": "immer-ooe-operator-algebra-candidate/v1",
            }
        )
        legacy = OperatorAlgebraCandidate.from_bytes(legacy_payload)
        self.assertEqual(legacy.to_bytes(), legacy_payload)
        self.assertEqual(
            legacy.sha256,
            "1f068f3b22059b5f66a08503a6113770686d69186cef70ed3077addd03b7028d",
        )

        separated = replace(
            candidate,
            discovery_verifier_sha256=_hash("separate-affine-discovery-verifier"),
        )
        self.assertEqual(
            json.loads(separated.to_bytes())["body"]["program_runtime"],
            "affine-monoid",
        )
        self.assertEqual(
            OperatorAlgebraCandidate.from_bytes(separated.to_bytes()), separated
        )

    def test_compute_candidate_and_program_binding_roundtrip_exactly(self) -> None:
        candidate = _compute_candidate()
        self.assertEqual(candidate.program_runtime, "compute-crystal-vm")
        self.assertNotEqual(
            candidate.discovery_verifier_sha256, candidate.verifier_sha256
        )
        self.assertIsInstance(candidate.program_receipt, ComputeProgramBindingReceipt)
        receipt = candidate.program_receipt
        self.assertEqual(
            ComputeProgramBindingReceipt.from_bytes(receipt.to_bytes()), receipt
        )
        self.assertEqual(receipt.program_sha256, candidate.program_sha256)
        self.assertEqual(
            receipt.schema_contract_sha256, candidate.schema_contract_sha256
        )
        self.assertEqual(receipt.program_bytes_sha256, candidate.program_bytes_sha256)
        restored = OperatorAlgebraCandidate.from_bytes(candidate.to_bytes())
        self.assertEqual(restored, candidate)
        document = json.loads(candidate.to_bytes())
        self.assertEqual(document["schema"], "immer-ooe-operator-algebra-candidate/v2")
        self.assertEqual(document["body"]["program_runtime"], "compute-crystal-vm")

    def test_compute_candidate_rejects_resealed_binding_and_runtime_tag_tamper(
        self,
    ) -> None:
        candidate = _compute_candidate()
        document = json.loads(candidate.to_bytes())
        document["body"]["schema_contract_sha256"] = _hash("foreign-contract")
        document["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(document["body"])
        ).hexdigest()
        with self.assertRaisesRegex(AlgebraAgentIntegrityError, "derived"):
            OperatorAlgebraCandidate.from_bytes(canonical_json_bytes(document))

        document = json.loads(candidate.to_bytes())
        document["body"]["program_runtime"] = "affine-monoid"
        document["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(document["body"])
        ).hexdigest()
        with self.assertRaisesRegex(AlgebraAgentIntegrityError, "invalid"):
            OperatorAlgebraCandidate.from_bytes(canonical_json_bytes(document))

        with self.assertRaisesRegex(ValueError, "discovery verifier"):
            OperatorAlgebraCandidate(
                family="unbound-compute",
                program=_compute_program(),
                verifier_sha256=_hash("execution-only"),
                evidence_sha256s=(_hash("evidence"),),
                behavior_descriptor=(0,),
                objectives=(1.0,),
            )


class ContextualAlgebraRouterTests(unittest.TestCase):
    def test_compute_candidate_router_choose_observe_and_restart(self) -> None:
        candidate = _compute_candidate()
        state = AlgebraRouterState.bootstrap(
            (candidate,),
            seed_sha256=_hash("compute-router-seed"),
            bin_counts=(1,),
            objective_count=1,
        )
        chosen, selection, advanced = state.choose("compute-context")
        self.assertEqual(chosen, candidate)
        self.assertEqual(selection.program_sha256, candidate.program_sha256)
        self.assertEqual(selection.schema_sha256, candidate.schema_contract_sha256)
        outcome = VerifierBoundOutcome.issue(
            selection,
            chosen,
            receipt_payload=b"compute-vm-execution-receipt",
            success=True,
        )
        updated, receipt = advanced.observe(selection, outcome)
        self.assertEqual(receipt.posterior_successes, 1)
        restored = AlgebraRouterState.from_bytes(updated.to_bytes())
        self.assertEqual(restored, updated)
        next_candidate, _, _ = restored.choose("compute-context")
        self.assertEqual(next_candidate, candidate)

    def test_contexts_learn_different_stack_fingerprint_and_decimal_winners(
        self,
    ) -> None:
        winners = {
            "context-stack": "stack",
            "context-fingerprint": "fingerprint",
            "context-decimal": "decimal",
        }
        trained = _train(_router(), winners, rounds=80)
        correct, _ = _evaluate(trained, winners, rounds=40)
        self.assertGreaterEqual(correct, 116)
        for context, family in winners.items():
            candidate = next(
                item for item in trained.candidates if item.family == family
            )
            posterior = trained.bandit.posterior(context, candidate.sha256)
            self.assertGreater(posterior.successes, posterior.failures)

    def test_unseen_context_explores_and_replays_deterministically(self) -> None:
        first = _router()
        second = AlgebraRouterState.from_bytes(first.to_bytes())
        first_families: list[str] = []
        second_families: list[str] = []
        for _ in range(32):
            candidate, selection, first = first.choose("unseen-context")
            first_families.append(candidate.family)
            self.assertEqual(
                AlgebraSelectionReceipt.from_bytes(selection.to_bytes()), selection
            )
            candidate, _, second = second.choose("unseen-context")
            second_families.append(candidate.family)
        self.assertGreaterEqual(len(set(first_families)), 2)
        self.assertEqual(first_families, second_families)
        self.assertEqual(first, second)

    def test_verifier_bound_updates_reject_swap_replay_and_recover_from_failure(
        self,
    ) -> None:
        state = _train(_router(), {"recovering-context": "stack"}, rounds=45)
        before_correct, state = _evaluate(
            state, {"recovering-context": "stack"}, rounds=20
        )
        self.assertGreaterEqual(before_correct, 18)

        # The environment changes.  Verified negative evidence drives the old
        # algebra down while positive evidence raises the replacement.
        state = _train(state, {"recovering-context": "decimal"}, rounds=140)
        recovered, state = _evaluate(
            state, {"recovering-context": "decimal"}, rounds=30
        )
        self.assertGreaterEqual(recovered, 27)

        candidate, selection, advanced = state.choose("recovering-context")
        outcome = VerifierBoundOutcome.issue(
            selection,
            candidate,
            receipt_payload=b"exact-verifier-receipt",
            success=True,
        )
        wrong = replace(outcome, verifier_sha256=_hash("wrong-verifier"))
        with self.assertRaisesRegex(AlgebraAgentIntegrityError, "binding"):
            advanced.observe(selection, wrong)
        updated, receipt = advanced.observe(selection, outcome)
        self.assertEqual(AlgebraUpdateReceipt.from_bytes(receipt.to_bytes()), receipt)
        with self.assertRaisesRegex(AlgebraAgentIntegrityError, "selected next state"):
            updated.observe(selection, outcome)

    def test_handcrafted_current_state_selection_cannot_poison_context(self) -> None:
        state = _router()
        candidate, selection, advanced = state.choose("real-context")
        forged = replace(
            selection,
            context="forged-context",
            decision_index=999,
            router_parent_sha256=_hash("forged-parent"),
            thompson_choice_sha256=_hash("forged-choice"),
        )
        outcome = VerifierBoundOutcome.issue(
            forged,
            candidate,
            receipt_payload=b"forged-selection-outcome",
            success=True,
        )
        with self.assertRaisesRegex(
            AlgebraAgentIntegrityError, "decision index|reproducible|deterministic"
        ):
            advanced.observe(forged, outcome)
        self.assertEqual(
            advanced.bandit.posterior("forged-context", candidate.sha256).successes,
            0,
        )

    def test_shuffled_context_placebo_degrades_routing(self) -> None:
        winners = {
            "ctx-stack": "stack",
            "ctx-fingerprint": "fingerprint",
            "ctx-decimal": "decimal",
        }
        trained = _train(_router(), winners, rounds=70)
        exact, state = _evaluate(trained, winners, rounds=30)
        shuffled = {
            "ctx-stack": "decimal",
            "ctx-fingerprint": "stack",
            "ctx-decimal": "fingerprint",
        }
        placebo, _ = _evaluate(state, shuffled, rounds=30)
        self.assertGreaterEqual(exact, 86)
        self.assertLessEqual(placebo, 4)

    def test_map_elites_diversity_dominance_and_exact_posterior_retention(self) -> None:
        state = _train(_router(), {"ctx-stack": "stack"}, rounds=20)
        stack = next(item for item in state.candidates if item.family == "stack")
        stack_posterior = state.bandit.posterior("ctx-stack", stack.sha256)
        programs = _programs()

        dominated = OperatorAlgebraCandidate(
            family="dominated-stack",
            program=programs["stack"],
            verifier_sha256=_hash("dominated-verifier"),
            evidence_sha256s=(_hash("dominated-evidence"),),
            behavior_descriptor=stack.behavior_descriptor,
            objectives=(0.5, 0.5),
        )
        unchanged, rejected = state.admit(dominated)
        self.assertIs(unchanged, state)
        self.assertFalse(rejected.admitted)

        dominant = OperatorAlgebraCandidate(
            family="dominant-stack",
            program=programs["stack"],
            verifier_sha256=_hash("dominant-verifier"),
            evidence_sha256s=(_hash("dominant-evidence"),),
            behavior_descriptor=stack.behavior_descriptor,
            objectives=(2.0, 2.0),
        )
        replaced_state, admitted = state.admit(dominant)
        self.assertTrue(admitted.admitted)
        self.assertIn(stack.sha256, admitted.evicted_candidate_sha256s)
        self.assertNotIn(
            stack.sha256, {item.sha256 for item in replaced_state.candidates}
        )

        diverse = OperatorAlgebraCandidate(
            family="diverse-decimal",
            program=programs["decimal"],
            verifier_sha256=_hash("diverse-verifier"),
            evidence_sha256s=(_hash("diverse-evidence"),),
            behavior_descriptor=(2, 1),
            objectives=(0.2, 0.2),
        )
        diverse_state, diversity_receipt = replaced_state.admit(diverse)
        self.assertTrue(diversity_receipt.admitted)
        self.assertEqual(diverse_state.archive.occupied_cells, 4)
        fingerprint = next(
            item for item in state.candidates if item.family == "fingerprint"
        )
        self.assertEqual(
            diverse_state.bandit.posterior("ctx-stack", fingerprint.sha256),
            state.bandit.posterior("ctx-stack", fingerprint.sha256),
        )
        self.assertGreater(stack_posterior.successes, 0)

    def test_router_roundtrip_restart_and_derived_tamper_rejection(self) -> None:
        trained = _train(_router(), {"restart-context": "decimal"}, rounds=25)
        restored = AlgebraRouterState.from_bytes(trained.to_bytes())
        self.assertEqual(restored, trained)
        first, _, first_next = trained.choose("restart-context")
        second, _, second_next = restored.choose("restart-context")
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(first_next, second_next)

        document = json.loads(trained.to_bytes())
        document["body"]["decision_index"] += 1
        document["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(document["body"])
        ).hexdigest()
        with self.assertRaisesRegex(AlgebraAgentIntegrityError, "derived"):
            AlgebraRouterState.from_bytes(canonical_json_bytes(document))

    def test_crystal_store_atomically_persists_router_and_catalog_with_cas(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CrystalStore(temporary)
            bank = AlgebraRouterBank(store)
            state = _router()
            first = bank.publish("primary", state)
            self.assertTrue(first.changed)
            self.assertEqual(bank.restore("primary"), state)
            second = bank.publish("primary", state, expected_sha256=state.sha256)
            self.assertFalse(second.changed)

            updated = _train(state, {"persistent-context": "stack"}, rounds=5)
            with self.assertRaises(ManifestConflictError):
                bank.publish("primary", updated, expected_sha256=_hash("stale"))
            bank.publish("primary", updated, expected_sha256=state.sha256)
            self.assertEqual(bank.restore("primary"), updated)

            store.publish_state(bank.state_name("primary"), b"{}")
            with self.assertRaises(AlgebraAgentIntegrityError):
                bank.restore("primary")


class HeterogeneousEnsembleTests(unittest.TestCase):
    @staticmethod
    def _ensemble() -> tuple[HeterogeneousProgramEnsemble, ExactVerifierArtifact]:
        programs = _programs()
        verifier = _hash("ensemble-fingerprint-verifier")
        artifact = ExactVerifierArtifact(
            verifier,
            canonical_json_bytes(
                {
                    "exact_equal": True,
                    "left_sha256": _hash("x"),
                    "right_sha256": _hash("x"),
                }
            ),
        )
        lanes = (
            ParallelProgramLane(
                "decimal",
                programs["decimal"],
                programs["decimal"].schema.state(),
            ),
            ParallelProgramLane(
                "fingerprint",
                programs["fingerprint"],
                programs["fingerprint"].schema.state(),
                (verifier,),
            ),
            ParallelProgramLane(
                "stack",
                programs["stack"],
                programs["stack"].schema.state(),
            ),
        )
        return HeterogeneousProgramEnsemble("ThreeIndependentABIs", lanes), artifact

    def test_parallel_ensemble_exact_outputs_receipts_and_no_sequential_abi(
        self,
    ) -> None:
        ensemble, artifact = self._ensemble()
        self.assertTrue(ensemble.heterogeneous)
        self.assertEqual(ensemble.abi_mode, "parallel-independent")
        with self.assertRaises(HeterogeneousABIError):
            ensemble.as_sequential_program()
        result = ensemble.execute({"fingerprint": (artifact,)})
        self.assertEqual(set(result.outputs), {"decimal", "fingerprint", "stack"})
        for lane in ensemble.lanes:
            direct = AffineMonoidRuntime.execute(
                lane.program,
                initial_state=lane.initial_state,
                verifier_sha256s=(artifact.sha256,)
                if lane.lane_id == "fingerprint"
                else (),
            )
            self.assertEqual(result.outputs[lane.lane_id], direct.state)
            lane_result = next(
                item for item in result.lane_results if item.lane_id == lane.lane_id
            )
            self.assertEqual(lane_result.program_receipt, direct.program_receipt)
            self.assertEqual(lane_result.execution_receipt, direct.execution_receipt)
        self.assertEqual(
            result.receipt.total_work_units,
            sum(item.execution_receipt.work_units for item in result.lane_results),
        )

    def test_parallel_affine_ensemble_rejects_compute_program_lanes(self) -> None:
        stack = _programs()["stack"]
        with self.assertRaisesRegex(TypeError, "AffineProgram"):
            ParallelProgramLane(
                "compute",
                _compute_program(),  # type: ignore[arg-type]
                stack.schema.state(),
            )

    def test_ensemble_roundtrip_deterministic_replay_and_artifact_join(self) -> None:
        ensemble, artifact = self._ensemble()
        first = ensemble.execute({"fingerprint": (artifact,)}, max_workers=1)
        second = ensemble.execute({"fingerprint": (artifact,)}, max_workers=3)
        self.assertEqual(first, second)
        self.assertEqual(
            HeterogeneousProgramEnsemble.from_bytes(ensemble.to_bytes()), ensemble
        )
        self.assertEqual(
            HeterogeneousEnsembleResult.from_bytes(first.to_bytes()), first
        )
        fingerprint = next(
            item for item in first.lane_results if item.lane_id == "fingerprint"
        )
        self.assertEqual(fingerprint.verifier_artifacts, (artifact,))
        self.assertIn(artifact.sha256, first.receipt.lane_joins[1][4])

    def test_swapped_missing_and_wrong_verifier_results_are_rejected(self) -> None:
        ensemble, artifact = self._ensemble()
        result = ensemble.execute({"fingerprint": (artifact,)})
        swapped = (
            result.lane_results[1],
            result.lane_results[0],
            result.lane_results[2],
        )
        with self.assertRaisesRegex(AlgebraAgentIntegrityError, "swapped"):
            HeterogeneousEnsembleResult(ensemble, swapped, result.receipt)
        with self.assertRaisesRegex(AlgebraAgentIntegrityError, "missing"):
            HeterogeneousEnsembleResult(
                ensemble, result.lane_results[:-1], result.receipt
            )
        wrong = ExactVerifierArtifact(_hash("other-verifier"), b"other")
        with self.assertRaisesRegex(AlgebraAgentIntegrityError, "does not match"):
            ensemble.execute({"fingerprint": (wrong,)})
        with self.assertRaisesRegex(AlgebraAgentIntegrityError, "does not match"):
            ensemble.execute()

    def test_standalone_lane_result_rejects_cross_program_receipt_splice(self) -> None:
        ensemble, artifact = self._ensemble()
        result = ensemble.execute({"fingerprint": (artifact,)})
        decimal = next(
            item for item in result.lane_results if item.lane_id == "decimal"
        )
        stack = next(item for item in result.lane_results if item.lane_id == "stack")
        decimal_lane = next(
            lane for lane in ensemble.lanes if lane.lane_id == "decimal"
        )
        with self.assertRaisesRegex(AlgebraAgentIntegrityError, "spliced"):
            ParallelLaneResult(
                lane_id="decimal",
                program=decimal_lane.program,
                initial_state=decimal_lane.initial_state,
                final_state=decimal.final_state,
                program_receipt=stack.program_receipt,
                execution_receipt=stack.execution_receipt,
            )

        expected_lane = next(lane for lane in ensemble.lanes if lane.lane_id == "stack")
        stack_machine = build_stack_machine(capacity=4, symbols=(1, 2, 3))
        substituted_program = AffineProgram(
            stack_machine.schema,
            (stack_machine.push(3),),
            "SubstitutedSameSchemaStack",
        )
        substituted = AffineMonoidRuntime.execute(substituted_program)
        substituted_record = ParallelLaneResult(
            lane_id="stack",
            program=substituted_program,
            initial_state=stack_machine.schema.state(),
            final_state=substituted.state,
            program_receipt=substituted.program_receipt,
            execution_receipt=substituted.execution_receipt,
        ).to_record()
        with self.assertRaisesRegex(AlgebraAgentIntegrityError, "same-schema"):
            ParallelLaneResult.from_record(
                substituted_record,
                schema_program=expected_lane.program,
            )

    def test_serialized_missing_lane_and_rehashed_output_tamper_fail_closed(
        self,
    ) -> None:
        ensemble, artifact = self._ensemble()
        result = ensemble.execute({"fingerprint": (artifact,)})
        document = json.loads(result.to_bytes())
        document["body"]["lane_results"].pop()
        document["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(document["body"])
        ).hexdigest()
        with self.assertRaisesRegex(AlgebraAgentIntegrityError, "count"):
            HeterogeneousEnsembleResult.from_bytes(canonical_json_bytes(document))

        document = json.loads(result.to_bytes())
        document["body"]["lane_results"][0]["final_state_sha256"] = _hash(
            "forged-output"
        )
        document["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(document["body"])
        ).hexdigest()
        with self.assertRaisesRegex(AlgebraAgentIntegrityError, "derived"):
            HeterogeneousEnsembleResult.from_bytes(canonical_json_bytes(document))


if __name__ == "__main__":
    unittest.main()
