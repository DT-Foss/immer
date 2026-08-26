from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from immer.runtimes.ooe.algebra_agents import (
    AlgebraRouterState,
    OperatorAlgebraCandidate,
    VerifierBoundOutcome,
)
from immer.runtimes.ooe.compute_crystals import (
    ComputeCrystal,
    ComputeCrystalBank,
    ComputeProgram,
)
from immer.runtimes.ooe.compute_graph import (
    ComputeOperatorGraph,
    ComputeOperatorGraphState,
    OperatorEdge,
)
from immer.runtimes.ooe.crystal import CrystalStore
from immer.runtimes.ooe.demand_execution import (
    DemandExecutionVerification,
    DemandRoutedExecutor,
)
from immer.runtimes.ooe.demand_scheduler import OperatorDemandScheduler
from immer.runtimes.ooe.executable_lexicon import (
    ExecutableLexiconBank,
    ExecutableLexiconState,
    ExecutableWordCompiler,
)
from immer.runtimes.ooe.language_bridge import (
    ALGEBRA_LANGUAGE_REWARD_POLICY_SHA256,
    DEMAND_LANGUAGE_REWARD_POLICY_SHA256,
    ConsequenceLanguageBridge,
    ConsequenceLanguageStateBank,
    FrontierMigrationReceipt,
    LanguageBridgeConflictError,
    LanguageBridgeIntegrityError,
    LanguageMacroDiscoveryReceipt,
    LanguageOutcomeCommitReceipt,
    LanguageRevisionPromotionReceipt,
    LanguageRoutingDecision,
    MacroActionPromotionReceipt,
    RouteFrontierTransitionReceipt,
    RouteWordResolutionReceipt,
    SnapshotComputeResolutionReceipt,
    VerifiedWordTrajectory,
    build_algebra_action_frontier,
    build_route_action_frontier,
    candidate_action_id,
    definition_from_macro_option,
    discover_language_macros,
    migrate_language_frontier,
    promote_and_migrate_language_revision,
    promote_compiled_word_action,
    prove_route_frontier_transition,
    resolve_route_words_to_programs,
    resolve_snapshot_compute_bindings,
    route_action_id,
)
from immer.runtimes.ooe.markov_language import (
    ActionBinding,
    ActionFrontier,
    ConsequenceMarkovLanguage,
    LanguageSnapshot,
    MarkovLanguageConflictError,
)
from immer.runtimes.ooe.options import MacroOption, OptionIdentity


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _RouteFixture:
    def __init__(self, root: Path, count: int = 2) -> None:
        self.bank = ComputeCrystalBank(root / "compute")
        self.graph = ComputeOperatorGraph(self.bank)
        self.crystals = []
        states = tuple(f"state-{index}" for index in range(count + 3))
        for index in range(count + 2):
            crystal = ComputeCrystal.affine(
                [[float(index + 1)]],
                [float(index + 1)],
            )
            self.crystals.append(crystal)
            self.bank.publish_crystal(crystal)
            self.graph.append_edge(
                OperatorEdge(
                    source_state=states[index],
                    target_state=states[index + 1],
                    crystal_sha256=crystal.sha256,
                    verifier_sha256=_sha(f"route-verifier-{index}"),
                    evidence_sha256=_sha(f"route-evidence-{index}"),
                )
            )
        for prefix_length in range(2, count + 2):
            plan = self.graph.plan_route(states[0], states[prefix_length]).plan
            assert plan is not None
            self.graph.charge_route(plan)
        self.state = self.graph.state()
        self.routes = tuple(self.state.materialized_routes)
        self.full_plan = self.graph.plan_route(states[0], states[-1]).plan
        assert self.full_plan is not None
        self.scheduler = OperatorDemandScheduler(self.bank.store)
        self.value = np.array([[-2.0], [0.5], [7.0]], dtype=np.float64)
        expected = self.value
        for crystal in self.crystals:
            expected = crystal.apply(expected)
        self.expected = expected
        self.frontier = build_route_action_frontier(
            self.state,
            context_schema_sha256=_sha("route-context-schema"),
            extra_authorities={"episode-terminal-verifier": _sha("terminal-verifier")},
        )

    def language(self, *, seed: int = 0) -> ConsequenceMarkovLanguage:
        language = ConsequenceMarkovLanguage(
            self.frontier,
            tuple(f"route-word-{index}" for index in range(len(self.routes) + 2)),
            seed=seed,
        )
        language.sender_q.fill(-1.0)
        language.receiver_q.fill(-1.0)
        language.receiver_visits.fill(0)
        for index, action_id in enumerate(language.action_ids):
            language.sender_q[index, index] = 1.0
            language.receiver_q[index, index] = 1.0
            language.receiver_visits[index, index] = 20
        return language

    def execute(
        self,
        route_sha256: str,
        *,
        ordinal: int,
        success: bool = True,
        episode_sha256: str | None = None,
    ):
        expected = self.expected

        def verifier(execution):
            accepted = success and np.array_equal(execution.output, expected)
            return DemandExecutionVerification.create(
                execution,
                accepted=accepted,
                verifier_sha256=_sha("outcome-verifier"),
                evidence_sha256=_sha(f"outcome-evidence-{ordinal}:{accepted}"),
                reason="accepted" if accepted else "rejected",
            )

        executor = DemandRoutedExecutor(
            graph=self.graph,
            scheduler=self.scheduler,
            verifier=verifier,
        )
        return executor.execute(
            self.full_plan,
            self.value,
            candidate_route_sha256s=(route_sha256,),
            episode_sha256=(
                _sha(f"episode-{ordinal}") if episode_sha256 is None else episode_sha256
            ),
        )


class DemandLanguageBridgeTests(unittest.TestCase):
    def test_route_frontier_and_verified_outcome_commit_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _RouteFixture(Path(temporary))
            language = fixture.language(seed=1)
            state_bank = ConsequenceLanguageStateBank(
                CrystalStore(Path(temporary) / "language")
            )
            bridge = ConsequenceLanguageBridge(language, state_bank=state_bank)
            intended = language.action_ids[0]
            decision = bridge.begin(intended, "route-context", epsilon=0.0)
            self.assertEqual(decision.receiver.action_id, intended)
            self.assertEqual(
                LanguageRoutingDecision.from_bytes(decision.to_bytes()), decision
            )
            route_sha = decision.selected_artifact_sha256
            routed = fixture.execute(route_sha, ordinal=1)
            commit = bridge.settle_demand(decision, routed.receipt, fixture.state)
            self.assertTrue(commit.accepted)
            self.assertGreater(commit.reward, 1.0)
            self.assertEqual(commit.external_outcome_sha256, routed.receipt.sha256)
            self.assertEqual(
                LanguageOutcomeCommitReceipt.from_bytes(commit.to_bytes()), commit
            )
            self.assertEqual(state_bank.current_state_sha256(), bridge.language.sha256)
            restored = state_bank.restore(
                expected_frontier=fixture.frontier,
                trusted_state_sha256=commit.language_state_after_sha256,
            )
            self.assertEqual(restored.to_bytes(), bridge.language.to_bytes())
            self.assertEqual(
                dict(fixture.frontier.authority_hashes)["language-reward-policy"],
                DEMAND_LANGUAGE_REWARD_POLICY_SHA256,
            )

    def test_wrong_route_revision_is_rejected_then_correct_outcome_can_settle(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _RouteFixture(Path(temporary))
            bridge = ConsequenceLanguageBridge(fixture.language(seed=2))
            decision = bridge.begin(
                fixture.frontier.action_ids[0], "route-context", epsilon=0.0
            )
            other = next(
                route
                for route in fixture.routes
                if route.sha256 != decision.selected_artifact_sha256
            )
            other_routed = fixture.execute(other.sha256, ordinal=2)
            with self.assertRaisesRegex(
                LanguageBridgeIntegrityError, "selected exact route"
            ):
                bridge.settle_demand(decision, other_routed.receipt, fixture.state)
            correct = fixture.execute(decision.selected_artifact_sha256, ordinal=3)
            commit = bridge.settle_demand(decision, correct.receipt, fixture.state)
            self.assertTrue(commit.accepted)

            next_decision = bridge.begin(
                fixture.frontier.action_ids[1], "route-context", epsilon=0.0
            )
            stale_routed = fixture.execute(
                next_decision.selected_artifact_sha256, ordinal=4
            )

            extra = ComputeCrystal.affine([[9.0]], [3.0])
            fixture.bank.publish_crystal(extra)
            fixture.graph.append_edge(
                OperatorEdge(
                    "stale-source",
                    "stale-target",
                    extra.sha256,
                    _sha("stale-verifier"),
                    _sha("stale-evidence"),
                )
            )
            stale_state = fixture.graph.state()
            with self.assertRaisesRegex(LanguageBridgeIntegrityError, "another graph"):
                bridge.settle_demand(
                    next_decision,
                    stale_routed.receipt,
                    stale_state,
                )
            bridge.abort(next_decision)

    def test_state_bank_cas_failure_rolls_back_the_in_memory_learner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _RouteFixture(root)
            initial = fixture.language(seed=3)
            initial_payload = initial.to_bytes()
            bank = ConsequenceLanguageStateBank(CrystalStore(root / "language"))
            first = ConsequenceLanguageBridge(initial, state_bank=bank)
            second = ConsequenceLanguageBridge(
                ConsequenceMarkovLanguage.from_bytes(
                    initial_payload,
                    expected_frontier=fixture.frontier,
                ),
                state_bank=bank,
            )
            first_decision = first.begin(
                fixture.frontier.action_ids[0], "route-context", epsilon=0.0
            )
            first.settle_demand(
                first_decision,
                fixture.execute(
                    first_decision.selected_artifact_sha256, ordinal=5
                ).receipt,
                fixture.state,
            )
            second_decision = second.begin(
                fixture.frontier.action_ids[1], "route-context", epsilon=0.0
            )
            with self.assertRaisesRegex(LanguageBridgeConflictError, "CAS"):
                second.settle_demand(
                    second_decision,
                    fixture.execute(
                        second_decision.selected_artifact_sha256, ordinal=6
                    ).receipt,
                    fixture.state,
                )
            self.assertEqual(second.language.to_bytes(), initial_payload)
            second.language.to_bytes()
            with self.assertRaisesRegex(
                LanguageBridgeIntegrityError, "rollback anchor"
            ):
                bank.restore(
                    expected_frontier=fixture.frontier,
                    trusted_state_sha256=_sha("stale-external-anchor"),
                )

    def test_bound_abort_is_persisted_without_quality_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _RouteFixture(root)
            bank = ConsequenceLanguageStateBank(CrystalStore(root / "language"))
            bridge = ConsequenceLanguageBridge(
                fixture.language(seed=4), state_bank=bank
            )
            before_episodes = bridge.language.training_episodes
            decision = bridge.begin(
                fixture.frontier.action_ids[0], "route-context", epsilon=0.0
            )
            after = bridge.abort(decision)
            self.assertEqual(after, bank.current_state_sha256())
            self.assertEqual(bridge.language.training_episodes, before_episodes)
            bridge.language.to_bytes()
            with self.assertRaises(LanguageBridgeConflictError):
                bridge.abort(decision)

    def test_verified_negative_is_real_feedback_and_failed_begin_is_atomic(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _RouteFixture(Path(temporary))
            bridge = ConsequenceLanguageBridge(fixture.language(seed=11))
            before = bridge.language.to_bytes()
            with self.assertRaises(ValueError):
                bridge.begin(
                    fixture.frontier.action_ids[0],
                    "invalid context with spaces",
                    epsilon=0.0,
                )
            self.assertEqual(bridge.language.to_bytes(), before)

            decision = bridge.begin(
                fixture.frontier.action_ids[0], "route-context", epsilon=0.0
            )
            word_index = bridge.language.vocabulary.index(decision.receiver.word_id)
            action_index = bridge.language.action_ids.index(decision.receiver.action_id)
            prior_q = float(bridge.language.receiver_q[word_index, action_index])
            rejected = fixture.execute(
                decision.selected_artifact_sha256,
                ordinal=20,
                success=False,
            )
            commit = bridge.settle_demand(decision, rejected.receipt, fixture.state)
            self.assertFalse(commit.accepted)
            self.assertEqual(commit.reward, -1.0)
            self.assertLess(
                float(bridge.language.receiver_q[word_index, action_index]),
                prior_q,
            )


class AlgebraLanguageBridgeTests(unittest.TestCase):
    def test_single_candidate_algebra_outcome_uses_pinned_binary_reward(self) -> None:
        crystal = ComputeCrystal.affine([[2.0]], [1.0])
        candidate = OperatorAlgebraCandidate(
            family="affine-test",
            program=ComputeProgram.compose((crystal,)),
            verifier_sha256=_sha("algebra-execution-verifier"),
            discovery_verifier_sha256=_sha("algebra-discovery-verifier"),
            evidence_sha256s=(_sha("algebra-evidence"),),
            behavior_descriptor=(0,),
            objectives=(1.0,),
        )
        router = AlgebraRouterState.bootstrap(
            (candidate,),
            seed_sha256=_sha("algebra-router-seed"),
            bin_counts=(1,),
            objective_count=1,
        )
        frontier = build_algebra_action_frontier(
            router,
            context_schema_sha256=_sha("algebra-context-schema"),
        )
        language = ConsequenceMarkovLanguage(frontier, ("algebra-word",), seed=5)
        bridge = ConsequenceLanguageBridge(language)
        decision = bridge.begin(
            candidate_action_id(candidate.sha256),
            "algebra-context",
            epsilon=0.0,
        )
        chosen, selection, _advanced = router.choose("algebra-context")
        self.assertEqual(chosen, candidate)
        outcome = VerifierBoundOutcome.issue(
            selection,
            candidate,
            receipt_payload=b"verified-algebra-result",
            success=True,
        )
        commit = bridge.settle_algebra(decision, selection, candidate, outcome)
        self.assertEqual(commit.reward, 1.0)
        self.assertTrue(commit.accepted)
        self.assertEqual(
            dict(frontier.authority_hashes)["language-reward-policy"],
            ALGEBRA_LANGUAGE_REWARD_POLICY_SHA256,
        )


class FrontierMigrationTests(unittest.TestCase):
    def test_revision_growth_retains_exact_actions_and_resets_new_or_changed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _RouteFixture(root, count=2)
            language = fixture.language(seed=6)
            local_q, local_visits = language._ensure_context("route-context")
            local_q[:] = language.receiver_q
            local_visits[:] = language.receiver_visits

            crystal = ComputeCrystal.affine([[5.0]], [7.0])
            fixture.bank.publish_crystal(crystal)
            fixture.graph.append_edge(
                OperatorEdge(
                    "source-new",
                    "target-new",
                    crystal.sha256,
                    _sha("new-route-verifier"),
                    _sha("new-route-evidence"),
                )
            )
            edge_state = fixture.graph.state()
            plan = fixture.graph.plan_route("source-new", "target-new").plan
            assert plan is not None
            fixture.graph.charge_route(plan)
            next_state = fixture.graph.state()
            next_frontier = build_route_action_frontier(
                next_state,
                context_schema_sha256=fixture.frontier.context_schema_sha256,
                extra_authorities={
                    "episode-terminal-verifier": _sha("terminal-verifier")
                },
            )
            unproved, unproved_receipt = migrate_language_frontier(
                language, next_frontier, seed=7
            )
            self.assertFalse(unproved_receipt.authority_evidence_retained)
            self.assertEqual(
                set(unproved_receipt.reset_action_ids), set(language.action_ids)
            )
            self.assertEqual(int(unproved.receiver_visits.sum()), 0)
            transition, proofs = prove_route_frontier_transition(
                fixture.frontier,
                next_frontier,
                (fixture.state, edge_state, next_state),
            )
            self.assertEqual(
                RouteFrontierTransitionReceipt.from_bytes(transition.to_bytes()),
                transition,
            )
            self.assertEqual(set(proofs), {"graph-state", "route-contract-set"})
            forged_transition = replace(
                transition,
                changed_authority_names=("graph-state",),
            )
            with self.assertRaisesRegex(
                LanguageBridgeIntegrityError,
                "does not justify authority retention",
            ):
                migrate_language_frontier(
                    language,
                    next_frontier,
                    seed=7,
                    authority_transition_receipts=(forged_transition,),
                )
            migrated, receipt = migrate_language_frontier(
                language,
                next_frontier,
                seed=7,
                authority_transition_receipts=(transition,),
            )
            self.assertEqual(
                FrontierMigrationReceipt.from_bytes(receipt.to_bytes()), receipt
            )
            self.assertEqual(len(receipt.retained_action_ids), 2)
            self.assertEqual(len(receipt.added_action_ids), 1)
            self.assertEqual(receipt.reset_action_ids, ())
            self.assertTrue(receipt.context_evidence_retained)
            self.assertTrue(receipt.reward_policy_retained)
            self.assertTrue(receipt.authority_evidence_retained)
            for action in receipt.retained_action_ids:
                old_index = language.action_ids.index(action)
                new_index = migrated.action_ids.index(action)
                np.testing.assert_array_equal(
                    migrated.receiver_q[:, new_index],
                    language.receiver_q[:, old_index],
                )
                np.testing.assert_array_equal(
                    migrated.context_q["route-context"][:, new_index],
                    language.context_q["route-context"][:, old_index],
                )
            new_action = receipt.added_action_ids[0]
            new_index = migrated.action_ids.index(new_action)
            self.assertEqual(int(migrated.receiver_visits[:, new_index].sum()), 0)

            tight = ConsequenceMarkovLanguage(
                fixture.frontier,
                ("tight-word-0", "tight-word-1"),
                seed=70,
            )
            tight.sender_q.fill(-1.0)
            tight.receiver_q.fill(-1.0)
            tight.receiver_visits.fill(0)
            for index in range(2):
                tight.sender_q[index, index] = 1.0
                tight.receiver_q[index, index] = 1.0
                tight.receiver_visits[index, index] = 20
            tight_migrated, tight_receipt = migrate_language_frontier(
                tight,
                next_frontier,
                seed=71,
                authority_transition_receipts=(transition,),
            )
            self.assertEqual(len(tight_receipt.added_word_ids), 1)
            self.assertEqual(len(tight_migrated.vocabulary), 3)
            self.assertTrue(
                tight_receipt.added_word_ids[0].startswith("frontier-word-")
            )

            old_snapshot = LanguageSnapshot.freeze(
                language, contexts=("route-context",)
            )
            with self.assertRaises(MarkovLanguageConflictError):
                old_snapshot.primitive_artifacts(next_frontier)

            shifted_context = ActionFrontier.create(
                next_frontier.actions,
                action_schema_sha256=next_frontier.action_schema_sha256,
                context_schema_sha256=_sha("shifted-context-schema"),
                authority_hashes=dict(next_frontier.authority_hashes),
            )
            shifted, shifted_receipt = migrate_language_frontier(
                migrated, shifted_context, seed=8
            )
            self.assertFalse(shifted_receipt.context_evidence_retained)
            self.assertEqual(shifted.context_q, {})

            changed_authorities = dict(next_frontier.authority_hashes)
            changed_authorities["language-reward-policy"] = _sha("new-reward-policy")
            changed_reward = ActionFrontier.create(
                next_frontier.actions,
                action_schema_sha256=next_frontier.action_schema_sha256,
                context_schema_sha256=next_frontier.context_schema_sha256,
                authority_hashes=changed_authorities,
            )
            reset_language, reset_receipt = migrate_language_frontier(
                migrated, changed_reward, seed=9
            )
            self.assertFalse(reset_receipt.reward_policy_retained)
            self.assertEqual(
                set(reset_receipt.reset_action_ids), set(migrated.action_ids)
            )
            self.assertEqual(int(reset_language.receiver_visits.sum()), 0)


class MacroPromotionTests(unittest.TestCase):
    @staticmethod
    def _commit(
        *,
        ordinal: int,
        frontier: ActionFrontier,
        action_id: str,
        word_id: str,
    ) -> LanguageOutcomeCommitReceipt:
        return LanguageOutcomeCommitReceipt(
            routing_decision_sha256=_sha(f"routing-{ordinal}"),
            frontier_sha256=frontier.sha256,
            external_outcome_kind="demand-outcome",
            external_outcome_sha256=_sha(f"external-{ordinal}"),
            feedback_sha256=_sha(f"feedback-{ordinal}"),
            language_state_before_sha256=_sha(f"before-{ordinal}"),
            language_state_after_sha256=_sha(f"after-{ordinal}"),
            action_id=action_id,
            word_id=word_id,
            context_id="route-context",
            accepted=True,
            reward=2.0,
        )

    def test_demand_episode_order_and_terminal_length_are_authenticated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _RouteFixture(Path(temporary), count=2)
            bridge = ConsequenceLanguageBridge(fixture.language(seed=14))
            snapshot = LanguageSnapshot.freeze(
                bridge.language, contexts=("route-context",)
            )
            episode_sha = _sha("shared-demand-episode")
            action_sequence = (
                fixture.frontier.action_ids[0],
                fixture.frontier.action_ids[1],
                fixture.frontier.action_ids[0],
            )
            pairs = []
            for ordinal, action in enumerate(action_sequence, start=40):
                decision = bridge.begin(action, "route-context", epsilon=0.0)
                routed = fixture.execute(
                    decision.selected_artifact_sha256,
                    ordinal=ordinal,
                    episode_sha256=episode_sha,
                )
                commit = bridge.settle_demand(decision, routed.receipt, fixture.state)
                pairs.append((commit, routed.receipt))
            trajectory = VerifiedWordTrajectory.from_demand_episode(
                tuple(reversed(pairs)),
                snapshot,
                episode_sha256=episode_sha,
                terminal_verifier_sha256=_sha("terminal-verifier"),
                terminal_verification_sha256=_sha("terminal-demand-proof"),
                expected_steps=3,
                frontier=fixture.frontier,
            )
            self.assertEqual(trajectory.action_ids, action_sequence)
            with self.assertRaisesRegex(ValueError, "expected_steps"):
                VerifiedWordTrajectory.from_demand_episode(
                    pairs,
                    snapshot,
                    episode_sha256=episode_sha,
                    terminal_verifier_sha256=_sha("terminal-verifier"),
                    terminal_verification_sha256=_sha("terminal-demand-proof"),
                    expected_steps=4,
                    frontier=fixture.frontier,
                )

    def test_verified_repetition_promotes_and_compiles_without_route_flattening(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _RouteFixture(root, count=2)
            language = fixture.language(seed=10)
            snapshot = LanguageSnapshot.freeze(language, contexts=("route-context",))
            action_words = dict(snapshot.sender_action_words)
            sequence = (
                fixture.frontier.action_ids[0],
                fixture.frontier.action_ids[1],
                fixture.frontier.action_ids[0],
            )
            words = tuple(action_words[action] for action in sequence)
            trajectories = []
            ordinal = 0
            for episode in range(3):
                commits = []
                for action, word in zip(sequence, words, strict=True):
                    commits.append(
                        self._commit(
                            ordinal=ordinal,
                            frontier=fixture.frontier,
                            action_id=action,
                            word_id=word,
                        )
                    )
                    ordinal += 1
                trajectory = VerifiedWordTrajectory.create(
                    commits,
                    snapshot,
                    episode_sha256=_sha(f"verified-episode-{episode}"),
                    terminal_verifier_sha256=_sha("terminal-verifier"),
                    terminal_verification_sha256=_sha(
                        f"terminal-verification-{episode}"
                    ),
                    expected_steps=3,
                    frontier=fixture.frontier,
                )
                self.assertEqual(
                    VerifiedWordTrajectory.from_bytes(trajectory.to_bytes()),
                    trajectory,
                )
                trajectories.append(trajectory)
            discovery = discover_language_macros(
                trajectories,
                snapshot,
                fixture.frontier,
                min_support=3,
                min_macro_length=3,
                max_macro_length=3,
            )
            self.assertEqual(
                LanguageMacroDiscoveryReceipt.from_bytes(discovery.to_bytes()),
                discovery,
            )
            self.assertEqual(len(discovery.definitions), 1)
            definition = discovery.definitions[0]
            self.assertEqual(definition.child_word_ids, words)

            primitive_bindings, resolution = resolve_route_words_to_programs(
                snapshot,
                fixture.frontier,
                fixture.state,
            )
            self.assertEqual(
                RouteWordResolutionReceipt.from_bytes(resolution.to_bytes()),
                resolution,
            )
            initial = ExecutableLexiconState.initial(
                primitive_bindings,
                language_snapshot_sha256=snapshot.sha256,
                frontier_sha256=fixture.frontier.sha256,
                authority_hashes=fixture.frontier.authority_hashes,
            )
            lexicon_bank = ExecutableLexiconBank(
                root / "lexicon", initial_state=initial
            )
            defined = lexicon_bank.append_definition(definition)
            compiler = ExecutableWordCompiler(defined, fixture.bank)
            compiled = compiler.compile(definition.new_word_id)
            self.assertTrue(compiled.constant_discharge)
            value = np.array([[2.0], [7.0]], dtype=np.float64)
            execution = compiler.execute(compiled, value)
            expected = value
            route_by_action = {
                route_action_id(route.sha256): route for route in fixture.routes
            }
            for action in sequence:
                route = route_by_action[action]
                program, crystals = fixture.bank.resolve_program(
                    route.executable_program_sha256
                )
                self.assertEqual(
                    program.crystal_sha256s, tuple(c.sha256 for c in crystals)
                )
                for crystal in crystals:
                    expected = crystal.apply(expected)
            np.testing.assert_array_equal(execution.output, expected)

            promoted_frontier, promotion = promote_compiled_word_action(
                fixture.frontier,
                discovery,
                definition,
                compiled,
            )
            self.assertEqual(
                MacroActionPromotionReceipt.from_bytes(promotion.to_bytes()),
                promotion,
            )
            self.assertIn(promotion.promoted_action_id, promoted_frontier.action_ids)
            migrated, migration = migrate_language_frontier(
                language,
                promoted_frontier,
                seed=13,
                authority_transition_receipts=(promotion,),
            )
            self.assertEqual(
                migration.added_action_ids,
                (promotion.promoted_action_id,),
            )
            self.assertEqual(
                set(migration.retained_action_ids),
                set(language.action_ids),
            )
            promoted_index = migrated.action_ids.index(promotion.promoted_action_id)
            self.assertEqual(int(migrated.receiver_visits[:, promoted_index].sum()), 0)

            language_state_bank = ConsequenceLanguageStateBank(
                CrystalStore(root / "promoted-language")
            )
            language_state_bank.initialize(language)
            (
                persisted_language,
                persisted_frontier,
                persisted_promotion,
                persisted_migration,
                revision_receipt,
            ) = promote_and_migrate_language_revision(
                language,
                language_state_bank,
                discovery,
                definition,
                compiled,
                seed=13,
            )
            self.assertEqual(persisted_frontier, promoted_frontier)
            self.assertEqual(persisted_promotion, promotion)
            self.assertEqual(persisted_migration, migration)
            self.assertEqual(
                LanguageRevisionPromotionReceipt.from_bytes(
                    revision_receipt.to_bytes()
                ),
                revision_receipt,
            )
            self.assertEqual(
                language_state_bank.current_state_sha256(),
                persisted_language.sha256,
            )
            recovered_language = language_state_bank.restore(
                trusted_state_sha256=persisted_language.sha256
            )
            self.assertEqual(recovered_language.frontier, persisted_frontier)
            self.assertEqual(
                language_state_bank.restore_revision(persisted_language.sha256),
                revision_receipt,
            )
            with self.assertRaises(LanguageBridgeConflictError):
                language_state_bank.restore(expected_frontier=fixture.frontier)

    def test_verified_macro_option_translates_through_the_snapshot(self) -> None:
        graph_sha = _sha("option-world-graph")
        kernels = {
            "action-a": np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float64),
            "action-b": np.array([[0.4, 0.6], [0.3, 0.7]], dtype=np.float64),
        }
        identity = OptionIdentity.create(
            action_sequence=("action-a", "action-b", "action-a"),
            action_kernels=kernels,
            source_world_model_sha256=_sha("option-world-model"),
            graph_revision_sha256=graph_sha,
            verifier_hashes={"option": _sha("option-verifier")},
        )
        option = MacroOption.from_identity(
            identity,
            action_kernels=kernels,
            source_trajectory_sha256s=(
                _sha("option-trajectory-1"),
                _sha("option-trajectory-2"),
            ),
        )
        frontier = ActionFrontier.create(
            (
                ActionBinding("action-a", "crystal", _sha("option-crystal-a")),
                ActionBinding("action-b", "crystal", _sha("option-crystal-b")),
            ),
            action_schema_sha256=_sha("option-action-schema"),
            context_schema_sha256=_sha("option-context-schema"),
            authority_hashes={"option-graph": graph_sha},
        )
        language = ConsequenceMarkovLanguage(frontier, ("word-a", "word-b"), seed=12)
        language.sender_q.fill(-1.0)
        language.receiver_q.fill(-1.0)
        language.receiver_visits.fill(0)
        for index in range(2):
            language.sender_q[index, index] = 1.0
            language.receiver_q[index, index] = 1.0
            language.receiver_visits[index, index] = 20
        snapshot = LanguageSnapshot.freeze(language, contexts=("option-context",))
        direct_bindings, direct_resolution = resolve_snapshot_compute_bindings(
            snapshot,
            frontier,
        )
        self.assertEqual(len(direct_bindings), 2)
        self.assertEqual(
            SnapshotComputeResolutionReceipt.from_bytes(direct_resolution.to_bytes()),
            direct_resolution,
        )
        with self.assertRaisesRegex(
            LanguageBridgeIntegrityError,
            "not authorized",
        ):
            resolve_snapshot_compute_bindings(
                snapshot,
                frontier,
                graph_state=ComputeOperatorGraphState.empty(),
            )
        definition = definition_from_macro_option(
            option,
            snapshot,
            frontier,
            context_id="option-context",
            graph_authority_name="option-graph",
        )
        self.assertEqual(
            definition.child_word_ids,
            ("word-a", "word-b", "word-a"),
        )
        wrong_frontier = ActionFrontier.create(
            frontier.actions,
            action_schema_sha256=frontier.action_schema_sha256,
            context_schema_sha256=frontier.context_schema_sha256,
            authority_hashes={"option-graph": _sha("wrong-option-graph")},
        )
        with self.assertRaisesRegex(LanguageBridgeIntegrityError, "graph authority"):
            definition_from_macro_option(
                option,
                snapshot,
                wrong_frontier,
                context_id="option-context",
                graph_authority_name="option-graph",
            )


if __name__ == "__main__":
    unittest.main()
