from __future__ import annotations

from dataclasses import fields
import hashlib
import inspect
import json
import unittest

import numpy as np

from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.markov_language import (
    ActionBinding,
    ActionFrontier,
    ConsequenceFeedback,
    ConsequenceMarkovLanguage,
    FactorizedConsequenceGrammar,
    HolisticConsequenceTable,
    LanguageSnapshot,
    MarkovLanguageConflictError,
    MarkovLanguageIntegrityError,
    ReceiverDecision,
    SenderEmission,
    assert_receiver_information_boundary,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _frontier(count: int = 4, *, suffix: str = "base") -> ActionFrontier:
    return ActionFrontier.create(
        [
            ActionBinding(
                action_id=f"action-{index}",
                artifact_kind="crystal",
                artifact_sha256=_sha(f"{suffix}:crystal:{index}"),
            )
            for index in range(count)
        ],
        action_schema_sha256=_sha(f"{suffix}:action-schema"),
        context_schema_sha256=_sha(f"{suffix}:context-schema"),
        authority_hashes={"verifier": _sha(f"{suffix}:verifier")},
    )


def _vocabulary(count: int = 8) -> tuple[str, ...]:
    return tuple(f"tau-{index}" for index in range(count))


def _train_language(
    *,
    seed: int,
    episodes: int = 6_000,
) -> ConsequenceMarkovLanguage:
    frontier = _frontier()
    language = ConsequenceMarkovLanguage(frontier, _vocabulary(), seed=seed)
    rng = np.random.default_rng(seed + 100)
    for step in range(episodes):
        intent = frontier.action_ids[int(rng.integers(len(frontier.actions)))]
        fraction = step / max(1, episodes - 1)
        epsilon = 0.40 * (1.0 - fraction) + 0.01 * fraction
        language.run_episode(
            intent,
            "context-main",
            lambda chosen, intent=intent, step=step: (
                1.0 if chosen == intent else -0.25,
                chosen == intent,
                _sha(f"train:{seed}:{step}:{chosen}"),
            ),
            epsilon=epsilon,
        )
    return language


class ActionFrontierTests(unittest.TestCase):
    def test_frontier_is_content_addressed_without_a_universal_model_pin(self) -> None:
        frontier = ActionFrontier.create(
            [
                ActionBinding("route-a", "materialized-route", _sha("route-a")),
                ActionBinding("organ-b", "organ", _sha("organ-b")),
                ActionBinding("qwen-c", "qwen-path", _sha("qwen-c")),
            ],
            action_schema_sha256=_sha("actions"),
            context_schema_sha256=_sha("contexts"),
            authority_hashes={"graph": _sha("graph"), "verifier": _sha("v")},
        )
        restored = ActionFrontier.from_record(frontier.to_record())
        self.assertEqual(restored, frontier)
        self.assertEqual(restored.sha256, frontier.sha256)
        self.assertNotIn("model_pin_sha256", frontier.to_record())

    def test_duplicate_semantic_artifacts_are_rejected_by_default(self) -> None:
        digest = _sha("same-artifact")
        with self.assertRaisesRegex(ValueError, "duplicate executable"):
            ActionFrontier.create(
                [
                    ActionBinding("left", "program", digest),
                    ActionBinding("right", "program", digest),
                ],
                action_schema_sha256=_sha("actions"),
                context_schema_sha256=_sha("contexts"),
            )
        allowed = ActionFrontier.create(
            [
                ActionBinding("left", "program", digest),
                ActionBinding("right", "program", digest),
            ],
            action_schema_sha256=_sha("actions"),
            context_schema_sha256=_sha("contexts"),
            allow_duplicate_artifacts=True,
        )
        self.assertTrue(allowed.allow_duplicate_artifacts)


class ConsequenceLanguageTests(unittest.TestCase):
    def test_receiver_api_and_feedback_contain_no_latent_semantics(self) -> None:
        assert_receiver_information_boundary()
        parameters = set(
            inspect.signature(ConsequenceMarkovLanguage.receiver_decide).parameters
        )
        self.assertEqual(parameters, {"self", "word_id", "context_id", "epsilon"})
        feedback_fields = {field.name for field in fields(ConsequenceFeedback)}
        self.assertFalse(
            feedback_fields
            & {"intent", "intent_action_id", "target", "target_action", "label"}
        )

    def test_feedback_is_one_shot_and_bound_to_the_exact_pending_decision(self) -> None:
        language = ConsequenceMarkovLanguage(_frontier(2), _vocabulary(4), seed=5)
        emission = language.emit("action-0", epsilon=1.0)
        receiver = language.receiver_decide(
            emission.word_id, "context-main", epsilon=1.0
        )
        sender = language.bind_sender(emission, receiver)
        feedback = ConsequenceFeedback.for_decision(
            receiver,
            reward=1.0,
            accepted=True,
            outcome_receipt_sha256=_sha("outcome"),
        )
        language.observe_receiver(receiver, feedback)
        with self.assertRaisesRegex(MarkovLanguageConflictError, "replayed"):
            language.observe_receiver(receiver, feedback)
        language.observe_sender(sender, feedback)
        with self.assertRaisesRegex(MarkovLanguageConflictError, "replayed"):
            language.observe_sender(sender, feedback)

        unrelated = ReceiverDecision(
            frontier_sha256=language.frontier.sha256,
            word_id=emission.word_id,
            context_id="context-main",
            action_id="action-0",
            sequence=999,
        )
        wrong = ConsequenceFeedback.for_decision(
            unrelated,
            reward=1.0,
            accepted=True,
            outcome_receipt_sha256=_sha("wrong"),
        )
        with self.assertRaises(MarkovLanguageConflictError):
            language.observe_receiver(unrelated, wrong)

    def test_forged_sender_emission_cannot_poison_another_intent(self) -> None:
        language = ConsequenceMarkovLanguage(_frontier(2), _vocabulary(4), seed=6)
        emission = language.emit("action-0", epsilon=0.0)
        receiver = language.receiver_decide(
            emission.word_id, "context-main", epsilon=1.0
        )
        forged = SenderEmission(
            frontier_sha256=emission.frontier_sha256,
            intent_action_id="action-1",
            word_id=emission.word_id,
            sequence=emission.sequence,
        )
        with self.assertRaisesRegex(MarkovLanguageConflictError, "share an episode"):
            language.bind_sender(forged, receiver)
        sender = language.bind_sender(emission, receiver)
        feedback = ConsequenceFeedback.for_decision(
            receiver,
            reward=1.0,
            accepted=True,
            outcome_receipt_sha256=_sha("bound-outcome"),
        )
        language.observe_receiver(receiver, feedback)
        language.observe_sender(sender, feedback)

    def test_same_word_cannot_bind_two_concurrent_sender_intents(self) -> None:
        language = ConsequenceMarkovLanguage(_frontier(2), _vocabulary(4), seed=7)
        language.sender_q.fill(-1.0)
        language.sender_q[:, 0] = 1.0
        first = language.emit("action-0", epsilon=0.0)
        with self.assertRaisesRegex(
            MarkovLanguageConflictError,
            "already has an uncommitted",
        ):
            language.emit("action-1", epsilon=0.0)
        language.abort_pending_episode(first)
        second = language.emit("action-1", epsilon=0.0)
        self.assertEqual(second.word_id, first.word_id)
        language.abort_pending_episode(second)

    def test_partial_episode_cannot_be_serialized_and_can_abort_cleanly(self) -> None:
        language = ConsequenceMarkovLanguage(_frontier(2), _vocabulary(4), seed=8)
        emission = language.emit("action-0", epsilon=0.0)
        with self.assertRaisesRegex(MarkovLanguageConflictError, "complete episode"):
            language.to_bytes()
        language.abort_pending_episode(emission)
        language.to_bytes()

        emission = language.emit("action-0", epsilon=0.0)
        receiver = language.receiver_decide(
            emission.word_id, "context-main", epsilon=0.0
        )
        sender = language.bind_sender(emission, receiver)
        feedback = ConsequenceFeedback.for_decision(
            receiver,
            reward=1.0,
            accepted=True,
            outcome_receipt_sha256=_sha("partial-outcome"),
        )
        language.observe_receiver(receiver, feedback)
        with self.assertRaisesRegex(MarkovLanguageConflictError, "complete episode"):
            language.to_bytes()
        language.observe_sender(sender, feedback)
        language.to_bytes()

    def test_reward_only_language_composes_unseen_programs(self) -> None:
        language = _train_language(seed=2)
        self.assertEqual(language.accuracy("context-main"), 1.0)
        rng = np.random.default_rng(44)
        exact = shuffled = fixed_no_message = no_action_abstain = 0
        action_ids = language.action_ids
        deranged = {
            action: action_ids[(index + 1) % len(action_ids)]
            for index, action in enumerate(action_ids)
        }
        matrices = {
            "action-0": (np.eye(2), np.array([1.0, 0.0])),
            "action-1": (np.eye(2), np.array([0.0, 1.0])),
            "action-2": (np.array([[1.0, 1.0], [0.0, 1.0]]), np.zeros(2)),
            "action-3": (np.array([[0.0, 1.0], [1.0, 0.0]]), np.zeros(2)),
        }

        def execute(state: np.ndarray, program: tuple[str, ...]) -> np.ndarray:
            result = state.copy()
            for action in program:
                matrix, bias = matrices[action]
                result = matrix @ result + bias
            return result

        for _ in range(300):
            program = tuple(
                action_ids[int(index)]
                for index in rng.integers(0, len(action_ids), size=6)
            )
            words = tuple(
                language.encode_action(action, "context-main") for action in program
            )
            self.assertNotIn(None, words)
            decoded = tuple(
                cast
                for word in words
                if (cast := language.decode_word(str(word), "context-main")) is not None
            )
            state = rng.normal(size=2)
            target = execute(state, program)
            exact += int(
                decoded == program and np.array_equal(execute(state, decoded), target)
            )
            shuffled += int(
                np.array_equal(
                    execute(state, tuple(deranged[action] for action in decoded)),
                    target,
                )
            )
            fixed_no_message += int(
                np.array_equal(execute(state, ("action-0",) * len(program)), target)
            )
            no_action_abstain += int(np.array_equal(state, target))
        self.assertEqual(exact, 300)
        self.assertLess(shuffled, 30)
        self.assertLess(fixed_no_message, 30)
        self.assertLess(no_action_abstain, 30)

    def test_context_residual_learns_opposite_meanings_for_the_same_word(self) -> None:
        language = ConsequenceMarkovLanguage(
            _frontier(2),
            ("tau-shared", "tau-unused"),
            seed=7,
            context_full_weight_visits=8,
        )
        for step in range(4_000):
            context = "context-left" if step % 2 == 0 else "context-right"
            target = "action-0" if context == "context-left" else "action-1"
            fraction = step / 3_999
            epsilon = 0.35 * (1.0 - fraction) + 0.01 * fraction
            decision = language.receiver_decide("tau-shared", context, epsilon=epsilon)
            feedback = ConsequenceFeedback.for_decision(
                decision,
                reward=1.0 if decision.action_id == target else -0.25,
                accepted=decision.action_id == target,
                outcome_receipt_sha256=_sha(f"context:{step}"),
            )
            language.observe_receiver(decision, feedback)
        self.assertEqual(language.decode_word("tau-shared", "context-left"), "action-0")
        self.assertEqual(
            language.decode_word("tau-shared", "context-right"), "action-1"
        )

    def test_in_vocabulary_unvisited_and_ambiguous_words_abstain(self) -> None:
        language = ConsequenceMarkovLanguage(_frontier(2), _vocabulary(4), seed=9)
        self.assertIsNone(language.decode_word("tau-0", "context-main"))
        language.receiver_q[0] = np.array([0.8, 0.8], dtype=np.float64)
        language.receiver_visits[0] = np.array([20, 20], dtype=np.int64)
        self.assertIsNone(language.decode_word("tau-0", "context-unseen"))
        self.assertIsNone(language.decode_word("not-in-vocabulary", "context-main"))

    def test_one_local_visit_cannot_hijack_a_globally_mature_word(self) -> None:
        language = ConsequenceMarkovLanguage(
            _frontier(2),
            _vocabulary(4),
            seed=10,
            context_full_weight_visits=1,
        )
        language.receiver_q[0] = np.array([0.9, 0.1], dtype=np.float64)
        language.receiver_visits[0] = np.array([100, 10], dtype=np.int64)
        local_q, local_visits = language._ensure_context("context-hijack")
        local_q[0] = np.array([0.0, 1.0], dtype=np.float64)
        local_visits[0] = np.array([0, 1], dtype=np.int64)
        self.assertIsNone(language.decode_word("tau-0", "context-hijack"))

    def test_state_snapshot_resume_tamper_and_stale_frontier(self) -> None:
        language = _train_language(seed=12, episodes=3_000)
        payload = language.to_bytes()
        restored = ConsequenceMarkovLanguage.from_bytes(
            payload, expected_frontier=language.frontier
        )
        self.assertEqual(restored.to_bytes(), payload)
        restored_emission = restored.emit("action-0", epsilon=0.0)
        original_emission = language.emit("action-0", epsilon=0.0)
        self.assertEqual(restored_emission, original_emission)
        restored.abort_pending_episode(restored_emission)
        language.abort_pending_episode(original_emission)
        with self.assertRaisesRegex(MarkovLanguageConflictError, "stale"):
            ConsequenceMarkovLanguage.from_bytes(
                payload, expected_frontier=_frontier(suffix="new-revision")
            )

        document = json.loads(payload)
        document["body"]["receiver_q"][0][0] = float(999.0).hex()
        tampered = canonical_json_bytes(document)
        with self.assertRaisesRegex(MarkovLanguageIntegrityError, "hash"):
            ConsequenceMarkovLanguage.from_bytes(tampered)

        snapshot = LanguageSnapshot.freeze(language, contexts=("context-main",))
        self.assertEqual(LanguageSnapshot.from_bytes(snapshot.to_bytes()), snapshot)
        artifacts = snapshot.primitive_artifacts(language.frontier)
        self.assertEqual(len(set(artifacts.values())), len(language.action_ids))
        with self.assertRaisesRegex(MarkovLanguageConflictError, "stale"):
            snapshot.primitive_artifacts(_frontier(suffix="stale"))

    def test_snapshot_exports_context_specific_words_without_global_collapse(
        self,
    ) -> None:
        language = ConsequenceMarkovLanguage(
            _frontier(2),
            ("tau-shared", "tau-other"),
            seed=14,
            context_full_weight_visits=1,
        )
        language.receiver_q[0] = np.array([0.5, 0.5], dtype=np.float64)
        language.receiver_visits[0] = np.array([20, 20], dtype=np.int64)
        left_q, left_visits = language._ensure_context("context-left")
        right_q, right_visits = language._ensure_context("context-right")
        left_q[0] = np.array([1.0, 0.0])
        right_q[0] = np.array([0.0, 1.0])
        left_visits[0] = np.array([10, 0])
        right_visits[0] = np.array([0, 10])
        snapshot = LanguageSnapshot.freeze(
            language,
            contexts=("context-left", "context-right"),
            require_complete=False,
        )
        self.assertNotIn("tau-shared", dict(snapshot.global_word_actions))
        self.assertEqual(snapshot.decode_word("tau-shared", "context-left"), "action-0")
        self.assertEqual(
            snapshot.decode_word("tau-shared", "context-right"), "action-1"
        )
        self.assertEqual(
            snapshot.primitive_artifacts(language.frontier, context_id="context-right")[
                "tau-shared"
            ],
            ("crystal", language.frontier.binding("action-1").artifact_sha256),
        )

    def test_blank_child_recovers_the_exact_teacher_dialect_by_consequence(
        self,
    ) -> None:
        teacher = _train_language(seed=32)
        teacher_snapshot = LanguageSnapshot.freeze(teacher, contexts=("context-main",))
        child = ConsequenceMarkovLanguage(teacher.frontier, teacher.vocabulary, seed=33)
        rng = np.random.default_rng(34)
        episodes = 5_000
        for step in range(episodes):
            intent = teacher.action_ids[int(rng.integers(len(teacher.action_ids)))]
            fraction = step / (episodes - 1)
            epsilon = 0.35 * (1.0 - fraction) + 0.01 * fraction
            child.learn_from_teacher_episode(
                teacher_snapshot,
                intent,
                "context-main",
                lambda chosen, intent=intent, step=step: (
                    1.0 if chosen == intent else -0.25,
                    chosen == intent,
                    _sha(f"culture:{step}:{chosen}"),
                ),
                epsilon=epsilon,
            )
        child.induce_sender_from_receiver("context-main")
        child_snapshot = LanguageSnapshot.freeze(child, contexts=("context-main",))
        self.assertEqual(
            child_snapshot.sender_action_words,
            teacher_snapshot.sender_action_words,
        )


class FactorizedGrammarTests(unittest.TestCase):
    def test_heldout_cartesian_combinations_recombine_from_whole_reward(self) -> None:
        factors: dict[str, tuple[str, str]] = {}
        bindings = []
        for verb in range(4):
            for argument in range(4):
                action = f"verb-{verb}:arg-{argument}"
                factors[action] = (f"verb-{verb}", f"arg-{argument}")
                bindings.append(ActionBinding(action, "program", _sha(action)))
        frontier = ActionFrontier.create(
            bindings,
            action_schema_sha256=_sha("grammar-actions"),
            context_schema_sha256=_sha("grammar-context"),
            authority_hashes={"verifier": _sha("grammar-verifier")},
        )
        grammar = FactorizedConsequenceGrammar(
            frontier,
            slot_names=("verb", "argument"),
            action_factors=factors,
            slot_vocabularies=(
                tuple(f"opaque-verb-{index}" for index in range(8)),
                tuple(f"opaque-arg-{index}" for index in range(8)),
            ),
            seed=10,
        )
        holdout = {f"verb-{index}:arg-{index}" for index in range(4)}
        train = tuple(action for action in frontier.action_ids if action not in holdout)
        holistic = HolisticConsequenceTable(set(train))
        rng = np.random.default_rng(11)
        episodes = 8_000
        for step in range(episodes):
            intent = train[int(rng.integers(len(train)))]
            fraction = step / (episodes - 1)
            epsilon = 0.40 * (1.0 - fraction) + 0.01 * fraction
            grammar.run_episode(
                intent,
                "grammar-context",
                lambda chosen, intent=intent, step=step: (
                    1.0 if chosen == intent else -0.25,
                    chosen == intent,
                    _sha(f"grammar:{step}:{chosen}"),
                ),
                epsilon=epsilon,
            )
        for action in sorted(holdout):
            words = grammar.encode_action(action)
            self.assertEqual(grammar.decode_words(words, "grammar-context"), action)
            self.assertIsNone(holistic.encode(action))

        shuffled_correct = 0
        for action in sorted(holdout):
            words = list(grammar.encode_action(action))
            words[0] = grammar.slot_vocabularies[0][
                (grammar.slot_vocabularies[0].index(words[0]) + 1)
                % len(grammar.slot_vocabularies[0])
            ]
            shuffled_correct += int(
                grammar.decode_words(words, "grammar-context") == action
            )
        self.assertLess(shuffled_correct, len(holdout))

    def test_incomplete_factor_product_is_rejected(self) -> None:
        frontier = _frontier(3)
        with self.assertRaisesRegex(ValueError, "Cartesian"):
            FactorizedConsequenceGrammar(
                frontier,
                slot_names=("left", "right"),
                action_factors={
                    "action-0": ("x0", "y0"),
                    "action-1": ("x0", "y1"),
                    "action-2": ("x1", "y0"),
                },
                slot_vocabularies=(("a", "b"), ("c", "d")),
            )


if __name__ == "__main__":
    unittest.main()
