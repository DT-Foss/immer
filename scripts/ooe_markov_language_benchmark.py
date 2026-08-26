#!/usr/bin/env python3
"""Benchmark consequence-grounded Markov language on real ComputeCrystal APIs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import tempfile
from typing import Any, Sequence

import numpy as np

from immer.runtimes.ooe.compute_crystals import (
    ComputeCrystal,
    ComputeCrystalBank,
    ComputeCrystalVM,
    ComputeProgram,
)
from immer.runtimes.ooe.executable_lexicon import (
    ExecutableLexicon,
    ExecutableLexiconBank,
    ExecutableLexiconState,
    ExecutableWordCompiler,
    ExecutableWordDefinition,
    PrimitiveWordBinding,
)
from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.markov_language import (
    ActionBinding,
    ActionFrontier,
    ConsequenceFeedback,
    ConsequenceMarkovLanguage,
    FactorizedConsequenceGrammar,
    HolisticConsequenceTable,
    LanguageSnapshot,
)


REPORT_SCHEMA = "immer-ooe-markov-language-benchmark/v1"
SOURCE_POC_SHA256 = "7bcd5e9455d60604ddcbbdc2f813d5ae9bfaf2094061fbd788629e67037f6c7c"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _epsilon(step: int, total: int, start: float) -> float:
    fraction = step / max(1, total - 1)
    return start * (1.0 - fraction) + 0.01 * fraction


def _action_state(rng: np.random.Generator) -> np.ndarray:
    return np.array(
        [int(rng.integers(1, 11)), int(rng.integers(11, 21))],
        dtype=np.float64,
    )


def _primitive_crystals() -> tuple[ComputeCrystal, ...]:
    return (
        ComputeCrystal.affine([[1.0, 0.0], [0.0, 1.0]], [1.0, 0.0]),
        ComputeCrystal.affine([[1.0, 0.0], [0.0, 1.0]], [0.0, 1.0]),
        ComputeCrystal.affine([[0.0, 1.0], [1.0, 0.0]], [0.0, 0.0]),
        ComputeCrystal.affine([[-1.0, 0.0], [0.0, 1.0]], [0.0, 0.0]),
    )


def _execute_crystals(
    crystals: Sequence[ComputeCrystal],
    state: np.ndarray,
    program: Sequence[int],
) -> np.ndarray:
    result = state.copy()
    for action in program:
        result = crystals[int(action)].apply(result)
    return result


def _train_primitive_language(
    frontier: ActionFrontier,
    crystals: Sequence[ComputeCrystal],
    *,
    seed: int,
    episodes: int,
) -> ConsequenceMarkovLanguage:
    language = ConsequenceMarkovLanguage(
        frontier,
        tuple(f"tau-{index}" for index in range(8)),
        seed=seed,
    )
    rng = np.random.default_rng(seed + 100)
    for step in range(episodes):
        intent_index = int(rng.integers(len(frontier.actions)))
        intent = frontier.action_ids[intent_index]
        state = _action_state(rng)
        context = f"state-bucket-{int(state[0]) % 4}"
        target = crystals[intent_index].apply(state)
        language.run_episode(
            intent,
            context,
            lambda chosen, state=state, target=target, step=step: (
                1.0
                if np.array_equal(
                    crystals[frontier.action_ids.index(chosen)].apply(state), target
                )
                else -0.25,
                np.array_equal(
                    crystals[frontier.action_ids.index(chosen)].apply(state), target
                ),
                _sha(f"primitive:{seed}:{step}:{chosen}"),
            ),
            epsilon=_epsilon(step, episodes, 0.40),
        )
    return language


def _grammar_trial(*, seed: int, episodes: int) -> tuple[float, float, float]:
    factors: dict[str, tuple[str, str]] = {}
    bindings = []
    for verb in range(4):
        for argument in range(4):
            action = f"verb-{verb}:arg-{argument}"
            factors[action] = (f"verb-{verb}", f"arg-{argument}")
            bindings.append(ActionBinding(action, "program", _sha(action)))
    frontier = ActionFrontier.create(
        bindings,
        action_schema_sha256=_sha("grammar-action-schema"),
        context_schema_sha256=_sha("grammar-context-schema"),
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
        seed=seed + 200,
    )
    holdout = {f"verb-{index}:arg-{index}" for index in range(4)}
    train = tuple(action for action in frontier.action_ids if action not in holdout)
    holistic = HolisticConsequenceTable(set(train))
    rng = np.random.default_rng(seed + 300)

    def apply(action: str, state: np.ndarray) -> np.ndarray:
        verb, argument = factors[action]
        verb_index = int(verb.removeprefix("verb-"))
        value = int(argument.removeprefix("arg-")) + 2
        result = state.copy()
        if verb_index == 0:
            result[0] += value
        elif verb_index == 1:
            result[1] += value
        elif verb_index == 2:
            result[0] *= value
        else:
            result[1] *= value
        return result

    for step in range(episodes):
        intent = train[int(rng.integers(len(train)))]
        state = _action_state(rng)
        target = apply(intent, state)
        grammar.run_episode(
            intent,
            "grammar-context",
            lambda chosen, state=state, target=target, step=step: (
                1.0 if np.array_equal(apply(chosen, state), target) else -0.25,
                np.array_equal(apply(chosen, state), target),
                _sha(f"grammar:{seed}:{step}:{chosen}"),
            ),
            epsilon=_epsilon(step, episodes, 0.40),
        )
    exact = sum(
        grammar.decode_words(grammar.encode_action(action), "grammar-context") == action
        for action in holdout
    ) / len(holdout)
    holistic_exact = sum(
        holistic.encode(action) is not None for action in holdout
    ) / len(holdout)
    shuffled = 0
    for action in holdout:
        words = list(grammar.encode_action(action))
        words[0] = grammar.slot_vocabularies[0][
            (grammar.slot_vocabularies[0].index(words[0]) + 1)
            % len(grammar.slot_vocabularies[0])
        ]
        shuffled += int(grammar.decode_words(words, "grammar-context") == action)
    return exact, holistic_exact, shuffled / len(holdout)


def run_seed(
    seed: int,
    *,
    tasks: int,
    primitive_episodes: int,
    grammar_episodes: int,
    recursive_depth: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 1_000)
    with tempfile.TemporaryDirectory(prefix="immer-markov-language-") as temporary:
        compute_bank = ComputeCrystalBank(Path(temporary) / "compute")
        crystals = _primitive_crystals()
        for crystal in crystals:
            compute_bank.publish_crystal(crystal)
        frontier = ActionFrontier.create(
            [
                ActionBinding(
                    action_id=f"action-{index}",
                    artifact_kind="crystal",
                    artifact_sha256=crystal.sha256,
                )
                for index, crystal in enumerate(crystals)
            ],
            action_schema_sha256=_sha("primitive-action-schema"),
            context_schema_sha256=_sha("state-bucket-schema"),
            authority_hashes={"verifier": _sha("exact-affine-consequence")},
        )
        language = _train_primitive_language(
            frontier,
            crystals,
            seed=seed,
            episodes=primitive_episodes,
        )
        contexts = tuple(f"state-bucket-{index}" for index in range(4))
        primitive_exact = sum(language.accuracy(context) for context in contexts) / 4

        exact = shuffled = no_message_fixed = no_message_abstain = 0
        deranged = {
            index: (index + 1) % len(crystals) for index in range(len(crystals))
        }
        for _ in range(tasks):
            length = int(rng.integers(2, 9))
            program = tuple(
                int(value) for value in rng.integers(0, len(crystals), size=length)
            )
            state = _action_state(rng)
            context = f"state-bucket-{int(state[0]) % 4}"
            target = _execute_crystals(crystals, state, program)
            words = tuple(
                language.encode_action(f"action-{action}", context)
                for action in program
            )
            decoded_ids = tuple(
                language.decode_word(str(word), context) for word in words
            )
            decoded = tuple(
                int(str(action).removeprefix("action-"))
                for action in decoded_ids
                if action is not None
            )
            exact += int(
                decoded == program
                and np.array_equal(_execute_crystals(crystals, state, decoded), target)
            )
            shuffled += int(
                len(decoded) == len(program)
                and np.array_equal(
                    _execute_crystals(
                        crystals, state, tuple(deranged[action] for action in decoded)
                    ),
                    target,
                )
            )
            no_message_fixed += int(
                np.array_equal(
                    _execute_crystals(crystals, state, (0,) * len(program)),
                    target,
                )
            )
            no_message_abstain += int(np.array_equal(state, target))

        options = (
            (0, 1, 2),
            (2, 3),
            (3, 0, 3, 1),
            (2, 2, 1),
        )
        option_programs = tuple(
            ComputeProgram.compose(tuple(crystals[index] for index in option))
            for option in options
        )
        for program in option_programs:
            compute_bank.publish_program(program)
        option_frontier = ActionFrontier.create(
            [
                ActionBinding(
                    f"option-{index}",
                    "program",
                    program.sha256,
                )
                for index, program in enumerate(option_programs)
            ],
            action_schema_sha256=_sha("option-action-schema"),
            context_schema_sha256=_sha("option-context-schema"),
            authority_hashes={"verifier": _sha("option-final-consequence")},
        )
        option_language = ConsequenceMarkovLanguage(
            option_frontier,
            tuple(f"option-word-{index}" for index in range(8)),
            seed=seed + 350,
        )
        option_episodes = max(4_000, primitive_episodes)
        for step in range(option_episodes):
            intent_index = int(rng.integers(len(options)))
            intent = f"option-{intent_index}"
            state = _action_state(rng)
            target = _execute_crystals(crystals, state, options[intent_index])
            option_language.run_episode(
                intent,
                "option-context",
                lambda chosen, state=state, target=target, step=step: (
                    1.0
                    if np.array_equal(
                        _execute_crystals(
                            crystals,
                            state,
                            options[int(chosen.removeprefix("option-"))],
                        ),
                        target,
                    )
                    else -0.25,
                    np.array_equal(
                        _execute_crystals(
                            crystals,
                            state,
                            options[int(chosen.removeprefix("option-"))],
                        ),
                        target,
                    ),
                    _sha(f"option:{seed}:{step}:{chosen}"),
                ),
                epsilon=_epsilon(step, option_episodes, 0.40),
            )
        option_exact = option_language.accuracy("option-context")

        context_language = ConsequenceMarkovLanguage(
            ActionFrontier.create(
                [
                    ActionBinding("left-action", "crystal", crystals[0].sha256),
                    ActionBinding("right-action", "crystal", crystals[1].sha256),
                ],
                action_schema_sha256=_sha("context-action-schema"),
                context_schema_sha256=_sha("left-right-context-schema"),
            ),
            ("tau-shared", "tau-unused"),
            seed=seed + 400,
            context_full_weight_visits=8,
        )
        context_episodes = max(2_000, primitive_episodes // 2)
        for step in range(context_episodes):
            context = "left-context" if step % 2 == 0 else "right-context"
            target = "left-action" if context == "left-context" else "right-action"
            decision = context_language.receiver_decide(
                "tau-shared",
                context,
                epsilon=_epsilon(step, context_episodes, 0.35),
            )
            feedback = ConsequenceFeedback.for_decision(
                decision,
                reward=1.0 if decision.action_id == target else -0.25,
                accepted=decision.action_id == target,
                outcome_receipt_sha256=_sha(f"context:{seed}:{step}"),
            )
            context_language.observe_receiver(decision, feedback)
        context_exact = float(
            context_language.decode_word("tau-shared", "left-context") == "left-action"
            and context_language.decode_word("tau-shared", "right-context")
            == "right-action"
        )
        in_vocab_unknown_abstains = (
            context_language.decode_word("tau-unused", "left-context") is None
        )

        grammar_exact, grammar_holistic, grammar_shuffled = _grammar_trial(
            seed=seed,
            episodes=grammar_episodes,
        )

        snapshot = LanguageSnapshot.freeze(language, contexts=contexts)
        child = ConsequenceMarkovLanguage(
            frontier, language.vocabulary, seed=seed + 500
        )
        cultural_episodes = max(3_000, primitive_episodes * 3 // 4)
        for step in range(cultural_episodes):
            intent_index = int(rng.integers(len(frontier.actions)))
            intent = frontier.action_ids[intent_index]
            state = _action_state(rng)
            context = f"state-bucket-{int(state[0]) % 4}"
            target = crystals[intent_index].apply(state)
            child.learn_from_teacher_episode(
                snapshot,
                intent,
                context,
                lambda chosen, state=state, target=target, step=step: (
                    1.0
                    if np.array_equal(
                        crystals[frontier.action_ids.index(chosen)].apply(state),
                        target,
                    )
                    else -0.25,
                    np.array_equal(
                        crystals[frontier.action_ids.index(chosen)].apply(state),
                        target,
                    ),
                    _sha(f"culture:{seed}:{step}:{chosen}"),
                ),
                epsilon=_epsilon(step, cultural_episodes, 0.35),
            )
        child.induce_sender_from_receiver(contexts[0])
        child_snapshot = LanguageSnapshot.freeze(child, contexts=contexts)
        cultural_exact = float(
            child_snapshot.sender_action_words == snapshot.sender_action_words
        )

        primitive_bindings = tuple(
            PrimitiveWordBinding(word, kind, digest)
            for word, (kind, digest) in sorted(
                snapshot.primitive_artifacts(frontier).items()
            )
        )
        lexicon_initial = ExecutableLexiconState.initial(
            primitive_bindings,
            language_snapshot_sha256=snapshot.sha256,
            frontier_sha256=frontier.sha256,
            authority_hashes=dict(frontier.authority_hashes),
        )
        sender_lexicon = ExecutableLexicon(lexicon_initial)
        receiver_lexicon = ExecutableLexicon(lexicon_initial)
        sender_words = dict(snapshot.sender_action_words)
        inner = ExecutableWordDefinition.create(
            "tau-inner",
            (
                sender_words["action-0"],
                sender_words["action-1"],
                sender_words["action-2"],
            ),
            language_snapshot_sha256=snapshot.sha256,
            frontier_sha256=frontier.sha256,
            authority_hashes=frontier.authority_hashes,
        )
        sender_lexicon.install(inner)
        outer = ExecutableWordDefinition.create(
            "tau-outer",
            ("tau-inner", sender_words["action-3"], "tau-inner"),
            language_snapshot_sha256=snapshot.sha256,
            frontier_sha256=frontier.sha256,
            authority_hashes=frontier.authority_hashes,
        )
        sender_lexicon.install(outer)
        _, first_repairs = receiver_lexicon.decode_or_request(
            ("tau-outer",), sender_lexicon
        )
        _, second_repairs = receiver_lexicon.decode_or_request(
            ("tau-outer",), sender_lexicon
        )
        macro_expanded = receiver_lexicon.state.expanded_length("tau-outer")
        macro_exact = float(
            receiver_lexicon.state.expand_word("tau-outer", max_actions=macro_expanded)
            == (
                sender_words["action-0"],
                sender_words["action-1"],
                sender_words["action-2"],
                sender_words["action-3"],
                sender_words["action-0"],
                sender_words["action-1"],
                sender_words["action-2"],
            )
        )

        lexicon_bank = ExecutableLexiconBank(
            Path(temporary) / "lexicon",
            initial_state=lexicon_initial,
        )
        current_word = sender_words["action-0"]
        bridge_words = tuple(sender_words[f"action-{index}"] for index in range(1, 4))
        for depth in range(1, recursive_depth + 1):
            definition = ExecutableWordDefinition.create(
                f"tau-dag-{depth}",
                (
                    current_word,
                    bridge_words[(depth - 1) % len(bridge_words)],
                    current_word,
                ),
                language_snapshot_sha256=snapshot.sha256,
                frontier_sha256=frontier.sha256,
                authority_hashes=frontier.authority_hashes,
            )
            lexicon_bank.append_definition(definition)
            current_word = definition.new_word_id
        lexicon_state = lexicon_bank.head()
        start = perf_counter()
        compiler = ExecutableWordCompiler(lexicon_state, compute_bank)
        compiled_receipt = compiler.compile(current_word)
        compile_seconds = perf_counter() - start
        expanded_words = lexicon_state.expand_word(
            current_word,
            max_actions=lexicon_state.expanded_length(current_word),
        )
        primitive_map = lexicon_state.primitive_map
        restored_primitives = {
            word: compute_bank.restore_crystal(binding.artifact_sha256)
            for word, binding in primitive_map.items()
        }
        flat_program = ComputeProgram.compose(
            tuple(restored_primitives[word] for word in expanded_words)
        )
        compute_bank.publish_program(flat_program)
        states = np.stack([_action_state(rng) for _ in range(25)])
        vm = ComputeCrystalVM(compute_bank)
        start = perf_counter()
        flat_execution = vm.execute(flat_program.sha256, states)
        flat_seconds = perf_counter() - start
        start = perf_counter()
        compiled_execution = vm.execute(
            compiled_receipt.program_sha256,
            states,
            charge_basis_sha256=compiled_receipt.charge_sha256,
        )
        compiled_seconds = perf_counter() - start
        recursive_exact = float(
            np.array_equal(flat_execution.output, compiled_execution.output)
        )
        compute_manifest = compute_bank.manifest()

        return {
            "context_dependent_word_exact": context_exact,
            "cultural_child_exact_surface": cultural_exact,
            "first_nested_definition_requests": first_repairs,
            "grammar_heldout_exact": grammar_exact,
            "grammar_holistic_heldout": grammar_holistic,
            "grammar_shuffled_exact": grammar_shuffled,
            "in_vocab_unknown_abstains": in_vocab_unknown_abstains,
            "macro_symbol_reduction": 1.0 - 1.0 / macro_expanded,
            "no_message_abstain_exact": no_message_abstain / tasks,
            "no_message_fixed_policy_exact": no_message_fixed / tasks,
            "novel_program_exact": exact / tasks,
            "option_word_exact": option_exact,
            "primitive_reward_exact": primitive_exact,
            "recursive_compile_seconds": compile_seconds,
            "recursive_compiled_seconds": compiled_seconds,
            "recursive_compute_charge_count": len(compute_manifest.charge_sha256s),
            "recursive_compute_crystal_count": len(compute_manifest.crystal_sha256s),
            "recursive_compute_program_count": len(compute_manifest.program_sha256s),
            "recursive_constant_discharge": compiled_receipt.constant_discharge,
            "recursive_deployment_symbol_reduction": (
                1.0 - 1.0 / compiled_receipt.expanded_primitive_actions
            ),
            "recursive_definition_references": (
                compiled_receipt.stored_definition_references
            ),
            "recursive_equivalent_source_work": (
                compiled_execution.receipt.equivalent_unfused_source_work
            ),
            "recursive_execution_speedup": flat_seconds / max(compiled_seconds, 1e-12),
            "recursive_expanded_actions": (compiled_receipt.expanded_primitive_actions),
            "recursive_flat_seconds": flat_seconds,
            "recursive_historical_work_released": (
                compiled_execution.receipt.historical_work_released
            ),
            "recursive_knowledge_reference_reduction": (
                1.0
                - compiled_receipt.stored_definition_references
                / compiled_receipt.expanded_primitive_actions
            ),
            "recursive_live_discharge_work": (
                compiled_execution.receipt.live_discharge_work
            ),
            "recursive_word_exact": recursive_exact,
            "second_nested_definition_requests": second_repairs,
            "seed": seed,
            "shuffled_semantics_exact": shuffled / tasks,
            "self_hosted_macro_exact": macro_exact,
        }


def build_report(
    *,
    seeds: int = 5,
    tasks: int = 1_000,
    primitive_episodes: int = 6_000,
    grammar_episodes: int = 8_000,
    recursive_depth: int = 12,
) -> dict[str, Any]:
    rows = [
        run_seed(
            seed,
            tasks=tasks,
            primitive_episodes=primitive_episodes,
            grammar_episodes=grammar_episodes,
            recursive_depth=recursive_depth,
        )
        for seed in range(seeds)
    ]
    numeric = tuple(
        key
        for key, value in rows[0].items()
        if key != "seed" and not isinstance(value, bool)
    )
    aggregate = {
        key: float(np.mean([float(row[key]) for row in rows])) for key in numeric
    }
    for key, value in rows[0].items():
        if isinstance(value, bool):
            aggregate[f"all_{key}"] = all(bool(row[key]) for row in rows)
    aggregate.update({"seeds": seeds, "tasks_per_seed": tasks})
    body = {
        "aggregate": aggregate,
        "grammar_episodes": grammar_episodes,
        "primitive_episodes": primitive_episodes,
        "recursive_depth": recursive_depth,
        "seeds": rows,
        "source_poc_sha256": SOURCE_POC_SHA256,
    }
    return {
        "body": body,
        "schema": REPORT_SCHEMA,
        "sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", type=_positive_int, default=5)
    parser.add_argument("--tasks", type=_positive_int, default=1_000)
    parser.add_argument("--primitive-episodes", type=_positive_int, default=6_000)
    parser.add_argument("--grammar-episodes", type=_positive_int, default=8_000)
    parser.add_argument("--recursive-depth", type=_positive_int, default=12)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_report(
        seeds=args.seeds,
        tasks=args.tasks,
        primitive_episodes=args.primitive_episodes,
        grammar_episodes=args.grammar_episodes,
        recursive_depth=args.recursive_depth,
    )
    rendered = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
