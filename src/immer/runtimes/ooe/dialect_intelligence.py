"""Mechanism benchmark for cross-dialect executable-language transfer."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .compute_crystals import ComputeCrystal, ComputeCrystalBank
from .dialect_mesh import (
    DialectMeshIntegrityError,
    align_dialects,
    localize_portable_program,
    portable_program_from_discovery,
    translate_words,
)
from .executable_lexicon import (
    ExecutableLexiconState,
    ExecutableWordCompiler,
    ExecutableWordDefinition,
    PrimitiveWordBinding,
)
from .identity import canonical_json_bytes
from .language_bridge import LanguageMacroDiscoveryReceipt
from .markov_language import (
    ActionBinding,
    ActionFrontier,
    ConsequenceMarkovLanguage,
    LanguageSnapshot,
    MarkovLanguageError,
)


DIALECT_MESH_INTELLIGENCE_SCHEMA = "immer-ooe-dialect-mesh-intelligence/v2"
MIN_CONTEXT_TRAINING_EPISODES = 800
MAX_DIALECT_TRAINING_ATTEMPTS = 16


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _epsilon(step: int, total: int) -> float:
    fraction = step / max(1, total - 1)
    return 0.40 * (1.0 - fraction) + 0.01 * fraction


def _crystals() -> tuple[ComputeCrystal, ...]:
    return (
        ComputeCrystal.affine([[1.0]], [1.0]),
        ComputeCrystal.affine([[-1.0]], [0.0]),
        ComputeCrystal.affine([[2.0]], [3.0]),
    )


def _frontier(crystals: tuple[ComputeCrystal, ...]) -> ActionFrontier:
    return ActionFrontier.create(
        tuple(
            ActionBinding(f"action-{index}", "crystal", crystal.sha256)
            for index, crystal in enumerate(crystals)
        ),
        action_schema_sha256=_sha("mesh-action-schema"),
        context_schema_sha256=_sha("mesh-context-schema"),
        authority_hashes={"verifier": _sha("mesh-consequence-verifier")},
    )


def _dialect_contexts(dialect_seed: int, count: int) -> tuple[str, ...]:
    return tuple(
        f"mesh-dialect-{dialect_seed}-context-{index}" for index in range(count)
    )


def _shared_vocabulary() -> tuple[str, ...]:
    """Return one colliding surface inventory shared by every dialect."""

    return tuple(f"mesh-word-{index}" for index in range(6))


def _non_identity_permutation(
    rng: np.random.Generator,
    size: int,
) -> tuple[int, ...]:
    if size < 2:
        raise ValueError("a permutation placebo needs at least two actions")
    identity = tuple(range(size))
    permutation = identity
    while permutation == identity:
        permutation = tuple(int(value) for value in rng.permutation(size))
    return permutation


def _train_language(
    frontier: ActionFrontier,
    crystals: tuple[ComputeCrystal, ...],
    *,
    context_ids: tuple[str, ...],
    vocabulary: tuple[str, ...] | None = None,
    seed: int,
    episodes: int,
) -> ConsequenceMarkovLanguage:
    if not context_ids:
        raise ValueError("language training needs at least one context")
    language = ConsequenceMarkovLanguage(
        frontier,
        _shared_vocabulary() if vocabulary is None else vocabulary,
        seed=seed,
    )
    rng = np.random.default_rng(seed + 50_000)
    for step in range(episodes):
        intent_index = int(rng.integers(len(crystals)))
        intent = frontier.action_ids[intent_index]
        value = np.array([int(rng.integers(1, 20))], dtype=np.float64)
        target = crystals[intent_index].apply(value)
        context_id = context_ids[int(rng.integers(len(context_ids)))]
        language.run_episode(
            intent,
            context_id,
            lambda chosen, value=value, target=target, step=step: (
                1.0
                if np.array_equal(
                    crystals[frontier.action_ids.index(chosen)].apply(value),
                    target,
                )
                else -0.25,
                np.array_equal(
                    crystals[frontier.action_ids.index(chosen)].apply(value),
                    target,
                ),
                _sha(f"mesh:{seed}:{step}:{chosen}"),
            ),
            epsilon=_epsilon(step, episodes),
        )
    return language


def _merge_context_snapshots(
    frontier: ActionFrontier,
    components: tuple[LanguageSnapshot, ...],
) -> LanguageSnapshot:
    """Merge independently learned one-context snapshots without averaging them."""

    if not components:
        raise ValueError("a merged dialect needs context components")
    vocabulary = _shared_vocabulary()
    context_word_actions = []
    context_action_words = []
    component_evidence = []
    seen_contexts: set[str] = set()
    for component in components:
        if (
            component.frontier_sha256 != frontier.sha256
            or set(component.vocabulary) != set(vocabulary)
            or len(component.context_word_actions) != 1
            or len(component.context_action_words) != 1
        ):
            raise RuntimeError("context component contract is invalid")
        context_id, word_actions = component.context_word_actions[0]
        sender_context_id, action_words = component.context_action_words[0]
        if context_id != sender_context_id or context_id in seen_contexts:
            raise RuntimeError("context component identity is invalid")
        seen_contexts.add(context_id)
        context_word_actions.append((context_id, word_actions))
        context_action_words.append((context_id, action_words))
        component_evidence.append(
            {
                "context_id": context_id,
                "snapshot_sha256": component.sha256,
            }
        )

    global_word_actions = []
    receiver_maps = tuple(dict(rows) for _, rows in context_word_actions)
    for word in vocabulary:
        actions = tuple(mapping.get(word) for mapping in receiver_maps)
        if actions[0] is not None and len(set(actions)) == 1:
            global_word_actions.append((word, str(actions[0])))

    sender_action_words = []
    sender_maps = tuple(dict(rows) for _, rows in context_action_words)
    for action in frontier.action_ids:
        words = tuple(mapping.get(action) for mapping in sender_maps)
        if words[0] is not None and len(set(words)) == 1:
            sender_action_words.append((action, str(words[0])))

    learner_state_sha256 = hashlib.sha256(
        canonical_json_bytes(
            {
                "components": component_evidence,
                "format": "immer-ooe-independent-context-language-merge/v1",
                "frontier_sha256": frontier.sha256,
            }
        )
    ).hexdigest()
    return LanguageSnapshot(
        frontier_sha256=frontier.sha256,
        learner_state_sha256=learner_state_sha256,
        vocabulary=vocabulary,
        sender_action_words=tuple(sender_action_words),
        global_word_actions=tuple(global_word_actions),
        context_word_actions=tuple(context_word_actions),
        context_action_words=tuple(context_action_words),
    )


def _distinct_context_mappings(snapshot: LanguageSnapshot) -> int:
    return len({tuple(rows) for _, rows in snapshot.context_action_words})


def _learn_dialect_snapshot(
    frontier: ActionFrontier,
    crystals: tuple[ComputeCrystal, ...],
    *,
    context_ids: tuple[str, ...],
    dialect_seed: int,
    train_episodes: int,
    globally_stable: bool,
    require_no_global_action_words: bool,
) -> tuple[LanguageSnapshot, tuple[LanguageSnapshot, ...], int]:
    """Learn each context separately, then merge their exact frozen mappings."""

    episodes_per_context = max(
        MIN_CONTEXT_TRAINING_EPISODES,
        (train_episodes + len(context_ids) - 1) // len(context_ids),
    )
    attempts = 1 if globally_stable else MAX_DIALECT_TRAINING_ATTEMPTS
    for attempt in range(attempts):
        components = []
        try:
            for context_index, context_id in enumerate(context_ids):
                if globally_stable:
                    component_seed = dialect_seed
                else:
                    component_seed = int(
                        _sha(
                            "mesh-context-learner:"
                            f"{dialect_seed}:{attempt}:{context_index}"
                        )[:15],
                        16,
                    )
                language = _train_language(
                    frontier,
                    crystals,
                    context_ids=(context_id,),
                    seed=component_seed,
                    episodes=episodes_per_context,
                )
                components.append(
                    LanguageSnapshot.freeze(language, contexts=(context_id,))
                )
        except MarkovLanguageError:
            continue
        merged = _merge_context_snapshots(frontier, tuple(components))
        distinct = _distinct_context_mappings(merged)
        if globally_stable:
            if distinct == 1 and len(merged.sender_action_words) == len(
                frontier.action_ids
            ):
                return merged, tuple(components), episodes_per_context
            continue
        if distinct != len(context_ids):
            continue
        if require_no_global_action_words and merged.sender_action_words:
            continue
        return merged, tuple(components), episodes_per_context
    raise RuntimeError("could not learn the required independent context dialect")


def _portable_program(
    snapshot: LanguageSnapshot,
    frontier: ActionFrontier,
    *,
    context_id: str,
) -> tuple[ExecutableWordDefinition, LanguageMacroDiscoveryReceipt]:
    actions = ("action-0", "action-1", "action-2", "action-0")
    words = tuple(snapshot.encode_action(action, context_id) for action in actions)
    if any(word is None for word in words):
        raise RuntimeError("source dialect is incomplete")
    definition = ExecutableWordDefinition.create(
        "mesh-source-macro",
        tuple(str(word) for word in words),
        language_snapshot_sha256=snapshot.sha256,
        frontier_sha256=frontier.sha256,
        authority_hashes=frontier.authority_hashes,
    )
    supports = (
        _sha("mesh-trajectory-0"),
        _sha("mesh-trajectory-1"),
        _sha("mesh-trajectory-2"),
    )
    discovery = LanguageMacroDiscoveryReceipt(
        language_snapshot_sha256=snapshot.sha256,
        frontier_sha256=frontier.sha256,
        authority_hashes=frontier.authority_hashes,
        trajectory_sha256s=supports,
        min_support=3,
        min_macro_length=4,
        max_macro_length=4,
        candidate_count=1,
        definitions=(definition,),
        definition_supports=((definition.sha256, supports),),
    )
    return definition, discovery


def run_dialect_mesh_seed(
    seed: int,
    *,
    dialects: int = 5,
    contexts_per_dialect: int = 3,
    train_episodes: int = 4_000,
    programs_per_pair: int = 200,
) -> dict[str, Any]:
    if (
        dialects < 2
        or contexts_per_dialect < 2
        or train_episodes < 1
        or programs_per_pair < 1
    ):
        raise ValueError("dialect benchmark bounds are invalid")
    with tempfile.TemporaryDirectory(prefix="immer-dialect-mesh-") as temporary:
        root = Path(temporary)
        bank = ComputeCrystalBank(root / "compute")
        crystals = _crystals()
        for crystal in crystals:
            bank.publish_crystal(crystal)
        frontier = _frontier(crystals)
        dialect_seeds = tuple(seed * 1_000 + index for index in range(dialects))
        context_inventories = tuple(
            _dialect_contexts(dialect_seed, contexts_per_dialect)
            for dialect_seed in dialect_seeds
        )
        learned_dialects = tuple(
            _learn_dialect_snapshot(
                frontier,
                crystals,
                context_ids=context_inventories[index],
                dialect_seed=dialect_seed,
                train_episodes=train_episodes,
                globally_stable=index == 0,
                require_no_global_action_words=index == 1,
            )
            for index, dialect_seed in enumerate(dialect_seeds)
        )
        snapshots = tuple(row[0] for row in learned_dialects)
        component_snapshots = tuple(row[1] for row in learned_dialects)
        context_training_episodes = tuple(row[2] for row in learned_dialects)
        surface_digests = {
            _sha(str(tuple(rows for _, rows in snapshot.context_action_words)))
            for snapshot in snapshots
        }
        rng = np.random.default_rng(seed + 60_000)
        translated_exact = 0
        direct_surface_exact = 0
        permutation_placebo_exact = 0
        translated_tokens_exact = 0
        direct_surface_tokens_exact = 0
        permutation_placebo_tokens_exact = 0
        surface_overlap_tokens = 0
        surface_semantic_collisions = 0
        surface_semantic_comparisons = 0
        total = 0
        total_tokens = 0
        translation_hashes = []
        target_context_translation_counts: dict[str, list[int]] = {}
        for source_index, source in enumerate(snapshots):
            for target_index, target in enumerate(snapshots):
                if source_index == target_index:
                    continue
                source_context = context_inventories[source_index][0]
                for target_context in context_inventories[target_index]:
                    receipt = align_dialects(
                        source,
                        target,
                        frontier,
                        frontier,
                        source_context_id=source_context,
                        target_context_id=target_context,
                    )
                    translation_hashes.append(receipt.sha256)
                    target_context_translation_counts.setdefault(target_context, [0, 0])
                    action_target_words = {
                        action: target_word
                        for _, target_word, action in receipt.word_translations
                    }
                    permutation = _non_identity_permutation(
                        rng, len(frontier.action_ids)
                    )
                    permuted_actions = {
                        action: frontier.action_ids[permutation[index]]
                        for index, action in enumerate(frontier.action_ids)
                    }
                    placebo_mapping = {
                        source_word: action_target_words[permuted_actions[action]]
                        for source_word, _, action in receipt.word_translations
                    }
                    for action in frontier.action_ids:
                        source_word = source.encode_action(action, source_context)
                        if source_word is None:
                            raise RuntimeError("source dialect is incomplete")
                        surface_overlap_tokens += int(source_word in target.vocabulary)
                        target_action = target.decode_word(source_word, target_context)
                        if target_action is not None:
                            surface_semantic_comparisons += 1
                            surface_semantic_collisions += int(target_action != action)
                    for _ in range(programs_per_pair):
                        length = int(rng.integers(2, 9))
                        actions = tuple(
                            frontier.action_ids[int(value)]
                            for value in rng.integers(
                                0, len(frontier.actions), size=length
                            )
                        )
                        source_words = tuple(
                            str(source.encode_action(action, source_context))
                            for action in actions
                        )
                        translated = translate_words(
                            source_words,
                            receipt,
                            source_snapshot=source,
                            target_snapshot=target,
                            source_frontier=frontier,
                            target_frontier=frontier,
                        )
                        if translated is None:
                            raise RuntimeError("complete dialect failed translation")
                        decoded = tuple(
                            target.decode_word(word, target_context)
                            for word in translated
                        )
                        direct = tuple(
                            target.decode_word(word, target_context)
                            for word in source_words
                        )
                        placebo = tuple(
                            target.decode_word(placebo_mapping[word], target_context)
                            for word in source_words
                        )
                        translated_exact += int(decoded == actions)
                        direct_surface_exact += int(direct == actions)
                        permutation_placebo_exact += int(placebo == actions)
                        translated_tokens_exact += sum(
                            actual == expected
                            for actual, expected in zip(decoded, actions, strict=True)
                        )
                        direct_surface_tokens_exact += sum(
                            actual == expected
                            for actual, expected in zip(direct, actions, strict=True)
                        )
                        permutation_placebo_tokens_exact += sum(
                            actual == expected
                            for actual, expected in zip(placebo, actions, strict=True)
                        )
                        total += 1
                        total_tokens += length
                        target_context_translation_counts[target_context][0] += int(
                            decoded == actions
                        )
                        target_context_translation_counts[target_context][1] += 1

        source_context = context_inventories[0][0]
        source_definition, discovery = _portable_program(
            snapshots[0], frontier, context_id=source_context
        )
        portable = portable_program_from_discovery(
            discovery,
            source_definition,
            snapshots[0],
            frontier,
        )
        portable_exact = 0
        historical_work = []
        localization_hashes = []
        values = np.arange(1, 26, dtype=np.float64).reshape(25, 1)
        expected = values
        for action in portable.action_sequence:
            expected = crystals[int(action.removeprefix("action-"))].apply(expected)
        target_context_localization_counts: dict[str, int] = {}
        for index, snapshot in enumerate(snapshots):
            for target_context in context_inventories[index]:
                localized, localization = localize_portable_program(
                    portable,
                    snapshot,
                    frontier,
                    source_discovery=discovery,
                    source_definition=source_definition,
                    source_snapshot=snapshots[0],
                    source_frontier=frontier,
                    target_context_id=target_context,
                )
                localization_hashes.append(localization.sha256)
                primitives = tuple(
                    PrimitiveWordBinding(word, kind, digest)
                    for word, (kind, digest) in sorted(
                        snapshot.primitive_artifacts(
                            frontier, context_id=target_context
                        ).items()
                    )
                )
                state = ExecutableLexiconState.initial(
                    primitives,
                    language_snapshot_sha256=snapshot.sha256,
                    frontier_sha256=frontier.sha256,
                    authority_hashes=frontier.authority_hashes,
                    context_id=target_context,
                ).with_definition(localized)
                compiler = ExecutableWordCompiler(state, bank)
                compiled = compiler.compile(localized.new_word_id)
                execution = compiler.execute(compiled, values)
                exact = int(np.array_equal(execution.output, expected))
                portable_exact += exact
                target_context_localization_counts[target_context] = exact
                historical_work.append(execution.receipt.historical_work_released)

        changed_frontier = ActionFrontier.create(
            (
                frontier.actions[0],
                ActionBinding("action-1", "crystal", _sha("changed-mesh-crystal")),
                frontier.actions[2],
            ),
            action_schema_sha256=_sha("changed-mesh-action-schema"),
            context_schema_sha256=frontier.context_schema_sha256,
            authority_hashes=dict(frontier.authority_hashes),
        )
        changed_language = _train_language(
            changed_frontier,
            (
                crystals[0],
                ComputeCrystal.affine([[3.0]], [5.0]),
                crystals[2],
            ),
            context_ids=_dialect_contexts(seed + 70_000, contexts_per_dialect),
            seed=seed + 70_000,
            episodes=train_episodes,
        )
        changed_contexts = _dialect_contexts(seed + 70_000, contexts_per_dialect)
        changed_snapshot = LanguageSnapshot.freeze(
            changed_language, contexts=changed_contexts
        )
        binding_change_rejected = False
        try:
            localize_portable_program(
                portable,
                changed_snapshot,
                changed_frontier,
                source_discovery=discovery,
                source_definition=source_definition,
                source_snapshot=snapshots[0],
                source_frontier=frontier,
                target_context_id=changed_contexts[0],
            )
        except DialectMeshIntegrityError:
            binding_change_rejected = True
        target_context_accuracies = tuple(
            exact / count for exact, count in target_context_translation_counts.values()
        )
        portable_localizations = dialects * contexts_per_dialect
        semantic_comparison_denominator = max(1, surface_semantic_comparisons)
        target_distinct_context_mappings = tuple(
            _distinct_context_mappings(snapshot) for snapshot in snapshots[1:]
        )
        context_specific_dialects = sum(
            count > 1 for count in target_distinct_context_mappings
        )
        target_dialects_without_global_action_words = sum(
            not snapshot.sender_action_words for snapshot in snapshots[1:]
        )
        return {
            "binding_change_rejected": binding_change_rejected,
            "context_component_sha256s": tuple(
                tuple(component.sha256 for component in components)
                for components in component_snapshots
            ),
            "context_specific_dialects": context_specific_dialects,
            "context_training_episodes_per_learner": min(context_training_episodes),
            "contexts_per_dialect": contexts_per_dialect,
            "dialects": dialects,
            "direct_surface_accuracy": direct_surface_exact / total,
            "direct_surface_token_accuracy": direct_surface_tokens_exact / total_tokens,
            "localization_sha256s": tuple(localization_hashes),
            "mean_historical_work_released": float(np.mean(historical_work)),
            "minimum_distinct_context_mappings": min(target_distinct_context_mappings),
            "minimum_target_context_translation_accuracy": min(
                target_context_accuracies
            ),
            "pairwise_programs": total,
            "permutation_placebo_accuracy": permutation_placebo_exact / total,
            "permutation_placebo_token_accuracy": permutation_placebo_tokens_exact
            / total_tokens,
            "portable_localization_accuracy": portable_exact / portable_localizations,
            "portable_localizations": portable_localizations,
            "portable_program_sha256": portable.sha256,
            "seed": seed,
            "surface_semantic_collision_rate": surface_semantic_collisions
            / semantic_comparison_denominator,
            "surface_token_overlap_rate": surface_overlap_tokens
            / (len(translation_hashes) * len(frontier.action_ids)),
            "source_globally_stable_action_words": len(
                snapshots[0].sender_action_words
            ),
            "target_dialects_without_global_action_words": (
                target_dialects_without_global_action_words
            ),
            "target_context_localization_exact": all(
                target_context_localization_counts.values()
            ),
            "target_context_translation_exact": all(
                exact == count
                for exact, count in target_context_translation_counts.values()
            ),
            "target_contexts_exercised": len(target_context_translation_counts),
            "translated_program_accuracy": translated_exact / total,
            "translated_token_accuracy": translated_tokens_exact / total_tokens,
            "translation_receipts": len(translation_hashes),
            "translation_sha256s": tuple(translation_hashes),
            "unique_surface_dialects": len(surface_digests),
        }


def run_dialect_mesh_intelligence_benchmark(
    *,
    seeds: int = 5,
    dialects: int = 5,
    contexts_per_dialect: int = 3,
    train_episodes: int = 4_000,
    programs_per_pair: int = 200,
) -> dict[str, Any]:
    if seeds < 1:
        raise ValueError("seeds must be positive")
    rows = tuple(
        run_dialect_mesh_seed(
            seed,
            dialects=dialects,
            contexts_per_dialect=contexts_per_dialect,
            train_episodes=train_episodes,
            programs_per_pair=programs_per_pair,
        )
        for seed in range(seeds)
    )
    aggregate: dict[str, object] = {"seeds": seeds}
    for key, value in rows[0].items():
        if key == "seed" or key.endswith("sha256") or key.endswith("sha256s"):
            continue
        if isinstance(value, bool):
            aggregate[f"all_{key}"] = all(bool(row[key]) for row in rows)
        elif isinstance(value, (int, float)):
            aggregate[key] = float(np.mean([float(row[key]) for row in rows]))
    body = {
        "aggregate": aggregate,
        "controls": {
            "direct_surface": "identity transfer over a fully shared token inventory",
            "independent_context_dialects": (
                "separately learned one-context snapshots merged without averaging"
            ),
            "permutation_placebo": (
                "non-identity action-word permutation per ordered dialect-context pair"
            ),
        },
        "contexts_per_dialect": contexts_per_dialect,
        "dialects": dialects,
        "programs_per_pair": programs_per_pair,
        "seeds": list(rows),
        "train_episodes": train_episodes,
    }
    return {
        "body": body,
        "schema": DIALECT_MESH_INTELLIGENCE_SCHEMA,
        "sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }


__all__ = [
    "DIALECT_MESH_INTELLIGENCE_SCHEMA",
    "run_dialect_mesh_intelligence_benchmark",
    "run_dialect_mesh_seed",
]
