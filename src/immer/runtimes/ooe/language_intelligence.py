"""Mechanism benchmark for continual self-extension of executable language."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .compute_crystals import (
    ComputeCrystal,
    ComputeCrystalBank,
)
from .crystal import CrystalStore
from .executable_lexicon import (
    ExecutableLexiconBank,
    ExecutableLexiconState,
    ExecutableWordCompiler,
    ExecutableWordDefinition,
    PrimitiveWordBinding,
)
from .identity import canonical_json_bytes
from .language_bridge import (
    ConsequenceLanguageStateBank,
    LanguageMacroDiscoveryReceipt,
    LanguageOutcomeCommitReceipt,
    VerifiedWordTrajectory,
    discover_language_macros,
    migrate_language_frontier,
    promote_and_migrate_language_revision,
    resolve_snapshot_compute_bindings,
)
from .markov_language import (
    ActionBinding,
    ActionFrontier,
    ConsequenceMarkovLanguage,
    LanguageSnapshot,
)


LANGUAGE_GROWTH_INTELLIGENCE_SCHEMA = "immer-ooe-language-growth-intelligence/v1"
DIRECT_CONSEQUENCE_REWARD_POLICY_SHA256 = hashlib.sha256(
    canonical_json_bytes(
        {
            "failure": -0.25,
            "format": "immer-ooe-direct-consequence-reward/v1",
            "success": 1.0,
        }
    )
).hexdigest()
TERMINAL_EPISODE_VERIFIER_SHA256 = hashlib.sha256(
    b"immer-ooe-language-growth-terminal-verifier/v1"
).hexdigest()


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _epsilon(step: int, total: int, *, start: float) -> float:
    fraction = step / max(1, total - 1)
    return start * (1.0 - fraction) + 0.01 * fraction


def _crystals() -> tuple[ComputeCrystal, ...]:
    return (
        ComputeCrystal.affine(
            [[1.0, 0.0], [0.0, 1.0]],
            [1.0, 0.0],
        ),
        ComputeCrystal.affine(
            [[0.0, 1.0], [1.0, 0.0]],
            [0.0, 0.0],
        ),
        ComputeCrystal.affine(
            [[-1.0, 0.0], [0.0, 1.0]],
            [0.0, 0.0],
        ),
    )


def _frontier(crystals: tuple[ComputeCrystal, ...]) -> ActionFrontier:
    return ActionFrontier.create(
        tuple(
            ActionBinding(f"action-{index}", "crystal", crystal.sha256)
            for index, crystal in enumerate(crystals)
        ),
        action_schema_sha256=_sha("growth-action-schema"),
        context_schema_sha256=_sha("growth-context-schema"),
        authority_hashes={
            "episode-terminal-verifier": TERMINAL_EPISODE_VERIFIER_SHA256,
            "language-reward-policy": DIRECT_CONSEQUENCE_REWARD_POLICY_SHA256,
        },
    )


def _value(rng: np.random.Generator) -> np.ndarray:
    return np.array(
        [int(rng.integers(1, 10)), int(rng.integers(11, 20))],
        dtype=np.float64,
    )


def _artifact_execution(
    action_id: str,
    frontier: ActionFrontier,
    bank: ComputeCrystalBank,
    value: np.ndarray,
    cache: dict[str, tuple[ComputeCrystal, ...]],
) -> np.ndarray:
    crystals = cache.get(action_id)
    if crystals is None:
        binding = frontier.binding(action_id)
        if binding.artifact_kind == "crystal":
            crystals = (bank.restore_crystal(binding.artifact_sha256),)
        elif binding.artifact_kind == "program":
            _program, crystals = bank.resolve_program(binding.artifact_sha256)
        else:
            raise ValueError("growth benchmark received a non-compute action")
        cache[action_id] = crystals
    result = value
    for crystal in crystals:
        result = crystal.apply(result)
    return result


def _train(
    language: ConsequenceMarkovLanguage,
    bank: ComputeCrystalBank,
    *,
    rng: np.random.Generator,
    episodes: int,
    contexts: tuple[str, ...],
    nonce_prefix: str,
    watch_action_id: str | None = None,
) -> int | None:
    cache: dict[str, tuple[ComputeCrystal, ...]] = {}
    first_confident_step: int | None = None
    for step in range(episodes):
        intent = language.action_ids[int(rng.integers(len(language.action_ids)))]
        context = contexts[int(rng.integers(len(contexts)))]
        value = _value(rng)
        target = _artifact_execution(intent, language.frontier, bank, value, cache)
        language.run_episode(
            intent,
            context,
            lambda chosen, value=value, target=target, step=step: (
                1.0
                if np.array_equal(
                    _artifact_execution(chosen, language.frontier, bank, value, cache),
                    target,
                )
                else -0.25,
                np.array_equal(
                    _artifact_execution(chosen, language.frontier, bank, value, cache),
                    target,
                ),
                _sha(f"{nonce_prefix}:{step}:{chosen}"),
            ),
            epsilon=_epsilon(step, episodes, start=0.40),
        )
        if (
            watch_action_id is not None
            and first_confident_step is None
            and all(
                language.encode_action(watch_action_id, context) is not None
                for context in contexts
            )
        ):
            first_confident_step = step + 1
    return first_confident_step


def _mean_accuracy(
    language: ConsequenceMarkovLanguage,
    contexts: tuple[str, ...],
) -> float:
    return float(np.mean([language.accuracy(context) for context in contexts]))


def _selected_action_accuracy(
    language: ConsequenceMarkovLanguage,
    contexts: tuple[str, ...],
    action_ids: tuple[str, ...],
) -> float:
    return float(
        np.mean(
            [
                language.encode_action(action, context) is not None
                for context in contexts
                for action in action_ids
            ]
        )
    )


def _promotion_trajectories(
    snapshot: LanguageSnapshot,
    frontier: ActionFrontier,
    *,
    seed: int,
) -> tuple[VerifiedWordTrajectory, ...]:
    action_words = dict(snapshot.sender_action_words)
    actions = (
        frontier.action_ids[0],
        frontier.action_ids[1],
        frontier.action_ids[2],
    )
    words = tuple(action_words[action] for action in actions)
    trajectories = []
    ordinal = 0
    for episode in range(3):
        commits = []
        for action, word in zip(actions, words, strict=True):
            commits.append(
                LanguageOutcomeCommitReceipt(
                    routing_decision_sha256=_sha(f"growth-routing:{seed}:{ordinal}"),
                    frontier_sha256=frontier.sha256,
                    external_outcome_kind="direct-exact-consequence",
                    external_outcome_sha256=_sha(f"growth-outcome:{seed}:{ordinal}"),
                    feedback_sha256=_sha(f"growth-feedback:{seed}:{ordinal}"),
                    language_state_before_sha256=_sha(
                        f"growth-before:{seed}:{ordinal}"
                    ),
                    language_state_after_sha256=_sha(f"growth-after:{seed}:{ordinal}"),
                    action_id=action,
                    word_id=word,
                    context_id="growth-context-0",
                    accepted=True,
                    reward=1.0,
                )
            )
            ordinal += 1
        trajectories.append(
            VerifiedWordTrajectory.create(
                commits,
                snapshot,
                episode_sha256=_sha(f"growth-episode:{seed}:{episode}"),
                terminal_verifier_sha256=TERMINAL_EPISODE_VERIFIER_SHA256,
                terminal_verification_sha256=_sha(f"growth-terminal:{seed}:{episode}"),
                expected_steps=3,
                frontier=frontier,
            )
        )
    return tuple(trajectories)


def run_language_growth_seed(
    seed: int,
    *,
    base_episodes: int = 5_000,
    growth_episodes: int = 5_000,
) -> dict[str, Any]:
    if base_episodes < 1 or growth_episodes < 1:
        raise ValueError("episode counts must be positive")
    rng = np.random.default_rng(seed + 10_000)
    contexts = tuple(f"growth-context-{index}" for index in range(3))
    with tempfile.TemporaryDirectory(prefix="immer-language-growth-") as temporary:
        root = Path(temporary)
        bank = ComputeCrystalBank(root / "compute")
        crystals = _crystals()
        for crystal in crystals:
            bank.publish_crystal(crystal)
        frontier = _frontier(crystals)
        language = ConsequenceMarkovLanguage(
            frontier,
            tuple(f"growth-word-{index}" for index in range(8)),
            seed=seed,
        )
        _train(
            language,
            bank,
            rng=rng,
            episodes=base_episodes,
            contexts=contexts,
            nonce_prefix=f"base:{seed}",
        )
        base_accuracy = _mean_accuracy(language, contexts)
        snapshot = LanguageSnapshot.freeze(language, contexts=contexts)
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
            authority_hashes=frontier.authority_hashes,
        )
        lexicon_bank = ExecutableLexiconBank(
            root / "lexicon", initial_state=lexicon_initial
        )
        trajectories = _promotion_trajectories(snapshot, frontier, seed=seed)
        discovery: LanguageMacroDiscoveryReceipt = discover_language_macros(
            trajectories,
            snapshot,
            frontier,
            min_support=3,
            min_macro_length=3,
            max_macro_length=3,
            max_definitions=1,
        )
        definition = discovery.definitions[0]
        defined = lexicon_bank.append_definition(definition)
        compiler = ExecutableWordCompiler(defined, bank)
        compiled = compiler.compile(definition.new_word_id)
        language_state_bank = ConsequenceLanguageStateBank(
            CrystalStore(root / "language")
        )
        language_state_bank.initialize(language)
        (
            grown,
            grown_frontier,
            promotion,
            migration,
            revision,
        ) = promote_and_migrate_language_revision(
            language,
            language_state_bank,
            discovery,
            definition,
            compiled,
            seed=seed + 1_000,
        )
        retained_accuracy = _selected_action_accuracy(
            grown,
            contexts,
            language.action_ids,
        )
        overall_accuracy_after_promotion = _mean_accuracy(grown, contexts)
        promoted_action = promotion.promoted_action_id
        promoted_initially_abstains = all(
            grown.encode_action(promoted_action, context) is None
            for context in contexts
        )
        fresh = ConsequenceMarkovLanguage(
            grown_frontier,
            grown.vocabulary,
            seed=seed + 2_000,
        )
        no_memory_accuracy = _mean_accuracy(fresh, contexts)
        episodes_to_promoted_action = _train(
            grown,
            bank,
            rng=rng,
            episodes=growth_episodes,
            contexts=contexts,
            nonce_prefix=f"growth:{seed}",
            watch_action_id=promoted_action,
        )
        final_accuracy = _mean_accuracy(grown, contexts)
        promoted_final_exact = float(
            all(
                grown.encode_action(promoted_action, context) is not None
                for context in contexts
            )
        )

        grown_snapshot = LanguageSnapshot.freeze(grown, contexts=contexts)
        second_bindings, second_resolution = resolve_snapshot_compute_bindings(
            grown_snapshot,
            grown_frontier,
        )
        second_initial = ExecutableLexiconState.initial(
            second_bindings,
            language_snapshot_sha256=grown_snapshot.sha256,
            frontier_sha256=grown_frontier.sha256,
            authority_hashes=grown_frontier.authority_hashes,
        )
        promoted_word = grown_snapshot.encode_action(promoted_action, contexts[0])
        base_word = grown_snapshot.encode_action(grown.action_ids[0], contexts[0])
        if promoted_word is None or base_word is None:
            raise RuntimeError("grown snapshot lost a learned action word")
        second_definition = ExecutableWordDefinition.create(
            "second-generation-" + _sha(f"{seed}:{promoted_action}")[:24],
            (promoted_word, base_word, promoted_word),
            language_snapshot_sha256=grown_snapshot.sha256,
            frontier_sha256=grown_frontier.sha256,
            authority_hashes=grown_frontier.authority_hashes,
        )
        second_state = second_initial.with_definition(second_definition)
        second_compiler = ExecutableWordCompiler(second_state, bank)
        second_compiled = second_compiler.compile(second_definition.new_word_id)

        shifted_frontier = ActionFrontier.create(
            grown_frontier.actions,
            action_schema_sha256=grown_frontier.action_schema_sha256,
            context_schema_sha256=_sha("growth-shifted-context-schema"),
            authority_hashes=dict(grown_frontier.authority_hashes),
        )
        shifted, shifted_receipt = migrate_language_frontier(
            grown,
            shifted_frontier,
            seed=seed + 3_000,
        )
        shifted_global_accuracy = shifted.accuracy("new-context-after-shift")

        changed_authorities = dict(grown_frontier.authority_hashes)
        changed_authorities["language-reward-policy"] = _sha(
            "growth-changed-reward-policy"
        )
        changed_reward_frontier = ActionFrontier.create(
            grown_frontier.actions,
            action_schema_sha256=grown_frontier.action_schema_sha256,
            context_schema_sha256=grown_frontier.context_schema_sha256,
            authority_hashes=changed_authorities,
        )
        reset, reset_receipt = migrate_language_frontier(
            grown,
            changed_reward_frontier,
            seed=seed + 4_000,
        )
        reward_shift_accuracy = _mean_accuracy(reset, contexts)

        value = np.stack([_value(rng) for _ in range(25)])
        macro_execution = compiler.execute(compiled, value)
        second_execution = second_compiler.execute(second_compiled, value)
        return {
            "base_accuracy": base_accuracy,
            "compiled_constant_discharge": compiled.constant_discharge,
            "compiled_historical_work_released": (
                macro_execution.receipt.historical_work_released
            ),
            "definition_references": compiled.stored_definition_references,
            "discovery_sha256": discovery.sha256,
            "final_accuracy": final_accuracy,
            "episodes_to_promoted_action": (
                growth_episodes
                if episodes_to_promoted_action is None
                else episodes_to_promoted_action
            ),
            "frontier_migration_sha256": migration.sha256,
            "frontier_promotion_sha256": promotion.sha256,
            "macro_expanded_actions": compiled.expanded_primitive_actions,
            "no_memory_accuracy": no_memory_accuracy,
            "promoted_action_initially_abstains": promoted_initially_abstains,
            "promoted_action_learned": promoted_final_exact,
            "overall_accuracy_after_promotion": overall_accuracy_after_promotion,
            "retained_old_action_accuracy": retained_accuracy,
            "revision_sha256": revision.sha256,
            "reward_policy_shift_accuracy": reward_shift_accuracy,
            "reward_policy_shift_reset_count": len(reset_receipt.reset_action_ids),
            "second_generation_binding_count": len(second_bindings),
            "second_generation_constant_discharge": (
                second_compiled.constant_discharge
            ),
            "second_generation_historical_work_released": (
                second_execution.receipt.historical_work_released
            ),
            "second_generation_resolution_sha256": second_resolution.sha256,
            "seed": seed,
            "shifted_context_evidence_retained": (
                shifted_receipt.context_evidence_retained
            ),
            "shifted_context_global_accuracy": shifted_global_accuracy,
        }


def run_language_growth_intelligence_benchmark(
    *,
    seeds: int = 5,
    base_episodes: int = 5_000,
    growth_episodes: int = 5_000,
) -> dict[str, Any]:
    if seeds < 1:
        raise ValueError("seeds must be positive")
    rows = tuple(
        run_language_growth_seed(
            seed,
            base_episodes=base_episodes,
            growth_episodes=growth_episodes,
        )
        for seed in range(seeds)
    )
    aggregate: dict[str, object] = {"seeds": seeds}
    for key, value in rows[0].items():
        if key == "seed" or key.endswith("sha256"):
            continue
        if isinstance(value, bool):
            aggregate[f"all_{key}"] = all(bool(row[key]) for row in rows)
        elif isinstance(value, (int, float)):
            aggregate[key] = float(np.mean([float(row[key]) for row in rows]))
    body = {
        "aggregate": aggregate,
        "base_episodes": base_episodes,
        "growth_episodes": growth_episodes,
        "seeds": list(rows),
    }
    return {
        "body": body,
        "schema": LANGUAGE_GROWTH_INTELLIGENCE_SCHEMA,
        "sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }


__all__ = [
    "DIRECT_CONSEQUENCE_REWARD_POLICY_SHA256",
    "LANGUAGE_GROWTH_INTELLIGENCE_SCHEMA",
    "TERMINAL_EPISODE_VERIFIER_SHA256",
    "run_language_growth_intelligence_benchmark",
    "run_language_growth_seed",
]
