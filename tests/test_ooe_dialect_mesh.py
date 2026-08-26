from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from immer.runtimes.ooe.compute_crystals import ComputeCrystal, ComputeCrystalBank
from immer.runtimes.ooe.dialect_mesh import (
    DialectMeshIntegrityError,
    DialectTranslationReceipt,
    PortableWordLocalizationReceipt,
    PortableWordProgram,
    align_dialects,
    localize_portable_program,
    portable_program_from_discovery,
    translate_words,
)
from immer.runtimes.ooe.executable_lexicon import (
    ExecutableLexiconConflictError,
    ExecutableLexiconState,
    ExecutableWordCompiler,
    ExecutableWordDefinition,
    PrimitiveWordBinding,
)
from immer.runtimes.ooe.language_bridge import LanguageMacroDiscoveryReceipt
from immer.runtimes.ooe.markov_language import (
    ActionBinding,
    ActionFrontier,
    ConsequenceMarkovLanguage,
    LanguageSnapshot,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _language(
    frontier: ActionFrontier,
    vocabulary: tuple[str, ...],
    word_indices: tuple[int, ...],
    *,
    seed: int,
) -> ConsequenceMarkovLanguage:
    language = ConsequenceMarkovLanguage(frontier, vocabulary, seed=seed)
    language.sender_q.fill(-1.0)
    language.receiver_q.fill(-1.0)
    language.receiver_visits.fill(0)
    for action_index, word_index in enumerate(word_indices):
        language.sender_q[action_index, word_index] = 1.0
        language.receiver_q[word_index, action_index] = 1.0
        language.receiver_visits[word_index, action_index] = 20
    return language


class DialectMeshTests(unittest.TestCase):
    def test_independent_surface_dialects_translate_by_exact_action_semantics(
        self,
    ) -> None:
        frontier = ActionFrontier.create(
            tuple(
                ActionBinding(f"action-{index}", "crystal", _sha(f"crystal-{index}"))
                for index in range(3)
            ),
            action_schema_sha256=_sha("dialect-actions"),
            context_schema_sha256=_sha("dialect-context"),
        )
        source = _language(
            frontier,
            ("source-a", "source-b", "source-c", "source-unused"),
            (2, 0, 1),
            seed=1,
        )
        target = _language(
            frontier,
            ("target-a", "target-b", "target-c", "target-unused"),
            (1, 2, 0),
            seed=2,
        )
        source_snapshot = LanguageSnapshot.freeze(source, contexts=("source-context",))
        target_snapshot = LanguageSnapshot.freeze(target, contexts=("target-context",))
        receipt = align_dialects(
            source_snapshot,
            target_snapshot,
            frontier,
            frontier,
            source_context_id="source-context",
            target_context_id="target-context",
        )
        self.assertEqual(
            DialectTranslationReceipt.from_bytes(receipt.to_bytes()), receipt
        )
        source_words = tuple(
            source_snapshot.encode_action(action, "source-context")
            for action in ("action-2", "action-0", "action-1", "action-2")
        )
        self.assertNotIn(None, source_words)
        translated = translate_words(
            tuple(str(word) for word in source_words),
            receipt,
            source_snapshot=source_snapshot,
            target_snapshot=target_snapshot,
            source_frontier=frontier,
            target_frontier=frontier,
        )
        self.assertIsNotNone(translated)
        assert translated is not None
        decoded = tuple(
            target_snapshot.decode_word(word, "target-context") for word in translated
        )
        self.assertEqual(
            decoded,
            ("action-2", "action-0", "action-1", "action-2"),
        )
        self.assertIsNone(
            translate_words(
                ("source-unused",),
                receipt,
                source_snapshot=source_snapshot,
                target_snapshot=target_snapshot,
                source_frontier=frontier,
                target_frontier=frontier,
            )
        )
        first = receipt.word_translations[0]
        forged = replace(
            receipt,
            word_translations=(
                (first[0], "target-unused", first[2]),
                *receipt.word_translations[1:],
            ),
        )
        with self.assertRaisesRegex(DialectMeshIntegrityError, "bound snapshots"):
            translate_words(
                (first[0],),
                forged,
                source_snapshot=source_snapshot,
                target_snapshot=target_snapshot,
                source_frontier=frontier,
                target_frontier=frontier,
            )

    def test_portable_program_localizes_and_compiles_in_another_dialect(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = ComputeCrystalBank(Path(temporary) / "compute")
            crystals = (
                ComputeCrystal.affine([[1.0]], [1.0]),
                ComputeCrystal.affine([[-1.0]], [0.0]),
                ComputeCrystal.affine([[2.0]], [3.0]),
            )
            for crystal in crystals:
                bank.publish_crystal(crystal)
            frontier = ActionFrontier.create(
                tuple(
                    ActionBinding(f"action-{index}", "crystal", crystal.sha256)
                    for index, crystal in enumerate(crystals)
                ),
                action_schema_sha256=_sha("portable-actions"),
                context_schema_sha256=_sha("portable-context"),
                authority_hashes={"verifier": _sha("portable-verifier")},
            )
            source = _language(
                frontier,
                ("s0", "s1", "s2", "s3"),
                (2, 0, 1),
                seed=3,
            )
            target = _language(
                frontier,
                ("t0", "t1", "t2", "t3"),
                (1, 2, 0),
                seed=4,
            )
            source_snapshot = LanguageSnapshot.freeze(
                source, contexts=("source-context",)
            )
            target_base = LanguageSnapshot.freeze(target, contexts=("target-context",))
            target_rows = dict(target_base.context_word_actions)["target-context"]
            target_senders = dict(target_base.context_action_words)["target-context"]
            target_snapshot = LanguageSnapshot(
                frontier_sha256=frontier.sha256,
                learner_state_sha256=target_base.learner_state_sha256,
                vocabulary=target_base.vocabulary,
                sender_action_words=(),
                global_word_actions=(),
                context_word_actions=(
                    (
                        "other-context",
                        (("t0", "action-0"), ("t1", "action-1"), ("t2", "action-2")),
                    ),
                    ("target-context", target_rows),
                ),
                context_action_words=(
                    (
                        "other-context",
                        (("action-0", "t0"), ("action-1", "t1"), ("action-2", "t2")),
                    ),
                    ("target-context", target_senders),
                ),
            )
            self.assertEqual(target_snapshot.global_word_actions, ())
            source_words = tuple(
                source_snapshot.encode_action(action, "source-context")
                for action in ("action-0", "action-1", "action-2", "action-0")
            )
            self.assertNotIn(None, source_words)
            definition = ExecutableWordDefinition.create(
                "source-macro",
                tuple(str(word) for word in source_words),
                language_snapshot_sha256=source_snapshot.sha256,
                frontier_sha256=frontier.sha256,
                authority_hashes=frontier.authority_hashes,
            )
            supports = (
                _sha("trajectory-1"),
                _sha("trajectory-2"),
                _sha("trajectory-3"),
            )
            discovery = LanguageMacroDiscoveryReceipt(
                language_snapshot_sha256=source_snapshot.sha256,
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
            portable = portable_program_from_discovery(
                discovery,
                definition,
                source_snapshot,
                frontier,
            )
            self.assertEqual(
                PortableWordProgram.from_bytes(portable.to_bytes()), portable
            )
            forged_portable = replace(
                portable,
                source_word_sequence=tuple(reversed(portable.source_word_sequence)),
            )
            with self.assertRaisesRegex(
                DialectMeshIntegrityError, "source discovery evidence"
            ):
                localize_portable_program(
                    forged_portable,
                    target_snapshot,
                    frontier,
                    source_discovery=discovery,
                    source_definition=definition,
                    source_snapshot=source_snapshot,
                    source_frontier=frontier,
                    target_context_id="target-context",
                )
            localized, receipt = localize_portable_program(
                portable,
                target_snapshot,
                frontier,
                source_discovery=discovery,
                source_definition=definition,
                source_snapshot=source_snapshot,
                source_frontier=frontier,
                target_context_id="target-context",
            )
            self.assertEqual(
                PortableWordLocalizationReceipt.from_bytes(receipt.to_bytes()),
                receipt,
            )
            self.assertEqual(
                localized.child_word_ids,
                tuple(
                    str(target_snapshot.encode_action(action, "target-context"))
                    for action in portable.action_sequence
                ),
            )
            self.assertEqual(localized.context_id, "target-context")
            primitives = tuple(
                PrimitiveWordBinding(word, kind, digest)
                for word, (kind, digest) in sorted(
                    target_snapshot.primitive_artifacts(
                        frontier, context_id="target-context"
                    ).items()
                )
            )
            state = ExecutableLexiconState.initial(
                primitives,
                language_snapshot_sha256=target_snapshot.sha256,
                frontier_sha256=frontier.sha256,
                authority_hashes=frontier.authority_hashes,
                context_id="target-context",
            ).with_definition(localized)
            compiler = ExecutableWordCompiler(state, bank)
            compiled = compiler.compile(localized.new_word_id)
            value = np.array([[2.0], [-3.0], [7.0]], dtype=np.float64)
            execution = compiler.execute(compiled, value)
            expected = value
            for action in portable.action_sequence:
                expected = crystals[int(action.removeprefix("action-"))].apply(expected)
            np.testing.assert_array_equal(execution.output, expected)

            with self.assertRaisesRegex(
                ExecutableLexiconConflictError, "contract is stale"
            ):
                ExecutableLexiconState.initial(
                    primitives,
                    language_snapshot_sha256=target_snapshot.sha256,
                    frontier_sha256=frontier.sha256,
                    authority_hashes=frontier.authority_hashes,
                ).with_definition(localized)

            changed_authorities = ActionFrontier.create(
                frontier.actions,
                action_schema_sha256=frontier.action_schema_sha256,
                context_schema_sha256=frontier.context_schema_sha256,
                authority_hashes={"verifier": _sha("different-verifier")},
            )
            changed_authority_snapshot = LanguageSnapshot(
                frontier_sha256=changed_authorities.sha256,
                learner_state_sha256=_sha("changed-authority-language"),
                vocabulary=target_snapshot.vocabulary,
                sender_action_words=target_snapshot.sender_action_words,
                global_word_actions=target_snapshot.global_word_actions,
                context_word_actions=target_snapshot.context_word_actions,
                context_action_words=target_snapshot.context_action_words,
            )
            with self.assertRaisesRegex(
                DialectMeshIntegrityError, "verifier authorities"
            ):
                localize_portable_program(
                    portable,
                    changed_authority_snapshot,
                    changed_authorities,
                    source_discovery=discovery,
                    source_definition=definition,
                    source_snapshot=source_snapshot,
                    source_frontier=frontier,
                    target_context_id="target-context",
                )

            changed = ActionFrontier.create(
                (
                    frontier.actions[0],
                    ActionBinding("action-1", "crystal", _sha("changed-crystal")),
                    frontier.actions[2],
                ),
                action_schema_sha256=_sha("changed-portable-actions"),
                context_schema_sha256=frontier.context_schema_sha256,
                authority_hashes=dict(frontier.authority_hashes),
            )
            changed_target = _language(
                changed,
                ("u0", "u1", "u2", "u3"),
                (0, 1, 2),
                seed=5,
            )
            changed_snapshot = LanguageSnapshot.freeze(
                changed_target, contexts=("changed-context",)
            )
            with self.assertRaisesRegex(DialectMeshIntegrityError, "binding changed"):
                localize_portable_program(
                    portable,
                    changed_snapshot,
                    changed,
                    source_discovery=discovery,
                    source_definition=definition,
                    source_snapshot=source_snapshot,
                    source_frontier=frontier,
                    target_context_id="changed-context",
                )


if __name__ == "__main__":
    unittest.main()
