from __future__ import annotations

import unittest

from immer.runtimes.qwen3_8.markov_composition import (
    CompositionBounds,
    ConfirmedTokenEpisode,
    LiteralAtom,
    MarkovCompositionProgram,
    RelativeCopyAtom,
    derive_programs,
    match_programs,
)


class MarkovCompositionTests(unittest.TestCase):
    def test_one_slot_executes_an_unseen_token_binding(self) -> None:
        episodes = (
            ConfirmedTokenEpisode((10, 11, 20), (20,)),
            ConfirmedTokenEpisode((10, 11, 21), (21,)),
        )

        programs = derive_programs(episodes)

        self.assertTrue(programs)
        self.assertEqual(match_programs(programs, (10, 11, 22)), (22,))
        self.assertTrue(all(row.support == row.total == 2 for row in programs))
        self.assertTrue(all(row.distinct_bindings == 2 for row in programs))

    def test_two_slots_and_literal_delimiter_compose(self) -> None:
        episodes = (
            ConfirmedTokenEpisode((1, 20, 2, 30), (20, 9, 30)),
            ConfirmedTokenEpisode((1, 21, 2, 31), (21, 9, 31)),
        )

        programs = derive_programs(episodes)
        result = match_programs(programs, (1, 22, 2, 32))

        self.assertEqual(result, (22, 9, 32))
        self.assertTrue(
            any(
                sum(isinstance(atom, RelativeCopyAtom) for atom in row.atoms) == 2
                and any(
                    isinstance(atom, LiteralAtom) and atom.token_ids == (9,)
                    for atom in row.atoms
                )
                for row in programs
            )
        )

    def test_reordered_slots_compose(self) -> None:
        episodes = (
            ConfirmedTokenEpisode((1, 20, 2, 30), (30, 9, 20)),
            ConfirmedTokenEpisode((1, 21, 2, 31), (31, 9, 21)),
        )

        programs = derive_programs(episodes)

        self.assertEqual(match_programs(programs, (1, 22, 2, 32)), (32, 9, 22))

    def test_relative_suffix_supports_different_prompt_lengths(self) -> None:
        episodes = (
            ConfirmedTokenEpisode((70, 1, 20, 2, 30), (20, 9, 30)),
            ConfirmedTokenEpisode((80, 81, 1, 21, 2, 31), (21, 9, 31)),
        )

        programs = derive_programs(episodes)
        prompt = (90, 91, 92, 1, 22, 2, 32)

        self.assertEqual(match_programs(programs, prompt), (22, 9, 32))
        self.assertTrue(any(row.context_width == 4 for row in programs))

    def test_partial_confirmed_output_resumes_at_exact_offset(self) -> None:
        programs = derive_programs(
            (
                ConfirmedTokenEpisode((1, 20, 2, 30), (20, 9, 30)),
                ConfirmedTokenEpisode((1, 21, 2, 31), (21, 9, 31)),
            )
        )
        prompt = (1, 22, 2, 32)

        self.assertEqual(
            match_programs(programs, prompt, (22, 9), limit=1),
            (32,),
        )
        self.assertIsNone(match_programs(programs, prompt, (22, 8)))
        self.assertEqual(match_programs(programs, prompt, (22, 9, 32)), ())

    def test_duplicate_conflict_ambiguity_and_context_mismatch_abstain(self) -> None:
        duplicate = (
            ConfirmedTokenEpisode((10, 11, 20), (20,)),
            ConfirmedTokenEpisode((10, 11, 20), (20,)),
        )
        self.assertEqual(derive_programs(duplicate), ())

        conflict = (
            ConfirmedTokenEpisode((10, 11, 20), (20,)),
            ConfirmedTokenEpisode((10, 11, 21), (21,)),
            ConfirmedTokenEpisode((10, 11, 22), (99,)),
        )
        self.assertEqual(derive_programs(conflict), ())

        ambiguous = (
            ConfirmedTokenEpisode((10, 20, 20, 11), (20, 20)),
            ConfirmedTokenEpisode((10, 21, 21, 11), (21, 21)),
        )
        ambiguous_programs = derive_programs(ambiguous)
        self.assertTrue(ambiguous_programs)
        self.assertEqual(
            match_programs(ambiguous_programs, (10, 22, 22, 11)),
            (22, 22),
        )
        self.assertIsNone(match_programs(ambiguous_programs, (10, 22, 23, 11)))

        programs = derive_programs(
            (
                ConfirmedTokenEpisode((10, 11, 20), (20,)),
                ConfirmedTokenEpisode((10, 11, 21), (21,)),
            )
        )
        self.assertIsNone(match_programs(programs, (10, 12, 22)))

    def test_every_variable_prompt_position_must_be_consumed(self) -> None:
        episodes = (
            ConfirmedTokenEpisode((10, 20, 30, 11), (20,)),
            ConfirmedTokenEpisode((10, 21, 31, 11), (21,)),
        )
        self.assertEqual(derive_programs(episodes), ())

    def test_derivation_order_and_candidate_bound_are_deterministic(self) -> None:
        episodes = tuple(
            ConfirmedTokenEpisode(
                (1, 20 + index, 2, 40 + index), (20 + index, 9, 40 + index)
            )
            for index in range(8)
        )
        bounds = CompositionBounds(max_candidates=3)

        first = derive_programs(episodes, bounds)
        second = derive_programs(tuple(reversed(episodes)), bounds)

        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 3)
        self.assertEqual(match_programs(first, (1, 99, 2, 109)), (99, 9, 109))

    def test_program_dataclass_executes_only_guarded_relative_spans(self) -> None:
        program = MarkovCompositionProgram(
            context_width=4,
            guards=((-4, 1), (-2, 2)),
            atoms=(
                RelativeCopyAtom(-3, 1),
                LiteralAtom((9,)),
                RelativeCopyAtom(-1, 1),
            ),
            support=2,
            total=2,
            distinct_bindings=2,
        )
        self.assertEqual(program.execute((7, 1, 20, 2, 30)), (20, 9, 30))
        self.assertIsNone(program.execute((7, 8, 20, 2, 30)))


if __name__ == "__main__":
    unittest.main()
