from __future__ import annotations

from fractions import Fraction
import unittest

from immer.cognition.fertig.arithmetic_ir import SolveStatus, solve
from immer.cognition.fertig.signed_event_frontend import (
    FrontendStatus,
    compile_signed_events,
    lex,
)
from immer.cognition.fertig.structural import parse_structural_problem


CASES = {
    8: (
        "It costs $194 per meter to repave a street. Monica's street is 150 "
        "meters long. How much more does it cost to repave Lewis' street, "
        "which is 490 meters long?",
        Fraction(65960),
    ),
    10: (
        "Each pole on a road intersection has 4 street lights. If the number "
        "of poles at each intersection is 6, and the road has 4 intersections, "
        "calculate the total number of functioning street lights if 20 "
        "streetlights from the total number are not working.",
        Fraction(76),
    ),
    13: (
        "A news website publishes an average of 20 political and weather news "
        "articles every day. Its sister company publishes an average of 10 "
        "business news articles daily. Calculate the total number of articles "
        "the two websites published together in February if there are 28 days "
        "in the month.",
        Fraction(840),
    ),
    27: (
        "A tomato vendor decides to switch who he buys his tomatoes for. He "
        "sells 500 tomatoes a day. He used to buy them for $.5 each but he gets "
        "a new vendor who sells them for $.4 each. How much money does he save "
        "a week?",
        Fraction(350),
    ),
    30: (
        "A perfume company is trying to create new scents. They already have 4 "
        "vanilla scents and 8 fruity scents available and they need to decide "
        "which kind of scent to focus on. They decide to focus on whichever "
        "scent sells the most and monitor their number of sales as part of their "
        "research. By the end of the day, they sell 5 of each of the vanilla "
        "scents and 2 of each of the fruity scents available. How many more "
        "vanilla scents sold compared with the fruity scents?",
        Fraction(4),
    ),
    34: (
        'Carrie is planning the caroling schedule. The choir plans to sing "Deck '
        'the Halls" twice and "Jingle Bells" once. If "Deck the Halls" is 150 '
        'seconds long and "Jingle Bells" is 240 seconds long, how long will they '
        "be caroling?",
        Fraction(540),
    ),
}

CALENDAR_CASE = (
    "A Reddit group has 1000 members. If each member posts an average of 3 "
    "posts per day, what's the total number of posts that the group will have "
    "in March?"
)


class SignedEventFrontendTests(unittest.TestCase):
    def test_six_shared_event_families_compile_and_solve_exactly(self) -> None:
        for offset, (question, expected) in CASES.items():
            with self.subTest(offset=offset):
                result = compile_signed_events(question)
                self.assertTrue(result.ok, result.reason)
                assert result.compiled is not None
                self.assertEqual(result.compiled.solution.target_value, expected)
                self.assertTrue(result.compiled.certificate.verified)
                numeric_tokens = [
                    token
                    for token in lex(question)
                    if token.number is not None
                    and token.text[0].isdigit()
                    or token.text.lstrip().startswith("$")
                ]
                self.assertGreaterEqual(
                    len(result.compiled.evidence_projection), len(numeric_tokens)
                )

    def test_structural_entrypoint_accepts_only_complete_signed_programs(self) -> None:
        for offset, (question, expected) in CASES.items():
            with self.subTest(offset=offset):
                parsed = parse_structural_problem(question)
                self.assertTrue(parsed.ok, parsed.reason)
                assert parsed.problem is not None
                solution = solve(parsed.problem)
                self.assertIs(solution.status, SolveStatus.UNIQUE)
                self.assertEqual(solution.target_value, expected)

    def test_domain_renames_preserve_operator_grammar(self) -> None:
        street = CASES[8][0]
        renamed = (
            street.replace("Monica", "Aria")
            .replace("Lewis", "Bram")
            .replace("street", "canal")
            .replace("repave", "restore")
        )
        result = compile_signed_events(renamed)
        self.assertTrue(result.ok, result.reason)
        assert result.compiled is not None
        self.assertEqual(result.compiled.solution.target_value, CASES[8][1])

        categories = (
            CASES[30][0]
            .replace("vanilla", "floral")
            .replace("fruity", "citrus")
            .replace("perfume", "soap")
        )
        result = compile_signed_events(categories)
        self.assertTrue(result.ok, result.reason)
        assert result.compiled is not None
        self.assertEqual(result.compiled.solution.target_value, CASES[30][1])

    def test_operator_and_state_mutations_fail_closed(self) -> None:
        mutations = (
            CASES[8][0].replace("per meter", "for material"),
            CASES[10][0].replace("are not working", "are working"),
            CASES[13][0].replace("published together", "published separately"),
            CASES[27][0].replace("a week", "eventually"),
            CASES[30][0].replace("5 of each", "5 near"),
            CASES[34][0].replace("twice", "often"),
        )
        for question in mutations:
            with self.subTest(question=question):
                self.assertFalse(compile_signed_events(question).ok)

    def test_extra_or_duplicated_numeric_surfaces_fail_closed(self) -> None:
        for question in (
            CASES[8][0] + " Reference 99.",
            CASES[8][0].replace(
                "Monica's street is 150 meters long.",
                "Monica's street is 150 meters long and half the crew rested.",
            ),
            CASES[10][0].replace(
                "calculate the total number",
                "2 supervisors are present, calculate the total number",
            ),
            CASES[10][0].replace(
                "calculate the total number",
                "each supervisor checks 2 gauges, calculate the total number",
            ),
            CASES[27][0].replace("500 tomatoes", "500 tomatoes and 500 tomatoes"),
        ):
            with self.subTest(question=question):
                result = compile_signed_events(question)
                self.assertFalse(result.ok)
                self.assertIn(
                    result.status,
                    {FrontendStatus.UNSUPPORTED, FrontendStatus.INVALID},
                )

    def test_reordered_duration_bindings_preserve_item_identity(self) -> None:
        question = CASES[34][0].replace(
            'If "Deck the Halls" is 150 seconds long and "Jingle Bells" is 240',
            'If "Jingle Bells" is 240 seconds long and "Deck the Halls" is 150',
        )
        result = compile_signed_events(question)
        self.assertTrue(result.ok, result.reason)
        assert result.compiled is not None
        self.assertEqual(result.compiled.solution.target_value, Fraction(540))

    def test_calendar_daily_rate_compiles_with_exact_month_basis(self) -> None:
        for month, days in (("March", 31), ("April", 30), ("December", 31)):
            with self.subTest(month=month):
                question = CALENDAR_CASE.replace("March", month)
                result = compile_signed_events(question)
                self.assertTrue(result.ok, result.reason)
                self.assertEqual(result.family, "calendar_daily_total")
                assert result.compiled is not None
                self.assertEqual(
                    result.compiled.solution.target_value,
                    Fraction(1000 * 3 * days),
                )
                self.assertTrue(result.compiled.certificate.verified)

                parsed = parse_structural_problem(question)
                self.assertTrue(parsed.ok, parsed.reason)
                assert parsed.problem is not None
                self.assertEqual(solve(parsed.problem).target_value, 1000 * 3 * days)

    def test_calendar_daily_rate_rejects_unproven_or_noisy_bindings(self) -> None:
        mutations = (
            CALENDAR_CASE.replace("March", "February"),
            CALENDAR_CASE.replace("March", "March or April"),
            CALENDAR_CASE.replace("each member", "each moderator"),
            CALENDAR_CASE.replace(
                "has 1000 members", "has 1000 members and 5 moderators"
            ),
            CALENDAR_CASE.replace("in March", "in an unspecified month"),
        )
        for question in mutations:
            with self.subTest(question=question):
                self.assertFalse(compile_signed_events(question).ok)

    def test_ambiguous_comparison_orientation_is_rejected(self) -> None:
        question = CASES[8][0].replace("Lewis' street", "the other street")
        result = compile_signed_events(question)
        self.assertFalse(result.ok)
        self.assertIs(result.status, FrontendStatus.AMBIGUOUS)

    def test_unrelated_and_deferred_families_do_not_match(self) -> None:
        deferred = (
            "Sarah has a 20 meter rope and a new rope costs $1.5 a meter. How "
            "much money remains after buying an unstated length?",
            "Tasha earned $80 and mowed three lawns. How much came from lemonade?",
            "What is the capital of France?",
        )
        for question in deferred:
            with self.subTest(question=question):
                result = compile_signed_events(question)
                self.assertFalse(result.ok)
                self.assertFalse(result.family_matched)


if __name__ == "__main__":
    unittest.main()
