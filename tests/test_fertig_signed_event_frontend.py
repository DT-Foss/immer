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

SECOND_CASES = {
    22: (
        "Ron is fed up with the pothole in front of his house. If it doesn't "
        "get fixed, it's going to do $450 worth of damage to his car. "
        "Unfortunately, the city council refuses to fix it, and will fine Ron "
        "$120 for unauthorized road maintenance if he fixes it himself. Ron "
        "will also have to buy 3 buckets of asphalt that each cost $25. How "
        "much money does Ron save by fixing the pothole?",
        Fraction(255),
        "avoided_cost_transaction",
    ),
    25: (
        "Mark decides to buy packs of old magic cards and open them to sell. "
        "He buys 3 packs for $1800 each. He gets 1 card that is worth $4000 "
        "and another card worth $1000. There are 30 more cards worth an "
        "average of $50 each. How much money profit did he make?",
        Fraction(1100),
        "profit_contribution_ledger",
    ),
    26: (
        "Jake's family wants to compare the cost of the two different "
        "amusement parks. The first amusement park has a $26 fee for each "
        "adult and a $12 fee for each child; while the second amusement park "
        "has a $14 fee for each adult and $10 for each child. If there are 2 "
        "adults and 2 children in their family, how much will they be able to "
        "save if they choose the second amusement park over the first?",
        Fraction(28),
        "alternative_cost_savings",
    ),
    37: (
        "Twenty students are working together to raise money for a charity. "
        "Each earns the same amount. The charity raises a total of $175,000. "
        "$50,000 comes from organizations and the rest from the students. How "
        "much did each student raise?",
        Fraction(6250),
        "equal_share_residual",
    ),
    53: (
        "Ellen is on a diet. She eats two carrots, a salad, and a yogurt every "
        "day. The salad costs her $6, while the yogurt is half the price. How "
        "much does Ellen pay for one carrot every day when in total she pays "
        "$11 for her goods?",
        Fraction(1),
        "unit_cost_residual",
    ),
    61: (
        "When Sophie watches her nephew, she gets out a variety of toys for "
        "him. The bag of building blocks has 31 blocks in it. The bin of "
        "stuffed animals has 8 stuffed animals inside. The tower of stacking "
        "rings has 9 multicolored rings on it. Sophie recently bought a tube "
        "of bouncy balls, bringing her total number of toys for her nephew up "
        "to 62. How many bouncy balls came in the tube?",
        Fraction(14),
        "inventory_total_residual",
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


class SignedEventContributionFrontendTests(unittest.TestCase):
    def test_second_six_families_compile_with_exact_certificates(self) -> None:
        for offset, (question, expected, family) in SECOND_CASES.items():
            with self.subTest(offset=offset):
                result = compile_signed_events(question)
                self.assertTrue(result.ok, result.reason)
                self.assertEqual(result.family, family)
                assert result.compiled is not None
                self.assertIs(result.compiled.solution.status, SolveStatus.UNIQUE)
                self.assertEqual(result.compiled.solution.target_value, expected)
                self.assertTrue(result.compiled.certificate.verified)
                projection = result.compiled.evidence_projection
                self.assertTrue(projection)
                self.assertEqual(
                    len({row.evidence.evidence_id for row in projection}),
                    len(projection),
                )
                for row in projection:
                    span = row.evidence.span
                    self.assertEqual(span.source, question)
                    self.assertTrue(question[span.start : span.end])

    def test_second_six_support_domain_and_owner_renames(self) -> None:
        variants = (
            SECOND_CASES[22][0]
            .replace("Ron", "Miro")
            .replace("pothole", "culvert")
            .replace("asphalt", "gravel"),
            SECOND_CASES[25][0]
            .replace("Mark", "Niko")
            .replace("packs", "bundles")
            .replace("pack", "bundle")
            .replace("cards", "stamps")
            .replace("card", "stamp"),
            SECOND_CASES[53][0]
            .replace("Ellen", "Mara")
            .replace("carrots", "radishes")
            .replace("carrot", "radish")
            .replace("salad", "soup")
            .replace("yogurt", "juice"),
            SECOND_CASES[61][0]
            .replace("Sophie", "Talia")
            .replace("toys", "trinkets")
            .replace("blocks", "tiles")
            .replace("balls", "cubes"),
        )
        expected = (Fraction(255), Fraction(1100), Fraction(1), Fraction(14))
        for question, answer in zip(variants, expected, strict=True):
            with self.subTest(question=question):
                result = compile_signed_events(question)
                self.assertTrue(result.ok, result.reason)
                assert result.compiled is not None
                self.assertEqual(result.compiled.solution.target_value, answer)

    def test_declarative_reordering_preserves_bound_ledgers(self) -> None:
        variants = (
            (
                "Ron is fed up with the pothole in front of his house. Ron will "
                "also have to buy 3 buckets of asphalt that each cost $25. "
                "Unfortunately, the city council refuses to fix it, and will "
                "fine Ron $120 for unauthorized road maintenance if he fixes it "
                "himself. If it doesn't get fixed, it's going to do $450 worth "
                "of damage to his car. How much money does Ron save by fixing "
                "the pothole?"
            ),
            (
                "Mark decides to buy packs of old magic cards and open them to "
                "sell. There are 30 more cards worth an average of $50 each. He "
                "gets 1 card that is worth $4000 and another card worth $1000. "
                "He buys 3 packs for $1800 each. How much money profit did he make?"
            ),
            (
                "Twenty students are working together to raise money for a "
                "charity. $50,000 comes from organizations and the rest from the "
                "students. The charity raises a total of $175,000. Each earns "
                "the same amount. How much did each student raise?"
            ),
            (
                "When Sophie watches her nephew, she gets out a variety of toys "
                "for him. The tower of stacking rings has 9 multicolored rings "
                "on it. The bag of building blocks has 31 blocks in it. The bin "
                "of stuffed animals has 8 stuffed animals inside. Sophie "
                "recently bought a tube of bouncy balls, bringing her total "
                "number of toys for her nephew up to 62. How many bouncy balls "
                "came in the tube?"
            ),
        )
        expected = (Fraction(255), Fraction(1100), Fraction(6250), Fraction(14))
        for question, answer in zip(variants, expected, strict=True):
            with self.subTest(question=question):
                result = compile_signed_events(question)
                self.assertTrue(result.ok, result.reason)
                assert result.compiled is not None
                self.assertEqual(result.compiled.solution.target_value, answer)

    def test_operator_direction_and_scope_mutations_abstain(self) -> None:
        mutations = (
            SECOND_CASES[22][0].replace("that each cost", "that cost"),
            SECOND_CASES[22][0].replace("fixing the pothole", "fixing the roof"),
            SECOND_CASES[25][0].replace("30 more cards", "30 more coins"),
            SECOND_CASES[25][0].replace("$1800 each", "$1800 total"),
            SECOND_CASES[26][0].replace("2 children", "2 teachers"),
            SECOND_CASES[26][0].replace(
                "choose the second amusement park over the first",
                "choose the first amusement park over the second",
            ),
            SECOND_CASES[37][0].replace(
                "Each earns the same", "Each earns a different"
            ),
            SECOND_CASES[37][0].replace("each student raise", "each teacher raise"),
            SECOND_CASES[53][0].replace("half the price", "double the price"),
            SECOND_CASES[53][0].replace("two carrots", "two apples"),
            SECOND_CASES[61][0].replace(
                "total number of toys", "total number of tools"
            ),
            SECOND_CASES[61][0].replace(
                "How many bouncy balls", "How many glass beads"
            ),
            SECOND_CASES[61][0].replace("has 8 stuffed animals", "has 8 wooden boxes"),
        )
        for question in mutations:
            with self.subTest(question=question):
                self.assertFalse(compile_signed_events(question).ok)

    def test_unrelated_numeric_surfaces_fail_evidence_closure(self) -> None:
        for offset, (question, _, _) in SECOND_CASES.items():
            with self.subTest(offset=offset):
                noisy = question.replace("?", " with reference number 99?")
                result = compile_signed_events(noisy)
                self.assertFalse(result.ok)
                self.assertIn(
                    result.status,
                    {FrontendStatus.UNSUPPORTED, FrontendStatus.INVALID},
                )

    def test_cross_scope_contributions_never_enter_the_ledger(self) -> None:
        repair = (
            SECOND_CASES[22][0]
            .replace(
                "will fine Ron $120 for unauthorized road maintenance if he fixes it himself",
                "will fine Ron $120 for parking in the wrong place",
            )
            .replace(
                "3 buckets of asphalt that each cost $25",
                "3 buckets of asphalt that each cost $25 for his garden path",
            )
        )
        foreign_actor = (
            SECOND_CASES[25][0]
            .replace("He gets 1 card", "Luke gets 1 card")
            .replace("did he make", "did Mark make")
        )
        mixed_actor_pronouns = SECOND_CASES[25][0].replace(
            "He gets 1 card", "She gets 1 card"
        )
        foreign_damage = SECOND_CASES[22][0].replace(
            "If it doesn't get fixed", "If his roof doesn't get fixed"
        )
        foreign_damage_car = SECOND_CASES[22][0].replace(
            "If it doesn't get fixed", "If the car doesn't get fixed"
        )
        foreign_inventory = SECOND_CASES[61][0].replace(
            "Sophie recently bought",
            "The shelf of toy cars has 2 cars on it. Sophie recently bought",
        )
        foreign_share = SECOND_CASES[37][0].replace(
            "How much did each student raise?",
            "Another club says each earns the same amount too. How much did "
            "each student raise?",
        )
        foreign_price = SECOND_CASES[53][0].replace(
            "The salad costs her", "Ben says the salad costs her"
        )
        foreign_price_anchor = SECOND_CASES[53][0].replace(
            "half the price", "half the smoothie price"
        )
        foreign_unit_cost_owner = SECOND_CASES[53][0].replace(
            "Ellen is on a diet", "Mara is on a diet"
        )
        foreign_total_owner = SECOND_CASES[61][0].replace(
            "Sophie recently bought", "Mia recently bought"
        )
        foreign_total_possessive = SECOND_CASES[61][0].replace(
            "bringing her total", "bringing Mia's total"
        )
        for question in (
            repair,
            foreign_actor,
            mixed_actor_pronouns,
            foreign_damage,
            foreign_damage_car,
            foreign_inventory,
            foreign_share,
            foreign_price,
            foreign_price_anchor,
            foreign_unit_cost_owner,
            foreign_total_owner,
            foreign_total_possessive,
        ):
            with self.subTest(question=question):
                self.assertFalse(compile_signed_events(question).ok)

    def test_two_and_twenty_are_numeric_only_inside_bound_productions(self) -> None:
        unrelated = lex("Two notes and twenty labels are on a desk.")
        self.assertTrue(
            all(
                token.number is None
                for token in unrelated
                if token.norm in {"two", "twenty"}
            )
        )
        for offset, surface, value in ((37, "Twenty", 20), (53, "two", 2)):
            result = compile_signed_events(SECOND_CASES[offset][0])
            self.assertTrue(result.ok, result.reason)
            assert result.compiled is not None
            matches = [
                row.evidence
                for row in result.compiled.evidence_projection
                if row.evidence.span.source[
                    row.evidence.span.start : row.evidence.span.end
                ]
                == surface
            ]
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0].value, value)


if __name__ == "__main__":
    unittest.main()
