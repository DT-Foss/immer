from __future__ import annotations

from fractions import Fraction
import unittest

from immer.cognition.fertig.arithmetic_ir import SolveStatus, solve
from immer.cognition.fertig.signed_event_frontend import compile_signed_events
from immer.cognition.fertig.structural import parse_structural_problem

VALUE_TRANSITION = (
    "Josh decides to try flipping a house. He buys a house for $80,000 and "
    "then puts in $50,000 in repairs. This increased the value of the house "
    "by 150%. How much profit did he make?"
)

REPEATED_UNIT_LEDGER = (
    "Toula went to the bakery and bought various types of pastries. She bought "
    "3 dozen donuts which cost $68 per dozen, 2 dozen mini cupcakes which cost "
    "$80 per dozen, and 6 dozen mini cheesecakes for $55 per dozen. How much "
    "was the total cost?"
)

TWO_PART_PARTITION = (
    "Gretchen has 110 coins. There are 30 more gold coins than silver coins. "
    "How many gold coins does Gretchen have?"
)

TWO_LINK_SCALE = (
    "Brandon's iPhone is four times as old as Ben's iPhone. Ben's iPhone is "
    "two times older than Suzy's iPhone. If Suzy’s iPhone is 1 year old, how "
    "old is Brandon’s iPhone?"
)

AUDIT_SCHEDULE_CASES = (
    (
        "On a particular week, a tow truck pulled ten cars for each of the "
        "first three days and then four fewer cars on each of the remaining "
        "days of the week. Calculate the total number of cars it towed that "
        "week.",
        Fraction(54),
        "closed_week_complement_schedule",
    ),
    (
        "Chase and Rider can ride their bikes thrice a day for 5 days; but on "
        "two other days, they ride twice the times they do on usual days. How "
        "many times do they ride their bikes a week?",
        Fraction(27),
        "closed_disjoint_week_schedule",
    ),
    (
        "Buford writes many checks every year. Once per month he writes a check "
        "to pay the electric bill. He also writes a check every month for the "
        "gas bill. Twice per month he writes a check to the church. And "
        "quarterly, he writes a check to the pest and lawn service. How many "
        "checks does Buford write per year?",
        Fraction(52),
        "calendar_frequency_ledger",
    ),
    (
        "Jen works for 7.5 hours a day 6 days a week. Her hourly rate is $1.5. "
        "Jen also receives an additional $10 if she has complete attendance. "
        "Suppose Jen did not incur any absences for April, and there are exactly "
        "4 weeks in April, how much will she receive?",
        Fraction(280),
        "explicit_weekly_pay_schedule",
    ),
    (
        "Pauline visits her favorite local museum three times a year. The cost "
        "of one visit is $2. After 5 years, the cost of one visit has increased "
        "by 150%, but Pauline decided not to give up any visit and continued to "
        "go to the museum for 3 more years. How much did Pauline spend on all "
        "visits to the museum?",
        Fraction(75),
        "closed_piecewise_period_cost",
    ),
    (
        "Ara joined the school basketball team four years ago. She has been "
        "playing 40 games every year. If her score for every game is 21 points, "
        "calculate the total number of points she has scored in the four years.",
        Fraction(3360),
        "explicit_period_score_total",
    ),
    (
        "Tim spends 6 hours each day at work answering phones. It takes him 15 "
        "minutes to deal with a call. How many calls does he deal with during "
        "his 5 day work week?",
        Fraction(120),
        "canonical_duration_rate_conversion",
    ),
    (
        "A company produces chocolate in bars. In one day, it can produce 5000 "
        "bars. The company sells all the produced bars for $2 per bar. How much "
        "money will the company receive for selling produced chocolate bars "
        "during two weeks?",
        Fraction(140000),
        "canonical_weekly_sales_total",
    ),
    (
        "Josh runs a car shop and services 3 cars a day. He is open every day "
        "of the week except Sunday and Wednesday. He gets paid $4 per car. How "
        "much does he make in 2 weeks?",
        Fraction(120),
        "explicit_weekday_exception_schedule",
    ),
)


class ClosedScheduleAlgebraTests(unittest.TestCase):
    def test_four_question_only_derivations_are_exact_and_certified(self) -> None:
        cases = (
            (
                VALUE_TRANSITION,
                Fraction(70000),
                "closed_value_transition_profit",
            ),
            (
                REPEATED_UNIT_LEDGER,
                Fraction(694),
                "closed_repeated_unit_price_ledger",
            ),
            (
                TWO_PART_PARTITION,
                Fraction(70),
                "closed_affine_two_part_partition",
            ),
            (TWO_LINK_SCALE, Fraction(8), "typed_two_link_scale_chain"),
        )
        for question, expected, family in cases:
            with self.subTest(family=family):
                result = compile_signed_events(question)
                self.assertTrue(result.ok, result.reason)
                self.assertEqual(result.family, family)
                assert result.compiled is not None
                self.assertIs(result.compiled.solution.status, SolveStatus.UNIQUE)
                self.assertEqual(result.compiled.solution.target_value, expected)
                self.assertTrue(result.compiled.certificate.verified)
                evidence = result.compiled.evidence_projection
                self.assertEqual(
                    len(evidence),
                    len({row.evidence.evidence_id for row in evidence}),
                )
                parsed = parse_structural_problem(question)
                self.assertTrue(parsed.ok, parsed.reason)
                assert parsed.problem is not None
                self.assertEqual(solve(parsed.problem).target_value, expected)

    def test_names_assets_items_and_categories_are_not_memorized(self) -> None:
        variants = (
            (
                VALUE_TRANSITION.replace("Josh", "Mira")
                .replace(" he ", " she ")
                .replace(" He ", " She ")
                .replace("house", "condo"),
                Fraction(70000),
            ),
            (
                REPEATED_UNIT_LEDGER.replace("Toula", "Miro")
                .replace("She bought", "He bought")
                .replace("donuts", "bagels")
                .replace("cupcakes", "muffins")
                .replace("cheesecakes", "croissants"),
                Fraction(694),
            ),
            (
                TWO_PART_PARTITION.replace("Gretchen", "Mara")
                .replace("coins", "tokens")
                .replace("gold", "blue")
                .replace("silver", "red"),
                Fraction(70),
            ),
            (
                TWO_LINK_SCALE.replace("Brandon", "Mara")
                .replace("Ben", "Miro")
                .replace("Suzy", "Talia")
                .replace("iPhone", "watch"),
                Fraction(8),
            ),
        )
        for question, expected in variants:
            with self.subTest(question=question):
                result = compile_signed_events(question)
                self.assertTrue(result.ok, result.reason)
                assert result.compiled is not None
                self.assertEqual(result.compiled.solution.target_value, expected)

    def test_nine_gold_free_calendar_and_schedule_analogues_are_exact(self) -> None:
        for question, expected, family in AUDIT_SCHEDULE_CASES:
            with self.subTest(family=family):
                result = compile_signed_events(question)
                self.assertTrue(result.ok, result.reason)
                self.assertEqual(result.family, family)
                assert result.compiled is not None
                self.assertEqual(result.compiled.solution.target_value, expected)
                self.assertTrue(result.compiled.certificate.verified)
                parsed = parse_structural_problem(question)
                self.assertTrue(parsed.ok, parsed.reason)
                assert parsed.problem is not None
                self.assertEqual(solve(parsed.problem).target_value, expected)

    def test_schedule_names_and_domain_nouns_are_not_memorized(self) -> None:
        variants = (
            (
                AUDIT_SCHEDULE_CASES[0][0]
                .replace("tow truck", "rescue boat")
                .replace("cars", "rafts")
                .replace("towed", "pulled"),
                Fraction(54),
            ),
            (
                AUDIT_SCHEDULE_CASES[1][0]
                .replace("Chase", "Mara")
                .replace("Rider", "Talia"),
                Fraction(27),
            ),
            (
                AUDIT_SCHEDULE_CASES[2][0].replace("Buford", "Miro"),
                Fraction(52),
            ),
            (
                AUDIT_SCHEDULE_CASES[4][0]
                .replace("Pauline", "Mara")
                .replace("museum", "gallery"),
                Fraction(75),
            ),
            (
                AUDIT_SCHEDULE_CASES[6][0].replace("Tim", "Miro"),
                Fraction(120),
            ),
        )
        for question, expected in variants:
            with self.subTest(question=question):
                result = compile_signed_events(question)
                self.assertTrue(result.ok, result.reason)
                assert result.compiled is not None
                self.assertEqual(result.compiled.solution.target_value, expected)

    def test_value_transition_scope_and_numeric_noise_fail_closed(self) -> None:
        mutations = (
            VALUE_TRANSITION.replace("value of the house", "value of the apartment"),
            VALUE_TRANSITION.replace("did he make", "did she make"),
            VALUE_TRANSITION.replace("$50,000 in repairs", "$50,000 in taxes"),
            VALUE_TRANSITION.replace("increased", "decreased"),
            VALUE_TRANSITION.replace("150%", "150 dollars"),
            VALUE_TRANSITION + " Reference 99.",
        )
        for question in mutations:
            with self.subTest(question=question):
                self.assertFalse(compile_signed_events(question).ok)

    def test_repeated_unit_boundaries_and_denominators_fail_closed(self) -> None:
        mutations = (
            REPEATED_UNIT_LEDGER.replace("$68 per dozen", "$68 per donut"),
            REPEATED_UNIT_LEDGER.replace("mini cupcakes", "donuts"),
            REPEATED_UNIT_LEDGER.replace("She bought", "They bought"),
            REPEATED_UNIT_LEDGER.replace(
                "How much was the total cost?",
                "How much was the total cost and how many pastries were bought?",
            ),
            REPEATED_UNIT_LEDGER + " Reference 99.",
        )
        for question in mutations:
            with self.subTest(question=question):
                self.assertFalse(compile_signed_events(question).ok)

    def test_two_part_exhaustiveness_target_and_integrality_fail_closed(self) -> None:
        mutations = (
            TWO_PART_PARTITION.replace("110 coins", "111 coins"),
            TWO_PART_PARTITION.replace("gold coins does", "silver coins does"),
            TWO_PART_PARTITION.replace("Gretchen have?", "Mara have?"),
            TWO_PART_PARTITION.replace("silver coins", "silver cards"),
            TWO_PART_PARTITION + " There are 5 bronze coins.",
        )
        for question in mutations:
            with self.subTest(question=question):
                self.assertFalse(compile_signed_events(question).ok)

    def test_scale_chain_branches_name_drift_and_noise_fail_closed(self) -> None:
        mutations = (
            TWO_LINK_SCALE.replace("Ben's iPhone. Ben's", "Bob's iPhone. Ben's"),
            TWO_LINK_SCALE.replace("Suzy's iPhone", "Suzy's watch", 1),
            TWO_LINK_SCALE.replace("four times", "four plus times"),
            TWO_LINK_SCALE.replace("Brandon’s iPhone?", "Ben’s iPhone?"),
            TWO_LINK_SCALE.replace("two times", "two three times"),
            TWO_LINK_SCALE + " Reference 99.",
        )
        for question in mutations:
            with self.subTest(question=question):
                self.assertFalse(compile_signed_events(question).ok)

    def test_calendar_segments_coverage_and_frequency_fail_closed(self) -> None:
        mutations = (
            AUDIT_SCHEDULE_CASES[0][0].replace("first three days", "first four days"),
            AUDIT_SCHEDULE_CASES[0][0].replace("remaining days", "some days"),
            AUDIT_SCHEDULE_CASES[1][0].replace("two other days", "three other days"),
            AUDIT_SCHEDULE_CASES[1][0].replace("usual days", "different days"),
            AUDIT_SCHEDULE_CASES[2][0].replace("quarterly", "occasionally"),
            AUDIT_SCHEDULE_CASES[2][0].replace("per year", "per month"),
            AUDIT_SCHEDULE_CASES[2][0].replace(
                "He also writes a check every month",
                "Sam also writes a check every month",
            ),
            AUDIT_SCHEDULE_CASES[2][0].replace(
                "He also writes a check every month",
                "She also writes a check every month",
            ),
            AUDIT_SCHEDULE_CASES[2][0] + " Reference 99.",
        )
        for question in mutations:
            with self.subTest(question=question):
                self.assertFalse(compile_signed_events(question).ok)

    def test_period_pay_conversion_and_exception_mutations_fail_closed(self) -> None:
        mutations = (
            AUDIT_SCHEDULE_CASES[3][0].replace(
                "did not incur any absences", "did incur an absence"
            ),
            AUDIT_SCHEDULE_CASES[3][0].replace("Her hourly rate", "His hourly rate"),
            AUDIT_SCHEDULE_CASES[3][0].replace("exactly 4 weeks", "about 4 weeks"),
            AUDIT_SCHEDULE_CASES[4][0].replace("increased", "decreased"),
            AUDIT_SCHEDULE_CASES[4][0].replace("150%", "150 dollars"),
            AUDIT_SCHEDULE_CASES[4][0].replace("3 more years", "3 overlapping years"),
            AUDIT_SCHEDULE_CASES[5][0].replace("the four years", "the three years"),
            AUDIT_SCHEDULE_CASES[6][0].replace("15 minutes", "15 hours"),
            AUDIT_SCHEDULE_CASES[6][0].replace("It takes him", "It takes her"),
            AUDIT_SCHEDULE_CASES[6][0].replace("answering phones", "answering emails"),
            AUDIT_SCHEDULE_CASES[7][0].replace("sells all", "sells some"),
            AUDIT_SCHEDULE_CASES[7][0].replace(
                "selling produced chocolate bars",
                "selling produced chocolate cookies",
            ),
            AUDIT_SCHEDULE_CASES[7][0].replace("two weeks", "two months"),
            AUDIT_SCHEDULE_CASES[8][0].replace(
                "services 3 cars a day", "services 3 bikes a day"
            ),
            AUDIT_SCHEDULE_CASES[8][0].replace("He gets paid", "She gets paid"),
            AUDIT_SCHEDULE_CASES[8][0].replace(
                "Sunday and Wednesday", "Sunday and Sunday"
            ),
            AUDIT_SCHEDULE_CASES[8][0].replace(
                "Sunday and Wednesday", "Sunday, Wednesday, and Thursday"
            ),
            AUDIT_SCHEDULE_CASES[8][0] + " Reference 99.",
        )
        for question in mutations:
            with self.subTest(question=question):
                self.assertFalse(compile_signed_events(question).ok)


if __name__ == "__main__":
    unittest.main()
