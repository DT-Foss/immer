from __future__ import annotations

from fractions import Fraction
import unittest

from immer.cognition.fertig.arithmetic_ir import solve
from immer.cognition.fertig.signed_event_frontend import compile_signed_events
from immer.cognition.fertig.structural import parse_structural_problem


DAILY_BUDGET = (
    "Peter has $70 and wishes to spend an equal amount each day for one week. "
    "From Sunday through Wednesday, he spent his money on wooden action figures "
    "which cost $5 each. For the rest of the week, he will buy plastic action "
    "figures which cost $2 each. How many total action figures will he have by "
    "the end of the week?"
)

TWO_DAY_DISCOUNT = (
    "On Tuesday, Clara bought 20 pomegranates at $20 each. At the till she got "
    "$2 off because she had a voucher. The next day, the price shot to $30 per "
    "fruit, but the store also offered a 10% discount on the total cost. Sheila "
    "took advantage of the discount and bought 20 pomegranates. What is the "
    "difference between the final prices paid for the pomegranates on the two "
    "days?"
)

MONTHLY_DURATION = (
    "Britany records 18 4-minute TikTok videos each week. She spends 2 hours a "
    "week writing amateur songs to sing on TikTok, and 15 minutes six days a "
    "week doing her makeup before filming herself for TikTok. How much time "
    "does Britany spend on TikTok in a month with four weeks?"
)

INSTALLATION = (
    "An interior design firm offers installation for $129.00. It includes "
    "hanging 4 mirrors, 2 shelves, 1 chandelier, and 10 pictures. They will "
    "install additional items for an extra $15.00 per item. Angela has 6 mirrors "
    "and 2 chandeliers and 20 pictures that she needs installed/hung. How much "
    "will this cost her?"
)

CASES = (
    (DAILY_BUDGET, Fraction(23), "equal_daily_budget_item_schedule"),
    (TWO_DAY_DISCOUNT, Fraction(142), "two_day_discount_price_difference"),
    (MONTHLY_DURATION, Fraction(94, 5), "closed_monthly_duration_ledger"),
    (INSTALLATION, Fraction(324), "typed_installation_overage_cost"),
)


class FertigBundleTariffAlgebraTests(unittest.TestCase):
    def test_four_closed_families_compile_and_certify_exactly(self) -> None:
        for question, expected, family in CASES:
            with self.subTest(family=family):
                frontend = compile_signed_events(question)
                self.assertTrue(frontend.ok, frontend.reason)
                self.assertEqual(frontend.family, family)
                assert frontend.compiled is not None
                self.assertEqual(frontend.compiled.solution.target_value, expected)
                self.assertTrue(frontend.compiled.certificate.verified)

                structural = parse_structural_problem(question)
                self.assertTrue(structural.ok, structural.reason)
                assert structural.problem is not None
                solution = solve(structural.problem)
                self.assertEqual(solution.target_value, expected)
                self.assertTrue(solution.certificate and solution.certificate.verified)

    def test_names_items_and_platform_are_not_memorized(self) -> None:
        renamed = (
            (
                DAILY_BUDGET.replace("Peter", "Mara")
                .replace("he spent his", "she spent her")
                .replace("he will", "she will")
                .replace("will he have", "will she have")
                .replace("action figures", "model robots"),
                Fraction(23),
            ),
            (
                TWO_DAY_DISCOUNT.replace("Clara", "Noah")
                .replace("she got", "he got")
                .replace("she had", "he had")
                .replace("Sheila", "Omar")
                .replace("pomegranates", "pears"),
                Fraction(142),
            ),
            (
                MONTHLY_DURATION.replace("Britany", "Arlo")
                .replace("She spends", "He spends")
                .replace("her makeup", "his makeup")
                .replace("herself", "himself")
                .replace("TikTok", "Clipster"),
                Fraction(94, 5),
            ),
            (
                INSTALLATION.replace("Angela", "Noah")
                .replace("she needs", "he needs")
                .replace("cost her", "cost him")
                .replace("mirrors", "panels")
                .replace("shelves", "racks")
                .replace("chandelier", "lamp")
                .replace("pictures", "frames"),
                Fraction(324),
            ),
        )
        for question, expected in renamed:
            with self.subTest(question=question):
                result = compile_signed_events(question)
                self.assertTrue(result.ok, result.reason)
                assert result.compiled is not None
                self.assertEqual(result.compiled.solution.target_value, expected)

    def test_closed_clause_and_sku_reordering_preserves_results(self) -> None:
        intro, ranged, rest, query = DAILY_BUDGET.split(". ")
        reordered_week = ". ".join((intro, rest, ranged, query))
        result = compile_signed_events(reordered_week)
        self.assertTrue(result.ok, result.reason)
        assert result.compiled is not None
        self.assertEqual(result.compiled.solution.target_value, 23)

        reordered_skus = INSTALLATION.replace(
            "4 mirrors, 2 shelves, 1 chandelier, and 10 pictures",
            "10 pictures, 1 chandelier, 4 mirrors, and 2 shelves",
        ).replace(
            "6 mirrors and 2 chandeliers and 20 pictures",
            "20 pictures and 6 mirrors and 2 chandeliers",
        )
        result = compile_signed_events(reordered_skus)
        self.assertTrue(result.ok, result.reason)
        assert result.compiled is not None
        self.assertEqual(result.compiled.solution.target_value, 324)

    def test_positive_overages_are_per_sku_and_never_pool_deficits(self) -> None:
        below_one_quota = INSTALLATION.replace("has 6 mirrors", "has 3 mirrors")
        result = compile_signed_events(below_one_quota)
        self.assertTrue(result.ok, result.reason)
        assert result.compiled is not None
        # Mirrors contribute zero; chandelier contributes one and pictures ten.
        self.assertEqual(result.compiled.solution.target_value, 294)

    def test_owner_item_scope_and_target_drift_fail_closed(self) -> None:
        mutations = (
            DAILY_BUDGET.replace("he will buy", "she will buy"),
            DAILY_BUDGET.replace("total action figures", "total toy cars"),
            DAILY_BUDGET.replace(
                "through Wednesday,", "through Wednesday and Thursday,"
            ),
            TWO_DAY_DISCOUNT.replace(
                "she got $2 off because she had", "he got $2 off because she had"
            ),
            TWO_DAY_DISCOUNT.replace(
                "bought 20 pomegranates. What", "bought 20 oranges. What"
            ),
            TWO_DAY_DISCOUNT.replace(
                "discount on the total cost", "discount on one selected fruit"
            ),
            MONTHLY_DURATION.replace("does Britany spend", "does Mara spend"),
            MONTHLY_DURATION.replace("filming herself for TikTok", "filming herself for Reels"),
            MONTHLY_DURATION.replace("six days a week", "five days a week"),
            INSTALLATION.replace("20 pictures", "20 windows"),
            INSTALLATION.replace("that she needs", "that he needs"),
            INSTALLATION.replace("per item", "per package"),
        )
        for question in mutations:
            with self.subTest(question=question):
                self.assertFalse(compile_signed_events(question).ok)

        for question, _, _ in CASES:
            with self.subTest(extra_evidence=question):
                self.assertFalse(
                    compile_signed_events(question + " Unrelated reference 99.").ok
                )
            with self.subTest(multiple_target=question):
                self.assertFalse(
                    compile_signed_events(
                        question.replace("?", " and what is the subtotal?")
                    ).ok
                )


if __name__ == "__main__":
    unittest.main()
