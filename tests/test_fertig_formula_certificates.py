from __future__ import annotations

from fractions import Fraction
import hashlib
import unittest

from immer.cognition.fertig.formula_certificates import solve_guarded_formula


CASES = (
    (
        "John fills a 6 foot by 4 foot pool that is 5 feet deep. It cost $.1 "
        "per cubic foot to fill. How much does it cost to fill?",
        Fraction(12),
        "rectangular_prism_unit_cost",
    ),
    (
        "The ratio of popsicles that Betty and Sam have is 5:6. If the total "
        "number of popsicles they have together is 165, how many more popsicles "
        "does Sam have more than Betty?",
        Fraction(15),
        "ratio_total_difference",
    ),
    (
        "Ali had $21. Leila gave him half of her $100. How much does Ali have now?",
        Fraction(71),
        "initial_plus_fractional_transfer",
    ),
    (
        "Keegan was running a car wash with his friend Tashay to raise money for "
        "a baseball camp. They needed to raise $200 for the two of them. By 3 pm, "
        "Keegan had earned $83 and Tasha had earned $91. How much more did they "
        "need to earn to reach their goal?",
        Fraction(26),
        "target_remainder_after_contributions",
    ),
    (
        "Jillian's handbag cost $20 less than 3 times as much as her shoes cost. "
        "If her shoes cost $80, how much did her bag cost?",
        Fraction(220),
        "affine_relative_price",
    ),
    (
        "The highest temperature ever recorded in Southlandia is -48 degrees "
        "Fahrenheit. The highest temperature ever recorded in Northlandia is 21 "
        "degrees Fahrenheit. The highest temperature recorded in Midlandia is -3 "
        "degrees Fahrenheit. What is the average highest temperature of these 3 "
        "countries?",
        Fraction(-10),
        "signed_three_value_mean",
    ),
    (
        "Elvis has a monthly saving target of $1125. In April, he wants to save "
        "twice as much daily in the second half as he saves in the first half in "
        "order to hit his target. How much does he have to save for each day in "
        "the second half of the month?",
        Fraction(50),
        "even_month_split_daily_saving",
    ),
    (
        "A mechanic charges different rates to repair the tires of trucks and "
        "cars. For each truck tire that is repaired, the mechanic will charge "
        "$60 and for each car tire that is repaired, the mechanic will charge "
        "$40. On Thursday, the mechanic repairs 6 truck tires and 4 car tires. "
        "On Friday, the mechanic repairs 12 car tries and doesn't repair any "
        "truck tires. How much more revenue did the mechanic earn on the day "
        "with higher revenue?",
        Fraction(40),
        "two_day_mixed_rate_revenue_difference",
    ),
    (
        "Well's mother sells watermelons, peppers, and oranges at the local "
        "store. A watermelon costs three times what each pepper costs. An "
        "orange costs 5 less than what a watermelon cost. Dillon is sent to the "
        "store to buy 4 watermelons, 20 peppers, and 10 oranges. What's the "
        "total amount of money he will spend if each pepper costs 15$?",
        Fraction(880),
        "dependent_price_purchase_ledger",
    ),
    (
        "Sara wants to buy herself a new jacket and 2 pairs of shoes. The jacket "
        "she wants costs $30 and each pair of shoes cost $20. Sara babysits the "
        "neighbor's kids 4 times, earning $5 each time she babysits them. Her "
        "parents pay her $4 each time she mows the lawn. If Sara already had $10 "
        "saved before she started babysitting, how many times must she mow the "
        "lawn before she can afford the jacket and shoes?",
        Fraction(10),
        "savings_plus_work_balance",
    ),
    (
        "Lori wants to buy a $320.00 pair of shoes and a matching belt that is "
        "$32.00. Her part-time job pays her $8.00 an hour. How many hours will "
        "she have to work before she can make her purchase?",
        Fraction(44),
        "hourly_wage_purchase_time",
    ),
    (
        "Janeth borrowed $2000 and promised to return it with an additional 10% "
        "of the amount. If she is going to pay $165 a month for 12 months, how "
        "much will be Janeth's remaining balance by then?",
        Fraction(220),
        "simple_interest_loan_balance",
    ),
    (
        "Each person in a certain household consumes 0.2 kg of rice every meal. "
        "Supposing 5 members of the household eat rice every lunch and dinner, "
        "how many weeks will a 42 kg bag of rice last?",
        Fraction(3),
        "household_consumption_duration",
    ),
    (
        "John hires a driving service to get him to work each day. His work is "
        "30 miles away and he has to go there and back each day. He goes to work "
        "5 days a week for 50 weeks a year. He gets charged $2 per mile driven "
        "and he also gives his driver a $150 bonus per month. How much does he "
        "pay a year for driving?",
        Fraction(31800),
        "annual_commute_plus_monthly_bonus",
    ),
    (
        "At the beginning of the party, there were 25 men and 15 women. After "
        "an hour, 1/4 of the total number of people left. How many women are left "
        "if 22 men stayed at the party?",
        Fraction(8),
        "party_fraction_departure_remainder",
    ),
    (
        "Nick is choosing between two jobs. Job A pays $15 an hour for 2000 "
        "hours a year, and is in a state with a 20% total tax rate. Job B pays "
        "$42,000 a year and is in a state that charges $6,000 in property tax "
        "and a 10% tax rate on net income after property tax. How much more "
        "money will Nick make at the job with a higher net pay rate, compared "
        "to the other job?",
        Fraction(8400),
        "two_job_after_tax_difference",
    ),
)


class FormulaCertificateTests(unittest.TestCase):
    def test_general_formula_families_return_verified_exact_certificates(self) -> None:
        for question, expected, family in CASES:
            with self.subTest(family=family):
                solution = solve_guarded_formula(question)
                self.assertIsNotNone(solution)
                assert solution is not None
                self.assertEqual(solution.answer, expected)
                self.assertEqual(solution.certificate.family, family)
                certificate = solution.certificate.to_dict()
                self.assertTrue(certificate["verified"])
                self.assertTrue(certificate["numeric_coverage"])
                self.assertTrue(certificate["numeric_spans"])

    def test_unconsumed_numbers_and_reference_drift_abstain(self) -> None:
        ratio = CASES[1][0]
        self.assertIsNone(solve_guarded_formula(ratio + " Reference 99."))
        self.assertIsNone(
            solve_guarded_formula(ratio.replace("more than Betty", "more than Alice"))
        )
        self.assertIsNone(solve_guarded_formula(CASES[3][0].replace("$200", "$100")))
        self.assertIsNone(solve_guarded_formula(CASES[6][0].replace("April", "May")))
        self.assertIsNone(
            solve_guarded_formula(CASES[7][0].replace("12 car tries", "12 bus tires"))
        )
        self.assertIsNone(
            solve_guarded_formula(CASES[11][0].replace("Janeth's", "Alice's"))
        )
        self.assertIsNone(solve_guarded_formula(CASES[14][0].replace("1/4", "1/0")))

    def test_unrelated_question_abstains(self) -> None:
        self.assertIsNone(solve_guarded_formula("What is the capital of France?"))

    def test_source_hash_and_numeric_spans_bind_the_original_text(self) -> None:
        source = f"  {CASES[11][0]}\n"
        solution = solve_guarded_formula(source)
        self.assertIsNotNone(solution)
        assert solution is not None
        certificate = solution.certificate
        self.assertEqual(
            certificate.source_sha256,
            hashlib.sha256(source.encode("utf-8")).hexdigest(),
        )
        for span in certificate.numeric_spans:
            self.assertEqual(source[span.start : span.end], span.text)


if __name__ == "__main__":
    unittest.main()
