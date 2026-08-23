from __future__ import annotations

import unittest
from fractions import Fraction
from itertools import permutations

from immer.cognition.fertig.arithmetic_ir import (
    Affine,
    Balance,
    Mean,
    Part,
    Rate,
    SolveStatus,
    Sum,
    solve,
)
from immer.cognition.fertig.structural import ParseStatus, parse_structural_problem


def _solve(text: str):
    parsed = parse_structural_problem(text)
    if not parsed.ok:
        raise AssertionError(f"parse failed: {parsed.status.value}: {parsed.reason}")
    assert parsed.problem is not None
    return parsed, solve(parsed.problem)


class AffineStructuralParserTests(unittest.TestCase):
    def test_age_relation_is_exact_and_keeps_source_evidence(self) -> None:
        source = (
            "Chenny is 10 years old. "
            "Alyana is 4 years younger than Chenny. "
            "How old is Anne if she is 2 years older than Alyana?"
        )
        parsed, solution = _solve(source)

        self.assertEqual(solution.status, SolveStatus.UNIQUE)
        self.assertEqual(solution.target_value, 8)
        assert parsed.problem is not None
        for constraint in parsed.problem.constraints:
            self.assertIsNotNone(constraint.span)
            assert constraint.span is not None
            self.assertEqual(
                constraint.span.source[constraint.span.start : constraint.span.end],
                source[constraint.span.start : constraint.span.end],
            )

    def test_entity_rename_and_number_perturbation_do_not_change_grammar(self) -> None:
        first = (
            "Lina has 31 marbles. Omar has 7 fewer marbles than Lina. "
            "How many marbles does Omar have?"
        )
        second = (
            "Keiko has 46 marbles. Pavel has 9 fewer marbles than Keiko. "
            "How many marbles does Pavel have?"
        )

        self.assertEqual(_solve(first)[1].target_value, 24)
        self.assertEqual(_solve(second)[1].target_value, 37)

    def test_comparison_direction_is_not_symmetric(self) -> None:
        more = (
            "Lina has 18 shells. Omar has 5 more shells than Lina. "
            "How many shells does Omar have?"
        )
        fewer = (
            "Lina has 18 shells. Omar has 5 fewer shells than Lina. "
            "How many shells does Omar have?"
        )

        self.assertEqual(_solve(more)[1].target_value, 23)
        self.assertEqual(_solve(fewer)[1].target_value, 13)

    def test_declarative_sentence_order_is_irrelevant(self) -> None:
        forward = (
            "Mina has 6 badges. Theo has 3 times as many badges as Mina. "
            "How many badges does Theo have?"
        )
        reversed_facts = (
            "Theo has 3 times as many badges as Mina. Mina has 6 badges. "
            "How many badges does Theo have?"
        )

        self.assertEqual(_solve(forward)[1].target_value, 18)
        self.assertEqual(_solve(reversed_facts)[1].target_value, 18)

    def test_possessive_property_chain_normalises_paraphrased_subjects(self) -> None:
        source = (
            "Martin's weight is 55 kg. Carl’s weight is 16 kg more than Martin’s weight. "
            "Christian’s weight is 8 kg more than Carl’s weight. "
            "Harry is 5 kg less than Christian’s weight. What is the weight of Harry, in kg?"
        )

        parsed, solution = _solve(source)
        self.assertEqual(solution.target_value, 74)
        assert parsed.problem is not None
        self.assertEqual(parsed.problem.target.unit.dimensions, (("mass", 1),))


class LedgerStructuralParserTests(unittest.TestCase):
    def test_typed_weighted_ledger_sums_contributions_not_item_counts(self) -> None:
        source = (
            "An eraser costs $2 and a pencil costs $3. "
            "How much do 6 erasers and 8 pencils cost?"
        )
        parsed, solution = _solve(source)

        self.assertEqual(solution.status, SolveStatus.UNIQUE)
        self.assertEqual(solution.target_value, 36)
        assert parsed.problem is not None
        self.assertEqual(len(parsed.problem.constraints), 5)
        self.assertEqual(parsed.problem.target.unit.symbol, "USD")

    def test_ledger_supports_item_renaming_decimals_and_numeric_perturbation(
        self,
    ) -> None:
        source = (
            "Each rivet sells for $1.25. One gasket is priced at 2.50 dollars. "
            "How much will 12 rivets and 3 gaskets cost?"
        )

        _, solution = _solve(source)
        self.assertEqual(solution.target_value, Fraction(45, 2))

    def test_unicode_fraction_is_exact_and_span_points_to_original_glyph(self) -> None:
        source = "A ribbon costs $8 each. How much does ½ ribbon cost?"
        parsed, solution = _solve(source)

        self.assertEqual(solution.target_value, 4)
        assert parsed.problem is not None
        rate = parsed.problem.constraints[-2]
        self.assertEqual(rate.duration.value, Fraction(1, 2))
        self.assertEqual(
            source[rate.duration.span.start : rate.duration.span.end],
            "½",
        )

    def test_price_declarations_can_follow_the_question(self) -> None:
        source = (
            "How much do 2 widgets and 4 sprockets cost? "
            "A sprocket costs $3. A widget costs $5."
        )

        self.assertEqual(_solve(source)[1].target_value, 22)


class RateStructuralParserTests(unittest.TestCase):
    def test_explicit_rate_times_duration_uses_typed_rate_constraint(self) -> None:
        source = (
            "Mira earns $12 per hour. Mira works for 7 hours. "
            "How much money does Mira earn?"
        )

        parsed, solution = _solve(source)

        self.assertEqual(solution.status, SolveStatus.UNIQUE)
        self.assertEqual(solution.target_value, 84)
        assert parsed.problem is not None
        self.assertEqual(len(parsed.problem.constraints), 1)
        self.assertIsInstance(parsed.problem.constraints[0], Rate)

    def test_rate_paraphrase_and_number_perturbation_preserve_binding(self) -> None:
        source = (
            "Pavel is paid $35 each day. Pavel worked for 6 days. "
            "How much did Pavel get paid?"
        )

        self.assertEqual(_solve(source)[1].target_value, 210)

    def test_rate_subject_mismatch_is_ambiguous(self) -> None:
        parsed = parse_structural_problem(
            "Mira earns $12 per hour. Pavel works for 7 hours. "
            "How much money does Mira earn?"
        )

        self.assertEqual(parsed.status, ParseStatus.AMBIGUOUS)
        self.assertIsNone(parsed.problem)

    def test_rate_does_not_invent_hours_worked_per_day(self) -> None:
        parsed = parse_structural_problem(
            "Mira earns $12 per hour. Mira works for 7 days. "
            "How much money does Mira earn?"
        )

        self.assertEqual(parsed.status, ParseStatus.AMBIGUOUS)
        self.assertIsNone(parsed.problem)

    def test_rate_rejects_an_extra_unbound_number(self) -> None:
        result = parse_structural_problem(
            "Mira earns $12 per hour. Mira works for 7 hours and takes 2 breaks. "
            "How much money does Mira earn?"
        )

        self.assertEqual(result.status, ParseStatus.UNSUPPORTED)
        self.assertIsNone(result.problem)


class PartStructuralParserTests(unittest.TestCase):
    def test_percent_part_has_an_explicit_whole_and_exact_count(self) -> None:
        source = (
            "Mira has 80 marbles. 25% of those marbles are blue. "
            "How many blue marbles does Mira have?"
        )

        parsed, solution = _solve(source)

        self.assertEqual(solution.target_value, 20)
        assert parsed.problem is not None
        self.assertIsInstance(parsed.problem.constraints[0], Part)

    def test_fraction_paraphrase_and_number_perturbation_preserve_basis(self) -> None:
        first = (
            "Pavel has 56 badges. 3/8 of the badges are silver. "
            "How many silver badges does Pavel have?"
        )
        second = (
            "Keiko has 36 shells. one third of these shells are striped. "
            "How many striped shells does Keiko have?"
        )

        self.assertEqual(_solve(first)[1].target_value, 21)
        self.assertEqual(_solve(second)[1].target_value, 12)

    def test_original_length_percent_is_linear_with_named_basis(self) -> None:
        source = (
            "Mira extends a ribbon by 25% of its original length and adds 5 cm. "
            "The final length is 55 cm. "
            "What was the original length of the ribbon in centimeters?"
        )

        parsed, solution = _solve(source)

        self.assertEqual(solution.target_value, 40)
        assert parsed.problem is not None
        self.assertTrue(
            any(
                isinstance(constraint, Part)
                for constraint in parsed.problem.constraints
            )
        )

    def test_part_basis_noun_mismatch_is_ambiguous(self) -> None:
        result = parse_structural_problem(
            "Mira has 80 marbles. 25% of those shells are blue. "
            "How many blue marbles does Mira have?"
        )

        self.assertEqual(result.status, ParseStatus.AMBIGUOUS)
        self.assertIsNone(result.problem)

    def test_part_rejects_fractional_people(self) -> None:
        result = parse_structural_problem(
            "Mira has 10 students. 25% of those students are absent. "
            "How many absent students does Mira have?"
        )

        self.assertEqual(result.status, ParseStatus.INVALID)
        self.assertIsNone(result.problem)

    def test_bare_number_of_whole_is_not_reinterpreted_as_a_fraction(self) -> None:
        result = parse_structural_problem(
            "Mira has 80 marbles. 1 of those marbles are blue. "
            "How many blue marbles does Mira have?"
        )

        self.assertIn(result.status, {ParseStatus.UNSUPPORTED, ParseStatus.AMBIGUOUS})
        self.assertIsNone(result.problem)

    def test_part_rejects_a_fractional_whole_count(self) -> None:
        result = parse_structural_problem(
            "Mira has 2.5 marbles. 40% of those marbles are blue. "
            "How many blue marbles does Mira have?"
        )

        self.assertEqual(result.status, ParseStatus.INVALID)
        self.assertIsNone(result.problem)

    def test_original_length_rejects_a_different_query_object(self) -> None:
        result = parse_structural_problem(
            "Mira extends a ribbon by 25% of its original length and adds 5 cm. "
            "The final length is 55 cm. "
            "What was the original length of the curtain in centimeters?"
        )

        self.assertEqual(result.status, ParseStatus.AMBIGUOUS)
        self.assertIsNone(result.problem)


class BalanceStructuralParserTests(unittest.TestCase):
    def test_inventory_changes_compile_to_one_exact_balance(self) -> None:
        source = (
            "Mira had some marbles. Mira received 11 more marbles and used 4 "
            "marbles. Mira now has 29 marbles. "
            "How many marbles did Mira have at first?"
        )

        parsed, solution = _solve(source)

        self.assertEqual(solution.target_value, 22)
        assert parsed.problem is not None
        self.assertEqual(len(parsed.problem.constraints), 1)
        self.assertIsInstance(parsed.problem.constraints[0], Balance)

    def test_balance_paraphrase_and_number_perturbation_preserve_signs(self) -> None:
        source = (
            "Pavel had some tickets. Pavel found 17 more tickets and lost 9 "
            "tickets. Pavel now has 41 tickets. "
            "How many tickets did Pavel have originally?"
        )

        self.assertEqual(_solve(source)[1].target_value, 33)

    def test_multi_stop_inventory_has_no_hidden_initial_guess(self) -> None:
        source = (
            "Some passengers got on a bus at the terminal. At the first bus stop, "
            "9 more passengers got in. Then at the second bus stop, 4 passengers "
            "got down and 6 more passengers got in. If there were a total of 31 "
            "passengers heading to the third stop, how many passengers got on the "
            "bus at the terminal?"
        )

        self.assertEqual(_solve(source)[1].target_value, 20)

    def test_balance_rejects_entity_or_noun_drift(self) -> None:
        entity_drift = parse_structural_problem(
            "Mira had some marbles. Pavel received 11 more marbles and used 4 "
            "marbles. Mira now has 29 marbles. "
            "How many marbles did Mira have at first?"
        )
        noun_drift = parse_structural_problem(
            "Mira had some marbles. Mira received 11 more shells and used 4 "
            "marbles. Mira now has 29 marbles. "
            "How many marbles did Mira have at first?"
        )

        self.assertEqual(entity_drift.status, ParseStatus.AMBIGUOUS)
        self.assertEqual(noun_drift.status, ParseStatus.AMBIGUOUS)

    def test_balance_rejects_negative_implied_start(self) -> None:
        result = parse_structural_problem(
            "Mira had some marbles. Mira received 50 more marbles and used 2 "
            "marbles. Mira now has 10 marbles. "
            "How many marbles did Mira have at first?"
        )

        self.assertEqual(result.status, ParseStatus.INVALID)
        self.assertIsNone(result.problem)


class ClosedSystemStructuralParserTests(unittest.TestCase):
    RESOURCE = (
        "Peter wants to make different sized ice cubes with 32 ounces of water. "
        "He can make giant cubes that use 4 ounces per cube, medium cubes that "
        "use 2 ounces, and small cubes that use 1/2 an ounce. If he makes 3 giant "
        "cubes, 7 medium cubes, and 8 small cubes, how many ounces of water does "
        "he have left?"
    )
    COMPONENTS = (
        "Tanya makes a salt scrub from salt, oil, fragrance, citrus zest, and "
        "sugar. She makes enough to fill a 10-ounce jar each time. She uses the "
        "same amount of citrus zest as fragrance and the same amount of salt as "
        "sugar. She uses twice as much oil as salt and twice as much salt as zest. "
        "How many ounces of oil does she use?"
    )
    PERIODS = (
        "In the first half of a soccer match, team A scores 4 goals while team B "
        "scores 2 goals fewer than team A. In the second half, team A scores 1/4 "
        "of the number of goals scored by team B, which scores 4 times the number "
        "of goals it scored in the first half. What's the total number of goals "
        "scored in the match?"
    )

    def assert_five_by_five_certificate(self, source: str, expected: Fraction) -> None:
        parsed, solution = _solve(source)
        self.assertEqual(solution.status, SolveStatus.UNIQUE)
        self.assertEqual(solution.target_value, expected)
        self.assertIsNotNone(solution.certificate)
        assert solution.certificate is not None
        self.assertTrue(solution.certificate.verified)
        self.assertEqual(solution.certificate.rank, 5)
        self.assertEqual(solution.certificate.variable_count, 5)
        self.assertEqual(solution.certificate.equation_count, 5)
        self.assertTrue(
            all(residual.value == 0 for residual in solution.certificate.residuals)
        )
        assert parsed.problem is not None
        self.assertEqual(len(parsed.problem.constraints), 5)

    def test_three_qwen_disagreements_have_exact_symbolic_certificates(self) -> None:
        for source, expected in (
            (self.RESOURCE, Fraction(2)),
            (self.COMPONENTS, Fraction(4)),
            (self.PERIODS, Fraction(16)),
        ):
            with self.subTest(expected=expected):
                self.assert_five_by_five_certificate(source, expected)

    def test_resource_family_survives_renaming_and_number_perturbation(self) -> None:
        source = (
            "Keiko wants to make different sized wax blocks with 40 cups of wax. "
            "She can make large blocks that use 3 cups per block, medium blocks "
            "that use 2 cups, and small blocks that use 1 cup. If she makes 4 "
            "large blocks, 5 medium blocks, and 6 small blocks, how many cups of "
            "wax does she have left?"
        )

        self.assertEqual(_solve(source)[1].target_value, 12)

    def test_component_family_resolves_one_unique_suffix_only(self) -> None:
        source = (
            "Mira makes a lotion from water, oil, wax, and scent. She makes enough "
            "to fill an 18-ounce jar each time. She uses the same amount of scent "
            "as wax. She uses twice as much oil as wax and 3 times as much water "
            "as oil. How many ounces of water does she use?"
        )

        self.assertEqual(_solve(source)[1].target_value, Fraction(54, 5))

    def test_component_equalities_get_distinct_functional_orientations(self) -> None:
        source = (
            "Mira makes a lotion from water, oil, wax, and scent. She makes enough "
            "to fill an 18-ounce jar each time. She uses the same amount of scent "
            "as wax and the same amount of scent as oil. She uses 3 times as much "
            "water as oil. How many ounces of water does she use?"
        )

        self.assertEqual(_solve(source)[1].target_value, 9)

    def test_period_family_keeps_team_and_period_scopes(self) -> None:
        source = (
            "In the first period of a hockey game, team Red scores 7 goals while "
            "team Blue scores 3 goals fewer than team Red. In the second period, "
            "team Red scores 1/2 of the number of goals scored by team Blue, which "
            "scores 2 times the number of goals it scored in the first period. "
            "What's the total number of goals scored in the game?"
        )

        self.assertEqual(_solve(source)[1].target_value, 23)

    def test_resource_family_rejects_label_unit_and_numeric_drift(self) -> None:
        sources = (
            self.RESOURCE.replace("8 small cubes", "8 tiny cubes"),
            self.RESOURCE.replace(
                "medium cubes that use 2 ounces", "medium cubes that use 2 cups"
            ),
            self.RESOURCE.replace(
                "If he makes 3 giant cubes",
                "He discards 1 ounce. If he makes 3 giant cubes",
            ),
        )

        for source in sources:
            with self.subTest(source=source):
                result = parse_structural_problem(source)
                self.assertFalse(result.ok, result)
                self.assertIsNone(result.problem)

    def test_component_family_rejects_unknown_duplicate_or_ambiguous_refs(self) -> None:
        unknown = self.COMPONENTS.replace("ounces of oil", "ounces of cream")
        duplicate = (
            "Mira makes a lotion from water, oil, wax, and scent. She makes enough "
            "to fill an 18-ounce jar each time. She uses the same amount of scent "
            "as wax and the same amount of wax as scent. She uses twice as much oil "
            "as wax. How many ounces of water does she use?"
        )
        ambiguous = (
            "Nora makes a scrub from lemon zest, citrus zest, oil, and salt. She "
            "makes enough to fill a 12-ounce jar each time. She uses the same "
            "amount of zest as salt. She uses twice as much oil as salt and twice "
            "as much lemon zest as oil. How many ounces of citrus zest does she use?"
        )

        for source in (unknown, duplicate, ambiguous):
            with self.subTest(source=source):
                result = parse_structural_problem(source)
                self.assertEqual(result.status, ParseStatus.AMBIGUOUS, result.reason)
                self.assertIsNone(result.problem)

    def test_component_family_rejects_a_cycle_with_an_ungrounded_component(
        self,
    ) -> None:
        source = (
            "Mira makes a lotion from water, oil, wax, and scent. She makes enough "
            "to fill an 18-ounce jar each time. She uses the same amount of water "
            "as oil and the same amount of oil as wax. She uses twice as much "
            "water as wax. How many ounces of scent does she use?"
        )

        result = parse_structural_problem(source)
        self.assertEqual(result.status, ParseStatus.UNSUPPORTED, result.reason)
        self.assertIsNone(result.problem)

    def test_component_family_rejects_two_directed_scales_for_one_target(
        self,
    ) -> None:
        source = (
            "Mira makes a lotion from water, oil, wax, and scent. She makes enough "
            "to fill an 18-ounce jar each time. She uses the same amount of scent "
            "as wax. She uses twice as much water as oil and 3 times as much water "
            "as wax. How many ounces of water does she use?"
        )

        result = parse_structural_problem(source)
        self.assertEqual(result.status, ParseStatus.AMBIGUOUS, result.reason)
        self.assertIsNone(result.problem)

    def test_period_family_rejects_team_reference_drift(self) -> None:
        result = parse_structural_problem(
            self.PERIODS.replace("fewer than team A", "fewer than team C")
        )

        self.assertEqual(result.status, ParseStatus.AMBIGUOUS, result.reason)
        self.assertIsNone(result.problem)


class NaturalChainStructuralParserTests(unittest.TestCase):
    SATIETY = (
        "Grandpa loves to eat jelly beans, but how many jelly beans he can eat "
        "depends on the size of the beans. It takes 75 large jelly beans to fill "
        "Grandpa up. He can eat twice as many medium-sized beans as large beans. "
        "And eating 3 small beans is the same as eating 1 medium-sized bean. How "
        "many small beans can Grandpa eat?"
    )
    FLOW = (
        "The amount of water passing through a river at one point in time is 4000 "
        "gallons. After a day of heavy rain, the amount of water passing through "
        "the river doubles at the same point. If the volume of water passing "
        "through the river at that point increases by 6000 gallons on the third "
        "day, calculate the total amount of water passing through the river at "
        "that point."
    )

    def assert_rank_three(self, source: str, expected: Fraction) -> None:
        parsed, solution = _solve(source)
        self.assertEqual(solution.status, SolveStatus.UNIQUE)
        self.assertEqual(solution.target_value, expected)
        self.assertIsNotNone(solution.certificate)
        assert solution.certificate is not None
        self.assertTrue(solution.certificate.verified)
        self.assertEqual(solution.certificate.rank, 3)
        self.assertEqual(solution.certificate.variable_count, 3)
        self.assertEqual(solution.certificate.equation_count, 3)
        assert parsed.problem is not None
        self.assertEqual(len(parsed.problem.constraints), 3)

    def test_qwen_scale_and_timeline_rows_gain_exact_certificates(self) -> None:
        for source, expected in (
            (self.SATIETY, Fraction(450)),
            (self.FLOW, Fraction(14000)),
        ):
            with self.subTest(expected=expected):
                self.assert_rank_three(source, expected)

    def test_satiety_chain_survives_renaming_and_ratio_perturbation(self) -> None:
        source = (
            "Grandmother loves to eat candy pieces, but how many candy pieces she can eat "
            "depends on the size of the pieces. It takes 40 jumbo candy pieces to "
            "fill Grandmother up. She can eat 3 times as many regular pieces as jumbo "
            "pieces. And eating 2 tiny pieces is the same as eating 1 regular "
            "piece. How many tiny pieces can Grandmother eat?"
        )

        self.assertEqual(_solve(source)[1].target_value, 240)

    def test_flow_chain_survives_material_channel_and_number_renaming(self) -> None:
        source = (
            "The amount of oil passing through a pipe at one point in time is 120 "
            "liters. After a day of maintenance, the amount of oil passing through "
            "the pipe triples at the same point. If the volume of oil passing "
            "through the pipe at that point increases by 30 liters on the third "
            "day, calculate the total amount of oil passing through the pipe at "
            "that point."
        )

        self.assertEqual(_solve(source)[1].target_value, 390)

    def test_satiety_chain_rejects_owner_label_noun_and_fractional_drift(self) -> None:
        sources = (
            self.SATIETY.replace("fill Grandpa up", "fill Pavel up"),
            self.SATIETY.replace("he can eat", "she can eat").replace(
                "He can eat", "She can eat"
            ),
            self.SATIETY.replace("he can eat", "they can eat").replace(
                "He can eat", "They can eat"
            ),
            self.SATIETY.replace("as large beans", "as tiny beans"),
            self.SATIETY.replace("1 medium-sized bean", "1 medium-sized berry"),
            self.SATIETY.replace("1 medium-sized bean", "4 medium-sized beans"),
        )

        for source in sources:
            with self.subTest(source=source):
                result = parse_structural_problem(source)
                self.assertFalse(result.ok, result)
                self.assertIsNone(result.problem)

    def test_flow_chain_rejects_material_channel_unit_and_timeline_drift(self) -> None:
        sources = (
            self.FLOW.replace("volume of water", "volume of oil"),
            self.FLOW.replace("through the river doubles", "through the pipe doubles"),
            self.FLOW.replace("6000 gallons", "6000 liters"),
            self.FLOW.replace("on the third day", "on the third week"),
        )

        for source in sources:
            with self.subTest(source=source):
                result = parse_structural_problem(source)
                self.assertEqual(result.status, ParseStatus.AMBIGUOUS, result.reason)
                self.assertIsNone(result.problem)


class MeanStructuralParserTests(unittest.TestCase):
    def test_explicit_score_list_compiles_to_mean_constraint(self) -> None:
        source = (
            "Nora received the following scores on her science quizzes: "
            "55, 65, 75, 85, and 95. Find her mean score."
        )

        parsed, solution = _solve(source)

        self.assertEqual(solution.target_value, 75)
        assert parsed.problem is not None
        self.assertEqual(len(parsed.problem.constraints), 1)
        self.assertIsInstance(parsed.problem.constraints[0], Mean)

    def test_mean_paraphrase_handles_exact_signed_measurements(self) -> None:
        source = (
            "Pavel recorded measurements: -5, 7, 10, and 0. "
            "What is Pavel's average measurement?"
        )

        self.assertEqual(_solve(source)[1].target_value, 3)

    def test_mean_number_perturbation_changes_only_the_arithmetic(self) -> None:
        first = "Mira got scores: 10, 20, and 30. What is her mean score?"
        second = "Mira got scores: 13, 22, and 31. What is her mean score?"

        self.assertEqual(_solve(first)[1].target_value, 20)
        self.assertEqual(_solve(second)[1].target_value, 22)

    def test_mean_rejects_unbound_extra_observation(self) -> None:
        result = parse_structural_problem(
            "Mira got scores: 10, 20, and 30. Mira skipped 2 quizzes. "
            "What is her mean score?"
        )

        self.assertEqual(result.status, ParseStatus.UNSUPPORTED)
        self.assertIsNone(result.problem)


class RecurrenceStructuralParserTests(unittest.TestCase):
    START = "Mira's sequence has value 3 at step 0"
    RULE = (
        "At each step, the next value in Mira's sequence is 2 times the "
        "current value in Mira's sequence plus 1"
    )

    def test_value_change_and_inclusive_sum_are_distinct_certified_targets(
        self,
    ) -> None:
        queries = {
            "What is the value in Mira's sequence at step 4": 63,
            "What is the net change in Mira's sequence from step 0 to step 4": 60,
            (
                "What is the cumulative sum of the values in Mira's sequence "
                "from step 0 through step 4"
            ): 119,
        }

        for query, expected in queries.items():
            with self.subTest(query=query):
                parsed, solution = _solve(f"{self.START}. {self.RULE}. {query}?")
                self.assertEqual(solution.status, SolveStatus.UNIQUE)
                self.assertEqual(solution.target_value, expected)
                self.assertIsNotNone(solution.certificate)
                assert solution.certificate is not None
                self.assertTrue(solution.certificate.verified)
                assert parsed.problem is not None
                self.assertTrue(
                    all(
                        constraint.span is not None
                        for constraint in parsed.problem.constraints
                    )
                )

        parsed, _ = _solve(
            f"{self.START}. {self.RULE}. "
            "What is the value in Mira's sequence at step 4?"
        )
        assert parsed.problem is not None
        self.assertEqual(
            sum(
                isinstance(constraint, Affine)
                for constraint in parsed.problem.constraints
            ),
            4,
        )

    def test_current_and_original_percent_bases_produce_different_ir(self) -> None:
        start = "Mira's sequence has value 10 at step 0"
        query = "What is the value in Mira's sequence at step 2?"
        current = (
            "At each step, the next value in Mira's sequence is the current value "
            "in Mira's sequence plus 10% of the current value in Mira's sequence "
            "plus 2"
        )
        original = (
            "At each step, the next value in Mira's sequence is the current value "
            "in Mira's sequence plus 10% of the original value in Mira's sequence "
            "plus 2"
        )

        current_parsed, current_solution = _solve(f"{start}. {current}. {query}")
        original_parsed, original_solution = _solve(f"{start}. {original}. {query}")

        self.assertEqual(current_solution.target_value, Fraction(163, 10))
        self.assertEqual(original_solution.target_value, 16)
        assert current_parsed.problem is not None
        assert original_parsed.problem is not None
        self.assertEqual(
            sum(
                isinstance(constraint, Part)
                for constraint in current_parsed.problem.constraints
            ),
            2,
        )
        self.assertEqual(
            sum(
                isinstance(constraint, Part)
                for constraint in original_parsed.problem.constraints
            ),
            1,
        )
        self.assertTrue(
            any(
                isinstance(constraint, Sum)
                for constraint in current_parsed.problem.constraints
            )
        )

    def test_generated_renames_and_number_perturbations_preserve_recurrence(
        self,
    ) -> None:
        cases = (
            ("Mira", 3, Fraction(2), 1, 0, 4),
            ("Pavel", 5, Fraction(3, 2), -2, 2, 6),
            ("Keiko", -4, Fraction(-1), 3, 7, 12),
            ("Nora", 9, Fraction(1, 3), 0, 11, 14),
        )

        for owner, initial, factor, offset, first, last in cases:
            factor_text = (
                str(factor.numerator)
                if factor.denominator == 1
                else f"{factor.numerator}/{factor.denominator}"
            )
            direction = "plus" if offset >= 0 else "minus"
            source = (
                f"{owner}'s sequence has value {initial} at step {first}. "
                f"At each step, the next value in {owner}'s sequence is "
                f"{factor_text} times the current value in {owner}'s sequence "
                f"{direction} {abs(offset)}. "
                f"What is the value in {owner}'s sequence at step {last}?"
            )
            expected = Fraction(initial)
            for _ in range(last - first):
                expected = factor * expected + offset

            with self.subTest(owner=owner, first=first, last=last):
                self.assertEqual(_solve(source)[1].target_value, expected)

    def test_all_sentence_orders_preserve_absolute_step_indexing(self) -> None:
        clauses = (
            "Pavel's sequence has value 4 at step 2.",
            (
                "At each step, the next value in Pavel's sequence equals 3/2 times "
                "the current value in Pavel's sequence minus 1."
            ),
            "What is the value in Pavel's sequence at step 5?",
        )

        for order in permutations(clauses):
            with self.subTest(order=order):
                self.assertEqual(
                    _solve(" ".join(order))[1].target_value,
                    Fraction(35, 4),
                )

    def test_ambiguous_or_incomplete_recurrences_abstain(self) -> None:
        cases = (
            (
                "Mira's sequence has value 3 at step 0. At each step, the next "
                "value in Mira's sequence is 2 times the current value in Pavel's "
                "sequence plus 1. What is the value in Mira's sequence at step 4?",
                ParseStatus.AMBIGUOUS,
            ),
            (
                "Mira's sequence has value 3 at step 0. At each step, the next "
                "value in Mira's sequence is the current value in Mira's sequence "
                "plus 10% plus 1. What is the value in Mira's sequence at step 4?",
                ParseStatus.AMBIGUOUS,
            ),
            (
                f"{self.START}. {self.RULE}. What is the net change in Mira's "
                "sequence from step 1 to step 4?",
                ParseStatus.AMBIGUOUS,
            ),
            (
                f"{self.START}. {self.RULE}. What is the value in Pavel's "
                "sequence at step 4?",
                ParseStatus.AMBIGUOUS,
            ),
            (
                f"{self.START}. {self.RULE}. What is the value in Mira's "
                "sequence at step 1.5?",
                ParseStatus.INVALID,
            ),
            (
                "Mira's sequence has value 3 at step 5. At each step, the next "
                "value in Mira's sequence is 2 times the current value in Mira's "
                "sequence plus 1. What is the value in Mira's sequence at step 4?",
                ParseStatus.INVALID,
            ),
            (
                f"{self.START}. What is the value in Mira's sequence at step 4?",
                ParseStatus.UNSUPPORTED,
            ),
            (
                f"{self.START}. At each step, the next value in Mira's sequence "
                "is 2 times the current value in Mira's sequence. What is the "
                "value in Mira's sequence at step 4?",
                ParseStatus.UNSUPPORTED,
            ),
            (
                f"{self.START}. {self.RULE}. {self.RULE}. What is the value in "
                "Mira's sequence at step 4?",
                ParseStatus.AMBIGUOUS,
            ),
            (
                f"{self.START}. {self.RULE}. What is the value in Mira's "
                "sequence at step 65?",
                ParseStatus.UNSUPPORTED,
            ),
        )

        for source, expected in cases:
            with self.subTest(source=source):
                result = parse_structural_problem(source)
                self.assertEqual(result.status, expected, result.reason)
                self.assertIsNone(result.problem)


class FailClosedStructuralParserTests(unittest.TestCase):
    def assert_not_parsed(self, source: str, expected: ParseStatus) -> None:
        result = parse_structural_problem(source)
        self.assertEqual(result.status, expected, result.reason)
        self.assertIsNone(result.problem)

    def test_pronoun_binding_is_ambiguous(self) -> None:
        self.assert_not_parsed(
            "Lina has 10 shells. She has 4 more shells than Omar. "
            "How many shells does Lina have?",
            ParseStatus.AMBIGUOUS,
        )

    def test_comparison_with_two_possible_bases_is_not_truncated(self) -> None:
        self.assert_not_parsed(
            "Lina has 10 shells. Omar has 4 more shells than Lina and Nia. "
            "How many shells does Omar have?",
            ParseStatus.UNSUPPORTED,
        )

    def test_unit_mismatch_fails_before_solver(self) -> None:
        self.assert_not_parsed(
            "Lina is 10 years old. Lina is 8 months old. How old is Lina?",
            ParseStatus.AMBIGUOUS,
        )

    def test_unparsed_numeric_clause_cannot_be_silently_ignored(self) -> None:
        self.assert_not_parsed(
            "Lina has 10 shells. A box hides 3 extra shells. "
            "How many shells does Lina have?",
            ParseStatus.UNSUPPORTED,
        )

    def test_multiple_targets_are_ambiguous(self) -> None:
        self.assert_not_parsed(
            "Lina has 10 shells. Omar has 4 shells. "
            "How many shells does Lina have? How many shells does Omar have?",
            ParseStatus.AMBIGUOUS,
        )

    def test_missing_ledger_rate_is_not_guessed(self) -> None:
        self.assert_not_parsed(
            "An eraser costs $2. How much do 2 erasers and 3 pencils cost?",
            ParseStatus.UNSUPPORTED,
        )

    def test_duplicate_price_requires_an_explicit_scope(self) -> None:
        self.assert_not_parsed(
            "A ticket costs $5. A ticket costs $7. How much do 2 tickets cost?",
            ParseStatus.AMBIGUOUS,
        )

    def test_known_semantic_quarantine_remains_unparsed(self) -> None:
        sources = (
            "Janeth borrowed $2000 and promised to return it with an additional 10% "
            "of the amount. If she is going to pay $165 a month for 12 months, how "
            "much will be Janeth's remaining balance by then?",
            "Shania is designing her own dress, and decides to make it a longer dress "
            "by extending the dress by 50% of its original length. She also adds 20cm "
            "to the bottom of the dress with a lace trim. If the final design is "
            "140cm long then how long, in centimeters, was the dress in its original "
            "design?",
            "Sasha and Julie are best friends playing on opposing basketball teams. "
            "The teams have two practice games scheduled. In the first game, Sasha "
            "had the home court advantage and scored 14 points. Julie scored 4 fewer "
            "points than Sasha in the same game. Sasha always struggles during away "
            "games and their second match was at Julie's home court. Sasha scored 6 "
            "fewer points in the second game than Julie's score in the first game. "
            "How many total points did Sasha score during both games?",
            "Dijana and Anis live near a lake, and every weekend they go out rowing "
            "into the lake. On a Sunday morning, both went out rowing, and Dijana "
            "rowed for 50 miles the whole day. Anis rowed 1/5 times more miles than "
            "Dijana. Calculate the total distance the two of them rowed on that day.",
            "The combined age of Peter, Paul and Jean is 100 years old. Find the age "
            "of Peter knowing that Paul is 10 years older than John and that Peter’s "
            "age is equal to the sum of Paul and John's age.",
            "Jim decides to go to college to earn some more money. It takes him 4 "
            "years to finish and he gets $50,000 in loans per year. If he had a 25k "
            "a year job before college and his college degree tripled his income, how "
            "long would it take to earn the money equivalent to the loans and the "
            "money lost from not working while in school.",
            "Johnny's dad brought him to watch some horse racing and his dad bet "
            "money. On the first race, he lost $5. On the second race, he won $1 "
            "more than twice the amount he previously lost. On the third race, he "
            "lost 1.5 times as much as he won in the second race. How much did he "
            "lose on average that day?",
            "Ben bought a car for $20000 in 2007. The price of the car depreciates at "
            "a constant rate of 21% per year. Find the price of the car in the year "
            "2010.",
            "Amalia, Megan, and Dior divided the home chores so that each person had "
            "something to do while the others were working. Amalia's work was to mow "
            "the lawn, which took her 4 hours. Megan had to walk the dog and this took "
            "her 2 hours longer than Amalia to complete her chore. Dior's work was to "
            "do laundry and she took well over 4 hours longer than the time Amalia "
            "took to mow the lawn. Calculate the total time they all took to do their "
            "chores altogether.",
            "Mark buys one lottery ticket with a 20% chance of winning and a second "
            "lottery ticket that's three times more likely to win. What is the "
            "probability, expressed as a percentage, that both tickets are winners?",
            "The girls are trying to raise money for a carnival. Kim raises $320 more "
            "than Alexandra, who raises $430, and Maryam raises $400 more than Sarah, "
            "who raises $300. How much money, in dollars, did they all raise in total?",
        )

        for source in sources:
            with self.subTest(source=source[:50]):
                result = parse_structural_problem(source)
                self.assertFalse(result.ok, result)
                self.assertIsNone(result.problem)


if __name__ == "__main__":
    unittest.main()
