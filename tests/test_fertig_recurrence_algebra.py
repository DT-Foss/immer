from __future__ import annotations

from fractions import Fraction
import unittest

from immer.cognition.fertig.adapter import FertigSolver
from immer.cognition.fertig.arithmetic_ir import Rate, SolveStatus, Span, Unit, solve
from immer.cognition.fertig.clause_compiler import SymbolKey
from immer.cognition.fertig.discourse_ssa import TypedDiscourseSSA
from immer.cognition.fertig.signed_event_frontend import compile_signed_events
from immer.cognition.fertig.signed_expression import (
    ExpressionCompileError,
    ExpressionProgram,
    ExpressionTarget,
    IterateExpr,
    LiteralExpr,
    NumericEvidence,
    RefExpr,
    compile_expression,
)
from immer.cognition.fertig.structural import parse_structural_problem


COUNT = Unit.count()
SCALAR = Unit.scalar()
MONEY = Unit.base("money", symbol="USD")
PERCENT = Unit("%", (), Fraction(1, 100))

PHONE_TREE = (
    "A phone tree is used to contact families and relatives of Ali's deceased "
    "coworker. Ali decided to call 3 families. Then each family calls 3 other "
    "families, and so on. How many families will be notified during the fourth "
    "round of calls?"
)

MONTHLY_DELTA = (
    "In one year, the number of students on campus doubles at the end of every "
    "month. If there are 10 students on campus at the beginning of the year, "
    "how many additional students would have joined by the end of May, above "
    "and beyond the number of students already on campus at the beginning of "
    "the year?"
)

DAILY_TOTAL = (
    "Alice likes to count the puffs of clouds in the sky while she eats her "
    "lunch outside at school. On Monday she counts just 3 puffs of clouds. "
    "Each day after that through Friday, though, she sees double the number of "
    "clouds in the sky as the day before. At the end of the week, how many "
    "clouds will she have counted in the sky at lunch across all five days?"
)

ORIGINAL_BASE_PERCENTAGE = (
    "Mrs. Tatiana owns a grocery store that sells different fruits and "
    "vegetables, which includes carrots. The price of carrots in the grocery "
    "store increases by 5% of the original price every year. What would be the "
    "price of carrots after three years if it was $120 initially? (Round to the "
    "nearest integer)"
)

WEIGHTED_THRESHOLD = (
    "Donny can only drink water if it's at least 40 degrees. He has two mugs of "
    "water. One mug is 33 degrees. The other is an unknown temperature. If he "
    "pours 4 ounces of water from the 33-degree mug into his water bottle and "
    "one ounce from the other bottle, he is now able to drink the water. At "
    "least how many degrees is the second bottle?"
)

AMBIGUOUS_PERCENTAGE_THRESHOLD = (
    "A teacher uses a 5-inch piece of chalk to write math equations on a "
    "chalkboard for his students. The teacher likes to conserve chalk, so he "
    "tries to only use 20% of the chalk each day. Since the teacher cannot "
    "write with a very small piece of chalk, he recycles the chalk when it is "
    "smaller than 2 inches. On Monday the teacher used a new piece of chalk. "
    "His students need extra help that day, so he ended up writing more than "
    "usual. He used up 45% of the chalk by the end of the day. If the teacher "
    "goes back to using only 20% of the chalk each day, how many days does he "
    "have before he has to recycle this piece?"
)


class _ExpressionBuilder:
    def __init__(self) -> None:
        self.evidence: list[NumericEvidence] = []

    def literal(self, value: int, unit: Unit) -> LiteralExpr:
        text = str(value)
        span = Span(0, len(text), text)
        evidence_id = f"n-{len(self.evidence)}"
        self.evidence.append(NumericEvidence(evidence_id, value, unit, span))
        return LiteralExpr(value, unit, span, (evidence_id,))

    def compile(self, expression: IterateExpr):
        target = SymbolKey("question", "answer", "value", "iteration", "current")
        return compile_expression(
            ExpressionProgram(
                (),
                ExpressionTarget(target, expression, expression.span),
                tuple(self.evidence),
            )
        )


class IterateExpressionTests(unittest.TestCase):
    def test_terminal_delta_cumulative_and_fixed_base_rules_unroll_exactly(
        self,
    ) -> None:
        cases = (
            (3, 3, 4, "multiply", "state_count", "final", Fraction(81)),
            (10, 2, 5, "multiply", "updates", "increase", Fraction(310)),
            (3, 2, 5, "multiply", "state_count", "cumulative", Fraction(93)),
        )
        for initial, factor, count, rule, count_mode, output, expected in cases:
            with self.subTest(output=output):
                builder = _ExpressionBuilder()
                expression = IterateExpr(
                    builder.literal(initial, COUNT),
                    builder.literal(factor, SCALAR),
                    builder.literal(count, SCALAR),
                    rule,
                    count_mode,
                    output,
                    Span(0, 0),
                )
                compiled = builder.compile(expression)
                self.assertEqual(compiled.solution.target_value, expected)
                self.assertTrue(compiled.certificate.verified)
                expected_updates = count - (count_mode == "state_count")
                transition_rates = [
                    row
                    for row in compiled.problem.constraints
                    if isinstance(row, Rate)
                    and row.duration.unit.dimensions == ()
                    and row.duration.value != 0
                ]
                self.assertEqual(len(transition_rates), expected_updates)

        builder = _ExpressionBuilder()
        fixed = IterateExpr(
            builder.literal(120, MONEY),
            builder.literal(5, PERCENT),
            builder.literal(3, SCALAR),
            "add_initial_fraction",
            "updates",
            "final",
            Span(0, 0),
        )
        compiled = builder.compile(fixed)
        self.assertEqual(compiled.solution.target_value, 138)
        self.assertEqual(
            len([row for row in compiled.problem.constraints if isinstance(row, Rate)]),
            2,
        )

    def test_original_base_and_current_base_percentage_are_distinct(self) -> None:
        values = {}
        for rule in ("add_initial_fraction", "add_current_fraction"):
            builder = _ExpressionBuilder()
            expression = IterateExpr(
                builder.literal(120, MONEY),
                builder.literal(5, PERCENT),
                builder.literal(3, SCALAR),
                rule,
                "updates",
                "final",
                Span(0, 0),
            )
            values[rule] = builder.compile(expression).solution.target_value
        self.assertEqual(values["add_initial_fraction"], 138)
        self.assertEqual(values["add_current_fraction"], Fraction(27783, 200))

    def test_iteration_count_and_mode_fail_closed(self) -> None:
        for count in (-1, 4097):
            builder = _ExpressionBuilder()
            expression = IterateExpr(
                builder.literal(2, COUNT),
                builder.literal(2, SCALAR),
                builder.literal(count, SCALAR),
                "multiply",
                "updates",
                "final",
                Span(0, 0),
            )
            with self.subTest(count=count), self.assertRaises(ExpressionCompileError):
                builder.compile(expression)

        builder = _ExpressionBuilder()
        with self.assertRaisesRegex(ValueError, "cumulative"):
            IterateExpr(
                builder.literal(2, COUNT),
                builder.literal(2, SCALAR),
                builder.literal(2, SCALAR),
                "multiply",
                "updates",
                "cumulative",
                Span(0, 0),
            )

        reference = RefExpr(
            SymbolKey("source", "factor", "item", "iteration", "current"),
            Span(0, 0),
        )
        builder = _ExpressionBuilder()
        initial = builder.literal(2, COUNT)
        factor = builder.literal(2, SCALAR)
        count = builder.literal(3, SCALAR)
        with self.assertRaisesRegex(TypeError, "factor.*ground LiteralExpr"):
            IterateExpr(
                initial,
                reference,
                count,
                "multiply",
                "updates",
                "final",
                Span(0, 0),
            )
        with self.assertRaisesRegex(TypeError, "count.*ground LiteralExpr"):
            IterateExpr(
                initial,
                factor,
                reference,
                "multiply",
                "updates",
                "final",
                Span(0, 0),
            )

    def test_discourse_ssa_traverses_iteration_dependencies(self) -> None:
        source = "seed factor count"
        span = Span(0, len(source), source)
        builder = _ExpressionBuilder()
        ssa = TypedDiscourseSSA(source)
        owner = ssa.entity("Aria", "owner", span)
        seed = ssa.symbol(
            owner,
            property="quantity",
            item="item",
            scope="recurrence",
            state="initial",
            role="state",
            unit=COUNT,
            span=span,
        )
        ssa.define(
            seed,
            builder.literal(3, COUNT),
            span,
            relation_id="initial_state",
        )
        target = IterateExpr(
            ssa.ref(seed, span, role="state"),
            builder.literal(2, SCALAR),
            builder.literal(4, SCALAR),
            "multiply",
            "state_count",
            "final",
            span,
        )
        definitions = ssa.finalize(target)
        key = SymbolKey("question", "answer", "value", "iteration", "current")
        compiled = compile_expression(
            ExpressionProgram(
                definitions,
                ExpressionTarget(key, target, span),
                tuple(builder.evidence),
            )
        )
        self.assertEqual(compiled.solution.target_value, 24)
        self.assertTrue(compiled.certificate.verified)


class RecurrenceFrontendTests(unittest.TestCase):
    CASES = (
        (PHONE_TREE, Fraction(81), "closed_phone_tree_recurrence"),
        (MONTHLY_DELTA, Fraction(310), "closed_monthly_state_recurrence"),
        (DAILY_TOTAL, Fraction(93), "closed_daily_geometric_total"),
        (
            ORIGINAL_BASE_PERCENTAGE,
            Fraction(138),
            "fixed_base_percentage_recurrence",
        ),
    )

    def test_four_question_only_recurrences_are_exact_and_publicly_certified(
        self,
    ) -> None:
        solver = FertigSolver()
        for question, expected, family in self.CASES:
            with self.subTest(family=family):
                frontend = compile_signed_events(question)
                self.assertTrue(frontend.ok, frontend.reason)
                self.assertEqual(frontend.family, family)
                assert frontend.compiled is not None
                self.assertEqual(frontend.compiled.solution.target_value, expected)
                self.assertTrue(frontend.compiled.certificate.verified)
                parsed = parse_structural_problem(question)
                self.assertTrue(parsed.ok, parsed.reason)
                assert parsed.problem is not None
                solution = solve(parsed.problem)
                self.assertIs(solution.status, SolveStatus.UNIQUE)
                self.assertEqual(solution.target_value, expected)
                certified = solver.certify(question)
                self.assertIsNotNone(certified)
                assert certified is not None
                self.assertEqual(certified.answer, str(expected))
                self.assertEqual(
                    certified.evidence["certificates"][0]["kind"],
                    "fraction_rref/v1",
                )

    def test_entities_domains_and_horizons_are_structural(self) -> None:
        variants = (
            (PHONE_TREE.replace("Ali", "Mara"), Fraction(81)),
            (
                MONTHLY_DELTA.replace("students", "workers").replace(
                    "campus", "factory"
                ),
                Fraction(310),
            ),
            (DAILY_TOTAL.replace("Alice", "Mara").replace("clouds", "birds"), 93),
            (
                ORIGINAL_BASE_PERCENTAGE.replace("Tatiana", "Mira").replace(
                    "carrots", "turnips"
                ),
                138,
            ),
            (PHONE_TREE.replace("fourth", "fifth"), 243),
            (PHONE_TREE.replace("fourth", "thirteenth"), 1594323),
            (PHONE_TREE.replace("fourth", "13th"), 1594323),
            (MONTHLY_DELTA.replace("May", "June"), 630),
            (ORIGINAL_BASE_PERCENTAGE.replace("three years", "four years"), 144),
            (ORIGINAL_BASE_PERCENTAGE.replace("three years", "3 years"), 138),
        )
        for question, expected in variants:
            with self.subTest(question=question):
                result = compile_signed_events(question)
                self.assertTrue(result.ok, result.reason)
                assert result.compiled is not None
                self.assertEqual(result.compiled.solution.target_value, expected)

    def test_scope_timing_base_and_target_mutations_abstain(self) -> None:
        mutations = (
            PHONE_TREE.replace("Ali decided", "Mira decided"),
            PHONE_TREE.replace("3 other families", "3 families"),
            PHONE_TREE.replace("during the fourth round", "through the fourth round"),
            PHONE_TREE.replace("fourth", "13rd"),
            PHONE_TREE.replace("fourth", "4097th"),
            PHONE_TREE + " How many relatives were contacted?",
            MONTHLY_DELTA.replace("at the end of every month", "during every month"),
            MONTHLY_DELTA.replace("by the end of May", "by the beginning of May"),
            MONTHLY_DELTA.replace("above and beyond", "together with"),
            DAILY_TOTAL.replace("through Friday", "through Thursday"),
            DAILY_TOTAL.replace("all five days", "all four days"),
            DAILY_TOTAL.replace("number of clouds", "number of birds"),
            ORIGINAL_BASE_PERCENTAGE.replace("original price", "current price"),
            ORIGINAL_BASE_PERCENTAGE.replace("price of carrots", "price of turnips", 1),
            ORIGINAL_BASE_PERCENTAGE.replace("5%", "6%"),
            ORIGINAL_BASE_PERCENTAGE.replace("three years", "13rd years"),
            ORIGINAL_BASE_PERCENTAGE.replace("three years", "13th years"),
        )
        for question in mutations:
            with self.subTest(question=question):
                self.assertFalse(compile_signed_events(question).ok)

    def test_numeric_noise_unknown_factors_and_inequality_without_witness_abstain(
        self,
    ) -> None:
        for question, _, _ in self.CASES:
            with self.subTest(question=question):
                self.assertFalse(
                    compile_signed_events(question + " Unrelated reference 99.").ok
                )
        unknown_factor = PHONE_TREE.replace(
            "each family calls 3 other families",
            "each family calls an unknown number of other families",
        )
        self.assertFalse(compile_signed_events(unknown_factor).ok)
        self.assertFalse(compile_signed_events(WEIGHTED_THRESHOLD).ok)
        self.assertFalse(compile_signed_events(AMBIGUOUS_PERCENTAGE_THRESHOLD).ok)


if __name__ == "__main__":
    unittest.main()
