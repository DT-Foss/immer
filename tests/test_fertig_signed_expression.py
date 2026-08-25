from __future__ import annotations

from dataclasses import FrozenInstanceError
from fractions import Fraction
import unittest

from immer.cognition.fertig.arithmetic_ir import (
    Assign,
    Mean as IRMean,
    Rate,
    SolveStatus,
    Span,
    Sum,
    Unit,
)
from immer.cognition.fertig.clause_compiler import SymbolKey
from immer.cognition.fertig.signed_expression import (
    Definition,
    ExpressionCompileError,
    ExpressionProgram,
    ExpressionTarget,
    LiteralExpr,
    MeanExpr,
    NumericEvidence,
    ProductExpr,
    QuotientExpr,
    RefExpr,
    SignedTerm,
    SumExpr,
    compile_expression,
)


COUNT = Unit.count()
SCALAR = Unit.scalar()
MONEY = Unit.base("money", symbol="USD")
PRICE = MONEY / COUNT
SECOND = Unit.base("time", symbol="s")
MINUTE = Unit("min", (("time", 1),), Fraction(60))


def _key(name: str, property_name: str = "quantity") -> SymbolKey:
    return SymbolKey(name, property_name, "item", "scope", "current")


class _Builder:
    def __init__(self) -> None:
        self.evidence: list[NumericEvidence] = []

    def literal(
        self, value: int | Fraction, unit: Unit = COUNT, *, evidence: bool = True
    ) -> LiteralExpr:
        exact = Fraction(value)
        source = str(exact)
        span = Span(0, len(source), source)
        if not evidence:
            return LiteralExpr(exact, unit, span)
        evidence_id = f"number-{len(self.evidence):03d}"
        self.evidence.append(NumericEvidence(evidence_id, exact, unit, span))
        return LiteralExpr(exact, unit, span, (evidence_id,))

    def program(
        self,
        expr,
        *,
        definitions: tuple[Definition, ...] = (),
        target: SymbolKey | None = None,
    ) -> ExpressionProgram:
        span = Span(0, 0)
        return ExpressionProgram(
            definitions,
            ExpressionTarget(target or _key("answer"), expr, span),
            tuple(self.evidence),
        )


def _signed(sign: int, expr, role: str = "contribution") -> SignedTerm:
    return SignedTerm(sign, expr, role, expr.span)


def _sum(*terms: SignedTerm) -> SumExpr:
    return SumExpr(tuple(terms), Span(0, 0))


def _compile(builder: _Builder, expr, **kwargs):
    return compile_expression(builder.program(expr, **kwargs))


class CanonicalExpressionTests(unittest.TestCase):
    def test_actual_twelve_mapped_expressions_plus_mean63_are_exact(self) -> None:
        cases = []

        builder = _Builder()  # 8: 194 * (490 - 150)
        distance = _sum(
            _signed(1, builder.literal(490)),
            _signed(-1, builder.literal(150)),
        )
        cases.append(
            (
                8,
                builder,
                ProductExpr((builder.literal(194, SCALAR), distance), Span(0, 0)),
                Fraction(65960),
            )
        )

        builder = _Builder()  # 10: 4 * 6 * 4 - 20
        lights = ProductExpr(
            (
                builder.literal(4, SCALAR),
                builder.literal(6, SCALAR),
                builder.literal(4),
            ),
            Span(0, 0),
        )
        cases.append(
            (
                10,
                builder,
                _sum(_signed(1, lights), _signed(-1, builder.literal(20))),
                Fraction(76),
            )
        )

        builder = _Builder()  # 13: (20 + 10) * 28
        daily = _sum(_signed(1, builder.literal(20)), _signed(1, builder.literal(10)))
        cases.append(
            (
                13,
                builder,
                ProductExpr((daily, builder.literal(28, SCALAR)), Span(0, 0)),
                Fraction(840),
            )
        )

        builder = _Builder()  # 22: 450 - 120 - 3 * 25
        asphalt = ProductExpr(
            (builder.literal(3, SCALAR), builder.literal(25)), Span(0, 0)
        )
        cases.append(
            (
                22,
                builder,
                _sum(
                    _signed(1, builder.literal(450)),
                    _signed(-1, builder.literal(120)),
                    _signed(-1, asphalt),
                ),
                Fraction(255),
            )
        )

        builder = _Builder()  # 25: 4000 + 1000 + 30 * 50 - 3 * 1800
        cards = ProductExpr(
            (builder.literal(30, SCALAR), builder.literal(50)), Span(0, 0)
        )
        packs = ProductExpr(
            (builder.literal(3, SCALAR), builder.literal(1800)), Span(0, 0)
        )
        cases.append(
            (
                25,
                builder,
                _sum(
                    _signed(1, builder.literal(4000)),
                    _signed(1, builder.literal(1000)),
                    _signed(1, cards),
                    _signed(-1, packs),
                ),
                Fraction(1100),
            )
        )

        builder = _Builder()  # 26: 2*26 + 2*12 - 2*14 - 2*10
        first_park = _sum(
            _signed(
                1,
                ProductExpr(
                    (builder.literal(2, SCALAR), builder.literal(26)), Span(0, 0)
                ),
            ),
            _signed(
                1,
                ProductExpr(
                    (builder.literal(2, SCALAR), builder.literal(12)), Span(0, 0)
                ),
            ),
        )
        second_park = _sum(
            _signed(
                1,
                ProductExpr(
                    (builder.literal(2, SCALAR), builder.literal(14)), Span(0, 0)
                ),
            ),
            _signed(
                1,
                ProductExpr(
                    (builder.literal(2, SCALAR), builder.literal(10)), Span(0, 0)
                ),
            ),
        )
        cases.append(
            (
                26,
                builder,
                _sum(_signed(1, first_park), _signed(-1, second_park)),
                Fraction(28),
            )
        )

        builder = _Builder()  # 27: 500 * 7 * (1/2 - 2/5)
        saving = _sum(
            _signed(1, builder.literal(Fraction(1, 2), SCALAR)),
            _signed(-1, builder.literal(Fraction(2, 5), SCALAR)),
        )
        cases.append(
            (
                27,
                builder,
                ProductExpr(
                    (builder.literal(500), builder.literal(7, SCALAR), saving),
                    Span(0, 0),
                ),
                Fraction(350),
            )
        )

        builder = _Builder()  # 30: 4 * 5 - 8 * 2
        vanilla = ProductExpr(
            (builder.literal(4, SCALAR), builder.literal(5)), Span(0, 0)
        )
        fruity = ProductExpr(
            (builder.literal(8, SCALAR), builder.literal(2)), Span(0, 0)
        )
        cases.append(
            (30, builder, _sum(_signed(1, vanilla), _signed(-1, fruity)), Fraction(4))
        )

        builder = _Builder()  # 34: 2 * 150 + 240
        deck = ProductExpr(
            (builder.literal(2, SCALAR), builder.literal(150)), Span(0, 0)
        )
        cases.append(
            (
                34,
                builder,
                _sum(_signed(1, deck), _signed(1, builder.literal(240))),
                Fraction(540),
            )
        )

        builder = _Builder()  # 37: (175000 - 50000) / 20
        student_total = _sum(
            _signed(1, builder.literal(175000)),
            _signed(-1, builder.literal(50000)),
        )
        cases.append(
            (
                37,
                builder,
                QuotientExpr(student_total, builder.literal(20, SCALAR), Span(0, 0)),
                Fraction(6250),
            )
        )

        builder = _Builder()  # 53: (11 - 6 - (1/2)*6) / 2
        yogurt = ProductExpr(
            (
                builder.literal(Fraction(1, 2), SCALAR),
                builder.literal(6),
            ),
            Span(0, 0),
        )
        carrot_total = _sum(
            _signed(1, builder.literal(11)),
            _signed(-1, builder.literal(6)),
            _signed(-1, yogurt),
        )
        cases.append(
            (
                53,
                builder,
                QuotientExpr(carrot_total, builder.literal(2, SCALAR), Span(0, 0)),
                Fraction(1),
            )
        )

        builder = _Builder()  # 61: 62 - 31 - 8 - 9
        cases.append(
            (
                61,
                builder,
                _sum(
                    _signed(1, builder.literal(62)),
                    _signed(-1, builder.literal(31)),
                    _signed(-1, builder.literal(8)),
                    _signed(-1, builder.literal(9)),
                ),
                Fraction(14),
            )
        )

        builder = _Builder()  # Mean63: mean(40*10, 35*12)
        zoey = ProductExpr(
            (builder.literal(40, SCALAR), builder.literal(10)), Span(0, 0)
        )
        sydney = ProductExpr(
            (builder.literal(35, SCALAR), builder.literal(12)), Span(0, 0)
        )
        cases.append((63, builder, MeanExpr((zoey, sydney), Span(0, 0)), Fraction(410)))

        for case_id, case_builder, expr, expected in cases:
            with self.subTest(case_id=case_id):
                result = _compile(case_builder, expr)
                self.assertIs(result.solution.status, SolveStatus.UNIQUE)
                self.assertEqual(result.solution.target_value, expected)
                self.assertTrue(result.certificate.verified)
                self.assertEqual(
                    len(result.evidence_projection), len(case_builder.evidence)
                )

    def test_definition_dag_is_bound_and_solved_exactly(self) -> None:
        builder = _Builder()
        a = _key("a")
        b = _key("b")
        span = Span(0, 0)
        definitions = (
            Definition(a, builder.literal(10), span),
            Definition(
                b,
                _sum(_signed(1, RefExpr(a, span)), _signed(1, builder.literal(5))),
                span,
            ),
        )
        target = _sum(_signed(1, RefExpr(b, span)), _signed(-1, builder.literal(3)))

        result = _compile(builder, target, definitions=definitions)

        self.assertEqual(result.solution.target_value, 12)
        self.assertEqual(result.solution.value(result.symbol_variables[a]), 10)
        self.assertEqual(result.solution.value(result.symbol_variables[b]), 15)
        self.assertEqual(
            result.certificate.variable_count, len(result.problem.variables)
        )

    def test_bound_refs_flow_through_product_quotient_and_mean_constraints(
        self,
    ) -> None:
        builder = _Builder()
        span = Span(0, 0)
        product_symbol = _key("product")
        quotient_symbol = _key("quotient")
        mean_symbol = _key("mean")
        definitions = (
            Definition(
                product_symbol,
                ProductExpr((builder.literal(6), builder.literal(4, SCALAR)), span),
                span,
            ),
            Definition(
                quotient_symbol,
                QuotientExpr(
                    RefExpr(product_symbol, span),
                    builder.literal(3, SCALAR),
                    span,
                ),
                span,
            ),
            Definition(
                mean_symbol,
                MeanExpr((RefExpr(quotient_symbol, span), builder.literal(10)), span),
                span,
            ),
        )

        result = _compile(builder, RefExpr(mean_symbol, span), definitions=definitions)

        self.assertEqual(result.solution.target_value, 9)
        self.assertEqual(
            result.solution.value(result.symbol_variables[product_symbol]), 24
        )
        self.assertEqual(
            result.solution.value(result.symbol_variables[quotient_symbol]), 8
        )
        self.assertEqual(result.solution.value(result.symbol_variables[mean_symbol]), 9)
        self.assertEqual(
            [type(constraint) for constraint in result.problem.constraints],
            [Rate, Assign, Rate, Assign, IRMean, Assign, Assign],
        )
        self.assertTrue(result.certificate.verified)

    def test_nested_operations_are_constraints_not_pre_evaluated_assignment(
        self,
    ) -> None:
        builder = _Builder()
        ledger = _sum(_signed(1, builder.literal(8)), _signed(-1, builder.literal(2)))
        expr = ProductExpr((ledger, builder.literal(4, SCALAR)), Span(0, 0))

        result = _compile(builder, expr)

        self.assertEqual(result.solution.target_value, 24)
        self.assertEqual(
            [type(constraint) for constraint in result.problem.constraints],
            [Sum, Rate, Assign],
        )

    def test_permuted_commutative_expressions_keep_the_exact_value(self) -> None:
        first = _Builder()
        first_expr = ProductExpr(
            (
                first.literal(2, SCALAR),
                first.literal(7),
                first.literal(3, SCALAR),
            ),
            Span(0, 0),
        )
        second = _Builder()
        second_expr = ProductExpr(
            (
                second.literal(3, SCALAR),
                second.literal(2, SCALAR),
                second.literal(7),
            ),
            Span(0, 0),
        )
        self.assertEqual(_compile(first, first_expr).solution.target_value, 42)
        self.assertEqual(_compile(second, second_expr).solution.target_value, 42)

    def test_auxiliary_names_and_operation_spans_are_deterministic(self) -> None:
        builder = _Builder()
        operation_span = Span(4, 9, "xxxx10xxx")
        expr = SumExpr(
            (_signed(1, builder.literal(2)), _signed(1, builder.literal(8))),
            operation_span,
        )
        program = builder.program(expr)

        first = compile_expression(program)
        second = compile_expression(program)

        self.assertEqual(first.problem, second.problem)
        auxiliaries = [
            variable
            for variable in first.problem.variables
            if variable.name.startswith("__")
        ]
        self.assertEqual(
            [variable.name for variable in auxiliaries], ["__signed_expr_0000"]
        )
        self.assertEqual(auxiliaries[0].span, operation_span)
        self.assertEqual(first.problem.constraints[0].span, operation_span)

    def test_ast_and_result_views_are_immutable(self) -> None:
        builder = _Builder()
        literal = builder.literal(4)
        result = _compile(builder, literal)
        with self.assertRaises(FrozenInstanceError):
            literal.value = Fraction(5)  # type: ignore[misc]
        with self.assertRaises(TypeError):
            result.symbol_variables[_key("intruder")] = result.problem.target  # type: ignore[index]


class ExpressionRejectionTests(unittest.TestCase):
    def test_duplicate_definition_and_target_definition_are_rejected(self) -> None:
        span = Span(0, 0)
        symbol = _key("x")
        builder = _Builder()
        first = Definition(symbol, builder.literal(1), span)
        second = Definition(symbol, builder.literal(2), span)
        with self.assertRaisesRegex(ExpressionCompileError, "duplicate definition"):
            _compile(
                builder,
                RefExpr(symbol, span),
                definitions=(first, second),
            )

        builder = _Builder()
        definition = Definition(symbol, builder.literal(1), span)
        program = builder.program(
            builder.literal(1), definitions=(definition,), target=symbol
        )
        with self.assertRaisesRegex(ExpressionCompileError, "must not also"):
            compile_expression(program)

    def test_undefined_cycles_and_disconnected_definitions_are_rejected(self) -> None:
        span = Span(0, 0)
        a, b, absent = _key("a"), _key("b"), _key("absent")

        with self.assertRaisesRegex(ExpressionCompileError, "undefined references"):
            _compile(_Builder(), RefExpr(absent, span))

        cyclic = ExpressionProgram(
            (
                Definition(a, RefExpr(b, span), span),
                Definition(b, RefExpr(a, span), span),
            ),
            ExpressionTarget(_key("answer"), RefExpr(a, span), span),
            (),
        )
        with self.assertRaisesRegex(ExpressionCompileError, "cyclic"):
            compile_expression(cyclic)

        builder = _Builder()
        disconnected = Definition(a, builder.literal(9), span)
        with self.assertRaisesRegex(ExpressionCompileError, "disconnected"):
            _compile(builder, builder.literal(4), definitions=(disconnected,))

    def test_evidence_must_be_present_consumed_once_and_value_bound(self) -> None:
        span = Span(0, 1, "7")
        literal = LiteralExpr(7, COUNT, span, ("missing",))
        with self.assertRaisesRegex(ExpressionCompileError, "missing evidence"):
            compile_expression(
                ExpressionProgram(
                    (), ExpressionTarget(_key("answer"), literal, span), ()
                )
            )

        evidence = NumericEvidence("n", 7, COUNT, span)
        duplicated = SumExpr(
            (
                _signed(1, LiteralExpr(7, COUNT, span, ("n",))),
                _signed(1, LiteralExpr(7, COUNT, span, ("n",))),
            ),
            span,
        )
        with self.assertRaisesRegex(ExpressionCompileError, "more than once"):
            compile_expression(
                ExpressionProgram(
                    (), ExpressionTarget(_key("answer"), duplicated, span), (evidence,)
                )
            )

        builder = _Builder()
        extra = NumericEvidence("extra", 3, COUNT, span)
        program = ExpressionProgram(
            (),
            ExpressionTarget(_key("answer"), builder.literal(2), span),
            (*builder.evidence, extra),
        )
        with self.assertRaisesRegex(ExpressionCompileError, "unconsumed"):
            compile_expression(program)

        mutated = LiteralExpr(8, COUNT, span, ("n",))
        with self.assertRaisesRegex(ExpressionCompileError, "value does not match"):
            compile_expression(
                ExpressionProgram(
                    (), ExpressionTarget(_key("answer"), mutated, span), (evidence,)
                )
            )

        wrong_span = NumericEvidence("n", 7, COUNT, Span(1, 2, "07"))
        bound = LiteralExpr(7, COUNT, span, ("n",))
        with self.assertRaisesRegex(ExpressionCompileError, "span does not match"):
            compile_expression(
                ExpressionProgram(
                    (), ExpressionTarget(_key("answer"), bound, span), (wrong_span,)
                )
            )

    def test_every_literal_requires_exactly_one_evidence_id(self) -> None:
        span = Span(0, 1, "2")
        zero_ids = LiteralExpr(2, COUNT, span)
        with self.assertRaisesRegex(ExpressionCompileError, "exactly one evidence"):
            compile_expression(
                ExpressionProgram(
                    (), ExpressionTarget(_key("answer"), zero_ids, span), ()
                )
            )

        combined = LiteralExpr(999, COUNT, Span(0, 3, "999"), ("two", "three"))
        evidence = (
            NumericEvidence("two", 2, COUNT, Span(0, 1, "2")),
            NumericEvidence("three", 3, COUNT, Span(0, 1, "3")),
        )
        with self.assertRaisesRegex(ExpressionCompileError, "exactly one evidence"):
            compile_expression(
                ExpressionProgram(
                    (),
                    ExpressionTarget(_key("answer"), combined, combined.span),
                    evidence,
                )
            )

    def test_sum_and_mean_reject_incompatible_units(self) -> None:
        for make_expr in (
            lambda builder: _sum(
                _signed(1, builder.literal(2, COUNT)),
                _signed(1, builder.literal(3, MONEY)),
            ),
            lambda builder: MeanExpr(
                (builder.literal(2, COUNT), builder.literal(3, MONEY)), Span(0, 0)
            ),
        ):
            builder = _Builder()
            with self.subTest(make_expr=make_expr):
                with self.assertRaisesRegex(ExpressionCompileError, "incompatible"):
                    _compile(builder, make_expr(builder))

    def test_unknown_times_unknown_is_rejected_even_when_definitions_are_ground(
        self,
    ) -> None:
        builder = _Builder()
        span = Span(0, 0)
        a, b = _key("a"), _key("b")
        definitions = (
            Definition(a, builder.literal(2, SCALAR), span),
            Definition(b, builder.literal(3, SCALAR), span),
        )
        expr = ProductExpr((RefExpr(a, span), RefExpr(b, span)), span)
        with self.assertRaisesRegex(ExpressionCompileError, r"unknown \* unknown"):
            _compile(builder, expr, definitions=definitions)

    def test_quotient_requires_a_nonzero_ground_literal(self) -> None:
        builder = _Builder()
        numerator = builder.literal(10)
        zero = builder.literal(0, SCALAR)
        with self.assertRaisesRegex(ExpressionCompileError, "nonzero"):
            _compile(builder, QuotientExpr(numerator, zero, Span(0, 0)))

        with self.assertRaisesRegex(TypeError, "ground LiteralExpr"):
            QuotientExpr(numerator, RefExpr(_key("x"), Span(0, 0)), Span(0, 0))  # type: ignore[arg-type]

    def test_count_literals_and_solutions_must_be_nonnegative_integers(self) -> None:
        for invalid in (Fraction(1, 2), Fraction(-1)):
            builder = _Builder()
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ExpressionCompileError, "count literal"):
                    _compile(builder, builder.literal(invalid))

        builder = _Builder()
        negative = _sum(_signed(1, builder.literal(2)), _signed(-1, builder.literal(3)))
        with self.assertRaisesRegex(ExpressionCompileError, "no unique exact solution"):
            _compile(builder, negative)

        builder = _Builder()
        fractional_mean = MeanExpr((builder.literal(1), builder.literal(2)), Span(0, 0))
        with self.assertRaisesRegex(ExpressionCompileError, "no unique exact solution"):
            _compile(builder, fractional_mean)

    def test_sign_and_term_mutations_fail_closed(self) -> None:
        builder = _Builder()
        literal = builder.literal(2)
        with self.assertRaisesRegex(ValueError, r"exactly -1 or \+1"):
            SignedTerm(0, literal, "contribution", literal.span)

        malformed = SumExpr((literal,), Span(0, 0))  # type: ignore[arg-type]
        with self.assertRaisesRegex(ExpressionCompileError, "not SignedTerm"):
            _compile(builder, malformed)

    def test_duplicate_numeric_evidence_ids_and_unit_mutation_fail(self) -> None:
        span = Span(0, 1, "2")
        duplicate = (
            NumericEvidence("n", 2, COUNT, span),
            NumericEvidence("n", 2, COUNT, span),
        )
        literal = LiteralExpr(2, COUNT, span, ("n",))
        with self.assertRaisesRegex(ExpressionCompileError, "duplicate numeric"):
            compile_expression(
                ExpressionProgram(
                    (), ExpressionTarget(_key("answer"), literal, span), duplicate
                )
            )

        wrong_unit = NumericEvidence("n", 2, MONEY, span)
        with self.assertRaisesRegex(ExpressionCompileError, "unit does not match"):
            compile_expression(
                ExpressionProgram(
                    (), ExpressionTarget(_key("answer"), literal, span), (wrong_unit,)
                )
            )


if __name__ == "__main__":
    unittest.main()
