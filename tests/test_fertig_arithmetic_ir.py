from __future__ import annotations

import unittest
from fractions import Fraction

from immer.cognition.fertig.arithmetic_ir import (
    Affine,
    Assign,
    Balance,
    Mean,
    Part,
    Problem,
    Quantity,
    Rate,
    SolveStatus,
    Span,
    Sum,
    Term,
    Unit,
    Variable,
    solve,
)


COUNT = Unit.count(symbol="items")
MONEY = Unit.base("money", symbol="EUR")
TIME = Unit.base("time", symbol="hour")
DISTANCE = Unit.base("distance", symbol="km")
SPEED = DISTANCE / TIME


def q(value: int | Fraction, unit: Unit = COUNT) -> Quantity:
    return Quantity(Fraction(value), unit)


class ArithmeticIRTests(unittest.TestCase):
    def test_unique_affine_sum_balance_and_exact_certificate(self) -> None:
        apples = Variable("apples", COUNT, count=True)
        pears = Variable("pears", COUNT, count=True)
        total = Variable("total", COUNT, count=True)
        constraints = (
            Assign(apples, q(7), Span(0, 5, "fixture")),
            Affine(pears, apples, 2, q(1), Span(6, 12, "fixture")),
            Sum(total, (apples, pears), Span(13, 20, "fixture")),
            Balance((total,), (q(22),), Span(21, 25, "fixture")),
        )

        result = solve(Problem((apples, pears, total), constraints, total))

        self.assertIs(result.status, SolveStatus.UNIQUE)
        self.assertEqual(result.target_value, 22)
        self.assertEqual(dict(result.values), {"apples": 7, "pears": 15, "total": 22})
        self.assertIsNotNone(result.certificate)
        self.assertTrue(result.certificate.verified)  # type: ignore[union-attr]
        self.assertEqual(
            [residual.value for residual in result.certificate.residuals],  # type: ignore[union-attr]
            [0, 0, 0, 0],
        )
        self.assertEqual(result.value(total), 22)

    def test_rank_deficient_component_is_underdetermined(self) -> None:
        x = Variable("x", COUNT, count=True)
        y = Variable("y", COUNT, count=True)

        result = solve(Problem((x, y), (Sum(q(10), (x, y)),), x))

        self.assertIs(result.status, SolveStatus.UNDERDETERMINED)
        self.assertIsNone(result.target_value)
        self.assertIn("rank 1", result.reason)

    def test_contradictory_component_is_inconsistent(self) -> None:
        x = Variable("x", COUNT, count=True)
        problem = Problem((x,), (Assign(x, q(2)), Assign(x, q(3))), x)

        result = solve(problem)

        self.assertIs(result.status, SolveStatus.INCONSISTENT)
        self.assertIsNone(result.certificate)

    def test_disconnected_contradiction_does_not_poison_target_component(self) -> None:
        target = Variable("target", COUNT, count=True)
        other = Variable("other", COUNT, count=True)
        result = solve(
            Problem(
                (target, other),
                (Assign(target, q(8)), Assign(other, q(1)), Assign(other, q(2))),
                target,
            )
        )

        self.assertIs(result.status, SolveStatus.UNIQUE)
        self.assertEqual(result.target_value, 8)
        self.assertEqual(dict(result.values), {"target": 8})

    def test_units_convert_exactly_and_reject_incompatible_addition(self) -> None:
        seconds = Unit("second", (("time", 1),), Fraction(1))
        minutes = Unit("minute", (("time", 1),), Fraction(60))
        elapsed = Variable("elapsed", seconds)
        exact = solve(Problem((elapsed,), (Assign(elapsed, q(2, minutes)),), elapsed))

        invalid = solve(
            Problem(
                (elapsed,),
                (Assign(elapsed, q(2, DISTANCE)),),
                elapsed,
            )
        )

        self.assertIs(exact.status, SolveStatus.UNIQUE)
        self.assertEqual(exact.target_value, 120)
        self.assertIs(invalid.status, SolveStatus.INVALID)
        self.assertIn("incompatible units", invalid.reason)

    def test_count_domain_is_integer_and_nonnegative(self) -> None:
        count = Variable("count", COUNT, count=True)
        fractional = solve(
            Problem((count,), (Assign(count, q(Fraction(3, 2))),), count)
        )
        negative = solve(Problem((count,), (Assign(count, q(-1)),), count))

        self.assertIs(fractional.status, SolveStatus.INVALID)
        self.assertIn("non-negative integer", fractional.reason)
        self.assertIs(negative.status, SolveStatus.INVALID)
        self.assertIn("non-negative integer", negative.reason)

    def test_rate_allows_one_unknown_factor_and_rejects_unknown_times_unknown(
        self,
    ) -> None:
        distance = Variable("distance", DISTANCE)
        speed = Variable("speed", SPEED)
        duration = Variable("duration", TIME)

        exact = solve(
            Problem(
                (distance,),
                (Rate(distance, q(12, SPEED), q(3, TIME)),),
                distance,
            )
        )
        nonlinear = solve(
            Problem(
                (distance, speed, duration),
                (Rate(distance, speed, duration),),
                distance,
            )
        )
        inferred_speed = solve(
            Problem(
                (distance, speed),
                (
                    Assign(distance, q(36, DISTANCE)),
                    Rate(distance, speed, q(3, TIME)),
                ),
                speed,
            )
        )

        self.assertIs(exact.status, SolveStatus.UNIQUE)
        self.assertEqual(exact.target_value, 36)
        self.assertIs(inferred_speed.status, SolveStatus.UNIQUE)
        self.assertEqual(inferred_speed.target_value, 12)
        self.assertIs(nonlinear.status, SolveStatus.INVALID)
        self.assertIn("unknown * unknown", nonlinear.reason)

    def test_part_and_mean_are_exact_linear_primitives(self) -> None:
        whole = Variable("whole", MONEY)
        part = Variable("part", MONEY)
        mean = Variable("mean", MONEY)
        problem = Problem(
            (whole, part, mean),
            (
                Assign(whole, q(20, MONEY)),
                Part(part, whole, Fraction(3, 10)),
                Mean(mean, (part, q(10, MONEY), q(8, MONEY))),
            ),
            mean,
        )

        result = solve(problem)

        self.assertIs(result.status, SolveStatus.UNIQUE)
        self.assertEqual(result.value(part), 6)
        self.assertEqual(result.target_value, 8)

    def test_weighted_ledger_is_representable_by_balance_terms(self) -> None:
        tickets = Variable("tickets", COUNT, count=True)
        revenue = Variable("revenue", MONEY)
        # A typed ledger can relate weighted quantities only within one unit.  Ticket
        # price is represented by an affine scale after the entity count is known.
        ticket_value = Variable("ticket_value", MONEY)
        result = solve(
            Problem(
                (tickets, ticket_value, revenue),
                (
                    Assign(tickets, q(4)),
                    Assign(ticket_value, q(3, MONEY)),
                    Balance(
                        (revenue,),
                        (Term(ticket_value, 4), q(2, MONEY)),
                    ),
                ),
                revenue,
            )
        )

        self.assertIs(result.status, SolveStatus.UNIQUE)
        self.assertEqual(result.target_value, 14)

    def test_variable_renaming_does_not_change_answer(self) -> None:
        def run(prefix: str) -> Fraction | None:
            a = Variable(f"{prefix}_a", COUNT, count=True)
            b = Variable(f"{prefix}_b", COUNT, count=True)
            return solve(
                Problem(
                    (a, b),
                    (Assign(a, q(9)), Affine(b, a, 3, q(2))),
                    b,
                )
            ).target_value

        self.assertEqual(run("first"), 29)
        self.assertEqual(run("renamed"), 29)

    def test_numeric_perturbation_propagates_instead_of_matching_a_template(
        self,
    ) -> None:
        def run(seed: int) -> Fraction | None:
            x = Variable("x", COUNT, count=True)
            y = Variable("y", COUNT, count=True)
            return solve(
                Problem((x, y), (Assign(x, q(seed)), Affine(y, x, 2, q(1))), y)
            ).target_value

        self.assertEqual(run(5), 11)
        self.assertEqual(run(8), 17)

    def test_constraint_order_does_not_change_solution(self) -> None:
        x = Variable("x", COUNT, count=True)
        y = Variable("y", COUNT, count=True)
        constraints = (Assign(x, q(4)), Affine(y, x, 5, q(2)))

        forward = solve(Problem((x, y), constraints, y))
        reversed_result = solve(Problem((x, y), tuple(reversed(constraints)), y))

        self.assertEqual(forward.target_value, 22)
        self.assertEqual(reversed_result.target_value, 22)
        self.assertTrue(forward.certificate.verified)  # type: ignore[union-attr]
        self.assertTrue(reversed_result.certificate.verified)  # type: ignore[union-attr]

    def test_duplicate_or_undeclared_variables_fail_closed(self) -> None:
        x = Variable("x", COUNT, count=True)
        alias = Variable("x", COUNT, count=True)
        missing = Variable("missing", COUNT, count=True)

        duplicate = solve(Problem((x, alias), (Assign(x, q(1)),), x))
        undeclared = solve(Problem((x,), (Sum(x, (missing, q(1))),), x))

        self.assertIs(duplicate.status, SolveStatus.INVALID)
        self.assertIn("duplicate variable", duplicate.reason)
        self.assertIs(undeclared.status, SolveStatus.INVALID)
        self.assertIn("undeclared", undeclared.reason)

    def test_empty_mean_and_malformed_span_are_rejected(self) -> None:
        x = Variable("x", COUNT, count=True)

        result = solve(Problem((x,), (Mean(x, ()),), x))

        self.assertIs(result.status, SolveStatus.INVALID)
        self.assertIn("at least one", result.reason)
        with self.assertRaises(ValueError):
            Span(3, 2)


if __name__ == "__main__":
    unittest.main()
