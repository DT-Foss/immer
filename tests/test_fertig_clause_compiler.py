from __future__ import annotations

from dataclasses import FrozenInstanceError
from fractions import Fraction
import unittest

from immer.cognition.fertig.arithmetic_ir import Rate, SolveStatus, Span, Sum, solve
from immer.cognition.fertig.clause_compiler import (
    CompileStatus,
    EvidenceLedger,
    NumericMention,
    SymbolKey,
    compile_clauses,
)


DEV64 = (
    "Becca, Smendrick, and PJ have collections of Magic Cards.  "
    "There is a total of 341 cards.  "
    "Becca has 12 more than Smendrick, and Smendrick has 3 times the "
    "amount of cards that PJ has.  How many cards does Becca have?"
)
DEV64_RATE_LEDGER = (
    "At the local Pick Your Own fruit orchard, you could pick your own peaches "
    "for $2.00 per pound, plums were $1.00 per pound and apricots were $3.00 "
    "per pound.  If Winston picked 6 pounds of peaches, 8 pounds of plums and "
    "6 pounds of apricots, how much did he spend on fruit?"
)


def _solution(source: str):
    compiled = compile_clauses(source)
    if not compiled.ok or compiled.problem is None:
        raise AssertionError(
            f"compile failed: {compiled.status.value}: {compiled.reason}"
        )
    return compiled, solve(compiled.problem)


class EvidenceLedgerTests(unittest.TestCase):
    def test_discovers_exact_numeric_surfaces_and_requires_one_consumption(
        self,
    ) -> None:
        source = "A has -1.25 widgets, B has 1/2 as many, and C has 2,000."
        ledger = EvidenceLedger(source)

        self.assertEqual(
            [mention.value for mention in ledger.mentions],
            [Fraction(-5, 4), Fraction(1, 2), Fraction(2000)],
        )
        self.assertEqual(
            [mention.text for mention in ledger.mentions], ["-1.25", "1/2", "2,000"]
        )
        self.assertFalse(ledger.closed)
        for mention in ledger.mentions:
            self.assertTrue(ledger.consume(mention))
        self.assertTrue(ledger.closed)

        self.assertFalse(ledger.consume(ledger.mentions[0]))
        self.assertEqual(ledger.duplicates, (ledger.mentions[0],))
        self.assertFalse(ledger.closed)

    def test_required_surface_types_are_immutable(self) -> None:
        span = Span(0, 1, "7")
        mention = NumericMention(Fraction(7), "items", span)
        symbol = SymbolKey("Mina", "count", "item", "collection", "current")

        with self.assertRaises(FrozenInstanceError):
            mention.value = Fraction(8)  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            symbol.owner = "Nora"  # type: ignore[misc]


class AffineClauseCompilerTests(unittest.TestCase):
    def test_dev64_item_compiles_without_gold_and_has_exact_certificate(self) -> None:
        compiled, solution = _solution(DEV64)

        self.assertIs(solution.status, SolveStatus.UNIQUE)
        self.assertEqual(solution.target_value, Fraction(153))
        self.assertIsNotNone(solution.certificate)
        self.assertTrue(solution.certificate.verified)  # type: ignore[union-attr]
        self.assertEqual(
            [mention.text for mention in compiled.diagnostics.observed_numeric],
            ["341", "12", "3"],
        )
        self.assertFalse(compiled.diagnostics.unconsumed_numeric)
        assert compiled.target is not None
        self.assertEqual(compiled.target.symbol.owner, "Becca")
        self.assertEqual(compiled.target.symbol.item, "magic card")

    def test_name_and_number_variants_use_the_same_affine_family(self) -> None:
        first = (
            "Aria, Bex, and Cy have collections of silver tokens. "
            "There is a total of 108 tokens. Aria has 8 more than Bex, and "
            "Bex has 2 times as many tokens as Cy. How many tokens does Aria have?"
        )
        perturbed = (
            "Nia, Oren, and Paz have collections of silver tokens. "
            "There is a total of 171 tokens. Nia has 9 more than Oren, and "
            "Oren has 4 times as many tokens as Paz. How many tokens does Nia have?"
        )

        self.assertEqual(_solution(first)[1].target_value, 48)
        self.assertEqual(_solution(perturbed)[1].target_value, 81)

    def test_assignments_and_additive_direction_lower_to_exact_ir(self) -> None:
        more = (
            "Lena has 14 marbles. Omar has 3 more marbles than Lena. "
            "How many marbles does Omar have?"
        )
        fewer = (
            "Lena has 14 marbles. Omar has 3 fewer marbles than Lena. "
            "How many marbles does Omar have?"
        )

        self.assertEqual(_solution(more)[1].target_value, 17)
        self.assertEqual(_solution(fewer)[1].target_value, 11)

    def test_source_divided_by_constant_is_a_reusable_affine_clause(self) -> None:
        source = (
            "Ana, Bo, and Cy have collections of marbles. "
            "There is a total of 55 marbles. Ana has Bo's marbles divided by 2. "
            "Bo has 3 times as many marbles as Cy. How many marbles does Ana have?"
        )

        compiled, solution = _solution(source)

        self.assertEqual(solution.target_value, 15)
        scales = [
            relation.args[2]
            for relation in compiled.relations
            if relation.kind == "affine"
        ]
        self.assertIn(Fraction(1, 2), scales)

    def test_entity_or_item_drift_abstains_as_ambiguous(self) -> None:
        entity_drift = compile_clauses(DEV64.replace("does Becca", "does Nora"))
        item_drift = compile_clauses(DEV64.replace("How many cards", "How many shells"))

        self.assertIs(entity_drift.status, CompileStatus.AMBIGUOUS)
        self.assertIsNone(entity_drift.problem)
        self.assertIs(item_drift.status, CompileStatus.AMBIGUOUS)
        self.assertIsNone(item_drift.problem)

    def test_unconsumed_extra_number_fails_closed(self) -> None:
        result = compile_clauses(DEV64 + " Reference 2024.")

        self.assertIs(result.status, CompileStatus.UNSUPPORTED)
        self.assertIsNone(result.problem)
        self.assertEqual(
            [mention.value for mention in result.diagnostics.unconsumed_numeric],
            [Fraction(2024)],
        )

    def test_underdetermined_total_remains_an_exact_nonunique_ir(self) -> None:
        source = (
            "Ava and Bo have collections of marbles. "
            "There is a total of 30 marbles. How many marbles does Ava have?"
        )
        compiled = compile_clauses(source)

        self.assertTrue(compiled.ok, compiled.reason)
        assert compiled.problem is not None
        solution = solve(compiled.problem)
        self.assertIs(solution.status, SolveStatus.UNDERDETERMINED)
        self.assertIsNone(solution.target_value)
        self.assertIsNone(solution.certificate)

    def test_multiple_or_missing_explicit_targets_abstain(self) -> None:
        missing = compile_clauses(DEV64.rsplit("  ", 1)[0] + ".")
        multiple = compile_clauses(DEV64 + " How many cards does PJ have?")

        self.assertIs(missing.status, CompileStatus.UNSUPPORTED)
        self.assertIsNone(missing.problem)
        self.assertIs(multiple.status, CompileStatus.AMBIGUOUS)
        self.assertIsNone(multiple.problem)


class RateLedgerClauseCompilerTests(unittest.TestCase):
    def test_dev64_rate_ledger_solves_through_rate_and_sum_ir(self) -> None:
        compiled, solution = _solution(DEV64_RATE_LEDGER)

        self.assertIs(solution.status, SolveStatus.UNIQUE)
        self.assertEqual(solution.target_value, Fraction(38))
        self.assertIsNotNone(solution.certificate)
        self.assertTrue(solution.certificate.verified)  # type: ignore[union-attr]
        assert compiled.problem is not None
        self.assertEqual(
            sum(
                isinstance(constraint, Rate)
                for constraint in compiled.problem.constraints
            ),
            3,
        )
        self.assertIsInstance(compiled.problem.constraints[-1], Sum)
        self.assertEqual(
            [relation.kind for relation in compiled.relations],
            ["unit_rate"] * 3 + ["acquire"] * 3,
        )
        self.assertEqual(
            [mention.text for mention in compiled.diagnostics.observed_numeric],
            ["2.00", "1.00", "3.00", "6", "8", "6"],
        )

    def test_names_items_and_exact_prices_are_perturbable(self) -> None:
        first = (
            "At the hardware store, rivets cost $1.25 per box and washers cost $2.50 "
            "per box. Mara bought 4 boxes of rivets and 2 boxes of washers. "
            "How much did Mara spend on hardware?"
        )
        perturbed = (
            "At the supplies store, beads cost $0.75 per bag and clasps cost $1.50 per "
            "bag. Niko purchased 8 bags of beads and 4 bags of clasps. "
            "How much did Niko spend on supplies?"
        )

        self.assertEqual(_solution(first)[1].target_value, Fraction(10))
        self.assertEqual(_solution(perturbed)[1].target_value, Fraction(12))

    def test_missing_price_abstains_without_partial_ir(self) -> None:
        source = DEV64_RATE_LEDGER.replace(" and apricots were $3.00 per pound", "")
        result = compile_clauses(source)

        self.assertIs(result.status, CompileStatus.UNSUPPORTED)
        self.assertIsNone(result.problem)
        self.assertIn("missing unit price for apricot", result.reason)

    def test_duplicate_item_abstains_as_ambiguous(self) -> None:
        source = DEV64_RATE_LEDGER.replace("plums were", "peaches were")
        result = compile_clauses(source)

        self.assertIs(result.status, CompileStatus.AMBIGUOUS)
        self.assertIsNone(result.problem)
        self.assertIn("duplicate unit price for peach", result.reason)

    def test_cost_target_entity_drift_abstains(self) -> None:
        source = DEV64_RATE_LEDGER.replace("did he spend", "did Nora spend")
        result = compile_clauses(source)

        self.assertIs(result.status, CompileStatus.AMBIGUOUS)
        self.assertIsNone(result.problem)
        self.assertIn("does not match the acquisition buyer", result.reason)

    def test_cost_target_item_drift_abstains(self) -> None:
        source = DEV64_RATE_LEDGER.replace("spend on fruit", "spend on bananas")
        result = compile_clauses(source)

        self.assertIs(result.status, CompileStatus.AMBIGUOUS)
        self.assertIsNone(result.problem)
        self.assertIn("not bound to a line item or explicit rate group", result.reason)

    def test_exact_line_item_target_returns_only_its_subtotal(self) -> None:
        source = (
            "At the grocery store, apples cost $2 per pound and pears cost $3 "
            "per pound. Mia bought 2 pounds of apples and 1 pound of pears. "
            "How much did Mia spend on apples?"
        )
        compiled, solution = _solution(source)

        self.assertIs(solution.status, SolveStatus.UNIQUE)
        self.assertEqual(solution.target_value, Fraction(4))
        self.assertIsNotNone(solution.certificate)
        self.assertTrue(solution.certificate.verified)  # type: ignore[union-attr]
        assert compiled.problem is not None
        self.assertFalse(
            any(isinstance(item, Sum) for item in compiled.problem.constraints)
        )

    def test_unattested_group_target_abstains(self) -> None:
        source = (
            "At the market, rivets cost $1.25 per box and washers cost $2.50 "
            "per box. Mara bought 4 boxes of rivets and 2 boxes of washers. "
            "How much did Mara spend on hardware?"
        )
        result = compile_clauses(source)

        self.assertIs(result.status, CompileStatus.AMBIGUOUS)
        self.assertIsNone(result.problem)

    def test_prices_from_multiple_local_scopes_do_not_mix(self) -> None:
        source = (
            "At the fruit store, apples cost $2 per pound. "
            "At the fruit market, pears cost $3 per pound. "
            "Mia bought 2 pounds of apples and 1 pound of pears. "
            "How much did Mia spend on fruit?"
        )
        result = compile_clauses(source)

        self.assertIs(result.status, CompileStatus.AMBIGUOUS)
        self.assertIsNone(result.problem)
        self.assertIn("multiple local rate scopes", result.reason)

    def test_repeated_identical_explicit_scope_can_span_price_clauses(self) -> None:
        source = (
            "At the fruit store, apples cost $2 per pound. "
            "At the fruit store, pears cost $3 per pound. "
            "Mia bought 2 pounds of apples and 1 pound of pears. "
            "How much did Mia spend on fruit?"
        )
        compiled, solution = _solution(source)

        self.assertIs(solution.status, SolveStatus.UNIQUE)
        self.assertEqual(solution.target_value, Fraction(7))
        self.assertIsNotNone(solution.certificate)
        self.assertTrue(solution.certificate.verified)  # type: ignore[union-attr]
        assert compiled.problem is not None
        self.assertEqual(
            sum(isinstance(item, Rate) for item in compiled.problem.constraints), 2
        )
        self.assertIsInstance(compiled.problem.constraints[-1], Sum)

    def test_unconsumed_extra_number_abstains_with_evidence(self) -> None:
        result = compile_clauses(DEV64_RATE_LEDGER + " Reference 2026.")

        self.assertIs(result.status, CompileStatus.UNSUPPORTED)
        self.assertIsNone(result.problem)
        self.assertEqual(
            [mention.value for mention in result.diagnostics.unconsumed_numeric],
            [Fraction(2026)],
        )

    def test_incompatible_rate_and_quantity_units_abstain(self) -> None:
        source = DEV64_RATE_LEDGER.replace(
            "6 pounds of apricots", "6 baskets of apricots"
        )
        result = compile_clauses(source)

        self.assertIs(result.status, CompileStatus.INVALID)
        self.assertIsNone(result.problem)
        self.assertIn("incompatible units for apricot", result.reason)


if __name__ == "__main__":
    unittest.main()
