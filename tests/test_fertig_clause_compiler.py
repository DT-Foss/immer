from __future__ import annotations

from dataclasses import FrozenInstanceError
from fractions import Fraction
import unittest

from immer.cognition.fertig.arithmetic_ir import SolveStatus, Span, solve
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


if __name__ == "__main__":
    unittest.main()
