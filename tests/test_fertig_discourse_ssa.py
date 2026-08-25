from __future__ import annotations

from fractions import Fraction
import unittest

from immer.cognition.fertig.arithmetic_ir import Span, Unit
from immer.cognition.fertig.discourse_ssa import (
    DiscourseNumber,
    DiscourseSSAError,
    TypedDiscourseSSA,
)
from immer.cognition.fertig.signed_expression import LiteralExpr


SCALAR = Unit.scalar()


class TypedDiscourseSSATests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = "Mara and Talia share one ledger."
        self.span = Span(0, len(self.source), self.source)

    def _symbol(self, ssa: TypedDiscourseSSA, owner, state: str):
        return ssa.symbol(
            owner,
            property="value",
            item="ledger",
            scope="test",
            state=state,
            role="ledger_value",
            unit=SCALAR,
            span=self.span,
        )

    def test_plural_and_singular_references_require_one_typed_antecedent(self) -> None:
        ssa = TypedDiscourseSSA(self.source)
        mara = ssa.entity("Mara", "reader", self.span)
        talia = ssa.entity("Talia", "reader", self.span)
        group = ssa.group("readers", (mara, talia), "reader", self.span)

        self.assertEqual(
            ssa.resolve(
                "they",
                role="reader",
                number=DiscourseNumber.PLURAL,
                members=("Mara", "Talia"),
            ),
            group,
        )
        with self.assertRaisesRegex(DiscourseSSAError, "unique typed antecedent"):
            ssa.resolve("she", role="reader", number=DiscourseNumber.SINGULAR)

    def test_names_are_exact_and_never_fuzzily_repaired(self) -> None:
        ssa = TypedDiscourseSSA(self.source)
        ssa.entity("Mara", "reader", self.span)
        with self.assertRaisesRegex(DiscourseSSAError, "unique typed antecedent"):
            ssa.resolve("Maria", role="reader")

    def test_definition_order_is_topological_not_clause_order(self) -> None:
        ssa = TypedDiscourseSSA(self.source)
        owner = ssa.entity("Mara", "reader", self.span)
        first = self._symbol(ssa, owner, "first")
        second = self._symbol(ssa, owner, "second")
        ssa.define(
            second,
            ssa.ref(first, self.span, role="ledger_value"),
            self.span,
            relation_id="second_from_first",
        )
        ssa.define(
            first,
            LiteralExpr(Fraction(3), SCALAR, self.span, ("three",)),
            self.span,
            relation_id="first_value",
        )
        target = ssa.ref(second, self.span, role="ledger_value")

        definitions = ssa.finalize(target)

        self.assertEqual([row.symbol for row in definitions], [first.key, second.key])

    def test_role_unit_duplicate_and_disconnected_mutations_fail_closed(self) -> None:
        ssa = TypedDiscourseSSA(self.source)
        owner = ssa.entity("Mara", "reader", self.span)
        first = self._symbol(ssa, owner, "first")
        second = self._symbol(ssa, owner, "second")
        literal = LiteralExpr(Fraction(3), SCALAR, self.span, ("three",))
        ssa.define(first, literal, self.span, relation_id="first")
        with self.assertRaisesRegex(DiscourseSSAError, "multiple definitions"):
            ssa.define(first, literal, self.span, relation_id="duplicate")
        with self.assertRaisesRegex(DiscourseSSAError, "role"):
            ssa.ref(first, self.span, role="foreign_role")
        ssa.define(second, literal, self.span, relation_id="second")
        with self.assertRaisesRegex(DiscourseSSAError, "outside target scope"):
            ssa.finalize(ssa.ref(first, self.span, role="ledger_value"))

    def test_cycles_are_rejected_as_non_affine_dependencies(self) -> None:
        ssa = TypedDiscourseSSA(self.source)
        owner = ssa.entity("Mara", "reader", self.span)
        first = self._symbol(ssa, owner, "first")
        second = self._symbol(ssa, owner, "second")
        ssa.define(
            first,
            ssa.ref(second, self.span, role="ledger_value"),
            self.span,
            relation_id="first_from_second",
        )
        ssa.define(
            second,
            ssa.ref(first, self.span, role="ledger_value"),
            self.span,
            relation_id="second_from_first",
        )
        with self.assertRaisesRegex(DiscourseSSAError, "cycle is non-affine"):
            ssa.finalize(ssa.ref(first, self.span, role="ledger_value"))


if __name__ == "__main__":
    unittest.main()
