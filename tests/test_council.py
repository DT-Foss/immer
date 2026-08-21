from __future__ import annotations

import unittest

from immer.contracts import Component, ExecutionStatus, Request, Result
from immer.council import Council


class YesBrain:
    name = "yes"
    capabilities = frozenset({"math"})

    def __init__(self, answer: str = "4") -> None:
        self.answer = answer

    def handle(self, request: Request) -> Result:
        return Result(ExecutionStatus.OK, self.name, output=self.answer)


class AbstainBrain:
    name = "abstainer"
    capabilities = frozenset({"math"})

    def handle(self, request: Request) -> Result:
        return Result(ExecutionStatus.ABSTAINED, self.name, reason="weiß ich nicht")


class BoomBrain:
    name = "boom"
    capabilities = frozenset({"math"})

    def handle(self, request: Request) -> Result:
        raise ValueError("Kurzschluss")


class OtherBrain:
    name = "other"
    capabilities = frozenset({"poetry"})

    def handle(self, request: Request) -> Result:
        return Result(ExecutionStatus.OK, self.name, output="roses")


class CouncilTests(unittest.TestCase):
    def test_majority_wins_with_evidence(self) -> None:
        council = Council((YesBrain("4"), YesBrain("4"), YesBrain("5")))
        result = council.deliberate("math", "2+2")
        self.assertTrue(result.ok)
        self.assertEqual(result.output, "4")
        self.assertEqual(result.evidence["agree"], 2)
        self.assertEqual(result.evidence["votes"], 3)

    def test_all_abstain_is_an_abstained_result(self) -> None:
        council = Council((AbstainBrain(), AbstainBrain()))
        result = council.deliberate("math", "2+2")
        self.assertIs(result.status, ExecutionStatus.ABSTAINED)

    def test_crashing_brain_counts_as_failed_vote_not_collapse(self) -> None:
        council = Council((BoomBrain(), YesBrain("4"), YesBrain("4")))
        result = council.deliberate("math", "2+2")
        self.assertTrue(result.ok)
        self.assertEqual(result.output, "4")

    def test_capability_nobody_holds_is_unavailable(self) -> None:
        council = Council((YesBrain(), AbstainBrain()))
        result = council.deliberate("quantum", "?")
        self.assertIs(result.status, ExecutionStatus.UNAVAILABLE)

    def test_council_is_a_component(self) -> None:
        council = Council((YesBrain(), OtherBrain()))
        self.assertIsInstance(council, Component)
        self.assertIn("math", council.capabilities)
        self.assertIn("poetry", council.capabilities)
        rejected = council.handle(Request("chemistry", "H2O"))
        self.assertIs(rejected.status, ExecutionStatus.REJECTED)

    def test_empty_council_is_rejected_at_construction(self) -> None:
        with self.assertRaises(ValueError):
            Council(())


if __name__ == "__main__":
    unittest.main()
