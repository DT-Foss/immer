from __future__ import annotations

import unittest

from immer.core.contracts import BackendStatus, SolveRequest
from immer.core.runtime import ImmerRuntime


class CoreTests(unittest.TestCase):
    def test_bootstrap_solver_starts_without_external_model(self) -> None:
        result = ImmerRuntime().solve(
            "A bakery sold 12 cakes on Monday and 15 cakes on Tuesday. How many cakes did they sell in total?"
        )
        self.assertEqual(result.answer, "27")
        self.assertIn(result.backend, {"FERTIG.unified_solver", "IMMER.bootstrap_solver"})

    def test_unsupported_question_abstains(self) -> None:
        result = ImmerRuntime().solve("Please invent a fact about an unknown world.")
        self.assertTrue(result.abstained)
        self.assertIsNone(result.answer)

    def test_unknown_capability_is_held(self) -> None:
        result = ImmerRuntime().dispatch(SolveRequest("x", capability="mystery"))
        self.assertEqual(result.status, BackendStatus.HELD)


if __name__ == "__main__":
    unittest.main()
