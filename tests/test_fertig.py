from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from immer.cognition.fertig import FertigSolver
from immer.contracts import ExecutionStatus, Request

ROOT = Path(__file__).resolve().parent.parent
GSM8K = ROOT / "evals" / "gsm8k_test.parquet"
QUARANTINED_MATH_ROWS = (
    216,
    485,
    682,
    963,
    1012,
    1016,
    1047,
    1158,
    1176,
    1204,
    1206,
    1213,
    1215,
    1244,
    1252,
    1272,
    1306,
)


class FertigAdapterTests(unittest.TestCase):
    def test_loads_solver_from_explicit_checkout_without_absolute_defaults(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "fertig"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "solver.py").write_text(
                "def solve(question):\n    return 42 if 'answer' in question else None\n",
                encoding="utf-8",
            )
            solver = FertigSolver(root)
            result = solver.handle(Request("exact_math", "answer please"))
            self.assertEqual(result.status, ExecutionStatus.OK)
            self.assertEqual(result.output, 42)

    def test_missing_checkout_is_unavailable(self) -> None:
        solver = FertigSolver("/path/that/does/not/exist")
        result = solver.handle(Request("exact_math", "x"))
        self.assertEqual(result.status, ExecutionStatus.UNAVAILABLE)

    def test_unverified_math_template_candidate_is_not_an_answer(self) -> None:
        # fertig.math can propose 10 for this template, but neither bindings
        # nor the semantic graph independently proves the relation. The public
        # adjudicator therefore keeps the candidate out of the answer path.
        result = FertigSolver().handle(Request("exact_math", "20 percent of 50"))
        self.assertEqual(result.status, ExecutionStatus.ABSTAINED)
        self.assertIsNone(result.output)

    @unittest.skipUnless(GSM8K.is_file(), "vendored GSM8K split unavailable")
    def test_all_observed_unverified_math_failures_are_must_abstain(self) -> None:
        import pandas as pd

        rows = pd.read_parquet(GSM8K)
        solver = FertigSolver()
        for index in QUARANTINED_MATH_ROWS:
            with self.subTest(index=index):
                result = solver.handle(
                    Request("exact_math", str(rows.iloc[index]["question"]))
                )
                self.assertEqual(result.status, ExecutionStatus.ABSTAINED)
                self.assertIsNone(result.output)


if __name__ == "__main__":
    unittest.main()
