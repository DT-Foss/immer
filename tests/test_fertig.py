from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from immer.cognition.fertig import FertigSolver
from immer.contracts import ExecutionStatus, Request


class FertigAdapterTests(unittest.TestCase):
    def test_loads_solver_from_explicit_checkout_without_absolute_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "fertig"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "solver.py").write_text("def solve(question):\n    return 42 if 'answer' in question else None\n", encoding="utf-8")
            solver = FertigSolver(root)
            result = solver.handle(Request("exact_math", "answer please"))
            self.assertEqual(result.status, ExecutionStatus.OK)
            self.assertEqual(result.output, 42)

    def test_missing_checkout_is_unavailable(self) -> None:
        solver = FertigSolver("/path/that/does/not/exist")
        result = solver.handle(Request("exact_math", "x"))
        self.assertEqual(result.status, ExecutionStatus.UNAVAILABLE)


if __name__ == "__main__":
    unittest.main()
