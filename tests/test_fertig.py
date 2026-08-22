from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
FORMERLY_MASKED_BINDING_ERRORS = (210, 215, 299, 570, 1261, 1295)
SELECTIVE_EXTERNAL_RESOLVER_CASES = (
    (
        "Jen got 3 fish. They each need $1 worth of food a day. "
        "How much does she spend on food in the month of May?",
        "93",
    ),
    (
        "Sid traveled 110 miles in 2 hours. If Sid then traveled an "
        "additional 140 miles in 3 hours, what's the average speed he was "
        "traveling?",
        "50",
    ),
    (
        "Mark buys one lottery ticket with a 20% chance of winning and a "
        "second lottery ticket that's three times more likely to win. What is "
        "the probability, expressed as a percentage, that both tickets are "
        "winners?",
        "12",
    ),
    (
        "John buys a cassette with 2 songs. The first song is 5 minutes and "
        "the second song is 60% longer. How much time was the total cassette?",
        "13",
    ),
    (
        "Carl has a cane that is half as long as he is tall. Carl is one foot "
        "taller than his brother, Ned. And Ned is two feet shorter than his "
        "cousin, Isabel. If Isabel is 7 feet tall, how long is Carl's cane, "
        "in feet?",
        "3",
    ),
    (
        "Geb is 10 less than half the age of Haley. If Haley is 26 years old, "
        "how old is Geb?",
        "3",
    ),
    (
        "Tyrion changes his face mask two times every time he goes out. If he "
        "goes out three times a day, how many face masks does he use every 2 "
        "days?",
        "12",
    ),
    (
        "The red rope was four times the length of the blue rope. The blue "
        "rope was 7 centimeters shorter than the yellow rope. If the 3 ropes "
        "had a combined length of 37 centimeters, what was the length of the "
        "red rope in centimeters?",
        "20",
    ),
)


class FertigAdapterTests(unittest.TestCase):
    def test_selective_external_resolvers_add_only_bound_answers(self) -> None:
        solver = FertigSolver()
        for question, expected in SELECTIVE_EXTERNAL_RESOLVER_CASES:
            with self.subTest(expected=expected):
                result = solver.handle(Request("exact_math", question))
                self.assertEqual(result.status, ExecutionStatus.OK)
                self.assertEqual(result.output, expected)

    def test_selective_external_resolvers_reject_incomplete_evidence(self) -> None:
        solver = FertigSolver()
        questions = (
            (
                "Geb is 10 less than half the age of Haley. If Alice is 26 "
                "years old, how old is Geb?"
            ),
            (
                "Mark buys one lottery ticket with a 60% chance of winning and "
                "a second lottery ticket that's three times more likely to win. "
                "What is the probability, expressed as a percentage, that both "
                "tickets are winners?"
            ),
            (
                "John buys a cassette with 3 songs. The first song is 5 minutes "
                "and the second song is 60% longer. How much time was the total "
                "cassette?"
            ),
            (
                "Carl has a cane that is half as long as he is tall. Carl is "
                "one foot taller than his brother, Ned. And Max is two feet "
                "shorter than his cousin, Isabel. If Isabel is 7 feet tall, "
                "how long is Carl's cane, in feet?"
            ),
        )
        for question in questions:
            with self.subTest(question=question):
                result = solver.handle(Request("exact_math", question))
                self.assertEqual(result.status, ExecutionStatus.ABSTAINED)
                self.assertIsNone(result.output)

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

    def test_structural_ir_only_promotes_exact_certified_legacy_abstentions(
        self,
    ) -> None:
        solver = FertigSolver()
        examples = (
            (
                "An eraser costs $2 and a pencil costs $3. "
                "How much do 6 erasers and 8 pencils cost?",
                "36",
            ),
            (
                "Chenny is 10 years old. Alyana is 4 years younger than Chenny. "
                "How old is Anne if she is 2 years older than Alyana?",
                "8",
            ),
            (
                "Martin's weight is 55 kg. Carl’s weight is 16 kg more than "
                "Martin’s weight. Christian’s weight is 8 kg more than Carl’s "
                "weight. Harry is 5 kg less than Christian’s weight. "
                "What is the weight of Harry, in kg?",
                "74",
            ),
            (
                "Mira's sequence has value 3 at step 0. At each step, the next "
                "value in Mira's sequence is 2 times the current value in Mira's "
                "sequence plus 1. What is the cumulative sum of the values in "
                "Mira's sequence from step 0 through step 4?",
                "119",
            ),
        )
        for question, expected in examples:
            with self.subTest(expected=expected):
                result = solver.handle(Request("exact_math", question))
                self.assertEqual(result.status, ExecutionStatus.OK)
                self.assertEqual(result.output, expected)

    def test_internal_binding_exception_is_typed_error_not_abstention(self) -> None:
        solver = FertigSolver()
        solver.handle(Request("exact_math", "not a supported problem"))
        with mock.patch("fertig.bindings._resolve", side_effect=RuntimeError("boom")):
            result = solver.handle(Request("exact_math", "still unsupported"))
        self.assertEqual(result.status, ExecutionStatus.ERROR)
        self.assertIn("BindingParserError", result.reason)

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

    @unittest.skipUnless(GSM8K.is_file(), "vendored GSM8K split unavailable")
    def test_formerly_masked_binding_handlers_totalize_without_guessing(self) -> None:
        import pandas as pd

        rows = pd.read_parquet(GSM8K)
        solver = FertigSolver()
        for index in FORMERLY_MASKED_BINDING_ERRORS:
            with self.subTest(index=index):
                result = solver.handle(
                    Request("exact_math", str(rows.iloc[index]["question"]))
                )
                self.assertEqual(result.status, ExecutionStatus.ABSTAINED)
                self.assertIsNone(result.output)


if __name__ == "__main__":
    unittest.main()
