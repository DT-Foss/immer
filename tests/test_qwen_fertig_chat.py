from __future__ import annotations

import hashlib
import json
import unittest
from contextlib import contextmanager
from unittest import mock

from immer.cognition.fertig.adapter import (
    CandidateVerification,
    CandidateVerificationStatus,
    CertifiedAnswer,
    FertigSolver,
)
from immer.cognition.qwen_fertig_chat import QwenFertigChat
from immer.contracts import ExecutionStatus, Request, Result


MATH_QUESTION = "What is 500?"
OVEN_QUESTION = (
    "Maggie's oven is malfunctioning. When she sets it to 450 the actual "
    "temperature is 468. If it's off by the same percentage for any recipe, "
    "what temperature should she set it at if her recipe calls for 520 degrees?"
)


def _certificate(
    answer: str = "500", *, question: str = MATH_QUESTION
) -> CertifiedAnswer:
    source_sha256 = hashlib.sha256(question.encode("utf-8")).hexdigest()
    start = question.index("500")
    return CertifiedAnswer(
        answer,
        {
            "answer": answer,
            "certificates": [
                {
                    "context": {},
                    "equation": f"answer = {answer}",
                    "family": "test_fixture",
                    "inputs": {"answer": answer},
                    "kind": "guarded_formula/v1",
                    "numeric_coverage": True,
                    "numeric_spans": [
                        {"end": start + 3, "start": start, "text": "500"}
                    ],
                    "source_sha256": source_sha256,
                    "verified": True,
                }
            ],
            "kind": "fertig-exact-solution/v1",
            "source_sha256": source_sha256,
            "verified": True,
        },
    )


def _verification(
    status: CandidateVerificationStatus,
    *,
    candidate: str | None,
    expected: str | None,
    question: str = MATH_QUESTION,
) -> CandidateVerification:
    certificate = (
        None if expected is None else _certificate(expected, question=question).evidence
    )
    return CandidateVerification(
        status,
        candidate,
        expected,
        {
            "candidate_numeric": candidate is not None,
            "exact_solution": certificate,
            "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
        },
    )


@contextmanager
def _patched_solver(
    certificate: CertifiedAnswer | None | Exception,
    verification: CandidateVerification | Exception,
):
    solver = FertigSolver()
    certify_options = (
        {"side_effect": certificate}
        if isinstance(certificate, Exception)
        else {"return_value": certificate}
    )
    verify_options = (
        {"side_effect": verification}
        if isinstance(verification, Exception)
        else {"return_value": verification}
    )
    with (
        mock.patch.object(solver, "certify", **certify_options) as certify,
        mock.patch.object(
            solver, "verify_candidate", **verify_options
        ) as verify_candidate,
    ):
        yield solver, certify, verify_candidate


class _Qwen:
    name = "qwen.fixture"
    capabilities = frozenset({"chat"})

    def __init__(self, result: Result | Exception) -> None:
        self.result = result
        self.requests: list[Request] = []
        self.loaded = False
        self.close_calls = 0

    def handle(self, request: Request) -> Result:
        self.requests.append(request)
        self.loaded = True
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def close(self) -> None:
        self.close_calls += 1


def _qwen_ok(output: str) -> Result:
    return Result(
        ExecutionStatus.OK,
        "qwen.fixture",
        output=output,
        evidence={
            "execution": "local-authenticated-causal-bundle/v1",
            "generation": {
                "forward_passes": 3,
                "generated_tokens": 2,
                "prompt_tokens": 8,
                "token_trace_sha256": "c" * 64,
            },
            "model": "Qwen/Qwen3.8-27B",
        },
    )


def _receipt(result: Result) -> dict:
    receipt = result.evidence["receipt"]
    assert isinstance(receipt, dict)
    core = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    encoded = json.dumps(
        core,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert receipt["receipt_sha256"] == hashlib.sha256(encoded).hexdigest()
    return receipt


class QwenFertigChatTests(unittest.TestCase):
    def test_real_formula_and_rref_certificates_short_circuit_without_qwen(
        self,
    ) -> None:
        cases = (
            (OVEN_QUESTION, "500"),
            (
                "Lina has 10 shells. Omar has 4 fewer shells than her. "
                "How many shells does Omar have?",
                "6",
            ),
        )
        for question, expected in cases:
            with self.subTest(expected=expected):
                qwen = _Qwen(_qwen_ok("wrong"))
                result = QwenFertigChat(qwen, FertigSolver()).handle(
                    Request("chat", question)
                )

                self.assertEqual(result.output, expected)
                self.assertFalse(qwen.loaded)
                self.assertFalse(qwen.requests)
                receipt = _receipt(result)
                self.assertEqual(receipt["route"], "fertig_exact_short_circuit")
                self.assertIsNone(receipt["qwen"])
                self.assertEqual(receipt["fertig"]["status"], "certified")

    def test_forged_or_incomplete_certificate_cannot_short_circuit_qwen(
        self,
    ) -> None:
        question = MATH_QUESTION
        source_sha256 = hashlib.sha256(question.encode("utf-8")).hexdigest()
        skeleton = {
            "answer": "500",
            "certificates": [{"kind": "guarded_formula/v1", "verified": True}],
            "kind": "fertig-exact-solution/v1",
            "source_sha256": source_sha256,
            "verified": True,
        }
        invalid_certificates = (
            CertifiedAnswer("500", skeleton),
            CertifiedAnswer(
                "500",
                {
                    **_certificate("500", question=question).evidence,
                    "certificates": [],
                },
            ),
            CertifiedAnswer(
                "500",
                {
                    **_certificate("500", question=question).evidence,
                    "source_sha256": "f" * 64,
                },
            ),
            CertifiedAnswer(
                "500",
                {
                    **_certificate("500", question=question).evidence,
                    "certificates": [{"kind": "guarded_formula/v1", "verified": False}],
                },
            ),
        )
        for certificate in invalid_certificates:
            with self.subTest(evidence=certificate.evidence):
                qwen = _Qwen(_qwen_ok("ordinary answer"))
                with _patched_solver(
                    certificate,
                    _verification(
                        CandidateVerificationStatus.ABSTAINED,
                        candidate=None,
                        expected=None,
                        question=question,
                    ),
                ) as (fertig, _, _):
                    result = QwenFertigChat(qwen, fertig).handle(
                        Request("chat", question)
                    )

                self.assertEqual(result.output, "ordinary answer")
                self.assertEqual(len(qwen.requests), 1)
                receipt = _receipt(result)
                self.assertEqual(receipt["route"], "qwen_verification_abstained")
                self.assertEqual(receipt["fertig"]["preflight"]["status"], "error")

    def test_verified_qwen_value_is_returned_once_with_full_output_candidate(
        self,
    ) -> None:
        qwen = _Qwen(_qwen_ok("500"))
        with _patched_solver(
            None,
            _verification(
                CandidateVerificationStatus.VERIFIED,
                candidate="500",
                expected="500",
            ),
        ) as (fertig, certify, verify):
            result = QwenFertigChat(qwen, fertig).handle(
                Request("chat", MATH_QUESTION, {"trace_id": "t-1"})
            )

        self.assertEqual(result.output, "500")
        self.assertEqual(len(qwen.requests), 1)
        self.assertEqual(qwen.requests[0].metadata["trace_id"], "t-1")
        certify.assert_called_once_with(MATH_QUESTION)
        verify.assert_called_once_with(MATH_QUESTION, "500")
        receipt = _receipt(result)
        self.assertEqual(receipt["route"], "qwen_verified")
        self.assertEqual(receipt["candidate"], "500")
        self.assertEqual(receipt["fertig"]["status"], "verified")
        self.assertEqual(receipt["qwen"]["status"], "ok")

    def test_mismatch_overrides_qwen_with_exact_expected_answer(self) -> None:
        qwen = _Qwen(_qwen_ok("504"))
        with _patched_solver(
            None,
            _verification(
                CandidateVerificationStatus.MISMATCH,
                candidate="504",
                expected="500",
            ),
        ) as (fertig, _, _):
            result = QwenFertigChat(qwen, fertig).handle(Request("chat", MATH_QUESTION))

        self.assertEqual(result.output, "500")
        self.assertEqual(len(qwen.requests), 1)
        receipt = _receipt(result)
        self.assertEqual(receipt["route"], "fertig_mismatch_override")
        self.assertEqual(receipt["candidate"], "504")
        self.assertEqual(receipt["fertig"]["verification"]["expected"], "500")

    def test_inconsistent_judgments_cannot_verify_or_override_qwen(self) -> None:
        question = MATH_QUESTION
        exact_500 = _certificate("500", question=question).evidence
        question_sha256 = hashlib.sha256(question.encode("utf-8")).hexdigest()
        wrong_exact = {**exact_500, "answer": "499"}
        skeleton_exact = {
            **exact_500,
            "certificates": [{"kind": "guarded_formula/v1", "verified": True}],
        }
        cases = (
            (
                "504",
                _verification(
                    CandidateVerificationStatus.VERIFIED,
                    candidate="500",
                    expected="500",
                    question=question,
                ),
            ),
            (
                "The answer is 500",
                _verification(
                    CandidateVerificationStatus.VERIFIED,
                    candidate="500",
                    expected="500",
                    question=question,
                ),
            ),
            (
                "504",
                _verification(
                    CandidateVerificationStatus.MISMATCH,
                    candidate="504",
                    expected="504",
                    question=question,
                ),
            ),
            (
                "504",
                CandidateVerification(
                    CandidateVerificationStatus.MISMATCH,
                    "504",
                    "500",
                    {
                        "candidate_numeric": True,
                        "exact_solution": wrong_exact,
                        "question_sha256": question_sha256,
                    },
                ),
            ),
            (
                "504",
                CandidateVerification(
                    CandidateVerificationStatus.ABSTAINED,
                    "504",
                    "500",
                    {
                        "candidate_numeric": True,
                        "exact_solution": exact_500,
                        "question_sha256": question_sha256,
                    },
                ),
            ),
            (
                "500",
                CandidateVerification(
                    CandidateVerificationStatus.VERIFIED,
                    "500",
                    "500",
                    {
                        "candidate_numeric": True,
                        "exact_solution": skeleton_exact,
                        "question_sha256": question_sha256,
                    },
                ),
            ),
        )
        for output, judgment in cases:
            with self.subTest(output=output, judgment=judgment.status.value):
                qwen = _Qwen(_qwen_ok(output))
                with _patched_solver(None, judgment) as (fertig, _, _):
                    result = QwenFertigChat(qwen, fertig).handle(
                        Request("chat", question)
                    )

                self.assertEqual(result.output, output)
                self.assertEqual(len(qwen.requests), 1)
                receipt = _receipt(result)
                self.assertEqual(receipt["route"], "qwen_fertig_error")
                self.assertEqual(receipt["fertig"]["status"], "error")

    def test_ordinary_chat_survives_exact_abstention_without_exact_claim(self) -> None:
        qwen = _Qwen(_qwen_ok("Paris is the capital of France."))
        with _patched_solver(
            None,
            _verification(
                CandidateVerificationStatus.ABSTAINED,
                candidate=None,
                expected=None,
                question="What is France's capital?",
            ),
        ) as (fertig, _, _):
            result = QwenFertigChat(qwen, fertig).handle(
                Request("chat", "What is France's capital?")
            )

        self.assertEqual(result.output, "Paris is the capital of France.")
        receipt = _receipt(result)
        self.assertEqual(receipt["route"], "qwen_verification_abstained")
        self.assertEqual(receipt["fertig"]["status"], "abstained")
        self.assertNotIn("exact_math", json.dumps(result.evidence, sort_keys=True))

    def test_failures_are_contained_and_never_trigger_a_second_qwen_call(self) -> None:
        cases = (
            (
                RuntimeError("certify broke"),
                _verification(
                    CandidateVerificationStatus.ABSTAINED,
                    candidate=None,
                    expected=None,
                ),
                _qwen_ok("ordinary answer"),
                ExecutionStatus.OK,
                "qwen_verification_abstained",
            ),
            (
                None,
                RuntimeError("verify broke"),
                _qwen_ok("ordinary answer"),
                ExecutionStatus.OK,
                "qwen_fertig_error",
            ),
            (
                None,
                _verification(
                    CandidateVerificationStatus.ABSTAINED,
                    candidate=None,
                    expected=None,
                ),
                RuntimeError("decoder broke"),
                ExecutionStatus.ERROR,
                "qwen_failure",
            ),
        )
        for certificate, verification, qwen_outcome, status, route in cases:
            with self.subTest(route=route):
                qwen = _Qwen(qwen_outcome)
                with _patched_solver(certificate, verification) as (fertig, _, _):
                    result = QwenFertigChat(qwen, fertig).handle(
                        Request("chat", MATH_QUESTION)
                    )
                self.assertIs(result.status, status)
                self.assertEqual(len(qwen.requests), 1)
                self.assertEqual(_receipt(result)["route"], route)

    def test_rejections_and_close_do_not_touch_qwen_or_duplicate_capability(
        self,
    ) -> None:
        qwen = _Qwen(_qwen_ok("answer"))
        chat = QwenFertigChat(qwen, FertigSolver())

        unsupported = chat.handle(Request("exact_math", "1+1"))
        blank = chat.handle(Request("chat", "  "))
        chat.close()
        chat.close()
        closed = chat.handle(Request("chat", "hello"))

        self.assertEqual(chat.capabilities, frozenset({"chat"}))
        self.assertIs(unsupported.status, ExecutionStatus.REJECTED)
        self.assertIs(blank.status, ExecutionStatus.REJECTED)
        self.assertIs(closed.status, ExecutionStatus.UNAVAILABLE)
        self.assertFalse(qwen.requests)
        self.assertEqual(qwen.close_calls, 1)

    def test_custom_or_subclass_verifier_is_rejected(self) -> None:
        class CustomFertig(FertigSolver):
            pass

        qwen = _Qwen(_qwen_ok("answer"))
        with self.assertRaisesRegex(TypeError, "production FertigSolver"):
            QwenFertigChat(qwen, CustomFertig())


if __name__ == "__main__":
    unittest.main()
