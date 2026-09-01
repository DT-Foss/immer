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
from immer.runtimes.ooe.chat import OoeChatAttempt, OoeChatHook
from immer.runtimes.ooe.controller import WarmAccountingReceipt
from immer.runtimes.qwen3_8.inference_economics import receipt_from_result
from immer.runtimes.qwen3_8.action_bank import InferenceActionDirective


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
        self.release_warm_calls = 0

    def handle(self, request: Request) -> Result:
        self.requests.append(request)
        self.loaded = True
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def close(self) -> None:
        self.close_calls += 1

    def release_warm_bypass_state(self) -> None:
        self.release_warm_calls += 1


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
    def test_semantic_warm_hit_releases_native_state_and_never_calls_qwen(
        self,
    ) -> None:
        warm = _qwen_ok("500")
        attempt = OoeChatAttempt(warm, {"status": "semantic-hit"})
        accounting = WarmAccountingReceipt(
            transaction_sha256="1" * 64,
            decision_binding_sha256="2" * 64,
            execution_receipt_sha256="3" * 64,
            disposition="committed",
            saved_qwen_forwards=3,
        )
        hook = object.__new__(OoeChatHook)
        hook.try_warm = lambda _question, _metadata: attempt
        hook.commit_warm = lambda _attempt: accounting
        hook.reject_warm = lambda _attempt: accounting
        hook.abstention_commit_authorized = lambda _attempt: True
        hook.observe_cold = lambda *_args, **_kwargs: {}
        qwen = _Qwen(RuntimeError("Qwen must not run on semantic replay"))
        directive = InferenceActionDirective(
            question_sha256=hashlib.sha256(MATH_QUESTION.encode("utf-8")).hexdigest(),
            runtime_profile_sha256="4" * 64,
            primary_actions=("stored_result",),
            fallback_actions=("qwen_target",),
            draft_enabled=None,
            source_signature_sha256s=("5" * 64,),
            support=1,
            saved_qwen_forwards=3,
        )
        with _patched_solver(
            None,
            _verification(
                CandidateVerificationStatus.VERIFIED,
                candidate="500",
                expected="500",
            ),
        ) as (fertig, _, verify):
            result = QwenFertigChat(qwen, fertig, ooe_hook=hook).handle(
                Request(
                    "chat",
                    MATH_QUESTION,
                    {"qwen_inference_action_directive": directive.to_document()},
                )
            )

        self.assertEqual(result.output, "500")
        self.assertEqual(qwen.requests, [])
        self.assertFalse(qwen.loaded)
        self.assertEqual(qwen.release_warm_calls, 1)
        verify.assert_called_once_with(MATH_QUESTION, "500")
        receipt = _receipt(result)
        self.assertEqual(receipt["route"], "ooe_verified")
        self.assertEqual(
            receipt["ooe"]["accounting"]["saved_qwen_forwards"],
            3,
        )
        self.assertEqual(
            result.evidence["inference_action_directive"]["applied"],
            {"actions": ["stored_result"], "route": "ooe_verified"},
        )

    def test_qwen_receipt_keeps_compact_markov_atlas_runtime_progress(self) -> None:
        base = _qwen_ok("ordinary answer")
        qwen_result = Result(
            base.status,
            base.component,
            output=base.output,
            evidence={
                **dict(base.evidence),
                "generation": {
                    **dict(base.evidence["generation"]),
                    "linear_calls": 90,
                    "seconds": 2.5,
                    "source_body_bytes": 300,
                    "time_to_first_token_seconds": 1.25,
                },
                "draft": {
                    "accepted_draft_tokens": 5,
                    "mode": "hybrid",
                    "rounds": 3,
                    "provider": {
                        "atlas_consensus_confidence_gain": 0.125,
                        "atlas_consensus_rounds": 1,
                        "atlas_consensus_tokens": 2,
                        "markov_rounds": 2,
                        "markov_selections": 1,
                        "mtp_rounds": 1,
                        "mtp_selections": 1,
                        "online_consensus_confidence_gain": 0.05,
                        "online_consensus_rounds": 1,
                        "online_consensus_tokens": 1,
                        "provider_switches": 1,
                        "markov": {
                            "atlas_accepted_tokens": 2,
                            "atlas_contexts": 500_000,
                            "atlas_corpus_tokens": 4_000_000,
                            "atlas_draft_tokens": 3,
                            "atlas_option_calls": 1,
                            "atlas_vote_calls": 2,
                            "atlas_vote_max_score": 0.8,
                            "atlas_vote_score_sum": 1.1,
                            "atlas_vote_supported_tokens": 3,
                            "atlas_vote_tokens": 6,
                            "history_capacity_tokens": 65_536,
                            "learned_tokens": 2_111,
                            "online_vote_calls": 2,
                            "online_vote_max_score": 0.6,
                            "online_vote_score_sum": 0.9,
                            "online_vote_supported_tokens": 2,
                            "online_vote_tokens": 6,
                            "crystal_accepted_tokens": 5,
                            "crystal_bank_cells": 12,
                            "crystal_bank_support": 19,
                            "crystal_captures": 8,
                            "crystal_enabled": True,
                            "crystal_failures": 0,
                            "crystal_last_cell_sha256": "e" * 64,
                            "crystal_last_cosine": 0.98,
                            "crystal_last_margin": 0.31,
                            "crystal_mismatches": 1,
                            "crystal_option_calls": 2,
                            "crystal_proposed_tokens": 9,
                            "crystal_queries": 4,
                            "crystal_query_hits": 2,
                            "crystal_verified_tokens": 6,
                        },
                    },
                    "window_size": 8,
                },
                "o1_markov_retention": {
                    "last_score": {"priority": 7.5},
                    "sequence": 3,
                },
                "contextual_continuation": {
                    "capture_count": 8,
                    "cell_count": 12,
                    "feedback_count": 2,
                    "hit_positions": 5,
                    "identity_sha256": "f" * 64,
                    "settlements": 2,
                    "state_sha256": "1" * 64,
                    "support": 19,
                    "verified_positions": 6,
                },
                "mlp_page_route": {
                    "request": {
                        "adaptive_width_pages_saved": 11,
                        "dynamic_route_calls": 64,
                    },
                },
                "q4": {
                    "request": {
                        "logical_weight_bytes": 400,
                        "page_mlp_prefetch_bytes": 50,
                        "page_mlp_rows": 1,
                        "page_mlp_selected_pages": 17,
                        "page_mlp_weight_bytes": 200,
                    },
                },
                "runtime_metrics": {
                    "generation_wall_seconds": 2.75,
                    "physical_read_bytes": 123,
                    "process_peak_rss_bytes": 1024,
                },
                "runtime_reward": {
                    "accepted_draft_tokens": 5,
                    "o1_priority": 4.5,
                    "page_actions": 17,
                    "page_actions_saved": 11,
                    "receipt_sha256": "d" * 64,
                    "reward": 1.75,
                    "schema": "immer.qwen3.8-joint-runtime-reward/v1",
                },
            },
        )
        qwen = _Qwen(qwen_result)
        with _patched_solver(
            None,
            _verification(
                CandidateVerificationStatus.ABSTAINED,
                candidate=None,
                expected=None,
            ),
        ) as (fertig, _, _):
            result = QwenFertigChat(qwen, fertig).handle(
                Request("chat", MATH_QUESTION)
            )

        draft = _receipt(result)["qwen"]["draft"]
        self.assertEqual(draft["accepted_draft_tokens"], 5)
        self.assertEqual(draft["provider"]["markov_selections"], 1)
        self.assertEqual(draft["provider"]["atlas_consensus_tokens"], 2)
        self.assertEqual(draft["atlas"]["atlas_contexts"], 500_000)
        self.assertEqual(draft["atlas"]["atlas_accepted_tokens"], 2)
        self.assertEqual(draft["atlas"]["atlas_vote_supported_tokens"], 3)
        self.assertEqual(draft["online_memory"]["history_capacity_tokens"], 65_536)
        self.assertEqual(draft["provider"]["online_consensus_tokens"], 1)
        self.assertEqual(draft["context_crystal"]["crystal_accepted_tokens"], 5)
        self.assertEqual(draft["context_crystal"]["crystal_bank_cells"], 12)
        self.assertEqual(
            _receipt(result)["qwen"]["o1_markov_retention"]["sequence"],
            3,
        )
        qwen_receipt = _receipt(result)["qwen"]
        self.assertEqual(qwen_receipt["generation"]["seconds"], 2.5)
        self.assertEqual(
            qwen_receipt["runtime_metrics"]["process_peak_rss_bytes"],
            1024,
        )
        self.assertEqual(
            qwen_receipt["q4"]["request"]["page_mlp_weight_bytes"],
            200,
        )
        self.assertEqual(qwen_receipt["q4"]["request"]["page_mlp_rows"], 1)
        self.assertEqual(
            qwen_receipt["mlp_page_route"]["request"][
                "adaptive_width_pages_saved"
            ],
            11,
        )
        self.assertEqual(qwen_receipt["runtime_reward"]["o1_priority"], 4.5)
        self.assertEqual(
            qwen_receipt["contextual_continuation"]["cell_count"],
            12,
        )
        economics = receipt_from_result(
            result,
            question_sha256=hashlib.sha256(
                MATH_QUESTION.encode("utf-8")
            ).hexdigest(),
            runtime_profile_sha256="e" * 64,
        )
        self.assertEqual(economics.target_forwards, 3)
        self.assertEqual(economics.logical_weight_bytes, 400)
        self.assertEqual(economics.page_mlp_weight_bytes, 200)
        self.assertEqual(economics.saved_pages, 11)
        self.assertEqual(economics.process_peak_rss_bytes, 1024)

    def test_qwen_receipt_keeps_compact_delta_head_execution(self) -> None:
        base = _qwen_ok("ordinary answer")
        qwen_result = Result(
            base.status,
            base.component,
            output=base.output,
            evidence={
                **dict(base.evidence),
                "delta_head_router": {
                    "bank_root": "/private/q4",
                    "head_dim": 128,
                    "layers": [0, 1],
                    "max_selected_heads": 40,
                    "q4_manifest_sha256": "a" * 64,
                    "route_policy": "mean-square+sinkhorn-first-order/v1",
                    "schema": "immer.qwen3.8-packed-delta-head-router/v1",
                    "state_path": "/private/delta-head.json",
                    "state_persistent": True,
                    "value_heads": 48,
                    "width_actions": [24, 32, 40],
                    "request": {
                        "calls": 2,
                        "full_equivalent_bytes": 600,
                        "logical_bytes_saved": 200,
                        "rows": 3,
                        "schema": "immer.qwen3.8-delta-head-request/v1",
                        "selected_blocks": 384,
                        "selected_heads": 96,
                        "sinkhorn_projections": 1,
                        "transitions": 2,
                        "width_32": 2,
                    },
                },
            },
        )
        qwen = _Qwen(qwen_result)
        with _patched_solver(
            None,
            _verification(
                CandidateVerificationStatus.ABSTAINED,
                candidate=None,
                expected=None,
            ),
        ) as (fertig, _, _):
            result = QwenFertigChat(qwen, fertig).handle(
                Request("chat", MATH_QUESTION)
            )

        compact = _receipt(result)["qwen"]["delta_head_router"]
        self.assertEqual(compact["layers"], [0, 1])
        self.assertEqual(compact["width_actions"], [24, 32, 40])
        self.assertEqual(compact["request"]["calls"], 2)
        self.assertEqual(compact["request"]["rows"], 3)
        self.assertEqual(compact["request"]["logical_bytes_saved"], 200)
        self.assertNotIn("bank_root", compact)
        self.assertNotIn("state_path", compact)
        self.assertNotIn("width_32", compact["request"])

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

    def test_outer_action_receipt_keeps_executed_physical_prefix_sinkhorn(
        self,
    ) -> None:
        question = "Explain causal graphs."
        base = _qwen_ok("They index causal structure.")
        qwen = _Qwen(
            Result(
                base.status,
                base.component,
                output=base.output,
                evidence={
                    **dict(base.evidence),
                    "prefix_sinkhorn": {
                        "action_identity_sha256": "a" * 64,
                        "active": True,
                        "available": True,
                        "configuration": {
                            "alpha": 1.0,
                            "replace_base_softmax": True,
                        },
                        "request": {
                            "base_softmax_head_rows_skipped": 12,
                            "base_softmax_probability_elements_skipped": 144,
                        },
                        "schema": (
                            "immer.qwen3.8-prefix-sinkhorn-action-evidence/v1"
                        ),
                    },
                },
            )
        )
        directive = InferenceActionDirective(
            question_sha256=hashlib.sha256(question.encode("utf-8")).hexdigest(),
            runtime_profile_sha256="4" * 64,
            primary_actions=("prefix_sinkhorn", "qwen_target"),
            fallback_actions=("qwen_target",),
            draft_enabled=None,
            source_signature_sha256s=("5" * 64,),
            support=1,
            saved_qwen_forwards=0,
        )
        with _patched_solver(
            None,
            _verification(
                CandidateVerificationStatus.ABSTAINED,
                candidate=None,
                expected=None,
                question=question,
            ),
        ) as (fertig, _, _):
            result = QwenFertigChat(qwen, fertig).handle(
                Request(
                    "chat",
                    question,
                    {"qwen_inference_action_directive": directive.to_document()},
                )
            )

        self.assertTrue(result.ok, result.reason)
        self.assertEqual(
            result.evidence["inference_action_directive"]["applied"]["actions"],
            ["prefix_sinkhorn", "qwen_target"],
        )

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
