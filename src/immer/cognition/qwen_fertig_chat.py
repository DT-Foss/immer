"""Certified chat policy joining the local Qwen runtime and exact FERTIG.

The component owns only ``chat``.  It gives a closed exact certificate first
refusal, then lets Qwen answer once and adjudicates that complete answer as one
numeric candidate.  It never extracts a convenient number from prose.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Mapping
from typing import Any

from ..contracts import Component, ExecutionStatus, Request, Result
from .fertig.adapter import (
    CandidateVerification,
    CandidateVerificationStatus,
    CertifiedAnswer,
    FertigSolver,
    canonical_numeric_candidate,
)


_EXACT_CERTIFICATE_KINDS = frozenset({"fraction_rref/v1", "guarded_formula/v1"})
_EXACT_EVIDENCE_KEYS = frozenset(
    {"answer", "certificates", "kind", "source_sha256", "verified"}
)
_FORMULA_KEYS = frozenset(
    {
        "context",
        "equation",
        "family",
        "inputs",
        "kind",
        "numeric_coverage",
        "numeric_spans",
        "source_sha256",
        "verified",
    }
)
_FORMULA_SPAN_KEYS = frozenset({"end", "start", "text"})
_RREF_KEYS = frozenset(
    {
        "component_variables",
        "equation_count",
        "kind",
        "pivot_columns",
        "rank",
        "residuals",
        "variable_count",
        "verified",
        "zero_residuals",
    }
)
_RREF_RESIDUAL_KEYS = frozenset({"constraint_index", "span", "value"})
_RREF_SPAN_KEYS = frozenset({"end", "start"})


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _text_sha256(value: str) -> str:
    return _sha256(value.encode("utf-8"))


def _stable_value(value: object) -> object:
    """Convert arbitrary component evidence to deterministic receipt data."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"invalid_float": str(value)}
    if isinstance(value, Mapping):
        rows: dict[str, object] = {}
        for key in sorted(value, key=lambda item: str(item)):
            rows[str(key)] = _stable_value(value[key])
        return rows
    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in value]
    value_type = type(value)
    return {"unsupported_type": f"{value_type.__module__}.{value_type.__qualname__}"}


def _failure(exc: Exception) -> dict[str, str]:
    return {
        "detail": str(exc),
        "error": f"{type(exc).__module__}.{type(exc).__qualname__}",
    }


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _span_coordinates(
    span: Mapping[str, Any],
    *,
    keys: frozenset[str],
    question: str,
) -> tuple[int, int]:
    if frozenset(span) != keys:
        raise ValueError("FERTIG certificate span schema is invalid")
    start, end = span.get("start"), span.get("end")
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or start < 0
        or end <= start
        or end > len(question)
    ):
        raise ValueError("FERTIG certificate span coordinates are invalid")
    return start, end


def _validate_formula_certificate(
    certificate: Mapping[str, Any], *, question: str, question_sha256: str
) -> None:
    if frozenset(certificate) != _FORMULA_KEYS:
        raise ValueError("guarded-formula certificate schema is invalid")
    if certificate.get("verified") is not True:
        raise ValueError("guarded-formula certificate is not verified")
    if certificate.get("numeric_coverage") is not True:
        raise ValueError("guarded-formula certificate lacks numeric coverage")
    if certificate.get("source_sha256") != question_sha256:
        raise ValueError("guarded-formula certificate is bound to another question")
    if not isinstance(certificate.get("family"), str) or not certificate["family"]:
        raise ValueError("guarded-formula certificate family is invalid")
    if not isinstance(certificate.get("equation"), str) or not certificate["equation"]:
        raise ValueError("guarded-formula certificate equation is invalid")
    inputs = certificate.get("inputs")
    if not isinstance(inputs, Mapping) or not inputs:
        raise ValueError("guarded-formula certificate inputs are invalid")
    if any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, str)
        or canonical_numeric_candidate(value) is None
        for key, value in inputs.items()
    ):
        raise ValueError("guarded-formula certificate input binding is invalid")
    context = certificate.get("context")
    if not isinstance(context, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in context.items()
    ):
        raise ValueError("guarded-formula certificate context is invalid")
    spans = certificate.get("numeric_spans")
    if not isinstance(spans, list) or not spans:
        raise ValueError("guarded-formula certificate numeric spans are invalid")
    previous_end = -1
    for span in spans:
        if not isinstance(span, Mapping):
            raise TypeError("guarded-formula numeric span must be a mapping")
        start, end = _span_coordinates(span, keys=_FORMULA_SPAN_KEYS, question=question)
        text = span.get("text")
        if (
            start < previous_end
            or not isinstance(text, str)
            or text != question[start:end]
            or canonical_numeric_candidate(text) is None
        ):
            raise ValueError("guarded-formula numeric spans are inconsistent")
        previous_end = end


def _validate_rref_certificate(
    certificate: Mapping[str, Any], *, question: str
) -> None:
    if frozenset(certificate) != _RREF_KEYS:
        raise ValueError("fraction-RREF certificate schema is invalid")
    if (
        certificate.get("verified") is not True
        or certificate.get("zero_residuals") is not True
    ):
        raise ValueError("fraction-RREF certificate is not verified")
    variable_count = certificate.get("variable_count")
    equation_count = certificate.get("equation_count")
    rank = certificate.get("rank")
    if not _positive_int(variable_count) or not _positive_int(equation_count):
        raise ValueError("fraction-RREF certificate dimensions are invalid")
    if rank != variable_count:
        raise ValueError("fraction-RREF certificate is not full rank")
    variables = certificate.get("component_variables")
    if (
        not isinstance(variables, list)
        or len(variables) != variable_count
        or len(set(variables)) != variable_count
        or any(not isinstance(value, str) or not value for value in variables)
    ):
        raise ValueError("fraction-RREF component variables are invalid")
    pivots = certificate.get("pivot_columns")
    if (
        not isinstance(pivots, list)
        or len(pivots) != variable_count
        or len(set(pivots)) != variable_count
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            or value >= variable_count
            for value in pivots
        )
    ):
        raise ValueError("fraction-RREF pivot columns are invalid")
    residuals = certificate.get("residuals")
    if not isinstance(residuals, list) or len(residuals) != equation_count:
        raise ValueError("fraction-RREF residual count is invalid")
    indices: set[int] = set()
    for residual in residuals:
        if (
            not isinstance(residual, Mapping)
            or frozenset(residual) != _RREF_RESIDUAL_KEYS
        ):
            raise ValueError("fraction-RREF residual schema is invalid")
        index = residual.get("constraint_index")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index in indices
            or residual.get("value") != "0"
        ):
            raise ValueError("fraction-RREF residual is invalid")
        indices.add(index)
        span = residual.get("span")
        if span is not None:
            if not isinstance(span, Mapping):
                raise TypeError("fraction-RREF residual span must be a mapping")
            _span_coordinates(span, keys=_RREF_SPAN_KEYS, question=question)


def _validate_exact_evidence(
    evidence: object,
    *,
    answer: str,
    question: str,
) -> Mapping[str, Any]:
    if not isinstance(evidence, Mapping):
        raise TypeError("FERTIG exact evidence must be a mapping")
    if frozenset(evidence) != _EXACT_EVIDENCE_KEYS:
        raise ValueError("FERTIG exact evidence schema is invalid")
    if evidence.get("kind") != "fertig-exact-solution/v1":
        raise ValueError("FERTIG exact evidence kind is invalid")
    if evidence.get("verified") is not True:
        raise ValueError("FERTIG exact evidence is not verified")
    if evidence.get("answer") != answer:
        raise ValueError("FERTIG exact evidence answer differs from its judgment")
    question_sha256 = _text_sha256(question)
    source_sha256 = evidence.get("source_sha256")
    if not _is_sha256(source_sha256):
        raise ValueError("FERTIG exact evidence source digest is invalid")
    if source_sha256 != question_sha256:
        raise ValueError("FERTIG exact evidence is bound to another question")
    certificates = evidence.get("certificates")
    if not isinstance(certificates, list) or not certificates:
        raise ValueError("FERTIG exact evidence has no certificates")
    for certificate in certificates:
        if not isinstance(certificate, Mapping):
            raise TypeError("FERTIG exact certificate must be a mapping")
        if certificate.get("verified") is not True:
            raise ValueError("FERTIG exact certificate is not verified")
        kind = certificate.get("kind")
        if kind not in _EXACT_CERTIFICATE_KINDS:
            raise ValueError("FERTIG exact certificate kind is not recognized")
        if kind == "guarded_formula/v1":
            _validate_formula_certificate(
                certificate, question=question, question_sha256=question_sha256
            )
        else:
            _validate_rref_certificate(certificate, question=question)
    return evidence


def _qwen_summary(result: Result) -> dict[str, Any]:
    evidence = _stable_value(dict(result.evidence))
    assert isinstance(evidence, dict)
    output = result.output if isinstance(result.output, str) else None
    summary: dict[str, Any] = {
        "component": result.component,
        "evidence_sha256": _sha256(_canonical_json(evidence)),
        "output_sha256": None if output is None else _text_sha256(output),
        "reason": result.reason,
        "status": result.status.value,
    }
    for key in ("execution", "model", "revision", "thinking", "tokenizer_sha256"):
        if key in evidence:
            summary[key] = evidence[key]
    bundle = evidence.get("bundle")
    if isinstance(bundle, dict):
        summary["bundle_sha256"] = _sha256(_canonical_json(bundle))
    generation = evidence.get("generation")
    if isinstance(generation, dict):
        summary["generation"] = {
            key: generation[key]
            for key in (
                "forward_passes",
                "generated_tokens",
                "linear_calls",
                "prompt_tokens",
                "source_body_bytes",
                "stopped_on_eos",
                "token_trace_sha256",
            )
            if key in generation
        }
        summary["generation_sha256"] = _sha256(_canonical_json(generation))
    return summary


def _certified_payload(certificate: CertifiedAnswer) -> dict[str, Any]:
    return {
        "answer": certificate.answer,
        "evidence": _stable_value(certificate.evidence),
        "kind": "fertig-certified-answer/v1",
    }


class QwenFertigChat:
    """One chat route with exact FERTIG first refusal and adjudication."""

    name = "qwen3.8.fertig-chat"
    capabilities = frozenset({"chat"})

    def __init__(self, qwen: Component, fertig: FertigSolver) -> None:
        qwen_capabilities = getattr(qwen, "capabilities", frozenset())
        if "chat" not in qwen_capabilities:
            raise TypeError("qwen component must own the chat capability")
        if "exact_math" in qwen_capabilities:
            raise ValueError("wrapped Qwen component must not own exact_math")
        if not callable(getattr(qwen, "handle", None)):
            raise TypeError("qwen component must provide handle(request)")
        if type(fertig) is not FertigSolver:
            raise TypeError("fertig must be the production FertigSolver")
        self._qwen = qwen
        self._fertig = fertig
        self._closed = False
        self._close_error: str | None = None
        self._lock = threading.RLock()

    @property
    def qwen(self) -> Component:
        return self._qwen

    @property
    def fertig(self) -> FertigSolver:
        return self._fertig

    @property
    def loaded(self) -> bool:
        return bool(getattr(self._qwen, "loaded", False))

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def model_id(self) -> str | None:
        value = getattr(self._qwen, "model_id", None)
        return value if isinstance(value, str) else None

    @property
    def revision(self) -> str | None:
        value = getattr(self._qwen, "revision", None)
        return value if isinstance(value, str) else None

    def _result(
        self,
        *,
        question: str,
        route: str,
        output: object,
        status: ExecutionStatus,
        reason: str | None,
        candidate: str | None,
        qwen: dict[str, Any] | None,
        fertig: dict[str, Any],
    ) -> Result:
        stable_output = _stable_value(output)
        output_sha256 = (
            _text_sha256(output)
            if isinstance(output, str)
            else _sha256(_canonical_json(stable_output))
        )
        core = {
            "candidate": candidate,
            "fertig": _stable_value(fertig),
            "kind": "qwen-fertig-chat-receipt/v1",
            "output_sha256": output_sha256,
            "question_sha256": _text_sha256(question),
            "qwen": qwen,
            "route": route,
        }
        receipt = {**core, "receipt_sha256": _sha256(_canonical_json(core))}
        return Result(
            status,
            self.name,
            output=output,
            reason=reason,
            evidence={"receipt": receipt},
        )

    @staticmethod
    def _certificate(value: object, *, question: str) -> CertifiedAnswer:
        if not isinstance(value, CertifiedAnswer):
            raise TypeError("FERTIG certify returned an invalid certificate")
        if not value.answer or not isinstance(value.evidence, dict):
            raise TypeError("FERTIG certify returned an incomplete certificate")
        _validate_exact_evidence(
            value.evidence,
            answer=value.answer,
            question=question,
        )
        return value

    @staticmethod
    def _verification(
        value: object,
        *,
        candidate: str,
        question: str,
    ) -> CandidateVerification:
        if not isinstance(value, CandidateVerification):
            raise TypeError("FERTIG verify_candidate returned an invalid judgment")
        if not isinstance(value.status, CandidateVerificationStatus):
            raise TypeError("FERTIG judgment has an invalid status")
        if not isinstance(value.evidence, Mapping):
            raise TypeError("FERTIG judgment evidence must be a mapping")
        canonical_candidate = canonical_numeric_candidate(candidate)
        if value.candidate != canonical_candidate:
            raise ValueError("FERTIG judgment is bound to another candidate")
        question_sha256 = _text_sha256(question)
        if value.evidence.get("question_sha256") != question_sha256:
            raise ValueError("FERTIG judgment is bound to another question")
        exact_solution = value.evidence.get("exact_solution")
        if value.status is CandidateVerificationStatus.ABSTAINED:
            if value.expected is not None or exact_solution is not None:
                raise ValueError("abstained FERTIG judgment contains an exact claim")
            return value
        if not isinstance(value.expected, str) or not value.expected:
            raise ValueError("exact FERTIG judgment lacks an expected answer")
        _validate_exact_evidence(
            exact_solution,
            answer=value.expected,
            question=question,
        )
        if value.status is CandidateVerificationStatus.VERIFIED:
            if canonical_candidate is None or value.expected != canonical_candidate:
                raise ValueError(
                    "verified FERTIG judgment disagrees with its candidate"
                )
        elif value.expected == canonical_candidate:
            raise ValueError("mismatch FERTIG judgment equals its candidate")
        return value

    def handle(self, request: Request) -> Result:
        if request.capability not in self.capabilities:
            return Result(
                ExecutionStatus.REJECTED,
                self.name,
                reason="unsupported capability",
            )
        if not isinstance(request.payload, str) or not request.payload.strip():
            return Result(
                ExecutionStatus.REJECTED,
                self.name,
                reason="chat payload must be non-empty text",
            )
        question = request.payload.strip()

        with self._lock:
            if self._closed:
                return Result(
                    ExecutionStatus.UNAVAILABLE,
                    self.name,
                    reason="Qwen-FERTIG chat component is closed",
                )

            preflight: dict[str, Any]
            try:
                raw_certificate = self._fertig.certify(question)
                certificate = (
                    None
                    if raw_certificate is None
                    else self._certificate(raw_certificate, question=question)
                )
                preflight = {"status": "abstained"}
            except Exception as exc:
                certificate = None
                preflight = {"status": "error", **_failure(exc)}
            if certificate is not None:
                certified = _certified_payload(certificate)
                return self._result(
                    question=question,
                    route="fertig_exact_short_circuit",
                    output=certificate.answer,
                    status=ExecutionStatus.OK,
                    reason=None,
                    candidate=None,
                    qwen=None,
                    fertig={"certificate": certified, "status": "certified"},
                )

            qwen_request = Request("chat", question, request.metadata)
            try:
                raw_qwen_result = self._qwen.handle(qwen_request)
                if not isinstance(raw_qwen_result, Result):
                    raise TypeError("Qwen handle returned a non-Result value")
                qwen_result = raw_qwen_result
            except Exception as exc:
                qwen_result = Result(
                    ExecutionStatus.ERROR,
                    str(getattr(self._qwen, "name", "qwen")),
                    reason=f"Qwen chat failed: {type(exc).__name__}: {exc}",
                    evidence={"failure": _failure(exc)},
                )
            try:
                qwen_summary = _qwen_summary(qwen_result)
            except Exception as exc:
                qwen_summary = {
                    "component": str(qwen_result.component),
                    "evidence": {"status": "invalid", **_failure(exc)},
                    "output_sha256": (
                        _text_sha256(qwen_result.output)
                        if isinstance(qwen_result.output, str)
                        else None
                    ),
                    "reason": qwen_result.reason,
                    "status": qwen_result.status.value,
                }
            if not qwen_result.ok:
                return self._result(
                    question=question,
                    route="qwen_failure",
                    output=qwen_result.output,
                    status=qwen_result.status,
                    reason=qwen_result.reason,
                    candidate=(
                        qwen_result.output
                        if isinstance(qwen_result.output, str)
                        else None
                    ),
                    qwen=qwen_summary,
                    fertig={"preflight": preflight, "status": "not_run"},
                )
            if not isinstance(qwen_result.output, str) or not qwen_result.output:
                return self._result(
                    question=question,
                    route="qwen_failure",
                    output=None,
                    status=ExecutionStatus.ERROR,
                    reason="Qwen chat returned an invalid successful output",
                    candidate=None,
                    qwen=qwen_summary,
                    fertig={"preflight": preflight, "status": "not_run"},
                )

            candidate = qwen_result.output
            try:
                verification = self._verification(
                    self._fertig.verify_candidate(question, candidate),
                    candidate=candidate,
                    question=question,
                )
            except Exception as exc:
                return self._result(
                    question=question,
                    route="qwen_fertig_error",
                    output=candidate,
                    status=ExecutionStatus.OK,
                    reason=None,
                    candidate=candidate,
                    qwen=qwen_summary,
                    fertig={
                        "preflight": preflight,
                        "status": "error",
                        **_failure(exc),
                    },
                )

            fertig_receipt = {
                "preflight": preflight,
                "status": verification.status.value,
                "verification": _stable_value(verification.to_dict()),
            }
            if verification.status is CandidateVerificationStatus.VERIFIED:
                return self._result(
                    question=question,
                    route="qwen_verified",
                    output=candidate,
                    status=ExecutionStatus.OK,
                    reason=None,
                    candidate=candidate,
                    qwen=qwen_summary,
                    fertig=fertig_receipt,
                )
            if verification.status is CandidateVerificationStatus.MISMATCH:
                assert verification.expected is not None
                return self._result(
                    question=question,
                    route="fertig_mismatch_override",
                    output=verification.expected,
                    status=ExecutionStatus.OK,
                    reason=None,
                    candidate=candidate,
                    qwen=qwen_summary,
                    fertig=fertig_receipt,
                )
            return self._result(
                question=question,
                route="qwen_verification_abstained",
                output=candidate,
                status=ExecutionStatus.OK,
                reason=None,
                candidate=candidate,
                qwen=qwen_summary,
                fertig=fertig_receipt,
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            close = getattr(self._qwen, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    self._close_error = f"{type(exc).__name__}: {exc}"

    def __enter__(self) -> QwenFertigChat:
        with self._lock:
            if self._closed:
                raise RuntimeError("Qwen-FERTIG chat component is closed")
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


__all__ = ["QwenFertigChat"]
