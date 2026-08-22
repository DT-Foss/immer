"""Guarded exact-math composition over cold S3 organs and FERTIG.

The cascade is deliberately the *only* public owner of ``exact_math``.
Its children are evidence sources, not independently registered services:

1. ask the cold, frozen-host S3 arithmetic path;
2. ask FERTIG as verifier/fallback;
3. accept disagreement only when S3 supplied crystal-verification evidence.

This keeps a fluent but wrong fallback from silently overriding a measured
structural result.  It also gives quantity-changing word problems a small,
deterministic guard because that is a known unsafe edge of the current FERTIG
solver.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Protocol

from ..contracts import ExecutionStatus, Request, Result


class ExactBackend(Protocol):
    """Minimal child interface accepted by :class:`ExactCascade`."""

    name: str

    def handle(self, request: Request) -> Result: ...


_NUMBER_WORDS: Mapping[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "null": 0,
    "eins": 1,
    "ein": 1,
    "eine": 1,
    "zwei": 2,
    "drei": 3,
    "vier": 4,
    "fuenf": 5,
    "fünf": 5,
    "sechs": 6,
    "sieben": 7,
    "acht": 8,
    "neun": 9,
    "zehn": 10,
}
_NUMBER_RE = re.compile(
    r"(?<![\w.])-?\d+(?:[.,]\d+)?(?![\w.])|\b(?:"
    + "|".join(sorted(map(re.escape, _NUMBER_WORDS), key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)
_QUANTITY_QUESTION_RE = re.compile(
    r"\b(?:how\s+many|how\s+much|wie\s+viele?|wieviel(?:e)?)\b", re.IGNORECASE
)
_ADD_CHANGE_RE = re.compile(
    r"\b(?:buys?|bought|gets?|got|receives?|received|finds?|found|earns?|earned|"
    r"gains?|gained|adds?|added|more|kauft|bekommt|bekam|erhaelt|erhält|findet|"
    r"verdient|dazu)\b",
    re.IGNORECASE,
)
_SUB_CHANGE_RE = re.compile(
    r"\b(?:loses?|lost|gives?|gave|spends?|spent|eats?|ate|sells?|sold|removes?|"
    r"removed|fewer|verliert|verlor|gibt|gab|isst|aß|verkauft|entfernt|weniger)\b",
    re.IGNORECASE,
)


class ExactCascade:
    """One guarded owner for the runtime's exact-math capability."""

    name = "immer.exact-cascade"
    capabilities = frozenset({"exact_math"})

    def __init__(self, s3_arithmetic: object, fertig: object) -> None:
        if s3_arithmetic is fertig:
            raise ValueError("S3 arithmetic and FERTIG must be distinct backends")
        self.s3_arithmetic = s3_arithmetic
        self.fertig = fertig

    def handle(self, request: Request) -> Result:
        if request.capability not in self.capabilities:
            return Result(ExecutionStatus.REJECTED, self.name, reason="unsupported capability")
        if not isinstance(request.payload, str) or not request.payload.strip():
            return Result(
                ExecutionStatus.REJECTED,
                self.name,
                reason="exact_math payload must be non-empty text",
            )

        # Evaluation order is an architectural property: the cold structural
        # path always sees the request before the general grounded fallback.
        s3 = self._invoke(self.s3_arithmetic, request)
        fertig = self._invoke(self.fertig, request)
        evidence = {
            "order": ("s3", "fertig"),
            "s3": self._summary(s3),
            "fertig": self._summary(fertig),
        }

        if s3.ok and fertig.ok:
            if self._equivalent(s3.output, fertig.output):
                return Result(
                    ExecutionStatus.OK,
                    self.name,
                    output=s3.output,
                    evidence={**evidence, "route": "consensus"},
                )
            if self._crystal_verified(s3):
                return Result(
                    ExecutionStatus.OK,
                    self.name,
                    output=s3.output,
                    reason="FERTIG disagreed; crystal-verified S3 result retained",
                    evidence={
                        **evidence,
                        "route": "s3_crystal_override",
                        "disagreement_guarded": True,
                    },
                )
            return Result(
                ExecutionStatus.ABSTAINED,
                self.name,
                reason="unverified exact backends disagreed",
                evidence={**evidence, "route": "guarded_abstention", "disagreement_guarded": True},
            )

        if s3.ok:
            if self._crystal_verified(s3):
                return Result(
                    ExecutionStatus.OK,
                    self.name,
                    output=s3.output,
                    reason="crystal-verified S3 result",
                    evidence={**evidence, "route": "s3_crystal"},
                )
            return Result(
                ExecutionStatus.ABSTAINED,
                self.name,
                reason="S3 result lacked crystal-verification evidence",
                evidence={**evidence, "route": "guarded_abstention"},
            )

        if fertig.ok:
            guard = self._guard_fertig_word_problem(request.payload, fertig.output)
            if guard["safe"]:
                return Result(
                    ExecutionStatus.OK,
                    self.name,
                    output=fertig.output,
                    evidence={**evidence, "route": "fertig_fallback", "fallback_guard": guard},
                )
            return Result(
                ExecutionStatus.ABSTAINED,
                self.name,
                reason=str(guard["reason"]),
                evidence={**evidence, "route": "guarded_abstention", "fallback_guard": guard},
            )

        status = self._empty_status(s3, fertig)
        return Result(
            status,
            self.name,
            reason=self._empty_reason(s3, fertig),
            evidence={**evidence, "route": "no_answer"},
        )

    @staticmethod
    def _invoke(backend: object, request: Request) -> Result:
        name = str(getattr(backend, "name", type(backend).__name__))
        try:
            handler = getattr(backend, "handle", None)
            if callable(handler):
                answer = handler(request)
            else:
                solver = getattr(backend, "solve", None)
                if not callable(solver):
                    return Result(
                        ExecutionStatus.UNAVAILABLE,
                        name,
                        reason="backend exposes neither handle(request) nor solve(text)",
                    )
                answer = solver(request.payload)
        except Exception as exc:  # noqa: BLE001 - a fallback cascade must contain child failure
            return Result(ExecutionStatus.ERROR, name, reason=f"{type(exc).__name__}: {exc}")

        if isinstance(answer, Result):
            return answer
        if answer is None:
            return Result(ExecutionStatus.ABSTAINED, name, reason=f"{name} abstained")
        return Result(ExecutionStatus.OK, name, output=answer)

    @staticmethod
    def _summary(result: Result) -> Mapping[str, Any]:
        summary: dict[str, Any] = {
            "status": result.status.value,
            "component": result.component,
        }
        if result.output is not None:
            summary["output"] = result.output
        if result.reason:
            summary["reason"] = result.reason
        if result.evidence:
            summary["evidence"] = dict(result.evidence)
        return summary

    @staticmethod
    def _crystal_verified(result: Result) -> bool:
        evidence = result.evidence
        if evidence.get("crystal_verified") is True:
            return True
        # Compatibility with early S3 prototypes.  Only literal booleans or a
        # literal verifier status are accepted; truthy strings are not proof.
        if evidence.get("verified") is True or evidence.get("exact") is True:
            return True
        verification = evidence.get("verification")
        return isinstance(verification, Mapping) and verification.get("status") == "verified"

    @classmethod
    def _equivalent(cls, left: Any, right: Any) -> bool:
        if isinstance(left, bool) or isinstance(right, bool):
            return left is right
        left_number = cls._decimal(left)
        right_number = cls._decimal(right)
        if left_number is not None and right_number is not None:
            return left_number == right_number
        return " ".join(str(left).split()).casefold() == " ".join(str(right).split()).casefold()

    @staticmethod
    def _decimal(value: Any) -> Decimal | None:
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, (int, float, Decimal)):
            text = str(value)
        elif isinstance(value, str):
            text = value.strip().casefold()
            if text in _NUMBER_WORDS:
                return Decimal(_NUMBER_WORDS[text])
            text = text.replace(",", ".")
        else:
            return None
        try:
            number = Decimal(text)
        except (InvalidOperation, ValueError):
            return None
        return number if number.is_finite() else None

    @classmethod
    def _guard_fertig_word_problem(cls, question: str, output: Any) -> Mapping[str, Any]:
        """Validate the narrow quantity-change family or refuse it.

        This is intentionally not a general word-problem solver.  The guard is
        activated only when the utterance asks for a quantity and contains an
        additive/subtractive change cue.  Ambiguity then means abstention.
        """

        add = bool(_ADD_CHANGE_RE.search(question))
        subtract = bool(_SUB_CHANGE_RE.search(question))
        guarded_family = bool(_QUANTITY_QUESTION_RE.search(question)) and (add or subtract)
        if not guarded_family:
            return {"safe": True, "kind": "not_quantity_change"}

        numbers = cls._numbers(question)
        if len(numbers) != 2 or add == subtract:
            return {
                "safe": False,
                "kind": "quantity_change",
                "reason": "ambiguous quantity-change problem; FERTIG answer withheld",
            }
        expected = numbers[0] + numbers[1] if add else numbers[0] - numbers[1]
        actual = cls._decimal(output)
        if actual is None or actual != expected:
            return {
                "safe": False,
                "kind": "quantity_change",
                "expected": str(expected),
                "reason": "FERTIG failed deterministic quantity-change validation",
            }
        return {
            "safe": True,
            "kind": "quantity_change",
            "expected": str(expected),
            "validated": True,
        }

    @staticmethod
    def _numbers(text: str) -> list[Decimal]:
        numbers: list[Decimal] = []
        for match in _NUMBER_RE.finditer(text):
            token = match.group(0).casefold()
            if token in _NUMBER_WORDS:
                numbers.append(Decimal(_NUMBER_WORDS[token]))
                continue
            try:
                numbers.append(Decimal(token.replace(",", ".")))
            except InvalidOperation:
                continue
        return numbers

    @staticmethod
    def _empty_status(s3: Result, fertig: Result) -> ExecutionStatus:
        statuses = {s3.status, fertig.status}
        if ExecutionStatus.ABSTAINED in statuses:
            return ExecutionStatus.ABSTAINED
        if statuses == {ExecutionStatus.UNAVAILABLE}:
            return ExecutionStatus.UNAVAILABLE
        if ExecutionStatus.ERROR in statuses:
            return ExecutionStatus.ERROR
        if ExecutionStatus.REJECTED in statuses:
            return ExecutionStatus.REJECTED
        return ExecutionStatus.UNAVAILABLE

    @staticmethod
    def _empty_reason(s3: Result, fertig: Result) -> str:
        reasons = [reason for reason in (s3.reason, fertig.reason) if reason]
        return "; ".join(reasons) if reasons else "both exact backends declined"


__all__ = ["ExactBackend", "ExactCascade"]
