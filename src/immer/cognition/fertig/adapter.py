from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from fractions import Fraction
import hashlib
import importlib
import importlib.util
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

from ...contracts import ExecutionStatus, Request, Result
from .arithmetic_ir import SolveStatus, solve as solve_arithmetic_ir
from .formula_certificates import solve_guarded_formula
from .structural import ParseStatus, parse_structural_problem


_CANDIDATE_NUMBER = r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
_CANDIDATE_NUMERIC = re.compile(
    rf"(?:{_CANDIDATE_NUMBER})(?:\s*/\s*(?:{_CANDIDATE_NUMBER}))?"
)


class FertigStructuralError(RuntimeError):
    """The closed structural frontend encountered invalid internal input."""


class CandidateVerificationStatus(str, Enum):
    """Gold-free outcome of checking one candidate against exact FERTIG proof."""

    VERIFIED = "verified"
    MISMATCH = "mismatch"
    ABSTAINED = "abstained"


@dataclass(frozen=True, slots=True)
class CertifiedAnswer:
    """Canonical answer and the complete evidence of an exact FERTIG proof."""

    answer: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "evidence": self.evidence,
            "kind": "fertig-certified-answer/v1",
        }


@dataclass(frozen=True, slots=True)
class CandidateVerification:
    """Exact candidate judgment plus machine-readable proof evidence."""

    status: CandidateVerificationStatus
    candidate: str | None
    expected: str | None
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "evidence": self.evidence,
            "expected": self.expected,
            "kind": "fertig-candidate-verification/v1",
            "status": self.status.value,
        }


@dataclass(frozen=True, slots=True)
class _CertifiedAnswer:
    answer: Fraction
    evidence: dict[str, Any]


def _format_fraction(value: object) -> str:
    if not isinstance(value, Fraction):
        raise FertigStructuralError("structural solver returned a non-Fraction target")
    if value.denominator == 1:
        return str(value.numerator)
    return str(float(value))


def _canonical_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    denominator = value.denominator
    twos = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    fives = 0
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        return f"{value.numerator}/{value.denominator}"
    places = max(twos, fives)
    scaled = value.numerator * 2 ** (places - twos) * 5 ** (places - fives)
    sign = "-" if scaled < 0 else ""
    digits = str(abs(scaled)).rjust(places + 1, "0")
    return f"{sign}{digits[:-places]}.{digits[-places:]}".rstrip("0").rstrip(".")


def _structural_certificate_evidence(solution: object) -> dict[str, Any]:
    certificate = getattr(solution, "certificate", None)
    if certificate is None or not certificate.verified:
        raise FertigStructuralError("unique structural solution lacks a certificate")
    return {
        "component_variables": list(certificate.component_variables),
        "equation_count": certificate.equation_count,
        "kind": "fraction_rref/v1",
        "pivot_columns": list(certificate.pivot_columns),
        "rank": certificate.rank,
        "residuals": [
            {
                "constraint_index": residual.constraint_index,
                "span": (
                    None
                    if residual.span is None
                    else {
                        "end": residual.span.end,
                        "start": residual.span.start,
                    }
                ),
                "value": _canonical_fraction(residual.value),
            }
            for residual in certificate.residuals
        ],
        "variable_count": certificate.variable_count,
        "verified": True,
        "zero_residuals": all(row.value == 0 for row in certificate.residuals),
    }


def _solve_certified_evidence(question: str) -> _CertifiedAnswer | None:
    formula = solve_guarded_formula(question)
    parsed = parse_structural_problem(question)
    if parsed.status is ParseStatus.INVALID:
        raise FertigStructuralError(f"invalid structural parse: {parsed.reason}")
    if not parsed.ok:
        if formula is None:
            return None
        answer = formula.answer
        certificates = [formula.certificate.to_dict()]
        return _CertifiedAnswer(
            answer,
            {
                "answer": _canonical_fraction(answer),
                "certificates": certificates,
                "kind": "fertig-exact-solution/v1",
                "source_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
                "verified": True,
            },
        )
    assert parsed.problem is not None
    solution = solve_arithmetic_ir(parsed.problem)
    if solution.status is SolveStatus.INVALID:
        raise FertigStructuralError(f"invalid arithmetic IR: {solution.reason}")
    if not solution.unique:
        if formula is None:
            return None
        answer = formula.answer
        return _CertifiedAnswer(
            answer,
            {
                "answer": _canonical_fraction(answer),
                "certificates": [formula.certificate.to_dict()],
                "kind": "fertig-exact-solution/v1",
                "source_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
                "verified": True,
            },
        )
    if (
        solution.target_value is None
        or solution.certificate is None
        or not solution.certificate.verified
    ):
        raise FertigStructuralError("unique structural solution lacks a certificate")
    if formula is not None and formula.answer != solution.target_value:
        raise FertigStructuralError(
            "independent exact certificates disagree on the target value"
        )
    certificates = [_structural_certificate_evidence(solution)]
    if formula is not None:
        certificates.append(formula.certificate.to_dict())
    answer = solution.target_value
    return _CertifiedAnswer(
        answer,
        {
            "answer": _canonical_fraction(answer),
            "certificates": certificates,
            "kind": "fertig-exact-solution/v1",
            "source_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
            "verified": True,
        },
    )


def _solve_certified(question: str) -> str | None:
    certified = _solve_certified_evidence(question)
    return _format_fraction(certified.answer) if certified is not None else None


def _candidate_fraction(candidate: object) -> Fraction | None:
    if isinstance(candidate, bool) or candidate is None:
        return None
    if isinstance(candidate, Fraction):
        return candidate
    if isinstance(candidate, int):
        return Fraction(candidate)
    if isinstance(candidate, Decimal):
        return Fraction(candidate) if candidate.is_finite() else None
    if not isinstance(candidate, str):
        return None
    raw = candidate.strip()
    if not raw or _CANDIDATE_NUMERIC.fullmatch(raw) is None:
        return None
    parts = raw.split("/")
    if len(parts) > 2:
        return None
    try:
        if len(parts) == 2:
            numerator = Decimal(parts[0].strip().replace(",", ""))
            denominator = Decimal(parts[1].strip().replace(",", ""))
            if (
                not numerator.is_finite()
                or not denominator.is_finite()
                or denominator == 0
            ):
                return None
            return Fraction(numerator) / Fraction(denominator)
        decimal = Decimal(raw.replace(",", ""))
        return Fraction(decimal) if decimal.is_finite() else None
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def canonical_numeric_candidate(candidate: object) -> str | None:
    """Canonicalize one complete numeric candidate without extracting prose."""

    parsed = _candidate_fraction(candidate)
    return _canonical_fraction(parsed) if parsed is not None else None


def verify_candidate(question: str, candidate: object) -> CandidateVerification:
    """Verify a proposed numeric answer using only exact, gold-free certificates."""

    question_sha256 = (
        hashlib.sha256(question.encode("utf-8")).hexdigest()
        if isinstance(question, str)
        else None
    )
    parsed_candidate = _candidate_fraction(candidate)
    canonical_candidate = canonical_numeric_candidate(candidate)
    if not isinstance(question, str) or not question.strip():
        return CandidateVerification(
            CandidateVerificationStatus.ABSTAINED,
            canonical_candidate,
            None,
            {
                "exact_solution": None,
                "question_sha256": question_sha256,
                "reason": "question_must_be_non_empty_text",
            },
        )
    certified = _solve_certified_evidence(question)
    if certified is None:
        return CandidateVerification(
            CandidateVerificationStatus.ABSTAINED,
            canonical_candidate,
            None,
            {
                "exact_solution": None,
                "question_sha256": question_sha256,
                "reason": "no_exact_certificate",
            },
        )
    expected = _canonical_fraction(certified.answer)
    matches = parsed_candidate is not None and parsed_candidate == certified.answer
    return CandidateVerification(
        (
            CandidateVerificationStatus.VERIFIED
            if matches
            else CandidateVerificationStatus.MISMATCH
        ),
        canonical_candidate,
        expected,
        {
            "candidate_numeric": parsed_candidate is not None,
            "exact_solution": certified.evidence,
            "question_sha256": question_sha256,
        },
    )


@contextmanager
def _solver_from_checkout(root: Path) -> Iterator[ModuleType]:
    package_dir = root / "fertig"
    package_file = package_dir / "__init__.py"
    solver_file = package_dir / "solver.py"
    if not package_file.is_file() or not solver_file.is_file():
        raise FileNotFoundError(f"incomplete FERTIG checkout: {root}")

    suffix = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:12]
    package_name = f"_immer_fertig_{suffix}"
    package = ModuleType(package_name)
    package.__file__ = str(package_file)
    package.__path__ = [str(package_dir)]
    package.__package__ = package_name
    sys.modules[package_name] = package
    try:
        module_name = f"{package_name}.solver"
        spec = importlib.util.spec_from_file_location(module_name, solver_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {solver_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        yield module
    finally:
        prefix = f"{package_name}."
        for name in list(sys.modules):
            if name == package_name or name.startswith(prefix):
                sys.modules.pop(name, None)


class FertigSolver:
    """Exact-math capability backed by a FERTIG checkout or installed package."""

    name = "fertig.solver"
    capabilities = frozenset({"exact_math"})

    def __init__(self, root: str | Path | None = None) -> None:
        configured = root if root is not None else os.environ.get("IMMER_FERTIG_ROOT")
        self.root = Path(configured).expanduser().resolve() if configured else None

    def _solve(self, question: str):
        certified = _solve_certified(question)
        if certified is not None:
            return certified
        if self.root is not None:
            with _solver_from_checkout(self.root) as solver:
                answer = solver.solve(question)
            return answer
        # The vendored copy ships with immer and always wins over an
        # ambient installation — reproducibility beats environment luck.
        vendor = Path(__file__).resolve().parent / "_vendor"
        if (vendor / "fertig" / "solver.py").is_file():
            if str(vendor) not in sys.path:
                sys.path.insert(0, str(vendor))
            try:
                solver = importlib.import_module("fertig.solver")
            except ModuleNotFoundError as exc:
                raise FileNotFoundError(
                    f"broken vendored FERTIG under {vendor}"
                ) from exc
            return solver.solve(question)
        try:
            solver = importlib.import_module("fertig.solver")
        except ModuleNotFoundError as exc:
            raise FileNotFoundError(
                "FERTIG is not installed, not vendored, and IMMER_FERTIG_ROOT is not configured"
            ) from exc
        return solver.solve(question)

    def verify_candidate(
        self, question: str, candidate: object
    ) -> CandidateVerification:
        """Return an exact gold-free judgment without invoking legacy heuristics."""

        return verify_candidate(question, candidate)

    def certify(self, question: str) -> CertifiedAnswer | None:
        """Return only a canonical, independently certified exact answer.

        This surface never invokes the checkout, vendored, or installed legacy
        solver.  ``None`` therefore means that the closed exact frontend did
        not produce a proof, not that a heuristic failed to guess an answer.
        """

        if not isinstance(question, str):
            raise TypeError("question must be text")
        if not question.strip():
            return None
        certified = _solve_certified_evidence(question)
        if certified is None:
            return None
        return CertifiedAnswer(
            answer=_canonical_fraction(certified.answer),
            evidence=certified.evidence,
        )

    def handle(self, request: Request) -> Result:
        if request.capability not in self.capabilities:
            return Result(
                ExecutionStatus.REJECTED, self.name, reason="unsupported capability"
            )
        if not isinstance(request.payload, str) or not request.payload.strip():
            return Result(
                ExecutionStatus.REJECTED,
                self.name,
                reason="exact_math payload must be non-empty text",
            )
        try:
            answer = self._solve(request.payload)
        except FileNotFoundError as exc:
            return Result(ExecutionStatus.UNAVAILABLE, self.name, reason=str(exc))
        except Exception as exc:
            return Result(
                ExecutionStatus.ERROR,
                self.name,
                reason=f"FERTIG solver failed: {type(exc).__name__}: {exc}",
            )
        if answer is None:
            return Result(
                ExecutionStatus.ABSTAINED, self.name, reason="FERTIG abstained"
            )
        return Result(ExecutionStatus.OK, self.name, output=answer)


__all__ = [
    "CandidateVerification",
    "CandidateVerificationStatus",
    "CertifiedAnswer",
    "FertigSolver",
    "FertigStructuralError",
    "canonical_numeric_candidate",
    "verify_candidate",
]
