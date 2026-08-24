from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterator

from ...contracts import ExecutionStatus, Request, Result
from .arithmetic_ir import SolveStatus, solve as solve_arithmetic_ir
from .formula_certificates import solve_guarded_formula
from .structural import ParseStatus, parse_structural_problem


class FertigStructuralError(RuntimeError):
    """The closed structural frontend encountered invalid internal input."""


def _format_fraction(value: object) -> str:
    from fractions import Fraction

    if not isinstance(value, Fraction):
        raise FertigStructuralError("structural solver returned a non-Fraction target")
    if value.denominator == 1:
        return str(value.numerator)
    return str(float(value))


def _solve_certified(question: str) -> str | None:
    formula = solve_guarded_formula(question)
    parsed = parse_structural_problem(question)
    if parsed.status is ParseStatus.INVALID:
        raise FertigStructuralError(f"invalid structural parse: {parsed.reason}")
    if not parsed.ok:
        return _format_fraction(formula.answer) if formula is not None else None
    assert parsed.problem is not None
    solution = solve_arithmetic_ir(parsed.problem)
    if solution.status is SolveStatus.INVALID:
        raise FertigStructuralError(f"invalid arithmetic IR: {solution.reason}")
    if not solution.unique:
        return _format_fraction(formula.answer) if formula is not None else None
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
    return _format_fraction(solution.target_value)


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


__all__ = ["FertigSolver", "FertigStructuralError"]
