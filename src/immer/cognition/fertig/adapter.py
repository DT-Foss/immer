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
        if self.root is not None:
            with _solver_from_checkout(self.root) as solver:
                return solver.solve(question)
        try:
            solver = importlib.import_module("fertig.solver")
        except ModuleNotFoundError as exc:
            raise FileNotFoundError(
                "FERTIG is not installed and IMMER_FERTIG_ROOT is not configured"
            ) from exc
        return solver.solve(question)

    def handle(self, request: Request) -> Result:
        if request.capability not in self.capabilities:
            return Result(ExecutionStatus.REJECTED, self.name, reason="unsupported capability")
        if not isinstance(request.payload, str) or not request.payload.strip():
            return Result(ExecutionStatus.REJECTED, self.name, reason="exact_math payload must be non-empty text")
        try:
            answer = self._solve(request.payload)
        except FileNotFoundError as exc:
            return Result(ExecutionStatus.UNAVAILABLE, self.name, reason=str(exc))
        if answer is None:
            return Result(ExecutionStatus.ABSTAINED, self.name, reason="FERTIG abstained")
        return Result(ExecutionStatus.OK, self.name, output=answer)
