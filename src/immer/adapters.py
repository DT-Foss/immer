"""Adapters for external projects.

Adapters are deliberately conservative: missing code, missing artifacts, or
rejected evidence becomes an explicit status and never a guessed answer.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterator

from .contracts import BackendStatus, CapabilityRoute, SolveResult, sha256_file


DEFAULT_FERTIG_ROOT = Path("/Users/bhkmie/Documents/Forschung/LanguageModel/FERTIG")
DEFAULT_O1_ROOT = Path("/Users/bhkmie/Documents/Forschung/O1_juli")
DEFAULT_FLCA_ROOT = Path("/Users/bhkmie/Downloads/Forschung/Alte AI Projekte/FLCA")


@contextmanager
def _isolated_fertig_package(root: Path) -> Iterator[ModuleType]:
    """Load FERTIG under a unique name, avoiding namespace collisions."""
    package_dir = root / "fertig"
    package_file = package_dir / "__init__.py"
    solver_file = package_dir / "solver.py"
    if not package_file.is_file() or not solver_file.is_file():
        raise FileNotFoundError(f"FERTIG solver package is incomplete under {root}")

    suffix = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:12]
    package_name = f"_immer_fertig_{suffix}"
    # Do not execute FERTIG's broad package initializer: it imports optional
    # desktop/vision modules and several historical surfaces. The solver only
    # needs relative imports from this package directory.
    package = ModuleType(package_name)
    package.__file__ = str(package_file)
    package.__path__ = [str(package_dir)]
    package.__package__ = package_name
    sys.modules[package_name] = package
    try:
        solver_name = f"{package_name}.solver"
        solver_spec = importlib.util.spec_from_file_location(solver_name, solver_file)
        if solver_spec is None or solver_spec.loader is None:
            raise ImportError(f"cannot create solver loader for {solver_file}")
        solver = importlib.util.module_from_spec(solver_spec)
        sys.modules[solver_name] = solver
        solver_spec.loader.exec_module(solver)
        yield solver
    finally:
        prefix = f"{package_name}."
        for name in list(sys.modules):
            if name == package_name or name.startswith(prefix):
                sys.modules.pop(name, None)


class FertigAdapter:
    """Invoke the current FERTIG unified solver without copying its package."""

    name = "FERTIG.unified_solver"

    def __init__(self, root: str | Path | None = None) -> None:
        configured = root or os.environ.get("IMMER_FERTIG_ROOT")
        self.root = Path(configured).expanduser().resolve() if configured else DEFAULT_FERTIG_ROOT

    @property
    def available(self) -> bool:
        return (self.root / "fertig" / "solver.py").is_file()

    def solve(self, question: str) -> SolveResult:
        if not self.available:
            return SolveResult(
                status=BackendStatus.UNAVAILABLE,
                backend=self.name,
                reason=f"solver.py not found under {self.root}",
            )
        try:
            with _isolated_fertig_package(self.root) as solver:
                answer = solver.solve(question)
        except Exception as exc:  # boundary: external code must not crash the host
            return SolveResult(
                status=BackendStatus.ERROR,
                backend=self.name,
                reason=f"external solver error: {type(exc).__name__}: {exc}",
            )
        if answer is None:
            return SolveResult(
                status=BackendStatus.ABSTAINED,
                backend=self.name,
                reason="all configured solver engines abstained",
                evidence={"root": str(self.root), "policy": "no guessing"},
            )
        return SolveResult(
            status=BackendStatus.VERIFIED,
            backend=self.name,
            answer=str(answer),
            evidence={
                "root": str(self.root),
                "solver_order": ["bindings", "semantic", "math", "miner"],
            },
        )

    def verify(self, question: str, draft: str) -> SolveResult:
        result = self.solve(question)
        normalized = draft.strip().replace(",", "")
        evidence = dict(result.evidence)
        evidence["draft"] = draft
        evidence["exact_match"] = result.answer is not None and normalized == result.answer
        if result.status is BackendStatus.VERIFIED and not evidence["exact_match"]:
            return SolveResult(
                status=BackendStatus.ABSTAINED,
                backend=self.name,
                reason="draft does not exactly match the external solver",
                evidence=evidence,
            )
        return SolveResult(
            status=result.status,
            backend=result.backend,
            answer=result.answer,
            reason=result.reason,
            evidence=evidence,
        )


class O1StateAdapter:
    """Describe the host boundary without guessing a private runtime API."""

    name = "o1-state"

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or os.environ.get("IMMER_O1_ROOT", DEFAULT_O1_ROOT)).expanduser().resolve()

    def route(self) -> CapabilityRoute:
        if not self.root.exists():
            return CapabilityRoute(
                capability="persistent_state",
                backend=self.name,
                status=BackendStatus.UNAVAILABLE,
                reason=f"host checkout not found: {self.root}",
            )
        return CapabilityRoute(
            capability="persistent_state",
            backend=self.name,
            status=BackendStatus.HELD,
            reason="host-specific execution API must be selected explicitly",
            evidence={"root": str(self.root), "weight_transfer": "none"},
        )


class FlcaAdapter:
    """Map known FLCA evidence families to conservative compiler roles."""

    name = "FLCA"
    ROUTE_POLICY = {
        "nonreversible_lift": ("crsa_local", 3),
        "reversible_control": ("crsa_balance", 4),
        "multiplicative_ds2_candidate": ("crsa_free", 6),
    }

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or os.environ.get("IMMER_FLCA_ROOT", DEFAULT_FLCA_ROOT)).expanduser().resolve()

    def route(self, family: str) -> CapabilityRoute:
        if family not in self.ROUTE_POLICY:
            return CapabilityRoute(
                capability="evidence_route",
                backend=self.name,
                status=BackendStatus.HELD,
                reason="operator family is not admitted by the local policy",
                evidence={"family": family, "fallback_bits": 4},
            )
        role, bits = self.ROUTE_POLICY[family]
        return CapabilityRoute(
            capability="evidence_route",
            backend=self.name,
            status=BackendStatus.VERIFIED,
            evidence={"family": family, "role": role, "bits": bits, "root": str(self.root)},
        )


class OrganBankAdapter:
    """Verify a small organ manifest; actual weights remain local artifacts."""

    name = "OrganBank"

    def load_manifest(self, path: str | Path, expected_sha256: str | None = None) -> CapabilityRoute:
        manifest = Path(path).expanduser().resolve()
        if not manifest.is_file():
            return CapabilityRoute(
                capability="organ_sidecar",
                backend=self.name,
                status=BackendStatus.UNAVAILABLE,
                reason=f"organ manifest not found: {manifest}",
            )
        digest = sha256_file(manifest)
        if expected_sha256 and digest != expected_sha256:
            return CapabilityRoute(
                capability="organ_sidecar",
                backend=self.name,
                status=BackendStatus.HELD,
                reason="organ manifest digest mismatch",
                evidence={"actual_sha256": digest, "expected_sha256": expected_sha256},
            )
        return CapabilityRoute(
            capability="organ_sidecar",
            backend=self.name,
            status=BackendStatus.VERIFIED,
            evidence={"manifest": str(manifest), "sha256": digest, "host_mutation": "none"},
        )
