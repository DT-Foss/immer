"""Canonical core runtime path."""

from __future__ import annotations

from ..adapters import FertigAdapter
from ..contracts import BackendStatus, SolveRequest, SolveResult
from ..router import CapabilityRouter
from . import bootstrap_solver


class ImmerRuntime:
    """Dispatch through FERTIG, with a deliberately narrow no-download fallback."""

    def __init__(self, *, fertig: FertigAdapter | None = None, router: CapabilityRouter | None = None) -> None:
        self.fertig = fertig or FertigAdapter()
        self.router = router or CapabilityRouter()

    def solve(self, question: str) -> SolveResult:
        return self.dispatch(SolveRequest(question=question, capability="exact_math"))

    def dispatch(self, request: SolveRequest) -> SolveResult:
        route = self.router.route(request)
        if request.capability != "exact_math":
            return SolveResult(route.status, route.backend, reason=route.reason, evidence={"route": route.to_dict()})
        result = self.fertig.solve(request.question)
        if result.status is BackendStatus.UNAVAILABLE:
            fallback = bootstrap_solver.solve(request.question)
            if fallback is not None:
                result = SolveResult(
                    BackendStatus.VERIFIED,
                    "IMMER.bootstrap_solver",
                    answer=fallback,
                    evidence={"policy": "narrow fixture grammar; otherwise abstain"},
                )
            else:
                result = SolveResult(
                    BackendStatus.ABSTAINED,
                    result.backend,
                    reason="FERTIG unavailable and bootstrap grammar abstained",
                )
        evidence = dict(result.evidence)
        evidence["route"] = route.to_dict()
        return SolveResult(result.status, result.backend, result.answer, result.reason, evidence)
