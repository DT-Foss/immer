"""Explicit capability routing policy."""

from __future__ import annotations

from .adapters import FlcaAdapter, O1StateAdapter
from .contracts import BackendStatus, CapabilityRoute, SolveRequest


class CapabilityRouter:
    """Route only known capabilities; unknown work is held for evidence."""

    def __init__(self, *, o1: O1StateAdapter | None = None, flca: FlcaAdapter | None = None) -> None:
        self.o1 = o1 or O1StateAdapter()
        self.flca = flca or FlcaAdapter()

    def route(self, request: SolveRequest) -> CapabilityRoute:
        if request.capability == "exact_math":
            return CapabilityRoute("exact_math", "FERTIG.unified_solver", BackendStatus.VERIFIED)
        if request.capability == "persistent_state":
            return self.o1.route()
        if request.capability == "evidence_route":
            family = str(request.metadata.get("operator_family", ""))
            return self.flca.route(family)
        if request.capability in {"crsa_attention", "deployment_compile"}:
            return CapabilityRoute(
                request.capability,
                "CRSA/Liquid-QAD",
                BackendStatus.HELD,
                reason="external measured plan required",
            )
        return CapabilityRoute(
            request.capability,
            "IMMER",
            BackendStatus.HELD,
            reason="unknown capability; no safe default route",
        )
