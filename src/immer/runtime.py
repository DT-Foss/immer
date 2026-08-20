from __future__ import annotations

from collections.abc import Iterable

from .contracts import Component, ExecutionStatus, Request, Result
from .registry import ComponentRegistry


class ImmerRuntime:
    """Dispatch requests across explicitly registered cognitive components."""

    def __init__(self, components: Iterable[Component] = ()) -> None:
        self.registry = ComponentRegistry()
        for component in components:
            self.register(component)

    def register(self, component: Component) -> None:
        self.registry.register(component)

    def dispatch(self, request: Request) -> Result:
        component = self.registry.get(request.capability)
        if component is None:
            return Result(
                ExecutionStatus.UNAVAILABLE,
                "immer",
                reason=f"no component registered for capability {request.capability!r}",
                evidence={"registered": self.registry.capabilities()},
            )
        try:
            return component.handle(request)
        except Exception as exc:
            return Result(
                ExecutionStatus.ERROR,
                component.name,
                reason=f"{type(exc).__name__}: {exc}",
            )
