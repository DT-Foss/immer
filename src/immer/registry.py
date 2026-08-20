from __future__ import annotations

from .contracts import Component


class ComponentRegistry:
    """One explicit component owner per capability."""

    def __init__(self) -> None:
        self._by_capability: dict[str, Component] = {}

    def register(self, component: Component) -> None:
        if not component.capabilities:
            raise ValueError(f"component {component.name!r} declares no capabilities")
        for capability in component.capabilities:
            if capability in self._by_capability:
                owner = self._by_capability[capability]
                raise ValueError(
                    f"capability {capability!r} already owned by {owner.name!r}; "
                    "selection between multiple owners must be explicit"
                )
        for capability in component.capabilities:
            self._by_capability[capability] = component

    def get(self, capability: str) -> Component | None:
        return self._by_capability.get(capability)

    def capabilities(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_capability))
