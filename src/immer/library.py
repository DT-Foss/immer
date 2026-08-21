"""Library: knowledge that grows through life.

The measured pattern (Kaskade wordlen 0.875 @ 8 calls): a small brain plus
lookup beats a big head alone. When recall misses, the brain harvests the
answer itself and the library keeps it — with provenance. Every card knows
where it came from; fuzzy 1.5B knowledge is tagged as such.
"""

from __future__ import annotations

from typing import Any

from .contracts import Component, ExecutionStatus, Request, Result
from .memory import SpanStore


class Library:
    """SpanStore + harvest: misses become cards instead of dead ends."""

    def __init__(self, store: SpanStore, harvester: Component | None = None) -> None:
        self.store = store
        self.harvester = harvester

    def teach(self, text: str) -> dict[str, Any]:
        span = self.store.teach(text)
        span["source"] = "david"
        self._rewrite_last_source(span)
        return span

    def recall(self, query: str) -> list[dict[str, Any]]:
        return self.store.recall(query)

    def grow(self, query: str) -> dict[str, Any] | None:
        """Harvest knowledge for a query from the brain; returns the new card."""
        if self.harvester is None:
            return None
        result = self.harvester.handle(
            Request(
                "chat",
                f"Antworte in einem einzigen kurzen sachlichen Satz: {query}",
                metadata={"life": "bibliotheks-erntefahrt"},
            )
        )
        if result.status is not ExecutionStatus.OK or not result.output:
            return None
        span = self.store.teach(str(result.output))
        span["source"] = f"harvest:{self.harvester.name}"
        if hasattr(self.harvester, "model_id"):
            span["source"] += f":{Path_safe(self.harvester.model_id)}"
        self._rewrite_last_source(span)
        return span

    def _rewrite_last_source(self, span: dict[str, Any]) -> None:
        if self.store.spans and self.store.spans[-1]["text"] == span["text"]:
            self.store.spans[-1]["source"] = span["source"]
            self.store._persist()


def Path_safe(model_id: str) -> str:
    """Short provenance tag from a model path or HF id."""
    tail = model_id.rstrip("/").split("/")[-1]
    return tail[:40]
