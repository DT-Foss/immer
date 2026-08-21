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
        result = self.ask_harvester(query)
        if result is None:
            return None
        span = self.store.teach(str(result), question=query)
        span["source"] = f"harvest:{self.harvester.name}"
        model_id = getattr(self.harvester, "model_id", "")
        if "snapshots" in str(model_id):  # lokale Snapshots: kurzer Modellname
            span["source"] += f":{Path_safe(str(model_id))}"
        self._rewrite_last_source(span)
        return span

    def capture(self, question: str, answer: str) -> dict[str, Any]:
        """Write a spoken answer into the library — repeats become free."""
        span = self.store.teach(answer, question=question)
        span["source"] = f"gespraech:{self.harvester.name if self.harvester else 'lokal'}"
        self._rewrite_last_source(span)
        return span

    def ask_harvester(self, query: str) -> str | None:
        """One harvest call; returns clean answer text or None."""
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
        return str(result.output)

    def refine(self, query: str, old_text: str) -> dict[str, Any] | None:
        """Upgrade an existing card with a fresh harvest (provenance kept)."""
        answer = self.ask_harvester(query)
        if answer is None:
            return None
        for span in self.store.spans:
            if span["text"] == old_text:
                span["text"] = answer
                span["source"] = f"harvest:{self.harvester.name}"
                self.store._persist()
                return span
        return None

    def _rewrite_last_source(self, span: dict[str, Any]) -> None:
        if self.store.spans and self.store.spans[-1]["text"] == span["text"]:
            self.store.spans[-1]["source"] = span["source"]
            self.store._persist()


def Path_safe(model_id: str) -> str:
    """Short provenance tag from a model path or HF id."""
    tail = model_id.rstrip("/").split("/")[-1]
    return tail[:40]
