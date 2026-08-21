"""SpanStore: taught facts survive restarts. The organism's card index.

David teaches ("merke: ..."), the organism recalls ("was weißt du über ...").
Knowledge lives OUTSIDE the weights — the library grows through life.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any


class SpanStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.spans: list[dict[str, Any]] = []
        if self.path.is_file():
            document = json.loads(self.path.read_text(encoding="utf-8"))
            self.spans = list(document.get("spans", []))

    def teach(self, text: str, *, question: str | None = None) -> dict[str, Any]:
        span = {
            "text": text.strip(),
            "key": _key_of(question if question else text),
            "ts": time.time(),
        }
        if question:
            span["q"] = question.strip()
        self.spans.append(span)
        self._persist()
        return span

    def recall(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        needle = query.strip().lower()
        if not needle:
            return []
        # Prefix-Stemming pro Wort: "schwarzes"/"Schwarzen"/"schwarze" → "schw"
        words = {_stem(w) for w in re.findall(r"\w+", needle) if len(w) > 3}
        hits: list[tuple[int, dict[str, Any]]] = []
        for span in self.spans:
            source_text = f"{span.get('q', '')} {span['key']} {span['text']}"
            hay_words = {_stem(w) for w in re.findall(r"\w+", source_text.lower())}
            score = sum(1 for w in words if w in hay_words)
            if score:
                hits.append((score, span))
        hits.sort(key=lambda pair: -pair[0])
        return [span for _, span in hits[:limit]]

    def count(self) -> int:
        return len(self.spans)

    def last(self, n: int = 3) -> list[dict[str, Any]]:
        return self.spans[-n:]

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"spans": self.spans}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )


def _key_of(text: str) -> str:
    words = re.findall(r"\w+", text.lower())
    return " ".join(words[:4])


def _stem(word: str) -> str:
    """Prefix-Stemmung: die ersten 4 Zeichen tragen den Stamm der meisten
    deutschen/englischen Wortformen (schwarzes→schw, apples→appl)."""
    return word[:4]
