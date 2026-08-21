"""
fertig.learn — das Lern-Modul: gemessene Lexika, die wachsen.

Stufe 3 der Kette: wer versteht, kann lernen. Deterministisches Zählen —
kein Training, kein Backprop:

  Verb-Score  : Anteil der Vorkommen in transitiver Position
                (vor "the/a/my/your/..." = Verb-Slot)  -> Aktions-Lexikon
  Nomen-Score : Anteil der Vorkommen nach Determiner        -> Ziel-Lexikon

Gespeichert als JSON (data/lexicon.json). fertig.intent nutzt es als
Coverage-Erweiterung: neue Verben werden Aktionen, neue Nomen werden
Ziel-Kandidaten — der Compounding-Loop des Systems.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

DEFAULT_LEXICON = Path(__file__).resolve().parent.parent / "data" / "lexicon.json"

_DETERMINERS = {"the", "a", "an", "my", "your", "his", "her", "our", "their"}


@dataclass
class Lexicon:
    """Gemessenes Lexikon: Verb->(Aktion, Gewicht), Nomen->Gewicht."""
    actions: Dict[str, Dict] = field(default_factory=dict)
    nouns: Dict[str, float] = field(default_factory=dict)
    tokens: int = 0

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({
            "actions": self.actions, "nouns": self.nouns,
            "tokens": self.tokens,
        }, indent=1, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Lexicon":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(actions=d.get("actions", {}), nouns=d.get("nouns", {}),
                   tokens=d.get("tokens", 0))


def learn(text: str, lexicon: Lexicon | None = None,
          min_count: int = 2) -> Lexicon:
    """Korpus -> wachsende Lexika (deterministisch)."""
    if lexicon is None:
        lexicon = Lexicon()
    words = re.findall(r"[a-z]+", text.lower())

    verb_pos: Dict[str, int] = {}    # Vorkommen in Verb-Slot
    noun_pos: Dict[str, int] = {}    # Vorkommen nach Determiner
    total: Dict[str, int] = {}

    for i, w in enumerate(words):
        total[w] = total.get(w, 0) + 1
        nxt = words[i + 1] if i + 1 < len(words) else ""
        prev = words[i - 1] if i > 0 else ""
        # Verb-Slot: direkt vor einem Determiner + Nomen ("reduce the risk")
        if nxt in _DETERMINERS and i + 2 < len(words):
            verb_pos[w] = verb_pos.get(w, 0) + 1
        # Nomen-Slot: direkt nach Determiner ("the flame")
        if prev in _DETERMINERS:
            noun_pos[w] = noun_pos.get(w, 0) + 1

    for w, n in total.items():
        if n < min_count:
            continue
        v_share = verb_pos.get(w, 0) / n
        n_share = noun_pos.get(w, 0) / n
        if v_share >= 0.5 and v_share > n_share:
            # transitives Verb: Aktion = das Wort selbst, Gewicht = Anteil
            lexicon.actions.setdefault(w, {"action": w, "weight": round(v_share, 3)})
        elif n_share >= 0.3:
            lexicon.nouns[w] = round(n_share, 3)

    lexicon.tokens += len(words)
    return lexicon


def learn_from_file(path, lexicon: Lexicon | None = None,
                    min_count: int = 2) -> Lexicon:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return learn(text, lexicon, min_count=min_count)
