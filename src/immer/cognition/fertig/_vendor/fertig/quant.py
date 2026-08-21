"""
fertig.quant — quantitative QA: Fragen, die nur mit Nicht-Wort-Ankern
beantwortbar sind. Der sichtbare Grounding-Beweis.

"How fast can a cheetah run?" -> Konzept (cheetah) + Messgröße (speed)
-> Graph-Kante (cheetah, running_at, "104 km/h") -> Antwort.

Messgrößen-Map: Frage-Wörter -> Mechanismus-Muster im Graphen.
Keine LLM, keine Wörterbücher — die Antwort IST die Messung.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from . import inference as inf
from .gaps import _load_world

# Frage-Wörter -> (Messgrößen-Muster, erforderliche Einheiten)
_MEASURES: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    "speed": (("run", "speed", "race", "travel", "fast"),
              ("km/h", "kmh", "kph", "mph", "m/s")),
    "weight": (("weigh", "mass", "weight"),
                ("kg", "g", "ton", "tons", "tonnes")),
    "size": (("length", "height", "size", "measure", "grow", "tall",
               "long"), ("cm", "m", "km", "mm", "meters", "metres",
                          "kilometers", "kilometres", "feet", "inches")),
    "lifespan": (("live", "lifespan", "age", "life"),
                  ("years", "year", "months", "days", "hours")),
    "capacity": (("capacity", "hold", "contain"),
                  ("liters", "litres", "l", "ml", "gallons")),
}
_QUESTION_WORDS = {"how", "what", "much", "many", "fast", "big", "heavy",
                   "tall", "long", "old", "is", "are", "does", "do", "can",
                   "the", "a", "an", "of", "in", "at", "to", "it", "its"}


def measure_of(question: str) -> Optional[str]:
    q = question.lower()
    for measure, (kws, _) in _MEASURES.items():
        if any(k in q for k in kws):
            return measure
    return None


def concept_of(question: str, symbols: List[str]) -> Optional[str]:
    """Konzept der Frage = bestes Jaro-Match gegen die Graph-Symbole."""
    toks = [t for t in re.findall(r"[a-z]+", question.lower())
            if t not in _QUESTION_WORDS and len(t) > 2]
    best, best_j = None, 0.0
    for t in toks:
        for s in symbols:
            j = inf.jaro_winkler(t, s)
            if j > best_j:
                best, best_j = s, j
    return best if best_j >= 0.8 else None


def answer(question: str) -> Tuple[Optional[str], Optional[str], float]:
    """Frage -> (Antwort, Mechanismus, Konfidenz). None wenn nicht
    beantwortbar (das ist die ehrliche Lücke = nächster ground-Kandidat)."""
    measure = measure_of(question)
    if measure is None:
        return None, None, 0.0
    trips = _load_world()
    symbols = set()
    for a, b, c, _ in trips:
        symbols.add(a)
        symbols.add(c)
    concept = concept_of(question, sorted(symbols))
    if concept is None:
        return None, None, 0.0
    patterns, units = _MEASURES[measure]
    best = None
    for a, b, c, conf in trips:
        if a != concept:
            continue
        if not any(p in b for p in patterns):
            continue
        # Einheiten-Check: die Antwort MUSS die Messgröße tragen
        if not any(u in c for u in units):
            continue
        if best is None or conf > best[2]:
            best = (c, b, conf)
    if best is None:
        return None, None, 0.0
    return best[0], best[1], best[2]


# Mini-Arena: präregistrierte quantitative Fragen
QUANT_SET = [
    ("How fast can a cheetah run?", "cheetah", "speed"),
    ("How much does an elephant weigh?", "elephant", "weight"),
    ("How fast can a human run?", "human", "speed"),
    ("How fast can a horse run?", "horse", "speed"),
    ("How long is a blue whale?", "blue whale", "size"),
]


def run_quant(verbose: bool = True) -> dict:
    hits = covered = total = 0
    rows = []
    for q, concept, measure in QUANT_SET:
        ans, mech, conf = answer(q)
        total += 1
        ok = ans is not None
        covered += int(ok)
        symbols = {a for a, _, _, _ in _load_world()} | \
            {c for _, _, c, _ in _load_world()}
        if ok and concept.lower() in [s.lower() for s in symbols]:
            hits += 1
        rows.append((q, concept, ans, mech, ok))
        if verbose:
            print(f"  {q:45s} -> {ans if ans else '(nicht beantwortbar)'}")
    return {"hits": hits, "covered": covered, "total": total, "rows": rows}


if __name__ == "__main__":
    for q, _, _ in QUANT_SET:
        a, m, c = answer(q)
        print(f"{q:45s} -> {a} [{m} conf={c:.2f}]" if a
              else f"{q:45s} -> (nicht beantwortbar)")
