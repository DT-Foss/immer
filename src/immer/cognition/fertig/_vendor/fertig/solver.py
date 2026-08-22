"""
fertig.solver — der Unified-Solver: vier Engine-Achsen, eine Antwort.

  0. bindings: Bindungs-Parser (Zahl->Objekt->Einheit->Rolle) — NEU,
     die schwere richtige Lösung für Textaufgaben
  1. semantic: Entity-Relations-Graph (sprachagnostisch, abstinent)
  2. math:     Operationsketten-Templates (Kandidaten; nicht adjudiziert)
  3. miner:    automatisch gelernte Regeln (Backtracking, abstinent)

Bindings und semantischer Graph duerfen eine Endantwort liefern. Die
Operationsketten bleiben als Kandidaten-Instrument vorhanden, werden ohne
unabhaengigen Verifier aber nicht adjudiziert. Falsche Antworten sind Schulden.
"""

from __future__ import annotations

from typing import Dict, Optional

from .semantic import parse_semantic
from . import miner as miner_mod
from . import bindings as bindings_mod


def solve(question: str, rules: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Unified: bindings -> semantic -> verified rules, else abstain."""
    # 0. Bindungs-Parser: Objekt-gebundene Mengen (die Grundlage)
    bv = bindings_mod.solve(question)
    if bv is not None:
        return bv
    # 1. Semantischer Graph
    g = parse_semantic(question)
    sv = g.solve()
    if sv is not None:
        return str(sv.numerator) if sv.denominator == 1 else str(float(sv))
    # 2. Operationsketten bleiben in fertig.math als messbares
    # Kandidaten-Instrument erhalten. Ohne einen unabhaengigen Strukturbeweis
    # duerfen Template-Treffer hier keine Endantwort werden.
    # 3. Explizit gelernte Regeln (Backtracking)
    if rules:
        return miner_mod.apply_rules(question, rules)
    return None
