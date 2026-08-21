"""
fertig.solver — der Unified-Solver: vier Engine-Achsen, eine Antwort.

  0. bindings: Bindungs-Parser (Zahl->Objekt->Einheit->Rolle) — NEU,
     die schwere richtige Lösung für Textaufgaben
  1. semantic: Entity-Relations-Graph (sprachagnostisch, abstinent)
  2. math:     Operationsketten-Templates (exakt, abstinent)
  3. miner:    automatisch gelernte Regeln (Backtracking, abstinent)

Best-of mit Abstinenz: die erste Engine, die eine Antwort hat, gewinnt.
Keine Engine rät — falsche Antworten sind Schulden.
"""

from __future__ import annotations

from typing import Dict, Optional

from .semantic import parse_semantic
from . import math as math_mod
from . import miner as miner_mod
from . import bindings as bindings_mod


def solve(question: str, rules: Optional[Dict[str, str]] = None
          ) -> Optional[str]:
    """Unified: bindings -> semantic -> math -> miner."""
    # 0. Bindungs-Parser: Objekt-gebundene Mengen (die Grundlage)
    bv = bindings_mod.solve(question)
    if bv is not None:
        return bv
    # 1. Semantischer Graph
    g = parse_semantic(question)
    sv = g.solve()
    if sv is not None:
        return str(sv.numerator) if sv.denominator == 1 else str(float(sv))
    # 2. Operationsketten
    mv = math_mod.solve(question)
    if mv is not None:
        return mv
    # 3. Gelernte Regeln (Backtracking)
    if rules:
        return miner_mod.apply_rules(question, rules)
    return None
