"""
fertig.arena — der Selbst-Benchmark (Stufe 4 der Kette).

Präregistrierter Evaluations-Satz: (Befehl, erwartete Aktion, erwartetes Ziel).
Die Zahlen sind deterministisch — jeder Lauf produziert identische Werte.
Metriken: Aktions-Präzision, Ziel-Präzision, volle Präzision, Ambiguitäts-,
Grounding- und ehrliche-Fehlschläge-Rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .intent import parse_command
from .pipeline import DEFAULT_GRAPH, load_graph


@dataclass
class ArenaResult:
    total: int
    action_hits: int
    target_hits: int
    full_hits: int
    ambiguous: int
    grounded: int
    unknown: int

    def report(self) -> str:
        pct = lambda x: f"{100.0 * x / max(self.total, 1):.1f}%"
        return "\n".join([
            f"Arena (präregistriert, {self.total} Befehle):",
            f"  Aktion korrekt : {self.action_hits}/{self.total}  {pct(self.action_hits)}",
            f"  Ziel korrekt   : {self.target_hits}/{self.total}  {pct(self.target_hits)}",
            f"  Volltreffer    : {self.full_hits}/{self.total}  {pct(self.full_hits)}",
            f"  Ambiguität     : {self.ambiguous}  (ehrliche Rückfragen)",
            f"  Grounded       : {self.grounded}/{self.total}  {pct(self.grounded)}",
            f"  Unbekannt      : {self.unknown}  (ehrliche Fehlschläge)",
        ])


# präregistrierter Evaluations-Satz (Befehl, Aktion, Ziel)
EVAL_SET: List[Tuple[str, str, str]] = [
    ("explain how smoking affects health", "explain", "smoking"),
    ("what does smoking cause", "find", "smoking"),
    ("how can i prevent lung damage", "prevent", "lung damage"),
    ("list what tar buildup causes", "find", "tar buildup"),
    ("tell me about breathlessness", "explain", "breathlessness"),
    ("what improves health", "find", "health"),
    ("show me the effects of exercise", "explain", "exercise"),
    ("stop smoking", "prevent", "smoking"),
    ("explain how caffeine affects sleep", "explain", "caffeine"),
    ("which activities reduce breathlessness", "find", "breathlessness"),
    ("verify the effects of tar buildup", "consult", "tar buildup"),
    ("what can help with lung damage", "help", "lung damage"),
]


def run_arena(graph_path=None, verbose: bool = True) -> ArenaResult:
    graph_path = graph_path or str(DEFAULT_GRAPH)
    vocab, stoi, adj, mech = load_graph(graph_path)
    res = ArenaResult(len(EVAL_SET), 0, 0, 0, 0, 0, 0)

    for cmd, want_action, want_target in EVAL_SET:
        it = parse_command(cmd, vocab)
        ok_action = it.action == want_action
        ok_target = it.target == want_target
        res.action_hits += int(ok_action)
        res.target_hits += int(ok_target)
        res.full_hits += int(ok_action and ok_target)
        res.grounded += int(it.grounded)
        if it.status == "ambiguous":
            res.ambiguous += 1
        if it.status in ("unknown-action", "unknown-target"):
            res.unknown += 1
        if verbose:
            mark = "OK " if (ok_action and ok_target) else ".. "
            print(f"{mark} {cmd!r:48s} -> {it.action}/{it.target} "
                  f"(conf {it.confidence:.2f}, {it.status})")
    return res
