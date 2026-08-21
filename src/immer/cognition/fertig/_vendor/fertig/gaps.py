"""
fertig.gaps — der Gap-Loop: Lücke erkennen -> Weltwissen beschaffen ->
in den Graphen speichern -> neu messen.

Der Kreislauf (F4 + Compounding, mit dem Web als Substrat):
  1. detect_gaps: Befehle parsen, unknown-target-Fälle sammeln
  2. grow(target): Wikipedia holen -> Tripletts extrahieren -> in den
     wachsenden Welt-Graphen (data/world.causal) mergen (Dedupe, max-conf)
  3. Der Fortschritt ist messbar: denselben Befehl erneut parsen -> grounded?

Der Welt-Graph ist ein normaler .causal (CausalWriter-Format) und wird
damit sofort von pipeline/intent/tools konsumiert.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ._vendor.dotcausal import CausalReader, CausalWriter
from .intent import parse_command
from .pipeline import load_graph, DEFAULT_GRAPH
from . import sources

WORLD_GRAPH = Path(__file__).resolve().parent.parent / "data" / "world.causal"


def _load_world() -> List[Tuple[str, str, str, float]]:
    """Vorhandene Welt-Tripletts (max-Konfidenz-Dedupe)."""
    if not WORLD_GRAPH.exists():
        return []
    best: Dict[Tuple[str, str, str], float] = {}
    try:
        for t in CausalReader(str(WORLD_GRAPH)).get_all_triplets():
            key = (str(t.get("trigger", "")), str(t.get("mechanism", "")),
                   str(t.get("outcome", "")))
            best[key] = max(best.get(key, 0.0),
                            float(t.get("confidence", 0.5) or 0.5))
    except Exception:
        return []
    return [(a, b, c, conf) for (a, b, c), conf in best.items()]


def _save_world(triplets: List[Tuple[str, str, str, float]]) -> int:
    w = CausalWriter(api_id="fertig")
    for a, b, c, conf in triplets:
        w.add_triplet(a, b, c, confidence=conf)
    w.save(str(WORLD_GRAPH))
    return len(triplets)


def grow(target: str, verbose: bool = True,
         source_names: Optional[List[str]] = None
         ) -> List[Tuple[str, str, str, float]]:
    """Ein Ziel: alle Quellen -> Extraktion -> Merge in den Welt-Graphen."""
    if verbose:
        print(f"[grow] {target!r}: Quellen {', '.join(source_names or sources.SOURCES)}")
    new_trips = sources.fetch_all(target, sources=source_names,
                                  verbose=verbose)
    if not new_trips:
        if verbose:
            print(f"[grow] {target!r}: keine Tripletts aus keiner Quelle")
        return []

    merged = {t[:3]: t[3] for t in _load_world()}
    added = 0
    for a, b, c, conf in new_trips:
        key = (a, b, c)
        if key not in merged or conf > merged[key]:
            merged[key] = conf
            added += 1
    _save_world([(a, b, c, conf) for (a, b, c), conf in merged.items()])

    if verbose:
        print(f"[grow] {target!r}: {len(new_trips)} extrahiert, "
              f"{added} neu, Graph jetzt {len(merged)} Tripletts")
    return new_trips


def detect_gaps(commands: List[str], graph_path=None) -> List[str]:
    """Befehle parsen; unbekannte Ziele als Lücken sammeln (dedupliziert)."""
    graph_path = graph_path or str(DEFAULT_GRAPH)
    vocab = load_graph(graph_path)[0]
    gaps: List[str] = []
    seen: Set[str] = set()
    for cmd in commands:
        it = parse_command(cmd, vocab)
        if it.status == "unknown-target":
            # Ziel-Kandidaten = Inhaltswörter des Befehls
            from .pipeline import _toks
            for w in _toks(cmd):
                if w in seen or len(w) < 3:
                    continue
                seen.add(w)
                gaps.append(w)
    return gaps


def grow_gaps(commands: List[str], max_targets: int = 3,
              verbose: bool = True) -> int:
    """Gap-Loop über die Lücken der gegebenen Befehle."""
    gaps = detect_gaps(commands)
    if verbose:
        print(f"[gaps] {len(gaps)} Kandidaten: {', '.join(gaps[:8])}")
    grown = 0
    for target in gaps[:max_targets]:
        trips = grow(target, verbose=verbose)
        grown += len(trips)
    return grown
