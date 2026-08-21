"""
fertig.tools — die Tool-Schicht (Stufe 5 der Foss-Lernhierarchie:
Umgebungs-Manipulation).

Ein Intent ist verstanden, wenn er einen registrierten Tool-Call auslösen
kann. Jedes Tool nimmt (Intent, Graph) und liefert ein strukturiertes
Ergebnis. Deterministisch, Graph-gegroundet, ehrliche Fehlschläge.

Aktion -> Tool:
  explain -> speech   (Walk + Verbalisierung)
  find    -> chain    (Ketten/Inferenz herleiten)
  prevent -> prevent  (negativer Pfad: was REDUZIERT das Ziel)
  consult -> consult  (Index-Konsultation)
  help    -> help     (was kann ich)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import state_init
from .intent import Intent
from .pipeline import load_graph, walk_chain, verbalize, derive_chains, _NEG_VERBS


@dataclass
class ToolResult:
    """Strukturiertes Tool-Ergebnis — maschinenprüfbar."""
    tool: str
    ok: bool
    text: str = ""
    hops: List[tuple] = field(default_factory=list)
    detail: Dict = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ToolResult {self.tool} ok={self.ok} text={self.text[:60]!r}>"


def _graph(path):
    vocab, stoi, adj, mech = load_graph(path)
    SM = state_init.initialize_symbol_state(len(vocab))
    return vocab, stoi, adj, mech, SM


def speech_tool(intent: Intent, graph_path: str, n: int = 8) -> ToolResult:
    """Aktion 'explain': Walk vom Ziel + Verbalisierung."""
    vocab, stoi, adj, mech, SM = _graph(graph_path)
    if intent.target is None or intent.target not in stoi:
        return ToolResult("speech", False,
                          text=f"'{intent.target}' nicht im Graphen",
                          detail={"target": intent.target})
    hops = walk_chain(intent.target, vocab, stoi, adj, SM, n=n, tau=0.3)
    if not hops:
        return ToolResult("speech", False, text="kein Kausalpfad von dort",
                          detail={"target": intent.target})
    chain = " -> ".join([vocab[hops[0][0]]] + [vocab[b] for _, b in hops])
    return ToolResult(
        "speech", True, text=verbalize(hops, vocab, mech, seed=0),
        hops=hops, detail={"chain": chain, "target": intent.target})


def chain_tool(intent: Intent, graph_path: str, k: int = 5) -> ToolResult:
    """Aktion 'find': abgeleitete Ketten + direkte Nachbarn des Ziels."""
    vocab, stoi, adj, mech, _ = _graph(graph_path)
    if intent.target is None or intent.target not in stoi:
        return ToolResult("chain", False,
                          text=f"'{intent.target}' nicht im Graphen",
                          detail={"target": intent.target})
    tid = stoi[intent.target]
    nbrs = adj.get(tid, {})
    out = sorted(((vocab[b], c) for b, c in nbrs.items()),
                 key=lambda kv: -kv[1])
    chains = derive_chains(adj, vocab)
    relevant = [(c, cf) for c, cf in chains.items()
                if len(c) >= 2 and c[0] == tid][:k]
    text_parts = [f"{intent.target}: {len(out)} direkte Kanten"]
    text_parts += [f"  -> {b} ({c:.2f})" for b, c in out[:k]]
    if relevant:
        text_parts.append("abgeleitete Ketten:")
        text_parts += [f"  [{' -> '.join(vocab[i] for i in c)}] ({cf:.2f})"
                       for c, cf in relevant]
    return ToolResult(
        "chain", True, text="\n".join(text_parts),
        detail={"neighbors": out, "chains": relevant, "target": intent.target})


def prevent_tool(intent: Intent, graph_path: str, n: int = 6) -> ToolResult:
    """Aktion 'prevent': der negative Pfad — welche Kanten REDUZIEREN
    das Ziel, und was treibt diese Reduktoren? (pass2-Vorzeichenlogik)."""
    vocab, stoi, adj, mech, SM = _graph(graph_path)
    if intent.target is None or intent.target not in stoi:
        return ToolResult("prevent", False,
                          text=f"'{intent.target}' nicht im Graphen",
                          detail={"target": intent.target})
    tid = stoi[intent.target]
    reducers = []
    for a, nbrs in adj.items():
        for b, c in nbrs.items():
            if b == tid:
                verb = mech.get((a, b), "")
                if set(verb.split()) & _NEG_VERBS:
                    reducers.append((vocab[a], verb, c))
    if not reducers:
        return ToolResult(
            "prevent", False,
            text=f"kein Reduktor für '{intent.target}' im Graphen",
            detail={"target": intent.target})
    lines = [f"Reduktoren von {intent.target} (gemessene Kanten):"]
    for src, verb, c in sorted(reducers, key=lambda kv: -kv[2]):
        lines.append(f"  {src} {verb} {intent.target} ({c:.2f})")
        # was treibt den Reduktor? Walk rückwärts suchen wir nicht — aber
        # vorwärts vom Reduktor gibt es den nächsten Schritt
        hops = walk_chain(src, vocab, stoi, adj, SM, n=n, tau=0.3)
        if hops:
            chain = " -> ".join([vocab[hops[0][0]]] +
                                [vocab[b] for _, b in hops])
            lines.append(f"    Pfad: {chain}")
    return ToolResult("prevent", True, text="\n".join(lines),
                      detail={"reducers": reducers, "target": intent.target})


def consult_tool(intent: Intent, graph_path: str) -> ToolResult:
    """Aktion 'consult': Index-Konsultation — Ziel-Statistik im Graphen."""
    vocab, stoi, adj, mech, _ = _graph(graph_path)
    if intent.target is None or intent.target not in stoi:
        return ToolResult("consult", False,
                          text=f"'{intent.target}' nicht im Index",
                          detail={"target": intent.target})
    tid = stoi[intent.target]
    out_deg = len(adj.get(tid, {}))
    in_deg = sum(1 for a in adj if tid in adj[a])
    return ToolResult(
        "consult", True,
        text=f"{intent.target}: out={out_deg}, in={in_deg} "
             f"(Index: {len(vocab)} Symbole)",
        detail={"out": out_deg, "in": in_deg, "target": intent.target})


def video_tool(intent: Intent, graph_path: str, n: int = 8) -> ToolResult:
    """Aktion 'erkennen': Video/Stream gegen die VideoBank klassifizieren.
    Verbindet Sehen mit Sprache — die Antwort ist die gelernte Kategorie.
    Das Video kommt aus intent.arguments['video'] (Pfad oder URL)."""
    from . import stream as stream_mod
    from . import video as video_mod
    from pathlib import Path as _P
    import os
    path = intent.arguments.get("video", "")
    if not path or not os.path.exists(path):
        return ToolResult("video", False,
                          text="kein Video-Pfad in den Argumenten "
                               "(--video <datei>)")
    bank = video_mod.VideoBank().load(
        _P(__file__).resolve().parent.parent / "data" / "video_bank.json")
    if not bank.prototypes:
        return ToolResult("video", False,
                          text="VideoBank leer — erst Streams mit --name lernen")
    learner = stream_mod.learn_from(path, seconds=6, fps=4, verbose=False)
    word, d = bank.recognize_signature(learner.sequence_signature())
    if word is None:
        return ToolResult("video", False,
                          text=f"unbekanntes Video (Distanz {d:.4f} über "
                               f"Schwelle {bank.threshold:.2f})",
                          detail={"distance": d})
    return ToolResult("video", True,
                      text=f"Das Video zeigt: {word}",
                      detail={"category": word, "distance": d})


def help_tool(intent: Intent, graph_path: str) -> ToolResult:
    """Aktion 'help': registrierte Tools anzeigen."""
    lines = ["registrierte Tools (Aktion -> Werkzeug):"]
    for action, tool in sorted(_TOOLS.items()):
        lines.append(f"  {action:9s} -> {tool.__name__}")
    return ToolResult("help", True, text="\n".join(lines),
                      detail={"tools": sorted(_TOOLS)})


_TOOLS = {
    "speech": speech_tool,
    "chain": chain_tool,
    "prevent": prevent_tool,
    "consult": consult_tool,
    "video": video_tool,
    "help": help_tool,
}


def execute(intent: Intent, graph_path: str, n: int = 8) -> ToolResult:
    """Intent -> Tool-Ausführung. Nur wenn grounded/status ok."""
    if intent.status != "ok":
        return ToolResult("none", False,
                          text=f"Intent nicht ausführbar (status={intent.status}): "
                               f"{intent.tree}")
    tool = intent.tool
    if tool is None or tool not in _TOOLS:
        return ToolResult("none", False,
                          text=f"kein Tool für Aktion '{intent.action}'")
    return _TOOLS[tool](intent, graph_path, n=n)
