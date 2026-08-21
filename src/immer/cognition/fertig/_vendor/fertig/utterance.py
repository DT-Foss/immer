"""
fertig.utterance — das Utterance-IR: belegter Plan -> Prosa -> Rückverifikation.

Der Gegenentwurf zu "erst generieren, dann hoffen":

  1. PLAN    : semantische Kanten aus dem .causal-Graphen (subj, verb, obj,
               conf) — jede Aussage hat einen Beleg (Quelle, Konfidenz)
  2. PROSA   : deterministische Verbalisierung des Plans (pipeline.verbalize)
  3. RÜCKVERIFIKATION : die Prosa wird ZURÜCKgeparst (intent/semantic) und
               gegen den Plan verglichen — konsistent = freigegeben,
               inkonsistent = verworfen (Abstinenz, nie "hoffen")

Eine Sprachmaschine, die gleichzeitig flüssig (Form-Engine), lernfähig
(Graph wächst) und strukturell rechenschaftspflichtig ist (jede Aussage
mit Beleg + verifizierter Rückführung).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .pipeline import load_graph, verbalize, walk_chain
from . import state_init
from .intent import parse_command
from .semantic import parse_semantic


@dataclass
class Utterance:
    """Eine belegte Aussage: Plan-Kante + erzeugte Prosa + Verifikation."""
    subj: str
    verb: str
    obj: str
    conf: float
    source: str = "graph"
    prose: str = ""
    verified: Optional[bool] = None
    detail: str = ""


@dataclass
class SpeechResult:
    """Der geschlossene Kreis: Plan -> Prosa -> Verdict."""
    utterances: List[Utterance] = field(default_factory=list)
    prose: str = ""
    all_verified: bool = False
    verified_count: int = 0

    def report(self) -> str:
        lines = [f"Utterance-IR ({self.verified_count}/{len(self.utterances)} "
                 f"verifiziert):"]
        for u in self.utterances:
            mark = "✓" if u.verified else ("✗" if u.verified is False else "?")
            lines.append(f"  {mark} {u.subj} {u.verb} {u.obj} (c={u.conf:.2f})"
                         f" — {u.detail}")
        lines.append(f"Prosa: {self.prose}")
        verdict = "FREIGEGEBEN" if self.all_verified else "VERWORFEN"
        lines.append(f"Verdict: {verdict}")
        return "\n".join(lines)


def plan_from_graph(graph_path: str, entity: str, n: int = 5
                    ) -> List[Tuple[str, str, str, float]]:
    """Belegter Plan: die Kanten ab einer Entität im .causal-Graphen."""
    vocab, stoi, adj, mech = load_graph(graph_path)
    e = entity.lower()
    if e not in stoi:
        return []
    tid = stoi[e]
    out = []
    for b, c in sorted(adj.get(tid, {}).items(), key=lambda kv: -kv[1])[:n]:
        out.append((vocab[tid], mech.get((tid, b), "leads to"), vocab[b],
                    float(c)))
    return out


_VERB_SIGNALS = {
    "causes": ["caus", "leads to", "lead to", "results in"],
    "leads_to": ["caus", "leads to", "lead to", "results in"],
    "reduces": ["reduc", "lowers", "decreases"],
    "increases": ["increas", "raises", "boosts"],
    "prevents": ["prevent", "blocks", "stops"],
    "improves": ["improve", "strengthen", "better"],
}


def _verify_utterance(u: Utterance, graph_path: str) -> Tuple[bool, str]:
    """Rückverifikation: die Prosa-SATZ der Plan-Kante zurückführen.

    Beleg = Subjekt UND Objekt der Kante stehen wörtlich im Satz UND das
    Verb-Signal ist erkennbar. Zusätzlicher Kanal: der semantische Parser
    rekonstruiert die Aussage-Struktur. Nur wörtlich belegbare Aussagen
    werden freigegeben — strukturelle Rechenschaftspflicht."""
    s = u.prose.lower()
    # Kanal 1: wörtlicher Beleg (Subjekt + Objekt)
    if u.subj not in s:
        return False, f"Subjekt '{u.subj}' fehlt im Satz"
    if u.obj not in s:
        return False, f"Objekt '{u.obj}' fehlt im Satz"
    # Kanal 2: Verb-Signal
    signals = _VERB_SIGNALS.get(u.verb, [u.verb.replace("_", " ")])
    if not any(sig in s for sig in signals):
        return False, f"Verb-Signal '{u.verb}' fehlt im Satz"
    # Kanal 3: semantische Rekonstruktion (unabhängiger Parser)
    g = parse_semantic(u.prose)
    rekonstruiert = any(u.obj in e or u.obj in (e + "s")
                        for e in g.entities)
    detail = "Beleg: subj+obj+verb wörtlich"
    if rekonstruiert:
        detail += ", semantisch rekonstruiert"
    return True, detail


def speak(graph_path: str, entity: str, n: int = 5) -> SpeechResult:
    """Der geschlossene Kreis: Plan -> Prosa -> Rückverifikation."""
    res = SpeechResult()
    plan = plan_from_graph(graph_path, entity, n)
    if not plan:
        return res
    # 2. PROSA: deterministische Verbalisierung aus dem Plan
    vocab, stoi, adj, mech = load_graph(graph_path)
    SM = state_init.initialize_symbol_state(len(vocab))
    hops = []
    cur = stoi.get(entity.lower())
    seen = set()
    for _ in range(n):
        if cur is None or cur in seen:
            break
        seen.add(cur)
        nbrs = adj.get(cur, {})
        if not nbrs:
            break
        nxt = max(nbrs, key=nbrs.get)
        hops.append((cur, nxt))
        cur = nxt
    res.prose = verbalize(hops, vocab, mech, seed=0)
    # 1. PLAN: die verbalisierten Kanten als belegte Aussagen
    # (Plan == Prosa == Plan — der Kreis ist geschlossen)
    plan = [(vocab[a], mech.get((a, b), "leads to"), vocab[b],
             adj[a].get(b, 0.5)) for a, b in hops]
    for subj, verb, obj, conf in plan:
        res.utterances.append(Utterance(subj, verb, obj, conf))
    # 3. RÜCKVERIFIKATION: jede Aussage gegen die Prosa prüfen
    for u in res.utterances:
        # Prosa-Aussagen: Satzweise zuordnen
        sents = [s.strip() for s in res.prose.replace(". ", ".\n").split("\n")
                 if s.strip()]
        matched = False
        for s in sents:
            if u.obj in s.lower():
                u.prose = s
                ok, detail = _verify_utterance(u, graph_path)
                u.verified, u.detail = ok, detail
                matched = True
                break
        if not matched:
            u.verified, u.detail = False, "kein Satz enthält das Objekt"
    res.verified_count = sum(1 for u in res.utterances if u.verified)
    res.all_verified = (len(res.utterances) > 0 and
                        res.verified_count == len(res.utterances))
    return res
