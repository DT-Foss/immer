"""
fertig.pipeline — die gewicht-freie Kausal-Pipeline.

.causal-Graph -> Symbol-Vokabular + Adjazenz (exakte Kanten, nichts erfunden)
Adjazenz      -> Walk (Kontraktions-Sampler, tau-kontrolliert,
                 Berry-Phasen-Schleifenwächter)
Walk          -> gesprochene Form (Polaritäts-bewusste Verknüpfungen)

Fakten exakt (Graph), Form generiert (deterministisches Gerüst). Gleiche
Eingabe -> gleiche Ausgabe.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import sampler, state_init, bphm, inference
from ._vendor.dotcausal import CausalReader

# Diskurs-Verknüpfungen nach Relations-Polarität — FORM, nicht Fakt.
_CAUSE_OPENERS = [
    "As a result,", "Therefore,", "In turn, this", "This",
    "Consequently,", "And so", "Which",
]
_POS_VERBS = {"improves", "enables", "helps", "increases", "promotes", "boosts"}
_NEG_VERBS = {"prevents", "reduces", "damages", "blocks", "harms", "impairs"}
_CONTRAST = ["However,", "On the other hand,", "Yet"]

DEFAULT_GRAPH = Path(__file__).resolve().parent.parent / "data" / "chained.causal"


def _toks(s) -> List[str]:
    return re.findall(r"[a-z]+", str(s).lower())


def load_graph(path) -> Tuple[List[str], Dict[str, int], Dict[int, Dict[int, float]], Dict[Tuple[int, int], str]]:
    """Symbol-Vokabular + Adjazenz aus einer .causal-Datei.

    symbols  = alle Entitäts-Phrasen (trigger/outcome)
    adj[a]   = {b: conf} — die EXAKTE Kantenmenge des Graphen, nichts erfunden.
    mech     = {(a, b): Mechanismus-Wort} für die Verbalisierung.
    """
    trips = CausalReader(str(path)).get_all_triplets()
    vocab: List[str] = []
    stoi: Dict[str, int] = {}

    def sym(phrase: str) -> Optional[int]:
        p = " ".join(_toks(phrase))
        if p and p not in stoi:
            stoi[p] = len(vocab)
            vocab.append(p)
        return stoi.get(p)

    adj: Dict[int, Dict[int, float]] = {}
    mech: Dict[Tuple[int, int], str] = {}
    for t in trips:
        a, b = sym(t.get("trigger", "")), sym(t.get("outcome", ""))
        if a is None or b is None:
            continue
        c = float(t.get("confidence", 0.5) or 0.5)
        adj.setdefault(a, {})[b] = max(adj.get(a, {}).get(b, 0), c)
        mech[(a, b)] = " ".join(_toks(t.get("mechanism", ""))) or "leads to"
    return vocab, stoi, adj, mech


def load_graph_merged(path=None):
    """Basis-Graph + Welt-Graph (data/world.causal) gemerged.
    Der Welt-Graph wächst durch `fertig grow` — der Gap-Loop."""
    path = path or DEFAULT_GRAPH
    vocab, stoi, adj, mech = load_graph(path)
    world = Path(__file__).resolve().parent.parent / "data" / "world.causal"
    if world.exists():
        try:
            wv, ws, wa, wm = load_graph(world)
        except Exception:
            return vocab, stoi, adj, mech
        offset = len(vocab)
        for old_i, new_i in zip(range(len(wv)), range(offset, offset + len(wv))):
            stoi[wv[old_i]] = new_i
        vocab = vocab + wv
        for a, nbrs in wa.items():
            for b, c in nbrs.items():
                adj.setdefault(a + offset, {})[b + offset] = max(
                    adj.get(a + offset, {}).get(b + offset, 0), c)
        for (a, b), m in wm.items():
            mech[(a + offset, b + offset)] = m
    return vocab, stoi, adj, mech


def derive_chains(adj: Dict[int, Dict[int, float]], vocab: List[str]) -> Dict[Tuple[int, ...], float]:
    """3-Pass-Inferenz (pass1: exakte Ketten) über der Adjazenz.

    Zeigt abgeleitete transitiv Kanten, die der Graph nicht explizit nennt.
    """
    all_ids = list(range(len(vocab)))
    return inference.pass1_exact_chains(all_ids, adjacency=adj)


def walk_chain(
    start: str, vocab: List[str], stoi: Dict[str, int],
    adj: Dict[int, Dict[int, float]], SM: np.ndarray,
    n: int = 8, tau: float = 0.3,
) -> List[Tuple[int, int]]:
    """Der gewicht-freie Walk: von `start` entlang der Kanten-Gewichte,
    kontraktions-gesampelt, mit Berry-Phasen-Wächter gegen Schleifen."""
    cur = stoi.get(" ".join(_toks(start)))
    if cur is None:
        return []
    hops: List[Tuple[int, int]] = []
    hist: List[np.ndarray] = []
    for _ in range(n):
        nbrs = adj.get(cur, {})
        if not nbrs:
            break  # ehrliche Sackgasse — der Graph hat keine weitere Kante
        logits = np.full(len(vocab), -1e9)
        for b, c in nbrs.items():
            logits[b] = np.log(c + 1e-9)
        nxt = sampler.contraction_sample(logits, tau=tau, top_k=10)
        hist.append(state_init.state_for_symbol(nxt, SM))
        if len(hist) >= 5 and bphm.detect_repetition(hist[-6:]):
            break  # Berry-Phase sagt: wir kreisen — stoppen
        hops.append((cur, int(nxt)))
        cur = int(nxt)
    return hops


def walk(
    start: str, vocab: List[str], stoi: Dict[str, int],
    adj: Dict[int, Dict[int, float]], mech: Dict[Tuple[int, int], str],
    SM: np.ndarray, n: int = 8, tau: float = 0.3,
) -> str:
    """Kette als Text: entity -[mechanism]-> entity -[mechanism]-> ..."""
    cur = stoi.get(" ".join(_toks(start)))
    if cur is None:
        return f"('{start}' nicht im Graphen)"
    path = [cur]
    hist: List[np.ndarray] = []
    for _ in range(n):
        nbrs = adj.get(cur, {})
        if not nbrs:
            break
        logits = np.full(len(vocab), -1e9)
        for b, c in nbrs.items():
            logits[b] = np.log(c + 1e-9)
        nxt = sampler.contraction_sample(logits, tau=tau, top_k=10)
        hist.append(state_init.state_for_symbol(nxt, SM))
        if len(hist) >= 5 and bphm.detect_repetition(hist[-6:]):
            break
        path.append(int(nxt))
        cur = int(nxt)
    parts = [vocab[path[0]]]
    for a, b in zip(path, path[1:]):
        m = mech.get((a, b), "leads to") or "leads to"
        parts.append(f"{m} {vocab[b]}")
    return " ".join(parts)


_MASS_NOUNS = {
    "health", "exercise", "smoking", "sleep", "damage", "stress",
    "breathlessness", "buildup", "caffeine",
}


def _det(noun: str) -> str:
    """Gemessene-ish Artikelwahl: Massennomen nackt, Zählnomen mit 'the'.
    Deterministisch und winzig — Form, nicht Fakt."""
    head = noun.split()[-1]
    return noun if head in _MASS_NOUNS else f"the {noun}"


def verbalize(hops: List[Tuple[int, int]], vocab: List[str],
              mech: Dict[Tuple[int, int], str], seed: int = 0) -> str:
    """Kette von Hops -> verbundene Prosa.

    Erster Hop ist ein voller Satz; spätere Hops steigen mit einem
    Diskurs-Verknüpfer und einem Pronomen für das getragene Subjekt wieder ein.
    Verknüpfer-Polarität folgt dem Verb (pass2-Vorzeichenregel).
    """
    if not hops:
        return "(kein Kausalpfad von dort.)"
    rng = np.random.RandomState(seed)
    sents = []
    prev_neg = False
    for i, (a, b) in enumerate(hops):
        subj, obj = vocab[a], vocab[b]
        verb = mech.get((a, b), "leads to")
        cur_neg = bool(set(verb.split()) & _NEG_VERBS)
        if i == 0:
            s = f"{_det(subj).capitalize()} {verb} {_det(obj)}."
        elif prev_neg and not cur_neg:
            # Polaritäts-Umschwung: das Subjekt wurde gerade REDUZIERT,
            # ein nacktes "This improves X" würde die Kette umkehren.
            s = f"Yet {_det(subj)} is exactly what {verb} {_det(obj)}."
        else:
            if cur_neg:
                opener = rng.choice(_CONTRAST)
            else:
                opener = rng.choice(_CAUSE_OPENERS)
            if opener.endswith(("This", "Which")):
                s = f"{opener} {verb} {_det(obj)}."
            else:
                s = f"{opener} {_det(subj)} {verb} {_det(obj)}."
        sents.append(s)
        prev_neg = cur_neg
    return " ".join(sents)


def top_starts(adj: Dict[int, Dict[int, float]], vocab: List[str], k: int = 4) -> List[str]:
    """Die Entitäten mit den meisten ausgehenden Kanten — sinnvolle Startpunkte."""
    ranked = sorted(adj.items(), key=lambda kv: -len(kv[1]))[:k]
    return [vocab[i] for i, _ in ranked]
