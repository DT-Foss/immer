"""
fertig.corpus — Korpus-Modus: gemessene Übergänge, gewicht-frei.

Korpus -> Wort-Vokabular + Bigramm-/Trigramm-Zählungen (deterministisch)
Logits  -> log-counts (Trigramm, Backoff Bigramm, Backoff Unigramm)
Sampler -> Kontraktions-Sampling mit Anti-Wiederholungs-Penalty + Berry-Phase

Kein Training, kein Backprop: reines Zählen und tau-kontrolliertes Ziehen.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import sampler, state_init, bphm

DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "data" / "faraday_candle.txt"


def build_vocab(text: str, max_vocab: int = 2000):
    """Wort-Vokabular + gemessene Bigramm/Trigramm-Übergänge."""
    words = re.findall(r"[a-z]+", text.lower())
    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    vocab = [w for w, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))][:max_vocab]
    stoi = {w: i for i, w in enumerate(vocab)}

    adjacency: Dict[int, Dict[int, int]] = {}
    trigram: Dict[Tuple[int, int], Dict[int, int]] = {}
    for a, b in zip(words, words[1:]):
        ia, ib = stoi.get(a), stoi.get(b)
        if ia is None or ib is None:
            continue
        row = adjacency.setdefault(ia, {})
        row[ib] = row.get(ib, 0) + 1
    for a, b, c in zip(words, words[1:], words[2:]):
        ia, ib, ic = stoi.get(a), stoi.get(b), stoi.get(c)
        if ia is None or ib is None or ic is None:
            continue
        row = trigram.setdefault((ia, ib), {})
        row[ic] = row.get(ic, 0) + 1
    unigram = np.array([freq[w] for w in vocab], dtype=float)
    return vocab, stoi, adjacency, trigram, unigram


def generate(
    prompt: str, vocab: List[str], stoi: Dict[str, int],
    adjacency: Dict[int, Dict[int, int]], trigram: Dict[Tuple[int, int], Dict[int, int]],
    unigram: np.ndarray, state_matrix: np.ndarray,
    n: int = 30, tau: float = 0.4, recency: int = 8,
) -> str:
    """Prompt Symbol für Symbol fortsetzen, gewicht-frei.

    Logits aus den gemessenen Trigramm-Zählungen des letzten Wortpaars,
    Backoff auf Bigramm, dann Unigramm an Sackgassen (alles log-counts).
    Kürzlich emittierte Wörter bekommen eine deterministische Penalty, damit
    der Kontraktions-Sampler nicht auf ein Token einfriert — dieselbe Rolle,
    die der Berry-Phasen-Wächter auf Zustandsebene spielt.
    """
    ids = [stoi[w] for w in re.findall(r"[a-z]+", prompt.lower()) if w in stoi]
    if not ids:
        ids = [0]
    out_ids = list(ids)
    state_history: List[np.ndarray] = []
    V = len(vocab)
    for _ in range(n):
        nbrs = None
        if len(out_ids) >= 2:
            nbrs = trigram.get((out_ids[-2], out_ids[-1]))
        if not nbrs:
            nbrs = adjacency.get(out_ids[-1])  # Backoff: Bigramm
        if nbrs:
            logits = np.full(V, -30.0)
            for b, c in nbrs.items():
                logits[b] = np.log(c)
        else:
            logits = np.log(unigram + 1e-9)  # ehrlicher Backoff: Unigramm
        # deterministische Anti-Wiederholung: kürzlich emittierte Symbole
        for k, sym in enumerate(out_ids[-recency:]):
            logits[sym] -= 2.0 * (k + 1) / recency
        nxt = sampler.contraction_sample(logits, tau=tau, top_k=40)
        # Berry-Phasen-Wächter: wenn die Zustands-Trajektorie kreist,
        # das Schleifen-Symbol hart maskieren und neu ziehen
        state_history.append(state_init.state_for_symbol(nxt, state_matrix))
        if len(state_history) >= 5 and bphm.detect_repetition(state_history[-6:]):
            logits[nxt] = -1e9
            nxt = sampler.contraction_sample(logits, tau=min(0.9, tau + 0.2), top_k=40)
        out_ids.append(int(nxt))
    return " ".join(vocab[i] for i in out_ids)


def stats(vocab: List[str], adjacency: Dict[int, Dict[int, int]],
          trigram: Dict[Tuple[int, int], Dict[int, int]], unigram: np.ndarray) -> str:
    n_edges = sum(len(v) for v in adjacency.values())
    n_tri = sum(len(v) for v in trigram.values())
    return (f"Korpus: {int(unigram.sum())} Wort-Token, Vokabular {len(vocab)}, "
            f"{n_edges} Bigramm- + {n_tri} Trigramm-Kanten")
