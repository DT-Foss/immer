#!/usr/bin/env python3 -u
"""
GSSM-OWN causal builder + FOV reader — the file adapted to the project, not the project to the file.
=====================================================================================================
fabel's rule_extractor needs explicit causal connectives ("X causes Y") — WikiText is encyclopedic
and starves it. We don't bend the project to that limit; we build OUR OWN structure-finder that fits
GSSM: knowledge lives in CO-OCCURRENCE and NEIGHBOURHOOD, not only in surface causal verbs.

Two human analogies drive the design:
  - IMPLICIT LEARNING = the CS 1.6 aimbot FOV: you aim at ONE target (the query), but everything in
    the field-of-view cone gets acquired for free. So retrieval isn't a point lookup — it ingests
    the whole NEIGHBOURHOOD around the answer (the FOV), which is the subconscious pickup.
  - ASSOCIATION = things seen together get linked (Hebbian: "fire together, wire together"). Two
    terms co-occurring in a window become an edge, weighted by proximity — the brain's basic glue.

Builder: slide a window over text; entities = salient content words; edges = co-occurrence within the
window, weighted by 1/distance (closer = stronger), like a receptive field. This makes a DENSE graph
from any prose (no causal verbs required) — the file adapted to the project.

Reader: `fov(term, radius)` returns everything in the field-of-view cone around a term — direct
neighbours AND their neighbours out to `radius` hops, ranked by association weight. That is the
implicit-learning operator: aim at one, acquire the cone.
"""
import os, sys, json, re, math, argparse
from collections import defaultdict

_STOP = set("the a an and or but of to in on at for with as by from into over under is are was were "
            "be been being it its this that these those he she they we you i his her their our your "
            "which who whom whose what when where why how not no nor so than then thus also can may "
            "will would should could has have had do does did up out off down then there here".split())


def _content_words(text):
    return [w for w in re.findall(r"[a-z][a-z\-]{2,}", text.lower()) if w not in _STOP]


class GSSMCausal:
    """A co-occurrence association graph with an FOV reader. Built to fit GSSM, not the other way.
    MULTI-CHANNEL (David's 'one substrate, many states' applied at BUILD time): each edge carries
    several superposed channels in ONE graph; different read operators pull different views out —
    the separate graphs never exist as data, only as read-operators (maximal compression):
      - 'fwd'  : directed a→b (what FOLLOWS a)      ┐ the wire pulled forward vs backward
      - 'bwd'  : directed b→a (what LEADS TO a)      ┘
      - 'near' : tight co-occurrence (dist 1-2) = fixed phrase / entity
      - 'far'  : loose co-occurrence (dist 3-window) = thematic proximity
    `adj` stays the combined view; the channels are read via fov(..., channel=)."""
    def __init__(self):
        self.edges = defaultdict(float)          # (a,b) -> combined weight  (a<b canonical)
        self.adj = defaultdict(lambda: defaultdict(float))               # combined view
        self.chan = {c: defaultdict(lambda: defaultdict(float))          # the superposed states
                     for c in ("fwd", "bwd", "near", "far")}
        self.freq = defaultdict(int)
        self.path = None

    # ---- build / grow -------------------------------------------------------
    def add_text(self, text, window=8):
        """Slide a window; link co-occurring content words. Writes ALL channels at once (one pass,
        one substrate): directed fwd/bwd + proximity near/far, each proximity-weighted 1/dist =
        receptive-field kernel. The combined `adj` is their sum."""
        ws = _content_words(text)
        n0 = len(self.edges)
        for i, w in enumerate(ws):
            self.freq[w] += 1
            for j in range(i + 1, min(i + window, len(ws))):
                u = ws[j]
                if u == w:
                    continue
                d = j - i
                wt = 1.0 / d
                a, b = (w, u) if w < u else (u, w)
                self.edges[(a, b)] += wt
                self.adj[w][u] += wt; self.adj[u][w] += wt              # combined (symmetric)
                self.chan["fwd"][w][u] += wt                            # w precedes u
                self.chan["bwd"][u][w] += wt                            # u is preceded by w
                band = "near" if d <= 2 else "far"
                self.chan[band][w][u] += wt; self.chan[band][u][w] += wt
        return len(self.edges) - n0

    def reinforce(self, a, b, amount=1.0):
        """Hebbian path-strengthening: when a→b proves USEFUL (a query path the loop actually used),
        bump it. Paths form by REPETITION (not instant) — usefulness verified over iterations makes
        the edge stick (David: 'the paths must form'). Strengthens the combined + fwd channel."""
        a, b = a.lower(), b.lower()
        self.adj[a][b] = self.adj[a].get(b, 0.0) + amount
        self.adj[b][a] = self.adj[b].get(a, 0.0) + amount
        self.chan["fwd"][a][b] = self.chan["fwd"][a].get(b, 0.0) + amount
        key = (a, b) if a < b else (b, a)
        self.edges[key] = self.edges.get(key, 0.0) + amount

    def knows(self, term):
        return term.lower() in self.adj and len(self.adj[term.lower()]) > 0

    def _pmi(self, a, b):
        """Pointwise mutual information weight: how much MORE often a,b co-occur than chance, given
        their individual frequencies. The brain's filter for the ubiquitous — 'the/of/after' link to
        everything so PMI is ~0; a specific pair like music↔album has high PMI. Filters function-word
        noise out of the FOV without a stopword list doing all the work."""
        ab = self.adj[a].get(b, 0.0)
        if ab <= 0:
            return 0.0
        fa, fb = self.freq.get(a, 1), self.freq.get(b, 1)
        # co-occur weight normalized by the product of marginals (log-PMI, clamped ≥0)
        return ab / ((fa * fb) ** 0.5 + 1e-6)

    # ---- FOV reader (the implicit-learning operator) ------------------------
    def fov(self, term, radius=2, top=12, pmi=True, channel="combined"):
        """Field-of-view acquisition: everything in the cone around `term` out to `radius` hops,
        ranked by PMI-weighted association × proximity falloff. The aimbot FOV — aim at one, acquire
        the cone (the subconscious neighbourhood) — with the brain's filter for ubiquitous words.

        `channel` picks the READ-OPERATOR (David's one-substrate-many-states at read time):
          'combined' (default), 'fwd' (what follows), 'bwd' (what leads to), 'near', 'far'.
        Same graph, different operator → different view, no extra storage."""
        term = term.lower()
        src = self.adj if channel == "combined" else self.chan.get(channel, self.adj)
        if term not in src:
            return []
        seen = {term: 0.0}
        frontier = [(term, 1.0, 0)]
        while frontier:
            node, w, depth = frontier.pop()
            if depth >= radius:
                continue
            for nb in src[node]:
                ew = self._pmi(node, nb) if pmi else src[node][nb]
                score = w * ew / (1 + depth)     # falloff with hop distance (cone narrows)
                if nb not in seen or score > seen[nb]:
                    seen[nb] = max(seen.get(nb, 0.0), score)
                    frontier.append((nb, score, depth + 1))
        seen.pop(term, None)
        ranked = sorted(seen.items(), key=lambda kv: -kv[1])[:top]
        return [{"term": t, "assoc": round(s, 4)} for t, s in ranked]

    def neighbours(self, term):
        return set(self.adj.get(term.lower(), {}).keys())

    def stats(self):
        return {"entities": len(self.adj), "edges": len(self.edges),
                "avg_degree": round(sum(len(v) for v in self.adj.values()) / max(1, len(self.adj)), 1)}

    # ---- persistence (our own tiny format — file fits the project) ----------
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump({"edges": [[a, b, round(w, 4)] for (a, b), w in self.edges.items()],
                   "freq": dict(self.freq)}, open(path, "w"))
        self.path = path

    @classmethod
    def load(cls, path):
        g = cls(); d = json.load(open(path))
        for a, b, w in d["edges"]:
            g.edges[(a, b)] = w; g.adj[a][b] += w; g.adj[b][a] += w
        g.freq = defaultdict(int, d.get("freq", {})); g.path = path
        return g


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chars", type=int, default=2_000_000)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--out", default="demo/causal_assets/gssm_assoc.json")
    args = ap.parse_args()
    sys.path.insert(0, "src"); sys.path.insert(0, "reference")
    from length_extrap_v2 import load_wikitext2
    import time
    text, _ = load_wikitext2()
    text = text[:args.max_chars]
    g = GSSMCausal()
    t0 = time.time()
    # build in chunks (paragraph-ish) so windows don't span unrelated sections
    for para in re.split(r"\n\s*\n", text):
        if len(para) > 40:
            g.add_text(para, window=args.window)
    g.save(args.out)
    print(f"[gssm-causal] {args.max_chars:,} chars → {g.stats()} in {time.time()-t0:.1f}s (LLM-free, co-occurrence)")
    # FOV demo
    for term in ["damage", "war", "music", "engine"]:
        cone = g.fov(term, radius=2, top=6)
        if cone:
            print(f"  FOV('{term}'): {[c['term'] for c in cone]}")
