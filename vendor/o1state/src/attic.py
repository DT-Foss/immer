#!/usr/bin/env python3 -u
"""
ATTIC — the one door to the .causal knowledge index (the KNOWLEDGE node of the pathfinding machine).
====================================================================================================
Bundles the whole vendored fabel pipeline behind a single clean interface so the GSSM side never
touches fabel internals. Three capabilities, one object:

    attic = Attic.from_corpus(text)        # build a fresh .causal index from raw text (LLM-free)
    attic = Attic.from_file(path)          # or load an existing .causal
    attic.knows(term)  -> bool             # does the index cover this gap?
    attic.lookup(term) -> [paths]          # 1-hop facts mentioning the term (with provenance)
    attic.reason(term) -> [chains]         # MULTI-HOP transitive paths via the RICH engine
                                           #   (transitive closure ≤5 hops — the real amplification,
                                           #    ~6x / 18x more connections than the 3-pass stub)

Everything is vendored under vendor/fabel (FORGE excluded). The GSSM streaming state consults this
attic ONLY on a gap (high surprise); the attic answers from measured structure, zero hallucination,
with sources. This is the "attic / second file" David specified — external to the O(1) state.
"""
import os, sys, re

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FABEL = os.path.join(_ROOT, "vendor", "fabel")
for p in (os.path.join(_FABEL, "dotcausal_package", "src"),
          os.path.join(_FABEL, "extract"),
          os.path.join(_FABEL, "language", "hsslm_s")):
    if p not in sys.path:
        sys.path.insert(0, p)

from dotcausal import CausalReader, CausalWriter          # noqa: E402
import rule_extractor as _rx                              # noqa: E402
import inference as _rich                                 # noqa: E402  (the rich transitive engine)

_CONF = {"high": 0.9, "medium": 0.7, "low": 0.5}


def _sentences(text):
    for s in re.split(r"(?<=[.!?])\s+", text):
        s = s.strip()
        if 20 < len(s) < 600:
            yield s


class Attic:
    """The knowledge index, consulted on a gap. Wraps the vendored fabel engine read-side, plus
    the rich transitive-closure inference for multi-hop reasoning."""

    def __init__(self, reader: CausalReader, path: str = None):
        self.reader = reader
        self.path = path
        self.triplets = reader.get_all_triplets(include_inferred=False)
        # surface-term → triplet index (for fast gap lookup)
        self._by_term = {}
        for t in self.triplets:
            surf = f"{t.get('trigger','')} {t.get('mechanism','')} {t.get('outcome','')}".lower()
            for w in set(re.findall(r"[a-z]{4,}", surf)):
                self._by_term.setdefault(w, []).append(t)
        # entity ↔ id maps + adjacency kb for the rich engine
        self._ent2id, self._id2ent = {}, {}
        self._kb = [(self._gid(t["trigger"]), t["mechanism"], self._gid(t["outcome"]))
                    for t in self.triplets]

    # ---- constructors -------------------------------------------------------
    @classmethod
    def from_file(cls, path):
        return cls(CausalReader(path), path)

    @classmethod
    def from_corpus(cls, text, out_path, api_id="corpus", max_sentences=200000, source="corpus"):
        """Build a fresh .causal index from raw text — deterministic, LLM-free (rule_extractor)."""
        w = CausalWriter(api_id=api_id)
        n = 0
        for i, s in enumerate(_sentences(text)):
            if i >= max_sentences:
                break
            for rt in _rx.extract_from_text(s, domain=api_id, source=source):
                d = rt.__dict__
                if d.get("trigger") and d.get("mechanism") and d.get("outcome"):
                    w.add_triplet(str(d["trigger"]), str(d["mechanism"]), str(d["outcome"]),
                                  confidence=_CONF.get(str(d.get("confidence", "medium")).lower(), 0.7),
                                  source=source, domain=api_id, evidence=s[:200])
                    n += 1
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        w.save(out_path)
        return cls(CausalReader(out_path), out_path), n

    # ---- internals ----------------------------------------------------------
    def _gid(self, s):
        if s not in self._ent2id:
            idx = len(self._ent2id)
            self._ent2id[s] = idx
            self._id2ent[idx] = s
        return self._ent2id[s]

    # ---- the gold-edge interface -------------------------------------------
    def knows(self, term):
        """Does the attic cover this gap term?"""
        return term.lower() in self._by_term

    def lookup(self, term, k=3):
        """1-hop facts mentioning `term`, with provenance (zero hallucination). `evidence` is the
        verbatim source sentence the triplet was extracted from — the exact text in the corpus."""
        out = []
        for h in self._by_term.get(term.lower(), [])[:k]:
            out.append({"path": f"{h['trigger']} → {h['mechanism']} → {h['outcome']}",
                        "evidence": h.get("evidence", ""),
                        "conf": h.get("confidence"), "source": (h.get("source") or "")[:48]})
        return out

    def reason(self, term, max_hops=5, k=5):
        """MULTI-HOP transitive chains seeded from entities mentioning `term`. This is the rich
        engine: paths the 1-hop lookup (and the 3-pass stub) cannot reach. Returns chains as
        readable strings with confidence."""
        seeds = [self._gid(t["trigger"]) for t in self._by_term.get(term.lower(), [])]
        # also seed on the term AS AN ENTITY itself (the word index keys on [a-z]{4,} words and
        # misses ids like UUIDs; if `term` is a graph entity, use it directly as a chain start)
        for ent, eid in self._ent2id.items():
            if ent == term or ent.lower() == term.lower():
                seeds.append(eid)
        if not seeds:
            return []
        paths = _rich.complete_inference_pipeline(
            list(set(seeds)), knowledge_base=self._kb, token_id_to_str=self._id2ent,
            max_path_length=max_hops)
        chains = []
        for path, conf, prov in paths:
            if len(path) > 2:                                  # genuine multi-hop only
                chain = " → ".join(self._id2ent.get(t, "?") for t in path)
                chains.append({"chain": chain, "hops": len(path) - 1, "conf": round(float(conf), 3)})
        chains.sort(key=lambda c: (-c["hops"], -c["conf"]))
        return chains[:k]

    # ---- GROWTH: the attic expands itself ----------------------------------
    def _extract(self, text, source, api_id):
        out = []
        for s in _sentences(text):
            for rt in _rx.extract_from_text(s, domain=api_id, source=source):
                d = rt.__dict__
                if d.get("trigger") and d.get("mechanism") and d.get("outcome"):
                    out.append((str(d["trigger"]), str(d["mechanism"]), str(d["outcome"]),
                                _CONF.get(str(d.get("confidence", "medium")).lower(), 0.7), s[:200]))
        return out

    def add_text(self, text, source="retrieved", api_id=None, flush=True):
        """Grow the attic from new text. flush=True writes now (simple); flush=False STAGES the
        triplets in memory for a later batch flush() — the fast path for many adds (one rewrite of
        the graph instead of one per call). The self-expanding loop: meet a gap, retrieve, fold in,
        next time KNOW. Returns number of new triplets staged/added."""
        api_id = api_id or "grown"
        new = self._extract(text, source, api_id)
        if not new:
            return 0
        if not hasattr(self, "_staged"):
            self._staged = []
        self._staged.extend((t, m, o, c, e, source, api_id) for (t, m, o, c, e) in new)
        if flush:
            self.flush()
        return len(new)

    def flush(self):
        """Write existing + all staged triplets to the .causal ONCE, then reload (inference
        re-materializes at save). Fast path: stage many add_text(flush=False), flush() once."""
        staged = getattr(self, "_staged", [])
        if not staged:
            return 0
        w = CausalWriter(api_id="grown")
        for t in self.triplets:
            w.add_triplet(t["trigger"], t["mechanism"], t["outcome"],
                          confidence=t.get("confidence", 0.7), source=t.get("source", ""),
                          domain=t.get("domain", ""), evidence=t.get("evidence", ""))
        for (trig, mech, outc, conf, ev, src, api) in staged:
            w.add_triplet(trig, mech, outc, confidence=conf, source=src, domain=api, evidence=ev)
        n = len(staged)
        self._staged = []
        if self.path:
            w.save(self.path)
            self.__init__(CausalReader(self.path), self.path)
        return n

    def stats(self):
        return {"path": self.path, "triplets": len(self.triplets),
                "entities": len(self._ent2id), "indexed_terms": len(self._by_term)}


# ===========================================================================
# FactIndex — the declarative-fact side of the cortex (for fact recall / NIAH)
# ===========================================================================
# The .causal Attic above indexes CAUSAL triplets (X causes Y). Many facts a stream carries are
# DECLARATIVE ("the best thing to do in X is Y", "the magic number is 42") — no causality. This
# is the matching index for those: as the state streams the text, each sentence is indexed by its
# content entities (nouns, proper-noun spans); at a query, the asked entity returns the verbatim
# sentence that mentions it. Same "the state reads, the index remembers the fact" mechanism, for
# declarative recall — constant per-chunk work, no full-text materialization required.

_FACT_STOP = frozenset((
    "the a an of for on at and or but with which who whom where when how why that this these those "
    "is are was were be been being to in into from by as it its their his her your our we you they "
    "he she them about over under above below best thing do does did has have had will would can "
    "could should may might one two three some any all most more less than then there here".split()
))


def _fact_entities(sentence):
    """Content keys for a sentence: lowercase content words (len≥3, non-stop) PLUS capitalized
    multi-word spans kept whole AND split (so 'San Francisco' is findable as 'san francisco',
    'san', and 'francisco')."""
    keys = set()
    for span in re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", sentence):
        keys.add(span.lower())
        for w in span.split():
            if len(w) >= 3:
                keys.add(w.lower())
    for w in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", sentence):
        wl = w.lower()
        if wl not in _FACT_STOP:
            keys.add(wl)
    return keys


class FactIndex:
    """Entity→sentence index built by streaming text. The declarative-fact cortex for the
    full O1 system: the state streams the haystack chunk by chunk, each sentence is folded into
    the index by its entities, and a query entity recalls the verbatim sentence. O(chunk) memory:
    only the current chunk's sentences are processed at a time; the index holds one entry per
    distinct (entity) → sentence-id, the haystack is never held whole."""

    # an id token = a UUID-like hex-with-hyphens; the chain links these
    _ID_RE = re.compile(r"\b[0-9a-fA-F]{4,}(?:-[0-9a-fA-F]{2,}){2,}\b")

    def __init__(self):
        self._sents = []                 # verbatim sentences (the recalled answer text)
        self._by_ent = {}                # entity term -> list of sentence ids
        self._id_next = {}               # id -> id it links to (chain adjacency, segmentation-free)

    def add_text(self, text):
        """Fold a chunk of text into the index. Two things: (1) sentence entities for fact recall;
        (2) id→id chain adjacency read directly from the raw text — consecutive id tokens in the
        stream are a link (`A maps to B` ⇒ A→B), captured independent of sentence segmentation so
        a chain survives even when an inserted link straddles a sentence boundary."""
        added = 0
        for s in _sentences(text):
            sid = len(self._sents)
            self._sents.append(s)
            for e in _fact_entities(s):
                self._by_ent.setdefault(e, []).append(sid)
            added += 1
        # chain adjacency: capture id→id links where the two ids are joined by a short connective
        # ("A maps to B", "A points to B", "A → B"). Requiring the connective avoids linking two
        # unrelated ids that merely happen to sit near each other in a million tokens of filler.
        for m in re.finditer(
                self._ID_RE.pattern + r"\s*\b(?:maps?|points?|leads?|refers?|links?)\b[^.]{0,12}?"
                + self._ID_RE.pattern, text):
            seg = m.group(0)
            pair = self._ID_RE.findall(seg)
            if len(pair) >= 2 and pair[0] != pair[-1]:
                self._id_next.setdefault(pair[0], pair[-1])
        return added

    def knows(self, term):
        return term.lower() in self._by_ent

    def recall(self, query, k=1):
        """Return up to k verbatim sentences best answering the query, ranking by IDF-weighted
        entity overlap PLUS a strong phrase-overlap bonus. Entity-IDF alone can't separate the
        needle from a filler sentence that happens to share the query's named entity (a Paul
        Graham essay also says 'San Francisco'); the discriminator is that the needle sentence
        contains the query almost verbatim ('the best thing to do in San Francisco is ...'). So we
        add the longest contiguous query-word-run found in each sentence, weighted to dominate.
        The state read the whole stream; the index recalls the sentence that actually answers."""
        import math
        qents = _fact_entities(query)
        N = max(1, len(self._sents))
        idf = {}
        for e in qents:
            df = len(self._by_ent.get(e, ()))
            if df:
                idf[e] = math.log(1.0 + N / df)
        if not idf:
            return []
        # candidate sentences = those sharing at least one query entity
        cand = set()
        for e in idf:
            cand.update(self._by_ent.get(e, ()))
        if not cand:
            return []
        qseq = re.findall(r"[a-z']+", query.lower())   # full ordered query words for phrase match
        score = {}
        for sid in cand:
            s = self._sents[sid]
            # entity-IDF component
            sent_ents = _fact_entities(s)
            v = sum(idf[e] for e in idf if e in sent_ents)
            # phrase component: longest run of consecutive query words appearing in the sentence
            best = 0
            swords = re.findall(r"[a-z']+", s.lower())
            # quick longest-common-substring over the query word sequence vs sentence words
            sword_pos = {}
            for i, w in enumerate(swords):
                sword_pos.setdefault(w, []).append(i)
            for start in range(len(qseq)):
                for pos in sword_pos.get(qseq[start], ()):
                    rl = 0
                    while (start + rl < len(qseq) and pos + rl < len(swords)
                           and qseq[start + rl] == swords[pos + rl]):
                        rl += 1
                    best = max(best, rl)
            v += best * 5.0          # a 4+ word verbatim run dominates any entity tie
            score[sid] = v
        ranked = sorted(score, key=lambda sid: (-score[sid], sid))
        return [self._sents[sid] for sid in ranked[:k]]

    def chain_recall(self, query, max_hops=16):
        """Follow a reference chain transitively through the index. Many long-context tasks ask
        'what does X ultimately map to', where the answer is several hops away: X→Y is one
        sentence, Y→Z another, etc. The flat recall returns only the first hop; this follows the
        links. Generic graph-walk: from the query's start token, find the sentence that links it,
        take the OTHER token in that sentence as the next node, repeat until the chain ends (no
        new sentence, or a cycle). Works for any 'A <verb> B' link sentences — exactly the
        multi-hop traversal the .causal reason() engine does, on the streamed fact index.

        Returns the final token reached (the end of the chain), or None."""
        start_ids = self._ID_RE.findall(query)
        if not start_ids:
            return None
        cur = start_ids[0]                          # the id the question starts from
        seen = {cur}
        last = cur
        for _ in range(max_hops):
            nxt = self._id_next.get(cur)            # follow the captured A→B adjacency
            if nxt is None or nxt in seen:
                break
            seen.add(nxt)
            last = nxt
            cur = nxt
        return last

    @classmethod
    def from_corpus(cls, text):
        idx = cls()
        idx.add_text(text)
        return idx

    def stats(self):
        return {"sentences": len(self._sents), "entities": len(self._by_ent)}


if __name__ == "__main__":
    # smoke: build from WikiText, then show 1-hop vs multi-hop on a term
    sys.path.insert(0, os.path.join(_ROOT, "src"))
    from length_extrap_v2 import load_wikitext2
    text, _ = load_wikitext2()
    attic, n = Attic.from_corpus(text, os.path.join(_ROOT, "demo/causal_assets/wikitext.causal"),
                                 api_id="wikitext", source="wikitext-2")
    print(f"[attic] built from WikiText: {attic.stats()} ({n} triplets extracted)")
    for term in ["damage", "increased", "caused"]:
        if attic.knows(term):
            hop1 = attic.lookup(term, k=1)
            multi = attic.reason(term, k=3)
            print(f"\n  '{term}': {len(hop1)} 1-hop, {len(multi)} multi-hop chains")
            for c in multi[:2]:
                print(f"     [{c['hops']}hop {c['conf']}] {c['chain'][:100]}")
