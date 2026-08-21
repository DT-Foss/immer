#!/usr/bin/env python3 -u
"""
THE CURATOR PAYS IN TRIPLETS, OR IT DOES NOT (P70).

P58 (src/surprise_filter_run.py) measured the gate's selection in TOKENS
(first-ever token-types / bigrams). The builder loop needs it in the
graph's OWN currency: VALIDATED TRIPLETS. Same protocol, one extra band:
the organism streams WT-103 exactly as in P58, but this harness ALSO
carries the raw word sequence alongside the token tape (C4Stream/HFStream
tokenize-and-drop the text; a local copy here keeps both), so every
128-token window can be handed to the fabel deterministic extractor +
14-step Foss validation gate as its own verbatim (unk-free) text.

Top-M post-ignition surprise windows vs M seeded-random windows (matched
counts, matched sizes) both go through vendor/fabel's rule_extractor +
validate_triplet_v2 — no LLM, nothing stochastic. Metrics: validated
triplets per arm, triplets/kilotoken, first-ever-entity rate, and three
clause booleans (yield ratio, entity-novelty ratio, double-pass
determinism).

REGISTRY SEMANTICS (fixed after review): "first-ever" is judged against
the WHOLE stream's word tape, not just the selected windows — an entity
counts as first-ever only if its normalized word sequence never occurred
anywhere in the stream text BEFORE the window's tape position. This is
a cheap membership check, not a full-stream extraction pass: every
n-gram (n=1..5, the span entity strings realistically cover) over the
ENTIRE word tape is indexed once for its first occurrence position
(mirrors P58's first_tok_pos/first_bi_pos registry pattern), and an
extracted entity is first-ever iff its own n-gram's first occurrence
position equals (or exceeds) the window's own tape position — i.e.
this window IS where it first appears in true stream order.
"""
import argparse
import json
import os
import random
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
FABEL_EXTRACT = os.path.join(REPO_ROOT, "vendor", "fabel", "extract")
sys.path.insert(0, FABEL_EXTRACT)

import re
import torch
torch.set_num_threads(1)

import portable_organism as po
from length_extrap_v2 import tokenize as _lel_tokenize

# fabel deterministic extractor + 14-step Foss validation gate, in-process.
from rule_extractor import extract_from_text          # noqa: E402
from extract_causal_triplets_v2 import validate_triplet_v2  # noqa: E402

_WORD_RE = re.compile(r"[a-zA-Z]+")


class HFStreamWithText(po.C4Stream):
    """Local copy of source_swap_run.HFStream that ALSO keeps the raw
    ORIGINAL-CASE text a document's tokens came from, segment-per-token, so
    a window of token positions can be mapped back to its verbatim source
    text — punctuation and capitalization intact, not a flattened lowercase
    word list. C4Stream's next_block() tokenizes-and-discards the text;
    this override keeps a parallel `pending_segs` buffer in lockstep with
    `pending` (both grown and drained the same way, so index i in a
    returned token block lines up with index i in the matching segment
    block). Each segment is the ORIGINAL text span from the end of the
    previous [a-zA-Z]+ match through the end of this one — i.e. it carries
    its own leading whitespace/punctuation and true casing, so joining
    segments verbatim reconstructs the exact source text (sentence
    boundaries, capitalization) the rule_extractor's sentence splitter and
    connective patterns need. Nothing about the gate/model path changes —
    this only widens what the stream hands back."""

    SOURCES = {
        "c4": ("allenai/c4", "en"),
        "wt103": ("Salesforce/wikitext", "wikitext-103-raw-v1"),
    }

    def __init__(self, source, stoi, unk, block=8192, skip_docs=0):
        super().__init__(stoi, unk, block=block, skip_docs=skip_docs)
        self.source = source
        self.pending_segs = []

    def _connect(self):
        from datasets import load_dataset
        path, name = self.SOURCES[self.source]
        ds = load_dataset(path, name, split="train", streaming=True)
        if self.docs:
            ds = ds.skip(self.docs)
        self._it = iter(ds)

    def next_block(self):
        while len(self.pending) < self.block:
            try:
                if self._it is None:
                    self._connect()
                row = next(self._it)
            except StopIteration:
                self.docs = 0
                self._it = None
                continue
            except Exception as e:
                self.reconnects += 1
                print(f"[stream] {type(e).__name__}: {e} — reconnect #{self.reconnects} "
                      f"at doc {self.docs:,}", flush=True)
                self._it = None
                time.sleep(min(30, 2 * self.reconnects))
                continue
            self.docs += 1
            t = row.get("text", "") if isinstance(row, dict) else ""
            if t.strip():
                # tokenize() matches [a-zA-Z]+ on t.lower() — NOT on t. For
                # most text len(t) == len(t.lower()) so matching directly on
                # t would agree, but Python's .lower() is not always
                # length-preserving (e.g. U+0130 LATIN CAPITAL LETTER I WITH
                # DOT ABOVE -> 'i' + COMBINING DOT ABOVE, one codepoint
                # becomes two), which silently shifts match boundaries and
                # desyncs the token/segment counts (crashed a live run on
                # WT-103 prose containing "İstanbul"). Fix: match on
                # t.lower() (byte-for-byte what tokenize() does) and build
                # an index map back to t, so segments still carry ORIGINAL
                # casing/punctuation but boundaries are guaranteed aligned.
                lowered_chars = []
                orig_idx = []
                for i, ch in enumerate(t):
                    lc = ch.lower()
                    lowered_chars.append(lc)
                    orig_idx.extend([i] * len(lc))
                orig_idx.append(len(t))   # sentinel for end-of-string matches
                t_lower = "".join(lowered_chars)
                matches = list(_WORD_RE.finditer(t_lower))
                toks = [self.stoi.get(m.group(0), self.unk) for m in matches]
                # cross-check against the shared tokenize() so any future
                # drift between the two [a-zA-Z]+ passes fails loudly here,
                # at the source document, not deep inside a 3000-chunk run.
                assert toks == _lel_tokenize(t, self.stoi, self.unk)
                # each segment = ORIGINAL text from the end of the PREVIOUS
                # match through the end of THIS match (mapped through
                # orig_idx), so it carries its own leading whitespace/
                # punctuation and true casing; the first segment in a
                # document also carries any leading text.
                segs = []
                prev_end = 0
                for m in matches:
                    orig_end = orig_idx[m.end()]
                    segs.append(t[prev_end:orig_end])
                    prev_end = orig_end
                self.pending.extend(toks)
                self.pending_segs.extend(segs)
        out_t, self.pending = self.pending[:self.block], self.pending[self.block:]
        out_s, self.pending_segs = self.pending_segs[:self.block], self.pending_segs[self.block:]
        return out_t, out_s


class ChunkFeederWithText:
    """ChunkFeeder that also returns lane-0's raw original-case text
    segments for the emitted x-window, so the caller can grow a segment
    tape in lockstep with the token tape (mirrors surprise_filter_run's
    `tape.extend(...)` pattern)."""

    def __init__(self, stream, batch, chunk):
        self.stream, self.B, self.K = stream, batch, chunk
        self.bufs = [[] for _ in range(batch)]
        self.sbufs = [[] for _ in range(batch)]

    def next_xy(self):
        B, K = self.B, self.K
        for b in range(B):
            while len(self.bufs[b]) < K + 1:
                toks, segs = self.stream.next_block()
                self.bufs[b].extend(toks)
                self.sbufs[b].extend(segs)
        x = torch.tensor([self.bufs[b][:K] for b in range(B)], dtype=torch.long)
        y = torch.tensor([self.bufs[b][1:K + 1] for b in range(B)], dtype=torch.long)
        lane0_segs = self.sbufs[0][:K]
        for b in range(B):
            del self.bufs[b][:K]
            del self.sbufs[b][:K]
        return x, y, lane0_segs


def window_source_text(seg_tape, start, wlen):
    """Verbatim (punctuation- and case-intact) source text for a window:
    joining the original per-token segments reconstructs the exact source
    span, since each segment already carries its own leading
    whitespace/punctuation (see HFStreamWithText.next_block)."""
    return "".join(seg_tape[start:min(start + wlen, len(seg_tape))]).strip()


def iter_windows(substrate="wt103", chunks=3000, window_tokens=128,
                  d_model=128, batch=8, chunk_size=64, seed=42,
                  q=0.75, window=500, min_window=100, ignition_chunks=100,
                  stream_cls=None):
    """MVP-4 CONTRACT (builder_run.py depends on this exact signature and
    yield shape — do not change without flagging the team lead).

    Streams `substrate` through a fresh Organism with the SAME cadence
    knobs P70 uses (gate q/window/min_window/ignition_chunks — see
    src/portable_organism.py's Organism.step_gated), and yields one
    (tape_pos, window_text, surprise, gated) tuple per chunk, in stream
    order, for chunks 1..`chunks`:

        tape_pos    — int, this chunk's start position in the token/segment
                      tape (same coordinate space P70's chunk_surprise/
                      first_pos use)
        window_text — str, verbatim (punctuation/case-intact) source text
                      for the `window_tokens`-token window starting at
                      tape_pos (via window_source_text on a running segment
                      tape — grows every chunk, so windows never read past
                      what has streamed so far)
        surprise    — float, this chunk's gate surprise s (org.step_gated's
                      own scalar; NOT windowed/smoothed)
        gated       — bool, whether the organism actually trained on this
                      chunk (new builder policy: extraction runs on EVERY
                      window regardless of gated — the gate curates what
                      the organism LEARNS from, not what the curator may
                      extract from; gated is exposed so the builder loop
                      can use it as a routing signal, e.g. weighting or
                      logging, without re-deriving it)

    This is the same streaming/cadence path main() drives for the P70
    measurement (HFStreamWithText + ChunkFeederWithText + Organism.
    step_gated) — one implementation, so a builder loop built on this
    generator sees exactly the stream P70 was scored against, not a
    re-implementation that could drift from it.

    `stream_cls` lets a caller substitute a stream (e.g. a FakeStream in a
    test) without touching this function; defaults to HFStreamWithText."""
    if stream_cls is None:
        stream_cls = HFStreamWithText

    po.D_MODEL, po.BATCH, po.CHUNK = d_model, batch, chunk_size
    po.GATE_Q, po.GATE_WINDOW, po.MIN_WINDOW, po.IGNITION_CHUNKS = \
        q, window, min_window, ignition_chunks
    vocab, stoi, unk, mask, val_ids = po.get_vocab()

    torch.manual_seed(seed)
    org = po.Organism("curator", len(vocab), mask, seed=seed)
    stream = stream_cls(substrate, stoi, unk)
    feeder = ChunkFeederWithText(stream, po.BATCH, po.CHUNK)

    seg_tape = []
    for ci in range(1, chunks + 1):
        x, y, lane0_segs = feeder.next_xy()
        s, gated, nll = org.step_gated(x, y)
        tape_pos = len(seg_tape)
        seg_tape.extend(lane0_segs)
        text = window_source_text(seg_tape, tape_pos, window_tokens)
        yield tape_pos, text, float(s), bool(gated)


def normalize_entity(s):
    """The one normalization used everywhere an entity string is compared:
    the registry (build_ngram_first_pos/is_first_ever), entity_keys(), and
    the MVP-4 contract's trigger_key/outcome_key all call this — one
    normalization, so a builder-loop membership check against the registry
    means the same thing it means inside this harness's own measurement."""
    return s.strip().lower()


def extract_validated(text, domain="wt103", source="p70"):
    """MVP-4 CONTRACT (builder_run.py depends on this exact signature and
    schema — do not change without flagging the team lead).

    fabel in-process path: rule_extractor.extract_from_text -> the 14-step
    Foss validation gate (validate_triplet_v2). Mirrors
    vendor/fabel/extract/extract_to_db.py's main loop exactly (sequential
    seen_kept accumulation within one text, deterministic — no LLM, nothing
    stochastic). This is the SAME function P70's own measurement calls
    (run_arm below), so the builder loop's extraction is byte-identical to
    what P70 scored — one source of truth, not a re-implementation that
    could drift from it.

    Returns a list[dict], each with at least:
        trigger, mechanism, outcome   — the raw fabel triplet fields
        trigger_key, outcome_key      — normalize_entity(trigger/outcome),
                                         the graph's own entity currency
                                         (matches entity_keys() below and
                                         the registry's n-gram normalization)
    plus whatever extract_from_text/validate_triplet_v2 attach (confidence,
    quality_score, evidence_sentence, quantification, domain, source, and
    optional condition/co_causes/effect_size/population/temporal)."""
    triplets = []
    seen_kept = []
    for rt in extract_from_text(text, domain=domain, source=source):
        d = rt.as_dict()
        res = validate_triplet_v2(d, seen_kept, rt.evidence_sentence)
        if not res.is_valid:
            continue
        d["confidence"] = res.confidence
        d["quality_score"] = res.quality_score
        d["trigger_key"] = normalize_entity(d["trigger"])
        d["outcome_key"] = normalize_entity(d["outcome"])
        seen_kept.append(d)
        triplets.append(d)
    return triplets


def entity_keys(triplet):
    """The graph's own entity currency: normalized trigger/outcome strings.
    Reads trigger_key/outcome_key when present (the extract_validated
    contract always sets them); falls back to normalizing trigger/outcome
    directly for any triplet dict built outside that path."""
    if "trigger_key" in triplet and "outcome_key" in triplet:
        return {triplet["trigger_key"], triplet["outcome_key"]}
    return {normalize_entity(triplet["trigger"]), normalize_entity(triplet["outcome"])}


MAX_ENTITY_NGRAM = 5   # entity strings are short noun phrases; 5 words covers them


def build_ngram_first_pos(word_tape, max_n=MAX_ENTITY_NGRAM):
    """First-occurrence tape position of every word n-gram (n=1..max_n) over
    the WHOLE stream's word tape — one linear pass, mirrors P58's
    first_tok_pos/first_bi_pos registry. An entity string's first_pos tells
    us exactly where in true stream order it first appeared, independent of
    which window (if any) happened to select it."""
    first_pos = {}
    T = len(word_tape)
    for n in range(1, max_n + 1):
        for i in range(T - n + 1):
            gram = " ".join(word_tape[i:i + n])
            if gram not in first_pos:
                first_pos[gram] = i
    return first_pos


def is_first_ever(entity_str, window_pos, first_pos):
    """True iff entity_str's first occurrence in the WHOLE stream is at or
    after window_pos — i.e. this window's tape position is where it first
    appears in stream order (not merely the first SELECTED window to
    contain it)."""
    fp = first_pos.get(entity_str)
    if fp is None:
        return False   # not found as an indexed n-gram (e.g. > max_n words) — conservative
    return fp >= window_pos


def run_arm(windows, seg_tape, first_pos, window_tokens):
    """windows: list of (chunk_idx, surprise, tape_pos). Runs the extractor
    over each window's verbatim (punctuation/case-intact) source text;
    first-ever is judged against first_pos, the whole-stream n-gram
    registry over normalized words (see build_ngram_first_pos), not
    against windows seen so far."""
    n_triplets = 0
    n_tokens = 0
    n_first_ever = 0
    all_triplets = []
    for _, _, pos in windows:
        text = window_source_text(seg_tape, pos, window_tokens)
        n_tokens += min(window_tokens, len(seg_tape) - pos)
        trips = extract_validated(text)
        for t in trips:
            is_first = any(is_first_ever(k, pos, first_pos) for k in entity_keys(t))
            if is_first:
                n_first_ever += 1
            n_triplets += 1
            all_triplets.append(t)
    return {"triplets": n_triplets, "tokens": n_tokens,
            "first_ever": n_first_ever, "raw": all_triplets}


def main():
    ap = argparse.ArgumentParser(description="P70: the curator pays in triplets")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--chunks", type=int, default=3000)
    ap.add_argument("--top-m", type=int, default=150)
    ap.add_argument("--window-tokens", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--q", type=float, default=0.75)
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--min-window", type=int, default=100)
    ap.add_argument("--ignition-chunks", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "curator_yield.json"))
    args = ap.parse_args()
    if args.smoke:
        args.chunks, args.top_m = 300, 20

    po.D_MODEL, po.BATCH, po.CHUNK = args.d_model, args.batch, args.chunk_size
    po.GATE_Q, po.GATE_WINDOW, po.MIN_WINDOW, po.IGNITION_CHUNKS = \
        args.q, args.window, args.min_window, args.ignition_chunks
    vocab, stoi, unk, mask, val_ids = po.get_vocab()
    V = len(vocab)
    K, B = po.CHUNK, po.BATCH

    torch.manual_seed(args.seed)
    org = po.Organism("curator", V, mask, seed=args.seed)
    stream = HFStreamWithText("wt103", stoi, unk)
    feeder = ChunkFeederWithText(stream, B, K)

    # lane-0 token tape + parallel ORIGINAL-CASE segment tape (index-aligned,
    # verbatim source text incl. punctuation) + per-chunk surprise
    tape = []
    seg_tape = []
    chunk_surprise = []          # (chunk_idx, s, tape_pos_of_chunk_start)
    t0 = time.time()
    for ci in range(1, args.chunks + 1):
        x, y, lane0_segs = feeder.next_xy()
        s, gated, nll = org.step_gated(x, y)
        chunk_surprise.append((ci, float(s), len(tape)))
        tape.extend(int(v) for v in x[0])
        seg_tape.extend(lane0_segs)
        if ci % 500 == 0:
            print(f"[stream] {ci}/{args.chunks} | s {s:.3f} | tape {len(tape):,} tok "
                  f"| {time.time()-t0:.0f}s", flush=True)
    assert len(tape) == len(seg_tape), "token/segment tapes must stay index-aligned"
    # normalized word tape (lowercase, punctuation-stripped) derived from the
    # same segments, for the entity registry — each segment carries exactly
    # one [a-zA-Z]+ match (see HFStreamWithText.next_block), so extracting
    # it again with the same regex recovers the single normalized word.
    norm_word_tape = ["".join(_WORD_RE.findall(seg.lower())) for seg in seg_tape]

    post = [c for c in chunk_surprise if c[0] > args.ignition_chunks]
    top = sorted(post, key=lambda c: -c[1])[:args.top_m]
    rng = random.Random(args.seed + 7)
    rand1 = rng.sample(post, min(args.top_m, len(post)))

    print(f"[p70] windows selected: surprise={len(top)} random={len(rand1)} "
          f"| tape {len(tape):,} tok | indexing whole-stream n-gram registry...",
          flush=True)

    # whole-stream registry: first occurrence position of every 1..5-word
    # n-gram over the FULL normalized word tape (not just selected windows)
    # — this is what "first-ever in the stream" means, per review.
    t1 = time.time()
    first_pos = build_ngram_first_pos(norm_word_tape)
    print(f"[p70] registry indexed | {len(first_pos):,} n-grams | {time.time()-t1:.0f}s "
          f"| extracting...", flush=True)

    # deterministic processing order: both arms in true STREAM order (stable
    # even though it no longer affects first-ever, which is judged against
    # the whole-stream registry independent of processing order).
    top_by_pos = sorted(top, key=lambda c: c[2])
    rand_by_pos = sorted(rand1, key=lambda c: c[2])

    arm_results = {
        "surprise": run_arm(top_by_pos, seg_tape, first_pos, args.window_tokens),
        "random": run_arm(rand_by_pos, seg_tape, first_pos, args.window_tokens),
    }

    g_sur, g_rand = arm_results["surprise"], arm_results["random"]

    def per_kt(n_trip, n_tok):
        return (n_trip / n_tok * 1000) if n_tok else 0.0

    def novelty_rate(n_first, n_trip):
        return (n_first / n_trip) if n_trip else 0.0

    yield_sur = per_kt(g_sur["triplets"], g_sur["tokens"])
    yield_rand = per_kt(g_rand["triplets"], g_rand["tokens"])
    yield_ratio = yield_sur / yield_rand if yield_rand else float("inf")

    nov_sur = novelty_rate(g_sur["first_ever"], g_sur["triplets"])
    nov_rand = novelty_rate(g_rand["first_ever"], g_rand["triplets"])
    novelty_ratio = nov_sur / nov_rand if nov_rand else (float("inf") if nov_sur else 0.0)

    # (c) double-pass determinism: re-run the extractor over the SAME
    # surprise windows' text and require byte-identical triplet output.
    def pass_again(windows):
        out = []
        for w in windows:
            text = window_source_text(seg_tape, w[2], args.window_tokens)
            out.extend(extract_validated(text))
        return out

    pass1_text = json.dumps(g_sur["raw"], sort_keys=True)
    pass2_raw = pass_again(top_by_pos)
    pass2_text = json.dumps(pass2_raw, sort_keys=True)
    p70c_pass = (pass1_text == pass2_text)

    out = {"p70": True, "smoke": args.smoke,
           "cadence": {"d_model": po.D_MODEL, "batch": B, "chunk": K,
                       "q": po.GATE_Q, "window": po.GATE_WINDOW,
                       "min_window": po.MIN_WINDOW, "ignition_chunks": po.IGNITION_CHUNKS},
           "config": {"chunks": args.chunks, "top_m": args.top_m,
                      "window_tokens": args.window_tokens, "substrate": "wt103",
                      "vocab_source": "wikitext-2/5000",
                      "vocab_note": "the token stoi/vocab is built on WikiText-2 "
                                     "(5000 words, portable_organism.get_vocab), "
                                     "not WT-103 — most WT-103 tokens hit unk, so "
                                     "window text/entities use the raw word tape "
                                     "(HFStreamWithText), never vocab[token_id]",
                      "registry": "whole-stream n-gram (n=1..%d) first-occurrence "
                                   "index over the full word tape, not window-only"
                                   % MAX_ENTITY_NGRAM},
           "surprise_windows": {"triplets": g_sur["triplets"], "tokens": g_sur["tokens"],
                                 "first_ever": g_sur["first_ever"],
                                 "triplets_per_kilotoken": round(yield_sur, 4),
                                 "first_ever_entity_rate": round(nov_sur, 4)},
           "random_windows": {"triplets": g_rand["triplets"], "tokens": g_rand["tokens"],
                               "first_ever": g_rand["first_ever"],
                               "triplets_per_kilotoken": round(yield_rand, 4),
                               "first_ever_entity_rate": round(nov_rand, 4)},
           "p70a_yield_ratio": round(yield_ratio, 4) if yield_ratio != float("inf") else None,
           "p70a_pass": bool(yield_ratio >= 1.3) if yield_ratio != float("inf") else True,
           "p70b_entity_novelty_ratio": round(novelty_ratio, 4) if novelty_ratio != float("inf") else None,
           "p70b_pass": bool(novelty_ratio >= 1.5) if novelty_ratio != float("inf") else True,
           "p70c_pass": bool(p70c_pass)}
    path = args.out if not args.smoke else args.out.replace(".json", "_smoke.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[p70] yield ratio {out['p70a_yield_ratio']} (bar 1.3) | novelty ratio "
          f"{out['p70b_entity_novelty_ratio']} (bar 1.5) | double-pass det. {p70c_pass} | "
          f"a:{out['p70a_pass']} b:{out['p70b_pass']} c:{out['p70c_pass']} -> {path}", flush=True)


if __name__ == "__main__":
    main()
