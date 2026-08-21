#!/usr/bin/env python3 -u
"""
POS DREAM — does the organism learn from ITS OWN generated stream, without any
stored memory at all?
============================================================================
pos_sleep.py proved that gradient training on REPLAYED stored surprise spans
(results/pos_index.jsonl) beats fresh C4 data, per gradient token. That is
sleep consolidation via memory: the organism stores real surprises and
replays them.

This file (P19, analysis/PREDICTIONS.md) asks the harder question: what if
storage is not required at all? What if the model can generate its OWN
training stream — dream instead of remember — and gradient-train on that? If
dreaming buys real heldout improvement, memory becomes optional (generative
consolidation, cheaper than storing 20k spans). If dreaming loses to stored
replay, that is proof that real surprises carry information self-sampling
cannot invent: the spans in pos_index.jsonl are not just "any hard tokens",
they are irreplaceable.

Four arms, all forked (deepcopy) from the IDENTICAL A3 snapshot in
results/pos_ckpt.pt, same warm Adam (restored moments, same convention as
pos_sleep), same number of gradient tokens:

  S1 sleep            replay the stored spans (pos_sleep's S1, RE-RUN here
                       against the current snapshot for exact comparability
                       — the live run has advanced past pos_sleep_trainfar's
                       snapshot, so timestamps don't line up otherwise)
  D  dream             the model generates its OWN training stream: ancestral
                       sampling (temperature=1.0) from a FROZEN eval-mode copy
                       of the base snapshot (the generator), seeded by an
                       8-token spark drawn from a small C4-val buffer (<5% of
                       tokens, ignition only — the vast majority of every
                       chunk is self-generated). Sampled in blocks of 64
                       tokens (chunk length), generator state detach-carried
                       across blocks, no_grad throughout generation. The
                       TRAINING arm (a separate, trainable deepcopy) trains on
                       these dreamed tokens exactly like real data.
  S2 fresh             C4-train, 5M docs ahead (train-far, pos_sleep's S2
                       control) — the "fresh data per gradient token"
                       reference.
  D-shuf dream-shuffled the exact dream token stream, chunk-permuted
                       (seeded) — the structure falsifier: if dreaming only
                       helps via unigram statistics, D-shuf ≈ D; if the
                       model needs its own COHERENT continuations, D-shuf
                       loses to D.

Design choice, stated once and held to: the generator is a FROZEN copy of the
snapshot, not the live-training model itself. Sampling from the model you are
simultaneously updating would let training drift feed back into the sampling
distribution mid-run (the dream chases a moving target, and any gain could be
an artifact of that coupling rather than of dreaming per se). Freezing the
generator isolates the question this run asks: can a FIXED snapshot's own
sampled continuations serve as useful training data. The drift variant (dream
from the live-updating model) is explicitly OUT OF SCOPE here.

Measurement: WT-2 heldout + a fixed C4-val slice, evaluated before/after each
arm (pos_sleep's idiom, including its skip-the-eval-slice offset guard). Also
recorded: the generator's own mean NLL on the dreamed tokens (how "surprised"
the frozen generator is by its own samples — by construction, ancestral
sampling makes this LOW, dreams are self-consistent almost by definition) set
beside the mean NLL of the stored spans under the SAME frozen snapshot (which
is HIGH — that's why they were stored as surprises). That contrast is the
information-theoretic core of the experiment: dreams are drawn from the
model's own belief and are therefore unsurprising to it, while stored spans
are exactly the tokens the model found hardest. Any dream-arm win has to come
from somewhere other than "hard tokens", by construction.

This file is READ-ONLY with respect to the live run's outputs (pos_ckpt.pt,
pos_index.jsonl, pos_status.json, ...): it only ever opens them for reading.
All outputs go to results/pos_dream*.json.

Usage:
  python src/pos_dream.py                      # full run (chunks=200) against the live ckpt/index
  python src/pos_dream.py --smoke              # small chunks=40 run
  python src/pos_dream.py --self-test           # fully offline synthetic self-test
"""
import os
import sys
import json
import copy
import time
import argparse
import tempfile
import shutil

sys.path.insert(0, "reference")
sys.path.insert(0, "src")

import numpy as np
import torch
import torch.nn.functional as F

torch.backends.mps.is_available = lambda: False          # force CPU (same as the run)
torch.set_num_threads(1)
try:
    os.nice(19)
except (AttributeError, PermissionError, OSError):
    pass                                                   # best-effort niceness (POSIX only)

from pos_sleep import (
    SpanStream, ChunkFeeder, C4ValStream, train_arm, load_snapshot, load_spans,
)
from length_extrap_v2 import load_wikitext2, build_vocab, tokenize
from pos_run import build_eval_set, heldout


# ───────────────────────────────────────────────────────────────────────────
#  DreamStream — the model generates its own training stream.
# ───────────────────────────────────────────────────────────────────────────
class DreamStream:
    """Ancestral-samples a flat token stream from a FROZEN generator model
    (eval mode, no_grad throughout). Ignited by an 8-token seed drawn from a
    small C4-val buffer (spark only, <5% of any produced block once the
    stream runs past its first ~160 tokens); thereafter purely
    self-generated, autoregressive, temperature=1.0. Generation proceeds in
    blocks of `gen_block` tokens (default 64, the chunk length) with the
    generator's own recurrent state detach-carried across blocks — the
    generator dreams one continuous, coherent sequence, not independent
    64-token fragments. Records every sampled token's NLL under the SAME
    frozen generator (self-consistency: how "surprised" is the generator by
    its own samples), and total wall-clock spent sampling, for a tok/s
    figure (the cost driver of this experiment)."""

    def __init__(self, generator, seed_tokens, seed, gen_block=64, temperature=1.0):
        assert len(seed_tokens) > 0, "DreamStream needs a non-empty ignition seed"
        self.generator = generator
        self.generator.eval()
        self.rng = np.random.default_rng(seed)
        self.gen_block = gen_block
        self.temperature = temperature
        self.pending = []
        self.state = None
        self.total_nll_sum = 0.0
        self.total_nll_n = 0
        self.gen_wall_s = 0.0
        self.ignition_tokens = 0
        self.total_tokens = 0
        self._ignite(seed_tokens)

    @torch.no_grad()
    def _ignite(self, seed_tokens):
        """Feed the seed through the generator to prime its state, WITHOUT
        counting the seed itself as a dreamed token (it's a spark, not a
        sample) — but its tokens ARE emitted into `pending` so the stream is
        continuous from token 1. NLL is not scored for ignition tokens since
        they are not self-generated."""
        t0 = time.time()
        x = torch.tensor([seed_tokens], dtype=torch.long)
        logits, self.state = self.generator(x, None)
        self.gen_wall_s += time.time() - t0
        self.pending.extend(int(t) for t in seed_tokens)
        self.ignition_tokens += len(seed_tokens)
        self.total_tokens += len(seed_tokens)
        # last-position logits become the seed for the first sampled token
        self._next_logits = logits[:, -1, :]

    @torch.no_grad()
    def _generate_block(self, n):
        """Autoregressively sample n tokens, one at a time (state carried),
        appending to `pending` and accumulating self-NLL."""
        t0 = time.time()
        out_tokens = []
        logits = self._next_logits
        for _ in range(n):
            probs = F.softmax(logits.squeeze(0) / self.temperature, dim=-1)
            probs_np = probs.numpy().astype(np.float64)
            probs_np /= probs_np.sum()                      # renormalize fp32->fp64 drift
            tok = int(self.rng.choice(len(probs_np), p=probs_np))
            nll = -float(np.log(max(probs_np[tok], 1e-12)))
            self.total_nll_sum += nll
            self.total_nll_n += 1
            out_tokens.append(tok)
            x = torch.tensor([[tok]], dtype=torch.long)
            logits, self.state = self.generator(x, self.state)
            logits = logits[:, -1, :]
        self._next_logits = logits
        self.state = [s.detach() for s in self.state]
        self.gen_wall_s += time.time() - t0
        self.pending.extend(out_tokens)
        self.total_tokens += len(out_tokens)

    def next_block(self, n):
        while len(self.pending) < n:
            self._generate_block(self.gen_block)
        out, self.pending = self.pending[:n], self.pending[n:]
        return out

    def stats(self):
        mean_nll = self.total_nll_sum / max(1, self.total_nll_n)
        tok_per_s = self.total_nll_n / self.gen_wall_s if self.gen_wall_s > 0 else 0.0
        return {
            "sampled_tokens": self.total_nll_n,
            "ignition_tokens": self.ignition_tokens,
            "ignition_frac": round(self.ignition_tokens / max(1, self.total_tokens), 6),
            "mean_self_nll": round(mean_nll, 6),
            "gen_wall_s": round(self.gen_wall_s, 3),
            "tok_per_s": round(tok_per_s, 2),
        }


class ChunkPermutedRelay:
    """Wraps an already-materialized flat token list and hands it out via
    next_block, chunk-permuting each drawn block (seeded) — the structure
    falsifier for the dream arm. Mirrors SpanStream(permute_chunks=True)'s
    contract, but over a fixed pre-generated stream instead of stored spans:
    dream tokens must be fully generated up front so D and D-shuf train on
    the IDENTICAL token multiset, differing only in intra-chunk order."""

    def __init__(self, tokens, seed):
        self.tokens = list(tokens)
        self.i = 0
        self.rng = np.random.default_rng(seed)

    def next_block(self, n):
        out = []
        while len(out) < n:
            if self.i >= len(self.tokens):
                self.i = 0                                   # loop if exhausted (shouldn't happen: pre-sized)
            take = min(n - len(out), len(self.tokens) - self.i)
            out.extend(self.tokens[self.i:self.i + take])
            self.i += take
        perm = self.rng.permutation(len(out))
        return [out[i] for i in perm]


def mean_nll_under(model, spans, K, bs=64):
    """Mean per-token NLL of `spans` (list[list[int]]) under `model`, using
    the SAME chunk convention as heldout: each span sliced into (K,K) x/y
    pairs via build_eval_set-style framing, but spans vary in length so we
    just chunk each span independently at chunk length K and average
    token-weighted. K is the run's cfg["chunk"] (64 live, injectable for the
    self-test's smaller synthetic chunk length)."""
    model.eval()
    xs, ys = [], []
    for span in spans:
        n = (len(span) - 1) // K
        for r in range(n):
            xs.append(span[r * K:(r + 1) * K])
            ys.append(span[r * K + 1:(r + 1) * K + 1])
    if not xs:
        return float("nan")
    X = torch.tensor(xs, dtype=torch.long)
    Y = torch.tensor(ys, dtype=torch.long)
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            lg, _ = model(X[i:i + bs], None)
            tot += float(F.cross_entropy(lg.reshape(-1, lg.size(-1)),
                                         Y[i:i + bs].reshape(-1), reduction="sum"))
            n += Y[i:i + bs].numel()
    return tot / max(1, n)


# ───────────────────────────────────────────────────────────────────────────
#  Main measurement — base heldout, four arms, sampling stats, verdict.
# ───────────────────────────────────────────────────────────────────────────
def run_dream(args, eval_fn, c4val_fn, seed_source_factory, fresh_source_factory,
              vocab_fn=None):
    """eval_fn(model) -> WT-2 heldout loss.
    c4val_fn(model) -> loss on the fixed C4-val slice.
    seed_source_factory(stoi, unk) -> token source to draw the 8-token
      ignition spark from (a SEPARATE small C4-val buffer, never overlapping
      the fixed c4val eval slice — same offset-protection idiom as pos_sleep).
    fresh_source_factory(stoi, unk, seed) -> S2 control source (train-far).
    vocab_fn() -> (vocab, stoi, unk, mask); None = pos_sleep's real WT-2 vocab.
    """
    if vocab_fn is None:
        from pos_sleep import _real_vocab
        vocab_fn = _real_vocab

    ck, cfg, base_model, opt_sd, stoi, unk, mask, V = load_snapshot(args.ckpt, vocab_fn)

    spans = load_spans(args.index)
    n_spans = len(spans)
    if n_spans < args.min_spans:
        print(f"[pos_dream] index too young, {n_spans} spans — rerun later "
              f"(need >= {args.min_spans})", flush=True)
        sys.exit(3)

    B, K = cfg["batch"], cfg["chunk"]
    grad_tokens_per_arm = args.chunks * B * K
    lr = args.replay_lr if args.replay_lr > 0 else cfg["lr"]

    base_heldout = eval_fn(base_model)
    base_c4val = c4val_fn(base_model)
    print(f"[pos_dream] ckpt n_streamed={ck['n_streamed']:,} | spans={n_spans} | "
          f"chunks={args.chunks} grad_tokens/arm={grad_tokens_per_arm:,} | "
          f"base_heldout={base_heldout:.6f} base_c4val={base_c4val:.6f}", flush=True)

    out = {
        "ckpt_n_streamed": ck["n_streamed"],
        "n_spans": n_spans,
        "grad_tokens_per_arm": grad_tokens_per_arm,
        "base_heldout": round(base_heldout, 6),
        "c4val": {"base": round(base_c4val, 6)},
        "arms": {},
    }

    def run_one(tag, source):
        model = copy.deepcopy(base_model)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        if not args.cold_opt and opt_sd is not None:
            # WARM optimizer: restore A3's live Adam moments, same rationale
            # as pos_sleep — isolates the data-source effect from a cold-Adam
            # restart artifact.
            opt.load_state_dict(opt_sd)
            for g in opt.param_groups:
                g["lr"] = lr
        feeder = ChunkFeeder(source, B, K)
        train_arm(model, opt, feeder, args.chunks)
        hl = eval_fn(model)
        cv = c4val_fn(model)
        rec = {
            "final_heldout": round(hl, 6), "delta": round(base_heldout - hl, 6),
        }
        out["c4val"][tag] = {"final": round(cv, 6), "delta": round(base_c4val - cv, 6)}
        return rec

    # ── S1 sleep — replay stored spans (re-run here for exact comparability
    #    against the CURRENT snapshot; the live run has advanced past the
    #    pos_sleep_trainfar.json reference snapshot)
    sleep_src = SpanStream(spans, seed=args.seed, permute_chunks=False)
    out["arms"]["sleep"] = run_one("sleep", sleep_src)

    # ── generator setup: a FROZEN eval-mode copy of the base snapshot.
    #    Never trained on, never updated — see module docstring for why.
    generator = copy.deepcopy(base_model)
    generator.eval()
    for p in generator.parameters():
        p.requires_grad_(False)

    seed_src = seed_source_factory(stoi, unk)
    ignition_seed = seed_src.next_block(8)

    dream_stream = DreamStream(generator, ignition_seed, seed=args.seed + 2,
                                gen_block=K, temperature=1.0)

    # ── D dream — materialize exactly grad_tokens_per_arm dreamed tokens up
    #    front (so D and D-shuf train on the IDENTICAL multiset), THEN train.
    #    ChunkFeeder needs (n_chunks * (K+1) worst-case with its own buffering,
    #    but per-b buffering can draw slightly more than B*K per next_xy call
    #    in general; here it's exact because next_block(K+1) is called until
    #    each of the B buffers has >= K+1, draining exactly K per call after
    #    the first — over n_chunks calls each of B buffers nets K*n_chunks+1
    #    tokens net drawn. Oversize the pre-materialized pool slightly and
    #    hand out via ChunkPermutedRelay-style flat draw for both arms so
    #    "same multiset" is exact regardless of buffering edge effects.
    total_dream_tokens = grad_tokens_per_arm + B * (K + 1)   # small safety margin
    print(f"[pos_dream] generating {total_dream_tokens:,} dream tokens "
          f"(ignition={len(ignition_seed)} spark tokens)...", flush=True)
    dream_tokens = dream_stream.next_block(total_dream_tokens)
    dream_stats = dream_stream.stats()
    print(f"[pos_dream] dream generation done: {dream_stats}", flush=True)

    class FixedRelay:
        """Flat-list next_block, no permutation — D's un-shuffled source."""
        def __init__(self, tokens):
            self.tokens, self.i = list(tokens), 0

        def next_block(self, n):
            out = []
            while len(out) < n:
                if self.i >= len(self.tokens):
                    self.i = 0
                take = min(n - len(out), len(self.tokens) - self.i)
                out.extend(self.tokens[self.i:self.i + take])
                self.i += take
            return out

    out["arms"]["dream"] = run_one("dream", FixedRelay(dream_tokens))

    # ── S2 fresh — C4 train-far control (pos_sleep's S2 recipe)
    fresh_src = fresh_source_factory(stoi, unk, args.seed + 1)
    out["arms"]["fresh"] = run_one("fresh", fresh_src)

    # ── D-shuf dream-shuffled — IDENTICAL dream tokens, chunk-permuted
    dreamshuf_src = ChunkPermutedRelay(dream_tokens, seed=args.seed + 3)
    out["arms"]["dream_shuffled"] = run_one("dream_shuffled", dreamshuf_src)

    # ── information-theoretic core: mean NLL of dreamed tokens vs stored
    #    spans, BOTH scored under the same frozen generator/base snapshot.
    dream_chunks_for_nll = [dream_tokens[i:i + K + 1]
                            for i in range(0, len(dream_tokens) - K, K + 1)]
    nll_dream_under_gen = mean_nll_under(generator, dream_chunks_for_nll, K)
    nll_spans_under_gen = mean_nll_under(generator, spans, K)
    out["self_consistency"] = {
        "mean_nll_dream_tokens_under_generator": round(nll_dream_under_gen, 6),
        "mean_nll_stored_spans_under_generator": round(nll_spans_under_gen, 6),
        "gap": round(nll_spans_under_gen - nll_dream_under_gen, 6),
        "note": ("dreams are sampled FROM the generator's own distribution, so "
                 "their NLL under it is low by construction; stored spans were "
                 "selected BECAUSE they were high-NLL surprises. A positive gap "
                 "here is expected and is not itself evidence about which arm "
                 "wins on heldout — it's the mechanism explanation."),
    }
    out["dream_sampling"] = dream_stats

    d_sleep = out["arms"]["sleep"]["delta"]
    d_dream = out["arms"]["dream"]["delta"]
    d_fresh = out["arms"]["fresh"]["delta"]
    d_dshuf = out["arms"]["dream_shuffled"]["delta"]

    dream_vs_fresh = d_dream - d_fresh
    sleep_vs_dream = d_sleep - d_dream
    dream_vs_dshuf = d_dream - d_dshuf

    # P19 scoring: dream beats fresh by >= 0.03 AND loses to sleep -> CONFIRMED
    # dream >= sleep -> SENSATION (storage-free consolidation)
    # dream <= fresh -> FALSIFIED (self-sampling adds nothing over fresh data)
    if d_dream >= d_sleep:
        p19 = "SENSATION (dream >= sleep: storage-free consolidation)"
    elif dream_vs_fresh >= 0.03:
        p19 = "CONFIRMED (dream beats fresh by >=0.03, loses to sleep)"
    elif dream_vs_fresh > 0:
        p19 = "PARTIAL (dream beats fresh by <0.03)"
    else:
        p19 = "FALSIFIED (dream does not beat fresh)"

    out["p19_scoring"] = p19
    out["verdict"] = (
        f"sleep delta={d_sleep:+.6f} dream delta={d_dream:+.6f} fresh delta={d_fresh:+.6f} "
        f"dream_shuffled delta={d_dshuf:+.6f} | "
        f"dream-vs-fresh={dream_vs_fresh:+.6f} ({'dream ahead' if dream_vs_fresh > 0 else 'fresh ahead' if dream_vs_fresh < 0 else 'tied'}) | "
        f"sleep-vs-dream={sleep_vs_dream:+.6f} ({'sleep ahead' if sleep_vs_dream > 0 else 'dream ahead' if sleep_vs_dream < 0 else 'tied'}) | "
        f"dream-vs-dream_shuffled={dream_vs_dshuf:+.6f} ({'structure matters' if dream_vs_dshuf > 0 else 'unigrams suffice' if dream_vs_dshuf < 0 else 'tied'}) | "
        f"P19={p19}"
    )
    print(f"[pos_dream] {out['verdict']}", flush=True)

    d = os.path.dirname(args.out) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_")
    with os.fdopen(fd, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, args.out)
    print(f"[pos_dream] -> {args.out}", flush=True)
    return out


# ───────────────────────────────────────────────────────────────────────────
#  CLI
# ───────────────────────────────────────────────────────────────────────────
def build_argparser():
    ap = argparse.ArgumentParser(description="POS DREAM: self-generated stream vs stored-span replay vs fresh data")
    ap.add_argument("--ckpt", default="results/pos_ckpt.pt")
    ap.add_argument("--index", default="results/pos_index.jsonl")
    ap.add_argument("--chunks", type=int, default=200)
    ap.add_argument("--min-spans", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true",
                    help="chunks=40, min-spans=20, out=results/pos_dream_smoke.json")
    ap.add_argument("--out", default="results/pos_dream.json")
    ap.add_argument("--cold-opt", action="store_true",
                    help="fresh Adam instead of restoring A3's moments")
    ap.add_argument("--replay-lr", type=float, default=0.0,
                    help="override lr for all arms (0 = the run's config lr)")
    ap.add_argument("--self-test", action="store_true",
                    help="fully offline synthetic self-test (no ckpt/index/HF/network access)")
    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    if args.self_test:
        run_self_test()
        return

    if args.smoke:
        args.chunks = 40
        args.min_spans = 20
        if args.out == "results/pos_dream.json":
            args.out = "results/pos_dream_smoke.json"

    def fresh_source_factory(stoi, unk, seed):
        # same train-far recipe as pos_sleep's --fresh-split train-far: 5M
        # docs ahead of anything the live 40h run can reach, same domain.
        return C4ValStream(stoi, unk, split="train", skip_docs=5_000_000)

    def seed_source_factory(stoi, unk):
        # small independent C4-val buffer for the ignition spark ONLY (8
        # tokens). Uses a large skip so it can never overlap the fixed
        # 20k-token c4val EVAL slice (which reads from the start of
        # validation) — same offset-protection idiom as pos_sleep's
        # 40k-token skip on the fresh-validation source.
        return C4ValStream(stoi, unk, split="validation", skip_docs=1000)

    def eval_fn(model):
        train_text, val_text = load_wikitext2()
        _, stoi, unk, mask = build_vocab(train_text)
        val_ids = tokenize(val_text, stoi, unk)
        cfg = torch.load(args.ckpt, weights_only=False)["config"]
        evX, evY = build_eval_set(val_ids, cfg["eval_tokens"], cfg["chunk"])
        return heldout(model, evX, evY)

    def c4val_eval_fn(model):
        # fixed 20k-token C4-val slice, same for every arm before/after —
        # identical convention to pos_sleep's c4val_eval_fn.
        cache = c4val_eval_fn._cache
        if cache is None:
            train_text, _ = load_wikitext2()
            _, stoi, unk, mask = build_vocab(train_text)
            src = C4ValStream(stoi, unk)
            ids = src.next_block(20_000)
            cfg = torch.load(args.ckpt, weights_only=False)["config"]
            K = cfg["chunk"]
            X, Y = build_eval_set(ids, len(ids) - 1, K)
            cache = c4val_eval_fn._cache = (X, Y)
        X, Y = cache
        return heldout(model, X, Y)
    c4val_eval_fn._cache = None

    run_dream(args, eval_fn, c4val_eval_fn, seed_source_factory, fresh_source_factory)


# ───────────────────────────────────────────────────────────────────────────
#  Self-test — fully offline, writes only into a tempfile.mkdtemp() dir.
# ───────────────────────────────────────────────────────────────────────────
def run_self_test():
    tmpdir = tempfile.mkdtemp()
    try:
        torch.manual_seed(0)
        from streaming_train import StreamingNoPELM
        V, MASK, D = 200, 199, 32
        cfg = {"d_model": D, "batch": 4, "chunk": 16, "lr": 3e-3, "seed": 42, "eval_tokens": 2000}

        model = StreamingNoPELM(V, MASK, d_model=D, n_layers=2, n_heads=4,
                                d_head=D // 4, seq_len=32, dropout=0.0, causal=True)
        ck = {"n_streamed": 6_000_000, "config": cfg,
              "arms": {"A3": {"model": model.state_dict()}}}
        ckpt_path = os.path.join(tmpdir, "pos_ckpt.pt")
        torch.save(ck, ckpt_path)

        rng = np.random.default_rng(1)
        index_path = os.path.join(tmpdir, "pos_index.jsonl")
        with open(index_path, "w") as f:
            for i in range(30):
                span_len = int(rng.integers(40, 66))
                span = [int(t) for t in rng.integers(0, V, size=span_len)]
                f.write(json.dumps({"key": [0, 0, 0, 0], "n": i, "w": float(i),
                                    "row": 0, "pos": 0, "nll": 8.0, "span": span}) + "\n")

        eval_rng = np.random.default_rng(2)
        eval_ids = [int(t) for t in eval_rng.integers(0, V, size=cfg["eval_tokens"] + 1)]
        evX, evY = build_eval_set(eval_ids, cfg["eval_tokens"], cfg["chunk"])

        def eval_fn(m):
            return heldout(m, evX, evY)

        c4_rng = np.random.default_rng(3)
        c4_ids = [int(t) for t in c4_rng.integers(0, V, size=2000 + 1)]
        c4X, c4Y = build_eval_set(c4_ids, 2000, cfg["chunk"])

        def c4val_fn(m):
            return heldout(m, c4X, c4Y)

        class SyntheticSource:
            def __init__(self, seed):
                self.rng = np.random.default_rng(seed)

            def next_block(self, n):
                return [int(t) for t in self.rng.integers(0, V, size=n)]

        def fresh_source_factory(stoi, unk, seed):
            return SyntheticSource(seed)

        def seed_source_factory(stoi, unk):
            return SyntheticSource(999)

        synth_vocab = (list(range(V)), {i: i for i in range(V)}, V - 1, MASK)

        def vocab_fn():
            return synth_vocab

        args = argparse.Namespace(ckpt=ckpt_path, index=index_path, chunks=6, min_spans=20,
                                  seed=42, cold_opt=False, replay_lr=0.0,
                                  out=os.path.join(tmpdir, "pos_dream_selftest.json"))

        out = run_dream(args, eval_fn, c4val_fn, seed_source_factory, fresh_source_factory,
                        vocab_fn=vocab_fn)

        assert set(out["arms"].keys()) == {"sleep", "dream", "fresh", "dream_shuffled"}, \
            f"expected 4 arms, got {list(out['arms'].keys())}"
        gt = out["grad_tokens_per_arm"]
        assert gt == args.chunks * cfg["batch"] * cfg["chunk"], f"grad_tokens_per_arm mismatch: {gt}"
        for tag, rec in out["arms"].items():
            assert "final_heldout" in rec and "delta" in rec, f"arm {tag} incomplete: {rec}"
            assert abs(rec["delta"] - (out["base_heldout"] - rec["final_heldout"])) < 1e-6, \
                f"delta mismatch for {tag}: {rec}"
            assert tag in out["c4val"], f"c4val missing for arm {tag}"
        assert "self_consistency" in out and "gap" in out["self_consistency"]
        assert "dream_sampling" in out and out["dream_sampling"]["sampled_tokens"] > 0
        assert out["p19_scoring"] in {
            "SENSATION (dream >= sleep: storage-free consolidation)",
            "CONFIRMED (dream beats fresh by >=0.03, loses to sleep)",
            "PARTIAL (dream beats fresh by <0.03)",
            "FALSIFIED (dream does not beat fresh)",
        }
        assert os.path.exists(args.out), "dream JSON not written"
        with open(args.out) as f:
            written = json.load(f)
        assert written == out, "written JSON does not match returned dict"
        assert isinstance(out["verdict"], str) and len(out["verdict"]) > 0

        # base_heldout reproducibility check (same idiom as pos_sleep's self-test)
        _, _, base_model2, *_ = load_snapshot(ckpt_path, vocab_fn)
        base_hl2 = eval_fn(base_model2)
        assert abs(base_hl2 - out["base_heldout"]) < 1e-6, \
            f"base_heldout not reproducible: {base_hl2} vs {out['base_heldout']}"

        print("SELF-TEST PASS")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
