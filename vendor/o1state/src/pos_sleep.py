#!/usr/bin/env python3 -u
"""
POS SLEEP — does the organism learn more from replaying its own surprises?
============================================================================
The POS run (pos_run.py) stores surprise spans (+/- span_half tokens around
surprise spikes) into results/pos_index.jsonl — the self-selected "memories"
of the A3 (surprise-gated) arm. This script asks: does gradient training on
those REPLAYED memories buy more heldout improvement PER GRADIENT TOKEN than
fresh stream data? That would be sleep consolidation as a measurable
mechanism — the POS thesis squared: the system doesn't just choose WHEN to
learn, it can draw a second dividend from WHAT it chose to remember.

Three arms, all forked (deepcopy) from the IDENTICAL A3 snapshot in
results/pos_ckpt.pt, same fresh Adam (lr from the run's config), same number
of gradient tokens:

  S1 sleep           replay the stored spans, shuffled order, concatenated
                      into a stream (re-shuffle on exhaustion = new replay
                      epoch; epochs are counted and logged)
  S2 fresh            control: same recipe, but the stream is C4 VALIDATION
                      (never seen by the run's training stream) — measures
                      "fresh data per gradient token"
  S3 shuffled-sleep   falsifier: S1's exact token stream, but each chunk is
                      token-permuted (seeded) — destroys sequence structure,
                      keeps unigram statistics. Separates "surprising tokens"
                      from "structured memory".

This file is READ-ONLY with respect to the live 40h run's outputs
(pos_ckpt.pt, pos_index.jsonl, pos_status.json, ...): it only ever opens them
for reading. All outputs go to results/pos_sleep_*.

Usage:
  python src/pos_sleep.py                      # full run against the live ckpt/index
  python src/pos_sleep.py --smoke              # small chunks=40 / min-spans=20 run
  python src/pos_sleep.py --self-test          # fully offline synthetic self-test
"""
import os
import sys
import json
import copy
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

from streaming_train import StreamingNoPELM
from length_extrap_v2 import load_wikitext2, build_vocab, tokenize
from pos_run import build_eval_set, heldout                # safe: pos_run's main() only runs under __main__


# ───────────────────────────────────────────────────────────────────────────
#  Span stream — replay the index's surprise spans as a flat token stream.
# ───────────────────────────────────────────────────────────────────────────
class SpanStream:
    """Concatenates spans (lists of token ids) in a seeded-shuffled order into a
    flat stream. Re-shuffles and starts a new "replay epoch" once exhausted.
    If `permute_chunks` is set, each drawn chunk of K+1 tokens is independently
    token-permuted (seeded) before being handed out — S3's structure-destroying
    falsifier control. Unigram statistics are preserved because the same token
    multiset is permuted, just the order within a chunk is randomized."""

    def __init__(self, spans, seed, permute_chunks=False):
        self.spans = list(spans)
        assert len(self.spans) > 0, "SpanStream needs at least one span"
        self.rng = np.random.default_rng(seed)
        self.permute_rng = np.random.default_rng(seed + 1)
        self.permute_chunks = permute_chunks
        self.epoch = 0
        self.pending = []
        self._reshuffle()

    def _reshuffle(self):
        order = self.rng.permutation(len(self.spans))
        self.pending.extend(int(t) for i in order for t in self.spans[i])
        self.epoch += 1

    def next_block(self, n):
        """Return exactly n tokens, refilling (and counting replay epochs) as needed."""
        out = []
        while len(out) < n:
            if not self.pending:
                self._reshuffle()
            take = min(n - len(out), len(self.pending))
            out.extend(self.pending[:take])
            del self.pending[:take]
        if self.permute_chunks:
            perm = self.permute_rng.permutation(len(out))
            out = [out[i] for i in perm]
        return out


# ───────────────────────────────────────────────────────────────────────────
#  Generic bounded-buffer chunk stream (B parallel buffers, K+1 tokens/chunk).
#  Shared by all three arms — only the underlying token source differs.
# ───────────────────────────────────────────────────────────────────────────
class ChunkFeeder:
    """Wraps a `next_block(n) -> list[int]` source into (x, y) chunk batches of
    shape (B, K), matching pos_run.py's C4Stream buffering convention."""

    def __init__(self, source, batch, chunk):
        self.source = source
        self.B, self.K = batch, chunk
        self.bufs = [[] for _ in range(batch)]

    def next_xy(self):
        B, K = self.B, self.K
        for b in range(B):
            while len(self.bufs[b]) < K + 1:
                self.bufs[b].extend(self.source.next_block(K + 1))
        x = torch.tensor([self.bufs[b][:K] for b in range(B)], dtype=torch.long)
        y = torch.tensor([self.bufs[b][1:K + 1] for b in range(B)], dtype=torch.long)
        for b in range(B):
            del self.bufs[b][:K]
        return x, y


class C4ValStream:
    """Token source over the C4 'validation' split — guaranteed never seen by the
    run's training stream (which reads split='train'). Same tokenize/block
    machinery as streaming_train.c4_block_stream, just split='validation'."""

    def __init__(self, stoi, unk, block=65536, split="validation", skip_docs=0):
        from datasets import load_dataset
        self.stoi, self.unk = stoi, unk
        self.split, self.skip_docs = split, skip_docs
        ds = load_dataset("allenai/c4", "en", split=split, streaming=True)
        if skip_docs:
            ds = ds.skip(skip_docs)
        self._it = iter(ds)
        self.pending = []
        self.block = block

    def _refill(self, target=None):
        # target > self.block is required when a caller asks next_block(n) with
        # n > block -- otherwise _refill returns at `block` tokens and
        # next_block busy-loops forever (found live: a 100k skip against the
        # 65536 default spun at 99% cpu indefinitely).
        if target is None:
            target = self.block
        while len(self.pending) < target:
            try:
                row = next(self._it)
            except StopIteration:
                # validation split is small enough that in principle it could be
                # exhausted; loop it (mirrors streaming_train's corpus-loop policy)
                from datasets import load_dataset
                ds = load_dataset("allenai/c4", "en", split=self.split, streaming=True)
                self._it = iter(ds)
                continue
            t = row.get("text", "") if isinstance(row, dict) else ""
            if t.strip():
                self.pending.extend(tokenize(t, self.stoi, self.unk))

    def next_block(self, n):
        while len(self.pending) < n:
            self._refill(target=n)
        out, self.pending = self.pending[:n], self.pending[n:]
        return out


# ───────────────────────────────────────────────────────────────────────────
#  Training loop shared by all three arms — identical recipe to pos_run's
#  step_a2 (full-gradient backward every chunk), just driven by a ChunkFeeder
#  instead of the live C4 stream.
# ───────────────────────────────────────────────────────────────────────────
def train_arm(model, opt, feeder, n_chunks, clip=5.0):
    """Runs n_chunks backward steps; detach-carries state across chunks (same
    truncated-BPTT convention as pos_run/streaming_train). Returns grad_tokens."""
    model.train()
    states = None
    grad_tokens = 0
    for _ in range(n_chunks):
        x, y = feeder.next_xy()
        logits, st = model(x, states)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        states = [s.detach() for s in st]
        grad_tokens += x.numel()
    return grad_tokens


# ───────────────────────────────────────────────────────────────────────────
#  Snapshot / index loading (read-only w.r.t. the live run's files)
# ───────────────────────────────────────────────────────────────────────────
def _real_vocab():
    """Default vocab source: WT-2 train text, exactly as pos_run.py builds it."""
    train_text, _ = load_wikitext2()
    return build_vocab(train_text)


def load_snapshot(ckpt_path, vocab_fn=_real_vocab):
    """vocab_fn() -> (vocab, stoi, unk, mask), injectable so --self-test can supply a
    synthetic vocab matching its fixture checkpoint instead of touching WT-2/HF."""
    ck = torch.load(ckpt_path, weights_only=False)          # read-only: never written back
    cfg = ck["config"]
    vocab, stoi, unk, mask = vocab_fn()
    V = len(vocab)
    model = StreamingNoPELM(V, mask, d_model=cfg["d_model"], n_layers=2, n_heads=4,
                            d_head=cfg["d_model"] // 4, seq_len=32, dropout=0.0, causal=True)
    model.load_state_dict(ck["arms"]["A3"]["model"])
    opt_sd = ck["arms"]["A3"].get("opt")
    return ck, cfg, model, opt_sd, stoi, unk, mask, V


def load_spans(index_path):
    if not os.path.exists(index_path):
        return []
    spans = []
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            span = rec.get("span")
            if span:
                spans.append(span)
    return spans


# ───────────────────────────────────────────────────────────────────────────
#  Main measurement — base heldout, three arms, verdict.
# ───────────────────────────────────────────────────────────────────────────
def run_sleep(args, fresh_source_factory, eval_fn, c4val_fn=None, vocab_fn=_real_vocab):
    """fresh_source_factory(stoi, unk, seed) -> token source for the S2 control.
    eval_fn(model) -> float heldout loss.
    c4val_fn(model) -> float loss on a fixed C4-val slice, or None to skip.
    vocab_fn() -> (vocab, stoi, unk, mask), forwarded to load_snapshot.
    Kept as injectable hooks so --self-test never touches HF/network."""
    ck, cfg, base_model, opt_sd, stoi, unk, mask, V = load_snapshot(args.ckpt, vocab_fn)

    spans = load_spans(args.index)
    n_spans = len(spans)
    if n_spans < args.min_spans:
        print(f"[pos_sleep] index too young, {n_spans} spans — rerun later "
              f"(need >= {args.min_spans})", flush=True)
        sys.exit(3)

    B, K = cfg["batch"], cfg["chunk"]
    grad_tokens_per_arm = args.chunks * B * K
    lr = args.replay_lr if args.replay_lr > 0 else cfg["lr"]

    base_heldout = eval_fn(base_model)
    base_c4val = c4val_fn(base_model) if c4val_fn else None
    print(f"[pos_sleep] ckpt n_streamed={ck['n_streamed']:,} | spans={n_spans} | "
          f"chunks={args.chunks} grad_tokens/arm={grad_tokens_per_arm:,} | "
          f"base_heldout={base_heldout:.6f}", flush=True)

    out = {
        "ckpt_n_streamed": ck["n_streamed"],
        "n_spans": n_spans,
        "grad_tokens_per_arm": grad_tokens_per_arm,
        "base_heldout": round(base_heldout, 6),
        "arms": {},
    }
    if base_c4val is not None:
        out["c4val"] = {"base": round(base_c4val, 6)}

    replay_epochs = {}

    def run_one(tag, source):
        model = copy.deepcopy(base_model)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        if not args.cold_opt and opt_sd is not None:
            # WARM optimizer: restore A3's live Adam moments from the checkpoint.
            # The smoke showed a cold Adam restart perturbs the converged model
            # (every arm regressed at small budgets) — warm-starting removes the
            # restart artifact so the arms differ ONLY in their data source.
            opt.load_state_dict(opt_sd)
            for g in opt.param_groups:
                g["lr"] = lr
        feeder = ChunkFeeder(source, B, K)
        train_arm(model, opt, feeder, args.chunks)
        hl = eval_fn(model)
        rec = {"final_heldout": round(hl, 6), "delta": round(base_heldout - hl, 6)}
        if c4val_fn:
            cv = c4val_fn(model)
            out.setdefault("c4val", {})[tag] = {
                "final": round(cv, 6), "delta": round(base_c4val - cv, 6)}
        return rec

    # S1: sleep — replay stored spans, shuffled order
    sleep_src = SpanStream(spans, seed=args.seed, permute_chunks=False)
    out["arms"]["sleep"] = run_one("sleep", sleep_src)
    replay_epochs["sleep"] = sleep_src.epoch

    # S2: fresh — control stream (C4 validation split, or the injected fixture source)
    fresh_src = fresh_source_factory(stoi, unk, args.seed + 1)
    out["arms"]["fresh"] = run_one("fresh", fresh_src)

    # S3: shuffled-sleep — same span tokens, token-permuted per chunk (falsifier)
    shuffled_src = SpanStream(spans, seed=args.seed, permute_chunks=True)
    out["arms"]["shuffled"] = run_one("shuffled", shuffled_src)
    replay_epochs["shuffled"] = shuffled_src.epoch

    out["replay_epochs"] = replay_epochs

    d_sleep = out["arms"]["sleep"]["delta"]
    d_fresh = out["arms"]["fresh"]["delta"]
    d_shuf = out["arms"]["shuffled"]["delta"]
    consolidation = d_sleep - d_fresh
    structure = d_sleep - d_shuf
    out["verdict"] = (
        f"sleep delta={d_sleep:+.6f} fresh delta={d_fresh:+.6f} shuffled delta={d_shuf:+.6f} | "
        f"sleep-vs-fresh={consolidation:+.6f} ({'sleep ahead' if consolidation > 0 else 'fresh ahead' if consolidation < 0 else 'tied'}) | "
        f"sleep-vs-shuffled={structure:+.6f} ({'structure matters' if structure > 0 else 'unigrams suffice' if structure < 0 else 'tied'})"
    )
    print(f"[pos_sleep] {out['verdict']}", flush=True)

    d = os.path.dirname(args.out) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_")
    with os.fdopen(fd, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, args.out)
    print(f"[pos_sleep] -> {args.out}", flush=True)
    return out


# ───────────────────────────────────────────────────────────────────────────
#  CLI
# ───────────────────────────────────────────────────────────────────────────
def build_argparser():
    ap = argparse.ArgumentParser(description="POS SLEEP: replayed-surprise vs fresh-data consolidation test")
    ap.add_argument("--ckpt", default="results/pos_ckpt.pt")
    ap.add_argument("--index", default="results/pos_index.jsonl")
    ap.add_argument("--chunks", type=int, default=200)
    ap.add_argument("--min-spans", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true",
                    help="chunks=40, min-spans=20, out=results/pos_sleep_smoke.json")
    ap.add_argument("--out", default="results/pos_sleep.json")
    ap.add_argument("--fresh-split", choices=["validation", "train-far"], default="validation",
                    help="S2 fresh-arm source: C4 validation, or C4 train skipped 5M docs ahead (same-domain control)")
    ap.add_argument("--cold-opt", action="store_true",
                    help="fresh Adam instead of restoring A3's moments (the smoke-measured restart artifact)")
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
        if args.out == "results/pos_sleep.json":
            args.out = "results/pos_sleep_smoke.json"

    def c4val_source_factory(stoi, unk, seed):
        if args.fresh_split == "train-far":
            # fresh tokens from the SAME split the spans came from (kills the
            # validation-domain confound), but 5M docs ahead of anything the
            # live run can reach in 40h — genuinely unseen, same distribution
            return C4ValStream(stoi, unk, split="train", skip_docs=5_000_000)
        src = C4ValStream(stoi, unk)
        # discard the first 40k tokens: the fixed c4val EVAL slice is the first 20k
        # tokens of the same split — without this skip, the fresh arm would train on
        # its own eval slice and inflate its c4val delta (WT-2 heldout is unaffected)
        src.next_block(40_000)
        return src

    def eval_fn(model):
        train_text, val_text = load_wikitext2()
        _, stoi, unk, mask = build_vocab(train_text)
        val_ids = tokenize(val_text, stoi, unk)
        cfg = torch.load(args.ckpt, weights_only=False)["config"]
        evX, evY = build_eval_set(val_ids, cfg["eval_tokens"], cfg["chunk"])
        return heldout(model, evX, evY)

    def c4val_eval_fn(model):
        # fixed 20k-token C4-val slice, same for every arm before/after
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

    run_sleep(args, c4val_source_factory, eval_fn, c4val_eval_fn)


# ───────────────────────────────────────────────────────────────────────────
#  Self-test — fully offline, writes only into a tempfile.mkdtemp() dir.
#  Builds a tiny fixture checkpoint (pos_ckpt.pt format) + fixture index.jsonl,
#  runs the full three-arm loop with a synthetic eval set and a synthetic S2
#  stream (no HF/network access anywhere), then asserts structural invariants.
# ───────────────────────────────────────────────────────────────────────────
def run_self_test():
    tmpdir = tempfile.mkdtemp()
    try:
        torch.manual_seed(0)
        V, MASK, D = 200, 199, 32
        cfg = {"d_model": D, "batch": 4, "chunk": 16, "lr": 3e-3, "seed": 42, "eval_tokens": 2000}

        model = StreamingNoPELM(V, MASK, d_model=D, n_layers=2, n_heads=4,
                                d_head=D // 4, seq_len=32, dropout=0.0, causal=True)
        ck = {"n_streamed": 6_000_000, "config": cfg,
              "arms": {"A3": {"model": model.state_dict()}}}
        ckpt_path = os.path.join(tmpdir, "pos_ckpt.pt")
        torch.save(ck, ckpt_path)

        # fixture index: 30 synthetic spans of 40-65 tokens
        rng = np.random.default_rng(1)
        index_path = os.path.join(tmpdir, "pos_index.jsonl")
        with open(index_path, "w") as f:
            for i in range(30):
                span_len = int(rng.integers(40, 66))
                span = [int(t) for t in rng.integers(0, V, size=span_len)]
                f.write(json.dumps({"key": [0, 0, 0, 0], "n": i, "w": float(i),
                                    "row": 0, "pos": 0, "nll": 8.0, "span": span}) + "\n")

        # synthetic eval set (random tokens, not WT-2)
        eval_rng = np.random.default_rng(2)
        eval_ids = [int(t) for t in eval_rng.integers(0, V, size=cfg["eval_tokens"] + 1)]
        evX, evY = build_eval_set(eval_ids, cfg["eval_tokens"], cfg["chunk"])

        def eval_fn(m):
            return heldout(m, evX, evY)

        # synthetic S2 "fresh" stream — a second independent synthetic token source,
        # standing in for C4-validation (no HF/network access in the self-test)
        class SyntheticFreshSource:
            def __init__(self, seed):
                self.rng = np.random.default_rng(seed)

            def next_block(self, n):
                return [int(t) for t in self.rng.integers(0, V, size=n)]

        def fresh_source_factory(stoi, unk, seed):
            return SyntheticFreshSource(seed)

        # synthetic vocab matching the fixture checkpoint's V=200/MASK=199 — stands in
        # for build_vocab(load_wikitext2()[0]) so the self-test never touches HF/network
        synth_vocab = (list(range(V)), {i: i for i in range(V)}, V - 1, MASK)

        def vocab_fn():
            return synth_vocab

        args = argparse.Namespace(ckpt=ckpt_path, index=index_path, chunks=8, min_spans=20,
                                  seed=42, cold_opt=False, replay_lr=0.0,
                                  out=os.path.join(tmpdir, "pos_sleep_selftest.json"))

        out = run_sleep(args, fresh_source_factory, eval_fn, c4val_fn=None, vocab_fn=vocab_fn)

        assert set(out["arms"].keys()) == {"sleep", "fresh", "shuffled"}, \
            f"expected 3 arms, got {list(out['arms'].keys())}"
        gt = out["grad_tokens_per_arm"]
        assert gt == args.chunks * cfg["batch"] * cfg["chunk"], f"grad_tokens_per_arm mismatch: {gt}"
        for tag, rec in out["arms"].items():
            assert "final_heldout" in rec and "delta" in rec, f"arm {tag} incomplete: {rec}"
            assert abs(rec["delta"] - (out["base_heldout"] - rec["final_heldout"])) < 1e-6, \
                f"delta mismatch for {tag}: {rec}"
        assert os.path.exists(args.out), "sleep JSON not written"
        with open(args.out) as f:
            written = json.load(f)
        assert written == out, "written JSON does not match returned dict"
        assert isinstance(out["verdict"], str) and len(out["verdict"]) > 0

        # base_heldout is identical for all arms as the starting point: verify by
        # re-running eval on a fresh copy of the base model and comparing.
        _, _, base_model2, *_ = load_snapshot(ckpt_path, vocab_fn)
        base_hl2 = eval_fn(base_model2)
        assert abs(base_hl2 - out["base_heldout"]) < 1e-6, \
            f"base_heldout not reproducible: {base_hl2} vs {out['base_heldout']}"

        print("SELF-TEST PASS")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
