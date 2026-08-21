#!/usr/bin/env python3 -u
"""
POS SLEEP CYCLES — does the closed wake/sleep loop compound?
==============================================================================
pos_sleep.py measured a ONE-SHOT consolidation dividend: replaying the run's
own stored surprise spans beat fresh same-domain data by +0.078 heldout
(paired, results/pos_sleep_trainfar.json). That is a single sleep episode
bolted onto a frozen snapshot. The dream question this file asks: does the
LOOP itself pay off — wake and collect your own surprises, sleep and replay
them, wake again on fresh data, sleep on the (now larger) memory pool again —
and does a closed loop like that beat pure continuous wake training at the
SAME total gradient budget?

Two arms, both forked (deepcopy) from the IDENTICAL A3 snapshot in
results/pos_ckpt.pt (A3 weights + A3's warm Adam moments, exactly as
pos_sleep.py does it), same total budget of N_total chunks (default 600):

  W  wake-only control    N_total chunks straight through on fresh C4
                          (train split, 5M docs ahead of the live run's
                          reach — pos_sleep's "train-far" recipe). Own
                          stream cursor, continuous, never interrupted.

  C  wake/sleep cycles    3 cycles of [wake: N_wake chunks on the SAME
                          fresh C4 stream positions W sees + collect this
                          arm's OWN surprise spans in-memory while waking]
                          then [sleep: replay the spans collected SO FAR
                          (SpanStream mechanics, spans accumulate across
                          cycles), budget CAPPED so replay does not exceed
                          --max-replay-epochs over the pool -- see "sleep
                          budget coupling" below] -- any sleep budget left
                          unused because the pool is still small is spent
                          on MORE wake instead, so every cycle still spends
                          exactly (N_wake_nominal + N_sleep_max) chunks and
                          3 * that = N_total, unchanged (same budget as W).

Honesty note (per the P15 brief): the spans C sleeps on come from data W ALSO
saw (both arms read the identical wake-stream positions — see "shared wake
stream" below). The only difference between the two arms is a REALLOCATION
of up-to-(N_sleep_max)-per-cycle budget chunks from "train once on new data"
to "replay your own past picks again". That reallocation, not access to
extra data, is exactly what this experiment isolates. If C wins, the win is
from WHAT the organism chose to revisit, not from seeing more.

Sleep budget coupling (v2 fix -- see "why v1 failed" below): a fixed
N_sleep=50 chunks of replay over a pool of ~8-24 spans (v1's smoke pool
size) is ~20 replay epochs over a few thousand tokens -- pure overfitting,
not consolidation. pos_sleep.py's one-shot result replayed 20,000 spans in
under one epoch and WON; the mechanism this file is testing needs the same
regime. So the sleep budget is now COUPLED to how much material is actually
in the pool:

    span_tokens          = sum(len(s) for s in collected_spans)
    n_sleep_chunks_used   = min(n_sleep_max,
                                ceil(max_replay_epochs * span_tokens / (B*K)))

--max-replay-epochs (default 2.0) caps how many times the pool may be
replayed in one sleep segment. A small pool sleeps briefly (few chunks,
<=2 epochs over what little there is); a large pool can use the full
n_sleep_max budget. The chunks NOT spent sleeping do not vanish -- they are
added to THIS cycle's wake segment (extra fresh-data training, spike
collection stays on), so every cycle still consumes exactly
(n_wake + n_sleep_max) chunks in total and the W/C total-budget equality
(3 * (n_wake+n_sleep_max) = chunks_total, identical for both arms) holds
exactly as before. Concretely: wake first for n_wake chunks (this cycle's
spike harvest happens here), THEN compute the capped sleep budget from the
resulting pool, THEN wake for the leftover (n_sleep_max - n_sleep_used)
chunks (spike collection stays on -- it's still wake time, on fresh data),
THEN sleep for n_sleep_used chunks. The wake segment is therefore reported
as a single merged segment per cycle (n_wake_nominal + leftover chunks);
the curve records ONE heldout point after all of that cycle's wake chunks
(nominal + leftover) and ONE after that cycle's (possibly shorter) sleep.

Why v1 failed (kept for the record): the first smoke (spike-quantile=0.90,
fixed n_sleep=10) collected only 8/10/6 spans/cycle (24 total, ~40 tokens
avg span). Every 10-chunk sleep phase replayed that tiny pool ~20 times --
all three dividends were negative and DEEPENING (-0.054, -0.092, -0.120),
i.e. textbook overfitting on a handful of memorized spans, not the
consolidation regime pos_sleep.py measured. v2 lowers the spike quantile to
0.75 (--spike-quantile, more material collected per wake segment) and adds
the epoch cap above so short sleep segments are only ever assigned to pools
too small to productively use more.

Shared wake stream: implemented as two independently instantiated
C4ValStream(train-far) objects created with identical (stoi, unk, skip_docs)
at the same call site — since C4 streaming with a fixed skip_docs replays
the exact same document sequence deterministically, instantiating the same
source twice yields byte-identical token order. W then consumes N_total
tokens straight through; C consumes (n_wake + leftover) tokens per cycle at
the SAME cumulative stream position (interleaved with sleep chunks that draw
from the in-memory span pool instead, never advancing the shared C4
cursor). Because neither arm's C4 cursor is perturbed by the other arm or by
C's sleep phases, at any matching "total chunks consumed" checkpoint both
arms have trained on the identical fresh-data token windows (C additionally
spends some of its wake-time chunks on sleep-derived replay, which W never
sees -- that reallocation is the entire experimental question).

Spike collection (arm C, in-flight during wake, no extra forward pass): the
per-token NLL already computed by the wake TRAINING forward (cross_entropy
with reduction="none", same tensor shape as pos_run.py's step_gated) is
reused directly -- no separate no_grad pre-pass. A chunk-mean surprise value
is pushed onto a rolling deque (maxlen=200); once >=2 values are in the
window, chunks whose mean surprise clears the CURRENT rolling quantile
threshold (--spike-quantile, default 0.75; computed from window contents
BEFORE this chunk, matching pos_run.py's "threshold uses previous chunks
only" convention) contribute spans: every token position in that chunk with
per-token NLL >= spike_min_nll becomes a span center, extended +/-32 tokens
(span_half, matching pos_run/pos_sleep), capped at 2 spans/chunk
(highest-NLL positions first), appended to an in-memory list that persists
across all 3 cycles. The rolling window itself is NOT reset between the
nominal-wake and leftover-wake sub-segments of a cycle (both are the same
continuous wake activity), but IS reset fresh at the start of each new
cycle's wake (each cycle re-learns its own recent-surprise baseline rather
than carrying a stale one from a very different point in the sleep-replay
schedule).

Measurement: WT-2 heldout (build_eval_set/heldout, pos_run.py machinery,
eval_tokens from the checkpoint's own config) is taken before any training,
after each of C's wake/sleep segment pairs and at W's matching cumulative
chunk-counts, and at the end for both arms. Each sleep phase's isolated
heldout delta (measured immediately before -> immediately after that sleep
segment) is recorded as that cycle's "consolidation dividend" -- note this
delta is computed relative to the post-wake (post-leftover) heldout, i.e.
sleep's OWN marginal contribution, not conflated with that cycle's wake
gains.

Output: results/pos_sleep_cycles.json (or --smoke -> pos_sleep_cycles_smoke.json)
  {ckpt_n_streamed, budget: {chunks_total, n_wake, n_sleep_max, n_cycles,
   max_replay_epochs, spike_quantile},
   base_heldout,
   arms: {
     W: {curve: [[chunks_consumed, heldout], ...], final},
     C: {curve: [[chunks_consumed, segment_tag, heldout], ...], final,
         sleep_dividends: [d1, d2, d3], n_spans_collected_per_cycle: [...],
         sleep_detail: [{n_spans, span_tokens, replay_epochs_effective,
                         n_sleep_chunks_used}, ...]}
   },
   verdict}

P15 verdict criteria: C beats W by more than noise (C_final < W_final - 0.02
in loss, i.e. C's heldout LOWER by >0.02) AND the per-cycle dividends d1/d2/d3
do not collapse to ~0 (the loop keeps paying, not just cycle 1).

This file is READ-ONLY with respect to the live run's outputs (pos_ckpt.pt,
pos_index.jsonl, pos_snapshots/*): it only ever opens them for reading. All
outputs go to results/pos_sleep_cycles*.

Usage:
  python src/pos_sleep_cycles.py                 # full: chunks-total=600 (3x(150+50))
  python src/pos_sleep_cycles.py --smoke          # chunks-total=120 (3x(30+10))
"""
import os
import sys
import json
import math
import copy
import argparse
import tempfile
from collections import deque

sys.path.insert(0, "reference")
sys.path.insert(0, "src")

try:
    os.nice(19)
except OSError:
    pass          # already at max niceness (launched under `nice -n 19`)

import numpy as np
import torch
import torch.nn.functional as F

torch.backends.mps.is_available = lambda: False          # force CPU (same as the live run)
torch.set_num_threads(1)

from streaming_train import StreamingNoPELM
from length_extrap_v2 import load_wikitext2, build_vocab, tokenize
from pos_run import build_eval_set, heldout                  # safe: pos_run's main() only runs under __main__
from pos_sleep import (SpanStream, ChunkFeeder, C4ValStream, train_arm,
                        load_snapshot, load_spans, _real_vocab)


# ───────────────────────────────────────────────────────────────────────────
#  Wake step with in-flight spike collection (arm C only). Mirrors
#  pos_sleep.train_arm's recipe (full-gradient backward every chunk,
#  detach-carried state, clip=5.0) but additionally returns the chunk-mean
#  surprise and per-token NLL so the caller can harvest spans without a
#  second forward pass.
# ───────────────────────────────────────────────────────────────────────────
def wake_step(model, opt, feeder, states, clip=5.0):
    x, y = feeder.next_xy()
    logits, st = model(x, states)
    nll_flat = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                               reduction="none")
    nll = nll_flat.view(x.shape)                              # (B, K), same graph as loss
    loss = nll_flat.mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step()
    new_states = [s.detach() for s in st]
    grad_tokens = x.numel()
    return new_states, grad_tokens, float(loss), x.detach(), nll.detach()


def harvest_spans(x, nll, spike_min_nll, span_half, max_per_chunk=2):
    """Extract up to max_per_chunk spans (+/- span_half tokens) centered on the
    highest-NLL token positions in this chunk (x is (B, K); spans are cut from
    the row the spike occurred in, clamped to that row's bounds -- rows are
    independent stream lanes in ChunkFeeder, so spans never cross a lane).
    Returns a list of token-id lists."""
    spans = []
    B, K = x.shape
    flat_nll = nll.reshape(-1)
    order = torch.argsort(flat_nll, descending=True)
    taken_rows = set()
    for idx in order.tolist():
        if len(spans) >= max_per_chunk:
            break
        val = float(flat_nll[idx])
        if val < spike_min_nll:
            break                                            # sorted descending: rest are lower
        b, k = idx // K, idx % K
        if b in taken_rows:
            continue                                          # <=1 span/row keeps spans disjoint
        lo, hi = max(0, k - span_half), min(K, k + span_half + 1)
        span = x[b, lo:hi].tolist()
        if len(span) > 1:
            spans.append(span)
            taken_rows.add(b)
    return spans


def run_wake_segment(model, opt, feeder, n_chunks, states, collect_spans, args, window=None):
    """Runs n_chunks of wake training. If collect_spans, maintains a rolling
    quantile-of-last-200 threshold (window contents seeded by the caller --
    pass the SAME deque across the nominal+leftover sub-segments of one
    cycle to keep gating continuous within a cycle; pass a fresh deque at
    the start of each new cycle) and harvests spans from chunks whose mean
    surprise clears args.spike_quantile (threshold computed from window
    contents BEFORE this chunk, pos_run.py convention)."""
    if window is None:
        window = deque(maxlen=200)
    new_spans = []
    grad_tokens = 0
    for _ in range(n_chunks):
        states, gt, loss, x, nll = wake_step(model, opt, feeder, states)
        grad_tokens += gt
        if collect_spans:
            chunk_mean = float(nll.mean())
            if len(window) >= 2:
                thresh = float(np.quantile(np.fromiter(window, dtype=np.float64), args.spike_quantile))
                if chunk_mean > thresh:
                    new_spans.extend(harvest_spans(x, nll, args.spike_min_nll, args.span_half))
            window.append(chunk_mean)
    return states, grad_tokens, new_spans, window


def sleep_budget(collected_spans, n_sleep_max, max_replay_epochs, B, K):
    """Couples the sleep segment's chunk count to how much material is
    actually in the pool: n_sleep_chunks_used = min(n_sleep_max,
    ceil(max_replay_epochs * span_tokens / (B*K))). Returns
    (n_sleep_chunks_used, span_tokens, replay_epochs_effective)."""
    span_tokens = sum(len(s) for s in collected_spans)
    if span_tokens == 0:
        return 0, 0, 0.0
    tokens_per_chunk = B * K
    n_capped_by_epochs = math.ceil(max_replay_epochs * span_tokens / tokens_per_chunk)
    n_used = max(1, min(n_sleep_max, n_capped_by_epochs))
    replay_epochs_effective = (n_used * tokens_per_chunk) / span_tokens
    return n_used, span_tokens, replay_epochs_effective


# ───────────────────────────────────────────────────────────────────────────
#  Main measurement
# ───────────────────────────────────────────────────────────────────────────
def run_cycles(args, eval_fn, vocab_fn=_real_vocab):
    ck, cfg, base_model, opt_sd, stoi, unk, mask, V = load_snapshot(args.ckpt, vocab_fn)
    B, K = cfg["batch"], cfg["chunk"]
    lr = cfg["lr"]

    n_cycles = 3
    assert args.n_wake * n_cycles + args.n_sleep * n_cycles == args.chunks_total, (
        f"budget mismatch: 3*({args.n_wake}+{args.n_sleep}) != {args.chunks_total}")

    base_heldout = eval_fn(base_model)
    print(f"[cycles] ckpt n_streamed={ck['n_streamed']:,} | budget={args.chunks_total} "
          f"(3x({args.n_wake}+{args.n_sleep})) | spike_quantile={args.spike_quantile} "
          f"max_replay_epochs={args.max_replay_epochs} | base_heldout={base_heldout:.6f}", flush=True)

    out = {
        "ckpt_n_streamed": ck["n_streamed"],
        "budget": {"chunks_total": args.chunks_total, "n_wake": args.n_wake,
                   "n_sleep_max": args.n_sleep, "n_cycles": n_cycles,
                   "max_replay_epochs": args.max_replay_epochs,
                   "spike_quantile": args.spike_quantile},
        "base_heldout": round(base_heldout, 6),
        "arms": {},
    }

    def fresh_model_opt():
        m = copy.deepcopy(base_model)
        o = torch.optim.Adam(m.parameters(), lr=lr)
        if opt_sd is not None:
            o.load_state_dict(opt_sd)                          # warm Adam moments (pos_sleep.py recipe)
            for g in o.param_groups:
                g["lr"] = lr
        return m, o

    # Shared wake stream: two independently instantiated C4ValStream(train-far)
    # objects at the IDENTICAL (stoi, unk, skip_docs) call site. C4 streaming
    # with a fixed skip_docs replays the same document sequence deterministically,
    # so instantiating it twice gives W and C byte-identical token order at
    # matching cumulative wake-chunk counts (see module docstring).
    def make_wake_source():
        return C4ValStream(stoi, unk, split="train", skip_docs=5_000_000)

    n_wake_nominal, n_sleep_max = args.n_wake, args.n_sleep
    cycle_budget = n_wake_nominal + n_sleep_max               # fixed total per cycle, both arms

    # ── Arm W: wake-only control, N_total chunks straight through ──────────
    w_model, w_opt = fresh_model_opt()
    w_source = make_wake_source()
    w_feeder = ChunkFeeder(w_source, B, K)
    w_states = None
    w_curve = [[0, round(base_heldout, 6)]]
    w_grad_tokens = 0
    w_chunks_done = 0
    w_checkpoints = [cycle_budget * (i + 1) for i in range(n_cycles)]
    for cp in w_checkpoints:
        n_this = cp - w_chunks_done
        w_states, gt, _, _ = run_wake_segment(w_model, w_opt, w_feeder, n_this, w_states,
                                              collect_spans=False, args=args)
        w_grad_tokens += gt
        w_chunks_done = cp
        hl = eval_fn(w_model)
        w_curve.append([w_chunks_done, round(hl, 6)])
        print(f"[cycles][W] chunks={w_chunks_done} heldout={hl:.6f}", flush=True)
    w_final = w_curve[-1][1]
    out["arms"]["W"] = {"curve": w_curve, "final": w_final,
                        "delta": round(base_heldout - w_final, 6),
                        "grad_tokens": w_grad_tokens}

    # ── Arm C: wake/sleep cycles, same total budget PER CYCLE (sleep budget
    #    coupled to pool size; unused sleep budget is spent on extra wake) ──
    c_model, c_opt = fresh_model_opt()
    c_wake_source = make_wake_source()                          # SAME skip_docs -> identical stream to W's
    c_wake_feeder = ChunkFeeder(c_wake_source, B, K)
    c_states = None
    c_curve = [[0, "base", round(base_heldout, 6)]]
    c_grad_tokens = 0
    c_chunks_done = 0
    collected_spans = []
    n_spans_per_cycle = []
    sleep_dividends = []
    sleep_detail = []

    for cyc in range(n_cycles):
        # nominal wake: n_wake_nominal chunks, spike collection on, fresh window this cycle
        c_states, gt, new_spans, window = run_wake_segment(
            c_model, c_opt, c_wake_feeder, n_wake_nominal, c_states,
            collect_spans=True, args=args)
        c_grad_tokens += gt
        c_chunks_done += n_wake_nominal
        collected_spans.extend(new_spans)

        # sleep budget coupled to the pool NOW (after this cycle's nominal wake)
        n_sleep_used, span_tokens, replay_epochs_eff = sleep_budget(
            collected_spans, n_sleep_max, args.max_replay_epochs, B, K)
        n_leftover_wake = n_sleep_max - n_sleep_used            # unused sleep budget -> more wake, same cycle

        # leftover wake: same cycle, spike collection stays on, window carries over
        # (still the same continuous wake activity as the nominal segment above)
        if n_leftover_wake > 0:
            c_states, gt_lo, more_spans, window = run_wake_segment(
                c_model, c_opt, c_wake_feeder, n_leftover_wake, c_states,
                collect_spans=True, args=args, window=window)
            c_grad_tokens += gt_lo
            c_chunks_done += n_leftover_wake
            collected_spans.extend(more_spans)
            new_spans = new_spans + more_spans

        n_spans_per_cycle.append(len(new_spans))
        hl_wake = eval_fn(c_model)
        c_curve.append([c_chunks_done, f"wake{cyc+1}", round(hl_wake, 6)])
        print(f"[cycles][C] cycle={cyc+1} wake done (nominal={n_wake_nominal}+leftover={n_leftover_wake}) "
              f"chunks={c_chunks_done} heldout={hl_wake:.6f} new_spans={len(new_spans)} "
              f"pool={len(collected_spans)}", flush=True)

        # sleep segment: replay ALL spans collected so far, budget = n_sleep_used
        if n_sleep_used == 0:
            print(f"[cycles][C] cycle={cyc+1}: empty pool, sleep segment skipped "
                  f"(0 chunks; all {n_sleep_max} sleep-budget chunks went to wake)", flush=True)
            hl_sleep = hl_wake
        else:
            sleep_src = SpanStream(collected_spans, seed=args.seed + cyc, permute_chunks=False)
            sleep_feeder = ChunkFeeder(sleep_src, B, K)
            c_states, gt2, _, _ = run_wake_segment(
                c_model, c_opt, sleep_feeder, n_sleep_used, c_states,
                collect_spans=False, args=args)
            c_grad_tokens += gt2
            hl_sleep = eval_fn(c_model)
        c_chunks_done += n_sleep_used
        c_curve.append([c_chunks_done, f"sleep{cyc+1}", round(hl_sleep, 6)])
        dividend = round(hl_wake - hl_sleep, 6)                 # loss DROP during sleep = this cycle's dividend
        sleep_dividends.append(dividend)
        sleep_detail.append({"n_spans": len(collected_spans), "span_tokens": span_tokens,
                             "replay_epochs_effective": round(replay_epochs_eff, 3),
                             "n_sleep_chunks_used": n_sleep_used,
                             "n_leftover_wake_chunks": n_leftover_wake})
        print(f"[cycles][C] cycle={cyc+1} sleep done chunks={c_chunks_done} heldout={hl_sleep:.6f} "
              f"dividend={dividend:+.6f} | pool_spans={len(collected_spans)} "
              f"pool_tokens={span_tokens} replay_epochs={replay_epochs_eff:.3f} "
              f"sleep_chunks_used={n_sleep_used}/{n_sleep_max}", flush=True)

        # cycle budget invariant: nominal wake + leftover wake + sleep == cycle_budget, always
        assert n_wake_nominal + n_leftover_wake + n_sleep_used == cycle_budget

    c_final = c_curve[-1][2]
    out["arms"]["C"] = {"curve": c_curve, "final": c_final,
                        "delta": round(base_heldout - c_final, 6),
                        "grad_tokens": c_grad_tokens,
                        "sleep_dividends": sleep_dividends,
                        "n_spans_collected_per_cycle": n_spans_per_cycle,
                        "n_spans_total": len(collected_spans),
                        "sleep_detail": sleep_detail}

    assert w_grad_tokens == c_grad_tokens, (
        f"budget mismatch: W spent {w_grad_tokens} grad tokens, C spent {c_grad_tokens}")

    gap = round(w_final - c_final, 6)                            # positive = C's loss is lower = C wins
    collapsing = (len(sleep_dividends) >= 2 and
                 abs(sleep_dividends[-1]) < 0.25 * (abs(sleep_dividends[0]) + 1e-9))
    p15_pass = gap > 0.02
    out["verdict"] = (
        f"W_final={w_final:.6f} C_final={c_final:.6f} | C-vs-W gap={gap:+.6f} "
        f"({'C ahead (closed loop wins)' if gap > 0 else 'W ahead (continuous wake wins)' if gap < 0 else 'tied'}) | "
        f"sleep dividends={sleep_dividends} "
        f"({'collapsing toward 0' if collapsing else 'sustained across cycles'}) | "
        f"P15: {'PASS' if p15_pass and not collapsing else 'FAIL'} "
        f"(gap>0.02: {p15_pass}, dividends sustained: {not collapsing})"
    )
    print(f"[cycles] {out['verdict']}", flush=True)

    d = os.path.dirname(args.out) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_")
    with os.fdopen(fd, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, args.out)
    print(f"[cycles] -> {args.out}", flush=True)
    return out


# ───────────────────────────────────────────────────────────────────────────
#  CLI
# ───────────────────────────────────────────────────────────────────────────
def build_argparser():
    ap = argparse.ArgumentParser(
        description="POS SLEEP CYCLES: closed wake/sleep loop vs continuous wake, same gradient budget")
    ap.add_argument("--ckpt", default="results/pos_ckpt.pt")
    ap.add_argument("--chunks-total", type=int, default=600)
    ap.add_argument("--n-wake", type=int, default=150)
    ap.add_argument("--n-sleep", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--spike-min-nll", type=float, default=7.0,
                    help="per-token NLL floor for a span center (matches the live run's pos_run.py default)")
    ap.add_argument("--span-half", type=int, default=32,
                    help="span extends +/- this many tokens around a spike (matches pos_run.py/pos_sleep.py)")
    ap.add_argument("--spike-quantile", type=float, default=0.75,
                    help="rolling-window quantile a chunk's mean surprise must clear to harvest spans "
                         "(v1 used a fixed 0.90 and starved the pool; 0.75 matches the live run's own q)")
    ap.add_argument("--max-replay-epochs", type=float, default=2.0,
                    help="caps sleep-segment length so replay never exceeds this many epochs over the "
                         "pool collected so far; unused sleep budget is spent on extra wake instead")
    ap.add_argument("--smoke", action="store_true",
                    help="chunks-total=120 (3x(30+10)), out=results/pos_sleep_cycles_smoke.json")
    ap.add_argument("--out", default="results/pos_sleep_cycles.json")
    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    if args.smoke:
        args.chunks_total = 120
        args.n_wake = 30
        args.n_sleep = 10
        if args.out == "results/pos_sleep_cycles.json":
            args.out = "results/pos_sleep_cycles_smoke.json"

    def eval_fn(model):
        train_text, val_text = load_wikitext2()
        _, stoi, unk, mask = build_vocab(train_text)
        val_ids = tokenize(val_text, stoi, unk)
        cfg = torch.load(args.ckpt, weights_only=False)["config"]
        evX, evY = build_eval_set(val_ids, cfg["eval_tokens"], cfg["chunk"])
        return heldout(model, evX, evY)

    run_cycles(args, eval_fn)


if __name__ == "__main__":
    main()
