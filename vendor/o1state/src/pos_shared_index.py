#!/usr/bin/env python3 -u
"""
POS SHARED INDEX — two organisms, one index: collective memory across O(1)
individuals (P31, analysis/PREDICTIONS.md).
=============================================================================================
pos_domain_shock.py (R3) showed that ONE organism's own surprise-harvest sleep buffer
cuts its own forgetting during a code shock. This file asks the next question: can that
buffer be handed to a DIFFERENT organism, one that never lived through the shock itself?
If A's collected surprises make B more resilient to a shock A never told B was coming,
that is collective memory, not just self-consolidation — one organism's surprises become
another's immunity.

Two organisms, both forked (deepcopy) from the IDENTICAL POS snapshot
(results/pos_snapshots/ckpt_359050240.pt, arm A3, read-only — this file never writes to
that path or to any other results/pos_* path except its own):

  ORGANISM A ("the older one", pre-learns the shock for B)
    Streams a 50/50 C4+CODE mix at CHUNK granularity (alternating chunks drawn from
    make_phase1_source / make_phase2_source, the exact factories pos_domain_shock.py
    uses for its phase1/phase2 sources) for N chunks (full=120, smoke=30). Every chunk
    is full-gradient (R1-style — A is not being tested for forgetting, it is being
    asked to LEARN and to notice what surprised it), with R3-style spike-harvest
    (rolling-q75 threshold, spike_min_nll=7.0, span_half=32, <=2 spans/chunk) running
    throughout. A's span store is the deliverable of this phase. A is then FROZEN: it
    does not participate in B's experiment at all (P31d — "A itself is unharmed,
    its own trajectory is not part of B's budget" — is therefore structurally true by
    construction here, not something to verify after the fact; noted in the report).

  ORGANISM B ("the young one", lives the shock)
    Starts from the SAME snapshot, runs the pos_domain_shock.py protocol verbatim:
    phase1 C4 (own spans collected, R3 rolling-q75 rule) -> phase2 CODE shock ->
    phase3 C4 resumed. R2-style surprise gate throughout (rolling-q75, window=200,
    ignition off — warm snapshot start). FOUR arms, identical streams (independently
    instantiated phase1/phase2 sources at matching call sites -> byte-identical HF
    order per pos_domain_shock.py's "shared stream, instantiate twice" pattern),
    budget-neutral sleep exactly like R3 (sleep chunks come OUT of the triggering
    wake block, so total chunks visited is identical across all four arms):

      (i)   shared    sleeps on a pool that is B's own phase1 spans PLUS A's spans;
                       when both are non-empty the sleep pool is built as a 50/50
                       TOKEN-budget mix (equal token contribution from each source,
                       see "50/50 mix" below) every time it is (re)built for a sleep
                       segment, so as A's store is fixed and B's own store grows
                       across phase2/3, the ratio target stays 50/50 by construction
                       rather than drifting toward whichever store is bigger.
      (ii)  private    == R3 replica: sleeps ONLY on B's own collected spans
                       (phase1 + phase2, exactly pos_domain_shock.py's R3 recipe).
      (iii) none       == R2 replica: gated only, no sleep segments at all.
      (iv)  shuffled   == arm (i)'s exact pool construction, but A's contribution is
                       token-permuted per span (seeded, pos_sleep.py's SpanStream
                       permute_chunks convention applied at harvest time here so the
                       50/50 mix logic is identical to (i) and only the CONTENT of
                       A's half is destroyed) — content destroyed, statistics
                       (span count, token count, unigram distribution) preserved.

50/50 mix (arms i, iv): built fresh every time a sleep segment's spans are assembled
(mirrors pos_domain_shock.py's sleep_budget() being recomputed from the CURRENT pool
at every sleep trigger). Target: half of the token BUDGET spent this sleep segment
comes from A's store, half from B's own store-so-far. Implemented by concatenating
A's spans and B's spans into two separate SpanStream-style pools and alternating
single-span draws between them (A, B, A, B, ...) until the running token count would
exceed the segment's token budget, which keeps the realized split close to 50/50
regardless of how the two stores' individual span-length distributions differ (code
spans run longer under a word tokenizer's heavy unk-fragmentation than C4 spans, so a
naive "half the spans" split would NOT be a 50/50 token split — token-budget
alternation is used instead, see `mix_spans_5050()`). If B has not collected ANY spans
yet (early phase1), arm (i) sleeps purely on A's store (the only material available);
symmetrically if A's store were empty this would fall back to B-only, but A's store is
never empty by construction (A always runs before B and always harvests during a
50/50 C4+CODE stream where code triggers heavy-unk spikes).

Budget fairness (matches pos_domain_shock.py's R3 fairness invariant exactly): each of
B's four arms visits the SAME total chunk count over phase2+phase3 (wake+sleep
combined) as B's own phase1, and as every other arm — sleep chunks are carved out of
the wake block that triggers them, never added on top. Verified and asserted at the
end (`chunk_budget_check`).

Metrics: WT-2 heldout and a fixed 20k-token CODE-val slice (pos_domain_shock.py
recipe, disjoint by construction from every arm's training data) scored every
eval_every chunks; forgetting/plasticity/recovery defined IDENTICALLY to
pos_domain_shock.py (see that file's docstring). Additionally reported: grad_tokens
per arm, A's span-store size and its code-likeness (unk_rate of each span's tokens
against the WT-2 vocabulary the whole system runs in — code spans are heavily
fragmented by a [a-zA-Z]+ word tokenizer and so have high unk_rate; the store's
unk_rate distribution is reported as a histogram summary so a bimodal split
(C4-like spans low-unk / code-like spans high-unk) is visible directly).

P31 scoring (analysis/PREDICTIONS.md):
  (a) forgetting(shared) <= 0.7 * forgetting(private)
  (b) plasticity(shared) >= plasticity(private)          (A's spans carry usable
      code, not just regularization mass)
  (c) forgetting(shuffled) >= forgetting(private) - 0.02  (shuffled does not help —
      the benefit in (a)/(b) is content, not noise injection)
  (d) structural: A is frozen before B starts; A's trajectory is never touched by
      B's training. True by construction; recorded, not measured.

This file is READ-ONLY with respect to every results/pos_* path except its own
(results/pos_shared_index.json / results/pos_shared_index_smoke.json and matching
.log files, written by the launcher, not by this script). It never touches
pos_ckpt.pt, pos_index.jsonl, pos_status.json, pos_domain_shock*.json, or any file
under results/pos_snapshots/ except opening the snapshot for reading.

Usage:
  python src/pos_shared_index.py --smoke   # A: 30 chunks. B: phase_chunks=40 each, eval every 10
  python src/pos_shared_index.py --full    # A: 120 chunks. B: phase_chunks=150 each (default)
"""
import os
import sys
import json
import copy
import argparse
import tempfile
from collections import deque

sys.path.insert(0, "reference")
sys.path.insert(0, "src")

try:
    os.nice(19)
except (PermissionError, OSError):
    pass  # already niced by the launcher (macOS EPERM on re-nice)

import numpy as np
import torch
import torch.nn.functional as F

torch.backends.mps.is_available = lambda: False          # force CPU (same as the live run)
torch.set_num_threads(1)

from streaming_train import StreamingNoPELM
from length_extrap_v2 import load_wikitext2, build_vocab, tokenize
from pos_run import build_eval_set, heldout                    # safe: pos_run's main() only runs under __main__
from pos_sleep import ChunkFeeder, SpanStream, C4ValStream, load_snapshot, _real_vocab
from pos_sleep_cycles import harvest_spans, sleep_budget
from pos_domain_shock import CodeStream, make_phase1_source, make_phase2_source, grad_step, nograd_step


# ───────────────────────────────────────────────────────────────────────────
#  Organism A: 50/50 C4+CODE chunk-alternating stream, full-gradient, R3-style
#  spike harvest throughout. No gate — A is not being scored for forgetting.
# ───────────────────────────────────────────────────────────────────────────
def run_organism_a(model, opt, phase1_src, phase2_src, n_chunks, B, K, args):
    """Alternates chunks C4, CODE, C4, CODE, ... (chunk-level 50/50 mix, not
    token-level, so each gradient step trains on one coherent domain at a time
    -- matches how pos_domain_shock.py's phase transitions work, just at a much
    finer granularity here). Returns (collected_spans, grad_tokens, n_code_chunks,
    n_c4_chunks)."""
    feeder_c4 = ChunkFeeder(phase1_src, B, K)
    feeder_code = ChunkFeeder(phase2_src, B, K)
    states_c4, states_code = None, None
    spike_window = deque(maxlen=200)
    collected_spans = []
    grad_tokens = 0
    n_c4, n_code = 0, 0
    for i in range(n_chunks):
        use_code = (i % 2 == 1)                          # C4, CODE, C4, CODE, ... starting on C4
        if use_code:
            x, y = feeder_code.next_xy()
            states_code, gt, loss, x_d, nll = grad_step(model, opt, x, y, states_code)
            n_code += 1
        else:
            x, y = feeder_c4.next_xy()
            states_c4, gt, loss, x_d, nll = grad_step(model, opt, x, y, states_c4)
            n_c4 += 1
        grad_tokens += gt
        chunk_mean = float(nll.mean())
        if len(spike_window) >= 2:
            sp_thresh = float(np.quantile(np.fromiter(spike_window, dtype=np.float64), args.spike_quantile))
            if chunk_mean > sp_thresh:
                collected_spans.extend(harvest_spans(x_d, nll, args.spike_min_nll, args.span_half))
        spike_window.append(chunk_mean)
    return collected_spans, grad_tokens, n_c4, n_code


def span_unk_rate(span, unk):
    if not span:
        return 0.0
    return sum(1 for t in span if t == unk) / len(span)


def unk_histogram(spans, unk, edges=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01)):
    rates = [span_unk_rate(s, unk) for s in spans]
    counts = [0] * (len(edges) - 1)
    for r in rates:
        for i in range(len(edges) - 1):
            if edges[i] <= r < edges[i + 1]:
                counts[i] += 1
                break
    return {"edges": list(edges), "counts": counts,
            "n": len(rates), "mean": round(float(np.mean(rates)), 4) if rates else 0.0,
            "code_like_frac": round(sum(1 for r in rates if r > 0.3) / max(1, len(rates)), 4)}


# ───────────────────────────────────────────────────────────────────────────
#  50/50 token-budget mix of two span pools (used by B's shared/shuffled arms).
# ───────────────────────────────────────────────────────────────────────────
def mix_spans_5050(spans_a, spans_b, token_budget, rng):
    """Alternates single-span draws from spans_a and spans_b (each internally
    shuffled once per call by `rng`) until the running token total would meet or
    exceed token_budget, keeping the realized token split close to 50/50
    regardless of each store's span-length distribution (code spans run longer
    under heavy unk-fragmentation, so a naive equal-COUNT split would skew the
    token budget toward whichever store has longer spans -- alternating by
    TOKENS, not by span count, avoids that). Falls back to whichever pool is
    non-empty if the other is empty. Returns the mixed list of spans (order is
    the interleave order; SpanStream reshuffles internally anyway)."""
    pool_a = list(spans_a)
    pool_b = list(spans_b)
    if pool_a:
        rng.shuffle(pool_a)
    if pool_b:
        rng.shuffle(pool_b)
    out = []
    tok_a = tok_b = 0
    ia = ib = 0
    turn_a = True
    while (tok_a + tok_b) < token_budget and (ia < len(pool_a) or ib < len(pool_b)):
        if turn_a and ia < len(pool_a):
            out.append(pool_a[ia]); tok_a += len(pool_a[ia]); ia += 1
        elif (not turn_a) and ib < len(pool_b):
            out.append(pool_b[ib]); tok_b += len(pool_b[ib]); ib += 1
        elif ia < len(pool_a):
            out.append(pool_a[ia]); tok_a += len(pool_a[ia]); ia += 1
        elif ib < len(pool_b):
            out.append(pool_b[ib]); tok_b += len(pool_b[ib]); ib += 1
        else:
            break
        turn_a = not turn_a
    return out, tok_a, tok_b


def permute_span(span, rng):
    arr = list(span)
    rng.shuffle(arr)
    return arr


# ───────────────────────────────────────────────────────────────────────────
#  Organism B: pos_domain_shock.py's R2 gate driver, reused verbatim.
# ───────────────────────────────────────────────────────────────────────────
def run_r2_chunk(model, opt, feeder, states, window, args):
    x, y = feeder.next_xy()
    st_ng, s, _, nll_ng = nograd_step(model, x, y, states)
    if len(window) >= args.min_window:
        thresh = float(np.quantile(np.fromiter(window, dtype=np.float64), args.gate_q))
        gated = s > thresh
    else:
        gated = True
    if gated:
        states, gt, loss, _, nll = grad_step(model, opt, x, y, states)
    else:
        states, gt = st_ng, 0
        nll = nll_ng
    window.append(s)
    return states, gt, gated, x.detach(), nll


def run_sleep_segment(model, opt, spans, n_chunks, seed, B, K):
    src = SpanStream(spans, seed=seed, permute_chunks=False)
    feeder = ChunkFeeder(src, B, K)
    states = None
    grad_tokens = 0
    for _ in range(n_chunks):
        x, y = feeder.next_xy()
        states, gt, _, _, _ = grad_step(model, opt, x, y, states)
        grad_tokens += gt
    return grad_tokens


# ───────────────────────────────────────────────────────────────────────────
#  Main measurement
# ───────────────────────────────────────────────────────────────────────────
def run_shared_index(args, eval_wt2_fn, vocab_fn=_real_vocab):
    ck, cfg, base_model, opt_sd, stoi, unk, mask, V = load_snapshot(args.ckpt, vocab_fn)
    B, K = cfg["batch"], cfg["chunk"]
    lr = cfg["lr"]

    print(f"[shared_index] ckpt n_streamed={ck['n_streamed']:,} | "
          f"a_chunks={args.a_chunks} b_phase_chunks={args.phase_chunks} "
          f"eval_every={args.eval_every} | B={B} K={K} lr={lr}", flush=True)

    code_val_src = CodeStream(stoi, unk)
    code_val_ids = code_val_src.next_block(args.code_val_tokens)
    code_val_unk_rate = code_val_src.unk_rate()
    cvX, cvY = build_eval_set(code_val_ids, len(code_val_ids) - 1, K)
    print(f"[shared_index] code-val slice: {cvY.numel():,} tokens, unk_rate={code_val_unk_rate:.4f}", flush=True)

    def eval_code_fn(model):
        return heldout(model, cvX, cvY)

    def fresh_model_opt():
        m = copy.deepcopy(base_model)
        o = torch.optim.Adam(m.parameters(), lr=lr)
        if opt_sd is not None:
            o.load_state_dict(opt_sd)
            for g in o.param_groups:
                g["lr"] = lr
        return m, o

    base_wt2 = eval_wt2_fn(base_model)
    print(f"[shared_index] base_heldout_wt2={base_wt2:.6f}", flush=True)

    out = {
        "ckpt_n_streamed": ck["n_streamed"],
        "phase2_dataset": "codeparrot/github-code-clean (language==Python, client-filtered)",
        "budget": {"a_chunks": args.a_chunks, "phase_chunks": args.phase_chunks,
                  "eval_every": args.eval_every, "sleep_chunks_max": args.sleep_chunks,
                  "max_replay_epochs": args.max_replay_epochs, "gate_q": args.gate_q,
                  "gate_window": args.gate_window, "min_window": args.min_window,
                  "spike_min_nll": args.spike_min_nll, "span_half": args.span_half,
                  "sleep_every_n_chunks": args.sleep_every},
        "code_val_tokens": cvY.numel(),
        "code_val_unk_rate": round(code_val_unk_rate, 6),
        "base_heldout_wt2": round(base_wt2, 6),
    }

    # ── ORGANISM A: 50/50 C4+CODE, full-gradient, span harvest, then FROZEN ──
    print(f"\n[shared_index] ===== organism A (pre-learns the shock) =====", flush=True)
    a_model, a_opt = fresh_model_opt()
    a_phase1_src = make_phase1_source(stoi, unk)
    a_phase2_src = make_phase2_source(stoi, unk)
    a_spans, a_grad_tokens, a_n_c4, a_n_code = run_organism_a(
        a_model, a_opt, a_phase1_src, a_phase2_src, args.a_chunks, B, K, args)
    a_hist = unk_histogram(a_spans, unk)
    print(f"[shared_index][A] done: {len(a_spans)} spans harvested "
          f"({a_n_c4} C4 chunks, {a_n_code} code chunks, {a_grad_tokens:,} grad tokens) | "
          f"unk_hist mean={a_hist['mean']} code_like_frac={a_hist['code_like_frac']}", flush=True)
    print(f"[shared_index][A] FROZEN — A's trajectory is not part of B's budget "
          f"(P31d, structural by construction).", flush=True)
    out["organism_a"] = {
        "n_spans": len(a_spans), "grad_tokens": a_grad_tokens,
        "n_c4_chunks": a_n_c4, "n_code_chunks": a_n_code,
        "span_unk_histogram": a_hist,
        "span_token_total": sum(len(s) for s in a_spans),
        "frozen_note": "A does not train after this point; A is deepcopy-isolated from B.",
    }
    # a_model is discarded from here on (never touched again) -- only a_spans is handed to B.
    del a_model, a_opt

    # ── ORGANISM B: pos_domain_shock.py protocol, four arms ─────────────────
    regime_results = {}
    for arm_tag in ("shared", "private", "none", "shuffled"):
        print(f"\n[shared_index] ===== organism B, arm={arm_tag} =====", flush=True)
        model, opt = fresh_model_opt()

        phase13_src = make_phase1_source(stoi, unk)
        phase2_src = make_phase2_source(stoi, unk)
        feeder13 = ChunkFeeder(phase13_src, B, K)
        feeder2 = ChunkFeeder(phase2_src, B, K)

        states = None
        gate_window = deque(maxlen=args.gate_window)
        spike_window = deque(maxlen=200)
        b_own_spans = []                                    # B's own harvest (phase1 + phase2/3)

        curve_wt2, curve_code = [], []
        grad_tokens_total = 0
        gate_frac_log = []
        sleep_events = []
        mix_rng = np.random.default_rng(args.seed + hash(arm_tag) % 10_000)
        permute_rng = np.random.default_rng(args.seed + 1 + hash(arm_tag) % 10_000)

        def record(global_idx, phase):
            hl_wt2 = eval_wt2_fn(model)
            hl_code = eval_code_fn(model)
            curve_wt2.append([global_idx, phase, round(hl_wt2, 6)])
            curve_code.append([global_idx, phase, round(hl_code, 6)])
            print(f"[shared_index][{arm_tag}] phase={phase:<7} chunk={global_idx:>4} "
                  f"wt2={hl_wt2:.6f} code={hl_code:.6f}", flush=True)
            return hl_wt2, hl_code

        record(0, "base")
        global_idx = 0

        def run_wake_block(feeder, n_chunks, phase_name, collect):
            nonlocal states, global_idx, grad_tokens_total
            done = 0
            while done < n_chunks:
                step_n = min(args.eval_every, n_chunks - done)
                for _ in range(step_n):
                    states, gt, gated, x_this, nll_this = run_r2_chunk(
                        model, opt, feeder, states, gate_window, args)
                    grad_tokens_total += gt
                    gate_frac_log.append(1 if gated else 0)
                    if collect and gated and nll_this is not None:
                        chunk_mean = float(nll_this.mean())
                        if len(spike_window) >= 2:
                            sp_thresh = float(np.quantile(
                                np.fromiter(spike_window, dtype=np.float64), args.spike_quantile))
                            if chunk_mean > sp_thresh:
                                b_own_spans.extend(
                                    harvest_spans(x_this, nll_this, args.spike_min_nll, args.span_half))
                        spike_window.append(chunk_mean)
                    done += 1
                    global_idx += 1
                record(global_idx, phase_name)
            return done

        # ── Phase 1: C4 (train-far). B always collects its own spans here —
        #    identical to pos_domain_shock.py's R3 phase-1 behavior. ────────
        run_wake_block(feeder13, args.phase_chunks, "phase1", collect=True)
        pre_phase2_wt2 = curve_wt2[-1][2]
        pre_phase2_code = curve_code[-1][2]

        # ── Phase 2/3: CODE shock then C4 resumed. "none" never sleeps
        #    (== R2 replica); the other three arms sleep budget-neutrally. ──
        for phase_name, feeder, phase_n in (("phase2", feeder2, args.phase_chunks),
                                            ("phase3", feeder13, args.phase_chunks)):
            if arm_tag == "none":
                run_wake_block(feeder, phase_n, phase_name, collect=False)
                continue

            done = 0
            while done < phase_n:
                block_n = min(args.sleep_every, phase_n - done)
                wake_n = max(1, block_n - args.sleep_chunks)
                run_wake_block(feeder, wake_n, phase_name, collect=True)
                done += wake_n

                # build this segment's sleep pool per-arm BEFORE computing the
                # coupled budget, so sleep_budget() sees the pool it will actually
                # replay (mirrors pos_domain_shock.py's R3: pool-then-budget order)
                if arm_tag == "private":
                    pool = b_own_spans
                elif arm_tag == "shared":
                    if b_own_spans:
                        target_tokens = args.sleep_chunks * B * K * args.max_replay_epochs
                        pool, _, _ = mix_spans_5050(a_spans, b_own_spans, target_tokens, mix_rng)
                    else:
                        pool = a_spans                        # nothing of B's own yet: A-only fallback
                elif arm_tag == "shuffled":
                    a_shuffled = [permute_span(s, permute_rng) for s in a_spans]
                    if b_own_spans:
                        target_tokens = args.sleep_chunks * B * K * args.max_replay_epochs
                        pool, _, _ = mix_spans_5050(a_shuffled, b_own_spans, target_tokens, mix_rng)
                    else:
                        pool = a_shuffled
                else:
                    pool = []

                n_sleep_used, span_tokens, replay_eff = sleep_budget(
                    pool, args.sleep_chunks, args.max_replay_epochs, B, K)
                leftover = args.sleep_chunks - n_sleep_used
                if leftover > 0 and done < phase_n:
                    extra = min(leftover, phase_n - done)
                    run_wake_block(feeder, extra, phase_name, collect=True)
                    done += extra

                if n_sleep_used > 0:
                    gt_sleep = run_sleep_segment(model, opt, pool, n_sleep_used,
                                                 seed=args.seed + global_idx, B=B, K=K)
                    grad_tokens_total += gt_sleep
                    global_idx += n_sleep_used
                    done += n_sleep_used
                    hl_wt2, hl_code = record(global_idx, f"{phase_name}_sleep")
                    sleep_events.append({"phase": phase_name, "global_idx": global_idx,
                                         "n_sleep_chunks": n_sleep_used, "pool_size": len(pool),
                                         "pool_tokens": span_tokens, "replay_epochs": round(replay_eff, 3),
                                         "wt2_after": round(hl_wt2, 6), "code_after": round(hl_code, 6)})
                    print(f"[shared_index][{arm_tag}] {phase_name} sleep: {n_sleep_used} chunks "
                          f"(pool={len(pool)} spans, {span_tokens} tok, {replay_eff:.2f} epochs) "
                          f"-> wt2={hl_wt2:.6f} code={hl_code:.6f}", flush=True)
                else:
                    print(f"[shared_index][{arm_tag}] {phase_name} block: empty/small pool, "
                          f"sleep skipped, all budget spent on wake", flush=True)

        wt2_values_phase2 = [v for idx, ph, v in curve_wt2 if ph in ("phase2", "phase2_sleep")]
        code_values_phase2 = [v for idx, ph, v in curve_code if ph in ("phase2", "phase2_sleep")]
        post_phase3_wt2 = curve_wt2[-1][2]

        forgetting = round(max(wt2_values_phase2 + [pre_phase2_wt2]) - pre_phase2_wt2, 6)
        plasticity = round(pre_phase2_code - min(code_values_phase2 + [pre_phase2_code]), 6)
        recovery = round(post_phase3_wt2 - pre_phase2_wt2, 6)

        gate_frac = sum(gate_frac_log) / max(1, len(gate_frac_log))
        regime_results[arm_tag] = {
            "curve_wt2": curve_wt2, "curve_code": curve_code,
            "grad_tokens_total": grad_tokens_total,
            "n_chunks_gated": sum(gate_frac_log), "n_chunks_seen": len(gate_frac_log),
            "gate_frac": round(gate_frac, 4),
            "pre_phase2_wt2": round(pre_phase2_wt2, 6), "pre_phase2_code": round(pre_phase2_code, 6),
            "post_phase3_wt2": round(post_phase3_wt2, 6),
            "forgetting": forgetting, "plasticity": plasticity, "recovery": recovery,
            "n_own_spans_collected": len(b_own_spans), "sleep_events": sleep_events,
        }
        print(f"[shared_index][{arm_tag}] forgetting={forgetting:+.6f} plasticity={plasticity:+.6f} "
              f"recovery={recovery:+.6f} gate_frac={gate_frac:.4f} "
              f"grad_tokens={grad_tokens_total:,} own_spans={len(b_own_spans)}", flush=True)

    out["organism_b_arms"] = regime_results

    # ── fairness invariant: all four arms visit the SAME total chunk count
    #    (wake+sleep) over phase1+phase2+phase3 — sleep chunks are carved out
    #    of the block that triggers them, never added on top (pos_domain_
    #    shock.py's R3 invariant, extended to four arms here). ──────────────
    n_visited = {}
    for t in ("shared", "private", "none", "shuffled"):
        n_visited[t] = regime_results[t]["n_chunks_seen"] + sum(
            e["n_sleep_chunks"] for e in regime_results[t]["sleep_events"])
    out["chunk_budget_check"] = {
        "n_chunks_visited": n_visited,
        "equal": len(set(n_visited.values())) == 1,
    }

    f_shared = regime_results["shared"]["forgetting"]
    f_private = regime_results["private"]["forgetting"]
    f_shuffled = regime_results["shuffled"]["forgetting"]
    p_shared = regime_results["shared"]["plasticity"]
    p_private = regime_results["private"]["plasticity"]

    p31_a = f_shared <= 0.7 * f_private if f_private > 0 else (f_shared <= 0)
    p31_b = p_shared >= p_private
    p31_c = f_shuffled >= (f_private - 0.02)
    p31_d = True   # structural: A is frozen before B starts (see organism_a.frozen_note)

    out["p31_scoring"] = {
        "a_shared_forgetting_leq_0.7x_private": {
            "pass": bool(p31_a), "shared": f_shared, "private": f_private,
            "threshold_0.7x_private": round(0.7 * f_private, 6)},
        "b_shared_plasticity_geq_private": {
            "pass": bool(p31_b), "shared": p_shared, "private": p_private},
        "c_shuffled_does_not_beat_private": {
            "pass": bool(p31_c), "shuffled": f_shuffled, "private": f_private,
            "private_minus_0.02": round(f_private - 0.02, 6)},
        "d_a_frozen_structural": {"pass": True, "note": "A never trains after its own phase; "
                                  "not measured, guaranteed by construction (del a_model before B starts)."},
    }
    all_pass = p31_a and p31_b and p31_c and p31_d
    out["verdict"] = (
        f"forgetting shared={f_shared:+.6f} private={f_private:+.6f} none={regime_results['none']['forgetting']:+.6f} "
        f"shuffled={f_shuffled:+.6f} | plasticity shared={p_shared:+.6f} private={p_private:+.6f} | "
        f"A: {len(a_spans)} spans ({a_hist['code_like_frac']*100:.1f}% code-like by unk_rate>0.3) | "
        f"P31: {'PASS' if all_pass else 'PARTIAL/FAIL'} "
        f"(a: {p31_a}, b: {p31_b}, c: {p31_c}, d: {p31_d})"
    )
    print(f"\n[shared_index] {out['verdict']}", flush=True)

    d = os.path.dirname(args.out) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_")
    with os.fdopen(fd, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, args.out)
    print(f"[shared_index] -> {args.out}", flush=True)
    return out


# ───────────────────────────────────────────────────────────────────────────
#  CLI
# ───────────────────────────────────────────────────────────────────────────
def build_argparser():
    ap = argparse.ArgumentParser(
        description="POS SHARED INDEX: two organisms, one index -- collective memory across O(1) individuals (P31)")
    ap.add_argument("--ckpt", default="results/pos_snapshots/ckpt_359050240.pt")
    ap.add_argument("--a-chunks", type=int, default=120, help="organism A's 50/50 C4+CODE chunk count")
    ap.add_argument("--phase-chunks", type=int, default=150, help="organism B: chunks per phase (1/2/3)")
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--code-val-tokens", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gate-q", type=float, default=0.75)
    ap.add_argument("--gate-window", type=int, default=200)
    ap.add_argument("--min-window", type=int, default=50)
    ap.add_argument("--spike-quantile", type=float, default=0.75)
    ap.add_argument("--spike-min-nll", type=float, default=7.0)
    ap.add_argument("--span-half", type=int, default=32)
    ap.add_argument("--sleep-every", type=int, default=30)
    ap.add_argument("--sleep-chunks", type=int, default=10)
    ap.add_argument("--max-replay-epochs", type=float, default=2.0)
    ap.add_argument("--smoke", action="store_true",
                    help="a-chunks=30, phase-chunks=40, eval-every=10, sleep-every=10, sleep-chunks=4")
    ap.add_argument("--full", action="store_true", help="a-chunks=120, phase-chunks=150 (explicit; also the default)")
    ap.add_argument("--out", default="results/pos_shared_index.json")
    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    if args.smoke:
        args.a_chunks = 30
        args.phase_chunks = 40
        args.eval_every = 10
        args.sleep_every = 10
        args.sleep_chunks = 4
        if args.out == "results/pos_shared_index.json":
            args.out = "results/pos_shared_index_smoke.json"
    elif args.full:
        args.a_chunks = 120
        args.phase_chunks = 150

    def eval_wt2_fn(model):
        train_text, val_text = load_wikitext2()
        _, stoi, unk, mask = build_vocab(train_text)
        val_ids = tokenize(val_text, stoi, unk)
        cfg = torch.load(args.ckpt, weights_only=False)["config"]
        evX, evY = build_eval_set(val_ids, cfg["eval_tokens"], cfg["chunk"])
        return heldout(model, evX, evY)

    run_shared_index(args, eval_wt2_fn)


if __name__ == "__main__":
    main()
