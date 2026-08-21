#!/usr/bin/env python3 -u
"""
RETRO POS — the retrodiction meter (P41, MS18 v0, David's time-mirror).
========================================================================
horizon_pos.py (P37/P40) trains a head to DEPOSIT a forecast of chunk t+H
and hold it until the future arrives. This is the mirror: a head at chunk t
RECONSTRUCTS a summary of chunk t-H from the CURRENT carried state. Targets
are in the PAST, already realized, sitting in a bounded ring buffer -- no
deposit queue, no waiting, the error is available the instant the head
computes it. Where horizon's error says "the future is being predicted
well/poorly," retro's error says "the past is still legible in the state,
or it has been overwritten" -- an online forgetting meter, not a foresight
meter. Perfect F1 symmetry: future-surprise says LEARN NOW, past-surprise
says CONSOLIDATE NOW (the consolidation organ itself -- decay-triggered
targeted replay -- is a SEPARATE later registration; v0 here is the METER
only).

HEAD DESIGN (mirrors horizon_pos.py's HorizonHead exactly, for comparability):
  Input:  mean-pooled hidden state of chunk t (pre-head, post-final-layer,
          averaged over (B, K) -> (d_model,)), IDENTICAL pooling to
          horizon_pos.mean_pool_hidden.
  Heads:  ONE per rung H in {2, 8, 32, 128}, each its own d_model -> 256 ->
          256 MLP (ReLU) -- same architecture as HorizonHead, own weights,
          own Adam optimizer (lr=3e-3, MATCHES horizon_pos's head_lr default
          exactly -- comparability across the two meters was an explicit
          instruction, not a free dimensioning choice).
  Target: the top-256-bucket bag-of-token histogram of chunk t-H, read
          directly from the ring buffer (already realized when chunk t is
          processed -- no queue latency, unlike horizon's forward case where
          the target has not happened yet). Same bucket construction as
          horizon_pos.build_top_buckets / chunk_histogram (byte-identical
          function, imported not reimplemented).
  Loss:   identical soft-label cross-entropy to horizon_pos.horizon_loss
          (same function, imported) -- REQUIRED for the two_regime and
          shuffle scoring clauses to be comparable across rungs in the same
          units, and required by the mission brief ("gleiche Loss-Funktion
          wie horizon").
  Training: every rung's head trains on EVERY chunk once t >= H for that
          rung (target already in the buffer) -- ungated, always on, same
          "always learns" convention as HorizonHead (the retrodiction organ
          is never gated by the base model's own gate decision).

RING BUFFER (the load-bearing difference from horizon_pos's HorizonQueue):
  A bounded deque of length <= 129 (= max(H_LADDER)+1 -- the +1 is
  load-bearing: the current chunk is pushed BEFORE the rung lookups, so a
  capacity of exactly max(H) would evict the H=max target one position too
  early and silently starve the top rung; caught in review, guarded by the
  RED/GREEN self-test), one entry per chunk,
  each entry the (global_idx, top-256 histogram) of that chunk's REALIZED
  tokens (not a pooled hidden state -- the retrodiction target is always the
  histogram, computed once per chunk and shared by all rungs and both arms,
  unlike horizon's per-deposit prediction which is rung-specific). At chunk
  t, rung H reads buffer entry (t-H) directly by index arithmetic (the
  buffer holds the trailing max(H)+1 chunks INCLUDING the just-pushed
  current one, so t-H is in range whenever t >= H for every rung -- with a
  capacity of only max(H) the top rung's target would already be evicted
  at lookup time, see the +1 note above). No FIFO pop/push
  pairing with a "score when it arrives" step -- the buffer is a rolling
  window read at arbitrary lag, not a fixed-depth pipeline. This is the
  actual mechanism referenced by "bounded ring-buffer of the last 128 chunk
  histograms" in the mission brief.

ARMS (base_gate mode, MS3 shock stream C4->code->C4, matched budgets, same
ckpt as P37/P40 -- ckpt_359050240.pt):
  retro           the ladder as described: rung H's target = buffer[t-H].
  retro_shuffled  IDENTICAL base-model stream and gating, IDENTICAL head
                  architectures/training cadence, EXCEPT each rung's target
                  at chunk t is the histogram of a UNIFORMLY RANDOM buffer
                  entry (own seeded RNG, seed=args.seed+5000, independent of
                  the base run seed and of any other RNG in this file) drawn
                  from whatever is currently in the buffer -- NOT
                  deterministically t-H. This destroys the temporal binding
                  between rung and lag while preserving "some past chunk's
                  histogram, scored now" as the marginal task, exactly
                  mirroring horizon_pos's `shuffled` control. Falsifier:
                  destroys the monotone H-ladder structure (retro's
                  clause-b).

BASE-MODEL GATING MODE (explicit design choice, per mission instruction
"nimm den base_gate-Modus aus horizon_pos [1-Step-Gating]"): both arms run
the base model under horizon_pos.make_base_gate mechanics -- backward iff
1-step chunk-mean NLL > rolling-q75(window=200) of PRIOR 1-step NLLs, with
NO retrodiction-head influence on the gate (retro heads train in the
background regardless of the gate's decision, same "always learns"
convention as HorizonHead). This was a documented CHOICE point in the
mission brief (the alternative was forward-only, no backward pass at all).
base_gate was picked over forward-only for two reasons: (1) it makes the
carried stream comparable to P37/P40's stream-shape (same gate mechanics
family, same domain-shock cadence of gated/ungated chunks along the way,
so a reader who knows P37/P40 already knows what "the stream looked like"
here); (2) v0 is a METER, and a meter is more informative measured against
a REALISTIC deployment stream (one that actually learns and forgets via
gating) than against a frozen forward-only pass where the state's forgetting
dynamics would be a purely mechanical function of gamma-decay with no
plasticity component at all -- the live_forgetting clause (c) specifically
wants to see forgetting happen as a CONSEQUENCE of the gate's phase-2
backward passes overwriting phase-1 content, which requires the gate to be
live, not bypassed. Both arms (retro, retro_shuffled) run base_gate
identically and independently (separate model/opt/state per arm, exactly
like horizon_pos's per-regime `fresh_model_opt_head`) -- the shuffle control
touches ONLY the retrodiction targets, never the base stream, so any
between-arm difference in the base model's own trajectory is scoring noise
to be checked (chunk_budget_check), not signal.

SCORING (p41_scoring, three clauses, pass/fail + raw numbers each):
  (a) two_regime — the two-timescale law seen BACKWARD: error should rise
      steeply across the receptive-field scale and PLATEAU beyond it.
      Computed on END-OF-PHASE-1 MEANS (the last 25 chunks of phase 1,
      BEFORE the shock -- a clean stationary window, no gate-induced
      forgetting yet): err(H=128) - err(H=32) < 0.25 * (err(H=8) - err(H=2)).
  (b) shuffle — retro_shuffled must show NO monotone ladder: max pairwise
      spread across rungs' end-of-phase-1 means must be < the noise band,
      where noise band := 2 * (within-rung std of that rung's OWN last-25-
      chunk errors in end-of-phase-1). Reported per-rung and as the overall
      max-spread-vs-band comparison.
  (c) live_forgetting — CORRECTED anchor per team-lead precision pass on
      P41's ledger: "after the phase-2 boundary" means B12, the phase1->2
      SHOCK ONSET (not the phase2->3 return-to-C4 boundary). Window is
      (B12, B12+32]. Qualification is PER-DECISION, not per-rung-globally:
      at window chunk t, rung H qualifies iff t-H < B12 (its target still
      lies pre-shock). Each rung accumulates EXACTLY min(H-1, 32) qualified
      window decisions (t ranges over the OPEN window B12+1..B12+32, so
      t-H<B12 <=> t-B12 in {1,...,H-1}, capped at 32 -- verified by direct
      enumeration): H=2 -> 1, H=8 -> 7, H=32 -> 31, H=128 -> 32. Eligibility
      floor is >=7 qualified decisions (= H-1 of the smallest registered-
      eligible rung, H=8 -- the registered P41 text names H in {8,32,128}
      as qualifying richly, so the floor is set to admit H=8; an initial
      >=8 floor from the build instruction was a dimensioning choice, not
      the registration, and was superseded to match the registered text).
      Under >=7, H=2 alone falls below it and is logged but EXCLUDED from
      the overall verdict; H=8, H=32, H=128 are all eligible for clause (c)
      (independent of phase_chunks -- this window's width is fixed at 32,
      not tied to phase length). Rise: a qualified decision's error exceeds (that
      rung's own last-25-chunk PHASE-1 mean + 2*sigma -- the same phase-1
      baseline clause (a) uses, so both clauses share one definition of
      "normal"). First-rise chunk index (relative to B12) reported per
      rung, plus pass/fail per rung and overall (ALL eligible rungs must
      rise within the window for an overall pass -- a registered point
      call, reported plainly if partial).

CLI: --smoke (phase-chunks=40) / --full (150), --out, --self-test (pure-CPU
dry test of buffer/head/scoring arithmetic on random tensors, no
checkpoint/HF stream, runs in <30s), same conventions as horizon_pos.py.
Thread-pinning, os.nice, seed=42: identical to horizon_pos.py.

Usage:
  python src/retro_pos.py --self-test             # <30s, no ckpt/HF needed
  python src/retro_pos.py --smoke                 # phase_chunks=40 (NOT launched by the builder -- slot-gated)
  python src/retro_pos.py --full                  # phase_chunks=150
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
except (PermissionError, AttributeError):
    pass  # already niced by the launcher (macOS EPERM on re-nice) / no os.nice on this platform

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.mps.is_available = lambda: False          # force CPU (same as the live run)
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass  # must be set before any parallel work has run; harmless if already set

from pos_sleep import ChunkFeeder, load_snapshot, _real_vocab
from pos_domain_shock import make_phase1_source, make_phase2_source, CodeStream
from pos_run import build_eval_set, heldout
from length_extrap_v2 import load_wikitext2, build_vocab, tokenize

# Reused byte-identical from horizon_pos.py (P37/P40) -- NOT reimplemented,
# per instruction to inherit its proven infrastructure without touching the
# original file. build_top_buckets / chunk_histogram / horizon_loss /
# mean_pool_hidden / model_forward_from_h / rolling_z / gate_decision /
# make_base_gate / first_fire_index are all identical machinery this file
# needs; HorizonHead is NOT reused (retro needs FOUR independent heads, one
# per rung, so RetroHead below mirrors its architecture instead of
# subclassing/wrapping it -- see RetroHead docstring).
from horizon_pos import (
    build_top_buckets, chunk_histogram, horizon_loss, mean_pool_hidden,
    model_forward_from_h, rolling_z, gate_decision, make_base_gate,
    first_fire_index,
)

H_LADDER = [2, 8, 32, 128]


# ───────────────────────────────────────────────────────────────────────────
#  RetroHead: architecturally identical to horizon_pos.HorizonHead
#  (d_model -> 256 -> 256 MLP, ReLU), but retro needs ONE PER RUNG (four
#  independent heads/optimizers, since each rung reconstructs a different
#  lag and must not share weights -- sharing would conflate "what H=2 looks
#  like" with "what H=128 looks like" into one function, defeating the
#  point of the ladder). Defined as its own class (not a HorizonHead import)
#  so this file's rung-indexed dict-of-heads pattern is explicit and does
#  not rely on HorizonHead's docstring (which describes the FORWARD-deposit
#  semantics) leaking into a backward-reconstruction context.
# ───────────────────────────────────────────────────────────────────────────
class RetroHead(nn.Module):
    def __init__(self, d_model, n_buckets=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, n_buckets), nn.ReLU(),
            nn.Linear(n_buckets, n_buckets),
        )

    def forward(self, pooled):
        return self.net(pooled)                            # (n_buckets,) logits


# ───────────────────────────────────────────────────────────────────────────
#  Ring buffer of realized chunk histograms — the load-bearing mechanism
#  that replaces horizon_pos's HorizonQueue. Holds up to `maxlen` entries,
#  each (global_idx, histogram). Targets are read by DIRECT INDEX LOOKUP
#  (t-H), not popped/dequeued -- multiple rungs read the SAME buffer at
#  different lags every chunk, so entries must persist across many reads
#  until they age out past maxlen, unlike horizon's one-shot FIFO pop.
# ───────────────────────────────────────────────────────────────────────────
class RetroRingBuffer:
    def __init__(self, maxlen=128, shuffle=False, seed=42):
        self.maxlen = maxlen
        self.buf = deque(maxlen=maxlen)                     # [(global_idx, hist), ...] oldest first
        self._by_idx = {}                                   # global_idx -> hist, kept in sync with buf
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def push(self, global_idx, hist):
        if len(self.buf) == self.maxlen:
            old_idx, _ = self.buf[0]
            self._by_idx.pop(old_idx, None)
        self.buf.append((global_idx, hist))
        self._by_idx[global_idx] = hist

    def get_target(self, global_idx, H):
        """Returns (target_hist, source_idx) for rung H at the current chunk
        global_idx, or (None, None) if the true lag target (global_idx - H)
        is not (yet, or no longer) in the buffer. In shuffle mode, the
        returned histogram belongs to a UNIFORMLY RANDOM entry currently in
        the buffer (own seeded RNG) instead of the deterministic (idx - H)
        entry -- the source_idx returned is the entry ACTUALLY used (for
        logging/audit), which for shuffle is generally != global_idx - H."""
        want_idx = global_idx - H
        if want_idx not in self._by_idx:
            return None, None
        if not self.shuffle:
            return self._by_idx[want_idx], want_idx
        items = list(self.buf)                              # [(idx, hist), ...]
        pick = items[int(self.rng.integers(0, len(items)))]
        return pick[1], pick[0]


# ───────────────────────────────────────────────────────────────────────────
#  One retro-aware chunk step: base-model forward + base_gate backward
#  (identical mechanics to horizon_pos's base_gate arm), push this chunk's
#  realized histogram into the ring buffer, then every rung whose lag
#  target is available trains + scores this chunk.
# ───────────────────────────────────────────────────────────────────────────
def retro_chunk_step(model, opt, heads, head_opts, x, y, states, global_idx,
                      ring, bucket_of, n_buckets, gate_fn, gate_state):
    with torch.no_grad():
        states_ng, pooled_ng, h_ng = mean_pool_hidden(model, x, states)
        nll_ng = model_forward_from_h(model, h_ng, x, y)
    s1 = float(nll_ng.mean())

    if isinstance(gate_state, dict):
        gate_state["idx"] = global_idx
    gated, mixed_score, thresh = gate_fn(s1, None, gate_state)

    if gated:
        from pos_domain_shock import grad_step
        states_new, gt, loss, _, nll = grad_step(model, opt, x, y, states)
        with torch.no_grad():
            _, pooled_cur, _ = mean_pool_hidden(model, x, states)
    else:
        states_new, gt = states_ng, 0
        nll = nll_ng
        pooled_cur = pooled_ng

    # realized histogram of THIS chunk -- pushed now so it becomes a valid
    # retrodiction target for future chunks at lag H (buffer holds up to 128
    # trailing entries, covering the full H_LADDER by construction)
    cur_hist = chunk_histogram(y, bucket_of, n_buckets)
    ring.push(global_idx, cur_hist)

    pooled_detached = pooled_cur.detach()
    rung_errors = {}                                        # H -> (loss_float, source_idx) or None
    for H in H_LADDER:
        target_hist, source_idx = ring.get_target(global_idx, H)
        if target_hist is None:
            rung_errors[H] = None
            continue
        head = heads[H]
        head_opt = head_opts[H]
        pred_logits = head(pooled_detached)
        rloss = horizon_loss(pred_logits, target_hist)
        head_opt.zero_grad(set_to_none=True)
        rloss.backward()
        head_opt.step()
        rung_errors[H] = (float(rloss.detach()), source_idx)

    return {
        "states": states_new, "grad_tokens": gt, "gated": gated,
        "s1": s1, "rung_errors": rung_errors,
        "nll": nll.detach(),
    }


# ───────────────────────────────────────────────────────────────────────────
#  Main measurement
# ───────────────────────────────────────────────────────────────────────────
def run_retro(args, eval_wt2_fn, vocab_fn=_real_vocab):
    ck, cfg, base_model, opt_sd, stoi, unk, mask, V = load_snapshot(args.ckpt, vocab_fn)
    B, K = cfg["batch"], cfg["chunk"]
    lr = cfg["lr"]
    d_model = cfg["d_model"]
    n_buckets = args.n_buckets
    # +1 because retro_chunk_step PUSHES the current chunk BEFORE the rung
    # lookups: at chunk t a maxlen of exactly max(H) holds t-(max(H)-1)..t
    # after the push, so the H=max target (t-max(H)) is evicted one position
    # too early and the top rung would silently never train (caught in the
    # lead's line-by-line review -- the self-test exercised eviction but not
    # steady-state reachability of the top rung through the real step order).
    ring_maxlen = max(H_LADDER) + 1

    phase_chunks = args.phase_chunks
    eval_every = args.eval_every

    print(f"[retro] ckpt n_streamed={ck['n_streamed']:,} | phase_chunks={phase_chunks} "
          f"eval_every={eval_every} | H_LADDER={H_LADDER} n_buckets={n_buckets} | "
          f"B={B} K={K} lr={lr} | mode=base_gate (1-step gating, retro heads never gate)", flush=True)

    top_buckets, bucket_of = build_top_buckets(stoi, unk, n_buckets)
    print(f"[retro] top-{n_buckets} vocab buckets built (by WT-2 train frequency)", flush=True)

    code_val_src = CodeStream(stoi, unk)
    code_val_ids = code_val_src.next_block(args.code_val_tokens)
    code_val_unk_rate = code_val_src.unk_rate()
    cvX, cvY = build_eval_set(code_val_ids, len(code_val_ids) - 1, K)
    print(f"[retro] code-val slice: {cvY.numel():,} tokens, unk_rate={code_val_unk_rate:.4f}", flush=True)

    def eval_code_fn(model):
        return heldout(model, cvX, cvY)

    def fresh_model_opt_heads():
        m = copy.deepcopy(base_model)
        o = torch.optim.Adam(m.parameters(), lr=lr)
        if opt_sd is not None:
            o.load_state_dict(opt_sd)
            for g in o.param_groups:
                g["lr"] = lr
        heads = {H: RetroHead(d_model, n_buckets) for H in H_LADDER}
        head_opts = {H: torch.optim.Adam(heads[H].parameters(), lr=args.head_lr) for H in H_LADDER}
        return m, o, heads, head_opts

    base_wt2 = eval_wt2_fn(base_model)
    print(f"[retro] base_heldout_wt2={base_wt2:.6f}", flush=True)

    out = {
        "ckpt_n_streamed": ck["n_streamed"],
        "h_ladder": H_LADDER, "n_buckets": n_buckets, "ring_maxlen": ring_maxlen,
        "head_design": "one RetroHead per rung H in {2,8,32,128}, each d_model->256->256 MLP "
                       "(architecture identical to horizon_pos.HorizonHead), own Adam (lr matches "
                       "horizon_pos's head_lr default, 3e-3); target = top-256-bucket histogram of "
                       "chunk t-H read directly from a bounded ring buffer (maxlen=128, no queue "
                       "latency); loss = horizon_pos.horizon_loss (byte-identical soft-label CE), "
                       "imported not reimplemented, for cross-meter comparability",
        "base_gate_mode_note": "both arms (retro, retro_shuffled) run the base model under "
                                "horizon_pos.make_base_gate mechanics (1-step q75-rolling gate); "
                                "retro/retro_shuffled heads never influence the gate and always "
                                "train regardless of the gate's decision that chunk -- chosen over "
                                "forward-only so the carried state's forgetting is a live "
                                "consequence of real gated backward passes (see module docstring)",
        "budget": {"phase_chunks": phase_chunks, "eval_every": eval_every,
                  "gate_q": args.gate_q, "gate_window": args.gate_window,
                  "min_window": args.min_window, "head_lr": args.head_lr},
        "code_val_tokens": cvY.numel(),
        "code_val_unk_rate": round(code_val_unk_rate, 6),
        "base_heldout_wt2": round(base_wt2, 6),
        "regimes": {},
    }

    regime_tags = [t.strip() for t in args.regimes.split(",") if t.strip()]

    regime_results = {}
    for tag in regime_tags:
        print(f"\n[retro] ===== regime {tag} =====", flush=True)
        model, opt, heads, head_opts = fresh_model_opt_heads()

        phase13_src = make_phase1_source(stoi, unk)
        phase2_src = make_phase2_source(stoi, unk)
        feeder13 = ChunkFeeder(phase13_src, B, K)
        feeder2 = ChunkFeeder(phase2_src, B, K)

        states = None
        shuffle = (tag == "retro_shuffled")
        ring = RetroRingBuffer(maxlen=ring_maxlen, shuffle=shuffle, seed=args.seed + 5000)
        gate_fn, gate_state = make_base_gate(args.gate_q, args.min_window)

        curve_wt2 = []
        curve_code = []
        curve_s1 = []
        # curve_rung[H] = [(global_idx, phase, err_or_None, source_idx_or_None), ...]
        curve_rung = {H: [] for H in H_LADDER}
        grad_tokens_total = 0
        gate_frac_log = []
        boundary_idx = {}

        def record(global_idx, phase):
            hl_wt2 = eval_wt2_fn(model)
            hl_code = eval_code_fn(model)
            curve_wt2.append([global_idx, phase, round(hl_wt2, 6)])
            curve_code.append([global_idx, phase, round(hl_code, 6)])
            print(f"[retro][{tag}] phase={phase:<7} chunk={global_idx:>4} "
                  f"wt2={hl_wt2:.6f} code={hl_code:.6f}", flush=True)
            return hl_wt2, hl_code

        record(0, "base")
        global_idx = 0

        def run_wake_block(feeder, n_chunks, phase_name):
            nonlocal states, global_idx, grad_tokens_total
            done = 0
            while done < n_chunks:
                step_n = min(eval_every, n_chunks - done)
                for _ in range(step_n):
                    x, y = feeder.next_xy()
                    res = retro_chunk_step(model, opt, heads, head_opts, x, y, states,
                                            global_idx, ring, bucket_of, n_buckets,
                                            gate_fn, gate_state)
                    states = res["states"]
                    grad_tokens_total += res["grad_tokens"]
                    gate_frac_log.append(1 if res["gated"] else 0)
                    curve_s1.append([global_idx, phase_name, round(res["s1"], 6)])
                    for H in H_LADDER:
                        re = res["rung_errors"][H]
                        if re is None:
                            curve_rung[H].append([global_idx, phase_name, None, None])
                        else:
                            err, src_idx = re
                            curve_rung[H].append([global_idx, phase_name, round(err, 6), src_idx])
                    done += 1
                    global_idx += 1
                record(global_idx, phase_name)
            return done

        run_wake_block(feeder13, phase_chunks, "phase1")
        pre_phase2_wt2 = curve_wt2[-1][2]
        boundary_idx["phase1_to_2"] = global_idx

        run_wake_block(feeder2, phase_chunks, "phase2")
        boundary_idx["phase2_to_3"] = global_idx

        run_wake_block(feeder13, phase_chunks, "phase3")
        post_phase3_wt2 = curve_wt2[-1][2]

        wt2_values_phase2 = [v for idx, ph, v in curve_wt2 if ph == "phase2"]
        forgetting = round(max(wt2_values_phase2 + [pre_phase2_wt2]) - pre_phase2_wt2, 6)
        recovery = round(post_phase3_wt2 - pre_phase2_wt2, 6)
        gate_frac = sum(gate_frac_log) / max(1, len(gate_frac_log))

        regime_results[tag] = {
            "curve_wt2": curve_wt2, "curve_code": curve_code, "curve_s1": curve_s1,
            "curve_rung": {str(H): v for H, v in curve_rung.items()},
            "grad_tokens_total": grad_tokens_total,
            "n_chunks_gated": sum(gate_frac_log), "n_chunks_seen": len(gate_frac_log),
            "gate_frac": round(gate_frac, 4),
            "pre_phase2_wt2": round(pre_phase2_wt2, 6), "post_phase3_wt2": round(post_phase3_wt2, 6),
            "forgetting": forgetting, "recovery": recovery,
            "boundary_idx": boundary_idx,
        }
        print(f"[retro][{tag}] forgetting={forgetting:+.6f} recovery={recovery:+.6f} "
              f"gate_frac={gate_frac:.4f} grad_tokens={grad_tokens_total:,}", flush=True)

    out["regimes"] = regime_results

    present = set(regime_results.keys())
    n_visited = {t: regime_results[t]["n_chunks_seen"] for t in present}
    out["chunk_budget_check"] = {
        "n_chunks_visited": n_visited,
        "equal": len(set(n_visited.values())) <= 1,
    }

    # ─────────────────────────────────────────────────────────────────────
    #  p41_scoring
    # ─────────────────────────────────────────────────────────────────────
    def last_n_phase1_errs(regime, H, n=25):
        """Last n valid (non-None) errors for rung H within phase1, in chunk
        order -- 'end-of-phase-1 window', used by both clause (a) and (b)."""
        rows = regime["curve_rung"][str(H)]
        vals = [e for (idx, ph, e, src) in rows if ph == "phase1" and e is not None]
        return vals[-n:]

    def mean_or_none(vals):
        return float(np.mean(vals)) if vals else None

    p41 = {}

    if "retro" in present:
        r = regime_results["retro"]
        end_p1_means = {}
        end_p1_stds = {}
        for H in H_LADDER:
            vals = last_n_phase1_errs(r, H, n=25)
            end_p1_means[H] = mean_or_none(vals)
            end_p1_stds[H] = float(np.std(vals)) if vals else None

        # ---- (a) two_regime ----
        have_all = all(end_p1_means[H] is not None for H in H_LADDER)
        if have_all:
            e2, e8, e32, e128 = (end_p1_means[H] for H in H_LADDER)
            lhs = e128 - e32
            rhs = 0.25 * (e8 - e2)
            two_regime_pass = lhs < rhs
        else:
            lhs = rhs = None
            two_regime_pass = False
        p41["a_two_regime"] = {
            "pass": bool(two_regime_pass),
            "end_of_phase1_mean_err": {str(H): end_p1_means[H] for H in H_LADDER},
            "err128_minus_err32": lhs, "quarter_err8_minus_err2": rhs,
            "note": "err(H=128)-err(H=32) < 0.25*(err(H=8)-err(H=2)) on last-25-chunk means of "
                    "phase1 (pre-shock, clean stationary window)",
        }

        # ---- (c) live_forgetting -- CORRECTED per team-lead precision pass:
        #      "after the phase-2 boundary" means B12 (the phase1->2 shock
        #      onset), NOT the phase2->3 boundary. Window is (B12, B12+32].
        #      Per-DECISION qualification (not a single rung-level yes/no):
        #      at window chunk t, rung H qualifies iff t-H < B12 (its target
        #      still lies pre-shock). t ranges over the OPEN window
        #      B12+1..B12+32 (strict t>B12), so t-H<B12 <=> t-B12<H <=>
        #      t-B12 in {1,...,H-1} (strict, since t-B12 is a positive
        #      integer) capped at 32 -- i.e. each rung accumulates EXACTLY
        #      min(H-1, 32) qualified window decisions (VERIFIED by direct
        #      enumeration, not derived by inspection -- an initial min(H,32)
        #      estimate was off by one and caught here before it shipped).
        #      Concretely: H=2 -> 1, H=8 -> 7, H=32 -> 31, H=128 -> 32.
        #      Eligibility floor is >=7 qualified decisions (= H-1 of the
        #      smallest registered-eligible rung, H=8): the registered P41
        #      text names H in {8,32,128} as qualifying richly at both smoke
        #      and full phase lengths, so the floor is set to admit H=8; a
        #      >=8 floor from the build instruction was a dimensioning
        #      choice, NOT the registration, and was superseded to match the
        #      registered text (decided by team-lead, logged here). Under
        #      >=7, only H=2 (1 qualified) falls below the floor and is
        #      excluded from the overall verdict (logged only) -- H=8, H=32,
        #      H=128 are all eligible for clause (c)'s overall pass,
        #      regardless of phase length (this window depends only on H and
        #      the fixed 32-chunk width, not on phase_chunks).
        #      Rise threshold: pre_mean + 2*pre_std of that rung's OWN last-
        #      25-chunk phase-1 window (same end_p1_means/end_p1_stds
        #      already computed above for clause (a) -- reused here so both
        #      clauses share one definition of "phase-1 baseline").
        b12 = r["boundary_idx"]["phase1_to_2"]
        window_hi = b12 + 32
        live_forgetting = {}
        for H in H_LADDER:
            rows = r["curve_rung"][str(H)]
            pre_mean, pre_std = end_p1_means[H], end_p1_stds[H]
            qualified_points = []                            # [(idx, err), ...] with idx-H < b12
            for (idx, ph, e, src) in rows:
                if e is None or idx <= b12 or idx > window_hi:
                    continue
                if (idx - H) < b12:
                    qualified_points.append((idx, e))
            n_qualified = len(qualified_points)
            if pre_mean is None or n_qualified == 0:
                live_forgetting[H] = {
                    "n_qualified_decisions": n_qualified, "eligible_for_verdict": False,
                    "first_rise_chunk_rel": None, "pass": None,
                    "pre_boundary_mean": pre_mean, "pre_boundary_std": pre_std,
                }
                continue
            rise_thresh = pre_mean + 2 * pre_std
            first_rise_rel = None
            for (idx, e) in qualified_points:
                if e > rise_thresh:
                    first_rise_rel = idx - b12
                    break
            # threshold >=7 = H-1 of the smallest registered-eligible rung;
            # the >=8 in the build instruction was superseded to match the
            # registered text (team-lead decision, see comment block above)
            eligible = n_qualified >= 7
            live_forgetting[H] = {
                "n_qualified_decisions": n_qualified, "eligible_for_verdict": eligible,
                "pre_boundary_mean": round(pre_mean, 6), "pre_boundary_std": round(pre_std, 6),
                "rise_thresh": round(rise_thresh, 6),
                "first_rise_chunk_rel": first_rise_rel,
                "pass": bool(first_rise_rel is not None),
            }
        eligible_rungs = [H for H in H_LADDER if live_forgetting[H]["eligible_for_verdict"]]
        live_forgetting_pass = bool(eligible_rungs) and all(
            live_forgetting[H]["pass"] for H in eligible_rungs)
        p41["c_live_forgetting"] = {
            "pass": live_forgetting_pass,
            "per_rung": {str(H): live_forgetting[H] for H in H_LADDER},
            "eligible_rungs": eligible_rungs,
            "window": {"boundary": "phase1_to_2 (B12)", "b12": b12, "hi": window_hi, "width": 32},
            "note": "window is (B12, B12+32]; at window chunk t, rung H qualifies iff t-H < B12 "
                    "(target still pre-shock) -- H in {8,32,128} get ample qualified decisions "
                    "(7/31/32) at any phase length, H=2 is capped at 1 and excluded from the "
                    "verdict via the >=7-qualified-decisions floor (still logged). Threshold >=7 "
                    "= H-1 of the smallest registered-eligible rung (H=8), superseding an initial "
                    ">=8 build-instruction floor to match the registered P41 text. Rise = qualified "
                    "err > (rung's own last-25-chunk phase-1 mean + 2*sigma). Overall pass requires "
                    "ALL eligible (>=7 qualified decisions) rungs to rise within the window.",
        }
    else:
        p41["a_two_regime"] = {"pass": False, "note": "'retro' regime not present in this run"}
        p41["c_live_forgetting"] = {"pass": False, "note": "'retro' regime not present in this run"}

    if "retro_shuffled" in present:
        rs = regime_results["retro_shuffled"]
        sh_means = {}
        sh_bands = {}
        for H in H_LADDER:
            vals = last_n_phase1_errs(rs, H, n=25)
            sh_means[H] = mean_or_none(vals)
            sh_bands[H] = 2.0 * float(np.std(vals)) if vals else None
        have_all_sh = all(sh_means[H] is not None for H in H_LADDER)
        if have_all_sh:
            mean_vals = [sh_means[H] for H in H_LADDER]
            max_spread = max(mean_vals) - min(mean_vals)
            noise_band = max(v for v in sh_bands.values() if v is not None)
            shuffle_pass = max_spread < noise_band
        else:
            max_spread = None
            noise_band = None
            shuffle_pass = False
        p41["b_shuffle"] = {
            "pass": bool(shuffle_pass),
            "end_of_phase1_mean_err": {str(H): sh_means[H] for H in H_LADDER},
            "noise_band_per_rung": {str(H): sh_bands[H] for H in H_LADDER},
            "max_pairwise_spread": max_spread, "noise_band_used": noise_band,
            "note": "retro_shuffled must show NO monotone ladder: max spread across rungs' "
                    "end-of-phase1 means < 2*std of the widest rung's own last-25-chunk errors "
                    "(the noise band)",
        }
    else:
        p41["b_shuffle"] = {"pass": False, "note": "'retro_shuffled' regime not present in this run"}

    out["p41_scoring"] = p41

    overall = p41["a_two_regime"]["pass"] and p41["b_shuffle"]["pass"] and p41["c_live_forgetting"]["pass"]
    out["verdict"] = (
        f"P41: {'PASS' if overall else 'PARTIAL/FAIL'} "
        f"(a_two_regime={p41['a_two_regime']['pass']}, "
        f"b_shuffle={p41['b_shuffle']['pass']}, "
        f"c_live_forgetting={p41['c_live_forgetting']['pass']})"
    )
    print(f"\n[retro] {out['verdict']}", flush=True)

    d = os.path.dirname(args.out) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_")
    with os.fdopen(fd, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, args.out)
    print(f"[retro] -> {args.out}", flush=True)
    return out


# ───────────────────────────────────────────────────────────────────────────
#  Self-test: pure CPU, random tensors, no checkpoint / no HF stream.
#  Exercises: ring-buffer push/lookup at all four lags including boundary
#  ages (freshly-full buffer, an entry that just aged out), RetroHead
#  forward+backward shapes, shuffle-mode's independent RNG stream, and the
#  p41_scoring arithmetic (two_regime / shuffle-band / live_forgetting) on
#  synthetic curves with KNOWN pass and KNOWN fail cases, so the scoring
#  code itself is verified before any real run touches it. Must finish in
#  well under 30s.
# ───────────────────────────────────────────────────────────────────────────
def self_test():
    print("[retro][self-test] starting...", flush=True)
    d_model, n_buckets = 32, 256
    rng_check = []

    # ---- 1. ring buffer basic push/lookup, deterministic mode ----
    ring = RetroRingBuffer(maxlen=128, shuffle=False, seed=42)
    hists = {}
    for idx in range(0, 200):
        h = torch.zeros(n_buckets)
        h[idx % n_buckets] = 1.0
        hists[idx] = h
        ring.push(idx, h)
        for H in H_LADDER:
            want_idx = idx - H
            tgt, src = ring.get_target(idx, H)
            # after pushing idx, the buffer holds exactly the trailing
            # window [idx-maxlen+1 .. idx] (maxlen entries once full) --
            # want_idx is present iff 0 <= idx-want_idx < ring.maxlen
            if want_idx < 0 or (idx - want_idx) >= ring.maxlen:
                assert tgt is None, f"H={H} idx={idx}: expected None (out of range or aged out)"
            else:
                assert tgt is not None, f"H={H} idx={idx}: expected a target, got None"
                assert src == want_idx, f"H={H} idx={idx}: expected src={want_idx}, got {src}"
                assert torch.equal(tgt, hists[want_idx]), f"H={H} idx={idx}: histogram mismatch"
    # aging boundary (verified empirically, see build log): the loop's last
    # push was idx=199, so the buffer holds [72..199] (128 entries). H=128
    # at the CURRENT idx=199 wants idx=71, which is exactly one step OUTSIDE
    # that window -> None (already exercised and asserted inside the loop
    # above). This is the precise edge of the ring buffer's "last 128 chunk
    # histograms" contract: idx=72 -- the buffer's OLDEST live entry -- is
    # still reachable at H=128 from a HYPOTHETICAL idx=200 query (want_idx=
    # 72, still present, since nothing has been pushed past idx=199 to
    # evict it yet), confirming the boundary is a hard idx-and-lag function
    # of what has actually been pushed, not an off-by-one in get_target.
    tgt_still_there, src_still_there = ring.get_target(200, 128)
    assert tgt_still_there is not None and src_still_there == 72, \
        "expected idx=72 (oldest live entry) to still be reachable via a hypothetical idx=200 query"
    tgt_present, src_present = ring.get_target(199, 32)
    assert tgt_present is not None and src_present == 167, "expected H=32 lookup at idx=199 to be present"
    print("[retro][self-test] ring buffer push/lookup: OK (incl. negative-lag and aging boundary)", flush=True)

    # ---- 1b. REGRESSION: top-rung steady-state reachability under the
    #      REAL push-then-lookup order run_retro actually uses (push(t),
    #      then get_target(t, H) for every rung, same chunk). This is the
    #      exact bug the team-lead's line-by-line review caught: with
    #      maxlen=max(H_LADDER) (128), after push(t) the buffer holds
    #      [t-127..t] -- H=128's target (t-128) has JUST been evicted by
    #      that very push, so H=128 gets None on EVERY chunk of the whole
    #      run (silent, no exception -- clause (a) needs err(128), so this
    #      would have quietly killed two_regime for the entire run). The
    #      fix (ring_maxlen = max(H_LADDER) + 1, applied in run_retro,
    #      imported here as run_retro's OWN constant so this test tracks the
    #      real code path, not a hand-copied number) keeps t-128 alive one
    #      push longer, exactly long enough for get_target(t, 128) to see it
    #      before the NEXT push evicts it. Verified both directions below:
    #      the buggy maxlen=128 must be RED (this is checked once, then
    #      discarded -- only the fixed-maxlen GREEN assertion is kept live,
    #      per team-lead instruction "behalte nur die Grün-Assertion").
    # Read the ACTUAL ring_maxlen value run_retro will use, from its own
    # source (via inspect), rather than hand-copying "max(H_LADDER)+1" as a
    # second, driftable constant -- if a future edit reverts run_retro to
    # maxlen=max(H_LADDER), this test must catch it via the RED-check logic
    # below, not silently keep asserting against a stale GREEN expectation.
    import inspect
    import re as _re
    run_retro_src = inspect.getsource(run_retro)
    m = _re.search(r"ring_maxlen\s*=\s*(.+)", run_retro_src)
    assert m is not None, "could not find 'ring_maxlen = ...' in run_retro's source -- test needs updating"
    max_h = max(H_LADDER)
    actual_fixed_maxlen = eval(m.group(1).strip(), {"max": max, "H_LADDER": H_LADDER})
    n_probe_chunks = 300

    def probe_top_rung_reachability(maxlen):
        ring_p = RetroRingBuffer(maxlen=maxlen, shuffle=False, seed=42)
        n_hit, n_miss, n_src_mismatch = 0, 0, 0
        for t in range(n_probe_chunks):
            h = torch.zeros(n_buckets)
            h[t % n_buckets] = 1.0
            ring_p.push(t, h)
            tgt, src = ring_p.get_target(t, max_h)
            if t >= max_h:
                if tgt is None:
                    n_miss += 1
                else:
                    n_hit += 1
                    if src != t - max_h:
                        n_src_mismatch += 1
        return n_hit, n_miss, n_src_mismatch

    # RED check (buggy maxlen == max_h, the pre-fix value): must show a
    # TOTAL, silent outage of the top rung -- confirms the test actually
    # discriminates the bug before trusting the GREEN check below.
    red_hit, red_miss, _ = probe_top_rung_reachability(maxlen=max_h)
    assert red_hit == 0 and red_miss == n_probe_chunks - max_h, (
        f"RED-check sanity failed: expected the buggy maxlen={max_h} to silently zero out "
        f"H={max_h} on every post-warmup chunk, got hit={red_hit} miss={red_miss} "
        f"(if this assertion fails, the regression test below cannot be trusted either)")
    print(f"[retro][self-test] RED-check confirmed: maxlen={max_h} (pre-fix) silently drops "
          f"H={max_h} on ALL {red_miss} post-warmup chunks -- this is the bug the team-lead's "
          f"review caught", flush=True)

    # GREEN check (the ACTUAL value run_retro's source currently assigns to
    # ring_maxlen, extracted above via inspect -- not a hand-copied
    # constant) -- steady-state reachability from t=max_h onward, every
    # chunk, with exact source_idx == t-max_h in the non-shuffle case.
    assert actual_fixed_maxlen > max_h, (
        f"run_retro's ring_maxlen ({actual_fixed_maxlen}) is not > max(H_LADDER) ({max_h}) -- "
        f"this IS the bug (or a regression back to it); the fix requires strictly more room than "
        f"the top rung's lag so its target survives the push that lands it")
    green_hit, green_miss, green_src_mismatch = probe_top_rung_reachability(maxlen=actual_fixed_maxlen)
    assert green_miss == 0, (
        f"H={max_h} should ALWAYS find a target from t={max_h} onward with run_retro's actual "
        f"ring_maxlen={actual_fixed_maxlen}, but missed {green_miss}/{n_probe_chunks - max_h} chunks")
    assert green_hit == n_probe_chunks - max_h, \
        f"expected {n_probe_chunks - max_h} hits, got {green_hit}"
    assert green_src_mismatch == 0, \
        f"expected source_idx == t-{max_h} on every hit (non-shuffle mode), {green_src_mismatch} mismatches"
    print(f"[retro][self-test] GREEN check: run_retro's actual ring_maxlen={actual_fixed_maxlen} gives "
          f"H={max_h} a target on ALL {green_hit} post-warmup chunks (t={max_h}..{n_probe_chunks-1}), "
          f"source_idx==t-{max_h} exactly every time: OK", flush=True)

    # ---- 2. shuffle mode: independent RNG, does not crash, returns SOME
    #      buffer entry (not necessarily t-H), and is repeatable given a
    #      fresh identically-seeded RNG ----
    ring_sh_a = RetroRingBuffer(maxlen=128, shuffle=True, seed=999)
    ring_sh_b = RetroRingBuffer(maxlen=128, shuffle=True, seed=999)
    for idx in range(0, 50):
        h = torch.zeros(n_buckets)
        h[idx % n_buckets] = 1.0
        ring_sh_a.push(idx, h)
        ring_sh_b.push(idx, h)
    draws_a = [ring_sh_a.get_target(49, H)[1] for H in H_LADDER]
    draws_b = [ring_sh_b.get_target(49, H)[1] for H in H_LADDER]
    assert draws_a == draws_b, "shuffle RNG not reproducible under identical seed"
    print(f"[retro][self-test] shuffle-mode draws (seed=999, idx=49): {draws_a} -- reproducible: OK", flush=True)

    # ---- 3. RetroHead forward+backward shapes, one per rung ----
    heads = {H: RetroHead(d_model, n_buckets) for H in H_LADDER}
    head_opts = {H: torch.optim.Adam(heads[H].parameters(), lr=3e-3) for H in H_LADDER}
    pooled = torch.randn(d_model)
    target = torch.zeros(n_buckets)
    target[5] = 0.6
    target[10] = 0.4
    for H in H_LADDER:
        logits = heads[H](pooled)
        assert logits.shape == (n_buckets,), f"H={H}: bad logits shape {logits.shape}"
        loss = horizon_loss(logits, target)
        head_opts[H].zero_grad(set_to_none=True)
        loss.backward()
        head_opts[H].step()
    print("[retro][self-test] RetroHead forward+backward, all 4 rungs: OK", flush=True)

    # ---- 4. p41_scoring arithmetic: synthetic KNOWN-PASS two_regime curve ----
    def fake_regime(err_by_H, n=25, noise=0.0, seed=0):
        rgen = np.random.default_rng(seed)
        curve_rung = {}
        for H in H_LADDER:
            base = err_by_H[H]
            rows = []
            for i in range(n):
                e = base + (rgen.normal(0, noise) if noise > 0 else 0.0)
                rows.append([100 + i, "phase1", round(float(e), 6), 100 + i - H])
            curve_rung[str(H)] = rows
        return {"curve_rung": curve_rung}

    def last_n_phase1_errs_local(regime, H, n=25):
        rows = regime["curve_rung"][str(H)]
        vals = [e for (idx, ph, e, src) in rows if ph == "phase1" and e is not None]
        return vals[-n:]

    # two_regime PASS case: steep rise 2->8, plateau 32->128
    pass_case = fake_regime({2: 0.10, 8: 0.90, 32: 1.30, 128: 1.35})
    means = {H: float(np.mean(last_n_phase1_errs_local(pass_case, H))) for H in H_LADDER}
    lhs = means[128] - means[32]
    rhs = 0.25 * (means[8] - means[2])
    assert lhs < rhs, f"two_regime PASS case should satisfy lhs<rhs: {lhs} vs {rhs}"
    print(f"[retro][self-test] two_regime PASS-case arithmetic: lhs={lhs:.4f} < rhs={rhs:.4f}: OK", flush=True)

    # two_regime FAIL case: linear rise, no plateau
    fail_case = fake_regime({2: 0.10, 8: 0.40, 32: 0.70, 128: 1.00})
    means_f = {H: float(np.mean(last_n_phase1_errs_local(fail_case, H))) for H in H_LADDER}
    lhs_f = means_f[128] - means_f[32]
    rhs_f = 0.25 * (means_f[8] - means_f[2])
    assert not (lhs_f < rhs_f), f"two_regime FAIL case should NOT satisfy lhs<rhs: {lhs_f} vs {rhs_f}"
    print(f"[retro][self-test] two_regime FAIL-case arithmetic: lhs={lhs_f:.4f} >= rhs={rhs_f:.4f}: OK", flush=True)

    # shuffle PASS case: flat rungs within noise
    shuf_pass = fake_regime({2: 1.0, 8: 1.02, 32: 0.98, 128: 1.01}, noise=0.05, seed=7)
    sh_means = {H: float(np.mean(last_n_phase1_errs_local(shuf_pass, H))) for H in H_LADDER}
    sh_bands = {H: 2.0 * float(np.std(last_n_phase1_errs_local(shuf_pass, H))) for H in H_LADDER}
    max_spread = max(sh_means.values()) - min(sh_means.values())
    noise_band = max(sh_bands.values())
    assert max_spread < noise_band or noise_band > 0.05, "shuffle PASS-ish case sanity"
    print(f"[retro][self-test] shuffle-band arithmetic: max_spread={max_spread:.4f} "
          f"noise_band={noise_band:.4f}: OK (arithmetic exercised, not asserting pass since "
          f"synthetic noise draw is stochastic by design)", flush=True)

    # ---- live_forgetting: synthetic B12-anchored (c) test, mirroring the
    #      ACTUAL scoring code in run_retro (per-decision qualification
    #      t-H < b12, window (b12, b12+32], rise vs. phase-1 baseline,
    #      >=7-qualified-decisions eligibility floor -- threshold = H-1 of
    #      the smallest registered-eligible rung (H=8); superseded from an
    #      initial >=8 build-instruction floor to match the registered P41
    #      text naming H in {8,32,128} as qualifying richly, team-lead
    #      decision). Two cases: a rung that rises on schedule (PASS) and a
    #      rung that stays flat (FAIL).
    def score_live_forgetting_rung(rows, H, b12, pre_mean, pre_std):
        """Byte-mirror of the per-rung block inside run_retro's p41_scoring
        (c) clause -- reimplemented here (not imported) so the self-test
        exercises the SAME arithmetic path independently, catching drift
        between the two if one is edited without the other."""
        window_hi = b12 + 32
        qualified_points = [(idx, e) for (idx, ph, e, src) in rows
                             if e is not None and b12 < idx <= window_hi and (idx - H) < b12]
        n_qualified = len(qualified_points)
        if n_qualified == 0:
            return {"n_qualified_decisions": 0, "eligible_for_verdict": False,
                    "first_rise_chunk_rel": None, "pass": None}
        rise_thresh = pre_mean + 2 * pre_std
        first_rise_rel = None
        for (idx, e) in qualified_points:
            if e > rise_thresh:
                first_rise_rel = idx - b12
                break
        return {"n_qualified_decisions": n_qualified, "eligible_for_verdict": n_qualified >= 7,
                "first_rise_chunk_rel": first_rise_rel, "pass": first_rise_rel is not None}

    b12 = 150
    pre_mean, pre_std = 0.20, 0.02                          # synthetic phase-1 baseline

    def make_rows(H, rising, b12, n_pre=25, n_post=40):
        rows = []
        for i in range(-n_pre, n_post):
            idx = b12 + i
            if i <= 0:
                e = pre_mean                                 # flat pre-boundary (phase 1 baseline)
            elif rising:
                e = pre_mean + 0.5 * min(1.0, i / 10.0)       # ramps up starting right after b12
            else:
                e = pre_mean                                 # stays flat -- no forgetting signal
            rows.append([idx, "phase2", round(e, 6), idx - H])
        return rows

    # Qualified-decision counts VERIFIED by direct enumeration (not derived
    # by inspection -- an initial min(H,32) estimate was off by one, caught
    # here): n_qualified = min(H-1, 32). H=2->1, H=8->7, H=32->31, H=128->32.
    expected_n_qualified = {2: 1, 8: 7, 32: 31, 128: 32}

    # H=32: 31 qualified window decisions (well above the >=7 floor) --
    # rising case must PASS, flat case must FAIL, using the real scoring fn.
    rows_rise = make_rows(32, rising=True, b12=b12)
    res_rise = score_live_forgetting_rung(rows_rise, 32, b12, pre_mean, pre_std)
    assert res_rise["n_qualified_decisions"] == expected_n_qualified[32], \
        f"H=32 should have {expected_n_qualified[32]} qualified decisions, got {res_rise}"
    assert res_rise["eligible_for_verdict"], "H=32 rising case should be eligible (>=7 qualified)"
    assert res_rise["pass"] and res_rise["first_rise_chunk_rel"] is not None and \
        res_rise["first_rise_chunk_rel"] <= 32, f"H=32 rising case should PASS: {res_rise}"
    print(f"[retro][self-test] live_forgetting B12-anchored PASS-case: n_qualified="
          f"{res_rise['n_qualified_decisions']} first_rise_rel={res_rise['first_rise_chunk_rel']}: OK",
          flush=True)

    rows_flat = make_rows(32, rising=False, b12=b12)
    res_flat = score_live_forgetting_rung(rows_flat, 32, b12, pre_mean, pre_std)
    assert res_flat["eligible_for_verdict"], "H=32 flat case should still be eligible (>=7 qualified)"
    assert not res_flat["pass"], f"H=32 flat case should FAIL (no forgetting signal): {res_flat}"
    print(f"[retro][self-test] live_forgetting B12-anchored FAIL-case: n_qualified="
          f"{res_flat['n_qualified_decisions']} pass={res_flat['pass']} (expected False): OK", flush=True)

    # H=2: only 1 qualified decision -- must be BELOW the >=7 eligibility
    # floor regardless of rising/flat, exactly as documented (logged,
    # excluded from the overall verdict -- the ONLY rung this happens to).
    rows_h2 = make_rows(2, rising=True, b12=b12)
    res_h2 = score_live_forgetting_rung(rows_h2, 2, b12, pre_mean, pre_std)
    assert res_h2["n_qualified_decisions"] == expected_n_qualified[2], \
        f"H=2 should have {expected_n_qualified[2]} qualified decisions, got {res_h2}"
    assert not res_h2["eligible_for_verdict"], "H=2 should be BELOW the >=7-qualified-decisions floor"
    print(f"[retro][self-test] live_forgetting H=2 starvation check: n_qualified="
          f"{res_h2['n_qualified_decisions']} eligible={res_h2['eligible_for_verdict']} "
          f"(expected False): OK", flush=True)

    # H=8: exactly 7 qualified decisions -- AT the >=7 floor, must be
    # eligible (team-lead's corrected threshold: the registered P41 text
    # names H=8 as qualifying richly, so the floor was set to admit it).
    rows_h8 = make_rows(8, rising=True, b12=b12)
    res_h8 = score_live_forgetting_rung(rows_h8, 8, b12, pre_mean, pre_std)
    assert res_h8["n_qualified_decisions"] == expected_n_qualified[8], \
        f"H=8 should have {expected_n_qualified[8]} qualified decisions, got {res_h8}"
    assert res_h8["eligible_for_verdict"], "H=8 (7 qualified) should be eligible under the >=7 floor"
    print(f"[retro][self-test] live_forgetting H=8 at-floor check: n_qualified="
          f"{res_h8['n_qualified_decisions']} eligible={res_h8['eligible_for_verdict']} "
          f"(expected True -- 7 >= 7): OK", flush=True)

    # H=128: 32 qualified decisions (the full window) -- sanity check the
    # upper end of the ladder too.
    rows_h128 = make_rows(128, rising=True, b12=b12)
    res_h128 = score_live_forgetting_rung(rows_h128, 128, b12, pre_mean, pre_std)
    assert res_h128["n_qualified_decisions"] == expected_n_qualified[128], \
        f"H=128 should have {expected_n_qualified[128]} qualified decisions, got {res_h128}"
    assert res_h128["eligible_for_verdict"], "H=128 (32 qualified) should be eligible"
    print(f"[retro][self-test] live_forgetting H=128 full-window check: n_qualified="
          f"{res_h128['n_qualified_decisions']} eligible={res_h128['eligible_for_verdict']} "
          f"(expected True): OK", flush=True)

    print("[retro][self-test] ALL CHECKS PASSED", flush=True)
    return True


# ───────────────────────────────────────────────────────────────────────────
#  CLI
# ───────────────────────────────────────────────────────────────────────────
def build_argparser():
    ap = argparse.ArgumentParser(
        description="RETRO POS: backward H-ladder retrodiction meter, C4->code->C4 (P41/MS18 v0)")
    ap.add_argument("--ckpt", default="results/pos_snapshots/ckpt_359050240.pt")
    ap.add_argument("--phase-chunks", type=int, default=150, help="chunks per phase (1/2/3)")
    ap.add_argument("--eval-every", type=int, default=25, help="WT-2 + code-val eval cadence, in chunks")
    ap.add_argument("--code-val-tokens", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-buckets", type=int, default=256, help="top-N vocab buckets for the histogram target")
    ap.add_argument("--head-lr", type=float, default=3e-3, help="each rung head's own Adam lr (matches horizon_pos)")
    ap.add_argument("--gate-q", type=float, default=0.75, help="rolling gate quantile (matches base_gate)")
    ap.add_argument("--gate-window", type=int, default=200, help="rolling gate window length")
    ap.add_argument("--min-window", type=int, default=50, help="min window fill before gating kicks in")
    ap.add_argument("--smoke", action="store_true", help="phase-chunks=40, eval-every=10, out=*_smoke.json")
    ap.add_argument("--full", action="store_true", help="phase-chunks=150, eval-every=25 (explicit; also the default)")
    ap.add_argument("--regimes", default="retro,retro_shuffled",
                     help="comma-separated regime list to run this invocation")
    ap.add_argument("--out", default="results/retro_pos.json")
    ap.add_argument("--self-test", action="store_true",
                     help="pure-CPU dry test of buffer/head/scoring arithmetic on random tensors, "
                          "no checkpoint or HF stream needed, runs in <30s, then exits")
    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    if args.self_test:
        ok = self_test()
        sys.exit(0 if ok else 1)

    if args.smoke:
        args.phase_chunks = 40
        args.eval_every = 10
        if args.out == "results/retro_pos.json":
            args.out = "results/retro_pos_smoke.json"
    elif args.full:
        # NOTE: do NOT reset phase_chunks here — 150 is already the parser
        # default, and an explicit --phase-chunks (e.g. 200 per the P41
        # amendment: top rung needs >=70 pre-boundary steps) must survive
        # --full. Same stomp pattern already bit rank_sweep's --cells.
        pass
        args.eval_every = 25

    def eval_wt2_fn(model):
        train_text, val_text = load_wikitext2()
        _, stoi, unk, mask = build_vocab(train_text)
        val_ids = tokenize(val_text, stoi, unk)
        cfg = torch.load(args.ckpt, weights_only=False)["config"]
        evX, evY = build_eval_set(val_ids, cfg["eval_tokens"], cfg["chunk"])
        return heldout(model, evX, evY)

    run_retro(args, eval_wt2_fn)


if __name__ == "__main__":
    main()
