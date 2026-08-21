#!/usr/bin/env python3 -u
"""
POS AUTO-Q — curiosity homeostasis: a self-regulated gating quantile (P28).
=============================================================================================
pos_domain_shock.py's R2 arm gates on a FIXED rolling quantile (q=0.75). That is a
static thermostat setting: it works at whatever gate rate the live stream happens
to produce, but under a domain shock (measured in pos_domain_shock's R2: gate_frac
0.58 during the CODE phase, more than double the ~0.25 the live 40h run settles
into) a fixed q lets the shock blow the gate wide open — everything looks
surprising, so everything gets gradient, so the organism forgets more, not less.

P28's claim: the organism doesn't need a fixed threshold, it needs a REGULATED
one. A homeostat that watches its OWN gate rate and adjusts q to hold a target
(~25%) turns "surprising" into a moving target that tracks the stream's current
volatility, not an absolute NLL level fixed at snapshot time. This file builds
that regulator (auto) and races it against the fixed-q recipe (fixed) on the
IDENTICAL 3-phase C4->code->C4 shock stream and snapshot pos_domain_shock.py
uses — same sources, same feeder, same eval cadence, same forgetting/
plasticity/recovery definitions, so the P28 verdict is a clean two-arm
extension of P20/P24's harness, not a new experiment design.

Homeostat (P-controller, deliberately simple — a P-term is sufficient to show
the mechanism; documenting the constants here rather than tuning further):
  r*      = 0.25         target gate rate
  r_hat   = EMA(gated ? 1 : 0), half-life ~50 chunks -> alpha = 1 - 2**(-1/50)
  q      <- clip(q + k * (r_hat - r*), 0.5, 0.95),  k = 0.5, applied once per chunk
  q0      = 0.75          (same starting point as R2's fixed q, so phase 1 opens
                           with both arms behaviorally identical before the
                           controller has had time to move)
Read r_hat > r* (gating MORE than target) -> q rises -> harder to clear -> gate
rate falls back toward target. Symmetric the other way. This is exactly a
leaky-integrator thermostat on the gate indicator.

Two arms, same snapshot start, same phase-length/eval-cadence stream position
per arm (own cursor each, byte-identical phase transitions and chunk counts,
mirroring pos_domain_shock.py's regime symmetry):

  fixed   R2's exact recipe: rolling-q75 gate, window=200, min_window=50,
          ignition OFF (warm snapshot start).
  auto    identical gating MECHANICS (rolling-window quantile threshold,
          forward always runs, backward iff this chunk's surprise clears the
          CURRENT threshold), but q is homeostat-controlled per the above
          instead of fixed at 0.75. Full q and r_hat trajectory logged per
          chunk (global index, phase, q, r_hat, gated).

Fairness (P28b needs a MATCHED total gradient tokens comparison): the two arms
gate at different rates by construction (that IS the mechanism under test), so
grad_tokens_total is logged per arm and the P28b forgetting/plasticity check is
reported BOTH ways — absolute (raw curves) AND per-gradient-token (normalized
by each arm's own grad_tokens_total) — no result is hidden by picking only the
flattering axis.

Measurement: identical to pos_domain_shock.py (same eval_every cadence, same
WT-2 heldout + fixed CODE-val slice, same forgetting/plasticity/recovery
definitions):
  forgetting  = max WT-2 heldout INCREASE during phase 2 vs. pre-phase-2 WT-2
  plasticity  = CODE-val heldout DROP during phase 2 (pre-phase-2 minus phase-2 min)
  recovery    = WT-2 heldout at END of phase 3 vs. pre-phase-2 WT-2

P28 scoring (analysis/PREDICTIONS.md):
  (a) auto holds gate_frac in [0.18, 0.32], PER PHASE, in all three phases,
      while fixed overshoots in phase 2 (R2's own measured phase-2 rate was 0.58).
  (b) auto forgetting <= fixed forgetting, AND auto plasticity >= 0.9x fixed
      plasticity — both checked absolute and per-gradient-token.
  (c) q trajectory rises during phase 2 (median q phase2 > median q phase1 + 0.03)
      and relaxes back down in phase 3 (median q phase3 < median q phase2).
Verdict is computed directly from the measured numbers, not asserted.

This file is READ-ONLY with respect to the live run's outputs (pos_ckpt.pt,
pos_index.jsonl, pos_status.json, ...) and with respect to pos_domain_shock's
own in-flight outputs (results/pos_domain_shock*.json/.log — a full run of
that harness may be running concurrently; this file never opens those paths).
All outputs go to results/pos_auto_q*.json.

Usage:
  python src/pos_auto_q.py --smoke   # phase_chunks=40, eval_every=10
  python src/pos_auto_q.py --full    # phase_chunks=150, eval_every=25
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
except PermissionError:
    pass  # already niced by the launcher (macOS EPERM on re-nice)

import numpy as np
import torch
import torch.nn.functional as F

torch.backends.mps.is_available = lambda: False          # force CPU (same as the live run)
torch.set_num_threads(1)

from streaming_train import StreamingNoPELM
from length_extrap_v2 import load_wikitext2, build_vocab, tokenize
from pos_run import build_eval_set, heldout                    # safe: pos_run's main() only runs under __main__
from pos_sleep import ChunkFeeder, C4ValStream, load_snapshot, _real_vocab

# CodeStream / make_phase1_source / make_phase2_source are byte-identical to
# pos_domain_shock.py's — redefined here (not imported) so this file never
# imports pos_domain_shock (which would tie import-time behavior to whatever
# is currently mid-flight in that module's own __main__ run).


# ───────────────────────────────────────────────────────────────────────────
#  Phase-2 CODE stream — codeparrot/github-code-clean, client-side filtered to
#  Python rows, tokenized with the WT-2 word vocabulary (heavy unk on code IS
#  the shock). Identical to pos_domain_shock.py's CodeStream.
# ───────────────────────────────────────────────────────────────────────────
class CodeStream:
    def __init__(self, stoi, unk, block=65536):
        from datasets import load_dataset
        self.stoi, self.unk = stoi, unk
        ds = load_dataset("codeparrot/github-code-clean", streaming=True, split="train",
                          trust_remote_code=True)
        self._it = iter(ds)
        self.pending = []
        self.block = block
        self.n_tokens_seen = 0
        self.n_unk_seen = 0
        self.rows_seen = 0
        self.rows_kept = 0

    def _refill(self):
        while len(self.pending) < self.block:
            try:
                row = next(self._it)
            except StopIteration:
                from datasets import load_dataset
                ds = load_dataset("codeparrot/github-code-clean", streaming=True, split="train",
                                  trust_remote_code=True)
                self._it = iter(ds)
                continue
            self.rows_seen += 1
            if row.get("language") != "Python":
                continue
            self.rows_kept += 1
            code = row.get("code", "") or ""
            if not code.strip():
                continue
            toks = tokenize(code, self.stoi, self.unk)
            self.n_unk_seen += sum(1 for t in toks if t == self.unk)
            self.n_tokens_seen += len(toks)
            self.pending.extend(toks)

    def next_block(self, n):
        while len(self.pending) < n:
            self._refill()
        out, self.pending = self.pending[:n], self.pending[n:]
        return out

    def unk_rate(self):
        return self.n_unk_seen / max(1, self.n_tokens_seen)


def make_phase1_source(stoi, unk):
    """Phase 1 / Phase 3 shared source: C4 train, 5M docs ahead of the live run's
    reach (pos_sleep_cycles.py's "train-far" recipe, same as pos_domain_shock.py)."""
    return C4ValStream(stoi, unk, split="train", skip_docs=5_000_000)


def make_phase2_source(stoi, unk):
    return CodeStream(stoi, unk)


# ───────────────────────────────────────────────────────────────────────────
#  One gradient step / one forward-only step — identical to pos_domain_shock.py
# ───────────────────────────────────────────────────────────────────────────
def grad_step(model, opt, x, y, states, clip=5.0):
    logits, st = model(x, states)
    nll_flat = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                               reduction="none")
    loss = nll_flat.mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step()
    new_states = [s.detach() for s in st]
    return new_states, x.numel(), float(loss)


def nograd_step(model, x, y, states):
    with torch.no_grad():
        logits, st = model(x, states)
        nll_flat = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                   reduction="none")
    return st, float(nll_flat.mean())


# ───────────────────────────────────────────────────────────────────────────
#  Homeostat: P-controller on the gate rate EMA.
# ───────────────────────────────────────────────────────────────────────────
class GateHomeostat:
    """Regulates q to hold a target gate rate r_star. r_hat is an EMA of the
    gated indicator (0/1 per chunk) with half-life `halflife_chunks`. q is
    updated once per chunk by a P-controller: q += k * (r_hat - r_star),
    clipped to [q_min, q_max]. q starts at q0 (matches the fixed arm's q, so
    both arms are behaviorally identical at t=0)."""

    def __init__(self, r_star=0.25, halflife_chunks=50.0, k=0.5, q0=0.75,
                q_min=0.5, q_max=0.95):
        self.r_star = r_star
        self.alpha = 1.0 - 2.0 ** (-1.0 / halflife_chunks)   # EMA update weight for the half-life
        self.k = k
        self.q = q0
        self.q_min, self.q_max = q_min, q_max
        self.r_hat = r_star                                   # neutral start: no bias before data arrives

    def update(self, gated, forced=False):
        # Anti-windup (P28 smoke finding): while gating is FORCED (window not
        # yet at min_window), the gated-indicator carries no information about
        # the policy -- integrating it drives q to the clip and it never
        # recovers. Freeze both r_hat and q until gating is free.
        if forced:
            return self.q, self.r_hat
        self.r_hat = (1 - self.alpha) * self.r_hat + self.alpha * (1.0 if gated else 0.0)
        self.q = float(np.clip(self.q + self.k * (self.r_hat - self.r_star), self.q_min, self.q_max))
        return self.q, self.r_hat


# ───────────────────────────────────────────────────────────────────────────
#  Regime drivers — both arms share the SAME gating mechanics (rolling-window
#  quantile threshold, forward always runs, backward iff surprise clears the
#  CURRENT threshold); only how q is obtained differs (fixed vs. homeostat).
# ───────────────────────────────────────────────────────────────────────────
def run_gated_chunk(model, opt, feeder, states, window, q, min_window):
    """One gated chunk at a GIVEN q (caller supplies q, fixed or homeostat-derived).
    Returns (states, grad_tokens, gated, surprise_s)."""
    x, y = feeder.next_xy()
    st_ng, s = nograd_step(model, x, y, states)
    if len(window) >= min_window:
        thresh = float(np.quantile(np.fromiter(window, dtype=np.float64), q))
        gated = s > thresh
    else:
        gated = True                                          # not enough window yet: learn
    if gated:
        states, gt, _ = grad_step(model, opt, x, y, states)
    else:
        states, gt = st_ng, 0
    window.append(s)
    return states, gt, gated, s


# ───────────────────────────────────────────────────────────────────────────
#  Main measurement
# ───────────────────────────────────────────────────────────────────────────
def run_auto_q(args, eval_wt2_fn, vocab_fn=_real_vocab):
    ck, cfg, base_model, opt_sd, stoi, unk, mask, V = load_snapshot(args.ckpt, vocab_fn)
    B, K = cfg["batch"], cfg["chunk"]
    lr = cfg["lr"]

    phase_chunks = args.phase_chunks
    eval_every = args.eval_every

    print(f"[autoq] ckpt n_streamed={ck['n_streamed']:,} | phase_chunks={phase_chunks} "
          f"eval_every={eval_every} | B={B} K={K} lr={lr}", flush=True)
    print(f"[autoq] homeostat: r_star={args.r_star} halflife={args.halflife} chunks "
          f"k={args.k} q0={args.q0} clip=[{args.q_min},{args.q_max}]", flush=True)

    # ── fixed CODE-val slice: independent CodeStream instance, read BEFORE any
    #    arm's Phase-2 stream is touched (disjoint from every arm's training
    #    data by construction — matches pos_domain_shock.py's pattern). ──
    code_val_src = CodeStream(stoi, unk)
    code_val_ids = code_val_src.next_block(args.code_val_tokens)
    code_val_unk_rate = code_val_src.unk_rate()
    cvX, cvY = build_eval_set(code_val_ids, len(code_val_ids) - 1, K)
    print(f"[autoq] code-val slice: {cvY.numel():,} tokens, unk_rate={code_val_unk_rate:.4f}", flush=True)

    def eval_code_fn(model):
        return heldout(model, cvX, cvY)

    def fresh_model_opt():
        m = copy.deepcopy(base_model)
        o = torch.optim.Adam(m.parameters(), lr=lr)
        if opt_sd is not None:
            o.load_state_dict(opt_sd)                          # warm Adam moments
            for g in o.param_groups:
                g["lr"] = lr
        return m, o

    base_wt2 = eval_wt2_fn(base_model)
    print(f"[autoq] base_heldout_wt2={base_wt2:.6f}", flush=True)

    out = {
        "ckpt_n_streamed": ck["n_streamed"],
        "phase2_dataset": "codeparrot/github-code-clean (language==Python, client-filtered)",
        "budget": {"phase_chunks": phase_chunks, "eval_every": eval_every,
                  "gate_window": args.gate_window, "min_window": args.min_window,
                  "fixed_q": args.fixed_q},
        "homeostat": {"r_star": args.r_star, "halflife_chunks": args.halflife,
                     "k": args.k, "q0": args.q0, "q_min": args.q_min, "q_max": args.q_max},
        "code_val_tokens": cvY.numel(),
        "code_val_unk_rate": round(code_val_unk_rate, 6),
        "base_heldout_wt2": round(base_wt2, 6),
        "arms": {},
    }

    arm_results = {}
    for tag in ("fixed", "auto"):
        print(f"\n[autoq] ===== arm {tag} =====", flush=True)
        model, opt = fresh_model_opt()

        # Independently instantiated but identically-parameterized sources per
        # arm (byte-identical HF stream order at matching call sites — the
        # pos_sleep_cycles.py "shared stream, instantiate twice" pattern).
        phase13_src = make_phase1_source(stoi, unk)
        phase2_src = make_phase2_source(stoi, unk)
        feeder13 = ChunkFeeder(phase13_src, B, K)
        feeder2 = ChunkFeeder(phase2_src, B, K)

        states = None
        gate_window = deque(maxlen=args.gate_window)
        homeostat = GateHomeostat(r_star=args.r_star, halflife_chunks=args.halflife,
                                  k=args.k, q0=args.q0, q_min=args.q_min, q_max=args.q_max)

        curve_wt2 = []                                          # [(global_chunk_idx, phase, heldout_wt2)]
        curve_code = []                                         # [(global_chunk_idx, phase, heldout_code)]
        grad_tokens_total = 0
        gate_log = []                                           # [(global_idx, phase, gated, q_used, r_hat)]

        def record(global_idx, phase):
            hl_wt2 = eval_wt2_fn(model)
            hl_code = eval_code_fn(model)
            curve_wt2.append([global_idx, phase, round(hl_wt2, 6)])
            curve_code.append([global_idx, phase, round(hl_code, 6)])
            print(f"[autoq][{tag}] phase={phase:<7} chunk={global_idx:>4} "
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
                    if tag == "fixed":
                        q_used = args.fixed_q
                    else:
                        q_used = homeostat.q                     # q as it stood BEFORE this chunk's gate decision
                    states, gt, gated, s = run_gated_chunk(
                        model, opt, feeder, states, gate_window, q_used, args.min_window)
                    grad_tokens_total += gt
                    if tag == "auto":
                        q_after, r_hat_after = homeostat.update(
                            gated, forced=len(gate_window) < args.min_window)
                    else:
                        r_hat_after = None
                    global_idx += 1
                    gate_log.append([global_idx, phase_name, bool(gated), round(q_used, 6),
                                     round(r_hat_after, 6) if r_hat_after is not None else None])
                    done += 1
                record(global_idx, phase_name)
            return done

        # ── Phase 1: C4 (train-far) ─────────────────────────────────────────
        run_wake_block(feeder13, phase_chunks, "phase1")
        pre_phase2_wt2 = curve_wt2[-1][2]
        pre_phase2_code = curve_code[-1][2]

        # ── Phase 2: CODE, then Phase 3: C4 resumed ─────────────────────────
        run_wake_block(feeder2, phase_chunks, "phase2")
        run_wake_block(feeder13, phase_chunks, "phase3")

        wt2_values_phase2 = [v for idx, ph, v in curve_wt2 if ph == "phase2"]
        code_values_phase2 = [v for idx, ph, v in curve_code if ph == "phase2"]
        post_phase3_wt2 = curve_wt2[-1][2]

        forgetting = round(max(wt2_values_phase2 + [pre_phase2_wt2]) - pre_phase2_wt2, 6)
        plasticity = round(pre_phase2_code - min(code_values_phase2 + [pre_phase2_code]), 6)
        recovery = round(post_phase3_wt2 - pre_phase2_wt2, 6)

        # per-phase gate_frac (P28a needs each phase checked separately)
        gate_frac_by_phase = {}
        for ph in ("phase1", "phase2", "phase3"):
            gs = [1 if g else 0 for _, p, g, _, _ in gate_log if p == ph]
            gate_frac_by_phase[ph] = round(sum(gs) / max(1, len(gs)), 4)
        gate_frac_overall = round(sum(1 if g else 0 for _, _, g, _, _ in gate_log) / max(1, len(gate_log)), 4)

        # q trajectory summary per phase (auto only meaningful; fixed is constant)
        q_by_phase = {}
        for ph in ("phase1", "phase2", "phase3"):
            qs = [q for _, p, _, q, _ in gate_log if p == ph]
            if qs:
                q_by_phase[ph] = {"min": round(min(qs), 6), "median": round(float(np.median(qs)), 6),
                                  "max": round(max(qs), 6)}

        arm_results[tag] = {
            "curve_wt2": curve_wt2, "curve_code": curve_code,
            "grad_tokens_total": grad_tokens_total,
            "n_chunks_gated": sum(1 for _, _, g, _, _ in gate_log if g),
            "n_chunks_seen": len(gate_log),
            "gate_frac_overall": gate_frac_overall,
            "gate_frac_by_phase": gate_frac_by_phase,
            "q_by_phase": q_by_phase,
            "pre_phase2_wt2": round(pre_phase2_wt2, 6), "pre_phase2_code": round(pre_phase2_code, 6),
            "post_phase3_wt2": round(post_phase3_wt2, 6),
            "forgetting": forgetting, "plasticity": plasticity, "recovery": recovery,
            "gate_trajectory": gate_log,
        }
        print(f"[autoq][{tag}] forgetting={forgetting:+.6f} plasticity={plasticity:+.6f} "
              f"recovery={recovery:+.6f} gate_frac={gate_frac_overall:.4f} "
              f"(by phase: {gate_frac_by_phase}) grad_tokens={grad_tokens_total:,}", flush=True)
        if tag == "auto":
            print(f"[autoq][auto] q_by_phase={q_by_phase}", flush=True)

    out["arms"] = arm_results

    # ── P28 scoring ──────────────────────────────────────────────────────────
    fx, au = arm_results["fixed"], arm_results["auto"]

    # (a) auto holds gate_frac in [0.18, 0.32] in ALL three phases
    p28a_bounds = {ph: (0.18 <= au["gate_frac_by_phase"][ph] <= 0.32) for ph in ("phase1", "phase2", "phase3")}
    p28a = all(p28a_bounds.values())

    # (b) auto forgetting <= fixed forgetting AND auto plasticity >= 0.9x fixed
    #     plasticity — absolute AND per-gradient-token.
    fx_gt, au_gt = fx["grad_tokens_total"], au["grad_tokens_total"]
    fx_forget_pgt = fx["forgetting"] / max(1, fx_gt) * 1e6      # per-million-gradient-tokens, for readability
    au_forget_pgt = au["forgetting"] / max(1, au_gt) * 1e6
    fx_plast_pgt = fx["plasticity"] / max(1, fx_gt) * 1e6
    au_plast_pgt = au["plasticity"] / max(1, au_gt) * 1e6

    p28b_forget_abs = au["forgetting"] <= fx["forgetting"]
    p28b_forget_pgt = au_forget_pgt <= fx_forget_pgt
    p28b_plast_abs = au["plasticity"] >= 0.9 * fx["plasticity"] if fx["plasticity"] > 0 else (au["plasticity"] >= 0)
    p28b_plast_pgt = au_plast_pgt >= 0.9 * fx_plast_pgt if fx_plast_pgt > 0 else (au_plast_pgt >= 0)
    p28b = p28b_forget_abs and p28b_plast_abs

    # (c) q trajectory: median q phase2 > median q phase1 + 0.03; median phase3 < median phase2
    q1, q2, q3 = (au["q_by_phase"].get(ph, {}).get("median") for ph in ("phase1", "phase2", "phase3"))
    p28c_rise = (q1 is not None and q2 is not None) and (q2 > q1 + 0.03)
    p28c_relax = (q2 is not None and q3 is not None) and (q3 < q2)
    p28c = p28c_rise and p28c_relax

    all_pass = p28a and p28b and p28c
    out["p28_scoring"] = {
        "a_auto_gate_frac_in_range_all_phases": {
            "pass": bool(p28a), "per_phase_in_range": {k: bool(v) for k, v in p28a_bounds.items()},
            "auto_gate_frac_by_phase": au["gate_frac_by_phase"],
            "fixed_gate_frac_by_phase": fx["gate_frac_by_phase"],
        },
        "b_auto_forgetting_leq_and_plasticity_geq_0.9x_fixed": {
            "pass": bool(p28b),
            "absolute": {"auto_forgetting": au["forgetting"], "fixed_forgetting": fx["forgetting"],
                        "forget_pass": bool(p28b_forget_abs),
                        "auto_plasticity": au["plasticity"], "fixed_plasticity": fx["plasticity"],
                        "plast_pass": bool(p28b_plast_abs)},
            "per_gradient_token_per_million": {
                        "auto_forgetting_pgt": round(au_forget_pgt, 6), "fixed_forgetting_pgt": round(fx_forget_pgt, 6),
                        "forget_pgt_pass": bool(p28b_forget_pgt),
                        "auto_plasticity_pgt": round(au_plast_pgt, 6), "fixed_plasticity_pgt": round(fx_plast_pgt, 6),
                        "plast_pgt_pass": bool(p28b_plast_pgt)},
            "grad_tokens_total": {"auto": au_gt, "fixed": fx_gt},
        },
        "c_q_trajectory_rises_phase2_relaxes_phase3": {
            "pass": bool(p28c), "rise": bool(p28c_rise), "relax": bool(p28c_relax),
            "median_q_phase1": q1, "median_q_phase2": q2, "median_q_phase3": q3,
        },
    }
    out["verdict"] = (
        f"gate_frac auto(phase1/2/3)={au['gate_frac_by_phase']} fixed(phase1/2/3)={fx['gate_frac_by_phase']} | "
        f"forgetting auto={au['forgetting']:+.6f} fixed={fx['forgetting']:+.6f} | "
        f"plasticity auto={au['plasticity']:+.6f} fixed={fx['plasticity']:+.6f} | "
        f"q_median phase1/2/3={q1}/{q2}/{q3} | "
        f"P28: {'PASS' if all_pass else 'PARTIAL/FAIL'} (a={p28a}, b={p28b}, c={p28c})"
    )
    print(f"\n[autoq] {out['verdict']}", flush=True)

    d = os.path.dirname(args.out) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_")
    with os.fdopen(fd, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, args.out)
    print(f"[autoq] -> {args.out}", flush=True)
    return out


# ───────────────────────────────────────────────────────────────────────────
#  CLI
# ───────────────────────────────────────────────────────────────────────────
def build_argparser():
    ap = argparse.ArgumentParser(
        description="POS AUTO-Q: homeostat-regulated gating quantile vs fixed-q, under the C4->code->C4 shock (P28)")
    ap.add_argument("--ckpt", default="results/pos_ckpt.pt")
    ap.add_argument("--phase-chunks", type=int, default=150, help="chunks per phase (1/2/3)")
    ap.add_argument("--eval-every", type=int, default=25, help="WT-2 + code-val eval cadence, in chunks")
    ap.add_argument("--code-val-tokens", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=42)
    # shared gate mechanics (rolling window, matches pos_domain_shock R2 defaults)
    ap.add_argument("--fixed-q", type=float, default=0.75, help="fixed arm's constant gate quantile")
    ap.add_argument("--gate-window", type=int, default=200, help="rolling surprise window length")
    ap.add_argument("--min-window", type=int, default=50, help="min window fill before gating kicks in")
    # homeostat constants (auto arm)
    ap.add_argument("--r-star", type=float, default=0.25, help="target gate rate")
    ap.add_argument("--halflife", type=float, default=50.0, help="EMA half-life, in chunks")
    ap.add_argument("--k", type=float, default=0.5, help="P-controller gain")
    ap.add_argument("--q0", type=float, default=0.75, help="homeostat starting q")
    ap.add_argument("--q-min", type=float, default=0.5)
    ap.add_argument("--q-max", type=float, default=0.95)
    ap.add_argument("--smoke", action="store_true", help="phase-chunks=40, eval-every=10, out=*_smoke.json")
    ap.add_argument("--full", action="store_true", help="phase-chunks=150, eval-every=25 (explicit; also the default)")
    ap.add_argument("--out", default="results/pos_auto_q.json")
    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    if args.smoke:
        args.phase_chunks = 40
        args.eval_every = 10
        if args.out == "results/pos_auto_q.json":
            args.out = "results/pos_auto_q_smoke.json"
    elif args.full:
        args.phase_chunks = 150
        args.eval_every = 25

    def eval_wt2_fn(model):
        train_text, val_text = load_wikitext2()
        _, stoi, unk, mask = build_vocab(train_text)
        val_ids = tokenize(val_text, stoi, unk)
        cfg = torch.load(args.ckpt, weights_only=False)["config"]
        evX, evY = build_eval_set(val_ids, cfg["eval_tokens"], cfg["chunk"])
        return heldout(model, evX, evY)

    run_auto_q(args, eval_wt2_fn)


if __name__ == "__main__":
    main()
