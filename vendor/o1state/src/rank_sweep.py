#!/usr/bin/env python3 -u
"""
RANK-SWEEP — does the holographic phase lift binding rank per channel to >=2? (P34, AMENDED)
===============================================================================================
Registered prediction (analysis/PREDICTIONS.md, P34): an Eckart-Young-style capacity
sweep (K keys vs D channels, the rank1_capacity method) on the COMPLEX holographic
state. ORIGINAL absolute criteria (D_eff_phase >= 1.8/channel where scalar inverts to
~1.0, anchored at the theorem doc's measured D_eff~=1.02) could not be used as-is: the
0.1406 anchor's generating script is lost (paper/evidence_companion/hybrid_B.json is
the artifact of record, but is not re-runnable -- two independent reconstruction
attempts, documented in RANK1_CAPACITY_THEOREM.md's "Reproduction debt" section, both
land at the mqar.py scalar floor ~0.017, not 0.1406). Team decision (recorded in
PREDICTIONS.md's P34 amendment, day 4 ~08:15): run on the REPRODUCIBLE mqar.py
instrument (src/mqar.py canonical K-V layout) with RELATIONAL criteria instead of the
lost absolute anchor:

  (a') D_eff_phase >= 2x D_eff_scalar at every K where phase clears 3x chance, AND
       D_eff_phase(K=8) is consistent with the measured 8.9% holographic ceiling
       (results/holographic_mqar.json, 5 seeds, n_heads=4/d_head=32/train_len=64/
       steps=2500: holo_on train_mean=0.0889) -- model-wide D_eff at that recall is
       ~0.6 (see ANCHOR VALIDATION below), so the K=8 phase cell should land near there.
  (b') the phase arm's capacity cliff-K sits at >= 2x the scalar arm's cliff-K (same
       "does complex double the binding load" claim as the original b, now stated as
       a ratio instead of against an absolute D).
  (c)  attention validity gate at ~1.0 throughout -- UNCHANGED.

If (a') fails at every K, that is scored honestly as "no rank lift measured on this
instrument" -- the SNR-based alternative from the original P34 text remains the honest
fallback reading.

IGNITION-AWARE EVALUATION (added after the 3-seed anchor check found the phase arm
does not always ignite -- see analysis/DECISIONS.md, day4 ~11:00). The anchor
validation showed the phase arm reaches the 8.9% ceiling on SOME seeds and sits at
chance on others (1/3 seeds at K=8, multithread, full budget) -- a bimodal
ignition/non-ignition split, not a tight unimodal spread around 0.089. Averaging
across seeds without accounting for this would silently blend "the mechanism has
higher rank" with "the mechanism sometimes fails to find its own basin," corrupting
both the D_eff numbers and the a'/b' verdicts. So EVERY (K, arm) cell now reports:
  - ignition_rate: fraction of seeds whose train recall clears 3x chance.
  - D_eff computed ONLY over ignited seeds, with n_ignited reported alongside (a
    cell's "D_eff_per_channel_mean" is silently meaningless if n_ignited=0 -- it is
    reported as null in that case, not as a misleading chance-level number).
  - recall_seeds: the raw per-seed recalls (already present), the direct evidence
    for the bimodality itself -- kept as its own field so the raw shape survives
    into the JSON even after aggregation.
  - IGNITION_DEAD flag: True when ignition_rate==0 for that cell. Such cells are
    EXCLUDED from the a'/b' relational checks (not scored as "no rank effect" --
    that would misattribute an ignition failure to a capacity finding). The a'/b'
    checks only run over cells with n_ignited >= 2 (need at least 2 ignited seeds
    to speak of a D_eff estimate at all).
4 seeds per cell (not 2): at an observed ignition rate ~1/3, 2 seeds/cell would give
~44% chance of an all-dead cell (misread as "no effect"); 4 seeds drops that to ~20%
and keeps scalar/attn (cheap arms) riding along for free. Team decision, day4 ~11:00.

ARCHITECTURE (per team decision): n_heads=4, d_head=32 -- the SAME channel budget as
holographic_mqar.json's anchor config, NOT swept. This is a departure from the first
draft of this script (which held n_heads=1 so D_eff could be inverted "per head" with
no cross-head pooling ambiguity). The team's call: D_eff is a property of the WHOLE
MODEL's channel pool (n_heads*d_head = 128 channels total), not of an artificially
isolated single head -- the theorem's D_eff~=1 anchor was ALSO measured on a multi-head
stack, so a fair comparison keeps multi-head. D_total = n_heads*d_head = 128 is FIXED
across the whole K-sweep (not calibrated per-K).

ANCHOR VALIDATION (replaces the old d_head-calibration sweep). Before trusting the
K-sweep, the K=8 cell of EACH arm is checked against results/holographic_mqar.json's
5-seed numbers at the identical config (n_heads=4,d_head=32,d_model=128,train_len=64,
steps=2500,n_pairs=8): scalar/holo_off ~0.017, phase/holo_on ~0.089 (+-2sigma~=0.037),
attn ~0.994. This IS the self-smoke: run --smoke (K=8 only, 1 seed, reduced steps for
speed) and compare against these numbers before committing to the full grid. A K=8
cell that lands far from these three numbers at FULL steps is a recipe break and must
be reported, not papered over -- but note the smoke itself uses reduced steps so it
will not hit the full-budget targets; it is a "does the machine run, roughly in the
right basin" check, not the full validation (see --verify-anchor for that, at full
--steps).

THREE ARMS (+ablation) at FIXED D_total=128 (n_heads=4,d_head=32), swept over
K in {2,4,8,16,32}:
  (a) scalar   : SelectiveRapiditySqrtTransformerLM (reference/) -- the scalar floor.
  (b) phase    : HolographicLM(use_phase=True) (src/holographic_gssm.py) -- key-
      conditioned complex write, the phase arm under test.
  (c) phase_off: HolographicLM(use_phase=False) -- ablation, byte-identical reduction
      to Selective by construction (holographic_gssm.py's REDUCTION GUARANTEE); rides
      alongside phase as an internal-consistency check (should track "scalar").
  (d) attn     : TinyCausalTransformerLM (src/mqar.py) -- validity gate, expect ~1.0.

D_EFF COUNTING (fairness, per the brief; UNCHANGED from the original design). A complex
leaky accumulator S_t in C is, as a REAL vector, two coupled real leaky scans (Re, Im)
-- see holographic_gssm.py's docstring ("decomposes into TWO REAL leaky scans"). So the
phase arm's 128 complex channels are 256 real numbers of state, vs 128 real numbers for
the scalar arm. We invert D_eff from recall using the SAME closed form for both arms
(a function of recall & K only) and report it under two labels:
  - D_eff_per_channel  : the raw inversion (in "bindings recovered" units) -- this is
                          what P34(a') compares 2x against for the phase-vs-scalar ratio.
  - D_eff_per_real_dof : D_eff_per_channel / 2 for the phase arm ONLY (it spends 2x the
                          real numbers per channel) -- "does the phase arm's rank merely
                          track its extra real numbers, no free lunch?" alternative read.

CLIFF POSITION (P34 b'). Data-driven cliff-K per arm: the largest K in the swept range
whose recall still clears the midpoint between chance and that arm's own plateau. b'
checks phase_cliff_K >= 2 * scalar_cliff_K (ratio, not an absolute D-based prediction).

TECHNIQUE. CPU-only (mps forced off, repo convention), torch's DEFAULT threading
(deliberately NOT pinned to 1 -- see the instrument-fidelity note by the imports:
threads=1 measurably prevents the phase arm from igniting on this workload, so
this script departs from holo_rent_map.py's threads=1 convention to stay faithful
to the multithread regime the anchor was measured in), os.nice(19) best-effort,
fixed seeds, argparse --smoke/--full/--cells 'K-list'/--out (stripe-parallelization
pattern copied from src/holo_rent_map.py's --cells/--merge idiom, minus the thread
pinning). Outputs: results/rank_sweep{_smoke,_K<k>}.json + --merge.
"""
import os
try:
    os.nice(19)
except Exception:
    pass

import sys
import json
import math
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(REPO, "results")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "reference"))

import torch
import torch.nn.functional as F

torch.backends.mps.is_available = lambda: False   # force CPU (repo convention)
# NOTE: deliberately NOT calling torch.set_num_threads(1) here (departure from
# the holo_rent_map.py --cells stripe-parallelization convention this script
# otherwise follows). Instrument-fidelity finding (analysis/DECISIONS.md, day4
# ~10:30): threads=1 does not just change wall time on this workload -- it
# changes whether the phase arm IGNITES at all. Verified directly: identical
# code, identical seed, 2500 steps -- threads=1 gives a flat loss curve and
# recall~=0.014-0.02 (chance) across 4/5 of the historical seeds; torch's
# multi-thread default (4 threads on this machine) gives a clean loss descent
# and recall~=0.073, inside 1sigma of the 5-seed 0.089+-0.019 anchor
# (results/holographic_mqar.json, whose generating script also never set
# num_threads). The anchor was measured in the multithread regime, so this
# instrument must reproduce that regime, not a faster-but-different one.
# Consequence for --cells stripe-parallelization on beast: each stripe process
# now competes for its own multi-thread pool (torch default) rather than
# pinning to 1 -- plan concurrent stripe COUNT accordingly (fewer parallel
# processes than cores, not one process per core).

from mqar import make_mqar_batch, mqar_accuracy, TinyCausalTransformerLM   # noqa: E402
from moebius_scan_transformer_selective import SelectiveRapiditySqrtTransformerLM  # noqa: E402
from holographic_gssm import HolographicLM   # noqa: E402

DEVICE = torch.device("cpu")
N_KEYS = N_VALUES = 64
CHANCE = 1.0 / N_VALUES

# ═══════════════════════════════════════════════════════════════════════════
# Fixed recipe -- matches results/holographic_mqar.json's anchor config exactly
# (the still-reproducible instrument). n_heads/d_head are NOT swept: D_total =
# n_heads*d_head = 128 channels is the model-wide capacity, held fixed across
# the whole K-grid per the team's amendment.
# ═══════════════════════════════════════════════════════════════════════════
DEFAULT_STEPS = 2500
DEFAULT_LR = 3e-3
DEFAULT_BATCH = 32
DEFAULT_TRAIN_LEN = 64
DEFAULT_TEST_LEN = 256
DEFAULT_D_MODEL = 128
DEFAULT_N_LAYERS = 2
DEFAULT_N_HEADS = 4
DEFAULT_D_HEAD = 32
D_TOTAL = DEFAULT_N_HEADS * DEFAULT_D_HEAD    # 128 -- model-wide channel budget

# results/holographic_mqar.json, 5 seeds, IDENTICAL config (n_heads=4,d_head=32,
# d_model=128,train_len=64,steps=2500,n_pairs=8) -- the anchor-validation targets.
ANCHOR_K8 = {
    "scalar":    {"mean": 0.0170, "std": 0.0022},   # "selective" arm there
    "phase_off": {"mean": 0.0167, "std": 0.0019},   # "holo_off" arm there
    "phase":     {"mean": 0.0889, "std": 0.0186},   # "holo_on" arm there
    "attn":      {"mean": 0.9940, "std": 0.0090},
}


def build_arm(arm, vocab_size, mask_idx, d_model, n_layers, n_heads, d_head, seq_len):
    if arm == "attn":
        return TinyCausalTransformerLM(
            vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            max_len=max(seq_len, 1024))
    if arm == "scalar":
        return SelectiveRapiditySqrtTransformerLM(
            vocab_size, mask_idx, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, d_head=d_head, seq_len=seq_len, dropout=0.0, causal=True)
    if arm == "phase":
        return HolographicLM(
            vocab_size, mask_idx, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, d_head=d_head, seq_len=seq_len, dropout=0.0,
            causal=True, phase_scale=math.pi, use_phase=True)
    if arm == "phase_off":
        return HolographicLM(
            vocab_size, mask_idx, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, d_head=d_head, seq_len=seq_len, dropout=0.0,
            causal=True, phase_scale=math.pi, use_phase=False)
    raise ValueError(arm)


def train_arm(model, cfg, steps, lr, seed, device):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(steps):
        tokens, targets, mask, _ = make_mqar_batch(generator=gen, device=device, **cfg)
        logits = model(tokens)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
        loss = (loss * mask.reshape(-1).float()).sum() / (mask.sum() + 1e-6)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    return model


def eval_arm(model, train_cfg, test_cfg, seed, device):
    model.eval()
    tr_overall, tr_gap, _ = mqar_accuracy(model, train_cfg, 8, seed + 1, device)
    te_overall, te_gap, _ = mqar_accuracy(model, test_cfg, 8, seed + 2, device)
    return {"train_len": {"overall": round(tr_overall, 4), "by_gap": tr_gap},
            "test_len": {"overall": round(te_overall, 4), "by_gap": te_gap}}


def d_eff_from_recall(recall, K, V=N_VALUES):
    """Invert RANK1_CAPACITY_THEOREM.md's closed form:
        recall = D_eff/K + (1 - D_eff/K)/V
    =>  D_eff = (recall - 1/V) / (1 - 1/V) * K
    Not clamped (negative / >K values are informative: below-chance or super-rank
    behavior, reported as-is, not hidden by clipping)."""
    return (recall - 1.0 / V) / (1.0 - 1.0 / V) * K


def run_one(arm, K, steps, lr, batch, train_len, test_len, d_model, n_layers,
            n_heads, d_head, seed, vocab_size, mask_idx):
    # mqar geometry: kv_len(2K) + n_queries(K) must fit in seq_len. K=32 at the
    # grid default train_len=64 is infeasible (96 > 64). Lift ONLY infeasible
    # cells to 4*K (filler stays proportional to the K16/64 cell); feasible
    # cells keep the exact length they already ran at, so completed grid cells
    # stay comparable and mergeable.
    if 3 * K > train_len:
        train_len = 4 * K
    train_cfg = dict(batch_size=batch, seq_len=train_len, n_pairs=K,
                      n_queries=K, n_keys=N_KEYS, n_values=N_VALUES)
    test_cfg = dict(batch_size=batch, seq_len=test_len, n_pairs=K,
                     n_queries=K, n_keys=N_KEYS, n_values=N_VALUES)
    torch.manual_seed(seed)
    model = build_arm(arm, vocab_size, mask_idx, d_model, n_layers, n_heads, d_head, train_len)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    t0 = time.time()
    train_arm(model, train_cfg, steps, lr, seed, DEVICE)
    res = eval_arm(model, train_cfg, test_cfg, seed, DEVICE)
    res["params"] = n_params
    res["wall_s"] = round(time.time() - t0, 1)
    return res


# ═══════════════════════════════════════════════════════════════════════════
# Anchor validation: train K=8 for each arm at the CURRENT --steps and compare
# against ANCHOR_K8 (the 5-seed holographic_mqar.json numbers at full budget).
# This is the self-smoke's core check. At reduced --steps (--smoke) the model
# may not have ignited yet -- reported as such, not silently passed.
# ═══════════════════════════════════════════════════════════════════════════
def verify_anchor(args, vocab_size, mask_idx):
    print("=" * 78)
    print(f"ANCHOR VALIDATION: K=8, n_heads={args.n_heads}, d_head={args.d_head}, "
          f"steps={args.steps} (full-budget targets: steps={DEFAULT_STEPS})")
    print("target (results/holographic_mqar.json, 5 seeds, full steps):")
    for arm, v in ANCHOR_K8.items():
        print(f"  {arm:10s}  {v['mean']:.4f} +- {v['std']:.4f}")
    print("=" * 78)
    rows = {}
    for arm in ["scalar", "phase_off", "phase", "attn"]:
        res = run_one(arm, 8, args.steps, args.lr, args.batch, args.train_len,
                      args.test_len, args.d_model, args.n_layers, args.n_heads,
                      args.d_head, args.seed, vocab_size, mask_idx)
        recall = res["train_len"]["overall"]
        target = ANCHOR_K8[arm]
        within_2sigma = abs(recall - target["mean"]) <= 2 * max(target["std"], 1e-4)
        note = ""
        if args.steps < DEFAULT_STEPS:
            note = f" (REDUCED steps={args.steps} vs anchor's {DEFAULT_STEPS} -- basin check only)"
        print(f"  {arm:10s}  recall={recall:.4f}  target={target['mean']:.4f}+-{target['std']:.4f}  "
              f"{'within 2sigma' if within_2sigma else 'OUTSIDE 2sigma'}{note}  ({res['wall_s']}s)")
        rows[arm] = {"recall": recall, "target_mean": target["mean"],
                     "target_std": target["std"], "within_2sigma": within_2sigma,
                     "wall_s": res["wall_s"], "params": res["params"]}
    all_ok = all(r["within_2sigma"] for r in rows.values())
    print(f"\n-> anchor validation: {'ALL WITHIN 2sigma' if all_ok else 'AT LEAST ONE OUTSIDE 2sigma'}"
          f"{' (expected at reduced steps -- rerun with --steps ' + str(DEFAULT_STEPS) + ' to confirm)' if not all_ok and args.steps < DEFAULT_STEPS else ''}")
    return {"rows": rows, "all_within_2sigma": all_ok, "steps_used": args.steps,
            "full_budget_steps": DEFAULT_STEPS}


# ═══════════════════════════════════════════════════════════════════════════
# Grid: K in {2,4,8,16,32} x arm in {scalar, phase, phase_off, attn}, D_total
# fixed at n_heads*d_head=128 throughout (not swept).
# ═══════════════════════════════════════════════════════════════════════════
def parse_ks(spec, all_ks):
    if not spec:
        return list(all_ks)
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def run_grid(args):
    vocab_size = N_KEYS + N_VALUES + 1
    mask_idx = vocab_size

    anchor_block = None
    if args.verify_anchor or args.smoke:
        anchor_block = verify_anchor(args, vocab_size, mask_idx)

    all_ks = [int(x) for x in args.k_grid.split(",")]
    ks = parse_ks(args.cells, all_ks)
    seeds = [int(x) for x in args.seeds.split(",")]
    arms = args.arms.split(",")

    print("=" * 78)
    print("RANK-SWEEP — K-sweep at fixed D_total (P34, amended/relational)")
    print(f"K={ks}  D_total={args.n_heads * args.d_head} (n_heads={args.n_heads},"
          f"d_head={args.d_head})  arms={arms}  seeds={seeds}  steps={args.steps}  lr={args.lr}")
    print("=" * 78)

    out = {
        "config": vars(args), "D_total": args.n_heads * args.d_head,
        "anchor_validation": anchor_block,
        "chance": CHANCE, "n_keys": N_KEYS, "n_values": N_VALUES,
        "ks_run": ks, "grid": {},
    }

    t0 = time.time()
    for K in ks:
        print(f"\n{'='*78}\nK={K}\n{'='*78}")
        k_key = f"K{K}"
        out["grid"][k_key] = {"K": K, "seeds": {}}
        for seed in seeds:
            print(f"  -- seed={seed} --")
            seed_rec = {}
            for arm in arms:
                res = run_one(arm, K, args.steps, args.lr, args.batch,
                               args.train_len, args.test_len, args.d_model,
                               args.n_layers, args.n_heads, args.d_head,
                               seed, vocab_size, mask_idx)
                recall = res["train_len"]["overall"]
                recall_test = res["test_len"]["overall"]
                deff = d_eff_from_recall(recall, K)
                seed_rec[arm] = {
                    "train_recall": recall, "test_recall": recall_test,
                    "D_eff_per_channel": round(deff, 3),
                    "D_eff_per_real_dof": round(deff / 2.0, 3) if arm == "phase" else None,
                    "params": res["params"], "wall_s": res["wall_s"],
                    "by_gap_train": res["train_len"]["by_gap"],
                }
                print(f"    [{arm:10s}] recall={recall:.4f} (test {recall_test:.4f})  "
                      f"D_eff={deff:.3f}  ({res['wall_s']}s)")
            out["grid"][k_key]["seeds"][str(seed)] = seed_rec
        _write(out, args.out)

    out["elapsed_s"] = round(time.time() - t0, 1)
    out["analysis"] = analyze(out, ks, arms, args.n_heads * args.d_head)
    _write(out, args.out)

    print("\n" + "=" * 78)
    print("VERDICT")
    print(out["analysis"]["verdict"])
    print(f"\n-> {args.out}  ({out['elapsed_s']}s)")


def _write(out, path):
    os.makedirs(RESULTS, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# Analysis: recall(K) curves per arm, D_eff per arm/K, P34-AMENDED relational
# checks: (a') D_eff_phase >= 2x D_eff_scalar where phase clears 3x chance,
# + K=8 consistency with the 8.9% ceiling (~0.6 model-wide D_eff); (b') phase
# cliff-K >= 2x scalar cliff-K; (c) attention ~1.0 unchanged.
# ═══════════════════════════════════════════════════════════════════════════
def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def analyze(out, ks, arms, d_total):
    grid = out["grid"]
    per_arm_curve = {arm: {} for arm in arms}
    three_chance = 3 * CHANCE
    for K in ks:
        k_key = f"K{K}"
        cell = grid.get(k_key)
        if cell is None:
            continue
        for arm in arms:
            recalls = [seed_rec[arm]["train_recall"]
                       for seed_rec in cell["seeds"].values() if arm in seed_rec]
            deffs_all = [seed_rec[arm]["D_eff_per_channel"]
                         for seed_rec in cell["seeds"].values() if arm in seed_rec]
            if not recalls:
                continue
            n_seeds = len(recalls)
            ignited_mask = [r >= three_chance for r in recalls]
            n_ignited = sum(ignited_mask)
            ignition_rate = n_ignited / n_seeds
            deffs_ignited = [d for d, ig in zip(deffs_all, ignited_mask) if ig]
            per_arm_curve[arm][K] = {
                "recall_mean": round(_mean(recalls), 4),
                "recall_seeds": [round(r, 4) for r in recalls],   # raw per-seed shape (bimodality evidence)
                "n_seeds": n_seeds,
                "n_ignited": n_ignited,
                "ignition_rate": round(ignition_rate, 3),
                "IGNITION_DEAD": n_ignited == 0,
                # D_eff over ALL seeds (for reference / transparency) vs over IGNITED
                # seeds only (the number that should be trusted as a capacity estimate).
                "D_eff_per_channel_mean_all_seeds": round(_mean(deffs_all), 3),
                "D_eff_per_channel_mean_ignited": (
                    round(_mean(deffs_ignited), 3) if deffs_ignited else None),
                "D_eff_seeds_ignited_only": [round(d, 3) for d in deffs_ignited],
            }

    # ── (c) attention validity gate — unchanged (attention has no ignition-failure
    #        mode observed; still reported via its own recall_mean, not filtered) ──
    attn_vals = [v["recall_mean"] for v in per_arm_curve.get("attn", {}).values()]
    attn_ok = bool(attn_vals) and min(attn_vals) >= 0.90
    check_c = {"attn_recall_by_K": {K: v["recall_mean"] for K, v in per_arm_curve.get("attn", {}).items()},
               "min_attn_recall": round(min(attn_vals), 4) if attn_vals else None,
               "passes_ge_0_90": attn_ok}

    scalar_curve = per_arm_curve.get("scalar", {})
    phase_curve = per_arm_curve.get("phase", {})

    # ── (a') D_eff_phase >= 2x D_eff_scalar at every K where phase clears 3x chance
    #        AND has >=2 ignited seeds (ignition-aware: dead cells are EXCLUDED, not
    #        scored as "no rank effect"), + K=8 consistency with the 8.9% ceiling
    #        (model-wide D_eff at recall=0.089, K=8, V=64: D_eff ~= 0.586 ~= 0.6). ──
    MIN_IGNITED = 2
    ratio_rows = {}
    for K in sorted(set(scalar_curve) & set(phase_curve)):
        p_cell = phase_curve[K]
        s_cell = scalar_curve[K]
        p_recall = p_cell["recall_mean"]
        p_deff = p_cell["D_eff_per_channel_mean_ignited"]
        s_deff = s_cell["D_eff_per_channel_mean_ignited"]
        # eligibility now requires BOTH the mean-recall gate AND enough ignited
        # seeds to trust the D_eff estimate at all; scalar's own ignition (it
        # rarely if ever clears 3x chance, so s_deff will usually be null/~0)
        # is tracked separately but does not by itself exclude the K (the ratio
        # is guarded below against a null/~0 denominator).
        enough_ignited = (p_cell["n_ignited"] >= MIN_IGNITED)
        clears_3x_chance = p_recall >= three_chance
        eligible_k = clears_3x_chance and enough_ignited and p_deff is not None
        ratio = None
        if eligible_k and s_deff is not None and s_deff > 1e-6:
            ratio = p_deff / s_deff
        ratio_rows[K] = {
            "phase_recall_mean": p_recall,
            "phase_ignition_rate": p_cell["ignition_rate"], "phase_n_ignited": p_cell["n_ignited"],
            "phase_IGNITION_DEAD": p_cell["IGNITION_DEAD"],
            "clears_3x_chance": clears_3x_chance, "enough_ignited_seeds (>=2)": enough_ignited,
            "eligible_for_a_prime": eligible_k,
            "phase_D_eff_ignited": p_deff, "scalar_D_eff_ignited": s_deff,
            "ratio_phase_over_scalar": round(ratio, 3) if ratio is not None else None,
            "passes_2x_at_this_K": (ratio is not None and ratio >= 2.0) if eligible_k else None,
        }
    eligible = {K: v for K, v in ratio_rows.items() if v["eligible_for_a_prime"]}
    dead_cells = {K: v for K, v in ratio_rows.items() if v["phase_IGNITION_DEAD"]}
    passes_a_ratio = (all(v["passes_2x_at_this_K"] for v in eligible.values())
                       if eligible else None)

    k8_deff = phase_curve.get(8, {}).get("D_eff_per_channel_mean_ignited")
    k8_n_ignited = phase_curve.get(8, {}).get("n_ignited", 0)
    ceiling_target = d_eff_from_recall(0.0889, 8)   # ~0.586, the 8.9%-ceiling's model-wide D_eff
    k8_consistent = (abs(k8_deff - ceiling_target) <= 0.3
                      if k8_deff is not None and k8_n_ignited >= MIN_IGNITED else None)

    check_a = {
        "three_x_chance": round(three_chance, 4),
        "min_ignited_seeds_required": MIN_IGNITED,
        "per_K": ratio_rows,
        "eligible_Ks (clears 3x chance AND >=2 ignited seeds)": list(eligible.keys()),
        "IGNITION_DEAD_Ks (excluded, not scored as null-effect)": list(dead_cells.keys()),
        "passes_2x_ratio_at_all_eligible_Ks": passes_a_ratio,
        "K8_phase_D_eff_ignited": k8_deff, "K8_n_ignited": k8_n_ignited,
        "K8_ceiling_target_D_eff (from 8.9pct anchor)": round(ceiling_target, 3),
        "K8_consistent_with_ceiling (+-0.3, requires >=2 ignited)": k8_consistent,
        "passes_a_prime": bool(passes_a_ratio) and bool(k8_consistent) if (
            passes_a_ratio is not None and k8_consistent is not None) else None,
    }

    # ── (b') cliff position ratio: phase_cliff_K >= 2 * scalar_cliff_K.
    #        Ignition-aware: the plateau and the midpoint threshold are computed
    #        only from cells with n_ignited>=2 (a dead cell must not drag the
    #        plateau down and manufacture a false early cliff). ──
    def cliff_k(curve):
        usable = {K: v for K, v in curve.items() if v["n_ignited"] >= MIN_IGNITED}
        if not usable:
            return {"cliff_K": None, "note": "no cell with >=2 ignited seeds -- cliff undefined"}
        Ks_sorted = sorted(usable.keys())
        plateau = max(v["recall_mean"] for v in usable.values())
        mid = (CHANCE + plateau) / 2.0
        above = [K for K in Ks_sorted if usable[K]["recall_mean"] >= mid]
        below = [K for K in Ks_sorted if usable[K]["recall_mean"] < mid]
        if not above:
            return {"cliff_K": Ks_sorted[0], "note": "never clears midpoint; cliff at/below smallest usable K",
                     "n_dead_excluded": len(curve) - len(usable)}
        if not below:
            return {"cliff_K": Ks_sorted[-1], "note": "never falls below midpoint among usable Ks",
                     "n_dead_excluded": len(curve) - len(usable)}
        return {"cliff_K": max(above), "note": f"last usable K clearing recall midpoint {mid:.3f}",
                 "n_dead_excluded": len(curve) - len(usable)}

    scalar_cliff = cliff_k(scalar_curve)
    phase_cliff = cliff_k(phase_curve)
    cliff_ratio = None
    passes_b_prime = None
    if (scalar_cliff and phase_cliff and scalar_cliff["cliff_K"] is not None
            and phase_cliff["cliff_K"] is not None and scalar_cliff["cliff_K"] > 0):
        cliff_ratio = phase_cliff["cliff_K"] / scalar_cliff["cliff_K"]
        passes_b_prime = cliff_ratio >= 2.0
    check_b = {
        "D_total": d_total,
        "scalar_cliff": scalar_cliff, "phase_cliff": phase_cliff,
        "cliff_ratio_phase_over_scalar": round(cliff_ratio, 3) if cliff_ratio is not None else None,
        "passes_b_prime (ratio>=2)": passes_b_prime,
    }

    verdict_bits = []
    verdict_bits.append(f"(c) attention validity: {'PASS' if attn_ok else 'FAIL/VOID'} "
                         f"(min recall {check_c['min_attn_recall']})")
    if dead_cells:
        verdict_bits.append(f"IGNITION_DEAD phase cells (excluded from a'/b'): {list(dead_cells.keys())}")
    if check_a["passes_a_prime"] is not None:
        verdict_bits.append(f"(a') D_eff_phase>=2x D_eff_scalar + K8~ceiling: "
                             f"{'PASS' if check_a['passes_a_prime'] else 'FAIL -> rent is not (2x) rank on this instrument'}")
    else:
        verdict_bits.append("(a') insufficient data (no K clears 3x chance for phase, or K=8 missing) -- cannot score")
    if passes_b_prime is not None:
        verdict_bits.append(f"(b') cliff ratio phase/scalar = {check_b['cliff_ratio_phase_over_scalar']} "
                             f"(>=2 required): {'PASS' if passes_b_prime else 'FAIL'}")
    else:
        verdict_bits.append("(b') cliff ratio undefined (scalar cliff_K=0 or missing data)")

    return {
        "per_arm_curve": per_arm_curve,
        "checks": {"a_prime_relational_rank": check_a, "b_prime_cliff_ratio": check_b,
                   "c_attention_validity": check_c},
        "verdict": " | ".join(verdict_bits),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Merge partial --cells JSONs into one results/rank_sweep.json.
# ═══════════════════════════════════════════════════════════════════════════
def run_merge(args):
    paths = [p.strip() for p in args.merge.split(",") if p.strip()]
    merged = {"config": None, "D_total": None, "anchor_validation": None,
              "chance": CHANCE, "n_keys": N_KEYS, "n_values": N_VALUES,
              "ks_run": [], "grid": {}, "elapsed_s": 0.0}
    ks_all = []
    arms = None
    d_total = None
    for p in paths:
        with open(p) as f:
            part = json.load(f)
        if merged["config"] is None:
            merged["config"] = part["config"]
            merged["D_total"] = part["D_total"]
            merged["anchor_validation"] = part.get("anchor_validation")
            d_total = part["D_total"]
            arms = part["config"]["arms"].split(",")
        merged["ks_run"].extend(part.get("ks_run", []))
        merged["grid"].update(part.get("grid", {}))
        merged["elapsed_s"] += part.get("elapsed_s", 0.0)
        ks_all.extend(part.get("ks_run", []))
    merged["elapsed_s"] = round(merged["elapsed_s"], 1)
    ks_all = sorted(set(ks_all))
    merged["analysis"] = analyze(merged, ks_all, arms, d_total)
    _write(merged, args.out)
    print(f"merged {len(paths)} partial file(s) -> {args.out}")
    print(merged["analysis"]["verdict"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                     help="K=8 only, 1 seed, reduced steps, includes anchor validation -- "
                          "<15min sanity check (reduced steps = basin check, not full match)")
    ap.add_argument("--full", action="store_true",
                     help="full K-grid, 4 seeds (0,1,7,42 -- team decision day4 ~11:00: at "
                          "the observed ~1/3 phase-ignition rate, 2 seeds/cell risked ~44%% "
                          "chance of an all-dead cell; 4 seeds drops that to ~20%%), anchor steps")
    ap.add_argument("--k-grid", default="2,4,8,16,32")
    ap.add_argument("--arms", default="scalar,phase,phase_off,attn")
    ap.add_argument("--seeds", default="0,1,7,42")
    ap.add_argument("--n-heads", type=int, default=DEFAULT_N_HEADS,
                     help="fixed across the whole K-grid (not swept) -- model-wide D_total="
                          "n_heads*d_head is the channel budget the team decided to hold fixed")
    ap.add_argument("--d-head", type=int, default=DEFAULT_D_HEAD)
    ap.add_argument("--verify-anchor", action="store_true",
                     help="run the K=8 anchor-validation cell (all 4 arms) before the grid, "
                          "compared against results/holographic_mqar.json's 5-seed numbers")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--train-len", type=int, default=DEFAULT_TRAIN_LEN)
    ap.add_argument("--test-len", type=int, default=DEFAULT_TEST_LEN)
    ap.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL)
    ap.add_argument("--n-layers", type=int, default=DEFAULT_N_LAYERS)
    ap.add_argument("--seed", type=int, default=0, help="single seed used during --verify-anchor")
    ap.add_argument("--cells", default="",
                     help="subset of K values as 'K,K,...' (stripe-parallelization across "
                          "processes, holo_rent_map.py's --cells idiom); empty = full --k-grid")
    ap.add_argument("--out", default=os.path.join(RESULTS, "rank_sweep.json"))
    ap.add_argument("--merge", default="",
                     help="comma list of partial JSON paths to merge into --out")
    args = ap.parse_args()

    if args.merge:
        run_merge(args)
        return

    if args.smoke:
        args.cells = "8"
        args.seeds = "0"
        args.steps = min(args.steps, 400)
        if args.out == os.path.join(RESULTS, "rank_sweep.json"):
            args.out = os.path.join(RESULTS, "rank_sweep_smoke.json")
    elif args.full:
        # NOTE: do NOT reset args.cells here -- an explicit --cells stripe must
        # survive --full (it did not, which silently reran the whole grid).
        args.seeds = "0,1,7,42"
        args.steps = max(args.steps, DEFAULT_STEPS)

    run_grid(args)


if __name__ == "__main__":
    main()
