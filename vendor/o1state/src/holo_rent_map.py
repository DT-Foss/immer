#!/usr/bin/env python3 -u
"""
HOLO-RENT-MAP — where does the holographic phase "rent" (holo_on - holo_off) live
in the (P_max, d_model) plane?
======================================================================================
Two anchors on record, same v3 recipe (train-short-eval-long, StreamingHolographicScanLayer,
gap-curriculum, chunked+carried eval), wildly different outcome:
  - results/holo_stream_recall_v3.json: P_max=16, d_model=64 (n_heads=4,d_head=16,
    2*d_model=128 state channels) -> carried beats off by +25-30pp at P=2 (0.83 vs 0.54
    at G=0, ignition_detail line "P=2 G=0: carried=0.830 ... off=0.540").
  - results/holo_heldout_keys.json: P_max=64, d_model=64 (SAME channel budget) -> both
    arms plateau ~0.55, rent ~0 ("BOTH ARMS GENERALIZE ... embedding geometry alone
    appears to carry enough key identity for the magnitude-only baseline to bind").
  - results/holo_capreturn.json (M6/P17): P_max=64, d_model in {64,128,256} -> scanned
    whether capacity alone brings the advantage back; also the source of the
    lr-control idiom (run_offlr_control) used below at the corner cells, because a
    larger d_model can make the holo_off ARM collapse at a fixed lr as an optimization
    artifact, not a genuine binding failure -- a "Scheinriese" (paper giant) that looks
    like phase superiority but is really an under-tuned baseline.

QUESTION (P32, registered): is "rent" (holo_on advantage over holo_off) a function of
the RATIO d_model / P_max (channels-per-key), collapsing onto one curve regardless of
the absolute scale of either axis? If channel-budget-per-key is the real variable, cells
sharing a ratio should show similar rent, and the transition from ~0 to full rent should
sit in a specific ratio band.

GRID: P_max in {8,16,32,64} x d_model in {32,64,128,256} x arm in {holo_on,holo_off} x
seed in {0,1} = 4*4*2*2 = 64 (P_max,d,arm,seed) cells, each trained with the v3 recipe
(iters/lr/chunk taken verbatim from results/holo_stream_recall_v3.json's config, since
the P_max=16,d=64 cell must reproduce that anchor) then evaluated at P in {2,4} x
G in {8,32} chunked+carried, plus a zeroed-null @ G=32 (the holo_zeroed_at_gap arm,
decisive-null idiom from holo_stream_recall.py, run alongside holo_on at every cell --
it is not swept independently, it rides on the holo_on model).

LR CONTROL (the Scheinriesen lesson from holo_capacity_return.py's run_offlr_control):
at the four corner cells (P_max,d) in {(8,32),(8,256),(64,32),(64,256)}, holo_off is
ADDITIONALLY trained at lr in {1e-3,3e-3} (v3's own lr=3e-3 is always included, so this
adds one extra candidate); the best-of-off accuracy is substituted into that corner's
rent computation IF it changes the accuracy. If using best-of-off flips the sign of rent
at that corner (vs using only the v3 lr), the cell is flagged lr_sensitive=True and
EXCLUDED from the ratio-collapse means/Spearman (not silently averaged in) -- reported
separately instead.

d_head is fixed at d_model/4 (n_heads=4, exactly holo_capacity_return.py's scaling:
total state channels = n_heads*d_head*n_layers = 2*d_model). P_max in {32,64} exceed the
default v_max/f_fillers=16 vocab convention only in the KEY range (_gap_vocab keeps
V_max, F fixed at their v3 defaults; only P_max, i.e. the key range, is swept) -- keys
and values/fillers stay in disjoint id ranges exactly as _gap_vocab guarantees for any
P_max.

CPU-only, single thread (torch.set_num_threads(1) AND torch.set_num_interop_threads(1),
both wrapped in try/except since this also runs on beast/Linux where interop-threads
may already be set), os.nice(19) best-effort. Results -> results/holo_rent_map*.json.
"""
import os
try:
    os.nice(19)
except Exception:
    pass

import sys
import json
import time
import argparse
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference"))

import torch
import torch.nn as nn

torch.backends.mps.is_available = lambda: False   # force CPU (repo convention)
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass  # already set / not settable post-init on this build (observed on beast/Linux)

from holo_stream_recall import (   # noqa: E402
    _gap_vocab, _build_lm, chunked_forward, make_gap_mqar_batch,
    train_gap_curriculum, eval_gap_recall,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(REPO, "results")

# ═══════════════════════════════════════════════════════════════════════════
# v3 anchor config, taken verbatim from results/holo_stream_recall_v3.json's
# "config" block (p_max=16,d_model=64 cell) -- this run MUST reproduce that
# cell's rent to trust the rest of the grid.
# ═══════════════════════════════════════════════════════════════════════════
V3_ITERS = 2000
V3_LR = 3e-3
V3_BATCH = 32
V3_CHUNK = 16
V3_G_START = 2
V3_PATIENCE = 25
V3_EVAL_BATCH = 200
V3_V_MAX = 16
V3_F_FILLERS = 16
V3_N_HEADS = 4
V3_N_LAYERS = 2

CORNER_CELLS = [(8, 32), (8, 256), (64, 32), (64, 256)]


# ═══════════════════════════════════════════════════════════════════════════
# One (P_max, d_model, arm, seed, P) training+eval unit. Reuses the v3 machinery
# unmodified (chunked_forward / train_gap_curriculum / eval_gap_recall all
# imported straight from holo_stream_recall.py -- no reimplementation).
#
# IMPORTANT (fixed after the first self-smoke, which reproduced ~0 rent even at
# the P_max=16,d=64 anchor): holo_stream_recall.run() trains a FRESH model PER P
# (its `for P in Ps:` loop rebuilds the model every iteration) -- it never trains
# one model and asks it to serve multiple P values. My first cut here shared one
# model across eval_Ps=[2,4], training at the harder P=4 cap (G<=6) and evaluating
# it at P=2 too; that is a different, harder, and never-validated regime, not the
# v3 recipe, and it collapsed the anchor's rent from +25-30pp to <1pp. Fixed: one
# model PER (P_max, d_model, arm, seed, P) -- exactly v3's loop nesting, just with
# P_max/d_model as extra outer axes.
# ═══════════════════════════════════════════════════════════════════════════
def run_cell_for_P(P_max, d_model, use_phase, seed, lr, iters, P, eval_Gs,
                    log_every=0, zero_at_gap_too=False):
    """Trains ONE model (arm=use_phase) at (P_max,d_model,seed,lr) FOR THIS P
    (v3's exact per-P loop body), then evaluates at every G in eval_Gs,
    chunked+carried. If zero_at_gap_too, ALSO evaluates the same trained model
    under the decisive-null (state zeroed at gap onset) at every G.
    Returns (curriculum_dict, {G: acc}, {G: zeroed_acc}|None, train_seconds).
    """
    V_max, F = V3_V_MAX, V3_F_FILLERS
    key_lo, val_lo, fill_lo, vocab_size = _gap_vocab(P_max, V_max, F)
    mask_idx = vocab_size
    n_heads = V3_N_HEADS
    d_head = d_model // n_heads
    assert d_head * n_heads == d_model, f"d_model={d_model} not divisible by n_heads={n_heads}"

    torch.manual_seed(seed)
    model = _build_lm(vocab_size, mask_idx, use_phase=use_phase,
                       d_model=d_model, n_layers=V3_N_LAYERS,
                       n_heads=n_heads, d_head=d_head)

    # v3 train-short-eval-long: cap training gap at THIS P's single-chunk max
    # (holo_stream_recall.run()'s per-P cap, verbatim: chunk - 2P - 2).
    Gmax_eval = max(eval_Gs) if eval_Gs else 0
    cap = V3_CHUNK - 2 * P - 2
    Gmax_train = min(Gmax_eval, max(2, cap))

    t0 = time.time()
    curr = train_gap_curriculum(
        model, P, Gmax_train, iters, lr, seed, V3_BATCH, V3_CHUNK,
        P_max, V_max, F, log_every=log_every, g_start=V3_G_START,
        patience=V3_PATIENCE)
    curr["train_gap_cap"] = Gmax_train
    curr["train_P"] = P
    train_s = time.time() - t0

    accs = {}
    zeroed_accs = {} if zero_at_gap_too else None
    for G in eval_Gs:
        acc, margin = eval_gap_recall(
            model, P, G, V3_EVAL_BATCH, seed + 1000 + G, P_max, V_max, F,
            V3_CHUNK, zero_at_gap=False)
        accs[G] = {"accuracy": round(acc, 4), "phi_margin": margin}
        if zero_at_gap_too:
            zacc, _ = eval_gap_recall(
                model, P, G, V3_EVAL_BATCH, seed + 1000 + G, P_max, V_max, F,
                V3_CHUNK, zero_at_gap=True)
            zeroed_accs[G] = {"accuracy": round(zacc, 4)}

    return curr, accs, zeroed_accs, round(train_s, 1)


# ═══════════════════════════════════════════════════════════════════════════
# Grid orchestration.
# ═══════════════════════════════════════════════════════════════════════════
def parse_cells(spec, all_pmax, all_d):
    """--cells 'Pmax:d,Pmax:d,...' -> list[(P_max,d_model)]. Empty spec = full grid."""
    if not spec:
        return [(p, d) for p in all_pmax for d in all_d]
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        p_s, d_s = tok.split(":")
        out.append((int(p_s), int(d_s)))
    return out


def run_grid(args):
    all_pmax = [int(x) for x in args.pmax_grid.split(",")]
    all_d = [int(x) for x in args.d_grid.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    eval_Ps = [int(x) for x in args.eval_pairs.split(",")]
    eval_Gs = [int(x) for x in args.eval_gaps.split(",")]

    cells = parse_cells(args.cells, all_pmax, all_d)

    print("=" * 78)
    print("HOLO-RENT-MAP — rent(P_max, d_model) sweep")
    print(f"cells={cells}")
    print(f"seeds={seeds}  eval P={eval_Ps}  eval G={eval_Gs}  iters={args.iters}  lr={args.lr}")
    print("=" * 78)

    t0 = time.time()
    out = {"config": vars(args), "cells_run": cells, "grid": {}, "corner_lr_control": {}}

    for (P_max, d_model) in cells:
        is_corner = (P_max, d_model) in CORNER_CELLS
        print(f"\n{'='*78}\nP_max={P_max}  d_model={d_model}  (corner={is_corner})\n{'='*78}")
        cell_key = f"Pmax{P_max}_d{d_model}"
        out["grid"][cell_key] = {"P_max": P_max, "d_model": d_model, "is_corner": is_corner,
                                  "seeds": {}}

        for seed in seeds:
            print(f"  -- seed={seed} --")
            seed_rec = {"holo_on": {"per_P": {}}, "holo_off": {"per_P": {}}}

            accs_on_all, zeroed_on_all, accs_off_all = {}, {}, {}
            for P in eval_Ps:
                # holo_on: also spot-check the decisive null (zeroed-at-gap),
                # riding on the same trained model (no extra training cost).
                curr_on, accs_on, zeroed_on, s_on = run_cell_for_P(
                    P_max, d_model, use_phase=True, seed=seed, lr=args.lr, iters=args.iters,
                    P=P, eval_Gs=eval_Gs, log_every=args.log_every, zero_at_gap_too=True)
                print(f"    [holo_on  P={P}] curriculum: final_gap={curr_on['final_train_gap']} "
                      f"final_acc={curr_on['final_train_acc']:.3f}  ({s_on}s)")
                seed_rec["holo_on"]["per_P"][str(P)] = {
                    "curriculum": curr_on, "train_s": s_on,
                    "eval": {f"G{g}": v for g, v in accs_on.items()},
                    "zeroed_eval": {f"G{g}": v for g, v in (zeroed_on or {}).items()},
                }
                for g, v in accs_on.items():
                    accs_on_all[(P, g)] = v
                for g, v in (zeroed_on or {}).items():
                    zeroed_on_all[(P, g)] = v

                # holo_off, always at v3 lr first.
                curr_off, accs_off, _, s_off = run_cell_for_P(
                    P_max, d_model, use_phase=False, seed=seed, lr=args.lr, iters=args.iters,
                    P=P, eval_Gs=eval_Gs, log_every=args.log_every, zero_at_gap_too=False)
                print(f"    [holo_off P={P}] curriculum: final_gap={curr_off['final_train_gap']} "
                      f"final_acc={curr_off['final_train_acc']:.3f}  ({s_off}s)")
                seed_rec["holo_off"]["per_P"][str(P)] = {
                    "curriculum": curr_off, "train_s": s_off,
                    "eval": {f"G{g}": v for g, v in accs_off.items()},
                    "lr": args.lr,
                }
                for g, v in accs_off.items():
                    accs_off_all[(P, g)] = v

            seed_rec["holo_on"]["eval"] = {f"P{p}_G{g}": v for (p, g), v in accs_on_all.items()}
            seed_rec["holo_on"]["zeroed_eval"] = {f"P{p}_G{g}": v for (p, g), v in zeroed_on_all.items()}
            seed_rec["holo_off"]["eval"] = {f"P{p}_G{g}": v for (p, g), v in accs_off_all.items()}
            seed_rec["holo_off"]["lr"] = args.lr

            # ── corner cells: additional holo_off lr candidates (Scheinriesen check) ──
            if is_corner and args.corner_lrs:
                corner_lrs = [float(x) for x in args.corner_lrs.split(",")]
                corner_lrs = [lr for lr in corner_lrs if abs(lr - args.lr) > 1e-12]
                off_by_lr = {round(args.lr, 6): accs_off_all}
                for lr in corner_lrs:
                    accs_off_lr_all = {}
                    for P in eval_Ps:
                        print(f"    [holo_off lr-control P={P}] lr={lr}")
                        _, accs_off_lr, _, s_lr = run_cell_for_P(
                            P_max, d_model, use_phase=False, seed=seed, lr=lr, iters=args.iters,
                            P=P, eval_Gs=eval_Gs, log_every=0, zero_at_gap_too=False)
                        for g, v in accs_off_lr.items():
                            accs_off_lr_all[(P, g)] = v
                    off_by_lr[round(lr, 6)] = accs_off_lr_all

                # best-of-off per (P,G): max accuracy across ALL lr candidates tried
                best_of_off = {}
                for P in eval_Ps:
                    for G in eval_Gs:
                        per_lr = {lr_k: v[(P, G)]["accuracy"] for lr_k, v in off_by_lr.items()}
                        best_lr = max(per_lr, key=per_lr.get)
                        best_of_off[f"P{P}_G{G}"] = {
                            "best_lr": best_lr, "best_acc": per_lr[best_lr], "per_lr": per_lr}
                seed_rec["holo_off_lr_control"] = {
                    "candidates": corner_lrs, "best_of_off": best_of_off}
                out["corner_lr_control"][f"{cell_key}_seed{seed}"] = seed_rec["holo_off_lr_control"]

            out["grid"][cell_key]["seeds"][str(seed)] = seed_rec

        # write partial output after each cell (cheap insurance against a long grid dying midway)
        _write(out, args.out)

    out["elapsed_s"] = round(time.time() - t0, 1)
    analysis = analyze(out, eval_Ps, eval_Gs)
    out["analysis"] = analysis
    _write(out, args.out)

    print("\n" + "=" * 78)
    print("VERDICT")
    print(analysis["verdict"])
    print(f"\n-> {args.out}  ({out['elapsed_s']}s)")


def _write(out, path):
    os.makedirs(RESULTS, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# Analysis: rent per cell, ratio-collapse (rent vs d/P_max, Spearman), P32 checks.
# ═══════════════════════════════════════════════════════════════════════════
def _spearman(xs, ys):
    """Spearman rank correlation, stdlib-only (no scipy dependency in this repo path)."""
    n = len(xs)
    if n < 2:
        return None

    def _ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks

    rx = _ranks(xs)
    ry = _ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    varx = sum((r - mx) ** 2 for r in rx)
    vary = sum((r - my) ** 2 for r in ry)
    if varx == 0 or vary == 0:
        return None
    return cov / (varx * vary) ** 0.5


def analyze(out, eval_Ps, eval_Gs):
    grid = out["grid"]
    corner_keys = {f"Pmax{p}_d{d}" for p, d in CORNER_CELLS}

    per_cell = {}       # cell_key -> {P_max, d_model, ratio, rent, lr_sensitive, off_profile}
    rent_table = {}      # "P_max=..,d=.." -> rent
    for cell_key, cell in grid.items():
        P_max, d_model = cell["P_max"], cell["d_model"]
        ratio = d_model / P_max
        seed_rents = []
        off_accs_all = []
        lr_sensitive = False
        for seed_k, seed_rec in cell["seeds"].items():
            on_eval = seed_rec["holo_on"]["eval"]
            off_eval = seed_rec["holo_off"]["eval"]

            # per-(P,G) off accuracy: substitute best-of-off if lr control ran AND it
            # actually changes accuracy at this corner/seed.
            eff_off_eval = dict(off_eval)
            lrc = seed_rec.get("holo_off_lr_control")
            if lrc is not None:
                for pg_key, best in lrc["best_of_off"].items():
                    base_acc = off_eval[pg_key]["accuracy"]
                    if abs(best["best_acc"] - base_acc) > 1e-9:
                        # does substituting best-of-off flip the SIGN of rent at this (P,G)?
                        on_acc = on_eval[pg_key]["accuracy"]
                        rent_base = on_acc - base_acc
                        rent_best = on_acc - best["best_acc"]
                        if (rent_base > 0) != (rent_best > 0):
                            lr_sensitive = True
                        eff_off_eval[pg_key] = {"accuracy": best["best_acc"]}

            diffs = [on_eval[k]["accuracy"] - eff_off_eval[k]["accuracy"] for k in on_eval]
            seed_rents.append(sum(diffs) / len(diffs))
            off_accs_all.extend(v["accuracy"] for v in eff_off_eval.values())

        rent = sum(seed_rents) / len(seed_rents)
        per_cell[cell_key] = {
            "P_max": P_max, "d_model": d_model, "ratio_d_over_pmax": round(ratio, 4),
            "rent_pp": round(rent * 100, 2), "seed_rents_pp": [round(r * 100, 2) for r in seed_rents],
            "lr_sensitive": lr_sensitive, "off_mean_acc": round(sum(off_accs_all) / len(off_accs_all), 4),
            "is_corner": cell_key in corner_keys,
        }
        rent_table[f"P_max={P_max},d={d_model}"] = round(rent * 100, 2)

    # ratio-collapse analysis: EXCLUDE lr_sensitive cells from the correlation/means,
    # report them separately (per the brief: "not averaged in").
    clean = {k: v for k, v in per_cell.items() if not v["lr_sensitive"]}
    flagged = {k: v for k, v in per_cell.items() if v["lr_sensitive"]}

    ratios = [v["ratio_d_over_pmax"] for v in clean.values()]
    rents = [v["rent_pp"] for v in clean.values()]
    spearman_rho = _spearman(ratios, rents) if len(clean) >= 2 else None

    # group clean cells by ratio (rounded to 3 decimals to merge float-identical ratios)
    by_ratio = {}
    for v in clean.values():
        by_ratio.setdefault(v["ratio_d_over_pmax"], []).append(v["rent_pp"])
    collapse_spread = {r: {"n": len(vs), "mean_pp": round(sum(vs) / len(vs), 2),
                            "spread_pp": round(max(vs) - min(vs), 2) if len(vs) > 1 else 0.0}
                        for r, vs in sorted(by_ratio.items())}
    collapse_holds = all(s["spread_pp"] < 10.0 for s in collapse_spread.values())

    # P32 check (a): transition band d/P_max ~ 2-4 (rent<5pp at ratio<=1, rent>=10pp at ratio>=4)
    low_ratio_cells = [v for v in clean.values() if v["ratio_d_over_pmax"] <= 1.0]
    high_ratio_cells = [v for v in clean.values() if v["ratio_d_over_pmax"] >= 4.0]
    low_ok = all(v["rent_pp"] < 5.0 for v in low_ratio_cells) if low_ratio_cells else None
    high_ok = all(v["rent_pp"] >= 10.0 for v in high_ratio_cells) if high_ratio_cells else None

    # P32 check (c): off-arm profile (does off degrade with larger d, i.e. the
    # capacity_return Scheinriese pattern, or stay flat?)
    off_by_d = {}
    for v in clean.values():
        off_by_d.setdefault(v["d_model"], []).append(v["off_mean_acc"])
    off_profile = {d: round(sum(a) / len(a), 4) for d, a in sorted(off_by_d.items())}

    checks = {
        "a_transition_band": {
            "low_ratio_le_1_all_under_5pp": low_ok,
            "high_ratio_ge_4_all_over_10pp": high_ok,
            "low_ratio_cells": [(v["P_max"], v["d_model"], v["rent_pp"]) for v in low_ratio_cells],
            "high_ratio_cells": [(v["P_max"], v["d_model"], v["rent_pp"]) for v in high_ratio_cells],
        },
        "b_collapse": {
            "holds_lt_10pp_spread_within_ratio": collapse_holds,
            "by_ratio": collapse_spread,
        },
        "c_off_arm_profile_by_d": off_profile,
    }

    verdict_bits = []
    verdict_bits.append(f"Spearman(rent, d/P_max) = {spearman_rho}" if spearman_rho is not None
                         else "Spearman: insufficient clean cells")
    verdict_bits.append(f"ratio-collapse (<10pp spread within ratio): {'HOLDS' if collapse_holds else 'FAILS'}")
    verdict_bits.append(f"transition band a: low-ratio<5pp={low_ok}, high-ratio>=10pp={high_ok}")
    if flagged:
        verdict_bits.append(f"lr-sensitive cells excluded from collapse stats: {list(flagged.keys())}")
    verdict = " | ".join(verdict_bits)

    return {
        "per_cell": per_cell,
        "rent_table": rent_table,
        "clean_cells": list(clean.keys()),
        "lr_sensitive_cells": list(flagged.keys()),
        "spearman_rent_vs_ratio": spearman_rho,
        "collapse_by_ratio": collapse_spread,
        "checks": checks,
        "verdict": verdict,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Merge: combine several partial --cells JSONs (written by parallel processes)
# into one results/holo_rent_map.json and recompute the analysis over the union.
# ═══════════════════════════════════════════════════════════════════════════
def run_merge(args):
    paths = [p.strip() for p in args.merge.split(",") if p.strip()]
    merged = {"config": None, "cells_run": [], "grid": {}, "corner_lr_control": {}, "elapsed_s": 0.0}
    eval_Ps = eval_Gs = None
    for p in paths:
        with open(p) as f:
            part = json.load(f)
        if merged["config"] is None:
            merged["config"] = part["config"]
            eval_Ps = [int(x) for x in part["config"]["eval_pairs"].split(",")]
            eval_Gs = [int(x) for x in part["config"]["eval_gaps"].split(",")]
        merged["cells_run"].extend(part.get("cells_run", []))
        merged["grid"].update(part.get("grid", {}))
        merged["corner_lr_control"].update(part.get("corner_lr_control", {}))
        merged["elapsed_s"] += part.get("elapsed_s", 0.0)
    merged["elapsed_s"] = round(merged["elapsed_s"], 1)
    merged["analysis"] = analyze(merged, eval_Ps, eval_Gs)
    _write(merged, args.out)
    print(f"merged {len(paths)} partial file(s) -> {args.out}")
    print(merged["analysis"]["verdict"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="4 anchor cells {(16,64),(64,64),(8,32),(64,256)}, 1 seed, reduced iters")
    ap.add_argument("--full", action="store_true", help="the full 4x4 grid, 2 seeds, v3 iters")
    ap.add_argument("--pmax-grid", default="8,16,32,64")
    ap.add_argument("--d-grid", default="32,64,128,256")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--eval-pairs", default="2,4", help="P values to evaluate at each cell")
    ap.add_argument("--eval-gaps", default="8,32", help="G values to evaluate at each cell")
    ap.add_argument("--iters", type=int, default=V3_ITERS)
    ap.add_argument("--lr", type=float, default=V3_LR)
    ap.add_argument("--corner-lrs", default="1e-3,3e-3",
                    help="comma list of lr candidates for the holo_off Scheinriesen "
                         "control at the 4 corner cells (v3 --lr is always included)")
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--cells", default="",
                    help="subset as 'Pmax:d,Pmax:d,...' (parallelization across processes); "
                         "empty = full pmax-grid x d-grid")
    ap.add_argument("--out", default=os.path.join(RESULTS, "holo_rent_map.json"))
    ap.add_argument("--merge", default="",
                    help="comma list of partial JSON paths to merge into --out "
                         "(runs the merge path instead of a sweep)")
    args = ap.parse_args()

    if args.merge:
        run_merge(args)
        return

    if args.smoke:
        args.cells = "16:64,64:64,8:32,64:256"
        args.seeds = "0"
        args.iters = min(args.iters, 500)
        if args.out == os.path.join(RESULTS, "holo_rent_map.json"):
            args.out = os.path.join(RESULTS, "holo_rent_map_smoke.json")
    elif args.full:
        args.cells = ""
        args.seeds = "0,1"
        args.iters = max(args.iters, V3_ITERS)

    run_grid(args)


if __name__ == "__main__":
    main()
