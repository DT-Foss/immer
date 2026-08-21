#!/usr/bin/env python3 -u
"""
Ginibre λ sweep — close the open thread decisively
====================================================

Sweeps λ_rep ∈ {0.03 (control), 0.1, 0.3, 1.0} on β=3 Ginibre repulsion,
5 seeds {1, 7, 42, 123, 2024}, 1500 steps, n_pairs=8, tanh_m readout, CPU.

CRITICAL DIAGNOSTIC: logs key-cloud ⟨s²⟩ for each λ.
- ⟨s²⟩ ≈ 1.0   → Poisson (keys not spreading — λ too small)
- ⟨s²⟩ ≈ 1.087 → Ginibre target (correct repulsion regime)
- ⟨s²⟩ >> 1.4  → over-regularized lattice (λ too big)

METHODOLOGY FIX (the overnight bug):
  - 5 seeds per arm instead of 3 (shrinks ±σ by ~40%)
  - baseline_1d arm runs in SAME RUN as repulsion arms (same seeds, same harness state)
  - "beats baseline" = beats SAME-RUN baseline by >1σ, not a remembered number.

VERDICT LOGIC:
  - If at the λ where ⟨s²⟩ ≈ 1.087, recall > baseline + 1σ → Ginibre wins (real signal).
  - If ⟨s²⟩ reaches 1.087 but recall STILL doesn't beat baseline → CLOSED NEGATIVE.
  - If no λ reaches ⟨s²⟩ ≈ 1.087 → sweep range insufficient (need even higher λ).

Usage:
    nohup python3 -u src/ginibre_lambda_sweep.py > /tmp/ginibre_lambda.log 2>&1 &
    # or with JSON output:
    python3 src/ginibre_lambda_sweep.py --json /tmp/ginibre_lambda_results.json
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Path setup ────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "reference"))
sys.path.insert(0, str(HERE))

from mqar import (make_mqar_batch, mqar_accuracy, TinyCausalTransformerLM,
                  mqar_train, DEVICE as MQAR_DEVICE)
from holographic_ginibre import (GinibreHolographicLM, _verify_reduction,
                                 key_cloud_variance)

# Always CPU for reproducibility
DEVICE = torch.device("cpu")

# ── Hyperparameters (must match the holographic baseline that gave 8.89%) ────
N_KEYS      = 64
N_VALUES    = 64
N_PAIRS     = 8
N_QUERIES   = 8
SEQ_LEN     = 64
BATCH_SIZE  = 32
D_MODEL     = 128
N_LAYERS    = 2
N_HEADS     = 4
D_HEAD      = 32
LR          = 3e-3
STEPS       = 1500
LOG_EVERY   = 300

# The λ sweep — control + 3 escalating values
LAMBDA_SWEEP = [0.03, 0.1, 0.3, 1.0]

# 5 seeds (fixes overnight ±1.8% noise floor, brings σ down ~40%)
SEEDS = [1, 7, 42, 123, 2024]

# Ginibre diagnostic target
GINIBRE_TARGET_S2 = 1.087
# Tolerance for "reached target"
GINIBRE_TOL = 0.05   # ⟨s²⟩ in [1.037, 1.137] counts as "at target"


def make_cfg():
    return dict(batch_size=BATCH_SIZE, seq_len=SEQ_LEN, n_pairs=N_PAIRS,
                n_queries=N_QUERIES, n_keys=N_KEYS, n_values=N_VALUES)


VOCAB_SIZE = N_KEYS + N_VALUES + 1
MASK_IDX   = VOCAB_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────────────────────────────────────

def build_baseline_1d() -> GinibreHolographicLM:
    """Arm A — per-channel read, effective D=2. Reproduces ~8-9% wall."""
    return GinibreHolographicLM(
        vocab_size=VOCAB_SIZE, mask_idx=MASK_IDX,
        d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS, d_head=D_HEAD,
        seq_len=SEQ_LEN, dropout=0.0, causal=True,
        phase_scale=math.pi,
        use_phase=True, repulsion=False, baseline_1d=True,
    )


def build_ginibre_repulsion(lambda_rep: float) -> GinibreHolographicLM:
    """Arm B/C — vector-key matched-filter + β=3 Ginibre repulsion at λ."""
    return GinibreHolographicLM(
        vocab_size=VOCAB_SIZE, mask_idx=MASK_IDX,
        d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS, d_head=D_HEAD,
        seq_len=SEQ_LEN, dropout=0.0, causal=True,
        phase_scale=math.pi,
        use_phase=True, repulsion=True, lambda_rep=lambda_rep, baseline_1d=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ⟨s²⟩ extraction — sample the key-cloud from a real forward pass
# ─────────────────────────────────────────────────────────────────────────────

def sample_key_cloud_s2(model: GinibreHolographicLM, cfg: dict,
                         seed: int) -> float:
    """Run one eval batch through the model, collect key-cloud ⟨s²⟩ from layer 0."""
    model.eval()
    gen = torch.Generator().manual_seed(seed + 9999)
    with torch.no_grad():
        tokens, targets, mask, _ = make_mqar_batch(generator=gen, device=DEVICE, **cfg)
        h = model.pos(model.embed(tokens))   # (B, T, d_model)
        # Extract phi from layer 0's scan module
        scan0 = model.layers[0].scan
        phi_raw = model.layers[0].scan.W_key(h)  # (B, T, n_heads*d_head)
        phi = (math.pi * torch.tanh(phi_raw)
               .view(tokens.size(0), SEQ_LEN, N_HEADS, D_HEAD))
        # Use b=0, all T positions, h=0 — (T, D) array of key phase vectors
        phi_sample = phi[0, :, 0, :].detach()   # (T=64, D=32)
        return key_cloud_variance(phi_sample)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model: GinibreHolographicLM, cfg: dict, steps: int,
                seed: int, arm_label: str) -> dict:
    """Train for `steps` steps. Returns dict with final ⟨s²⟩ and training stats."""
    model.to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(seed)

    task_losses  = []
    rep_losses   = []
    s2_log       = []    # ⟨s²⟩ checkpoints at LOG_EVERY intervals

    for step in range(steps):
        tokens, targets, mask, _ = make_mqar_batch(generator=gen, device=DEVICE, **cfg)
        logits = model(tokens)
        task_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1), reduction='none')
        task_loss = (task_loss * mask.reshape(-1).float()).sum() / (mask.sum() + 1e-6)

        rep_loss   = model.get_repulsion_loss()
        total_loss = task_loss + rep_loss

        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if (step + 1) % LOG_EVERY == 0:
            rl = rep_loss.item() if isinstance(rep_loss, torch.Tensor) else 0.0
            tl = task_loss.item()
            task_losses.append(tl)
            rep_losses.append(rl)

            # ⟨s²⟩ from REAL batch phi (not dummy)
            scan0 = model.layers[0].scan
            if scan0.use_phase and not scan0.baseline_1d:
                with torch.no_grad():
                    phi_raw = scan0.W_key(
                        model.pos(model.embed(tokens)).detach()
                    )   # (B, T, H*D)
                    phi = (math.pi * torch.tanh(phi_raw)
                           .view(BATCH_SIZE, SEQ_LEN, N_HEADS, D_HEAD))
                    phi_sample = phi[0, :, 0, :]   # (T, D)
                    s2 = key_cloud_variance(phi_sample)
                s2_log.append({"step": step + 1, "s2": round(s2, 4)})
                print(f"    [{arm_label} seed={seed} step={step+1}/{steps}] "
                      f"task={tl:.4f} rep={rl:.5f} ⟨s²⟩={s2:.4f}")
            else:
                print(f"    [{arm_label} seed={seed} step={step+1}/{steps}] "
                      f"task={tl:.4f}")

    return {
        "task_losses":  task_losses,
        "rep_losses":   rep_losses,
        "s2_log":       s2_log,
        "final_s2":     s2_log[-1]["s2"] if s2_log else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single seed run: baseline + one λ value
# ─────────────────────────────────────────────────────────────────────────────

def run_seed(lambda_rep: float, cfg: dict, seed: int) -> dict:
    """Train baseline_1d + ginibre_repulsion(λ) for ONE seed. Return both recalls."""
    print(f"\n  ── seed={seed}  λ={lambda_rep} ──────────────────────────────────────")

    # Baseline arm (same seed → same data ordering → fair comparison)
    t0 = time.time()
    baseline_model = build_baseline_1d()
    train_stats_b = train_model(baseline_model, cfg, STEPS, seed, f"baseline λ_ctrl")
    baseline_model.eval()
    baseline_recall, _, _ = mqar_accuracy(baseline_model, cfg, n_batches=8,
                                           seed=seed + 100, device=DEVICE)
    elapsed_b = time.time() - t0
    print(f"    baseline recall = {baseline_recall:.4f}  [{elapsed_b:.1f}s]")

    # Ginibre arm
    t1 = time.time()
    gin_model = build_ginibre_repulsion(lambda_rep)
    train_stats_g = train_model(gin_model, cfg, STEPS, seed, f"ginibre λ={lambda_rep}")
    gin_model.eval()

    # ⟨s²⟩ from real eval batch AFTER training
    final_s2 = sample_key_cloud_s2(gin_model, cfg, seed)
    train_stats_g["final_s2_eval"] = round(final_s2, 4)

    gin_recall, _, _ = mqar_accuracy(gin_model, cfg, n_batches=8,
                                      seed=seed + 100, device=DEVICE)
    elapsed_g = time.time() - t1
    print(f"    ginibre recall  = {gin_recall:.4f}  ⟨s²⟩={final_s2:.4f}  [{elapsed_g:.1f}s]")
    print(f"    delta = {(gin_recall - baseline_recall)*100:+.2f}pp  "
          f"(ginibre − baseline, same seed)")

    return {
        "seed":             seed,
        "lambda_rep":       lambda_rep,
        "baseline_recall":  round(float(baseline_recall), 4),
        "gin_recall":       round(float(gin_recall), 4),
        "delta_recall":     round(float(gin_recall) - float(baseline_recall), 4),
        "final_s2":         round(final_s2, 4),
        "s2_log":           train_stats_g["s2_log"],
        "baseline_train":   train_stats_b,
        "gin_train":        train_stats_g,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Attention validity gate
# ─────────────────────────────────────────────────────────────────────────────

def run_attn_gate(cfg: dict) -> float:
    print("\n[gate] Training attention baseline for validity check...")
    model = TinyCausalTransformerLM(VOCAB_SIZE, d_model=64, n_layers=2, n_heads=4,
                                    max_len=SEQ_LEN)
    mqar_train(model, cfg, steps=800, lr=3e-3, seed=999, device=DEVICE, log_every=400)
    model.eval()
    overall, _, _ = mqar_accuracy(model, cfg, n_batches=8, seed=1000, device=DEVICE)
    ok = overall >= 0.90
    print(f"[gate] Attention recall={overall:.4f}  {'PASS ≥0.90' if ok else 'FAIL <0.90 — HARNESS BROKEN'}")
    return overall


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="Write results JSON to path")
    ap.add_argument("--skip-gate", action="store_true",
                    help="Skip attention validity gate (for debugging only)")
    args = ap.parse_args()

    print("=" * 74)
    print("Ginibre λ Sweep — CLOSE the open thread")
    print(f"  λ sweep:  {LAMBDA_SWEEP}")
    print(f"  seeds:    {SEEDS}")
    print(f"  steps:    {STEPS}")
    print(f"  n_pairs:  {N_PAIRS}")
    print(f"  device:   {DEVICE}")
    print(f"  target ⟨s²⟩: {GINIBRE_TARGET_S2}")
    print("=" * 74)

    cfg = make_cfg()
    results = {
        "config": {
            "lambda_sweep": LAMBDA_SWEEP,
            "seeds": SEEDS,
            "steps": STEPS,
            "n_pairs": N_PAIRS,
            "n_keys": N_KEYS,
            "n_values": N_VALUES,
            "d_head": D_HEAD,
            "ginibre_target_s2": GINIBRE_TARGET_S2,
        }
    }

    # ── 0. Reduction self-test ────────────────────────────────────────────────
    print("\n--- Reduction self-test ---")
    red_ok, red_err = _verify_reduction(device="cpu", tol=1e-5)
    results["reduction_test"] = {"max_delta": round(red_err, 8), "ok": red_ok}
    if not red_ok:
        print(f"  FATAL: reduction test FAILED (max|Δ|={red_err:.3e}). Abort.")
        sys.exit(1)

    # ── 1. Attention validity gate ────────────────────────────────────────────
    if not args.skip_gate:
        attn_recall = run_attn_gate(cfg)
        results["attn_gate"] = {"recall": round(float(attn_recall), 4),
                                "ok": bool(attn_recall >= 0.90)}
        if attn_recall < 0.90:
            print("FATAL: Attention gate FAILED. Abort.")
            sys.exit(1)
    else:
        print("\n[skip-gate] Skipping attention gate.")
        results["attn_gate"] = {"recall": None, "ok": None, "note": "skipped"}

    # ── 2. Lambda sweep ───────────────────────────────────────────────────────
    sweep_results = {}

    for lam in LAMBDA_SWEEP:
        label = f"lambda_{lam:.2f}".replace(".", "_")
        print(f"\n{'='*74}")
        print(f"λ = {lam}  (β=3 repulsion, 5 seeds)")
        print('='*74)

        seed_results = []
        for seed in SEEDS:
            r = run_seed(lam, cfg, seed)
            seed_results.append(r)

        baseline_recalls = [r["baseline_recall"] for r in seed_results]
        gin_recalls      = [r["gin_recall"]      for r in seed_results]
        deltas           = [r["delta_recall"]     for r in seed_results]
        s2_finals        = [r["final_s2"]         for r in seed_results]

        def mean_std(xs):
            m = sum(xs) / len(xs)
            s = (sum((x - m)**2 for x in xs) / max(1, len(xs))) ** 0.5
            return round(m, 4), round(s, 4)

        bm, bs = mean_std(baseline_recalls)
        gm, gs = mean_std(gin_recalls)
        dm, ds = mean_std(deltas)
        s2m, s2s = mean_std(s2_finals)

        beats_baseline_1sigma = dm > ds   # delta > 1σ(delta) → real signal
        s2_at_target = abs(s2m - GINIBRE_TARGET_S2) <= GINIBRE_TOL

        print(f"\n  λ={lam}  SUMMARY:")
        print(f"    baseline recall = {bm:.4f} ± {bs:.4f}")
        print(f"    ginibre  recall = {gm:.4f} ± {gs:.4f}")
        print(f"    delta           = {dm:+.4f} ± {ds:.4f}  "
              f"{'> 1σ — REAL SIGNAL' if beats_baseline_1sigma else '≤ 1σ — NOISE'}")
        print(f"    ⟨s²⟩           = {s2m:.4f} ± {s2s:.4f}  "
              f"({'AT GINIBRE TARGET ✓' if s2_at_target else f'AWAY from {GINIBRE_TARGET_S2}'})")

        sweep_results[label] = {
            "lambda":             lam,
            "baseline_recall_mean": bm,
            "baseline_recall_std":  bs,
            "gin_recall_mean":    gm,
            "gin_recall_std":     gs,
            "delta_mean":         dm,
            "delta_std":          ds,
            "s2_mean":            s2m,
            "s2_std":             s2s,
            "beats_baseline_1sigma": beats_baseline_1sigma,
            "s2_at_ginibre_target":  s2_at_target,
            "seeds":              seed_results,
        }

    results["sweep"] = sweep_results

    # ── 3. Verdict ────────────────────────────────────────────────────────────
    print("\n" + "="*74)
    print("FINAL VERDICT — Ginibre λ thread")
    print("="*74)

    # Find the λ where ⟨s²⟩ is closest to target
    best_lambda_for_s2 = None
    best_s2_dist = float('inf')
    for label, res in sweep_results.items():
        dist = abs(res["s2_mean"] - GINIBRE_TARGET_S2)
        if dist < best_s2_dist:
            best_s2_dist = dist
            best_lambda_for_s2 = res

    print(f"\n  λ values tested:  {LAMBDA_SWEEP}")
    print(f"\n  Per-λ ⟨s²⟩ and recall vs same-run baseline:")
    print(f"  {'λ':>8} | {'⟨s²⟩':>8} | {'baseline':>10} | {'ginibre':>10} | {'Δ':>8} | {'vs 1σ':>10}")
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}")
    for label, res in sweep_results.items():
        flag = "REAL" if res["beats_baseline_1sigma"] else "noise"
        s2_flag = "TARGET" if res["s2_at_ginibre_target"] else ""
        print(f"  {res['lambda']:>8.2f} | {res['s2_mean']:>8.4f} | "
              f"{res['baseline_recall_mean']*100:>9.2f}% | "
              f"{res['gin_recall_mean']*100:>9.2f}% | "
              f"{res['delta_mean']*100:>+7.2f}pp | "
              f"{flag:>10}  {s2_flag}")

    # Decisive verdict
    if best_lambda_for_s2 is not None:
        lam_val  = best_lambda_for_s2["lambda"]
        s2_val   = best_lambda_for_s2["s2_mean"]
        delta    = best_lambda_for_s2["delta_mean"]
        delta_s  = best_lambda_for_s2["delta_std"]
        beats    = best_lambda_for_s2["beats_baseline_1sigma"]
        on_target = best_lambda_for_s2["s2_at_ginibre_target"]

        print(f"\n  Closest λ to Ginibre target: λ={lam_val}  ⟨s²⟩={s2_val:.4f}")
        if not on_target:
            print(f"  WARNING: ⟨s²⟩={s2_val:.4f} NEVER reached {GINIBRE_TARGET_S2} (±{GINIBRE_TOL}).")
            print(f"  The sweep did not push keys to the Ginibre regime.")
            print(f"  Cannot definitively close — need even higher λ (try λ=3.0, 10.0).")
            verdict = "INCONCLUSIVE — ⟨s²⟩ never reached Ginibre target; need larger λ"
        elif beats:
            print(f"\n  RESULT: AT ⟨s²⟩={s2_val:.4f} (Ginibre regime), "
                  f"recall delta = {delta*100:+.2f}pp ± {delta_s*100:.2f}pp > 1σ.")
            print(f"  VERDICT: Ginibre key spreading HELPS. Thread OPEN — "
                  f"real win at λ={lam_val}. Fold into stack.")
            verdict = f"OPEN — Ginibre wins at λ={lam_val}: delta={delta*100:+.2f}pp > 1σ at ⟨s²⟩={s2_val:.4f}"
        else:
            print(f"\n  RESULT: AT ⟨s²⟩={s2_val:.4f} (Ginibre regime), "
                  f"recall delta = {delta*100:+.2f}pp ± {delta_s*100:.2f}pp ≤ 1σ.")
            print(f"  VERDICT: CLOSED NEGATIVE. Keys reached Ginibre spacing but "
                  f"recall did NOT improve. Ginibre repulsion is not load-bearing here.")
            verdict = f"CLOSED NEGATIVE — ⟨s²⟩={s2_val:.4f} at target but delta={delta*100:+.2f}pp ≤ 1σ"
    else:
        verdict = "ERROR — no λ results found"

    results["verdict"] = {
        "statement":           verdict,
        "best_lambda_for_s2":  best_lambda_for_s2["lambda"] if best_lambda_for_s2 else None,
        "best_s2":             best_lambda_for_s2["s2_mean"] if best_lambda_for_s2 else None,
        "s2_reached_target":   bool(best_lambda_for_s2["s2_at_ginibre_target"]) if best_lambda_for_s2 else False,
        "beats_baseline_at_target": bool(best_lambda_for_s2["beats_baseline_1sigma"]) if best_lambda_for_s2 else False,
    }

    # ── 4. Write JSON ─────────────────────────────────────────────────────────
    json_path = args.json or str(HERE / "ginibre_lambda_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to: {json_path}")

    print("\n" + "="*74)
    print("DONE")
    print("="*74)
    return results


if __name__ == "__main__":
    main()
