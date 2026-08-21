#!/usr/bin/env python3 -u
"""
Stack MQAR Run — Baseline vs K2-Only vs Ginibre-Only vs BOTH
=============================================================

Compares four arms on MQAR (n_pairs=8, n_keys=n_values=64, train_len=64,
1500 steps, tanh_m readout, 5 seeds {1,7,42,123,2024}, CPU-deterministic):

  ARM 0  baseline   : use_phase=True, use_vector_key=False, n_freqs=1, λ=0
                      Per-channel 1D holographic read. Expected: ~8-9%.
                      This IS the ~8.89% wall arm (same as the original run).

  ARM 1  K2_only    : use_phase=True, use_vector_key=True,  n_freqs=2, λ=0
                      D-vec matched-filter + K=2 harmonics, no repulsion.
                      Tests whether dual-harmonic write survives D-vec read.

  ARM 2  gin_only   : use_phase=True, use_vector_key=True,  n_freqs=1, λ=λ_best
                      D-vec matched-filter + single write + Ginibre repulsion.
                      λ swept over {0.1, 0.3, 1.0} — use best by ⟨s²⟩ metric.

  ARM 3  both       : use_phase=True, use_vector_key=True,  n_freqs=2, λ=λ_best
                      Full stack: D-vec + K=2 harmonics + Ginibre repulsion.
                      Question: synergy (+) or cancellation (~)?

GATES (all must pass before trusting numbers):
  1. Attention baseline ≥ 0.90 (harness validity — mqar.py spec sanity #4).
  2. holo_off arm (use_phase=False) ≤ 2% (Selective floor, no holographic path).
  3. Baseline arm ∈ [6%, 14%] (reproduce the ~8-9% wall).

KEY DIAGNOSTIC: ⟨s²⟩ of the key cloud per arm (Ginibre target ≈ 1.087).
Arms without repulsion: ⟨s²⟩ ≈ 1.0 (Poisson / random).
Arms with repulsion: we want ⟨s²⟩ to actually REACH ~1.087. Only trust a
Ginibre recall verdict from a run where ⟨s²⟩ ≥ 1.05.

SAME-RUN COMPARISON: every arm is run with the same 5 seeds. An arm "beats
baseline" only if arm_recall > baseline_recall + 1σ(baseline_std) on the
SAME seed set (within-run comparison, not against a remembered number).

Usage:
    python3 stack_mqar_run.py [--smoke] [--steps N] [--lambda-rep F]
    nohup python3 -u stack_mqar_run.py > /tmp/stack.log 2>&1 &

    --smoke        : 2 seeds, 300 steps (fast sanity)
    --steps N      : override step count (default 1500)
    --lambda-rep F : fixed λ for gin/both arms (default: sweep {0.1,0.3,1.0})
    --no-lambda-sweep : skip λ sweep, use --lambda-rep directly
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
import torch.nn.functional as F

# ── Path setup ────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "reference"))
sys.path.insert(0, str(HERE))

from mqar import (make_mqar_batch, mqar_accuracy, TinyCausalTransformerLM,
                  mqar_train)
from holographic_stack import (StackedHolographicLM, key_cloud_variance)

# Always CPU for determinism
DEVICE = torch.device("cpu")

# ── Experiment constants ──────────────────────────────────────────────────────
N_KEYS     = 64
N_VALUES   = 64
N_PAIRS    = 8
N_QUERIES  = 8
SEQ_LEN    = 64
BATCH_SIZE = 32
D_MODEL    = 128
N_LAYERS   = 2
N_HEADS    = 4
D_HEAD     = 32
LR         = 3e-3

VOCAB_SIZE = N_KEYS + N_VALUES + 1
MASK_IDX   = VOCAB_SIZE     # never collides with a real token id

SEEDS_FULL  = [1, 7, 42, 123, 2024]
SEEDS_SMOKE = [1, 7]

# Ginibre repulsion: λ sweep candidates (push past the λ=0.03 failure)
LAMBDA_SWEEP = [0.1, 0.3, 1.0]

CHANCE = 1.0 / N_VALUES   # 1/64 = 1.5625%


def make_cfg():
    return dict(batch_size=BATCH_SIZE, seq_len=SEQ_LEN, n_pairs=N_PAIRS,
                n_queries=N_QUERIES, n_keys=N_KEYS, n_values=N_VALUES)


# ─────────────────────────────────────────────────────────────────────────────
# Model factories
# ─────────────────────────────────────────────────────────────────────────────

def build_model(arm: str, lambda_rep: float = 0.0) -> StackedHolographicLM:
    """
    arm one of: "holo_off", "baseline", "k2_only", "gin_only", "both"
    """
    common = dict(
        vocab_size=VOCAB_SIZE, mask_idx=MASK_IDX,
        d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS, d_head=D_HEAD,
        seq_len=SEQ_LEN, dropout=0.0, causal=True, phase_scale=math.pi,
    )
    if arm == "holo_off":
        # use_phase=False → exact GSSM-Selective (no holographic path at all)
        return StackedHolographicLM(**common, use_phase=False,
                                    use_vector_key=False, n_freqs=1, lambda_rep=0.0)
    elif arm == "baseline":
        # Per-channel 1D holographic read (the ~8-9% wall arm).
        # use_vector_key=False → W_match is None, W_out mixes channels independently.
        # n_freqs=1 → single harmonic write (K=1).
        return StackedHolographicLM(**common, use_phase=True,
                                    use_vector_key=False, n_freqs=1, lambda_rep=0.0)
    elif arm == "k2_only":
        # D-vec matched-filter + K=2 harmonics, no repulsion.
        return StackedHolographicLM(**common, use_phase=True,
                                    use_vector_key=True, n_freqs=2, lambda_rep=0.0)
    elif arm == "gin_only":
        # D-vec matched-filter + single write + β=3 Ginibre repulsion.
        return StackedHolographicLM(**common, use_phase=True,
                                    use_vector_key=True, n_freqs=1, lambda_rep=lambda_rep)
    elif arm == "both":
        # Full stack: D-vec + K=2 harmonics + Ginibre repulsion.
        return StackedHolographicLM(**common, use_phase=True,
                                    use_vector_key=True, n_freqs=2, lambda_rep=lambda_rep)
    else:
        raise ValueError(f"unknown arm {arm!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one(model: StackedHolographicLM, cfg: dict, steps: int, seed: int,
              log_every: int = 300) -> dict:
    """Train for `steps` steps. Returns diagnostics dict."""
    model.to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(seed)

    task_losses = []
    rep_losses  = []
    s2_history  = []   # ⟨s²⟩ key-cloud variance at each log step

    for step in range(steps):
        tokens, targets, mask, _ = make_mqar_batch(generator=gen, device=DEVICE, **cfg)
        logits    = model(tokens)
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

        if log_every and (step + 1) % log_every == 0:
            rl = rep_loss.item() if isinstance(rep_loss, torch.Tensor) else 0.0
            task_losses.append(task_loss.item())
            rep_losses.append(rl)

            # ⟨s²⟩ diagnostic: key cloud geometry of layer 0, head 0
            scan0 = model.layers[0].scan
            if scan0.use_phase and scan0.use_vector_key:
                with torch.no_grad():
                    dummy = torch.randn(1, N_PAIRS, D_MODEL)
                    phi_d = math.pi * torch.tanh(
                        scan0.W_key(dummy)
                    ).view(1, N_PAIRS, N_HEADS, D_HEAD)[0, :, 0, :]   # (N_PAIRS, D_HEAD)
                    s2 = key_cloud_variance(phi_d)
                s2_history.append(s2)
                print(f"      [step {step+1}/{steps}] task={task_loss.item():.4f} "
                      f"rep={rl:.5f}  ⟨s²⟩={s2:.4f}")
            else:
                print(f"      [step {step+1}/{steps}] task={task_loss.item():.4f} "
                      f"rep={rl:.5f}")

    return {"task_losses": task_losses, "rep_losses": rep_losses,
            "s2_history": s2_history}


def eval_model(model: StackedHolographicLM, cfg: dict, seed: int,
               n_batches: int = 8) -> tuple:
    """Returns (overall_recall, by_gap)."""
    model.eval()
    overall, by_gap, _ = mqar_accuracy(model, cfg, n_batches=n_batches,
                                        seed=seed + 100, device=DEVICE)
    return overall, by_gap


# ─────────────────────────────────────────────────────────────────────────────
# Attention validity gate
# ─────────────────────────────────────────────────────────────────────────────

def run_attn_gate(cfg: dict, threshold: float = 0.90) -> float:
    print("[gate] Training attention baseline for harness validity...")
    attn = TinyCausalTransformerLM(VOCAB_SIZE, d_model=64, n_layers=2, n_heads=4,
                                    max_len=SEQ_LEN)
    mqar_train(attn, cfg, steps=1000, lr=3e-3, seed=999, device=DEVICE, log_every=500)
    attn.eval()
    overall, _, _ = mqar_accuracy(attn, cfg, n_batches=8, seed=1000, device=DEVICE)
    ok = overall >= threshold
    print(f"[gate] attention recall={overall:.4f}  "
          f"{'PASS ≥0.90' if ok else 'FAIL <0.90 — HARNESS BROKEN'}")
    if not ok:
        raise RuntimeError(
            f"HARNESS BROKEN: attention={overall:.4f} < {threshold}. "
            "Do NOT trust holographic numbers from this run.")
    return overall


# ─────────────────────────────────────────────────────────────────────────────
# Lambda sweep: find best λ for Ginibre arms
# ─────────────────────────────────────────────────────────────────────────────

def lambda_sweep(cfg: dict, steps: int, seed: int, lambdas: list) -> dict:
    """
    Quick sweep over λ values using gin_only arm (1 seed) to find the λ
    that actually pushes ⟨s²⟩ toward ~1.087.

    Returns dict: {lambda: {"s2_final": float, "recall": float}}
    """
    print(f"\n── λ sweep  lambdas={lambdas}  seed={seed}  steps={steps} ──")
    sweep_results = {}
    for lam in lambdas:
        print(f"\n  λ={lam}")
        torch.manual_seed(seed)
        model = build_model("gin_only", lambda_rep=lam)
        train_stats = train_one(model, cfg, steps=steps, seed=seed, log_every=steps)
        recall, _ = eval_model(model, cfg, seed=seed)
        # Final ⟨s²⟩
        scan0 = model.layers[0].scan
        with torch.no_grad():
            dummy = torch.randn(1, N_PAIRS, D_MODEL)
            phi_d = math.pi * torch.tanh(
                scan0.W_key(dummy)
            ).view(1, N_PAIRS, N_HEADS, D_HEAD)[0, :, 0, :]
            s2_final = key_cloud_variance(phi_d)
        print(f"    λ={lam}: recall={recall:.4f}  ⟨s²⟩_final={s2_final:.4f}  "
              f"(target ~1.087)")
        sweep_results[lam] = {"s2_final": s2_final, "recall": recall}
    return sweep_results


def pick_best_lambda(sweep_results: dict) -> float:
    """Pick λ where ⟨s²⟩ is closest to 1.087 from below (not over-regularized)."""
    TARGET = 1.087
    best_lam = None
    best_dist = float('inf')
    for lam, r in sweep_results.items():
        s2 = r["s2_final"]
        # Prefer ⟨s²⟩ in [1.05, 1.3] — reached target without lattice collapse
        if math.isnan(s2):
            continue
        dist = abs(s2 - TARGET)
        if dist < best_dist:
            best_dist = dist
            best_lam = lam
    if best_lam is None:
        best_lam = LAMBDA_SWEEP[-1]  # fallback: strongest λ
    print(f"\n  Best λ={best_lam} (⟨s²⟩={sweep_results[best_lam]['s2_final']:.4f}, "
          f"target={TARGET})")
    return best_lam


# ─────────────────────────────────────────────────────────────────────────────
# Full arm sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_arm_seeds(arm: str, cfg: dict, steps: int, seeds: list,
                  lambda_rep: float = 0.0, log_every: int = 500) -> dict:
    """Run one arm across all seeds. Returns dict with per-seed and aggregate stats."""
    print(f"\n{'='*70}")
    print(f"ARM: {arm}  n_freqs={'2' if 'k2' in arm or arm=='both' else '1'}  "
          f"vec_key={'True' if arm!='baseline' and arm!='holo_off' else 'False'}  "
          f"λ={lambda_rep if 'gin' in arm or arm=='both' else 0.0}")
    print(f"  seeds={seeds}  steps={steps}")
    print('='*70)

    seed_recalls = []
    seed_s2_finals = []
    t0 = time.time()

    for seed in seeds:
        print(f"\n  [seed={seed}]")
        torch.manual_seed(seed)
        model = build_model(arm, lambda_rep=lambda_rep)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    params: {n_params:,}")

        train_stats = train_one(model, cfg, steps=steps, seed=seed, log_every=log_every)
        recall, by_gap = eval_model(model, cfg, seed=seed)
        seed_recalls.append(recall)

        # Final ⟨s²⟩ for this seed
        scan0 = model.layers[0].scan
        s2 = float('nan')
        if scan0.use_phase and scan0.use_vector_key:
            with torch.no_grad():
                dummy = torch.randn(1, N_PAIRS, D_MODEL)
                phi_d = math.pi * torch.tanh(
                    scan0.W_key(dummy)
                ).view(1, N_PAIRS, N_HEADS, D_HEAD)[0, :, 0, :]
                s2 = key_cloud_variance(phi_d)
        seed_s2_finals.append(s2)

        print(f"    recall={recall:.4f}  ⟨s²⟩={s2:.4f}")

    elapsed = time.time() - t0
    mean_r = sum(seed_recalls) / len(seed_recalls)
    std_r  = (sum((r - mean_r)**2 for r in seed_recalls) / len(seed_recalls))**0.5
    s2_vals = [v for v in seed_s2_finals if not math.isnan(v)]
    mean_s2 = sum(s2_vals) / len(s2_vals) if s2_vals else float('nan')

    print(f"\n  {arm}:  recall {mean_r:.4f}±{std_r:.4f}  "
          f"⟨s²⟩_mean={mean_s2:.4f}  seeds={[round(r,4) for r in seed_recalls]}  "
          f"elapsed={elapsed:.0f}s")

    return {
        "arm": arm,
        "lambda_rep": lambda_rep,
        "mean_recall": round(mean_r, 4),
        "std_recall":  round(std_r, 4),
        "seed_recalls": [round(r, 4) for r in seed_recalls],
        "mean_s2":      round(mean_s2, 4) if not math.isnan(mean_s2) else None,
        "seed_s2":      [round(v, 4) if not math.isnan(v) else None for v in seed_s2_finals],
        "seeds":        seeds,
        "elapsed_s":    round(elapsed, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke",           action="store_true",
                    help="Fast smoke: 2 seeds, 300 steps")
    ap.add_argument("--steps",           type=int,   default=None)
    ap.add_argument("--lambda-rep",      type=float, default=None,
                    help="Fixed λ for gin/both arms (skips λ sweep)")
    ap.add_argument("--no-lambda-sweep", action="store_true",
                    help="Skip λ sweep; use --lambda-rep directly (default 0.3)")
    ap.add_argument("--json",            default=None,
                    help="Output JSON path (default: src/stack_mqar_results.json)")
    args = ap.parse_args()

    SEEDS = SEEDS_SMOKE if args.smoke else SEEDS_FULL
    STEPS = args.steps or (300 if args.smoke else 1500)
    LOG_EVERY = 150 if args.smoke else 500

    print("=" * 74)
    print(f"Stack MQAR Experiment  {'(SMOKE)' if args.smoke else '(FULL)'}")
    print(f"  seeds={SEEDS}  steps={STEPS}  n_pairs={N_PAIRS}  device={DEVICE}")
    print(f"  arms: baseline | k2_only | gin_only | both")
    print("=" * 74)

    cfg = make_cfg()
    results = {
        "config": {
            "n_pairs": N_PAIRS, "n_keys": N_KEYS, "n_values": N_VALUES,
            "train_len": SEQ_LEN, "steps": STEPS, "seeds": SEEDS,
            "d_model": D_MODEL, "n_heads": N_HEADS, "d_head": D_HEAD,
            "n_layers": N_LAYERS, "lr": LR, "batch": BATCH_SIZE,
            "chance": CHANCE,
        }
    }

    # ── 0. Attention validity gate ────────────────────────────────────────────
    print("\n--- Attention validity gate ---")
    attn_recall = run_attn_gate(cfg, threshold=0.90)
    results["attn_gate"] = {"recall": round(attn_recall, 4), "ok": True}

    # ── 1. holo_off floor (Selective, use_phase=False) ────────────────────────
    print("\n--- holo_off floor (use_phase=False = Selective) ---")
    holo_off_res = run_arm_seeds("holo_off", cfg, STEPS, SEEDS, log_every=LOG_EVERY)
    results["holo_off"] = holo_off_res
    if holo_off_res["mean_recall"] > 0.02:
        print(f"  WARNING: holo_off={holo_off_res['mean_recall']:.4f} > 2% — "
              f"Selective path has non-zero recall. Check data leak.")

    # ── 2. Determine λ for Ginibre arms ──────────────────────────────────────
    if args.no_lambda_sweep or args.lambda_rep is not None:
        best_lambda = args.lambda_rep if args.lambda_rep is not None else 0.3
        lambda_sweep_results = None
        print(f"\n  Using fixed λ={best_lambda} (λ sweep skipped).")
    else:
        sweep_steps = min(STEPS, 500)   # quick sweep: 500 steps, 1 seed
        lambda_sweep_results = lambda_sweep(cfg, steps=sweep_steps,
                                             seed=SEEDS[0], lambdas=LAMBDA_SWEEP)
        best_lambda = pick_best_lambda(lambda_sweep_results)
        results["lambda_sweep"] = {
            str(lam): r for lam, r in lambda_sweep_results.items()
        }
        results["best_lambda"] = best_lambda

    print(f"\n  Using λ={best_lambda} for gin_only and both arms.")

    # ── 3. Four arm comparison (same seeds) ───────────────────────────────────
    arm_results = {}

    # ARM 0: baseline (1D per-channel, the ~8-9% wall)
    arm_results["baseline"] = run_arm_seeds(
        "baseline", cfg, STEPS, SEEDS, log_every=LOG_EVERY)

    # ARM 1: K2-only (D-vec matched-filter + K=2 harmonics, no repulsion)
    arm_results["k2_only"] = run_arm_seeds(
        "k2_only", cfg, STEPS, SEEDS, lambda_rep=0.0, log_every=LOG_EVERY)

    # ARM 2: Ginibre-only (D-vec + single write + repulsion)
    arm_results["gin_only"] = run_arm_seeds(
        "gin_only", cfg, STEPS, SEEDS, lambda_rep=best_lambda, log_every=LOG_EVERY)

    # ARM 3: BOTH (D-vec + K=2 + repulsion)
    arm_results["both"] = run_arm_seeds(
        "both", cfg, STEPS, SEEDS, lambda_rep=best_lambda, log_every=LOG_EVERY)

    results["arms"] = arm_results

    # ── 4. Summary & verdicts ─────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("RESULTS SUMMARY")
    print("=" * 74)

    baseline_r = arm_results["baseline"]["mean_recall"]
    baseline_s = arm_results["baseline"]["std_recall"]
    k2_r       = arm_results["k2_only"]["mean_recall"]
    k2_s       = arm_results["k2_only"]["std_recall"]
    gin_r      = arm_results["gin_only"]["mean_recall"]
    gin_s      = arm_results["gin_only"]["std_recall"]
    both_r     = arm_results["both"]["mean_recall"]
    both_s     = arm_results["both"]["std_recall"]
    gin_s2     = arm_results["gin_only"]["mean_s2"]
    both_s2    = arm_results["both"]["mean_s2"]
    hoff_r     = holo_off_res["mean_recall"]

    print(f"  chance:         {CHANCE*100:.2f}%")
    print(f"  holo_off:       {hoff_r*100:.2f}%  (Selective, expect ≤2%)")
    print(f"  baseline:       {baseline_r*100:.2f}%±{baseline_s*100:.2f}pp  "
          f"(1D holographic wall, expect 8-9%)")
    print(f"  K2-only:        {k2_r*100:.2f}%±{k2_s*100:.2f}pp  "
          f"(D-vec + K=2 harmonics)")
    print(f"  gin-only:       {gin_r*100:.2f}%±{gin_s*100:.2f}pp  "
          f"(D-vec + λ={best_lambda} repulsion)  ⟨s²⟩={gin_s2}")
    print(f"  BOTH:           {both_r*100:.2f}%±{both_s*100:.2f}pp  "
          f"(D-vec + K=2 + λ={best_lambda})  ⟨s²⟩={both_s2}")

    # Within-run verdicts (>1σ above baseline)
    threshold = baseline_r + baseline_s
    verdicts = {}

    def verdict_arm(name, r, s, s2):
        beats = r > threshold
        delta = r - baseline_r
        s2_ok = (s2 is not None and not math.isnan(s2) and s2 >= 1.05) if s2 else None
        label = ("SYNERGY" if beats and name == "both" and
                  r > max(k2_r, gin_r) + max(k2_s, gin_s)
                  else "BEATS_BASELINE" if beats else "FLAT/CANCEL")
        print(f"\n  {name}: Δ={delta*100:+.2f}pp vs baseline  "
              f"threshold={threshold*100:.2f}%  → {label}")
        if s2 is not None and not math.isnan(s2):
            trusted = s2_ok
            print(f"    ⟨s²⟩={s2:.4f}  Ginibre spread {'REACHED (≥1.05)' if trusted else 'NOT REACHED (<1.05) — verdict unreliable'}")
        verdicts[name] = {
            "beats_baseline_1sigma": beats,
            "delta_pp": round(delta * 100, 2),
            "threshold_pp": round(threshold * 100, 2),
            "mean_recall": round(r, 4),
            "std_recall": round(s, 4),
            "mean_s2": round(s2, 4) if s2 and not math.isnan(s2) else None,
            "s2_reached_target": s2_ok,
            "label": label,
        }

    verdict_arm("k2_only",  k2_r,   k2_s,   None)
    verdict_arm("gin_only", gin_r,   gin_s,  gin_s2)
    verdict_arm("both",     both_r,  both_s, both_s2)

    # Synergy check: BOTH > best single lever + 1σ of that lever
    best_single_r = max(k2_r, gin_r)
    best_single_s = k2_s if k2_r >= gin_r else gin_s
    synergy = both_r > best_single_r + best_single_s
    print(f"\n  Synergy check: BOTH ({both_r*100:.2f}%) vs best-single "
          f"({best_single_r*100:.2f}%±{best_single_s*100:.2f}pp) → "
          f"{'SYNERGY' if synergy else 'NO SYNERGY (gains cancel or flat)'}")
    verdicts["synergy"] = {
        "both_beats_best_single_by_1sigma": synergy,
        "best_single": "k2_only" if k2_r >= gin_r else "gin_only",
        "best_single_recall": round(best_single_r, 4),
        "delta_pp": round((both_r - best_single_r) * 100, 2),
    }

    # Ginibre spread check
    if gin_s2 and not math.isnan(gin_s2) and gin_s2 < 1.05:
        print(f"\n  NOTE: gin_only ⟨s²⟩={gin_s2:.4f} < 1.05. "
              f"Ginibre spread did NOT reach target even at λ={best_lambda}. "
              f"The repulsion verdict is from a run where keys were NOT spread. "
              f"Push λ higher (try 3.0, 10.0) or increase training steps.")

    results["verdicts"] = verdicts
    results["summary"] = {
        "chance_pct": round(CHANCE * 100, 2),
        "holo_off_pct": round(hoff_r * 100, 2),
        "baseline_pct": round(baseline_r * 100, 2),
        "k2_only_pct": round(k2_r * 100, 2),
        "gin_only_pct": round(gin_r * 100, 2),
        "both_pct": round(both_r * 100, 2),
        "best_lambda": best_lambda,
        "gin_s2": round(gin_s2, 4) if gin_s2 and not math.isnan(gin_s2) else None,
        "both_s2": round(both_s2, 4) if both_s2 and not math.isnan(both_s2) else None,
        "synergy": synergy,
        "attn_gate": round(attn_recall, 4),
    }

    # ── 5. Write JSON ─────────────────────────────────────────────────────────
    json_path = args.json or str(HERE / "stack_mqar_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to: {json_path}")

    return results


if __name__ == "__main__":
    main()
