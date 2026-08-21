#!/usr/bin/env python3 -u
"""
K=2 vs K=1 confirmation run — 5 seeds, 2000 steps, same-run baseline.
======================================================================

Spec from THREAD 2:
- K ∈ {1, 2} only
- Seeds {1, 7, 42, 123, 2024}
- 2000 steps (more budget for tighter estimate)
- n_pairs=8, n_keys=64, n_values=64, train_len=64
- tanh_m readout (load-bearing)
- CPU-deterministic
- Same-run baseline: K=1 arm IS the baseline; every comparison is within-run
- Attn validity gate ≥0.90 or VOID
- Question: is K=2 > K=1 by >1σ?

Output → /tmp/k2_confirm_results.json
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import os
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "reference"))

DEVICE = torch.device("cpu")

from mqar import make_mqar_batch, mqar_accuracy, mqar_train, TinyCausalTransformerLM
from holographic_multifreq import HolographicMultiFreqLM

# ── Config ────────────────────────────────────────────────────────────────────
N_PAIRS   = 8
N_KEYS    = 64
N_VALUES  = 64
TRAIN_LEN = 64
BATCH     = 32
LR        = 3e-3
STEPS     = 2000
SEEDS     = [1, 7, 42, 123, 2024]
KS        = [1, 2]

VOCAB_SIZE = N_KEYS + N_VALUES + 1
MASK_IDX   = VOCAB_SIZE

TRAIN_CFG = dict(batch_size=BATCH, seq_len=TRAIN_LEN, n_pairs=N_PAIRS,
                 n_queries=N_PAIRS, n_keys=N_KEYS, n_values=N_VALUES)

D_MODEL  = 128
N_HEADS  = 4
D_HEAD   = 32
N_LAYERS = 2


def build_model(K, use_phase=True):
    return HolographicMultiFreqLM(
        vocab_size=VOCAB_SIZE, mask_idx=MASK_IDX,
        d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS, d_head=D_HEAD,
        seq_len=TRAIN_LEN, dropout=0.0, causal=True,
        phase_scale=math.pi, use_phase=use_phase, readout="tanh_m", n_freqs=K)


def run_one(K, seed, steps):
    torch.manual_seed(seed)
    model = build_model(K, use_phase=True)
    model.to(DEVICE).train()

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(seed)

    for step in range(steps):
        tokens, targets, mask, _ = make_mqar_batch(
            generator=gen, device=DEVICE, **TRAIN_CFG)
        logits = model(tokens)
        loss   = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1), reduction='none')
        loss   = (loss * mask.reshape(-1).float()).sum() / (mask.sum() + 1e-6)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if (step + 1) % 500 == 0:
            print(f"    [K={K} seed={seed}] step {step+1}/{steps} | loss {loss.item():.4f}",
                  flush=True)

    overall, _, _ = mqar_accuracy(model, TRAIN_CFG, 8, seed=seed + 100, device=DEVICE)
    return overall


def run_attn_gate():
    print("\n── Attention validity gate ──")
    vocab_size = N_KEYS + N_VALUES + 1
    attn = TinyCausalTransformerLM(vocab_size, d_model=64, n_layers=2, n_heads=4,
                                    max_len=TRAIN_LEN)
    attn.to(DEVICE).train()
    mqar_train(attn, TRAIN_CFG, steps=1000, lr=3e-3, seed=0, device=DEVICE, log_every=500)
    overall, _, _ = mqar_accuracy(attn, TRAIN_CFG, 8, seed=1, device=DEVICE)
    ok = overall >= 0.90
    print(f"  attention recall: {overall:.3f}  gate={'PASS' if ok else 'FAIL'}")
    return overall, ok


def stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    std  = (sum((v - mean)**2 for v in vals) / n) ** 0.5
    return mean, std


def main():
    print("=" * 70)
    print(f"K=2 vs K=1 CONFIRM — {len(SEEDS)} seeds, {STEPS} steps, same-run baseline")
    print(f"Seeds: {SEEDS}")
    print(f"K∈{KS}, n_pairs={N_PAIRS}, tanh_m, CPU")
    print("=" * 70)

    # Attention gate
    attn_acc, gate_ok = run_attn_gate()
    if not gate_ok:
        print(f"ABORT: attn gate FAILED ({attn_acc:.3f} < 0.90). Results would be VOID.")
        sys.exit(1)

    # Per-seed results for K=1 and K=2
    recalls = {K: [] for K in KS}

    for K in KS:
        print(f"\n{'='*60}")
        print(f"K={K}")
        t0 = time.time()
        for seed in SEEDS:
            acc = run_one(K, seed, STEPS)
            recalls[K].append(acc)
            print(f"  K={K} seed={seed:5d}  recall={acc:.4f}  ({acc*100:.2f}%)",
                  flush=True)
        elapsed = time.time() - t0
        m, s = stats(recalls[K])
        print(f"  K={K} SUMMARY: {m:.4f}±{s:.4f}  ({m*100:.2f}%±{s*100:.2f}%)  "
              f"elapsed={elapsed:.0f}s")

    # ── Same-run comparison (K=2 vs K=1, paired by seed) ──────────────────────
    print("\n" + "=" * 70)
    print("SAME-RUN COMPARISON (paired by seed)")
    print("=" * 70)

    diffs = [recalls[2][i] - recalls[1][i] for i in range(len(SEEDS))]
    mean1, std1 = stats(recalls[1])
    mean2, std2 = stats(recalls[2])
    mean_diff, std_diff = stats(diffs)

    print(f"K=1:  {mean1:.4f}±{std1:.4f}  seeds={[round(r,4) for r in recalls[1]]}")
    print(f"K=2:  {mean2:.4f}±{std2:.4f}  seeds={[round(r,4) for r in recalls[2]]}")
    print(f"diff: {mean_diff:+.4f}±{std_diff:.4f}  "
          f"(K=2 − K=1, per seed: {[round(d,4) for d in diffs]})")

    # >1σ criterion: mean_diff > std_diff of differences
    beats_1sigma = mean_diff > std_diff
    print(f"\nmean_diff ({mean_diff:+.4f}) > std_diff ({std_diff:.4f}): "
          f"{'YES — K=2 beats K=1 by >1σ' if beats_1sigma else 'NO — gap within noise'}")

    # Fraction of seeds where K=2 > K=1
    wins = sum(1 for d in diffs if d > 0)
    print(f"K=2 > K=1 in {wins}/{len(SEEDS)} seeds")

    result = {
        "config": {
            "n_pairs": N_PAIRS, "n_keys": N_KEYS, "n_values": N_VALUES,
            "train_len": TRAIN_LEN, "steps": STEPS, "seeds": SEEDS,
            "d_model": D_MODEL, "n_heads": N_HEADS, "d_head": D_HEAD,
            "n_layers": N_LAYERS, "lr": LR, "batch": BATCH,
            "readout": "tanh_m",
        },
        "attention_gate": round(attn_acc, 4),
        "gate_passed": gate_ok,
        "chance": round(1.0 / N_VALUES, 4),
        "K1": {
            "mean": round(mean1, 4),
            "std": round(std1, 4),
            "by_seed": [round(r, 4) for r in recalls[1]],
        },
        "K2": {
            "mean": round(mean2, 4),
            "std": round(std2, 4),
            "by_seed": [round(r, 4) for r in recalls[2]],
        },
        "diff_K2_minus_K1": {
            "mean": round(mean_diff, 4),
            "std": round(std_diff, 4),
            "by_seed": [round(d, 4) for d in diffs],
        },
        "beats_1sigma": beats_1sigma,
        "K2_wins": f"{wins}/{len(SEEDS)}",
    }

    out_path = "/tmp/k2_confirm_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults → {out_path}")
    return result


if __name__ == "__main__":
    main()
