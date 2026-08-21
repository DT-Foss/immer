#!/usr/bin/env python3 -u
"""
ADVERSARIAL RE-VERIFICATION of the K=2-confirm thread.
======================================================
Re-run with FRESH seeds {13, 99} that the original 5-seed run never saw,
plus baseline seed {1} for an anchor against the recorded run.

K=1 IS the same-run baseline (byte-identical to the 8.89% holographic write;
reduction self-test PASS, max|Δ|=0). K=2 is the multi-freq harmonic test arm.

Checks:
  - Does K=2 beat K=1 (same-run, paired by seed)?
  - Attn validity gate >= 0.90 (else VOID)?
  - Does seed 1 anchor reproduce the recorded by-seed value?
Identical config to k2_confirm_run.py: 2000 steps, n_pairs=8, n_keys=n_values=64,
train_len=64, d_model=128, tanh_m readout, CPU-deterministic.

Output -> /tmp/k2_verify_freshseeds.json
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

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

# Identical config to k2_confirm_run.py
N_PAIRS, N_KEYS, N_VALUES, TRAIN_LEN = 8, 64, 64, 64
BATCH, LR, STEPS = 32, 3e-3, 2000
SEEDS = [13, 99]          # FRESH seeds, never in the recorded {1,7,42,123,2024}
ANCHOR_SEED = 1           # recorded by-seed: K1=0.0527, K2=0.0703
KS = [1, 2]

VOCAB_SIZE = N_KEYS + N_VALUES + 1
MASK_IDX = VOCAB_SIZE
TRAIN_CFG = dict(batch_size=BATCH, seq_len=TRAIN_LEN, n_pairs=N_PAIRS,
                 n_queries=N_PAIRS, n_keys=N_KEYS, n_values=N_VALUES)
D_MODEL, N_HEADS, D_HEAD, N_LAYERS = 128, 4, 32, 2


def build_model(K):
    return HolographicMultiFreqLM(
        vocab_size=VOCAB_SIZE, mask_idx=MASK_IDX,
        d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS, d_head=D_HEAD,
        seq_len=TRAIN_LEN, dropout=0.0, causal=True,
        phase_scale=math.pi, use_phase=True, readout="tanh_m", n_freqs=K)


def run_one(K, seed, steps):
    torch.manual_seed(seed)
    model = build_model(K).to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(seed)
    for step in range(steps):
        tokens, targets, mask, _ = make_mqar_batch(generator=gen, device=DEVICE, **TRAIN_CFG)
        logits = model(tokens)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1), reduction='none')
        loss = (loss * mask.reshape(-1).float()).sum() / (mask.sum() + 1e-6)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if (step + 1) % 1000 == 0:
            print(f"    [K={K} seed={seed}] step {step+1}/{steps} | loss {loss.item():.4f}", flush=True)
    overall, _, _ = mqar_accuracy(model, TRAIN_CFG, 8, seed=seed + 100, device=DEVICE)
    return overall


def run_attn_gate():
    print("\n-- Attention validity gate --")
    attn = TinyCausalTransformerLM(VOCAB_SIZE, d_model=64, n_layers=2, n_heads=4, max_len=TRAIN_LEN)
    attn.to(DEVICE).train()
    mqar_train(attn, TRAIN_CFG, steps=1000, lr=3e-3, seed=0, device=DEVICE, log_every=500)
    overall, _, _ = mqar_accuracy(attn, TRAIN_CFG, 8, seed=1, device=DEVICE)
    ok = overall >= 0.90
    print(f"  attention recall: {overall:.3f}  gate={'PASS' if ok else 'FAIL'}")
    return overall, ok


def stats(vals):
    n = len(vals); mean = sum(vals) / n
    std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5
    return mean, std


def main():
    print("=" * 70)
    print(f"K=2 RE-VERIFICATION — fresh seeds {SEEDS} + anchor seed {ANCHOR_SEED}")
    print(f"K in {KS}, {STEPS} steps, n_pairs={N_PAIRS}, tanh_m, CPU")
    print("=" * 70)

    attn_acc, gate_ok = run_attn_gate()
    if not gate_ok:
        print(f"ABORT: attn gate FAILED ({attn_acc:.3f} < 0.90). VOID.")
        sys.exit(1)

    all_seeds = [ANCHOR_SEED] + SEEDS
    recalls = {K: {} for K in KS}
    for K in KS:
        print(f"\n{'='*60}\nK={K}")
        for seed in all_seeds:
            t0 = time.time()
            acc = run_one(K, seed, STEPS)
            recalls[K][seed] = acc
            print(f"  K={K} seed={seed:5d}  recall={acc:.4f} ({acc*100:.2f}%)  ({time.time()-t0:.0f}s)", flush=True)

    # Anchor reproducibility vs recorded run
    rec_anchor = {1: {"K1": 0.0527, "K2": 0.0703}}
    print("\n-- Anchor seed 1 vs recorded --")
    print(f"  K=1: now={recalls[1][1]:.4f}  recorded={rec_anchor[1]['K1']:.4f}")
    print(f"  K=2: now={recalls[2][1]:.4f}  recorded={rec_anchor[1]['K2']:.4f}")

    # Same-run paired comparison on FRESH seeds (the real test)
    print("\n" + "=" * 70)
    print("SAME-RUN PAIRED COMPARISON (fresh seeds 13,99)")
    print("=" * 70)
    fresh_k1 = [recalls[1][s] for s in SEEDS]
    fresh_k2 = [recalls[2][s] for s in SEEDS]
    diffs = [recalls[2][s] - recalls[1][s] for s in SEEDS]
    m1, s1 = stats(fresh_k1); m2, s2 = stats(fresh_k2); md, sd = stats(diffs)
    print(f"K=1 (baseline): {m1:.4f}±{s1:.4f}  seeds={[round(r,4) for r in fresh_k1]}")
    print(f"K=2 (test):     {m2:.4f}±{s2:.4f}  seeds={[round(r,4) for r in fresh_k2]}")
    print(f"diff K2-K1:     {md:+.4f}±{sd:.4f}  per-seed={[round(d,4) for d in diffs]}")
    beats_1sigma = md > sd
    print(f"\nK=2 beats K=1 by >1sigma: {'YES' if beats_1sigma else 'NO'}")

    # Also report ALL three seeds pooled (1,13,99) for a 3-seed mean
    allk1 = [recalls[1][s] for s in all_seeds]
    allk2 = [recalls[2][s] for s in all_seeds]
    am1, as1 = stats(allk1); am2, as2 = stats(allk2)
    print(f"\nPooled 3 seeds {all_seeds}:")
    print(f"  K=1: {am1:.4f}±{as1:.4f}  ({am1*100:.2f}%)")
    print(f"  K=2: {am2:.4f}±{as2:.4f}  ({am2*100:.2f}%)")

    result = {
        "config": {"n_pairs": N_PAIRS, "n_keys": N_KEYS, "n_values": N_VALUES,
                   "train_len": TRAIN_LEN, "steps": STEPS, "fresh_seeds": SEEDS,
                   "anchor_seed": ANCHOR_SEED, "d_model": D_MODEL, "readout": "tanh_m"},
        "attention_gate": round(attn_acc, 4), "gate_passed": gate_ok,
        "chance": round(1.0 / N_VALUES, 4),
        "anchor_seed_1": {"K1_now": round(recalls[1][1], 4), "K1_recorded": 0.0527,
                          "K2_now": round(recalls[2][1], 4), "K2_recorded": 0.0703},
        "fresh_K1": {"by_seed": {str(s): round(recalls[1][s], 4) for s in SEEDS},
                     "mean": round(m1, 4), "std": round(s1, 4)},
        "fresh_K2": {"by_seed": {str(s): round(recalls[2][s], 4) for s in SEEDS},
                     "mean": round(m2, 4), "std": round(s2, 4)},
        "diff_K2_minus_K1_fresh": {"mean": round(md, 4), "std": round(sd, 4),
                                   "by_seed": [round(d, 4) for d in diffs]},
        "beats_1sigma_fresh": beats_1sigma,
        "pooled_3seed": {"seeds": all_seeds,
                         "K1_mean": round(am1, 4), "K1_std": round(as1, 4),
                         "K2_mean": round(am2, 4), "K2_std": round(as2, 4)},
    }
    out = "/tmp/k2_verify_freshseeds.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults -> {out}")


if __name__ == "__main__":
    main()
