#!/usr/bin/env python3 -u
"""
ONSET campaign — Lever 1: Hopfield learned-β readout (reopening entry 18).
==========================================================================
Same-run paired baselines (rms, tanh_m) vs the two reopened power readouts
(hopfield_beta, poly_before_norm), multi-seed, attn validity gate.

Readout-crossover discipline (standing fact): rms needs 2500 steps to ignite;
the two hopfield readouts START as rms (λ=0), so they ALSO need 2500 steps.
tanh_m carries at ~1500 but we run everyone at 2500 for a fair paired read.

Gates (pre-registered, log-style):
  CONFIRM (wall broken)  : a hopfield readout mean ≥ 15%  (beats the 8.89% wall).
  POSITIVE (worth 5-seed): hopfield mean > best_baseline mean by > 1σ (paired).
  KILL                   : hopfield mean ≤ best_baseline mean (no lift over rms/tanh_m).

Screen: seeds {1,7,42}.  If POSITIVE/CONFIRM → escalate to 5-seed confirm run.
CPU-deterministic, 4 threads.  Output → results/onset_hopfield.json + logfile.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import os
import json
import math
import time
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(4)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "reference"))

from mqar import make_mqar_batch, mqar_accuracy, TinyCausalTransformerLM  # noqa: E402
from holographic_hopfield_beta import HopfieldBetaLM  # noqa: E402

DEVICE = torch.device("cpu")

N_KEYS = N_VALUES = 64
VOCAB = N_KEYS + N_VALUES + 1
MASK_IDX = VOCAB
CHANCE = 1.0 / N_VALUES

D_MODEL, N_HEADS, D_HEAD, N_LAYERS = 128, 4, 32, 2


def build(arm, train_len, p=3.0):
    if arm == "attn":
        return TinyCausalTransformerLM(VOCAB, d_model=64, n_layers=2, n_heads=4,
                                       max_len=max(train_len, 1024))
    if arm == "holo_off":
        return HopfieldBetaLM(VOCAB, MASK_IDX, d_model=D_MODEL, n_layers=N_LAYERS,
                              n_heads=N_HEADS, d_head=D_HEAD, seq_len=train_len,
                              use_phase=False, readout="rms")
    # holo_<readout>
    ro = arm.split("_", 1)[1]
    return HopfieldBetaLM(
        VOCAB, MASK_IDX, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
        d_head=D_HEAD, seq_len=train_len, use_phase=True, readout=ro, p=p)


def train(model, cfg, steps, lr, seed):
    model.to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for step in range(steps):
        tok, tgt, mask, _ = make_mqar_batch(generator=gen, device=DEVICE, **cfg)
        logits = model(tok)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               tgt.reshape(-1), reduction="none")
        loss = (loss * mask.reshape(-1).float()).sum() / (mask.sum() + 1e-6)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if (step + 1) % 500 == 0:
            print(f"      step {step+1}/{steps} | loss {loss.item():.4f}", flush=True)
    return model


def mean_std(xs):
    mu = sum(xs) / len(xs)
    return mu, (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--seeds", default="1,7,42")
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--train-len", type=int, default=64)
    ap.add_argument("--p", type=float, default=3.0)
    ap.add_argument("--out", default=str(REPO / "results" / "onset_hopfield.json"))
    args = ap.parse_args()

    cfg = dict(batch_size=32, seq_len=args.train_len, n_pairs=args.n_pairs,
               n_queries=args.n_pairs, n_keys=N_KEYS, n_values=N_VALUES)
    seeds = [int(s) for s in args.seeds.split(",")]
    arms = ["attn", "holo_off", "holo_rms", "holo_tanh_m",
            "holo_hopfield_beta", "holo_poly_before_norm"]

    print("=" * 74)
    print(f"ONSET Lever 1: Hopfield learned-β  steps={args.steps} seeds={seeds} "
          f"p={args.p} chance={CHANCE:.4f}")
    print("=" * 74)

    acc = {a: [] for a in arms}
    t0 = time.time()
    for seed in seeds:
        print(f"\n--- seed {seed} ---", flush=True)
        for a in arms:
            torch.manual_seed(seed)
            m = build(a, args.train_len, p=args.p)
            train(m, cfg, args.steps, 3e-3, seed)
            m.eval()
            ov, _, _ = mqar_accuracy(m, cfg, 8, seed + 1, DEVICE)
            acc[a].append(ov)
            print(f"  {a:24s} {ov:.4f}  ({ov*100:.2f}%)", flush=True)

    print("\n" + "=" * 74)
    print("AGGREGATE (mean ± std)")
    summ = {}
    for a in arms:
        mu, sd = mean_std(acc[a])
        summ[a] = {"mean": mu, "std": sd, "per_seed": acc[a]}
        print(f"  {a:24s} {mu:.4f} ± {sd:.4f}")

    # ── verdicts ──
    base_rms = summ["holo_rms"]["mean"]
    base_tanh = summ["holo_tanh_m"]["mean"]
    best_base = max(base_rms, base_tanh)
    best_base_name = "rms" if base_rms >= base_tanh else "tanh_m"
    print(f"\n  best same-run baseline: {best_base_name} = {best_base:.4f}")
    verdicts = {}
    for a in ("holo_hopfield_beta", "holo_poly_before_norm"):
        mu = summ[a]["mean"]
        # paired std of (arm - best_base) per seed
        base_arm = "holo_rms" if best_base_name == "rms" else "holo_tanh_m"
        diffs = [acc[a][i] - acc[base_arm][i] for i in range(len(seeds))]
        dmu, dsd = mean_std(diffs)
        if mu >= 0.15:
            v = "CONFIRM (wall broken ≥15%)"
        elif dmu > dsd and dmu > 0:
            v = f"POSITIVE (>1σ over {best_base_name}, Δ={dmu*100:+.2f}pp) → 5-seed"
        else:
            v = f"KILL (no lift over {best_base_name}, Δ={dmu*100:+.2f}pp)"
        verdicts[a] = {"verdict": v, "delta_vs_best_base_pp": round(dmu * 100, 3),
                       "delta_std_pp": round(dsd * 100, 3), "per_seed_diff": diffs}
        print(f"  {a:24s} → {v}")
    print(f"  validity (attn): {summ['attn']['mean']:.4f} "
          f"{'PASS' if summ['attn']['mean']>=0.9 else 'FAIL — VOID'}")

    payload = {
        "config": {"lever": "1_hopfield_learned_beta", "steps": args.steps,
                   "seeds": ",".join(str(s) for s in seeds), "n_pairs": args.n_pairs,
                   "train_len": args.train_len, "d_model": D_MODEL, "d_head": D_HEAD,
                   "p": args.p, "readout_note": "hopfield readouts start==rms (λ=0), need 2500 steps",
                   "out": args.out},
        "chance": CHANCE,
        "summary": summ,
        "verdicts": verdicts,
        "best_baseline": {"name": best_base_name, "mean": best_base},
        "elapsed_s": round(time.time() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWritten {args.out}  ({payload['elapsed_s']}s)")


if __name__ == "__main__":
    main()
