#!/usr/bin/env python3 -u
"""
ONSET campaign — Lever 4: Phase-confound clean re-run.
======================================================
results/phase_mqar_capacity.json (+_REPRO) measured phase_true vs phase_false
with the NATIVE raw readout at 1500/3000 steps → all three GSSM arms byte-
identical at chance. That violated the readout-crossover rule (raw read is
under-trained; tanh_m carries the effect). This re-run uses a tanh_m-equivalent
readout at 2500 steps, 3 seeds, same-run paired, before the phase channel is
declared null.

Arms (per seed, same MQAR config):
    attn         : validity gate (>=0.90)
    selective    : reference scalar magnitude (native readout floor)
    phase_false  : PhaseReadoutLM(use_phase=False) == Selective (control)
    phase_true   : PhaseReadoutLM(use_phase=True, readout=tanh_m)  ← the fair test

Gate (pre-registered):
    LIVE  : phase_true − phase_false > 1σ (paired) AND phase_true mean > chance+2σ
    NULL  : phase_true ≈ phase_false within 1σ → additive-phase channel confirmed
            null under a fair readout (not a readout artefact).

MPS if available. Output → results/onset_phase_confound.json + logfile.
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
from phase_gssm_readout import PhaseReadoutLM  # noqa: E402
from moebius_scan_transformer_selective import SelectiveRapiditySqrtTransformerLM  # noqa: E402

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

N_KEYS = N_VALUES = 64
VOCAB = N_KEYS + N_VALUES + 1
MASK_IDX = VOCAB
CHANCE = 1.0 / N_VALUES
D_MODEL, N_HEADS, D_HEAD, N_LAYERS = 128, 4, 32, 2


def build(arm, train_len, readout):
    if arm == "attn":
        return TinyCausalTransformerLM(VOCAB, d_model=64, n_layers=2, n_heads=4,
                                       max_len=max(train_len, 1024))
    if arm == "selective":
        return SelectiveRapiditySqrtTransformerLM(
            VOCAB, MASK_IDX, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
            d_head=D_HEAD, seq_len=train_len, dropout=0.0, causal=True)
    if arm == "phase_false":
        return PhaseReadoutLM(VOCAB, MASK_IDX, d_model=D_MODEL, n_layers=N_LAYERS,
                              n_heads=N_HEADS, d_head=D_HEAD, seq_len=train_len,
                              use_phase=False, readout=readout)
    if arm == "phase_true":
        return PhaseReadoutLM(VOCAB, MASK_IDX, d_model=D_MODEL, n_layers=N_LAYERS,
                              n_heads=N_HEADS, d_head=D_HEAD, seq_len=train_len,
                              use_phase=True, readout=readout)
    raise ValueError(arm)


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
    ap.add_argument("--readout", default="tanh_m")
    ap.add_argument("--out", default=str(REPO / "results" / "onset_phase_confound.json"))
    args = ap.parse_args()

    cfg = dict(batch_size=32, seq_len=args.train_len, n_pairs=args.n_pairs,
               n_queries=args.n_pairs, n_keys=N_KEYS, n_values=N_VALUES)
    seeds = [int(s) for s in args.seeds.split(",")]
    arms = ["attn", "selective", "phase_false", "phase_true"]

    print("=" * 74)
    print(f"ONSET Lever 4: Phase-confound re-run  device={DEVICE} steps={args.steps} "
          f"seeds={seeds} readout={args.readout} chance={CHANCE:.4f}")
    print("=" * 74)

    acc = {a: [] for a in arms}
    t0 = time.time()
    for seed in seeds:
        print(f"\n--- seed {seed} ---", flush=True)
        for a in arms:
            torch.manual_seed(seed)
            m = build(a, args.train_len, args.readout)
            train(m, cfg, args.steps, 3e-3, seed)
            m.eval()
            ov, _, _ = mqar_accuracy(m, cfg, 8, seed + 1, DEVICE)
            acc[a].append(ov)
            print(f"  {a:14s} {ov:.4f}  ({ov*100:.2f}%)", flush=True)

    print("\n" + "=" * 74)
    print("AGGREGATE (mean ± std)")
    summ = {}
    for a in arms:
        mu, sd = mean_std(acc[a])
        summ[a] = {"mean": mu, "std": sd, "per_seed": acc[a]}
        print(f"  {a:14s} {mu:.4f} ± {sd:.4f}")

    diffs = [acc["phase_true"][i] - acc["phase_false"][i] for i in range(len(seeds))]
    dmu, dsd = mean_std(diffs)
    pt_mean = summ["phase_true"]["mean"]
    live = (dmu > dsd and dmu > 0) and (pt_mean > CHANCE + 2 * summ["phase_true"]["std"])
    verdict = ("LIVE (phase channel contributes >1σ under fair readout)" if live
               else f"NULL (phase_true−phase_false Δ={dmu*100:+.2f}pp within noise; "
                    f"additive-phase confirmed null, not a readout artefact)")
    print(f"\n  phase_true − phase_false = {dmu*100:+.2f}pp ± {dsd*100:.2f}pp  "
          f"(per seed: {[round(d*100,2) for d in diffs]})")
    print(f"  VERDICT: {verdict}")
    print(f"  validity (attn): {summ['attn']['mean']:.4f} "
          f"{'PASS' if summ['attn']['mean']>=0.9 else 'FAIL — VOID'}")

    payload = {
        "config": {"lever": "4_phase_confound_rerun", "steps": args.steps,
                   "seeds": ",".join(str(s) for s in seeds), "n_pairs": args.n_pairs,
                   "train_len": args.train_len, "readout": args.readout,
                   "device": str(DEVICE),
                   "note": "re-tests phase_mqar_capacity.json which used native raw readout at 1500/3000 steps",
                   "out": args.out},
        "chance": CHANCE,
        "summary": summ,
        "phase_true_minus_false": {"mean_pp": round(dmu * 100, 3),
                                    "std_pp": round(dsd * 100, 3),
                                    "per_seed_pp": [round(d * 100, 3) for d in diffs]},
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWritten {args.out}  ({payload['elapsed_s']}s)")


if __name__ == "__main__":
    main()
