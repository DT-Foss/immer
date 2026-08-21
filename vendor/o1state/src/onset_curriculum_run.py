#!/usr/bin/env python3 -u
"""
ONSET campaign — Lever 3: Curriculum over n_pairs (2 → 4 → 8).
=============================================================
Standing fact: at n_pairs=2 the holographic mechanism IGNITES (25.8% recall);
at n_pairs=8 crosstalk drops it to 8.89%. The wall is CROSSTALK/ONSET, not
capacity. Direct attack on the onset wall: ignite where it ignites (n=2), then
raise the crosstalk load in stages, carrying the trained mechanism forward.

The hypothesis this tests: onset is a BASIN-FINDING problem. If the optimizer
finds the erase/superpose mechanism at low crosstalk (n=2) and we anneal the load
up, the mechanism may survive to n=8 at a recall the COLD n=8 start never reaches
(the cold start sits in the ln(64) plateau on 2/3 of seeds — entries 17/20).

Protocol (single HolographicLM, tanh_m, same model carried across stages):
  stage A: n_pairs=2, steps_A  → confirm ignition (>15%)
  stage B: n_pairs=4, steps_B  → continue same weights
  stage C: n_pairs=8, steps_C  → continue same weights, FINAL eval at n=8
Baseline (same seed, same total steps): COLD start directly at n_pairs=8.

seq_len fixed at 64 for every stage (only n_pairs / n_queries change) so the
model sees the same sequence geometry; only the crosstalk load grows.

Gate (pre-registered):
  CONFIRM : curriculum FINAL recall @ n=8 > cold-n=8 mean by >1σ AND > 8.89% wall.
  PARTIAL : curriculum ignites @ n=8 (>4%) on seeds where cold stays at chance
            (plateau) — proves annealing rescues onset even if it doesn't beat the wall.
  KILL    : curriculum @ n=8 ≈ cold-n=8 → carrying the low-crosstalk mechanism
            forward does not survive the crosstalk increase.

CPU-deterministic, 4 threads. Output → results/onset_curriculum.json + logfile.
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
from holographic_gssm import HolographicLM  # noqa: E402

DEVICE = torch.device("cpu")
N_KEYS = N_VALUES = 64
VOCAB = N_KEYS + N_VALUES + 1
MASK_IDX = VOCAB
CHANCE = 1.0 / N_VALUES
D_MODEL, N_HEADS, D_HEAD, N_LAYERS = 128, 4, 32, 2
READOUT = "tanh_m"
TRAIN_LEN = 64


def cfg_for(n_pairs):
    return dict(batch_size=32, seq_len=TRAIN_LEN, n_pairs=n_pairs,
                n_queries=n_pairs, n_keys=N_KEYS, n_values=N_VALUES)


def train_stage(model, n_pairs, steps, lr, seed, tag):
    """Continue-train `model` at a given n_pairs for `steps`. Returns final recall
    at that n_pairs."""
    cfg = cfg_for(n_pairs)
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
            model.eval()
            with torch.no_grad():
                acc, _, _ = mqar_accuracy(model, cfg, 6, seed + 1, DEVICE)
            model.train()
            print(f"      [{tag} n={n_pairs}] step {step+1}/{steps} loss {loss.item():.3f} "
                  f"recall {acc*100:.2f}%", flush=True)
    model.eval()
    with torch.no_grad():
        acc, _, _ = mqar_accuracy(model, cfg, 8, seed + 1, DEVICE)
    return acc


def mean_std(xs):
    mu = sum(xs) / len(xs)
    return mu, (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-a", type=int, default=1500)   # n=2 ignition
    ap.add_argument("--steps-b", type=int, default=1000)   # n=4
    ap.add_argument("--steps-c", type=int, default=1500)   # n=8
    ap.add_argument("--seeds", default="1,7,42")
    ap.add_argument("--out", default=str(REPO / "results" / "onset_curriculum.json"))
    args = ap.parse_args()

    total_steps = args.steps_a + args.steps_b + args.steps_c
    seeds = [int(s) for s in args.seeds.split(",")]

    print("=" * 74)
    print(f"ONSET Lever 3: Curriculum n_pairs 2→4→8  "
          f"A={args.steps_a} B={args.steps_b} C={args.steps_c} (cold={total_steps}) "
          f"seeds={seeds}")
    print("=" * 74)

    # attn validity gate once
    print("\n── attn validity gate (n=8) ──", flush=True)
    torch.manual_seed(0)
    attn = TinyCausalTransformerLM(VOCAB, d_model=64, n_layers=2, n_heads=4, max_len=64)
    attn.to(DEVICE).train()
    opt = torch.optim.Adam(attn.parameters(), lr=3e-3)
    gen = torch.Generator(device="cpu").manual_seed(0)
    for _ in range(1000):
        tok, tgt, mask, _ = make_mqar_batch(generator=gen, device=DEVICE, **cfg_for(8))
        lo = attn(tok)
        l = F.cross_entropy(lo.reshape(-1, lo.size(-1)), tgt.reshape(-1), reduction="none")
        l = (l * mask.reshape(-1).float()).sum() / (mask.sum() + 1e-6)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(attn.parameters(), 5.0); opt.step()
    attn.eval()
    attn_acc, _, _ = mqar_accuracy(attn, cfg_for(8), 8, 1, DEVICE)
    print(f"  attn recall {attn_acc:.4f}  {'PASS' if attn_acc>=0.9 else 'FAIL — VOID'}",
          flush=True)

    curric, cold = [], []
    stageA_igni = []
    t0 = time.time()
    for seed in seeds:
        print(f"\n{'='*60}\n--- seed {seed} ---", flush=True)

        # CURRICULUM: one model carried across stages
        print("  [curriculum] stage A n=2 (ignite)...", flush=True)
        torch.manual_seed(seed)
        model = HolographicLM(VOCAB, MASK_IDX, d_model=D_MODEL, n_layers=N_LAYERS,
                              n_heads=N_HEADS, d_head=D_HEAD, seq_len=TRAIN_LEN,
                              use_phase=True, readout=READOUT)
        aA = train_stage(model, 2, args.steps_a, 3e-3, seed, "A")
        stageA_igni.append(aA)
        print(f"  [curriculum] stage A n=2 recall {aA*100:.2f}%", flush=True)
        print("  [curriculum] stage B n=4...", flush=True)
        train_stage(model, 4, args.steps_b, 3e-3, seed + 100, "B")
        print("  [curriculum] stage C n=8...", flush=True)
        aC = train_stage(model, 8, args.steps_c, 3e-3, seed + 200, "C")
        curric.append(aC)
        print(f"  [curriculum] FINAL n=8 recall {aC*100:.2f}%", flush=True)

        # COLD baseline: same seed, total steps directly at n=8
        print("  [cold] direct n=8 start (same total steps)...", flush=True)
        torch.manual_seed(seed)
        cold_m = HolographicLM(VOCAB, MASK_IDX, d_model=D_MODEL, n_layers=N_LAYERS,
                               n_heads=N_HEADS, d_head=D_HEAD, seq_len=TRAIN_LEN,
                               use_phase=True, readout=READOUT)
        cC = train_stage(cold_m, 8, total_steps, 3e-3, seed, "cold")
        cold.append(cC)
        print(f"  [cold] n=8 recall {cC*100:.2f}%", flush=True)

    print("\n" + "=" * 74)
    print("AGGREGATE (mean ± std)")
    cur_mu, cur_sd = mean_std(curric)
    cold_mu, cold_sd = mean_std(cold)
    a_mu, a_sd = mean_std(stageA_igni)
    print(f"  stageA_ignition(n=2) {a_mu:.4f} ± {a_sd:.4f}  {[round(x,4) for x in stageA_igni]}")
    print(f"  curriculum(n=8)      {cur_mu:.4f} ± {cur_sd:.4f}  {[round(x,4) for x in curric]}")
    print(f"  cold(n=8)            {cold_mu:.4f} ± {cold_sd:.4f}  {[round(x,4) for x in cold]}")

    diffs = [curric[i] - cold[i] for i in range(len(seeds))]
    dmu, dsd = mean_std(diffs)
    if cur_mu > cold_mu + dsd and cur_mu > 0.0889 and dmu > 0:
        verdict = f"CONFIRM (curriculum beats cold by >1σ, Δ={dmu*100:+.2f}pp, >8.89% wall)"
    elif cur_mu > 0.04 and cold_mu < 0.04:
        verdict = (f"PARTIAL (curriculum ignites {cur_mu*100:.2f}% where cold "
                   f"stays {cold_mu*100:.2f}%)")
    else:
        verdict = f"KILL (curriculum ≈ cold @ n=8, Δ={dmu*100:+.2f}pp)"
    print(f"\n  curriculum − cold = {dmu*100:+.2f}pp ± {dsd*100:.2f}pp  "
          f"(per seed: {[round(d*100,2) for d in diffs]})")
    print(f"  VERDICT: {verdict}")

    payload = {
        "config": {"lever": "3_curriculum_npairs", "steps_a": args.steps_a,
                   "steps_b": args.steps_b, "steps_c": args.steps_c,
                   "cold_steps": total_steps, "seeds": ",".join(str(s) for s in seeds),
                   "readout": READOUT, "train_len": TRAIN_LEN, "out": args.out},
        "chance": CHANCE,
        "attn_gate": round(attn_acc, 4),
        "summary": {
            "stageA_ignition_n2": {"mean": a_mu, "std": a_sd, "per_seed": stageA_igni},
            "curriculum_n8": {"mean": cur_mu, "std": cur_sd, "per_seed": curric},
            "cold_n8": {"mean": cold_mu, "std": cold_sd, "per_seed": cold},
        },
        "curriculum_minus_cold": {"mean_pp": round(dmu * 100, 3),
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
