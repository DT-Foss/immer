#!/usr/bin/env python3 -u
"""
Holographic readout shootout — Hopfield nonlinear / MMSE vs rms baseline — 2026-06-25
=====================================================================================
Tests Kimi Dim-07 (bounded form): a NONLINEARITY ON THE OVERLAP (poly) and a learned
MMSE/decorrelating read matrix, vs the linear rms read. All bounded, same complex
holographic state, zero extra STATE (poly=0 params, mmse=D²·H·L params, NO key-loop,
NO alphabet index, NO value-cache).

Arms (same-run, within-seed):
  rms        — linear de-rotation read (THE BASELINE; must reproduce ~7-9% or run void)
  poly3      — rectified cube on the rms-normalised read (sharpen matched contrast)
  poly5      — rectified 5th power
  mmse       — learned D×D decorrelating read matrix (identity-init → == rms at init)
  poly_mmse  — mmse then poly3 (combine decorrelation + sharpening)

Rigor: 2500 steps (rms needs ≥2500 or it under-trains — the documented confound).
Same-run within-seed comparison. attn validity gate. CPU deterministic.
"""
import os, sys, json, math, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "reference"))
sys.path.insert(0, HERE)

import torch
import torch.nn as nn
import torch.nn.functional as F
from mqar import make_mqar_batch, mqar_accuracy, TinyCausalTransformerLM
from holographic_gssm import HolographicLM


def build(arm, vocab, mask, seq_len):
    if arm == "attn":
        return TinyCausalTransformerLM(vocab, d_model=128, n_layers=2, n_heads=4,
                                       max_len=max(seq_len, 1024))
    # all holographic arms share identical config except the readout string
    return HolographicLM(vocab, mask, d_model=128, n_layers=2, n_heads=4, d_head=32,
                         seq_len=seq_len, dropout=0.0, causal=True,
                         phase_scale=math.pi, use_phase=True, readout=arm)


def train(model, cfg, steps, lr, seed, dev, tag):
    model.to(dev).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator(device="cpu").manual_seed(seed)
    t0 = time.time()
    for s in range(steps):
        tok, tgt, m, _ = make_mqar_batch(generator=g, device=dev, **cfg)
        lo = model(tok)
        l = F.cross_entropy(lo.reshape(-1, lo.size(-1)), tgt.reshape(-1),
                            reduction="none")
        l = (l * m.reshape(-1).float()).sum() / (m.sum() + 1e-6)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if (s + 1) % 500 == 0:
            print(f"      [{tag}] step {s+1}/{steps} loss {l.item():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--seeds", default="1,7,42")
    ap.add_argument("--arms", default="rms,poly3,poly5,mmse,poly_mmse,attn")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default=os.path.join(REPO, "results",
                                                  "holo_readout_hopfield.json"))
    args = ap.parse_args()

    dev = torch.device("cpu")
    nk = nv = 64
    vocab = nk + nv + 1
    mask = vocab
    chance = 1.0 / nv
    seq_len = 64
    cfg = dict(batch_size=32, seq_len=seq_len, n_pairs=8, n_queries=8,
               n_keys=nk, n_values=nv)
    seeds = [int(s) for s in args.seeds.split(",")]
    arms = args.arms.split(",")

    print("=" * 78)
    print("HOLO READOUT SHOOTOUT — Hopfield nonlinear / MMSE vs rms")
    print(f"steps={args.steps} seeds={seeds} arms={arms} chance={chance:.4f}")
    print("=" * 78)

    per_seed = {a: [] for a in arms}
    for seed in seeds:
        print(f"\n{'='*28} seed {seed} {'='*28}")
        for arm in arms:
            torch.manual_seed(seed)
            model = build(arm, vocab, mask, seq_len)
            np_ = sum(p.numel() for p in model.parameters())
            print(f"  [arm={arm}] params={np_:,}")
            train(model, cfg, args.steps, args.lr, seed, dev, arm)
            model.eval()
            acc, _, _ = mqar_accuracy(model, cfg, 8, seed + 1, dev)
            per_seed[arm].append(round(acc * 100, 2))
            print(f"    {arm:10s} recall {acc*100:.2f}%", flush=True)

    def ms(xs):
        mu = sum(xs) / len(xs)
        sd = (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5
        return round(mu, 2), round(sd, 2)

    print("\n" + "=" * 78)
    print("AGGREGATE (recall %, mean ± std)")
    summary = {}
    for a in arms:
        mu, sd = ms(per_seed[a])
        summary[a] = {"mean": mu, "std": sd, "per_seed": per_seed[a]}
        print(f"  {a:10s} {mu:5.2f} ± {sd:4.2f}   {per_seed[a]}")

    base = summary.get("rms", {}).get("mean", 0.0)
    attn = summary.get("attn", {}).get("mean", 0.0)
    print(f"\n  baseline rms = {base:.2f}%  (alive if > {3*chance*100:.2f}%)")
    print(f"  attn gate    = {attn:.2f}%  ({'PASS' if attn >= 90 else 'FAIL→VOID'})")
    best = max((a for a in arms if a not in ("attn",)),
               key=lambda a: summary[a]["mean"])
    print(f"  BEST non-attn arm: {best} = {summary[best]['mean']:.2f}%  "
          f"(vs rms {base:.2f}%, Δ={summary[best]['mean']-base:+.2f}pp)")

    out = {"config": {"steps": args.steps, "seeds": seeds, "arms": arms,
                      "chance": chance, "n_pairs": 8},
           "summary": summary,
           "baseline_rms": base, "attn_gate": attn, "best_arm": best}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
