#!/usr/bin/env python3 -u
"""
Phase-GSSM MQAR capacity run — Experiment 2 (Frontier #3).
==========================================================

Runs the complex+selective quadrant on the MQAR gap-sweep, exactly per
analysis/NEXT_EXPERIMENTS.md §2.  Four arms:

  1. selective    : reference SelectiveRapiditySqrtTransformerLM (scalar magnitude)
  2. phase_true   : PhaseSelectiveLM(use_phase=True,  omega_scale=pi)  — complex+selective
  3. phase_false  : PhaseSelectiveLM(use_phase=False)  — ablation, == Selective by construction
  4. attn         : TinyCausalTransformerLM             — validity gate (>=0.90) + recall ceiling

Same training loop as mqar.run_mqar (train -> freeze -> eval at train AND test len,
binned by gap). All arms share the identical MQAR vocab/config so phase_true - phase_false
isolates the phase channel inside ONE codepath.

NO downloads. Offline synthetic MQAR only. python3, MPS-safe.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "reference"))

from mqar import (  # noqa: E402
    make_mqar_batch, mqar_accuracy, run_mqar, GAP_BINS, TinyCausalTransformerLM,
)
from phase_gssm import PhaseSelectiveLM  # noqa: E402
from moebius_scan_transformer_selective import SelectiveRapiditySqrtTransformerLM  # noqa: E402

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def build_arm(arm, vocab_size, mask_idx, d_model, n_layers, n_heads, d_head,
              seq_len, omega_scale):
    if arm == "selective":
        return SelectiveRapiditySqrtTransformerLM(
            vocab_size, mask_idx, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, d_head=d_head, seq_len=seq_len, dropout=0.0, causal=True)
    if arm == "phase_true":
        return PhaseSelectiveLM(
            vocab_size, mask_idx, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, d_head=d_head, seq_len=seq_len, dropout=0.0,
            causal=True, omega_scale=omega_scale, use_phase=True)
    if arm == "phase_false":
        return PhaseSelectiveLM(
            vocab_size, mask_idx, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, d_head=d_head, seq_len=seq_len, dropout=0.0,
            causal=True, omega_scale=omega_scale, use_phase=False)
    if arm == "attn":
        return TinyCausalTransformerLM(
            vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            max_len=max(seq_len, 1024))
    raise ValueError(arm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--train-len", type=int, default=64)
    ap.add_argument("--test-len", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-head", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "results" / "phase_mqar_capacity.json"))
    ap.add_argument("--arms", default="attn,selective,phase_false,phase_true")
    args = ap.parse_args()

    n_keys = n_values = 64
    vocab_size = n_keys + n_values + 1
    mask_idx = vocab_size
    omega_scale = math.pi

    train_cfg = dict(batch_size=32, seq_len=args.train_len, n_pairs=args.n_pairs,
                     n_queries=args.n_pairs, n_keys=n_keys, n_values=n_values)
    test_cfg = dict(batch_size=32, seq_len=args.test_len, n_pairs=args.n_pairs,
                    n_queries=args.n_pairs, n_keys=n_keys, n_values=n_values)

    print(f"=== Phase-GSSM MQAR capacity run ===")
    print(f"device={DEVICE}  steps={args.steps}  train_len={args.train_len} "
          f"test_len={args.test_len}  n_pairs={args.n_pairs}")
    print(f"d_model={args.d_model} n_layers={args.n_layers} n_heads={args.n_heads} "
          f"d_head={args.d_head} omega_scale={omega_scale:.4f}")
    print(f"vocab_size={vocab_size}  SEP_ID={n_keys+n_values}\n")

    results = {}
    arms = args.arms.split(",")
    for arm in arms:
        torch.manual_seed(args.seed)
        model = build_arm(arm, vocab_size, mask_idx, args.d_model, args.n_layers,
                          args.n_heads, args.d_head, args.train_len, omega_scale)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"── arm={arm}  params={n_params:,} ──")
        t0 = time.time()
        res = run_mqar(model, train_cfg, test_cfg, args.steps, lr=args.lr,
                       seed=args.seed, device=DEVICE)
        dt = time.time() - t0
        res["params"] = n_params
        res["wall_s"] = round(dt, 1)
        results[arm] = res
        tr = res["train_len"]["overall"]
        te = res["test_len"]["overall"]
        print(f"   train-len overall: {tr:.4f}   test-len overall: {te:.4f}   ({dt:.0f}s)")
        # long-gap bins on train-len (the recall-cliff region)
        tg = res["train_len"]["by_gap"]
        longbins = {k: round(v, 3) for k, v in tg.items()
                    if v is not None and k in ("9-12", "13-16", "17-24", "25-32", "33-48")}
        print(f"   train-len long-gap bins: {longbins}\n")

    # validity gate
    attn_acc = results.get("attn", {}).get("train_len", {}).get("overall", 0.0)
    gate_pass = attn_acc >= 0.90 if "attn" in results else None

    payload = {
        "header": "Phase-GSSM MQAR capacity — complex+selective quadrant (Exp 2 / Frontier #3)",
        "device": str(DEVICE),
        "config": {
            "steps": args.steps, "n_pairs": args.n_pairs,
            "train_len": args.train_len, "test_len": args.test_len,
            "d_model": args.d_model, "n_layers": args.n_layers,
            "n_heads": args.n_heads, "d_head": args.d_head,
            "lr": args.lr, "seed": args.seed,
            "n_keys": n_keys, "n_values": n_values,
            "omega_scale": omega_scale, "vocab_size": vocab_size,
        },
        "validity_gate": {
            "attn_train_acc": round(attn_acc, 4) if "attn" in results else None,
            "passed": gate_pass,
            "note": "attn must reach >=0.90 at train len or ALL GSSM numbers are VOID",
        },
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {args.out}")

    # headline contrast
    print("\n=== HEADLINE CONTRAST (train-len overall) ===")
    for arm in arms:
        if arm in results:
            print(f"  {arm:14s}: {results[arm]['train_len']['overall']:.4f}")
    if "phase_true" in results and "phase_false" in results:
        d = results["phase_true"]["train_len"]["overall"] - results["phase_false"]["train_len"]["overall"]
        print(f"  phase_true - phase_false = {d:+.4f}  (the phase channel's contribution)")
    print(f"  validity gate (attn>=0.90): {gate_pass}")


if __name__ == "__main__":
    main()
