"""
FRONT 3 — DEPTH & HEADS sweep for the working tanh_m Holographic-GSSM — by Opus 4.8
==================================================================================

The key-conditioned holographic write (src/holographic_gssm.py, readout="tanh_m")
clears the scalar-recall wall: +5.6pp over Selective (7.18% vs 1.61%), attn gate 0.997.
But it plateaus at ~7%. FRONT 3 tests ONE plateau cause: capacity of the retrieval
*stack*, not the write rule.

  HYPOTHESIS. A single holographic layer WRITES the (key,value) superposition, but
  retrieving-and-comparing the de-rotated read may need a SECOND stage — the
  induction-head story: attention-based MQAR solvers famously need 2 layers
  (a previous-token head feeding a match head). More LAYERS = more retrieval stages.
  More HEADS = more parallel key-value memories (more independent phase circles), so a
  given (key,value) load spreads over more channels and collides less at read time.

  QUESTION. Does recall climb with DEPTH (2 -> 3 -> 4 layers, induction-head-style
  multi-stage retrieval) or with HEADS (4 -> 8, more parallel memories) — or neither
  (the cap is the write/read key-sharing, not stack capacity)?

DESIGN (locked to the working baseline so the (2,4) cell reproduces 7.18%):
  * readout = "tanh_m"  (the LOAD-BEARING readout — m is the learned relevance gate;
    rms/layernorm FAILED at +0.7/+0.4pp. Do NOT touch the readout here.)
  * n_pairs=8, train_len=64, d_head=32 FIXED. Sweep n_layers in {2,3,4} x n_heads in {4,8}.
  * d_model = n_heads * d_head (128 at 4 heads, 256 at 8 heads) — the natural widening,
    same convention as Selective/Holographic (total_dim = n_heads*d_head, W_out: total->d_model).
  * Per (layers,heads) cell: holo_on (tanh_m) is the arm under test; attn is the
    per-cell VALIDITY GATE (must hit >=0.90 or that cell's holo number is VOID);
    holo_off (use_phase=False == Selective) is the per-cell FLOOR.
  * multi-seed (default 3: 1,7,42 — same seeds the shootout used), CPU-deterministic,
    mean +- std. A single seed is NOISE on a chance-flat loss (1/64=1.56%); this is how
    the original phase-GSSM positive was a FALSE positive. Multi-seed is mandatory.
  * steps=1200, lr=3e-3 — identical to holographic_readout_shootout.py so the (2,4)
    holo_on cell is an apples-to-apples reproduction of the 7.18% baseline.

READING THE RESULT.
  - holo_on climbs monotonically with n_layers at fixed heads  -> retrieval is
    depth-limited (induction-head-style 2+ stage recall). Recommend deeper stack.
  - holo_on climbs with n_heads at fixed depth -> memory is collision-limited
    (capacity per phase circle). Recommend more parallel memories / wider state.
  - holo_on flat across the whole grid -> the cap is the WRITE/READ KEY SHARING, not
    stack capacity. Next lever moves to the binding mechanism (separate W_key_write /
    W_key_read, learned phase_scale), NOT depth/width.

Self-contained, offline, python3. Writes results/holographic_depth.json.
"""

import os
import sys
import math
import json
import time
import argparse

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "reference"))
sys.path.insert(0, HERE)

from mqar import make_mqar_batch, mqar_accuracy, TinyCausalTransformerLM  # noqa: E402
from holographic_gssm import HolographicLM  # noqa: E402


def build(arm, vocab_size, mask_idx, d_model, n_layers, n_heads, d_head, seq_len):
    """arm in {attn, holo_off, holo_on}. holo_on is the working tanh_m readout."""
    if arm == "attn":
        return TinyCausalTransformerLM(vocab_size, d_model=d_model, n_layers=n_layers,
                                       n_heads=n_heads, max_len=max(seq_len, 1024))
    if arm == "holo_off":
        return HolographicLM(vocab_size, mask_idx, d_model=d_model, n_layers=n_layers,
                             n_heads=n_heads, d_head=d_head, seq_len=seq_len,
                             use_phase=False, readout="tanh_m")
    if arm == "holo_on":
        return HolographicLM(vocab_size, mask_idx, d_model=d_model, n_layers=n_layers,
                             n_heads=n_heads, d_head=d_head, seq_len=seq_len,
                             phase_scale=math.pi, use_phase=True, readout="tanh_m")
    raise ValueError(arm)


def train(model, cfg, steps, lr, seed, device):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(steps):
        tok, tgt, mask, _ = make_mqar_batch(generator=gen, device=device, **cfg)
        logits = model(tok)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               tgt.reshape(-1), reduction="none")
        loss = (loss * mask.reshape(-1).float()).sum() / (mask.sum() + 1e-6)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    return model


def mean_std(xs):
    mu = sum(xs) / len(xs)
    return mu, (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--seeds", default="1,7,42")
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--train-len", type=int, default=64)
    ap.add_argument("--d-head", type=int, default=32)
    ap.add_argument("--layers", default="2,3,4")
    ap.add_argument("--heads", default="4,8")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default=os.path.join(REPO, "results", "holographic_depth.json"))
    args = ap.parse_args()

    device = torch.device("cpu")  # deterministic; MPS nondeterminism is poison on a chance-flat loss
    nk = nv = 64
    vocab = nk + nv + 1
    mask_idx = vocab
    chance = 1.0 / nv
    cfg = dict(batch_size=32, seq_len=args.train_len, n_pairs=args.n_pairs,
               n_queries=args.n_pairs, n_keys=nk, n_values=nv)

    seeds = [int(s) for s in args.seeds.split(",")]
    layers_grid = [int(x) for x in args.layers.split(",")]
    heads_grid = [int(x) for x in args.heads.split(",")]
    arms = ["attn", "holo_off", "holo_on"]

    print("=" * 78)
    print("FRONT 3 — DEPTH & HEADS sweep, tanh_m Holographic-GSSM, multi-seed, CPU")
    print(f"steps={args.steps} seeds={seeds} n_pairs={args.n_pairs} "
          f"train_len={args.train_len} d_head={args.d_head}")
    print(f"layers={layers_grid}  heads={heads_grid}  chance=1/{nv}={chance:.4f}")
    print("=" * 78)

    cells = {}            # "L{l}_H{h}" -> {arm: {mean,std,per_seed}, d_model, valid}
    t0 = time.time()

    for nl in layers_grid:
        for nh in heads_grid:
            d_model = nh * args.d_head        # natural widening: 4h->128, 8h->256
            key = f"L{nl}_H{nh}"
            print(f"\n=== cell {key}  (d_model={d_model}) ===")
            acc = {a: [] for a in arms}
            for seed in seeds:
                line = f"  seed {seed:>4d}: "
                for a in arms:
                    torch.manual_seed(seed)
                    m = build(a, vocab, mask_idx, d_model, nl, nh, args.d_head, args.train_len)
                    train(m, cfg, args.steps, args.lr, seed, device)
                    m.eval()
                    ov, _, _ = mqar_accuracy(m, cfg, 8, seed + 1, device)
                    acc[a].append(ov)
                    line += f"{a}={ov:.4f}  "
                print(line)
            cell = {"d_model": d_model, "n_layers": nl, "n_heads": nh}
            for a in arms:
                mu, sd = mean_std(acc[a])
                cell[a] = {"mean": mu, "std": sd, "per_seed": acc[a]}
            cell["valid"] = bool(cell["attn"]["mean"] >= 0.90)
            cell["holo_contribution_pp"] = round(
                100 * (cell["holo_on"]["mean"] - cell["holo_off"]["mean"]), 2)
            cells[key] = cell
            print(f"  -> holo_on {cell['holo_on']['mean']:.4f} ± {cell['holo_on']['std']:.4f}"
                  f"   holo_off {cell['holo_off']['mean']:.4f}"
                  f"   attn {cell['attn']['mean']:.4f}"
                  f"   contrib {cell['holo_contribution_pp']:+.2f}pp"
                  f"   {'VALID' if cell['valid'] else 'VOID(attn<0.90)'}")

    # ── grid summary ──
    print("\n" + "=" * 78)
    print("GRID — holo_on recall mean ± std  (contrib over holo_off floor; * = attn gate failed)")
    print(f"{'':6s}" + "".join(f"  H{h:<14d}" for h in heads_grid))
    base_key = "L2_H4"
    base_mean = cells.get(base_key, {}).get("holo_on", {}).get("mean", None)
    for nl in layers_grid:
        row = f"L{nl:<5d}"
        for nh in heads_grid:
            c = cells[f"L{nl}_H{nh}"]
            star = "" if c["valid"] else "*"
            row += f"  {c['holo_on']['mean']:.4f}±{c['holo_on']['std']:.4f}{star:1s}"
        print(row)

    print("\nholo_on vs the L2_H4 baseline (7.18% reference):")
    for nl in layers_grid:
        for nh in heads_grid:
            c = cells[f"L{nl}_H{nh}"]
            delta = (100 * (c["holo_on"]["mean"] - base_mean)) if base_mean is not None else float("nan")
            print(f"  L{nl}_H{nh}: holo_on {c['holo_on']['mean']:.4f}  "
                  f"(Δ vs L2_H4 = {delta:+.2f}pp)  contrib {c['holo_contribution_pp']:+.2f}pp  "
                  f"{'VALID' if c['valid'] else 'VOID'}")

    # ── verdict: which axis (if any) moves recall ──
    def cell_mean(l, h):
        return cells[f"L{l}_H{h}"]["holo_on"]["mean"]

    # depth effect at fixed heads (averaged over heads): L_max - L_min
    depth_gain = {}
    for nh in heads_grid:
        depth_gain[nh] = cell_mean(max(layers_grid), nh) - cell_mean(min(layers_grid), nh)
    # head effect at fixed depth: H_max - H_min
    head_gain = {}
    for nl in layers_grid:
        head_gain[nl] = cell_mean(nl, max(heads_grid)) - cell_mean(nl, min(heads_grid))

    mean_depth_gain = sum(depth_gain.values()) / len(depth_gain)
    mean_head_gain = sum(head_gain.values()) / len(head_gain)

    # noise scale: typical 2σ across holo_on cells
    typ_sd = sum(cells[k]["holo_on"]["std"] for k in cells) / len(cells)
    band = 2 * typ_sd

    depth_helps = mean_depth_gain > band
    heads_help = mean_head_gain > band

    if depth_helps and not heads_help:
        interp = ("DEPTH-LIMITED: recall climbs with n_layers (induction-head-style "
                  "multi-stage retrieval), not with heads. Next lever: deeper stack.")
    elif heads_help and not depth_helps:
        interp = ("COLLISION-LIMITED: recall climbs with n_heads (more parallel memories), "
                  "not with depth. Next lever: more parallel key-value memories / wider state.")
    elif depth_helps and heads_help:
        interp = ("BOTH depth AND heads lift recall — retrieval stack is undersized on both "
                  "axes. Next lever: jointly scale layers and heads.")
    else:
        interp = ("FLAT across the whole grid (no axis clears 2σ). The cap is the WRITE/READ "
                  "KEY SHARING, not stack capacity. Next lever: separate W_key_write/W_key_read "
                  "and/or learned phase_scale — the binding mechanism, NOT depth/width.")

    verdict = {
        "mean_depth_gain_pp": round(100 * mean_depth_gain, 2),
        "mean_head_gain_pp": round(100 * mean_head_gain, 2),
        "depth_gain_per_heads_pp": {f"H{h}": round(100 * v, 2) for h, v in depth_gain.items()},
        "head_gain_per_layers_pp": {f"L{l}": round(100 * v, 2) for l, v in head_gain.items()},
        "noise_band_2sigma_pp": round(100 * band, 2),
        "depth_helps": bool(depth_helps),
        "heads_help": bool(heads_help),
        "all_cells_valid": all(cells[k]["valid"] for k in cells),
        "interpretation": interp,
    }

    print("\n" + "=" * 78)
    print("VERDICT")
    print(f"  mean depth gain (L{max(layers_grid)}-L{min(layers_grid)}, avg over heads): "
          f"{100*mean_depth_gain:+.2f} pp   per-heads {verdict['depth_gain_per_heads_pp']}")
    print(f"  mean head  gain (H{max(heads_grid)}-H{min(heads_grid)}, avg over layers): "
          f"{100*mean_head_gain:+.2f} pp   per-layers {verdict['head_gain_per_layers_pp']}")
    print(f"  noise band (2σ, typical)         : {100*band:.2f} pp")
    print(f"  all cells valid (attn ≥ 0.90)    : {verdict['all_cells_valid']}")
    print(f"  >>> {interp}")

    out = {
        "config": {"steps": args.steps, "seeds": seeds, "n_pairs": args.n_pairs,
                   "train_len": args.train_len, "d_head": args.d_head,
                   "layers_grid": layers_grid, "heads_grid": heads_grid,
                   "lr": args.lr, "chance": chance, "device": "cpu",
                   "readout": "tanh_m", "d_model_rule": "n_heads*d_head"},
        "cells": cells,
        "baseline_L2_H4_holo_on": base_mean,
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults written to {args.out}  ({out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
