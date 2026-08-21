"""
Holographic-GSSM FRONT 2 — separate WRITE/READ key projections — by Opus 4.8
============================================================================

PLATEAU CAUSE UNDER TEST.  The working holographic write (readout="tanh_m") plateaus at
~7% recall.  In that baseline a SINGLE W_key does double duty: it sets the WRITE angle
φ_write (where token t deposits its value on the phase circle) AND the READ angle φ_read
(how a query de-rotates the accumulator).  Attention does NOT do this — it learns a Key
projection and a SEPARATE Query projection, precisely so the match Q·Kᵀ can be sharpened
independently of how keys are laid out.  Front 2 asks: is the shared W_key a binding
bottleneck?  Give the read its own learned query angle (W_read_key) and see if matching
sharpens past 7%.

ARMS
  * attn       — TinyCausalTransformer, the validity gate (must reach ≥0.90 or all VOID).
  * holo_off   — HolographicLM use_phase=False == GSSM-Selective, the recall FLOOR (~1.6%).
  * holo_shared— readout="tanh_m", separate_qk=False — the WORKING 7% baseline (φ_read==φ_write).
  * holo_sepqk — readout="tanh_m", separate_qk=True  — W_key writes φ_write, W_read_key reads φ_read.

Everything else is held fixed (same drive, same γ, same m·tanh readout — the m-gate is the
load-bearing relevance trigger and is NOT touched).  The ONLY change between shared and
sepqk is whether the read de-rotation angle comes from W_key or from a separate W_read_key.

DECISION RULE (committed before reading results):
  - attn mean ≥ 0.90 or ALL GSSM numbers are VOID.
  - holo_off ≈ chance (internal consistency: it IS Selective).
  - holo_shared must reproduce ~7% (sanity that the harness matches the known baseline).
  - sepqk BEATS shared iff  sepqk_mean − shared_mean > 2·max(std)  (clears the noise band).
    A flat/negative Δ within the band is a clean NEGATIVE: separating Q/K does not break
    the 7% wall on this readout — report it, move to the next lever.

CPU-deterministic, multi-seed, offline.  Mirrors holographic_readout_shootout.py settings
(3 seeds 1,7,42 / 1200 steps / d_model=128 / d_head=32) so holo_shared is directly
comparable to the recorded 7.18% tanh_m baseline.
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


def build_arm(arm, vocab_size, mask_idx, d_model, n_layers, n_heads, d_head, seq_len):
    if arm == "attn":
        return TinyCausalTransformerLM(
            vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            max_len=max(seq_len, 1024))
    if arm == "holo_off":
        return HolographicLM(
            vocab_size, mask_idx, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, d_head=d_head, seq_len=seq_len, dropout=0.0,
            causal=True, phase_scale=math.pi, use_phase=False, readout="tanh_m")
    if arm == "holo_shared":
        # The working 7% baseline: one W_key for write AND read.
        return HolographicLM(
            vocab_size, mask_idx, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, d_head=d_head, seq_len=seq_len, dropout=0.0,
            causal=True, phase_scale=math.pi, use_phase=True,
            readout="tanh_m", separate_qk=False)
    if arm == "holo_sepqk":
        # Front 2: W_key writes φ_write, separate W_read_key reads φ_read.
        return HolographicLM(
            vocab_size, mask_idx, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, d_head=d_head, seq_len=seq_len, dropout=0.0,
            causal=True, phase_scale=math.pi, use_phase=True,
            readout="tanh_m", separate_qk=True)
    raise ValueError(arm)


def train_arm(model, cfg, steps, lr, seed, device):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for step in range(steps):
        tokens, targets, mask, _ = make_mqar_batch(generator=gen, device=device, **cfg)
        logits = model(tokens)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
        loss = (loss * mask.reshape(-1).float()).sum() / (mask.sum() + 1e-6)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    return model


def mean_std(xs):
    n = len(xs)
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / n
    return mu, var ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--train-len", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-head", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seeds", default="1,7,42")
    ap.add_argument("--arms", default="attn,holo_off,holo_shared,holo_sepqk")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "holographic_qk.json"))
    args = ap.parse_args()

    device = torch.device("cpu")  # deterministic

    n_keys = n_values = 64
    vocab_size = n_keys + n_values + 1
    mask_idx = vocab_size
    chance = 1.0 / n_values

    train_cfg = dict(batch_size=32, seq_len=args.train_len, n_pairs=args.n_pairs,
                     n_queries=args.n_pairs, n_keys=n_keys, n_values=n_values)

    seeds = [int(s) for s in args.seeds.split(",")]
    arms = args.arms.split(",")

    print("=" * 78)
    print("Holographic-GSSM FRONT 2 — separate write/read key, multi-seed, CPU-deterministic")
    print(f"device={device} steps={args.steps} train_len={args.train_len} "
          f"n_pairs={args.n_pairs} d_model={args.d_model} d_head={args.d_head}")
    print(f"seeds={seeds}  chance=1/{n_values}={chance:.4f}")
    print("=" * 78)

    per_seed = {arm: [] for arm in arms}
    t0 = time.time()

    for seed in seeds:
        print(f"\n--- seed {seed} ---")
        for arm in arms:
            torch.manual_seed(seed)
            model = build_arm(arm, vocab_size, mask_idx, args.d_model, args.n_layers,
                              args.n_heads, args.d_head, args.train_len)
            train_arm(model, train_cfg, args.steps, args.lr, seed, device)
            model.eval()
            acc, _, _ = mqar_accuracy(model, train_cfg, 8, seed + 1, device)
            per_seed[arm].append(acc)
            print(f"  {arm:12s}  train-len recall {acc:.4f}")

    # ── aggregate ──
    print("\n" + "=" * 78)
    print("AGGREGATE (train-len recall, mean ± std over seeds)")
    summary = {}
    for arm in arms:
        mu, sd = mean_std(per_seed[arm])
        summary[arm] = {"mean": mu, "std": sd, "per_seed": per_seed[arm]}
        print(f"  {arm:12s}  {mu:.4f} ± {sd:.4f}")

    attn_mu = summary.get("attn", {}).get("mean", 0.0)
    validity = attn_mu >= 0.90

    verdict = {}
    if "holo_sepqk" in summary and "holo_shared" in summary:
        sep_mu, sep_sd = summary["holo_sepqk"]["mean"], summary["holo_sepqk"]["std"]
        sh_mu, sh_sd = summary["holo_shared"]["mean"], summary["holo_shared"]["std"]
        off_mu = summary.get("holo_off", {}).get("mean", float("nan"))
        delta = sep_mu - sh_mu
        band = 2 * max(sep_sd, sh_sd, 1e-6)
        beats = (delta > band) and validity
        verdict = {
            "separate_qk_mean": round(sep_mu, 4),
            "shared_qk_mean": round(sh_mu, 4),
            "holo_off_floor": round(off_mu, 4),
            "delta_pp": round(100 * delta, 2),
            "noise_band_pp": round(100 * band, 2),
            "chance": round(chance, 4),
            "validity_gate_attn": round(attn_mu, 4),
            "validity_passed": bool(validity),
            "sepqk_beats_shared": bool(beats),
            "interpretation": (
                "SEPARATE Q/K HELPS — sepqk clears shared + 2σ"
                if beats else
                ("VOID — attention validity gate failed" if not validity else
                 "NEGATIVE — separate Q/K does not beat shared-key on tanh_m (Δ within band)")),
        }
        print("\n" + "=" * 78)
        print("VERDICT (Front 2: separate write/read key)")
        print(f"  validity gate (attn ≥ 0.90)   : {attn_mu:.4f}  "
              f"{'PASS' if validity else 'FAIL → numbers VOID'}")
        print(f"  holo_off floor (==Selective)  : {off_mu:.4f}")
        print(f"  holo_shared (7% baseline)     : {sh_mu:.4f} ± {sh_sd:.4f}")
        print(f"  holo_sepqk  (separate Q/K)    : {sep_mu:.4f} ± {sep_sd:.4f}")
        print(f"  Δ (sepqk − shared)            : {100*delta:+.2f} pp")
        print(f"  noise band (2σ)               : {100*band:.2f} pp")
        print(f"  >>> {verdict['interpretation']}")

    out = {
        "config": {"steps": args.steps, "n_pairs": args.n_pairs,
                   "train_len": args.train_len, "d_model": args.d_model,
                   "n_heads": args.n_heads, "d_head": args.d_head,
                   "lr": args.lr, "seeds": seeds, "chance": chance, "device": "cpu",
                   "readout": "tanh_m"},
        "summary": summary, "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults written to {args.out}  ({out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
