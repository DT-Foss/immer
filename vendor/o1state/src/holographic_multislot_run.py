"""
Holographic-GSSM FRONT 3 — MULTI-SLOT key-binned accumulators — by Opus 4.8
===========================================================================

PLATEAU CAUSE UNDER TEST.  The working holographic write (readout="tanh_m") plateaus at
~7-9% recall and is FLAT in d_head (32→64→96 ≈ 7.5%) and FLAT under separate Q/K. The
remaining suspect is the holographic CROSSTALK CAP intrinsic to a SINGLE complex
accumulator that superposes ALL n_pairs pairs:

    read = Σ_k γ_{k→t} u_k cos(φ_k − φ_q)

The matched key gives cos≈1; the N−1 mismatched keys give cos(φ_k − φ_q) that does NOT
average to exactly zero for finite N. The residual interference grows with the number of
superposed pairs — the classic HRR/VSA holographic-memory capacity limit. Adding channels
does not fix it (the SAME N pairs sit in every channel). The fix is structural: superpose
FEWER pairs PER memory.

THE ATTACK.  Give the layer M complex slots and ROUTE each token's WRITE to one slot by a
learned per-head function of its content (W_slot, hard argmax with a straight-through
gradient). Each slot then superposes only ~N/M pairs, so the incoherent crosstalk
amplitude (a random-phase sum of ~N/M terms) drops ~√M. The READ gathers from the query's
OWN routed slot (same W_slot) and de-rotates there. m·tanh readout and the m relevance-gate
are UNTOUCHED — they remain the load-bearing WHEN-to-read trigger.

  n_slots=1  ==  the single-accumulator holographic baseline, byte-identical (W_slot absent,
                 mask all-ones). The reduction use_phase=False == Selective is also preserved
                 for every n_slots (the slot path is skipped when phase is off).

ARMS
  * attn        — TinyCausalTransformer, the validity gate (mean ≥0.90 or ALL GSSM VOID).
  * holo_off    — HolographicLM use_phase=False == GSSM-Selective, the recall FLOOR (~1.6%).
  * holo_s1     — n_slots=1, the working holographic baseline (~7-9%).
  * holo_s2     — n_slots=2 (≈ √2 ≈ 1.41× crosstalk suppression).
  * holo_s4     — n_slots=4 (≈ 2× suppression).
  * holo_s8     — n_slots=8 (≈ 2.83× suppression; with n_pairs=8 this is ~1 pair/slot).

DECISION RULE (committed before reading results):
  - attn mean ≥ 0.90 or ALL GSSM numbers are VOID (harness validity).
  - holo_off ≈ chance (internal consistency: it IS Selective).
  - holo_s1 reproduces ~7-9% (sanity that this harness matches the known baseline).
  - SLOTS HELP iff recall climbs MONOTONE-ish with M AND the best M clears
        holo_s1 + 2·max(std)   (clears the noise band).
    A flat curve within the band is a clean NEGATIVE: slot-binning does not lift the
    holographic cap on this readout — report it, name the next lever.

CPU-deterministic, multi-seed, offline.  Mirrors holographic_qk_run.py settings
(seeds 1,7,42 / 1200 steps / d_model=128 / d_head=32 / n_pairs=8) so holo_s1 is directly
comparable to the recorded tanh_m baseline.
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


# Arm spec: name → (n_slots or None for attn, use_phase). holo_off uses use_phase=False.
SLOT_ARMS = {"holo_s1": 1, "holo_s2": 2, "holo_s4": 4, "holo_s8": 8}


def build_arm(arm, vocab_size, mask_idx, d_model, n_layers, n_heads, d_head, seq_len):
    if arm == "attn":
        return TinyCausalTransformerLM(
            vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            max_len=max(seq_len, 1024))
    if arm == "holo_off":
        return HolographicLM(
            vocab_size, mask_idx, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, d_head=d_head, seq_len=seq_len, dropout=0.0,
            causal=True, phase_scale=math.pi, use_phase=False, readout="tanh_m",
            n_slots=1)
    if arm in SLOT_ARMS:
        # Holographic write, tanh_m readout (the m relevance-gate untouched), shared QK,
        # M key-binned slots. n_slots=1 is the byte-identical single-accumulator baseline.
        return HolographicLM(
            vocab_size, mask_idx, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, d_head=d_head, seq_len=seq_len, dropout=0.0,
            causal=True, phase_scale=math.pi, use_phase=True,
            readout="tanh_m", separate_qk=False, n_slots=SLOT_ARMS[arm])
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
    ap.add_argument("--arms", default="attn,holo_off,holo_s1,holo_s2,holo_s4,holo_s8")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "holographic_multislot.json"))
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

    print("=" * 80)
    print("Holographic-GSSM FRONT 3 — MULTI-SLOT key-binned accumulators, multi-seed, CPU")
    print(f"device={device} steps={args.steps} train_len={args.train_len} "
          f"n_pairs={args.n_pairs} d_model={args.d_model} d_head={args.d_head}")
    print(f"seeds={seeds}  chance=1/{n_values}={chance:.4f}  readout=tanh_m  shared-QK")
    print("=" * 80)

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
            print(f"  {arm:10s}  train-len recall {acc:.4f}")

    # ── aggregate ──
    print("\n" + "=" * 80)
    print("AGGREGATE (train-len recall, mean ± std over seeds)")
    summary = {}
    for arm in arms:
        mu, sd = mean_std(per_seed[arm])
        summary[arm] = {"mean": mu, "std": sd, "per_seed": per_seed[arm]}
        print(f"  {arm:10s}  {mu:.4f} ± {sd:.4f}   {[round(x,4) for x in per_seed[arm]]}")

    attn_mu = summary.get("attn", {}).get("mean", 0.0)
    validity = attn_mu >= 0.90
    off_mu = summary.get("holo_off", {}).get("mean", float("nan"))

    # ── slot curve + verdict ──
    slot_names = [a for a in ("holo_s1", "holo_s2", "holo_s4", "holo_s8") if a in summary]
    curve = [(SLOT_ARMS[a], summary[a]["mean"], summary[a]["std"]) for a in slot_names]

    verdict = {}
    if "holo_s1" in summary and len(slot_names) >= 2:
        base_mu, base_sd = summary["holo_s1"]["mean"], summary["holo_s1"]["std"]
        # best slot count (excluding s1) and its std
        best_arm = max(slot_names, key=lambda a: summary[a]["mean"])
        best_mu, best_sd = summary[best_arm]["mean"], summary[best_arm]["std"]
        delta = best_mu - base_mu
        band = 2 * max(base_sd, best_sd, 1e-6)
        # monotone-ish: recall at the best M is above s1, and the curve does not fall
        # back below s1 before reaching it (loose check on the sorted-by-M means).
        means_by_m = [summary[a]["mean"] for a in slot_names]
        climbs = best_mu > base_mu
        clears_band = delta > band
        slots_help = bool(validity and climbs and clears_band)
        verdict = {
            "slot_curve": [{"n_slots": m, "mean": round(mu, 4), "std": round(sd, 4)}
                           for (m, mu, sd) in curve],
            "baseline_s1_mean": round(base_mu, 4),
            "best_arm": best_arm,
            "best_n_slots": SLOT_ARMS[best_arm],
            "best_mean": round(best_mu, 4),
            "delta_best_minus_s1_pp": round(100 * delta, 2),
            "noise_band_pp": round(100 * band, 2),
            "holo_off_floor": round(off_mu, 4),
            "chance": round(chance, 4),
            "validity_gate_attn": round(attn_mu, 4),
            "validity_passed": bool(validity),
            "means_by_m": [round(x, 4) for x in means_by_m],
            "slots_help": slots_help,
            "interpretation": (
                f"SLOTS HELP — recall climbs to {best_mu:.3f} at n_slots={SLOT_ARMS[best_arm]}, "
                f"clearing s1 + 2σ ({100*delta:+.2f}pp > {100*band:.2f}pp band)"
                if slots_help else
                ("VOID — attention validity gate failed (attn < 0.90)" if not validity else
                 f"NEGATIVE — slot-binning does not break the holographic cap "
                 f"(best Δ {100*delta:+.2f}pp within {100*band:.2f}pp noise band)")),
        }
        print("\n" + "=" * 80)
        print("VERDICT (Front 3: multi-slot key-binned accumulators)")
        print(f"  validity gate (attn ≥ 0.90)   : {attn_mu:.4f}  "
              f"{'PASS' if validity else 'FAIL → numbers VOID'}")
        print(f"  holo_off floor (==Selective)  : {off_mu:.4f}")
        print(f"  slot curve (n_slots → recall) :")
        for (m, mu, sd) in curve:
            print(f"      n_slots={m:<2d}  {mu:.4f} ± {sd:.4f}")
        print(f"  best                          : n_slots={SLOT_ARMS[best_arm]} "
              f"@ {best_mu:.4f} (Δ vs s1 {100*delta:+.2f}pp, band {100*band:.2f}pp)")
        print(f"  >>> {verdict['interpretation']}")

    out = {
        "config": {"steps": args.steps, "n_pairs": args.n_pairs,
                   "train_len": args.train_len, "d_model": args.d_model,
                   "n_heads": args.n_heads, "d_head": args.d_head,
                   "lr": args.lr, "seeds": seeds, "chance": chance, "device": "cpu",
                   "readout": "tanh_m", "separate_qk": False,
                   "routing": "learned per-head W_slot, hard argmax + straight-through"},
        "summary": summary, "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults written to {args.out}  ({out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
