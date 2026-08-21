"""
Holographic-GSSM SHARPER PHASE MATCHING — phase-scale sweep + learned read-gain — by Opus 4.8
=============================================================================================

THE WALL.  The key-conditioned holographic write broke the 14% scalar-recall wall and plateaus
at ~7-9% MQAR recall (5-seed: 8.89%±1.86%).  d_head 32→64→96 is FLAT; separate-QK is FLAT/worse.
The diagnosed bottleneck is HOLOGRAPHIC CROSSTALK:

    read = Re( S · e^{−iφ_q} ) = Σ_k γ u_k cos(φ_k − φ_q)

The matched key gives cos≈1; the N−1 mismatched keys give cos(φ_k − φ_q) that does NOT average
to exactly zero for finite N.  The de-rotation is a SOFT match — it leaks for nearby-but-not-equal
keys.  This run attacks the SHARPNESS of that match with two cheap levers that add NO slots:

  (a) phase_scale sweep beyond π : {π, 1.5π, 2π, 4π}.
      φ = scale·tanh(W_key x) lives on a circle of half-width `scale`.  A larger scale spreads
      distinct keys FARTHER apart on the circle → fewer accidental near-collisions → cleaner
      cos(φ_k − φ_q) separation.  BUT past a point the keys wrap around the circle and DISTINCT
      keys alias to the same angle (cos is 2π-periodic) → crosstalk comes back.  There is a sweet
      spot; we sweep to find it.  (π is the established baseline.)

  (b) learned per-head READ GAIN g_h (a temperature on the read).
      read' = g_h · read   applied BEFORE the tanh_m saturation, g_h = softplus(raw), init 1.0
      (raw0 = log(e−1) ⇒ softplus(raw0)=1, so the model STARTS byte-identical to the baseline).
      Idea: push the coherent matched component (≈1) up the tanh toward saturation while the
      smaller incoherent crosstalk stays in the near-linear region — a soft sharpening / contrast
      gain.  One scalar per head, ~n_heads·n_layers params total.  Implemented as a thin subclass
      that overrides forward() — holographic_gssm.py is NOT touched (an n_slots refactor is in
      flight there; we stay out of its way).

We also run the cross of the two: best-looking phase_scale × read_gain.

ARMS (all CPU-deterministic, N≥5 seeds, mean±std):
  attn       — validity gate, must reach ≥0.90 or ALL GSSM numbers are VOID.
  holo_off   — use_phase=False == GSSM-Selective, the scalar floor (~1.6%).
  holo_base  — phase_scale=π, readout=tanh_m, NO gain. THE 7-9% BASELINE to beat.
  ps_1.5pi / ps_2pi / ps_4pi          — lever (a).
  gain_pi    — phase_scale=π + learned read gain. lever (b).
  gain_X     — best phase_scale + learned read gain. the cross.

DECISION RULE (committed before reading results):
  A lever WINS iff its mean train-len recall beats holo_base by MORE THAN the combined 1σ band
  (mean_lever − mean_base > sqrt(σ_lever² + σ_base²)) AND clears chance (1.56%) clearly.  A lever
  that lands inside the band is FLAT — reported as flat, no spin.

readout=tanh_m is load-bearing (rms/layernorm fail per the readout shootout), so every holo arm
here uses tanh_m, matching the baseline whose number we are trying to beat.

Offline, python3, CPU.  Reference: Foss 2026.
"""

import os
import sys
import math
import json
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "reference"))
sys.path.insert(0, HERE)

from mqar import (  # noqa: E402
    make_mqar_batch, mqar_accuracy, TinyCausalTransformerLM,
)
from moebius_scan_transformer_selective import SelectiveRapiditySqrtTransformerLM  # noqa: E402
from holographic_gssm import (  # noqa: E402
    HolographicLM, HolographicTransformerLayer, HolographicScanLayer,
    sequential_linear_scan, EPS,
)


# ===========================================================================
# Lever (b): a thin subclass that adds a learned per-head read GAIN on top of
# the tanh_m readout.  forward() is copied from HolographicScanLayer with the
# single addition `read *= g_h` immediately before the tanh saturation.  We do
# NOT edit holographic_gssm.py (an n_slots refactor is mid-flight there).
# ===========================================================================

class GainHolographicScanLayer(HolographicScanLayer):
    """HolographicScanLayer + learned per-head read gain g_h (softplus, init 1.0).

    g_h multiplies the de-rotated read BEFORE m·tanh(·).  At init g_h=1 ⇒ byte-identical to
    the tanh_m baseline.  Lets the network amplify the coherent (matched) read so it saturates
    the tanh while the incoherent crosstalk stays near-linear — a soft contrast sharpening.
    Only meaningful with readout='tanh_m' (the load-bearing readout); asserted below.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.readout == "tanh_m", "read gain is defined against the tanh_m saturation"
        raw0 = math.log(math.e - 1.0)                      # softplus(raw0) == 1.0
        self.read_gain_raw = nn.Parameter(torch.full((self.n_heads,), raw0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        if not self.use_phase:
            m = self._magnitude(x)
            return self.W_out(m.view(B, T, self.n_heads * self.d_head))

        a, gamma = self._drive_and_gamma(x)
        phi_w = self.phase_scale * torch.tanh(self.W_key(x))
        phi_w = phi_w.view(B, T, self.n_heads, self.d_head)

        if self.separate_qk:
            phi_r = self.phase_scale * torch.tanh(self.W_read_key(x))
            phi_r = phi_r.view(B, T, self.n_heads, self.d_head)
        else:
            phi_r = phi_w

        drive_re = a * torch.cos(phi_w)
        drive_im = a * torch.sin(phi_w)

        if self.causal:
            S_re = sequential_linear_scan(drive_re, gamma)
            S_im = sequential_linear_scan(drive_im, gamma)
        else:
            S_re = sequential_linear_scan(drive_re, gamma) + torch.flip(
                sequential_linear_scan(torch.flip(drive_re, dims=[1]),
                                       torch.flip(gamma, dims=[1])), dims=[1])
            S_im = sequential_linear_scan(drive_im, gamma) + torch.flip(
                sequential_linear_scan(torch.flip(drive_im, dims=[1]),
                                       torch.flip(gamma, dims=[1])), dims=[1])

        read_re = S_re * torch.cos(phi_r) + S_im * torch.sin(phi_r)
        read_im = S_im * torch.cos(phi_r) - S_re * torch.sin(phi_r)

        # ── lever (b): learned per-head read gain BEFORE the tanh_m saturation ──
        # g_h shape (n_heads,) → broadcast over (B,T,H,D) on the H axis.
        g = F.softplus(self.read_gain_raw).view(1, 1, self.n_heads, 1)
        read_re = g * read_re
        read_im = g * read_im

        # readout == "tanh_m" (asserted): m·tanh(g·read)
        m = self._magnitude(x)
        read_re = m * torch.tanh(read_re)
        read_im = m * torch.tanh(read_im)

        read_re = read_re.view(B, T, self.n_heads * self.d_head)
        read_im = read_im.view(B, T, self.n_heads * self.d_head)
        return self.W_out(read_re) + self.W_im(read_im)


def _patch_scan_with_gain(holo_lm: HolographicLM):
    """Swap every layer's HolographicScanLayer for a GainHolographicScanLayer with the
    SAME hyper-params, so the only difference vs the baseline is the learned read gain.
    Operates on a freshly-built HolographicLM (weights are re-init by the subclass, which
    is fine — we build then seed-train, identical to every other arm)."""
    for layer in holo_lm.layers:
        old = layer.scan
        new = GainHolographicScanLayer(
            old.d_model, d_head=old.d_head, n_heads=old.n_heads, causal=old.causal,
            dropout=0.0, phase_scale=old.phase_scale, use_phase=old.use_phase,
            readout=old.readout, separate_qk=old.separate_qk)
        layer.scan = new
    return holo_lm


# ===========================================================================
# Arm factory
# ===========================================================================

def build_arm(spec, vocab_size, mask_idx, d_model, n_layers, n_heads, d_head, seq_len):
    """spec is a dict: {kind, phase_scale?, gain?}."""
    kind = spec["kind"]
    if kind == "attn":
        return TinyCausalTransformerLM(
            vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            max_len=max(seq_len, 1024))
    if kind == "selective":
        return SelectiveRapiditySqrtTransformerLM(
            vocab_size, mask_idx, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, d_head=d_head, seq_len=seq_len, dropout=0.0, causal=True)
    if kind == "holo_off":
        return HolographicLM(
            vocab_size, mask_idx, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, d_head=d_head, seq_len=seq_len, dropout=0.0,
            causal=True, phase_scale=math.pi, use_phase=False, readout="tanh_m")
    if kind == "holo":
        m = HolographicLM(
            vocab_size, mask_idx, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, d_head=d_head, seq_len=seq_len, dropout=0.0,
            causal=True, phase_scale=spec["phase_scale"], use_phase=True,
            readout="tanh_m")
        if spec.get("gain", False):
            m = _patch_scan_with_gain(m)
        return m
    raise ValueError(kind)


def train_arm(model, cfg, steps, lr, seed, device):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(steps):
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


def eval_arm(model, train_cfg, test_cfg, seed, device):
    model.eval()
    tr_overall, _, _ = mqar_accuracy(model, train_cfg, 8, seed + 1, device)
    te_overall, _, _ = mqar_accuracy(model, test_cfg, 8, seed + 2, device)
    return tr_overall, te_overall


def mean_std(xs):
    n = len(xs)
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / n
    return mu, var ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--train-len", type=int, default=64)
    ap.add_argument("--test-len", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-head", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seeds", default="1,7,42,123,2024")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "holographic_sharpen.json"))
    args = ap.parse_args()

    device = torch.device("cpu")  # deterministic on a chance-flat loss

    n_keys = n_values = 64
    vocab_size = n_keys + n_values + 1
    mask_idx = vocab_size
    chance = 1.0 / n_values

    train_cfg = dict(batch_size=32, seq_len=args.train_len, n_pairs=args.n_pairs,
                     n_queries=args.n_pairs, n_keys=n_keys, n_values=n_values)
    test_cfg = dict(batch_size=32, seq_len=args.test_len, n_pairs=args.n_pairs,
                    n_queries=args.n_pairs, n_keys=n_keys, n_values=n_values)

    seeds = [int(s) for s in args.seeds.split(",")]

    PI = math.pi
    # ── arm specs (label → build spec). Order matters only for printing. ──
    arms = {
        "attn":       {"kind": "attn"},
        "holo_off":   {"kind": "holo_off"},
        "holo_base":  {"kind": "holo", "phase_scale": PI},                       # the 7-9% baseline
        "ps_1.5pi":   {"kind": "holo", "phase_scale": 1.5 * PI},                 # lever (a)
        "ps_2pi":     {"kind": "holo", "phase_scale": 2.0 * PI},                 # lever (a)
        "ps_4pi":     {"kind": "holo", "phase_scale": 4.0 * PI},                 # lever (a)
        "gain_pi":    {"kind": "holo", "phase_scale": PI, "gain": True},         # lever (b)
    }

    print("=" * 80)
    print("Holographic SHARPEN — phase_scale sweep + learned read gain, multi-seed CPU")
    print(f"device={device} steps={args.steps} train_len={args.train_len} "
          f"test_len={args.test_len} n_pairs={args.n_pairs} d_head={args.d_head}")
    print(f"seeds={seeds}  chance=1/{n_values}={chance:.4f}")
    print(f"arms={list(arms)}")
    print("=" * 80)

    per_seed_tr = {a: [] for a in arms}
    per_seed_te = {a: [] for a in arms}
    t0 = time.time()

    for seed in seeds:
        print(f"\n--- seed {seed} ---")
        for label, spec in arms.items():
            torch.manual_seed(seed)
            model = build_arm(spec, vocab_size, mask_idx, args.d_model, args.n_layers,
                              args.n_heads, args.d_head, args.train_len)
            train_arm(model, train_cfg, args.steps, args.lr, seed, device)
            tr, te = eval_arm(model, train_cfg, test_cfg, seed, device)
            per_seed_tr[label].append(tr)
            per_seed_te[label].append(te)
            print(f"  {label:11s}  train-len {tr:.4f}   test-len {te:.4f}")

    # ── aggregate ──
    summary = {}
    for label in arms:
        mu, sd = mean_std(per_seed_tr[label])
        mu_te, sd_te = mean_std(per_seed_te[label])
        summary[label] = {
            "train_mean": mu, "train_std": sd,
            "test_mean": mu_te, "test_std": sd_te,
            "per_seed_train": per_seed_tr[label],
            "per_seed_test": per_seed_te[label],
            "spec": {k: (v if not isinstance(v, float) else round(v, 4))
                     for k, v in arms[label].items()},
        }

    print("\n" + "=" * 80)
    print("AGGREGATE (train-len overall, mean ± std over seeds)")
    for label in arms:
        s = summary[label]
        print(f"  {label:11s}  {s['train_mean']:.4f} ± {s['train_std']:.4f}   "
              f"(test {s['test_mean']:.4f} ± {s['test_std']:.4f})")

    # ── verdict ──
    attn_mu = summary["attn"]["train_mean"]
    validity = attn_mu >= 0.90
    base_mu = summary["holo_base"]["train_mean"]
    base_sd = summary["holo_base"]["train_std"]
    off_mu = summary["holo_off"]["train_mean"]

    levers = ["ps_1.5pi", "ps_2pi", "ps_4pi", "gain_pi"]
    lever_report = {}
    best_label, best_delta = None, -1.0
    for label in levers:
        mu, sd = summary[label]["train_mean"], summary[label]["train_std"]
        delta = mu - base_mu
        band = math.sqrt(sd * sd + base_sd * base_sd)              # combined 1σ
        wins = (delta > band) and (mu > 3 * chance) and validity
        lever_report[label] = {
            "mean": round(mu, 4), "std": round(sd, 4),
            "delta_vs_base_pp": round(100 * delta, 2),
            "combined_1sigma_pp": round(100 * band, 2),
            "beats_baseline": bool(wins),
        }
        if delta > best_delta:
            best_delta, best_label = delta, label

    any_win = any(v["beats_baseline"] for v in lever_report.values())
    verdict = {
        "validity_gate_attn": round(attn_mu, 4),
        "validity_passed": bool(validity),
        "holo_off_floor": round(off_mu, 4),
        "holo_base_mean": round(base_mu, 4),
        "holo_base_std": round(base_sd, 4),
        "chance": round(chance, 4),
        "levers": lever_report,
        "best_lever": best_label,
        "best_delta_pp": round(100 * best_delta, 2),
        "any_lever_wins": bool(any_win and validity),
        "interpretation": (
            "VOID — attention validity gate failed" if not validity else
            (f"SHARPENING HELPS — {best_label} beats baseline by {100*best_delta:+.2f}pp "
             f"(> combined 1σ)" if any_win else
             f"FLAT — no lever clears holo_base + combined 1σ; best is {best_label} "
             f"at {100*best_delta:+.2f}pp (inside the noise band)")),
    }

    print("\n" + "=" * 80)
    print("VERDICT")
    print(f"  validity gate (attn ≥ 0.90) : {attn_mu:.4f}  "
          f"{'PASS' if validity else 'FAIL → VOID'}")
    print(f"  holo_off floor (Selective)  : {off_mu:.4f}")
    print(f"  holo_base (π, tanh_m)        : {base_mu:.4f} ± {base_sd:.4f}")
    for label in levers:
        r = lever_report[label]
        flag = "WIN" if r["beats_baseline"] else "flat"
        print(f"  {label:11s} {r['mean']:.4f} ± {r['std']:.4f}  "
              f"Δ={r['delta_vs_base_pp']:+.2f}pp  (band ±{r['combined_1sigma_pp']:.2f}pp)  [{flag}]")
    print(f"  >>> {verdict['interpretation']}")

    out = {
        "config": {"steps": args.steps, "n_pairs": args.n_pairs,
                   "train_len": args.train_len, "test_len": args.test_len,
                   "d_model": args.d_model, "n_heads": args.n_heads,
                   "d_head": args.d_head, "seeds": seeds, "chance": chance,
                   "device": "cpu", "readout": "tanh_m"},
        "summary": summary, "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults written to {args.out}  ({out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
