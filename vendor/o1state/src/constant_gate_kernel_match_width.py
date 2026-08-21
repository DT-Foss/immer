"""
Constant-Gate Kernel-Match Falsifier AT FULL WIDTH (FRONT 5) — by Opus 4.8
==========================================================================

FRONT 5 question: the constant-gate kernel match was confirmed to 8.88e-16 at
d_small (d_model=32).  d512 was the HISTORICAL COLLAPSE WIDTH (FINAL_REPORT M1).
Does the machine-precision identity — trained constant-gate read map == closed-form
geometric-Toeplitz kernel — SURVIVE at d=512, or does width introduce drift?

This is the same protocol as constant_gate_kernel_match.py, parameterised over
width and run as a sweep d_model ∈ {512, 256, 128} (largest first).  For each
width that trains in a couple of minutes on this M4 we:

  1. BUILD a GSSM-Selective LM with W_gamma/W_alpha FROZEN (γ_t≡γ, α_t≡α constant).
  2. TRAIN BPTT on offline synthetic tokens (next-token CE).  Gates provably frozen.
  3. KERNEL MATCH: trained live scan read map z_scan  vs  constant_gamma_closed_form.
  4. KERNEL-RIDGE cross-check (explicit Toeplitz K, z=K·a, per-channel scale≈1).
  5. NEGATIVE CONTROL: a SELECTIVE (time-varying γ_t) model at the SAME width —
     its read map must NOT match any single geometric kernel.

The headline is the max abs match error at the LARGEST width that actually ran,
and the control/match contrast at that width.  No downloads, CPU, float64.

    python3 src/constant_gate_kernel_match_width.py
Writes constant_gate_kernel_match_width_results.json next to this file.

Reference: Foss 2026, "From Markov Chains to Minkowski Space".
"""

import os
import sys
import json
import time
import traceback

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
REF = os.path.join(REPO, "reference")

sys.path.insert(0, REF)
sys.path.insert(0, HERE)

import moebius_scan_transformer_selective as ref  # noqa: E402
from moebius_scan_transformer_selective import (  # noqa: E402
    SelectiveRapiditySqrtTransformerLM,
)
from parallel_scan import constant_gamma_closed_form  # noqa: E402

# Reuse the constant-gate machinery from the d-small experiment verbatim — same
# layer, same builder, same offline-token generator, same trainer, same frozen
# check.  Only the driver (width sweep) is new, so the kernel-match logic is
# byte-identical to the matched-to-8.88e-16 experiment.
from constant_gate_kernel_match import (  # noqa: E402
    build_constant_gate_lm,
    make_offline_tokens,
    train_bptt,
    assert_gates_frozen,
    kernel_match_constant_gate,
    kernel_ridge_crosscheck,
    selective_control_match,
)


def run_one_width(width_cfg, dtype, device):
    """Run the full constant-gate kernel-match protocol at one width.
    Returns a results dict, or raises on OOM/failure (caught by the driver)."""
    vocab_size = width_cfg["vocab_size"]
    mask_idx = vocab_size + 1
    seq_len = width_cfg["seq_len"]
    n_seqs = width_cfg["n_seqs"]
    d_model = width_cfg["d_model"]
    n_layers = width_cfg["n_layers"]
    n_heads = width_cfg["n_heads"]
    d_head = width_cfg["d_head"]
    steps = width_cfg["steps"]
    lr = width_cfg["lr"]
    batch = width_cfg["batch"]
    gamma_const = width_cfg["gamma_const"]
    alpha_const = width_cfg["alpha_const"]

    assert n_heads * d_head == d_model, (
        f"n_heads*d_head ({n_heads*d_head}) must equal d_model ({d_model})")

    print("\n" + "#" * 74)
    print(f"#  WIDTH d_model={d_model}  (n_heads={n_heads} x d_head={d_head}, "
          f"L={n_layers}, T={seq_len})")
    print("#" * 74)

    out = {"config": dict(width_cfg)}

    X = make_offline_tokens(vocab_size, seq_len, n_seqs, seed=1234).to(device)

    # ---- 1+2: build + train ----
    print(f"[1] BUILD constant-gate LM d_model={d_model} (W_gamma/W_alpha FROZEN)")
    model = build_constant_gate_lm(
        vocab_size, mask_idx, d_model, n_layers, n_heads, d_head, seq_len,
        gamma_const, alpha_const, causal=True,
    ).to(device=device, dtype=dtype)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"    trainable params {n_train:,} | frozen params {n_frozen:,}")

    print(f"[2] TRAIN BPTT {steps} steps (next-token CE)")
    t0 = time.time()
    losses = train_bptt(model, X, steps=steps, lr=lr, batch=batch, seed=7,
                        log_every=max(1, steps // 6))
    train_s = time.time() - t0
    print(f"    loss {losses[0]:.4f} -> {losses[-1]:.4f}  in {train_s:.1f}s")
    out["training"] = {
        "loss_first": losses[0], "loss_last": losses[-1],
        "loss_min": min(losses), "steps": len(losses),
        "n_trainable": n_train, "n_frozen": n_frozen, "wall_s": train_s,
    }

    frozen_report = assert_gates_frozen(model)
    print(f"    gates provably frozen after training: {frozen_report['all_frozen']}")
    out["frozen_gate_check_all_frozen"] = frozen_report["all_frozen"]

    # ---- 3: THE KERNEL MATCH ----
    print("[3] KERNEL MATCH — trained read map z  vs  constant_gamma_closed_form")
    x_probe = make_offline_tokens(vocab_size, seq_len, 8, seed=9999).to(device)
    km = kernel_match_constant_gate(model, x_probe)
    for r in km:
        print(f"    layer {r['layer']}: max|Δ| = {r['max_abs_err']:.3e}  "
              f"mean|Δ| = {r['mean_abs_err']:.3e}  rel = {r['rel_err_mean']:.3e}  "
              f"(z scale {r['z_scale_mean_abs']:.3e})")
    out["kernel_match_constant_gate"] = km
    km_max = max(r["max_abs_err"] for r in km)
    km_match = km_max < 1e-9  # float64 machine-precision identity threshold
    print(f"    -> max over layers = {km_max:.3e}  "
          f"({'MATCH (machine precision)' if km_match else 'NO MATCH'})")

    # ---- 4: kernel-ridge cross-check ----
    print("[4] KERNEL-RIDGE CROSS-CHECK (explicit Toeplitz K; z=K·a; per-chan scale)")
    kr = kernel_ridge_crosscheck(model, x_probe)
    print(f"    exact K·a vs scan z   max|Δ| = {kr['exact_K_dot_a_vs_scan_max_abs_err']:.3e}")
    print(f"    scaled-kernel ridge residual (rel) = {kr['ridge_scaled_kernel_residual_rel']:.3e}")
    print(f"    per-channel scale: mean {kr['per_channel_scale_mean']:.4f} "
          f"std {kr['per_channel_scale_std']:.2e}")
    out["kernel_ridge_crosscheck"] = kr

    # ---- 5: negative control at the SAME width ----
    print("[5] NEGATIVE CONTROL — SELECTIVE (time-varying γ_t) vs best single kernel")
    ctrl = selective_control_match(
        device, vocab_size, mask_idx, d_model, n_layers, n_heads, d_head,
        seq_len, x_probe, dtype,
    )
    print(f"    selective scan vs best-mean-γ kernel: max|Δ| = {ctrl['max_abs_err']:.3e}  "
          f"mean|Δ| = {ctrl['mean_abs_err']:.3e}")
    print(f"    γ_t time-std (mean over chan) = {ctrl['gamma_time_std_mean']:.3e}")
    out["selective_negative_control"] = ctrl

    ratio = ctrl["max_abs_err"] / (km_max + 1e-300)
    out["summary"] = {
        "d_model": d_model,
        "constant_gate_matches_kernel": bool(km_match),
        "kernel_match_max_abs_err": km_max,
        "kernel_match_mean_abs_err": max(r["mean_abs_err"] for r in km),
        "z_scale_mean_abs": max(r["z_scale_mean_abs"] for r in km),
        "selective_control_max_abs_err": ctrl["max_abs_err"],
        "control_over_match_ratio": ratio,
        "train_loss_first": losses[0], "train_loss_last": losses[-1],
        "wall_s_train": train_s,
    }
    print(f"    => d{d_model}: match {km_max:.3e} | control {ctrl['max_abs_err']:.3e} "
          f"| ratio {ratio:.2e}")
    return out


def main():
    device = "cpu"
    dtype = torch.float64  # machine-precision identity claim ⇒ exact arithmetic
    torch.manual_seed(0)

    print("=" * 74)
    print("Constant-Gate Kernel-Match AT FULL WIDTH (FRONT 5) — GSSM-Selective")
    print(f"torch {torch.__version__}  |  device={device}  dtype={dtype}")
    print("d512 = historical collapse width (FINAL_REPORT M1). Does the match hold?")
    print("=" * 74)

    # Width ladder.  d512 FIRST (the target).  n_heads*d_head == d_model.
    # seq_len kept at the d-small value (24) so the ONLY thing changing is width;
    # this isolates 'does width introduce drift' from 'does length introduce drift'.
    # Modest steps so each width trains in ~a minute on the M4 (it's a numerical
    # identity test, not a convergence test — the loss only has to move).
    common = dict(vocab_size=40, seq_len=24, n_seqs=64, n_layers=2,
                  lr=3e-3, batch=16, gamma_const=0.9, alpha_const=0.5)
    width_ladder = [
        {**common, "d_model": 512, "n_heads": 8, "d_head": 64, "steps": 120},
        {**common, "d_model": 256, "n_heads": 4, "d_head": 64, "steps": 150},
        {**common, "d_model": 128, "n_heads": 4, "d_head": 32, "steps": 200},
    ]

    results = {
        "front": "FRONT 5 — constant-gate kernel match at full width",
        "device": device, "dtype": str(dtype),
        "torch_version": torch.__version__,
        "d_small_baseline_max_abs_err": 8.881784197001252e-16,
        "widths_attempted": [w["d_model"] for w in width_ladder],
        "per_width": {},
        "widths_succeeded": [],
        "widths_failed": {},
    }

    overall_t0 = time.time()
    for wcfg in width_ladder:
        d = wcfg["d_model"]
        try:
            wt0 = time.time()
            out = run_one_width(wcfg, dtype, device)
            out["wall_s_total"] = time.time() - wt0
            results["per_width"][str(d)] = out
            results["widths_succeeded"].append(d)
        except Exception as e:  # noqa: BLE001  (OOM / anything → record, continue)
            tb = traceback.format_exc()
            print(f"\n!!! WIDTH d_model={d} FAILED: {type(e).__name__}: {e}")
            print(tb)
            results["widths_failed"][str(d)] = f"{type(e).__name__}: {e}"
            # keep going to the fallback width
    results["wall_s_all"] = time.time() - overall_t0

    # ---- headline: the LARGEST width that actually ran ----
    if results["widths_succeeded"]:
        largest = max(results["widths_succeeded"])
        s = results["per_width"][str(largest)]["summary"]
        results["headline"] = {
            "largest_width_run": largest,
            "kernel_match_max_abs_err_at_largest_width": s["kernel_match_max_abs_err"],
            "match_at_largest_width": s["constant_gate_matches_kernel"],
            "selective_control_max_abs_err_at_largest_width": s["selective_control_max_abs_err"],
            "control_over_match_ratio_at_largest_width": s["control_over_match_ratio"],
            "d_small_baseline_max_abs_err": 8.881784197001252e-16,
            "drift_vs_d_small": (s["kernel_match_max_abs_err"]
                                 / 8.881784197001252e-16),
        }
        print("\n" + "=" * 74)
        print("HEADLINE (FRONT 5)")
        print(f"  Largest width that RAN: d_model = {largest}")
        print(f"  kernel-match max|Δ| at d{largest} : "
              f"{s['kernel_match_max_abs_err']:.3e}  "
              f"({'MATCH (machine precision)' if s['constant_gate_matches_kernel'] else 'NO MATCH'})")
        print(f"  d_small baseline (d32)            : 8.882e-16")
        print(f"  drift factor vs d_small           : "
              f"{results['headline']['drift_vs_d_small']:.2f}x")
        print(f"  selective control gap at d{largest}  : "
              f"{s['selective_control_max_abs_err']:.3e}")
        print(f"  control/match ratio at d{largest}    : "
              f"{s['control_over_match_ratio']:.3e}")
        # cross-width table
        print("\n  Per-width:")
        for d in sorted(results["widths_succeeded"]):
            ss = results["per_width"][str(d)]["summary"]
            print(f"    d{d:<4d}: match {ss['kernel_match_max_abs_err']:.3e}  "
                  f"control {ss['selective_control_max_abs_err']:.3e}  "
                  f"ratio {ss['control_over_match_ratio']:.2e}  "
                  f"loss {ss['train_loss_first']:.2f}->{ss['train_loss_last']:.2f}")
        print("=" * 74)
    else:
        results["headline"] = {"error": "no width ran successfully"}
        print("\n!!! NO WIDTH RAN — see widths_failed")

    out_path = os.path.join(HERE, "constant_gate_kernel_match_width_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults JSON written to: {out_path}")


if __name__ == "__main__":
    main()
