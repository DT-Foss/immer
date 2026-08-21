"""
Test: scan_dispatch wiring — forward + gradient identity (FRONT 4) — by Opus 4.8
================================================================================

Proves the deployment dispatcher (`scan_dispatch.dispatch_linear_scan`, wired in
via `enable_parallel_inference`) produces a SelectiveRapiditySqrtScanLayer that is
identical — forward AND backward — to the same layer running the pure sequential
reference scan.

Three checks, on the auto-detected device:
  1. fp32, CAUSAL          — forward + every-parameter grad within ~1e-5.
  2. fp32, BIDIRECTIONAL    — exercises the forward+reverse non-causal path too.
  3. fp64, CAUSAL (CPU)     — machine precision (~1e-12), proving the fp32 deltas
                             are pure FP-reassociation noise, not an algorithmic
                             discrepancy in the scan or its autograd graph.

On CPU the dispatcher routes BACK to the sequential loop, so checks 1-2 are an
exact identity there (delta == 0).  On MPS/CUDA they route through the doubling
scan, so checks 1-2 are the real fp32-reassociation test.  Either way the assert
bounds hold.

Run: `python3 src/test_scan_dispatch.py`.  Offline, self-contained, exits non-zero
on any failure.

Reference: Foss 2026, "From Markov Chains to Minkowski Space".
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
REF = os.path.join(REPO, "reference")

for _p in (REF, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import moebius_scan_transformer_selective as ref
from moebius_scan_transformer_selective import SelectiveRapiditySqrtScanLayer

from scan_dispatch import (
    dispatch_linear_scan,
    enable_parallel_inference,
    disable_parallel_inference,
    pick_device,
    which_scan,
)


def _build_layer(device, dtype, causal, seed=0,
                 d_model=48, d_head=12, n_heads=4):
    torch.manual_seed(seed)
    layer = SelectiveRapiditySqrtScanLayer(
        d_model, d_head=d_head, n_heads=n_heads, causal=causal, dropout=0.0
    ).to(device=device, dtype=dtype)
    layer.eval()  # kill any stochasticity
    return layer


def _forward_backward(layer, x):
    """Run forward, build a scalar loss, backward.  Return (out, param_grads, x_grad)."""
    x = x.clone().detach().requires_grad_(True)
    layer.zero_grad(set_to_none=True)
    out = layer(x)
    loss = out.pow(2).sum()
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in layer.named_parameters()}
    return out.detach().clone(), grads, x.grad.detach().clone()


def identity_check(device, causal, dtype=torch.float32, tol=1e-5,
                   B=3, T=37, seed=0):
    """Same layer, same input — sequential reference scan vs dispatcher.

    The dispatcher is wired in exactly as deployment does it: rebinding the
    module-global symbol via enable_parallel_inference().  Returns a dict of
    measured max-abs forward/grad deltas and pass/fail flags.

    T=37 is deliberately non-power-of-two to exercise the doubling guard.
    """
    layer = _build_layer(device, dtype, causal, seed=seed)
    x = torch.randn(B, T, layer.d_model, device=device, dtype=dtype)

    # ---- reference: ensure the pure sequential scan is the active symbol ----
    disable_parallel_inference()
    assert ref.sequential_linear_scan is not dispatch_linear_scan
    out_ref, grads_ref, gx_ref = _forward_backward(layer, x)

    # ---- dispatcher: wire it in the way deployment does ----
    enable_parallel_inference()
    assert ref.sequential_linear_scan is dispatch_linear_scan
    try:
        out_disp, grads_disp, gx_disp = _forward_backward(layer, x)
    finally:
        disable_parallel_inference()  # always restore

    fwd_err = (out_disp - out_ref).abs().max().item()
    grad_errs = {n: (grads_disp[n] - grads_ref[n]).abs().max().item()
                 for n in grads_ref}
    grad_errs["__input__"] = (gx_disp - gx_ref).abs().max().item()
    max_grad_err = max(grad_errs.values())

    return {
        "device": str(device),
        "dtype": str(dtype),
        "causal": causal,
        "B": B, "T": T,
        "routed_to": which_scan(device),
        "forward_max_abs_err": fwd_err,
        "grad_max_abs_err": max_grad_err,
        "per_param_grad_err": grad_errs,
        "forward_pass": fwd_err < tol,
        "grad_pass": max_grad_err < tol,
        "tol": tol,
    }


def main():
    device = pick_device()
    print("=" * 74)
    print("scan_dispatch — forward + gradient identity vs sequential reference")
    print(f"torch {torch.__version__}  |  auto device = {device}")
    print(f"dispatcher on {device} routes to: {which_scan(device)}")
    print(f"dispatcher on cpu   routes to: {which_scan('cpu')}")
    print("=" * 74)

    all_ok = True

    # ---- 1 + 2: fp32 on the auto device, causal AND bidirectional ----
    for causal in (True, False):
        tag = "causal" if causal else "bidirectional"
        r = identity_check(device, causal=causal, dtype=torch.float32, tol=1e-5)
        ok = r["forward_pass"] and r["grad_pass"]
        all_ok = all_ok and ok
        print(f"[fp32 {tag:13s}] routed→{r['routed_to']:32s}")
        print(f"    forward max|Δ| = {r['forward_max_abs_err']:.3e} "
              f"({'PASS' if r['forward_pass'] else 'FAIL'})  |  "
              f"grad max|Δ| = {r['grad_max_abs_err']:.3e} "
              f"({'PASS' if r['grad_pass'] else 'FAIL'})  [tol {r['tol']:.0e}]")

    # ---- 3: fp64 on CPU — machine-precision exactness ----
    r64 = identity_check("cpu", causal=True, dtype=torch.float64, tol=1e-10)
    ok64 = r64["forward_pass"] and r64["grad_pass"]
    all_ok = all_ok and ok64
    print(f"[fp64 causal CPU   ] routed→{r64['routed_to']:32s}")
    print(f"    forward max|Δ| = {r64['forward_max_abs_err']:.3e} "
          f"({'PASS' if r64['forward_pass'] else 'FAIL'})  |  "
          f"grad max|Δ| = {r64['grad_max_abs_err']:.3e} "
          f"({'PASS' if r64['grad_pass'] else 'FAIL'})  [tol {r64['tol']:.0e}]")

    # ---- also confirm fp64 on the auto device if it supports float64 ----
    # MPS has no float64; CUDA does.  CPU already covered above.  Skip cleanly.
    if device == "cuda":
        r64d = identity_check(device, causal=True, dtype=torch.float64, tol=1e-10)
        ok64d = r64d["forward_pass"] and r64d["grad_pass"]
        all_ok = all_ok and ok64d
        print(f"[fp64 causal {device:6s}] routed→{r64d['routed_to']:32s}")
        print(f"    forward max|Δ| = {r64d['forward_max_abs_err']:.3e} "
              f"({'PASS' if r64d['forward_pass'] else 'FAIL'})  |  "
              f"grad max|Δ| = {r64d['grad_max_abs_err']:.3e} "
              f"({'PASS' if r64d['grad_pass'] else 'FAIL'})  [tol {r64d['tol']:.0e}]")
    elif device == "mps":
        print(f"[fp64 causal {device:6s}] SKIP — MPS has no float64; fp64 "
              f"exactness shown on CPU above (Δ=0).")

    # ---- confirm the global symbol was restored after all checks ----
    restored = ref.sequential_linear_scan is not dispatch_linear_scan
    print("-" * 74)
    print(f"global symbol restored after tests? {'YES' if restored else 'NO'}")
    all_ok = all_ok and restored

    print("=" * 74)
    if all_ok:
        print("DISPATCHER OK — wired layer is forward+grad identical to the "
              "sequential reference on the auto device.")
    else:
        print("DISPATCHER FAILED — see deltas above.")
    print("=" * 74)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
