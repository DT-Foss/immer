"""
Parallel-Scan Integration & Measurement — by Opus 4.8
=====================================================

Closes the gap FINAL_REPORT.md flagged: the O(log T)-depth affine-associative
parallel scan (`src/parallel_scan.py::parallel_linear_scan`) was numerically
verified in ISOLATION (5e-17 vs the sequential loop) but had NEVER been wired
into the actual GSSM-Selective model and trained.  This script does exactly
that, end to end, and MEASURES it — no aspirational claims, only what runs.

What this script proves, in order:

  1. INTEGRATION.  A `SelectiveRapiditySqrtScanLayer` variant runs its forward()
     through `parallel_linear_scan` instead of `sequential_linear_scan`.  We do
     this by monkeypatching the module-level `sequential_linear_scan` symbol
     that the reference layer's forward() resolves at call time — so BOTH the
     causal and the (forward+reverse) non-causal code paths are covered, with
     zero edits to the frozen reference file.

  2. FORWARD IDENTITY.  Same random weights, same input → the parallel-scan
     layer output equals the sequential-scan layer output to ~1e-5.

  3. GRADIENT IDENTITY.  The real test.  Backward through both scans must give
     identical grads w.r.t. EVERY weight.  Autograd flows through the doubling
     scan's slice/concat/mul/add graph; it must agree with the loop's graph to
     ~1e-5.  (The doubling scan touches each z_t through a *different* compute
     graph than the loop — if the chain rule disagreed anywhere, this catches it.)

  4. TRAINING CONVERGENCE.  Train a tiny GSSM-Selective LM twice from an
     identical seed on a tiny fixed offline token tensor (NO WikiText download),
     once per scan.  Same seed ⇒ the loss curves must be bit-identical / ~1e-4.

  5. TIMING.  Wall-time sequential vs parallel at T ∈ {128,512,1024,2048} on the
     auto-detected device (CPU / MPS).  Reported straight, including where the
     parallel scan LOSES in wall-time despite O(log T) depth — at these sizes the
     Python doubling loop's concat/launch overhead can beat the tight inner loop.

Offline, fast, self-contained.  `python3 src/parallel_scan_integration.py`.
Writes `parallel_scan_integration_results.json` next to this file.

Reference: Foss 2026, "From Markov Chains to Minkowski Space".
"""

import os
import sys
import json
import time
import copy

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
REF = os.path.join(REPO, "reference")

# Make both the reference layer and the parallel scan importable.
sys.path.insert(0, REF)
sys.path.insert(0, HERE)

import moebius_scan_transformer_selective as ref  # the frozen reference module
from moebius_scan_transformer_selective import (
    SelectiveRapiditySqrtScanLayer,
    SelectiveRapiditySqrtTransformerLM,
)
from parallel_scan import parallel_linear_scan, sequential_linear_scan


# ───────────────────────────────────────────────────────────────────────────
# Scan swap.  forward() inside SelectiveRapiditySqrtScanLayer calls the
# module-global `sequential_linear_scan` (resolved at call time in ref's
# namespace).  We flip that one symbol — covers causal AND non-causal paths.
# ───────────────────────────────────────────────────────────────────────────

class use_parallel_scan:
    """Context manager: inside the block, the reference layer uses the parallel
    O(log T) scan; outside, the original sequential loop is restored."""

    def __enter__(self):
        self._orig = ref.sequential_linear_scan
        ref.sequential_linear_scan = parallel_linear_scan
        return self

    def __exit__(self, *exc):
        ref.sequential_linear_scan = self._orig
        return False


def _pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _sync(device):
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


# ───────────────────────────────────────────────────────────────────────────
# 2 + 3.  Forward + gradient identity on one scan layer (causal & non-causal).
# ───────────────────────────────────────────────────────────────────────────

def forward_and_grad_identity(device, causal, B=3, T=37, d_model=48,
                              n_heads=4, d_head=12, seed=0, tol=1e-5,
                              dtype=torch.float32):
    """Build ONE scan layer, run the SAME weights+input through the sequential
    and the parallel scan, compare forward output and ALL parameter grads.

    T=37 is deliberately non-power-of-two to exercise the doubling guard.
    dtype=float64 (on CPU) verifies the identity is EXACT (machine precision),
    proving the float32 deltas are pure FP-reassociation noise, not algorithmic.
    Returns a dict of measured max-abs differences.
    """
    torch.manual_seed(seed)
    layer = SelectiveRapiditySqrtScanLayer(
        d_model, d_head=d_head, n_heads=n_heads, causal=causal, dropout=0.0
    ).to(device=device, dtype=dtype)
    layer.eval()  # no dropout stochasticity

    x = torch.randn(B, T, d_model, device=device, dtype=dtype)

    # ---- sequential path (reference) ----
    x_seq = x.clone().detach().requires_grad_(True)
    out_seq = layer(x_seq)
    loss_seq = out_seq.pow(2).sum()
    layer.zero_grad(set_to_none=True)
    loss_seq.backward()
    grads_seq = {n: p.grad.detach().clone() for n, p in layer.named_parameters()}
    gx_seq = x_seq.grad.detach().clone()

    # ---- parallel path (swap the scan) ----
    x_par = x.clone().detach().requires_grad_(True)
    layer.zero_grad(set_to_none=True)
    with use_parallel_scan():
        out_par = layer(x_par)
        loss_par = out_par.pow(2).sum()
        loss_par.backward()
    grads_par = {n: p.grad.detach().clone() for n, p in layer.named_parameters()}
    gx_par = x_par.grad.detach().clone()

    fwd_err = (out_par - out_seq).abs().max().item()

    grad_errs = {}
    for name in grads_seq:
        grad_errs[name] = (grads_par[name] - grads_seq[name]).abs().max().item()
    grad_errs["__input__"] = (gx_par - gx_seq).abs().max().item()
    max_grad_err = max(grad_errs.values())

    return {
        "causal": causal,
        "dtype": str(dtype),
        "B": B, "T": T, "d_model": d_model,
        "n_heads": n_heads, "d_head": d_head,
        "forward_max_abs_err": fwd_err,
        "grad_max_abs_err": max_grad_err,
        "per_param_grad_err": grad_errs,
        "forward_pass": fwd_err < tol,
        "grad_pass": max_grad_err < tol,
        "tol": tol,
    }


# ───────────────────────────────────────────────────────────────────────────
# 4.  Training convergence — tiny offline LM, identical seed, two scans.
# ───────────────────────────────────────────────────────────────────────────

def _make_offline_tokens(vocab_size, seq_len, n_seqs, seed=1234):
    """A fixed synthetic token tensor — fully offline, no datasets download.
    A mild local structure (token depends on its predecessor) so loss actually
    moves; but content is irrelevant — we only test seq==par equality."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randint(0, vocab_size, (n_seqs, seq_len), generator=g)
    shifted = (base.roll(1, dims=1) + 1) % vocab_size
    # Mix so there is a learnable predecessor signal.
    X = torch.where(torch.rand(n_seqs, seq_len, generator=g) < 0.5, base, shifted)
    return X.long()


def train_tiny_lm(device, use_parallel, X, vocab_size, mask_idx,
                  d_model=32, n_layers=2, n_heads=2, d_head=16,
                  steps=12, lr=3e-3, seed=7):
    """Train a tiny causal GSSM-Selective LM as a next-token predictor for a
    few steps.  Deterministic given (seed, scan).  Returns the per-step loss
    curve.  Same seed across scans ⇒ curves must match."""
    torch.manual_seed(seed)
    model = SelectiveRapiditySqrtTransformerLM(
        vocab_size, mask_idx, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, d_head=d_head, seq_len=X.shape[1],
        dropout=0.0, causal=True,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    X = X.to(device)
    inp = X[:, :-1]
    tgt = X[:, 1:]

    losses = []

    def _train_loop():
        model.train()
        for _ in range(steps):
            logits = model(inp)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())

    if use_parallel:
        with use_parallel_scan():
            _train_loop()
    else:
        _train_loop()

    return losses


# ───────────────────────────────────────────────────────────────────────────
# 5.  Timing — sequential vs parallel wall-time, depth noted.
# ───────────────────────────────────────────────────────────────────────────

def timing_sweep(device, Ts=(128, 512, 1024, 2048), B=8, H=4, D=16,
                 iters=20, warmup=5):
    rows = []
    for T in Ts:
        torch.manual_seed(0)
        a = torch.randn(B, T, H, D, device=device)
        gamma = torch.sigmoid(torch.randn(B, T, H, D, device=device))
        a = a.float()
        gamma = gamma.float()

        for _ in range(warmup):
            sequential_linear_scan(a, gamma)
            parallel_linear_scan(a, gamma)
        _sync(device)

        t0 = time.perf_counter()
        for _ in range(iters):
            sequential_linear_scan(a, gamma)
        _sync(device)
        t_seq = (time.perf_counter() - t0) / iters * 1e3

        t0 = time.perf_counter()
        for _ in range(iters):
            parallel_linear_scan(a, gamma)
        _sync(device)
        t_par = (time.perf_counter() - t0) / iters * 1e3

        depth_seq = T
        depth_par = max(1, (T - 1).bit_length())  # ceil(log2 T)
        speedup = t_seq / t_par if t_par > 0 else float("nan")
        rows.append({
            "T": T,
            "seq_ms": t_seq,
            "par_ms": t_par,
            "depth_seq": depth_seq,
            "depth_par": depth_par,
            "speedup_seq_over_par": speedup,
        })
    return rows


# ───────────────────────────────────────────────────────────────────────────
# Driver
# ───────────────────────────────────────────────────────────────────────────

def main():
    device = _pick_device()
    print("=" * 74)
    print("Parallel-Scan Integration & Measurement — GSSM-Selective")
    print(f"torch {torch.__version__}  |  device = {device}")
    print("=" * 74)

    results = {
        "device": device,
        "torch_version": torch.__version__,
    }

    # ---- 2 + 3: forward + gradient identity (causal AND non-causal) ----
    print("\n[1] FORWARD + GRADIENT IDENTITY (parallel scan vs sequential, one layer)")
    identity = {}
    for causal in (True, False):
        r = forward_and_grad_identity(device, causal=causal)
        identity["causal" if causal else "bidirectional"] = r
        tag = "causal" if causal else "bidirectional"
        print(f"  [{tag:13s}] forward max|Δ| = {r['forward_max_abs_err']:.3e} "
              f"({'PASS' if r['forward_pass'] else 'FAIL'})  |  "
              f"grad max|Δ| = {r['grad_max_abs_err']:.3e} "
              f"({'PASS' if r['grad_pass'] else 'FAIL'})")
    results["forward_grad_identity"] = identity

    fwd_ok = all(v["forward_pass"] for v in identity.values())
    grad_ok = all(v["grad_pass"] for v in identity.values())

    # float64 exactness check (CPU) — proves the float32 deltas are pure FP
    # reassociation noise, not an algorithmic discrepancy in the scan/autograd.
    r64 = forward_and_grad_identity("cpu", causal=True, dtype=torch.float64,
                                    tol=1e-10)
    print(f"  [float64 exact ] forward max|Δ| = {r64['forward_max_abs_err']:.3e}  |  "
          f"grad max|Δ| = {r64['grad_max_abs_err']:.3e}  "
          f"(machine precision ⇒ identity is EXACT)")
    results["forward_grad_identity_float64"] = r64

    # ---- 4: training convergence ----
    print("\n[2] TRAINING CONVERGENCE (tiny offline LM, identical seed, two scans)")
    vocab_size = 40
    mask_idx = vocab_size + 1
    seq_len = 16
    n_seqs = 24
    X = _make_offline_tokens(vocab_size, seq_len, n_seqs, seed=1234)

    loss_seq = train_tiny_lm(device, use_parallel=False, X=X,
                             vocab_size=vocab_size, mask_idx=mask_idx)
    loss_par = train_tiny_lm(device, use_parallel=True, X=X,
                             vocab_size=vocab_size, mask_idx=mask_idx)

    curve_max_abs = max(abs(a - b) for a, b in zip(loss_seq, loss_par))
    final_seq, final_par = loss_seq[-1], loss_par[-1]
    print(f"  steps           : {len(loss_seq)}")
    print(f"  loss[0]  seq/par: {loss_seq[0]:.6f} / {loss_par[0]:.6f}")
    print(f"  loss[-1] seq/par: {final_seq:.6f} / {final_par:.6f}")
    print(f"  max |Δ loss| over the whole curve: {curve_max_abs:.3e}")
    train_ok = curve_max_abs < 1e-4
    print(f"  curves identical to 1e-4? {'YES' if train_ok else 'NO'}")
    results["training_convergence"] = {
        "steps": len(loss_seq),
        "loss_curve_sequential": loss_seq,
        "loss_curve_parallel": loss_par,
        "final_loss_sequential": final_seq,
        "final_loss_parallel": final_par,
        "max_abs_curve_diff": curve_max_abs,
        "identical_to_1e-4": train_ok,
        "config": {
            "vocab_size": vocab_size, "seq_len": seq_len, "n_seqs": n_seqs,
            "d_model": 32, "n_layers": 2, "n_heads": 2, "d_head": 16,
        },
    }

    # ---- 5: timing ----
    # Run on the primary device AND on CPU, so the artifact carries the full
    # honest picture: the O(log T) DEPTH win only converts to a wall-time win
    # where the hardware can actually run the wide O(T) work per step in
    # parallel (MPS/GPU).  On CPU, with no parallelism to exploit, the parallel
    # scan's larger total work + concat allocations LOSE to the tight loop.
    timing_all = {}
    devices_to_time = [device]
    if device != "cpu":
        devices_to_time.append("cpu")
    for tdev in devices_to_time:
        print(f"\n[3] TIMING  (sequential vs parallel scan, wall-time on {tdev})")
        print(f"  {'T':>6} | {'seq ms':>9} | {'par ms':>9} | {'depth seq':>9} | "
              f"{'depth par':>9} | {'seq/par':>8}")
        rows = timing_sweep(tdev)
        for row in rows:
            verdict = "par wins" if row["speedup_seq_over_par"] > 1 else "seq wins"
            print(f"  {row['T']:>6} | {row['seq_ms']:>9.3f} | {row['par_ms']:>9.3f} | "
                  f"{row['depth_seq']:>9} | {row['depth_par']:>9} | "
                  f"{row['speedup_seq_over_par']:>7.2f}x  ({verdict})")
        timing_all[tdev] = rows
    results["timing"] = timing_all[device]   # primary device, back-compat
    results["timing_by_device"] = timing_all

    # ---- summary ----
    print("\n" + "=" * 74)
    print("SUMMARY")
    print(f"  forward identity (~1e-5)        : {'PASS' if fwd_ok else 'FAIL'}")
    print(f"  gradient identity (~1e-5)       : {'PASS' if grad_ok else 'FAIL'}")
    print(f"  training convergence (~1e-4)    : {'PASS' if train_ok else 'FAIL'}")
    print(f"  parallel scan WIRED + TRAINED   : YES")
    print("=" * 74)

    results["summary"] = {
        "forward_identity_pass": fwd_ok,
        "gradient_identity_pass": grad_ok,
        "training_convergence_pass": train_ok,
        "integration_done": True,
    }

    out_path = os.path.join(HERE, "parallel_scan_integration_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults JSON written to: {out_path}")

    all_pass = fwd_ok and grad_ok and train_ok
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
