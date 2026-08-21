"""
Affine-Associative Parallel Scan (substrate Lead 1) — by Opus 4.8
================================================================

Closes the honest training-throughput caveat of GSSM-Selective.

GSSM-Selective's core recurrence (moebius_scan_transformer_selective.py) is the
first-order linear recurrence

    z_t = γ_t · z_{t-1} + a_t        with z_{-1} = 0,   a_t = α_t · log(1 − v_t²)

computed today by an O(T)-depth sequential Python loop (`sequential_linear_scan`).
The readout s_t = √(1 − exp(z_t)) is elementwise and untouched here.

LEAD 1 (substrate, numerically verified by the mining agent to 5e-17):
the recurrence is EXACTLY associative under the affine operator

        (A₂, B₂) ⊗ (A₁, B₁) = (A₂·A₁,  A₂·B₁ + B₂)            identity (1, 0)

where element 1 is the EARLIER token (applied first) and element 2 the LATER one.
Each token contributes (A_t, B_t) = (γ_t, a_t).  The inclusive prefix combine of
tokens 1..t is (Γ_{1:t}, Z_{1:t}); applied to the initial state z_{-1}=0 it yields
Γ_{1:t}·0 + Z_{1:t} = z_t.  So the B-component of the inclusive affine prefix scan
IS the sequence z_t — to machine precision, no approximation.

Because ⊗ is associative, the whole sequence is computable in O(log T) sequential
DEPTH / O(T) work via a doubling (Hillis–Steele) scan.  γ being per-channel AND
input-dependent changes nothing: the carry is simply A_t = γ_t, the operator is the
same.  This is the standard "linear SSM = associative scan" identity (Mamba's
selective scan, S5, LRU), specialised to the project's exact (γ, a) parametrisation.

This file is BUILD + SMOKE only (M6 is on the GPU per the runner contract).  The
point of Lead 1 is ZERO RISK: the parallel scan must match the reference sequential
scan to machine precision.  `verify_against_sequential()` is the gate; if it does
not match, the build FAILS and says so.

MPS NOTE: everything here is real-valued (no torch.complex).  The doubling scan uses
only slicing + elementwise mul/add + concat, all of which are MPS-native and avoid
the in-place index-scatter that the Blelloch up/down-sweep needs (kept here as an
alternative, `parallel_linear_scan_blelloch`, mirroring ps_lifted_scan.py's style).

Reference: Foss 2026, "From Markov Chains to Minkowski Space".
"""

import sys
import os
import time
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ───────────────────────────────────────────────────────────────────────────
# Reference sequential scan (vendored verbatim from
# moebius_scan_transformer_selective.py so this file is self-contained and the
# correctness gate compares against the *exact* reference semantics: z_{-1}=0,
# first output = a[:, 0]).
# ───────────────────────────────────────────────────────────────────────────

def sequential_linear_scan(a: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """Sequential scan for z_t = γ_t · z_{t-1} + a_t.  Shapes (B, T, H, D)."""
    B, T, H, D = a.shape
    Z = torch.zeros(B, H, D, device=a.device, dtype=a.dtype)
    out = []
    for t in range(T):
        Z = gamma[:, t] * Z + a[:, t]
        out.append(Z)
    return torch.stack(out, dim=1)


def _next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


# ───────────────────────────────────────────────────────────────────────────
# Lead 1 — affine-associative parallel scan (doubling / Hillis–Steele).
# ───────────────────────────────────────────────────────────────────────────

def parallel_linear_scan(a: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """O(log T)-depth inclusive prefix scan for z_t = γ_t · z_{t-1} + a_t.

    Computes the SAME z_t as `sequential_linear_scan`, but via the associative
    affine operator (A₂,B₂)⊗(A₁,B₁) = (A₂·A₁, A₂·B₁ + B₂) using a doubling
    (Hillis–Steele) scan along the time axis (dim=1).

    Shapes: a, gamma are (B, T, H, D); returns z of shape (B, T, H, D).

    Carry A = γ (the per-token, per-channel, input-dependent forget gate),
    carry B = a (the additive log-complement drive).  The B-component of the
    inclusive prefix, applied to z_{-1}=0, equals z_t exactly.

    Depth: ceil(log2 T) sequential steps; work: O(T) per step → O(T log T) total
    ops, O(T) if a Blelloch up/down-sweep is used instead (see below).  T need not
    be a power of two — the doubling loop runs while the shift `d < T`, no padding.
    """
    assert a.shape == gamma.shape, (a.shape, gamma.shape)
    B, T, H, D = a.shape
    if T == 0:
        return a
    if T == 1:
        # z_0 = a_0 (γ multiplies z_{-1}=0).
        return a.clone()

    # A holds the prefix forget-product, Bc holds the prefix drive (the z's).
    A = gamma
    Bc = a

    d = 1
    while d < T:
        # Earlier operand at offset d (tokens [0:T-d]); leading d positions have
        # no earlier neighbour → combine with identity (1, 0), i.e. unchanged.
        A_prev = A[:, :T - d]      # (B, T-d, H, D)  earlier A
        B_prev = Bc[:, :T - d]     # (B, T-d, H, D)  earlier B
        A_cur = A[:, d:]           # (B, T-d, H, D)  later A
        B_cur = Bc[:, d:]          # (B, T-d, H, D)  later B

        # (A_cur, B_cur) ⊗ (A_prev, B_prev) = (A_cur·A_prev, A_cur·B_prev + B_cur)
        A_comb = A_cur * A_prev
        B_comb = A_cur * B_prev + B_cur

        # Positions [0:d] are unchanged this round (identity earlier operand).
        A = torch.cat([A[:, :d], A_comb], dim=1)
        Bc = torch.cat([Bc[:, :d], B_comb], dim=1)
        d *= 2

    return Bc


def parallel_linear_scan_blelloch(a: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """Work-efficient O(T)-work Blelloch up/down-sweep variant (alternative).

    Same result as `parallel_linear_scan`; mirrors the up/down-sweep style of
    ps_lifted_scan.py but on the affine semiring (A, B) instead of 2x2 matmul, so
    each combine is 2 muls + 1 add rather than an 8-flop matmul.  Pads time to the
    next power of two with the identity element (1, 0).

    Kept as an alternative / cross-check; the doubling scan above is the default
    because its pure-slice/concat form is the cleanest on MPS (no in-place
    index_copy_ scatter).  This routine produces an INCLUSIVE scan to match the
    reference (z_t includes token t).
    """
    assert a.shape == gamma.shape, (a.shape, gamma.shape)
    B, T, H, D = a.shape
    if T == 0:
        return a
    if T == 1:
        return a.clone()

    n = _next_power_of_two(T)
    # Identity element (A=1, B=0).
    A = torch.ones(B, n, H, D, device=a.device, dtype=a.dtype)
    Bc = torch.zeros(B, n, H, D, device=a.device, dtype=a.dtype)
    A[:, :T] = gamma
    Bc[:, :T] = a

    def combine(Al, Bl, Ar, Br):
        # right (later) ⊗ left (earlier):  (Ar·Al, Ar·Bl + Br)
        return Ar * Al, Ar * Bl + Br

    # ── up-sweep (reduce): build block totals on the right edge of each block ──
    step = 1
    while step < n:
        idx = torch.arange(2 * step - 1, n, 2 * step, device=a.device)
        left = idx - step
        Al, Bl = A[:, left], Bc[:, left]
        Ar, Br = A[:, idx], Bc[:, idx]
        Ac, Bcc = combine(Al, Bl, Ar, Br)
        A = A.index_copy(1, idx, Ac)
        Bc = Bc.index_copy(1, idx, Bcc)
        step *= 2

    # Save the inclusive total before the exclusive down-sweep clobbers the root.
    total_A = A[:, n - 1].clone()
    total_B = Bc[:, n - 1].clone()

    # ── down-sweep: standard Blelloch EXCLUSIVE scan ──
    A = A.index_copy(1, torch.tensor([n - 1], device=a.device),
                     torch.ones(B, 1, H, D, device=a.device, dtype=a.dtype))
    Bc = Bc.index_copy(1, torch.tensor([n - 1], device=a.device),
                       torch.zeros(B, 1, H, D, device=a.device, dtype=a.dtype))
    step = n // 2
    while step >= 1:
        idx = torch.arange(2 * step - 1, n, 2 * step, device=a.device)
        left = idx - step
        Al, Bl = A[:, left].clone(), Bc[:, left].clone()   # old left = left-subtree reduction (earlier segment)
        Ar, Br = A[:, idx].clone(), Bc[:, idx].clone()     # old right = exclusive prefix flowing DOWN (the context before this block)
        # Blelloch exclusive down-sweep:
        #   left child  ← incoming exclusive prefix (= old right)
        #   right child ← incoming exclusive prefix ⊗ left-subtree reduction
        #                 (earlier = incoming prefix, later = left-subtree)
        new_left_A, new_left_B = Ar, Br
        # combine(earlier=old_right(=Ar,Br), later=old_left(=Al,Bl)) = (Al·Ar, Al·Br + Bl)
        new_right_A, new_right_B = combine(Ar, Br, Al, Bl)
        A = A.index_copy(1, left, new_left_A)
        Bc = Bc.index_copy(1, left, new_left_B)
        A = A.index_copy(1, idx, new_right_A)
        Bc = Bc.index_copy(1, idx, new_right_B)
        step //= 2

    # Now (A, Bc) is the EXCLUSIVE scan (prefix BEFORE each element).  Convert to
    # INCLUSIVE by combining the exclusive prefix with each element:
    #   inclusive_t = element_t ⊗ exclusive_t
    A_excl, B_excl = A[:, :T], Bc[:, :T]
    A_el, B_el = gamma, a
    _, Bc_incl = combine(A_excl, B_excl, A_el, B_el)
    return Bc_incl


# ───────────────────────────────────────────────────────────────────────────
# Constant-γ closed form (Lead 1 optimisation): geometric causal convolution.
# z_t = γ^t z_{-1} + Σ_{k=0..t} γ^{t-k} a_k.  With z_{-1}=0 and a SCALAR γ this is
# a causal convolution of a with the geometric kernel {γ^j}_{j≥0}.
# NOTE: this is ONLY valid when γ is a single constant (or per-channel constant)
# shared across time.  The GENERAL per-token-γ recurrence needs the ⊗ scan above;
# this closed form is the constant-γ special-case optimisation.
# ───────────────────────────────────────────────────────────────────────────

def constant_gamma_closed_form(a: torch.Tensor, gamma_const) -> torch.Tensor:
    """z_t = Σ_{k≤t} γ^{t-k} a_k for time-constant γ (scalar or per-(H,D) channel).

    a: (B, T, H, D).  gamma_const: python float, or a tensor broadcastable to
    (H, D) / (1, 1, H, D) holding a per-channel CONSTANT forget rate.

    Implemented as a causal geometric convolution via cumulative rescaling:
        z_t = γ^t · Σ_{k≤t} γ^{-k} a_k
    which is an exact O(T) cumsum (numerically guarded by factoring γ^t back out
    per-t so the γ^{-k} blow-up cancels).  This matches the ⊗ scan when γ_t≡γ.
    """
    B, T, H, D = a.shape
    if not torch.is_tensor(gamma_const):
        gamma_const = torch.tensor(float(gamma_const), device=a.device, dtype=a.dtype)
    g = gamma_const.to(device=a.device, dtype=a.dtype)
    # Broadcast g to (H, D).
    g = g.expand(H, D) if g.dim() else g.expand(H, D)

    # Exponents j = 0..T-1 along time.
    j = torch.arange(T, device=a.device, dtype=a.dtype).view(1, T, 1, 1)
    g_b = g.view(1, 1, H, D)

    # Numerically stable geometric prefix sum:
    #   z_t = Σ_{k≤t} g^{t-k} a_k
    # Compute log-domain shifting to avoid g^{-k} overflow for small g:
    #   z_t = g^t · cumsum_k( g^{-k} a_k )  is unstable; instead use the identity
    #   z_t = a_t + g · z_{t-1}, but vectorised as a matmul with a lower-triangular
    #   Toeplitz geometric matrix (exact, O(T²) but closed-form & fully parallel).
    # For the modest T used in verification this is the cleanest exact closed form.
    t_idx = torch.arange(T, device=a.device)
    diff = t_idx.view(T, 1) - t_idx.view(1, T)          # (T, T): t - k
    mask = (diff >= 0)                                   # causal lower triangle
    # Kernel K[t,k] = g^{t-k} for t>=k else 0.  g is per-(H,D), so build per channel.
    # Shape (T, T, H, D).
    diff_f = diff.to(a.dtype)
    K = torch.where(
        mask.view(T, T, 1, 1),
        torch.pow(g_b.view(1, 1, H, D), diff_f.view(T, T, 1, 1)),
        torch.zeros(1, 1, 1, 1, device=a.device, dtype=a.dtype),
    )
    # z[b,t,h,d] = Σ_k K[t,k,h,d] · a[b,k,h,d]
    z = torch.einsum('tkhd,bkhd->bthd', K, a)
    return z


# ───────────────────────────────────────────────────────────────────────────
# Correctness gate — THE WHOLE POINT.
# ───────────────────────────────────────────────────────────────────────────

def verify_against_sequential(B=3, T=37, H=4, D=8, seed=0, device="cpu",
                              tol=1e-5, verbose=True):
    """Build random (a, γ), run reference sequential + parallel, assert match.

    T=37 is deliberately NOT a power of two to exercise the non-PoT guard.
    Returns (ok, max_err).  ok is False (build FAILS) if max_err >= tol.
    """
    torch.manual_seed(seed)
    dev = torch.device(device)
    a = torch.randn(B, T, H, D, device=dev)
    # γ in (0,1) like the model's sigmoid forget gate.
    gamma = torch.sigmoid(torch.randn(B, T, H, D, device=dev))

    z_seq = sequential_linear_scan(a, gamma)
    z_par = parallel_linear_scan(a, gamma)
    z_bl = parallel_linear_scan_blelloch(a, gamma)

    err_par = (z_par - z_seq).abs().max().item()
    err_bl = (z_bl - z_seq).abs().max().item()
    max_err = max(err_par, err_bl)

    ok = (err_par < tol) and (err_bl < tol)

    if verbose:
        print(f"[verify] B={B} T={T} H={H} D={D} device={device} (T power-of-2? "
              f"{T == _next_power_of_two(T)})")
        print(f"[verify] doubling  scan  max abs err vs sequential = {err_par:.3e}")
        print(f"[verify] blelloch  scan  max abs err vs sequential = {err_bl:.3e}")
        print(f"[verify] tolerance = {tol:.0e}  →  {'PASS' if ok else 'FAIL'}")
        if not ok:
            print("[verify] *** BUILD FAILS: parallel scan does NOT match sequential "
                  "to machine precision. ***")

    return ok, max_err


def verify_constant_gamma(B=2, T=24, H=3, D=5, seed=1, device="cpu",
                          tol=1e-5, verbose=True):
    """Verify the constant-γ closed form matches the sequential scan."""
    torch.manual_seed(seed)
    dev = torch.device(device)
    a = torch.randn(B, T, H, D, device=dev)

    results = []
    # (a) global scalar γ
    g_scalar = 0.9
    gamma_full = torch.full((B, T, H, D), g_scalar, device=dev)
    z_seq = sequential_linear_scan(a, gamma_full)
    z_cf = constant_gamma_closed_form(a, g_scalar)
    err_s = (z_cf - z_seq).abs().max().item()
    results.append(("scalar γ=0.9", err_s))

    # (b) per-channel constant γ ∈ (0,1)
    torch.manual_seed(seed + 100)
    g_chan = torch.sigmoid(torch.randn(H, D, device=dev))
    gamma_full2 = g_chan.view(1, 1, H, D).expand(B, T, H, D).contiguous()
    z_seq2 = sequential_linear_scan(a, gamma_full2)
    z_cf2 = constant_gamma_closed_form(a, g_chan)
    err_c = (z_cf2 - z_seq2).abs().max().item()
    results.append(("per-channel constant γ", err_c))

    max_err = max(e for _, e in results)
    ok = max_err < tol
    if verbose:
        for name, e in results:
            print(f"[verify-const] {name:24s} max abs err vs sequential = {e:.3e}")
        print(f"[verify-const] tolerance = {tol:.0e}  →  {'PASS' if ok else 'FAIL'}")
    return ok, max_err


# ───────────────────────────────────────────────────────────────────────────
# Honest timing: sequential vs parallel wall-time at T ∈ {128, 512, 1024}.
# The DEPTH win is O(log T); at these sizes on MPS the parallel kernels carry
# launch overhead and may not beat the tight sequential loop in wall-time.
# Report it straight — no overclaiming.
# ───────────────────────────────────────────────────────────────────────────

def _sync(device):
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def timing_comparison(Ts=(128, 512, 1024), B=8, H=4, D=16, device="cpu",
                      iters=10, warmup=3):
    dev = torch.device(device)
    print(f"\n[timing] device={device}  B={B} H={H} D={D}  iters={iters} "
          f"(warmup {warmup})")
    print(f"[timing] {'T':>6} | {'seq (ms)':>10} | {'par (ms)':>10} | "
          f"{'depth seq':>10} | {'depth par':>10} | speedup")
    rows = []
    for T in Ts:
        torch.manual_seed(0)
        a = torch.randn(B, T, H, D, device=dev)
        gamma = torch.sigmoid(torch.randn(B, T, H, D, device=dev))

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
        depth_par = (T - 1).bit_length()  # ceil(log2 T)
        spd = t_seq / t_par if t_par > 0 else float('nan')
        print(f"[timing] {T:>6} | {t_seq:>10.3f} | {t_par:>10.3f} | "
              f"{depth_seq:>10} | {depth_par:>10} | {spd:>6.2f}x")
        rows.append((T, t_seq, t_par, depth_seq, depth_par))
    return rows


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def _pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    ap = argparse.ArgumentParser(
        description="Affine-associative parallel scan (substrate Lead 1)."
    )
    ap.add_argument("--smoke", action="store_true",
                    help="Run correctness gate + tiny timing.")
    ap.add_argument("--device", default=None,
                    help="cpu | mps | cuda (default: auto).")
    ap.add_argument("--full", action="store_true",
                    help="Larger timing sweep (build-host only).")
    args = ap.parse_args()

    device = args.device or _pick_device()
    print("=" * 74)
    print("Affine-Associative Parallel Scan (substrate Lead 1) — by Opus 4.8")
    print(f"device = {device}")
    print("=" * 74)

    # Correctness gate is cheap and runs on CPU regardless (machine-precision
    # check is the whole point; keep it deterministic / device-independent).
    ok_main, err_main = verify_against_sequential(device="cpu")
    print()
    ok_const, err_const = verify_constant_gamma(device="cpu")

    all_ok = ok_main and ok_const

    if args.smoke or args.full:
        Ts = (128, 512, 1024) if not args.full else (128, 512, 1024, 2048, 4096)
        timing_comparison(Ts=Ts, device=device)

    print()
    print("=" * 74)
    if all_ok:
        print(f"BUILD OK — parallel scan matches sequential to {err_main:.1e} "
              f"(< 1e-5), constant-γ closed form to {err_const:.1e}.")
    else:
        print("BUILD FAILED — parallel scan does NOT match the reference "
              "sequential scan. See [verify] output above.")
        sys.exit(1)
    print("=" * 74)


if __name__ == "__main__":
    main()
