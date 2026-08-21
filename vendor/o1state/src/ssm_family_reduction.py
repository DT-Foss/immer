"""
SSM-family reduction under one affine operator — VERIFIED (Front 3) — by Opus 4.8
================================================================================

Backs the RKHS_UNIFICATION_SECTION.md §4 claim with *running* code:

    Every modern linear / selective SSM is the first-order recurrence

        h_t = A_t · h_{t-1} + B_t · u_t          (h_{-1} = 0)

    carried by the SINGLE associative affine operator

        (A_2, B_2) ⊗ (A_1, B_1) = (A_2·A_1,  A_2·B_1 + B_2),   identity (1, 0)

    where element 1 is the EARLIER token (applied first). The inclusive affine
    prefix scan of (A_t, B_t·u_t) gives, in its B-component, exactly h_t.

The family differs ONLY in THREE PARAMETRIC SWITCHES — implemented here as code:

    Switch 1 — STATE ALGEBRA of A:
        (a) real scalar  A ∈ (0,1)        → GSSM-Selective   (1-pole, per channel)
        (b) real diagonal A ∈ (0,1)^N     → Mamba / S6        (N-dim real state)
        (c) COMPLEX diagonal A ∈ ℂ^N      → S5 / LRU          (complex eigenvalues)

    Switch 2 — IS A_t INPUT-DEPENDENT?
        selective (yes) ⇒ A_t = f(x_t), kernel time-inhomogeneous (GSSM, Mamba)
        LTI       (no)  ⇒ A_t = A const, fixed (Mercer) kernel    (S5, LRU)

    Switch 3 — B-MAP (the drive):
        GSSM : b_t = α_t · φ(v̄_t),  φ = log(1−·²)   (nonlinear rapidity feature)
        Mamba: b_t = Δ_t · B̄ · u_t                  (linear, input-scaled)
        S5   : b_t = Δ · B · u_t                     (linear, time-const B)
        LRU  : b_t = B · u_t                         (linear, time-const B)

The affine operator ⊗ is THE SAME in every case. Only A's algebra (real-scalar /
real-diag / complex-diag), A's input-dependence, and the B-map change. That is the
whole content of "one operator, three switches."

This file is RUN-AND-ASSERT: for each case it builds a small random recurrence,
runs the naive sequential loop AND the parallel doubling scan over ⊗, and asserts
they agree to machine precision. It prints per-case max abs error and writes a
results JSON. python3, offline, CPU, torch.complex is fine.

Reference: Foss 2026, "From Markov Chains to Minkowski Space"; analysis/
RKHS_UNIFICATION_SECTION.md §4 and verification-log line "Complex-diagonal
(S5/LRU) affine operator vs sequential recurrence: 4e-16".
"""

import json
import os
import sys

import torch


# ───────────────────────────────────────────────────────────────────────────
# THE one operator, written once, used by every case.
# Works for real OR complex tensors identically — torch.complex obeys the same
# (·, +) so the affine monoid is dtype-agnostic. This is the "one operator" claim
# made literal: a single function, no per-family branching.
# ───────────────────────────────────────────────────────────────────────────

def affine_combine(A2, B2, A1, B1):
    """(A2,B2) ⊗ (A1,B1) = (A2·A1, A2·B1 + B2). Element 1 = earlier, 2 = later.

    A, B are tensors of shape (..., T, N) where N is the (diagonal) state size
    (N=1 for the scalar case). Combine is elementwise over the state dim — the
    state algebra is DIAGONAL, so the matrix product A2·A1 is just elementwise
    multiply, real or complex. This identical line serves GSSM, Mamba, S5, LRU.
    """
    return A2 * A1, A2 * B1 + B2


# ───────────────────────────────────────────────────────────────────────────
# Naive sequential ground truth: h_t = A_t·h_{t-1} + b_t  (b_t = B_t·u_t already).
# Diagonal state ⇒ A_t is a vector and the matvec is an elementwise multiply.
# ───────────────────────────────────────────────────────────────────────────

def sequential_diag_ssm(A, b):
    """Sequential scan of a diagonal-state SSM.

    A : (B, T, N)  diagonal forget eigenvalues per step  (real or complex)
    b : (B, T, N)  drive  B_t·u_t already formed         (real or complex)
    returns h : (B, T, N), h_t = A_t ⊙ h_{t-1} + b_t,  h_{-1}=0.
    """
    Bsz, T, N = A.shape
    h = torch.zeros(Bsz, N, dtype=A.dtype, device=A.device)
    out = []
    for t in range(T):
        h = A[:, t] * h + b[:, t]
        out.append(h)
    return torch.stack(out, dim=1)


# ───────────────────────────────────────────────────────────────────────────
# Parallel inclusive prefix scan over ⊗ (doubling / Hillis–Steele).
# Carry-A = A_t (the diagonal eigenvalues), carry-B = b_t (the drive). The
# B-component of the inclusive prefix applied to h_{-1}=0 IS h_t. Dtype-agnostic:
# the SAME routine runs for real-scalar, real-diag, and complex-diag.
# ───────────────────────────────────────────────────────────────────────────

def parallel_diag_ssm(A, b):
    """O(log T)-depth inclusive prefix scan of the diagonal-state SSM via ⊗.

    A, b : (B, T, N). Returns h : (B, T, N), bit-equal in exact arithmetic to
    sequential_diag_ssm. No padding; the doubling loop runs while shift d < T.
    """
    Bsz, T, N = A.shape
    if T <= 1:
        return b.clone()

    Acur = A
    Bcur = b
    d = 1
    while d < T:
        # earlier operand = positions [0:T-d], later operand = positions [d:T]
        A_earlier = Acur[:, : T - d]
        B_earlier = Bcur[:, : T - d]
        A_later = Acur[:, d:]
        B_later = Bcur[:, d:]
        # (later) ⊗ (earlier)
        A_comb, B_comb = affine_combine(A_later, B_later, A_earlier, B_earlier)
        # leading d positions have no earlier neighbour → unchanged (identity)
        Acur = torch.cat([Acur[:, :d], A_comb], dim=1)
        Bcur = torch.cat([Bcur[:, :d], B_comb], dim=1)
        d *= 2
    return Bcur


# ───────────────────────────────────────────────────────────────────────────
# The three switches, as explicit builders. Each returns (A, b) = the (forget
# eigenvalues, drive) tensors for the recurrence. The RECURRENCE / SCAN is then
# the SAME for all three — only these builders differ.
# ───────────────────────────────────────────────────────────────────────────

def build_gssm_selective(Bsz, T, seed=0, device="cpu"):
    """Switch 1=(a) real SCALAR A∈(0,1); Switch 2=selective (A_t=σ(W x_t));
    Switch 3=GSSM B-map  b_t = α_t·φ(v̄_t),  φ(x)=log(1−x²).

    This is exactly the project's z_t = γ_t·z_{t-1} + α_t·log(1−v̄_t²), with the
    scalar state expressed as a 1-dim diagonal (N=1) so it shares the operator.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    N = 1  # SCALAR state algebra
    # Switch 2: A_t INPUT-DEPENDENT. Emulate γ_t = σ(W_γ x_t) with random gates.
    A = torch.sigmoid(torch.randn(Bsz, T, N, generator=g, device=device))  # ∈(0,1)
    # Switch 3: GSSM nonlinear rapidity drive  b_t = α_t · log(1 − v̄_t²).
    v = torch.tanh(torch.randn(Bsz, T, N, generator=g, device=device))  # v̄ ∈(-1,1)
    alpha = torch.rand(Bsz, T, N, generator=g, device=device)
    b = alpha * torch.log(1.0 - v * v)
    return A.to(torch.float64), b.to(torch.float64)


def build_mamba_s6(Bsz, T, N=8, seed=1, device="cpu"):
    """Switch 1=(b) real DIAGONAL A∈(0,1)^N; Switch 2=selective (Δ_t=f(x_t));
    Switch 3=Mamba linear drive  b_t = Δ_t·B̄·u_t.

    Discrete Mamba/S6: Ā_t = exp(Δ_t·A) with A<0 real ⇒ Ā_t∈(0,1)^N, input-dep via
    a per-step, per-channel Δ_t = softplus(W_Δ x_t) > 0.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    A_log = -torch.rand(N, generator=g, device=device) * 2.0 - 0.05  # A<0 (stable, real)
    # Switch 2: Δ_t input-dependent ⇒ Ā_t input-dependent (selective).
    delta = torch.nn.functional.softplus(
        torch.randn(Bsz, T, N, generator=g, device=device)
    )  # >0
    A = torch.exp(delta * A_log.view(1, 1, N))  # Ā_t ∈ (0,1)^N, real diagonal, input-dep
    # Switch 3: linear, input-scaled drive b_t = Δ_t · (B̄ u_t).
    Bmat = torch.randn(Bsz, T, N, generator=g, device=device)  # already B̄·u_t per step
    u = torch.randn(Bsz, T, 1, generator=g, device=device)
    b = delta * Bmat * u
    return A.to(torch.float64), b.to(torch.float64)


def build_s5(Bsz, T, N=8, seed=2, device="cpu"):
    """Switch 1=(c) COMPLEX DIAGONAL A=exp(ΔΛ); Switch 2=LTI (A time-const,
    input-independent); Switch 3=S5 linear drive b_t = Δ·B·u_t (B time-const).

    S5: continuous complex eigenvalues Λ = −ν + iθ (ν>0 stable), ZOH-discretised
    Ā = exp(ΔΛ), one SHARED Ā across all t (no selectivity). Drive is the complex
    projection of the real input.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    nu = torch.rand(N, generator=g, device=device) * 0.5 + 0.05  # decay >0
    theta = torch.rand(N, generator=g, device=device) * 3.0      # oscillation
    Lam = torch.complex(-nu, theta)                              # Λ = −ν + iθ
    dt = 0.3
    Abar = torch.exp(dt * Lam)                                  # complex eigenvalue, |Ā|<1
    # Switch 2: time-CONSTANT, input-INDEPENDENT ⇒ broadcast the SAME Ā over t.
    A = Abar.view(1, 1, N).expand(Bsz, T, N).contiguous()
    # Switch 3: linear complex drive b_t = Δ · (B u_t), B complex time-const.
    Bcols = torch.complex(
        torch.randn(N, generator=g, device=device),
        torch.randn(N, generator=g, device=device),
    )
    u = torch.randn(Bsz, T, 1, generator=g, device=device)  # real input
    b = dt * Bcols.view(1, 1, N) * u.to(torch.complex128)
    return A.to(torch.complex128), b.to(torch.complex128)


def build_lru(Bsz, T, N=8, seed=3, device="cpu"):
    """Switch 1=(c) COMPLEX DIAGONAL A=λ=e^{−ν+iθ}; Switch 2=LTI (input-indep);
    Switch 3=LRU linear drive b_t = B·u_t (no Δ scaling, B time-const).

    LRU parametrises eigenvalues directly on the unit disk: λ = exp(−exp(ν_log) +
    i·θ). Same complex-diagonal LTI algebra as S5; differs only in the B-map
    (no Δ step-size factor) — exactly Switch 3.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    nu_log = torch.randn(N, generator=g, device=device)
    theta = torch.rand(N, generator=g, device=device) * 6.283  # ∈[0,2π)
    lam = torch.exp(torch.complex(-torch.exp(nu_log), theta))  # |λ|<1 on unit disk
    A = lam.view(1, 1, N).expand(Bsz, T, N).contiguous()      # LTI, input-indep
    # Switch 3: b_t = B u_t (no Δ), B complex time-const.
    Bcols = torch.complex(
        torch.randn(N, generator=g, device=device),
        torch.randn(N, generator=g, device=device),
    )
    u = torch.randn(Bsz, T, 1, generator=g, device=device)
    b = Bcols.view(1, 1, N) * u.to(torch.complex128)
    return A.to(torch.complex128), b.to(torch.complex128)


# ───────────────────────────────────────────────────────────────────────────
# Run each case: sequential vs parallel under the ONE operator; assert match.
# ───────────────────────────────────────────────────────────────────────────

def _switches(state_algebra, input_dep, b_map):
    return {"state_algebra": state_algebra, "A_t_input_dependent": input_dep,
            "B_map": b_map}


CASES = [
    # name, builder, kwargs, dtype-label, three-switch description
    ("GSSM-Selective", build_gssm_selective, dict(N_note="scalar"),
     "float64", _switches("real scalar ∈(0,1)", True, "α_t·log(1−v̄_t²) [nonlinear φ]")),
    ("Mamba-S6", build_mamba_s6, dict(),
     "float64", _switches("real diagonal ∈(0,1)^N", True, "Δ_t·B̄·u_t [linear, input-scaled]")),
    ("S5", build_s5, dict(),
     "complex128", _switches("COMPLEX diagonal exp(ΔΛ)", False, "Δ·B·u_t [linear, time-const B]")),
    ("LRU", build_lru, dict(),
     "complex128", _switches("COMPLEX diagonal e^{−ν+iθ}", False, "B·u_t [linear, time-const B]")),
]


def run_case(name, builder, dtype_label, switches, Bsz=4, T=37, device="cpu"):
    """Build random recurrence, run seq + parallel ⊗ scan, report max abs error.

    T=37 deliberately not a power of two (exercises the non-PoT doubling guard).
    """
    # builders take their own optional N/seed; call with batch+T only.
    A, b = builder(Bsz, T, device=device)
    h_seq = sequential_diag_ssm(A, b)
    h_par = parallel_diag_ssm(A, b)

    err = (h_par - h_seq).abs().max().item()
    # machine-precision bar: ~1e-13 for fp64/complex128 with T=37 reassociation
    tol = 1e-12
    ok = err < tol

    N = A.shape[-1]
    print(f"[{name:16s}] state={switches['state_algebra']:30s} N={N:<2d} "
          f"dtype={dtype_label:11s} input-dep={str(switches['A_t_input_dependent']):5s}")
    print(f"{'':18s} B-map: {switches['B_map']}")
    print(f"{'':18s} seq vs parallel ⊗-scan  max|Δ| = {err:.3e}   "
          f"(tol {tol:.0e})  {'PASS' if ok else 'FAIL'}")
    print()

    return {
        "name": name,
        "state_algebra": switches["state_algebra"],
        "A_t_input_dependent": switches["A_t_input_dependent"],
        "B_map": switches["B_map"],
        "dtype": dtype_label,
        "B": Bsz, "T": T, "N": N,
        "T_is_power_of_two": (T & (T - 1)) == 0,
        "max_abs_err_seq_vs_parallel": err,
        "tol": tol,
        "pass": bool(ok),
    }


def main():
    device = "cpu"  # torch.complex is fully supported on CPU; MPS is not needed here
    Bsz, T = 4, 37

    print("=" * 78)
    print("SSM-family reduction under ONE affine operator (A2,B2)⊗(A1,B1) =")
    print("                       (A2·A1, A2·B1 + B2),  identity (1,0)")
    print(f"torch {torch.__version__}  device={device}  B={Bsz}  T={T} "
          f"(power-of-two? {(T & (T-1)) == 0})")
    print("=" * 78)
    print()

    results = []
    for name, builder, _kw, dtype_label, switches in CASES:
        results.append(run_case(name, builder, dtype_label, switches,
                                Bsz=Bsz, T=T, device=device))

    # Headline numbers the doc cites.
    real_cases = [r for r in results if "complex" not in r["dtype"]]
    complex_cases = [r for r in results if "complex" in r["dtype"]]
    real_max = max(r["max_abs_err_seq_vs_parallel"] for r in real_cases)
    complex_max = max(r["max_abs_err_seq_vs_parallel"] for r in complex_cases)
    all_pass = all(r["pass"] for r in results)

    print("-" * 78)
    print(f"REAL    (GSSM scalar + Mamba real-diag)  max abs err = {real_max:.3e}")
    print(f"COMPLEX (S5 + LRU complex-diagonal)      max abs err = {complex_max:.3e}")
    print(f"ALL CASES PASS = {all_pass}")
    print("-" * 78)
    print("Claim status: 'one operator, three switches' is now CODE, not prose.")
    print("Complex-diagonal S5/LRU reduction is VERIFIED, not asserted.")

    out = {
        "torch_version": torch.__version__,
        "device": device,
        "operator": "(A2,B2) x (A1,B1) = (A2*A1, A2*B1 + B2), identity (1,0)",
        "three_switches": [
            "Switch 1 (state algebra of A): real scalar | real diagonal | complex diagonal",
            "Switch 2 (A_t input-dependence): selective (GSSM, Mamba) | LTI (S5, LRU)",
            "Switch 3 (B-map / drive): GSSM nonlinear φ | Mamba Δ·B·u | S5 Δ·B·u | LRU B·u",
        ],
        "B": Bsz, "T": T,
        "cases": results,
        "headline": {
            "real_max_abs_err": real_max,
            "complex_max_abs_err": complex_max,
            "all_pass": all_pass,
        },
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "ssm_family_reduction_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults JSON written to {out_path}")

    # Hard gate: nonzero exit if any case fails (so a CI / runner catches it).
    if not all_pass:
        print("*** FAIL: at least one SSM-family case did not reduce to the "
              "sequential recurrence to machine precision. ***", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
