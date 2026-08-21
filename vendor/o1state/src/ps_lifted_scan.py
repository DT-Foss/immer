"""
PS-Lifted Möbius Scan.

"PS" stands for parallel scan.  The Möbius coupling

    f(a, b) = (a + b) / (1 + a b)

is the projection of ordinary matrix multiplication on the group of
Lorentz boosts.  Represent a value x in (-1, 1) by the 2x2 matrix

    M(x) = [[1, x],
            [x, 1]]

Then the Möbius sum of a sequence is obtained from the prefix product

    P_t = M(x_t) @ M(x_{t-1}) @ ... @ M(x_1)

and projecting back:

    f(x_1, ..., x_t) = P_t[1, 0] / P_t[0, 0].

Because matrix multiplication is associative, the prefix product can be
computed with a parallel (Blelloch-style) scan in O(log T) sequential
steps instead of the O(T^2) nested Python loop of the original
MoebiusAttention.  The whole operation is fully vectorised over batch,
head and feature dimensions.

Reference: Foss 2026, "From Markov Chains to Minkowski Space".
"""

import math
from typing import Union

import torch
import torch.nn as nn


MOEBIUS_CLAMP = 0.95  # tighter bound prevents overflow in prefix products


def moebius_couple(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Elementwise Möbius coupling (a + b) / (1 + a b), clamped to (-1, 1)."""
    a = torch.clamp(a, -MOEBIUS_CLAMP, MOEBIUS_CLAMP)
    b = torch.clamp(b, -MOEBIUS_CLAMP, MOEBIUS_CLAMP)
    denom = 1.0 + a * b
    denom = torch.sign(denom) * torch.clamp(torch.abs(denom), min=eps)
    return torch.clamp((a + b) / denom, -MOEBIUS_CLAMP, MOEBIUS_CLAMP)


def lift_matrix(x: torch.Tensor) -> torch.Tensor:
    """Lift scalar x to 2x2 Möbius matrix [[1, x], [x, 1]].

    Shape: (*shape) -> (*shape, 2, 2)
    """
    x = torch.clamp(x, -MOEBIUS_CLAMP, MOEBIUS_CLAMP)
    shape = x.shape
    M = torch.zeros(*shape, 2, 2, dtype=x.dtype, device=x.device)
    M[..., 0, 0] = 1.0
    M[..., 1, 1] = 1.0
    M[..., 0, 1] = x
    M[..., 1, 0] = x
    return M


def project_from_lorentz(u: torch.Tensor) -> torch.Tensor:
    """Project lifted 2-vector u = [u0, u1] back to Möbius scalar u1/u0."""
    u0 = u[..., 0]
    u1 = u[..., 1]
    denom = torch.sign(u0) * torch.clamp(torch.abs(u0), min=1e-4)
    return torch.clamp(u1 / denom, -MOEBIUS_CLAMP, MOEBIUS_CLAMP)


def prefix_scan_2x2_iterative(M: torch.Tensor) -> torch.Tensor:
    """Iterative but fully vectorised prefix product of 2x2 matrices.

    M  : tensor of shape (..., T, 2, 2)
    Returns P where P[..., i, :, :] = M[..., i] @ ... @ M[..., 0].

    Complexity is O(T) sequential PyTorch ops, but each op is batched over
    all leading dimensions.  This is already much faster than a Python
    double loop; for true O(log T) depth see `prefix_scan_2x2_parallel`.
    """
    T = M.shape[-3]
    if T == 0:
        return M

    # Start with the first matrix, then accumulate.
    pieces = [M[..., 0:1, :, :]]
    acc = M[..., 0, :, :]
    for t in range(1, T):
        acc = acc @ M[..., t, :, :]
        pieces.append(acc.unsqueeze(-3))

    return torch.cat(pieces, dim=-3)


def _next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


def prefix_scan_2x2_parallel(M: torch.Tensor) -> torch.Tensor:
    """Blelloch-style parallel prefix product of 2x2 matrices.

    M  : tensor of shape (..., T, 2, 2)
    Returns P[..., i, :, :] = M[..., i] @ ... @ M[..., 0].

    Sequential depth is O(log T); all operations are batched.
    """
    T = M.shape[-3]
    if T <= 1:
        return M

    n = _next_power_of_two(T)
    if n != T:
        pad = torch.zeros(*M.shape[:-3], n - T, 2, 2, dtype=M.dtype, device=M.device)
        pad[..., 0, 0] = 1.0
        pad[..., 1, 1] = 1.0
        M = torch.cat([M, pad], dim=-3)
    else:
        M = M.clone()

    # ---------- upsweep: build partial products on the right edge of blocks ----------
    step = 1
    while step < n:
        # indices of right edge elements
        right = torch.arange(2 * step - 1, n, 2 * step, device=M.device)
        left = right - step
        # M[..., right, :, :] = M[..., left, :, :] @ M[..., right, :, :]
        M[..., right, :, :] = torch.matmul(
            M[..., left, :, :], M[..., right, :, :]
        )
        step *= 2

    # The last element now holds the total product; set it to identity so the
    # downsweep does not include it twice.
    M[..., -1, :, :] = torch.eye(2, dtype=M.dtype, device=M.device)

    # ---------- downsweep: propagate prefix products to the interior ----------
    step = n // 2
    while step >= 1:
        right = torch.arange(2 * step - 1, n, 2 * step, device=M.device)
        left = right - step
        # old right becomes left @ old right
        old_right = M[..., right, :, :].clone()
        M[..., right, :, :] = torch.matmul(
            M[..., left, :, :], M[..., right, :, :]
        )
        M[..., left, :, :] = old_right
        step //= 2

    return M[..., :T, :, :]


def ps_lifted_moebius_scan(
    lam: torch.Tensor,
    v_gated: torch.Tensor,
    causal: Union[bool, str] = True,
    parallel: bool = True,
) -> torch.Tensor:
    """Lifted parallel Möbius scan.

    Parameters
    ----------
    lam : (B, T, H, Dh)
        Per-position rest eigenvalue (tanh-bounded).
    v_gated : (B, T, H, Dh)
        Per-position gated velocity (tanh-bounded).
    causal : bool or str
        If True compute a causal prefix scan; if False compute the full
        associative sum and broadcast it to every position.
        If ``"bidirectional"``, compute a left prefix and a right suffix at
        each position and combine them via the Möbius coupling.
    parallel : bool
        Use O(log T) parallel scan; if False use iterative O(T) scan.

    Returns
    -------
    out : (B, T, H, Dh)
        state_t = lam_t coupled with the Möbius sum of v_gated up to t.
    """
    assert lam.shape == v_gated.shape
    B, T, H, Dh = lam.shape

    # Build lifted matrices for every scalar: (B, T, H, Dh, 2, 2)
    M = lift_matrix(v_gated)

    # Collapse all non-time dimensions so the scan runs along dim -3.
    M = M.permute(0, 2, 3, 1, 4, 5).reshape(B * H * Dh, T, 2, 2)

    scan_fn = prefix_scan_2x2_parallel if parallel else prefix_scan_2x2_iterative

    if causal == "bidirectional":
        # Forward prefix scan (left context, including t).
        P_fwd = scan_fn(M)  # (B*H*Dh, T, 2, 2)
        u_fwd = P_fwd[..., :, 0]

        # Backward suffix scan (right context, including t): reverse time,
        # run the same prefix scan, then reverse the result back.
        M_rev = torch.flip(M, dims=[1])
        P_rev = scan_fn(M_rev)
        P_bwd = torch.flip(P_rev, dims=[1])
        u_bwd = P_bwd[..., :, 0]

        # Restore shape and project both directions.
        u_fwd = u_fwd.view(B, H, Dh, T, 2).permute(0, 3, 1, 2, 4)
        u_bwd = u_bwd.view(B, H, Dh, T, 2).permute(0, 3, 1, 2, 4)
        prefix_fwd = project_from_lorentz(u_fwd)
        prefix_bwd = project_from_lorentz(u_bwd)
        prefix = moebius_couple(prefix_fwd, prefix_bwd)
    elif causal:
        P = scan_fn(M)  # (B*H*Dh, T, 2, 2)
        # Apply each prefix matrix to the canonical vector [1, 0]^T.
        u = P[..., :, 0]  # (B*H*Dh, T, 2)
        # Restore shape: (B, H, Dh, T, 2) -> (B, T, H, Dh, 2)
        u = u.view(B, H, Dh, T, 2).permute(0, 3, 1, 2, 4)
        prefix = project_from_lorentz(u)  # (B, T, H, Dh)
    else:
        # Non-causal: associative sum over all positions.  Same result for every t.
        total = M
        n = _next_power_of_two(T)
        if n != T:
            pad = torch.zeros(B * H * Dh, n - T, 2, 2, dtype=M.dtype, device=M.device)
            pad[..., 0, 0] = 1.0
            pad[..., 1, 1] = 1.0
            total = torch.cat([total, pad], dim=1)
        while total.shape[1] > 1:
            half = total.shape[1] // 2
            left = total[:, :half, :, :]
            right = total[:, half:, :, :]
            # left[i] = right[i] @ left[i]  (right is later in time)
            total = torch.matmul(right, left)
        u = total[:, 0, :, 0]  # (B*H*Dh, 2), first column of the final 2x2 matrix
        u = u.unsqueeze(1).expand(-1, T, -1)
        # Restore shape: (B, H, Dh, T, 2) -> (B, T, H, Dh, 2)
        u = u.view(B, H, Dh, T, 2).permute(0, 3, 1, 2, 4)
        prefix = project_from_lorentz(u)  # (B, T, H, Dh)

    # Final coupling with the per-position rest eigenvalue.
    out = moebius_couple(lam, prefix)
    return out


def _reference_moebius_scan(lam: torch.Tensor, v_gated: torch.Tensor) -> torch.Tensor:
    """Reference O(T^2) Python-loop scan for correctness checks."""
    B, T, H, Dh = lam.shape
    outputs = []
    for i in range(T):
        state = lam[:, i, :, :].clone()
        for j in range(i + 1):
            denom = 1.0 + state * v_gated[:, j, :, :]
            denom = torch.sign(denom) * torch.clamp(torch.abs(denom), min=1e-6)
            state = torch.clamp((state + v_gated[:, j, :, :]) / denom, -0.999, 0.999)
        outputs.append(state)
    return torch.stack(outputs, dim=1)


if __name__ == "__main__":
    B, T, H, Dh = 2, 16, 4, 8
    lam = torch.tanh(torch.randn(B, T, H, Dh))
    v_gated = torch.tanh(torch.randn(B, T, H, Dh))

    ref = _reference_moebius_scan(lam, v_gated)
    out_iter = ps_lifted_moebius_scan(lam, v_gated, causal=True, parallel=False)
    out_para = ps_lifted_moebius_scan(lam, v_gated, causal=True, parallel=True)

    print("PS-Lifted scan smoke test")
    print(f"  shapes: ref={ref.shape}, iterative={out_iter.shape}, parallel={out_para.shape}")
    print(f"  iterative vs ref max abs diff: {(out_iter - ref).abs().max().item():.2e}")
    print(f"  parallel vs ref max abs diff:  {(out_para - ref).abs().max().item():.2e}")

    # ── Bidirectional mode smoke test ──────────────────────────────────────
    out_bi_para = ps_lifted_moebius_scan(lam, v_gated, causal="bidirectional", parallel=True)
    out_bi_iter = ps_lifted_moebius_scan(lam, v_gated, causal="bidirectional", parallel=False)
    assert out_bi_para.shape == (B, T, H, Dh), out_bi_para.shape
    assert out_bi_iter.shape == (B, T, H, Dh), out_bi_iter.shape

    # The bidirectional scan must produce different values for different time steps.
    time_variation = (out_bi_para[:, 1:] - out_bi_para[:, :-1]).abs().max().item()
    assert time_variation > 1e-6, "bidirectional scan output must vary over time"

    bi_diff = (out_bi_para - out_bi_iter).abs().max().item()
    print("  bidirectional smoke test")
    print(f"    shape: {out_bi_para.shape}")
    print(f"    parallel vs iterative max abs diff: {bi_diff:.2e}")
    print(f"    time-step variation: {time_variation:.2e}")

    import time
    # Timing (CPU)
    t0 = time.time()
    for _ in range(20):
        _reference_moebius_scan(lam, v_gated)
    t_ref = (time.time() - t0) / 20

    t0 = time.time()
    for _ in range(200):
        ps_lifted_moebius_scan(lam, v_gated, causal=True, parallel=False)
    t_iter = (time.time() - t0) / 200

    t0 = time.time()
    for _ in range(200):
        ps_lifted_moebius_scan(lam, v_gated, causal=True, parallel=True)
    t_para = (time.time() - t0) / 200

    print(f"  reference O(T^2) scan: {t_ref*1000:.3f} ms")
    print(f"  lifted iterative scan: {t_iter*1000:.3f} ms")
    print(f"  lifted parallel scan:  {t_para*1000:.3f} ms")
