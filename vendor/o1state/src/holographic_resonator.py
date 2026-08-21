"""
Holographic-GSSM + Bounded Resonator Cleanup — by Claude (Sonnet 4.6)
======================================================================

DERIVATION — WHY A RESONATOR AND WHY IT STAYS BOUNDED
------------------------------------------------------
The standard holographic read gives:

    r_0 = Re(S · e^{−iφ_q}) = Σ_k γ_{k→t} u_k cos(φ_k − φ_q)

The matched key (φ_k ≈ φ_q) contributes u_k·cos(0) = u_k.
All N−1 other keys add crosstalk: u_k·cos(φ_k − φ_q) which only averages
toward zero for large N. With n_pairs=8 this crosstalk is the capacity wall.

The IMAGINARY part of the same de-rotation is:

    im_0 = Im(S · e^{−iφ_q}) = Σ_k γ_{k→t} u_k sin(φ_k − φ_q)

At the matched key, sin(0) = 0. For a SLIGHTLY mismatched query angle
(φ_q ≈ φ_k + ε), im_0 ≈ −u_k · ε. So im_0 carries the GRADIENT of the
real read w.r.t. φ_q — pointing from the current query phase toward the
nearest stored phase maximum.

RESONATOR CLEANUP STEP (k = 0, 1, ..., K_iter−1):
    φ_{k+1} = φ_q + λ · norm(im_k)          ← phase correction
    r_{k+1}  = Re(S · e^{−iφ_{k+1}})        ← re-read at corrected angle
    im_{k+1} = Im(S · e^{−iφ_{k+1}})        ← residual for next step

where norm(im_k) = im_k / (|im_k| + ε) is a sign-scaled step (bounded to
(-λ, +λ) per channel) and λ is a learned per-head scalar (init near 0).

BOUNDEDNESS PROOF:
  1. S_re, S_im are already computed by the leaky scan (bounded |S|≤|u|/(1-γ)).
  2. Each cleanup step computes re(S·e^{-iφ}) and im(S·e^{-iφ}) — O(1)
     elementwise trig on the EXISTING state, no new recurrence, no new scan.
  3. The total phase shift Σ_{k} λ·norm(im_k) is at most K_iter · λ — bounded
     by construction since λ is a learned scalar (initialized small).
  4. No attention over the sequence: the operation at position t uses ONLY
     (S_re_t, S_im_t, φ_q_t) — quantities already available per-position.
  5. K_iter ∈ {0, 1, 2, 3} fixed at build time: deterministic compute budget.

K_iter=0 → EXACT baseline (no cleanup, identical to HolographicScanLayer).

CONNECTION TO VSA RESONATOR NETWORKS:
  Classical VSA resonators (Frady et al. 2020, Kent 2020) iterate:
      x_{k+1} = φ(W x_k)  where W encodes the codebook
  That requires O(V) codebook size → explicit attention. Our version AVOIDS
  the codebook by exploiting the Fourier structure of phase binding: the
  imaginary residual IS the local gradient of the binding function w.r.t.
  the query phase, so one step of gradient ascent on the phase circle moves
  toward the nearest stored key without materializing the codebook.

REDUCTION GUARANTEE:
  cleanup_steps=0  →  identical output to HolographicScanLayer with same
  weights (the cleanup module is a no-op pass-through). use_phase=False →
  EXACT Selective (magnitude only, no phase path).
"""

import os
import sys
import math

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "reference"))
sys.path.insert(0, HERE)

D_MODEL = 128
N_HEADS = 4
D_HEAD = 32
N_LAYERS = 2

LOG_COMPLEMENT_CLAMP = 0.999
EPS = 1e-6


def sequential_linear_scan(a: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """z_t = γ_t·z_{t-1} + a_t, shapes (B,T,H,D)."""
    B, T, H, D = a.shape
    Z = torch.zeros(B, H, D, device=a.device, dtype=a.dtype)
    out = []
    for t in range(T):
        Z = gamma[:, t] * Z + a[:, t]
        out.append(Z)
    return torch.stack(out, dim=1)


class ResonatorReadout(nn.Module):
    """
    Bounded resonator cleanup on top of a holographic read.

    Given (S_re, S_im) — the real and imaginary parts of the complex
    holographic state — and φ_q — the query/read angle — this module
    performs K_iter Newton-like phase-correction steps:

        im_k = S_im·cos(φ_k) − S_re·sin(φ_k)     ← imaginary residual
        Δφ_k = λ · im_k / (|im_k| + ε)            ← bounded phase step
        φ_{k+1} = φ_q + Σ_{j≤k} Δφ_j              ← cumulative correction

    Then reads r_K = S_re·cos(φ_K) + S_im·sin(φ_K).

    K_iter=0: no correction, r_0 = Re(S·e^{-iφ_q}) exactly. REDUCTION EXACT.
    λ is a per-head learned scalar, init near 0 so the cleanup starts passive.
    """

    def __init__(self, n_heads: int, d_head: int, cleanup_steps: int = 1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.cleanup_steps = cleanup_steps

        # λ: per-head step-size scalar. One scalar per head (shared across d_head
        # channels within a head). Init at 0.01 — small but non-zero so gradient flows.
        # Shape (1, 1, n_heads, 1) for broadcasting over (B, T, H, D).
        if cleanup_steps > 0:
            self.lam = nn.Parameter(
                torch.full((1, 1, n_heads, 1), 0.01)
            )
        else:
            self.lam = None

    def forward(self,
                S_re: torch.Tensor,   # (B, T, H, D)
                S_im: torch.Tensor,   # (B, T, H, D)
                phi_q: torch.Tensor   # (B, T, H, D) — per-channel query angle
                ) -> tuple:
        """
        Returns (read_re, read_im) after K_iter cleanup steps.
        K=0: exact holographic read, no cleanup applied.
        """
        if self.cleanup_steps == 0:
            # K=0: standard read, no resonator correction.
            r  = S_re * torch.cos(phi_q) + S_im * torch.sin(phi_q)
            im = S_im * torch.cos(phi_q) - S_re * torch.sin(phi_q)
            return r, im

        # Iterative phase cleanup
        phi_cur = phi_q                     # running corrected angle
        for _ in range(self.cleanup_steps):
            cos_phi = torch.cos(phi_cur)
            sin_phi = torch.sin(phi_cur)
            # imaginary residual: derivative of Re(S·e^{-iφ}) w.r.t. φ (up to sign)
            im_cur = S_im * cos_phi - S_re * sin_phi  # (B,T,H,D)
            # bounded step: λ · sign(im_cur) scaled by |im_cur|/(|im_cur|+ε)
            # = λ · tanh(im_cur / ε_scale) approximately for large ε_scale
            # We use the simple normalized step: im / (|im| + ε) ∈ (−1, +1)
            step = im_cur / (im_cur.abs() + EPS)      # soft-sign, bounded
            phi_cur = phi_cur + self.lam * step        # lam broadcast (1,1,H,1)

        # Final read at corrected angle
        cos_f = torch.cos(phi_cur)
        sin_f = torch.sin(phi_cur)
        r  = S_re * cos_f + S_im * sin_f
        im = S_im * cos_f - S_re * sin_f
        return r, im


class ResonatorHolographicScanLayer(nn.Module):
    """
    GSSM holographic scan layer with optional bounded resonator cleanup.

    use_phase=False  → exact GSSM-Selective (reduction control, no resonator).
    use_phase=True, cleanup_steps=0  → standard holographic baseline (= 8-9%).
    use_phase=True, cleanup_steps=K  → K resonator cleanup steps after initial read.

    The resonator module (ResonatorReadout) adds only:
      - 1 learned scalar λ per head per cleanup step (n_heads params per layer)
      - K_iter passes of O(D) elementwise trig per position
    No attention, no sequence scan, no codebook materialization.
    """

    def __init__(self, d_model: int, d_head: int = D_HEAD, n_heads: int = N_HEADS,
                 causal: bool = True, dropout: float = 0.0,
                 phase_scale: float = math.pi, use_phase: bool = True,
                 readout: str = "tanh_m", cleanup_steps: int = 0):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads
        self.causal = causal
        self.phase_scale = phase_scale
        self.use_phase = use_phase
        self.cleanup_steps = cleanup_steps
        assert readout in ("tanh_m", "rms", "layernorm"), readout
        self.readout = readout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        total_dim = n_heads * d_head

        # Magnitude / value projections (identical to Selective)
        self.W_v     = nn.Linear(d_model, total_dim, bias=False)
        self.W_gate  = nn.Linear(d_model, total_dim, bias=False)
        self.W_gamma = nn.Linear(d_model, total_dim, bias=False)
        self.W_alpha = nn.Linear(d_model, total_dim, bias=False)
        self.W_out   = nn.Linear(total_dim, d_model, bias=False)

        # Holographic projections
        self.W_key = nn.Linear(d_model, total_dim, bias=False)   # write+read angle
        self.W_im  = nn.Linear(total_dim, d_model, bias=False)   # imaginary channel out

        # Resonator module (no-op when cleanup_steps=0 but still initialised so
        # the module list is consistent; the forward is O(1) for K=0).
        self.resonator = ResonatorReadout(n_heads, d_head, cleanup_steps)

        self._reset_parameters()

    def _reset_parameters(self):
        for module in [self.W_gamma, self.W_alpha]:
            for p in module.parameters():
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p, gain=0.1)
        for module in [self.W_v, self.W_gate, self.W_out]:
            for p in module.parameters():
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p, gain=0.6)
        # W_key small → φ≈0 at init → starts near Selective regime.
        for p in self.W_key.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p, gain=0.1)
        for p in self.W_im.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p, gain=0.6)

    def _drive_and_gamma(self, x):
        B, T, _ = x.shape
        v = torch.tanh(self.W_v(x))
        gate = torch.sigmoid(self.W_gate(x))
        gamma = torch.sigmoid(self.W_gamma(x))
        alpha = torch.sigmoid(self.W_alpha(x))
        v_gated = v * gate
        if self.dropout is not None:
            v_gated = self.dropout(v_gated)
        v_gated = v_gated.view(B, T, self.n_heads, self.d_head)
        gamma   = gamma.view(B, T, self.n_heads, self.d_head)
        alpha   = alpha.view(B, T, self.n_heads, self.d_head)
        w = torch.clamp(v_gated * v_gated, max=LOG_COMPLEMENT_CLAMP)
        z_in = torch.log(1.0 - w + EPS)
        a = alpha * z_in
        return a, gamma

    def _magnitude(self, x):
        a, gamma = self._drive_and_gamma(x)
        if self.causal:
            Z = sequential_linear_scan(a, gamma)
        else:
            Z_fwd = sequential_linear_scan(a, gamma)
            Z_rev = torch.flip(sequential_linear_scan(
                torch.flip(a, dims=[1]), torch.flip(gamma, dims=[1])), dims=[1])
            Z = Z_fwd + Z_rev
        s_sq = torch.clamp(1.0 - torch.exp(Z), min=0.0)
        return torch.sqrt(s_sq + EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        if not self.use_phase:
            # Ablation: EXACT Selective, no phase path, no resonator.
            m = self._magnitude(x)
            return self.W_out(m.view(B, T, self.n_heads * self.d_head))

        # Key-conditioned holographic write
        a, gamma = self._drive_and_gamma(x)
        phi_q = self.phase_scale * torch.tanh(self.W_key(x))      # shared write/read angle
        phi_q = phi_q.view(B, T, self.n_heads, self.d_head)

        # Complex leaky scans: S_t = γ_t S_{t-1} + u_t e^{iφ_t}
        drive_re = a * torch.cos(phi_q)
        drive_im = a * torch.sin(phi_q)

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

        # Resonator cleanup: K_iter phase-correction steps on the complex state.
        # K=0 → exact standard holographic read (no-op resonator).
        read_re, read_im = self.resonator(S_re, S_im, phi_q)

        # Scale readout
        if self.readout == "tanh_m":
            m = self._magnitude(x)
            read_re = m * torch.tanh(read_re)
            read_im = m * torch.tanh(read_im)
        elif self.readout == "rms":
            rms_re = read_re.pow(2).mean(dim=-1, keepdim=True).add(EPS).sqrt()
            rms_im = read_im.pow(2).mean(dim=-1, keepdim=True).add(EPS).sqrt()
            read_re = read_re / rms_re
            read_im = read_im / rms_im

        read_re = read_re.view(B, T, self.n_heads * self.d_head)
        read_im = read_im.view(B, T, self.n_heads * self.d_head)
        return self.W_out(read_re) + self.W_im(read_im)


class ResonatorTransformerLayer(nn.Module):
    """Post-LN block around ResonatorHolographicScanLayer."""

    def __init__(self, d_model: int, d_head: int = D_HEAD, n_heads: int = N_HEADS,
                 ffn_dim: int = None, dropout: float = 0.0, causal: bool = True,
                 phase_scale: float = math.pi, use_phase: bool = True,
                 readout: str = "tanh_m", cleanup_steps: int = 0):
        super().__init__()
        self.scan = ResonatorHolographicScanLayer(
            d_model, d_head=d_head, n_heads=n_heads, causal=causal,
            dropout=dropout, phase_scale=phase_scale, use_phase=use_phase,
            readout=readout, cleanup_steps=cleanup_steps)
        self.ln1 = nn.LayerNorm(d_model)
        ffn_dim = ffn_dim or 4 * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(ffn_dim, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.ln1(x + self.scan(x))
        x = self.ln2(x + self.ffn(x))
        return x


class ResonatorHolographicLM(nn.Module):
    """Causal LM wrapper for the resonator holographic scan."""

    def __init__(self, vocab_size: int, mask_idx: int,
                 d_model: int = D_MODEL, n_layers: int = N_LAYERS,
                 n_heads: int = N_HEADS, d_head: int = D_HEAD,
                 seq_len: int = 64, dropout: float = 0.0, causal: bool = True,
                 phase_scale: float = math.pi, use_phase: bool = True,
                 readout: str = "tanh_m", cleanup_steps: int = 0):
        super().__init__()
        from moebius_attention import SinusoidalPositionalEncoding
        self.mask_idx = mask_idx
        self.embed = nn.Embedding(vocab_size + 2, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            ResonatorTransformerLayer(
                d_model, d_head=d_head, n_heads=n_heads, ffn_dim=4 * d_model,
                dropout=dropout, causal=causal, phase_scale=phase_scale,
                use_phase=use_phase, readout=readout, cleanup_steps=cleanup_steps)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size + 1)

    def forward(self, x):
        h = self.pos(self.embed(x))
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Reduction gate: use_phase=False + cleanup_steps=0 must equal Selective.
# ---------------------------------------------------------------------------

def _verify_reduction(device="cpu", tol=1e-5):
    """
    ResonatorHolographicScanLayer with use_phase=False must be byte-identical
    to GSSM-Selective (the ablation control must be exact).
    """
    from moebius_scan_transformer_selective import SelectiveRapiditySqrtScanLayer
    torch.manual_seed(0)
    d_model, n_heads, d_head = 48, 4, 12
    res = ResonatorHolographicScanLayer(
        d_model, d_head=d_head, n_heads=n_heads,
        use_phase=False, cleanup_steps=0).to(device).eval()
    sel = SelectiveRapiditySqrtScanLayer(
        d_model, d_head=d_head, n_heads=n_heads, dropout=0.0).to(device).eval()
    with torch.no_grad():
        sel.W_v.weight.copy_(res.W_v.weight)
        sel.W_gate.weight.copy_(res.W_gate.weight)
        sel.W_gamma.weight.copy_(res.W_gamma.weight)
        sel.W_alpha.weight.copy_(res.W_alpha.weight)
        sel.W_out.weight.copy_(res.W_out.weight)
    x = torch.randn(3, 37, d_model, device=device)
    err = (res(x) - sel(x)).abs().max().item()
    ok = err < tol
    print(f"[reduction] use_phase=False,K=0 vs Selective  max|Δ| = {err:.3e}  "
          f"{'PASS (exact reduction)' if ok else 'FAIL'}")
    return ok, err


def _verify_k0_equals_baseline(device="cpu", tol=1e-5):
    """
    cleanup_steps=0 with use_phase=True must produce the SAME output as the
    standard HolographicScanLayer (ResonatorReadout is a no-op at K=0).
    """
    from holographic_gssm import HolographicScanLayer
    torch.manual_seed(7)
    d_model, n_heads, d_head = 48, 4, 12
    res = ResonatorHolographicScanLayer(
        d_model, d_head=d_head, n_heads=n_heads,
        use_phase=True, readout="tanh_m", cleanup_steps=0).to(device).eval()
    base = HolographicScanLayer(
        d_model, d_head=d_head, n_heads=n_heads,
        use_phase=True, readout="tanh_m").to(device).eval()
    # Copy shared weights
    with torch.no_grad():
        for name in ["W_v", "W_gate", "W_gamma", "W_alpha", "W_out", "W_key", "W_im"]:
            getattr(base, name).weight.copy_(getattr(res, name).weight)
    x = torch.randn(3, 37, d_model, device=device)
    err = (res(x) - base(x)).abs().max().item()
    ok = err < tol
    print(f"[reduction] cleanup_steps=0 vs HolographicScanLayer  max|Δ| = {err:.3e}  "
          f"{'PASS (K=0 is exact baseline)' if ok else 'FAIL'}")
    return ok, err


if __name__ == "__main__":
    print("=" * 74)
    print("Holographic-GSSM + Bounded Resonator")
    print("=" * 74)
    ok1, err1 = _verify_reduction()
    ok2, err2 = _verify_k0_equals_baseline()

    # Sanity: K>0 produces a different, finite output
    torch.manual_seed(42)
    layer_k0 = ResonatorHolographicScanLayer(48, d_head=12, n_heads=4,
                                              use_phase=True, readout="tanh_m",
                                              cleanup_steps=0).eval()
    layer_k3 = ResonatorHolographicScanLayer(48, d_head=12, n_heads=4,
                                              use_phase=True, readout="tanh_m",
                                              cleanup_steps=3).eval()
    # Copy weights so only the resonator differs
    with torch.no_grad():
        for name in ["W_v", "W_gate", "W_gamma", "W_alpha", "W_out", "W_key", "W_im"]:
            getattr(layer_k3, name).weight.copy_(getattr(layer_k0, name).weight)
    x = torch.randn(2, 40, 48)
    y0 = layer_k0(x)
    y3 = layer_k3(x)
    print(f"[sanity]    K=0 finite={torch.isfinite(y0).all().item()}  std={y0.std():.3f}")
    print(f"[sanity]    K=3 finite={torch.isfinite(y3).all().item()}  std={y3.std():.3f}")
    print(f"[sanity]    K=3 vs K=0 max|Δ|={( y3-y0).abs().max():.3e} (nonzero=resonator active)")
    sys.exit(0 if (ok1 and ok2) else 1)
