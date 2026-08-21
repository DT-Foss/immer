"""
Holographic Stack — K=2 Multi-Freq Harmonics × Ginibre D-Vector Keys
======================================================================

Combines TWO independent levers that attack crosstalk on different axes:

  LEVER 1 — K=2 multi-freq harmonic write (holographic_multifreq.py):
    Write the key at 2 harmonic angles φ and 2φ.
    Signal at matched key sums coherently (2× amplitude).
    Mismatch terms at k=1 and k=2 decohere independently.
    SNR ∝ √K = √2 over single-harmonic.

  LEVER 2 — Ginibre D-dimensional vector keys (holographic_ginibre.py):
    Each key is a D-vector of phases [φ_1,...,φ_D] (one per channel).
    Matched-filter read collapses D channels → one scalar per head.
    Cross-talk RMS = √((N−1)/D) vs √(N−1) for 1D keys.
    β=3 repulsion (λ_rep) spreads the key cloud toward Ginibre ⟨s²⟩≈1.087.

MODES (set via constructor flags):
  use_phase=False                       → exact GSSM-Selective (ablation control)
  use_phase=True, n_freqs=1, no W_match → baseline_1d = 1D holographic (~8-9%)
  use_phase=True, n_freqs=2, W_match    → Ginibre-ONLY (D-vec keys, K=1 single write)
  use_phase=True, n_freqs=1, W_match    → K2-ONLY (1D keys, K=2 harmonics)
  use_phase=True, n_freqs=2, W_match    → BOTH (D-vec keys + K=2 harmonics)

NOTE: the "K2-ONLY" arm uses D-vector keys with matched-filter (W_match) but n_freqs=1.
      The "Ginibre-ONLY" arm uses D-vector keys with W_match and n_freqs=1 but with
      repulsion; "K2-ONLY" = D-vec + n_freqs=2 but no repulsion.
      The "baseline" arm: n_freqs=1, no W_match (baseline_1d path = per-channel read).

This file implements a single unified StackedHolographicScanLayer that parameterizes
all four comparison arms cleanly, plus the full LM wrapper.

MPS-native: no torch.complex, two real leaky scans per harmonic band.
Diagnostic: logs ⟨s²⟩ of the key cloud (target ~1.087 for Ginibre spread).
"""

import os
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "reference"))
sys.path.insert(0, HERE)

D_MODEL  = 128
N_HEADS  = 4
D_HEAD   = 32
N_LAYERS = 2

LOG_COMPLEMENT_CLAMP = 0.999
EPS = 1e-6


def sequential_linear_scan(a: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """z_t = γ_t·z_{t-1} + a_t, shapes (B,T,H,D). Sequential O(T) loop, CPU-native."""
    B, T, H, D = a.shape
    Z = torch.zeros(B, H, D, device=a.device, dtype=a.dtype)
    out = []
    for t in range(T):
        Z = gamma[:, t] * Z + a[:, t]
        out.append(Z)
    return torch.stack(out, dim=1)


def ginibre_repulsion(phi: torch.Tensor, margin: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """β=3 cubic-onset repulsion for a key cloud.

    phi : (M, D) phase angles.  Chordal distance in ℂ^D space.
    Returns scalar loss (0 if M < 2).
    """
    M, D = phi.shape
    if M < 2:
        return phi.new_zeros(())
    cos_e = torch.cos(phi)   # (M, D)
    sin_e = torch.sin(phi)   # (M, D)
    # Gram: G_ij = Re⟨e_i, e_j⟩ = (1/D) Σ_c cos(phi_i[c] − phi_j[c])
    G = (cos_e @ cos_e.T + sin_e @ sin_e.T) / D  # (M, M)
    d2 = (2.0 - 2.0 * G).clamp(min=0.0)           # chordal squared distance
    iu = torch.triu_indices(M, M, offset=1, device=phi.device)
    s = d2[iu[0], iu[1]].clamp(min=eps).sqrt()    # pairwise distances
    s_mean = s.mean().clamp(min=eps)
    s_norm = s / s_mean
    return torch.relu(margin - s_norm).pow(3).mean()


def key_cloud_variance(phi: torch.Tensor, eps: float = 1e-6) -> float:
    """Diagnostic: ⟨s²⟩ of nearest-neighbor distances (Ginibre target ≈ 1.087).

    phi : (M, D) phase angles (detached, on CPU).
    Returns float.
    """
    with torch.no_grad():
        M, D = phi.shape
        if M < 2:
            return float('nan')
        cos_e = torch.cos(phi)
        sin_e = torch.sin(phi)
        G = (cos_e @ cos_e.T + sin_e @ sin_e.T) / D
        d2 = (2.0 - 2.0 * G).clamp(min=0.0)
        d2_fill = d2.clone()
        d2_fill.fill_diagonal_(float('inf'))
        nnd = d2_fill.min(dim=1).values.clamp(min=eps).sqrt()
        nnd_mean = nnd.mean().clamp(min=eps)
        s = nnd / nnd_mean
        return s.pow(2).mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Main stacked scan layer
# ─────────────────────────────────────────────────────────────────────────────

class StackedHolographicScanLayer(nn.Module):
    """
    Unified scan layer for all four comparison arms.

    Constructor flags
    -----------------
    use_phase : bool
        False → exact GSSM-Selective (ablation control, all holographic paths skipped).
    use_vector_key : bool
        True → D-dimensional vector-key + matched-filter read (W_match path).
        False → per-channel 1D read (baseline holographic, W_out mixes channels).
    n_freqs : int
        K = number of harmonic bands (1 = single write, 2 = dual-harmonic write).
    lambda_rep : float
        Weight on β=3 Ginibre repulsion loss (0 = no repulsion).
    rep_margin : float
        Hinge margin for repulsion (default 1.0 → targets s≈1, Ginibre mode).

    Arms
    ----
    baseline: use_phase=True, use_vector_key=False, n_freqs=1, lambda_rep=0
    K2_only:  use_phase=True, use_vector_key=True,  n_freqs=2, lambda_rep=0
    gin_only: use_phase=True, use_vector_key=True,  n_freqs=1, lambda_rep=λ
    both:     use_phase=True, use_vector_key=True,  n_freqs=2, lambda_rep=λ
    """

    def __init__(self, d_model: int, d_head: int = D_HEAD, n_heads: int = N_HEADS,
                 causal: bool = True, dropout: float = 0.0,
                 phase_scale: float = math.pi,
                 use_phase: bool = True,
                 use_vector_key: bool = True,
                 n_freqs: int = 2,
                 lambda_rep: float = 0.0,
                 rep_margin: float = 1.0):
        super().__init__()
        self.d_model       = d_model
        self.d_head        = d_head
        self.n_heads       = n_heads
        self.causal        = causal
        self.phase_scale   = phase_scale
        self.use_phase     = use_phase
        self.use_vector_key = use_vector_key
        self.n_freqs       = n_freqs
        self.lambda_rep    = lambda_rep
        self.rep_margin    = rep_margin

        assert n_freqs >= 1, n_freqs
        total_dim = n_heads * d_head

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # ── Magnitude / value projections (identical to Selective baseline) ──
        self.W_v     = nn.Linear(d_model, total_dim, bias=False)
        self.W_gate  = nn.Linear(d_model, total_dim, bias=False)
        self.W_gamma = nn.Linear(d_model, total_dim, bias=False)
        self.W_alpha = nn.Linear(d_model, total_dim, bias=False)
        self.W_out   = nn.Linear(total_dim, d_model, bias=False)

        # ── Holographic key projection ──
        # φ_t[c] = phase_scale · tanh(W_key x_t)[c], one phase per channel.
        self.W_key = nn.Linear(d_model, total_dim, bias=False)

        # ── Imaginary-channel out-projection ──
        # Used by BOTH the per-channel (baseline) and vector-key arms.
        self.W_im  = nn.Linear(total_dim, d_model, bias=False)

        # ── W_match: matched-filter scalar → d_model (vector-key arm only) ──
        # Projects H matched-filter scalars (one per head) to d_model.
        # Zero-init: let the per-channel path stabilize first.
        if use_vector_key:
            self.W_match = nn.Linear(n_heads, d_model, bias=False)
        else:
            self.W_match = None

        self._reset_parameters()

        # Repulsion loss accumulated during forward (read by training loop)
        self._rep_loss = None

    def _reset_parameters(self):
        for m in [self.W_gamma, self.W_alpha]:
            nn.init.xavier_uniform_(m.weight, gain=0.1)
        for m in [self.W_v, self.W_gate, self.W_out]:
            nn.init.xavier_uniform_(m.weight, gain=0.6)
        # W_key small → φ≈0 at init → starts near real-write (Selective) regime
        nn.init.xavier_uniform_(self.W_key.weight, gain=0.1)
        nn.init.xavier_uniform_(self.W_im.weight, gain=0.6)
        if self.W_match is not None:
            nn.init.zeros_(self.W_match.weight)

    # ── Shared drive and forget ───────────────────────────────────────────────

    def _drive_and_gamma(self, x):
        B, T, _ = x.shape
        v      = torch.tanh(self.W_v(x))
        gate   = torch.sigmoid(self.W_gate(x))
        gamma  = torch.sigmoid(self.W_gamma(x))
        alpha  = torch.sigmoid(self.W_alpha(x))

        v_gated = v * gate
        if self.dropout is not None:
            v_gated = self.dropout(v_gated)

        v_gated = v_gated.view(B, T, self.n_heads, self.d_head)
        gamma   = gamma.view(B, T, self.n_heads, self.d_head)
        alpha   = alpha.view(B, T, self.n_heads, self.d_head)

        w    = torch.clamp(v_gated * v_gated, max=LOG_COMPLEMENT_CLAMP)
        z_in = torch.log(1.0 - w + EPS)
        a    = alpha * z_in       # bounded log-complement drive a_t ≤ 0
        return a, gamma

    def _magnitude(self, x):
        """m_t = √(1 − exp z_t) ∈ [0,1) — load-bearing tanh_m gate."""
        a, gamma = self._drive_and_gamma(x)
        Z = sequential_linear_scan(a, gamma)
        s_sq = torch.clamp(1.0 - torch.exp(Z), min=0.0)
        return torch.sqrt(s_sq + EPS)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        self._rep_loss = None

        # ── Ablation: exact GSSM-Selective ──────────────────────────────────
        if not self.use_phase:
            m = self._magnitude(x)
            return self.W_out(m.view(B, T, self.n_heads * self.d_head))

        # ── Build holographic state ──────────────────────────────────────────
        a, gamma = self._drive_and_gamma(x)         # (B,T,H,D)
        phi = self.phase_scale * torch.tanh(self.W_key(x))
        phi = phi.view(B, T, self.n_heads, self.d_head)  # (B,T,H,D)

        # Accumulate K harmonic bands.
        # Each band k writes at angle k·φ and reads at angle k·φ (shared-QK).
        # Two real scans per band (re, im), same γ for all bands (single forget).
        read_re = torch.zeros(B, T, self.n_heads, self.d_head,
                              device=x.device, dtype=x.dtype)
        read_im = torch.zeros_like(read_re)

        for k in range(1, self.n_freqs + 1):
            kphi     = k * phi
            cos_kphi = torch.cos(kphi)
            sin_kphi = torch.sin(kphi)

            # Complex drive: a_t · e^{i·k·φ_t}
            drive_re_k = a * cos_kphi
            drive_im_k = a * sin_kphi

            # Leaky scan per band
            Sre_k = sequential_linear_scan(drive_re_k, gamma)
            Sim_k = sequential_linear_scan(drive_im_k, gamma)

            # De-rotate at query angle k·φ:
            #   Re(S^(k) · e^{−i·k·φ_q}) = Sre·cos(k·φ) + Sim·sin(k·φ)
            #   Im(S^(k) · e^{−i·k·φ_q}) = Sim·cos(k·φ) − Sre·sin(k·φ)
            read_re = read_re + (Sre_k * cos_kphi + Sim_k * sin_kphi)
            read_im = read_im + (Sim_k * cos_kphi - Sre_k * sin_kphi)

        # ── Readout: m·tanh (load-bearing constraint) ───────────────────────
        m = self._magnitude(x)                            # (B,T,H,D)
        read_re_gated = m * torch.tanh(read_re)          # (B,T,H,D)
        read_im_gated = m * torch.tanh(read_im)          # (B,T,H,D)

        read_re_flat = read_re_gated.view(B, T, self.n_heads * self.d_head)
        read_im_flat = read_im_gated.view(B, T, self.n_heads * self.d_head)

        # ── Per-channel output (all arms) ────────────────────────────────────
        out = self.W_out(read_re_flat) + self.W_im(read_im_flat)

        # ── Vector-key matched-filter (use_vector_key=True arms) ─────────────
        if self.use_vector_key:
            # Collapse D channels → 1 scalar per head via matched-filter sum.
            # read_re[b,t,h,c] = Σ_k (Sre^(k)·cos(k·φ) + Sim^(k)·sin(k·φ)) [c]
            # The D-dimensional average removes cross-key interference:
            # mismatch Gram entry = (1/D) Σ_c cos(φ_k[c]−φ_q[c]) → 0 as D grows.
            read_match = read_re.sum(dim=-1) / math.sqrt(self.d_head)  # (B,T,H)
            out = out + self.W_match(read_match)                        # (B,T,d_model)

            # ── β=3 Ginibre repulsion (only if lambda_rep > 0 and training) ──
            if self.lambda_rep > 0.0 and self.training:
                # Sample the key-cloud geometry: first batch, first head, all T positions.
                phi_sample = phi[0, :, 0, :].detach()   # (T, D)
                rep = ginibre_repulsion(phi_sample, margin=self.rep_margin)
                self._rep_loss = self.lambda_rep * rep

        return out

    def get_repulsion_loss(self) -> torch.Tensor:
        if self._rep_loss is not None:
            return self._rep_loss
        return torch.tensor(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer layer
# ─────────────────────────────────────────────────────────────────────────────

class StackedHolographicTransformerLayer(nn.Module):
    """Post-LN block wrapping StackedHolographicScanLayer."""

    def __init__(self, d_model: int, d_head: int = D_HEAD, n_heads: int = N_HEADS,
                 ffn_dim: int = None, dropout: float = 0.0, causal: bool = True,
                 phase_scale: float = math.pi,
                 use_phase: bool = True,
                 use_vector_key: bool = True,
                 n_freqs: int = 2,
                 lambda_rep: float = 0.0,
                 rep_margin: float = 1.0):
        super().__init__()
        self.scan = StackedHolographicScanLayer(
            d_model, d_head=d_head, n_heads=n_heads, causal=causal,
            dropout=dropout, phase_scale=phase_scale,
            use_phase=use_phase, use_vector_key=use_vector_key,
            n_freqs=n_freqs, lambda_rep=lambda_rep, rep_margin=rep_margin)
        self.ln1 = nn.LayerNorm(d_model)
        ffn_dim  = ffn_dim or 4 * d_model
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

    def get_repulsion_loss(self):
        return self.scan.get_repulsion_loss()


# ─────────────────────────────────────────────────────────────────────────────
# Full LM wrapper
# ─────────────────────────────────────────────────────────────────────────────

class StackedHolographicLM(nn.Module):
    """
    Causal LM for the stacked holographic GSSM.

    Same envelope as the other GSSM LMs (vocab_size+2 embedding for mask_idx).
    Exposes get_repulsion_loss() for the training loop (0 if no repulsion).
    """

    def __init__(self, vocab_size: int, mask_idx: int,
                 d_model: int = D_MODEL, n_layers: int = N_LAYERS,
                 n_heads: int = N_HEADS, d_head: int = D_HEAD,
                 seq_len: int = 64, dropout: float = 0.0, causal: bool = True,
                 phase_scale: float = math.pi,
                 use_phase: bool = True,
                 use_vector_key: bool = True,
                 n_freqs: int = 2,
                 lambda_rep: float = 0.0,
                 rep_margin: float = 1.0):
        super().__init__()
        from moebius_attention import SinusoidalPositionalEncoding
        self.mask_idx = mask_idx
        self.embed    = nn.Embedding(vocab_size + 2, d_model)
        self.pos      = SinusoidalPositionalEncoding(d_model)
        self.layers   = nn.ModuleList([
            StackedHolographicTransformerLayer(
                d_model, d_head=d_head, n_heads=n_heads, ffn_dim=4 * d_model,
                dropout=dropout, causal=causal, phase_scale=phase_scale,
                use_phase=use_phase, use_vector_key=use_vector_key,
                n_freqs=n_freqs, lambda_rep=lambda_rep, rep_margin=rep_margin)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size + 1)

    def forward(self, x):
        h = self.pos(self.embed(x))
        for layer in self.layers:
            h = layer(h)
        return self.head(h)

    def get_repulsion_loss(self) -> torch.Tensor:
        total = torch.tensor(0.0)
        for layer in self.layers:
            total = total + layer.get_repulsion_loss()
        return total
