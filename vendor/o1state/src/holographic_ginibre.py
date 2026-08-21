"""
Holographic-Ginibre Layer — D-dimensional vector keys + β=3 repulsion
======================================================================

THE PROBLEM WITH THE EXISTING HOLOGRAPHIC LAYER.
The existing HolographicScanLayer in holographic_gssm.py uses ONE shared phase
angle per (head, channel):

    phi_w[b, t, h, c]  is computed from W_key(x)[h, c], each channel c has its
                        OWN tanh output — SEPARATE values — but the READ is then
                        done PER CHANNEL:
                            read_re[b,t,h,c] = S_re[c]*cos(phi_r[c]) + S_im[c]*sin(phi_r[c])
                        and W_out mixes these independently.

This means each channel is a SEPARATE 1-D holographic memory with its own angle,
and W_out aggregates them linearly.  The effective key for EACH channel is a
scalar on S^1 (the unit circle, ℝ², D=2).  Crosstalk per channel = O(1/√2).

THE FIX (derived from the theorist's arithmetic).
Instead of letting W_out mix D independent 1-channel memories, COLLAPSE the
channel dimension in the MATCHED-FILTER READ:

    read_match[b,t,h] = (1/√D) Σ_c [ S_re[c]·cos(φ_q[c]) + S_im[c]·sin(φ_q[c]) ]
                       = (1/√D) Re⟨S_t, e^{iφ_q}⟩   where e^{iφ_q} ∈ ℂ^D.

Now the key is a D-VECTOR of unit-modulus complex numbers (one phase per channel).
The Gram entry ⟨e_k, e_q⟩ = (1/D) Σ_c cos(φ_k[c] − φ_q[c]) averages D independent
cosines → crosstalk RMS = √((N−1)/D) instead of √(N−1).  At N=8, D=32: 0.47 vs 1.87.

MODES
-----
repulsion=False  → vector-key matched-filter read (dimensionality fix only, no loss term).
                   "Arm B" in the experiment.
repulsion=True   → Arm B + β=3 cubic-onset Ginibre repulsion regularizer on the key cloud.
                   "Arm C" in the experiment.
use_phase=False  → exact GSSM-Selective reduction (ablation control, byte-identical).

REDUCTION GUARANTEE.
When use_phase=False: W_key, phi computation, and the matched-filter sum are all
SKIPPED.  The layer degenerates to exactly:
    m = sqrt(1 − exp(z_t)),  output = W_out(m.view(...))
which is byte-identical to SelectiveRapiditySqrtScanLayer on matching weights.

BASELINE MODE (repulsion_only_baseline=True).
When this flag is True AND use_phase=True: uses the OLD per-channel read (not the
collapsed matched filter), making this equivalent to the 1D holographic baseline
(but with vector key phases).  This reproduces the ~8-9% baseline wall.

MPS-NATIVE. No torch.complex.  Two real leaky scans (re/im).  The repulsion loss
operates on an N×N Gram (N=8, N^2=64 ops) — negligible compute.

Reference: Foss 2026, "One Constant Rules All 2D Spectra", §VII Eq.(10),
           §XII.A Eq.(14), §XVII Δ-metric.  Derivation in the theorist's writeup.
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

D_MODEL = 128
N_HEADS = 4
D_HEAD = 32
N_LAYERS = 2

LOG_COMPLEMENT_CLAMP = 0.999
EPS = 1e-6


def sequential_linear_scan(a: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """z_t = γ_t·z_{t-1} + a_t, shapes (B,T,H,D).  Sequential O(T) loop."""
    B, T, H, D = a.shape
    Z = torch.zeros(B, H, D, device=a.device, dtype=a.dtype)
    out = []
    for t in range(T):
        Z = gamma[:, t] * Z + a[:, t]
        out.append(Z)
    return torch.stack(out, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# β=3 Ginibre repulsion regularizer
# ─────────────────────────────────────────────────────────────────────────────

def ginibre_repulsion(e: torch.Tensor, margin: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """Cubic-onset repulsion loss for a set of unit-modulus key vectors.

    e : (M, D) real tensor, where each row is a D-dimensional unit-phase key vector
        represented as [cos(phi_1), sin(phi_1), cos(phi_2), sin(phi_2), ..] or simply
        as the raw phi angles (we work in phase space; the Gram is computed via cos).
        Here we expect e to be the STACKED [cos(phi), sin(phi)] representation so that
        e[i] · e[j] = (1/D) Σ_c cos(phi_i[c] − phi_j[c]) = Re⟨e_i, e_j⟩ / D.

    Actually simpler: pass cos_phi and sin_phi separately and compute
        G_ij = (cos_phi_i · cos_phi_j + sin_phi_i · sin_phi_j).mean(dim=-1)
    which is Re⟨e_i, e_j⟩.  We accept phi (M, D) and compute internally.

    phi : (M, D) phase angles in [-π, π].

    The paper's metric (§XVII, Def. 1): s_i = d_i / d̄, where d_i = min_{j≠i} |e_i − e_j|.
    We use PAIRWISE distances s_ij = |e_i − e_j| / mean_pair_distance.
    The β=3 onset: p(s) ∼ s³  <=>  penalty ∼ relu(margin − s)³.
    margin=1 targets s≈1 (the kernel mode from Table IX).

    Returns a scalar loss term.
    """
    # e is (M, D) — phase angles. Compute Re Gram via cosine dot:
    # G_ij = (1/D) Σ_c cos(phi_i[c] − phi_j[c])
    #       = (1/D) [ cos_i·cos_j + sin_i·sin_j ]  (trig identity)
    cos_e = torch.cos(e)   # (M, D)
    sin_e = torch.sin(e)   # (M, D)
    M, D = e.shape
    # Gram: G_ij = (cos_e_i · cos_e_j + sin_e_i · sin_e_j) / D
    G = (cos_e @ cos_e.T + sin_e @ sin_e.T) / D  # (M, M), range [-1, 1]
    # Chordal squared distance: |e_i − e_j|² = 2 − 2 G_ij   (for unit-modulus vecs)
    d2 = (2.0 - 2.0 * G).clamp(min=0.0)          # (M, M)
    # Extract upper triangle (unique pairs)
    iu = torch.triu_indices(M, M, offset=1, device=e.device)
    s = d2[iu[0], iu[1]].clamp(min=eps).sqrt()   # pairwise distances
    # Normalize by mean spacing (paper Def. 1, §XVII)
    s_mean = s.mean().clamp(min=eps)
    s_norm = s / s_mean
    # Cubic-onset hinge: relu(margin − s_norm)³  (β=3 onset, not log/1/r)
    rep_loss = torch.relu(margin - s_norm).pow(3).mean()
    return rep_loss


def key_cloud_variance(phi: torch.Tensor, eps: float = 1e-6) -> float:
    """Compute ⟨s²⟩ of the key-cloud NND distribution — paper's §XVII diagnostic.

    Target: ⟨s²⟩ → 1.087 (Ginibre universal).
    ≈ 1.0 → Poisson (not repelling, λ_rep too small).
    >> 1.4 → over-regularized lattice (λ_rep too big).

    phi: (M, D) — all key phase vectors seen in a batch.
    Returns float (detached).
    """
    with torch.no_grad():
        cos_e = torch.cos(phi)
        sin_e = torch.sin(phi)
        M, D = phi.shape
        G = (cos_e @ cos_e.T + sin_e @ sin_e.T) / D
        d2 = (2.0 - 2.0 * G).clamp(min=0.0)
        # For each point, find its nearest-neighbor distance
        d2_fill = d2.clone()
        d2_fill.fill_diagonal_(float('inf'))
        nnd = d2_fill.min(dim=1).values.clamp(min=eps).sqrt()
        nnd_mean = nnd.mean().clamp(min=eps)
        s = nnd / nnd_mean
        return s.pow(2).mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Main layer
# ─────────────────────────────────────────────────────────────────────────────

class GinibreHolographicScanLayer(nn.Module):
    """GSSM-Selective magnitude + D-dimensional vector-key holographic write/read.

    Three modes (set at construction):
      use_phase=False                      → exact Selective reduction (ablation)
      use_phase=True, repulsion=False      → Arm B: vector-key, no repulsion loss
      use_phase=True, repulsion=True       → Arm C: vector-key + β=3 Ginibre loss

    One extra mode for reproducing the 1D holographic baseline:
      use_phase=True, baseline_1d=True     → OLD per-channel read (D=2 effective)
                                             Should reproduce ~8-9% wall.
    """

    def __init__(self, d_model: int, d_head: int = D_HEAD, n_heads: int = N_HEADS,
                 causal: bool = True, dropout: float = 0.0,
                 phase_scale: float = math.pi, use_phase: bool = True,
                 repulsion: bool = False, lambda_rep: float = 0.03,
                 rep_margin: float = 1.0,
                 baseline_1d: bool = False):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads
        self.causal = causal
        self.phase_scale = phase_scale
        self.use_phase = use_phase
        self.repulsion = repulsion
        self.lambda_rep = lambda_rep
        self.rep_margin = rep_margin
        # baseline_1d=True: use the old per-channel read (W_out mixes channels
        # independently), effective D=2 per channel.  This reproduces ~8-9%.
        self.baseline_1d = baseline_1d

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        total_dim = n_heads * d_head

        # ── Magnitude / value projections (byte-identical to Selective) ──
        self.W_v     = nn.Linear(d_model, total_dim, bias=False)
        self.W_gate  = nn.Linear(d_model, total_dim, bias=False)
        self.W_gamma = nn.Linear(d_model, total_dim, bias=False)
        self.W_alpha = nn.Linear(d_model, total_dim, bias=False)
        self.W_out   = nn.Linear(total_dim, d_model, bias=False)

        # ── Key projections (NEW) ──
        # W_key → per-channel phase φ_t[c] = π·tanh(W_key x_t)[c]
        # Shape: (d_model → n_heads*d_head), one phase PER CHANNEL.
        # This is already the case in holographic_gssm.py — the difference is in
        # the READOUT: we collapse channels via matched-filter sum, not W_out mixing.
        self.W_key = nn.Linear(d_model, total_dim, bias=False)

        # For the baseline_1d mode we also need W_im (the imaginary-channel projection).
        # In Arm B/C the per-channel read_re is also kept as a residual (see below).
        self.W_im = nn.Linear(total_dim, d_model, bias=False)

        # W_match: projects the matched-filter scalar per head (n_heads → d_model).
        # The matched-filter sum collapses the D channel dim to ONE scalar per head.
        # Feeding that scalar broadcast to W_out would give a rank-H effective input
        # (only H independent values across H*D inputs) — very poor gradient flow.
        # Instead, project the H matched-filter scalars directly to d_model with W_match.
        # This gives: output = W_match(matched_scalars) + W_out(per_channel_re)
        # where per_channel_re carries per-channel diversity for W_out (full rank H*D).
        self.W_match = nn.Linear(n_heads, d_model, bias=False) if not baseline_1d else None

        self._reset_parameters()

        # Storage for the repulsion loss (populated during forward, read by training loop)
        self._rep_loss = None

    def _reset_parameters(self):
        for m in [self.W_gamma, self.W_alpha]:
            nn.init.xavier_uniform_(m.weight, gain=0.1)
        for m in [self.W_v, self.W_gate, self.W_out]:
            nn.init.xavier_uniform_(m.weight, gain=0.6)
        # W_key small → φ≈0 at init → starts near Selective regime
        nn.init.xavier_uniform_(self.W_key.weight, gain=0.1)
        nn.init.xavier_uniform_(self.W_im.weight, gain=0.6)
        if self.W_match is not None:
            # Zero init: start from baseline_1d regime (only per-channel path active).
            # W_match grows from zero as the matched-filter scalar earns its gradient.
            # This avoids the init noise killing the per-channel path before it stabilizes.
            nn.init.zeros_(self.W_match.weight)

    # ── Shared magnitude drive (identical to Selective) ──────────────────────
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

        w   = torch.clamp(v_gated * v_gated, max=LOG_COMPLEMENT_CLAMP)
        z_in = torch.log(1.0 - w + EPS)
        a   = alpha * z_in      # bounded log-complement drive
        return a, gamma

    def _magnitude(self, x):
        """m_t = √(1−exp z_t) ∈ [0,1) — byte-identical to Selective's state."""
        a, gamma = self._drive_and_gamma(x)
        Z = sequential_linear_scan(a, gamma)
        s_sq = torch.clamp(1.0 - torch.exp(Z), min=0.0)
        return torch.sqrt(s_sq + EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        self._rep_loss = None   # reset each forward pass

        # ── Ablation: exact GSSM-Selective ──────────────────────────────────
        if not self.use_phase:
            m = self._magnitude(x)
            return self.W_out(m.view(B, T, self.n_heads * self.d_head))

        # ── Key-conditioned holographic write ────────────────────────────────
        a, gamma = self._drive_and_gamma(x)

        # Per-channel phase angles: φ_t[c] = π·tanh(W_key x_t)[c]
        # Shape: (B, T, n_heads, d_head) — DISTINCT angle per channel.
        phi = self.phase_scale * torch.tanh(self.W_key(x))
        phi = phi.view(B, T, self.n_heads, self.d_head)     # (B,T,H,D)

        # Decompose complex write into two real scans:
        #   S_t = γ_t ⊙ S_{t-1} + a_t · e^{iφ_t}
        # where e^{iφ_t}[c] = (cos φ_t[c], sin φ_t[c]) per channel.
        drive_re = a * torch.cos(phi)   # (B,T,H,D)
        drive_im = a * torch.sin(phi)   # (B,T,H,D)

        if self.causal:
            S_re = sequential_linear_scan(drive_re, gamma)   # (B,T,H,D)
            S_im = sequential_linear_scan(drive_im, gamma)
        else:
            S_re = sequential_linear_scan(drive_re, gamma) + torch.flip(
                sequential_linear_scan(torch.flip(drive_re, [1]), torch.flip(gamma, [1])), [1])
            S_im = sequential_linear_scan(drive_im, gamma) + torch.flip(
                sequential_linear_scan(torch.flip(drive_im, [1]), torch.flip(gamma, [1])), [1])

        # ── Readout branch ───────────────────────────────────────────────────

        if self.baseline_1d:
            # OLD per-channel read: each channel is an independent 1-D holographic
            # memory; W_out mixes them.  This should reproduce the ~8-9% wall.
            # (Effective D=2: each channel has 1 angle, lives on the unit circle ℝ².)
            read_re = S_re * torch.cos(phi) + S_im * torch.sin(phi)  # (B,T,H,D)
            read_im = S_im * torch.cos(phi) - S_re * torch.sin(phi)
            # tanh_m readout (same as holographic_gssm default)
            m = self._magnitude(x)
            read_re = m * torch.tanh(read_re)
            read_im = m * torch.tanh(read_im)
            read_re = read_re.view(B, T, self.n_heads * self.d_head)
            read_im = read_im.view(B, T, self.n_heads * self.d_head)
            return self.W_out(read_re) + self.W_im(read_im)

        else:
            # ── VECTOR-KEY matched-filter read (Arm B / Arm C) ───────────────
            # Per-channel derotation (same as baseline_1d):
            read_re = S_re * torch.cos(phi) + S_im * torch.sin(phi)  # (B,T,H,D)
            read_im = S_im * torch.cos(phi) - S_re * torch.sin(phi)  # (B,T,H,D)

            # Matched-filter: collapse D channels → ONE scalar per head.
            #   read_match_h = (1/√D) Σ_c [S_re[c]·cos φ[c] + S_im[c]·sin φ[c]]
            #                = (1/√D) Re⟨S_t, e^{iφ_q}⟩
            # Crosstalk RMS = √((N−1)/D) (vs √(N−1) for per-channel, since
            # each mismatch cosine averages independently across D channels).
            read_match = read_re.sum(dim=-1) / math.sqrt(self.d_head)  # (B,T,H)

            # Two-term output:
            #   1. W_match: projects the H matched-filter scalars → d_model.
            #      This gives the D-vector-key discrimination signal full-rank gradients.
            #      The matched key has read_match_h ~ O(√D · a_k) (coherent sum),
            #      mismatched keys ~ O(1) (D independent cosines cancel).
            #   2. W_out + W_im: projects the full per-channel (B,T,H*D) derotated
            #      state → d_model, providing capacity + per-channel diversity.
            #      tanh_m gate same as baseline_1d (LOAD-BEARING m-gate preserved).
            m = self._magnitude(x)                                     # (B,T,H,D)
            read_re_gated = m * torch.tanh(read_re)                   # (B,T,H,D)
            read_im_gated = m * torch.tanh(read_im)                   # (B,T,H,D)

            out_perchan = (self.W_out(read_re_gated.view(B, T, self.n_heads * self.d_head))
                           + self.W_im(read_im_gated.view(B, T, self.n_heads * self.d_head)))

            # W_match path: no tanh saturation on the matched-filter scalar
            # (the discrimination is IN the magnitude of read_match, not its sign).
            out_match = self.W_match(read_match)                       # (B,T,d_model)
            out = out_perchan + out_match

            # ── β=3 repulsion loss (Arm C only) ─────────────────────────────
            if self.repulsion and self.training:
                # Collect the N distinct key vectors actually used this batch.
                # phi has shape (B,T,H,D); collapse B,T,H → flatten unique contexts.
                # For a practical estimate, sample a small set: use the first batch,
                # first head, all T positions.  (The loss only cares about the
                # geometry of the key CLOUD, not per-position structure.)
                phi_sample = phi[0, :, 0, :].detach()    # (T, D) — b=0, h=0
                # Deduplicate or just use all T positions (T=64, fine for N^2=4096)
                rep = ginibre_repulsion(phi_sample, margin=self.rep_margin)
                self._rep_loss = self.lambda_rep * rep

            return out

    def get_repulsion_loss(self) -> torch.Tensor:
        """Return the scalar repulsion loss from the last forward pass (or 0)."""
        if self._rep_loss is not None:
            return self._rep_loss
        return torch.tensor(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer layer + LM wrappers
# ─────────────────────────────────────────────────────────────────────────────

class GinibreHolographicTransformerLayer(nn.Module):
    """Post-LN block wrapping GinibreHolographicScanLayer."""

    def __init__(self, d_model: int, d_head: int = D_HEAD, n_heads: int = N_HEADS,
                 ffn_dim: int = None, dropout: float = 0.0, causal: bool = True,
                 phase_scale: float = math.pi, use_phase: bool = True,
                 repulsion: bool = False, lambda_rep: float = 0.03,
                 rep_margin: float = 1.0, baseline_1d: bool = False):
        super().__init__()
        self.scan = GinibreHolographicScanLayer(
            d_model, d_head=d_head, n_heads=n_heads, causal=causal,
            dropout=dropout, phase_scale=phase_scale, use_phase=use_phase,
            repulsion=repulsion, lambda_rep=lambda_rep, rep_margin=rep_margin,
            baseline_1d=baseline_1d)
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

    def get_repulsion_loss(self):
        return self.scan.get_repulsion_loss()


class GinibreHolographicLM(nn.Module):
    """Causal LM — same envelope as the other GSSM LMs.

    Extra attributes exposed for the training loop:
      .get_repulsion_loss() → sum of rep losses across all layers (for Arm C).
    """

    def __init__(self, vocab_size: int, mask_idx: int,
                 d_model: int = D_MODEL, n_layers: int = N_LAYERS,
                 n_heads: int = N_HEADS, d_head: int = D_HEAD,
                 seq_len: int = 64, dropout: float = 0.0, causal: bool = True,
                 phase_scale: float = math.pi, use_phase: bool = True,
                 repulsion: bool = False, lambda_rep: float = 0.03,
                 rep_margin: float = 1.0, baseline_1d: bool = False):
        super().__init__()
        from moebius_attention import SinusoidalPositionalEncoding
        self.mask_idx = mask_idx
        self.embed = nn.Embedding(vocab_size + 2, d_model)
        self.pos   = SinusoidalPositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            GinibreHolographicTransformerLayer(
                d_model, d_head=d_head, n_heads=n_heads, ffn_dim=4 * d_model,
                dropout=dropout, causal=causal, phase_scale=phase_scale,
                use_phase=use_phase, repulsion=repulsion, lambda_rep=lambda_rep,
                rep_margin=rep_margin, baseline_1d=baseline_1d)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size + 1)

    def forward(self, x):
        h = self.pos(self.embed(x))
        for layer in self.layers:
            h = layer(h)
        return self.head(h)

    def get_repulsion_loss(self) -> torch.Tensor:
        """Aggregate β=3 repulsion loss across all layers."""
        total = torch.tensor(0.0)
        for layer in self.layers:
            total = total + layer.get_repulsion_loss()
        return total


# ─────────────────────────────────────────────────────────────────────────────
# Reduction verification
# ─────────────────────────────────────────────────────────────────────────────

def _verify_reduction(device="cpu", tol=1e-5):
    """use_phase=False must equal Selective exactly on matching weights."""
    from moebius_scan_transformer_selective import SelectiveRapiditySqrtScanLayer
    torch.manual_seed(0)
    d_model, n_heads, d_head = 48, 4, 12
    gin  = GinibreHolographicScanLayer(d_model, d_head=d_head, n_heads=n_heads,
                                       use_phase=False).to(device).eval()
    sel  = SelectiveRapiditySqrtScanLayer(d_model, d_head=d_head, n_heads=n_heads,
                                          dropout=0.0).to(device).eval()
    with torch.no_grad():
        sel.W_v.weight.copy_(gin.W_v.weight)
        sel.W_gate.weight.copy_(gin.W_gate.weight)
        sel.W_gamma.weight.copy_(gin.W_gamma.weight)
        sel.W_alpha.weight.copy_(gin.W_alpha.weight)
        sel.W_out.weight.copy_(gin.W_out.weight)
    x   = torch.randn(3, 37, d_model, device=device)
    err = (gin(x) - sel(x)).abs().max().item()
    ok  = err < tol
    print(f"[reduction]  use_phase=False vs Selective  max|Δ| = {err:.3e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok, err


def _verify_baseline_1d_differs(device="cpu"):
    """baseline_1d=True must differ from the vector-key read (different mechanisms)."""
    torch.manual_seed(7)
    d_model, n_heads, d_head = 48, 4, 12
    vec  = GinibreHolographicScanLayer(d_model, d_head=d_head, n_heads=n_heads,
                                       use_phase=True, baseline_1d=False).to(device).eval()
    b1d  = GinibreHolographicScanLayer(d_model, d_head=d_head, n_heads=n_heads,
                                       use_phase=True, baseline_1d=True).to(device).eval()
    # Copy all shared weights so the only difference is the read path.
    with torch.no_grad():
        b1d.W_v.weight.copy_(vec.W_v.weight)
        b1d.W_gate.weight.copy_(vec.W_gate.weight)
        b1d.W_gamma.weight.copy_(vec.W_gamma.weight)
        b1d.W_alpha.weight.copy_(vec.W_alpha.weight)
        b1d.W_out.weight.copy_(vec.W_out.weight)
        b1d.W_key.weight.copy_(vec.W_key.weight)
    x = torch.randn(2, 37, d_model, device=device)
    diff = (vec(x) - b1d(x)).abs().max().item()
    ok   = diff > 1e-3    # they SHOULD differ
    print(f"[baseline_1d vs vec-key]  max|Δ| = {diff:.3e}  "
          f"{'DIFFER (correct)' if ok else 'IDENTICAL (BUG)'}")
    return ok


if __name__ == "__main__":
    print("=" * 74)
    print("Holographic-Ginibre Layer — self-test")
    print("=" * 74)

    ok_red, _ = _verify_reduction()
    ok_b1d    = _verify_baseline_1d_differs()

    # Sanity: vector-key mode produces finite bounded output.
    torch.manual_seed(42)
    layer = GinibreHolographicScanLayer(48, d_head=12, n_heads=4,
                                        use_phase=True, repulsion=False).eval()
    x = torch.randn(2, 40, 48)
    y = layer(x)
    print(f"[sanity vec-key]  finite={torch.isfinite(y).all().item()}  "
          f"shape={tuple(y.shape)}  std={y.std().item():.3f}  "
          f"range=[{y.min().item():.3f}, {y.max().item():.3f}]")

    # Sanity: repulsion loss computed in training mode.
    layer_rep = GinibreHolographicScanLayer(48, d_head=12, n_heads=4,
                                            use_phase=True, repulsion=True,
                                            lambda_rep=0.03).train()
    _ = layer_rep(x)
    rep_loss = layer_rep.get_repulsion_loss()
    print(f"[repulsion loss]  value={rep_loss.item():.5f}  "
          f"{'finite' if torch.isfinite(rep_loss) else 'NONFINITE (BUG)'}")

    # Sanity: key-cloud variance diagnostic.
    phi_test = torch.randn(8, 12) * math.pi    # 8 keys, D=12 channels
    kv = key_cloud_variance(phi_test)
    print(f"[key_cloud_variance]  random init ⟨s²⟩ = {kv:.4f}  "
          f"(target after training: ~1.087)")

    all_ok = ok_red and ok_b1d
    print(f"\n{'ALL SELF-TESTS PASSED' if all_ok else 'SOME SELF-TESTS FAILED'}")
    sys.exit(0 if all_ok else 1)
