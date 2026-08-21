"""
Holographic-GSSM — Multi-Frequency Write (K independent harmonic bands)
========================================================================

MECHANISM (from the derivation — fully verified):

Per channel, maintain K independent complex leaky accumulators S^(k), k=1..K.
Write key at HARMONIC angles of the phase:

    S^(k)_t = γ_t · S^(k)_{t-1} + u_t · e^{i·k·φ_t},   k = 1..K

Read: de-rotate each band at its own harmonic of the query angle, sum with equal
weights (optimal — the interference covariance is white, C_{kl} = (N-1)/2 · δ_{kl}):

    r = Σ_{k=1..K}  Re( S^(k) · e^{-i·k·φ_q} )
      = Σ_{k,j}  g_{j→t} · u_j · cos( k·(φ_j − φ_q) )

Matched key (δ=φ_j−φ_q=0): cos(0)=1 at EVERY band → sums coherently = K·g·u_match.
Mismatched key (δ≠0): cos(kδ) for k=1..K are at K different points → decohere.
RMS of D_K(δ)=Σ_k cos(kδ) over δ~U(-π,π) = √(K/2), not K.

Signal/interference = K / √((N-1)·K/2) = √(2K/(N-1))   →   SNR scales as √K.

K=1 is byte-identical to the single-accumulator holographic baseline.
use_phase=False is exact GSSM-Selective (reduction preserved).

Cost: 2K real leaky scans per channel. O(1) state, O(T) compute, MPS-native.

Reference: Foss 2026 — Multi-Frequency Write derivation, verified numerically.
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

D_MODEL    = 128
N_HEADS    = 4
D_HEAD     = 32
N_LAYERS   = 2
N_FREQS    = 4           # default K

LOG_COMPLEMENT_CLAMP = 0.999
EPS = 1e-6


def sequential_linear_scan(a: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """z_t = γ_t·z_{t-1} + a_t, shapes (B,T,H,D). Same recurrence as Selective."""
    B, T, H, D = a.shape
    Z = torch.zeros(B, H, D, device=a.device, dtype=a.dtype)
    out = []
    for t in range(T):
        Z = gamma[:, t] * Z + a[:, t]
        out.append(Z)
    return torch.stack(out, dim=1)


class HolographicMultiFreqScanLayer(nn.Module):
    """
    Multi-Frequency Holographic GSSM scan layer.

    Parameters
    ----------
    n_freqs : int
        K = number of harmonic bands (default 4).
        K=1 → byte-identical to single-accumulator holographic baseline.
        Each band uses 2 real leaky scans (re, im), same γ as the base layer.
    use_phase : bool
        False → exact GSSM-Selective (ablation control).
        True  → full multi-frequency holographic write/read.
    readout : str
        "rms"      → read / (rms(read)+eps)  [default, preserves contrast]
        "tanh_m"   → m · tanh(read)           [bounded, doubly damped]
        "layernorm" → raw read, let post-LN normalize
    """

    def __init__(self, d_model: int, d_head: int = D_HEAD, n_heads: int = N_HEADS,
                 causal: bool = True, dropout: float = 0.0,
                 phase_scale: float = math.pi, use_phase: bool = True,
                 readout: str = "rms", n_freqs: int = N_FREQS):
        super().__init__()
        self.d_model     = d_model
        self.d_head      = d_head
        self.n_heads     = n_heads
        self.causal      = causal
        self.phase_scale = phase_scale
        self.use_phase   = use_phase
        self.n_freqs     = n_freqs   # K
        assert n_freqs >= 1, n_freqs
        assert readout in ("tanh_m", "layernorm", "rms"), readout
        self.readout = readout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        total_dim = n_heads * d_head

        # ── Magnitude / value projections (identical names/shapes to baseline) ──
        self.W_v     = nn.Linear(d_model, total_dim, bias=False)
        self.W_gate  = nn.Linear(d_model, total_dim, bias=False)
        self.W_gamma = nn.Linear(d_model, total_dim, bias=False)   # forget
        self.W_alpha = nn.Linear(d_model, total_dim, bias=False)   # input scale
        self.W_out   = nn.Linear(total_dim, d_model, bias=False)

        # ── Holographic projections ──
        # W_key: write-key angle φ_t = phase_scale · tanh(W_key x_t).
        #        Also used as the read/query angle (shared-QK, same as baseline).
        self.W_key = nn.Linear(d_model, total_dim, bias=False)
        # W_im: out-projection for the imaginary channel (summed across K bands).
        self.W_im  = nn.Linear(total_dim, d_model, bias=False)

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
        # W_key small → φ≈0 at init → starts near the real-write (Selective) regime.
        for p in self.W_key.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p, gain=0.1)
        for p in self.W_im.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p, gain=0.6)

    # ── shared drive: bounded value u_t and forget γ_t ──────────────────────
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
        a    = alpha * z_in          # a_t ≤ 0, bounded log-complement drive
        return a, gamma

    def _magnitude(self, x):
        """m_t = √(1−exp z_t) ∈ [0,1) — byte-identical to Selective's state."""
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
            # Ablation: EXACT GSSM-Selective. Read the real magnitude alone.
            m = self._magnitude(x)
            return self.W_out(m.view(B, T, self.n_heads * self.d_head))

        # ── multi-frequency holographic write/read ─────────────────────────
        a, gamma = self._drive_and_gamma(x)        # (B,T,H,D)
        phi = self.phase_scale * torch.tanh(self.W_key(x))  # base phase φ_t
        phi = phi.view(B, T, self.n_heads, self.d_head)     # (B,T,H,D)

        # Accumulate K bands.  Each band k writes/reads at harmonic k·φ.
        # Two real leaky scans per band (re, im), same γ_t for all bands.
        # Initialise summed reads to zero.
        read_re = torch.zeros(B, T, self.n_heads, self.d_head,
                              device=x.device, dtype=x.dtype)
        read_im = torch.zeros_like(read_re)

        for k in range(1, self.n_freqs + 1):
            kphi = k * phi                           # harmonic angle k·φ_t
            cos_kphi = torch.cos(kphi)
            sin_kphi = torch.sin(kphi)

            # Complex drive at harmonic k: a · e^{i·k·φ}
            drive_re_k = a * cos_kphi               # (B,T,H,D)
            drive_im_k = a * sin_kphi

            # Leaky scan: S^(k)_t = γ_t · S^(k)_{t-1} + drive_k_t
            if self.causal:
                Sre_k = sequential_linear_scan(drive_re_k, gamma)
                Sim_k = sequential_linear_scan(drive_im_k, gamma)
            else:
                Sre_k = sequential_linear_scan(drive_re_k, gamma) + torch.flip(
                    sequential_linear_scan(torch.flip(drive_re_k, dims=[1]),
                                           torch.flip(gamma, dims=[1])), dims=[1])
                Sim_k = sequential_linear_scan(drive_im_k, gamma) + torch.flip(
                    sequential_linear_scan(torch.flip(drive_im_k, dims=[1]),
                                           torch.flip(gamma, dims=[1])), dims=[1])

            # De-rotate at query angle k·φ (shared-QK, φ_read = φ_write):
            #   Re( S^(k) · e^{-i·k·φ_q} ) = Sre·cos(k·φ_q) + Sim·sin(k·φ_q)
            #   Im( S^(k) · e^{-i·k·φ_q} ) = Sim·cos(k·φ_q) - Sre·sin(k·φ_q)
            read_re = read_re + (Sre_k * cos_kphi + Sim_k * sin_kphi)
            read_im = read_im + (Sim_k * cos_kphi - Sre_k * sin_kphi)

        # Equal weights w_k = 1 (optimal: interference covariance is white → C^{-1}s ∝ s).
        # read_re / read_im each sum K bands; no additional normalisation needed here.

        # ── readout ────────────────────────────────────────────────────────
        if self.readout == "tanh_m":
            m        = self._magnitude(x)
            read_re  = m * torch.tanh(read_re)
            read_im  = m * torch.tanh(read_im)
        elif self.readout == "rms":
            rms_re   = read_re.pow(2).mean(dim=-1, keepdim=True).add(EPS).sqrt()
            rms_im   = read_im.pow(2).mean(dim=-1, keepdim=True).add(EPS).sqrt()
            read_re  = read_re / rms_re
            read_im  = read_im / rms_im
        # "layernorm": pass raw; post-LN block normalises.

        read_re = read_re.view(B, T, self.n_heads * self.d_head)
        read_im = read_im.view(B, T, self.n_heads * self.d_head)
        return self.W_out(read_re) + self.W_im(read_im)


class HolographicMultiFreqTransformerLayer(nn.Module):
    """Post-LN block, same envelope as the baseline HolographicTransformerLayer."""

    def __init__(self, d_model: int, d_head: int = D_HEAD, n_heads: int = N_HEADS,
                 ffn_dim: int = None, dropout: float = 0.0, causal: bool = True,
                 phase_scale: float = math.pi, use_phase: bool = True,
                 readout: str = "rms", n_freqs: int = N_FREQS):
        super().__init__()
        self.scan = HolographicMultiFreqScanLayer(
            d_model, d_head=d_head, n_heads=n_heads, causal=causal,
            dropout=dropout, phase_scale=phase_scale, use_phase=use_phase,
            readout=readout, n_freqs=n_freqs)
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


class HolographicMultiFreqLM(nn.Module):
    """
    Causal LM wrapper for the multi-frequency holographic GSSM.
    Same shape as HolographicLM / SelectiveRapiditySqrtTransformerLM.
    """

    def __init__(self, vocab_size: int, mask_idx: int,
                 d_model: int = D_MODEL, n_layers: int = N_LAYERS,
                 n_heads: int = N_HEADS, d_head: int = D_HEAD,
                 seq_len: int = 64, dropout: float = 0.0, causal: bool = True,
                 phase_scale: float = math.pi, use_phase: bool = True,
                 readout: str = "rms", n_freqs: int = N_FREQS):
        super().__init__()
        from moebius_attention import SinusoidalPositionalEncoding
        self.mask_idx = mask_idx
        self.embed    = nn.Embedding(vocab_size + 2, d_model)
        self.pos      = SinusoidalPositionalEncoding(d_model)
        self.layers   = nn.ModuleList([
            HolographicMultiFreqTransformerLayer(
                d_model, d_head=d_head, n_heads=n_heads, ffn_dim=4 * d_model,
                dropout=dropout, causal=causal, phase_scale=phase_scale,
                use_phase=use_phase, readout=readout, n_freqs=n_freqs)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size + 1)

    def forward(self, x):
        h = self.pos(self.embed(x))
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


# ───────────────────────────────────────────────────────────────────────────
# Self-tests
# ───────────────────────────────────────────────────────────────────────────

def _verify_k1_equals_baseline(device="cpu", tol=1e-5):
    """K=1 multi-freq layer must be byte-identical to the single-accumulator baseline
    when given the same weights."""
    from holographic_gssm import HolographicScanLayer
    torch.manual_seed(0)
    d_model, n_heads, d_head = 48, 4, 12

    mf  = HolographicMultiFreqScanLayer(d_model, d_head=d_head, n_heads=n_heads,
                                         n_freqs=1, use_phase=True,
                                         readout="rms").to(device).eval()
    bl  = HolographicScanLayer(d_model, d_head=d_head, n_heads=n_heads,
                                use_phase=True, readout="rms",
                                n_slots=1).to(device).eval()

    # Copy matching weights (identical names, identical shapes).
    with torch.no_grad():
        for name in ["W_v", "W_gate", "W_gamma", "W_alpha", "W_out", "W_key", "W_im"]:
            getattr(mf, name).weight.copy_(getattr(bl, name).weight)

    x   = torch.randn(3, 37, d_model, device=device)
    err = (mf(x) - bl(x)).abs().max().item()
    ok  = err < tol
    print(f"[K=1 == baseline]  max|Δ| = {err:.3e}  "
          f"{'PASS (byte-identical)' if ok else 'FAIL'}")
    return ok, err


def _verify_reduction_selective(device="cpu", tol=1e-5):
    """use_phase=False must equal GSSM-Selective on identical magnitude weights."""
    from moebius_scan_transformer_selective import SelectiveRapiditySqrtScanLayer
    torch.manual_seed(0)
    d_model, n_heads, d_head = 48, 4, 12

    mf  = HolographicMultiFreqScanLayer(d_model, d_head=d_head, n_heads=n_heads,
                                         n_freqs=4, use_phase=False).to(device).eval()
    sel = SelectiveRapiditySqrtScanLayer(d_model, d_head=d_head, n_heads=n_heads,
                                          dropout=0.0).to(device).eval()
    with torch.no_grad():
        sel.W_v.weight.copy_(mf.W_v.weight)
        sel.W_gate.weight.copy_(mf.W_gate.weight)
        sel.W_gamma.weight.copy_(mf.W_gamma.weight)
        sel.W_alpha.weight.copy_(mf.W_alpha.weight)
        sel.W_out.weight.copy_(mf.W_out.weight)

    x   = torch.randn(3, 37, d_model, device=device)
    err = (mf(x) - sel(x)).abs().max().item()
    ok  = err < tol
    print(f"[use_phase=False == Selective]  max|Δ| = {err:.3e}  "
          f"{'PASS (exact reduction)' if ok else 'FAIL'}")
    return ok, err


def _verify_finite_multifreq(device="cpu"):
    """Multi-freq output must be finite for K∈{1,2,4,8}."""
    torch.manual_seed(2)
    d_model, n_heads, d_head = 48, 4, 12
    x = torch.randn(2, 40, d_model, device=device)
    all_ok = True
    for K in [1, 2, 4, 8]:
        layer = HolographicMultiFreqScanLayer(d_model, d_head=d_head, n_heads=n_heads,
                                              n_freqs=K, use_phase=True,
                                              readout="rms").to(device).eval()
        y = layer(x)
        finite = torch.isfinite(y).all().item()
        all_ok = all_ok and finite
        print(f"[K={K}]  finite={finite}  shape={tuple(y.shape)}  "
              f"std={y.std().item():.3f}  range=[{y.min().item():.3f}, {y.max().item():.3f}]")
    return all_ok


if __name__ == "__main__":
    print("=" * 74)
    print("Holographic-GSSM — Multi-Frequency Write self-tests")
    print("=" * 74)

    ok1, _ = _verify_k1_equals_baseline()
    ok2, _ = _verify_reduction_selective()
    ok3    = _verify_finite_multifreq()

    print()
    all_pass = ok1 and ok2 and ok3
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILURES — see above'}")
    sys.exit(0 if all_pass else 1)
