"""
Holographic-GSSM — key-cloud ATTRACTION / target-spacing regularizer (entry 14).
================================================================================
Dead-end entry 14 closed the β=3 Ginibre REPULSION thread with a mechanism, and
named the one force never built:

    "⟨s²⟩ did NOT reach target 1.087 — it OVERSHOT it, monotone in λ: 1.605→1.746.
     The β=3 hinge repulsion drives keys PAST Ginibre spacing into lattice
     territory (>1.4), *away* from 1.087, not toward it.  Reaching 1.087 *from
     above* needs an ATTRACTION term, not stronger repulsion.  The repulsion lever
     cannot reach the spread-key regime by construction."

This file builds that attraction force.  It reuses the SAME per-channel key-phase
holographic write as src/holographic_gssm.py (NOT the D-vec matched-filter Ginibre
architecture, which entry 11 found inert; here the phases are the ordinary
per-channel φ that already carries the 8.89% effect) and adds a regularizer on the
key-cloud nearest-neighbour spacing with THREE modes:

    "attract"      : relu(s_norm − margin)³   — pulls neighbours TOGETHER when they
                     are TOO FAR (the mirror of repulsion). Tests: can attraction
                     bring ⟨s²⟩ DOWN to 1.087 from the lattice regime the repulsion
                     overshot into?  (Attraction alone → expect ⟨s²⟩ < 1: clustering.)
    "spacing"      : (s_norm − 1)²            — bilateral target-spacing: penalises
                     BOTH too-close and too-far, regulating ⟨s²⟩ toward 1 (Ginibre-
                     like), the regulator repulsion-alone cannot be.  THE key test.
    "repel_attract": relu(m_lo − s)³ + relu(s − m_hi)³ — double hinge with a DEAD
                     BAND [m_lo, m_hi]: repel inside m_lo, attract beyond m_hi, free
                     in between → settles the cloud in a target spacing window.

Gate is the SAME as the sweep (entry 14): read ⟨s²⟩ on real sequence phases; a mode
"wins" only if it (a) lands ⟨s²⟩ near 1.087 AND (b) beats same-run baseline_1d recall
by >1σ.  If ⟨s²⟩ hits 1.087 but recall does NOT beat baseline → the spread-key regime
is reachable but USELESS (closes entry-14 from the other side).

use_phase=False → exact GSSM-Selective (reduction preserved).  MPS-safe real ops.
Reference: Foss 2026; RECALL_DEADENDS_LOG entry 14.
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

from holographic_gssm import HolographicScanLayer, sequential_linear_scan  # noqa: E402
from holographic_ginibre import key_cloud_variance  # noqa: E402 (reuse the ⟨s²⟩ instrument)

D_MODEL = 128
N_HEADS = 4
D_HEAD = 32
N_LAYERS = 2
EPS = 1e-6


def _pairwise_snorm(phi: torch.Tensor, eps: float = 1e-6):
    """phi (M, D) phase angles → normalised pairwise chordal distances s_norm (upper tri)."""
    cos_e = torch.cos(phi)
    sin_e = torch.sin(phi)
    M, D = phi.shape
    G = (cos_e @ cos_e.T + sin_e @ sin_e.T) / D
    d2 = (2.0 - 2.0 * G).clamp(min=0.0)
    iu = torch.triu_indices(M, M, offset=1, device=phi.device)
    s = d2[iu[0], iu[1]].clamp(min=eps).sqrt()
    s_mean = s.mean().clamp(min=eps)
    return s / s_mean


def spacing_loss(phi: torch.Tensor, mode: str = "spacing",
                 margin: float = 1.0, m_lo: float = 0.8, m_hi: float = 1.2,
                 eps: float = 1e-6) -> torch.Tensor:
    """Key-cloud spacing regularizer. phi: (M, D) phase angles. Scalar loss."""
    s = _pairwise_snorm(phi, eps)
    if mode == "attract":
        return torch.relu(s - margin).pow(3).mean()
    if mode == "spacing":
        return (s - 1.0).pow(2).mean()
    if mode == "repel_attract":
        return (torch.relu(m_lo - s).pow(3).mean()
                + torch.relu(s - m_hi).pow(3).mean())
    raise ValueError(mode)


class AttractionHolographicScanLayer(HolographicScanLayer):
    """Ordinary per-channel holographic scan (inherits the 8.89% mechanism) that
    ALSO exposes the write-key phases so a spacing loss can be applied.

    attract_mode ∈ {"attract","spacing","repel_attract"} + lambda_attr weight.
    The loss from the last forward is stored in self._attr_loss and read by the
    training loop, exactly like Ginibre's get_repulsion_loss pattern.
    """

    def __init__(self, *args, attract_mode: str = "spacing", lambda_attr: float = 0.1,
                 attr_margin: float = 1.0, attr_lo: float = 0.8, attr_hi: float = 1.2,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.attract_mode = attract_mode
        self.lambda_attr = lambda_attr
        self.attr_margin = attr_margin
        self.attr_lo = attr_lo
        self.attr_hi = attr_hi
        self._attr_loss = None
        self._last_s2 = None

    def forward(self, x):
        self._attr_loss = None
        out = super().forward(x)
        if self.use_phase and self.training:
            # recompute the write-key phases (cheap) for the spacing loss + ⟨s²⟩.
            phi = self.phase_scale * torch.tanh(self.W_key(x))     # (B,T,total)
            B, T, _ = phi.shape
            phi = phi.view(B, T, self.n_heads, self.d_head)
            # take one head, all positions of the first batch item as the key cloud
            phi_sample = phi[0, :, 0, :]                            # (T, d_head)
            self._attr_loss = self.lambda_attr * spacing_loss(
                phi_sample, mode=self.attract_mode, margin=self.attr_margin,
                m_lo=self.attr_lo, m_hi=self.attr_hi)
            self._last_s2 = key_cloud_variance(phi_sample)
        return out

    def get_attraction_loss(self):
        return self._attr_loss if self._attr_loss is not None else torch.tensor(0.0)


class AttractionHolographicTransformerLayer(nn.Module):
    def __init__(self, d_model, d_head=D_HEAD, n_heads=N_HEADS, ffn_dim=None,
                 dropout=0.0, causal=True, phase_scale=math.pi, use_phase=True,
                 readout="tanh_m", attract_mode="spacing", lambda_attr=0.1,
                 attr_margin=1.0, attr_lo=0.8, attr_hi=1.2):
        super().__init__()
        self.scan = AttractionHolographicScanLayer(
            d_model, d_head=d_head, n_heads=n_heads, causal=causal, dropout=dropout,
            phase_scale=phase_scale, use_phase=use_phase, readout=readout,
            attract_mode=attract_mode, lambda_attr=lambda_attr,
            attr_margin=attr_margin, attr_lo=attr_lo, attr_hi=attr_hi)
        self.ln1 = nn.LayerNorm(d_model)
        ffn_dim = ffn_dim or 4 * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(),
            nn.Identity(),
            nn.Linear(ffn_dim, d_model))
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.ln1(x + self.scan(x))
        x = self.ln2(x + self.ffn(x))
        return x

    def get_attraction_loss(self):
        return self.scan.get_attraction_loss()


class AttractionHolographicLM(nn.Module):
    def __init__(self, vocab_size, mask_idx, d_model=D_MODEL, n_layers=N_LAYERS,
                 n_heads=N_HEADS, d_head=D_HEAD, seq_len=64, dropout=0.0, causal=True,
                 phase_scale=math.pi, use_phase=True, readout="tanh_m",
                 attract_mode="spacing", lambda_attr=0.1, attr_margin=1.0,
                 attr_lo=0.8, attr_hi=1.2):
        super().__init__()
        from moebius_attention import SinusoidalPositionalEncoding
        self.mask_idx = mask_idx
        self.embed = nn.Embedding(vocab_size + 2, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            AttractionHolographicTransformerLayer(
                d_model, d_head=d_head, n_heads=n_heads, ffn_dim=4 * d_model,
                dropout=dropout, causal=causal, phase_scale=phase_scale,
                use_phase=use_phase, readout=readout, attract_mode=attract_mode,
                lambda_attr=lambda_attr, attr_margin=attr_margin,
                attr_lo=attr_lo, attr_hi=attr_hi)
            for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab_size + 1)

    def forward(self, x):
        h = self.pos(self.embed(x))
        for l in self.layers:
            h = l(h)
        return self.head(h)

    def get_attraction_loss(self):
        total = torch.tensor(0.0)
        for l in self.layers:
            al = l.get_attraction_loss()
            total = total + al
        return total

    def last_s2(self):
        vals = [l.scan._last_s2 for l in self.layers if l.scan._last_s2 is not None]
        return sum(vals) / len(vals) if vals else None


def _verify_reduction(device="cpu", tol=1e-5):
    """use_phase=False must equal GSSM-Selective (inherited mechanism intact)."""
    from moebius_scan_transformer_selective import SelectiveRapiditySqrtScanLayer
    torch.manual_seed(0)
    d_model, n_heads, d_head = 48, 4, 12
    holo = AttractionHolographicScanLayer(d_model, d_head=d_head, n_heads=n_heads,
                                          use_phase=False).to(device).eval()
    sel = SelectiveRapiditySqrtScanLayer(d_model, d_head=d_head, n_heads=n_heads,
                                         dropout=0.0).to(device).eval()
    with torch.no_grad():
        sel.W_v.weight.copy_(holo.W_v.weight)
        sel.W_gate.weight.copy_(holo.W_gate.weight)
        sel.W_gamma.weight.copy_(holo.W_gamma.weight)
        sel.W_alpha.weight.copy_(holo.W_alpha.weight)
        sel.W_out.weight.copy_(holo.W_out.weight)
    x = torch.randn(3, 37, d_model, device=device)
    err = (holo(x) - sel(x)).abs().max().item()
    ok = err < tol
    print(f"[reduction] use_phase=False vs Selective  max|Δ|={err:.3e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok, err


if __name__ == "__main__":
    print("=" * 74)
    print("Holographic key-cloud ATTRACTION / spacing regularizer — entry 14")
    print("=" * 74)
    ok, _ = _verify_reduction()
    # sanity: each mode produces a finite loss and a finite ⟨s²⟩
    torch.manual_seed(1)
    for mode in ("attract", "spacing", "repel_attract"):
        layer = AttractionHolographicScanLayer(
            48, d_head=12, n_heads=4, use_phase=True, readout="tanh_m",
            attract_mode=mode, lambda_attr=0.1).train()
        x = torch.randn(2, 40, 48)
        y = layer(x)
        al = layer.get_attraction_loss()
        print(f"[sanity] mode={mode:14s} out.std={y.std().item():.3f} "
              f"attr_loss={al.item():.4f} ⟨s²⟩={layer._last_s2:.4f}")
    sys.exit(0 if ok else 1)
