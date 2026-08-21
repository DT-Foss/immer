"""
Phase-GSSM with SELECTABLE readout — the fair re-test of the phase channel.
===========================================================================
CONFOUND (ONSET campaign lever 4).  results/phase_mqar_capacity.json ran the
additive-phase Phase-GSSM (src/phase_gssm.py) with its NATIVE readout
`out = W_out(r_re) + W_im(r_im)` where r_re=m·cosΘ, r_im=m·sinΘ — a RAW,
un-normalised read — at 1500 steps (capacity json) AND 3000 steps (REPRO json).
In BOTH, selective / phase_false / phase_true came out BYTE-IDENTICAL at ~1.3%
(chance).  The standing readout-crossover rule says: a raw/rms read is under-
trained and collapses; tanh_m carries the effect.  The phase channel was
therefore never given the readout that lets the holographic effect show.

This module keeps the EXACT additive-phase magnitude+phase math of
src/phase_gssm.py (Θ_t = cumsum ω_t; NOT edited — this is a separate file) but
adds the same readout menu the holographic line uses, so phase_true vs
phase_false is measured under a tanh_m-equivalent read at 2500 steps.

NOTE ON WHAT THIS DOES AND DOES NOT TEST.  Entry 1 of the dead-end log already
KILLED the additive-phase mechanism on a *mechanistic* ground: "phase rotates
blindly with TIME, not key identity → all values in one shared rotating bin."
This re-run does NOT resurrect that mechanism — it only removes the READOUT
confound so the null is clean: if phase_true == phase_false even under tanh_m at
2500 steps, the additive-phase channel is confirmed null NOT because of an
under-trained readout but because the mechanism carries no key-separable signal.

readout ∈ {"native", "tanh_m", "rms"}:
    "native"   : W_out(m·cosΘ) + W_im(m·sinΘ)          (the confounded original)
    "tanh_m"   : W_out(m·tanh(cosΘ')) + W_im(m·tanh(sinΘ'))  where the read is the
                 de-rotation-free real/imag parts m·cosΘ, m·sinΘ passed through
                 m·tanh (matching the holographic tanh_m envelope)
    "rms"      : rms-normalise (m·cosΘ, m·sinΘ) over the channel dim then read.

use_phase=False → exact GSSM-Selective (reduction preserved, byte-identical).
MPS-safe real ops only.  Reference: Foss 2026; RECALL_DEADENDS_LOG entry 1.
"""

import sys
import math
from pathlib import Path

import torch
import torch.nn as nn

REF = Path(__file__).resolve().parent.parent / "reference"
sys.path.insert(0, str(REF))

from moebius_attention import SinusoidalPositionalEncoding  # noqa: E402

LOG_COMPLEMENT_CLAMP = 0.999
EPS = 1e-6

D_MODEL = 128
N_HEADS = 4
D_HEAD = 32
N_LAYERS = 2


def sequential_linear_scan(a: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    B, T, H, D = a.shape
    Z = torch.zeros(B, H, D, device=a.device, dtype=a.dtype)
    out = []
    for t in range(T):
        Z = gamma[:, t] * Z + a[:, t]
        out.append(Z)
    return torch.stack(out, dim=1)


class PhaseReadoutScanLayer(nn.Module):
    def __init__(self, d_model: int, d_head: int = D_HEAD, n_heads: int = N_HEADS,
                 causal: bool = True, dropout: float = 0.0,
                 omega_scale: float = math.pi, use_phase: bool = True,
                 readout: str = "tanh_m"):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads
        self.causal = causal
        self.omega_scale = omega_scale
        self.use_phase = use_phase
        assert readout in ("native", "tanh_m", "rms"), readout
        self.readout = readout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        total_dim = n_heads * d_head
        self.W_v = nn.Linear(d_model, total_dim, bias=False)
        self.W_gate = nn.Linear(d_model, total_dim, bias=False)
        self.W_gamma = nn.Linear(d_model, total_dim, bias=False)
        self.W_alpha = nn.Linear(d_model, total_dim, bias=False)
        self.W_out = nn.Linear(total_dim, d_model, bias=False)
        self.W_omega = nn.Linear(d_model, total_dim, bias=False)
        self.W_im = nn.Linear(total_dim, d_model, bias=False)
        self._reset_parameters()

    def _reset_parameters(self):
        for module in [self.W_gamma, self.W_alpha, self.W_omega]:
            for p in module.parameters():
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p, gain=0.1)
        for module in [self.W_v, self.W_gate, self.W_out, self.W_im]:
            for p in module.parameters():
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p, gain=0.6)

    def _magnitude(self, x):
        B, T, _ = x.shape
        v = torch.tanh(self.W_v(x))
        gate = torch.sigmoid(self.W_gate(x))
        gamma = torch.sigmoid(self.W_gamma(x))
        alpha = torch.sigmoid(self.W_alpha(x))
        v_gated = v * gate
        if self.dropout is not None:
            v_gated = self.dropout(v_gated)
        v_gated = v_gated.view(B, T, self.n_heads, self.d_head)
        gamma = gamma.view(B, T, self.n_heads, self.d_head)
        alpha = alpha.view(B, T, self.n_heads, self.d_head)
        w = torch.clamp(v_gated * v_gated, max=LOG_COMPLEMENT_CLAMP)
        z_in = torch.log(1.0 - w + EPS)
        a = alpha * z_in
        if self.causal:
            Z = sequential_linear_scan(a, gamma)
        else:
            Z_fwd = sequential_linear_scan(a, gamma)
            Z_rev = torch.flip(sequential_linear_scan(
                torch.flip(a, dims=[1]), torch.flip(gamma, dims=[1])), dims=[1])
            Z = Z_fwd + Z_rev
        s_sq = torch.clamp(1.0 - torch.exp(Z), min=0.0)
        return torch.sqrt(s_sq + EPS)

    def _phase(self, x):
        B, T, _ = x.shape
        omega = torch.tanh(self.W_omega(x)) * self.omega_scale
        omega = omega.view(B, T, self.n_heads, self.d_head)
        if self.causal:
            Theta = torch.cumsum(omega, dim=1)
        else:
            Theta_fwd = torch.cumsum(omega, dim=1)
            Theta_rev = torch.flip(torch.cumsum(torch.flip(omega, dims=[1]), dim=1), dims=[1])
            Theta = Theta_fwd + Theta_rev
        return Theta

    def _rms_norm(self, r):
        return r / r.pow(2).mean(dim=-1, keepdim=True).add(EPS).sqrt()

    def forward(self, x):
        B, T, _ = x.shape
        m = self._magnitude(x)
        if not self.use_phase:
            return self.W_out(m.view(B, T, self.n_heads * self.d_head))
        Theta = self._phase(x)
        r_re = m * torch.cos(Theta)
        r_im = m * torch.sin(Theta)
        if self.readout == "tanh_m":
            r_re = m * torch.tanh(r_re)
            r_im = m * torch.tanh(r_im)
        elif self.readout == "rms":
            r_re = self._rms_norm(r_re)
            r_im = self._rms_norm(r_im)
        # "native": raw m·cosΘ, m·sinΘ (the confounded original)
        r_re = r_re.view(B, T, self.n_heads * self.d_head)
        r_im = r_im.view(B, T, self.n_heads * self.d_head)
        return self.W_out(r_re) + self.W_im(r_im)


class PhaseReadoutTransformerLayer(nn.Module):
    def __init__(self, d_model, d_head=D_HEAD, n_heads=N_HEADS, ffn_dim=None,
                 dropout=0.0, causal=True, omega_scale=math.pi, use_phase=True,
                 readout="tanh_m"):
        super().__init__()
        self.scan = PhaseReadoutScanLayer(
            d_model, d_head=d_head, n_heads=n_heads, causal=causal,
            dropout=dropout, omega_scale=omega_scale, use_phase=use_phase,
            readout=readout)
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


class PhaseReadoutLM(nn.Module):
    def __init__(self, vocab_size, mask_idx, d_model=D_MODEL, n_layers=N_LAYERS,
                 n_heads=N_HEADS, d_head=D_HEAD, seq_len=64, dropout=0.0,
                 causal=True, omega_scale=math.pi, use_phase=True, readout="tanh_m"):
        super().__init__()
        self.mask_idx = mask_idx
        self.embed = nn.Embedding(vocab_size + 2, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            PhaseReadoutTransformerLayer(
                d_model, d_head=d_head, n_heads=n_heads, ffn_dim=4 * d_model,
                dropout=dropout, causal=causal, omega_scale=omega_scale,
                use_phase=use_phase, readout=readout)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size + 1)

    def forward(self, x):
        h = self.pos(self.embed(x))
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


def _verify_reduction(device="cpu", tol=1e-4):
    """use_phase=False must equal GSSM-Selective."""
    from moebius_scan_transformer_selective import SelectiveRapiditySqrtTransformerLM
    torch.manual_seed(123)
    vocab, mask = 200, 201
    phase_off = PhaseReadoutLM(vocab, mask, seq_len=16, use_phase=False,
                               readout="tanh_m").to(device)
    torch.manual_seed(123)
    sel = SelectiveRapiditySqrtTransformerLM(vocab, mask, d_model=D_MODEL,
                                             n_layers=N_LAYERS, n_heads=N_HEADS,
                                             d_head=D_HEAD, seq_len=16, dropout=0.0,
                                             causal=True).to(device)
    with torch.no_grad():
        sd_sel = sel.state_dict()
        sd_ph = phase_off.state_dict()
        copied = 0
        for k, v in sd_sel.items():
            if k in sd_ph and sd_ph[k].shape == v.shape:
                sd_ph[k].copy_(v); copied += 1
        phase_off.load_state_dict(sd_ph)
    phase_off.eval(); sel.eval()
    x = torch.randint(0, vocab, (4, 16), device=device)
    with torch.no_grad():
        err = (phase_off(x) - sel(x)).abs().max().item()
    ok = err < tol
    print(f"[reduction] use_phase=False vs Selective  max|Δ|={err:.3e} copied={copied} "
          f"{'PASS' if ok else 'FAIL'}")
    return ok, err


if __name__ == "__main__":
    print("=" * 74)
    print("Phase-GSSM with selectable readout — fair re-test of the phase channel")
    print("=" * 74)
    ok, _ = _verify_reduction()
    torch.manual_seed(1)
    for ro in ("native", "tanh_m", "rms"):
        layer = PhaseReadoutScanLayer(48, d_head=12, n_heads=4, use_phase=True,
                                      readout=ro).eval()
        with torch.no_grad():
            y = layer(torch.randn(2, 40, 48))
        print(f"[sanity] readout={ro:8s} finite={torch.isfinite(y).all().item()} "
              f"std={y.std().item():.3f}")
    sys.exit(0 if ok else 1)
