#!/usr/bin/env python3 -u
"""
Phase-GSSM — complex/phase extension of GSSM-Selective — by Opus 4.8
====================================================================

Phase-GSSM keeps the proven GSSM-Selective MAGNITUDE channel byte-for-byte
identical and adds an ORTHOGONAL phase channel.  The scalar state becomes a
bounded complex number  r_t = m_t · e^{iΘ_t}  carried as TWO REAL TENSORS
(r_re, r_im) — there is NO torch.complex anywhere (cfloat/cdouble silently
fall back to CPU on MPS, so they are forbidden).

    MAGNITUDE  (identical to Selective):
        v_t      = tanh(W_v x_t)
        gate_t   = sigmoid(W_gate x_t)
        γ_t      = sigmoid(W_gamma x_t)        forget gate
        α_t      = sigmoid(W_alpha x_t)        input  gate
        v_gated  = v_t · gate_t
        a_t      = α_t · log(1 − v_gated²)      ≤ 0   (clamp + EPS as in ref)
        z_t      = γ_t · z_{t-1} + a_t          ≤ 0   (sequential linear scan)
        m_t      = sqrt(1 − exp(z_t))           ∈ [0,1)   ← exactly Selective's s_t

    PHASE  (new — plain additive prefix-sum, an undamped/ungated integrator):
        ω_t      = tanh(W_omega x_t) · ω_scale
        Θ_t      = Θ_{t-1} + ω_t = Σ_{k≤t} ω_k   (torch.cumsum, the γ≡1 case)

    COMPLEX STATE  (two real tensors, pointwise AFTER both scans):
        r_re_t   = m_t · cos Θ_t
        r_im_t   = m_t · sin Θ_t

    READOUT  (Option A — clean ablation):
        out_t    = W_out(r_re_t) + W_im(r_im_t)

θ≡0 REDUCTION (the use_phase=False ablation):
    Setting ω_t ≡ 0 ∀t ⟹ Θ_t = Σ 0 = 0 ⟹ cos Θ_t = 1, sin Θ_t = 0
    ⟹ r_re_t = m_t·1 = m_t  and  r_im_t = m_t·0 = 0.
    The imaginary readout vanishes (W_im(0)=0, no bias), so
        out_t = W_out(m_t) = W_out(s_t)  — BIT-IDENTICAL to GSSM-Selective.
    cos 0 / sin 0 are exact in floating point, so the reduction needs no
    tolerance and W_out keeps the same parameter slot/shape as Selective.
    Phase-GSSM is therefore a STRICT GENERALIZATION of Selective: any gain
    over the ω≡0 arm is attributable to the phase channel alone.

BOUNDEDNESS:
    |r_t|² = m_t²(cos²Θ + sin²Θ) = m_t² ≤ 1.  The phase is an isometry of ℂ;
    it cannot change the modulus, so boundedness is by geometry, not clipping.
    The state always lives in the closed unit disk.  Θ itself is unbounded
    (an angle), but only ever appears inside cos/sin, so no overflow at large T.

The ORIGINAL Selective layer/LM are imported from ../reference ONLY for
reference/comparison in the smoke test — never edited (reference/ is chmod 444).

Reference: Foss 2026, "From Markov Chains to Minkowski Space".
"""

import sys
import os
import math
from pathlib import Path

import torch
import torch.nn as nn

# ── Import ORIGINAL, UNMODIFIED reference modules (read-only; chmod 444) ──
# Same sys.path pattern as the other src/ runners (instrumented_runner.py).
REF = Path(__file__).resolve().parent.parent / "reference"
sys.path.insert(0, str(REF))

from moebius_attention import SinusoidalPositionalEncoding          # noqa: E402
# Imported ONLY to reference/compare against in the smoke test — never edited.
from moebius_scan_transformer_selective import (                    # noqa: E402
    SelectiveRapiditySqrtScanLayer,
    SelectiveRapiditySqrtTransformerLM,
)

# ── Constants matching the reference Selective layer byte-for-byte ──
LOG_COMPLEMENT_CLAMP = 0.999
EPS = 1e-6

# Default envelope (matches the validated regime d_model=128, H=4, D=32).
D_MODEL = 128
N_HEADS = 4
D_HEAD = 32
N_LAYERS = 2
SEQ_LEN = 32
DROPOUT = 0.1


# ===========================================================================
# Sequential scans
# ===========================================================================

def sequential_linear_scan(a: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """First-order linear scan  z_t = γ_t · z_{t-1} + a_t.

    Identical to the reference Selective scan: a simple O(T) loop.  At the
    short T used here the overhead is negligible vs. the rest of the model.

    a, gamma : (B, T, H, D)  ->  Z : (B, T, H, D)
    """
    B, T, H, D = a.shape
    Z = torch.zeros(B, H, D, device=a.device, dtype=a.dtype)
    out = []
    for t in range(T):
        Z = gamma[:, t] * Z + a[:, t]
        out.append(Z)
    return torch.stack(out, dim=1)


# ===========================================================================
# Phase-Selective scan layer (two real tensors, MPS-safe, NO torch.complex)
# ===========================================================================

class PhaseSelectiveScanLayer(nn.Module):
    """GSSM-Selective magnitude channel + orthogonal additive phase channel.

    The magnitude path (v, gate, γ, α → z → m) is byte-identical to
    ``SelectiveRapiditySqrtScanLayer``.  A new projection ``W_omega`` produces a
    per-token angular velocity ω_t whose causal cumulative sum is the phase
    Θ_t.  The bounded complex state  m_t·e^{iΘ_t}  is materialized only as the
    two real tensors (r_re, r_im) = m_t·(cos Θ_t, sin Θ_t).  Readout (Option A):
    ``out = W_out(r_re) + W_im(r_im)``, where W_out occupies the SAME slot as
    Selective's W_out and W_im is the new imaginary readout.

    use_phase=False  →  the exact-Selective control: skip the W_omega / Θ / W_im
    path entirely and return ``W_out(m)`` (= GSSM-Selective by construction).
    No complex dtypes are ever created — only cos, sin, *, +, log, exp, sqrt,
    cumsum, clamp, tanh, sigmoid (all MPS-native real ops).
    """

    def __init__(self, d_model: int, d_head: int = D_HEAD, n_heads: int = N_HEADS,
                 causal: bool = True, dropout: float = 0.0,
                 omega_scale: float = math.pi, use_phase: bool = True):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads
        self.causal = causal
        self.omega_scale = omega_scale
        self.use_phase = use_phase
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        total_dim = n_heads * d_head

        # ── Magnitude projections (identical names/shapes to Selective) ──
        self.W_v = nn.Linear(d_model, total_dim, bias=False)
        self.W_gate = nn.Linear(d_model, total_dim, bias=False)
        self.W_gamma = nn.Linear(d_model, total_dim, bias=False)  # forget gate
        self.W_alpha = nn.Linear(d_model, total_dim, bias=False)  # input  gate
        self.W_out = nn.Linear(total_dim, d_model, bias=False)    # SAME slot as Selective

        # ── Phase projections (NEW). Always constructed so the parameter set
        #    is consistent across use_phase, but inert when use_phase=False. ──
        # W_omega: in-projection for the angular velocity ω_t.
        self.W_omega = nn.Linear(d_model, total_dim, bias=False)
        # W_im: out-projection for the imaginary channel.  NEVER add a bias —
        # the θ≡0 guarantee W_im(0)=0 depends on bias being absent.
        self.W_im = nn.Linear(total_dim, d_model, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        # Magnitude inits — match Selective exactly.
        #   value-carrying readouts/projections: gain 0.6
        #   gates (γ, α): gain 0.1  (small init → sigmoid ≈ 0.5, gates "open")
        for module in [self.W_gamma, self.W_alpha]:
            for p in module.parameters():
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p, gain=0.1)
        for module in [self.W_v, self.W_gate, self.W_out]:
            for p in module.parameters():
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p, gain=0.6)
        # Phase inits.
        #   W_omega: small (gain 0.1) so ω ≈ 0 at init → the model STARTS near
        #            the Selective regime (phase nearly constant) and *learns*
        #            to rotate, earning the phase channel.
        #   W_im:    value-carrying readout like W_out → gain 0.6.
        for p in self.W_omega.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p, gain=0.1)
        for p in self.W_im.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p, gain=0.6)

    # ── magnitude channel (identical to Selective) ──────────────────────────
    def _magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Return m : (B, T, H, D) — exactly Selective's bounded state s_t."""
        B, T, _ = x.shape

        v = torch.tanh(self.W_v(x))
        gate = torch.sigmoid(self.W_gate(x))
        gamma = torch.sigmoid(self.W_gamma(x))   # forget
        alpha = torch.sigmoid(self.W_alpha(x))   # input

        v_gated = v * gate
        if self.dropout is not None:
            v_gated = self.dropout(v_gated)

        v_gated = v_gated.view(B, T, self.n_heads, self.d_head)
        gamma = gamma.view(B, T, self.n_heads, self.d_head)
        alpha = alpha.view(B, T, self.n_heads, self.d_head)

        # a_t = α_t · log(1 − v_gated²)  ≤ 0
        w = v_gated * v_gated
        w = torch.clamp(w, max=LOG_COMPLEMENT_CLAMP)
        z_in = torch.log(1.0 - w + EPS)
        a = alpha * z_in

        if self.causal:
            Z = sequential_linear_scan(a, gamma)
        else:
            Z_fwd = sequential_linear_scan(a, gamma)
            a_rev = torch.flip(a, dims=[1])
            gamma_rev = torch.flip(gamma, dims=[1])
            Z_rev = torch.flip(sequential_linear_scan(a_rev, gamma_rev), dims=[1])
            Z = Z_fwd + Z_rev

        s_sq = 1.0 - torch.exp(Z)
        s_sq = torch.clamp(s_sq, min=0.0)
        m = torch.sqrt(s_sq + EPS)            # ∈ [0,1)   (byte-identical to Selective)
        return m

    # ── phase channel (new) ─────────────────────────────────────────────────
    def _phase(self, x: torch.Tensor) -> torch.Tensor:
        """Return Θ : (B, T, H, D) — cumulative phase from ω_t.

        ω is an angular RATE, not a value — it is NOT dropped out.
        """
        B, T, _ = x.shape
        omega = torch.tanh(self.W_omega(x)) * self.omega_scale
        omega = omega.view(B, T, self.n_heads, self.d_head)

        if self.causal:
            Theta = torch.cumsum(omega, dim=1)
        else:
            # Mirror the bidirectional magnitude path: forward + flip-reverse-sum.
            # Phase has no boundedness constraint, so the sum is safe; this keeps
            # the bidirectional ablation consistent (ω≡0 ⟹ Θ≡0 still holds).
            Theta_fwd = torch.cumsum(omega, dim=1)
            Theta_rev = torch.flip(
                torch.cumsum(torch.flip(omega, dims=[1]), dim=1), dims=[1]
            )
            Theta = Theta_fwd + Theta_rev
        return Theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        m = self._magnitude(x)                 # (B, T, H, D)  — Selective's s_t

        if not self.use_phase:
            # θ≡0 ablation: EXACTLY GSSM-Selective.  Skip the entire phase path
            # (do not even compute Θ) and read out the real magnitude alone.
            m_flat = m.view(B, T, self.n_heads * self.d_head)
            return self.W_out(m_flat)

        # Phase-GSSM: build the bounded complex state as two real tensors.
        Theta = self._phase(x)                 # (B, T, H, D)
        r_re = m * torch.cos(Theta)            # m·cos Θ   (real part)
        r_im = m * torch.sin(Theta)            # m·sin Θ   (imag part)

        r_re = r_re.view(B, T, self.n_heads * self.d_head)
        r_im = r_im.view(B, T, self.n_heads * self.d_head)

        # Readout (Option A): same W_out as Selective on r_re, plus new W_im on r_im.
        return self.W_out(r_re) + self.W_im(r_im)

    # ── helper for tests/instrumentation: reconstruct the complex state ──────
    @torch.no_grad()
    def state_modulus(self, x: torch.Tensor) -> torch.Tensor:
        """Return |r_t| = sqrt(r_re² + r_im²) : (B, T, H, D), for bound checks.

        With use_phase=False this is just m_t (phase ≡ 0).
        """
        m = self._magnitude(x)
        if not self.use_phase:
            return m
        Theta = self._phase(x)
        r_re = m * torch.cos(Theta)
        r_im = m * torch.sin(Theta)
        return torch.sqrt(r_re * r_re + r_im * r_im)


# ===========================================================================
# Transformer layer + LM wrapper (copy of Selective wrappers, scan swapped)
# ===========================================================================

class PhaseSelectiveTransformerLayer(nn.Module):
    """Post-LN block  LN(x + sublayer(x))  — identical envelope to Selective,
    only the scan class is swapped for PhaseSelectiveScanLayer."""

    def __init__(self, d_model: int, d_head: int = D_HEAD, n_heads: int = N_HEADS,
                 ffn_dim: int = None, dropout: float = 0.0, causal: bool = True,
                 omega_scale: float = math.pi, use_phase: bool = True):
        super().__init__()
        self.scan = PhaseSelectiveScanLayer(
            d_model, d_head=d_head, n_heads=n_heads, causal=causal,
            dropout=dropout, omega_scale=omega_scale, use_phase=use_phase,
        )
        self.ln1 = nn.LayerNorm(d_model)
        ffn_dim = ffn_dim or 4 * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(ffn_dim, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x + self.scan(x))         # post-LN
        x = self.ln2(x + self.ffn(x))
        return x


class PhaseSelectiveLM(nn.Module):
    """GSSM LM envelope: embed vocab+2, sinusoidal PE, post-LN blocks,
    head vocab+1 — identical to SelectiveRapiditySqrtTransformerLM apart from
    the Phase-GSSM scan and the (omega_scale, use_phase) knobs."""

    def __init__(self, vocab_size: int, mask_idx: int,
                 d_model: int = D_MODEL, n_layers: int = N_LAYERS,
                 n_heads: int = N_HEADS, d_head: int = D_HEAD,
                 seq_len: int = SEQ_LEN, dropout: float = DROPOUT,
                 causal: bool = True, omega_scale: float = math.pi,
                 use_phase: bool = True):
        super().__init__()
        self.mask_idx = mask_idx
        self.use_phase = use_phase
        self.embed = nn.Embedding(vocab_size + 2, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            PhaseSelectiveTransformerLayer(
                d_model, d_head=d_head, n_heads=n_heads,
                ffn_dim=4 * d_model, dropout=dropout, causal=causal,
                omega_scale=omega_scale, use_phase=use_phase,
            )
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pos(self.embed(x))
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


# ===========================================================================
# Smoke test  (run on MPS if available; fwd + bwd for both phase arms)
# ===========================================================================

def _smoke_test():
    torch.manual_seed(0)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    B, T = 4, 16
    vocab_size, mask_idx = 200, 201
    x = torch.randint(0, vocab_size, (B, T), device=device)

    for use_phase in (True, False):
        tag = "use_phase=True (Phase-GSSM)" if use_phase else "use_phase=False (=Selective)"
        print(f"\n── {tag} ──")

        model = PhaseSelectiveLM(
            vocab_size, mask_idx, d_model=D_MODEL, n_layers=N_LAYERS,
            n_heads=N_HEADS, d_head=D_HEAD, seq_len=T, dropout=0.0,
            causal=True, omega_scale=math.pi, use_phase=use_phase,
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # forward
        logits = model(x)
        print(f"  output shape : {tuple(logits.shape)}  (expect ({B}, {T}, {vocab_size + 1}))")
        print(f"  output device: {logits.device.type}")
        print(f"  params       : {n_params:,}")
        assert tuple(logits.shape) == (B, T, vocab_size + 1), "output shape mismatch"
        assert logits.device.type == device.type, "output off-device"

        # backward
        loss = logits.float().pow(2).mean()
        loss.backward()

        grad_devs = {p.grad.device.type for p in model.parameters() if p.grad is not None}
        n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
        print(f"  grad devices : {grad_devs}  ({n_with_grad} tensors with grad)")
        assert grad_devs == {device.type}, f"grads not all on {device.type}: {grad_devs}"

        # When phase is off, the imaginary path must receive NO gradient
        # (it is never touched), and when on it MUST receive gradient.
        scan0 = model.layers[0].scan
        wim_grad = scan0.W_im.weight.grad
        womega_grad = scan0.W_omega.weight.grad
        if use_phase:
            assert wim_grad is not None and wim_grad.abs().sum().item() > 0, \
                "W_im should receive gradient when use_phase=True"
            assert womega_grad is not None and womega_grad.abs().sum().item() > 0, \
                "W_omega should receive gradient when use_phase=True"
            print("  phase grads  : W_omega and W_im both received gradient (OK)")
        else:
            # inert: never used in the forward, so .grad stays None
            assert wim_grad is None, "W_im must be untouched when use_phase=False"
            assert womega_grad is None, "W_omega must be untouched when use_phase=False"
            print("  phase grads  : W_omega and W_im untouched (inert, OK)")

        # magnitude bound: reconstruct the complex state and assert |r| ≤ 1.001
        with torch.no_grad():
            h = model.pos(model.embed(x))
            mods = []
            for layer in model.layers:
                scan = layer.scan
                mods.append(scan.state_modulus(h).max().item())
                h = layer(h)
            max_mod = max(mods)
        print(f"  max |state|  : {max_mod:.6f}  (must be ≤ 1.001)")
        assert max_mod <= 1.001, f"state modulus bound violated: {max_mod}"

    # ── θ≡0 reduction: use_phase=False must equal vanilla Selective ──
    print("\n── θ≡0 reduction check: Phase-GSSM(use_phase=False) vs GSSM-Selective ──")
    torch.manual_seed(123)
    phase_off = PhaseSelectiveLM(
        vocab_size, mask_idx, d_model=D_MODEL, n_layers=N_LAYERS,
        n_heads=N_HEADS, d_head=D_HEAD, seq_len=T, dropout=0.0,
        causal=True, use_phase=False,
    ).to(device)
    torch.manual_seed(123)
    selective = SelectiveRapiditySqrtTransformerLM(
        vocab_size, mask_idx, d_model=D_MODEL, n_layers=N_LAYERS,
        n_heads=N_HEADS, d_head=D_HEAD, seq_len=T, dropout=0.0, causal=True,
    ).to(device)
    # Copy the shared magnitude weights so the two models are weight-identical
    # on every parameter that participates when use_phase=False.
    with torch.no_grad():
        sd_sel = selective.state_dict()
        sd_phase = phase_off.state_dict()
        copied = 0
        for k, v in sd_sel.items():
            if k in sd_phase and sd_phase[k].shape == v.shape:
                sd_phase[k].copy_(v)
                copied += 1
        phase_off.load_state_dict(sd_phase)
    phase_off.eval()
    selective.eval()
    with torch.no_grad():
        out_phase = phase_off(x)
        out_sel = selective(x)
        max_abs_diff = (out_phase - out_sel).abs().max().item()
    print(f"  weights copied: {copied} tensors")
    print(f"  max |Δlogits| : {max_abs_diff:.3e}  (use_phase=False ≡ Selective)")
    assert max_abs_diff < 1e-4, f"use_phase=False is NOT Selective: diff={max_abs_diff}"

    print("\nAll smoke-test assertions passed.")


if __name__ == "__main__":
    _smoke_test()
