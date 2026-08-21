"""
Holographic-GSSM — Hopfield rectified-power readout WITH learned β (temperature)
================================================================================
REOPENING dead-end entry 18.  The rectified-power readout  sign(x)·|x|^p  (poly3,
poly5) was KILLED as built: applied AFTER rms-normalisation, ~half the channels have
|x|<1, and |x|^p with p≥3 CRUSHES them toward 0 (gradient 3x²→0 there) → dead
channels → chance recall.  The entry itself names the missing ingredient:

    "Missing ingredient: a learned scale/temperature BEFORE the power (real Hopfield
     has the β in softmax(βx); pure |x|^p has none).  poly-after-rms is the wrong
     order.  NOT a refutation of nonlinear readout — a refutation of *this unscaled
     form*.  Reopen with learned-β or poly-before-norm."

This file builds BOTH reopened forms as two new readout modes on the SAME holographic
key-conditioned complex write as src/holographic_gssm.py (single band, shared-QK).
Nothing else changes — only the map read → residual.

THE TWO REOPENED FORMS
----------------------
Let the coherent read be `x` (per (B,T,H,D)), rms_x = rms over the channel dim.

(A) "hopfield_beta"  — learned temperature β BEFORE the power, applied to the
    rms-normalised read (fixes the crush by SPREADING the contrast first):

        x̂   = x / (rms_x + eps)               # unit-scale contrast (the rms baseline)
        y   = sign(x̂) · |β · x̂|^p            # β learned (per-head), p a fixed exponent

    β is a learned per-head positive scale (β = softplus(raw_β)).  The power now acts
    on β·x̂: the matched channel (large x̂) is pushed ABOVE 1 by β so its |·|^p GROWS,
    the incoherent channels (small x̂) still shrink → the coherent-vs-incoherent
    CONTRAST is sharpened instead of destroyed.  This is exactly the Hopfield β·x
    inside the nonlinearity.

    INIT so start == rms: init raw_β = softplus^{-1}(1) and use exponent p=1 in the
    warmup-equivalent — but to keep it a genuine power readout we set p=3 (fixed) and
    init β so β·x̂ has unit RMS at the start (β_init = 1), giving sign·|x̂|^3.  To make
    the START itself behave like rms (entry-18's requirement "init == rms/tanh_m") we
    add a learned MIX gate λ∈[0,1] (λ=0 at init) that blends power-readout with the
    plain rms read:  out = (1-λ)·x̂ + λ·y.  At init λ=0 → out == x̂ == the rms readout
    EXACTLY (verified in _verify_reduction_rms).  The model must EARN the power path.

(B) "poly_before_norm" — the power BEFORE the rms-normalisation (the other order the
    entry names).  The rms is then computed ON the powered signal, so it can never
    crush: whatever scale |βx|^p lands at, the following rms brings it back to unit
    contrast without a dead zone:

        y   = sign(x) · |β · x|^p             # power on the RAW read (β learned)
        out = y / (rms(y) + eps)              # normalise AFTER the power

    Same λ-mix to rms at init (λ=0 → out == plain-rms-of-x path… but note poly-before
    changes the argument; we blend against rms(x) so λ=0 is again exactly rms).

Both are single-band, shared-QK, use_phase=False ⇒ exact GSSM-Selective (reduction
preserved).  All ops MPS-native real (no torch.complex).  Offline.

Reference: Foss 2026, "From Markov Chains to Minkowski Space"; RECALL_DEADENDS_LOG entry 18.
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
    """z_t = γ_t·z_{t-1} + a_t, shapes (B,T,H,D). Same recurrence as Selective."""
    B, T, H, D = a.shape
    Z = torch.zeros(B, H, D, device=a.device, dtype=a.dtype)
    out = []
    for t in range(T):
        Z = gamma[:, t] * Z + a[:, t]
        out.append(Z)
    return torch.stack(out, dim=1)


class HopfieldBetaScanLayer(nn.Module):
    """Holographic key-conditioned complex write + Hopfield learned-β power readout.

    readout ∈ {"rms", "tanh_m", "hopfield_beta", "poly_before_norm"}
        "rms"              : baseline unit-scale read (entry-18 control).
        "tanh_m"           : m·tanh(read) baseline.
        "hopfield_beta"    : (A) β BEFORE power, power AFTER rms-norm, λ-mix to rms.
        "poly_before_norm" : (B) power BEFORE rms-norm, λ-mix to rms.

    p        : fixed power exponent for the two hopfield readouts (default 3).
    use_phase=False → exact GSSM-Selective (reduction control).
    """

    def __init__(self, d_model: int, d_head: int = D_HEAD, n_heads: int = N_HEADS,
                 causal: bool = True, dropout: float = 0.0,
                 phase_scale: float = math.pi, use_phase: bool = True,
                 readout: str = "hopfield_beta", p: float = 3.0):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads
        self.causal = causal
        self.phase_scale = phase_scale
        self.use_phase = use_phase
        assert readout in ("rms", "tanh_m", "hopfield_beta", "poly_before_norm"), readout
        self.readout = readout
        self.p = p
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        total_dim = n_heads * d_head

        # ── Magnitude / value projections (identical names/shapes to Selective) ──
        self.W_v = nn.Linear(d_model, total_dim, bias=False)
        self.W_gate = nn.Linear(d_model, total_dim, bias=False)
        self.W_gamma = nn.Linear(d_model, total_dim, bias=False)   # forget
        self.W_alpha = nn.Linear(d_model, total_dim, bias=False)   # input
        self.W_out = nn.Linear(total_dim, d_model, bias=False)

        # ── Holographic projections (single band, shared-QK) ──
        self.W_key = nn.Linear(d_model, total_dim, bias=False)
        self.W_im = nn.Linear(total_dim, d_model, bias=False)

        # ── Hopfield learned-β params (only for the two power readouts) ──
        #   raw_beta : per-head temperature; β = softplus(raw_beta), init β=1
        #              (softplus(raw)=1 → raw = log(e^1 - 1) ≈ 0.5413).
        #   raw_lambda: per-head power-mix gate; λ = sigmoid(raw_lambda), init λ=0
        #              (raw_lambda large negative), so START == plain rms read EXACTLY.
        if readout in ("hopfield_beta", "poly_before_norm"):
            beta0 = math.log(math.e - 1.0)          # softplus^{-1}(1)
            self.raw_beta = nn.Parameter(torch.full((n_heads,), beta0))
            self.raw_lambda = nn.Parameter(torch.full((n_heads,), -12.0))  # sigmoid(-12)≈6e-6 → start==rms
        else:
            self.raw_beta = None
            self.raw_lambda = None

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
        gamma = gamma.view(B, T, self.n_heads, self.d_head)
        alpha = alpha.view(B, T, self.n_heads, self.d_head)

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

    def _rms_norm(self, read):
        rms = read.pow(2).mean(dim=-1, keepdim=True).add(EPS).sqrt()
        return read / rms

    def _hopfield_beta(self, read):
        """(A) β BEFORE the power, power AFTER rms-norm, λ-mix to rms.

        read : (B,T,H,D).  β,λ per head → broadcast over D.
        """
        xhat = self._rms_norm(read)                          # unit-scale (the rms read)
        beta = torch.nn.functional.softplus(self.raw_beta)   # (H,)  >0, init 1
        lam = torch.sigmoid(self.raw_lambda)                 # (H,)  ∈[0,1], init≈0
        beta = beta.view(1, 1, self.n_heads, 1)
        lam = lam.view(1, 1, self.n_heads, 1)
        bx = beta * xhat
        y = torch.sign(bx) * bx.abs().clamp(min=EPS).pow(self.p)   # rectified power
        # re-normalise the powered signal so its scale matches the residual, THEN mix.
        y = self._rms_norm(y)
        return (1.0 - lam) * xhat + lam * y                  # λ=0 → exact rms

    def _poly_before_norm(self, read):
        """(B) power BEFORE the rms-norm, λ-mix to rms.

        read : (B,T,H,D).  power on the RAW read, rms AFTER (never crushes).
        """
        beta = torch.nn.functional.softplus(self.raw_beta)
        lam = torch.sigmoid(self.raw_lambda)
        beta = beta.view(1, 1, self.n_heads, 1)
        lam = lam.view(1, 1, self.n_heads, 1)
        bx = beta * read
        y = torch.sign(bx) * bx.abs().clamp(min=EPS).pow(self.p)   # power on RAW read
        y = self._rms_norm(y)                                 # normalise AFTER power
        xhat = self._rms_norm(read)                           # the plain rms read
        return (1.0 - lam) * xhat + lam * y                   # λ=0 → exact rms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        if not self.use_phase:
            m = self._magnitude(x)
            return self.W_out(m.view(B, T, self.n_heads * self.d_head))

        a, gamma = self._drive_and_gamma(x)
        phi = self.phase_scale * torch.tanh(self.W_key(x))
        phi = phi.view(B, T, self.n_heads, self.d_head)

        drive_re = a * torch.cos(phi)
        drive_im = a * torch.sin(phi)

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

        # shared-QK read (φ_read == φ_write)
        read_re = S_re * torch.cos(phi) + S_im * torch.sin(phi)
        read_im = S_im * torch.cos(phi) - S_re * torch.sin(phi)

        if self.readout == "tanh_m":
            m = self._magnitude(x)
            read_re = m * torch.tanh(read_re)
            read_im = m * torch.tanh(read_im)
        elif self.readout == "rms":
            read_re = self._rms_norm(read_re)
            read_im = self._rms_norm(read_im)
        elif self.readout == "hopfield_beta":
            read_re = self._hopfield_beta(read_re)
            read_im = self._hopfield_beta(read_im)
        elif self.readout == "poly_before_norm":
            read_re = self._poly_before_norm(read_re)
            read_im = self._poly_before_norm(read_im)

        read_re = read_re.view(B, T, self.n_heads * self.d_head)
        read_im = read_im.view(B, T, self.n_heads * self.d_head)
        return self.W_out(read_re) + self.W_im(read_im)


class HopfieldBetaTransformerLayer(nn.Module):
    def __init__(self, d_model: int, d_head: int = D_HEAD, n_heads: int = N_HEADS,
                 ffn_dim: int = None, dropout: float = 0.0, causal: bool = True,
                 phase_scale: float = math.pi, use_phase: bool = True,
                 readout: str = "hopfield_beta", p: float = 3.0):
        super().__init__()
        self.scan = HopfieldBetaScanLayer(
            d_model, d_head=d_head, n_heads=n_heads, causal=causal,
            dropout=dropout, phase_scale=phase_scale, use_phase=use_phase,
            readout=readout, p=p)
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


class HopfieldBetaLM(nn.Module):
    def __init__(self, vocab_size: int, mask_idx: int,
                 d_model: int = D_MODEL, n_layers: int = N_LAYERS,
                 n_heads: int = N_HEADS, d_head: int = D_HEAD,
                 seq_len: int = 64, dropout: float = 0.0, causal: bool = True,
                 phase_scale: float = math.pi, use_phase: bool = True,
                 readout: str = "hopfield_beta", p: float = 3.0):
        super().__init__()
        from moebius_attention import SinusoidalPositionalEncoding
        self.mask_idx = mask_idx
        self.embed = nn.Embedding(vocab_size + 2, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            HopfieldBetaTransformerLayer(
                d_model, d_head=d_head, n_heads=n_heads, ffn_dim=4 * d_model,
                dropout=dropout, causal=causal, phase_scale=phase_scale,
                use_phase=use_phase, readout=readout, p=p)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size + 1)

    def forward(self, x):
        h = self.pos(self.embed(x))
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


# ───────────────────────────────────────────────────────────────────────────
# Verification gates
# ───────────────────────────────────────────────────────────────────────────

def _verify_reduction_selective(device="cpu", tol=1e-5):
    """use_phase=False must equal GSSM-Selective (ablation control)."""
    from moebius_scan_transformer_selective import SelectiveRapiditySqrtScanLayer
    torch.manual_seed(0)
    d_model, n_heads, d_head = 48, 4, 12
    holo = HopfieldBetaScanLayer(d_model, d_head=d_head, n_heads=n_heads,
                                 use_phase=False, readout="hopfield_beta").to(device).eval()
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
    print(f"[reduction/selective] use_phase=False vs Selective  max|Δ| = {err:.3e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok, err


def _verify_reduction_rms(device="cpu", tol=1e-4):
    """At init (λ≈6e-6), the two hopfield readouts must ≈ the plain rms readout
    on the SAME weights (entry-18's requirement: start == rms). Tol 1e-4 because
    λ=sigmoid(-12)≈6e-6 leaks a FP-grade amount of the power path — the START is rms
    up to that; the SELECTIVE reduction (above) is the byte-exact hard guarantee."""
    torch.manual_seed(1)
    d_model, n_heads, d_head = 48, 4, 12
    results = {}
    for ro in ("hopfield_beta", "poly_before_norm"):
        torch.manual_seed(1)
        holo = HopfieldBetaScanLayer(d_model, d_head=d_head, n_heads=n_heads,
                                     use_phase=True, readout=ro).to(device).eval()
        torch.manual_seed(1)
        rms = HopfieldBetaScanLayer(d_model, d_head=d_head, n_heads=n_heads,
                                    use_phase=True, readout="rms").to(device).eval()
        # copy shared weights (same manual_seed → already identical, but be explicit)
        with torch.no_grad():
            for name in ("W_v", "W_gate", "W_gamma", "W_alpha", "W_out", "W_key", "W_im"):
                getattr(rms, name).weight.copy_(getattr(holo, name).weight)
        x = torch.randn(3, 37, d_model, device=device)
        err = (holo(x) - rms(x)).abs().max().item()
        ok = err < tol
        results[ro] = (ok, err)
        print(f"[reduction/rms] readout={ro:18s} init(λ=0) vs rms  max|Δ| = {err:.3e}  "
              f"{'PASS' if ok else 'FAIL'}")
    return all(v[0] for v in results.values()), results


if __name__ == "__main__":
    print("=" * 74)
    print("Holographic Hopfield learned-β readout — reopening entry 18")
    print("=" * 74)
    ok1, _ = _verify_reduction_selective()
    ok2, _ = _verify_reduction_rms()

    # sanity: power path produces finite output once λ is opened
    torch.manual_seed(2)
    layer = HopfieldBetaScanLayer(48, d_head=12, n_heads=4, use_phase=True,
                                  readout="hopfield_beta").eval()
    with torch.no_grad():
        layer.raw_lambda.fill_(3.0)   # open the power path (λ≈0.95)
        x = torch.randn(2, 40, 48)
        y = layer(x)
    print(f"[sanity] hopfield_beta λ-open finite={torch.isfinite(y).all().item()} "
          f"shape={tuple(y.shape)} std={y.std().item():.3f}")
    sys.exit(0 if (ok1 and ok2) else 1)
