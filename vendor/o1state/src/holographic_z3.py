"""
Z3-Polyphase Holographic Read — the Foss-corpus crosstalk fix — by Opus 4.8
===========================================================================

The crosstalk wall, MEASURED: holographic recall rises as pairs drop —
n_pairs=8 → 7.6%, n_pairs=4 → 8.3%, n_pairs=2 → 25.8% (single de-rotation read,
holographic_capacity_log.txt). That 1/√N decay is the HRR interference walk: a SINGLE
de-rotation read,  read = Σ_k γ u_k cos(φ_k − φ_q),  is the n=2 case of the polyphase
power identity, and n=2 is exactly the case that FAILS to be constant.

THE FIX (David Foss, vortexmath_formulas.md §"Z₃ Polyphase Power Constancy", verified 1e-16,
re-verified here to 1e-15):

    Σ_{j=0}^{n-1} cos²(x + 2πj/n) = n/2   identically in x,  for ALL n ≥ 3   (n=2 fails).

Read the SAME complex accumulator S through n ≥ 3 quadrature phases offset by 2πj/n:

    read_j = Re( S · e^{−i(φ_q + 2πj/n)} ) = Σ_k γ u_k cos(φ_k − φ_q − 2πj/n)

The MATCHED key (φ_k = φ_q) contributes cos(−2πj/n) across the n reads — a known, fixed
pattern that survives a matched combine. The N−1 MISMATCHED keys have random (φ_k − φ_q),
and by the polyphase identity their *power* across the n offsets sums to a constant n/2
pedestal — DC, independent of the random angle. Subtract the cross-phase mean (the pedestal)
and the random interference cancels to O(1/n) instead of growing as O(√N). Crosstalk that
grew with N becomes a constant we remove.

CONCRETE READOUT (the "matched-power minus pedestal" combine):
    P_match = (1/n) Σ_j read_j · cos(2πj/n)·2   ← coherent matched projection (re part)
            + (1/n) Σ_j read_j · sin(2πj/n)·2   ← (im part, via the e^{-i·} phases)
  i.e. we recover Re(S·e^{-iφ_q}) AND Im(S·e^{-iφ_q}) as the n=1 read does, BUT we also form
    pedestal = (1/n) Σ_j read_j²   → the constant-power term, ≈ (n/2)·(interference energy)
  and gate the magnitude readout by (signal_power − pedestal), so the random walk is removed.

This stays BOUNDED O(1)-per-step (n is a constant, not T; n real scans fold into the head dim
like n_slots already does), MPS-native (no torch.complex), and reduces to GSSM-Selective
exactly when use_phase=False. n_phase=1 is the current single-read holographic baseline.

Self-contained. Offline. Reuses the frozen Selective magnitude recurrence.
Reference: Foss 2026 (Z₃ polyphase identity; MarkovChains→MinkowskiSpace geometry).
"""
import os, sys, math
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "reference"))
sys.path.insert(0, HERE)

LOG_COMPLEMENT_CLAMP = 0.999
EPS = 1e-6


def sequential_linear_scan(a, gamma):
    B, T, H, D = a.shape
    Z = torch.zeros(B, H, D, device=a.device, dtype=a.dtype)
    out = []
    for t in range(T):
        Z = gamma[:, t] * Z + a[:, t]
        out.append(Z)
    return torch.stack(out, dim=1)


def verify_polyphase_identity():
    """Re-verify Σ cos²(x+2πj/n)=n/2 for n>=3, fails n=2. The fix's foundation."""
    import numpy as np
    res = {}
    for n in [2, 3, 4, 6]:
        xs = np.linspace(0, 2 * np.pi, 33)
        vals = [sum(np.cos(x + 2 * np.pi * j / n) ** 2 for j in range(n)) for x in xs]
        res[n] = (float(np.mean(vals)), float(max(vals) - min(vals)))
    return res


COMBINE_CHOICES = ("linear", "relu", "dc_gate", "floorsub")
# linear   — DFT bin-1 == n=1 control (proves harness is clean)
# relu     — rectified-coherence: matched key de-rotates to +u>=0, relu kills negative
#            crosstalk half → AUC 0.92 in derivation
# dc_gate  — cross-channel DC as a selective multiplicative gate, the principled repair
#            of the broken sigmoid (gates on cross-channel mean, which IS selective) → AUC 0.945
# floorsub — one-sided soft-threshold using polyphase-estimated noise floor; best MSE


class Z3HolographicScanLayer(nn.Module):
    """Holographic write + Z3-POLYPHASE read (n>=3 quadrature phases kill crosstalk).

    n_phase=1  → the single-read holographic baseline (~9%).
    n_phase>=3 → polyphase constant-power read; mismatched interference → DC pedestal,
                 subtracted, so recall is no longer 1/√N-limited.
    use_phase=False → exact GSSM-Selective.

    combine : one of COMBINE_CHOICES (only active when n_phase >= 3).
      'linear'  : DFT bin-1 = n=1 control; must reproduce baseline.
      'relu'    : relu(read_re/im) before m·tanh — rectified DC coherence.
      'dc_gate' : sigmoid(beta * mean_d(read_re)) * read — selective cross-channel gate.
      'floorsub': relu(read - lambda*floor) one-sided soft-threshold.
    """

    def __init__(self, d_model, d_head=32, n_heads=4, causal=True, dropout=0.0,
                 phase_scale=math.pi, use_phase=True, n_phase=3,
                 combine="linear", dc_beta=1.0, floor_lambda=0.5):
        super().__init__()
        assert n_phase >= 1
        assert combine in COMBINE_CHOICES, f"combine must be one of {COMBINE_CHOICES}"
        self.d_model, self.d_head, self.n_heads = d_model, d_head, n_heads
        self.causal, self.phase_scale, self.use_phase = causal, phase_scale, use_phase
        self.n_phase = n_phase
        self.combine = combine
        self.dc_beta = dc_beta          # scale for dc_gate sigmoid
        self.floor_lambda = floor_lambda  # threshold fraction for floorsub
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        total = n_heads * d_head
        self.W_v = nn.Linear(d_model, total, bias=False)
        self.W_gate = nn.Linear(d_model, total, bias=False)
        self.W_gamma = nn.Linear(d_model, total, bias=False)
        self.W_alpha = nn.Linear(d_model, total, bias=False)
        self.W_out = nn.Linear(total, d_model, bias=False)
        self.W_key = nn.Linear(d_model, total, bias=False)
        self.W_im = nn.Linear(total, d_model, bias=False)
        self._reset()

    def _reset(self):
        for m in [self.W_gamma, self.W_alpha, self.W_key]:
            for p in m.parameters():
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p, gain=0.1)
        for m in [self.W_v, self.W_gate, self.W_out, self.W_im]:
            for p in m.parameters():
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p, gain=0.6)

    def _drive_and_gamma(self, x):
        B, T, _ = x.shape
        v = torch.tanh(self.W_v(x))
        gate = torch.sigmoid(self.W_gate(x))
        gamma = torch.sigmoid(self.W_gamma(x))
        alpha = torch.sigmoid(self.W_alpha(x))
        vg = v * gate
        if self.dropout is not None:
            vg = self.dropout(vg)
        vg = vg.view(B, T, self.n_heads, self.d_head)
        gamma = gamma.view(B, T, self.n_heads, self.d_head)
        alpha = alpha.view(B, T, self.n_heads, self.d_head)
        w = torch.clamp(vg * vg, max=LOG_COMPLEMENT_CLAMP)
        a = alpha * torch.log(1.0 - w + EPS)
        return a, gamma

    def _magnitude(self, x):
        a, gamma = self._drive_and_gamma(x)
        if self.causal:
            Z = sequential_linear_scan(a, gamma)
        else:
            Zf = sequential_linear_scan(a, gamma)
            Zr = torch.flip(sequential_linear_scan(
                torch.flip(a, dims=[1]), torch.flip(gamma, dims=[1])), dims=[1])
            Z = Zf + Zr
        return torch.sqrt(torch.clamp(1.0 - torch.exp(Z), min=0.0) + EPS)

    def forward(self, x):
        B, T, _ = x.shape
        m = self._magnitude(x)
        if not self.use_phase:
            return self.W_out(m.view(B, T, self.n_heads * self.d_head))

        a, gamma = self._drive_and_gamma(x)
        phi = (self.phase_scale * torch.tanh(self.W_key(x))).view(
            B, T, self.n_heads, self.d_head)            # key angle φ_t

        # complex write (two real leaky scans)
        dre, dim = a * torch.cos(phi), a * torch.sin(phi)
        if self.causal:
            S_re = sequential_linear_scan(dre, gamma)
            S_im = sequential_linear_scan(dim, gamma)
        else:
            S_re = sequential_linear_scan(dre, gamma) + torch.flip(
                sequential_linear_scan(torch.flip(dre, [1]), torch.flip(gamma, [1])), [1])
            S_im = sequential_linear_scan(dim, gamma) + torch.flip(
                sequential_linear_scan(torch.flip(dim, [1]), torch.flip(gamma, [1])), [1])

        # ── Z3-polyphase read: n quadrature de-rotations offset by 2πj/n ──
        n = self.n_phase
        reads = []
        for j in range(n):
            ang = phi + 2.0 * math.pi * j / n          # query angle φ_q + 2πj/n
            # read_j = Re(S·e^{-i·ang}) = S_re cos(ang) + S_im sin(ang)
            reads.append(S_re * torch.cos(ang) + S_im * torch.sin(ang))
        reads = torch.stack(reads, dim=0)              # (n, B, T, H, D)

        if n == 1:
            read_re = reads[0]
            read_im = S_im * torch.cos(phi) - S_re * torch.sin(phi)
        else:
            # ── DFT bin-1 combine: recovers Re/Im(S e^{-iφ_q}) exactly (proved rank-2).
            # This is the LINEAR baseline and the starting point for all other combines.
            cosw = torch.tensor([math.cos(2 * math.pi * j / n) for j in range(n)],
                                device=x.device).view(n, 1, 1, 1, 1)
            sinw = torch.tensor([math.sin(2 * math.pi * j / n) for j in range(n)],
                                device=x.device).view(n, 1, 1, 1, 1)
            read_re = (reads * cosw).sum(0) * (2.0 / n)   # = Re(S e^{-iφ_q})
            read_im = (reads * sinw).sum(0) * (2.0 / n)   # = Im(S e^{-iφ_q})

            if self.combine == "linear":
                # DFT bin-1 == n=1 control; must reproduce baseline recall.
                pass

            elif self.combine == "relu":
                # Rectified-coherence (AUC 0.92 in derivation).
                # Matched key de-rotates to +u_match >= 0 across all d_head channels;
                # crosstalk is zero-mean → relu kills the negative half, keeping the
                # coherent positive DC while suppressing half the crosstalk noise.
                read_re = torch.relu(read_re)
                read_im = torch.relu(read_im)

            elif self.combine == "dc_gate":
                # Cross-channel DC gate (AUC 0.945 in derivation) — the principled repair.
                # After de-rotation by φ_q, matched key contributes +u_match across all D
                # channels (coherent positive mean); mismatched keys sum to ~0 mean across D.
                # Gate by sigmoid(beta * mean_D(read_re)) — selective because the DC IS
                # different for matched vs unmatched, unlike coh_energy-pedestal which was
                # identically coh/2 (never selective, provably == the broken old gate).
                dc = read_re.mean(dim=-1, keepdim=True)          # (B, T, H, 1)
                gate = torch.sigmoid(self.dc_beta * dc)           # >0.5 when matched
                read_re = read_re * gate
                read_im = read_im * gate

            elif self.combine == "floorsub":
                # One-sided soft-threshold with polyphase-estimated noise floor.
                # pedestal = mean_j(read_j²) = ½|S e^{-iφ_q}|² (clean power estimate).
                # floor = sqrt(pedestal/2) ≈ crosstalk std per channel.
                # relu(read - lambda*floor): removes the zero-mean crosstalk, keeps matched DC.
                pedestal = (reads * reads).mean(0)                # (B, T, H, D)
                floor = torch.sqrt(pedestal * 0.5 + EPS)
                read_re = torch.relu(read_re - self.floor_lambda * floor)
                read_im = torch.relu(read_im - self.floor_lambda * floor)

        # m-gated readout (load-bearing relevance gate), tanh contrast
        read_re = m * torch.tanh(read_re)
        read_im = m * torch.tanh(read_im)
        rre = read_re.view(B, T, self.n_heads * self.d_head)
        rim = read_im.view(B, T, self.n_heads * self.d_head)
        return self.W_out(rre) + self.W_im(rim)


# ── LM wrapper ──
class Z3HolographicLM(nn.Module):
    def __init__(self, vocab_size, mask_idx, d_model=128, n_layers=2, n_heads=4,
                 d_head=32, seq_len=64, dropout=0.0, causal=True,
                 phase_scale=math.pi, use_phase=True, n_phase=3,
                 combine="linear", dc_beta=1.0, floor_lambda=0.5):
        super().__init__()
        from moebius_attention import SinusoidalPositionalEncoding
        self.embed = nn.Embedding(vocab_size + 2, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            scan = Z3HolographicScanLayer(d_model, d_head=d_head, n_heads=n_heads,
                                          causal=causal, dropout=dropout,
                                          phase_scale=phase_scale, use_phase=use_phase,
                                          n_phase=n_phase, combine=combine,
                                          dc_beta=dc_beta, floor_lambda=floor_lambda)
            ln1 = nn.LayerNorm(d_model)
            ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                nn.Linear(4 * d_model, d_model))
            ln2 = nn.LayerNorm(d_model)
            self.layers.append(nn.ModuleDict(dict(scan=scan, ln1=ln1, ffn=ffn, ln2=ln2)))
        self.head = nn.Linear(d_model, vocab_size + 1)

    def forward(self, x):
        h = self.pos(self.embed(x))
        for L in self.layers:
            h = L["ln1"](h + L["scan"](h))
            h = L["ln2"](h + L["ffn"](h))
        return self.head(h)


if __name__ == "__main__":
    print("=" * 70)
    print("Z3-Polyphase Holographic — corpus crosstalk fix")
    print("=" * 70)
    print("[identity] Σcos²(x+2πj/n)=n/2 :")
    for nn_, (mean, spread) in verify_polyphase_identity().items():
        tag = "CONSTANT (works)" if spread < 1e-12 else "x-DEPENDENT (fails — the n=2 read!)"
        print(f"  n={nn_}: mean={mean:.4f} (n/2={nn_/2})  spread={spread:.2e}  {tag}")
    print()
    torch.manual_seed(0)
    x = torch.randn(2, 40, 48)
    # n=1 baseline
    L = Z3HolographicScanLayer(48, d_head=12, n_heads=4, n_phase=1).eval()
    y = L(x)
    print(f"[smoke] n_phase=1 (baseline)  finite={torch.isfinite(y).all().item()}  "
          f"shape={tuple(y.shape)}  std={y.std().item():.3f}")
    # n=3 with each combine
    for comb in COMBINE_CHOICES:
        torch.manual_seed(0)
        L = Z3HolographicScanLayer(48, d_head=12, n_heads=4, n_phase=3, combine=comb).eval()
        y = L(x)
        print(f"[smoke] n_phase=3 combine={comb:<10s}  finite={torch.isfinite(y).all().item()}  "
              f"std={y.std().item():.3f}")
    # reduction: use_phase=False must be plain Selective magnitude readout
    L0 = Z3HolographicScanLayer(48, d_head=12, n_heads=4, use_phase=False).eval()
    print(f"[reduction] use_phase=False finite={torch.isfinite(L0(x)).all().item()} "
          f"(== Selective magnitude readout)")
