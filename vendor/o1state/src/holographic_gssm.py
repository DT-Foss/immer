"""
Holographic-GSSM — key-conditioned complex write on the bounded GSSM state — by Opus 4.8
========================================================================================

WHY THIS EXISTS.  The additive-phase Phase-GSSM (src/phase_gssm.py) gave 0.00pp recall
contribution across seeds: its phase Θ_t = cumsum(ω_t) rotates *blindly with time*, so
every value lands in one shared rotating accumulator and the (key,value) pairs cannot be
disentangled at read time.  That is the weakest possible binder — the write does not know
*which key* it is writing.  The adversarial review located the real axis: **key-conditioned
write**, the complex analogue of attention's outer-product KV binding.  Bounded D scalar
channels with key-conditioned writes were shown to reach ~0.88 recall; the magnitude/phase
*number line* was never the limit — the *binding mechanism* was.

THE MECHANISM (holographic / Hopfield / HRR memory in a leaky-integrator track).

Per channel, carry a COMPLEX leaky accumulator S_t ∈ ℂ:

    φ_t  = key angle of token t   = π · tanh(W_key x_t)        (depends on token IDENTITY,
                                                                NOT on time — this is the fix)
    u_t  = value written by token t = α_t · log(1 − v̄_t²)       (same bounded drive as Selective,
                                                                so |contribution| is controlled)
    S_t  = γ_t · S_{t-1} + u_t · e^{i φ_t}                       (leaky complex write)

Read at a query token q with its OWN key angle φ_q (same W_key — a query re-derives the
key it is asking for):

    read_t = Re( S_t · e^{−i φ_q } )  =  Σ_{k≤t} γ_{k→t} u_k cos(φ_k − φ_q)

The matched key (φ_k ≈ φ_q) rotates its value coherently onto the real axis (cos ≈ 1);
mismatched keys carry cos(φ_k − φ_q) that averages toward zero over many pairs.  This is
exactly the holographic-memory retrieval rule: superpose values at key-specific phases,
read by de-rotating with the query phase.  A SINGLE complex track now stores several
(key,value) pairs separably — the binding the scalar magnitude channel provably lacks.

BOUNDEDNESS.  |S_t| ≤ Σ_k γ_{k→t}|u_k| ≤ |u|/(1−γ) (finite DC gain, the kernel statement);
the readout is linear in S_t, so the RKHS / Volterra characterization carries over with a
COMPLEX (rather than real) reproducing kernel — the kernel picks up a phase factor
e^{i(φ_s − φ_t)}.  The magnitude channel m_t = √(1−exp z_t) is untouched.

REDUCTION GUARANTEE.  use_phase=False  →  W_key/W_im path skipped, read = Selective's real
magnitude readout, byte-identical to GSSM-Selective (the ablation control).  At init W_key
is small (gain 0.1) so φ ≈ 0 and the model STARTS near the real-write regime, learning to
spread keys across the phase circle if it earns recall.

The complex write decomposes into TWO REAL leaky scans (real & imag parts), each the exact
affine recurrence src/parallel_scan.py already parallelizes — so this stays MPS-native and
O(log T)-parallelizable, no torch.complex needed.

Self-contained; reuses the frozen Selective magnitude recurrence.  Offline.
Reference: Foss 2026, "From Markov Chains to Minkowski Space".
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


class HolographicScanLayer(nn.Module):
    """GSSM-Selective magnitude + key-conditioned COMPLEX leaky write (holographic).

    use_phase=False  →  exact GSSM-Selective (the ablation control).
    use_phase=True   →  key-conditioned holographic memory write/read.
    """

    def __init__(self, d_model: int, d_head: int = D_HEAD, n_heads: int = N_HEADS,
                 causal: bool = True, dropout: float = 0.0,
                 phase_scale: float = math.pi, use_phase: bool = True,
                 readout: str = "rms", separate_qk: bool = False,
                 n_slots: int = 1):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads
        self.causal = causal
        self.phase_scale = phase_scale
        self.use_phase = use_phase
        # separate_qk: give the WRITE its own key angle (W_key) and the READ/query its
        #   own angle (W_read_key), the way attention separates K and Q.  When False the
        #   single W_key does double duty (write angle == read angle), the 7% baseline.
        self.separate_qk = separate_qk
        # ── n_slots: MULTI-SLOT key-binned accumulators (crosstalk attack) ──────────
        # The single complex accumulator superposes ALL n_pairs pairs → holographic
        # crosstalk: read = Σ_k γ u_k cos(φ_k − φ_q); the N−1 mismatched keys do NOT
        # average to exactly zero for finite N, and interference grows with n_pairs.
        # More channels (d_head) was FLAT against this. The fix is FEWER superposed
        # pairs PER memory: give the layer M complex slots and route each token's WRITE
        # to one slot by a learned function of its content (W_slot). Each slot then
        # superposes only ~N/M pairs → crosstalk amplitude drops ~√M. The READ gathers
        # from the query's OWN routed slot (same W_slot) and de-rotates there.
        #   n_slots=1  →  byte-identical to the single-accumulator holographic baseline
        #                 (mask is all-ones, W_slot skipped). Reduction is preserved.
        assert n_slots >= 1, n_slots
        self.n_slots = n_slots
        # readout: how the holographic complex read is brought into the residual scale.
        #   "tanh_m"   : m·tanh(read)  — bounded but doubly damped (original; weak signal)
        #   "layernorm": raw read, let the post-LN block normalize it (full signal)
        #   "rms"      : read / (rms(read)+eps) — unit-scale, no saturation (default)
        assert readout in ("tanh_m", "layernorm", "rms"), readout
        self.readout = readout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        total_dim = n_heads * d_head

        # ── Magnitude / value projections (identical names/shapes to Selective) ──
        self.W_v = nn.Linear(d_model, total_dim, bias=False)
        self.W_gate = nn.Linear(d_model, total_dim, bias=False)
        self.W_gamma = nn.Linear(d_model, total_dim, bias=False)   # forget
        self.W_alpha = nn.Linear(d_model, total_dim, bias=False)   # input
        self.W_out = nn.Linear(total_dim, d_model, bias=False)     # SAME slot as Selective

        # ── Holographic projections (NEW) ──
        # W_key: per-token WRITE KEY ANGLE φ_write_t (token identity → phase; NOT cumulative).
        #        Shared-QK: also serves as the read/query angle.  Separate-QK: write only.
        self.W_key = nn.Linear(d_model, total_dim, bias=False)
        # W_read_key: SEPARATE READ/QUERY ANGLE φ_read_t — only when separate_qk=True.
        #        Attention separates Q and K; this gives the read its own projection so the
        #        de-rotation angle is learned independently of the write angle.
        self.W_read_key = nn.Linear(d_model, total_dim, bias=False) if separate_qk else None
        # W_im: out-projection for the imaginary read channel.  NO bias (φ≡0 ⇒ W_im(0)=0).
        self.W_im = nn.Linear(total_dim, d_model, bias=False)

        # W_slot: per-token SLOT ROUTER → logits over n_slots PER HEAD (one routing
        #   decision per head, shared across that head's d_head channels). argmax picks
        #   the slot a token writes to / a query reads from. Only built when n_slots>1,
        #   so n_slots=1 carries ZERO extra parameters and is exactly the baseline.
        self.W_slot = nn.Linear(d_model, n_heads * n_slots, bias=False) if n_slots > 1 else None

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
        # W_read_key small too → φ_read≈0 at init.  At init both angles ≈0, so the model
        # STARTS at the shared-QK regime (read angle ≈ write angle ≈ 0) and can earn a
        # distinct query projection if it sharpens matching.
        if self.W_read_key is not None:
            for p in self.W_read_key.parameters():
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p, gain=0.1)
        for p in self.W_im.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p, gain=0.6)
        # W_slot: ordinary-gain router so the argmax is content-dependent from the start
        # (we WANT distinct keys to spread across slots; a tiny gain would collapse all
        # tokens into slot 0 at init and defeat the binning).
        if self.W_slot is not None:
            for p in self.W_slot.parameters():
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p, gain=1.0)

    # ── shared drive: the bounded Selective value u_t and gates ──────────────
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
        a = alpha * z_in                          # a_t ≤ 0, the bounded log-complement drive
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

        # ── key-conditioned holographic write ──
        a, gamma = self._drive_and_gamma(x)        # value drive u_t (=a) + forget γ_t
        phi_w = self.phase_scale * torch.tanh(self.W_key(x))   # WRITE KEY ANGLE φ_write_t
        phi_w = phi_w.view(B, T, self.n_heads, self.d_head)

        # Read/query angle: shared-QK → same projection; separate-QK → its own W_read_key.
        if self.separate_qk:
            phi_r = self.phase_scale * torch.tanh(self.W_read_key(x))  # READ ANGLE φ_read_t
            phi_r = phi_r.view(B, T, self.n_heads, self.d_head)
        else:
            phi_r = phi_w

        # Write value u_t at its WRITE key phase: complex drive  a·e^{iφ_w} = (a cos φ_w, a sin φ_w).
        # Two real leaky scans (real & imag) under the SAME forget γ_t.
        drive_re = a * torch.cos(phi_w)
        drive_im = a * torch.sin(phi_w)

        if self.n_slots > 1:
            # ── MULTI-SLOT route: each token writes only into ITS slot; query reads from ITS slot.
            # Per-head slot logits → hard argmax one-hot (straight-through so W_slot still trains).
            slot_logits = self.W_slot(x).view(B, T, self.n_heads, self.n_slots)
            slot_soft = torch.softmax(slot_logits, dim=-1)
            slot_hard = torch.zeros_like(slot_soft).scatter_(
                -1, slot_soft.argmax(dim=-1, keepdim=True), 1.0)
            # straight-through estimator: hard routing forward, soft gradient backward.
            slot_oh = slot_hard + (slot_soft - slot_soft.detach())   # (B,T,H,M)

            # Broadcast the slot one-hot across the d_head channels and FOLD slots into the
            # head dim so the existing (B,T,H,D) scan handles all M·H accumulators at once.
            # write_mask_t,h,s = 1 iff token t (head h) routes to slot s.
            wmask = slot_oh.unsqueeze(-1)                            # (B,T,H,M,1)
            dre = (drive_re.unsqueeze(3) * wmask).reshape(
                B, T, self.n_heads * self.n_slots, self.d_head)     # masked write per slot
            dim = (drive_im.unsqueeze(3) * wmask).reshape(
                B, T, self.n_heads * self.n_slots, self.d_head)
            # γ is per (token,head,channel); each slot decays with the SAME γ as its head.
            gslot = gamma.unsqueeze(3).expand(
                B, T, self.n_heads, self.n_slots, self.d_head).reshape(
                B, T, self.n_heads * self.n_slots, self.d_head)

            if self.causal:
                Sre_all = sequential_linear_scan(dre, gslot)
                Sim_all = sequential_linear_scan(dim, gslot)
            else:
                Sre_all = sequential_linear_scan(dre, gslot) + torch.flip(
                    sequential_linear_scan(torch.flip(dre, dims=[1]),
                                           torch.flip(gslot, dims=[1])), dims=[1])
                Sim_all = sequential_linear_scan(dim, gslot) + torch.flip(
                    sequential_linear_scan(torch.flip(dim, dims=[1]),
                                           torch.flip(gslot, dims=[1])), dims=[1])

            # GATHER the query's slot: at each position, read from the slot that position
            # routes to (same router). slot_oh selects ONE slot per (B,T,H).
            Sre_all = Sre_all.view(B, T, self.n_heads, self.n_slots, self.d_head)
            Sim_all = Sim_all.view(B, T, self.n_heads, self.n_slots, self.d_head)
            rmask = slot_oh.unsqueeze(-1)                            # (B,T,H,M,1)
            S_re = (Sre_all * rmask).sum(dim=3)                      # (B,T,H,D) query slot only
            S_im = (Sim_all * rmask).sum(dim=3)
        elif self.causal:
            S_re = sequential_linear_scan(drive_re, gamma)
            S_im = sequential_linear_scan(drive_im, gamma)
        else:
            S_re = sequential_linear_scan(drive_re, gamma) + torch.flip(
                sequential_linear_scan(torch.flip(drive_re, dims=[1]),
                                       torch.flip(gamma, dims=[1])), dims=[1])
            S_im = sequential_linear_scan(drive_im, gamma) + torch.flip(
                sequential_linear_scan(torch.flip(drive_im, dims=[1]),
                                       torch.flip(gamma, dims=[1])), dims=[1])

        # ── read at each position by de-rotating with that position's READ/query angle ──
        #   read = Re( S · e^{−iφ_r} ) = S_re·cos φ_r + S_im·sin φ_r   (matched key → coherent)
        #   imag = Im( S · e^{−iφ_r} ) = S_im·cos φ_r − S_re·sin φ_r   (extra channel for W_im)
        # Shared-QK: φ_r == φ_w (a query re-derives the key it wrote).  Separate-QK: φ_r is
        # the independent query angle from W_read_key — the matching is sharpened iff the
        # learned read angle aligns with the stored write phases better than self-derivation.
        read_re = S_re * torch.cos(phi_r) + S_im * torch.sin(phi_r)
        read_im = S_im * torch.cos(phi_r) - S_re * torch.sin(phi_r)

        # Bring the holographic read into the residual scale. The leaky complex
        # accumulator has finite DC gain (|S| ≤ |u|/(1−γ)), so the read is already
        # bounded; the question is only how much signal survives to the readout.
        if self.readout == "tanh_m":
            # Original: gate by magnitude envelope and saturate. Bounded but doubly damped.
            m = self._magnitude(x)
            read_re = m * torch.tanh(read_re)
            read_im = m * torch.tanh(read_im)
        elif self.readout == "rms":
            # Unit-scale per (B,T,H) over the channel dim, no saturation — preserves the
            # coherent-vs-incoherent contrast that tanh would crush.
            rms_re = read_re.pow(2).mean(dim=-1, keepdim=True).add(EPS).sqrt()
            rms_im = read_im.pow(2).mean(dim=-1, keepdim=True).add(EPS).sqrt()
            read_re = read_re / rms_re
            read_im = read_im / rms_im
        # "layernorm": pass the raw read straight through; the post-LN block normalizes it.

        read_re = read_re.view(B, T, self.n_heads * self.d_head)
        read_im = read_im.view(B, T, self.n_heads * self.d_head)
        return self.W_out(read_re) + self.W_im(read_im)


class HolographicTransformerLayer(nn.Module):
    """Post-LN block, identical envelope to Selective; scan = HolographicScanLayer."""

    def __init__(self, d_model: int, d_head: int = D_HEAD, n_heads: int = N_HEADS,
                 ffn_dim: int = None, dropout: float = 0.0, causal: bool = True,
                 phase_scale: float = math.pi, use_phase: bool = True,
                 readout: str = "rms", separate_qk: bool = False, n_slots: int = 1):
        super().__init__()
        self.scan = HolographicScanLayer(
            d_model, d_head=d_head, n_heads=n_heads, causal=causal,
            dropout=dropout, phase_scale=phase_scale, use_phase=use_phase,
            readout=readout, separate_qk=separate_qk, n_slots=n_slots)
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


class HolographicLM(nn.Module):
    """Causal LM wrapper, same shape as SelectiveRapiditySqrtTransformerLM."""

    def __init__(self, vocab_size: int, mask_idx: int,
                 d_model: int = D_MODEL, n_layers: int = N_LAYERS,
                 n_heads: int = N_HEADS, d_head: int = D_HEAD,
                 seq_len: int = 64, dropout: float = 0.0, causal: bool = True,
                 phase_scale: float = math.pi, use_phase: bool = True,
                 readout: str = "rms", separate_qk: bool = False, n_slots: int = 1):
        super().__init__()
        from moebius_attention import SinusoidalPositionalEncoding
        self.mask_idx = mask_idx
        self.embed = nn.Embedding(vocab_size + 2, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            HolographicTransformerLayer(
                d_model, d_head=d_head, n_heads=n_heads, ffn_dim=4 * d_model,
                dropout=dropout, causal=causal, phase_scale=phase_scale,
                use_phase=use_phase, readout=readout, separate_qk=separate_qk,
                n_slots=n_slots)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size + 1)

    def forward(self, x):
        h = self.pos(self.embed(x))
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


# ───────────────────────────────────────────────────────────────────────────
# Reduction gate: use_phase=False must be byte-identical to GSSM-Selective.
# ───────────────────────────────────────────────────────────────────────────

def _verify_reduction(device="cpu", tol=1e-5):
    """A Holographic layer with use_phase=False must equal the Selective layer
    on identical magnitude weights (the ablation control must be exact)."""
    from moebius_scan_transformer_selective import SelectiveRapiditySqrtScanLayer
    torch.manual_seed(0)
    d_model, n_heads, d_head = 48, 4, 12
    holo = HolographicScanLayer(d_model, d_head=d_head, n_heads=n_heads,
                                use_phase=False).to(device).eval()
    sel = SelectiveRapiditySqrtScanLayer(d_model, d_head=d_head, n_heads=n_heads,
                                         dropout=0.0).to(device).eval()
    # Copy the shared magnitude projections so the two compute the same state.
    with torch.no_grad():
        sel.W_v.weight.copy_(holo.W_v.weight)
        sel.W_gate.weight.copy_(holo.W_gate.weight)
        sel.W_gamma.weight.copy_(holo.W_gamma.weight)
        sel.W_alpha.weight.copy_(holo.W_alpha.weight)
        sel.W_out.weight.copy_(holo.W_out.weight)
    x = torch.randn(3, 37, d_model, device=device)
    err = (holo(x) - sel(x)).abs().max().item()
    ok = err < tol
    print(f"[reduction] use_phase=False vs Selective  max|Δ| = {err:.3e}  "
          f"{'PASS (exact reduction)' if ok else 'FAIL'}")
    return ok, err


if __name__ == "__main__":
    print("=" * 74)
    print("Holographic-GSSM — key-conditioned complex write")
    print("=" * 74)
    ok, err = _verify_reduction()

    # Sanity: phase ON produces a different, finite, bounded output.
    torch.manual_seed(1)
    layer = HolographicScanLayer(48, d_head=12, n_heads=4, use_phase=True).eval()
    x = torch.randn(2, 40, 48)
    y = layer(x)
    print(f"[sanity]    phase ON  output  finite={torch.isfinite(y).all().item()}  "
          f"shape={tuple(y.shape)}  std={y.std().item():.3f}")
    # State-modulus bound check: the holographic read after tanh·m is in [-1,1]-ish.
    print(f"[sanity]    read range [{y.min().item():.3f}, {y.max().item():.3f}]")

    # Sanity: separate-QK (Front 2) produces finite output with the extra W_read_key.
    torch.manual_seed(1)
    layer_qk = HolographicScanLayer(48, d_head=12, n_heads=4, use_phase=True,
                                    readout="tanh_m", separate_qk=True).eval()
    y_qk = layer_qk(x)
    has_rk = layer_qk.W_read_key is not None
    print(f"[sanity]    separate_qk output finite={torch.isfinite(y_qk).all().item()}  "
          f"shape={tuple(y_qk.shape)}  std={y_qk.std().item():.3f}  W_read_key={has_rk}")
    # At init both angles ≈ 0, but distinct projections → output differs from shared-QK.
    torch.manual_seed(1)
    layer_shared = HolographicScanLayer(48, d_head=12, n_heads=4, use_phase=True,
                                        readout="tanh_m", separate_qk=False).eval()
    y_shared = layer_shared(x)
    print(f"[sanity]    separate_qk vs shared (tanh_m) max|Δ| = "
          f"{(y_qk - y_shared).abs().max().item():.3e} (nonzero ⇒ read angle is distinct)")
    sys.exit(0 if ok else 1)
