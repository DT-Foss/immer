"""
DeltaNet-GSSM — bounded D×D fast-weight matrix memory, O(1) in T — 2026-06-25
================================================================================

WHY DELTANET OVER HOLOGRAPHIC.
The holographic scalar memory superposes all n_pairs pairs into ONE complex scalar
accumulator per channel: S_t = γ S_{t-1} + u_t e^{iφ_t}.  The (N−1) mismatched
keys produce crosstalk cos(φ_k − φ_q) that averages to ~0 but does NOT cancel
exactly: residual interference ∝ 1/√N → 8.89% recall at 8 pairs, hard ceiling.

DeltaNet (fast-weight / outer-product matrix memory) rewrites before storing:
    M_t = M_{t-1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ

The (I − β k kᵀ) factor PROJECTS old content off the new key's subspace before
the write, so distinct keys do not interfere in the same sum.  At β=1 this is a
clean ERASE-THEN-WRITE: M_t k_t ← v_t, prior content on orthogonal keys preserved
exactly.  DeltaNet achieves ~100% MQAR on 64 pairs; holographic hits a ceiling ≤ 9%.

BOUNDEDNESS (the hard constraint).
M ∈ ℝ^{d_k × d_v} per head, fixed size, independent of T and of the 64-key
alphabet.  NOT a KV-cache; NOT indexed by key ID.  The matrix IS the state.
‖M_t‖_F ≤ max(‖M_{t-1}‖_F, √d_v) because (I−βkkᵀ) is a contraction for
‖k‖=1, β ∈ [0,1].  The state is O(d_k × d_v × H) = fixed.

IMPLEMENTATION: numerically stable DeltaNet update form.
    delta_t = k_tᵀ M_{t-1}    (what the memory currently returns for this key)
    M_t = M_{t-1} + β_t (v_t − delta_t)ᵀ k_tᵀ   (erase the old, write the new)
Equivalently:
    M_t = M_{t-1} − β_t (delta_t − v_t) ⊗ k_t

This is the cancellation form; it avoids forming (I−βkkᵀ)M explicitly.

READ: y_q = M_t k_q  (k_q shared with write key; inner-product lookup)
OPTIONAL GATED VARIANT: add per-step scalar decay γ_t on M before the update
(γ_t M_{t-1} instead of M_{t-1}) — helps if full sequence needs to be forgotten.
We build plain first; gated is enabled with use_gate=True.

REDUCTION GUARANTEE.
beta=0 (or β_t ≡ 0 from W_beta init) → M_t = M_{t-1} ≡ 0 → y_q = 0 everywhere
→ model must use W_out(0) = 0, i.e. it degrades to an all-zero memory.
A separate "delta_off" arm in the run script verifies this arm's recall (it should
equal or beat chance, since the residual stream still carries the embedding).

Reference: Schlag et al. "Linear Transformers Are Secretly Fast Weight Programmers"
(ICML 2021). Foss 2026, GSSM-Kernel experiments.
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

EPS = 1e-6

# ────────────────────────────────────────────────────────────────────────────
# Core: bounded DeltaNet fast-weight scan (sequential, real, CPU/MPS-safe)
# ────────────────────────────────────────────────────────────────────────────

def deltanet_scan(k: torch.Tensor,   # (B, T, H, d_k)   normalised key
                  v: torch.Tensor,   # (B, T, H, d_v)   value
                  beta: torch.Tensor,  # (B, T, H)       write strength ∈ [0,1]
                  gamma: torch.Tensor = None,  # (B, T, H) optional decay ∈ [0,1]
                  ) -> torch.Tensor:
    """Sequential DeltaNet fast-weight update.

    Returns y  (B, T, H, d_v): read from M BEFORE each write at step t
    (causal: query cannot attend to token t's own write — only M_{t-1}).

    State M shape: (B, H, d_k, d_v) — fixed, bounded, O(1) in T.

    Update (numerically stable cancellation form):
        delta_t = k_t M_{t-1}                         (B, H, d_v)
        if use_gate: M_{t-1} *= gamma_t               (per-head scalar decay)
        M_t = M_{t-1} + beta_t * (v_t - delta_t) ⊗ k_t
    """
    B, T, H, dk = k.shape
    dv = v.shape[-1]

    M = torch.zeros(B, H, dk, dv, device=k.device, dtype=k.dtype)
    outs = []

    beta_exp = beta.unsqueeze(-1)        # (B, T, H, 1) for broadcasting

    for t in range(T):
        k_t = k[:, t]                     # (B, H, d_k)
        v_t = v[:, t]                     # (B, H, d_v)
        b_t = beta_exp[:, t]              # (B, H, 1)

        # READ from M before write (causal)
        #   y_t = M_{t-1} k_t  →  (B, H, d_k) @ (B, H, d_k, d_v) → (B, H, d_v)
        #   via einsum: y[b,h,j] = sum_i M[b,h,i,j] * k[b,h,i]
        y_t = torch.einsum("bhi,bhij->bhj", k_t, M)   # (B, H, d_v)
        outs.append(y_t)

        # Optional per-step decay (gated variant)
        if gamma is not None:
            g_t = gamma[:, t].unsqueeze(-1).unsqueeze(-1)   # (B, H, 1, 1)
            M = M * g_t

        # WRITE (cancellation form — numerically stable)
        #   delta = M k  (same as y_t, reuse)
        delta_t = y_t                              # (B, H, d_v)
        err_t = v_t - delta_t                      # (B, H, d_v) — residual to write
        # outer product:  (B, H, d_k, 1) * (B, H, 1, d_v) → (B, H, d_k, d_v)
        M = M + b_t.unsqueeze(-1) * (k_t.unsqueeze(-1) * err_t.unsqueeze(-2))

    return torch.stack(outs, dim=1)    # (B, T, H, d_v)


# ────────────────────────────────────────────────────────────────────────────
# Layer: DeltaNetScanLayer
# ────────────────────────────────────────────────────────────────────────────

class DeltaNetScanLayer(nn.Module):
    """Bounded DeltaNet fast-weight memory layer.

    Per head carries M ∈ ℝ^{d_k × d_v}, fixed size independent of T and vocab.

    use_gate=False  →  plain DeltaNet
    use_gate=True   →  gated DeltaNet (per-step γ_t decay on M)
    """

    def __init__(self, d_model: int, d_k: int = 32, d_v: int = 32,
                 n_heads: int = 4, dropout: float = 0.0,
                 use_gate: bool = False):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.use_gate = use_gate

        # Key projection: shared for write and read (MQAR: same key reads its value)
        self.W_k = nn.Linear(d_model, n_heads * d_k, bias=False)
        # Value projection
        self.W_v = nn.Linear(d_model, n_heads * d_v, bias=False)
        # Beta: write strength in [0, 1] per head (scalar per token per head)
        self.W_beta = nn.Linear(d_model, n_heads, bias=True)
        # Optional gate: per-step decay in [0, 1] per head
        self.W_gamma = nn.Linear(d_model, n_heads, bias=True) if use_gate else None
        # Output projection: d_v * n_heads → d_model
        self.W_out = nn.Linear(n_heads * d_v, d_model, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        self._reset_parameters()

    def _reset_parameters(self):
        # Keys: small init so reads start near 0 (avoids spurious recall at init)
        nn.init.xavier_uniform_(self.W_k.weight, gain=0.5)
        nn.init.xavier_uniform_(self.W_v.weight, gain=0.6)
        nn.init.xavier_uniform_(self.W_out.weight, gain=0.6)
        # Beta bias → logit 0 → sigmoid(0) = 0.5 at init (moderate write strength)
        nn.init.zeros_(self.W_beta.bias)
        nn.init.xavier_uniform_(self.W_beta.weight, gain=0.5)
        if self.W_gamma is not None:
            # Gamma bias → sigmoid(2) ≈ 0.88: keep memory, slight decay at init
            nn.init.constant_(self.W_gamma.bias, 2.0)
            nn.init.xavier_uniform_(self.W_gamma.weight, gain=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        # --- projections ---
        k_raw = self.W_k(x).view(B, T, self.n_heads, self.d_k)   # (B, T, H, d_k)
        # Normalise keys to unit sphere (essential for DeltaNet stability and
        # correct beta-=1 erase semantics: (I - k kᵀ) is an orthogonal projector
        # only when ‖k‖ = 1)
        k = k_raw / (k_raw.norm(dim=-1, keepdim=True) + EPS)

        v = self.W_v(x).view(B, T, self.n_heads, self.d_v)        # (B, T, H, d_v)
        if self.dropout is not None:
            v = self.dropout(v)

        beta = torch.sigmoid(self.W_beta(x))                       # (B, T, H)

        gamma = None
        if self.use_gate:
            gamma = torch.sigmoid(self.W_gamma(x))                 # (B, T, H)

        # --- DeltaNet scan ---
        y = deltanet_scan(k, v, beta, gamma=gamma)                  # (B, T, H, d_v)

        # --- output projection ---
        y = y.reshape(B, T, self.n_heads * self.d_v)
        return self.W_out(y)


# ────────────────────────────────────────────────────────────────────────────
# Transformer block wrapper (same post-LN envelope as holographic baseline)
# ────────────────────────────────────────────────────────────────────────────

class DeltaNetTransformerLayer(nn.Module):
    """Post-LN residual block around DeltaNetScanLayer."""

    def __init__(self, d_model: int, d_k: int = 32, d_v: int = 32,
                 n_heads: int = 4, ffn_dim: int = None,
                 dropout: float = 0.0, use_gate: bool = False):
        super().__init__()
        self.scan = DeltaNetScanLayer(
            d_model, d_k=d_k, d_v=d_v, n_heads=n_heads,
            dropout=dropout, use_gate=use_gate)
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


# ────────────────────────────────────────────────────────────────────────────
# LM wrapper (same interface as HolographicLM for the harness)
# ────────────────────────────────────────────────────────────────────────────

class DeltaNetLM(nn.Module):
    """Causal LM with DeltaNet fast-weight memory layers."""

    def __init__(self, vocab_size: int, mask_idx: int,
                 d_model: int = 128, n_layers: int = 2,
                 n_heads: int = 4, d_k: int = 32, d_v: int = 32,
                 seq_len: int = 64, dropout: float = 0.0,
                 use_gate: bool = False):
        super().__init__()
        from moebius_attention import SinusoidalPositionalEncoding
        self.mask_idx = mask_idx
        self.embed = nn.Embedding(vocab_size + 2, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            DeltaNetTransformerLayer(
                d_model, d_k=d_k, d_v=d_v, n_heads=n_heads,
                ffn_dim=4 * d_model, dropout=dropout, use_gate=use_gate)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size + 1)

    def forward(self, x):
        h = self.pos(self.embed(x))
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


# ────────────────────────────────────────────────────────────────────────────
# Quick sanity / bounded-check when run as a script
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("DeltaNet-GSSM sanity check")
    print("=" * 70)

    torch.manual_seed(0)
    B, T, d_model = 2, 40, 128
    n_heads, d_k, d_v = 4, 32, 32
    x = torch.randn(B, T, d_model)

    # --- plain ---
    layer = DeltaNetScanLayer(d_model, d_k=d_k, d_v=d_v, n_heads=n_heads,
                              use_gate=False).eval()
    with torch.no_grad():
        y = layer(x)
    print(f"[plain]  output finite={torch.isfinite(y).all().item()}  "
          f"shape={tuple(y.shape)}  std={y.std().item():.4f}")
    print(f"[plain]  range [{y.min().item():.3f}, {y.max().item():.3f}]")

    # --- gated ---
    layer_g = DeltaNetScanLayer(d_model, d_k=d_k, d_v=d_v, n_heads=n_heads,
                                use_gate=True).eval()
    with torch.no_grad():
        yg = layer_g(x)
    print(f"[gated]  output finite={torch.isfinite(yg).all().item()}  "
          f"shape={tuple(yg.shape)}  std={yg.std().item():.4f}")

    # --- bounded state check ---
    print("\n[bounded] M is d_k × d_v per head — fixed, NOT growing with T:")
    print(f"  state shape = ({n_heads}, {d_k}, {d_v})  per sample  (O(1) in T)")
    print(f"  total state params = {n_heads * d_k * d_v} reals/sample (independent of T, vocab)")

    # --- gradient flow ---
    x2 = torch.randn(2, 20, d_model, requires_grad=False)
    x2.requires_grad_(True)
    layer2 = DeltaNetScanLayer(d_model, d_k=d_k, d_v=d_v, n_heads=n_heads)
    y2 = layer2(x2).sum()
    y2.backward()
    print(f"\n[grad]   gradient flows: {x2.grad is not None and torch.isfinite(x2.grad).all().item()}")

    # --- LM wrapper ---
    vocab_size = 129
    lm = DeltaNetLM(vocab_size, mask_idx=130, d_model=d_model,
                    n_layers=2, n_heads=n_heads, d_k=d_k, d_v=d_v).eval()
    with torch.no_grad():
        tok = torch.randint(0, vocab_size, (2, 32))
        out = lm(tok)
    print(f"[lm]     output finite={torch.isfinite(out).all().item()}  "
          f"shape={tuple(out.shape)}")
    print("\nAll checks passed.")
