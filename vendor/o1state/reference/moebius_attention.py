"""
Möbius-Attention: A non-softmax, non-LayerNorm attention mechanism.

Instead of computing Q·K^T and softmax weights, each token projects to:
  - lambda: a "rest eigenvalue" state in (-1, 1)
  - v:      a "velocity" in (-1, 1)
  - g:      a gate in (0, 1)

For each query position i and each head h, the layer performs an associative
Möbius scan over the context (causal: j <= i; full: all j):

    state_{i,h} = lambda_{i,h}
    state_{i,h} = f(state_{i,h}, g_{j,h} * v_{j,h})   for j in context

where f(a, b) = (a + b) / (1 + ab) is the Möbius velocity-addition formula.

Properties:
  - Associative: scan order is irrelevant.
  - Bounded: |f(a,b)| < 1 for |a|,|b| < 1, so no LayerNorm is required.
  - No softmax, no pairwise scoring matrix.

Reference: Foss 2026, "From Markov Chains to Minkowski Space" — the
Möbius-Lorentz correspondence on doubly-stochastic eigenvalue spectra.
"""

import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ps_lifted_scan import ps_lifted_moebius_scan


class SinusoidalPositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MoebiusAttention(nn.Module):
    """Multi-head Möbius-coupling attention layer.

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of independent Möbius heads.
    d_head : int
        Möbius-state dimension per head.
    causal : bool or str
        If True, each position only scans previous positions (including itself).
        If ``"bidirectional"``, combine left and right Möbius scans.
    dropout : float
        Dropout applied to the gated velocity (regularizes the scan).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        d_head: int = 16,
        causal: Union[bool, str] = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.causal = causal
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        self.W_lambda = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_gate = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_out = nn.Linear(n_heads * d_head, d_model, bias=False)

        # Moderate Xavier init; tanh/sigmoid will keep Möbius arguments bounded
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p, gain=0.7)

    @staticmethod
    def moebius_couple(
        a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        """Elementwise Möbius coupling f(a,b) = (a+b)/(1+ab), clamped to (-1,1)."""
        denom = 1.0 + a * b
        denom = torch.sign(denom) * torch.clamp(torch.abs(denom), min=eps)
        return torch.clamp((a + b) / denom, -0.999, 0.999)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        returns: (B, T, d_model)
        """
        B, T, _ = x.shape

        lam = torch.tanh(self.W_lambda(x))           # (B, T, H*Dh)
        v = torch.tanh(self.W_v(x))                  # (B, T, H*Dh)
        gate = torch.sigmoid(self.W_gate(x))         # (B, T, H*Dh)
        v_gated = v * gate
        if self.dropout is not None:
            v_gated = self.dropout(v_gated)

        # Reshape to heads
        lam = lam.view(B, T, self.n_heads, self.d_head)
        v_gated = v_gated.view(B, T, self.n_heads, self.d_head)

        # PS-Lifted parallel scan (O(log T) instead of O(T^2) Python loop)
        out = ps_lifted_moebius_scan(lam, v_gated, causal=self.causal, parallel=True)
        out = out.view(B, T, self.n_heads * self.d_head)
        return self.W_out(out)


class MoebiusTransformerLayer(nn.Module):
    """Single transformer layer using Möbius attention and a tiny FFN.

    No LayerNorm is used because the Möbius state is naturally bounded.
    A residual connection is still applied.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        d_head: int = 16,
        causal: Union[bool, str] = True,
        ffn_dim: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn = MoebiusAttention(
            d_model, n_heads=n_heads, d_head=d_head, causal=causal, dropout=dropout
        )
        ffn_dim = ffn_dim or 4 * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x


class StandardAttention(nn.Module):
    """Multi-head standard scaled-dot-product attention for comparison."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        d_head: int = 16,
        causal: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.causal = causal
        self.dropout = nn.Dropout(dropout)

        self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_out = nn.Linear(n_heads * d_head, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, Dh = self.n_heads, self.d_head

        Q = self.W_q(x).view(B, T, H, Dh).transpose(1, 2) / math.sqrt(Dh)
        K = self.W_k(x).view(B, T, H, Dh).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, Dh).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, H, T, T)
        if self.causal:
            mask = torch.triu(
                torch.ones(T, T, device=x.device) * float("-inf"), diagonal=1
            )
            scores = scores + mask
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = torch.matmul(weights, V)  # (B, H, T, Dh)
        out = out.transpose(1, 2).contiguous().view(B, T, H * Dh)
        return self.W_out(out)


class StandardTransformerLayer(nn.Module):
    """Standard transformer layer with LayerNorm for comparison."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        d_head: int = 16,
        causal: bool = True,
        ffn_dim: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn = StandardAttention(
            d_model, n_heads=n_heads, d_head=d_head, causal=causal, dropout=dropout
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
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ffn(x))
        return x


# ── Smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, T, D = 2, 8, 64
    x = torch.randn(B, T, D)

    print("Möbius Attention smoke test")
    m_attn = MoebiusAttention(D, n_heads=4, d_head=16, causal=True)
    y = m_attn(x)
    print(f"  input shape:  {x.shape}")
    print(f"  output shape: {y.shape}")
    print(f"  output range: [{y.min():.3f}, {y.max():.3f}]")

    print("\nMöbius Transformer Layer smoke test")
    m_layer = MoebiusTransformerLayer(D, n_heads=4, d_head=16, causal=True)
    z = m_layer(x)
    print(f"  output shape: {z.shape}")
    print(f"  output range: [{z.min():.3f}, {z.max():.3f}]")

    print("\nStandard Attention smoke test")
    s_attn = StandardAttention(D, n_heads=4, d_head=16, causal=True)
    y2 = s_attn(x)
    print(f"  output shape: {y2.shape}")
