"""Production o1-state models with no research-module import side effects.

This is the deployed form of the canonical selective rapidity-sqrt recurrence::

    v_t = tanh(W_v x_t)
    g_t = sigmoid(W_gate x_t)
    gamma_t = sigmoid(W_gamma x_t)
    alpha_t = sigmoid(W_alpha x_t)
    a_t = alpha_t * log(1 - clamp((v_t * g_t)^2, max=.999) + 1e-6)
    z_t = gamma_t * z_(t-1) + a_t
    s_t = sqrt(clamp(1 - exp(z_t), min=0) + 1e-6)

Module and parameter names intentionally match the frozen canonical checkpoints.
Unlike the historical research scripts, importing or constructing these classes
does not modify ``sys.path``, torch's MPS predicate, or torch's thread count.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


LOG_COMPLEMENT_CLAMP = 0.999
EPS = 1e-6


def stateful_linear_scan(
    additive: torch.Tensor,
    gamma: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate ``z_t = gamma_t*z_(t-1) + additive_t`` exactly in token order."""

    if additive.ndim != 4 or gamma.shape != additive.shape:
        raise ValueError("additive and gamma must have equal [B,T,H,D] shapes")
    batch, length, heads, width = additive.shape
    if length < 1:
        raise ValueError("o1-state scan requires at least one token")
    if initial_state is None:
        state = torch.zeros(
            batch,
            heads,
            width,
            device=additive.device,
            dtype=additive.dtype,
        )
    else:
        expected = (batch, heads, width)
        if tuple(initial_state.shape) != expected:
            raise ValueError(
                f"incoming o1 state has shape {tuple(initial_state.shape)}, expected {expected}"
            )
        if initial_state.device != additive.device or initial_state.dtype != additive.dtype:
            initial_state = initial_state.to(device=additive.device, dtype=additive.dtype)
        state = initial_state
    sequence = []
    for index in range(length):
        state = gamma[:, index] * state + additive[:, index]
        sequence.append(state)
    return torch.stack(sequence, dim=1), state


class _SelectiveScanBase(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_head: int = 16,
        n_heads: int = 4,
        *,
        causal: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model < 1 or d_head < 1 or n_heads < 1:
            raise ValueError("o1-state dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads
        self.causal = causal
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        total_dim = n_heads * d_head
        # Declaration and initialisation order are checkpoint-significant.
        self.W_v = nn.Linear(d_model, total_dim, bias=False)
        self.W_gate = nn.Linear(d_model, total_dim, bias=False)
        self.W_gamma = nn.Linear(d_model, total_dim, bias=False)
        self.W_alpha = nn.Linear(d_model, total_dim, bias=False)
        self.W_out = nn.Linear(total_dim, d_model, bias=False)
        for module in (self.W_gamma, self.W_alpha):
            nn.init.xavier_uniform_(module.weight, gain=0.1)
        for module in (self.W_v, self.W_gate, self.W_out):
            nn.init.xavier_uniform_(module.weight, gain=0.6)

    def _terms(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"o1-state input must have shape [B,T,{self.d_model}]")
        batch, length, _ = x.shape
        velocity = torch.tanh(self.W_v(x))
        value_gate = torch.sigmoid(self.W_gate(x))
        gamma = torch.sigmoid(self.W_gamma(x))
        alpha = torch.sigmoid(self.W_alpha(x))
        gated = velocity * value_gate
        if self.dropout is not None:
            gated = self.dropout(gated)
        gated = gated.view(batch, length, self.n_heads, self.d_head)
        gamma = gamma.view(batch, length, self.n_heads, self.d_head)
        alpha = alpha.view(batch, length, self.n_heads, self.d_head)
        squared = torch.clamp(gated * gated, max=LOG_COMPLEMENT_CLAMP)
        additive = alpha * torch.log(1.0 - squared + EPS)
        return additive, gamma

    def _readout(self, states: torch.Tensor) -> torch.Tensor:
        bounded_squared = torch.clamp(1.0 - torch.exp(states), min=0.0)
        bounded = torch.sqrt(bounded_squared + EPS)
        flat = bounded.reshape(
            bounded.shape[0], bounded.shape[1], self.n_heads * self.d_head
        )
        return self.W_out(flat)


class SelectiveNoPEScanLayer(_SelectiveScanBase):
    """Canonical stateless scan, including the reference bidirectional control."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        additive, gamma = self._terms(x)
        forward_states, _ = stateful_linear_scan(additive, gamma)
        if self.causal:
            states = forward_states
        else:
            reverse_states, _ = stateful_linear_scan(
                torch.flip(additive, dims=(1,)),
                torch.flip(gamma, dims=(1,)),
            )
            states = forward_states + torch.flip(reverse_states, dims=(1,))
        return self._readout(states)


class StreamingNoPEScanLayer(_SelectiveScanBase):
    """Causal scan that accepts and returns the persistent final ``Z`` state."""

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        if not self.causal:
            raise ValueError("persistent o1-state streaming requires a causal scan")

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        additive, gamma = self._terms(x)
        states, final_state = stateful_linear_scan(additive, gamma, state)
        return self._readout(states), final_state


def _feed_forward(d_model: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_model, 4 * d_model),
        nn.GELU(),
        nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        nn.Linear(4 * d_model, d_model),
    )


class SelectiveNoPETransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_head: int = 16,
        n_heads: int = 4,
        *,
        dropout: float = 0.0,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.scan = SelectiveNoPEScanLayer(
            d_model,
            d_head=d_head,
            n_heads=n_heads,
            causal=causal,
            dropout=dropout,
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = _feed_forward(d_model, dropout)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x + self.scan(x))
        return self.ln2(x + self.ffn(x))


class StreamingNoPETransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_head: int = 16,
        n_heads: int = 4,
        *,
        dropout: float = 0.0,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.scan = StreamingNoPEScanLayer(
            d_model,
            d_head=d_head,
            n_heads=n_heads,
            causal=causal,
            dropout=dropout,
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = _feed_forward(d_model, dropout)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scanned, final_state = self.scan(x, state)
        x = self.ln1(x + scanned)
        return self.ln2(x + self.ffn(x)), final_state


class _LanguageModelBase(nn.Module):
    layer_type: type[nn.Module]

    def __init__(
        self,
        vocab_size: int,
        mask_idx: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        d_head: int = 32,
        seq_len: int = 32,
        dropout: float = 0.1,
        causal: bool = True,
    ) -> None:
        super().__init__()
        if vocab_size < 1 or n_layers < 1 or seq_len < 1:
            raise ValueError("vocabulary, layer count and sequence length must be positive")
        self.mask_idx = mask_idx
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab_size + 2, d_model)
        # Kept as an attribute for API compatibility; Identity has no state and
        # is the defining NoPE ablation used by the frozen A1 checkpoint.
        self.pos = nn.Identity()
        self.layers = nn.ModuleList(
            self.layer_type(
                d_model,
                d_head=d_head,
                n_heads=n_heads,
                dropout=dropout,
                causal=causal,
            )
            for _ in range(n_layers)
        )
        self.head = nn.Linear(d_model, vocab_size + 1)


class SelectiveNoPETransformerLM(_LanguageModelBase):
    """Stateless NoPE language model compatible with frozen A1 state dicts."""

    layer_type = SelectiveNoPETransformerLayer

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.pos(self.embed(tokens))
        for layer in self.layers:
            hidden = layer(hidden)
        return self.head(hidden)


class StreamingNoPELM(_LanguageModelBase):
    """Stateful NoPE model used by the continuously carried life stream."""

    layer_type = StreamingNoPETransformerLayer

    def forward(
        self,
        tokens: torch.Tensor,
        states: Sequence[torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if states is None:
            incoming: Sequence[torch.Tensor | None] = (None,) * len(self.layers)
        else:
            if len(states) != len(self.layers):
                raise ValueError(
                    f"received {len(states)} o1 states for {len(self.layers)} layers"
                )
            incoming = states
        hidden = self.pos(self.embed(tokens))
        outgoing: list[torch.Tensor] = []
        for layer, state in zip(self.layers, incoming, strict=True):
            hidden, final_state = layer(hidden, state)
            outgoing.append(final_state)
        return self.head(hidden), outgoing


__all__ = [
    "EPS",
    "LOG_COMPLEMENT_CLAMP",
    "SelectiveNoPEScanLayer",
    "SelectiveNoPETransformerLM",
    "StreamingNoPELM",
    "StreamingNoPEScanLayer",
    "stateful_linear_scan",
]
