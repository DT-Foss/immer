"""Trainable Seed v3 neural core and proposal-only inference surface."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .config import SeedV3Config


class CausalPrefixSinkhornAttention(nn.Module):
    """Causal Local/Prefix-Balanced/Free attention from Seed v0.3.

    The prefix-balanced arm divides each causal attention row by cumulative
    prefix usage before re-normalizing it.  Remaining heads stay Free, while
    every role retains a hard zero-future mask.
    """

    def __init__(self, config: SeedV3Config) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.n_local_heads = config.n_local_heads
        self.n_balanced_heads = config.n_balanced_heads
        self.n_free_heads = config.free_head_count
        self.local_window = config.local_window
        self.alpha = float(config.crsa_alpha)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    @staticmethod
    def _normalize(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=-1)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch, length, _ = x.shape
        qkv = self.qkv(x).view(
            batch, length, 3, self.n_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.clamp(-30.0, 30.0)

        positions = torch.arange(length, device=x.device)
        causal_2d = positions[None, :] <= positions[:, None]
        local_2d = causal_2d & (
            (positions[:, None] - positions[None, :]) < self.local_window
        )
        key_valid = attention_mask[:, None, None, :]
        causal = causal_2d[None, None] & key_valid
        local = local_2d[None, None] & key_valid

        base = self._normalize(scores, causal)
        weights = base.clone()
        if self.n_local_heads:
            local_slice = slice(0, self.n_local_heads)
            weights[:, local_slice] = self._normalize(
                scores[:, local_slice], local
            )

        balanced_start = self.n_local_heads
        balanced_end = balanced_start + self.n_balanced_heads
        if balanced_end > balanced_start:
            balanced_base = base[:, balanced_start:balanced_end]
            prefix_usage = torch.cumsum(balanced_base, dim=-2)
            balanced = balanced_base / prefix_usage.clamp_min(1e-12).pow(
                self.alpha
            )
            balanced = torch.where(causal, balanced, torch.zeros_like(balanced))
            balanced = balanced / balanced.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
            weights[:, balanced_start:balanced_end] = balanced

        weights = torch.where(causal, weights, torch.zeros_like(weights))
        query_valid = attention_mask[:, None, :, None]
        weights = weights * query_valid.to(weights.dtype)
        weights = torch.where(
            query_valid,
            weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12),
            weights,
        )
        context = self.dropout(weights) @ value
        context = context.transpose(1, 2).contiguous().view(
            batch, length, self.d_model
        )
        output = self.out(context) * attention_mask.unsqueeze(-1).to(context.dtype)
        if return_weights:
            return output, weights
        return output


class KeyedSwiGLU(nn.Module):
    """SwiGLU whose selected gate and up channels form a joint microkey."""

    def __init__(self, config: SeedV3Config) -> None:
        super().__init__()
        self.key_channels = config.key_channels
        self.gate = nn.Linear(config.d_model, config.mlp_hidden, bias=False)
        self.up = nn.Linear(config.d_model, config.mlp_hidden, bias=False)
        self.down = nn.Linear(config.mlp_hidden, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = self.gate(x)
        up = self.up(x)
        key = torch.cat(
            (gate[..., : self.key_channels], up[..., : self.key_channels]), dim=-1
        )
        return self.down(F.silu(gate) * up), key


class SeedV3Block(nn.Module):
    def __init__(self, config: SeedV3Config) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attention = CausalPrefixSinkhornAttention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.mlp = KeyedSwiGLU(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attended = self.attention(self.norm1(x), attention_mask)
        x = x + self.dropout(attended)
        mlp, key = self.mlp(self.norm2(x))
        return x + self.dropout(mlp), key


@dataclass(frozen=True, slots=True)
class SeedV3Output:
    hidden: torch.Tensor
    lm_logits: torch.Tensor
    operator_logits: torch.Tensor
    state_logits: torch.Tensor
    final_logits: torch.Tensor
    novelty_logits: torch.Tensor
    predictive_logits: torch.Tensor
    quotient_logits: torch.Tensor
    route_values: torch.Tensor
    expected_work: torch.Tensor
    mlp_keys: torch.Tensor


@dataclass(frozen=True, slots=True)
class SeedV3Proposal:
    """Uncommitted answer-position values for downstream IMMER verifiers."""

    operator_logits: torch.Tensor
    novelty_logits: torch.Tensor
    predictive_logits: torch.Tensor
    quotient_logits: torch.Tensor
    route_values: torch.Tensor
    expected_work: torch.Tensor
    mlp_keys: torch.Tensor
    authority: Literal["proposal-only"] = "proposal-only"


class ImmerSeedV3(nn.Module):
    """GRU + CRSA Seed that can propose routes but cannot authorize them."""

    def __init__(self, config: SeedV3Config) -> None:
        super().__init__()
        if not isinstance(config, SeedV3Config):
            raise TypeError("config must be a SeedV3Config")
        self.config = config
        # ``cfg`` keeps the established v0.3 training interface intact.
        self.cfg = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.receipt_projection = (
            nn.Linear(config.receipt_feature_dim, config.d_model, bias=False)
            if config.receipt_feature_dim is not None
            else None
        )
        self.gru = nn.GRU(
            config.d_model,
            config.d_model,
            num_layers=config.gru_layers,
            batch_first=True,
        )
        self.blocks = nn.ModuleList(
            SeedV3Block(config) for _ in range(config.n_layers)
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        # Operator and novelty retain the v0.3 contextual-state + token-residual
        # contract.  No raw embedding-only classifier can authorize a route.
        self.operator_head = nn.Linear(
            config.d_model, config.operator_classes, bias=False
        )
        self.novelty_head = nn.Linear(config.d_model, 2, bias=False)
        self.state_head = nn.Linear(config.d_model, config.state_classes)
        self.final_head = nn.Linear(config.d_model, config.state_classes)
        self.predictive_head = nn.Linear(
            config.d_model, config.predictive_classes
        )
        self.quotient_head = nn.Linear(config.d_model, config.quotient_classes)
        self.route_value_head = nn.Linear(config.d_model, config.route_count)
        self.expected_work_head = nn.Linear(config.d_model, config.route_count)

    @staticmethod
    def _validated_sequence(
        config: SeedV3Config,
        attention_mask: torch.Tensor,
        answer_pos: torch.Tensor,
        *,
        batch: int,
        length: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if length <= 0 or length > config.max_seq_len:
            raise ValueError("sequence length is outside the config")
        if attention_mask.shape != (batch, length):
            raise ValueError("attention_mask must match the input sequence")
        if attention_mask.device != device:
            attention_mask = attention_mask.to(device=device)
        if attention_mask.dtype != torch.bool:
            if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
                raise ValueError("attention_mask must contain only zero or one")
            attention_mask = attention_mask.to(torch.bool)
        if length > 1 and bool(
            ((~attention_mask[:, :-1]) & attention_mask[:, 1:]).any()
        ):
            raise ValueError("attention_mask must be a right-padded causal prefix")
        if answer_pos.shape != (batch,):
            raise ValueError("answer_pos must have shape [batch]")
        if answer_pos.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError("answer_pos must contain integers")
        answer_pos = answer_pos.to(device=device, dtype=torch.long)
        if bool(((answer_pos < 0) | (answer_pos >= length)).any()):
            raise ValueError("answer_pos is outside the sequence")
        rows = torch.arange(batch, device=device)
        if not bool(attention_mask[rows, answer_pos].all()):
            raise ValueError("answer_pos must select an unmasked token")
        return attention_mask, answer_pos

    @classmethod
    def _validated_inputs(
        cls,
        config: SeedV3Config,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        answer_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError("input_ids must contain integers")
        batch, length = input_ids.shape
        attention_mask, answer_pos = cls._validated_sequence(
            config,
            attention_mask,
            answer_pos,
            batch=batch,
            length=length,
            device=input_ids.device,
        )
        if bool(((input_ids < 0) | (input_ids >= config.vocab_size)).any()):
            raise ValueError("input_ids contain an out-of-vocabulary token")
        return input_ids.to(torch.long), attention_mask, answer_pos

    def _forward_projected(
        self,
        input_residual: torch.Tensor,
        attention_mask: torch.Tensor,
        answer_pos: torch.Tensor,
    ) -> SeedV3Output:
        batch, length, _ = input_residual.shape
        positions = torch.arange(length, device=input_residual.device)
        x, _ = self.gru(
            input_residual + self.position_embedding(positions).unsqueeze(0)
        )
        keys: list[torch.Tensor] = []
        for block in self.blocks:
            x, key = block(x, attention_mask)
            keys.append(key)
        x = self.norm(x)
        rows = torch.arange(batch, device=x.device)
        final_hidden = x[rows, answer_pos]
        semantic_hidden = x + input_residual
        route_values = self.route_value_head(final_hidden)
        expected_work = F.softplus(self.expected_work_head(final_hidden))
        return SeedV3Output(
            hidden=x,
            lm_logits=self.lm_head(x),
            operator_logits=self.operator_head(semantic_hidden),
            state_logits=self.state_head(x),
            final_logits=self.final_head(final_hidden),
            novelty_logits=self.novelty_head(semantic_hidden),
            predictive_logits=self.predictive_head(x),
            quotient_logits=self.quotient_head(x),
            route_values=route_values,
            expected_work=expected_work,
            mlp_keys=torch.cat(keys, dim=-1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        answer_pos: torch.Tensor,
    ) -> SeedV3Output:
        input_ids, attention_mask, answer_pos = self._validated_inputs(
            self.config, input_ids, attention_mask, answer_pos
        )
        token_only = self.token_embedding(input_ids)
        return self._forward_projected(token_only, attention_mask, answer_pos)

    def forward_receipts(
        self,
        receipt_features: torch.Tensor,
        attention_mask: torch.Tensor,
        answer_pos: torch.Tensor,
    ) -> SeedV3Output:
        """Run the same causal core over pinned continuous receipt features."""

        if self.receipt_projection is None or self.config.receipt_feature_dim is None:
            raise RuntimeError("receipt_feature_dim is not configured")
        if receipt_features.ndim != 3:
            raise ValueError("receipt_features must have shape [batch, sequence, feature]")
        if not receipt_features.is_floating_point():
            raise TypeError("receipt_features must be floating point")
        batch, length, feature_dim = receipt_features.shape
        if feature_dim != self.config.receipt_feature_dim:
            raise ValueError("receipt feature ABI does not match receipt_feature_dim")
        if not bool(torch.isfinite(receipt_features).all()):
            raise ValueError("receipt_features must be finite")
        attention_mask, answer_pos = self._validated_sequence(
            self.config,
            attention_mask,
            answer_pos,
            batch=batch,
            length=length,
            device=receipt_features.device,
        )
        projected = self.receipt_projection(receipt_features)
        return self._forward_projected(projected, attention_mask, answer_pos)

    @torch.no_grad()
    def propose(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        answer_pos: torch.Tensor,
    ) -> SeedV3Proposal:
        """Emit shadow values; authority remains with O1/Qwen/verifiers."""

        if self.training:
            raise RuntimeError("propose requires eval mode")
        output = self.forward(input_ids, attention_mask, answer_pos)
        positions = answer_pos.to(device=input_ids.device, dtype=torch.long)
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        return SeedV3Proposal(
            operator_logits=output.operator_logits[rows, positions],
            novelty_logits=output.novelty_logits[rows, positions],
            predictive_logits=output.predictive_logits[rows, positions],
            quotient_logits=output.quotient_logits[rows, positions],
            route_values=output.route_values,
            expected_work=output.expected_work,
            mlp_keys=output.mlp_keys[rows, positions],
        )

    @torch.no_grad()
    def propose_receipts(
        self,
        receipt_features: torch.Tensor,
        attention_mask: torch.Tensor,
        answer_pos: torch.Tensor,
    ) -> SeedV3Proposal:
        """Emit shadow proposals from authenticated continuous features."""

        if self.training:
            raise RuntimeError("propose_receipts requires eval mode")
        output = self.forward_receipts(
            receipt_features, attention_mask, answer_pos
        )
        positions = answer_pos.to(device=receipt_features.device, dtype=torch.long)
        rows = torch.arange(receipt_features.shape[0], device=receipt_features.device)
        return SeedV3Proposal(
            operator_logits=output.operator_logits[rows, positions],
            novelty_logits=output.novelty_logits[rows, positions],
            predictive_logits=output.predictive_logits[rows, positions],
            quotient_logits=output.quotient_logits[rows, positions],
            route_values=output.route_values,
            expected_work=output.expected_work,
            mlp_keys=output.mlp_keys[rows, positions],
        )


# Compatibility with the name used by the shared v0.3 training lab.
ImmerSeedModel = ImmerSeedV3
SeedOutput = SeedV3Output


__all__ = [
    "CausalPrefixSinkhornAttention",
    "ImmerSeedModel",
    "ImmerSeedV3",
    "KeyedSwiGLU",
    "SeedOutput",
    "SeedV3Block",
    "SeedV3Output",
    "SeedV3Proposal",
]
