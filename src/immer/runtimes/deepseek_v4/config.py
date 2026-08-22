"""Validated configuration for the official DeepSeek-V4 checkpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class DeepSeekV4ConfigError(ValueError):
    """A checkpoint configuration cannot drive the streamed runtime safely."""


def _positive(raw: Mapping[str, Any], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DeepSeekV4ConfigError(f"{key} must be a positive integer")
    return value


def _nonnegative(raw: Mapping[str, Any], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DeepSeekV4ConfigError(f"{key} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class DeepSeekV4Config:
    """Only the architecture fields used by the executable main decoder.

    Names follow the public ``config.json``.  ``from_mapping`` rejects silent
    architecture drift; a new V4 revision must be inspected before it can be
    executed with old assumptions.
    """

    vocab_size: int
    dim: int
    n_layers: int
    n_heads: int
    head_dim: int
    rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    moe_inter_dim: int
    n_routed_experts: int
    n_shared_experts: int
    n_activated_experts: int
    n_hash_layers: int
    norm_eps: float
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    window_size: int
    compress_ratios: tuple[int, ...]
    rope_theta: float
    compress_rope_theta: float
    max_position_embeddings: int
    original_seq_len: int
    rope_factor: float
    beta_fast: int
    beta_slow: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    score_func: str
    route_scale: float
    swiglu_limit: float
    expert_dtype: str
    target_layer_ids: tuple[int, ...]

    @classmethod
    def from_file(cls, path: str | Path) -> "DeepSeekV4Config":
        source = Path(path)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DeepSeekV4ConfigError(
                f"cannot read DeepSeek-V4 config: {source}"
            ) from exc
        if not isinstance(raw, Mapping):
            raise DeepSeekV4ConfigError("DeepSeek-V4 config root must be an object")
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DeepSeekV4Config":
        if raw.get("model_type") not in {None, "deepseek_v4"}:
            raise DeepSeekV4ConfigError(
                f"model_type must be 'deepseek_v4', got {raw.get('model_type')!r}"
            )
        architectures = raw.get("architectures")
        if architectures is not None and "DeepseekV4ForCausalLM" not in architectures:
            raise DeepSeekV4ConfigError("checkpoint is not DeepseekV4ForCausalLM")

        n_layers = _positive(raw, "num_hidden_layers")
        ratios_raw = raw.get("compress_ratios")
        if not isinstance(ratios_raw, list) or len(ratios_raw) < n_layers:
            raise DeepSeekV4ConfigError(
                "compress_ratios must contain at least num_hidden_layers entries"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in ratios_raw
        ):
            raise DeepSeekV4ConfigError(
                "compress_ratios must contain non-negative integers"
            )

        score_func = str(raw.get("scoring_func", ""))
        if score_func not in {"softmax", "sigmoid", "sqrtsoftplus"}:
            raise DeepSeekV4ConfigError(f"unsupported scoring_func: {score_func!r}")
        expert_dtype = str(raw.get("expert_dtype", ""))
        if expert_dtype != "fp4":
            raise DeepSeekV4ConfigError(
                f"streamed V4 runtime currently requires official fp4 experts, got {expert_dtype!r}"
            )

        rope = raw.get("rope_scaling")
        if not isinstance(rope, Mapping):
            raise DeepSeekV4ConfigError("rope_scaling object is required")
        target_ids = raw.get("dspark_target_layer_ids", [])
        if not isinstance(target_ids, list) or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < n_layers
            for value in target_ids
        ):
            raise DeepSeekV4ConfigError("invalid dspark_target_layer_ids")

        config = cls(
            vocab_size=_positive(raw, "vocab_size"),
            dim=_positive(raw, "hidden_size"),
            n_layers=n_layers,
            n_heads=_positive(raw, "num_attention_heads"),
            head_dim=_positive(raw, "head_dim"),
            rope_head_dim=_positive(raw, "qk_rope_head_dim"),
            q_lora_rank=_positive(raw, "q_lora_rank"),
            o_lora_rank=_positive(raw, "o_lora_rank"),
            o_groups=_positive(raw, "o_groups"),
            moe_inter_dim=_positive(raw, "moe_intermediate_size"),
            n_routed_experts=_positive(raw, "n_routed_experts"),
            n_shared_experts=_positive(raw, "n_shared_experts"),
            n_activated_experts=_positive(raw, "num_experts_per_tok"),
            n_hash_layers=_nonnegative(raw, "num_hash_layers"),
            norm_eps=float(raw.get("rms_norm_eps", 1e-6)),
            hc_mult=_positive(raw, "hc_mult"),
            hc_sinkhorn_iters=_positive(raw, "hc_sinkhorn_iters"),
            hc_eps=float(raw.get("hc_eps", 1e-6)),
            window_size=_positive(raw, "sliding_window"),
            compress_ratios=tuple(int(value) for value in ratios_raw[:n_layers]),
            rope_theta=float(raw.get("rope_theta", 10000.0)),
            compress_rope_theta=float(raw.get("compress_rope_theta", 40000.0)),
            max_position_embeddings=_positive(raw, "max_position_embeddings"),
            original_seq_len=_positive(rope, "original_max_position_embeddings"),
            rope_factor=float(rope.get("factor", 1.0)),
            beta_fast=_positive(rope, "beta_fast"),
            beta_slow=_positive(rope, "beta_slow"),
            index_n_heads=_positive(raw, "index_n_heads"),
            index_head_dim=_positive(raw, "index_head_dim"),
            index_topk=_positive(raw, "index_topk"),
            score_func=score_func,
            route_scale=float(raw.get("routed_scaling_factor", 1.0)),
            swiglu_limit=float(raw.get("swiglu_limit", 0.0)),
            expert_dtype=expert_dtype,
            target_layer_ids=tuple(target_ids),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.n_heads % self.o_groups:
            raise DeepSeekV4ConfigError(
                "num_attention_heads must be divisible by o_groups"
            )
        if self.head_dim <= self.rope_head_dim or self.rope_head_dim % 2:
            raise DeepSeekV4ConfigError(
                "head_dim/rope_head_dim are incompatible with RoPE"
            )
        if self.n_shared_experts != 1:
            raise DeepSeekV4ConfigError(
                "official V4 forward requires exactly one shared expert"
            )
        if not 0 < self.n_activated_experts <= self.n_routed_experts:
            raise DeepSeekV4ConfigError("invalid active/routed expert counts")
        if self.n_hash_layers > self.n_layers:
            raise DeepSeekV4ConfigError("num_hash_layers exceeds decoder depth")
        if self.dim % 128 or self.q_lora_rank % 128 or self.head_dim % 128:
            raise DeepSeekV4ConfigError(
                "official MXFP8 matrix dimensions must be 128-aligned"
            )
        if self.moe_inter_dim % 128:
            raise DeepSeekV4ConfigError(
                "official FP8-activation/MXFP4 expert dimension must be 128-aligned"
            )
        if self.norm_eps <= 0 or self.hc_eps <= 0:
            raise DeepSeekV4ConfigError("normalization epsilons must be positive")
