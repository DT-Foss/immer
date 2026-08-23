"""Strict text-decoder configuration for the pinned Qwen3.8-27B checkpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


OFFICIAL_REPO_ID = "Qwen/Qwen3.8-27B"
OFFICIAL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


class Qwen38ConfigError(ValueError):
    """A config or source identity cannot drive the Qwen3.8 runtime safely."""


def _positive(raw: Mapping[str, Any], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Qwen38ConfigError(f"{key} must be a positive integer")
    return value


def _nonnegative(raw: Mapping[str, Any], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Qwen38ConfigError(f"{key} must be a non-negative integer")
    return value


def _optional_int(raw: Mapping[str, Any], key: str) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Qwen38ConfigError(f"{key} must be a non-negative integer or null")
    return value


def _finite_float(raw: Mapping[str, Any], key: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Qwen38ConfigError(f"{key} must be a finite number")
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:
        raise Qwen38ConfigError(f"{key} must be a finite number")
    return result


def _boolean(raw: Mapping[str, Any], key: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise Qwen38ConfigError(f"{key} must be a boolean")
    return value


def _string(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise Qwen38ConfigError(f"{key} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class Qwen38Config:
    """Executable contract for the text-only part of Qwen3.8-27B.

    ``from_mapping`` accepts either the repository's outer multimodal config or
    its extracted ``text_config``.  The default is deliberately strict: a
    changed checkpoint must be inspected instead of silently running with stale
    shape or layer-layout assumptions.  Tiny formula tests may explicitly pass
    ``require_official=False`` while retaining all relational validation.
    """

    vocab_size: int
    dim: int
    intermediate_size: int
    n_layers: int
    layer_types: tuple[str, ...]
    n_heads: int
    n_kv_heads: int
    head_dim: int
    partial_rotary_factor: float
    full_attention_interval: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    mrope_interleaved: bool
    mrope_section: tuple[int, ...]
    attention_bias: bool
    attention_dropout: float
    attn_output_gate: bool
    output_gate_type: str
    hidden_act: str
    checkpoint_dtype: str
    mamba_ssm_dtype: str
    mtp_num_hidden_layers: int
    mtp_use_dedicated_embeddings: bool
    tie_word_embeddings: bool
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int | None

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def hidden_size(self) -> int:
        return self.dim

    @property
    def num_hidden_layers(self) -> int:
        return self.n_layers

    @property
    def num_attention_heads(self) -> int:
        return self.n_heads

    @property
    def num_key_value_heads(self) -> int:
        return self.n_kv_heads

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        require_official: bool = True,
    ) -> "Qwen38Config":
        source = Path(path)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise Qwen38ConfigError(f"cannot read Qwen3.8 config: {source}") from exc
        if not isinstance(raw, Mapping):
            raise Qwen38ConfigError("Qwen3.8 config root must be an object")
        return cls.from_mapping(raw, require_official=require_official)

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        require_official: bool = True,
    ) -> "Qwen38Config":
        if not isinstance(raw, Mapping):
            raise Qwen38ConfigError("Qwen3.8 config root must be an object")

        text_raw: Mapping[str, Any]
        if "text_config" in raw:
            if raw.get("model_type") != "qwen3_5":
                raise Qwen38ConfigError("outer model_type must be 'qwen3_5'")
            architectures = raw.get("architectures")
            if not isinstance(architectures, list) or (
                "Qwen3_5ForConditionalGeneration" not in architectures
            ):
                raise Qwen38ConfigError(
                    "checkpoint is not Qwen3_5ForConditionalGeneration"
                )
            candidate = raw.get("text_config")
            if not isinstance(candidate, Mapping):
                raise Qwen38ConfigError("text_config must be an object")
            text_raw = candidate
            outer_tie = raw.get("tie_word_embeddings")
            if outer_tie is not None and outer_tie is not False:
                raise Qwen38ConfigError("outer tied embeddings are unsupported")
        else:
            text_raw = raw

        if text_raw.get("model_type") != "qwen3_5_text":
            raise Qwen38ConfigError("text model_type must be 'qwen3_5_text'")

        n_layers = _positive(text_raw, "num_hidden_layers")
        raw_layer_types = text_raw.get("layer_types")
        if not isinstance(raw_layer_types, list) or len(raw_layer_types) != n_layers:
            raise Qwen38ConfigError(
                "layer_types must have exactly num_hidden_layers entries"
            )
        if any(
            value not in {"linear_attention", "full_attention"}
            for value in raw_layer_types
        ):
            raise Qwen38ConfigError("layer_types contains an unsupported layer")

        rope = text_raw.get("rope_parameters")
        if not isinstance(rope, Mapping):
            raise Qwen38ConfigError("rope_parameters object is required")
        raw_sections = rope.get("mrope_section")
        if not isinstance(raw_sections, list) or not raw_sections:
            raise Qwen38ConfigError("mrope_section must be a non-empty list")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in raw_sections
        ):
            raise Qwen38ConfigError(
                "mrope_section must contain positive integer sections"
            )

        partial_rotary_factor = _finite_float(text_raw, "partial_rotary_factor")
        rope_partial = _finite_float(rope, "partial_rotary_factor")
        if partial_rotary_factor != rope_partial:
            raise Qwen38ConfigError(
                "partial_rotary_factor disagrees with rope_parameters"
            )

        config = cls(
            vocab_size=_positive(text_raw, "vocab_size"),
            dim=_positive(text_raw, "hidden_size"),
            intermediate_size=_positive(text_raw, "intermediate_size"),
            n_layers=n_layers,
            layer_types=tuple(str(value) for value in raw_layer_types),
            n_heads=_positive(text_raw, "num_attention_heads"),
            n_kv_heads=_positive(text_raw, "num_key_value_heads"),
            head_dim=_positive(text_raw, "head_dim"),
            partial_rotary_factor=partial_rotary_factor,
            full_attention_interval=_positive(text_raw, "full_attention_interval"),
            linear_num_key_heads=_positive(text_raw, "linear_num_key_heads"),
            linear_num_value_heads=_positive(text_raw, "linear_num_value_heads"),
            linear_key_head_dim=_positive(text_raw, "linear_key_head_dim"),
            linear_value_head_dim=_positive(text_raw, "linear_value_head_dim"),
            linear_conv_kernel_dim=_positive(text_raw, "linear_conv_kernel_dim"),
            max_position_embeddings=_positive(text_raw, "max_position_embeddings"),
            rms_norm_eps=_finite_float(text_raw, "rms_norm_eps"),
            rope_theta=_finite_float(rope, "rope_theta"),
            mrope_interleaved=_boolean(rope, "mrope_interleaved"),
            mrope_section=tuple(int(value) for value in raw_sections),
            attention_bias=_boolean(text_raw, "attention_bias"),
            attention_dropout=_finite_float(text_raw, "attention_dropout"),
            attn_output_gate=_boolean(text_raw, "attn_output_gate"),
            output_gate_type=_string(text_raw, "output_gate_type"),
            hidden_act=_string(text_raw, "hidden_act"),
            checkpoint_dtype=_string(text_raw, "dtype"),
            mamba_ssm_dtype=_string(text_raw, "mamba_ssm_dtype"),
            mtp_num_hidden_layers=_positive(text_raw, "mtp_num_hidden_layers"),
            mtp_use_dedicated_embeddings=_boolean(
                text_raw, "mtp_use_dedicated_embeddings"
            ),
            tie_word_embeddings=_boolean(text_raw, "tie_word_embeddings"),
            bos_token_id=_nonnegative(text_raw, "bos_token_id"),
            eos_token_id=_nonnegative(text_raw, "eos_token_id"),
            pad_token_id=_optional_int(text_raw, "pad_token_id"),
        )
        config.validate()
        if require_official:
            config.validate_official()
        return config

    def layer_type(self, layer_index: int) -> str:
        if isinstance(layer_index, bool) or not isinstance(layer_index, int):
            raise TypeError("layer_index must be an integer")
        if not 0 <= layer_index < self.n_layers:
            raise IndexError(f"layer_index outside [0, {self.n_layers})")
        return self.layer_types[layer_index]

    def is_full_attention(self, layer_index: int) -> bool:
        return self.layer_type(layer_index) == "full_attention"

    def validate(self) -> None:
        if self.n_heads % self.n_kv_heads:
            raise Qwen38ConfigError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.linear_num_value_heads % self.linear_num_key_heads:
            raise Qwen38ConfigError(
                "linear value heads must be divisible by linear key heads"
            )
        rotary = self.head_dim * self.partial_rotary_factor
        if not 0 < self.partial_rotary_factor <= 1 or not rotary.is_integer():
            raise Qwen38ConfigError("partial rotary dimension must be integral")
        if self.rotary_dim <= 0 or self.rotary_dim % 2:
            raise Qwen38ConfigError(
                "partial rotary dimension must be positive and even"
            )
        if 2 * sum(self.mrope_section) != self.rotary_dim:
            raise Qwen38ConfigError(
                "mrope_section must partition half the rotary dimension"
            )
        expected_layers = tuple(
            "full_attention"
            if (index + 1) % self.full_attention_interval == 0
            else "linear_attention"
            for index in range(self.n_layers)
        )
        if self.layer_types != expected_layers:
            raise Qwen38ConfigError(
                "layer_types must follow the configured linear/full interval"
            )
        if self.rms_norm_eps <= 0 or self.rope_theta <= 0:
            raise Qwen38ConfigError(
                "normalization epsilon and RoPE theta must be positive"
            )
        if self.attention_bias or self.attention_dropout != 0.0:
            raise Qwen38ConfigError(
                "runtime requires bias-free, zero-dropout attention"
            )
        if not self.attn_output_gate or self.output_gate_type != "swish":
            raise Qwen38ConfigError("runtime requires the swish attention output gate")
        if self.hidden_act != "silu":
            raise Qwen38ConfigError("runtime requires the SiLU MLP")
        if self.checkpoint_dtype != "bfloat16":
            raise Qwen38ConfigError("runtime requires BF16 checkpoint tensors")
        if self.mamba_ssm_dtype != "float32":
            raise Qwen38ConfigError("DeltaNet recurrence must use float32 state")
        if self.tie_word_embeddings or self.mtp_use_dedicated_embeddings:
            raise Qwen38ConfigError(
                "runtime requires the published untied embedding layout"
            )

    def validate_official(self) -> None:
        expected: dict[str, Any] = {
            "vocab_size": 248320,
            "dim": 5120,
            "intermediate_size": 17408,
            "n_layers": 64,
            "n_heads": 24,
            "n_kv_heads": 4,
            "head_dim": 256,
            "partial_rotary_factor": 0.25,
            "full_attention_interval": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10_000_000.0,
            "mrope_interleaved": True,
            "mrope_section": (11, 11, 10),
            "mtp_num_hidden_layers": 1,
            "bos_token_id": 248044,
            "eos_token_id": 248044,
            "pad_token_id": None,
        }
        for field, wanted in expected.items():
            actual = getattr(self, field)
            if actual != wanted:
                raise Qwen38ConfigError(
                    f"official Qwen3.8-27B requires {field}={wanted!r}, got {actual!r}"
                )


def validate_source_identity(
    repo_id: object,
    revision: object,
    *,
    require_identity: bool = False,
) -> str:
    """Validate a remote source pin while allowing explicit local fixtures."""

    if repo_id is None and revision is None:
        if require_identity:
            raise Qwen38ConfigError("tensor source has no repo/revision identity")
        return "injected-unverified"
    if not isinstance(repo_id, str) or not isinstance(revision, str):
        raise Qwen38ConfigError("tensor source repo/revision identity is incomplete")
    if repo_id.startswith("local:"):
        return "local"
    if repo_id != OFFICIAL_REPO_ID:
        raise Qwen38ConfigError(
            f"remote Qwen3.8 source must be {OFFICIAL_REPO_ID!r}, got {repo_id!r}"
        )
    if revision != OFFICIAL_REVISION:
        raise Qwen38ConfigError(
            "remote Qwen3.8 source must use pinned revision "
            f"{OFFICIAL_REVISION}, got {revision!r}"
        )
    return "official-pinned"
