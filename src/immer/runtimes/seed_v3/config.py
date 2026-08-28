"""Configuration for the optional IMMER Seed v3 shadow proposer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class SeedV3Config:
    """Shape contract shared by training, inference, and checkpoint manifests.

    The defaults preserve the v0.3 micro architecture.  Local and balanced
    heads occupy the leading attention heads; every remaining head is a Free
    causal path.
    """

    vocab_size: int
    d_model: int = 96
    n_layers: int = 3
    n_heads: int = 4
    n_local_heads: int = 1
    n_balanced_heads: int = 1
    mlp_hidden: int = 256
    key_channels: int = 2
    key_code_bits: int = 4
    max_seq_len: int = 64
    local_window: int = 8
    crsa_alpha: float = 1.0
    dropout: float = 0.0
    operator_classes: int = 6
    state_classes: int = 17 * 17
    predictive_classes: int = 17
    quotient_classes: int = 17
    route_count: int = 4
    gru_layers: int = 1
    receipt_feature_dim: int | None = None

    def __post_init__(self) -> None:
        integer_fields = (
            "vocab_size",
            "d_model",
            "n_layers",
            "n_heads",
            "n_local_heads",
            "n_balanced_heads",
            "mlp_hidden",
            "key_channels",
            "key_code_bits",
            "max_seq_len",
            "local_window",
            "operator_classes",
            "state_classes",
            "predictive_classes",
            "quotient_classes",
            "route_count",
            "gru_layers",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")

        if self.receipt_feature_dim is not None:
            if isinstance(self.receipt_feature_dim, bool) or not isinstance(
                self.receipt_feature_dim, int
            ):
                raise TypeError("receipt_feature_dim must be an integer or None")
            if self.receipt_feature_dim <= 0:
                raise ValueError("receipt_feature_dim must be positive when configured")

        positive = (
            "vocab_size",
            "d_model",
            "n_layers",
            "n_heads",
            "mlp_hidden",
            "key_channels",
            "key_code_bits",
            "max_seq_len",
            "local_window",
            "operator_classes",
            "state_classes",
            "predictive_classes",
            "quotient_classes",
            "route_count",
            "gru_layers",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_local_heads < 0 or self.n_balanced_heads < 0:
            raise ValueError("head-role counts cannot be negative")
        if self.n_local_heads + self.n_balanced_heads >= self.n_heads:
            raise ValueError("at least one Free attention head is required")
        if self.key_channels > self.mlp_hidden:
            raise ValueError("key_channels must fit inside mlp_hidden")

        total_joint_key_width = 2 * self.key_channels * self.n_layers
        if total_joint_key_width % self.key_code_bits:
            raise ValueError(
                "concatenated gate/up key width must be divisible by key_code_bits"
            )
        if self.operator_classes > 2**self.key_code_bits:
            raise ValueError("operator_classes exceed the configured key code space")

        for name in ("crsa_alpha", "dropout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.crsa_alpha < 0:
            raise ValueError("crsa_alpha cannot be negative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def free_head_count(self) -> int:
        """Number of structurally unconstrained causal attention paths."""

        return self.n_heads - self.n_local_heads - self.n_balanced_heads

    @property
    def joint_key_width(self) -> int:
        """Joint gate/up microkey width emitted by one Seed block."""

        return 2 * self.key_channels

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SeedV3Config":
        if not isinstance(value, Mapping):
            raise TypeError("Seed v3 config must be a mapping")
        known = set(cls.__dataclass_fields__)
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"unknown Seed v3 config fields: {sorted(unknown)!r}")
        return cls(**dict(value))

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                self.to_dict(),
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "SeedV3Config":
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(document)


# The short name mirrors the v0.3 lab while the explicit name prevents
# ambiguity at wider IMMER runtime boundaries.
SeedConfig = SeedV3Config


__all__ = ["SeedConfig", "SeedV3Config"]
