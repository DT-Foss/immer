"""Deterministic routing on contextual states from the frozen A1 host.

This module deliberately keeps two concerns separate:

* :class:`RoleCompleteContext` is the fixed CRSA program (2 Local, one
  Balanced, one unrestricted Free head).  It has no learned parameters.
* :class:`RidgeRouteHead` is a small, serialisable readout fitted on those
  contextual features.  It is deterministic and carries its feature schema
  and calibration provenance in the saved payload.

The feature is a residual pair ``[A1 last scan state | CRSA last context]``.
Keeping the raw state is intentional: it is the same residual construction
used by the surrounding host, and makes the CRSA contribution directly
ablatable without pretending that attention replaces the host state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .crsa.operators import AttentionSpec, apply_attention, role_complete_attention
from ..resource_paths import crsa_router_manifest


ROUTER_STATE_SCHEMA = "immer.crsa-a1-router/v1"
DEFAULT_ROUTER_STATE = crsa_router_manifest()
FEATURE_SCHEMA = (
    "a1.layer0.scan:[raw_last|role_complete_last];"
    "heads=2local+1balanced+1free;slope=0.8;diagonal_debit=3"
)


class RoleCompleteContext(nn.Module):
    """Parameter-free fixed-role CRSA over contextual host states.

    Input states have shape ``[batch, time, d_model]``.  Q, K and V are the
    same contextual state split into four complete 32-dimensional heads for
    the canonical 128-dimensional A1 host.  No static embedding mean and no
    cross-model projection is used.
    """

    def __init__(
        self,
        d_model: int = 128,
        *,
        heads: int = 4,
        local_heads: int = 2,
        balanced_heads: int = 1,
        free_heads: int = 1,
        slope: float = 0.8,
        diagonal_debit: float = 3.0,
    ) -> None:
        super().__init__()
        if d_model < 1 or heads < 1 or d_model % heads:
            raise ValueError("d_model must be positive and divisible by heads")
        if (local_heads, balanced_heads, free_heads) != (2, 1, 1):
            raise ValueError("the measured router requires exactly 2 Local + 1 Balanced + 1 Free head")
        if heads != local_heads + balanced_heads + free_heads:
            raise ValueError("role counts must equal the head count")
        self.d_model = int(d_model)
        self.heads = int(heads)
        self.head_dim = self.d_model // self.heads
        self.spec = AttentionSpec(
            kind="role_complete",
            local_heads=local_heads,
            balanced_heads=balanced_heads,
            free_heads=free_heads,
            slope=float(slope),
            diagonal_debit=float(diagonal_debit),
        )

    def _heads(self, states: Tensor) -> Tensor:
        if states.ndim != 3 or states.shape[-1] != self.d_model:
            raise ValueError(
                f"expected contextual states [batch, time, {self.d_model}], got {tuple(states.shape)}"
            )
        if states.shape[0] < 1 or states.shape[1] < 1:
            raise ValueError("batch and time dimensions must be positive")
        return states.reshape(states.shape[0], states.shape[1], self.heads, self.head_dim).transpose(1, 2)

    def attention_logits(self, states: Tensor) -> Tensor:
        heads = self._heads(states)
        return torch.matmul(heads, heads.transpose(-1, -2)) / math.sqrt(self.head_dim)

    def attention_weights(self, states: Tensor, *, operator: str = "role_complete") -> Tensor:
        """Return weights for the measured operator or its softmax ablation."""
        logits = self.attention_logits(states)
        if operator == "role_complete":
            return role_complete_attention(logits, self.spec)
        if operator == "softmax":
            return apply_attention(logits, AttentionSpec(kind="softmax"))
        raise ValueError("operator must be 'role_complete' or 'softmax'")

    def forward(self, states: Tensor, *, operator: str = "role_complete") -> Tensor:
        heads = self._heads(states)
        weights = self.attention_weights(states, operator=operator)
        context = torch.matmul(weights, heads).transpose(1, 2).contiguous()
        return context.reshape(states.shape[0], states.shape[1], self.d_model)

    def route_features(self, states: Tensor, *, operator: str = "role_complete") -> Tensor:
        """Return the measured residual router feature ``[raw_last | context_last]``."""
        context = self(states, operator=operator)
        return torch.cat((states[:, -1], context[:, -1]), dim=-1)

    def description(self) -> dict[str, Any]:
        return {
            "feature_schema": FEATURE_SCHEMA,
            "d_model": self.d_model,
            "heads": self.heads,
            "head_dim": self.head_dim,
            "local_heads": self.spec.local_heads,
            "balanced_heads": self.spec.balanced_heads,
            "free_heads": self.spec.free_heads,
            "slope": self.spec.slope,
            "diagonal_debit": self.spec.diagonal_debit,
        }


def capture_a1_scan_states(host: nn.Module, token_ids: Tensor, *, layer: int = 0) -> Tensor:
    """Read contextual scan outputs from a frozen A1-compatible host.

    The host head is not evaluated: this follows the host forward path only as
    far as the requested scan layer.  For layer zero it is exactly the tensor
    captured by the forward hook used in the SHIP-v6 experiment, while avoiding
    hook lifetime and concurrency hazards.  Both the stateless frozen host's
    tensor return and the streaming twin's ``(tensor, state)`` return are
    accepted; with a zero incoming state their scan sequence is identical.
    """
    if token_ids.ndim != 2 or token_ids.shape[0] < 1 or token_ids.shape[1] < 1:
        raise ValueError("token_ids must have shape [batch, time] with positive dimensions")
    parameters = tuple(host.parameters())
    if any(parameter.requires_grad for parameter in parameters):
        raise ValueError("A1 host must be frozen before router state extraction")
    layers = getattr(host, "layers", None)
    embed = getattr(host, "embed", None)
    if layers is None or embed is None or not 0 <= layer < len(layers):
        raise ValueError("host does not expose the requested A1 scan layer")

    with torch.no_grad():
        hidden = embed(token_ids)
        for index, block in enumerate(layers):
            try:
                result = block.scan(hidden, None)
            except TypeError:
                # The frozen stateless SelectiveNoPE host is bit-equivalent for
                # a zero incoming state but exposes ``scan(hidden) -> Tensor``.
                result = block.scan(hidden)
            scan_states = result[0] if isinstance(result, tuple) else result
            if not isinstance(scan_states, Tensor) or scan_states.shape != hidden.shape:
                raise TypeError("A1 scan must return a contextual sequence matching its input")
            if index == layer:
                return scan_states.detach()
            hidden = block.ln1(hidden + scan_states)
            hidden = block.ln2(hidden + block.ffn(hidden))
    raise AssertionError("unreachable A1 layer traversal")


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """One deterministic binary routing decision."""

    label: str
    index: int
    score: float
    margin: float

    @property
    def is_arithmetic(self) -> bool:
        return self.index == 1


@dataclass(slots=True)
class RidgeRouteHead:
    """Closed-form binary readout with an auditable JSON state."""

    mean: Tensor
    scale: Tensor
    coefficient: Tensor
    intercept: float
    ridge: float = 10.0
    negative_label: str = "text"
    positive_label: str = "arithmetic"
    feature_schema: str = FEATURE_SCHEMA
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mean = self._vector(self.mean, "mean")
        self.scale = self._vector(self.scale, "scale")
        self.coefficient = self._vector(self.coefficient, "coefficient")
        if not (self.mean.shape == self.scale.shape == self.coefficient.shape):
            raise ValueError("mean, scale and coefficient dimensions must match")
        if torch.any(self.scale <= 0):
            raise ValueError("all feature scales must be positive")
        if not math.isfinite(float(self.intercept)) or not math.isfinite(float(self.ridge)):
            raise ValueError("intercept and ridge must be finite")
        if self.ridge <= 0:
            raise ValueError("ridge must be positive")
        if self.negative_label == self.positive_label:
            raise ValueError("route labels must be distinct")
        self.intercept = float(self.intercept)
        self.ridge = float(self.ridge)
        self.metadata = dict(self.metadata)

    @staticmethod
    def _vector(value: Tensor | Sequence[float], name: str) -> Tensor:
        out = torch.as_tensor(value, dtype=torch.float64, device="cpu").detach().clone()
        if out.ndim != 1 or out.numel() < 1 or not bool(torch.isfinite(out).all()):
            raise ValueError(f"{name} must be a non-empty finite vector")
        return out

    @classmethod
    def fit(
        cls,
        features: Tensor,
        labels: Tensor | Sequence[int],
        *,
        ridge: float = 10.0,
        metadata: Mapping[str, Any] | None = None,
        feature_schema: str = FEATURE_SCHEMA,
    ) -> "RidgeRouteHead":
        if not math.isfinite(float(ridge)) or ridge <= 0:
            raise ValueError("ridge must be a positive finite number")
        x = torch.as_tensor(features, dtype=torch.float64, device="cpu").detach()
        y = torch.as_tensor(labels, dtype=torch.int64, device="cpu").detach()
        if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
            raise ValueError("features must have shape [examples, features]")
        if y.shape != (x.shape[0],) or not bool(torch.all((y == 0) | (y == 1))):
            raise ValueError("labels must be a binary vector aligned with features")
        if int(torch.unique(y).numel()) != 2:
            raise ValueError("both route classes are required")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("features must be finite")

        mean = x.mean(dim=0)
        scale = x.std(dim=0, unbiased=False).clamp_min(1e-5)
        standardized = (x - mean) / scale
        design = torch.cat((standardized, torch.ones(x.shape[0], 1, dtype=x.dtype)), dim=1)
        penalty = torch.eye(design.shape[1], dtype=x.dtype)
        penalty[-1, -1] = 0.0
        target = y.to(x.dtype).mul(2.0).sub(1.0)
        solution = torch.linalg.solve(
            design.T @ design + float(ridge) * penalty,
            design.T @ target,
        )
        return cls(
            mean=mean,
            scale=scale,
            coefficient=solution[:-1],
            intercept=float(solution[-1]),
            ridge=float(ridge),
            feature_schema=feature_schema,
            metadata=dict(metadata or {}),
        )

    @property
    def feature_dim(self) -> int:
        return int(self.mean.numel())

    def scores(self, features: Tensor) -> Tensor:
        x = torch.as_tensor(features)
        one = x.ndim == 1
        if one:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[1] != self.feature_dim:
            raise ValueError(f"expected features [batch, {self.feature_dim}]")
        work = x.to(dtype=torch.float64)
        mean = self.mean.to(device=work.device)
        scale = self.scale.to(device=work.device)
        coefficient = self.coefficient.to(device=work.device)
        out = ((work - mean) / scale) @ coefficient + self.intercept
        return out[0] if one else out

    def predict(self, features: Tensor) -> Tensor:
        return (self.scores(features) >= 0.0).to(torch.int64)

    def decide(self, features: Tensor) -> list[RouteDecision]:
        score = self.scores(features)
        if score.ndim == 0:
            score = score.unsqueeze(0)
        decisions: list[RouteDecision] = []
        for value in score.detach().cpu().tolist():
            index = int(value >= 0.0)
            decisions.append(
                RouteDecision(
                    label=self.positive_label if index else self.negative_label,
                    index=index,
                    score=float(value),
                    margin=abs(float(value)),
                )
            )
        return decisions

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": ROUTER_STATE_SCHEMA,
            "feature_schema": self.feature_schema,
            "labels": {"0": self.negative_label, "1": self.positive_label},
            "ridge": self.ridge,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficient": self.coefficient.tolist(),
            "intercept": self.intercept,
            "metadata": self.metadata,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RidgeRouteHead":
        if payload.get("schema") != ROUTER_STATE_SCHEMA:
            raise ValueError("unsupported CRSA router state schema")
        labels = payload.get("labels")
        if not isinstance(labels, Mapping) or "0" not in labels or "1" not in labels:
            raise ValueError("router state is missing binary labels")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("router metadata must be an object")
        return cls(
            mean=payload["mean"],
            scale=payload["scale"],
            coefficient=payload["coefficient"],
            intercept=float(payload["intercept"]),
            ridge=float(payload["ridge"]),
            negative_label=str(labels["0"]),
            positive_label=str(labels["1"]),
            feature_schema=str(payload.get("feature_schema", "")),
            metadata=dict(metadata),
        )

    def digest(self) -> str:
        encoded = json.dumps(
            self.to_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        os.close(fd)
        temporary_path = Path(temporary)
        try:
            temporary_path.write_text(
                json.dumps(self.to_payload(), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_path, target)
        finally:
            temporary_path.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> "RidgeRouteHead":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("router state must be a JSON object")
        return cls.from_payload(payload)


class FrozenA1CrsaRouter:
    """Executable frozen-A1 → fixed-CRSA → deterministic route path."""

    def __init__(
        self,
        head: RidgeRouteHead,
        *,
        context: RoleCompleteContext | None = None,
        layer: int = 0,
    ) -> None:
        self.context = context or RoleCompleteContext()
        self.head = head
        self.layer = int(layer)
        if self.layer != 0:
            raise ValueError("the persisted router was measured only on A1 layer zero")
        if self.head.feature_schema != FEATURE_SCHEMA:
            raise ValueError("router head was fitted for a different feature schema")
        if self.head.feature_dim != 2 * self.context.d_model:
            raise ValueError("router head dimension does not match the CRSA residual feature")

    @classmethod
    def load(
        cls,
        path: str | Path = DEFAULT_ROUTER_STATE,
        *,
        context: RoleCompleteContext | None = None,
    ) -> "FrozenA1CrsaRouter":
        return cls(RidgeRouteHead.load(path), context=context)

    def features(self, host: nn.Module, token_ids: Tensor) -> Tensor:
        states = capture_a1_scan_states(host, token_ids, layer=self.layer)
        return self.context.route_features(states)

    def decide(self, host: nn.Module, token_ids: Tensor) -> list[RouteDecision]:
        return self.head.decide(self.features(host, token_ids))

    def decide_one(self, host: nn.Module, token_ids: Tensor) -> RouteDecision:
        decisions = self.decide(host, token_ids)
        if len(decisions) != 1:
            raise ValueError("decide_one requires a batch of exactly one sequence")
        return decisions[0]


__all__ = [
    "DEFAULT_ROUTER_STATE",
    "FEATURE_SCHEMA",
    "ROUTER_STATE_SCHEMA",
    "FrozenA1CrsaRouter",
    "RidgeRouteHead",
    "RoleCompleteContext",
    "RouteDecision",
    "capture_a1_scan_states",
]
