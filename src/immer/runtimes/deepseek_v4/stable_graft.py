"""Norm-stable CRSA sidecar for DeepSeek-V4 hyper-connection histories.

The original experimental graft applies raw residual histories directly as
queries, keys, and values.  Real V4 activations contain a shared first-token
state whose norm can exceed the answer-token norm by more than three orders of
magnitude.  A row-stochastic attention matrix is therefore not enough to keep
the sidecar bounded.

This implementation treats each CRSA head as a direction field:

``z = h / RMS(h)``
``c = P_CRSA(z)``
``c_hat = c / RMS(c) * RMS(h)``
``h' = RMS(h) * normalize((1-alpha) * h + alpha * c_hat)``

Q, K, and V are all RMS-normalised per token and per head.  The final line
preserves every token/head RMS exactly (up to arithmetic precision), while
``alpha`` remains a bounded convex dose in ``[0, 1]``.  The operator stays
parameter-free, strictly causal, and independent across V4 hyper-connections.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import Tensor

from ...attention.crsa.operators import apply_attention, role_complete_attention
from .graft import (
    DeepSeekV4CrsaGraft,
    _prefix_shuffle,
)


STABLE_GRAFT_EVIDENCE_SCHEMA = "immer.deepseek-v4-stable-crsa-graft/v1"
STABLE_GRAFT_POLICY = "rms-cosine-head-energy/v1"


@dataclass(frozen=True, slots=True)
class StableGraftEvidence:
    """Serializable evidence for one norm-stable graft application."""

    schema: str
    policy: str
    mode: str
    operator: str
    alpha: float
    input_shape: tuple[int, ...]
    independent_histories: int
    heads: int
    head_dim: int
    max_history: int
    identity: bool
    strict_causal: bool
    future_weight_max_abs: float
    row_sum_max_error: float
    input_rms: float
    raw_context_rms: float
    normalized_context_rms: float
    output_rms: float
    max_head_rms_relative_error: float
    quadratic_attention_bytes: int
    estimated_peak_bytes: int
    head_entropy: tuple[float, ...]
    learned_parameters: int = 0
    uses_a1_ridge: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_shape"] = list(self.input_shape)
        payload["head_entropy"] = list(self.head_entropy)
        return payload


def _cpu_rms(tensor: Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    values = tensor.detach().to(device="cpu", dtype=torch.float32).to(torch.float64)
    return float(values.square().mean().sqrt())


class DeepSeekV4StableCrsaGraft(DeepSeekV4CrsaGraft):
    """Cosine-routed, per-head energy-preserving CRSA graft."""

    policy = STABLE_GRAFT_POLICY
    evidence_schema = STABLE_GRAFT_EVIDENCE_SCHEMA

    def __init__(self, *args: Any, rms_eps: float = 1e-6, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not math.isfinite(float(rms_eps)) or float(rms_eps) <= 0.0:
            raise ValueError("rms_eps must be finite and positive")
        if self.alpha > 1.0:
            raise ValueError("stable graft alpha must be in [0, 1]")
        self.rms_eps = float(rms_eps)

    @staticmethod
    def _validate_alpha(alpha: float) -> float:
        value = DeepSeekV4CrsaGraft._validate_alpha(alpha)
        if value > 1.0:
            raise ValueError("stable graft alpha must be in [0, 1]")
        return value

    @staticmethod
    def estimate_peak_bytes(
        shape: tuple[int, ...],
        *,
        element_size: int = 4,
        heads: int = 4,
    ) -> int:
        """Bound FP32 attention matrices and normalised history workspaces."""

        if len(shape) == 3:
            batch, sequence, width = shape
            histories = 1
        elif len(shape) == 4:
            batch, sequence, histories, width = shape
        else:
            raise ValueError("shape must describe [B, S, D] or [B, S, HC, D]")
        if min(batch, sequence, histories, width, element_size, heads) < 1:
            raise ValueError("shape, element_size, and heads must be positive")
        independent = batch * histories
        matrix = independent * heads * sequence * sequence * 4
        history = independent * sequence * width * 4
        # Logits, weights, operator work; heads, unit values, context, scaled
        # context, mixed values, restored FP32 output, and one safety margin.
        return 3 * matrix + 7 * history

    def _normalised_heads(self, canonical: Tensor) -> tuple[Tensor, Tensor]:
        head_dim = canonical.shape[-1] // self.heads
        heads = canonical.reshape(
            canonical.shape[0], canonical.shape[1], self.heads, head_dim
        ).transpose(1, 2)
        heads_fp32 = heads.to(dtype=torch.float32)
        target_rms = heads_fp32.square().mean(-1, keepdim=True).sqrt()
        unit = heads_fp32 / target_rms.clamp_min(self.rms_eps)
        return heads_fp32, unit

    def _weights_for_unit(self, unit: Tensor, *, mode: str) -> Tensor:
        head_dim = unit.shape[-1]
        logits = torch.matmul(unit, unit.transpose(-1, -2)) / math.sqrt(head_dim)
        if mode == "softmax":
            return apply_attention(logits, self.spec.__class__(kind="softmax"))
        routed = role_complete_attention(logits, self.spec)
        if mode == "shuffle":
            return _prefix_shuffle(routed, self.shuffle_seed)
        return routed

    def _weights(self, canonical: Tensor, *, mode: str) -> Tensor:
        _heads, unit = self._normalised_heads(canonical)
        return self._weights_for_unit(unit, mode=mode)

    def _identity_evidence_stable(
        self,
        hidden: Tensor,
        *,
        mode: str,
        alpha: float,
        histories: int,
        width: int,
    ) -> StableGraftEvidence:
        head_dim = width // self.heads if width % self.heads == 0 else 0
        rms = _cpu_rms(hidden)
        return StableGraftEvidence(
            schema=self.evidence_schema,
            policy=self.policy,
            mode=mode,
            operator="off" if mode == "off" else "identity(alpha=0)",
            alpha=alpha,
            input_shape=tuple(hidden.shape),
            independent_histories=hidden.shape[0] * histories,
            heads=self.heads,
            head_dim=head_dim,
            max_history=self.max_history,
            identity=True,
            strict_causal=True,
            future_weight_max_abs=0.0,
            row_sum_max_error=0.0,
            input_rms=rms,
            raw_context_rms=0.0,
            normalized_context_rms=0.0,
            output_rms=rms,
            max_head_rms_relative_error=0.0,
            quadratic_attention_bytes=0,
            estimated_peak_bytes=0,
            head_entropy=(),
        )

    def forward(
        self,
        hidden: Tensor,
        *,
        mode: str | None = None,
        alpha: float | None = None,
        return_evidence: bool = False,
    ) -> Tensor | tuple[Tensor, StableGraftEvidence]:
        batch, sequence, histories, width, hyper = self._shape(hidden)
        selected = self._validate_mode(self.mode if mode is None else mode)
        strength = self._validate_alpha(self.alpha if alpha is None else alpha)
        if selected == "off" or strength == 0.0:
            evidence = self._identity_evidence_stable(
                hidden,
                mode=selected,
                alpha=strength,
                histories=histories,
                width=width,
            )
            return (hidden, evidence) if return_evidence else hidden

        if sequence > self.max_history:
            estimate = self.estimate_peak_bytes(tuple(hidden.shape), heads=self.heads)
            raise ValueError(
                f"history length {sequence} exceeds max_history={self.max_history}; "
                f"uncapped estimated peak would be {estimate} bytes"
            )
        if width % self.heads:
            raise ValueError(f"hidden width {width} must be divisible by {self.heads}")

        canonical = self._canonical(
            hidden, histories=histories, has_hyper_connections=hyper
        )
        head_dim = width // self.heads
        heads, unit = self._normalised_heads(canonical)
        target_rms = heads.square().mean(-1, keepdim=True).sqrt()
        weights = self._weights_for_unit(unit, mode=selected)
        raw_context = torch.matmul(weights, unit)
        context = (
            raw_context
            / raw_context.square().mean(-1, keepdim=True).sqrt().clamp_min(
                self.rms_eps
            )
            * target_rms
        )
        mixed = (1.0 - strength) * heads + strength * context
        output_heads = (
            mixed
            / mixed.square().mean(-1, keepdim=True).sqrt().clamp_min(self.rms_eps)
            * target_rms
        )
        output_canonical = (
            output_heads.transpose(1, 2)
            .contiguous()
            .reshape(batch * histories, sequence, width)
            .to(dtype=hidden.dtype)
        )
        output = self._restore(
            output_canonical,
            batch=batch,
            histories=histories,
            has_hyper_connections=hyper,
        )

        if not return_evidence:
            return output

        future = torch.ones(
            sequence, sequence, dtype=torch.bool, device=weights.device
        ).triu(diagonal=1)
        detached_weights = weights.detach()
        future_max = (
            float(detached_weights.masked_select(future).abs().max())
            if sequence > 1
            else 0.0
        )
        row_error = float((detached_weights.sum(-1) - 1.0).abs().max())
        entropy = -torch.where(
            detached_weights > 0,
            detached_weights
            * detached_weights.clamp_min(
                torch.finfo(detached_weights.dtype).tiny
            ).log(),
            torch.zeros_like(detached_weights),
        ).sum(-1)
        per_head_entropy = (
            entropy.to(device="cpu", dtype=torch.float32)
            .to(torch.float64)
            .mean(dim=(0, 2))
        )
        output_rms_by_head = output_heads.square().mean(-1, keepdim=True).sqrt()
        relative_error = (
            (output_rms_by_head - target_rms).abs()
            / target_rms.clamp_min(self.rms_eps)
        )
        evidence = StableGraftEvidence(
            schema=self.evidence_schema,
            policy=self.policy,
            mode=selected,
            operator={
                "crsa": "rms_role_complete(2local+1balanced+1free)",
                "softmax": "rms_causal_softmax",
                "shuffle": "rms_role_complete_prefix_shuffle_placebo",
            }[selected],
            alpha=strength,
            input_shape=tuple(hidden.shape),
            independent_histories=batch * histories,
            heads=self.heads,
            head_dim=head_dim,
            max_history=self.max_history,
            identity=False,
            strict_causal=future_max == 0.0,
            future_weight_max_abs=future_max,
            row_sum_max_error=row_error,
            input_rms=_cpu_rms(hidden),
            raw_context_rms=_cpu_rms(raw_context),
            normalized_context_rms=_cpu_rms(context),
            output_rms=_cpu_rms(output),
            max_head_rms_relative_error=float(relative_error.max()),
            quadratic_attention_bytes=weights.numel() * weights.element_size(),
            estimated_peak_bytes=self.estimate_peak_bytes(
                tuple(hidden.shape), heads=self.heads
            ),
            head_entropy=tuple(float(value) for value in per_head_entropy),
        )
        return output, evidence


StableCrsaGraft = DeepSeekV4StableCrsaGraft


__all__ = [
    "DeepSeekV4StableCrsaGraft",
    "STABLE_GRAFT_EVIDENCE_SCHEMA",
    "STABLE_GRAFT_POLICY",
    "StableCrsaGraft",
    "StableGraftEvidence",
]
