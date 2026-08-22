"""Ablatable CRSA residual graft for streamed DeepSeek-V4 hidden histories.

The graft is deliberately a sidecar, not a replacement decoder block.  It
applies immer's fixed 2-Local + 1-Balanced + 1-Free CRSA program to contextual
hidden histories and adds the resulting context through a scalar residual.
There is no fitted readout, A1 state, cross-model projection, or checkpoint
parameter in this module.

DeepSeek-V4 exposes hyper-connection histories as ``[B, S, HC, D]``.  Each HC
stream is routed independently (by folding HC into the batch dimension), so a
graft cannot leak information between hyper-connection branches.  Plain
``[B, S, D]`` histories are accepted as well.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Any

import torch
from torch import Tensor, nn

from ...attention.crsa.operators import (
    AttentionSpec,
    apply_attention,
    role_complete_attention,
)


GRAFT_EVIDENCE_SCHEMA = "immer.deepseek-v4-crsa-graft/v1"
GRAFT_MODES = frozenset({"off", "crsa", "softmax", "shuffle"})


@dataclass(frozen=True, slots=True)
class GraftEvidence:
    """Small, serialisable evidence record for one graft application."""

    schema: str
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
    sidecar_rms: float
    output_rms: float
    quadratic_attention_bytes: int
    estimated_peak_bytes: int
    head_entropy: tuple[float, ...]
    learned_parameters: int = 0
    uses_a1_ridge: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation without retaining tensors."""

        payload = asdict(self)
        payload["input_shape"] = list(self.input_shape)
        payload["head_entropy"] = list(self.head_entropy)
        return payload


def _rms(tensor: Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    # MPS has no float64 kernels.  Transfer in a supported dtype first; the
    # evidence-only reduction can then use float64 on CPU.
    values = tensor.detach().to(device="cpu", dtype=torch.float32).to(torch.float64)
    return float(values.square().mean().sqrt())


def _prefix_shuffle(weights: Tensor, seed: int) -> Tensor:
    """Permute each causal prefix while retaining support and row marginals.

    A separate deterministic permutation is used for every prefix length.  It
    is intentionally generated outside torch's global RNG, making the placebo
    reproducible without mutating application random state.  Only columns
    ``0..query`` are permuted; future columns remain exactly zero.
    """

    shuffled = torch.zeros_like(weights)
    sequence = weights.shape[-1]
    for query in range(sequence):
        prefix = query + 1
        order = list(range(prefix))
        # Mix the prefix length into the seed; Python's local RNG is stable for
        # integer seeds and does not touch torch/random module global state.
        random.Random(int(seed) ^ (prefix * 0x9E3779B1)).shuffle(order)
        indices = torch.tensor(order, dtype=torch.long, device=weights.device)
        shuffled[..., query, :prefix] = weights[..., query, :prefix].index_select(
            -1, indices
        )
    return shuffled


class DeepSeekV4CrsaGraft(nn.Module):
    """Parameter-free, strictly causal residual sidecar.

    ``crsa`` uses the repository's canonical role-complete operator, ``softmax``
    is its causal-softmax ablation, and ``shuffle`` preserves CRSA row weights
    while permuting their assignment inside each already-visible prefix.  The
    shuffle therefore acts as a causal alignment placebo rather than leaking a
    future token.

    ``max_history`` is a hard allocation guard.  The current CRSA operator is
    quadratic in the short history length; rejecting longer histories keeps a
    malformed request from consuming unbounded unified memory.
    """

    def __init__(
        self,
        *,
        mode: str = "off",
        alpha: float = 0.0,
        heads: int = 4,
        max_history: int = 256,
        slope: float = 0.8,
        diagonal_debit: float = 3.0,
        shuffle_seed: int = 17,
    ) -> None:
        super().__init__()
        self.mode = self._validate_mode(mode)
        self.alpha = self._validate_alpha(alpha)
        if heads != 4:
            raise ValueError(
                "the measured CRSA graft requires exactly four heads "
                "(2 Local + 1 Balanced + 1 Free)"
            )
        if (
            isinstance(max_history, bool)
            or int(max_history) != max_history
            or max_history < 1
        ):
            raise ValueError("max_history must be a positive integer")
        if not math.isfinite(float(slope)) or float(slope) < 0.0:
            raise ValueError("slope must be finite and non-negative")
        if not math.isfinite(float(diagonal_debit)) or float(diagonal_debit) < 0.0:
            raise ValueError("diagonal_debit must be finite and non-negative")
        self.heads = int(heads)
        self.max_history = int(max_history)
        self.shuffle_seed = int(shuffle_seed)
        self.spec = AttentionSpec(
            kind="role_complete",
            local_heads=2,
            balanced_heads=1,
            free_heads=1,
            slope=float(slope),
            diagonal_debit=float(diagonal_debit),
        )

    @staticmethod
    def _validate_mode(mode: str) -> str:
        value = str(mode).lower()
        if value not in GRAFT_MODES:
            choices = ", ".join(sorted(GRAFT_MODES))
            raise ValueError(f"mode must be one of {choices}, got {mode!r}")
        return value

    @staticmethod
    def _validate_alpha(alpha: float) -> float:
        value = float(alpha)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("alpha must be finite and non-negative")
        return value

    @staticmethod
    def _shape(hidden: Tensor) -> tuple[int, int, int, int, bool]:
        if hidden.ndim == 3:
            batch, sequence, width = hidden.shape
            histories = 1
            has_hyper_connections = False
        elif hidden.ndim == 4:
            batch, sequence, histories, width = hidden.shape
            has_hyper_connections = True
        else:
            raise ValueError("hidden must have shape [B, S, D] or [B, S, HC, D]")
        if min(batch, sequence, histories, width) < 1:
            raise ValueError("all hidden dimensions must be positive")
        if not hidden.is_floating_point():
            raise TypeError("hidden histories must use a floating-point dtype")
        return batch, sequence, histories, width, has_hyper_connections

    @staticmethod
    def estimate_peak_bytes(
        shape: tuple[int, ...],
        *,
        element_size: int = 4,
        heads: int = 4,
    ) -> int:
        """Conservative peak allocation estimate for admission/telemetry."""

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
        matrix = independent * heads * sequence * sequence * element_size
        history = independent * sequence * width * element_size
        # logits, weights, an operator work matrix, plus input/context/output.
        return 3 * matrix + 3 * history

    def _identity_evidence(
        self,
        hidden: Tensor,
        *,
        mode: str,
        alpha: float,
        histories: int,
        width: int,
    ) -> GraftEvidence:
        batch, sequence = hidden.shape[:2]
        independent = batch * histories
        head_dim = width // self.heads if width % self.heads == 0 else 0
        return GraftEvidence(
            schema=GRAFT_EVIDENCE_SCHEMA,
            mode=mode,
            operator="off" if mode == "off" else "identity(alpha=0)",
            alpha=alpha,
            input_shape=tuple(hidden.shape),
            independent_histories=independent,
            heads=self.heads,
            head_dim=head_dim,
            max_history=self.max_history,
            identity=True,
            strict_causal=True,
            future_weight_max_abs=0.0,
            row_sum_max_error=0.0,
            input_rms=_rms(hidden),
            sidecar_rms=0.0,
            output_rms=_rms(hidden),
            quadratic_attention_bytes=0,
            estimated_peak_bytes=0,
            head_entropy=(),
        )

    def _canonical(
        self,
        hidden: Tensor,
        *,
        histories: int,
        has_hyper_connections: bool,
    ) -> Tensor:
        if not has_hyper_connections:
            return hidden
        # [B,S,HC,D] -> [B,HC,S,D] -> [B*HC,S,D]
        return hidden.permute(0, 2, 1, 3).reshape(
            hidden.shape[0] * histories, hidden.shape[1], hidden.shape[-1]
        )

    @staticmethod
    def _restore(
        canonical: Tensor,
        *,
        batch: int,
        histories: int,
        has_hyper_connections: bool,
    ) -> Tensor:
        if not has_hyper_connections:
            return canonical
        return canonical.reshape(
            batch, histories, canonical.shape[1], canonical.shape[2]
        ).permute(0, 2, 1, 3)

    def _weights(self, canonical: Tensor, *, mode: str) -> Tensor:
        head_dim = canonical.shape[-1] // self.heads
        values = canonical.reshape(
            canonical.shape[0], canonical.shape[1], self.heads, head_dim
        ).transpose(1, 2)
        logits = torch.matmul(values, values.transpose(-1, -2)) / math.sqrt(head_dim)
        if mode == "softmax":
            return apply_attention(logits, AttentionSpec(kind="softmax"))
        routed = role_complete_attention(logits, self.spec)
        if mode == "shuffle":
            return _prefix_shuffle(routed, self.shuffle_seed)
        return routed

    def attention_weights(self, hidden: Tensor, *, mode: str | None = None) -> Tensor:
        """Expose the causal attention matrix for diagnostics and ablations.

        For a hyper-connection input the returned shape is ``[B, HC, H, S, S]``;
        otherwise it is ``[B, H, S, S]``.
        """

        batch, sequence, histories, width, hyper = self._shape(hidden)
        selected = self._validate_mode(self.mode if mode is None else mode)
        if selected == "off":
            raise ValueError("mode='off' has no attention weights")
        if sequence > self.max_history:
            raise ValueError(
                f"history length {sequence} exceeds max_history={self.max_history}"
            )
        if width % self.heads:
            raise ValueError(f"hidden width {width} must be divisible by {self.heads}")
        canonical = self._canonical(
            hidden, histories=histories, has_hyper_connections=hyper
        )
        weights = self._weights(canonical, mode=selected)
        if hyper:
            return weights.reshape(batch, histories, self.heads, sequence, sequence)
        return weights

    def forward(
        self,
        hidden: Tensor,
        *,
        mode: str | None = None,
        alpha: float | None = None,
        return_evidence: bool = False,
    ) -> Tensor | tuple[Tensor, GraftEvidence]:
        batch, sequence, histories, width, hyper = self._shape(hidden)
        selected = self._validate_mode(self.mode if mode is None else mode)
        strength = self._validate_alpha(self.alpha if alpha is None else alpha)

        # This early branch is intentional: the kill switch returns the exact
        # same object even for NaNs, unusual strides, or widths not divisible by
        # four.  It is a true zero-risk ablation, not a floating-point no-op.
        if selected == "off" or strength == 0.0:
            evidence = self._identity_evidence(
                hidden,
                mode=selected,
                alpha=strength,
                histories=histories,
                width=width,
            )
            return (hidden, evidence) if return_evidence else hidden

        if sequence > self.max_history:
            estimate = self.estimate_peak_bytes(
                tuple(hidden.shape),
                element_size=hidden.element_size(),
                heads=self.heads,
            )
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
        heads = canonical.reshape(
            batch * histories, sequence, self.heads, head_dim
        ).transpose(1, 2)
        weights = self._weights(canonical, mode=selected)
        context = (
            torch.matmul(weights, heads)
            .transpose(1, 2)
            .contiguous()
            .reshape(batch * histories, sequence, width)
        )
        restored = self._restore(
            context,
            batch=batch,
            histories=histories,
            has_hyper_connections=hyper,
        )
        sidecar = restored * strength
        output = hidden + sidecar

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
        matrix_bytes = weights.numel() * weights.element_size()
        operators = {
            "crsa": "role_complete(2local+1balanced+1free)",
            "softmax": "causal_softmax",
            "shuffle": "role_complete_prefix_shuffle_placebo",
        }
        evidence = GraftEvidence(
            schema=GRAFT_EVIDENCE_SCHEMA,
            mode=selected,
            operator=operators[selected],
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
            input_rms=_rms(hidden),
            sidecar_rms=_rms(sidecar),
            output_rms=_rms(output),
            quadratic_attention_bytes=matrix_bytes,
            estimated_peak_bytes=self.estimate_peak_bytes(
                tuple(hidden.shape),
                element_size=hidden.element_size(),
                heads=self.heads,
            ),
            head_entropy=tuple(float(value) for value in per_head_entropy),
        )
        return output, evidence


# Short alias for runtime call sites while retaining the architecture-specific
# public name for discovery and documentation.
CrsaResidualGraft = DeepSeekV4CrsaGraft


__all__ = [
    "CrsaResidualGraft",
    "DeepSeekV4CrsaGraft",
    "GRAFT_EVIDENCE_SCHEMA",
    "GRAFT_MODES",
    "GraftEvidence",
]
