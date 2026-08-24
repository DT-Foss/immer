"""Native, stateful Head-CRSA intervention for Qwen3.8 layer 27."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, ClassVar

import torch
from torch import Tensor

from ...attention.crsa.operators import AttentionSpec, streaming_prefix_log


NATIVE_HEAD_CRSA_EVIDENCE_SCHEMA = "immer.qwen3.8-native-head-crsa/v1"
NATIVE_HEAD_CRSA_LAYER = 27
NATIVE_HEAD_CRSA_QUERY_HEADS = (2, 8, 14, 20)
NATIVE_HEAD_CRSA_KV_HEADS = (0, 1, 2, 3)
NATIVE_HEAD_CRSA_FREE_HEADS = tuple(
    head for head in range(24) if head not in NATIVE_HEAD_CRSA_QUERY_HEADS
)
# Final Qwen probabilities use BF16.  Half one BF16 relative ULP is the tight
# dtype-independent row-mass bound after summing the stored values in FP32.
NATIVE_HEAD_CRSA_ROW_SUM_TOLERANCE = 2.0**-8


def _real(value: object, name: str, *, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be no greater than {maximum:g}")
    return result


def _integer(value: object, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _outside_support_max_abs(value: Tensor, support: Tensor) -> float:
    """Measure masked mass one head at a time to bound audit peak memory."""

    if bool(support.all().item()):
        return 0.0
    maximum = 0.0
    for head in range(value.shape[1]):
        outside = value[:, head].detach().masked_select(~support[:, head])
        if outside.numel():
            maximum = max(maximum, float(outside.abs().max()))
    return maximum


@dataclass(frozen=True, slots=True)
class NativeHeadCrsaEvidence:
    """Immutable receipt for one native probability intervention."""

    schema: str
    layer: int
    query_start: int
    query_length: int
    key_length: int
    selected_query_heads: tuple[int, ...]
    selected_kv_heads: tuple[int, ...]
    alpha_per_head: tuple[float, ...]
    argmax_changed_queries_per_head: tuple[int, ...]
    mean_l1_probability_delta_per_head: tuple[float, ...]
    free_heads: tuple[int, ...]
    free_head_max_abs_error: float
    future_weight_max_abs: float
    row_sum_max_error: float
    history_length_before: int
    history_length_after: int
    identity: bool

    def __post_init__(self) -> None:
        if self.schema != NATIVE_HEAD_CRSA_EVIDENCE_SCHEMA:
            raise ValueError("native Head-CRSA evidence schema is invalid")
        if self.layer != NATIVE_HEAD_CRSA_LAYER:
            raise ValueError("native Head-CRSA evidence must describe layer 27")
        query_start = _integer(self.query_start, "query_start")
        query_length = _integer(self.query_length, "query_length", positive=True)
        key_length = _integer(self.key_length, "key_length", positive=True)
        if key_length != query_start + query_length:
            raise ValueError("native Head-CRSA evidence is not a contiguous block")
        if self.selected_query_heads != NATIVE_HEAD_CRSA_QUERY_HEADS:
            raise ValueError("native Head-CRSA query-head evidence is invalid")
        if self.selected_kv_heads != NATIVE_HEAD_CRSA_KV_HEADS:
            raise ValueError("native Head-CRSA KV-head evidence is invalid")
        count = len(NATIVE_HEAD_CRSA_QUERY_HEADS)
        if len(self.alpha_per_head) != count:
            raise ValueError("alpha_per_head must cover every selected query head")
        alphas = tuple(
            _real(value, "alpha_per_head", maximum=1.0) for value in self.alpha_per_head
        )
        if len(self.argmax_changed_queries_per_head) != count:
            raise ValueError(
                "argmax_changed_queries_per_head must cover every selected query head"
            )
        for value in self.argmax_changed_queries_per_head:
            _integer(value, "argmax_changed_queries_per_head")
        if len(self.mean_l1_probability_delta_per_head) != count:
            raise ValueError(
                "mean_l1_probability_delta_per_head must cover every selected query head"
            )
        for value in self.mean_l1_probability_delta_per_head:
            _real(value, "mean_l1_probability_delta_per_head", maximum=2.0)
        if self.free_heads != NATIVE_HEAD_CRSA_FREE_HEADS:
            raise ValueError(
                "native Head-CRSA free heads must be the exact selected-head complement"
            )
        free_error = _real(self.free_head_max_abs_error, "free_head_max_abs_error")
        future = _real(self.future_weight_max_abs, "future_weight_max_abs")
        _real(
            self.row_sum_max_error,
            "row_sum_max_error",
            maximum=NATIVE_HEAD_CRSA_ROW_SUM_TOLERANCE,
        )
        history_before = _integer(self.history_length_before, "history_length_before")
        history_after = _integer(self.history_length_after, "history_length_after")
        if not isinstance(self.identity, bool):
            raise TypeError("identity must be boolean")
        if free_error != 0.0:
            raise ValueError("free native attention heads must be bit-exact")
        if future != 0.0:
            raise ValueError("native Head-CRSA evidence must be strictly causal")
        if self.identity:
            if any(alphas):
                raise ValueError("identity evidence cannot report a non-zero alpha")
            if any(self.argmax_changed_queries_per_head) or any(
                self.mean_l1_probability_delta_per_head
            ):
                raise ValueError("identity evidence cannot report probability changes")
            if history_before or history_after:
                raise ValueError("identity evidence cannot retain CRSA usage history")
        elif history_before != query_start or history_after != key_length:
            raise ValueError("native Head-CRSA usage-history evidence is inconsistent")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for name in (
            "selected_query_heads",
            "selected_kv_heads",
            "alpha_per_head",
            "argmax_changed_queries_per_head",
            "mean_l1_probability_delta_per_head",
            "free_heads",
        ):
            payload[name] = list(payload[name])
        return payload


@dataclass(frozen=True, slots=True)
class Qwen38NativeHeadCrsa:
    """Blend Prefix-Sinkhorn into one real query head per Qwen GQA group.

    The other 20 query heads remain the original causal-softmax tensors.  The
    blended probability matrix is consumed by the checkpoint's single native
    ``P @ V`` / output-gate / ``o_proj`` path; no second attention forward or
    hidden-state graft is involved.
    """

    layer: int = NATIVE_HEAD_CRSA_LAYER
    alpha: float = 0.01
    head_indices: tuple[int, ...] = NATIVE_HEAD_CRSA_QUERY_HEADS
    balance_alpha: float = 1.0
    diagonal_debit: float = 3.0

    evidence_schema: ClassVar[str] = NATIVE_HEAD_CRSA_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer, bool)
            or not isinstance(self.layer, int)
            or self.layer != NATIVE_HEAD_CRSA_LAYER
        ):
            raise ValueError("native Head-CRSA is validated only for layer 27")
        if not isinstance(self.head_indices, tuple):
            raise TypeError("head_indices must be a tuple")
        if self.head_indices != NATIVE_HEAD_CRSA_QUERY_HEADS:
            raise ValueError(
                "native Head-CRSA query heads must be exactly (2, 8, 14, 20)"
            )
        object.__setattr__(self, "alpha", _real(self.alpha, "alpha", maximum=1.0))
        object.__setattr__(
            self, "balance_alpha", _real(self.balance_alpha, "balance_alpha")
        )
        object.__setattr__(
            self, "diagonal_debit", _real(self.diagonal_debit, "diagonal_debit")
        )

    @property
    def active(self) -> bool:
        return self.alpha != 0.0

    @property
    def selected_kv_heads(self) -> tuple[int, ...]:
        return NATIVE_HEAD_CRSA_KV_HEADS

    @property
    def spec(self) -> AttentionSpec:
        return AttentionSpec(
            kind="prefix_log",
            alpha=self.balance_alpha,
            diagonal_debit=self.diagonal_debit,
        )

    @staticmethod
    def _work_dtype(dtype: torch.dtype) -> torch.dtype:
        return torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype

    def route(
        self,
        logits: Tensor,
        base_probabilities: Tensor,
        *,
        query_start: int,
        allowed: Tensor,
        prior_log_usage: Tensor | None,
    ) -> tuple[Tensor, Tensor | None, NativeHeadCrsaEvidence]:
        """Return native probabilities, staged usage state, and strict evidence."""

        if not isinstance(logits, Tensor) or not logits.is_floating_point():
            raise TypeError("logits must be a floating-point torch tensor")
        if logits.ndim != 4:
            raise ValueError("logits must have [batch, head, query, key] shape")
        if (
            not isinstance(base_probabilities, Tensor)
            or not base_probabilities.is_floating_point()
        ):
            raise TypeError("base_probabilities must be a floating-point torch tensor")
        if tuple(base_probabilities.shape) != tuple(logits.shape):
            raise ValueError("base probabilities must match the logits shape")
        if (
            base_probabilities.device != logits.device
            or base_probabilities.dtype != logits.dtype
        ):
            raise ValueError("base probabilities must match logits dtype and device")
        batch, heads, query_length, key_length = logits.shape
        if heads != 24:
            raise ValueError("native Head-CRSA requires exactly 24 query heads")
        if min(batch, query_length, key_length) < 1:
            raise ValueError("native Head-CRSA tensor axes must be non-empty")
        if (
            isinstance(query_start, bool)
            or not isinstance(query_start, int)
            or query_start < 0
        ):
            raise ValueError("query_start must be a non-negative integer")
        if key_length != query_start + query_length:
            raise ValueError(
                "key length must equal query_start + query length for native CRSA"
            )
        if not isinstance(allowed, Tensor) or allowed.dtype != torch.bool:
            raise TypeError("allowed must be a boolean torch tensor")
        if allowed.device != logits.device:
            raise ValueError("allowed must be on the logits device")
        try:
            support = torch.broadcast_to(allowed, logits.shape)
        except RuntimeError as exc:
            raise ValueError("allowed must broadcast to the logits shape") from exc
        if bool((~support.any(dim=-1)).any().item()):
            raise ValueError("every native attention row must allow at least one key")
        if bool((~torch.isfinite(base_probabilities)).any().item()):
            raise ValueError("base probabilities must be finite")
        if bool((base_probabilities < 0).any().item()):
            raise ValueError("base probabilities must be non-negative")
        if _outside_support_max_abs(base_probabilities, support) != 0.0:
            raise ValueError("base probabilities must be exactly zero outside support")

        alpha_per_head = (self.alpha,) * len(self.head_indices)
        if not self.active:
            if prior_log_usage is not None:
                raise ValueError("alpha=0 native Head-CRSA cannot retain usage history")
            evidence = NativeHeadCrsaEvidence(
                schema=self.evidence_schema,
                layer=self.layer,
                query_start=query_start,
                query_length=query_length,
                key_length=key_length,
                selected_query_heads=self.head_indices,
                selected_kv_heads=self.selected_kv_heads,
                alpha_per_head=alpha_per_head,
                argmax_changed_queries_per_head=(0, 0, 0, 0),
                mean_l1_probability_delta_per_head=(0.0, 0.0, 0.0, 0.0),
                free_heads=NATIVE_HEAD_CRSA_FREE_HEADS,
                free_head_max_abs_error=0.0,
                future_weight_max_abs=0.0,
                row_sum_max_error=float(
                    (base_probabilities.detach().sum(dim=-1, dtype=torch.float32) - 1.0)
                    .abs()
                    .max()
                ),
                history_length_before=0,
                history_length_after=0,
                identity=True,
            )
            # This return deliberately precedes every clone and dtype cast.
            return base_probabilities, None, evidence

        selected = torch.tensor(
            self.head_indices, dtype=torch.long, device=logits.device
        )
        work_dtype = self._work_dtype(logits.dtype)
        selected_logits = logits.index_select(1, selected).to(dtype=work_dtype)
        selected_base_native = base_probabilities.index_select(1, selected)
        selected_base = selected_base_native.to(dtype=work_dtype)
        selected_support = support.index_select(1, selected)
        expected_prior = (batch, len(self.head_indices), query_start)
        if prior_log_usage is not None:
            if (
                not isinstance(prior_log_usage, Tensor)
                or not prior_log_usage.is_floating_point()
            ):
                raise TypeError("prior_log_usage must be floating point")
            if tuple(prior_log_usage.shape) != expected_prior:
                raise ValueError(
                    f"prior_log_usage must have shape {expected_prior}, got "
                    f"{tuple(prior_log_usage.shape)}"
                )
            if prior_log_usage.device != logits.device:
                raise ValueError("prior_log_usage must be on the logits device")
            if prior_log_usage.dtype != work_dtype:
                raise ValueError(
                    f"prior_log_usage must have dtype {work_dtype}, got "
                    f"{prior_log_usage.dtype}"
                )
        routed, next_log_usage = streaming_prefix_log(
            selected_logits,
            self.spec,
            query_start=query_start,
            prior_log_usage=prior_log_usage,
            allowed=selected_support,
        )
        mixed_selected = selected_base + self.alpha * (routed - selected_base)
        mixed_selected = mixed_selected.to(dtype=base_probabilities.dtype)
        probabilities = base_probabilities.clone()
        probabilities[:, self.head_indices] = mixed_selected

        free = NATIVE_HEAD_CRSA_FREE_HEADS
        for head in free:
            if not torch.equal(probabilities[:, head], base_probabilities[:, head]):
                raise RuntimeError("native Head-CRSA modified a free attention head")
        detached_delta = (mixed_selected - selected_base_native).detach()
        argmax_changed = (
            mixed_selected.detach().argmax(-1)
            != selected_base_native.detach().argmax(-1)
        ).sum(dim=(0, 2))
        mean_l1 = detached_delta.abs().sum(-1).to(torch.float64).mean(dim=(0, 2))
        future_max = _outside_support_max_abs(mixed_selected, selected_support)
        evidence = NativeHeadCrsaEvidence(
            schema=self.evidence_schema,
            layer=self.layer,
            query_start=query_start,
            query_length=query_length,
            key_length=key_length,
            selected_query_heads=self.head_indices,
            selected_kv_heads=self.selected_kv_heads,
            alpha_per_head=alpha_per_head,
            argmax_changed_queries_per_head=tuple(
                int(value) for value in argmax_changed.to(device="cpu").tolist()
            ),
            mean_l1_probability_delta_per_head=tuple(
                float(value) for value in mean_l1.to(device="cpu").tolist()
            ),
            free_heads=free,
            free_head_max_abs_error=0.0,
            future_weight_max_abs=future_max,
            row_sum_max_error=float(
                (probabilities.detach().sum(dim=-1, dtype=torch.float32) - 1.0)
                .abs()
                .max()
            ),
            history_length_before=query_start,
            history_length_after=int(next_log_usage.shape[-1]),
            identity=False,
        )
        return probabilities, next_log_usage, evidence


__all__ = [
    "NATIVE_HEAD_CRSA_EVIDENCE_SCHEMA",
    "NATIVE_HEAD_CRSA_FREE_HEADS",
    "NATIVE_HEAD_CRSA_KV_HEADS",
    "NATIVE_HEAD_CRSA_LAYER",
    "NATIVE_HEAD_CRSA_QUERY_HEADS",
    "NATIVE_HEAD_CRSA_ROW_SUM_TOLERANCE",
    "NativeHeadCrsaEvidence",
    "Qwen38NativeHeadCrsa",
]
