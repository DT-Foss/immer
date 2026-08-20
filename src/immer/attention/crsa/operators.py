from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class AttentionSpec:
    kind: str = "softmax"
    alpha: float = 1.0
    usage_decay: float = 1.0
    eps: float = 0.0
    diagonal_debit: float = 0.0
    slope: float = 1.0
    reservoir_logit: float = -1.0
    local_heads: int = 0
    balanced_heads: int = 0
    free_heads: int = 1


def causal_mask(length: int, device: torch.device | str) -> Tensor:
    if length < 1:
        raise ValueError("length must be positive")
    return torch.ones(length, length, dtype=torch.bool, device=device).tril()


def _validate_logits(logits: Tensor) -> Tensor:
    if logits.ndim != 4 or logits.shape[-1] != logits.shape[-2]:
        raise ValueError("expected square [batch, heads, query, key] logits")
    return causal_mask(logits.shape[-1], logits.device)


def _masked_logits(logits: Tensor, mask: Tensor) -> Tensor:
    return logits.masked_fill(~mask, -torch.inf)


def _base(logits: Tensor) -> tuple[Tensor, Tensor]:
    mask = _validate_logits(logits)
    weights = torch.softmax(_masked_logits(logits, mask), dim=-1).masked_fill(~mask, 0.0)
    return weights, mask


def _log_base(logits: Tensor) -> tuple[Tensor, Tensor]:
    mask = _validate_logits(logits)
    log_weights = torch.log_softmax(_masked_logits(logits, mask), dim=-1)
    return log_weights.masked_fill(~mask, -torch.inf), mask


def _row_normalize(raw: Tensor, mask: Tensor) -> Tensor:
    raw = raw.masked_fill(~mask, 0.0)
    total = raw.sum(-1, keepdim=True).clamp_min(torch.finfo(raw.dtype).tiny)
    return (raw / total).masked_fill(~mask, 0.0)


def _debit_log_diagonal(log_weights: Tensor, debit: float, *, preserve_first: bool = True) -> Tensor:
    if debit <= 0:
        return log_weights
    t = log_weights.shape[-1]
    eye = torch.eye(t, dtype=torch.bool, device=log_weights.device).view(1, 1, t, t)
    out = torch.where(eye, log_weights - float(debit), log_weights)
    if preserve_first:
        out = out.clone()
        out[..., 0, 0] = log_weights[..., 0, 0]
    return out


def _debit_probability(weights: Tensor, debit: float, mask: Tensor) -> Tensor:
    if debit <= 0:
        return weights
    t = weights.shape[-1]
    eye = torch.eye(t, dtype=torch.bool, device=weights.device).view(1, 1, t, t)
    raw = torch.where(eye, weights * math.exp(-float(debit)), weights)
    raw = raw.clone()
    raw[..., 0, 0] = weights[..., 0, 0]
    return _row_normalize(raw, mask)


def prefix_probability(logits: Tensor, spec: AttentionSpec) -> Tensor:
    base, mask = _base(logits)
    usage = base.cumsum(-2)
    if spec.eps > 0:
        usage = usage + float(spec.eps)
    raw = base / usage.clamp_min(torch.finfo(base.dtype).tiny).pow(float(spec.alpha))
    return _debit_probability(_row_normalize(raw, mask), spec.diagonal_debit, mask)


def prefix_log(logits: Tensor, spec: AttentionSpec) -> Tensor:
    """Numerically stable causal Prefix-Sinkhorn / RAPS operator."""
    log_base, mask = _log_base(logits)
    log_usage = torch.logcumsumexp(log_base, dim=-2)
    if spec.eps > 0:
        log_eps = torch.as_tensor(math.log(float(spec.eps)), dtype=logits.dtype, device=logits.device)
        log_usage = torch.logaddexp(log_usage, log_eps)
    safe_base = torch.where(mask, log_base, torch.zeros_like(log_base))
    safe_usage = torch.where(mask, log_usage, torch.zeros_like(log_usage))
    log_raw = (safe_base - float(spec.alpha) * safe_usage).masked_fill(~mask, -torch.inf)
    t = logits.shape[-1]
    eye = torch.eye(t, dtype=torch.bool, device=logits.device).view(1, 1, t, t)
    log_raw = torch.where(eye, log_raw + 8.0 * torch.finfo(logits.dtype).eps, log_raw)
    log_raw = _debit_log_diagonal(log_raw, spec.diagonal_debit)
    return torch.softmax(log_raw, -1).masked_fill(~mask, 0.0)


def geometric_prefix_log(logits: Tensor, spec: AttentionSpec) -> Tensor:
    """Prefix balancing with discounted causal usage state U_i=A_i+lambda*U_{i-1}."""
    decay = float(spec.usage_decay)
    if not 0.0 < decay <= 1.0:
        raise ValueError("usage_decay must be in (0, 1]")
    if decay == 1.0:
        return prefix_log(logits, replace(spec, kind="prefix_log"))

    log_base, mask = _log_base(logits)
    t = logits.shape[-1]
    row = torch.arange(t, dtype=logits.dtype, device=logits.device).view(1, 1, t, 1)
    log_decay = math.log(decay)
    log_usage = torch.logcumsumexp(log_base - row * log_decay, dim=-2) + row * log_decay
    if spec.eps > 0:
        log_eps = torch.as_tensor(math.log(float(spec.eps)), dtype=logits.dtype, device=logits.device)
        log_usage = torch.logaddexp(log_usage, log_eps)
    safe_base = torch.where(mask, log_base, torch.zeros_like(log_base))
    safe_usage = torch.where(mask, log_usage, torch.zeros_like(log_usage))
    log_raw = (safe_base - float(spec.alpha) * safe_usage).masked_fill(~mask, -torch.inf)
    eye = torch.eye(t, dtype=torch.bool, device=logits.device).view(1, 1, t, t)
    log_raw = torch.where(eye, log_raw + 8.0 * torch.finfo(logits.dtype).eps, log_raw)
    log_raw = _debit_log_diagonal(log_raw, spec.diagonal_debit)
    return torch.softmax(log_raw, -1).masked_fill(~mask, 0.0)


def reservoir_prefix(logits: Tensor, spec: AttentionSpec) -> Tensor:
    """Causal prefix balancing with an explicit sub-stochastic null reservoir."""
    mask = _validate_logits(logits)
    masked = _masked_logits(logits, mask)
    rho = torch.full((*logits.shape[:-1], 1), float(spec.reservoir_logit), dtype=logits.dtype, device=logits.device)
    log_z = torch.logsumexp(torch.cat((masked, rho), dim=-1), dim=-1, keepdim=True)
    log_base = (masked - log_z).masked_fill(~mask, -torch.inf)
    log_reservoir = rho - log_z
    log_usage = torch.logcumsumexp(log_base, dim=-2)
    if spec.eps > 0:
        log_eps = torch.as_tensor(math.log(float(spec.eps)), dtype=logits.dtype, device=logits.device)
        log_usage = torch.logaddexp(log_usage, log_eps)
    safe_base = torch.where(mask, log_base, torch.zeros_like(log_base))
    safe_usage = torch.where(mask, log_usage, torch.zeros_like(log_usage))
    log_raw = (safe_base - float(spec.alpha) * safe_usage).masked_fill(~mask, -torch.inf)
    t = logits.shape[-1]
    eye = torch.eye(t, dtype=torch.bool, device=logits.device).view(1, 1, t, t)
    log_raw = torch.where(eye, log_raw + 8.0 * torch.finfo(logits.dtype).eps, log_raw)
    log_raw = _debit_log_diagonal(log_raw, spec.diagonal_debit)
    normalizer = torch.logsumexp(torch.cat((log_raw, log_reservoir), dim=-1), dim=-1, keepdim=True)
    return torch.exp(log_raw - normalizer).masked_fill(~mask, 0.0)


def recency_attention(logits: Tensor, slope: float, *, exclude_self: bool = False) -> Tensor:
    mask = _validate_logits(logits)
    t = logits.shape[-1]
    q = torch.arange(t, device=logits.device).view(t, 1)
    k = torch.arange(t, device=logits.device).view(1, t)
    age = (q - k).clamp_min(0).to(logits.dtype)
    allowed = mask
    if exclude_self:
        allowed = torch.ones(t, t, dtype=torch.bool, device=logits.device).tril(-1)
        allowed[0, 0] = True
    biased = logits - float(slope) * age
    return torch.softmax(biased.masked_fill(~allowed, -torch.inf), -1).masked_fill(~allowed, 0.0)


def identity_attention(logits: Tensor) -> Tensor:
    _validate_logits(logits)
    t = logits.shape[-1]
    eye = torch.eye(t, dtype=logits.dtype, device=logits.device).view(1, 1, t, t)
    return eye.expand(logits.shape[0], logits.shape[1], -1, -1)


def role_complete_attention(logits: Tensor, spec: AttentionSpec) -> Tensor:
    """Fixed Local | Balanced | Free causal head program."""
    h = logits.shape[1]
    counts = (spec.local_heads, spec.balanced_heads, spec.free_heads)
    if any(n < 0 for n in counts) or sum(counts) != h:
        raise ValueError("local_heads + balanced_heads + free_heads must equal the head count")
    a = spec.local_heads
    b = a + spec.balanced_heads
    parts: list[Tensor] = []
    if a:
        parts.append(recency_attention(logits[:, :a], spec.slope, exclude_self=True))
    if b > a:
        parts.append(prefix_log(logits[:, a:b], replace(spec, kind="raps")))
    if b < h:
        parts.append(_base(logits[:, b:])[0])
    return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]


def leaky_masked_sinkhorn(logits: Tensor, iterations: int = 1) -> Tensor:
    """Negative control: full-support masked Sinkhorn, retained for leak probes."""
    base, mask = _base(logits)
    out = base
    for _ in range(iterations):
        out = out / out.sum(-2, keepdim=True).clamp_min(torch.finfo(out.dtype).tiny)
        out = _row_normalize(out, mask)
    return out


def apply_attention(logits: Tensor, spec: AttentionSpec) -> Tensor:
    if spec.kind == "softmax":
        return _base(logits)[0]
    if spec.kind == "prefix":
        return prefix_probability(logits, spec)
    if spec.kind in {"prefix_log", "raps"}:
        return prefix_log(logits, spec)
    if spec.kind == "geometric_prefix":
        return geometric_prefix_log(logits, spec)
    if spec.kind == "reservoir_prefix":
        return reservoir_prefix(logits, spec)
    if spec.kind == "recency":
        return recency_attention(logits, spec.slope)
    if spec.kind == "past_recency":
        return recency_attention(logits, spec.slope, exclude_self=True)
    if spec.kind == "identity":
        return identity_attention(logits)
    if spec.kind == "role_complete":
        return role_complete_attention(logits, spec)
    if spec.kind == "leaky_sinkhorn":
        return leaky_masked_sinkhorn(logits)
    raise ValueError(f"unsupported attention kind: {spec.kind}")
