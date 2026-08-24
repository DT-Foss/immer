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
    local_heads: int = 0
    self_heads: int = 0
    balanced_heads: int = 0
    reservoir_logit: float = -1.0
    free_floor: float = 0.25
    free_heads: int = 1
    specialist_init: str = "llb"
    init_strength: float = 4.0
    anchor_pattern: str = "llb"
    adapt_budget: float = 0.25

    def label(self) -> str:
        p: list[str] = []
        if self.kind in {
            "prefix",
            "prefix_log",
            "raps",
            "reservoir_prefix",
            "geometric_prefix",
        }:
            p.append(f"a={self.alpha:g}")
        if self.kind == "quad_route" and self.alpha != 1.0:
            p.append(f"a={self.alpha:g}")
        if self.kind in {"geometric_prefix", "quad_route"} and self.usage_decay != 1.0:
            p.append(f"lam={self.usage_decay:g}")
        if self.diagonal_debit:
            p.append(f"dd={self.diagonal_debit:g}")
        if self.kind in {"recency", "past_recency"}:
            p.append(f"s={self.slope:g}")
        if self.kind == "dual_route":
            p.extend((f"lh={self.local_heads}", f"s={self.slope:g}"))
        if self.kind == "slg":
            p.extend(
                (f"sh={self.self_heads}", f"lh={self.local_heads}", f"s={self.slope:g}")
            )
        if self.kind == "quad_route":
            p.extend(
                (
                    f"sh={self.self_heads}",
                    f"lh={self.local_heads}",
                    f"bh={self.balanced_heads}",
                    f"s={self.slope:g}",
                )
            )
        if self.kind == "reservoir_prefix":
            p.append(f"rho={self.reservoir_logit:g}")
        if self.kind == "adaptive_route":
            p.extend((f"s={self.slope:g}", f"ff={self.free_floor:g}"))
        if self.kind == "adaptive_specialists":
            p.extend(
                (
                    f"fh={self.free_heads}",
                    f"s={self.slope:g}",
                    f"si={self.specialist_init}",
                    f"is={self.init_strength:g}",
                )
            )
        if self.kind == "anchor_residual":
            p.extend(
                (
                    f"fh={self.free_heads}",
                    f"s={self.slope:g}",
                    f"ap={self.anchor_pattern}",
                    f"ab={self.adapt_budget:g}",
                )
            )
        if self.kind == "marginal_residual":
            p.extend(
                (
                    f"fh={self.free_heads}",
                    f"s={self.slope:g}",
                    f"ab={self.adapt_budget:g}",
                )
            )
        if self.kind == "q_residual":
            p.extend(
                (
                    f"fh={self.free_heads}",
                    f"s={self.slope:g}",
                    f"is={self.init_strength:g}",
                    f"ab={self.adapt_budget:g}",
                )
            )
        return self.kind if not p else f"{self.kind}[{','.join(p)}]"


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
    weights = torch.softmax(_masked_logits(logits, mask), dim=-1).masked_fill(
        ~mask, 0.0
    )
    return weights, mask


def _log_base(logits: Tensor) -> tuple[Tensor, Tensor]:
    mask = _validate_logits(logits)
    return torch.log_softmax(_masked_logits(logits, mask), dim=-1).masked_fill(
        ~mask, -torch.inf
    ), mask


def _row_normalize(raw: Tensor, mask: Tensor) -> Tensor:
    raw = raw.masked_fill(~mask, 0.0)
    floor = torch.finfo(raw.dtype).tiny
    total = raw.sum(-1, keepdim=True).clamp_min(floor)
    return (raw / total).masked_fill(~mask, 0.0)


def _debit_log_diagonal(
    log_weights: Tensor, debit: float, *, preserve_first: bool = True
) -> Tensor:
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
    log_base, mask = _log_base(logits)
    log_usage = torch.logcumsumexp(log_base, dim=-2)
    if spec.eps > 0:
        log_eps = torch.as_tensor(
            math.log(float(spec.eps)), dtype=logits.dtype, device=logits.device
        )
        log_usage = torch.logaddexp(log_usage, log_eps)
    safe_base = torch.where(mask, log_base, torch.zeros_like(log_base))
    safe_usage = torch.where(mask, log_usage, torch.zeros_like(log_usage))
    log_raw = (safe_base - float(spec.alpha) * safe_usage).masked_fill(
        ~mask, -torch.inf
    )
    t = logits.shape[-1]
    eye = torch.eye(t, dtype=torch.bool, device=logits.device).view(1, 1, t, t)
    log_raw = torch.where(eye, log_raw + 8.0 * torch.finfo(logits.dtype).eps, log_raw)
    log_raw = _debit_log_diagonal(log_raw, spec.diagonal_debit)
    return torch.softmax(log_raw, -1).masked_fill(~mask, 0.0)


def streaming_prefix_log(
    logits: Tensor,
    spec: AttentionSpec,
    *,
    query_start: int,
    prior_log_usage: Tensor | None,
    allowed: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply Prefix-Sinkhorn to a rectangular streaming query block.

    ``prior_log_usage`` stores the log cumulative *causal-softmax* mass from
    every query preceding ``query_start``.  Keeping that state is what makes a
    ``[B, H, 1, past+1]`` decode step equivalent to the corresponding row of a
    one-shot square call.  The returned usage is the raw cumulative mass (the
    optional epsilon belongs only to the balancing denominator), ready for the
    next contiguous block.

    ``allowed`` is an explicit boolean support mask.  It is applied with
    ``-inf`` before either softmax, so masked logits have exactly zero mass and
    exactly zero gradient.  The diagonal is resolved in absolute coordinates;
    for a block beginning at ``query_start``, local row ``r`` debits key
    ``query_start + r``.
    """

    if logits.ndim != 4:
        raise ValueError("expected [batch, heads, query, key] logits")
    if not logits.is_floating_point():
        raise TypeError("logits must be floating point")
    if (
        isinstance(query_start, bool)
        or not isinstance(query_start, int)
        or query_start < 0
    ):
        raise ValueError("query_start must be a non-negative integer")
    batch, heads, query_length, key_length = logits.shape
    if min(batch, heads, query_length, key_length) < 1:
        raise ValueError("logits dimensions must be non-empty")
    if key_length != query_start + query_length:
        raise ValueError(
            "key length must equal query_start + query length for contiguous streaming"
        )
    if not isinstance(allowed, Tensor):
        raise TypeError("allowed must be a torch tensor")
    if allowed.dtype != torch.bool:
        raise TypeError("allowed must be boolean")
    if allowed.device != logits.device:
        raise ValueError("allowed must be on the logits device")
    try:
        support = torch.broadcast_to(allowed, logits.shape)
    except RuntimeError as exc:
        raise ValueError("allowed must broadcast to the logits shape") from exc
    if bool((~support.any(dim=-1)).any().item()):
        raise ValueError("every query row must allow at least one key")

    if not isinstance(spec, AttentionSpec):
        raise TypeError("spec must be an AttentionSpec")
    if spec.kind not in {"prefix_log", "raps"}:
        raise ValueError("streaming_prefix_log requires kind='prefix_log' or 'raps'")
    for value, name in (
        (spec.alpha, "alpha"),
        (spec.eps, "eps"),
        (spec.diagonal_debit, "diagonal_debit"),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real number")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    if prior_log_usage is None:
        if query_start:
            raise ValueError("prior_log_usage is required after query_start zero")
        prior = torch.full(
            (batch, heads, key_length),
            -torch.inf,
            dtype=logits.dtype,
            device=logits.device,
        )
    else:
        if not isinstance(prior_log_usage, Tensor):
            raise TypeError("prior_log_usage must be a torch tensor or None")
        if not prior_log_usage.is_floating_point():
            raise TypeError("prior_log_usage must be floating point")
        expected = (batch, heads, query_start)
        if tuple(prior_log_usage.shape) != expected:
            raise ValueError(
                f"prior_log_usage must have shape {expected}, got "
                f"{tuple(prior_log_usage.shape)}"
            )
        if prior_log_usage.device != logits.device:
            raise ValueError("prior_log_usage must be on the logits device")
        if prior_log_usage.dtype != logits.dtype:
            raise ValueError("prior_log_usage must have the logits dtype")
        if bool(
            (torch.isnan(prior_log_usage) | torch.isposinf(prior_log_usage))
            .any()
            .item()
        ):
            raise ValueError("prior_log_usage may contain only finite values or -inf")
        extension = torch.full(
            (batch, heads, query_length),
            -torch.inf,
            dtype=logits.dtype,
            device=logits.device,
        )
        prior = torch.cat((prior_log_usage, extension), dim=-1)

    log_base = torch.log_softmax(logits.masked_fill(~support, -torch.inf), dim=-1)
    log_base = log_base.masked_fill(~support, -torch.inf)
    block_log_usage = torch.logcumsumexp(log_base, dim=-2)
    raw_log_usage = torch.logaddexp(prior.unsqueeze(-2), block_log_usage)
    denominator_log_usage = raw_log_usage
    if spec.eps > 0:
        log_eps = torch.as_tensor(
            math.log(float(spec.eps)), dtype=logits.dtype, device=logits.device
        )
        denominator_log_usage = torch.logaddexp(denominator_log_usage, log_eps)

    safe_base = torch.where(support, log_base, torch.zeros_like(log_base))
    safe_usage = torch.where(
        support, denominator_log_usage, torch.zeros_like(denominator_log_usage)
    )
    log_raw = (safe_base - float(spec.alpha) * safe_usage).masked_fill(
        ~support, -torch.inf
    )

    query_index = query_start + torch.arange(query_length, device=logits.device)
    key_index = torch.arange(key_length, device=logits.device)
    diagonal = (query_index[:, None] == key_index[None, :]).reshape(
        1, 1, query_length, key_length
    )
    log_raw = torch.where(
        diagonal,
        log_raw + 8.0 * torch.finfo(logits.dtype).eps,
        log_raw,
    )
    if spec.diagonal_debit > 0:
        debit_diagonal = diagonal
        if query_start == 0:
            debit_diagonal = debit_diagonal.clone()
            debit_diagonal[..., 0, 0] = False
        log_raw = torch.where(
            debit_diagonal,
            log_raw - float(spec.diagonal_debit),
            log_raw,
        )
    weights = torch.softmax(log_raw, dim=-1).masked_fill(~support, 0.0)
    return weights, raw_log_usage[..., -1, :]


def geometric_prefix_log(logits: Tensor, spec: AttentionSpec) -> Tensor:
    """Prefix-Sinkhorn with geometrically discounted causal key exposure.

    ``usage_decay=1`` is exactly the original cumulative Prefix-Sinkhorn
    operator. Values below one implement the streaming recurrence
    ``U_i = A_i + usage_decay * U_{i-1}`` without exposing future rows.
    """
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
        log_eps = torch.as_tensor(
            math.log(float(spec.eps)), dtype=logits.dtype, device=logits.device
        )
        log_usage = torch.logaddexp(log_usage, log_eps)
    safe_base = torch.where(mask, log_base, torch.zeros_like(log_base))
    safe_usage = torch.where(mask, log_usage, torch.zeros_like(log_usage))
    log_raw = (safe_base - float(spec.alpha) * safe_usage).masked_fill(
        ~mask, -torch.inf
    )
    eye = torch.eye(t, dtype=torch.bool, device=logits.device).view(1, 1, t, t)
    log_raw = torch.where(eye, log_raw + 8.0 * torch.finfo(logits.dtype).eps, log_raw)
    log_raw = _debit_log_diagonal(log_raw, spec.diagonal_debit)
    return torch.softmax(log_raw, -1).masked_fill(~mask, 0.0)


def reservoir_prefix(logits: Tensor, spec: AttentionSpec) -> Tensor:
    """Prefix balancing with an explicit null reservoir.

    The returned token block is causal and sub-stochastic. Missing row mass is
    the reservoir allocation, so the residual stream carries the abstained mass.
    """
    mask = _validate_logits(logits)
    masked = _masked_logits(logits, mask)
    rho = torch.full(
        (*logits.shape[:-1], 1),
        float(spec.reservoir_logit),
        dtype=logits.dtype,
        device=logits.device,
    )
    log_z = torch.logsumexp(torch.cat((masked, rho), dim=-1), dim=-1, keepdim=True)
    log_base = (masked - log_z).masked_fill(~mask, -torch.inf)
    log_reservoir = rho - log_z
    log_usage = torch.logcumsumexp(log_base, dim=-2)
    if spec.eps > 0:
        log_eps = torch.as_tensor(
            math.log(float(spec.eps)), dtype=logits.dtype, device=logits.device
        )
        log_usage = torch.logaddexp(log_usage, log_eps)
    safe_base = torch.where(mask, log_base, torch.zeros_like(log_base))
    safe_usage = torch.where(mask, log_usage, torch.zeros_like(log_usage))
    log_raw = (safe_base - float(spec.alpha) * safe_usage).masked_fill(
        ~mask, -torch.inf
    )
    t = logits.shape[-1]
    eye = torch.eye(t, dtype=torch.bool, device=logits.device).view(1, 1, t, t)
    log_raw = torch.where(eye, log_raw + 8.0 * torch.finfo(logits.dtype).eps, log_raw)
    log_raw = _debit_log_diagonal(log_raw, spec.diagonal_debit)
    normalizer = torch.logsumexp(
        torch.cat((log_raw, log_reservoir), dim=-1), dim=-1, keepdim=True
    )
    return torch.exp(log_raw - normalizer).masked_fill(~mask, 0.0)


def recency_attention(
    logits: Tensor, slope: float, *, exclude_self: bool = False
) -> Tensor:
    mask = _validate_logits(logits)
    t = logits.shape[-1]
    q = torch.arange(t, device=logits.device).view(t, 1)
    k = torch.arange(t, device=logits.device).view(1, t)
    age = (q - k).clamp_min(0).to(logits.dtype)
    allowed = mask.clone()
    if exclude_self:
        allowed = torch.ones(t, t, dtype=torch.bool, device=logits.device).tril(-1)
        allowed[0, 0] = True
    biased = logits - float(slope) * age
    return torch.softmax(biased.masked_fill(~allowed, -torch.inf), -1).masked_fill(
        ~allowed, 0.0
    )


def identity_attention(logits: Tensor) -> Tensor:
    t = logits.shape[-1]
    return (
        torch.eye(t, dtype=logits.dtype, device=logits.device)
        .view(1, 1, t, t)
        .expand(logits.shape[0], logits.shape[1], t, t)
    )


def adaptive_route_attention(
    logits: Tensor,
    gate_probs: Tensor,
    spec: AttentionSpec,
    *,
    return_routes: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Query-gated Self–Local–Balanced–Free attention with a hard free floor.

    ``gate_probs`` orders routes as self, strictly-past local, RAPS-balanced,
    and unrestricted causal softmax.  The free route receives at least
    ``spec.free_floor`` probability for every query and head.
    """
    if spec.kind != "adaptive_route":
        raise ValueError("kind='adaptive_route' required")
    if not 0.0 <= spec.free_floor <= 1.0:
        raise ValueError("free_floor must be in [0, 1]")
    _validate_logits(logits)
    expected = (*logits.shape[:-1], 4)
    if gate_probs.shape != expected:
        raise ValueError(f"expected gate probabilities with shape {expected}")

    self_route = identity_attention(logits)
    local_route = recency_attention(logits, spec.slope, exclude_self=True)
    balanced_route = prefix_log(logits, replace(spec, kind="raps"))
    free_route = _base(logits)[0]
    branches = torch.stack(
        (self_route, local_route, balanced_route, free_route), dim=-2
    )

    effective = (1.0 - float(spec.free_floor)) * gate_probs
    effective = effective.clone()
    effective[..., 3] = effective[..., 3] + float(spec.free_floor)
    weights = (branches * effective.unsqueeze(-1)).sum(dim=-2)
    if return_routes:
        return weights, effective
    return weights


def adaptive_specialist_attention(
    logits: Tensor,
    gate_probs: Tensor,
    spec: AttentionSpec,
) -> Tensor:
    """Query-gated specialist heads followed by whole unrestricted heads.

    Specialist heads mix self, strictly-past local, and RAPS-balanced routes.
    The final ``free_heads`` are untouched causal-softmax heads, preserving a
    complete long-range value subspace instead of only a fractional floor.
    """
    if spec.kind != "adaptive_specialists":
        raise ValueError("kind='adaptive_specialists' required")
    mask = _validate_logits(logits)
    del mask
    heads = logits.shape[1]
    if not 1 <= spec.free_heads < heads:
        raise ValueError("free_heads must be in 1..heads-1")
    specialist_heads = heads - spec.free_heads
    expected = (logits.shape[0], specialist_heads, logits.shape[-2], 3)
    if gate_probs.shape != expected:
        raise ValueError(f"expected gate probabilities with shape {expected}")

    specialist_logits = logits[:, :specialist_heads]
    self_route = identity_attention(specialist_logits)
    local_route = recency_attention(specialist_logits, spec.slope, exclude_self=True)
    balanced_route = prefix_log(specialist_logits, replace(spec, kind="raps"))
    branches = torch.stack((self_route, local_route, balanced_route), dim=-2)
    specialist = (branches * gate_probs.unsqueeze(-1)).sum(dim=-2)
    free = _base(logits[:, specialist_heads:])[0]
    return torch.cat((specialist, free), dim=1)


def specialist_anchor(
    pattern: str,
    specialist_heads: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> Tensor:
    """Return one-hot self/local/balanced anchors for specialist heads."""
    if specialist_heads < 1:
        raise ValueError("specialist_heads must be positive")
    if pattern not in {"lll", "llb"}:
        raise ValueError("anchor_pattern must be 'lll' or 'llb'")
    anchor = torch.zeros(specialist_heads, 3, device=device, dtype=dtype)
    anchor[:, 1] = 1.0
    if pattern == "llb":
        anchor[-1, 1] = 0.0
        anchor[-1, 2] = 1.0
    return anchor


def anchor_residual_attention(
    logits: Tensor,
    gate_probs: Tensor,
    spec: AttentionSpec,
    *,
    return_routes: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Anchor-preserving residual routing plus whole free softmax heads.

    Each specialist keeps ``1-adapt_budget`` of a fixed LLL/LLB role and
    allocates only the residual budget through a query-dependent gate.
    """
    if spec.kind != "anchor_residual":
        raise ValueError("kind='anchor_residual' required")
    if not 0.0 <= spec.adapt_budget <= 1.0:
        raise ValueError("adapt_budget must be in [0, 1]")
    _validate_logits(logits)
    heads = logits.shape[1]
    if not 1 <= spec.free_heads < heads:
        raise ValueError("free_heads must be in 1..heads-1")
    specialist_heads = heads - spec.free_heads
    expected = (logits.shape[0], specialist_heads, logits.shape[-2], 3)
    if gate_probs.shape != expected:
        raise ValueError(f"expected gate probabilities with shape {expected}")

    anchor = specialist_anchor(
        spec.anchor_pattern,
        specialist_heads,
        device=logits.device,
        dtype=logits.dtype,
    ).view(1, specialist_heads, 1, 3)
    budget = float(spec.adapt_budget)
    effective = (1.0 - budget) * anchor + budget * gate_probs

    specialist_logits = logits[:, :specialist_heads]
    branches = torch.stack(
        (
            identity_attention(specialist_logits),
            recency_attention(specialist_logits, spec.slope, exclude_self=True),
            prefix_log(specialist_logits, replace(spec, kind="raps")),
        ),
        dim=-2,
    )
    specialist = (branches * effective.unsqueeze(-1)).sum(dim=-2)
    free = _base(logits[:, specialist_heads:])[0]
    weights = torch.cat((specialist, free), dim=1)
    if return_routes:
        return weights, effective.expand(logits.shape[0], -1, logits.shape[-2], -1)
    return weights


def marginal_residual_attention(
    logits: Tensor,
    gate_probs: Tensor,
    spec: AttentionSpec,
    *,
    return_routes: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Bounded query-dependent interpolation from local to prefix-balanced routing.

    Specialist heads remain local by default and may divert at most
    ``adapt_budget`` of each query into the causal Prefix-Sinkhorn branch.
    The final ``free_heads`` remain exact causal-softmax heads.
    """
    if spec.kind != "marginal_residual":
        raise ValueError("kind='marginal_residual' required")
    if not 0.0 <= spec.adapt_budget <= 1.0:
        raise ValueError("adapt_budget must be in [0, 1]")
    _validate_logits(logits)
    heads = logits.shape[1]
    if not 1 <= spec.free_heads < heads:
        raise ValueError("free_heads must be in 1..heads-1")
    specialist_heads = heads - spec.free_heads
    expected = (logits.shape[0], specialist_heads, logits.shape[-2])
    if gate_probs.shape != expected:
        raise ValueError(f"expected gate probabilities with shape {expected}")
    if torch.any((gate_probs < 0.0) | (gate_probs > 1.0)):
        raise ValueError("gate probabilities must be in [0, 1]")

    specialist_logits = logits[:, :specialist_heads]
    local = recency_attention(specialist_logits, spec.slope, exclude_self=True)
    balanced = prefix_log(specialist_logits, replace(spec, kind="raps"))
    balance = float(spec.adapt_budget) * gate_probs
    specialist = local + balance.unsqueeze(-1) * (balanced - local)
    free = _base(logits[:, specialist_heads:])[0]
    weights = torch.cat((specialist, free), dim=1)
    if return_routes:
        return weights, balance
    return weights


def q_residual_attention(
    logits: Tensor,
    gate_probs: Tensor,
    spec: AttentionSpec,
    *,
    return_routes: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """LLB-anchored local/Prefix interpolation plus whole free heads.

    The final specialist head is anchored to Prefix-Sinkhorn balance and all
    preceding specialist heads are anchored to strictly-past local attention.
    ``adapt_budget`` bounds how far each query may move away from that fixed Q
    foundation, while the final ``free_heads`` remain exact causal softmax.
    """
    if spec.kind != "q_residual":
        raise ValueError("kind='q_residual' required")
    if not 0.0 <= spec.adapt_budget <= 1.0:
        raise ValueError("adapt_budget must be in [0, 1]")
    _validate_logits(logits)
    heads = logits.shape[1]
    if not 1 <= spec.free_heads < heads:
        raise ValueError("free_heads must be in 1..heads-1")
    specialist_heads = heads - spec.free_heads
    expected = (logits.shape[0], specialist_heads, logits.shape[-2])
    if gate_probs.shape != expected:
        raise ValueError(f"expected gate probabilities with shape {expected}")
    if torch.any((gate_probs < 0.0) | (gate_probs > 1.0)):
        raise ValueError("gate probabilities must be in [0, 1]")

    specialist_logits = logits[:, :specialist_heads]
    local = recency_attention(specialist_logits, spec.slope, exclude_self=True)
    balanced = prefix_log(specialist_logits, replace(spec, kind="raps"))
    anchor = torch.zeros_like(gate_probs)
    anchor[:, -1] = 1.0
    budget = float(spec.adapt_budget)
    balance = (1.0 - budget) * anchor + budget * gate_probs
    if budget == 0.0:
        specialist = torch.cat((local[:, :-1], balanced[:, -1:]), dim=1)
    else:
        specialist = local + balance.unsqueeze(-1) * (balanced - local)
    free = _base(logits[:, specialist_heads:])[0]
    weights = torch.cat((specialist, free), dim=1)
    if return_routes:
        return weights, balance
    return weights


def leaky_masked_sinkhorn(logits: Tensor, iterations: int = 1) -> Tensor:
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
    if spec.kind == "prefix_log":
        return prefix_log(logits, spec)
    if spec.kind == "raps":
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
    if spec.kind == "dual_route":
        h = logits.shape[1]
        if not 0 <= spec.local_heads <= h:
            raise ValueError("local_heads out of range")
        n = spec.local_heads
        parts = []
        if n:
            parts.append(recency_attention(logits[:, :n], spec.slope))
        if n < h:
            parts.append(prefix_log(logits[:, n:], replace(spec, kind="prefix_log")))
        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
    if spec.kind == "slg":
        h = logits.shape[1]
        if (
            spec.self_heads < 0
            or spec.local_heads < 0
            or spec.self_heads + spec.local_heads > h
        ):
            raise ValueError("self_heads + local_heads out of range")
        a = spec.self_heads
        b = a + spec.local_heads
        parts = []
        if a:
            parts.append(identity_attention(logits[:, :a]))
        if b > a:
            parts.append(
                recency_attention(logits[:, a:b], spec.slope, exclude_self=True)
            )
        if b < h:
            parts.append(prefix_log(logits[:, b:], replace(spec, kind="prefix_log")))
        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
    if spec.kind == "quad_route":
        h = logits.shape[1]
        counts = (spec.self_heads, spec.local_heads, spec.balanced_heads)
        if any(n < 0 for n in counts) or sum(counts) > h:
            raise ValueError("self_heads + local_heads + balanced_heads out of range")
        a = spec.self_heads
        b = a + spec.local_heads
        c = b + spec.balanced_heads
        parts = []
        if a:
            parts.append(identity_attention(logits[:, :a]))
        if b > a:
            parts.append(
                recency_attention(logits[:, a:b], spec.slope, exclude_self=True)
            )
        if c > b:
            balanced_spec = replace(spec, kind="geometric_prefix")
            parts.append(geometric_prefix_log(logits[:, b:c], balanced_spec))
        if c < h:
            parts.append(_base(logits[:, c:])[0])
        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
    if spec.kind == "role_complete":
        return role_complete_attention(logits, spec)
    if spec.kind == "leaky_sinkhorn":
        return leaky_masked_sinkhorn(logits)
    raise ValueError(f"unsupported attention kind: {spec.kind}")


def role_complete_attention(logits: Tensor, spec: AttentionSpec) -> Tensor:
    """Fixed Local | Balanced | Free causal head program."""
    h = logits.shape[1]
    counts = (spec.local_heads, spec.balanced_heads, spec.free_heads)
    if any(n < 0 for n in counts) or sum(counts) != h:
        raise ValueError(
            "local_heads + balanced_heads + free_heads must equal the head count"
        )
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
