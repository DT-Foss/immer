"""Portable Qwen3.5/Qwen3.8 text kernels expressed with ordinary PyTorch.

The production runtime streams projection matrices one at a time.  Accordingly,
the large-block entry points in this module consume *projected activations*, not
resident projection weights.  They implement the checkpoint's exact reference
math without depending on Transformers, FLA, causal-conv1d, or custom CUDA
kernels.

Tensor conventions follow the official Qwen3.5 implementation:

* projected activations use ``[batch, sequence, feature]``;
* full-attention cache tensors use ``[batch, kv_head, sequence, head_dim]``;
* DeltaNet recurrent state uses ``[batch, value_head, key_dim, value_dim]``;
* DeltaNet convolution state retains the last ``kernel_size`` raw projected
  inputs, matching the official dynamic cache.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .native_crsa import NativeHeadCrsaEvidence, Qwen38NativeHeadCrsa


__all__ = [
    "AttentionState",
    "DeltaNetProbe",
    "DeltaNetState",
    "apply_rotary_pos_emb",
    "causal_depthwise_conv",
    "full_attention_core",
    "full_attention_fork_core",
    "gated_delta_net_core",
    "l2_normalize",
    "recurrent_gated_delta_rule",
    "rms_norm",
    "rms_norm_gated",
    "rope_cos_sin",
    "split_query_gate",
    "swiglu",
]


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _floating_tensor(
    value: object, name: str, *, ndim: int | None = None
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point torch tensor")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {value.ndim}")
    return value


def _same_device_dtype(reference: torch.Tensor, value: torch.Tensor, name: str) -> None:
    if value.device != reference.device:
        raise ValueError(f"{name} must be on {reference.device}, got {value.device}")
    if value.dtype != reference.dtype:
        raise ValueError(f"{name} must have dtype {reference.dtype}, got {value.dtype}")


@dataclass(frozen=True, slots=True)
class AttentionState:
    """Persistent full-attention KV state and optional native CRSA usage."""

    key: torch.Tensor
    value: torch.Tensor
    crsa_log_usage: torch.Tensor | None = None

    def __post_init__(self) -> None:
        key = _floating_tensor(self.key, "key", ndim=4)
        value = _floating_tensor(self.value, "value", ndim=4)
        if key.shape != value.shape:
            raise ValueError(
                f"key/value state shapes must match, got {tuple(key.shape)} and "
                f"{tuple(value.shape)}"
            )
        _same_device_dtype(key, value, "value")
        if self.crsa_log_usage is not None:
            usage = _floating_tensor(self.crsa_log_usage, "crsa_log_usage", ndim=3)
            expected = (key.shape[0], 4, key.shape[2])
            if tuple(usage.shape) != expected:
                raise ValueError(
                    f"crsa_log_usage state shape must be {expected}, got "
                    f"{tuple(usage.shape)}"
                )
            if usage.device != key.device:
                raise ValueError("crsa_log_usage must be on the KV-state device")
            expected_dtype = (
                torch.float32
                if key.dtype in {torch.float16, torch.bfloat16}
                else key.dtype
            )
            if usage.dtype != expected_dtype:
                raise ValueError(
                    f"crsa_log_usage must have dtype {expected_dtype}, got "
                    f"{usage.dtype}"
                )
            if bool((torch.isnan(usage) | torch.isposinf(usage)).any().item()):
                raise ValueError(
                    "crsa_log_usage may contain only finite values or -inf"
                )

    @property
    def length(self) -> int:
        return int(self.key.shape[2])


@dataclass(frozen=True, slots=True)
class _FullAttentionWork:
    """Shared immutable tensors for one full-attention layer pass."""

    gate: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    repeated_value: torch.Tensor
    scores: torch.Tensor
    allowed: torch.Tensor
    base_probabilities: torch.Tensor
    batch_size: int
    sequence_length: int
    heads: int
    width: int
    past_length: int


@dataclass(frozen=True, slots=True)
class DeltaNetState:
    """Persistent convolution and recurrent state for one DeltaNet layer."""

    conv: torch.Tensor
    recurrent: torch.Tensor

    def __post_init__(self) -> None:
        conv = _floating_tensor(self.conv, "conv", ndim=3)
        recurrent = _floating_tensor(self.recurrent, "recurrent", ndim=4)
        if conv.shape[0] != recurrent.shape[0]:
            raise ValueError("conv and recurrent states must have the same batch size")
        if conv.device != recurrent.device:
            raise ValueError("conv and recurrent states must be on the same device")


@dataclass(frozen=True, slots=True)
class DeltaNetProbe:
    """Read-only Qwen DeltaNet component measurements for one layer pass.

    These are the nine component statistics measured on the original 27B
    probe run.  Probe collection never changes the tensors used by inference.
    """

    beta_mean: float
    beta_std: float
    decay_mean: float
    decay_std: float
    conv_norm: float
    q_norm: float
    k_norm: float
    v_norm: float
    delta_norm: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")


def _probe_mean_vector_norm(value: torch.Tensor, width: int) -> float:
    """Match the legacy float64 probe reduction with bounded host blocks."""

    flat = value.detach().reshape(-1, width)
    if flat.shape[0] == 0:
        raise ValueError("probe tensor must not be empty")
    total = 0.0
    for start in range(0, flat.shape[0], 256):
        block = flat[start : start + 256].to(device="cpu").to(dtype=torch.float64)
        total += float(torch.linalg.vector_norm(block, dim=-1).sum().item())
    return total / flat.shape[0]


def _probe_mean_sample_std(value: torch.Tensor) -> tuple[float, float]:
    """Return the float64 mean and sample standard deviation (ddof=1)."""

    flat = value.detach().reshape(-1)
    count = int(flat.numel())
    if count == 0:
        raise ValueError("probe tensor must not be empty")
    total = 0.0
    for start in range(0, count, 1_048_576):
        block = flat[start : start + 1_048_576].to(device="cpu").to(dtype=torch.float64)
        total += float(block.sum().item())
    mean = total / count
    if count == 1:
        return mean, 0.0
    squared = 0.0
    for start in range(0, count, 1_048_576):
        block = flat[start : start + 1_048_576].to(device="cpu").to(dtype=torch.float64)
        squared += float((block - mean).square().sum().item())
    return mean, math.sqrt(squared / (count - 1))


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply Qwen3.5's delta-centered RMSNorm in float32.

    Qwen3.5 stores zero-centered norm deltas, so the learned multiplier is
    ``1 + weight`` rather than ``weight``.
    """

    values = _floating_tensor(x, "x")
    if values.ndim == 0:
        raise ValueError("x must have a feature dimension")
    scale = _floating_tensor(weight, "weight", ndim=1)
    if scale.numel() != values.shape[-1]:
        raise ValueError(
            f"weight shape {tuple(scale.shape)} does not match feature size "
            f"{values.shape[-1]}"
        )
    if scale.device != values.device:
        raise ValueError("weight and x must be on the same device")
    epsilon = _positive_float(eps, "eps")

    normalized = values.float()
    normalized = normalized * torch.rsqrt(
        normalized.square().mean(dim=-1, keepdim=True) + epsilon
    )
    normalized = normalized * (1.0 + scale.float())
    return normalized.to(dtype=values.dtype)


def rms_norm_gated(
    x: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply the DeltaNet output RMSNorm followed by its SiLU gate.

    Unlike decoder/head RMSNorm, this norm's checkpoint weight is a conventional
    multiplier (initialized to one), not a zero-centered delta.
    """

    values = _floating_tensor(x, "x")
    gates = _floating_tensor(gate, "gate")
    scale = _floating_tensor(weight, "weight", ndim=1)
    if values.shape != gates.shape:
        raise ValueError(
            f"gate shape {tuple(gates.shape)} must equal x shape {tuple(values.shape)}"
        )
    if values.ndim == 0:
        raise ValueError("x must have a feature dimension")
    if scale.numel() != values.shape[-1]:
        raise ValueError(
            f"weight shape {tuple(scale.shape)} does not match feature size "
            f"{values.shape[-1]}"
        )
    if gates.device != values.device or scale.device != values.device:
        raise ValueError("x, gate, and weight must be on the same device")
    epsilon = _positive_float(eps, "eps")

    normalized = values.float()
    normalized = normalized * torch.rsqrt(
        normalized.square().mean(dim=-1, keepdim=True) + epsilon
    )
    # This cast/order mirrors Qwen3_5RMSNormGated rather than fusing all math
    # into float32; it matters for checkpoint inference in bfloat16.
    normalized = normalized.to(values.dtype) * scale.to(values.dtype)
    normalized = normalized * F.silu(gates.float())
    return normalized.to(values.dtype)


def swiglu(gate_projection: torch.Tensor, up_projection: torch.Tensor) -> torch.Tensor:
    """Combine separately streamed gate/up projections with dense SwiGLU."""

    gate = _floating_tensor(gate_projection, "gate_projection")
    up = _floating_tensor(up_projection, "up_projection")
    if gate.shape != up.shape:
        raise ValueError(
            f"gate/up projection shapes must match, got {tuple(gate.shape)} and "
            f"{tuple(up.shape)}"
        )
    _same_device_dtype(gate, up, "up_projection")
    return F.silu(gate) * up


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """FLA-compatible L2 normalization used before the gated delta rule."""

    values = _floating_tensor(x, "x")
    if isinstance(dim, bool) or not isinstance(dim, int):
        raise TypeError("dim must be an integer")
    if not -values.ndim <= dim < values.ndim:
        raise ValueError(f"dim {dim} is invalid for a {values.ndim}-D tensor")
    epsilon = _positive_float(eps, "eps")
    return values * torch.rsqrt((values * values).sum(dim=dim, keepdim=True) + epsilon)


def rope_cos_sin(
    position_ids: torch.Tensor,
    rotary_dim: int,
    *,
    theta: float = 10_000_000.0,
    mrope_section: Sequence[int] | None = None,
    mrope_interleaved: bool = True,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct partial RoPE tables, including Qwen3.5 interleaved MRoPE.

    ``position_ids`` may be text positions ``[batch, sequence]`` or multimodal
    temporal/height/width positions ``[3, batch, sequence]``.  Text positions
    use the temporal axis for every rotary component.  Returned tables have
    shape ``[batch, sequence, rotary_dim]`` and follow Qwen's rotate-half layout
    (the half-sized frequency vector is concatenated with itself).
    """

    if not isinstance(position_ids, torch.Tensor):
        raise TypeError("position_ids must be a torch tensor")
    if position_ids.dtype == torch.bool or position_ids.is_floating_point():
        raise TypeError("position_ids must use an integer dtype")
    dimension = _positive_int(rotary_dim, "rotary_dim")
    if dimension % 2:
        raise ValueError("rotary_dim must be even")
    base = _positive_float(theta, "theta")
    if not isinstance(mrope_interleaved, bool):
        raise TypeError("mrope_interleaved must be boolean")
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TypeError("dtype must be a floating-point torch dtype")

    plain_text = position_ids.ndim == 2
    if plain_text:
        axes = position_ids.unsqueeze(0).expand(3, -1, -1)
    elif position_ids.ndim == 3 and position_ids.shape[0] == 3:
        axes = position_ids
    else:
        raise ValueError(
            "position_ids must have shape [batch, sequence] or [3, batch, sequence]"
        )
    if axes.shape[1] <= 0 or axes.shape[2] <= 0:
        raise ValueError("position_ids must have non-empty batch and sequence axes")

    inverse_frequency = 1.0 / (
        base
        ** (
            torch.arange(0, dimension, 2, device=axes.device, dtype=torch.float32)
            / dimension
        )
    )
    frequencies = axes.float().unsqueeze(-1) * inverse_frequency

    if mrope_section is None:
        if not plain_text:
            raise ValueError("mrope_section is required for three-axis position_ids")
        selected = frequencies[0]
    else:
        if len(mrope_section) != 3:
            raise ValueError(
                "mrope_section must contain temporal, height, and width sizes"
            )
        sections = tuple(
            _positive_int(value, "mrope_section entry") for value in mrope_section
        )
        if sum(sections) != dimension // 2:
            raise ValueError(
                f"mrope_section sums to {sum(sections)}, expected {dimension // 2}"
            )
        if mrope_interleaved:
            selected = frequencies[0].clone()
            for axis, offset in ((1, 1), (2, 2)):
                index = slice(offset, sections[axis] * 3, 3)
                selected[..., index] = frequencies[axis, ..., index]
        else:
            chunks: list[torch.Tensor] = []
            start = 0
            for axis, width in enumerate(sections):
                stop = start + width
                chunks.append(frequencies[axis, ..., start:stop])
                start = stop
            selected = torch.cat(chunks, dim=-1)

    embedding = torch.cat((selected, selected), dim=-1)
    return embedding.cos().to(dtype=dtype), embedding.sin().to(dtype=dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    midpoint = x.shape[-1] // 2
    return torch.cat((-x[..., midpoint:], x[..., :midpoint]), dim=-1)


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Qwen's partial rotate-half RoPE to ``[B, H, S, D]`` Q/K."""

    q = _floating_tensor(query, "query", ndim=4)
    k = _floating_tensor(key, "key", ndim=4)
    cosine = _floating_tensor(cos, "cos")
    sine = _floating_tensor(sin, "sin")
    if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ValueError(
            "query/key must share batch, sequence, and head dimensions; only head "
            "count may differ"
        )
    _same_device_dtype(q, k, "key")
    if cosine.ndim == 2:
        cosine = cosine.unsqueeze(0).expand(q.shape[0], -1, -1)
    if sine.ndim == 2:
        sine = sine.unsqueeze(0).expand(q.shape[0], -1, -1)
    expected_prefix = (q.shape[0], q.shape[2])
    if cosine.ndim != 3 or cosine.shape[:2] != expected_prefix:
        raise ValueError(
            f"cos shape {tuple(cosine.shape)} does not match batch/sequence "
            f"{expected_prefix}"
        )
    if sine.shape != cosine.shape:
        raise ValueError("sin must have the same shape as cos")
    if cosine.device != q.device or sine.device != q.device:
        raise ValueError("query, key, cos, and sin must be on the same device")
    rotary_dim = cosine.shape[-1]
    if rotary_dim <= 0 or rotary_dim > q.shape[-1] or rotary_dim % 2:
        raise ValueError(
            "cos/sin rotary dimension must be positive, even, and no larger than head_dim"
        )

    cosine = cosine.to(dtype=q.dtype).unsqueeze(1)
    sine = sine.to(dtype=q.dtype).unsqueeze(1)
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = q_rot * cosine + _rotate_half(q_rot) * sine
    k_embed = k_rot * cosine + _rotate_half(k_rot) * sine
    return torch.cat((q_embed, q_pass), dim=-1), torch.cat((k_embed, k_pass), dim=-1)


def split_query_gate(
    projected_query_gate: torch.Tensor,
    num_attention_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split Qwen3.5's per-head interleaved query/output-gate projection."""

    projected = _floating_tensor(projected_query_gate, "projected_query_gate", ndim=3)
    heads = _positive_int(num_attention_heads, "num_attention_heads")
    width = _positive_int(head_dim, "head_dim")
    expected = heads * width * 2
    if projected.shape[-1] != expected:
        raise ValueError(
            f"projected_query_gate feature size {projected.shape[-1]} does not match "
            f"2 * {heads} * {width} = {expected}"
        )
    per_head = projected.reshape(*projected.shape[:-1], heads, width * 2)
    return torch.chunk(per_head, 2, dim=-1)


def _validate_attention_mask(
    attention_mask: torch.Tensor,
    *,
    batch_size: int,
    num_heads: int,
    query_length: int,
    key_length: int,
) -> tuple[torch.Tensor, bool]:
    if not isinstance(attention_mask, torch.Tensor):
        raise TypeError("attention_mask must be a torch tensor")
    if attention_mask.ndim == 2:
        if tuple(attention_mask.shape) != (batch_size, key_length):
            raise ValueError(
                f"2-D attention_mask must have shape {(batch_size, key_length)}, got "
                f"{tuple(attention_mask.shape)}"
            )
        return attention_mask[:, None, None, :], True
    if attention_mask.ndim == 3:
        if tuple(attention_mask.shape) != (batch_size, query_length, key_length):
            raise ValueError(
                "3-D attention_mask must have shape "
                f"{(batch_size, query_length, key_length)}, got "
                f"{tuple(attention_mask.shape)}"
            )
        return attention_mask[:, None, :, :], attention_mask.dtype == torch.bool
    if attention_mask.ndim == 4:
        if attention_mask.shape[0] != batch_size:
            raise ValueError("4-D attention_mask batch size does not match projections")
        if attention_mask.shape[1] not in (1, num_heads):
            raise ValueError(
                "4-D attention_mask head axis must be 1 or num_attention_heads"
            )
        if tuple(attention_mask.shape[2:]) != (query_length, key_length):
            raise ValueError(
                "4-D attention_mask query/key axes must have shape "
                f"{(query_length, key_length)}"
            )
        return attention_mask, attention_mask.dtype == torch.bool
    raise ValueError("attention_mask must have 2, 3, or 4 dimensions")


def _full_attention_work(
    projected_query_gate: torch.Tensor,
    projected_key: torch.Tensor,
    projected_value: torch.Tensor,
    *,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    position_ids: torch.Tensor | None = None,
    state: AttentionState | None = None,
    attention_mask: torch.Tensor | None = None,
    require_crsa_usage: bool,
    native_support: bool,
    rope_theta: float = 10_000_000.0,
    rotary_dim: int | None = None,
    partial_rotary_factor: float = 0.25,
    mrope_section: Sequence[int] | None = None,
    mrope_interleaved: bool = True,
    rms_norm_eps: float = 1e-6,
) -> _FullAttentionWork:
    """Compute tensors shared by ordinary and native full-attention arms."""

    query_gate = _floating_tensor(projected_query_gate, "projected_query_gate", ndim=3)
    key_projection = _floating_tensor(projected_key, "projected_key", ndim=3)
    value_projection = _floating_tensor(projected_value, "projected_value", ndim=3)
    heads = _positive_int(num_attention_heads, "num_attention_heads")
    kv_heads = _positive_int(num_key_value_heads, "num_key_value_heads")
    width = _positive_int(head_dim, "head_dim")
    if heads % kv_heads:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    if (
        query_gate.shape[:2] != key_projection.shape[:2]
        or query_gate.shape[:2] != value_projection.shape[:2]
    ):
        raise ValueError("all attention projections must share batch and sequence axes")
    _same_device_dtype(query_gate, key_projection, "projected_key")
    _same_device_dtype(query_gate, value_projection, "projected_value")
    if key_projection.shape[-1] != kv_heads * width:
        raise ValueError(
            f"projected_key feature size must be {kv_heads * width}, got "
            f"{key_projection.shape[-1]}"
        )
    if value_projection.shape[-1] != kv_heads * width:
        raise ValueError(
            f"projected_value feature size must be {kv_heads * width}, got "
            f"{value_projection.shape[-1]}"
        )
    batch_size, sequence_length = query_gate.shape[:2]
    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError(
            "attention projections require non-empty batch and sequence axes"
        )

    query, gate = split_query_gate(query_gate, heads, width)
    key = key_projection.reshape(batch_size, sequence_length, kv_heads, width)
    value = value_projection.reshape(batch_size, sequence_length, kv_heads, width)
    query = rms_norm(query, q_norm_weight, rms_norm_eps).transpose(1, 2)
    key = rms_norm(key, k_norm_weight, rms_norm_eps).transpose(1, 2)
    value = value.transpose(1, 2)

    past_length = 0
    if state is not None:
        if not isinstance(state, AttentionState):
            raise TypeError("state must be an AttentionState")
        expected = (batch_size, kv_heads, state.length, width)
        if tuple(state.key.shape) != expected:
            raise ValueError(
                f"attention state shape {tuple(state.key.shape)} does not match {expected}"
            )
        _same_device_dtype(key, state.key, "state.key")
        _same_device_dtype(value, state.value, "state.value")
        past_length = state.length
        if require_crsa_usage != (state.crsa_log_usage is not None):
            raise ValueError(
                "attention state CRSA usage does not match the native intervention"
            )

    if rotary_dim is None:
        factor = _positive_float(partial_rotary_factor, "partial_rotary_factor")
        resolved_rotary_dim = int(width * factor)
    else:
        resolved_rotary_dim = _positive_int(rotary_dim, "rotary_dim")
    if resolved_rotary_dim > width or resolved_rotary_dim % 2:
        raise ValueError("rotary_dim must be even and no larger than head_dim")

    if position_ids is None:
        positions = torch.arange(
            past_length,
            past_length + sequence_length,
            device=query.device,
            dtype=torch.long,
        ).expand(batch_size, -1)
    else:
        positions = position_ids
        if not isinstance(positions, torch.Tensor):
            raise TypeError("position_ids must be a torch tensor")
        if positions.ndim == 2 and tuple(positions.shape) != (
            batch_size,
            sequence_length,
        ):
            raise ValueError(
                f"position_ids must have shape {(batch_size, sequence_length)}, got "
                f"{tuple(positions.shape)}"
            )
        if positions.ndim == 3 and tuple(positions.shape) != (
            3,
            batch_size,
            sequence_length,
        ):
            raise ValueError(
                "three-axis position_ids must have shape "
                f"{(3, batch_size, sequence_length)}"
            )
        if positions.device != query.device:
            positions = positions.to(query.device)
    cosine, sine = rope_cos_sin(
        positions,
        resolved_rotary_dim,
        theta=rope_theta,
        mrope_section=mrope_section,
        mrope_interleaved=mrope_interleaved,
        dtype=query.dtype,
    )
    query, key = apply_rotary_pos_emb(query, key, cosine, sine)

    if state is not None:
        key = torch.cat((state.key, key), dim=2)
        value = torch.cat((state.value, value), dim=2)
    key_length = key.shape[2]

    repetitions = heads // kv_heads
    repeated_key = key.repeat_interleave(repetitions, dim=1)
    repeated_value = value.repeat_interleave(repetitions, dim=1)
    scores = torch.matmul(query, repeated_key.transpose(2, 3)) * (width**-0.5)
    query_index = torch.arange(sequence_length, device=scores.device)[:, None]
    key_index = torch.arange(key_length, device=scores.device)[None, :]
    causal = key_index <= (past_length + query_index)
    allowed = causal.reshape(1, 1, sequence_length, key_length)
    scores = scores.masked_fill(
        ~allowed,
        torch.finfo(scores.dtype).min,
    )

    if attention_mask is not None:
        mask, is_validity_mask = _validate_attention_mask(
            attention_mask,
            batch_size=batch_size,
            num_heads=heads,
            query_length=sequence_length,
            key_length=key_length,
        )
        if mask.device != scores.device:
            raise ValueError("attention_mask must be on the same device as projections")
        if is_validity_mask or not mask.is_floating_point():
            validity = mask.bool()
            scores = scores.masked_fill(~validity, torch.finfo(scores.dtype).min)
            if native_support:
                allowed = allowed & validity
        else:
            scores = scores + mask.to(dtype=scores.dtype)
            if native_support:
                if bool((torch.isnan(mask) | torch.isposinf(mask)).any().item()):
                    raise ValueError(
                        "floating attention_mask may not contain NaN or +inf"
                    )
                allowed = allowed & ~torch.isneginf(mask)
                scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)

    base_probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return _FullAttentionWork(
        gate=gate,
        key=key,
        value=value,
        repeated_value=repeated_value,
        scores=scores,
        allowed=allowed,
        base_probabilities=base_probabilities,
        batch_size=batch_size,
        sequence_length=sequence_length,
        heads=heads,
        width=width,
        past_length=past_length,
    )


def _full_attention_output(
    work: _FullAttentionWork,
    probabilities: torch.Tensor,
) -> torch.Tensor:
    output = torch.matmul(probabilities, work.repeated_value)
    output = output.transpose(1, 2).contiguous()
    output = output * torch.sigmoid(work.gate)
    return output.reshape(
        work.batch_size,
        work.sequence_length,
        work.heads * work.width,
    )


def _route_native_attention(
    work: _FullAttentionWork,
    intervention: Qwen38NativeHeadCrsa,
    prior_log_usage: torch.Tensor | None,
    *,
    tokenwise_usage: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, NativeHeadCrsaEvidence]:
    probabilities = work.base_probabilities
    if intervention.active:
        probabilities = probabilities.masked_fill(~work.allowed, 0.0)
    return intervention.route(
        work.scores,
        probabilities,
        query_start=work.past_length,
        allowed=work.allowed,
        prior_log_usage=prior_log_usage,
        tokenwise_usage=tokenwise_usage,
    )


def _validate_native_attention_hook(
    native_head_crsa: Qwen38NativeHeadCrsa | None,
    native_head_crsa_observer: Callable[[NativeHeadCrsaEvidence], None] | None,
    *,
    required: bool,
) -> Qwen38NativeHeadCrsa | None:
    if native_head_crsa is None:
        if required:
            raise TypeError("native_head_crsa must be a Qwen38NativeHeadCrsa")
        if native_head_crsa_observer is not None:
            raise ValueError(
                "native_head_crsa_observer requires an active native_head_crsa hook"
            )
        return None
    if not isinstance(native_head_crsa, Qwen38NativeHeadCrsa):
        suffix = "" if required else " or None"
        raise TypeError(f"native_head_crsa must be a Qwen38NativeHeadCrsa{suffix}")
    if native_head_crsa_observer is not None and not callable(
        native_head_crsa_observer
    ):
        raise TypeError("native_head_crsa_observer must be callable or None")
    return native_head_crsa


def full_attention_core(
    projected_query_gate: torch.Tensor,
    projected_key: torch.Tensor,
    projected_value: torch.Tensor,
    *,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    position_ids: torch.Tensor | None = None,
    state: AttentionState | None = None,
    attention_mask: torch.Tensor | None = None,
    native_head_crsa: Qwen38NativeHeadCrsa | None = None,
    native_head_crsa_observer: Callable[[NativeHeadCrsaEvidence], None] | None = None,
    native_head_crsa_tokenwise_usage: bool = False,
    rope_theta: float = 10_000_000.0,
    rotary_dim: int | None = None,
    partial_rotary_factor: float = 0.25,
    mrope_section: Sequence[int] | None = None,
    mrope_interleaved: bool = True,
    rms_norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, AttentionState]:
    """Run gated causal GQA from sequentially streamed projection outputs.

    The returned activation is pre-``o_proj`` with shape
    ``[batch, sequence, num_attention_heads * head_dim]``.  The returned state
    contains RoPE-applied keys and raw values for exact continuation.
    """

    if not isinstance(native_head_crsa_tokenwise_usage, bool):
        raise TypeError("native_head_crsa_tokenwise_usage must be a boolean")
    intervention = _validate_native_attention_hook(
        native_head_crsa,
        native_head_crsa_observer,
        required=False,
    )
    work = _full_attention_work(
        projected_query_gate,
        projected_key,
        projected_value,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        position_ids=position_ids,
        state=state,
        attention_mask=attention_mask,
        require_crsa_usage=intervention is not None and intervention.active,
        native_support=intervention is not None,
        rope_theta=rope_theta,
        rotary_dim=rotary_dim,
        partial_rotary_factor=partial_rotary_factor,
        mrope_section=mrope_section,
        mrope_interleaved=mrope_interleaved,
        rms_norm_eps=rms_norm_eps,
    )
    probabilities = work.base_probabilities
    next_log_usage = None
    if intervention is not None:
        probabilities, next_log_usage, evidence = _route_native_attention(
            work,
            intervention,
            None if state is None else state.crsa_log_usage,
            tokenwise_usage=native_head_crsa_tokenwise_usage,
        )
        if native_head_crsa_observer is not None:
            native_head_crsa_observer(evidence)
    output = _full_attention_output(work, probabilities)
    next_state = AttentionState(
        key=work.key,
        value=work.value,
        crsa_log_usage=next_log_usage,
    )
    return output, next_state


def full_attention_fork_core(
    projected_query_gate: torch.Tensor,
    projected_key: torch.Tensor,
    projected_value: torch.Tensor,
    *,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    native_head_crsa: Qwen38NativeHeadCrsa,
    position_ids: torch.Tensor | None = None,
    off_state: AttentionState | None = None,
    native_state: AttentionState | None = None,
    attention_mask: torch.Tensor | None = None,
    native_head_crsa_observer: Callable[[NativeHeadCrsaEvidence], None] | None = None,
    rope_theta: float = 10_000_000.0,
    rotary_dim: int | None = None,
    partial_rotary_factor: float = 0.25,
    mrope_section: Sequence[int] | None = None,
    mrope_interleaved: bool = True,
    rms_norm_eps: float = 1e-6,
) -> tuple[
    torch.Tensor,
    AttentionState,
    torch.Tensor,
    AttentionState,
    NativeHeadCrsaEvidence,
]:
    """Fork one shared attention pass into exact off and native Head-CRSA arms.

    Query/key/value normalization, RoPE, causal score construction, masking and
    base softmax are evaluated exactly once.  Each arm then performs its own
    ``P @ V`` and output gate without concatenating the two arithmetic paths.
    Returned K/V tensors are the same immutable-by-contract tensor objects;
    only the native state owns CRSA usage history.
    """

    intervention = _validate_native_attention_hook(
        native_head_crsa,
        native_head_crsa_observer,
        required=True,
    )
    assert intervention is not None
    if (off_state is None) != (native_state is None):
        raise ValueError("off_state and native_state must both be present or both None")
    if off_state is not None:
        if not isinstance(off_state, AttentionState):
            raise TypeError("off_state must be an AttentionState")
        if not isinstance(native_state, AttentionState):
            raise TypeError("native_state must be an AttentionState")
        if off_state.crsa_log_usage is not None:
            raise ValueError("off_state may not retain CRSA usage")
        needs_usage = intervention.active
        if needs_usage != (native_state.crsa_log_usage is not None):
            raise ValueError(
                "native_state CRSA usage does not match the native intervention"
            )
        if off_state.key.shape != native_state.key.shape:
            raise ValueError("fork state K/V shapes must match")
        _same_device_dtype(off_state.key, native_state.key, "native_state.key")
        _same_device_dtype(off_state.value, native_state.value, "native_state.value")
        if not torch.equal(off_state.key, native_state.key) or not torch.equal(
            off_state.value,
            native_state.value,
        ):
            raise ValueError("fork states must contain identical K/V history")

    work = _full_attention_work(
        projected_query_gate,
        projected_key,
        projected_value,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        position_ids=position_ids,
        state=off_state,
        attention_mask=attention_mask,
        require_crsa_usage=False,
        native_support=True,
        rope_theta=rope_theta,
        rotary_dim=rotary_dim,
        partial_rotary_factor=partial_rotary_factor,
        mrope_section=mrope_section,
        mrope_interleaved=mrope_interleaved,
        rms_norm_eps=rms_norm_eps,
    )
    native_probabilities, next_log_usage, evidence = _route_native_attention(
        work,
        intervention,
        None if native_state is None else native_state.crsa_log_usage,
    )
    off_output = _full_attention_output(work, work.base_probabilities)
    native_output = _full_attention_output(work, native_probabilities)
    off_next_state = AttentionState(key=work.key, value=work.value)
    native_next_state = AttentionState(
        key=work.key,
        value=work.value,
        crsa_log_usage=next_log_usage,
    )
    if native_head_crsa_observer is not None:
        native_head_crsa_observer(evidence)
    return (
        off_output,
        off_next_state,
        native_output,
        native_next_state,
        evidence,
    )


def causal_depthwise_conv(
    projected_qkv: torch.Tensor,
    weight: torch.Tensor,
    *,
    conv_state: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply exact causal depthwise convolution + SiLU and update its cache.

    ``projected_qkv`` uses ``[B, S, C]``.  The returned cache uses ``[B, C, K]``
    and stores raw (pre-convolution) inputs.  Keeping ``K`` rather than ``K-1``
    values intentionally matches Qwen's official cache/update kernel.
    """

    projected = _floating_tensor(projected_qkv, "projected_qkv", ndim=3)
    kernel = _floating_tensor(weight, "weight")
    if kernel.ndim == 3:
        if kernel.shape[1] != 1:
            raise ValueError(
                "3-D depthwise weight must have shape [channels, 1, kernel]"
            )
        kernel = kernel[:, 0, :]
    elif kernel.ndim != 2:
        raise ValueError(
            "weight must have shape [channels, kernel] or [channels, 1, kernel]"
        )
    batch_size, sequence_length, channels = projected.shape
    if batch_size <= 0 or sequence_length <= 0 or channels <= 0:
        raise ValueError(
            "projected_qkv must have non-empty batch, sequence, and channel axes"
        )
    if kernel.shape[0] != channels or kernel.shape[1] <= 0:
        raise ValueError(
            f"weight shape {tuple(kernel.shape)} does not match {channels} channels"
        )
    if kernel.device != projected.device:
        raise ValueError("weight and projected_qkv must be on the same device")
    kernel_size = kernel.shape[1]
    conv_bias = None
    if bias is not None:
        conv_bias = _floating_tensor(bias, "bias", ndim=1)
        if conv_bias.numel() != channels:
            raise ValueError("bias feature size must equal convolution channels")
        if conv_bias.device != projected.device:
            raise ValueError("bias and projected_qkv must be on the same device")

    channels_first = projected.transpose(1, 2)
    if conv_state is None:
        convolution_input = channels_first.to(dtype=kernel.dtype)
        convolved = F.conv1d(
            convolution_input,
            kernel.unsqueeze(1),
            None if conv_bias is None else conv_bias.to(kernel.dtype),
            padding=kernel_size - 1,
            groups=channels,
        )[..., :sequence_length]
        if sequence_length >= kernel_size:
            next_state = channels_first[..., -kernel_size:]
        else:
            next_state = F.pad(channels_first, (kernel_size - sequence_length, 0))
    else:
        previous = _floating_tensor(conv_state, "conv_state", ndim=3)
        expected = (batch_size, channels, kernel_size)
        if tuple(previous.shape) != expected:
            raise ValueError(
                f"conv_state shape {tuple(previous.shape)} does not match {expected}"
            )
        _same_device_dtype(channels_first, previous, "conv_state")
        combined = torch.cat((previous, channels_first), dim=-1)
        convolved = F.conv1d(
            combined.to(dtype=kernel.dtype),
            kernel.unsqueeze(1),
            None if conv_bias is None else conv_bias.to(kernel.dtype),
            padding=0,
            groups=channels,
        )[..., -sequence_length:]
        next_state = combined[..., -kernel_size:]

    activated = F.silu(convolved).to(dtype=projected.dtype)
    return activated.transpose(1, 2).contiguous(), next_state.contiguous()


def recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    l2_norm_eps: float = 1e-6,
    _probe_delta_norms: list[float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the official causal recurrent gated-delta inference rule.

    Q/K/V use ``[B, S, H, D]`` and ``g`` is the log decay.  Recurrence is
    deliberately accumulated in float32, including when activations are BF16.
    """

    q = _floating_tensor(query, "query", ndim=4)
    k = _floating_tensor(key, "key", ndim=4)
    v = _floating_tensor(value, "value", ndim=4)
    decay = _floating_tensor(g, "g", ndim=3)
    step = _floating_tensor(beta, "beta", ndim=3)
    if q.shape != k.shape:
        raise ValueError("query and key must have identical shapes")
    if q.shape[:3] != v.shape[:3]:
        raise ValueError("query/key/value must share batch, sequence, and head axes")
    if tuple(decay.shape) != q.shape[:3] or tuple(step.shape) != q.shape[:3]:
        raise ValueError("g and beta must have shape [batch, sequence, heads]")
    for name, tensor in (("key", k), ("value", v), ("g", decay), ("beta", step)):
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on the same device as query")
    batch_size, sequence_length, heads, key_width = q.shape
    value_width = v.shape[-1]
    if min(batch_size, sequence_length, heads, key_width, value_width) <= 0:
        raise ValueError("gated delta tensors must have non-empty dimensions")

    q = l2_normalize(q, eps=l2_norm_eps).float() * (key_width**-0.5)
    k = l2_normalize(k, eps=l2_norm_eps).float()
    v = v.float()
    decay = decay.float()
    step = step.float()
    expected_state = (batch_size, heads, key_width, value_width)
    if initial_state is None:
        recurrent = torch.zeros(expected_state, device=q.device, dtype=torch.float32)
    else:
        recurrent = _floating_tensor(initial_state, "initial_state", ndim=4)
        if tuple(recurrent.shape) != expected_state:
            raise ValueError(
                f"initial_state shape {tuple(recurrent.shape)} does not match "
                f"{expected_state}"
            )
        if recurrent.device != q.device:
            raise ValueError("initial_state must be on the same device as query")
        recurrent = recurrent.float()

    outputs: list[torch.Tensor] = []
    for index in range(sequence_length):
        q_t = q[:, index]
        k_t = k[:, index]
        v_t = v[:, index]
        recurrent = recurrent * decay[:, index].exp()[..., None, None]
        remembered_value = (recurrent * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - remembered_value) * step[:, index].unsqueeze(-1)
        if _probe_delta_norms is not None:
            _probe_delta_norms.append(_probe_mean_vector_norm(delta, delta.shape[-1]))
        recurrent = recurrent + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((recurrent * q_t.unsqueeze(-1)).sum(dim=-2))

    output = torch.stack(outputs, dim=1).to(dtype=query.dtype)
    return output, recurrent


def gated_delta_net_core(
    projected_qkv: torch.Tensor,
    projected_z: torch.Tensor,
    projected_b: torch.Tensor,
    projected_a: torch.Tensor,
    *,
    conv1d_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    state: DeltaNetState | None = None,
    conv1d_bias: torch.Tensor | None = None,
    rms_norm_eps: float = 1e-6,
    probe: Callable[[DeltaNetProbe], None] | None = None,
) -> tuple[torch.Tensor, DeltaNetState]:
    """Run Qwen3.5 Gated DeltaNet from individually streamed projections.

    The caller applies padding to hidden states before computing these four
    projections.  The returned activation is pre-``out_proj`` with shape
    ``[B, S, num_value_heads * value_head_dim]``.
    """

    qkv = _floating_tensor(projected_qkv, "projected_qkv", ndim=3)
    z = _floating_tensor(projected_z, "projected_z", ndim=3)
    b = _floating_tensor(projected_b, "projected_b", ndim=3)
    a = _floating_tensor(projected_a, "projected_a", ndim=3)
    key_heads = _positive_int(num_key_heads, "num_key_heads")
    value_heads = _positive_int(num_value_heads, "num_value_heads")
    key_width = _positive_int(key_head_dim, "key_head_dim")
    value_width = _positive_int(value_head_dim, "value_head_dim")
    if value_heads % key_heads:
        raise ValueError("num_value_heads must be divisible by num_key_heads")
    if (
        qkv.shape[:2] != z.shape[:2]
        or qkv.shape[:2] != b.shape[:2]
        or qkv.shape[:2] != a.shape[:2]
    ):
        raise ValueError("all DeltaNet projections must share batch and sequence axes")
    for name, tensor in (("projected_z", z), ("projected_b", b), ("projected_a", a)):
        _same_device_dtype(qkv, tensor, name)
    batch_size, sequence_length = qkv.shape[:2]
    key_features = key_heads * key_width
    value_features = value_heads * value_width
    expected_qkv = key_features * 2 + value_features
    if qkv.shape[-1] != expected_qkv:
        raise ValueError(
            f"projected_qkv feature size {qkv.shape[-1]} does not match "
            f"2 * {key_features} + {value_features} = {expected_qkv}"
        )
    if z.shape[-1] != value_features:
        raise ValueError(f"projected_z feature size must be {value_features}")
    if b.shape[-1] != value_heads or a.shape[-1] != value_heads:
        raise ValueError(f"projected_b/projected_a feature size must be {value_heads}")

    previous_conv = None
    previous_recurrent = None
    if state is not None:
        if not isinstance(state, DeltaNetState):
            raise TypeError("state must be a DeltaNetState")
        previous_conv = state.conv
        previous_recurrent = state.recurrent
    mixed, next_conv = causal_depthwise_conv(
        qkv,
        conv1d_weight,
        conv_state=previous_conv,
        bias=conv1d_bias,
    )
    query, key, value = torch.split(
        mixed,
        (key_features, key_features, value_features),
        dim=-1,
    )
    query = query.reshape(batch_size, sequence_length, key_heads, key_width)
    key = key.reshape(batch_size, sequence_length, key_heads, key_width)
    value = value.reshape(batch_size, sequence_length, value_heads, value_width)
    raw_query = query
    raw_key = key

    beta = b.sigmoid()
    a_log = _floating_tensor(A_log, "A_log", ndim=1)
    delta_bias = _floating_tensor(dt_bias, "dt_bias", ndim=1)
    if a_log.numel() != value_heads or delta_bias.numel() != value_heads:
        raise ValueError(f"A_log and dt_bias must each contain {value_heads} values")
    if a_log.device != qkv.device or delta_bias.device != qkv.device:
        raise ValueError("A_log, dt_bias, and projections must be on the same device")
    log_decay = -a_log.float().exp() * F.softplus(a.float() + delta_bias.float())

    repetitions = value_heads // key_heads
    if repetitions > 1:
        query = query.repeat_interleave(repetitions, dim=2)
        key = key.repeat_interleave(repetitions, dim=2)
    probe_delta_norms: list[float] | None = [] if probe is not None else None
    core_output, next_recurrent = recurrent_gated_delta_rule(
        query,
        key,
        value,
        log_decay,
        beta,
        initial_state=previous_recurrent,
        _probe_delta_norms=probe_delta_norms,
    )

    if probe is not None:
        if not probe_delta_norms:
            raise RuntimeError("DeltaNet probe captured no recurrent updates")
        beta_mean, beta_std = _probe_mean_sample_std(beta)
        decay = log_decay.exp()
        decay_mean, decay_std = _probe_mean_sample_std(decay)
        probe(
            DeltaNetProbe(
                beta_mean=beta_mean,
                beta_std=beta_std,
                decay_mean=decay_mean,
                decay_std=decay_std,
                conv_norm=_probe_mean_vector_norm(mixed, mixed.shape[-1]),
                q_norm=_probe_mean_vector_norm(raw_query, key_features),
                k_norm=_probe_mean_vector_norm(raw_key, key_features),
                v_norm=_probe_mean_vector_norm(value, value_width),
                delta_norm=math.fsum(probe_delta_norms) / len(probe_delta_norms),
            )
        )

    core_output = core_output.reshape(-1, value_width)
    z = z.reshape(-1, value_width)
    core_output = rms_norm_gated(core_output, norm_weight, z, rms_norm_eps)
    output = core_output.reshape(batch_size, sequence_length, value_features)
    return output, DeltaNetState(conv=next_conv, recurrent=next_recurrent)
