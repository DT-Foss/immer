"""Portable DeepSeek-V4 reference kernels for CPU and Apple MPS.

The official checkpoint ships TileLang/CUDA kernels.  This module keeps their
math and tensor conventions, but expresses them with ordinary PyTorch
operations so the streamed runtime can execute on Apple silicon.  Compute that
is numerically sensitive is deliberately performed in float32, matching the
published kernels.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _float_tensor_like(value: Any, reference: torch.Tensor, name: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    return tensor.to(device=reference.device, dtype=torch.float32)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply the checkpoint RMSNorm formula with float32 accumulation."""

    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        raise TypeError("x must be a floating-point torch tensor")
    if x.ndim == 0:
        raise ValueError("x must have a feature dimension")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    w = _float_tensor_like(weight, x, "weight")
    if w.ndim != 1 or w.numel() != x.shape[-1]:
        raise ValueError(
            f"weight shape {tuple(w.shape)} does not match feature size {x.shape[-1]}"
        )
    dtype = x.dtype
    values = x.float()
    values = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)
    return (values * w).to(dtype=dtype)


def precompute_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int = 0,
    base: float = 10_000.0,
    factor: float = 1.0,
    beta_fast: int = 32,
    beta_slow: int = 1,
) -> torch.Tensor:
    """Build the official complex RoPE/YaRN frequency table on CPU.

    Keeping the small complex table on CPU is intentional: MPS does not expose
    every complex operation on every supported macOS/PyTorch combination.
    :func:`apply_rotary_emb` consumes this table using real arithmetic and moves
    only its cosine/sine views to the activation device.
    """

    _positive_int(dim, "dim")
    _positive_int(seqlen, "seqlen")
    if dim % 2:
        raise ValueError("dim must be even")
    if isinstance(original_seq_len, bool) or not isinstance(original_seq_len, int):
        raise TypeError("original_seq_len must be an integer")
    if original_seq_len < 0:
        raise ValueError("original_seq_len must be non-negative")
    if not math.isfinite(base) or base <= 1:
        raise ValueError("base must be finite and greater than one")
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("factor must be finite and positive")
    _positive_int(beta_fast, "beta_fast")
    _positive_int(beta_slow, "beta_slow")

    freqs = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cpu") / dim)
    )
    if original_seq_len > 0:

        def correction_dim(num_rotations: int) -> float:
            return (
                dim
                * math.log(original_seq_len / (num_rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(correction_dim(beta_fast)), 0)
        high = min(math.ceil(correction_dim(beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        ramp = (torch.arange(dim // 2, dtype=torch.float32, device="cpu") - low) / (
            high - low
        )
        smooth = 1.0 - ramp.clamp(0.0, 1.0)
        freqs = freqs / factor * (1.0 - smooth) + freqs * smooth

    angles = torch.outer(torch.arange(seqlen, dtype=torch.float32, device="cpu"), freqs)
    return torch.polar(torch.ones_like(angles), angles)


def _cos_sin(
    freqs_cis: torch.Tensor, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(freqs_cis, torch.Tensor):
        freqs_cis = torch.as_tensor(freqs_cis)
    if freqs_cis.is_complex():
        cos = freqs_cis.real
        sin = freqs_cis.imag
    elif freqs_cis.ndim >= 1 and freqs_cis.shape[-1] == 2:
        cos, sin = freqs_cis.unbind(dim=-1)
    else:
        raise TypeError(
            "freqs_cis must be complex or have a final (cos, sin) pair axis"
        )
    return (
        cos.to(device=x.device, dtype=torch.float32),
        sin.to(device=x.device, dtype=torch.float32),
    )


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply RoPE in-place using real operations and return ``x``.

    ``x`` follows the published ``[batch, sequence, ..., rotary_dim]`` layout.
    Both the official complex frequency table and a real ``[..., 2]``
    ``(cos, sin)`` representation are accepted, making the operation safe on
    CPU and MPS.
    """

    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        raise TypeError("x must be a floating-point torch tensor")
    if x.ndim < 3:
        raise ValueError("x must have [batch, sequence, ..., feature] dimensions")
    if x.shape[-1] % 2:
        raise ValueError("rotary feature dimension must be even")
    cos, sin = _cos_sin(freqs_cis, x)
    pairs = x.shape[-1] // 2
    if cos.ndim != 2 or cos.shape != (x.shape[1], pairs):
        raise ValueError(
            f"frequency shape {tuple(cos.shape)} does not match "
            f"sequence/rotary shape {(x.shape[1], pairs)}"
        )
    broadcast = (1, x.shape[1], *((1,) * (x.ndim - 3)), pairs)
    cos = cos.reshape(broadcast)
    sin = sin.reshape(broadcast)
    if inverse:
        sin = -sin

    values = x.float().unflatten(-1, (-1, 2))
    real, imag = values.unbind(dim=-1)
    rotated = torch.stack(
        (real * cos - imag * sin, real * sin + imag * cos), dim=-1
    ).flatten(-2)
    x.copy_(rotated.to(dtype=x.dtype))
    return x


def hadamard_transform(x: torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
    """Portable Walsh-Hadamard transform used by the V4 indexer QAT path."""

    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        raise TypeError("x must be a floating-point torch tensor")
    if x.ndim == 0 or x.shape[-1] < 1:
        raise ValueError("x must have a non-empty feature dimension")
    width = x.shape[-1]
    if width & (width - 1):
        raise ValueError("Hadamard feature dimension must be a power of two")
    dtype = x.dtype
    values = x.float()
    step = 1
    while step < width:
        paired = values.reshape(*values.shape[:-1], -1, 2, step)
        left = paired[..., 0, :]
        right = paired[..., 1, :]
        values = torch.cat((left + right, left - right), dim=-1).reshape_as(values)
        step *= 2
    if normalize:
        values = values * (width**-0.5)
    return values.to(dtype=dtype)


def window_indices(
    window_size: int,
    batch_size: int,
    seqlen: int,
    start_pos: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build causal sliding-window indices for prefill or one-token decode.

    Decode indices address the circular KV buffer and are returned oldest to
    newest.  ``-1`` denotes padding, exactly as in the published sparse kernel.
    """

    _positive_int(window_size, "window_size")
    _positive_int(batch_size, "batch_size")
    _positive_int(seqlen, "seqlen")
    if isinstance(start_pos, bool) or not isinstance(start_pos, int) or start_pos < 0:
        raise ValueError("start_pos must be a non-negative integer")
    if start_pos > 0 and seqlen != 1:
        raise ValueError("decode index construction requires seqlen == 1")

    target = torch.device("cpu" if device is None else device)
    if start_pos >= window_size - 1 and start_pos > 0:
        slot = start_pos % window_size
        matrix = torch.cat(
            (
                torch.arange(slot + 1, window_size, device=target),
                torch.arange(0, slot + 1, device=target),
            )
        )
    elif start_pos > 0:
        matrix = F.pad(
            torch.arange(start_pos + 1, device=target),
            (0, window_size - start_pos - 1),
            value=-1,
        )
    else:
        width = min(seqlen, window_size)
        query = torch.arange(seqlen, device=target).unsqueeze(1)
        matrix = (query - window_size + 1).clamp(min=0) + torch.arange(
            width, device=target
        )
        matrix = torch.where(matrix > query, -1, matrix)
    return (
        matrix.to(dtype=torch.int32)
        .unsqueeze(0)
        .expand(batch_size, -1, -1)
        .contiguous()
    )


def compressed_indices(
    ratio: int,
    batch_size: int,
    seqlen: int,
    start_pos: int,
    offset: int = 0,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build causal indices for non-learned compressed KV blocks."""

    _positive_int(ratio, "ratio")
    _positive_int(batch_size, "batch_size")
    _positive_int(seqlen, "seqlen")
    if isinstance(start_pos, bool) or not isinstance(start_pos, int) or start_pos < 0:
        raise ValueError("start_pos must be a non-negative integer")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if start_pos > 0 and seqlen != 1:
        raise ValueError("decode index construction requires seqlen == 1")

    target = torch.device("cpu" if device is None else device)
    if start_pos > 0:
        matrix = torch.arange((start_pos + 1) // ratio, device=target) + offset
    else:
        n_blocks = seqlen // ratio
        matrix = torch.arange(n_blocks, device=target).repeat(seqlen, 1)
        complete = torch.arange(1, seqlen + 1, device=target).unsqueeze(1) // ratio
        matrix = torch.where(matrix >= complete, -1, matrix + offset)
    return (
        matrix.to(dtype=torch.int32)
        .unsqueeze(0)
        .expand(batch_size, -1, -1)
        .contiguous()
    )


def sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    indices: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Portable sparse MQA with the official learnable attention sink.

    The sink is an additional unscaled softmax logit whose value is zero.  It
    therefore contributes only to the denominator.  Duplicate indices retain
    duplicate mass, matching the gather-based CUDA kernel.
    """

    if not isinstance(q, torch.Tensor) or not q.is_floating_point() or q.ndim != 4:
        raise TypeError("q must be a floating [batch, sequence, heads, dim] tensor")
    if not isinstance(kv, torch.Tensor) or not kv.is_floating_point() or kv.ndim != 3:
        raise TypeError("kv must be a floating [batch, cache, dim] tensor")
    if q.shape[0] != kv.shape[0] or q.shape[-1] != kv.shape[-1]:
        raise ValueError("q and kv batch/feature dimensions must match")
    if not isinstance(indices, torch.Tensor) or indices.ndim != 3:
        raise TypeError("indices must be [batch, sequence, topk]")
    if tuple(indices.shape[:2]) != tuple(q.shape[:2]):
        raise ValueError("indices batch/sequence dimensions must match q")
    if indices.is_floating_point() or indices.is_complex():
        raise TypeError("indices must contain integers")
    sink = _float_tensor_like(attn_sink, q, "attn_sink")
    if sink.ndim != 1 or sink.numel() != q.shape[2]:
        raise ValueError("attn_sink must contain one logit per attention head")
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError("scale must be finite and positive")

    idx = indices.to(device=q.device, dtype=torch.long)
    valid = idx != -1
    if torch.any(idx < -1) or torch.any(idx >= kv.shape[1]):
        raise IndexError("sparse attention index is outside the KV cache")
    safe = idx.clamp(min=0)
    batch = torch.arange(q.shape[0], device=q.device).view(-1, 1, 1)
    values = kv.to(device=q.device, dtype=torch.float32)[batch, safe]

    # Match the published TileLang kernel's 64-wide online-softmax equation.
    # In particular, each block's *unnormalized* exponentials are rounded to
    # BF16 before the value GEMM; the denominator and running rescale remain
    # FP32, and the learnable sink is added only after the final block.  A
    # global torch.softmax (or casting normalized probabilities) changes logits.
    running_max = torch.full(
        q.shape[:-1], float("-inf"), dtype=torch.float32, device=q.device
    )
    denominator = torch.zeros_like(running_max)
    numerator = torch.zeros(
        (*q.shape[:-1], q.shape[-1]),
        dtype=torch.float32,
        device=q.device,
    )
    for start in range(0, values.shape[-2], 64):
        stop = min(start + 64, values.shape[-2])
        block_values = values[..., start:stop, :]
        block_valid = valid[..., start:stop].unsqueeze(2)
        # Score each 64-wide tile independently.  Computing the whole top-k
        # score matrix and slicing it afterwards can select a different GEMM
        # kernel/reduction order than the published tiled loop.
        block_scores = (
            torch.einsum("bshd,bskd->bshk", q.float(), block_values)
            * float(scale)
        ).masked_fill(~block_valid, float("-inf"))
        block_max = block_scores.amax(dim=-1)
        next_max = torch.maximum(running_max, block_max)
        finite = torch.isfinite(next_max)
        rescale = torch.where(
            finite,
            torch.exp(running_max - next_max),
            torch.ones_like(next_max),
        )
        exponentials = torch.where(
            block_valid,
            torch.exp(block_scores - next_max.unsqueeze(-1)),
            torch.zeros_like(block_scores),
        )
        denominator = denominator * rescale + exponentials.sum(dim=-1)
        numerator = numerator * rescale.unsqueeze(-1) + torch.einsum(
            "bshk,bskd->bshd",
            exponentials.to(dtype=q.dtype).float(),
            block_values,
        )
        running_max = next_max

    any_valid = valid.any(dim=-1).unsqueeze(2)
    sink_term = torch.exp(sink.view(1, 1, -1) - running_max)
    denominator = denominator + torch.where(
        any_valid, sink_term, torch.zeros_like(sink_term)
    )
    output = torch.where(
        any_valid.unsqueeze(-1),
        numerator / denominator.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1),
        torch.zeros_like(numerator),
    )
    return output.to(dtype=q.dtype)


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split HC controls and run the published alternating normalization."""

    _positive_int(hc_mult, "hc_mult")
    _positive_int(sinkhorn_iters, "sinkhorn_iters")
    if not isinstance(mixes, torch.Tensor) or not mixes.is_floating_point():
        raise TypeError("mixes must be a floating-point torch tensor")
    if mixes.ndim < 1:
        raise ValueError("mixes must have a final control dimension")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    mix_hc = (2 + hc_mult) * hc_mult
    if mixes.shape[-1] != mix_hc:
        raise ValueError(f"mixes final dimension must be {mix_hc}")
    scale = _float_tensor_like(hc_scale, mixes, "hc_scale")
    base = _float_tensor_like(hc_base, mixes, "hc_base")
    if scale.shape != (3,):
        raise ValueError("hc_scale must have shape (3,)")
    if base.shape != (mix_hc,):
        raise ValueError(f"hc_base must have shape ({mix_hc},)")

    controls = mixes.float()
    pre = torch.sigmoid(controls[..., :hc_mult] * scale[0] + base[:hc_mult]) + eps
    post = 2.0 * torch.sigmoid(
        controls[..., hc_mult : 2 * hc_mult] * scale[1] + base[hc_mult : 2 * hc_mult]
    )
    comb = (controls[..., 2 * hc_mult :] * scale[2] + base[2 * hc_mult :]).unflatten(
        -1, (hc_mult, hc_mult)
    )
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def hc_pre(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int | None = None,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce Hyper-Connection streams before an attention or FFN branch."""

    if not isinstance(x, torch.Tensor) or not x.is_floating_point() or x.ndim < 3:
        raise TypeError("x must be a floating [..., hc, dim] tensor")
    inferred_hc = x.shape[-2]
    hc_mult = inferred_hc if hc_mult is None else hc_mult
    _positive_int(hc_mult, "hc_mult")
    if inferred_hc != hc_mult:
        raise ValueError("x Hyper-Connection axis does not match hc_mult")
    if not math.isfinite(norm_eps) or norm_eps <= 0:
        raise ValueError("norm_eps must be finite and positive")
    mix_hc = (2 + hc_mult) * hc_mult
    fn = _float_tensor_like(hc_fn, x, "hc_fn")
    if fn.shape != (mix_hc, hc_mult * x.shape[-1]):
        raise ValueError(
            f"hc_fn shape {tuple(fn.shape)} does not match "
            f"{(mix_hc, hc_mult * x.shape[-1])}"
        )

    dtype = x.dtype
    flat = x.flatten(start_dim=-2).float()
    inv_rms = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + norm_eps)
    mixes = F.linear(flat, fn) * inv_rms
    pre, post, comb = hc_split_sinkhorn(
        mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps
    )
    reduced = torch.sum(pre.unsqueeze(-1) * flat.unflatten(-1, (hc_mult, -1)), dim=-2)
    return reduced.to(dtype=dtype), post, comb


def hc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Expand a branch result and mix old HC streams into destination streams."""

    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        raise TypeError("x must be a floating-point torch tensor")
    if not isinstance(residual, torch.Tensor) or not residual.is_floating_point():
        raise TypeError("residual must be a floating-point torch tensor")
    if residual.ndim != x.ndim + 1 or residual.shape[:-2] != x.shape[:-1]:
        raise ValueError("residual must have shape [..., hc, dim] matching x")
    if residual.shape[-1] != x.shape[-1]:
        raise ValueError("residual and x feature dimensions must match")
    hc_mult = residual.shape[-2]
    post32 = _float_tensor_like(post, x, "post")
    comb32 = _float_tensor_like(comb, x, "comb")
    if post32.shape != (*x.shape[:-1], hc_mult):
        raise ValueError("post shape does not match x/HC dimensions")
    if comb32.shape != (*x.shape[:-1], hc_mult, hc_mult):
        raise ValueError("comb shape does not match x/HC dimensions")
    result = post32.unsqueeze(-1) * x.float().unsqueeze(-2)
    result = result + torch.einsum(
        "...jk,...jd->...kd", comb32, residual.to(x.device).float()
    )
    return result.to(dtype=x.dtype)


def hc_head(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    eps: float = 1e-6,
    norm_eps: float = 1e-6,
) -> torch.Tensor:
    """Collapse final Hyper-Connection streams before the decoder norm/head."""

    if not isinstance(x, torch.Tensor) or not x.is_floating_point() or x.ndim < 3:
        raise TypeError("x must be a floating [..., hc, dim] tensor")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if not math.isfinite(norm_eps) or norm_eps <= 0:
        raise ValueError("norm_eps must be finite and positive")
    hc_mult, dim = x.shape[-2:]
    fn = _float_tensor_like(hc_fn, x, "hc_fn")
    scale = _float_tensor_like(hc_scale, x, "hc_scale")
    base = _float_tensor_like(hc_base, x, "hc_base")
    if fn.shape != (hc_mult, hc_mult * dim):
        raise ValueError("hc_fn shape does not match x")
    if scale.numel() != 1:
        raise ValueError("hc_scale must contain exactly one value")
    if base.shape != (hc_mult,):
        raise ValueError("hc_base must have shape (hc_mult,)")

    dtype = x.dtype
    flat = x.flatten(start_dim=-2).float()
    inv_rms = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + norm_eps)
    mixes = F.linear(flat, fn) * inv_rms
    pre = torch.sigmoid(mixes * scale.reshape(()) + base) + eps
    result = torch.sum(pre.unsqueeze(-1) * flat.unflatten(-1, (hc_mult, dim)), dim=-2)
    return result.to(dtype=dtype)


__all__ = [
    "apply_rotary_emb",
    "compressed_indices",
    "hadamard_transform",
    "hc_head",
    "hc_post",
    "hc_pre",
    "hc_split_sinkhorn",
    "precompute_freqs_cis",
    "rms_norm",
    "sparse_attention",
    "window_indices",
]
