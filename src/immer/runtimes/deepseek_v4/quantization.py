"""Portable reference math for DeepSeek-V4 MXFP8/MXFP4 weights.

DeepSeek's published CUDA kernels use OCP E4M3 values with one E8M0 scale
per 128x128 weight tile, and packed E2M1 experts with one E8M0 scale per
32 values along K.  These routines implement the same representation in
NumPy/Torch so Apple silicon can execute it without TileLang.
"""

from __future__ import annotations

from typing import Any

import numpy as np


FP4_E2M1_TABLE = np.asarray(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=np.float32,
)


def _float_array(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype.kind != "f":
        raise TypeError(f"{name} must contain decoded floating-point values")
    return np.ascontiguousarray(result, dtype=np.float32)


def dequantize_fp8_e4m3(
    weight: Any,
    scale: Any,
    *,
    block_size: int = 128,
    output_dtype: np.dtype[Any] | type = np.float32,
) -> np.ndarray:
    """Apply DeepSeek's 2D MXFP8 block scales to decoded E4M3 values."""

    values = _float_array(weight, "weight")
    scales = _float_array(scale, "scale")
    if values.ndim != 2 or scales.ndim != 2:
        raise ValueError("MXFP8 weight and scale must both be 2D")
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size <= 0
    ):
        raise ValueError("block_size must be a positive integer")
    expected = (
        (values.shape[0] + block_size - 1) // block_size,
        (values.shape[1] + block_size - 1) // block_size,
    )
    if scales.shape != expected:
        raise ValueError(f"MXFP8 scale shape {scales.shape} does not match {expected}")
    row_scale = np.repeat(scales, block_size, axis=0)[: values.shape[0]]
    full_scale = np.repeat(row_scale, block_size, axis=1)[:, : values.shape[1]]
    return np.ascontiguousarray(values * full_scale, dtype=output_dtype)


def unpack_fp4_e2m1(packed: Any) -> np.ndarray:
    """Unpack low-nibble-first E2M1 bytes along the final K dimension."""

    raw = np.asarray(packed)
    if raw.dtype not in (np.dtype(np.int8), np.dtype(np.uint8)):
        raise TypeError("packed FP4 tensor must use int8 or uint8 storage")
    if raw.ndim == 0:
        raise ValueError("packed FP4 tensor needs at least one dimension")
    codes = np.ascontiguousarray(raw).view(np.uint8)
    result = np.empty((*codes.shape[:-1], codes.shape[-1] * 2), dtype=np.float32)
    result[..., 0::2] = FP4_E2M1_TABLE[codes & np.uint8(0x0F)]
    result[..., 1::2] = FP4_E2M1_TABLE[(codes >> np.uint8(4)) & np.uint8(0x0F)]
    return result


def dequantize_fp4_e2m1(
    packed: Any,
    scale: Any,
    *,
    block_size: int = 32,
    output_dtype: np.dtype[Any] | type = np.float32,
) -> np.ndarray:
    """Decode one official packed expert matrix and apply per-row scales."""

    values = unpack_fp4_e2m1(packed)
    scales = _float_array(scale, "scale")
    if values.ndim != 2 or scales.ndim != 2:
        raise ValueError("MXFP4 weight and scale must both be 2D")
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size <= 0
    ):
        raise ValueError("block_size must be a positive integer")
    expected = (values.shape[0], (values.shape[1] + block_size - 1) // block_size)
    if scales.shape != expected:
        raise ValueError(f"MXFP4 scale shape {scales.shape} does not match {expected}")
    full_scale = np.repeat(scales, block_size, axis=1)[:, : values.shape[1]]
    return np.ascontiguousarray(values * full_scale, dtype=output_dtype)


def _power_of_two_scale(
    amax: np.ndarray,
    max_value: float,
    *,
    minimum_amax: float = 1e-4,
) -> np.ndarray:
    clipped = np.maximum(amax.astype(np.float32, copy=False), np.float32(minimum_amax))
    return np.exp2(np.ceil(np.log2(clipped / np.float32(max_value)))).astype(np.float32)


def quantize_fp8_e4m3_parts(
    x: Any, *, block_size: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """Return official E4M3 activation values and their UE8M0 scales.

    The first result contains decoded, *unscaled* E4M3 values with the same
    shape as ``x``.  The second contains one power-of-two scale per row and
    K block, with shape ``x.shape[:-1] + (K // block_size,)``.  Keeping these
    parts separate is required by DeepSeek's FP8/FP4 GEMMs: every raw-code
    tile is reduced in FP32 before its activation and weight scales are
    applied.
    """

    values = _float_array(x, "x")
    if values.ndim == 0:
        raise ValueError("activation tensor needs at least one dimension")
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size <= 0
    ):
        raise ValueError("block_size must be a positive integer")
    if values.shape[-1] % block_size:
        raise ValueError("activation K dimension must be divisible by block_size")
    grouped = values.reshape(*values.shape[:-1], -1, block_size)
    scale = _power_of_two_scale(np.max(np.abs(grouped), axis=-1), 448.0)
    normalized = np.clip(grouped / scale[..., None], -448.0, 448.0)
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - neural extra owns torch
        raise RuntimeError(
            "DeepSeek-V4 activation quantization requires torch"
        ) from exc
    rounded = (
        torch.from_numpy(np.ascontiguousarray(normalized))
        .to(torch.float8_e4m3fn)
        .to(torch.float32)
        .numpy()
    )
    return (
        np.ascontiguousarray(rounded.reshape(values.shape), dtype=np.float32),
        np.ascontiguousarray(scale, dtype=np.float32),
    )


def quantize_dequantize_fp8(x: Any, *, block_size: int = 128) -> np.ndarray:
    """Simulate the official UE8M0-scaled E4M3 activation quantizer.

    Conversion to E4M3 delegates only the final IEEE/OCP rounding step to
    PyTorch, which exposes the dtype on CPU even when TileLang/CUDA is absent.
    The returned tensor is float32 and has the original shape.
    """

    rounded, scale = quantize_fp8_e4m3_parts(x, block_size=block_size)
    grouped = rounded.reshape(*rounded.shape[:-1], -1, block_size)
    return np.ascontiguousarray(
        (grouped * scale[..., None]).reshape(rounded.shape), dtype=np.float32
    )


def quantize_dequantize_fp4(x: Any, *, block_size: int = 32) -> np.ndarray:
    """Simulate the official UE8M0-scaled E2M1 activation quantizer.

    PyTorch exposes the packed FP4 dtype but does not implement CPU casts to
    it.  Resolve the eight finite magnitudes explicitly and use round-to-nearest,
    ties-to-even on the E2M1 code, matching the cast used by DeepSeek's TileLang
    kernel.  The result is dequantized float32 with the original shape.
    """

    values = _float_array(x, "x")
    if values.ndim == 0:
        raise ValueError("activation tensor needs at least one dimension")
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size <= 0
    ):
        raise ValueError("block_size must be a positive integer")
    if values.shape[-1] % block_size:
        raise ValueError("activation K dimension must be divisible by block_size")
    grouped = values.reshape(*values.shape[:-1], -1, block_size)
    # The published FP4 kernel accepts the complete normal BF16 range.  Its
    # guard is ``fp4_max * 2**-126``; reusing the FP8 kernel's 1e-4 guard here
    # silently quantizes every sufficiently small but representable activation
    # to zero and changes downstream routing/logits.
    scale = _power_of_two_scale(
        np.max(np.abs(grouped), axis=-1, keepdims=True),
        6.0,
        minimum_amax=6.0 * 2.0**-126,
    )
    normalized = np.clip(grouped / scale, -6.0, 6.0)

    magnitudes = np.abs(normalized)[..., None]
    levels = FP4_E2M1_TABLE[:8]
    distances = np.abs(magnitudes - levels)
    minimum = distances.min(axis=-1, keepdims=True)
    tied = distances == minimum
    # Even code wins an exact midpoint tie.  Adding one makes odd codes lose
    # while leaving the unique-nearest case unchanged.
    codes = np.where(tied, np.arange(8, dtype=np.int16) & 1, 2).argmin(axis=-1)
    quantized = levels[codes]
    quantized = np.copysign(quantized, normalized)
    return np.ascontiguousarray(
        (quantized * scale).reshape(values.shape), dtype=np.float32
    )


__all__ = [
    "FP4_E2M1_TABLE",
    "dequantize_fp4_e2m1",
    "dequantize_fp8_e4m3",
    "quantize_dequantize_fp4",
    "quantize_dequantize_fp8",
    "quantize_fp8_e4m3_parts",
    "unpack_fp4_e2m1",
]
