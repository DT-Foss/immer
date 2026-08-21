"""Ginibre kernel weight initialization for HSSLM-C.

From "One Constant Rules All 2D Spectra":
- Universal constant: <s^2> = 1.08747... (2D NND second moment)
- Cubic repulsion: beta=3
- Sinkhorn renormalization for doubly stochastic projection
"""
import torch
import torch.nn as nn
from typing import Tuple


class GinibreInitializer:
    """Initialize weights using Ginibre 2D spectral statistics."""

    GINIBRE_KERNEL = 1.08746866652609
    CUBIC_REPULSION_BETA = 3.0

    @staticmethod
    def sinkhorn_renormalize(W: torch.Tensor, n_iter: int = 10, eps: float = 1e-6) -> torch.Tensor:
        W = W.abs() + eps
        for _ in range(n_iter):
            W = W / W.sum(dim=-1, keepdim=True).clamp(min=eps)
            W = W / W.sum(dim=-2, keepdim=True).clamp(min=eps)
        return W

    @staticmethod
    def initialize_2d_matrix(rows: int, cols: int, asymmetry: float = 0.5,
                              device='cpu', dtype=torch.float32) -> torch.Tensor:
        R = torch.randn(rows, cols, device=device, dtype=dtype)
        if asymmetry < 1.0 and rows == cols:
            S = (R + R.t()) / 2
            M = (1 - asymmetry) * S + asymmetry * R
        else:
            M = R
        if rows == cols:
            M = GinibreInitializer.sinkhorn_renormalize(M.abs())
            M = M * (1.0 / max(M.abs().max().item(), 0.1))
        else:
            fan_in, fan_out = cols, rows
            scale = (6.0 / (fan_in + fan_out)) ** 0.5
            M = M * scale * (1 + 0.1 * asymmetry)
        return M

    @staticmethod
    def compute_nnd_moment(W: torch.Tensor) -> float:
        if W.shape[0] != W.shape[1] or min(W.shape) < 3:
            return 0.0
        eigs = torch.linalg.eigvals(W)
        points = torch.stack([eigs.real, eigs.imag], dim=-1)
        n = points.shape[0]
        if n < 3:
            return 0.0
        dists = torch.cdist(points, points)
        dists.fill_diagonal_(float('inf'))
        nnd = dists.min(dim=1).values
        mean_nnd = nnd.mean()
        if mean_nnd < 1e-6:
            return 0.0
        s = nnd / mean_nnd
        return (s ** 2).mean().item()


def ginibre_init_(tensor: nn.Parameter, asymmetry: float = 0.5):
    """In-place Ginibre initialization for any weight tensor."""
    shape = tensor.shape
    if len(shape) == 2:
        W = GinibreInitializer.initialize_2d_matrix(
            shape[0], shape[1], asymmetry, device=tensor.device, dtype=tensor.dtype)
        with torch.no_grad():
            tensor.copy_(W)
    elif len(shape) == 1:
        nn.init.normal_(tensor, mean=0, std=0.02 * (1 + asymmetry))
    elif len(shape) >= 3:
        for idx in torch.cartesian_prod(*[torch.arange(s) for s in shape[:-2]]):
            idx_tuple = tuple(idx.tolist())
            W = GinibreInitializer.initialize_2d_matrix(
                shape[-2], shape[-1], asymmetry, device=tensor.device, dtype=tensor.dtype)
            with torch.no_grad():
                tensor[idx_tuple].copy_(W)
    else:
        nn.init.normal_(tensor, std=0.02)
