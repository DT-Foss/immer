"""
Contraction-based generation and Z2 topological lift for HSSLM-C.

From "Collapse is Contraction" and "Unitarity is the Boundary":
- Contraction coefficient τ as order parameter
- Zeno/anti-Zeno scheduling
- Z2 topological lift with quantum potential
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class ContractionSampler:
    """
    Token sampling controlled by contraction coefficient τ.
    T(τ) = (1-τ²)^(-1/2) - 1 maps τ to softmax temperature.
    """

    TAU_DEFAULT = 0.65
    ZENO_OPTIMAL_K = 5

    @staticmethod
    def temperature(tau: float) -> float:
        if tau >= 1.0:
            return 10.0
        return (1.0 - tau ** 2) ** (-0.5) - 1.0

    def __init__(self, tau_default: float = TAU_DEFAULT, zeno_k: int = ZENO_OPTIMAL_K):
        self.tau_default = tau_default
        self.zeno_k = zeno_k

    def sample(self, logits: torch.Tensor, tau: Optional[float] = None,
               top_k: Optional[int] = 50) -> torch.Tensor:
        if tau is None:
            tau = self.tau_default
        temp = self.temperature(tau)
        if temp > 0.01:
            logits = logits / temp
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(logits < v[..., [-1]], float('-inf'), logits)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def zeno_schedule(self, step: int) -> float:
        if step % self.zeno_k == 0:
            return self.tau_default
        return self.tau_default * 0.7


class Z2TopologicalLift(nn.Module):
    """
    Z2 topological lift: doubles latent space into physical + momentum.
    Quantum potential Q_i = m_i / x_i guides generation.
    Foss Topological Index target: F = 0.75 ± 0.05.
    """

    def __init__(self, d_model: int = 256, reduced_dim: int = 64):
        super().__init__()
        # Use reduced dimension for Z2 projections (not full d_model//2)
        self.reduced_dim = reduced_dim
        self.physical_proj = nn.Linear(d_model, reduced_dim)
        self.momentum_proj = nn.Linear(d_model, reduced_dim)
        self.potential_proj = nn.Sequential(
            nn.Linear(reduced_dim * 2, reduced_dim), nn.SiLU(), nn.Linear(reduced_dim, reduced_dim))
        self.recombine = nn.Linear(d_model, d_model)  # Skip connection: identity-like

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.tanh(self.physical_proj(h))
        m = torch.tanh(self.momentum_proj(h))
        eps = 1e-6
        Q = torch.tanh(m / (x.abs() + eps))
        return x, m, Q

    def recombine_with_potential(self, h: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        # Expand Q from reduced_dim to d_model by tiling
        d_model = h.shape[-1]
        repeat_factor = d_model // Q.shape[-1]
        remainder = d_model % Q.shape[-1]
        Q_expanded = Q.repeat_interleave(repeat_factor, dim=-1)
        if remainder > 0:
            padding = torch.zeros(*Q.shape[:-1], remainder, device=Q.device, dtype=Q.dtype)
            Q_expanded = torch.cat([Q_expanded, padding], dim=-1)
        guided = h * (1 + 0.1 * Q_expanded)
        return self.recombine(guided)

    def compute_topological_index(self, physical: torch.Tensor,
                                   momentum: torch.Tensor) -> float:
        B, L, rd = physical.shape
        n = B * L
        flat_p = physical.reshape(-1, rd)
        flat_m = momentum.reshape(-1, rd)
        cov = (flat_p.t() @ flat_m) / n
        s = torch.linalg.svdvals(cov)
        s_norm = s / (s.sum() + 1e-8)
        S_ent = -(s_norm * torch.log(s_norm + 1e-8)).sum().item()
        return S_ent / math.log(max(n, 2))


class BvNPathIntegralSampler:
    """
    Birkhoff-von Neumann path integral token selection.
    Decomposes token probabilities into computational paths.
    """

    def __init__(self, vocab_size: int = 16384, max_paths: int = 100):
        self.vocab_size = vocab_size
        self.max_paths = max_paths

    def bvn_decompose(self, prob_matrix: torch.Tensor) -> Tuple[List, List[float]]:
        n = min(prob_matrix.shape[0], self.max_paths)
        paths, weights = [], []
        residual = prob_matrix.clone()
        for _ in range(min(n, self.max_paths)):
            perm = self._greedy_permutation(residual)
            w = residual[perm, torch.arange(len(perm))].min().item()
            if w < 1e-6:
                break
            for i, j in enumerate(perm):
                residual[i, j] -= w
            paths.append(perm)
            weights.append(max(w, 0))
        return paths, weights

    def _greedy_permutation(self, matrix: torch.Tensor) -> torch.Tensor:
        n = matrix.shape[0]
        perm = torch.zeros(n, dtype=torch.long)
        used = set()
        for i in range(n):
            best_j, best_val = -1, -1
            for j in range(n):
                if j not in used and matrix[i, j] > best_val:
                    best_val, best_j = matrix[i, j], j
            perm[i] = best_j
            used.add(best_j)
        return perm
