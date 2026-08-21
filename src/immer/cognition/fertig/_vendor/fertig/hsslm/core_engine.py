"""
Core SSM engine: Selective State Space Model (Mamba-style) in pure PyTorch.

This module implements the complete Mamba block with selective SSM scan,
without any custom CUDA kernels. It runs in recurrent mode which is
slightly slower for training but much simpler and fully equivalent.

Components:
    - RMSNorm: Root mean square layer normalization
    - SelectiveSSM: Input-dependent state space model (S6)
    - MambaBlock: Complete block (proj -> conv -> SSM -> gate -> out)
    - StateSpaceCore: Stack of MambaBlocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Reference: Zhang & Sennrich (2019)
    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class SelectiveSSM(nn.Module):
    """Selective State Space Model (S6 / Mamba-style).

    Implements input-dependent state space transitions. Unlike the
    time-invariant S4, this model makes B, C, and Delta functions of
    the input, enabling content-dependent reasoning.

    State equations (continuous time):
        h'(t) = A * h(t) + B * x(t)
        y(t)  = C * h(t) + D * x(t)

    Discretized (zero-order hold, per Mamba paper):
        A_bar = exp(Delta * A)
        B_bar = Delta * B
        h[t]  = A_bar * h[t-1] + B_bar * x[t]
        y[t]  = C * h[t] + D * x[t]

    Selectivity: B, C, Delta are projected from the input x.
    """

    def __init__(
        self,
        d_inner: int = 512,
        d_state: int = 16,
        dt_rank: int = 8,
        bias: bool = False,
    ) -> None:
        """
        Args:
            d_inner: Expanded dimension (d_model * expand).
            d_state: SSM state dimension per channel (N in paper).
            dt_rank: Rank for Delta (step size) projection (R in paper).
            bias: Whether to use bias in linear layers.
        """
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank

        # Project input to Delta, B, C
        # Output: dt_rank + 2 * d_state values per channel
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=bias)
        # 512 * (8 + 32) = 20,480 params

        # Project rank-ranked Delta to full d_inner dimension
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        # 8 * 512 + 512 = 4,608 params

        # A_log: learned parameter initialized as HiPPO
        # Shape: (d_inner, d_state)
        # Initialized as log(1), log(2), ..., log(N) repeated d_inner times
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        # 512 * 16 = 8,192 params

        # D: skip connection parameter (learned scalar per channel)
        self.D = nn.Parameter(torch.ones(d_inner))
        # 512 params

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for stable SSM training."""
        # x_proj: small random values
        nn.init.xavier_uniform_(self.x_proj.weight)
        # dt_proj: small values for stability
        nn.init.xavier_uniform_(self.dt_proj.weight)
        # Delta bias: initialize to small positive values
        if self.dt_proj.bias is not None:
            nn.init.uniform_(self.dt_proj.bias, 0.01, 0.1)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Selective SSM forward pass (recurrent mode).

        Args:
            x: (B, L, d_inner) input sequence.
            state: Optional (B, d_inner, d_state) previous state.

        Returns:
            - y: (B, L, d_inner) output sequence.
            - state: (B, d_inner, d_state) final state.
        """
        B, L, _ = x.shape

        # 1. Project input to Delta, B, C
        x_proj_out = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        delta, B_ssm, C_ssm = torch.split(
            x_proj_out,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        # delta: (B, L, dt_rank)
        # B_ssm: (B, L, d_state)
        # C_ssm: (B, L, d_state)

        # 2. Project Delta to full dimension and apply softplus
        delta = F.softplus(self.dt_proj(delta))  # (B, L, d_inner)
        # Clamp to avoid numerical issues
        delta = torch.clamp(delta, min=1e-4, max=1.0)

        # 3. Get A (negative exponential ensures stability)
        # A: (d_inner, d_state)
        A = -torch.exp(self.A_log.float())  # negative for stability

        # 4. Discretize A and B (input-dependent via Delta)
        # A_discrete: (B, L, d_inner, d_state)
        A_discrete = torch.exp(
            A.unsqueeze(0).unsqueeze(0) * delta.unsqueeze(-1)
        )
        # Clamp to prevent explosion
        A_discrete = torch.clamp(A_discrete, min=0.0, max=1.0)

        # B_discrete: (B, L, 1, d_state)
        B_discrete = B_ssm.unsqueeze(2) * delta.unsqueeze(-1)

        # 5. Recurrent scan
        h = (
            torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
            if state is None else state
        )

        ys = []
        for t in range(L):
            # State update: h = A_bar * h + B_bar * x
            h = A_discrete[:, t] * h + B_discrete[:, t] * x[:, t].unsqueeze(-1)

            # Output: y = C * h + D * x
            # C[:, t]: (B, d_state), h: (B, d_inner, d_state)
            y = torch.sum(h * C_ssm[:, t].unsqueeze(1), dim=-1)  # (B, d_inner)
            y = y + self.D * x[:, t]  # skip connection
            ys.append(y)

        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        return y, h


class MambaBlock(nn.Module):
    """Complete Mamba block.

    Architecture (from Mamba paper, Figure 3):
        x --> [RMSNorm] --> [in_proj] --> split to x_branch, z_branch
                                                  |
        |-----------------------------------------|
        |                                         v
        |                              [conv1d] -> SiLU
        |                                         |
        |                                         v
        |                              [SelectiveSSM]
        |                                         |
        |                                         v
        |                              [gating: * SiLU(z)]
        |                                         |
        |                                         v
        |                              [out_proj] --> + --> y
        |                                             ^
        +---------------------------------------------+
                     (residual connection)

    The in_proj produces 2 * d_inner values:
        - x_branch: goes through conv + SSM
        - z_branch: used as gating signal (SiLU activation)
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int = 8,
        bias: bool = False,
        dropout: float = 0.0,
    ) -> None:
        """
        Args:
            d_model: Model dimension (D in paper).
            d_state: SSM state dimension (N in paper).
            d_conv: Local convolution kernel size.
            expand: Expansion factor for inner dimension (E in paper).
            dt_rank: Delta projection rank (R in paper).
            bias: Use bias in linear layers.
            dropout: Dropout rate.
        """
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_conv = d_conv

        # RMSNorm before the block
        self.norm = RMSNorm(d_model)

        # Input projection: d_model -> 2 * d_inner (for x_branch + z_branch)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        # 256 * 1024 = 262,144 params

        # Short convolution (depthwise separable) on x_branch
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,  # depthwise: each channel convolved separately
            bias=True,
        )
        # 512 * 4 + 512 = 2,560 params

        # Selective SSM
        self.ssm = SelectiveSSM(
            d_inner=self.d_inner,
            d_state=d_state,
            dt_rank=dt_rank,
            bias=bias,
        )
        # 33,792 params (see SelectiveSSM)

        # Output projection: d_inner -> d_model
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        # 512 * 256 = 131,072 params

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L, d_model) input.
            state: Optional previous SSM state.

        Returns:
            - output: (B, L, d_model).
            - state: Final SSM state.
        """
        B, L, D = x.shape
        assert D == self.d_model, f"Expected d_model={self.d_model}, got {D}"

        # Residual
        residual = x

        # RMSNorm
        x = self.norm(x)

        # Input projection to two branches
        xz = self.in_proj(x)  # (B, L, 2 * d_inner)
        x_branch, z_branch = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Short convolution on x_branch
        # Conv1d expects (B, C, L) so transpose
        x_conv = self.conv1d(x_branch.transpose(1, 2))  # (B, d_inner, L + pad)
        x_conv = x_conv[:, :, :L]  # crop padding: (B, d_inner, L)
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # Selective SSM
        x_ssm, state = self.ssm(x_conv, state)  # (B, L, d_inner)

        # Gating: multiply with activated z_branch
        x_gated = x_ssm * F.silu(z_branch)  # (B, L, d_inner)

        # Output projection
        output = self.out_proj(x_gated)  # (B, L, d_model)
        output = self.dropout(output)

        # Residual connection
        return output + residual, state


class StateSpaceCore(nn.Module):
    """Stack of MambaBlocks forming the core sequence processing engine.

    Processes token embeddings through multiple SSM layers to build
    contextualized representations. Each layer refines the representation
    by applying content-dependent state transitions.
    """

    def __init__(
        self,
        n_layers: int = 6,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int = 8,
        dropout: float = 0.0,
    ) -> None:
        """
        Args:
            n_layers: Number of Mamba blocks.
            d_model: Model dimension.
            d_state: SSM state dimension.
            d_conv: Convolution kernel size.
            expand: Expansion factor.
            dt_rank: Delta projection rank.
            dropout: Dropout rate.
        """
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        # Stack of Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dt_rank=dt_rank,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Final normalization
        self.final_norm = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (B, L, d_model) embedded input.
            states: Optional list of per-layer SSM states for recurrent inference.

        Returns:
            - output: (B, L, d_model) processed sequence.
            - states: List of final SSM states per layer.
        """
        if states is None:
            states = [None] * self.n_layers

        new_states = []
        for i, layer in enumerate(self.layers):
            x, new_state = layer(x, states[i])
            new_states.append(new_state)

        x = self.final_norm(x)
        return x, new_states

    def init_states(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        """Initialize empty states for recurrent generation.

        Args:
            batch_size: Batch size.
            device: Target device.

        Returns:
            List of (B, d_inner, d_state) zero tensors, one per layer.
        """
        return [
            torch.zeros(batch_size, self.d_inner, self.d_state, device=device)
            for _ in range(self.n_layers)
        ]
