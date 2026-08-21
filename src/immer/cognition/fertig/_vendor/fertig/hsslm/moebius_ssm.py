"""
Moebius Contractive SSM Core for HSSLM-C.

Based on David Tom Foss's mathematical frameworks:
- Moebius coupling: f(λ,v) = (λ+v)/(1+λv) (Lorentz velocity addition)
- Period function: g(λ) = (1-λ²)^(-1/2) (Lorentz factor)
- Contraction coefficient τ < 1 (Birkhoff contraction)
- PS-Lifted Z2 doubling with Fiedler orientation

Replaces SelectiveSSM: fewer parameters, O(1) convergence, contraction guarantees.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class MoebiusCoupling:
    """Static Moebius coupling functions from the Moebius-Lorentz correspondence."""
    EPS = 1e-6

    @staticmethod
    def forward(lam: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """f(λ,v) = (λ+v)/(1+λv) with numerical stability."""
        num = lam + v
        den = 1 + lam * v
        return num / (den + MoebiusCoupling.EPS)

    @staticmethod
    def period(lam: torch.Tensor) -> torch.Tensor:
        """g(λ) = (1-λ²)^(-1/2) -- Lorentz factor / intrinsic timescale."""
        return (1 - lam.pow(2)).clamp(min=MoebiusCoupling.EPS).pow(-0.5)

    @staticmethod
    def lorentz_factor(lam: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """γ(λ,v) = (1+λv)/sqrt(1-v²) -- time dilation factor."""
        return (1 + lam * v) / (1 - v.pow(2)).clamp(min=MoebiusCoupling.EPS).sqrt()


class ContractiveSSM(nn.Module):
    """
    Single Contractive SSM layer with Moebius coupling.

    REDUCED dimensions (contraction efficiency = fewer params needed):
    - d_inner: 256 (was 512 in SelectiveSSM -- 50% reduction)
    - d_state: 16
    - dt_rank: 8
    - tau_max: 0.95 (contraction coefficient upper bound)
    """

    def __init__(self, d_inner: int = 256, d_state: int = 16, dt_rank: int = 8,
                 tau_max: float = 0.95):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.tau_max = tau_max

        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # State transition A: log-parameterized for stability
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        # Moebius coupling projection
        self.v_proj = nn.Linear(d_inner, d_state, bias=False)

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, d_inner = x.shape

        if state is None:
            state = torch.zeros(B, self.d_state, d_inner, device=x.device, dtype=x.dtype)

        x_reshaped = x.reshape(B * L, d_inner)
        x_proj_out = self.x_proj(x_reshaped)
        delta = x_proj_out[:, :self.dt_rank]
        B_param = x_proj_out[:, self.dt_rank:self.dt_rank + self.d_state]
        C_param = x_proj_out[:, self.dt_rank + self.d_state:]

        dt = self.dt_proj(delta)
        dt = F.softplus(dt).reshape(B, L, d_inner)
        A = -torch.exp(self.A_log)
        v = torch.tanh(self.v_proj(x_reshaped)).reshape(B, L, self.d_state)
        v = v.unsqueeze(-1).expand(-1, -1, -1, d_inner)

        outputs = []
        for i in range(L):
            xi = x[:, i, :]
            dti = dt[:, i, :]
            vi = v[:, i, :, :]

            # Discretize lambda = exp(A * dt)
            dt_exp = dti.unsqueeze(-1)
            A_exp = A.unsqueeze(0)
            lam = torch.exp(A_exp * dt_exp)
            lam = lam.clamp(min=-self.tau_max, max=self.tau_max)

            # Moebius coupling. The Moebius/Lorentz addition f(lam,v) can drive
            # |lam_new| toward (or past) 1 even when lam itself is clamped — and
            # period(lam)=(1-lam^2)^(-1/2) then blows up, so the recurrent product
            # gate*lam_new diverges (state exploded to ~1e11). Clamp lam_new back
            # into the contractive region BEFORE period(): this is exactly the
            # tau_max contraction the layer promises, just enforced where the
            # recurrence actually multiplies — Birkhoff contraction, not blow-up.
            lam_t = lam.transpose(-2, -1)
            lam_new = MoebiusCoupling.forward(lam_t, vi)
            lam_new = lam_new.clamp(min=-self.tau_max, max=self.tau_max)
            # lam_new (|.|<=tau_max<1) IS the contraction factor of the state
            # recurrence — that is the Birkhoff/Lorentz-velocity guarantee. The
            # period g(lam)=(1-lam^2)^(-1/2) is the intrinsic TIMESCALE (>=1), not
            # a state multiplier: using it as one breaks contraction (state grew
            # to ~1e4). Keep the recurrence strictly contractive; let period scale
            # the input drive instead, so a slow mode admits more input per step.
            gate = MoebiusCoupling.period(lam_new)
            new_state = lam_new * state
            new_state = new_state + 0.01 * gate * xi.unsqueeze(1) * (1 - lam_new.abs())
            state = new_state

            # Output
            Ci = C_param[i * B:(i + 1) * B, :]
            out = (Ci.unsqueeze(-1) * state).sum(dim=1)
            out = out + self.D * xi
            outputs.append(out)

        return torch.stack(outputs, dim=1), state


class PSLiftedBlock(nn.Module):
    """
    PS-Lifted block with Z2 state-space doubling.
    Forward probability pc=0.65, self-loop ps=0.003, reverse pr=0.347.
    Physical projection: h = h_+ + h_-
    Momentum projection: m = h_+ - h_-
    """

    def __init__(self, d_model: int = 256, d_state: int = 16, dt_rank: int = 8,
                 pc: float = 0.65, ps: float = 0.003, tau_max: float = 0.95):
        super().__init__()
        self.d_model = d_model
        self.pc = pc
        self.ps = ps
        self.pr = 1.0 - pc - ps
        half_d = d_model // 2

        self.ssm_plus = ContractiveSSM(d_inner=half_d, d_state=d_state,
                                       dt_rank=dt_rank, tau_max=tau_max)
        self.ssm_minus = ContractiveSSM(d_inner=half_d, d_state=d_state,
                                        dt_rank=dt_rank, tau_max=tau_max)
        self.cross_forward = nn.Linear(half_d, half_d, bias=False)
        self.cross_backward = nn.Linear(half_d, half_d, bias=False)
        self.in_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.self_loop_gate = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, state: Optional[Tuple] = None
                ) -> Tuple[torch.Tensor, Tuple]:
        B, L, d_model = x.shape
        half_d = d_model // 2

        x_norm = self.norm(x)
        z = self.in_proj(x_norm)
        z_plus = z[:, :, :half_d]
        z_minus = z[:, :, half_d:]

        if state is None:
            state_plus = state_minus = None
        else:
            state_plus, state_minus = state

        gate = torch.sigmoid(self.self_loop_gate(x_norm))
        h_plus, new_state_plus = self.ssm_plus(z_plus, state_plus)
        h_minus, new_state_minus = self.ssm_minus(z_minus, state_minus)

        h_plus = h_plus + self.pc * self.cross_forward(h_minus)
        h_minus = h_minus + self.pr * self.cross_backward(h_plus)

        h_plus = gate[:, :, :half_d] * z_plus + (1 - gate[:, :, :half_d]) * h_plus
        h_minus = gate[:, :, half_d:] * z_minus + (1 - gate[:, :, half_d:]) * h_minus

        h = torch.cat([h_plus, h_minus], dim=-1)
        h = self.out_proj(h)
        return h + x, (new_state_plus, new_state_minus)


class MoebiusStateSpaceCore(nn.Module):
    """
    Stack of 4 PSLiftedBlocks (was 6 SelectiveSSM layers).
    Contraction efficiency allows fewer layers.
    """

    def __init__(self, n_layers: int = 4, d_model: int = 256, d_state: int = 16,
                 dt_rank: int = 8, tau_max: float = 0.95, pc: float = 0.65,
                 ps: float = 0.003):
        super().__init__()
        self.layers = nn.ModuleList([
            PSLiftedBlock(d_model=d_model, d_state=d_state, dt_rank=dt_rank,
                          pc=pc, ps=ps, tau_max=tau_max)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, states: Optional[list] = None
                ) -> Tuple[torch.Tensor, list]:
        if states is None:
            states = [None] * len(self.layers)
        new_states = []
        for i, layer in enumerate(self.layers):
            x, state = layer(x, states[i])
            new_states.append(state)
        return x, new_states

    def init_states(self, batch_size: int, device) -> list:
        """Null-Zustaende fuer alle PSLiftedBlock-Schichten (Z2-Doubling).

        Jede Schicht fuehrt zwei ContractiveSSM-Zustaende (plus/minus) mit
        je (B, d_state, d_inner/2). Ohne diese Initialisierung ist die
        rekursive Generierung kontextfrei (nur letztes Token) — der Bug,
        der alle Degenerations-Muster erzeugte.
        """
        states = []
        for layer in self.layers:
            half_d = layer.d_model // 2
            d_state = layer.ssm_plus.d_state
            zero = torch.zeros(batch_size, d_state, half_d, device=device)
            states.append((zero.clone(), zero.clone()))
        return states
