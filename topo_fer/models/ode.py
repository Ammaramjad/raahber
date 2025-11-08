"""Neural ODE components for the Geometric Scaffold Network."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import torch
import torch.nn as nn
from torchdiffeq import odeint

from ..config import ModelConfig


class DynamicsFunction(nn.Module):
    """Neural network parameterizing latent dynamics for expression transitions."""

    def __init__(self, latent_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        layers = []
        input_dim = latent_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ELU(inplace=True))
            layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        del t  # time is unused but required for interface
        return self.net(z)


class NeuralODEBlock(nn.Module):
    """Integrate latent dynamics over a time span."""

    def __init__(
        self,
        dynamics: DynamicsFunction,
        time_span: Tuple[float, float],
        solver: str = "dopri5",
    ) -> None:
        super().__init__()
        self.dynamics = dynamics
        self.time_span = torch.tensor(time_span, dtype=torch.float32)
        self.solver = solver

    def forward(self, z0: torch.Tensor) -> torch.Tensor:
        time_span = self.time_span.to(z0.device)
        z_traj = odeint(self.dynamics, z0, time_span, method=self.solver)
        return z_traj[-1]


def build_neural_ode(cfg: ModelConfig) -> NeuralODEBlock:
    """Factory function for NeuralODEBlock based on model config."""
    dynamics = DynamicsFunction(
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.ode_hidden_dim,
        num_layers=cfg.ode_num_layers,
    )
    return NeuralODEBlock(
        dynamics=dynamics,
        time_span=(cfg.ode_time_span[0], cfg.ode_time_span[-1]),
    )
