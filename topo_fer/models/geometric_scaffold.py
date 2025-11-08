from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

from .backbones import FeatureBackbone


@dataclass
class ODEConfig:
    hidden_dim: int = 128
    num_layers: int = 2
    atol: float = 1e-5
    rtol: float = 1e-4
    solver: str = "dopri5"
    time_span: Tuple[float, float] = (0.0, 1.0)
    steps: int = 10


class NeuralODEFunction(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        layers = []
        in_dim = latent_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class GeometricScaffoldNetwork(nn.Module):
    """Neural ODE-based manifold learner for facial expression dynamics."""

    def __init__(
        self,
        backbone: str,
        embedding_dim: int,
        latent_dim: int,
        ode_config: Dict[str, float],
        num_known_classes: int,
        classifier_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = FeatureBackbone(backbone)
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim

        self.feature_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.backbone.out_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        self.latent_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, latent_dim),
        )

        raw_ode_cfg = dict(ode_config)
        ode_cfg = {
            **raw_ode_cfg,
            "time_span": tuple(raw_ode_cfg.get("time_span", (0.0, 1.0))),
        }
        cfg = ODEConfig(**ode_cfg)
        self.ode_config = cfg
        self.ode_function = NeuralODEFunction(latent_dim, cfg.hidden_dim, cfg.num_layers)

        self.latent_decoder = nn.Sequential(
            nn.Linear(latent_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=classifier_dropout),
            nn.Linear(embedding_dim, num_known_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        time_span: Optional[Tuple[float, float]] = None,
        steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        embedding = self.feature_projection(features)
        embedding = F.normalize(embedding, dim=-1)

        z0 = self.latent_encoder(embedding)

        cfg = self.ode_config
        t0, t1 = time_span if time_span is not None else cfg.time_span
        n_steps = steps or cfg.steps
        time_points = torch.linspace(t0, t1, n_steps, device=x.device, dtype=x.dtype)

        ode_solution = odeint(
            self.ode_function,
            z0,
            time_points,
            method=cfg.solver,
            atol=cfg.atol,
            rtol=cfg.rtol,
        )
        latent_path = ode_solution.permute(1, 0, 2)
        zT = latent_path[:, -1]
        reconstructed = self.latent_decoder(zT)
        logits = self.classifier(reconstructed)

        return {
            "embedding": embedding,
            "latent_path": latent_path,
            "latent_initial": z0,
            "latent_final": zT,
            "reconstruction": reconstructed,
            "logits": logits,
            "time_points": time_points,
        }

    def manifold_energy(self, latent_path: torch.Tensor) -> torch.Tensor:
        """Compute energy of trajectories to regularize manifold smoothness."""
        velocity = latent_path[:, 1:] - latent_path[:, :-1]
        energy = (velocity.norm(dim=-1) ** 2).mean(dim=-1)
        return energy

