"""Geometric Scaffold Network definition."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ..config import ModelConfig
from .ode import build_neural_ode


def _build_backbone(cfg: ModelConfig) -> Tuple[nn.Module, int]:
    """Construct the convolutional backbone."""
    if cfg.backbone == "resnet18":
        try:
            weights = (
                models.ResNet18_Weights.DEFAULT if cfg.pretrained_backbone else None
            )
            backbone = models.resnet18(weights=weights)
        except AttributeError:
            backbone = models.resnet18(pretrained=cfg.pretrained_backbone)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif cfg.backbone == "resnet50":
        try:
            weights = (
                models.ResNet50_Weights.DEFAULT if cfg.pretrained_backbone else None
            )
            backbone = models.resnet50(weights=weights)
        except AttributeError:
            backbone = models.resnet50(pretrained=cfg.pretrained_backbone)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone: {cfg.backbone}")

    return backbone, feature_dim


class GeometricScaffoldNet(nn.Module):
    """Model that maps images to a manifold representation with Neural ODE dynamics."""

    def __init__(self, cfg: ModelConfig, num_known_classes: int) -> None:
        super().__init__()
        backbone, feature_dim = _build_backbone(cfg)
        self.backbone = backbone

        self.feature_dim = feature_dim
        self.latent_projection = nn.Sequential(
            nn.Linear(feature_dim, cfg.latent_dim),
            nn.LayerNorm(cfg.latent_dim),
            nn.SiLU(),
        )

        self.ode_block = build_neural_ode(cfg)

        self.manifold_projection = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.manifold_projection_dim),
            nn.LayerNorm(cfg.manifold_projection_dim),
            nn.Tanh(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(cfg.manifold_projection_dim, cfg.classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.classifier_hidden_dim, num_known_classes),
        )

        self.reconstruction_head = nn.Sequential(
            nn.Linear(cfg.manifold_projection_dim, cfg.latent_dim),
            nn.LayerNorm(cfg.latent_dim),
            nn.SiLU(),
            nn.Linear(cfg.latent_dim, feature_dim),
        )

    def forward(
        self,
        images: torch.Tensor,
        return_latent: bool = False,
    ) -> Dict[str, torch.Tensor]:
        features = self.backbone(images)
        latent = self.latent_projection(features)
        latent_ode = self.ode_block(latent)
        manifold = self.manifold_projection(latent_ode)
        logits = self.classifier(manifold)
        reconstruction = self.reconstruction_head(manifold)

        outputs = {
            "logits": logits,
            "manifold": manifold,
            "reconstruction": reconstruction,
        }
        if return_latent:
            outputs["features"] = features
            outputs["latent"] = latent
            outputs["latent_ode"] = latent_ode

        return outputs

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return manifold embeddings."""
        with torch.no_grad():
            features = self.backbone(images)
            latent = self.latent_projection(features)
            latent_ode = self.ode_block(latent)
            manifold = self.manifold_projection(latent_ode)
        return manifold
