"""Loss functions for TOPO-FER."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy over known classes, ignoring unlabeled samples."""
    mask = labels >= 0
    if mask.sum() == 0:
        return torch.zeros(1, device=logits.device, dtype=logits.dtype)
    return F.cross_entropy(logits[mask], labels[mask])


def manifold_smoothness_loss(latent: torch.Tensor, latent_ode: torch.Tensor) -> torch.Tensor:
    """Penalize divergence between latent code and ODE-integrated latent."""
    return F.mse_loss(latent_ode, latent)


def reconstruction_loss(
    reconstructed_features: torch.Tensor, target_features: torch.Tensor
) -> torch.Tensor:
    """Encourage manifold embeddings to preserve reconstructive information."""
    return F.mse_loss(reconstructed_features, target_features)


def supervised_contrastive_loss(
    manifold_embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Compute supervised contrastive loss over known classes."""
    mask = labels >= 0
    if mask.sum() <= 1:
        return torch.zeros(1, device=manifold_embeddings.device, dtype=manifold_embeddings.dtype)

    features = F.normalize(manifold_embeddings[mask], dim=-1)
    labels = labels[mask]

    similarity = torch.matmul(features, features.t()) / temperature
    logits_mask = torch.ones_like(similarity, dtype=torch.bool)
    logits_mask.fill_diagonal_(False)

    positives = labels.unsqueeze(0) == labels.unsqueeze(1)
    positives = positives & logits_mask

    log_prob = similarity - torch.logsumexp(similarity * logits_mask.float(), dim=1, keepdim=True)

    mean_log_prob_pos = (log_prob * positives.float()).sum(1) / positives.float().sum(1).clamp_min(1.0)
    loss = -mean_log_prob_pos.mean()
    return loss


def aggregate_losses(components: Dict[str, torch.Tensor], weights: Dict[str, float]) -> torch.Tensor:
    """Combine loss components using provided weights."""
    total = torch.zeros(1, device=next(iter(components.values())).device)
    for name, value in components.items():
        weight = weights.get(name, 1.0)
        total = total + weight * value
    return total
