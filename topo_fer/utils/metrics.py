"""Metric utilities for evaluation."""

from __future__ import annotations

from typing import Dict

import torch


def accuracy(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
     """Compute top-1 accuracy."""
     if target.numel() == 0:
         return torch.tensor(0.0, device=output.device)
     preds = output.argmax(dim=1)
     correct = (preds == target).float().sum()
     return correct / target.numel()


def novel_class_discovery_score(
     discovered_labels: torch.Tensor, ground_truth_labels: torch.Tensor
) -> Dict[str, float]:
     """Compute clustering metrics for novel class discovery."""
     from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

     cpu_discovered = discovered_labels.detach().cpu().numpy()
     cpu_ground_truth = ground_truth_labels.detach().cpu().numpy()
     return {
         "ari": float(adjusted_rand_score(cpu_ground_truth, cpu_discovered)),
         "nmi": float(
             normalized_mutual_info_score(cpu_ground_truth, cpu_discovered, average_method="max")
         ),
     }
