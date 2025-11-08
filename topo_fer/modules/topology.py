"""Topological Discovery Module for TOPO-FER."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

import hdbscan
from gtda.diagrams import Filtering, Scaler
from gtda.diagrams.features import PersistenceEntropy
from gtda.homology import VietorisRipsPersistence

from ..config import TopologyConfig
from ..utils.logging import get_logger


LOGGER = get_logger(__name__)


class TopologicalDiscoveryModule(nn.Module):
    """Performs persistent homology and clustering to discover novel expressions."""

    def __init__(self, cfg: TopologyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.persistence = VietorisRipsPersistence(
            metric="euclidean",
            max_edge_length=cfg.rips_epsilon,
            homology_dimensions=tuple(range(cfg.max_dimension + 1)),
        )
        self.scaler = Scaler()
        self.filtering = Filtering(
            epsilon=cfg.persistence_threshold, homology_dimensions=[1, 2]
        )
        self.entropy = PersistenceEntropy()

    @torch.no_grad()
    def compute_diagram(self, embeddings: torch.Tensor) -> np.ndarray:
        """Compute persistent homology diagram for a batch of embeddings."""
        point_cloud = embeddings.detach().cpu().numpy()
        point_cloud = np.expand_dims(point_cloud, axis=0)
        diagrams = self.persistence.fit_transform(point_cloud)
        diagrams = self.scaler.fit_transform(diagrams)
        diagrams = self.filtering.fit_transform(diagrams)
        return diagrams[0]

    def topological_regularizer(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Encourage high-persistence structures by penalizing short lifetimes."""
        diagram = self.compute_diagram(embeddings)
        if diagram.size == 0:
            return torch.zeros(1, device=embeddings.device)

        lifetimes = diagram[:, 2] - diagram[:, 1]
        penalty = np.clip(self.cfg.persistence_threshold - lifetimes, a_min=0, a_max=None)
        return torch.as_tensor(penalty.mean(), device=embeddings.device, dtype=embeddings.dtype)

    @torch.no_grad()
    def discover_novel_categories(
        self, embeddings: torch.Tensor, known_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Cluster embeddings into novel categories using persistent homology cues."""
        device = embeddings.device
        if known_mask is not None:
            discovery_embeddings = embeddings[~known_mask]
        else:
            discovery_embeddings = embeddings

        if discovery_embeddings.numel() == 0:
            LOGGER.warning("No embeddings available for novel discovery.")
            return {
                "novel_labels": torch.empty(0, dtype=torch.long, device=device),
                "scores": torch.empty(0, dtype=embeddings.dtype, device=device),
            }

        diagram = self.compute_diagram(discovery_embeddings)
        entropy = self.entropy.fit_transform(diagram[None, ...])
        stability_score = float(entropy.squeeze()) if entropy.size else 0.0

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.cfg.hdbscan_min_cluster_size,
            min_samples=self.cfg.hdbscan_min_samples,
            metric="euclidean",
        )
        cluster_labels = clusterer.fit_predict(
            discovery_embeddings.detach().cpu().numpy()
        )
        novel_labels = torch.as_tensor(cluster_labels, device=device, dtype=torch.long)

        if known_mask is not None:
            full_labels = torch.full(
                (embeddings.shape[0],), fill_value=-1, dtype=torch.long, device=device
            )
            full_labels[~known_mask] = novel_labels
            novel_labels = full_labels

        scores = torch.full_like(novel_labels, fill_value=stability_score, dtype=embeddings.dtype)

        return {
            "novel_labels": novel_labels,
            "scores": scores,
        }
