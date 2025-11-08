from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from ripser import ripser
from sklearn.preprocessing import StandardScaler

try:
    import hdbscan
except ImportError:  # pragma: no cover - optional dependency
    hdbscan = None


@dataclass
class TopologyConfig:
    persistence_threshold: float = 0.2
    max_dimension: int = 2
    subsample_size: int = 2048
    clustering: str = "hdbscan"
    hdbscan: Optional[Dict[str, Any]] = None


class TopologicalDiscoveryModule:
    """Persistent homology-based discovery of novel expression categories."""

    def __init__(self, config: Dict[str, Any]) -> None:
        cfg = TopologyConfig(**config)
        self.cfg = cfg
        if cfg.clustering == "hdbscan" and hdbscan is None:
            raise ImportError("hdbscan is required for clustering but is not installed.")

    def _subsample(self, embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings) <= self.cfg.subsample_size:
            return embeddings
        indices = np.random.choice(len(embeddings), self.cfg.subsample_size, replace=False)
        return embeddings[indices]

    def _compute_persistence(self, embeddings: np.ndarray) -> Dict[str, Any]:
        diagrams = ripser(embeddings, maxdim=self.cfg.max_dimension)
        return diagrams

    def _high_persistence_features(self, diagrams: Dict[str, Any]) -> Dict[int, np.ndarray]:
        high_persistence: Dict[int, np.ndarray] = {}
        for dim, diag in enumerate(diagrams["dgms"]):
            persistence = diag[:, 1] - diag[:, 0]
            mask = persistence >= self.cfg.persistence_threshold
            high_persistence[dim] = diag[mask]
        return high_persistence

    def _cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        if self.cfg.clustering != "hdbscan":
            raise ValueError(f"Unsupported clustering algorithm {self.cfg.clustering}")
        params = {
            "min_cluster_size": 20,
            "min_samples": 10,
            "cluster_selection_epsilon": 0.0,
        }
        if self.cfg.hdbscan:
            params.update(self.cfg.hdbscan)
        algo = hdbscan.HDBSCAN(**params)
        scaled = StandardScaler().fit_transform(embeddings)
        labels = algo.fit_predict(scaled)
        return labels

    def discover(self, embeddings: np.ndarray) -> Dict[str, Any]:
        sampled_embeddings = self._subsample(embeddings)
        diagrams = self._compute_persistence(sampled_embeddings)
        high_persistence = self._high_persistence_features(diagrams)
        clusters = self._cluster_embeddings(sampled_embeddings)
        return {
            "sampled_embeddings": sampled_embeddings,
            "persistence_diagrams": diagrams["dgms"],
            "cocycles": diagrams.get("cocycles"),
            "high_persistence": high_persistence,
            "clusters": clusters,
        }

    def score_novelty(self, embeddings: np.ndarray, reference_embeddings: np.ndarray) -> np.ndarray:
        """Compute novelty score based on distance to reference manifold."""
        from sklearn.neighbors import NearestNeighbors

        knn = NearestNeighbors(n_neighbors=5)
        knn.fit(reference_embeddings)
        distances, _ = knn.kneighbors(embeddings)
        novelty = distances.mean(axis=1)
        novelty = (novelty - novelty.min()) / (novelty.max() - novelty.min() + 1e-8)
        return novelty

