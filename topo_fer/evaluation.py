"""Evaluation utilities for TOPO-FER."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

from .models.scaffold import GeometricScaffoldNet
from .modules.topology import TopologicalDiscoveryModule
from .utils.metrics import accuracy


def evaluate_known_classes(
    model: GeometricScaffoldNet,
    dataloader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate known-class accuracy."""
    model.eval()
    total_accuracy = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            outputs = model(images)
            total_accuracy += accuracy(outputs["logits"], labels).item()
            num_batches += 1
    return {"accuracy": total_accuracy / max(1, num_batches)}


def build_expression_map(
    model: GeometricScaffoldNet,
    dataloader,
    device: torch.device,
    output_path: Path,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> None:
    """Generate a 2D t-SNE visualization of the expression manifold."""
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            outputs = model(images)
            embeddings.append(outputs["manifold"])
            labels.append(batch["label"])

    if not embeddings:
        raise RuntimeError("No embeddings were collected for visualization.")

    embedding_tensor = torch.cat(embeddings, dim=0).cpu().numpy()
    label_tensor = torch.cat(labels, dim=0).cpu().numpy()

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    tsne_coords = tsne.fit_transform(embedding_tensor)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_coords[:, 0], tsne_coords[:, 1], c=label_tensor, cmap="tab20", s=12)
    plt.colorbar(scatter, label="Expression label")
    plt.title("TOPO-FER Manifold Visualization")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def discover_novel_categories(
    topology_module: TopologicalDiscoveryModule,
    embeddings: torch.Tensor,
    known_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Convenience wrapper for topological discovery."""
    return topology_module.discover_novel_categories(embeddings, known_mask)
