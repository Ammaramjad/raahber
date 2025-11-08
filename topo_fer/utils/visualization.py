from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from persim import plot_diagrams


def plot_persistence_diagram(diagrams: Iterable[np.ndarray], save_path: Path | None = None) -> None:
    plt.figure(figsize=(6, 6))
    plot_diagrams(diagrams, show=False)
    plt.title("Persistence Diagram")
    plt.xlabel("Birth")
    plt.ylabel("Death")
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_mapper_graph(graph: nx.Graph, save_path: Path | None = None) -> None:
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph, seed=42)
    node_sizes = [graph.nodes[node].get("size", 50) * 10 for node in graph.nodes]
    node_colors = [graph.nodes[node].get("cluster", 0) for node in graph.nodes]
    nx.draw_networkx(
        graph,
        pos=pos,
        with_labels=False,
        node_size=node_sizes,
        node_color=node_colors,
        cmap="Spectral",
    )
    plt.title("Mapper Graph of Expression Manifold")
    plt.axis("off")
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def build_mapper_graph(embeddings: np.ndarray, clusters: np.ndarray) -> nx.Graph:
    graph = nx.Graph()
    for idx, cluster_id in enumerate(clusters):
        node_id = int(cluster_id)
        if node_id not in graph:
            graph.add_node(node_id, size=0, members=[])
        graph.nodes[node_id]["size"] += 1
        graph.nodes[node_id]["members"].append(idx)
    unique_clusters = np.unique(clusters)
    for i in unique_clusters:
        for j in unique_clusters:
            if i >= j:
                continue
            members_i = embeddings[clusters == i]
            members_j = embeddings[clusters == j]
            centroid_i = members_i.mean(axis=0)
            centroid_j = members_j.mean(axis=0)
            similarity = np.dot(centroid_i, centroid_j) / (
                np.linalg.norm(centroid_i) * np.linalg.norm(centroid_j) + 1e-8
            )
            if similarity > 0.9:
                graph.add_edge(int(i), int(j), weight=float(similarity))
    return graph

