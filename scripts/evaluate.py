"""Evaluation script for TOPO-FER."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from topo_fer.config import ExperimentConfig, load_experiment_config
from topo_fer.data import build_dataloaders
from topo_fer.evaluation import (
    build_expression_map,
    discover_novel_categories,
    evaluate_known_classes,
)
from topo_fer.models.scaffold import GeometricScaffoldNet
from topo_fer.modules.topology import TopologicalDiscoveryModule
from topo_fer.utils.logging import get_logger
from topo_fer.utils.random import set_seed


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the TOPO-FER model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file used for training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a trained checkpoint (.pth).",
    )
    parser.add_argument(
        "--known-classes",
        type=int,
        required=True,
        help="Number of known expression categories.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="known",
        choices=["known", "discover", "visualize"],
        help="Evaluation mode: known-class accuracy, novel discovery, or visualization.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/eval",
        help="Output directory for evaluation artifacts.",
    )
    return parser.parse_args()


def load_model(cfg: ExperimentConfig, checkpoint_path: str, num_known_classes: int) -> GeometricScaffoldNet:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeometricScaffoldNet(cfg.model, num_known_classes=num_known_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    LOGGER.info("Loaded checkpoint from %s (epoch %s)", checkpoint_path, checkpoint.get("epoch", "unknown"))
    return model


def main() -> None:
    args = parse_args()
    cfg: ExperimentConfig = load_experiment_config(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    dataloaders = build_dataloaders(
        cfg.dataset, batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers
    )

    model = load_model(cfg, args.checkpoint, num_known_classes=args.known_classes)
    device = next(model.parameters()).device

    if args.mode == "known":
        metrics = evaluate_known_classes(model, dataloaders["val"], device)
        LOGGER.info("Known-class evaluation: %s", metrics)
        with open(output_dir / "known_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    elif args.mode == "discover":
        topology = TopologicalDiscoveryModule(cfg.topology)
        embeddings = []
        known_mask = []
        with torch.no_grad():
            for batch in dataloaders["val"]:
                images = batch["image"].to(device, non_blocking=True)
                outputs = model(images)
                embeddings.append(outputs["manifold"])
                known_mask.append(batch["label"] >= 0)
        manifold = torch.cat(embeddings, dim=0)
        mask = torch.cat(known_mask, dim=0).to(device)
        discovery = discover_novel_categories(topology, manifold, mask)
        LOGGER.info("Novel discovery complete. Found %d clusters.", discovery["novel_labels"].unique().numel())
        with open(output_dir / "discovery.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "novel_labels": discovery["novel_labels"].tolist(),
                    "scores": discovery["scores"].tolist(),
                },
                f,
                indent=2,
            )
    elif args.mode == "visualize":
        map_path = output_dir / "expression_map.png"
        build_expression_map(model, dataloaders["val"], device, map_path)
        LOGGER.info("Saved expression manifold visualization to %s", map_path)


if __name__ == "__main__":
    main()
