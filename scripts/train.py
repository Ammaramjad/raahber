"""Training entry point for TOPO-FER."""

from __future__ import annotations

import argparse
from pathlib import Path

from topo_fer.config import ExperimentConfig, load_experiment_config
from topo_fer.trainer import TopoFERTrainer
from topo_fer.utils.logging import get_logger
from topo_fer.utils.random import set_seed


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TOPO-FER model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to experiment configuration file.",
    )
    parser.add_argument(
        "--known-classes",
        type=int,
        required=True,
        help="Number of known expression categories in the labeled set.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional override for output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg: ExperimentConfig = load_experiment_config(args.config)
    if args.output_dir:
        cfg.output_dir = args.output_dir

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    LOGGER.info("Loaded configuration from %s", args.config)
    trainer = TopoFERTrainer(cfg, num_known_classes=args.known_classes)
    trainer.train()


if __name__ == "__main__":
    main()
