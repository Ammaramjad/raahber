from __future__ import annotations

import argparse
from pathlib import Path

from lightning import Trainer

from topo_fer.data.datamodule import FacialExpressionDataModule
from topo_fer.training.module import TOPOFERLightningModule
from topo_fer.utils.config import load_config, merge_configs
from topo_fer.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TOPO-FER checkpoint.")
    parser.add_argument("--config", type=Path, required=True, help="Path to config used for training.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint file path.")
    parser.add_argument(
        "--overrides",
        type=str,
        nargs="*",
        help="Optional KEY=VALUE overrides (same semantics as training).",
    )
    parser.add_argument("--experiment_name", type=str, default="topo-fer-eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.config)
    override_cfg = {}
    if args.overrides:
        from topo_fer.scripts.train import parse_overrides

        override_cfg = parse_overrides(args.overrides)
    cfg = merge_configs(base_cfg, override_cfg)

    exp_cfg = cfg["experiment"]
    configure_logging(exp_cfg.get("output_dir", "outputs"), args.experiment_name)

    data_cfg = cfg["data"]
    label_mapping = data_cfg.get("label_mapping", {})
    num_known_classes = len(label_mapping) or cfg.get("model", {}).get("num_known_classes", 7)

    datamodule = FacialExpressionDataModule(data_cfg, label_mapping)
    module = TOPOFERLightningModule.load_from_checkpoint(
        checkpoint_path=str(args.checkpoint),
        config=cfg,
        num_known_classes=num_known_classes,
    )

    trainer = Trainer(
        accelerator=exp_cfg.get("accelerator", "gpu"),
        devices=exp_cfg.get("devices", 1),
    )
    trainer.test(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
