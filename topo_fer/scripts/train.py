from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from topo_fer.data.datamodule import FacialExpressionDataModule
from topo_fer.training.module import TOPOFERLightningModule
from topo_fer.utils.config import load_config, merge_configs
from topo_fer.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TOPO-FER.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("topo_fer/configs/default.yaml"),
        help="Path to configuration file.",
    )
    parser.add_argument(
        "--overrides",
        type=str,
        nargs="*",
        help="Optional list of KEY=VALUE overrides.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="topo-fer",
        help="Name of the experiment run.",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Run topological discovery over unlabeled data after training.",
    )
    return parser.parse_args()


def parse_overrides(overrides: list[str] | None) -> dict:
    if not overrides:
        return {}
    result = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Expected format KEY=VALUE.")
        key, value = override.split("=", 1)
        parts = key.split(".")
        ref = result
        for part in parts[:-1]:
            ref = ref.setdefault(part, {})
        try:
            value_loaded = json.loads(value)
        except json.JSONDecodeError:
            value_loaded = value
        ref[parts[-1]] = value_loaded
    return result


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.config)
    override_cfg = parse_overrides(args.overrides)
    cfg = merge_configs(base_cfg, override_cfg)

    exp_cfg = cfg["experiment"]
    seed_everything(exp_cfg.get("seed", 42), workers=True)
    log_path = configure_logging(exp_cfg.get("output_dir", "outputs"), args.experiment_name)

    data_cfg = cfg["data"]
    label_mapping = data_cfg.get("label_mapping", {})
    num_known_classes = len(label_mapping) or cfg.get("model", {}).get("num_known_classes", 7)

    datamodule = FacialExpressionDataModule(data_cfg, label_mapping)
    module = TOPOFERLightningModule(cfg, num_known_classes=num_known_classes)

    output_dir = Path(exp_cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename=f"{args.experiment_name}" + "-{epoch:03d}-{val_accuracy:.3f}",
        save_top_k=3,
        monitor="val/accuracy",
        mode="max",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = TensorBoardLogger(
        save_dir=str(output_dir / "tensorboard"),
        name=args.experiment_name,
    )

    trainer = Trainer(
        accelerator=exp_cfg.get("accelerator", "gpu"),
        devices=exp_cfg.get("devices", 1),
        max_epochs=cfg["optimization"].get("max_epochs", 100),
        log_every_n_steps=exp_cfg.get("log_every_n_steps", 50),
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        gradient_clip_val=cfg["optimization"].get("gradient_clip_val", 1.0),
    )

    trainer.fit(module, datamodule=datamodule)

    if args.discover:
        topo_results = module.run_topological_discovery(datamodule.unlabeled_dataloader())
        discovery_path = output_dir / "topology" / args.experiment_name
        discovery_path.mkdir(parents=True, exist_ok=True)
        npy_path = discovery_path / "clusters.npy"
        np.save(npy_path, topo_results["clusters"])
        with (discovery_path / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "log_file": str(log_path),
                    "persistence_threshold": cfg["discovery"]["persistence_threshold"],
                    "num_clusters": int(np.unique(topo_results["clusters"]).size),
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
