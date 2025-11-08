from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Mapping

from lightning import Trainer

from topo_fer.data.datamodule import FacialExpressionDataModule
from topo_fer.training.module import TOPOFERLightningModule
from topo_fer.utils.config import load_config, merge_configs
from topo_fer.utils.logging import configure_logging


def evaluate_model(
    config: str | Path | Dict[str, Any],
    checkpoint: str | Path,
    overrides: Mapping[str, Any] | None = None,
    *,
    experiment_name: str | None = "topo-fer-eval",
    trainer_kwargs: Dict[str, Any] | None = None,
    enable_logging: bool = True,
) -> Dict[str, Any]:
    """Run evaluation for a TOPO-FER checkpoint and return aggregated metrics.

    Args:
        config: Path to a configuration file or a configuration dictionary.
        checkpoint: Path to the Lightning checkpoint to evaluate.
        overrides: Optional configuration overrides that will be merged into the base config.
        experiment_name: Name used for logging directory creation.
        trainer_kwargs: Optional keyword arguments forwarded to ``lightning.Trainer``.
        enable_logging: Whether to configure file logging directories.

    Returns:
        A dictionary containing the metrics reported by ``Trainer.test``.
    """
    if isinstance(config, (str, Path)):
        base_cfg = load_config(config)
    elif isinstance(config, dict):
        base_cfg = copy.deepcopy(config)
    else:
        raise TypeError("config must be a path or a dictionary-like object.")

    override_cfg = dict(overrides) if overrides else None
    cfg = merge_configs(base_cfg, override_cfg)

    exp_cfg = cfg.get("experiment", {})
    data_cfg = cfg["data"]
    model_cfg = cfg.get("model", {})

    if enable_logging:
        exp_name = experiment_name or exp_cfg.get("name", "topo-fer-eval")
        configure_logging(exp_cfg.get("output_dir", "outputs"), exp_name)

    label_mapping = data_cfg.get("label_mapping", {})
    num_known_classes = len(label_mapping) or model_cfg.get("num_known_classes", 7)

    datamodule = FacialExpressionDataModule(data_cfg, label_mapping)
    module = TOPOFERLightningModule.load_from_checkpoint(
        checkpoint_path=str(checkpoint),
        config=cfg,
        num_known_classes=num_known_classes,
    )

    trainer_config: Dict[str, Any] = {
        "accelerator": exp_cfg.get("accelerator", "gpu"),
        "devices": exp_cfg.get("devices", 1),
    }
    if trainer_kwargs:
        trainer_config.update(trainer_kwargs)

    trainer = Trainer(**trainer_config)
    results = trainer.test(module, datamodule=datamodule)

    return results[0] if results else {}

