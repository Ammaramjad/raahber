from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from lightning import Trainer

from topo_fer.data.datamodule import FacialExpressionDataModule
from topo_fer.training.module import TOPOFERLightningModule
from topo_fer.utils.config import load_config, merge_configs, parse_overrides
from topo_fer.utils.logging import configure_logging


def _normalize_config(
    config: str | Path | Mapping[str, Any],
    overrides: Mapping[str, Any] | Sequence[str] | None,
) -> Dict[str, Any]:
    if isinstance(config, (str, Path)):
        base_cfg = load_config(config)
    elif isinstance(config, Mapping):
        base_cfg = dict(config)
    else:  # pragma: no cover - defensive branch
        raise TypeError(
            "config must be a path-like object or mapping, "
            f"got type {type(config).__name__}",
        )

    override_cfg: Dict[str, Any] = {}
    if overrides:
        if isinstance(overrides, Mapping):
            override_cfg = dict(overrides)
        elif isinstance(overrides, Sequence) and not isinstance(overrides, (str, bytes)):
            override_cfg = parse_overrides(list(overrides))
        else:  # pragma: no cover - defensive branch
            raise TypeError(
                "overrides must be a mapping or a sequence of KEY=VALUE strings, "
                f"got type {type(overrides).__name__}",
            )
    return merge_configs(base_cfg, override_cfg or None)


def evaluate_model(
    config: str | Path | Mapping[str, Any],
    checkpoint: str | Path,
    *,
    overrides: Mapping[str, Any] | Sequence[str] | None = None,
    experiment_name: str = "topo-fer-eval",
    dataloader_split: str = "test",
    trainer_kwargs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Evaluate a TOPO-FER checkpoint and return aggregated metrics.

    Args:
        config: Configuration dictionary or path to the original training config.
        checkpoint: Path to the Lightning checkpoint generated during training.
        overrides: Optional configuration overrides (dict or KEY=VALUE strings).
        experiment_name: Name used for logging directory structure.
        dataloader_split: Which dataloader to evaluate (`val` or `test`).
        trainer_kwargs: Extra keyword arguments forwarded to `lightning.Trainer`.

    Returns:
        A dictionary with metric names and their scalar values.
    """
    cfg = _normalize_config(config, overrides)

    exp_cfg = cfg.get("experiment", {})
    configure_logging(exp_cfg.get("output_dir", "outputs"), experiment_name)

    data_cfg = cfg["data"]
    label_mapping = data_cfg.get("label_mapping", {})
    num_known_classes = len(label_mapping) or cfg.get("model", {}).get("num_known_classes", 7)

    datamodule = FacialExpressionDataModule(data_cfg, label_mapping)
    datamodule.setup(stage=dataloader_split)

    module = TOPOFERLightningModule.load_from_checkpoint(
        checkpoint_path=str(checkpoint),
        config=cfg,
        num_known_classes=num_known_classes,
    )

    trainer_args = {
        "accelerator": exp_cfg.get("accelerator", "gpu"),
        "devices": exp_cfg.get("devices", 1),
        "logger": False,
        "enable_checkpointing": False,
    }
    if trainer_kwargs:
        trainer_args.update(trainer_kwargs)

    trainer = Trainer(**trainer_args)

    if dataloader_split not in {"val", "test"}:
        raise ValueError("dataloader_split must be 'val' or 'test'.")

    if dataloader_split == "val":
        dataloader = datamodule.val_dataloader()
    else:
        dataloader = datamodule.test_dataloader()

    results = trainer.validate(module, dataloaders=dataloader, verbose=False)
    return results[0] if results else {}
