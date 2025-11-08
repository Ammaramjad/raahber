from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Optional

from lightning import Trainer

from topo_fer.data.datamodule import FacialExpressionDataModule
from topo_fer.training.module import TOPOFERLightningModule
from topo_fer.utils.config import load_config, merge_configs
from topo_fer.utils.logging import configure_logging


def _parse_overrides(overrides: Sequence[str]) -> Dict[str, Any]:
    from topo_fer.scripts.train import parse_overrides

    return parse_overrides(list(overrides))


def _normalize_overrides(overrides: Mapping[str, Any] | Sequence[str] | str | None) -> Dict[str, Any]:
    if overrides is None:
        return {}
    if isinstance(overrides, Mapping):
        return dict(overrides)
    if isinstance(overrides, str):
        overrides = [overrides]
    if isinstance(overrides, Sequence):
        return _parse_overrides(overrides)
    raise TypeError(
        "Overrides must be a mapping, a string, a sequence of strings, or None."
    )


def evaluate_model(
    config: str | Path,
    checkpoint: str | Path,
    *,
    overrides: Mapping[str, Any] | Sequence[str] | str | None = None,
    experiment_name: str = "topo-fer-eval",
    trainer_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate a TOPO-FER checkpoint and return the aggregated metrics.

    Args:
        config: Path to the YAML configuration file used for training.
        checkpoint: Path to the Lightning checkpoint file.
        overrides: Optional configuration overrides. Accepts a mapping or
            sequence of ``KEY=VALUE`` strings following the CLI format.
        experiment_name: Name of the evaluation run, used for logging.
        trainer_kwargs: Optional keyword arguments forwarded to ``Trainer``.

    Returns:
        A dictionary with metric names as keys and floating-point values.
    """
    config_path = Path(config)
    checkpoint_path = Path(checkpoint)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    base_cfg = load_config(config_path)
    override_cfg = _normalize_overrides(overrides)
    cfg = merge_configs(base_cfg, override_cfg)

    exp_cfg = cfg["experiment"]
    configure_logging(exp_cfg.get("output_dir", "outputs"), experiment_name)

    data_cfg = cfg["data"]
    label_mapping = data_cfg.get("label_mapping", {})
    num_known_classes = len(label_mapping) or cfg.get("model", {}).get("num_known_classes", 7)

    datamodule = FacialExpressionDataModule(data_cfg, label_mapping)
    module = TOPOFERLightningModule.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        config=cfg,
        num_known_classes=num_known_classes,
    )

    trainer_params: Dict[str, Any] = dict(trainer_kwargs or {})
    trainer_params.setdefault("accelerator", exp_cfg.get("accelerator", "gpu"))
    trainer_params.setdefault("devices", exp_cfg.get("devices", 1))

    trainer = Trainer(**trainer_params)
    results = trainer.test(module, datamodule=datamodule)
    if not results:
        return {}
    return results[0]


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
    results = evaluate_model(
        config=args.config,
        checkpoint=args.checkpoint,
        overrides=args.overrides,
        experiment_name=args.experiment_name,
    )

    if results:
        print("Evaluation metrics:")
        for key, value in results.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
