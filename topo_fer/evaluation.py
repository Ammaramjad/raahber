"""High-level evaluation utilities for TOPO-FER models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping

import numpy as np
import torch
import torch.nn.functional as F

from topo_fer.data.datamodule import FacialExpressionDataModule
from topo_fer.training.module import TOPOFERLightningModule
from topo_fer.utils.config import load_config, merge_configs
from topo_fer.utils.logging import configure_logging
from topo_fer.utils.metrics import classification_metrics, open_set_metrics


def _normalize_stage(stage: str) -> str:
    normalized = stage.lower()
    if normalized in {"val", "validate", "validation"}:
        return "validate"
    if normalized in {"test", "eval", "evaluation"}:
        return "test"
    raise ValueError(f"Unsupported evaluation stage '{stage}'. Expected 'val' or 'test'.")


def _pick_metrics(
    available: Dict[str, float],
    requested: Iterable[str] | None,
) -> Dict[str, float]:
    if not requested:
        return available
    requested_set = set(requested)
    return {name: value for name, value in available.items() if name in requested_set}


def evaluate_model(
    config: str | Path | Mapping[str, Any],
    checkpoint: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
    experiment_name: str = "topo-fer-eval",
    stage: str = "test",
    device: str | torch.device | None = None,
) -> Dict[str, Any]:
    """Evaluate a TOPO-FER checkpoint and return summary metrics.

    Args:
        config: Path to a YAML configuration file or a configuration mapping.
        checkpoint: Path to the Lightning checkpoint file.
        overrides: Optional configuration overrides (same semantics as training).
        experiment_name: Name used for logging directory/file naming.
        stage: Which dataloader split to evaluate: ``"val"``/``"validation"`` or
            ``"test"``/``"evaluation"``.
        device: Optional torch device specifier. If omitted, CUDA is used when
            available, otherwise CPU.

    Returns:
        A dictionary with the evaluation stage, number of samples, resolved log file,
        and dictionaries of closed-set and open-set metrics.
    """
    if isinstance(config, (str, Path)):
        base_cfg: MutableMapping[str, Any] = load_config(config)
    else:
        base_cfg = dict(config)  # shallow copy
    cfg: MutableMapping[str, Any] = merge_configs(base_cfg, overrides or {})

    exp_cfg = cfg.get("experiment", {})
    log_path = configure_logging(exp_cfg.get("output_dir", "outputs"), experiment_name)

    stage_key = _normalize_stage(stage)
    data_cfg = cfg["data"]
    label_mapping = data_cfg.get("label_mapping", {})
    num_known_classes = len(label_mapping) or cfg.get("model", {}).get("num_known_classes", 7)

    datamodule = FacialExpressionDataModule(data_cfg, label_mapping)
    datamodule.setup(stage=stage_key)
    if stage_key == "validate":
        dataloader = datamodule.val_dataloader()
        metric_prefix = "val"
    else:
        dataloader = datamodule.test_dataloader()
        metric_prefix = "test"

    module = TOPOFERLightningModule.load_from_checkpoint(
        checkpoint_path=str(Path(checkpoint).expanduser()),
        config=cfg,
        num_known_classes=num_known_classes,
    )

    if device is None:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)
    module = module.to(torch_device)
    module.eval()

    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    known_scores: list[float] = []
    latent_norms: list[float] = []
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"]
            labels = batch["label"]
            if not isinstance(images, torch.Tensor):
                images = torch.as_tensor(images)
            if not isinstance(labels, torch.Tensor):
                labels = torch.as_tensor(labels)

            images = images.to(torch_device, non_blocking=True)
            labels = labels.to(torch_device)

            mask = labels >= 0
            if not mask.any():
                continue

            images = images[mask]
            labels = labels[mask]

            outputs = module(images)
            logits = outputs["logits"]
            probs = F.softmax(logits, dim=-1)

            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())

            confidence = probs.max(dim=-1).values.detach().cpu().numpy()
            known_scores.extend(confidence.tolist())

            if "latent_final" in outputs:
                latent = outputs["latent_final"]
                norms = latent.norm(dim=-1).detach().cpu().numpy()
                latent_norms.extend(norms.tolist())

            total_samples += labels.shape[0]

    if not all_labels:
        raise RuntimeError("No labeled samples encountered during evaluation.")

    labels_np = np.concatenate(all_labels)
    probs_np = np.concatenate(all_probs)
    logits_np = np.concatenate(all_logits)
    preds_np = np.argmax(probs_np, axis=-1)

    closed_metrics = classification_metrics(labels_np, preds_np, probs_np)
    eval_cfg = cfg.get("evaluation", {})
    closed_metrics = _pick_metrics(closed_metrics, eval_cfg.get("known_class_metrics"))

    open_metrics: Dict[str, float] = {}
    if known_scores and latent_norms:
        latent_arr = np.array(latent_norms)
        latent_norm = (latent_arr - latent_arr.min()) / (latent_arr.max() - latent_arr.min() + 1e-8)
        novel_knownness = 1.0 - latent_norm
        open_results = open_set_metrics(known_scores, novel_knownness)
        open_metrics = _pick_metrics(open_results, eval_cfg.get("open_set_metrics"))

    return {
        "stage": metric_prefix,
        "num_samples": total_samples,
        "log_path": str(log_path),
        "classification": closed_metrics,
        "open_set": open_metrics,
        "raw": {
            "labels": labels_np,
            "predictions": preds_np,
            "logits": logits_np,
            "probabilities": probs_np,
        },
    }

