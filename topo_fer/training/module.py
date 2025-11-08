from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from topo_fer.models.geometric_scaffold import GeometricScaffoldNetwork
from topo_fer.models.topological_discovery import TopologicalDiscoveryModule
from topo_fer.utils.metrics import classification_metrics, open_set_metrics


class TOPOFERLightningModule(LightningModule):
    def __init__(self, config: Dict[str, Any], num_known_classes: int) -> None:
        super().__init__()
        self.save_hyperparameters(config)
        model_cfg = config["model"]
        ode_cfg = dict(model_cfg.get("ode", {}))
        self.temperature = model_cfg.get("contrastive_temperature", 0.07)
        self.manifold_weight = model_cfg.get("manifold_regularization_weight", 0.1)
        self.topology_weight = model_cfg.get("topological_regularization_weight", 0.05)

        self.model = GeometricScaffoldNetwork(
            backbone=model_cfg.get("backbone", "resnet50"),
            embedding_dim=model_cfg.get("embedding_dim", 256),
            latent_dim=model_cfg.get("latent_dim", 16),
            ode_config=ode_cfg,
            num_known_classes=num_known_classes,
            classifier_dropout=model_cfg.get("classifier_dropout", 0.1),
        )
        self.discovery = TopologicalDiscoveryModule(config.get("discovery", {}))

        opt_cfg = config.get("optimization", {})
        self.lr = opt_cfg.get("lr", 1e-4)
        self.weight_decay = opt_cfg.get("weight_decay", 1e-2)
        self.warmup_steps = opt_cfg.get("warmup_steps", 0)
        self.max_epochs = opt_cfg.get("max_epochs", 100)
        self.scheduler_name = opt_cfg.get("scheduler", "cosine")

        self._criterion = nn.CrossEntropyLoss()
        self._eval_logits: List[np.ndarray] = []
        self._eval_labels: List[np.ndarray] = []
        self._eval_probs: List[np.ndarray] = []

        self._novel_scores: List[float] = []
        self._known_scores: List[float] = []

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(x)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        images = batch["image"]
        labels = batch["label"]
        labels = labels.to(torch.long)
        mask = labels >= 0
        if not mask.any():
            return torch.tensor(0.0, device=images.device, requires_grad=True)
        images = images[mask]
        labels = labels[mask]

        outputs = self(images)
        logits = outputs["logits"]

        classification_loss = self._criterion(logits, labels)
        recon_loss = F.mse_loss(outputs["reconstruction"], outputs["embedding"])
        manifold_energy = self.model.manifold_energy(outputs["latent_path"]).mean()

        loss = classification_loss + recon_loss + self.manifold_weight * manifold_energy

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/ce_loss", classification_loss, on_step=False, on_epoch=True)
        self.log("train/recon_loss", recon_loss, on_step=False, on_epoch=True)
        self.log("train/manifold_energy", manifold_energy, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        images = batch["image"]
        labels = batch["label"]
        labels = labels.to(torch.long)
        mask = labels >= 0
        if not mask.any():
            return
        images = images[mask]
        labels = labels[mask]
        outputs = self(images)
        logits = outputs["logits"]
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        self._eval_logits.append(logits.detach().cpu().numpy())
        self._eval_labels.append(labels.detach().cpu().numpy())
        self._eval_probs.append(probs.detach().cpu().numpy())

        confidence = probs.max(dim=-1).detach().cpu().numpy()
        novelty = outputs["latent_final"].norm(dim=-1).detach().cpu().numpy()
        self._known_scores.extend(confidence.tolist())
        self._novel_scores.extend(novelty.tolist())

        loss = self._criterion(logits, labels)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        if not self._eval_labels:
            return
        labels = np.concatenate(self._eval_labels)
        logits = np.concatenate(self._eval_logits)
        probs = np.concatenate(self._eval_probs)
        preds = np.argmax(probs, axis=-1)

        closed_metrics = classification_metrics(labels, preds, probs)
        for name, value in closed_metrics.items():
            self.log(f"val/{name}", value, prog_bar=True, sync_dist=True)

        if self._known_scores and self._novel_scores:
            known_scores = np.array(self._known_scores)
            novelty = np.array(self._novel_scores)
            novelty_norm = (novelty - novelty.min()) / (novelty.max() - novelty.min() + 1e-8)
            novel_knownness = 1.0 - novelty_norm
            open_metrics = open_set_metrics(known_scores, novel_knownness)
            for name, value in open_metrics.items():
                self.log(f"val/{name}", value, prog_bar=False, sync_dist=True)

        self._eval_logits.clear()
        self._eval_labels.clear()
        self._eval_probs.clear()
        self._novel_scores.clear()
        self._known_scores.clear()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.scheduler_name == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs)
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        if self.scheduler_name == "warmup":
            def lr_lambda(step: int) -> float:
                if step < self.warmup_steps:
                    return float(step + 1) / max(1, self.warmup_steps)
                return 1.0

            scheduler = LambdaLR(optimizer, lr_lambda)
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        return optimizer

    def extract_embeddings(self, dataloader) -> np.ndarray:
        self.eval()
        device = self.device
        embeddings: List[np.ndarray] = []
        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(device)
                outputs = self(images)
                embedding = outputs["embedding"].cpu().numpy()
                embeddings.append(embedding)
        return np.concatenate(embeddings, axis=0)

    def run_topological_discovery(self, dataloader) -> Dict[str, Any]:
        embeddings = self.extract_embeddings(dataloader)
        return self.discovery.discover(embeddings)

