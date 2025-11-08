"""Training pipeline for TOPO-FER."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.optim as optim
from torch.cuda import amp

from .config import ExperimentConfig
from .data import build_dataloaders
from .losses import (
    aggregate_losses,
    classification_loss,
    manifold_smoothness_loss,
    reconstruction_loss,
    supervised_contrastive_loss,
)
from .models.scaffold import GeometricScaffoldNet
from .modules.topology import TopologicalDiscoveryModule
from .utils.logging import configure_file_logger, get_logger
from .utils.metrics import accuracy


LOGGER = get_logger(__name__)


class TopoFERTrainer:
    """Trainer encapsulating optimization and evaluation routines."""

    def __init__(self, cfg: ExperimentConfig, num_known_classes: int) -> None:
       self.cfg = cfg
       self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       LOGGER.info("Using device: %s", self.device)

       self.dataloaders = build_dataloaders(
           cfg.dataset, cfg.training.batch_size, cfg.training.num_workers
       )

       self.model = GeometricScaffoldNet(cfg.model, num_known_classes=num_known_classes)
       self.model.to(self.device)

       self.topology = TopologicalDiscoveryModule(cfg.topology)
       self.optimizer = optim.AdamW(
           self.model.parameters(),
           lr=cfg.training.learning_rate,
           weight_decay=cfg.training.weight_decay,
       )
       self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
           self.optimizer, T_max=cfg.training.max_epochs
       )

       self.scaler: Optional[amp.GradScaler] = (
           amp.GradScaler(enabled=cfg.training.mixed_precision)
       )

       Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
       configure_file_logger(LOGGER, cfg.output_dir, "training.log")
       self.best_val_accuracy = 0.0

    def train(self) -> None:
       """Main training loop."""
       for epoch in range(1, self.cfg.training.max_epochs + 1):
           train_stats = self._train_one_epoch(epoch)
           val_stats = self._validate(epoch)

           self.lr_scheduler.step()

           LOGGER.info(
               "Epoch %d | Train Loss %.4f | Val Loss %.4f | Val Acc %.3f",
               epoch,
               train_stats["loss"],
               val_stats["loss"],
               val_stats["accuracy"],
           )

           if val_stats["accuracy"] > self.best_val_accuracy:
               self.best_val_accuracy = val_stats["accuracy"]
               self._save_checkpoint(epoch, is_best=True)

           if self.cfg.training.discovery_start_epoch > 0 and epoch >= self.cfg.training.discovery_start_epoch:
               self._run_topological_discovery(epoch)

           if epoch % self.cfg.training.checkpoint_interval == 0:
               self._save_checkpoint(epoch, is_best=False)

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
       """Train for a single epoch."""
       self.model.train()
       total_loss = 0.0
       total_accuracy = 0.0
       num_batches = 0

       for batch in self.dataloaders["train"]:
           images = batch["image"].to(self.device, non_blocking=True)
           labels = batch["label"].to(self.device, non_blocking=True)

           self.optimizer.zero_grad()

           with amp.autocast(enabled=self.cfg.training.mixed_precision):
               outputs = self.model(images, return_latent=True)
               loss_components = {
                   "classification": classification_loss(outputs["logits"], labels),
                   "manifold_smoothness": manifold_smoothness_loss(
                       outputs["latent"], outputs["latent_ode"]
                   ),
                   "reconstruction": reconstruction_loss(outputs["reconstruction"], outputs["features"]),
                   "contrastive": supervised_contrastive_loss(outputs["manifold"], labels),
                    "topology": self.topology.topological_regularizer(outputs["manifold"]),
               }
               loss = aggregate_losses(
                   loss_components,
                   weights={
                       "classification": self.cfg.training.known_classification_weight,
                       "manifold_smoothness": self.cfg.training.manifold_smoothness_weight,
                       "contrastive": self.cfg.training.contrastive_weight,
                       "reconstruction": self.cfg.topology.reconstruction_weight,
                       "topology": self.cfg.topology.topological_regularizer_weight,
                   },
               )

           if self.scaler is not None:
               self.scaler.scale(loss).backward()
               self.scaler.unscale_(self.optimizer)
               torch.nn.utils.clip_grad_norm_(
                   self.model.parameters(), self.cfg.training.gradient_clip_norm
               )
               self.scaler.step(self.optimizer)
               self.scaler.update()
           else:
               loss.backward()
               torch.nn.utils.clip_grad_norm_(
                   self.model.parameters(), self.cfg.training.gradient_clip_norm
               )
               self.optimizer.step()

           total_loss += loss.item()
           total_accuracy += accuracy(outputs["logits"], labels).item()
           num_batches += 1

       return {
           "loss": total_loss / max(1, num_batches),
           "accuracy": total_accuracy / max(1, num_batches),
       }

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
       """Validation loop."""
       self.model.eval()
       total_loss = 0.0
       total_accuracy = 0.0
       num_batches = 0

       for batch in self.dataloaders["val"]:
           images = batch["image"].to(self.device, non_blocking=True)
           labels = batch["label"].to(self.device, non_blocking=True)
           outputs = self.model(images, return_latent=True)

           loss_components = {
               "classification": classification_loss(outputs["logits"], labels),
               "manifold_smoothness": manifold_smoothness_loss(
                   outputs["latent"], outputs["latent_ode"]
               ),
               "reconstruction": reconstruction_loss(outputs["reconstruction"], outputs["features"]),
               "contrastive": supervised_contrastive_loss(outputs["manifold"], labels),
                "topology": self.topology.topological_regularizer(outputs["manifold"]),
           }
           loss = aggregate_losses(
               loss_components,
               weights={
                   "classification": self.cfg.training.known_classification_weight,
                   "manifold_smoothness": self.cfg.training.manifold_smoothness_weight,
                   "contrastive": self.cfg.training.contrastive_weight,
                   "reconstruction": self.cfg.topology.reconstruction_weight,
                   "topology": self.cfg.topology.topological_regularizer_weight,
               },
           )

           total_loss += loss.item()
           total_accuracy += accuracy(outputs["logits"], labels).item()
           num_batches += 1

       return {
           "loss": total_loss / max(1, num_batches),
           "accuracy": total_accuracy / max(1, num_batches),
       }

    @torch.no_grad()
    def _run_topological_discovery(self, epoch: int) -> None:
       """Perform unsupervised discovery on validation embeddings."""
       self.model.eval()
       all_embeddings = []
       all_known_mask = []

       for batch in self.dataloaders["val"]:
           images = batch["image"].to(self.device, non_blocking=True)
           labels = batch["label"].to(self.device, non_blocking=True)
           outputs = self.model(images, return_latent=False)
           all_embeddings.append(outputs["manifold"])
           all_known_mask.append(labels >= 0)

       embeddings = torch.cat(all_embeddings, dim=0)
       known_mask = torch.cat(all_known_mask, dim=0).to(self.device)
       discovery = self.topology.discover_novel_categories(embeddings, known_mask)

       stats_path = Path(self.cfg.output_dir) / f"discovery_epoch_{epoch:03d}.json"
       with open(stats_path, "w", encoding="utf-8") as f:
           json.dump(
               {
                   "epoch": epoch,
                   "stability_scores": discovery["scores"].tolist(),
                   "novel_labels": discovery["novel_labels"].tolist(),
               },
               f,
               indent=2,
           )
       LOGGER.info("Saved discovery statistics to %s", stats_path)

    def _save_checkpoint(self, epoch: int, is_best: bool) -> None:
       """Persist model and optimizer state."""
       checkpoint = {
           "epoch": epoch,
           "model_state": self.model.state_dict(),
           "optimizer_state": self.optimizer.state_dict(),
           "lr_scheduler_state": self.lr_scheduler.state_dict(),
           "best_val_accuracy": self.best_val_accuracy,
           "config": self.cfg,
       }
       path = Path(self.cfg.output_dir) / f"checkpoint_epoch_{epoch:03d}.pth"
       torch.save(checkpoint, path)
       LOGGER.info("Saved checkpoint: %s", path)
       if is_best:
           best_path = Path(self.cfg.output_dir) / "best.pth"
           torch.save(checkpoint, best_path)
           LOGGER.info("Updated best checkpoint: %s", best_path)
