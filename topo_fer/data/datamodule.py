from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader

from .datasets import BaseFacialExpressionDataset, build_dataset
from .transforms import build_transforms


@dataclass
class DatasetConfig:
    name: str
    root: str
    annotation: str
    train_split: str = "train"
    eval_split: str = "val"
    unlabeled_split: str = "unlabeled"


class FacialExpressionDataModule(LightningDataModule):
    def __init__(
        self,
        data_cfg: Dict[str, Any],
        label_mapping: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.cfg = data_cfg
        self.label_mapping = label_mapping or {}
        self.batch_size = data_cfg.get("batch_size", 32)
        self.num_workers = data_cfg.get("num_workers", 8)
        image_size = data_cfg.get("image_size", 112)
        aug_cfg = data_cfg.get("augmentations", [])
        self.train_transform, self.eval_transform = build_transforms(image_size, aug_cfg)
        self.datasets_cfg = [
            DatasetConfig(
                name=dataset["name"],
                root=dataset["root"],
                annotation=dataset["annotation"],
                train_split=dataset.get("train_split", "train"),
                eval_split=dataset.get("eval_split", "val"),
                unlabeled_split=dataset.get("unlabeled_split", "unlabeled"),
            )
            for dataset in data_cfg.get("datasets", [])
        ]
        if not self.datasets_cfg:
            raise ValueError("No datasets specified in the configuration.")

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        self._unlabeled_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        train_sets: List[BaseFacialExpressionDataset] = []
        val_sets: List[BaseFacialExpressionDataset] = []
        test_sets: List[BaseFacialExpressionDataset] = []
        unlabeled_sets: List[BaseFacialExpressionDataset] = []

        for dataset_cfg in self.datasets_cfg:
            train_sets.append(
                build_dataset(
                    dataset_cfg.name,
                    root=dataset_cfg.root,
                    annotation=dataset_cfg.annotation,
                    split=dataset_cfg.train_split,
                    transform=self.train_transform,
                    label_mapping=self.label_mapping,
                )
            )
            val_sets.append(
                build_dataset(
                    dataset_cfg.name,
                    root=dataset_cfg.root,
                    annotation=dataset_cfg.annotation,
                    split=dataset_cfg.eval_split,
                    transform=self.eval_transform,
                    label_mapping=self.label_mapping,
                )
            )
            test_sets.append(
                build_dataset(
                    dataset_cfg.name,
                    root=dataset_cfg.root,
                    annotation=dataset_cfg.annotation,
                    split=dataset_cfg.eval_split,
                    transform=self.eval_transform,
                    label_mapping=self.label_mapping,
                )
            )
            unlabeled_sets.append(
                build_dataset(
                    dataset_cfg.name,
                    root=dataset_cfg.root,
                    annotation=dataset_cfg.annotation,
                    split=dataset_cfg.unlabeled_split,
                    transform=self.eval_transform,
                    label_mapping=self.label_mapping,
                )
            )

        self._train_dataset = ConcatDataset(train_sets)
        self._val_dataset = ConcatDataset(val_sets)
        self._test_dataset = ConcatDataset(test_sets)
        self._unlabeled_dataset = ConcatDataset(unlabeled_sets)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def unlabeled_dataloader(self) -> DataLoader:
        return DataLoader(
            self._unlabeled_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

