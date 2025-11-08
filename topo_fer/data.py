"""Dataset and dataloader utilities for TOPO-FER."""

from __future__ import annotations

import csv
import os
from dataclasses import asdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from .config import DatasetConfig
from .utils.logging import get_logger


LOGGER = get_logger(__name__)


def _build_transforms(cfg: DatasetConfig, train: bool) -> Callable:
    """Construct torchvision transforms based on configuration."""
    base_transforms: List[Callable] = []
    augmentations = cfg.augmentations or []

    if train:
        for aug in augmentations:
            if aug == "random_horizontal_flip":
                base_transforms.append(transforms.RandomHorizontalFlip())
            elif aug == "color_jitter":
                base_transforms.append(
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
                    )
                )
            elif aug == "random_grayscale":
                base_transforms.append(transforms.RandomGrayscale(p=0.1))
            elif aug == "random_affine":
                base_transforms.append(
                    transforms.RandomAffine(
                        degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)
                    )
                )
            else:
                LOGGER.warning("Unsupported augmentation '%s' ignored.", aug)

    base_transforms.extend(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transforms.Compose(base_transforms)


class ExpressionRecord:
    """Simple structure for an expression sample."""

    __slots__ = ("path", "label", "split", "is_known")

    def __init__(self, path: str, label: int, split: str, is_known: bool) -> None:
        self.path = path
        self.label = label
        self.split = split
        self.is_known = is_known

    def __repr__(self) -> str:
        return (
            f"ExpressionRecord(path={self.path!r}, label={self.label}, "
            f"split={self.split!r}, is_known={self.is_known})"
        )


class ExpressionDataset(Dataset):
    """Dataset that loads expression samples based on a metadata CSV file."""

    def __init__(
        self,
        records: List[ExpressionRecord],
        transform: Optional[Callable] = None,
    ) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        image = Image.open(record.path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        sample = {
            "image": image,
            "label": torch.as_tensor(record.label, dtype=torch.long)
            if record.label >= 0
            else torch.as_tensor(-1, dtype=torch.long),
            "is_known": torch.as_tensor(record.is_known, dtype=torch.bool),
        }
        return sample


def _load_metadata(cfg: DatasetConfig) -> List[ExpressionRecord]:
    """Load metadata from CSV file containing dataset partitions."""
    metadata_path = os.path.join(cfg.root, "metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Could not locate metadata.csv for dataset '{cfg.name}'. "
            f"Expected at {metadata_path}. Please prepare the dataset according to "
            "the documentation."
        )

    records: List[ExpressionRecord] = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_columns = {"path", "label", "split"}
        missing = required_columns.difference(reader.fieldnames or set())
        if missing:
            raise ValueError(f"Metadata CSV missing columns: {missing}")

        known_class_ids = set(cfg.known_class_ids or [])
        for row in reader:
            image_path = os.path.join(cfg.root, row["path"])
            if not os.path.exists(image_path):
                LOGGER.warning("Image file missing: %s", image_path)
                continue

            label = int(row["label"]) if row["label"] != "" else -1
            is_known = label in known_class_ids if known_class_ids else label >= 0
            records.append(
                ExpressionRecord(
                    path=image_path, label=label, split=row["split"], is_known=is_known
                )
            )
    LOGGER.info(
        "Loaded %d samples for dataset '%s' with config %s",
        len(records),
        cfg.name,
        asdict(cfg),
    )
    return records


def _split_records(
    records: List[ExpressionRecord], split_ratio: float
) -> Tuple[List[ExpressionRecord], List[ExpressionRecord]]:
    """Split records into train and validation sets."""
    train_records: List[ExpressionRecord] = []
    val_records: List[ExpressionRecord] = []

    for record in records:
        if record.split.lower() == "train":
            train_records.append(record)
        elif record.split.lower() == "val":
            val_records.append(record)
        elif record.split.lower() == "test":
            val_records.append(record)

    if not train_records or not val_records:
        # fallback split based on ratio
        cutoff = int(len(records) * split_ratio)
        train_records = records[:cutoff]
        val_records = records[cutoff:]

    return train_records, val_records


def build_dataloaders(
    cfg: DatasetConfig,
    batch_size: int,
    num_workers: int,
) -> Dict[str, DataLoader]:
    """Create PyTorch dataloaders for train/val/test splits."""
    records = _load_metadata(cfg)
    train_records, val_records = _split_records(records, cfg.split_ratio)

    train_transform = _build_transforms(cfg, train=True)
    eval_transform = _build_transforms(cfg, train=False)

    train_known = [r for r in train_records if r.is_known]
    train_unlabeled = [r for r in train_records if not r.is_known]

    if cfg.unlabeled_ratio < 1.0 and train_unlabeled:
        cutoff = int(len(train_unlabeled) * cfg.unlabeled_ratio)
        train_unlabeled = train_unlabeled[:cutoff]

    train_dataset = ExpressionDataset(train_known + train_unlabeled, train_transform)
    val_dataset = ExpressionDataset(val_records, eval_transform)

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
        ),
    }
    return dataloaders


def build_inference_dataset(
    cfg: DatasetConfig, metadata_records: Iterable[Dict[str, str]]
) -> ExpressionDataset:
    """Create dataset from arbitrary metadata records for inference."""
    records: List[ExpressionRecord] = []
    for row in metadata_records:
        path = row["path"]
        label = int(row.get("label", "-1"))
        split = row.get("split", "inference")
        is_known = bool(int(row.get("is_known", "0")))
        records.append(ExpressionRecord(path, label, split, is_known))

    transform = _build_transforms(cfg, train=False)
    return ExpressionDataset(records, transform)


def subset_dataset(dataset: Dataset, indices: List[int]) -> Subset:
    """Return a subset of a dataset for flexible sampling."""
    return Subset(dataset, indices)
