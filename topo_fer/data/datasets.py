from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ExpressionRecord:
    path: Path
    label: Optional[int]
    split: str
    metadata: Dict[str, Any]


class BaseFacialExpressionDataset(Dataset):
    """Base dataset class for facial expression recognition benchmarks."""

    def __init__(
        self,
        root: str | Path,
        annotation_path: str | Path,
        split: str,
        transform: Any | None = None,
        label_mapping: Optional[Dict[str, int]] = None,
        allow_unlabeled: bool = True,
    ) -> None:
        self.root = Path(root)
        self.annotation_path = Path(annotation_path)
        self.split = split
        self.transform = transform
        self.label_mapping = label_mapping or {}
        self.allow_unlabeled = allow_unlabeled
        self.records = self._load_records()
        if not self.records:
            raise ValueError(f"No samples found for split '{split}' in {annotation_path}")

    def _read_table(self) -> pd.DataFrame:
        ext = self.annotation_path.suffix.lower()
        if ext in {".csv"}:
            return pd.read_csv(self.annotation_path)
        if ext in {".tsv"}:
            return pd.read_csv(self.annotation_path, sep="\t")
        if ext in {".parquet"}:
            return pd.read_parquet(self.annotation_path)
        if ext in {".json"}:
            return pd.read_json(self.annotation_path)
        if ext in {".txt"}:
            return self._read_txt()
        raise ValueError(f"Unsupported annotation format: {ext}")

    def _read_txt(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        with self.annotation_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 2:
                    path_rel, label_str = parts
                    rows.append({"path": path_rel, "label": int(label_str)})
                elif len(parts) >= 3:
                    path_rel, label_str, split = parts[:3]
                    rows.append({"path": path_rel, "label": int(label_str), "split": split})
                else:
                    raise ValueError(f"Cannot parse line: {line}")
        df = pd.DataFrame(rows)
        if "split" not in df.columns:
            df["split"] = np.where(df["path"].str.contains("train"), "train", "test")
        return df

    def _normalize_split_name(self, value: str) -> str:
        value = value.lower()
        if value in {"train", "training"}:
            return "train"
        if value in {"val", "valid", "validation"}:
            return "val"
        if value in {"test", "testing"}:
            return "test"
        if value in {"unlabeled", "unlabelled", "novel"}:
            return "unlabeled"
        return value

    def _derive_split(self, df: pd.DataFrame) -> pd.DataFrame:
        if "split" in df.columns:
            df["split"] = df["split"].map(self._normalize_split_name)
            return df
        if "usage" in df.columns:
            df["split"] = df["usage"].map(self._normalize_split_name)
            return df
        if "set" in df.columns:
            df["split"] = df["set"].map(self._normalize_split_name)
            return df
        df["split"] = "train"
        return df

    def _load_records(self) -> List[ExpressionRecord]:
        table = self._read_table()
        table = self._derive_split(table)
        records: List[ExpressionRecord] = []
        for _, row in table.iterrows():
            split = self._normalize_split_name(str(row["split"]))
            if split != self.split:
                continue
            rel_path = Path(str(row["path"]))
            label_value: Optional[int]
            if "label" in row and not pd.isna(row["label"]):
                if isinstance(row["label"], str) and row["label"] in self.label_mapping:
                    label_value = self.label_mapping[row["label"]]
                elif isinstance(row["label"], str):
                    label_value = int(row["label"])
                else:
                    label_value = int(row["label"])
            else:
                label_value = None
                if not self.allow_unlabeled:
                    continue
            metadata = {k: row[k] for k in row.index if k not in {"path", "label", "split"}}
            records.append(ExpressionRecord(self.root / rel_path, label_value, split, metadata))
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        image = self._load_image(record.path)
        label = record.label if record.label is not None else -1
        sample = {"image": image, "label": label, "path": str(record.path), "metadata": record.metadata}
        if self.transform:
            augmented = self.transform(image=image)
            sample["image"] = augmented["image"]
        return sample

    def _load_image(self, path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


class RAFDBDataset(BaseFacialExpressionDataset):
    def _read_txt(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        with self.annotation_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path_rel, label_str = line.split()
                split = "train" if "train" in path_rel else "test"
                rows.append({"path": f"Image/aligned/{path_rel}.jpg", "label": int(label_str) - 1, "split": split})
        return pd.DataFrame(rows)


class FERPlusDataset(BaseFacialExpressionDataset):
    def _read_table(self) -> pd.DataFrame:
        df = super()._read_table()
        if "Usage" in df.columns:
            df["split"] = df["Usage"]
        if "FERPlusTraining" in df.columns:
            df = df.rename(columns={"FERPlusTraining": "path"})
        return df


class AffectNetDataset(BaseFacialExpressionDataset):
    def _read_table(self) -> pd.DataFrame:
        df = super()._read_table()
        for candidate in ("subDirectory_filePath", "path"):
            if candidate in df.columns:
                df["path"] = df[candidate]
                break
        if "objective" in df.columns:
            df["label"] = df["objective"].astype(int)
        return df


class OVMERDDataset(BaseFacialExpressionDataset):
    """Dataset wrapper for the Open-World MERD benchmark."""

    def _read_table(self) -> pd.DataFrame:
        df = super()._read_table()
        if "filepath" in df.columns:
            df = df.rename(columns={"filepath": "path"})
        return df


DATASET_REGISTRY = {
    "raf_db": RAFDBDataset,
    "ferplus": FERPlusDataset,
    "affectnet": AffectNetDataset,
    "ov_merd": OVMERDDataset,
}


def build_dataset(
    name: str,
    root: str | Path,
    annotation: str | Path,
    split: str,
    transform: Any | None = None,
    label_mapping: Optional[Dict[str, int]] = None,
) -> BaseFacialExpressionDataset:
    dataset_cls = DATASET_REGISTRY.get(name.lower())
    if dataset_cls is None:
        raise KeyError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}")
    return dataset_cls(
        root=root,
        annotation_path=annotation,
        split=split,
        transform=transform,
        label_mapping=label_mapping,
    )

