from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import tifffile
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.config import LOCALIZATION_LABELS
from src.image_preprocessing import ensure_three_channels


@dataclass
class SplitData:
    train_df: pd.DataFrame
    val_df: pd.DataFrame


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = set(["sequence", "image_path", "gene_name"] + LOCALIZATION_LABELS)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def split_dataset(df: pd.DataFrame, val_split: float = 0.2, seed: int = 42) -> SplitData:
    train_df, val_df = train_test_split(df, test_size=val_split, random_state=seed, shuffle=True)
    return SplitData(train_df=train_df.reset_index(drop=True), val_df=val_df.reset_index(drop=True))


class ProteinLocalizationDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_root: str,
        transform=None,
        esm_cache_dir: Optional[str] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform
        self.esm_cache_dir = Path(esm_cache_dir) if esm_cache_dir else None

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, image_path: str) -> Path:
        p = Path(image_path)
        if p.is_absolute() and p.exists():
            return p
        p2 = self.image_root / image_path
        if p2.exists():
            return p2
        return p

    def _read_image(self, path: Path) -> np.ndarray:
        if path.suffix.lower() in {".tif", ".tiff"}:
            img = tifffile.imread(str(path))
        else:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Unable to read image: {path}")
        img = ensure_three_channels(img)
        if img.dtype != np.uint8:
            img_min, img_max = float(img.min()), float(img.max())
            if img_max <= img_min:
                img = np.zeros_like(img, dtype=np.uint8)
            else:
                img = ((img - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
        return img

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        image_path = self._resolve_image_path(str(row["image_path"]))
        image = self._read_image(image_path)
        image_tensor = self.transform(image) if self.transform else torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        labels = torch.tensor(row[LOCALIZATION_LABELS].values.astype(np.float32), dtype=torch.float32)

        item: Dict[str, torch.Tensor] = {
            "image": image_tensor,
            "labels": labels,
            "sequence": row["sequence"],
            "gene_name": row.get("gene_name", ""),
            "uniprot_id": row.get("uniprot_id", ""),
        }

        if self.esm_cache_dir is not None:
            uid = hashlib.sha256(str(row['sequence']).encode('utf-8')).hexdigest()
            cache_file = self.esm_cache_dir / f"{uid}.npy"
            if cache_file.exists():
                item["seq_embedding"] = torch.tensor(np.load(cache_file), dtype=torch.float32)

        if "alzheimers_label" in row.index:
            item["alzheimers_label"] = torch.tensor(float(row["alzheimers_label"]), dtype=torch.float32)
        return item
