import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .config import CLASS_ALIASES

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LABEL_CANDIDATES = ["label", "class", "diagnosis", "alzheimer", "dementia", "target"]
IMAGE_CANDIDATES = ["image", "filepath", "file_path", "path", "img", "filename"]


@dataclass
class DatasetPreparationResult:
    source: str
    train_csv: Path
    val_csv: Path
    test_csv: Path
    class_to_index: Dict[str, int]


class AlzheimerImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform=None):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label_index"])


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _find_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for key in lower_map:
        for candidate in candidates:
            if candidate in key:
                return lower_map[key]
    return None


def _normalize_class_name(label: str) -> str:
    compact = str(label).strip().replace(" ", "")
    return CLASS_ALIASES.get(compact, str(label).strip())


def extract_valid_alzheimer_records_from_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    if not csv_path.exists():
        return None
    frame = pd.read_csv(csv_path)
    label_col = _find_column(frame.columns.tolist(), LABEL_CANDIDATES)
    image_col = _find_column(frame.columns.tolist(), IMAGE_CANDIDATES)
    if label_col is None or image_col is None:
        return None
    data = frame[[image_col, label_col]].copy()
    data.columns = ["image_path", "label"]
    data["image_path"] = data["image_path"].astype(str)
    csv_parent = csv_path.parent
    data["image_path"] = data["image_path"].apply(
        lambda p: str((csv_parent / p).resolve()) if not Path(p).is_absolute() else p
    )
    data["label"] = data["label"].astype(str).map(_normalize_class_name)
    data = data[data["image_path"].map(lambda p: Path(p).exists() and Path(p).suffix.lower() in IMAGE_EXTENSIONS)]
    if data["label"].nunique() < 2 or len(data) < 100:
        return None
    return data.reset_index(drop=True)


def download_kaggle_dataset(dataset_slug: str, output_dir: Path) -> None:
    ensure_dir(output_dir)
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset_slug,
        "-p",
        str(output_dir),
        "--unzip",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Kaggle dataset download failed. Configure Kaggle API credentials and retry.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr: {result.stderr.strip()}"
        )


def build_records_from_image_folders(root_dir: Path) -> pd.DataFrame:
    records = []
    for path in root_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            label = _normalize_class_name(path.parent.name)
            records.append({"image_path": str(path.resolve()), "label": label})
    if not records:
        raise RuntimeError(f"No MRI images found in {root_dir}")
    data = pd.DataFrame(records).drop_duplicates()
    counts = data["label"].value_counts()
    valid_labels = counts[counts >= 2].index.tolist()
    data = data[data["label"].isin(valid_labels)].reset_index(drop=True)
    return data


def create_stratified_splits(
    data: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, temp = train_test_split(
        data,
        train_size=train_ratio,
        stratify=data["label"],
        random_state=random_state,
    )
    val_size = val_ratio / (1 - train_ratio)
    val, test = train_test_split(
        temp,
        train_size=val_size,
        stratify=temp["label"],
        random_state=random_state,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def encode_labels(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    classes = sorted(train_df["label"].unique().tolist())
    class_to_index = {label: idx for idx, label in enumerate(classes)}
    for frame in (train_df, val_df, test_df):
        frame["label_index"] = frame["label"].map(class_to_index).astype(int)
    return train_df, val_df, test_df, class_to_index


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_to_index: Dict[str, int],
    split_dir: Path,
) -> Tuple[Path, Path, Path]:
    ensure_dir(split_dir)
    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"
    test_csv = split_dir / "test.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    with (split_dir / "class_to_index.json").open("w", encoding="utf-8") as f:
        json.dump(class_to_index, f, indent=2)
    return train_csv, val_csv, test_csv


def load_splits(split_dir: Path):
    train_df = pd.read_csv(split_dir / "train.csv")
    val_df = pd.read_csv(split_dir / "val.csv")
    test_df = pd.read_csv(split_dir / "test.csv")
    with (split_dir / "class_to_index.json").open("r", encoding="utf-8") as f:
        class_to_index = json.load(f)
    return train_df, val_df, test_df, class_to_index

