from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import yaml

LOCALIZATION_LABELS: List[str] = [
    "centrosome",
    "chromatin",
    "cytoplasmic",
    "cytoskeleton",
    "er",
    "focal_adhesions",
    "golgi",
    "membrane",
    "mitochondria",
    "nuclear_membrane",
    "nuclear_punctae",
    "nucleolus_fc_dfc",
    "nucleolus_gc",
    "nucleoplasm",
    "vesicles",
]


@dataclass
class PathsConfig:
    project_root: str = "/home/lab3/min/project"
    output_root: str = "/home/lab3/min/output"
    dataset_csv: str = "/home/lab3/min/merged_dataset.csv"
    image_root: str = "/home/lab3/min/images"


@dataclass
class TrainingConfig:
    seed: int = 42
    batch_size: int = 16
    num_workers: int = 4
    epochs: int = 30
    lr: float = 2e-4
    weight_decay: float = 1e-4
    val_split: float = 0.2
    early_stopping_patience: int = 7
    amp: bool = True


@dataclass
class ModelConfig:
    image_backbone: str = "efficientnet_b0"
    image_emb_dim: int = 512
    seq_model: str = "esm2"  # esm2 | bilstm
    esm_name: str = "facebook/esm2_t12_35M_UR50D"
    seq_emb_dim: int = 480
    fusion_type: str = "attention"  # concat | attention
    fusion_dim: int = 512
    dropout: float = 0.2


@dataclass
class AppConfig:
    device: str = "cuda"


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    app: AppConfig = field(default_factory=AppConfig)


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(
        paths=PathsConfig(**data.get("paths", {})),
        training=TrainingConfig(**data.get("training", {})),
        model=ModelConfig(**data.get("model", {})),
        app=AppConfig(**data.get("app", {})),
    )
