from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import Config, load_config
from src.dataset import ProteinLocalizationDataset, load_dataset, split_dataset
from src.image_preprocessing import build_image_transforms
from src.fusion_model import MultimodalLocalizationModel
from src.trainer import Trainer
from src.evaluate import evaluate_localization
from src.utils import ensure_dirs, seed_everything


def main(config_path: str = "config.yaml"):
    cfg: Config = load_config(config_path)
    ensure_dirs(cfg.paths.output_root)
    seed_everything(cfg.training.seed)

    df = load_dataset(cfg.paths.dataset_csv)
    split = split_dataset(df, val_split=cfg.training.val_split, seed=cfg.training.seed)

    train_ds = ProteinLocalizationDataset(
        split.train_df,
        image_root=cfg.paths.image_root,
        transform=build_image_transforms(train=True),
        esm_cache_dir=str(Path(cfg.paths.output_root) / "esm_cache"),
    )
    val_ds = ProteinLocalizationDataset(
        split.val_df,
        image_root=cfg.paths.image_root,
        transform=build_image_transforms(train=False),
        esm_cache_dir=str(Path(cfg.paths.output_root) / "esm_cache"),
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers)

    has_ad_labels = "alzheimers_label" in df.columns
    model = MultimodalLocalizationModel(
        image_backbone=cfg.model.image_backbone,
        image_emb_dim=cfg.model.image_emb_dim,
        seq_emb_dim=cfg.model.seq_emb_dim,
        fusion_dim=cfg.model.fusion_dim,
        fusion_type=cfg.model.fusion_type,
        seq_model=cfg.model.seq_model,
        has_alzheimers_labels=has_ad_labels,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(model=model, config=cfg, device=device)
    trainer.fit(train_loader, val_loader)

    best_model = Path(cfg.paths.output_root) / "checkpoints" / "best_model.pt"
    model.load_state_dict(torch.load(best_model, map_location=device))
    metrics = evaluate_localization(model, val_loader, device=device, output_dir=str(Path(cfg.paths.output_root) / "evaluation"))
    print(metrics)


if __name__ == "__main__":
    main()
