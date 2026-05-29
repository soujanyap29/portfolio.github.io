from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils import multilabel_metrics, save_json


class Trainer:
    def __init__(self, model, config, device: str):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5, patience=2)
        self.scaler = GradScaler(enabled=config.training.amp and device.startswith("cuda"))
        self.writer = SummaryWriter(log_dir=str(Path(config.paths.output_root) / "logs"))

    def _run_epoch(self, loader, train: bool = True) -> Dict[str, float]:
        self.model.train(train)
        total_loss = 0.0
        all_true, all_prob = [], []

        pbar = tqdm(loader, desc="train" if train else "val")
        for batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["labels"].to(self.device)
            seq_emb = batch.get("seq_embedding")
            if seq_emb is not None:
                seq_emb = seq_emb.to(self.device)
            sequences = batch["sequence"]

            with torch.set_grad_enabled(train):
                with autocast(enabled=self.scaler.is_enabled()):
                    out = self.model(images=images, sequences=sequences, seq_embedding=seq_emb)
                    loss = self.criterion(out["localization_logits"], labels)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            total_loss += loss.item() * labels.size(0)
            probs = torch.sigmoid(out["localization_logits"]).detach().cpu().numpy()
            all_prob.append(probs)
            all_true.append(labels.detach().cpu().numpy())
            pbar.set_postfix(loss=loss.item())

        y_true = np.concatenate(all_true, axis=0)
        y_prob = np.concatenate(all_prob, axis=0)
        m = multilabel_metrics(y_true, y_prob)
        m["loss"] = total_loss / len(loader.dataset)
        return m

    def fit(self, train_loader, val_loader):
        output_dir = Path(self.config.paths.output_root)
        ckpt_dir = output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_f1 = -1.0
        patience = 0
        history = {"train": [], "val": []}

        for epoch in range(1, self.config.training.epochs + 1):
            train_m = self._run_epoch(train_loader, train=True)
            val_m = self._run_epoch(val_loader, train=False)
            self.scheduler.step(val_m["f1_macro"])

            history["train"].append(train_m)
            history["val"].append(val_m)
            self.writer.add_scalar("train/loss", train_m["loss"], epoch)
            self.writer.add_scalar("val/loss", val_m["loss"], epoch)
            self.writer.add_scalar("val/f1_macro", val_m["f1_macro"], epoch)

            if val_m["f1_macro"] > best_f1:
                best_f1 = val_m["f1_macro"]
                patience = 0
                torch.save(self.model.state_dict(), ckpt_dir / "best_model.pt")
            else:
                patience += 1

            torch.save(self.model.state_dict(), ckpt_dir / "last_model.pt")
            if patience >= self.config.training.early_stopping_patience:
                break

        save_json(history, output_dir / "training_history.json")
        return history
