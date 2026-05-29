from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc

from src.config import LOCALIZATION_LABELS
from src.utils import multilabel_metrics, save_json


@torch.no_grad()
def evaluate_localization(model, loader, device: str, output_dir: str) -> Dict[str, float]:
    model.eval()
    y_true, y_prob = [], []
    for batch in loader:
        images = batch["image"].to(device)
        seq_emb = batch.get("seq_embedding")
        if seq_emb is not None:
            seq_emb = seq_emb.to(device)
        out = model(images=images, sequences=batch["sequence"], seq_embedding=seq_emb)
        probs = torch.sigmoid(out["localization_logits"]).cpu().numpy()
        y_prob.append(probs)
        y_true.append(batch["labels"].numpy())

    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    metrics = multilabel_metrics(y_true, y_prob)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_json(metrics, out_path / "evaluation_metrics.json")

    for idx, label in enumerate(LOCALIZATION_LABELS):
        fpr, tpr, _ = roc_curve(y_true[:, idx], y_prob[:, idx])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.title(f"ROC - {label}")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path / f"roc_{label}.png")
        plt.close()

        cm = confusion_matrix(y_true[:, idx], (y_prob[:, idx] >= 0.5).astype(int))
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {label}")
        plt.tight_layout()
        plt.savefig(out_path / f"cm_{label}.png")
        plt.close()

    return metrics
