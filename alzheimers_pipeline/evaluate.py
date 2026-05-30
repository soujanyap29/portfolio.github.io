import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.transforms as T
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from .config import BEST_CHECKPOINT_PATH, METRICS_DIR, PLOTS_DIR, SPLIT_DIR
from .dataset_utils import AlzheimerImageDataset, ensure_dir, load_splits
from .train import create_model


def build_eval_loader(test_df: pd.DataFrame, batch_size: int, workers: int):
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = AlzheimerImageDataset(test_df, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_logits, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()
    probs = torch.softmax(logits, dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    metrics = {
        "loss": float(total_loss / len(labels)),
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
    }
    if probs.shape[1] > 2:
        metrics["roc_auc_ovr"] = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    else:
        metrics["roc_auc_ovr"] = float(roc_auc_score(labels, probs[:, 1]))
    return labels, preds, probs, metrics


def plot_confusion(labels, preds, class_names, output_path: Path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_roc(labels, probs, class_names, output_path: Path):
    y_bin = label_binarize(labels, classes=list(range(len(class_names))))
    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        auc_i = roc_auc_score(y_bin[:, i], probs[:, i])
        plt.plot(fpr, tpr, label=f"{class_name} (AUC={auc_i:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Alzheimer model.")
    parser.add_argument("--checkpoint", type=str, default=str(BEST_CHECKPOINT_PATH))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    ensure_dir(METRICS_DIR)
    ensure_dir(PLOTS_DIR)
    train_df, val_df, test_df, class_to_index = load_splits(SPLIT_DIR)
    index_to_class = {v: k for k, v in class_to_index.items()}
    class_names = [index_to_class[i] for i in range(len(index_to_class))]
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_name = checkpoint["model_name"]
    model = create_model(model_name, num_classes=len(class_to_index))
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loader = build_eval_loader(test_df, args.batch_size, args.num_workers)
    labels, preds, probs, metrics = evaluate(model, loader, device)
    plot_confusion(labels, preds, class_names, PLOTS_DIR / "confusion_matrix.png")
    plot_roc(labels, probs, class_names, PLOTS_DIR / "roc_curves.png")
    report = classification_report(labels, preds, target_names=class_names, output_dict=True)

    with (METRICS_DIR / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (METRICS_DIR / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

