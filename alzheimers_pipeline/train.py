import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
    efficientnet_b0,
    resnet50,
    vit_b_16,
)
from tqdm import tqdm

from .config import (
    BEST_CHECKPOINT_PATH,
    CHECKPOINT_DIR,
    LAST_CHECKPOINT_PATH,
    METRICS_DIR,
    MODEL_PATH,
    PLOTS_DIR,
    SPLIT_DIR,
    TENSORBOARD_DIR,
)
from .dataset_utils import AlzheimerImageDataset, ensure_dir, load_splits


def build_transforms(image_size: int = 224):
    train_transform = T.Compose(
        [
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def create_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "efficientnet_b0":
        try:
            model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        except Exception:
            model = efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
    if model_name == "resnet50":
        try:
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        except Exception:
            model = resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    if model_name == "vit_b_16":
        try:
            model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        except Exception:
            model = vit_b_16(weights=None)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model: {model_name}")


def make_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int,
    workers: int,
):
    train_tf, eval_tf = build_transforms()
    train_ds = AlzheimerImageDataset(train_df, transform=train_tf)
    val_ds = AlzheimerImageDataset(val_df, transform=eval_tf)
    test_ds = AlzheimerImageDataset(test_df, transform=eval_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def compute_scores(y_true, y_pred, y_prob, num_classes: int):
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }
    try:
        if num_classes > 2:
            result["roc_auc_ovr"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        else:
            result["roc_auc_ovr"] = float(roc_auc_score(y_true, y_prob[:, 1]))
    except Exception:
        result["roc_auc_ovr"] = float("nan")
    return result


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: torch.amp.GradScaler,
    device: torch.device,
    train: bool = True,
) -> Tuple[float, Dict[str, float]]:
    model.train(mode=train)
    losses: List[float] = []
    all_y, all_pred, all_prob = [], [], []
    autocast_enabled = device.type == "cuda"
    iterator = tqdm(loader, leave=False)
    for x, y in iterator:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
            logits = model(x)
            loss = criterion(logits, y)
        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        losses.append(loss.item())
        prob = torch.softmax(logits.detach(), dim=1).cpu().numpy()
        pred = np.argmax(prob, axis=1)
        all_prob.append(prob)
        all_pred.extend(pred.tolist())
        all_y.extend(y.cpu().numpy().tolist())
    y_prob = np.concatenate(all_prob, axis=0)
    metrics = compute_scores(all_y, all_pred, y_prob, num_classes=y_prob.shape[1])
    return float(np.mean(losses)), metrics


def save_history_plots(history: Dict[str, List[float]], out_dir: Path):
    ensure_dir(out_dir)
    for metric in ["loss", "accuracy", "f1_macro", "roc_auc_ovr"]:
        plt.figure(figsize=(7, 5))
        plt.plot(history[f"train_{metric}"], label=f"train_{metric}")
        plt.plot(history[f"val_{metric}"], label=f"val_{metric}")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}.png")
        plt.close()


def train_single_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    epochs: int,
    lr: float,
    patience: int,
    device: torch.device,
):
    model = create_model(model_name, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    writer = SummaryWriter(str(TENSORBOARD_DIR / model_name))
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "train_f1_macro": [],
        "train_roc_auc_ovr": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1_macro": [],
        "val_roc_auc_ovr": [],
    }
    best_state = deepcopy(model.state_dict())
    best_score = -np.inf
    bad_epochs = 0
    for epoch in range(1, epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, optimizer, scaler, device, train=False)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        for key, value in train_metrics.items():
            history[f"train_{key}"].append(value)
        for key, value in val_metrics.items():
            history[f"val_{key}"].append(value)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("f1_macro/train", train_metrics["f1_macro"], epoch)
        writer.add_scalar("f1_macro/val", val_metrics["f1_macro"], epoch)
        scheduler.step(val_metrics["f1_macro"])
        if val_metrics["f1_macro"] > best_score:
            best_score = val_metrics["f1_macro"]
            best_state = deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    writer.close()
    model.load_state_dict(best_state)
    return model, history, float(best_score)


def evaluate_on_test(model: nn.Module, loader: DataLoader, device: torch.device):
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=False)
    loss, metrics = run_epoch(model, loader, criterion, optimizer=None, scaler=scaler, device=device, train=False)
    metrics["loss"] = loss
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Alzheimer MRI classifier with model comparison.")
    parser.add_argument("--models", nargs="+", default=["efficientnet_b0", "resnet50", "vit_b_16"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    ensure_dir(CHECKPOINT_DIR)
    ensure_dir(PLOTS_DIR)
    ensure_dir(METRICS_DIR)
    ensure_dir(TENSORBOARD_DIR)

    train_df, val_df, test_df, class_to_index = load_splits(SPLIT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = make_loaders(
        train_df, val_df, test_df, batch_size=args.batch_size, workers=args.num_workers
    )
    num_classes = len(class_to_index)

    all_results = {}
    best_model_name = None
    best_model = None
    best_history = None
    best_score = -np.inf

    for model_name in args.models:
        model, history, val_score = train_single_model(
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            device=device,
        )
        test_metrics = evaluate_on_test(model, test_loader, device)
        all_results[model_name] = {"best_val_f1": val_score, "test_metrics": test_metrics}
        if val_score > best_score:
            best_score = val_score
            best_model_name = model_name
            best_model = model
            best_history = history

    torch.save(best_model.state_dict(), MODEL_PATH)
    torch.save(
        {
            "model_name": best_model_name,
            "state_dict": best_model.state_dict(),
            "class_to_index": class_to_index,
        },
        BEST_CHECKPOINT_PATH,
    )
    torch.save(
        {
            "all_results": all_results,
            "best_model_name": best_model_name,
            "class_to_index": class_to_index,
        },
        LAST_CHECKPOINT_PATH,
    )
    save_history_plots(best_history, PLOTS_DIR / "training_curves")
    with (METRICS_DIR / "model_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    with (CHECKPOINT_DIR / "model_meta.json").open("w", encoding="utf-8") as f:
        json.dump({"model_name": best_model_name, "class_to_index": class_to_index}, f, indent=2)
    summary = {
        "recommended_model": best_model_name,
        "selection_reason": "Highest validation macro-F1 across compared transfer-learning architectures.",
        "saved_model_path": str(MODEL_PATH),
    }
    with (METRICS_DIR / "recommended_model.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
