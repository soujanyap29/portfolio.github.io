from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import tifffile
import torch

from src.config import LOCALIZATION_LABELS
from src.image_preprocessing import build_image_transforms, ensure_three_channels
from src.utils import GradCAM


def load_image(image_path: str, image_size: int = 224) -> torch.Tensor:
    p = Path(image_path)
    image = tifffile.imread(str(p)) if p.suffix.lower() in {".tif", ".tiff"} else cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    image = ensure_three_channels(image)
    image = image.astype(np.uint8) if image.dtype != np.uint8 else image
    transform = build_image_transforms(image_size=image_size, train=False)
    return transform(image).unsqueeze(0)


def run_inference(model, image_tensor, sequence: str, seq_embedding: torch.Tensor | None, device: str) -> Dict:
    model.eval()
    image_tensor = image_tensor.to(device)
    seq_emb = seq_embedding.to(device) if seq_embedding is not None else None

    with torch.no_grad():
        out = model(images=image_tensor, sequences=[sequence], seq_embedding=seq_emb)
        loc_prob = torch.sigmoid(out["localization_logits"]).cpu().numpy()[0]

    predictions = {label: float(loc_prob[i]) for i, label in enumerate(LOCALIZATION_LABELS)}
    alz_prob = float(torch.sigmoid(out["alzheimers_logits"]).cpu().numpy().squeeze())
    return {
        "localization_probabilities": predictions,
        "alzheimers_probability": alz_prob,
    }


def generate_gradcam(model, image_tensor: torch.Tensor, sequence: str, seq_embedding: torch.Tensor | None, out_path: str, device: str):
    model.eval()
    image_tensor = image_tensor.to(device)
    seq_emb = seq_embedding.to(device) if seq_embedding is not None else None

    target_layer = model.image_encoder.encoder.conv_head if hasattr(model.image_encoder.encoder, "conv_head") else list(model.image_encoder.encoder.children())[-1]
    gradcam = GradCAM(model, target_layer)

    out = model(images=image_tensor, sequences=[sequence], seq_embedding=seq_emb)
    score = out["localization_logits"][0].max()
    cam = gradcam.generate(score)[0, 0].detach().cpu().numpy()

    cam_img = (cam * 255).astype(np.uint8)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, cam_img)
    return out_path
