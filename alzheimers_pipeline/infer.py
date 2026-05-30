import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import torchvision.transforms as T
from PIL import Image

from .config import BEST_CHECKPOINT_PATH, CHECKPOINT_DIR, MODEL_PATH
from .train import create_model


def _allowed_checkpoint_path(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    allowed_root = CHECKPOINT_DIR.resolve()
    if resolved.suffix.lower() not in {".pt", ".pth", ".bin", ".json"}:
        raise ValueError(f"Unsupported file extension: {resolved.suffix}")
    if allowed_root not in resolved.parents:
        raise ValueError(f"Checkpoint path must be inside {allowed_root}")
    return resolved


def _safe_torch_load(path: Path):
    resolved = _allowed_checkpoint_path(path)
    try:
        return torch.load(resolved, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(resolved, map_location="cpu")


def load_model_from_checkpoint():
    checkpoint = _safe_torch_load(BEST_CHECKPOINT_PATH)
    class_to_index = checkpoint["class_to_index"]
    model = create_model(checkpoint["model_name"], num_classes=len(class_to_index))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, class_to_index, checkpoint["model_name"]


def load_model_from_state_dict(model_name: str, class_to_index: Dict[str, int]):
    model = create_model(model_name=model_name, num_classes=len(class_to_index))
    state_dict = _safe_torch_load(MODEL_PATH)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_model_from_pt_with_meta():
    resolved_meta = _allowed_checkpoint_path(CHECKPOINT_DIR / "model_meta.json")
    with resolved_meta.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    class_to_index = meta["class_to_index"]
    model_name = meta["model_name"]
    model = load_model_from_state_dict(model_name, class_to_index)
    return model, class_to_index, model_name


def predict_image(model, image_path: Path, class_to_index: Dict[str, int], device: torch.device):
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    index_to_class = {v: k for k, v in class_to_index.items()}
    pred_index = int(probs.argmax())
    result = {
        "predicted_class": index_to_class[pred_index],
        "confidence": float(probs[pred_index]),
        "scores": {index_to_class[i]: float(probs[i]) for i in range(len(probs))},
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Inference for Alzheimer MRI classifier.")
    parser.add_argument("--image", type=str, required=True, help="Path to MRI image")
    parser.add_argument("--use-pt", action="store_true", help="Load alzheimers_model.pt with model_meta.json")
    args = parser.parse_args()

    if args.use_pt:
        model, class_to_index, model_name = load_model_from_pt_with_meta()
    else:
        model, class_to_index, model_name = load_model_from_checkpoint()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prediction = predict_image(model, Path(args.image), class_to_index, device)
    output = {"model_name": model_name, "model_state_dict_path": str(MODEL_PATH), **prediction}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
