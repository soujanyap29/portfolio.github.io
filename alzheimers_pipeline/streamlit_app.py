import json
from io import BytesIO
from typing import Dict

import matplotlib.cm as cm
import numpy as np
import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image

from .config import MODEL_PATH
from .infer import load_model_from_checkpoint, load_model_from_pt_with_meta


def preprocess(image: Image.Image):
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)


@st.cache_resource
def load_checkpoint(use_pt: bool):
    if use_pt:
        return load_model_from_pt_with_meta()
    return load_model_from_checkpoint()


def predict(model, tensor, class_to_index: Dict[str, int]):
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx_to_class = {v: k for k, v in class_to_index.items()}
    pred_idx = int(np.argmax(probs))
    return idx_to_class[pred_idx], probs, idx_to_class


def _target_layer(model):
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    if hasattr(model, "features"):
        return model.features[-1]
    return None


def gradcam_overlay(model, input_tensor, class_idx: int, raw_img: Image.Image):
    target = _target_layer(model)
    if target is None:
        return None
    feats = {}
    grads = {}

    def fwd_hook(_, __, output):
        feats["value"] = output

    def bwd_hook(_, _grad_input, grad_output):
        grads["value"] = grad_output[0]

    handle_fwd = target.register_forward_hook(fwd_hook)
    handle_bwd = target.register_full_backward_hook(bwd_hook)
    model.zero_grad(set_to_none=True)
    logits = model(input_tensor)
    score = logits[:, class_idx].sum()
    score.backward()
    handle_fwd.remove()
    handle_bwd.remove()
    if "value" not in feats or "value" not in grads:
        return None
    fmap = feats["value"].detach().cpu()[0]
    grad = grads["value"].detach().cpu()[0]
    weights = grad.mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((weights * fmap).sum(dim=0)).numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(raw_img.size)
    cam_np = np.array(cam_img) / 255.0
    color = cm.jet(cam_np)[..., :3]
    base = np.array(raw_img).astype(np.float32) / 255.0
    overlay = np.clip(0.6 * base + 0.4 * color, 0, 1)
    return (overlay * 255).astype(np.uint8)


def main():
    st.set_page_config(page_title="Alzheimer MRI Classifier", layout="wide")
    st.title("Alzheimer's Disease MRI Classifier")
    st.write("Upload one or more MRI images to predict dementia stage.")
    use_pt = st.checkbox("Load alzheimers_model.pt", value=True)
    model, class_to_index, model_name = load_checkpoint(use_pt=use_pt)
    st.sidebar.header("Model Information")
    st.sidebar.write(f"Architecture: {model_name}")
    st.sidebar.write(f"Classes: {list(class_to_index.keys())}")
    st.sidebar.write(f"Model file: {MODEL_PATH}")

    uploaded = st.file_uploader("Upload MRI image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if not uploaded:
        st.stop()

    for file in uploaded:
        image = Image.open(BytesIO(file.read())).convert("RGB")
        tensor = preprocess(image)
        pred_class, probs, idx_to_class = predict(model, tensor, class_to_index)
        pred_idx = int(np.argmax(probs))
        scores = {idx_to_class[i]: float(probs[i]) for i in range(len(probs))}
        st.subheader(file.name)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Input MRI", use_container_width=True)
        with col2:
            st.metric("Predicted Class", pred_class)
            st.write("Confidence Scores")
            st.json(scores)
        overlay = gradcam_overlay(model, tensor, pred_idx, image)
        if overlay is not None:
            st.image(overlay, caption="Grad-CAM Explanation", use_container_width=True)
        else:
            st.info("Visual explanation is available for CNN backbones (EfficientNet/ResNet).")
    st.download_button(
        "Download model metadata",
        data=json.dumps({"model_name": model_name, "class_to_index": class_to_index}, indent=2),
        file_name="model_info.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
