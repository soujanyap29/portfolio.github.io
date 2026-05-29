from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.config import LOCALIZATION_LABELS, load_config
from src.fusion_model import MultimodalLocalizationModel
from src.inference import generate_gradcam, load_image, run_inference

st.set_page_config(page_title="Protein Localization & Alzheimer's AI", page_icon="🧬", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 1.2rem;}
.metric-card {background: linear-gradient(145deg,#101820,#20313f); padding:12px; border-radius:12px; color:white;}
</style>
""",
    unsafe_allow_html=True,
)

cfg = load_config("config.yaml")
output_root = Path(cfg.paths.output_root)
history_csv = output_root / "predictions" / "prediction_history.csv"
history_csv.parent.mkdir(parents=True, exist_ok=True)

@st.cache_resource
def load_model(checkpoint_path: str):
    model = MultimodalLocalizationModel(
        image_backbone=cfg.model.image_backbone,
        image_emb_dim=cfg.model.image_emb_dim,
        seq_emb_dim=cfg.model.seq_emb_dim,
        fusion_dim=cfg.model.fusion_dim,
        fusion_type=cfg.model.fusion_type,
        seq_model=cfg.model.seq_model,
        has_alzheimers_labels=False,
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Inference", "Dashboard", "History", "Documentation"])
checkpoint = st.sidebar.text_input("Checkpoint path", value=str(output_root / "checkpoints" / "best_model.pt"))

if section == "Inference":
    st.title("Protein Subcellular Localization + Alzheimer's Association")
    c1, c2 = st.columns([1, 1])
    with c1:
        img_file = st.file_uploader("Upload TIFF microscopy image", type=["tif", "tiff"])
        sequence = st.text_area("Protein sequence", height=180)
    with c2:
        run_btn = st.button("Run Prediction", type="primary")

    if run_btn and img_file and sequence.strip():
        tmp_img = output_root / "temp_uploaded.tif"
        tmp_img.write_bytes(img_file.read())
        model = load_model(checkpoint)
        image_tensor = load_image(str(tmp_img))
        result = run_inference(model, image_tensor=image_tensor, sequence=sequence.strip(), seq_embedding=None, device="cpu")

        tabs = st.tabs(["Localization", "Alzheimer's", "Grad-CAM", "Export"])
        with tabs[0]:
            df = pd.DataFrame({"label": list(result["localization_probabilities"].keys()), "probability": list(result["localization_probabilities"].values())})
            st.bar_chart(df.set_index("label"))
        with tabs[1]:
            st.metric("Alzheimer's association probability", f"{result['alzheimers_probability']:.4f}")
        with tabs[2]:
            gradcam_path = output_root / "gradcam" / f"gradcam_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
            generate_gradcam(model, image_tensor=image_tensor, sequence=sequence.strip(), seq_embedding=None, out_path=str(gradcam_path), device="cpu")
            st.image(str(gradcam_path), caption="Grad-CAM")
        with tabs[3]:
            rec = {
                "timestamp": datetime.utcnow().isoformat(),
                **result["localization_probabilities"],
                "alzheimers_probability": result["alzheimers_probability"],
            }
            row = pd.DataFrame([rec])
            if history_csv.exists():
                row.to_csv(history_csv, mode="a", index=False, header=False)
            else:
                row.to_csv(history_csv, index=False)
            st.download_button("Download latest prediction JSON", data=json.dumps(rec, indent=2), file_name="prediction.json")

elif section == "Dashboard":
    st.title("Model Dashboard")
    eval_path = output_root / "evaluation" / "evaluation_metrics.json"
    if eval_path.exists():
        metrics = json.loads(eval_path.read_text())
        cols = st.columns(5)
        for idx, key in enumerate(["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_macro"]):
            cols[idx].markdown(f"<div class='metric-card'><h4>{key}</h4><p>{metrics.get(key,0):.4f}</p></div>", unsafe_allow_html=True)
    else:
        st.info("Run training/evaluation to populate dashboard metrics.")

elif section == "History":
    st.title("Prediction History")
    if history_csv.exists():
        hist = pd.read_csv(history_csv)
        st.dataframe(hist, use_container_width=True)
        st.download_button("Download history CSV", data=history_csv.read_bytes(), file_name="prediction_history.csv")
    else:
        st.warning("No history found.")

else:
    st.title("Project Documentation")
    st.markdown("""
- Multimodal model combines microscopy image embeddings and protein-sequence embeddings.
- Supports attention fusion and concatenation fusion.
- Includes placeholder Alzheimer's module for future supervised disease labels.
""")
