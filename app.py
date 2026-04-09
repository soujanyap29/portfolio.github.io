"""
🔬 Multimodal Protein Localization Predictor
=============================================
A Streamlit application that predicts protein subcellular localization
using both protein images (TIFF) and amino acid sequences together.

Architecture:
  - Image Branch  : CNN (ResNet-18 backbone) -> feature vector
  - Sequence Branch: Embedding -> BiGRU -> Attention -> feature vector
  - Fusion Layer  : Concatenate(image_features, seq_features) -> FC -> Sigmoid

Usage:
  streamlit run app.py

Dependencies:
  pip install streamlit torch torchvision pandas numpy pillow matplotlib tifffile scikit-learn
"""

import os
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# ---------- PIL ----------
from PIL import Image

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False

# ---------- PyTorch ----------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ---------- Sklearn ----------
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# =============================================================================
# CONSTANTS
# =============================================================================

LABEL_COLUMNS = [
    "Cytoplasm", "Nucleus", "Mitochondria", "Endoplasmic Reticulum",
    "Golgi Apparatus", "Lysosome", "Peroxisome", "Plasma Membrane",
    "Cell Junction", "Cytoskeleton", "Nucleolus", "Nuclear Membrane",
    "Vesicle", "Centrosome", "Lipid Droplet", "Aggresome", "Microtubule",
]
NUM_LABELS = len(LABEL_COLUMNS)

IMG_SIZE    = 224
MAX_SEQ_LEN = 1000
EMBED_DIM   = 64
HIDDEN_DIM  = 128
CNN_OUT_DIM = 256
SEQ_OUT_DIM = 256

DEFAULT_IMAGE_DIR = "/home/soujanya/opencell_project/images"
DEFAULT_CSV_PATH  = "/home/soujanya/opencell_project/data/opencell_multimodal_with_sequences.csv"

# Amino-acid vocabulary (20 canonical + <PAD> + <UNK>)
AA_VOCAB = {aa: idx + 2 for idx, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
AA_VOCAB["<PAD>"] = 0
AA_VOCAB["<UNK>"] = 1
VOCAB_SIZE = len(AA_VOCAB)  # 22

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# DATA UTILITIES
# =============================================================================

def load_tiff_image(path: str) -> np.ndarray:
    """Load a TIFF file and return a uint8 RGB numpy array (H, W, 3)."""
    if TIFFFILE_AVAILABLE:
        arr = tifffile.imread(path)
    else:
        arr = np.array(Image.open(path))

    # Normalise dimensions to (H, W, C)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[0] <= 4:  # (C,H,W)
        arr = np.transpose(arr, (1, 2, 0))

    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    # Rescale to [0, 255] uint8
    arr = arr.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    return (arr * 255).astype(np.uint8)


def load_protein_images(image_dir: str, gene_name: str):
    """Return list of RGB uint8 arrays for all TIFFs of a gene folder."""
    gene_path = Path(image_dir) / gene_name
    images = []
    if gene_path.exists():
        tif_files = sorted(gene_path.glob("*.tif")) + sorted(gene_path.glob("*.tiff"))
        for fp in tif_files:
            try:
                images.append(load_tiff_image(str(fp)))
            except Exception:
                pass
    return images


def average_protein_images(images, size: int = IMG_SIZE) -> np.ndarray:
    """
    Resize each image to (size, size) and average across all images.
    Returns float32 array in [0, 1] with shape (H, W, 3).
    """
    if not images:
        return np.zeros((size, size, 3), dtype=np.float32)
    resized = []
    for img in images:
        pil = Image.fromarray(img).convert("RGB").resize((size, size), Image.BILINEAR)
        resized.append(np.array(pil, dtype=np.float32) / 255.0)
    return np.mean(resized, axis=0).astype(np.float32)


def encode_sequence(seq: str, max_len: int = MAX_SEQ_LEN):
    """Encode amino acid string to a fixed-length integer list."""
    tokens = [AA_VOCAB.get(aa.upper(), AA_VOCAB["<UNK>"]) for aa in seq[:max_len]]
    tokens += [AA_VOCAB["<PAD>"]] * (max_len - len(tokens))
    return tokens


def image_to_tensor(img_array: np.ndarray) -> torch.Tensor:
    """Convert (H,W,3) float32 [0,1] array to normalised (3,H,W) tensor."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    pil = Image.fromarray((img_array * 255).astype(np.uint8)).convert("RGB")
    return transform(pil)


# =============================================================================
# DATASET
# =============================================================================

class ProteinDataset(Dataset):
    """
    Multimodal dataset yielding:
      image_tensor : (3, 224, 224) float
      seq_tensor   : (MAX_SEQ_LEN,) long
      label_tensor : (NUM_LABELS,) float
    """

    def __init__(self, df: pd.DataFrame, image_dir: str,
                 label_cols, max_seq_len: int = MAX_SEQ_LEN):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.label_cols = label_cols
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        gene_name = row.get("gene_name", "")
        sequence  = str(row.get("sequence", ""))

        # --- Image ---
        imgs = load_protein_images(self.image_dir, gene_name)
        img_arr = average_protein_images(imgs)
        img_tensor = image_to_tensor(img_arr)

        # --- Sequence ---
        tokens = encode_sequence(sequence, self.max_seq_len)
        seq_tensor = torch.tensor(tokens, dtype=torch.long)

        # --- Labels ---
        labels = [float(row.get(col, 0)) for col in self.label_cols]
        label_tensor = torch.tensor(labels, dtype=torch.float32)

        return img_tensor, seq_tensor, label_tensor


# =============================================================================
# MODEL COMPONENTS
# =============================================================================

class ImageBranch(nn.Module):
    """CNN image encoder based on ResNet-18 (pretrained backbone)."""

    def __init__(self, out_dim: int = CNN_OUT_DIM):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove the final classification layer
        in_features = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Sequential(
            nn.Linear(in_features, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        feat = self.backbone(x)   # (B, 512)
        return self.proj(feat)    # (B, CNN_OUT_DIM)


class AttentionLayer(nn.Module):
    """Additive (Bahdanau-style) attention over sequence of hidden states."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, encoder_outputs):
        # encoder_outputs: (B, T, hidden_size)
        scores = self.attention(encoder_outputs)          # (B, T, 1)
        weights = torch.softmax(scores, dim=1)            # (B, T, 1)
        context = (weights * encoder_outputs).sum(dim=1)  # (B, hidden_size)
        return context, weights.squeeze(-1)


class SequenceBranch(nn.Module):
    """Embedding -> BiGRU -> Attention -> feature vector."""

    def __init__(self, vocab_size: int = VOCAB_SIZE,
                 embed_dim: int = EMBED_DIM,
                 hidden_dim: int = HIDDEN_DIM,
                 out_dim: int = SEQ_OUT_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bigru = nn.GRU(embed_dim, hidden_dim, batch_first=True,
                            bidirectional=True)
        # BiGRU doubles the hidden dim
        self.attention = AttentionLayer(hidden_dim * 2)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        emb = self.embedding(x)                           # (B, T, embed_dim)
        gru_out, _ = self.bigru(emb)                      # (B, T, 2*hidden)
        context, attn_weights = self.attention(gru_out)   # (B, 2*hidden)
        return self.proj(context), attn_weights           # (B, out_dim)


class MultimodalFusionModel(nn.Module):
    """
    Full multimodal model:
      image_branch + sequence_branch -> fusion -> multi-label sigmoid output
    """

    def __init__(self, num_labels: int = NUM_LABELS,
                 cnn_out: int = CNN_OUT_DIM, seq_out: int = SEQ_OUT_DIM,
                 dropout: float = 0.3):
        super().__init__()
        self.image_branch = ImageBranch(out_dim=cnn_out)
        self.seq_branch   = SequenceBranch(out_dim=seq_out)

        fused = cnn_out + seq_out
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
            nn.Sigmoid(),
        )

    def forward(self, images, sequences):
        img_feat          = self.image_branch(images)
        seq_feat, attn_w  = self.seq_branch(sequences)
        fused             = torch.cat([img_feat, seq_feat], dim=-1)
        logits            = self.classifier(fused)
        return logits, attn_w


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

def train_model(model, train_loader, val_loader, epochs: int = 5,
                lr: float = 1e-3, device=DEVICE, progress_callback=None):
    """Train the multimodal model and return training history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    criterion = nn.BCELoss()
    model.to(device)

    history = {"train_loss": [], "val_loss": [],
               "val_acc": [], "val_precision": [], "val_recall": []}

    for epoch in range(1, epochs + 1):
        # ---- Training ----
        model.train()
        train_loss = 0.0
        for imgs, seqs, labels in train_loader:
            imgs, seqs, labels = imgs.to(device), seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            preds, _ = model(imgs, seqs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, seqs, labels in val_loader:
                imgs, seqs, labels = imgs.to(device), seqs.to(device), labels.to(device)
                preds, _ = model(imgs, seqs)
                val_loss += criterion(preds, labels).item() * imgs.size(0)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        all_preds  = np.vstack(all_preds) > 0.5
        all_labels = np.vstack(all_labels)

        acc  = accuracy_score(all_labels.flatten(), all_preds.flatten())
        prec = precision_score(all_labels, all_preds, average="micro", zero_division=0)
        rec  = recall_score(all_labels, all_preds, average="micro", zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(acc)
        history["val_precision"].append(prec)
        history["val_recall"].append(rec)

        scheduler.step()

        if progress_callback:
            progress_callback(epoch, epochs, train_loss, val_loss, acc, prec, rec)

    return history


# =============================================================================
# INFERENCE
# =============================================================================

@torch.no_grad()
def predict(model, image_arrays, sequence: str, device=DEVICE, threshold: float = 0.5):
    """
    Run multimodal inference for a single protein.

    Parameters
    ----------
    image_arrays : list of float32 np.ndarray (H,W,3) already in [0,1],
                   or empty list (zero image will be used).
    sequence     : raw amino acid string
    threshold    : decision threshold for each label

    Returns
    -------
    probs       : (NUM_LABELS,) numpy float32
    predictions : (NUM_LABELS,) bool array
    attn_weights: (MAX_SEQ_LEN,) attention map
    """
    model.eval()
    model.to(device)

    # Image
    avg_img = average_protein_images(image_arrays) if image_arrays else \
              np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    img_tensor = image_to_tensor(avg_img).unsqueeze(0).to(device)  # (1,3,224,224)

    # Sequence
    tokens     = encode_sequence(sequence)
    seq_tensor = torch.tensor([tokens], dtype=torch.long).to(device)  # (1, MAX_LEN)

    probs, attn = model(img_tensor, seq_tensor)
    probs = probs.squeeze(0).cpu().numpy()
    attn  = attn.squeeze(0).cpu().numpy()

    return probs, probs >= threshold, attn


# =============================================================================
# EXPLAINABLE AI
# =============================================================================

# Simple rule-based cues for each label
LOCATION_CUES = {
    "Cytoplasm":              ("cytoplasmic targeting signal",
                               "diffuse cytoplasmic staining pattern"),
    "Nucleus":                ("nuclear localisation sequence (NLS)",
                               "compact nuclear staining pattern"),
    "Mitochondria":           ("mitochondrial targeting sequence (MTS)",
                               "punctate mitochondrial staining pattern"),
    "Endoplasmic Reticulum":  ("ER retention signal (KDEL/HDEL)",
                               "reticular ER network staining"),
    "Golgi Apparatus":        ("Golgi retention signal",
                               "perinuclear Golgi staining"),
    "Lysosome":               ("lysosomal targeting motif",
                               "vesicular lysosomal staining"),
    "Peroxisome":             ("peroxisomal targeting signal (PTS1/PTS2)",
                               "small punctate peroxisomal staining"),
    "Plasma Membrane":        ("transmembrane domain or GPI anchor",
                               "peripheral plasma membrane staining"),
    "Cell Junction":          ("cell junction localisation motif",
                               "sharp cell-border staining at junctions"),
    "Cytoskeleton":           ("actin/tubulin binding domain",
                               "fibrous cytoskeletal staining"),
    "Nucleolus":              ("nucleolar localisation sequence",
                               "bright focal nucleolar staining"),
    "Nuclear Membrane":       ("nuclear envelope targeting signal",
                               "ring-shaped nuclear envelope staining"),
    "Vesicle":                ("vesicular trafficking motif",
                               "small vesicular staining puncta"),
    "Centrosome":             ("centrosomal targeting domain",
                               "bright focal centrosomal spot"),
    "Lipid Droplet":          ("lipid-droplet targeting sequence",
                               "round lipid-droplet staining"),
    "Aggresome":              ("aggresome-forming sequence",
                               "juxtanuclear aggresome staining"),
    "Microtubule":            ("microtubule-associated protein domain",
                               "linear microtubule staining"),
}


def generate_explanation(predicted_labels, probs, label_columns):
    """Return a human-readable explanation for the top predicted locations."""
    active = [(label_columns[i], float(probs[i]))
              for i in range(len(label_columns)) if probs[i] >= 0.5]
    active.sort(key=lambda x: x[1], reverse=True)

    if not active:
        top_i = int(np.argmax(probs))
        active = [(label_columns[top_i], float(probs[top_i]))]

    parts = []
    for loc, prob in active[:3]:      # explain top-3 at most
        seq_cue, img_cue = LOCATION_CUES.get(loc, ("characteristic sequence features",
                                                     "characteristic image features"))
        parts.append(
            f"**{loc}** (confidence {prob:.1%}):\n"
            f"  • Sequence patterns suggest *{seq_cue}*\n"
            f"  • Image features resemble *{img_cue}*"
        )

    return "\n\n".join(parts)


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def plot_probability_bar(probs, label_columns, threshold=0.5):
    """Horizontal bar chart of per-label probabilities."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2196F3" if p >= threshold else "#B0BEC5" for p in probs]
    y_pos  = range(len(label_columns))
    ax.barh(list(y_pos), probs, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(label_columns, fontsize=9)
    ax.set_xlim(0, 1)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1, label=f"Threshold ({threshold})")
    ax.set_xlabel("Confidence Score")
    ax.set_title("Subcellular Localization Confidence Scores")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def plot_multilabel_radar(probs, label_columns):
    """Radar / spider chart for multi-label predictions."""
    N = len(label_columns)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values = list(probs) + [probs[0]]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, "o-", linewidth=2, color="#4CAF50")
    ax.fill(angles, values, alpha=0.25, color="#4CAF50")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(label_columns, fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_title("Multi-label Prediction Radar", size=12, pad=16)
    plt.tight_layout()
    return fig


def plot_attention(attn_weights, sequence: str, max_show: int = 50):
    """Heatmap of attention weights over the first `max_show` residues."""
    n = min(len(sequence), max_show)
    weights = attn_weights[:n]
    labels  = list(sequence[:n])

    fig, ax = plt.subplots(figsize=(max(8, n * 0.25), 2))
    im = ax.imshow(weights[np.newaxis, :], aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=weights.max() or 1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks([])
    ax.set_title("Attention Weights on Sequence (first 50 residues)")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    plt.tight_layout()
    return fig


def plot_training_history(history):
    """Line plots of loss and metrics over training epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BCE Loss")
    axes[0].set_title("Training & Validation Loss"); axes[0].legend()

    # Metrics
    axes[1].plot(epochs, history["val_acc"],       "g-o", label="Accuracy")
    axes[1].plot(epochs, history["val_precision"], "b-o", label="Precision")
    axes[1].plot(epochs, history["val_recall"],    "r-o", label="Recall")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score")
    axes[1].set_title("Validation Metrics"); axes[1].legend()

    plt.tight_layout()
    return fig


# =============================================================================
# SESSION STATE HELPERS
# =============================================================================

def get_or_create_model():
    """Return cached model from session state, creating it if needed."""
    if "model" not in st.session_state:
        st.session_state["model"] = MultimodalFusionModel(num_labels=NUM_LABELS)
        st.session_state["model_trained"] = False
    return st.session_state["model"]


# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="Multimodal Protein Localization Predictor",
        page_icon="🔬",
        layout="wide",
    )

    # ── Header ──────────────────────────────────────────────────────────────
    st.title("🔬 Multimodal Protein Localization Predictor")
    st.markdown(
        "Predict protein subcellular localization using **protein images** "
        "and **amino acid sequences** jointly via a deep multimodal model."
    )
    st.divider()

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("📂 Dataset Paths")
        image_dir = st.text_input("Images directory", value=DEFAULT_IMAGE_DIR)
        csv_path  = st.text_input("CSV file path",    value=DEFAULT_CSV_PATH)

        # Resolve to absolute, normalised paths to prevent path traversal
        image_dir = os.path.realpath(os.path.abspath(image_dir.strip()))
        csv_path  = os.path.realpath(os.path.abspath(csv_path.strip()))

        st.subheader("🎓 Training Parameters")
        epochs   = st.slider("Epochs",        min_value=1,  max_value=20, value=5)
        lr       = st.select_slider("Learning rate",
                                    options=[1e-4, 5e-4, 1e-3, 5e-3], value=1e-3)
        batch_sz = st.select_slider("Batch size",
                                    options=[4, 8, 16, 32], value=8)
        train_btn = st.button("🚀 Train Model", use_container_width=True)

        st.divider()
        st.subheader("🔍 Inference Inputs")
        uploaded_images = st.file_uploader(
            "Upload protein images (TIFF / PNG / JPG)",
            type=["tif", "tiff", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )
        sequence_input = st.text_area(
            "Paste protein sequence (amino acids)",
            placeholder="MKTAYIAKQRQISFVKSHFSRQ...",
            height=120,
        )
        threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, step=0.05)
        predict_btn = st.button("🔮 Predict Localization", use_container_width=True)

    # ── Main Tabs ───────────────────────────────────────────────────────────
    tab_overview, tab_train, tab_predict = st.tabs(
        ["📖 Overview", "🎓 Train Model", "🔮 Predict"]
    )

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1 – Overview
    # ─────────────────────────────────────────────────────────────────────────
    with tab_overview:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🏗️ Model Architecture")
            st.markdown("""
| Component | Details |
|---|---|
| **Image Branch** | ResNet-18 backbone → Linear(256) → BN → ReLU |
| **Sequence Branch** | Embedding(64) → BiGRU(128) → Attention → Linear(256) → BN → ReLU |
| **Fusion** | Concatenate → Dropout → Linear(256) → ReLU → Linear(17) → Sigmoid |
| **Loss** | Binary Cross-Entropy (multi-label) |
| **Labels** | 17 subcellular compartments |
""")

        with col2:
            st.subheader("📊 Label Space (17 locations)")
            for i in range(0, NUM_LABELS, 2):
                c1, c2 = st.columns(2)
                c1.markdown(f"🔹 {LABEL_COLUMNS[i]}")
                if i + 1 < NUM_LABELS:
                    c2.markdown(f"🔹 {LABEL_COLUMNS[i+1]}")

        st.subheader("📁 Dataset Information")
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            st.success(f"CSV loaded: **{len(df)} proteins**")
            st.dataframe(df.head(5), use_container_width=True)
        else:
            st.warning(
                f"CSV not found at `{csv_path}`. "
                "Predictions can still be run using uploaded images + pasted sequence."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2 – Train
    # ─────────────────────────────────────────────────────────────────────────
    with tab_train:
        st.subheader("🎓 Train Multimodal Model")

        if train_btn:
            # Validate CSV
            if not os.path.isfile(csv_path):
                st.error(f"CSV file not found: `{csv_path}`")
                st.stop()

            df = pd.read_csv(csv_path)
            # Identify available label columns
            avail_labels = [c for c in LABEL_COLUMNS if c in df.columns]
            if not avail_labels:
                st.error("No label columns found in CSV. "
                         "Expected columns: " + ", ".join(LABEL_COLUMNS))
                st.stop()

            st.info(f"Using {len(avail_labels)} label columns and {len(df)} proteins.")

            # Split
            train_df, val_df = train_test_split(df, test_size=0.2,
                                                 random_state=42)
            train_ds = ProteinDataset(train_df, image_dir, avail_labels)
            val_ds   = ProteinDataset(val_df,   image_dir, avail_labels)
            train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=True,
                                      num_workers=0)
            val_loader   = DataLoader(val_ds,   batch_size=batch_sz, shuffle=False,
                                      num_workers=0)

            model = MultimodalFusionModel(num_labels=len(avail_labels))
            st.session_state["model"] = model
            st.session_state["avail_labels"] = avail_labels

            # Progress widgets
            progress_bar = st.progress(0)
            status_text  = st.empty()
            metrics_placeholder = st.empty()
            train_records = []

            def progress_callback(epoch, total, tl, vl, acc, prec, rec):
                pct = epoch / total
                progress_bar.progress(pct)
                status_text.text(
                    f"Epoch {epoch}/{total} — "
                    f"Train Loss: {tl:.4f} | Val Loss: {vl:.4f} | "
                    f"Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f}"
                )
                train_records.append({
                    "Epoch": epoch, "Train Loss": tl, "Val Loss": vl,
                    "Accuracy": acc, "Precision": prec, "Recall": rec
                })
                metrics_placeholder.dataframe(
                    pd.DataFrame(train_records), use_container_width=True
                )

            with st.spinner("Training in progress…"):
                history = train_model(
                    model, train_loader, val_loader,
                    epochs=epochs, lr=lr, device=DEVICE,
                    progress_callback=progress_callback,
                )

            st.session_state["model_trained"] = True
            st.session_state["history"] = history
            st.success("✅ Training complete!")

            # Training curves
            fig = plot_training_history(history)
            st.pyplot(fig)
            plt.close(fig)

        elif "history" in st.session_state:
            st.info("Showing results from previous training run.")
            fig = plot_training_history(st.session_state["history"])
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Configure parameters in the sidebar and click **Train Model** to begin.")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3 – Predict
    # ─────────────────────────────────────────────────────────────────────────
    with tab_predict:
        st.subheader("🔮 Subcellular Localization Prediction")

        model = get_or_create_model()
        label_cols = st.session_state.get("avail_labels", LABEL_COLUMNS)

        if predict_btn:
            if not sequence_input.strip():
                st.warning("Please paste a protein sequence in the sidebar.")
                st.stop()

            # ---- Load uploaded images ----
            user_images = []
            if uploaded_images:
                col_imgs = st.columns(min(len(uploaded_images), 5))
                for i, uf in enumerate(uploaded_images):
                    try:
                        raw = uf.read()
                        if uf.name.lower().endswith((".tif", ".tiff")):
                            if TIFFFILE_AVAILABLE:
                                arr = tifffile.imread(io.BytesIO(raw))
                            else:
                                arr = np.array(Image.open(io.BytesIO(raw)))
                            # normalise
                            if arr.ndim == 2:
                                arr = np.stack([arr]*3, axis=-1)
                            elif arr.ndim == 3 and arr.shape[0] <= 4:
                                arr = np.transpose(arr, (1, 2, 0))
                            if arr.shape[-1] > 3:
                                arr = arr[..., :3]
                            elif arr.shape[-1] == 1:
                                arr = np.repeat(arr, 3, axis=-1)
                            arr = arr.astype(np.float32)
                            mn, mx = arr.min(), arr.max()
                            if mx > mn:
                                arr = (arr - mn) / (mx - mn)
                            user_images.append(arr)
                        else:
                            pil = Image.open(io.BytesIO(raw)).convert("RGB")
                            arr = np.array(pil, dtype=np.float32) / 255.0
                            user_images.append(arr)

                        col_imgs[i % len(col_imgs)].image(
                            (arr * 255).astype(np.uint8),
                            caption=uf.name, use_container_width=True,
                        )
                    except Exception as e:
                        st.warning(f"Could not load {uf.name}: {e}")
            else:
                st.info("No images uploaded — using a zero-image placeholder for the image branch.")

            # ---- Display sequence ----
            seq = sequence_input.strip().upper()
            st.markdown(f"**Sequence** ({len(seq)} residues):  `{seq[:80]}{'…' if len(seq)>80 else ''}`")

            # ---- Run inference ----
            with st.spinner("Running multimodal inference…"):
                probs, preds, attn_weights = predict(
                    model, user_images, seq, device=DEVICE, threshold=threshold
                )

            # ---- Results ----
            st.divider()
            st.subheader("📊 Prediction Results")

            # Top prediction highlight
            top_idx  = int(np.argmax(probs))
            top_label = label_cols[top_idx] if top_idx < len(label_cols) else LABEL_COLUMNS[top_idx]
            st.success(f"🏆 **Top prediction:** {top_label}  "
                       f"(confidence: {probs[top_idx]:.1%})")

            # Predicted labels list
            predicted = [label_cols[i] for i in range(len(label_cols)) if preds[i]]
            if not predicted:
                predicted = [label_cols[top_idx]]
            st.markdown("**Predicted subcellular locations:** " +
                        ", ".join(f"`{l}`" for l in predicted))

            # ---- Probability chart ----
            col_a, col_b = st.columns(2)
            with col_a:
                fig_bar = plot_probability_bar(
                    probs[:len(label_cols)], label_cols, threshold=threshold
                )
                st.pyplot(fig_bar)
                plt.close(fig_bar)

            with col_b:
                fig_radar = plot_multilabel_radar(probs[:len(label_cols)], label_cols)
                st.pyplot(fig_radar)
                plt.close(fig_radar)

            # ---- Attention map ----
            st.subheader("🔍 Sequence Attention Map")
            fig_attn = plot_attention(attn_weights, seq)
            st.pyplot(fig_attn)
            plt.close(fig_attn)

            # ---- Explainable AI ----
            st.subheader("🧠 Explainable AI")
            explanation = generate_explanation(preds, probs, label_cols)
            st.markdown(explanation)

            # ---- Confidence table ----
            st.subheader("📋 Confidence Scores Table")
            score_df = pd.DataFrame({
                "Compartment": label_cols,
                "Confidence": [f"{p:.1%}" for p in probs[:len(label_cols)]],
                "Predicted":  ["✅" if b else "❌" for b in preds[:len(label_cols)]],
            })
            st.dataframe(score_df, use_container_width=True, hide_index=True)

        else:
            st.info(
                "Upload protein image(s), paste a sequence in the sidebar, "
                "then click **Predict Localization**."
            )


if __name__ == "__main__":
    main()
