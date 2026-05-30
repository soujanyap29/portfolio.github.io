from pathlib import Path

CSV_PATH = Path("/home/lab3/min/merged_dataset.csv")
DATA_ROOT = Path("/home/lab3/min/data")
RAW_DATA_DIR = DATA_ROOT / "raw"
OUTPUT_DIR = Path("/home/lab3/min/output")
SPLIT_DIR = OUTPUT_DIR / "splits"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
MODEL_PATH = CHECKPOINT_DIR / "alzheimers_model.pt"
BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "best_checkpoint.pt"
LAST_CHECKPOINT_PATH = CHECKPOINT_DIR / "last_checkpoint.pt"
TENSORBOARD_DIR = OUTPUT_DIR / "tensorboard"
PLOTS_DIR = OUTPUT_DIR / "plots"
METRICS_DIR = OUTPUT_DIR / "metrics"

PUBLIC_DATASET_SLUG = "sachinkumar413/alzheimer-mri-dataset"
CLASS_ALIASES = {
    "NonDemented": "Non Demented",
    "VeryMildDemented": "Very Mild Demented",
    "MildDemented": "Mild Demented",
    "ModerateDemented": "Moderate Demented",
}

