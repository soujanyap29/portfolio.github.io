"""
Configuration file for Protein Sub-Cellular Localization System
"""
import os

# Directory Paths
INPUT_DIR = "/mnt/d/5TH_SEM/CELLULAR/input"
OUTPUT_DIR = "/mnt/d/5TH_SEM/CELLULAR/output"
SEGMENTED_DIR = os.path.join(OUTPUT_DIR, "segmented")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
GRAPHS_DIR = os.path.join(OUTPUT_DIR, "graphs")

# Model Parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 5  # Adjust based on localization classes
LEARNING_RATE = 0.001
EPOCHS = 50

# Localization Classes
LOCALIZATION_CLASSES = [
    "Nucleus",
    "Cytoplasm",
    "Mitochondria",
    "Endoplasmic Reticulum",
    "Membrane"
]

# Segmentation Parameters
SLIC_N_SEGMENTS = 100
SLIC_COMPACTNESS = 10
SLIC_SIGMA = 1

# Graph Parameters
GNN_HIDDEN_CHANNELS = 64
GNN_NUM_LAYERS = 3
GNN_DROPOUT = 0.5

# Visualization Parameters
DPI = 300
FIGURE_SIZE = (12, 8)

# Model Weights (for fusion)
VGG16_WEIGHT = 0.6
GNN_WEIGHT = 0.4

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SEGMENTED_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)
