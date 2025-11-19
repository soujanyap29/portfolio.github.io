"""
Configuration file for Protein Sub-Cellular Localization System
"""
import os

# Directory Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
GRAPHS_DIR = os.path.join(OUTPUT_DIR, "graphs")
FRONTEND_DIR = os.path.join(OUTPUT_DIR, "frontend")
BACKEND_DIR = os.path.join(OUTPUT_DIR, "backend")

# Input/Output Paths
INPUT_PATH = "/mnt/d/5TH_SEM/CELLULAR/input"
OUTPUT_PATH = "/mnt/d/5TH_SEM/CELLULAR/output"
GRAPH_OUTPUT_PATH = "/mnt/d/5TH_SEM/CELLULAR/output/graphs"

# Model Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50

# Protein Localization Classes
PROTEIN_CLASSES = [
    "Nucleus",
    "Cytoplasm",
    "Membrane",
    "Mitochondria",
    "Endoplasmic Reticulum",
    "Golgi Apparatus",
    "Peroxisome",
    "Cytoskeleton"
]

# Segmentation Configuration
SEGMENTATION_METHOD = "CELLPOSE"  # Options: "CELLPOSE", "UNET", "SLIC", "WATERSHED"
USE_CELLPOSE_PREPROCESSING = True  # Use Cellpose's advanced normalization and preprocessing
CELLPOSE_DIAMETER = 30.0  # Estimated cell/nucleus diameter in pixels (None for auto-detect)
CELLPOSE_FLOW_THRESHOLD = 0.4  # Flow threshold (higher = more selective)
CELLPOSE_CELLPROB_THRESHOLD = 0.0  # Cell probability threshold
SLIC_N_SEGMENTS = 100
SLIC_COMPACTNESS = 10

# Visualization Configuration
DPI = 300
FIGURE_SIZE = (10, 8)
COLORMAP = "viridis"

# GNN Configuration
GNN_HIDDEN_DIM = 128
GNN_NUM_LAYERS = 3
GNN_DROPOUT = 0.5

# Evaluation Metrics
CONFIDENCE_THRESHOLD = 0.8
