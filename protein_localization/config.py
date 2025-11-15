"""
Configuration file for protein sub-cellular localization pipeline
"""
import os

# Directory paths
INPUT_DIR = "/mnt/d/5TH_SEM/CELLULAR/input"
OUTPUT_DIR = "/mnt/d/5TH_SEM/CELLULAR/output/output"
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, "visualizations")

# File extensions to process
TIFF_EXTENSIONS = ['.tif', '.tiff', '.TIF', '.TIFF']

# Segmentation parameters
CELLPOSE_MODEL = 'cyto2'  # or 'nuclei', 'cyto', etc.
CELLPOSE_DIAMETER = None  # Auto-detect
CELLPOSE_CHANNELS = [0, 0]  # grayscale

# Feature extraction parameters
SPATIAL_FEATURES = ['centroid_x', 'centroid_y', 'centroid_z', 'pairwise_distance']
MORPHOLOGICAL_FEATURES = ['area', 'perimeter', 'eccentricity', 'solidity', 'extent']
INTENSITY_FEATURES = ['mean_intensity', 'max_intensity', 'min_intensity', 'std_intensity']

# Graph construction parameters
PROXIMITY_THRESHOLD = 50  # pixels for edge creation
MAX_EDGES_PER_NODE = 10

# Model training parameters
TRAIN_TEST_SPLIT = 0.8
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# Graph CNN parameters
GRAPH_HIDDEN_CHANNELS = 64
GRAPH_NUM_LAYERS = 3
GRAPH_DROPOUT = 0.5

# VGG-16 parameters
VGG_PRETRAINED = True
VGG_NUM_CLASSES = 10  # Adjust based on your dataset

# Visualization parameters
FIGURE_DPI = 300
COLORMAP = 'viridis'
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# Interface parameters
MAX_UPLOAD_SIZE = None  # No restriction
ALLOWED_EXTENSIONS = TIFF_EXTENSIONS
