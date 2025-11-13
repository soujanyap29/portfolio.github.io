# Protein Sub-Cellular Localization in Neurons

An automated pipeline for processing and analyzing TIFF images from the OpenCell database to predict protein sub-cellular localization patterns in neurons.

## Overview

This project implements a complete end-to-end pipeline for:
- Loading and preprocessing TIFF images
- Constructing graph representations from images
- Training deep learning models (GNN/CNN)
- Running inference across multiple samples
- Evaluating and visualizing results

## Features

- **Batch Processing**: Handles multiple TIFF images automatically
- **Graph-Based Analysis**: Converts images to graph representations for spatial analysis
- **Deep Learning Models**: Supports both CNN and Graph Neural Network architectures
- **Flexible Pipeline**: Works with any TIFF images from OpenCell database
- **Visualization**: Comprehensive visualization of results and predictions
- **Jupyter Integration**: Interactive notebooks for exploration and analysis

## Installation

### Prerequisites
- Python 3.8 or higher
- Ubuntu system (or compatible Linux distribution)
- GPU recommended for training (CUDA-enabled)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/soujanyap29/portfolio.github.io.git
cd portfolio.github.io/protein_localization
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Ubuntu/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

1. **Prepare your data**: Place TIFF images in the `data/raw/` directory

2. **Run the complete pipeline**:
```bash
python main.py --input_dir data/raw --output_dir outputs
```

### Step-by-Step Processing

#### 1. Data Loading and Sanity Checks
```bash
python utils/data_loader.py --input_dir data/raw --validate
```

#### 2. Image Preprocessing
```bash
python utils/preprocessor.py --input_dir data/raw --output_dir data/processed
```

#### 3. Graph Construction
```bash
python utils/graph_builder.py --input_dir data/processed --output_dir data/graphs
```

#### 4. Train Model
```bash
python train.py --data_dir data/graphs --model_type gnn --epochs 100
```

#### 5. Run Inference
```bash
python inference.py --model_path outputs/models/best_model.pth --input_dir data/raw
```

#### 6. Evaluate and Visualize
```bash
python evaluate.py --predictions_dir outputs/results --ground_truth_dir data/labels
```

### Using Jupyter Lab

Start Jupyter Lab for interactive analysis:
```bash
jupyter lab
```

Open `notebooks/protein_localization_pipeline.ipynb` for a complete walkthrough.

## Project Structure

```
protein_localization/
├── data/                      # Data directory
│   ├── raw/                   # Raw TIFF images
│   ├── processed/             # Preprocessed images
│   ├── graphs/                # Graph representations
│   └── labels/                # Label files
├── models/                    # Model architectures
│   ├── cnn_model.py          # CNN architecture
│   ├── gnn_model.py          # Graph Neural Network
│   └── hybrid_model.py       # Hybrid CNN-GNN
├── utils/                     # Utility modules
│   ├── data_loader.py        # Data loading utilities
│   ├── preprocessor.py       # Image preprocessing
│   ├── graph_builder.py      # Graph construction
│   └── visualizer.py         # Visualization tools
├── notebooks/                 # Jupyter notebooks
│   └── protein_localization_pipeline.ipynb
├── outputs/                   # Output directory
│   ├── models/               # Saved models
│   ├── results/              # Prediction results
│   └── visualizations/       # Generated plots
├── main.py                   # Main orchestration script
├── train.py                  # Training script
├── inference.py              # Inference script
├── evaluate.py               # Evaluation script
├── config.yaml               # Configuration file
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Configuration

Edit `config.yaml` to customize the pipeline:

```yaml
data:
  input_dir: "data/raw"
  output_dir: "outputs"
  image_size: [512, 512]
  
preprocessing:
  normalize: true
  denoise: true
  enhance_contrast: true

graph:
  node_features: ["intensity", "texture", "morphology"]
  edge_threshold: 0.5

model:
  type: "gnn"  # Options: cnn, gnn, hybrid
  hidden_dim: 128
  num_layers: 3

training:
  batch_size: 16
  epochs: 100
  learning_rate: 0.001
  device: "cuda"
```

## Examples

### Process a single TIFF file
```python
from utils.data_loader import load_tiff
from utils.preprocessor import preprocess_image
from utils.graph_builder import build_graph

# Load image
image = load_tiff("path/to/image.tif")

# Preprocess
processed = preprocess_image(image)

# Build graph
graph = build_graph(processed)
```

### Batch processing multiple files
```python
from main import process_batch

results = process_batch(
    input_dir="data/raw",
    output_dir="outputs",
    model_path="outputs/models/best_model.pth"
)
```

## Requirements

- Compatible with Ubuntu + Jupyter Lab
- Processes all TIFF images in a given folder
- Builds correct graph representations for every image
- Valid outputs for all images
- Batch mode for multiple files

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{protein_localization_pipeline,
  title={Protein Sub-Cellular Localization Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/soujanyap29/portfolio.github.io}
}
```

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub.
