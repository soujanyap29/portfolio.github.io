# Protein Sub-Cellular Localization in Neurons - Complete Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Security: Checked](https://img.shields.io/badge/Security-Verified-green.svg)](https://github.com/soujanyap29/portfolio.github.io)

A complete, competition-ready pipeline for processing 4D TIFF images and accurately predicting the sub-cellular localization of proteins in neurons.

## 🎯 Project Overview

This project implements a comprehensive machine learning pipeline that combines:
- **Image Processing**: Cellpose-based segmentation of neuronal structures
- **Graph Neural Networks**: Biological graph construction and GNN-based classification
- **Deep Learning**: Graph-CNN, GAT, and Hybrid CNN+GNN architectures
- **Visualization**: Publication-ready plots and interactive visualizations
- **Web Interface**: User-friendly Flask application for predictions

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Pipeline Components](#pipeline-components)
- [Model Architecture](#model-architecture)
- [Outputs](#outputs)
- [Security](#security)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

### Data Processing
- ✅ Recursive TIFF file scanning across multiple subdirectories
- ✅ 4D TIFF image loading (time, z-stack, x, y)
- ✅ Cellpose segmentation with GPU acceleration
- ✅ Comprehensive feature extraction:
  - Spatial coordinates
  - Morphological measurements (area, perimeter, eccentricity, etc.)
  - Intensity distributions (mean, max, min)
  - Region-based statistics

### Graph Construction
- ✅ Biological graph representation of protein networks
- ✅ Nodes: protein puncta or sub-cellular compartments
- ✅ Edges: spatial and structural relationships
- ✅ PyTorch Geometric compatibility
- ✅ Node label preservation for visualization

### Model Training
- ✅ **Graph-CNN**: Graph Convolutional Network for classification
- ✅ **GAT**: Graph Attention Network with multi-head attention
- ✅ **Hybrid CNN+GNN**: VGG-16 + Graph-CNN combined architecture
- ✅ Comprehensive metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Specificity (per class and overall)
  - Confusion Matrix

### Visualization
- ✅ **Image Overlays**: Raw TIFF + segmentation outlines
- ✅ **Compartment Maps**: Colored segmentation by region
- ✅ **Statistical Plots**: Grouped bar plots with mean ± SEM
- ✅ **Distribution Plots**: Box and violin plots
- ✅ **Colocalization**: Scatter/hexbin plots with Pearson/Manders coefficients
- ✅ **Intensity Profiles**: Distance-dependent intensity analysis
- ✅ **Graph Visualization**: Network plots with labeled nodes
- ✅ **Training History**: Loss and accuracy curves
- ✅ **Confusion Matrix**: Performance visualization

### Web Interface
- ✅ File upload for TIFF images
- ✅ Real-time prediction
- ✅ Interactive result display
- ✅ Downloadable visualizations
- ✅ Metric summaries

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 16GB+ RAM (for large images)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/soujanyap29/portfolio.github.io.git
cd portfolio.github.io
```

2. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### GPU Setup (Optional but Recommended)

For CUDA support:
```bash
# Install PyTorch with CUDA (example for CUDA 11.8)
pip install torch==2.6.0+cu118 torchvision==0.17.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

## 🎬 Quick Start

### Option 1: Jupyter Notebook (Recommended)

The easiest way to run the complete pipeline:

```bash
jupyter notebook notebooks/final_pipeline.ipynb
```

Then execute all cells sequentially. The notebook includes:
- ✅ Step-by-step pipeline execution
- ✅ Inline visualizations
- ✅ Detailed explanations
- ✅ Sample predictions

### Option 2: Web Interface

Launch the Flask application:

```bash
cd src/frontend
export FLASK_DEBUG=false  # Secure production mode
python app.py
```

Open browser to: `http://localhost:5000`

Upload a TIFF file and get instant predictions!

### Option 3: Command Line

Run individual modules:

```bash
# Preprocessing
python src/preprocessing/preprocess.py

# Graph construction
python src/graph/graph_builder.py

# Model training
python src/models/train.py
```

## 📁 Project Structure

```
portfolio.github.io/
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── preprocess.py          # TIFF loading & segmentation
│   ├── graph/
│   │   ├── __init__.py
│   │   └── graph_builder.py       # Graph construction
│   ├── models/
│   │   ├── __init__.py
│   │   └── train.py               # Model architectures & training
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plots.py               # Publication-ready plots
│   └── frontend/
│       ├── app.py                 # Flask web interface
│       ├── templates/
│       │   └── index.html         # Web UI
│       └── static/
├── notebooks/
│   └── final_pipeline.ipynb       # Complete end-to-end notebook
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── README_PROJECT.md              # Additional documentation
├── QUICKSTART.md                  # Quick reference guide
└── .gitignore
```

## 📖 Usage

### Data Preparation

Place your TIFF images in:
```
/mnt/d/5TH_SEM/CELLULAR/input/
```

The pipeline will recursively scan all subdirectories for `.tif` and `.tiff` files.

### Running the Pipeline

#### 1. Preprocessing
```python
from preprocessing import PreprocessingPipeline

pipeline = PreprocessingPipeline(
    input_dir="/mnt/d/5TH_SEM/CELLULAR/input",
    output_dir="/mnt/d/5TH_SEM/CELLULAR/output"
)
results = pipeline.process_all()
```

#### 2. Graph Construction
```python
from graph import GraphDataset

dataset = GraphDataset("output/preprocessed_data.pkl")
dataset.load_and_build_graphs()
dataset.save_graphs("output/graphs.pkl")
```

#### 3. Model Training
```python
from models import GraphCNN, ModelTrainer, prepare_data_loaders

# Prepare data
train_loader, test_loader = prepare_data_loaders("output/graphs.pkl")

# Initialize model
model = GraphCNN(num_features=6, num_classes=6)
trainer = ModelTrainer(model)

# Train
metrics = trainer.train(train_loader, test_loader, num_epochs=100)
trainer.save_model("output/models/graph_cnn.pth")
```

#### 4. Visualization
```python
from visualization import VisualizationSuite

viz = VisualizationSuite(output_dir="output")
viz.plot_image_overlay(image, masks)
viz.plot_graph_visualization(graph)
viz.plot_confusion_matrix(cm, class_names)
```

## 🔬 Pipeline Components

### 1. Preprocessing (`src/preprocessing/preprocess.py`)

**Classes:**
- `TIFFProcessor`: Recursive file scanning and loading
- `CellposeSegmenter`: Neural network-based segmentation
- `FeatureExtractor`: Extract morphological and intensity features
- `PreprocessingPipeline`: Complete preprocessing workflow

**Key Features:**
- Handles 2D, 3D, and 4D TIFF files
- Automatic image normalization
- GPU-accelerated segmentation
- Batch processing support

### 2. Graph Construction (`src/graph/graph_builder.py`)

**Classes:**
- `GraphBuilder`: Convert segmentations to graphs
- `GraphDataset`: PyTorch Geometric dataset
- `CompartmentGraph`: Hierarchical compartment modeling

**Graph Properties:**
- Nodes: Individual protein puncta/compartments
- Node features: area, perimeter, intensity, morphology
- Edges: K-nearest neighbors or distance threshold
- Edge weights: Inverse distance

### 3. Models (`src/models/train.py`)

**Architectures:**

1. **Graph-CNN** (Graph Convolutional Network)
   - 3 GCN layers with batch normalization
   - Global mean pooling
   - Dropout for regularization
   - ~100K parameters

2. **GAT** (Graph Attention Network)
   - Multi-head attention mechanism
   - 4 attention heads per layer
   - Enhanced feature learning

3. **Hybrid CNN+GNN**
   - VGG-16 for image features
   - GCN for graph features
   - Feature fusion layer
   - Best for multi-modal data

**Training Features:**
- Adam optimizer
- Cross-entropy loss
- Learning rate: 0.001
- Batch size: 32
- Early stopping based on validation accuracy

### 4. Visualization (`src/visualization/plots.py`)

**Plot Types:**
- Image overlay (raw + segmentation)
- Compartment masks
- Grouped bar plots with error bars
- Box and violin plots
- Colocalization scatter/hexbin
- Intensity profiles
- Network graphs
- Confusion matrices
- Training curves

**Style:**
- Publication-ready quality
- 300 DPI resolution
- Scientific color schemes
- Clean, professional layout

### 5. Web Interface (`src/frontend/app.py`)

**Features:**
- Drag-and-drop file upload
- Real-time processing feedback
- Interactive result display
- Metric visualization
- Download results

**Security:**
- CORS protection
- File type validation
- Size limits (100MB)
- Debug mode disabled by default

## 🏗️ Model Architecture

### Graph-CNN Architecture

```
Input: Node features (6D) + Edge index

Layer 1: GCNConv(6, 64) + BatchNorm + ReLU + Dropout(0.5)
Layer 2: GCNConv(64, 64) + BatchNorm + ReLU + Dropout(0.5)
Layer 3: GCNConv(64, 64) + BatchNorm + ReLU + Dropout(0.5)

Global Mean Pool

FC1: Linear(64, 32) + ReLU + Dropout(0.5)
FC2: Linear(32, 6)

Output: Class predictions (6 classes)
```

**Classes:**
1. Soma
2. Dendrite
3. Axon
4. Nucleus
5. Synapse
6. Mitochondria

## 📊 Outputs

All outputs are saved to: `/mnt/d/5TH_SEM/CELLULAR/output/`

### Files Generated:
```
output/
├── preprocessed_data.pkl          # Processed images + features
├── graphs.pkl                     # Graph dataset
├── models/
│   ├── graph_cnn.pth             # Trained model weights
│   └── metrics.json              # Evaluation metrics
├── image_overlay.png             # Segmentation overlay
├── compartment_map.png           # Compartment visualization
├── graph_viz.png                 # Network graph
├── confusion_matrix.png          # Model performance
├── training_history.png          # Training curves
├── colocalization.png            # Channel analysis
└── ... (additional plots)
```

### Metrics Reported:
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class and weighted average
- **Recall**: Per-class and weighted average
- **F1-Score**: Harmonic mean of precision/recall
- **Specificity**: True negative rate per class
- **Confusion Matrix**: Detailed classification breakdown

## 🔒 Security

This project follows security best practices:

### Dependency Security
All dependencies are updated to latest secure versions:
- ✅ PyTorch 2.6.0+ (fixes RCE, heap overflow)
- ✅ Flask 2.3.2+ (fixes session disclosure)
- ✅ Pillow 10.2.0+ (fixes path traversal, RCE)

### Code Security
- ✅ CodeQL scanned (0 alerts)
- ✅ Flask debug mode disabled in production
- ✅ Input validation on file uploads
- ✅ CORS protection enabled
- ✅ File size limits enforced

### Recommended Practices
- Use environment variables for sensitive config
- Run behind reverse proxy (nginx) in production
- Enable HTTPS for web interface
- Regular dependency updates

## 📚 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{protein_localization_pipeline_2024,
  title={Protein Sub-Cellular Localization Pipeline},
  author={Patil, Soujanya},
  year={2024},
  url={https://github.com/soujanyap29/portfolio.github.io},
  note={A complete pipeline for protein localization prediction in neurons}
}
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup
```bash
pip install -r requirements.txt
pip install pytest black flake8  # Development tools
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Cellpose**: Stringer, C., et al. (2021). "Cellpose: a generalist algorithm for cellular segmentation." Nature Methods.
- **PyTorch Geometric**: Fey, M., & Lenssen, J. E. (2019). "Fast Graph Representation Learning with PyTorch Geometric."
- **Graph Neural Networks**: Kipf, T. N., & Welling, M. (2017). "Semi-Supervised Classification with Graph Convolutional Networks."

## 📞 Contact

Soujanya Patil
- GitHub: [@soujanyap29](https://github.com/soujanyap29)
- LinkedIn: [Soujanya Patil](https://www.linkedin.com/in/soujanya-patil-056a93306)

## 🐛 Known Issues

- Large 4D TIFF files (>1GB) may require significant RAM
- Cellpose segmentation is GPU-intensive
- First-time model loading may take 1-2 minutes

## 🗺️ Roadmap

- [ ] Add temporal analysis for time-series data
- [ ] Implement transfer learning capabilities
- [ ] Support additional image formats (CZI, ND2)
- [ ] Add batch processing via CLI
- [ ] Docker containerization
- [ ] Cloud deployment guide

---

**Built with ❤️ for the neuroscience community**

For detailed usage instructions, see [QUICKSTART.md](QUICKSTART.md)
