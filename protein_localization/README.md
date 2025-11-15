# Protein Sub-Cellular Localization Pipeline

A complete, competition-ready pipeline for analyzing 4D TIFF microscopy images to predict protein sub-cellular localization in neurons.

## 🔬 Overview

This pipeline implements a comprehensive workflow for:
- **Segmentation**: Cellpose-based detection of neuronal structures
- **Feature Extraction**: Spatial, morphological, and intensity features
- **Graph Construction**: Graph Neural Network compatible representations
- **Model Training**: Graph-CNN, VGG-16, and combined architectures
- **Visualization**: Publication-ready plots and analytics
- **Web Interface**: User-friendly Gradio app with no upload restrictions

## 📁 Project Structure

```
protein_localization/
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── preprocessing/             # Data loading and segmentation
│   ├── segmentation.py       # Cellpose segmentation
│   └── feature_extraction.py # Feature extraction
├── graph_construction/        # Graph building
│   └── graph_builder.py      # Graph construction and conversion
├── models/                    # Neural network models
│   ├── graph_cnn.py          # Graph Neural Networks
│   ├── vgg16.py              # CNN models
│   ├── combined_model.py     # Hybrid CNN + GNN
│   └── trainer.py            # Training framework
├── visualization/             # Plotting and visualization
│   ├── plotters.py           # Statistical plots
│   ├── graph_viz.py          # Graph visualization
│   └── metrics.py            # Evaluation metrics
├── interface/                 # Web interface
│   └── app.py                # Gradio application
└── notebooks/                 # Jupyter notebooks
    └── final_pipeline.ipynb  # Complete executable notebook
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional but recommended)
- At least 8GB RAM

### Install Dependencies

```bash
cd protein_localization
pip install -r requirements.txt
```

## 📊 Usage

### 1. Command Line Pipeline

Process TIFF images from directory:

```python
from preprocessing.segmentation import DirectoryHandler, TIFFLoader, CellposeSegmenter
from preprocessing.feature_extraction import FeatureExtractor
from graph_construction.graph_builder import GraphConstructor

# Scan directory
handler = DirectoryHandler("/mnt/d/5TH_SEM/CELLULAR/input")
tiff_files = handler.scan_directory()

# Load and segment
loader = TIFFLoader()
segmenter = CellposeSegmenter()
image = loader.load_tiff(tiff_files[0])
masks, info = segmenter.segment_image(image)

# Extract features
extractor = FeatureExtractor()
features = extractor.extract_all_features(image, masks)

# Build graph
constructor = GraphConstructor()
graph = constructor.construct_graph(features, masks)
```

### 2. Web Interface

Launch the Gradio interface:

```bash
cd protein_localization
python -m interface.app
```

Then open your browser to `http://localhost:7860`

Features:
- Upload TIFF files (any size, no restrictions)
- Automatic segmentation and analysis
- Real-time visualizations
- Download results

### 3. Jupyter Notebook

Open and run the complete pipeline:

```bash
cd protein_localization/notebooks
jupyter lab final_pipeline.ipynb
```

## 🎯 Pipeline Features

### Preprocessing
- ✅ Recursive directory scanning for all TIFF files
- ✅ Support for .tif and .tiff formats
- ✅ Cellpose segmentation (soma, dendrites, axons, puncta)
- ✅ Multi-channel and 4D image support

### Feature Extraction
- ✅ **Spatial Features**: Centroids, coordinates, pairwise distances
- ✅ **Morphological Features**: Area, perimeter, shape descriptors
- ✅ **Intensity Features**: Channel-wise intensities, histograms, distributions
- ✅ **Region-Level Descriptors**: Masks, neighborhoods, local interactions

### Graph Construction
- ✅ Nodes for protein puncta and cellular compartments
- ✅ Edges based on spatial proximity and adjacency
- ✅ Compatible with PyTorch Geometric and DGL
- ✅ Stable node labels throughout training

### Models
- ✅ **Graph-CNN**: GCN, GAT, GraphSAGE architectures
- ✅ **VGG-16**: Pre-trained and custom variants
- ✅ **Combined Model**: Hybrid CNN + Graph-CNN with multiple fusion strategies
- ✅ Complete training framework with early stopping

### Evaluation
- ✅ Accuracy, Precision, Recall, F1-Score
- ✅ Specificity per class
- ✅ Confusion matrix
- ✅ Per-class metrics

### Visualization
- ✅ Segmentation overlays
- ✅ Color-coded compartment maps
- ✅ Grouped bar plots with mean ± SEM
- ✅ Box plots and violin plots
- ✅ Scatter / hexbin plots
- ✅ Manders and Pearson co-localization metrics
- ✅ Intensity profile plots
- ✅ Graph visualizations with rounded nodes and clean styling

## 🔧 Configuration

Edit `config.py` to customize:

```python
INPUT_DIR = "/mnt/d/5TH_SEM/CELLULAR/input"
OUTPUT_DIR = "/mnt/d/5TH_SEM/CELLULAR/output/output"
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

# Model parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
```

## 📈 Model Training

```python
from models.trainer import ModelTrainer, create_data_loaders
from models.graph_cnn import GraphCNN

# Create model
model = GraphCNN(in_channels=20, hidden_channels=64, out_channels=10)

# Initialize trainer
trainer = ModelTrainer(model, learning_rate=0.001)

# Train
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    save_dir="./models"
)
```

Models are saved to: `/mnt/d/5TH_SEM/CELLULAR/output/output/models`

## 📊 Outputs

All outputs are saved to: `/mnt/d/5TH_SEM/CELLULAR/output/output`

### Directory Structure
```
output/
├── models/                    # Trained models
│   ├── best_model.pth
│   ├── final_model.pth
│   └── training_history.json
├── visualizations/            # All plots and figures
│   ├── segmentation_*.png
│   ├── graph_*.png
│   ├── confusion_matrix.png
│   └── metrics_*.png
└── features/                  # Extracted features
    └── features_*.csv
```

## 🎓 System Requirements (Detailed)

### Minimum Requirements
- **OS**: Ubuntu 18.04+ / Windows 10+ / macOS 10.14+
- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 10GB free space

### Recommended Requirements
- **OS**: Ubuntu 20.04+
- **CPU**: 8+ cores
- **GPU**: NVIDIA GPU with 8GB+ VRAM (CUDA 11.0+)
- **RAM**: 16GB+
- **Storage**: 50GB+ SSD

## 🐛 Troubleshooting

### Cellpose Installation Issues
```bash
pip install cellpose==2.0.0 --no-deps
pip install torch torchvision
```

### PyTorch Geometric Issues
```bash
pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
```

### Memory Issues
- Reduce `BATCH_SIZE` in config.py
- Process images in smaller batches
- Use CPU instead of GPU for smaller datasets

## 📝 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{protein_localization_pipeline,
  author = {Patil, Soujanya},
  title = {Protein Sub-Cellular Localization Pipeline},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/soujanyap29/portfolio.github.io}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Soujanya Patil**
- GitHub: [@soujanyap29](https://github.com/soujanyap29)
- LinkedIn: [Soujanya Patil](https://www.linkedin.com/in/soujanya-patil-056a93306)

## 🙏 Acknowledgments

- Cellpose for segmentation
- PyTorch and PyTorch Geometric for deep learning
- scikit-image for image processing
- Gradio for web interface

## 📮 Support

For issues and questions:
- Open an issue on GitHub
- Contact: [GitHub Profile](https://github.com/soujanyap29)

---

**Note**: This is a research pipeline. For production use, additional validation and testing is recommended.
