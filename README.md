# Protein Sub-Cellular Localization in Neurons

## 🔬 Advanced Deep Learning System for Neuronal Protein Analysis

A comprehensive scientific system for analyzing neuronal TIFF microscopy images and classifying protein sub-cellular localization using deep learning and graph neural networks.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Documentation](#documentation)
- [License](#license)

---

## 🎯 Overview

This system integrates state-of-the-art deep learning techniques to automatically analyze microscopy images and predict protein localization in neurons across five major cellular compartments:

1. **Nucleus**
2. **Cytoplasm**
3. **Mitochondria**
4. **Endoplasmic Reticulum**
5. **Membrane**

The system combines:
- **VGG16-based CNN** for global feature extraction
- **Graph Neural Networks (GNN)** for spatial relationship modeling
- **U-Net/SLIC segmentation** for cellular compartment delineation
- **Model fusion** for robust predictions
- **Publication-quality visualizations** (300+ DPI)

---

## ✨ Features

### Core Functionality
- ✅ Single TIFF image analysis
- ✅ Batch processing with recursive directory scanning
- ✅ Multiple segmentation methods (U-Net, SLIC, Watershed)
- ✅ Hybrid CNN + GNN architecture
- ✅ Model fusion with weighted combination
- ✅ Comprehensive evaluation metrics

### User Interface
- 🌐 Clean web interface (Flask-based)
- 📤 Drag-and-drop file upload
- 📊 Real-time result visualization
- 💾 Downloadable reports (JSON, images)

### Scientific Outputs
- 📈 Publication-ready plots (300+ DPI)
- 🎨 Segmentation overlays and masks
- 📉 Probability distributions
- 🔀 Confusion matrices
- 📊 Evaluation metrics (accuracy, precision, recall, F1, specificity)
- 🕸️ Graph structure visualizations
- 📄 Comprehensive JSON reports

### Advanced Features
- 🧬 Superpixel-based graph construction
- 🔬 Intensity profile analysis
- 📁 Batch summary generation
- 🎯 Per-class performance metrics
- 📚 Automated journal document generation

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Input: TIFF Images                      │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│              Image Preprocessing & Normalization            │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│              Segmentation (SLIC Superpixels)                │
└───────┬─────────────────────────────────────────────────────┘
        │
        ├─────────────────┬───────────────────┐
        ▼                 ▼                   ▼
┌───────────────┐  ┌──────────────┐  ┌──────────────────┐
│  CNN Branch   │  │  GNN Branch  │  │  Visualization   │
│   (VGG16)     │  │   (GCN/GAT)  │  │    Generation    │
└───────┬───────┘  └──────┬───────┘  └──────────────────┘
        │                 │
        └────────┬────────┘
                 ▼
         ┌───────────────┐
         │ Model Fusion  │
         └───────┬───────┘
                 ▼
         ┌───────────────┐
         │  Prediction   │
         └───────┬───────┘
                 ▼
         ┌───────────────┐
         │ Report & Save │
         └───────────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) NVIDIA GPU with CUDA for accelerated processing

### Step 1: Clone Repository

```bash
git clone https://github.com/soujanyap29/portfolio.github.io.git
cd portfolio.github.io
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Create Output Directories

The system will automatically create necessary directories, but you can also create them manually:

```bash
mkdir -p /mnt/d/5TH_SEM/CELLULAR/input
mkdir -p /mnt/d/5TH_SEM/CELLULAR/output/{segmented,predictions,reports,graphs}
```

**Note**: Update paths in `output/backend/config.py` if using different directories.

---

## 💻 Usage

### Option 1: Web Interface (Recommended)

1. Start the Flask server:

```bash
cd output/frontend
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Use the interface to:
   - Upload single TIFF images
   - Start batch processing
   - View results
   - Download reports

### Option 2: Python API

```python
from output.backend.pipeline import ProteinLocalizationPipeline

# Initialize pipeline
pipeline = ProteinLocalizationPipeline()

# Analyze single image
result = pipeline.analyze_single_image("path/to/image.tif")
print(f"Predicted class: {result['fused_prediction']['class']}")
print(f"Confidence: {result['fused_prediction']['confidence']:.2%}")

# Batch process directory
results = pipeline.batch_process("/path/to/input/directory")
print(f"Processed {len(results)} images")
```

### Option 3: Command Line

```bash
# Single image analysis
python -c "
from output.backend.pipeline import ProteinLocalizationPipeline
pipeline = ProteinLocalizationPipeline()
result = pipeline.analyze_single_image('image.tif')
print(result)
"

# Batch processing
python -c "
from output.backend.pipeline import ProteinLocalizationPipeline
pipeline = ProteinLocalizationPipeline()
pipeline.batch_process()
"
```

---

## 📁 Project Structure

```
portfolio.github.io/
├── requirements.txt                    # Python dependencies
├── JOURNAL_DOCUMENT.md                 # Complete scientific documentation
├── README.md                           # This file
│
├── output/
│   ├── frontend/                       # Web interface
│   │   ├── app.py                     # Flask application
│   │   ├── templates/
│   │   │   └── index.html             # Main web page
│   │   └── static/
│   │       ├── css/
│   │       └── js/
│   │
│   ├── backend/                        # Core analysis modules
│   │   ├── config.py                  # Configuration settings
│   │   ├── pipeline.py                # Main orchestrator
│   │   ├── image_processor.py         # TIFF loading & preprocessing
│   │   ├── segmentation.py            # U-Net, SLIC, Watershed
│   │   ├── cnn_classifier.py          # VGG16-based classifier
│   │   ├── gnn_classifier.py          # Graph neural network
│   │   ├── evaluation.py              # Metrics calculation
│   │   └── visualization.py           # Scientific plotting
│   │
│   └── results/                        # Output directory
│       ├── segmented/                 # Segmentation masks
│       ├── predictions/               # Prediction images
│       ├── reports/                   # JSON reports
│       └── graphs/                    # Visualizations
│
└── code                                # Original C++ hospital system
```

---

## 📦 Requirements

### Core Dependencies

```
# Deep Learning
torch>=1.9.0
torchvision>=0.10.0
torch-geometric>=2.0.0

# Image Processing
opencv-python>=4.5.0
scikit-image>=0.18.0
tifffile>=2021.7.0
Pillow>=8.3.0

# Scientific Computing
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
scikit-learn>=0.24.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0
networkx>=2.6.0

# Web Framework
flask>=2.0.0
streamlit>=1.0.0

# Utilities
tqdm>=4.62.0
PyYAML>=5.4.0
```

See `requirements.txt` for complete list.

---

## 📚 Documentation

### Scientific Paper

See [`JOURNAL_DOCUMENT.md`](./JOURNAL_DOCUMENT.md) for complete scientific documentation including:

- Abstract
- Introduction and literature survey
- Detailed methodology
- System architecture
- Mathematical formulations
- Evaluation metrics
- Mermaid diagrams
- Results and discussion
- References

### API Documentation

Each module contains detailed docstrings. Example:

```python
from output.backend.pipeline import ProteinLocalizationPipeline
help(ProteinLocalizationPipeline)
```

---

## 🎓 Citation

If you use this system in your research, please cite:

```bibtex
@software{protein_localization_2024,
  title={Protein Sub-Cellular Localization in Neurons Using Deep Learning},
  author={Your Name},
  year={2024},
  url={https://github.com/soujanyap29/portfolio.github.io}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- PyTorch and PyTorch Geometric teams
- OpenCV and scikit-image communities
- Scientific visualization libraries (Matplotlib, Seaborn)
- Public microscopy image databases (Human Protein Atlas, Allen Brain Atlas)

---

## 📞 Support

For questions, issues, or feature requests:

- 📧 Email: support@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/soujanyap29/portfolio.github.io/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/soujanyap29/portfolio.github.io/discussions)

---

## 🔄 Version History

- **v1.0.0** (2024-11-19)
  - Initial release
  - VGG16 + GNN hybrid architecture
  - SLIC superpixel segmentation
  - Web interface
  - Batch processing
  - Publication-quality visualizations
  - Complete journal documentation

---

## 🚀 Roadmap

- [ ] Multi-label classification for co-localized proteins
- [ ] 3D volumetric image support (z-stacks)
- [ ] Real-time analysis mode
- [ ] Additional GNN architectures (GraphSAGE, GAT)
- [ ] Self-supervised pre-training
- [ ] Docker containerization
- [ ] REST API for programmatic access
- [ ] Integration with image databases

---

**Made with ❤️ for the neuroscience research community**
