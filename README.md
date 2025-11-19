# Protein Sub-Cellular Localization in Neurons

## Machine Learning and Deep Learning Course Project

A complete scientific system for analyzing neuronal TIFF microscopy images and classifying protein sub-cellular localization using deep learning (CNN + GNN), biological segmentation, and high-quality scientific visualizations.

---

## 🌟 Features

### Machine Learning Models
- **VGG16-based Deep CNN**: Fine-tuned on neuronal microscopy datasets for global feature extraction
- **Graph Neural Network (GNN)**: Superpixel-based graph construction with GCN/GraphSAGE/GAT architectures
- **Model Fusion**: Late fusion and weighted score combination for improved accuracy

### Segmentation
- SLIC Superpixel Segmentation
- U-Net Deep Learning Segmentation
- Watershed Segmentation

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score, Specificity
- Confusion Matrix
- Per-class metrics
- Probability distributions

### Scientific Visualizations (300+ DPI)
- Raw TIFF + segmentation mask overlays
- Compartment mask maps
- Grouped bar plots with error bars
- Box/violin plots
- Colocalization scatter/hexbin plots
- Graph network visualizations
- Intensity profile plots

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/soujanyap29/portfolio.github.io.git
cd portfolio.github.io
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure paths** (Optional)
Edit `output/backend/config.py` to set your input/output directories:
```python
INPUT_PATH = "/mnt/d/5TH_SEM/CELLULAR/input"
OUTPUT_PATH = "/mnt/d/5TH_SEM/CELLULAR/output"
```

---

## 🚀 Usage

### Web Interface (Streamlit)

Launch the web interface:
```bash
cd output/frontend
streamlit run streamlit_app.py
```

The interface provides:
- **Single Image Analysis**: Upload and analyze individual TIFF files
- **Batch Processing**: Process entire directories recursively
- **Interactive Results**: View segmentation, predictions, probabilities, and visualizations
- **Downloadable Reports**: Export results as JSON

### Command Line Interface

**Process a single image:**
```bash
cd output/backend
python pipeline.py --image /path/to/image.tif --output /path/to/output
```

**Batch process a directory:**
```bash
python pipeline.py --batch /path/to/input/dir --output /path/to/output
```

---

## 📁 Project Structure

```
portfolio.github.io/
├── output/
│   ├── backend/
│   │   ├── config.py              # Configuration settings
│   │   ├── image_loader.py        # TIFF loading and preprocessing
│   │   ├── segmentation.py        # Segmentation module (U-Net/SLIC/Watershed)
│   │   ├── cnn_model.py           # VGG16 CNN classifier
│   │   ├── gnn_model.py           # Graph Neural Network models
│   │   ├── model_fusion.py        # Model ensemble methods
│   │   ├── evaluation.py          # Evaluation metrics
│   │   ├── visualization.py       # Scientific visualization
│   │   └── pipeline.py            # Main inference pipeline
│   ├── frontend/
│   │   └── streamlit_app.py       # Streamlit web interface
│   ├── results/
│   │   ├── segmented/             # Segmentation outputs
│   │   ├── predictions/           # Prediction results
│   │   └── reports/               # JSON reports
│   └── graphs/                    # Scientific visualizations
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🧬 Protein Localization Classes

The system classifies proteins into 8 sub-cellular locations:

1. **Nucleus** - Nuclear proteins
2. **Cytoplasm** - Cytoplasmic proteins
3. **Membrane** - Plasma membrane proteins
4. **Mitochondria** - Mitochondrial proteins
5. **Endoplasmic Reticulum** - ER-localized proteins
6. **Golgi Apparatus** - Golgi proteins
7. **Peroxisome** - Peroxisomal proteins
8. **Cytoskeleton** - Cytoskeletal proteins

---

## 🔬 System Architecture

```
Input TIFF Image
     ↓
Preprocessing & Normalization
     ↓
Segmentation (SLIC/U-Net/Watershed)
     ↓
├─→ VGG16 CNN ────────────┐
│                         ↓
└─→ Superpixel Graph → GNN ──→ Model Fusion → Final Prediction
                                     ↓
                          Visualization & Report Generation
```

### Pipeline Steps

1. **TIFF Image Loading**: Load and normalize microscopy images
2. **Segmentation**: Apply biological segmentation (SLIC superpixels by default)
3. **CNN Classification**: VGG16-based global feature extraction and classification
4. **Graph Construction**: Build superpixel graph with intensity, texture, and geometric features
5. **GNN Classification**: Graph-based spatial reasoning and classification
6. **Model Fusion**: Combine predictions using weighted fusion (default: 60% CNN, 40% GNN)
7. **Visualization**: Generate publication-ready plots and graphs
8. **Report Generation**: Export comprehensive JSON reports

---

## 📊 Output Files

### For Single Image Analysis

Each processed image generates:

1. **Segmentation**: `<filename>_segment.png` - Segmentation visualization
2. **Overlay**: `<filename>_overlay.png` - Image with segmentation overlay
3. **Probabilities**: `<filename>_probabilities.png` - Probability distribution plot
4. **Graph**: `<filename>_graph.png` - Superpixel graph network
5. **Compartments**: `<filename>_compartments.png` - Colored compartment map
6. **Report**: `<filename>_report.json` - Complete analysis results

### For Batch Processing

- `batch_summary.json` - Summary of all processed images
- Individual reports for each image

---

## 🔧 Configuration

Edit `output/backend/config.py` to customize:

```python
# Image processing
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Segmentation
SEGMENTATION_METHOD = "SLIC"  # Options: "UNET", "SLIC", "WATERSHED"
SLIC_N_SEGMENTS = 100
SLIC_COMPACTNESS = 10

# Model parameters
GNN_HIDDEN_DIM = 128
GNN_NUM_LAYERS = 3
GNN_DROPOUT = 0.5

# Visualization
DPI = 300
FIGURE_SIZE = (10, 8)

# Model fusion weights
CNN_WEIGHT = 0.6
GNN_WEIGHT = 0.4
```

---

## 🧪 Example Results

### Single Image Analysis Output

```json
{
  "filename": "neuron_001.tif",
  "cnn": {
    "predicted_class": "Nucleus",
    "confidence": 0.892
  },
  "gnn": {
    "predicted_class": "Nucleus",
    "confidence": 0.854
  },
  "fused": {
    "predicted_class": "Nucleus",
    "confidence": 0.876
  }
}
```

---

## 🔬 Scientific Applications

- **Neurodegenerative Disease Research**: Identify mislocalized proteins in disease models
- **Synaptic Protein Mapping**: Characterize protein distribution in synapses
- **Drug Discovery**: Assess drug effects on protein localization
- **Cell-Type Classification**: Distinguish neuronal subtypes based on protein patterns
- **Biomarker Studies**: Identify localization-based disease biomarkers

---

## 📚 Technical Stack

- **Deep Learning**: TensorFlow 2.14, Keras 2.14
- **Graph Learning**: PyTorch 2.1, PyTorch Geometric 2.4
- **Image Processing**: scikit-image, OpenCV, Pillow
- **Segmentation**: U-Net, SLIC, Watershed
- **Visualization**: Matplotlib, Seaborn, NetworkX
- **Web Interface**: Streamlit 1.29
- **Data Processing**: NumPy, Pandas, SciPy

---

## 🎓 Course Information

**Course**: Machine Learning and Deep Learning  
**Project Type**: Complete Scientific Analysis System  
**Semester**: 5th Semester

---

## 📝 Citation

If you use this system in your research, please cite:

```
@software{protein_localization_2025,
  title = {Protein Sub-Cellular Localization in Neurons: A Deep Learning Approach},
  author = {Your Name},
  year = {2025},
  institution = {Your Institution},
  course = {Machine Learning and Deep Learning}
}
```

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional GNN architectures (e.g., Graph Transformers)
- Support for 3D microscopy (Z-stacks)
- Multi-channel image analysis
- Active learning for annotation
- Self-supervised pre-training
- Real-time inference optimization

---

## 📄 License

This project is developed for educational purposes as part of a Machine Learning and Deep Learning course.

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the repository owner.

---

## 🙏 Acknowledgments

- VGG16 architecture from Visual Geometry Group, Oxford
- PyTorch Geometric for graph neural network implementations
- scikit-image for image processing utilities
- The computational neuroscience and cellular imaging community

---

**Built with ❤️ for advancing neuroscience research through machine learning**
