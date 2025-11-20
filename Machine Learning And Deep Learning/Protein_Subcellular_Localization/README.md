# Protein Sub-Cellular Localization in Neurons

**Student:** Soujanya  
**Course:** Machine Learning and Deep Learning  
**Project Type:** Research-Grade Computational Platform

## 🔬 Overview

This project implements a complete, automated system for analyzing neuronal TIFF microscopy images to determine protein sub-cellular localization using advanced deep learning techniques.

## 🎯 Features

- **Automated TIFF Image Processing**: Batch processing of real microscopy images
- **Cellpose Segmentation**: Biological segmentation of neuronal structures
- **Dual Model Architecture**:
  - VGG16 Convolutional Neural Network (CNN)
  - Graph Neural Networks (GCN/GraphSAGE/GAT)
- **Model Fusion**: Late fusion for improved prediction accuracy
- **High-Resolution Visualizations**: Publication-quality outputs (≥300 DPI)
- **Web Interface**: User-friendly dashboard for image upload and analysis
- **Automated Reports**: Journal-style PDF and JSON reports

## 📁 Project Structure

```
Protein_Subcellular_Localization/
├── backend/
│   ├── models/
│   │   ├── cnn_model.py         # VGG16 implementation
│   │   └── gnn_model.py         # GNN models (GCN/GraphSAGE/GAT)
│   ├── segmentation/
│   │   └── cellpose_segmentation.py  # Cellpose integration
│   ├── utils/
│   │   ├── image_preprocessing.py    # TIFF loading and preprocessing
│   │   ├── graph_construction.py     # Superpixel and graph generation
│   │   ├── model_fusion.py           # Prediction fusion
│   │   └── visualization.py          # Scientific visualizations
├── frontend/
│   ├── app.py                   # Flask web application
│   └── templates/
│       └── index.html           # Web interface
├── notebooks/
│   └── automated_pipeline.ipynb # Complete automated workflow
├── config.yaml                  # Configuration file
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for faster training)
- 8GB+ RAM recommended

### Setup

```bash
# Clone the repository
cd "Machine Learning And Deep Learning/Protein_Subcellular_Localization"

# Install dependencies
pip install -r requirements.txt

# Note: Cellpose may require additional setup
# Follow instructions at: https://github.com/MouseLand/cellpose
```

## 💻 Usage

### 1. Jupyter Notebook (Automated Pipeline)

The complete automated workflow is available in a Jupyter notebook:

```bash
cd notebooks
jupyter notebook automated_pipeline.ipynb
```

This notebook automatically:
- Scans `/mnt/d/5TH_SEM/CELLULAR/input` for TIFF files
- Performs segmentation, prediction, and fusion
- Generates all visualizations and reports
- Saves results to `/mnt/d/5TH_SEM/CELLULAR/output`

### 2. Web Interface

Start the Flask web application:

```bash
cd frontend
python app.py
```

Then open your browser to `http://localhost:5000`

Features:
- Upload single or multiple TIFF images
- View segmentation results
- Compare CNN, GNN, and fused predictions
- Download reports

### 3. Command Line (Batch Processing)

```python
from backend.utils.image_preprocessing import TIFFLoader
from backend.segmentation.cellpose_segmentation import CellposeSegmenter

# Load images
loader = TIFFLoader()
images = loader.batch_load("/path/to/input/directory")

# Segment
segmenter = CellposeSegmenter()
for filepath, original, processed in images:
    masks, info = segmenter.segment(original)
    # Continue with processing...
```

## 📊 Models

### CNN (VGG16)
- Transfer learning from ImageNet
- Fine-tuned on microscopy images
- Outputs: class predictions + probability distributions

### GNN (Graph Neural Networks)
- Superpixel-based graph construction
- Node features: intensity, texture, geometry
- Edge features: spatial adjacency
- Architectures: GCN, GraphSAGE, GAT

### Fusion
- Late fusion of CNN and GNN predictions
- Weighted averaging (configurable weights)
- Improved accuracy over individual models

## 📈 Evaluation Metrics

All models are evaluated using:
- Accuracy
- Precision (macro/micro/weighted)
- Recall (macro/micro/weighted)
- F1-Score (macro/micro/weighted)
- Specificity
- Confusion Matrix

## 🎨 Visualizations

Generated visualizations include:
- Raw TIFF images
- Segmentation overlays
- Superpixel graphs with curved edges
- Probability distribution plots
- Confusion matrices
- Training history plots
- Performance comparison charts

All visualizations are saved at ≥300 DPI for publication quality.

## 📝 Output Structure

```
/mnt/d/5TH_SEM/CELLULAR/output/
├── segmented/              # Segmentation visualizations
│   └── *_segment.png
├── predictions/            # Model predictions
│   └── combined_predictions.csv
├── reports/               # Individual image reports
│   └── *_report.json
├── graphs/                # All visualizations
│   ├── *_cnn_probs.png
│   ├── *_gnn_probs.png
│   ├── *_fused_probs.png
│   └── *_graph.png
└── final_pipeline.ipynb   # Completed notebook
```

## ⚙️ Configuration

Edit `config.yaml` to customize:
- Input/output directories
- Model hyperparameters
- Segmentation parameters
- Visualization settings
- Class names

## 🔧 Development

### Adding New Models

1. Create model class in `backend/models/`
2. Implement training and prediction methods
3. Update fusion logic in `backend/utils/model_fusion.py`
4. Update notebook and web interface

### Adding New Visualizations

1. Add method to `backend/utils/visualization.py`
2. Call from notebook or web app
3. Ensure ≥300 DPI output

## 📚 Dependencies

Key dependencies:
- PyTorch / TorchVision
- PyTorch Geometric
- Cellpose
- scikit-image
- Flask
- Matplotlib / Seaborn
- NetworkX

See `requirements.txt` for complete list.

## 🔬 Scientific Applications

This platform can be used for:
- Protein localization studies
- Neuronal morphology analysis
- Drug response screening
- Disease mechanism research
- Comparative cell biology

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@software{protein_localization_2023,
  title={Protein Sub-Cellular Localization in Neurons},
  author={Soujanya},
  year={2023},
  course={Machine Learning and Deep Learning}
}
```

## 🤝 Contributing

This is a student project. For questions or collaboration:
- Student: Soujanya
- Course: Machine Learning and Deep Learning

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- Cellpose team for segmentation model
- PyTorch Geometric team for GNN framework
- Course instructors and mentors

---

**Status**: ✅ Production Ready
**Last Updated**: 2023
**Version**: 1.0.0
