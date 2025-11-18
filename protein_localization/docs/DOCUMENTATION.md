# Project Documentation Index

## Protein Sub-Cellular Localization Pipeline

A complete bioinformatics pipeline for automated protein localization analysis.

---

## 📚 Documentation

### Getting Started
1. **[QUICKSTART.md](QUICKSTART.md)** - Fast setup and installation guide
2. **[README.md](README.md)** - Complete project documentation

### Core Documentation
- **[requirements.txt](requirements.txt)** - Python dependencies
- **[setup.py](setup.py)** - Package installation script

---

## 📁 Project Structure

```
protein_localization/
│
├── 📄 README.md              # Main documentation
├── 📄 QUICKSTART.md          # Quick start guide
├── 📄 requirements.txt       # Dependencies
├── 📄 setup.py              # Installation script
├── 📄 demo.py               # Demo script
│
├── 📂 src/                  # Core modules
│   ├── preprocessing.py     # TIFF loading & segmentation
│   ├── graph_builder.py     # Graph construction
│   ├── models.py           # Deep learning models
│   ├── visualization.py     # Scientific visualizations
│   └── __init__.py         # Package initialization
│
├── 📂 frontend/             # Web interface
│   ├── app.py              # Flask application
│   └── templates/
│       └── index.html       # Web UI
│
├── 📂 notebooks/            # Jupyter notebooks
│   └── final_pipeline.ipynb # Complete pipeline
│
├── 📂 tests/                # Unit tests (future)
└── 📂 docs/                 # Additional documentation (future)
```

---

## 🔧 Core Modules

### 1. Preprocessing (`src/preprocessing.py`)
**Purpose**: Load TIFF images, perform segmentation, extract features

**Key Classes**:
- `TIFFPreprocessor` - Main preprocessing class

**Key Functions**:
- `scan_tiff_files()` - Recursively find TIFF files
- `load_tiff()` - Load 3D/4D TIFF images
- `segment_cellpose()` - Cellpose segmentation with fallback
- `extract_features()` - Extract morphological and intensity features
- `process_single_tiff()` - Complete processing pipeline for one file

**Features Extracted**:
- Spatial: centroid, distance from center
- Morphological: area, perimeter, eccentricity, solidity
- Intensity: mean, max, min, std

---

### 2. Graph Builder (`src/graph_builder.py`)
**Purpose**: Construct biological graphs from segmented regions

**Key Classes**:
- `BiologicalGraphBuilder` - Graph construction
- `GraphFeatureExtractor` - Graph-level features

**Key Functions**:
- `build_graph()` - Create NetworkX graph from features
- `networkx_to_pyg()` - Convert to PyTorch Geometric format
- `process_results()` - Build graphs for multiple samples

**Graph Properties**:
- Nodes: Puncta/compartments with features
- Edges: Spatial relationships (K-NN or threshold)
- Compatible with PyTorch Geometric

---

### 3. Models (`src/models.py`)
**Purpose**: Deep learning models for classification

**Key Classes**:
- `GraphCNN` - Graph Convolutional Network
- `VGG16Classifier` - CNN-based classifier
- `HybridModel` - Combined CNN + Graph-CNN
- `ModelTrainer` - Training and evaluation

**Features**:
- Multi-class classification
- Comprehensive metrics (accuracy, precision, recall, F1)
- Early stopping
- Model saving/loading

---

### 4. Visualization (`src/visualization.py`)
**Purpose**: Create publication-ready scientific figures

**Key Classes**:
- `ProteinVisualization` - Visualization suite

**Key Functions**:
- `plot_segmentation_overlay()` - Raw + segmentation overlay
- `plot_compartment_masks()` - Compartment maps
- `plot_feature_distributions()` - Statistical distributions
- `plot_graph()` - Graph visualization with labels
- `plot_training_history()` - Training curves
- `plot_confusion_matrix()` - Model performance

**Output Quality**: 300 DPI, publication-ready

---

## 🌐 Web Interface

### Flask Application (`frontend/app.py`)

**Endpoints**:
- `GET /` - Main page
- `POST /upload` - File upload and processing
- `GET /results/<filename>` - Download results
- `POST /load_model` - Load trained model
- `GET /status` - System status

**Features**:
- Drag-and-drop upload
- Real-time processing
- Interactive results display
- Downloadable outputs

---

## 📓 Jupyter Notebook

### Complete Pipeline (`notebooks/final_pipeline.ipynb`)

**Sections**:
1. Setup and imports
2. Configuration
3. Data preprocessing
4. Graph construction
5. Model training
6. Visualization
7. Prediction demo
8. Summary and export
9. Web interface instructions

**Features**:
- Step-by-step execution
- Inline documentation
- Handles missing data gracefully
- Generates synthetic data for testing

---

## 🚀 Usage Modes

### 1. Quick Demo
```bash
python demo.py
```
- Tests installation
- Creates synthetic data
- Runs complete pipeline
- Generates sample outputs

### 2. Jupyter Notebook
```bash
jupyter lab notebooks/final_pipeline.ipynb
```
- Interactive execution
- Detailed documentation
- Visualization inline
- Best for learning

### 3. Web Interface
```bash
cd frontend && python app.py
```
- User-friendly GUI
- No coding required
- Real-time results
- Best for end users

### 4. Python API
```python
from src import preprocess_pipeline, build_graphs_pipeline

results = preprocess_pipeline(input_dir, output_dir)
graphs = build_graphs_pipeline(results, output_dir)
```
- Programmatic access
- Full customization
- Best for integration

---

## 📊 Output Structure

```
/mnt/d/5TH_SEM/CELLULAR/output/
│
├── models/
│   ├── graph_cnn_model.pt         # Trained model
│   └── graph_cnn_metrics.json     # Performance metrics
│
├── graphs/
│   ├── sample1_graph.gpickle      # NetworkX graph
│   └── sample1_pyg.pt             # PyTorch Geometric
│
├── visualizations/
│   ├── sample1_graph.png          # Graph visualization
│   ├── sample1_features.png       # Feature distributions
│   ├── sample1_intensity.png      # Intensity profile
│   ├── training_history.png       # Training curves
│   └── confusion_matrix.png       # Performance matrix
│
└── pipeline_summary.json          # Complete summary
```

---

## 🔬 Scientific Methods

### Segmentation
- **Primary**: Cellpose deep learning segmentation
- **Fallback**: Threshold-based segmentation
- **Supports**: 2D, 3D, 4D TIFF images

### Graph Construction
- **Method**: K-nearest neighbors or distance threshold
- **Nodes**: Segmented compartments with features
- **Edges**: Spatial proximity relationships

### Classification
- **Architecture**: Graph-CNN, VGG-16, or Hybrid
- **Training**: Adam optimizer, early stopping
- **Metrics**: Accuracy, precision, recall, F1, specificity

### Visualization
- **Style**: Publication-ready, scientific quality
- **Format**: PNG at 300 DPI
- **Types**: Overlays, distributions, graphs, matrices

---

## 📋 Requirements

### System Requirements
- Python 3.8+
- 4GB+ RAM (8GB+ recommended)
- GPU optional (10-100x speedup)

### Python Packages
- Core: numpy, pandas, scipy
- Image: tifffile, scikit-image, cellpose
- ML: torch, torch-geometric, scikit-learn
- Graph: networkx
- Viz: matplotlib, seaborn
- Web: flask

See [requirements.txt](requirements.txt) for complete list.

---

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **Cellpose Not Found**
   - Pipeline uses fallback segmentation automatically

3. **CUDA Out of Memory**
   - Reduce batch size
   - Use CPU mode

4. **No TIFF Files Found**
   - Check input directory path
   - Verify file permissions

See [QUICKSTART.md](QUICKSTART.md) for more solutions.

---

## 🤝 Contributing

This is a portfolio project, but contributions are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## 📞 Support

- **Documentation**: See README.md and QUICKSTART.md
- **Issues**: Open a GitHub issue
- **Questions**: Contact through portfolio website

---

## 📈 Performance

### Benchmarks
- **Preprocessing**: ~1-5 seconds per TIFF
- **Graph Building**: <1 second per sample
- **Training**: ~1-5 minutes (depends on data size)
- **Prediction**: <1 second per sample

### Scalability
- Handles hundreds of TIFF files
- GPU acceleration available
- Batch processing supported

---

## 🎯 Future Enhancements

- [ ] Additional segmentation algorithms
- [ ] Multi-channel TIFF support
- [ ] 4D time-series analysis
- [ ] REST API
- [ ] Docker containerization
- [ ] Cloud deployment
- [ ] 3D visualizations
- [ ] Automated hyperparameter tuning

---

## 📚 References

### Scientific Papers
- **Cellpose**: Stringer et al. (2020) - Nature Methods
- **Graph Neural Networks**: Various sources
- **PyTorch Geometric**: Fey & Lenssen (2019)

### Technologies
- PyTorch
- PyTorch Geometric
- NetworkX
- Flask
- Matplotlib/Seaborn

---

## 📄 License

MIT License - See [LICENSE](../LICENSE) file

---

## ✨ Acknowledgments

This project demonstrates:
- Scientific computing with Python
- Deep learning for biomedical imaging
- Graph neural networks
- Full-stack web development
- Production-ready ML pipelines

Built for the bioinformatics and neuroscience community.

---

**Last Updated**: November 2025
**Version**: 1.0.0
**Maintainer**: Portfolio Project
