# Protein Sub-Cellular Localization in Neurons

A complete, production-ready system for analyzing protein localization in neuronal cells using deep learning, graph neural networks, and advanced image processing.

## 🎯 Project Overview

This system provides end-to-end analysis of 4D TIFF microscopy images:
- **Automatic segmentation** using Cellpose
- **Biological graph construction** from segmented regions
- **Deep learning classification** with Graph-CNN
- **Scientific visualizations** for publication
- **Web interface** for easy file upload and prediction

## 📋 Features

### 1. Image Processing & Segmentation
- Recursive TIFF file scanning
- Support for 2D, 3D, and 4D images
- Cellpose-based segmentation
- Feature extraction (spatial, morphological, intensity)

### 2. Graph Construction
- Biological graph generation from segmented regions
- Spatial proximity-based edge creation
- Rich node and edge attributes
- PyTorch Geometric compatibility

### 3. Deep Learning Models
- **Graph-CNN**: Graph Convolutional Network for graph-based classification
- **VGG-16**: Traditional CNN for image classification
- **Hybrid Model**: Combined CNN + Graph-CNN architecture

### 4. Visualization Suite
- Segmentation overlays
- Compartment mask maps
- Grouped bar plots (mean ± SEM)
- Box/violin plots
- Colocalization scatter plots
- Intensity profiles
- Graph visualizations with node labels
- Confusion matrices

### 5. Web Interface
- Modern, responsive UI
- Drag-and-drop file upload
- Real-time processing feedback
- Interactive result visualization
- Download results

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Ubuntu (recommended) or macOS
- CUDA-capable GPU (optional, for faster processing)

### Setup

```bash
# Clone the repository
cd protein_localization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Cellpose (may require additional steps)
pip install cellpose[gui]
```

## 📂 Directory Structure

```
protein_localization/
├── preprocessing/          # Image loading and segmentation
├── graph_construction/     # Biological graph building
├── models/                 # Deep learning models
├── visualization/          # Scientific plotting
├── frontend/               # Web interface
│   ├── app.py             # Flask application
│   └── templates/         # HTML templates
├── utils/                  # Utility functions
├── output/                 # Output directory
│   ├── models/            # Trained models
│   ├── visualizations/    # Generated plots
│   ├── segmented/         # Segmentation results
│   ├── graphs/            # Saved graphs
│   ├── predictions/       # Prediction results
│   └── final_pipeline.ipynb  # Complete notebook
├── config.yaml            # Configuration file
└── requirements.txt       # Python dependencies
```

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
data:
  input_dir: "/mnt/d/5TH_SEM/CELLULAR/input"    # Your TIFF directory
  output_dir: "/mnt/d/5TH_SEM/CELLULAR/output"  # Output directory

segmentation:
  model_type: "cyto"        # Cellpose model
  diameter: 30              # Cell diameter (pixels)
  
training:
  model_type: "graph_cnn"   # Model architecture
  epochs: 100               # Training epochs
  batch_size: 16            # Batch size
  learning_rate: 0.001      # Learning rate
```

## 💻 Usage

### 1. Jupyter Notebook (Recommended for first-time users)

```bash
cd protein_localization
jupyter lab output/final_pipeline.ipynb
```

Run all cells to execute the complete pipeline:
1. Load and preprocess TIFF files
2. Segment with Cellpose
3. Build biological graphs
4. Train models
5. Generate visualizations
6. Make predictions

### 2. Web Interface

```bash
cd protein_localization/frontend
python app.py
```

Then open your browser to `http://localhost:5000`

**Features:**
- Upload TIFF files via drag-and-drop
- Automatic processing through pipeline
- View segmentation, graphs, and predictions
- Download results

### 3. Python API

```python
from preprocessing import TIFFProcessor
from graph_construction import BiologicalGraphBuilder
from models import GraphCNN, ModelTrainer
from visualization import ScientificVisualizer
from utils import load_config

# Load configuration
config = load_config('config.yaml')

# Process TIFF file
processor = TIFFProcessor(config)
img, masks, features = processor.process_single_tiff('path/to/file.tif')

# Build graph
graph_builder = BiologicalGraphBuilder(config)
G = graph_builder.build_graph(features)

# Visualize
visualizer = ScientificVisualizer(config)
visualizer.plot_segmentation_overlay(img, masks, 'output.png')
visualizer.plot_graph_visualization(G, 'graph.png')
```

## 📊 Model Training

### Train Graph-CNN

```python
from models import GraphCNN, ModelTrainer
import torch

# Initialize model
model = GraphCNN(num_node_features=7, num_classes=3, hidden_dim=64)
trainer = ModelTrainer(model, config)

# Train
for epoch in range(epochs):
    loss, acc = trainer.train_epoch(train_loader)
    print(f"Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.4f}")

# Save
trainer.save_model('output/models/model.pth')
```

## 📈 Evaluation Metrics

The system computes:
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class precision
- **Recall**: Per-class recall (sensitivity)
- **F1-score**: Harmonic mean of precision and recall
- **Specificity**: True negative rate
- **Confusion Matrix**: Detailed classification results

## 🎨 Visualizations

All visualizations are publication-ready with:
- High resolution (300 DPI)
- Scientific styling
- Clear labels and legends
- Multiple format support (PNG, PDF)

### Available Plots
1. **Segmentation Overlay**: Original + masks + overlay
2. **Compartment Map**: Color-coded regions
3. **Grouped Bar Plot**: Mean ± SEM comparisons
4. **Box/Violin Plots**: Distribution analysis
5. **Colocalization Scatter**: Channel correlation
6. **Intensity Profile**: Distance-based intensity
7. **Graph Visualization**: Biological network with labels
8. **Confusion Matrix**: Classification performance

## 🔬 Scientific Applications

This system can be used for:
- Protein localization studies
- Subcellular compartment analysis
- Neuron morphology research
- Drug screening assays
- High-throughput microscopy analysis

## 🐛 Troubleshooting

### Common Issues

**Issue**: Cellpose installation fails
```bash
# Try installing with conda
conda install -c conda-forge cellpose
```

**Issue**: CUDA out of memory
```bash
# Reduce batch size in config.yaml
training:
  batch_size: 8  # or smaller
```

**Issue**: TIFF files not loading
- Ensure files are valid TIFF format
- Check file permissions
- Verify path in config.yaml

## 📝 Citation

If you use this system in your research, please cite:

```bibtex
@software{protein_localization_2025,
  title={Protein Sub-Cellular Localization System},
  author={Your Name},
  year={2025},
  url={https://github.com/soujanyap29/portfolio.github.io}
}
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Cellpose**: For excellent segmentation models
- **PyTorch Geometric**: For graph neural network support
- **scikit-image**: For image processing utilities

## 📧 Contact

For questions or support:
- Create an issue on GitHub
- Email: [your-email@example.com]

## 🔮 Future Enhancements

- [ ] Multi-GPU training support
- [ ] Real-time video processing
- [ ] 3D visualization
- [ ] Additional model architectures
- [ ] Automated hyperparameter tuning
- [ ] Cloud deployment support
- [ ] REST API for programmatic access

---

**Last Updated**: November 2025  
**Version**: 1.0.0
