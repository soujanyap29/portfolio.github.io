# Quick Start Guide

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/soujanyap29/portfolio.github.io.git
cd portfolio.github.io
```

2. **Create Python environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Run Complete Pipeline via Jupyter Notebook (Recommended)

```bash
jupyter notebook notebooks/final_pipeline.ipynb
```

Then execute all cells sequentially. The notebook includes:
- Data loading and preprocessing
- Segmentation with Cellpose
- Graph construction
- Model training and evaluation
- Visualizations
- Prediction demo

### Option 2: Run Individual Modules

**Preprocessing:**
```bash
cd src/preprocessing
python preprocess.py
```

**Graph Construction:**
```bash
cd src/graph
python graph_builder.py
```

**Model Training:**
```bash
cd src/models
python train.py
```

### Option 3: Use Web Interface

```bash
cd src/frontend
python app.py
```

Then open browser to: `http://localhost:5000`

Upload a TIFF file and click "Process & Predict" to get:
- Predicted protein localization class
- Confidence scores
- Segmentation overlay
- Graph visualization
- Analysis metrics

## Data Structure

Place your TIFF images in:
```
/mnt/d/5TH_SEM/CELLULAR/input/
```

All outputs will be saved to:
```
/mnt/d/5TH_SEM/CELLULAR/output/
├── preprocessed_data.pkl
├── graphs.pkl
├── models/
│   ├── graph_cnn.pth
│   └── metrics.json
└── *.png (visualizations)
```

## Key Features

✅ **Preprocessing**
- Recursive TIFF scanning
- Cellpose segmentation
- Feature extraction (spatial, morphological, intensity)

✅ **Graph Construction**
- Biological graph representation
- Node features: area, perimeter, intensity, etc.
- Edge features: spatial relationships

✅ **Model Training**
- Graph-CNN architecture
- Optional: Graph Attention Network
- Optional: Hybrid CNN+GNN
- Comprehensive evaluation metrics

✅ **Visualization**
- Image overlays
- Compartment maps
- Statistical plots (bar, box, violin)
- Colocalization analysis
- Graph visualization with labels
- Training history plots

✅ **Web Interface**
- Upload TIFF files
- Real-time prediction
- Interactive visualizations

## Troubleshooting

**Issue: CUDA out of memory**
- Reduce batch size in configuration
- Use CPU instead: `device = 'cpu'`

**Issue: Cellpose segmentation too slow**
- Reduce image resolution
- Use GPU acceleration
- Adjust diameter parameter

**Issue: No TIFF files found**
- Check INPUT_DIR path
- Ensure .tif or .tiff extension
- Check file permissions

## Example Workflow

```python
# 1. Preprocess images
from preprocessing import PreprocessingPipeline
pipeline = PreprocessingPipeline()
results = pipeline.process_all()

# 2. Build graphs
from graph import GraphDataset
dataset = GraphDataset("output/preprocessed_data.pkl")
dataset.load_and_build_graphs()

# 3. Train model
from models import GraphCNN, ModelTrainer, prepare_data_loaders
train_loader, test_loader = prepare_data_loaders("output/graphs.pkl")
model = GraphCNN(num_features=6, num_classes=6)
trainer = ModelTrainer(model)
metrics = trainer.train(train_loader, test_loader, num_epochs=100)

# 4. Visualize
from visualization import VisualizationSuite
viz = VisualizationSuite()
viz.generate_all_plots(data_dict)
```

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

## Support

For issues and questions:
- GitHub Issues: https://github.com/soujanyap29/portfolio.github.io/issues
- Email: your.email@example.com

## License

MIT License - See LICENSE file for details
