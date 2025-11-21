# Quick Start Guide

## Installation

### 1. Prerequisites

- Python 3.8 or higher
- Ubuntu or Linux environment
- Optional: CUDA-capable GPU for faster processing

### 2. Setup

```bash
# Navigate to project directory
cd protein-localization

# Run automated setup
bash setup.sh

# Activate virtual environment
source venv/bin/activate
```

### 3. Verify Installation

```bash
# Run structure tests
cd scripts
python test_structure.py
```

## Quick Usage

### Option 1: Web Interface (Easiest)

```bash
# Start Streamlit app
streamlit run frontend/streamlit_app.py
```

Then:
1. Open browser at http://localhost:8501
2. Upload a TIFF file
3. Click "Run Pipeline"
4. View results and download outputs

### Option 2: Command Line

```bash
# Process all TIFF files in input directory
python scripts/pipeline.py \
  --input /mnt/d/5TH_SEM/CELLULAR/input \
  --output /mnt/d/5TH_SEM/CELLULAR/output \
  --model gcn \
  --epochs 50

# With GPU acceleration
python scripts/pipeline.py \
  --input /mnt/d/5TH_SEM/CELLULAR/input \
  --output /mnt/d/5TH_SEM/CELLULAR/output \
  --gpu \
  --model hybrid \
  --epochs 100

# Process limited files for testing
python scripts/pipeline.py \
  --input /mnt/d/5TH_SEM/CELLULAR/input \
  --output /mnt/d/5TH_SEM/CELLULAR/output \
  --max-files 5 \
  --no-train
```

### Option 3: Jupyter Notebook

```bash
# Start JupyterLab
jupyter lab

# Open final_pipeline.ipynb
# Run cells sequentially
```

## Individual Module Usage

### Load TIFF Files

```python
from tiff_loader import TIFFLoader

loader = TIFFLoader("/path/to/tiff/files", recursive=True)
loader.scan_directory()
data = loader.load_all(max_files=10)

for image, metadata in data:
    print(f"Loaded: {metadata['filename']}, Shape: {image.shape}")
```

### Segment and Extract Features

```python
from preprocessing import ImagePreprocessor

preprocessor = ImagePreprocessor(use_gpu=False)
masks, features, info = preprocessor.process_image(
    image,
    output_dir="/path/to/output",
    basename="my_image"
)

print(f"Found {info['n_regions']} regions")
print(f"Features: {list(features.columns)}")
```

### Build Graphs

```python
from graph_construction import GraphConstructor

constructor = GraphConstructor(
    distance_threshold=50,
    k_neighbors=5
)

G = constructor.build_spatial_graph(features, method='knn')
constructor.add_morphological_edges(G, features)

stats = constructor.get_graph_statistics(G)
print(f"Graph: {stats['n_nodes']} nodes, {stats['n_edges']} edges")
```

### Train Model

```python
from model_training import ModelTrainer

trainer = ModelTrainer(model_type='gcn', device='auto')
trainer.create_model(input_dim=20, output_dim=3, hidden_dim=64)

# Assuming you have train_data and val_data
trainer.train(train_data, val_data, epochs=50, lr=0.001)

# Evaluate
metrics = trainer.compute_metrics(test_loader)
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

### Visualize Results

```python
from visualization import Visualizer

visualizer = Visualizer(output_dir="/path/to/output")

# Segmentation overlay
visualizer.plot_segmentation_overlay(image, masks, save_name="seg")

# Feature distributions
visualizer.plot_feature_distributions(features, save_name="features")

# Graph
visualizer.plot_graph(G, save_name="graph")

# Training history
visualizer.plot_training_history(trainer.history, save_name="training")

# Metrics
visualizer.plot_metrics_summary(metrics, save_name="metrics")
```

## Expected Outputs

After running the pipeline, you'll find:

```
/mnt/d/5TH_SEM/CELLULAR/output/
├── features/               # Extracted features (CSV + JSON)
│   ├── image1_features.csv
│   ├── image1_features.json
│   └── ...
├── graphs/                 # Generated graphs
│   ├── image1_graph.pkl
│   └── ...
├── models/                 # Trained models
│   ├── gcn_model.pt
│   ├── gcn_metrics.json
│   └── ...
├── figures/                # All visualizations
│   ├── image1_segmentation.png
│   ├── image1_graph.png
│   ├── gcn_training.png
│   ├── gcn_metrics.png
│   └── ...
├── pipeline_results.json   # Overall results
└── final_pipeline.ipynb    # Complete workflow notebook
```

## Common Issues

### Issue: "Cellpose not found"
**Solution**: The system will use fallback segmentation. To install Cellpose:
```bash
pip install cellpose
```

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size or use CPU:
```bash
python scripts/pipeline.py --no-gpu
```

### Issue: "No TIFF files found"
**Solution**: Check input directory path and ensure TIFF files exist:
```bash
ls -R /mnt/d/5TH_SEM/CELLULAR/input/*.tif*
```

### Issue: "PyTorch Geometric not available"
**Solution**: Install PyG (requires PyTorch first):
```bash
pip install torch-geometric
```

## Performance Tips

1. **Use GPU acceleration** for faster processing
2. **Process in batches** for large datasets
3. **Adjust parameters** based on your specific data:
   - Cell diameter: Typical range 20-50 pixels
   - Distance threshold: 50-100 pixels for graphs
   - K neighbors: 3-7 for most applications

4. **Monitor memory usage** for large images:
   - Use batch processing
   - Process lower resolution first
   - Close unused programs

## Next Steps

- Read [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for detailed documentation
- Explore `final_pipeline.ipynb` for complete examples
- Check `scripts/test_structure.py` for validation
- Customize parameters for your specific data
- Train models with your own labeled data

## Support

- Documentation: `docs/` folder
- Examples: `final_pipeline.ipynb`
- Tests: `scripts/test_structure.py`
- GitHub: https://github.com/soujanyap29/portfolio.github.io
