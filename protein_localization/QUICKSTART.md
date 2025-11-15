# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### 1. Installation

```bash
cd protein_localization
pip install -r requirements.txt
```

### 2. Process Your First Image

#### Option A: Command Line

```bash
# Process a single TIFF file
python main.py process --input /path/to/your/image.tif --output ./output

# Process entire directory
python main.py process --input /mnt/d/5TH_SEM/CELLULAR/input --output ./output

# Limit to first 10 files
python main.py process --input /mnt/d/5TH_SEM/CELLULAR/input --output ./output --max-files 10
```

#### Option B: Web Interface

```bash
# Launch Gradio app
python main.py interface

# Then open: http://localhost:7860
```

#### Option C: Jupyter Notebook

```bash
# Open the complete notebook
python main.py notebook

# Or directly:
jupyter lab notebooks/final_pipeline.ipynb
```

### 3. Python API

```python
from preprocessing.segmentation import TIFFLoader, CellposeSegmenter
from preprocessing.feature_extraction import FeatureExtractor
from graph_construction.graph_builder import GraphConstructor

# Load image
loader = TIFFLoader()
image = loader.load_tiff("path/to/image.tif")

# Segment
segmenter = CellposeSegmenter()
masks, info = segmenter.segment_image(image)
print(f"Found {info['num_cells']} cells")

# Extract features
extractor = FeatureExtractor()
features = extractor.extract_all_features(image, masks)
print(f"Extracted {len(features.columns)} features")

# Build graph
constructor = GraphConstructor()
graph = constructor.construct_graph(features, masks)
print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
```

## 📊 Example Workflows

### Workflow 1: Basic Processing

```python
import os
from preprocessing.segmentation import DirectoryHandler, TIFFLoader, CellposeSegmenter
from preprocessing.feature_extraction import FeatureExtractor, FeatureStorage

# Setup
INPUT_DIR = "/mnt/d/5TH_SEM/CELLULAR/input"
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find all TIFF files
handler = DirectoryHandler(INPUT_DIR)
tiff_files = handler.scan_directory()

# Process each file
loader = TIFFLoader()
segmenter = CellposeSegmenter()
extractor = FeatureExtractor()
storage = FeatureStorage(OUTPUT_DIR)

for tiff_file in tiff_files[:5]:  # First 5 files
    print(f"Processing: {tiff_file}")
    
    # Load and segment
    image = loader.load_tiff(tiff_file)
    masks, _ = segmenter.segment_image(image)
    
    # Extract and save features
    features = extractor.extract_all_features(image, masks)
    filename = os.path.basename(tiff_file).replace('.tif', '')
    storage.save_features(features, filename)
    
    print(f"  Saved features: {len(features)} regions")
```

### Workflow 2: Visualization

```python
from visualization.plotters import SegmentationVisualizer, StatisticalPlotter
from visualization.graph_viz import GraphVisualizer

# Setup visualizers
seg_viz = SegmentationVisualizer(output_dir="./visualizations")
stat_viz = StatisticalPlotter(output_dir="./visualizations")
graph_viz = GraphVisualizer(output_dir="./visualizations")

# Segmentation overlay
seg_viz.plot_segmentation_overlay(
    image, masks,
    title="Neuronal Segmentation",
    filename="segmentation.png"
)

# Graph visualization
graph_viz.plot_graph(
    graph,
    title="Protein Localization Graph",
    filename="graph.png"
)

# Statistical plots
data = {
    'Soma': features[features['area'] > 500]['mean_intensity'].tolist(),
    'Dendrites': features[features['area'] < 500]['mean_intensity'].tolist()
}

stat_viz.plot_grouped_bar(
    data,
    title="Intensity by Compartment",
    ylabel="Mean Intensity",
    filename="intensity_comparison.png"
)
```

### Workflow 3: Model Training

```python
import torch
from models.graph_cnn import GraphCNN
from models.trainer import ModelTrainer, ProteinLocalizationDataset, create_data_loaders

# Prepare data (assuming you have preprocessed data)
# images = list of image tensors
# graphs = list of graph data objects
# labels = list of labels

dataset = ProteinLocalizationDataset(images, graphs, labels)
train_loader, val_loader = create_data_loaders(dataset, train_split=0.8, batch_size=16)

# Create model
model = GraphCNN(
    in_channels=20,
    hidden_channels=64,
    out_channels=10,
    num_layers=3
)

# Train
trainer = ModelTrainer(model, learning_rate=0.001)
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    save_dir="./models"
)
```

### Workflow 4: Inference

```python
from graph_construction.graph_builder import PyTorchGeometricConverter

# Load trained model
model = GraphCNN(in_channels=20, hidden_channels=64, out_channels=10)
model.load_state_dict(torch.load("./models/best_model.pth")['model_state_dict'])
model.eval()

# Process new image
image = loader.load_tiff("new_image.tif")
masks, _ = segmenter.segment_image(image)
features = extractor.extract_all_features(image, masks)
graph = constructor.construct_graph(features, masks)

# Convert to tensor
converter = PyTorchGeometricConverter()
graph_data = converter.to_pytorch_geometric(graph)

# Predict
with torch.no_grad():
    output = model(graph_data['x'], graph_data['edge_index'])
    prediction = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1).max().item()

print(f"Predicted class: {prediction}")
print(f"Confidence: {confidence:.2%}")
```

## 🎯 Common Use Cases

### Use Case 1: Batch Processing
```bash
# Process all files in a directory
python main.py process --input /data/microscopy --output /data/results
```

### Use Case 2: Interactive Analysis
```bash
# Launch web interface for manual analysis
python main.py interface
```

### Use Case 3: Custom Pipeline
```python
# Create custom processing pipeline
from preprocessing import *
from graph_construction import *
from models import *

# Your custom code here
```

## 📝 Configuration

Edit `config.py` to customize pipeline behavior:

```python
# Paths
INPUT_DIR = "/your/input/path"
OUTPUT_DIR = "/your/output/path"

# Segmentation
CELLPOSE_MODEL = 'cyto2'  # or 'nuclei', 'cyto'
CELLPOSE_DIAMETER = None  # Auto-detect

# Graph construction
PROXIMITY_THRESHOLD = 50  # pixels
MAX_EDGES_PER_NODE = 10

# Training
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
```

## 🔧 Troubleshooting

### Issue: Out of Memory
```python
# Reduce batch size in config.py
BATCH_SIZE = 8  # Instead of 32
```

### Issue: Slow Processing
```python
# Use GPU if available
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# Or process fewer files
handler.scan_directory()[:10]  # First 10 only
```

### Issue: Segmentation Failed
```python
# Try different Cellpose model
segmenter = CellposeSegmenter(model_type='nuclei')  # Instead of 'cyto2'

# Or adjust diameter
segmenter = CellposeSegmenter(diameter=30)
```

## 📚 Next Steps

1. **Explore the Notebook**: `notebooks/final_pipeline.ipynb`
2. **Read the Full README**: `README.md`
3. **Try the Web Interface**: `python main.py interface`
4. **Customize the Pipeline**: Edit modules in each subdirectory

## 🆘 Getting Help

- Check the README.md for detailed documentation
- Open an issue on GitHub
- Review the example notebook

---

Happy analyzing! 🔬
