# Complete System Documentation

## Architecture Overview

### System Components

```
Input TIFF Files
      ↓
[Preprocessing Module]
  - Load 4D TIFF
  - Cellpose Segmentation
  - Feature Extraction
      ↓
[Graph Construction Module]
  - Node Creation (regions)
  - Edge Creation (proximity)
  - Attribute Assignment
      ↓
[Model Training Module]
  - Graph-CNN
  - VGG-16
  - Hybrid Models
      ↓
[Visualization Module]
  - Scientific Plots
  - Graph Visualizations
      ↓
[Frontend Module]
  - Web Interface
  - File Upload
  - Results Display
```

## Module Details

### 1. Preprocessing Module (`preprocessing/__init__.py`)

**Purpose**: Load and segment TIFF images, extract features

**Key Classes**:
- `TIFFProcessor`: Main processor class

**Key Methods**:
- `load_tiff(file_path)`: Load TIFF file (2D/3D/4D support)
- `segment_cellpose(img)`: Segment using Cellpose
- `extract_features(img, masks)`: Extract region features
- `process_single_tiff(file_path)`: Complete pipeline for one file

**Features Extracted**:
- Spatial: centroids, coordinates
- Morphological: area, perimeter, eccentricity
- Intensity: mean, max, min, std

**Dependencies**: tifffile, cellpose, scikit-image, numpy

### 2. Graph Construction Module (`graph_construction/__init__.py`)

**Purpose**: Build biological graphs from segmented regions

**Key Classes**:
- `BiologicalGraphBuilder`: Graph builder class

**Key Methods**:
- `build_graph(features)`: Build NetworkX graph
- `_add_proximity_edges(G, features)`: Add spatial edges
- `graph_to_pytorch_geometric(G, labels)`: Convert to PyTorch Geometric
- `save_graph(G, path)`: Save graph to file
- `load_graph(path)`: Load graph from file

**Graph Structure**:
- **Nodes**: Represent puncta/compartments
- **Node Attributes**: All extracted features
- **Edges**: Spatial proximity (< threshold distance)
- **Edge Attributes**: distance, intensity_similarity

**Dependencies**: networkx, scipy, torch, torch-geometric

### 3. Models Module (`models/__init__.py`)

**Purpose**: Deep learning models for classification

**Key Classes**:

#### `GraphCNN`
- Graph Convolutional Network
- 3 GCN layers + 2 FC layers
- Global mean pooling
- Dropout for regularization

#### `VGG16Classifier`
- Pre-trained VGG-16
- Modified classifier head
- Transfer learning ready

#### `HybridModel`
- Combined CNN + Graph-CNN
- Parallel feature extraction
- Late fusion architecture

#### `ModelTrainer`
- Training loop management
- Evaluation metrics
- Model checkpointing

**Dependencies**: torch, torch-geometric, torchvision

### 4. Visualization Module (`visualization/__init__.py`)

**Purpose**: Scientific publication-quality visualizations

**Key Classes**:
- `ScientificVisualizer`: Main visualizer

**Available Plots**:

1. **Segmentation Overlay**
   - Original image, masks, overlay
   - 3-panel layout
   
2. **Compartment Map**
   - Color-coded regions
   - Colorbar with labels
   
3. **Grouped Bar Plot**
   - Mean ± SEM
   - Multiple groups
   
4. **Box/Violin Plots**
   - Distribution visualization
   - Side-by-side comparison
   
5. **Colocalization Scatter**
   - Hexbin density plot
   - Pearson correlation
   
6. **Intensity Profile**
   - Distance-based analysis
   - Moving average trend
   
7. **Graph Visualization**
   - Spring layout
   - Node size/color by attributes
   - Edge transparency
   
8. **Confusion Matrix**
   - Heatmap style
   - Annotated counts

**Style Settings**:
- DPI: 300 (publication quality)
- Figure size: Configurable
- Color schemes: Scientific palettes
- Fonts: Sans-serif, bold labels

**Dependencies**: matplotlib, seaborn, networkx, sklearn

### 5. Frontend Module (`frontend/app.py`)

**Purpose**: Web interface for easy access

**Technology Stack**:
- **Backend**: Flask
- **Frontend**: HTML5 + CSS3 + JavaScript
- **API**: RESTful endpoints

**Endpoints**:
- `GET /`: Main page
- `POST /upload`: File upload and processing
- `GET /health`: Health check

**Features**:
- Drag-and-drop upload
- Real-time progress
- Interactive results
- Base64 image encoding
- Error handling

**Dependencies**: flask, flask-cors

### 6. Utils Module (`utils/__init__.py`)

**Purpose**: Shared utility functions

**Key Functions**:
- `load_config(path)`: Load YAML configuration
- `ensure_dir(path)`: Create directories
- `get_tiff_files(dir, recursive)`: Find TIFF files
- `validate_tiff_file(path)`: Validate TIFF

## Configuration System

The `config.yaml` file controls all system parameters:

```yaml
# Data paths
data:
  input_dir: "/path/to/input"
  output_dir: "/path/to/output"

# Segmentation parameters
segmentation:
  model_type: "cyto"
  diameter: 30
  channels: [0, 0]
  flow_threshold: 0.4
  cellprob_threshold: 0.0

# Graph construction
graph:
  proximity_threshold: 50
  min_node_size: 10
  edge_features: ["distance", "intensity_similarity"]

# Model training
training:
  model_type: "graph_cnn"
  batch_size: 16
  epochs: 100
  learning_rate: 0.001
  validation_split: 0.2
  test_split: 0.1

# Visualization
visualization:
  dpi: 300
  figure_size: [10, 8]
  colormap: "viridis"
  save_formats: ["png", "pdf"]

# Frontend
frontend:
  host: "0.0.0.0"
  port: 5000
  max_upload_size_mb: 500
  allowed_extensions: [".tif", ".tiff"]
```

## Complete Pipeline Flow

### 1. Data Input
```python
# Scan directory for TIFF files
tiff_files = get_tiff_files(input_dir, recursive=True)
```

### 2. Preprocessing
```python
processor = TIFFProcessor(config)
for tiff_file in tiff_files:
    img, masks, features = processor.process_single_tiff(tiff_file)
```

### 3. Graph Construction
```python
graph_builder = BiologicalGraphBuilder(config)
G = graph_builder.build_graph(features)
graph_data = graph_builder.graph_to_pytorch_geometric(G, labels)
```

### 4. Model Training
```python
model = GraphCNN(num_features, num_classes)
trainer = ModelTrainer(model, config)

for epoch in range(epochs):
    loss, acc = trainer.train_epoch(train_loader)
    
trainer.save_model('model.pth')
```

### 5. Evaluation
```python
acc, preds, labels = trainer.evaluate(test_loader)
precision, recall, f1, _ = precision_recall_fscore_support(labels, preds)
```

### 6. Visualization
```python
visualizer = ScientificVisualizer(config)
visualizer.plot_segmentation_overlay(img, masks, 'output.png')
visualizer.plot_graph_visualization(G, 'graph.png')
visualizer.plot_confusion_matrix(labels, preds, class_names, 'cm.png')
```

### 7. Prediction
```python
# Load model
trainer.load_model('model.pth')

# Process new image
img, masks, features = processor.process_single_tiff('new_image.tif')
G = graph_builder.build_graph(features)
graph_data = graph_builder.graph_to_pytorch_geometric(G)

# Predict
model.eval()
with torch.no_grad():
    out = model(graph_data.x, graph_data.edge_index)
    pred_class = out.argmax(dim=1).item()
```

## Performance Optimization

### GPU Acceleration
```python
# Enable CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
```

### Batch Processing
```python
# Use DataLoader for efficient batching
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
```

### Memory Management
```python
# Clear cache periodically
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

## Testing

### Unit Tests (example)
```python
def test_tiff_loading():
    processor = TIFFProcessor(config)
    img = processor.load_tiff('test.tif')
    assert img is not None
    assert img.ndim >= 2

def test_graph_construction():
    builder = BiologicalGraphBuilder(config)
    G = builder.build_graph(features)
    assert G.number_of_nodes() > 0
    assert G.number_of_edges() >= 0
```

## Deployment

### Local Deployment
```bash
python frontend/app.py
```

### Docker Deployment (example)
```dockerfile
FROM python:3.8
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "frontend/app.py"]
```

## Troubleshooting Guide

### Issue: Cellpose not segmenting
**Solution**: Adjust diameter and threshold parameters

### Issue: Out of memory
**Solution**: Reduce batch size or use CPU

### Issue: Slow processing
**Solution**: Enable GPU, reduce image size, or use fewer epochs

### Issue: Poor model performance
**Solution**: Collect more data, tune hyperparameters, try different architectures

## API Reference

See individual module docstrings for detailed API documentation.

## Contributing Guidelines

1. Follow PEP 8 style guide
2. Add docstrings to all functions/classes
3. Write unit tests for new features
4. Update documentation
5. Submit pull requests with clear descriptions

## Version History

- **v1.0.0** (2025-11): Initial release
  - Complete pipeline implementation
  - Web interface
  - Jupyter notebook
  - Documentation

## References

1. Cellpose: Stringer et al., Nature Methods 2021
2. Graph Neural Networks: Kipf & Welling, ICLR 2017
3. VGG-16: Simonyan & Zisserman, ICLR 2015

---

For more information, see README.md or contact the development team.
