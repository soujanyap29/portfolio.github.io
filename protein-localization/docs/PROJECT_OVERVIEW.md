# Project Overview

## Protein Sub-Cellular Localization in Neurons

### Executive Summary

This project implements a comprehensive computational neuroscience pipeline for analyzing 4D neuronal TIFF microscopy images and predicting protein sub-cellular localization using state-of-the-art deep learning techniques, including Graph Convolutional Networks (GCNs) and hybrid CNN-GNN architectures.

## System Architecture

### Pipeline Flow

```
TIFF Images → Segmentation → Feature Extraction → Graph Construction → 
Deep Learning Models → Classification → Visualization → Results
```

### Components

1. **Data Loading** (`tiff_loader.py`)
   - Recursive scanning of directories
   - Multi-dimensional TIFF support (2D, 3D, 4D)
   - Multi-channel image handling
   - Metadata extraction

2. **Preprocessing** (`preprocessing.py`)
   - Cell segmentation using Cellpose
   - Fallback traditional segmentation
   - Comprehensive feature extraction
   - Export to CSV and JSON

3. **Graph Construction** (`graph_construction.py`)
   - Spatial relationship graphs
   - K-nearest neighbors
   - Distance-based edges
   - Delaunay triangulation
   - Morphological similarity
   - PyTorch Geometric compatibility
   - DGL compatibility

4. **Model Training** (`model_training.py`)
   - Graph Convolutional Networks
   - CNN feature extractors (VGG-16 style)
   - Hybrid CNN+GNN models
   - GPU support
   - Comprehensive metrics

5. **Visualization** (`visualization.py`)
   - Publication-ready figures
   - Segmentation overlays
   - Feature distributions
   - Graph visualizations
   - Training curves
   - Confusion matrices

6. **Pipeline Integration** (`pipeline.py`)
   - End-to-end automation
   - Command-line interface
   - Progress tracking
   - Result aggregation

7. **Web Interface** (`streamlit_app.py`)
   - Drag-and-drop file upload
   - Interactive parameter tuning
   - Real-time processing
   - Result visualization
   - Download capabilities

## Technical Details

### Segmentation

The system uses Cellpose, a generalist algorithm for cellular segmentation:

- **Model**: cyto2 (default) or nuclei
- **Detection**: Soma, neurites, protein puncta, sub-cellular compartments
- **Fallback**: Otsu thresholding with morphological operations

### Feature Extraction

For each segmented region, the system extracts:

**Spatial Features:**
- Centroid coordinates (x, y)
- Bounding box

**Morphological Features:**
- Area
- Perimeter
- Eccentricity
- Solidity
- Extent
- Orientation
- Major/minor axis lengths
- Convex area
- Equivalent diameter
- Circularity

**Intensity Features:**
- Mean intensity
- Max/min intensity
- Per-channel statistics

**Texture Features (GLCM):**
- Contrast
- Dissimilarity
- Homogeneity
- Energy
- Correlation

### Graph Construction

**Node Representation:**
Each node represents:
- A segmented region (protein puncta, vesicle, compartment)
- All extracted features as node attributes

**Edge Types:**

1. **Spatial Edges**:
   - K-nearest neighbors in coordinate space
   - Distance threshold-based
   - Delaunay triangulation

2. **Morphological Edges**:
   - Based on feature similarity
   - Normalized feature vectors
   - Similarity threshold

**Graph Formats:**
- NetworkX (default)
- PyTorch Geometric Data objects
- DGL graphs
- Export: pickle, GML, GraphML, JSON

### Deep Learning Models

#### 1. Graph Convolutional Network (GCN)

```
Input Features → GCN Layer 1 → ReLU → Dropout → 
GCN Layer 2 → Global Pooling → Fully Connected → 
Log Softmax → Classification
```

**Architecture:**
- Input: Node features (variable dimension)
- Hidden: 64-256 units
- Layers: 2-4 GCN layers
- Dropout: 0.5
- Output: Number of classes

#### 2. CNN Feature Extractor

```
Input Image → Conv Blocks → Max Pooling → 
Adaptive Pooling → Fully Connected → Features
```

**Architecture (VGG-16 inspired):**
- Conv1: 64 filters
- Conv2: 128 filters
- Conv3: 256 filters
- Conv4: 512 filters
- FC: 512 units

#### 3. Hybrid CNN-GNN

```
Image → CNN Features ─┐
                      ├→ Fusion → Classification
Graph → GNN Features ─┘
```

**Fusion Strategy:**
- Concatenate CNN and GNN feature vectors
- Fully connected fusion layer
- Joint classification

### Training

**Configuration:**
- Optimizer: Adam
- Learning rate: 0.001 (default)
- Batch size: 8-32
- Epochs: 50-100
- Loss: Negative Log Likelihood
- Device: Auto-detect (CUDA if available)

**Data Splitting:**
- Training: 64%
- Validation: 16%
- Testing: 20%

**Checkpointing:**
- Save best model based on validation accuracy
- Store optimizer state
- Save training history

### Evaluation Metrics

The system computes:

1. **Accuracy**: Overall correctness
2. **Precision**: True positives / (True positives + False positives)
3. **Recall**: True positives / (True positives + False negatives)
4. **F1-Score**: Harmonic mean of precision and recall
5. **Specificity**: True negatives / (True negatives + False positives)
6. **Confusion Matrix**: Detailed per-class performance
7. **ROC/AUC** (optional): Receiver Operating Characteristic curves

## Data Flow

### Input

```
/mnt/d/5TH_SEM/CELLULAR/input/
├── experiment1/
│   ├── image1.tif
│   ├── image2.tif
│   └── ...
├── experiment2/
│   └── ...
└── ...
```

### Output

```
/mnt/d/5TH_SEM/CELLULAR/output/
├── features/
│   ├── image1_features.csv
│   ├── image1_features.json
│   └── ...
├── graphs/
│   ├── image1_graph.pkl
│   └── ...
├── models/
│   ├── gcn_model.pt
│   ├── hybrid_model.pt
│   ├── gcn_metrics.json
│   └── ...
├── figures/
│   ├── image1_segmentation.png
│   ├── image1_compartments.png
│   ├── image1_graph.png
│   ├── gcn_training.png
│   ├── gcn_metrics.png
│   └── ...
├── pipeline_results.json
└── final_pipeline.ipynb
```

## Usage Modes

### 1. Web Interface

Best for:
- Interactive exploration
- Single image analysis
- Parameter tuning
- Quick visualization

### 2. Command Line

Best for:
- Batch processing
- Automated workflows
- Large datasets
- HPC environments

### 3. Jupyter Notebook

Best for:
- Learning and tutorials
- Custom analysis
- Research documentation
- Reproducible science

### 4. Python API

Best for:
- Integration with other tools
- Custom pipelines
- Programmatic control
- Advanced users

## Performance Characteristics

### Memory Requirements

- Small images (<1024x1024): 2-4 GB RAM
- Medium images (1024-2048): 4-8 GB RAM
- Large images (>2048): 8-16 GB RAM
- GPU: 4-8 GB VRAM recommended

### Processing Time

Approximate times per image (CPU, no GPU):
- Loading: 1-5 seconds
- Segmentation: 10-60 seconds
- Feature extraction: 5-15 seconds
- Graph construction: 1-5 seconds
- Visualization: 5-10 seconds

With GPU:
- Segmentation: 2-10 seconds (5-10x faster)
- Model training: 10-100x faster

### Scalability

- Handles 100s of images in batch mode
- Memory-efficient batch loading
- Parallel processing support
- Distributed training capable

## Scientific Applications

This pipeline enables research in:

1. **Neuroscience**
   - Synaptic protein localization
   - Axonal transport studies
   - Dendritic spine analysis
   - Neurotransmitter distribution

2. **Cell Biology**
   - Organelle tracking
   - Protein trafficking
   - Colocalization studies
   - Subcellular compartmentalization

3. **Drug Discovery**
   - Target validation
   - Drug distribution
   - Cellular response profiling
   - High-throughput screening

4. **Disease Modeling**
   - Protein mislocalization in disease
   - Neurodegeneration studies
   - Developmental disorders
   - Cellular dysfunction

## Extensions and Customization

### Adding New Features

1. Edit `preprocessing.py`
2. Add feature extraction in `_extract_region_features()`
3. Features automatically propagate to downstream analysis

### Custom Models

1. Implement model in `model_training.py`
2. Inherit from `nn.Module`
3. Add to `ModelTrainer.create_model()`

### New Visualizations

1. Add method to `Visualizer` class
2. Call from pipeline or notebook
3. Follows matplotlib/seaborn conventions

### Integration

The pipeline can integrate with:
- CellProfiler pipelines
- ImageJ/Fiji macros
- OMERO image databases
- Custom analysis tools

## Limitations and Future Work

### Current Limitations

1. **Segmentation**: Best for well-separated cells
2. **3D Analysis**: Currently uses 2D projections
3. **Time Series**: Limited 4D support
4. **Labels**: Requires labeled data for supervised training

### Planned Enhancements

1. Full 3D segmentation and analysis
2. Time-series tracking
3. Semi-supervised and unsupervised methods
4. Multi-modal imaging support
5. Cloud deployment
6. Real-time processing

## Dependencies

### Core

- Python 3.8+
- NumPy, Pandas, SciPy
- PyTorch
- scikit-image, OpenCV

### Optional

- Cellpose (segmentation)
- PyTorch Geometric (GNN)
- DGL (alternative GNN)
- CUDA (GPU acceleration)

### Visualization

- Matplotlib, Seaborn, Plotly

### Interface

- Streamlit
- Jupyter

## Validation and Testing

The system includes:
- Unit tests for each module
- Integration tests for pipeline
- Example data and notebooks
- Continuous validation

Run tests:
```bash
python scripts/test_structure.py
```

## Citation

If you use this pipeline, please cite:

```bibtex
@software{protein_localization_2024,
  title={Protein Sub-Cellular Localization in Neurons: 
         A Graph Neural Network Approach},
  author={Your Name},
  year={2024},
  url={https://github.com/soujanyap29/portfolio.github.io}
}
```

## License

MIT License - See LICENSE file for details

## Support and Community

- **Documentation**: This folder
- **Examples**: Jupyter notebooks
- **Issues**: GitHub issue tracker
- **Contributions**: Pull requests welcome

## References

1. Stringer, C., et al. (2021). Cellpose: a generalist algorithm for cellular segmentation. Nature Methods.
2. Kipf, T.N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR.
3. He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
4. Fey, M., & Lenssen, J.E. (2019). Fast Graph Representation Learning with PyTorch Geometric. ICLR Workshop.
