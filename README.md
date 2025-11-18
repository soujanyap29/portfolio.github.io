# Protein Sub-Cellular Localization in Neurons

## Overview

This project implements a complete competition-ready pipeline for predicting the sub-cellular localization of proteins in neurons using Graph Convolutional Neural Networks (Graph-CNN) and image processing techniques.

## Features

- **Recursive TIFF Loading**: Automatically scans and loads 4D TIFF images from nested subdirectories
- **Image Preprocessing**: Segments cellular structures using advanced thresholding and morphological operations
- **Graph Construction**: Converts segmented images into graph representations with nodes representing protein locations
- **Deep Learning**: Trains Graph-CNN models for protein localization classification
- **Hybrid Models**: Optional VGG-16 feature extraction combined with Graph-CNN
- **Visualization**: Generates intuitive visualizations of predictions and graph structures
- **Web Interface**: User-friendly front-end for uploading images and viewing results

## Project Structure

```
protein-localization/
├── scripts/
│   ├── tiff_loader.py           # TIFF file loading
│   ├── preprocessing.py         # Image segmentation and feature extraction
│   ├── graph_construction.py    # Graph creation from segmented images
│   ├── model_training.py        # Graph-CNN and Hybrid CNN models
│   ├── visualization.py         # Graph and result visualization
│   └── pipeline.py              # Complete integrated pipeline
├── frontend/
│   ├── index.html               # Web interface
│   ├── style.css                # Styling
│   └── app.js                   # Frontend logic
├── models/                      # Trained models (created during training)
├── output/                      # Pipeline outputs (created during training)
├── docs/                        # Additional documentation
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Setup

1. **Clone or download this project**

2. **Install Python dependencies**:
   ```bash
   cd protein-localization
   pip install -r requirements.txt
   ```

3. **For PyTorch Geometric**, you may need to install additional dependencies based on your system:
   ```bash
   # For CPU only
   pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cpu.html
   
   # For CUDA 11.3
   pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
   ```

## Usage

### Option 1: Jupyter Notebook (Recommended for Interactive Use) 🆕

**No need to copy-paste code!** Use the provided Jupyter notebook for interactive exploration:

1. **Open the notebook**:
   ```bash
   jupyter notebook Protein_Localization_Pipeline.ipynb
   ```

2. **Run cells step-by-step** or use the complete pipeline in one go

3. The notebook includes:
   - Easy import of all modules
   - Step-by-step execution with visualizations
   - Demo with synthetic data
   - Training and prediction examples

### Option 2: Command Line (For Batch Processing)

To run the entire pipeline from loading images to training and visualization:

```bash
cd scripts
python pipeline.py --input /path/to/input/directory --output /path/to/output/directory --epochs 50
```

**Arguments:**
- `--input`: Directory containing TIFF files (default: `D:\5TH_SEM\CELLULAR\input`)
- `--output`: Directory for saving outputs (default: `D:\5TH_SEM\CELLULAR\output`)
- `--epochs`: Number of training epochs (default: 50)
- `--max-files`: Maximum number of files to process (default: all files in all subdirectories)

**Examples:**

```bash
# Process ALL TIFF files from all protein folders (AAMP_*, AATF_*, etc.)
python pipeline.py --input D:\5TH_SEM\CELLULAR\input --output D:\5TH_SEM\CELLULAR\output --epochs 20

# Process first 50 files only (for testing)
python pipeline.py --input D:\5TH_SEM\CELLULAR\input --output D:\5TH_SEM\CELLULAR\output --max-files 50
```

**Note:** The pipeline now processes **ALL TIFF files** from **ALL subdirectories** by default. Your directory structure with multiple protein folders (AAMP_ENSG00000127837, AATF_ENSG00000275700, etc.) is fully supported!

### Option 3: Import as Python Modules

#### 1. Load TIFF Files

```python
from tiff_loader import TIFFLoader

loader = TIFFLoader("path/to/input")
tiff_files = loader.scan_directory()
images = loader.load_all_tiffs()
```

#### 2. Preprocess Images

```python
from preprocessing import ImagePreprocessor

preprocessor = ImagePreprocessor()
labeled_regions, features = preprocessor.process_image(image)
```

#### 3. Construct Graphs

```python
from graph_construction import GraphConstructor

constructor = GraphConstructor()
graph = constructor.create_graph_from_regions(features)
constructor.save_graph(graph, "output.gml")
```

#### 4. Train Model

```python
from model_training import GraphCNN, ModelTrainer

model = GraphCNN(num_features=4, num_classes=5)
trainer = ModelTrainer(model)
trainer.setup_training(learning_rate=0.001)
history = trainer.train(train_loader, test_loader, num_epochs=50)
trainer.save_model("model.pt")
```

#### 5. Visualize Results

```python
from visualization import GraphVisualizer

visualizer = GraphVisualizer()
visualizer.visualize_graph(graph, predictions, save_path="graph.png")

# Or generate a clean biological network diagram
visualizer.visualize_biological_network(
    graph,
    central_node=0,  # Highlight node 0 as central hub
    save_path="bio_network.png"
)
```

### Biological Network Diagram Generator 🆕

Generate clean, scientific-style biological network diagrams from **any TIFF image you provide**:

**Process YOUR TIFF Images (Recommended)**:
```bash
cd scripts
# Process any TIFF file - automatically segments, builds graph, and visualizes
python generate_biological_network.py --input /path/to/your/image.tif --output network.png

# Works with all TIFF formats: 2D, 3D, 4D, multi-page
python generate_biological_network.py --input "D:\5TH_SEM\CELLULAR\input\sample.tif" --output network.png
```

**Web Interface (Demo Only - Synthetic Networks)**:
```bash
cd frontend
python -m http.server 8000
# Open http://localhost:8000/biological_network_generator.html
# Note: Web version generates synthetic networks for quick demos
# For real TIFF processing, use the Python script above
```

**How It Works**:
1. Loads your TIFF image (any format/size)
2. Segments cellular structures automatically
3. Builds graph network from segmented regions
4. Generates diagram with requested aesthetic

**Features**:
- Processes **any TIFF image** you provide as input
- Soft grey rounded rectangle nodes with distinct blue central hub
- Thin, light-grey curved connection lines
- Translucent cluster groupings
- Minimal, scientific, bioinformatics-style aesthetic
- Soft shadows and light grey background

### Using the Web Interface

1. **Start a local web server** in the `frontend` directory:
   ```bash
   cd frontend
   python -m http.server 8000
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:8000
   ```

3. **Upload a TIFF file** using the drag-and-drop interface or file browser

4. **View results** including:
   - Predicted protein localization class
   - Confidence scores
   - Segmented regions
   - Graph representation
   - Feature distributions
   - Classification probabilities

## Input Data Format

The pipeline expects 4D TIFF images with the following characteristics:
- **Format**: TIFF (.tif or .tiff)
- **Dimensions**: Time × Z-stack × Height × Width (4D) or Height × Width (2D)
- **Organization**: Files can be organized in any subdirectory structure

Example directory structure:
```
input/
├── experiment1/
│   ├── image1.tif
│   └── image2.tif
├── experiment2/
│   ├── subfolder/
│   │   └── image3.tif
│   └── image4.tif
└── image5.tif
```

## Output

The pipeline generates the following outputs in the specified output directory:

### Directory Structure
```
output/
├── graphs/              # Saved graph structures (.gml format)
├── models/              # Trained models (.pt format)
├── visualizations/      # PNG images of graphs and results
└── data/                # Processed features and intermediate data
```

### Generated Files
- **Graphs**: `graph_0.gml`, `graph_1.gml`, ...
- **Model**: `graph_cnn.pt`
- **Visualizations**:
  - `training_history.png`: Loss and accuracy curves
  - `graph_0_prediction.png`, `graph_1_prediction.png`, ...: Graph visualizations with predictions
  - `confusion_matrix.png`: Model performance matrix

## Model Architecture

### Graph-CNN

The Graph Convolutional Neural Network consists of:
- 3 Graph Convolutional layers (GCN)
- Global mean pooling
- 2 Fully connected layers
- Dropout for regularization

**Node Features:**
- Area of segmented region
- Mean intensity
- Eccentricity
- Solidity

**Classes:**
1. Nucleus
2. Mitochondria
3. Endoplasmic Reticulum
4. Golgi Apparatus
5. Cytoplasm

### Hybrid CNN (Optional)

Combines:
- VGG-16 style convolutional layers for image feature extraction
- Graph-CNN for graph-based features
- Fusion layer combining both feature types

## Performance

Expected performance metrics:
- **Accuracy**: 75-90% (depending on data quality and quantity)
- **Training Time**: 5-15 minutes for 50 epochs (GPU)
- **Inference Time**: < 1 second per image

## Troubleshooting

### Common Issues

1. **PyTorch Geometric installation fails**
   - Install torch-scatter and torch-sparse separately
   - Check PyTorch version compatibility

2. **Out of memory errors**
   - Reduce batch size
   - Use smaller images
   - Enable CPU mode if GPU memory is limited

3. **TIFF files not loading**
   - Verify file format (.tif or .tiff)
   - Check file permissions
   - Ensure tifffile package is installed

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{protein_localization_2025,
  title={Protein Sub-Cellular Localization Pipeline},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/protein-localization}
}
```

## License

This project is provided as-is for educational and research purposes.

## Contact

For questions or issues, please contact: [your-email@example.com]

## Acknowledgments

- Cellpose for cell segmentation inspiration
- PyTorch Geometric for graph neural network implementations
- scikit-image for image processing utilities

## Future Improvements

- [ ] Integration with actual Cellpose models
- [ ] Support for 3D visualization
- [ ] Real-time prediction API
- [ ] Batch processing interface
- [ ] Model ensemble techniques
- [ ] Transfer learning from pre-trained models
- [ ] Multi-GPU training support
- [ ] Cloud deployment options

## Version History

- **v1.0.0** (2025-01): Initial release with core functionality
  - TIFF loading
  - Graph construction
  - Graph-CNN training
  - Web interface
  - Visualization tools
