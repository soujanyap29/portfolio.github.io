# Protein Sub-Cellular Localization in Neurons

A comprehensive computational neuroscience pipeline for processing 4D neuronal TIFF microscopy images and predicting protein sub-cellular localization using deep learning and graph neural networks.

## Overview

This system provides an end-to-end solution for:
- Loading and processing multi-dimensional TIFF microscopy images
- Automated cell segmentation using Cellpose
- Feature extraction and graph construction
- Deep learning models (CNN + GNN hybrid)
- Interactive visualization and analysis
- Web-based user interface for easy interaction

## Features

- **Automated TIFF Processing**: Recursively scan and load TIFF files from multiple folders
- **Advanced Segmentation**: Detect neuronal soma, neurites, protein puncta, and sub-cellular compartments
- **Graph Neural Networks**: Convert segmented images into biological graphs for GNN analysis
- **Hybrid Deep Learning**: Combine CNN and GNN architectures for enhanced accuracy
- **Rich Visualizations**: Generate publication-ready figures and analytics
- **Interactive Interface**: Upload and analyze images through a Streamlit web app
- **Complete Documentation**: Jupyter notebooks with step-by-step walkthroughs

## Directory Structure

```
protein-localization/
│
├── scripts/                    # Core Python modules
│   ├── tiff_loader.py         # TIFF file loading
│   ├── preprocessing.py       # Image segmentation & feature extraction
│   ├── graph_construction.py  # Graph creation
│   ├── model_training.py      # Deep learning models
│   ├── visualization.py       # Plots and analytics
│   ├── pipeline.py            # End-to-end pipeline
│   └── test_structure.py      # Verification scripts
│
├── frontend/                   # Web interface
│   └── streamlit_app.py       # Streamlit application
│
├── docs/                       # Documentation
│   ├── QUICKSTART.md          # Quick start guide
│   └── PROJECT_OVERVIEW.md    # Detailed project overview
│
├── models/                     # Trained model files
├── output/                     # Generated outputs
├── requirements.txt            # Python dependencies
├── setup.sh                    # Automated setup script
└── README.md                   # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for deep learning)
- Ubuntu or Linux environment

### Quick Setup

1. Clone the repository:
```bash
git clone https://github.com/soujanyap29/portfolio.github.io.git
cd portfolio.github.io/protein-localization
```

2. Run the setup script:
```bash
bash setup.sh
```

3. Activate the virtual environment:
```bash
source venv/bin/activate
```

## Usage

### Web Interface

Launch the Streamlit application:
```bash
streamlit run frontend/streamlit_app.py
```

Then:
1. Upload a TIFF file
2. Click "Run Pipeline"
3. View results, metrics, and visualizations
4. Download outputs

### Command Line

Run the complete pipeline:
```bash
python scripts/pipeline.py --input /path/to/tiff/files --output /path/to/output
```

### Jupyter Notebook

For a complete walkthrough:
```bash
jupyter lab output/final_pipeline.ipynb
```

## Input Data

The system expects TIFF microscopy images stored in:
```
/mnt/d/5TH_SEM/CELLULAR/input
```

The loader will recursively scan all sub-folders for `.tif` and `.tiff` files.

## Output

All outputs are saved to:
```
/mnt/d/5TH_SEM/CELLULAR/output
```

Including:
- Trained models (`/models`)
- Visualizations (`/figures`)
- Feature tables (CSV/JSON)
- Graph files
- Evaluation metrics
- Research manuscript
- Final pipeline notebook

## Model Architecture

The system implements a hybrid architecture:
1. **Segmentation**: Cellpose for cell and compartment detection
2. **CNN**: VGG-16 for feature extraction
3. **Graph Construction**: Build spatial relationship graphs
4. **GNN**: Graph Convolutional Networks for classification
5. **Fusion**: Combine CNN and GNN predictions

## Metrics

The system computes comprehensive metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Specificity
- Confusion Matrix
- ROC curves and AUC

## Citation

If you use this pipeline in your research, please cite:

```
@software{protein_localization_2024,
  title={Protein Sub-Cellular Localization in Neurons},
  author={Your Name},
  year={2024},
  url={https://github.com/soujanyap29/portfolio.github.io}
}
```

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- Open an issue on GitHub
- See documentation in `docs/`
- Check the Jupyter notebook for examples

## Acknowledgments

This project uses:
- Cellpose for segmentation
- PyTorch Geometric for GNN
- Streamlit for web interface
- Various scientific Python libraries
