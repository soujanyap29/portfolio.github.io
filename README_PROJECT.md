# Protein Sub-Cellular Localization Pipeline

A complete, competition-ready pipeline for processing 4D TIFF images and predicting protein sub-cellular localization in neurons.

## Project Structure

```
portfolio.github.io/
├── src/
│   ├── preprocessing/      # TIFF processing and segmentation
│   ├── graph/             # Graph construction
│   ├── models/            # ML/DL models
│   ├── visualization/     # Plotting utilities
│   └── frontend/          # Web interface
├── notebooks/
│   └── final_pipeline.ipynb
├── requirements.txt
└── README_PROJECT.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the complete pipeline via Jupyter notebook:
```bash
jupyter notebook notebooks/final_pipeline.ipynb
```

Or use the web interface:
```bash
python src/frontend/app.py
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for deep learning)
- Input data location: `/mnt/d/5TH_SEM/CELLULAR/input`
- Output location: `/mnt/d/5TH_SEM/CELLULAR/output`

## Features

1. **Preprocessing**: Cellpose segmentation, feature extraction
2. **Graph Construction**: Biological graph representation
3. **Model Training**: Graph-CNN with evaluation metrics
4. **Visualization**: Publication-ready plots
5. **Web Interface**: Upload and predict functionality
6. **Jupyter Notebook**: Complete end-to-end pipeline

## Outputs

- Trained models saved to `/mnt/d/5TH_SEM/CELLULAR/output/models`
- Visualizations and data saved to `/mnt/d/5TH_SEM/CELLULAR/output`
