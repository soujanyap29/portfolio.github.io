# Protein Sub-Cellular Localization Package

## Quick Start Guide

### Installation

```bash
cd protein_localization
bash setup.sh
source venv/bin/activate
```

### Basic Usage

1. **Place TIFF images in `data/raw/` directory**

2. **Run preprocessing**:
```bash
python main.py --input_dir data/raw --mode preprocess
```

3. **Train model**:
```bash
python train.py --data_dir data/graphs --epochs 50
```

4. **Run inference**:
```bash
python inference.py --model_path outputs/models/best_model.pth --input_dir data/raw
```

5. **Evaluate results**:
```bash
python evaluate.py --predictions_dir outputs/results
```

### Using Jupyter Lab

```bash
jupyter lab
# Open notebooks/protein_localization_pipeline.ipynb
```

## Requirements Met

✅ **Environment & Setup**: Complete setup script and requirements.txt  
✅ **Data Access & Sanity Checks**: Robust data loader with validation  
✅ **Image Preprocessing**: Comprehensive preprocessing pipeline  
✅ **Graph Construction**: Automatic graph building from images  
✅ **Labels Preparation**: Label handling infrastructure  
✅ **Model Design & Training**: GNN and CNN models with training scripts  
✅ **Training**: Complete training pipeline with early stopping  
✅ **Inference Across All Samples**: Batch inference capability  
✅ **Evaluation & Visualization**: Full evaluation suite with plots  

## Key Features

- ✅ Processes all TIFF images in folder
- ✅ Builds correct graph representations for every image
- ✅ Valid outputs for all images
- ✅ Batch mode for multiple files
- ✅ Ubuntu + Jupyter Lab compatible

## Project Structure

```
protein_localization/
├── config.yaml                 # Configuration
├── main.py                     # Main orchestration
├── train.py                    # Training script
├── inference.py                # Inference script
├── evaluate.py                 # Evaluation script
├── examples.py                 # Usage examples
├── setup.sh                    # Setup script
├── requirements.txt            # Dependencies
├── utils/                      # Utility modules
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── graph_builder.py
│   └── visualizer.py
├── models/                     # Model architectures
│   ├── gnn_model.py
│   └── cnn_model.py
└── notebooks/                  # Jupyter notebooks
    └── protein_localization_pipeline.ipynb
```

## Support

For issues or questions, please refer to the main README.md file.
