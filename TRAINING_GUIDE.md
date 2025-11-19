# Model Training & Evaluation Guide

## Overview

This system now includes comprehensive model training and evaluation capabilities with reproducible train-test splits, allowing you to train custom protein localization models from your labeled datasets.

## What's New

### 1. Training Module (`backend/training.py`)

A complete training framework that provides:

- **Reproducible Data Splits**: Train-validation-test splits with configurable ratios
- **Multiple Model Training**: Support for VGG16, ResNet50, and EfficientNet
- **Comprehensive Evaluation**: All metrics computed for train, validation, and test sets
- **Visualization Generation**: Training curves, model comparisons, and performance metrics
- **Result Persistence**: JSON export of all results for later analysis

### 2. Training Script (`train_models.py`)

A command-line tool for easy model training:

```bash
python output/train_models.py --data_dir /path/to/labeled/data --epochs 50
```

### 3. Jupyter Notebook Section

New section added to `final_pipeline.ipynb`:
- **Section 4**: Model Training & Evaluation (Optional)
- Demonstrates training workflow with code examples
- Shows how to load datasets and train models
- Includes visualization of training results

## Dataset Requirements

### Directory Structure

Your labeled dataset should be organized as:

```
labeled_data/
├── Nucleus/
│   ├── neuron_001.tif
│   ├── neuron_002.tif
│   └── ...
├── Cytoplasm/
│   ├── neuron_010.tif
│   ├── neuron_011.tif
│   └── ...
├── Membrane/
│   └── ...
├── Mitochondria/
│   └── ...
├── ER/
│   └── ...
├── Golgi/
│   └── ...
├── Peroxisome/
│   └── ...
└── Cytoskeleton/
    └── ...
```

### Image Format

- **Format**: TIFF files (`.tif` or `.tiff`)
- **Size**: Any size (will be resized to 224x224)
- **Channels**: Grayscale or RGB (will be converted to RGB)
- **Normalization**: Automatically normalized to [0, 1]

### Minimum Requirements

- At least 50-100 images per class recommended
- Balanced distribution across classes preferred
- Clear, high-quality microscopy images

## Training Workflow

### Step 1: Prepare Dataset

Organize your labeled TIFF images into class directories as shown above.

### Step 2: Train Models

```bash
cd output

# Train all models (VGG16, ResNet50, EfficientNet)
python train_models.py --data_dir /path/to/labeled_data --epochs 50

# Train specific models only
python train_models.py --data_dir /path/to/labeled_data --models vgg16 resnet50

# Adjust data splits (e.g., 80% train, 10% val, 10% test)
python train_models.py --data_dir /path/to/labeled_data --test_size 0.1 --val_size 0.1

# Custom hyperparameters
python train_models.py --data_dir /path/to/labeled_data --epochs 100 --batch_size 16
```

### Step 3: Review Results

After training completes, check:

**Trained Models**:
- `output/models/vgg16_trained.h5`
- `output/models/resnet50_trained.h5`
- `output/models/efficientnet_trained.h5`

**Visualizations** (`output/graphs/`):
- `vgg16_training_history.png` - Loss and accuracy curves
- `resnet50_training_history.png`
- `efficientnet_training_history.png`
- `all_models_comparison.png` - Side-by-side comparison

**Results JSON**:
- `output/results/reports/training_results.json` - Complete metrics

## Evaluation Metrics

For each model and each dataset split (train/val/test), the system computes:

### Overall Metrics
- **Accuracy**: Proportion of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Specificity**: True negatives / (True negatives + False positives)

### Per-Class Metrics
- Precision, Recall, F1-Score for each protein localization class
- Confusion matrix showing prediction distribution

### Visualizations
- **Training History**: Loss and accuracy curves over epochs
- **Model Comparison**: Bar charts comparing all metrics
- **Confusion Matrices**: Detailed prediction breakdowns

## Command-Line Options

```
usage: train_models.py [-h] --data_dir DATA_DIR
                       [--models {vgg16,resnet50,efficientnet,all} [{vgg16,resnet50,efficientnet,all} ...]]
                       [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                       [--test_size TEST_SIZE] [--val_size VAL_SIZE]
                       [--random_seed RANDOM_SEED] [--output_dir OUTPUT_DIR]

options:
  --data_dir DATA_DIR    Path to labeled dataset directory
  --models MODELS        Models to train (default: vgg16 resnet50 efficientnet)
  --epochs EPOCHS        Number of training epochs (default: 50)
  --batch_size BATCH_SIZE
                        Batch size for training (default: 32)
  --test_size TEST_SIZE  Proportion of data for test set (default: 0.2)
  --val_size VAL_SIZE    Proportion for validation set (default: 0.1)
  --random_seed SEED     Random seed for reproducibility (default: 42)
  --output_dir DIR       Output directory (defaults to config OUTPUT_DIR)
```

## Using in Jupyter Notebook

Open `output/final_pipeline.ipynb` and navigate to **Section 4: Model Training & Evaluation**.

The section includes:
1. Data preparation and splitting
2. Model training for each variant
3. Visualization generation
4. Results summary

**Note**: Uncomment the code blocks and replace with your actual dataset paths.

## Programmatic Usage

You can also use the training module in your own Python scripts:

```python
from backend.training import ModelTrainer
import numpy as np

# Initialize trainer
trainer = ModelTrainer(random_seed=42)

# Prepare data (assumes you have loaded images and labels)
data = trainer.prepare_data(images, labels, test_size=0.2, val_size=0.1)

# Train VGG16
vgg_results = trainer.train_cnn_model('vgg16', data, epochs=50, batch_size=32)

# Train ResNet50
resnet_results = trainer.train_cnn_model('resnet50', data, epochs=50, batch_size=32)

# Visualize results
trainer.visualize_training_history('vgg16', save_dir='output/graphs')
trainer.visualize_all_metrics(save_dir='output/graphs')

# Save results
trainer.save_results('output/results/reports/training_results.json')

# Print summary
trainer.print_summary()
```

## Best Practices

### Data Preparation
- Ensure balanced class distribution (similar number of images per class)
- Use high-quality, clear microscopy images
- Remove duplicate or low-quality images
- Consider data augmentation for small datasets

### Training
- Start with default hyperparameters (50 epochs, batch size 32)
- Monitor validation loss to detect overfitting
- Use early stopping (automatically enabled)
- Save checkpoints during training

### Evaluation
- Always evaluate on held-out test set
- Check confusion matrix for systematic errors
- Analyze per-class metrics for class-specific issues
- Compare multiple models before final selection

### Reproducibility
- Set fixed random seed (default: 42)
- Document all hyperparameters used
- Save complete training results JSON
- Version control your datasets

## Troubleshooting

### Out of Memory Errors
- Reduce batch size: `--batch_size 16` or `--batch_size 8`
- Train models one at a time instead of all together
- Use smaller image sizes if needed

### Poor Performance
- Increase number of training epochs: `--epochs 100`
- Check data quality and class balance
- Verify images are loading correctly
- Try different train/val/test splits

### Slow Training
- Reduce number of epochs for initial experiments
- Use GPU if available (automatically detected)
- Reduce batch size for faster iteration

## Output Files

After training, you'll have:

```
output/
├── models/
│   ├── vgg16_trained.h5           # Trained model weights
│   ├── resnet50_trained.h5
│   └── efficientnet_trained.h5
├── graphs/
│   ├── vgg16_training_history.png       # Training curves
│   ├── resnet50_training_history.png
│   ├── efficientnet_training_history.png
│   └── all_models_comparison.png         # Model comparison
└── results/
    └── reports/
        └── training_results.json         # Complete metrics
```

## Next Steps

After training your models:

1. **Use for inference**: Load trained models with `pipeline.py` for batch processing
2. **Fine-tune**: Adjust hyperparameters and retrain if needed
3. **Deploy**: Integrate best-performing model into production workflow
4. **Document**: Include results in your journal paper or report

## Integration with Existing System

The training module seamlessly integrates with existing components:

- **Models**: Uses same CNN architectures as inference pipeline
- **Evaluation**: Same metrics computation as `evaluation.py`
- **Visualization**: Consistent style with `visualization.py`
- **Configuration**: Respects paths and settings from `config.py`

## Support

For questions or issues:
- Check the examples in the Jupyter notebook (Section 4)
- Review the training script help: `python train_models.py --help`
- Examine the `backend/training.py` module for implementation details

---

**Ready to train your own protein localization models!**
