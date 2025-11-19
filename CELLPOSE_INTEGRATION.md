# Cellpose Integration Guide

## Overview

The Protein Sub-Cellular Localization System now integrates **Cellpose**, a state-of-the-art deep learning algorithm for cellular and nuclear segmentation. Cellpose provides superior segmentation quality compared to traditional methods and includes advanced preprocessing capabilities.

---

## What is Cellpose?

Cellpose is a generalist algorithm for cellular and nuclear instance segmentation developed by the Stringer Lab. It uses a deep neural network trained on diverse microscopy images to accurately segment cells and nuclei across different imaging modalities.

### Key Advantages:
- **Superior accuracy**: Trained on diverse datasets, works well on various cell types
- **Robust normalization**: Percentile-based normalization handles outliers and varying intensities
- **No parameter tuning**: Works out-of-the-box for most microscopy images
- **Instance segmentation**: Separates individual cells/nuclei, not just regions
- **GPU acceleration**: Optional GPU support for faster processing

---

## Integration Features

### 1. Cellpose Segmentation
**Location**: `output/backend/segmentation.py`

The `SegmentationModule` now supports Cellpose as a segmentation method:

```python
from segmentation import SegmentationModule

# Initialize with Cellpose (default)
segmenter = SegmentationModule(method="CELLPOSE")

# Segment image
mask = segmenter.segment(image, diameter=30.0)
```

**Parameters**:
- `diameter`: Estimated cell/nucleus diameter in pixels (default: 30.0, None for auto-detect)
- `flow_threshold`: Flow error threshold, higher = more selective (default: 0.4)
- `cellprob_threshold`: Cell probability threshold (default: 0.0)

### 2. Cellpose Preprocessing
**Location**: `output/backend/image_loader.py`

The `TIFFLoader` class now includes Cellpose-based preprocessing:

```python
from image_loader import TIFFLoader

# Initialize loader with Cellpose preprocessing
loader = TIFFLoader(use_cellpose=True)

# Preprocess image with advanced normalization
image_processed = loader.preprocess_with_cellpose(image, target_size=(224, 224))
```

**Features**:
- **Percentile-based normalization**: `transforms.normalize99()` - robust to outliers
- **Quality resizing**: Uses Cellpose's optimized resize functions
- **Automatic fallback**: Falls back to OpenCV if Cellpose unavailable

---

## Configuration

Edit `output/backend/config.py` to configure Cellpose:

```python
# Segmentation Configuration
SEGMENTATION_METHOD = "CELLPOSE"  # Use Cellpose by default
USE_CELLPOSE_PREPROCESSING = True  # Enable Cellpose preprocessing
CELLPOSE_DIAMETER = 30.0  # Cell diameter (pixels)
CELLPOSE_FLOW_THRESHOLD = 0.4  # Flow threshold
CELLPOSE_CELLPROB_THRESHOLD = 0.0  # Cell probability threshold
```

### Available Segmentation Methods:
- `"CELLPOSE"` - **Recommended**: State-of-the-art neural network segmentation
- `"SLIC"` - Fast superpixel segmentation
- `"UNET"` - U-Net deep learning segmentation
- `"WATERSHED"` - Traditional watershed algorithm

---

## Usage Examples

### Basic Usage

```python
import numpy as np
from image_loader import TIFFLoader
from segmentation import SegmentationModule

# Load and preprocess
loader = TIFFLoader(use_cellpose=True)
image = loader.load_tiff("neuron.tif")
image_preprocessed = loader.preprocess_with_cellpose(image, target_size=(512, 512))

# Segment with Cellpose
segmenter = SegmentationModule(method="CELLPOSE")
masks = segmenter.segment(image_preprocessed, diameter=30.0)

print(f"Found {masks.max()} cells/regions")
```

### Batch Processing with Cellpose

```python
import os
import glob
from image_loader import TIFFLoader
from segmentation import SegmentationModule

# Initialize
loader = TIFFLoader(use_cellpose=True)
segmenter = SegmentationModule(method="CELLPOSE")

# Process all TIFF files
tiff_files = glob.glob("/path/to/input/*.tif")
for tiff_path in tiff_files:
    # Load and preprocess
    image = loader.load_tiff(tiff_path)
    image_preprocessed = loader.preprocess_with_cellpose(image)
    
    # Segment
    masks = segmenter.segment(image_preprocessed, diameter=35.0)
    
    # Save results
    output_path = f"segmented/{os.path.basename(tiff_path)}"
    # ... save masks ...
```

### Jupyter Notebook

The enhanced Jupyter notebook (`output/final_pipeline.ipynb`) automatically uses Cellpose when configured:

```bash
jupyter notebook output/final_pipeline.ipynb
```

Run Section 3 for batch processing with Cellpose segmentation.

---

## Cellpose Models

The system uses the **cyto2** model by default, which is a general-purpose model for cells and nuclei:

- **cyto2**: General cytoplasm + nucleus model (recommended for neurons)
- **nuclei**: Specialized for nuclear segmentation
- **cyto**: Original cytoplasm model

To change the model, edit `segmentation.py`:

```python
self.cellpose_model = models.Cellpose(gpu=False, model_type='nuclei')
```

---

## Performance Comparison

### Segmentation Quality

| Method | Accuracy | Speed | Use Case |
|--------|----------|-------|----------|
| **Cellpose** | ★★★★★ | ★★★☆☆ | Best overall quality, diverse cell types |
| SLIC | ★★★☆☆ | ★★★★★ | Fast superpixels, good for graphs |
| U-Net | ★★★★☆ | ★★☆☆☆ | Custom training required |
| Watershed | ★★☆☆☆ | ★★★★☆ | Simple traditional method |

### Preprocessing Quality

| Method | Robustness | Quality | Speed |
|--------|------------|---------|-------|
| **Cellpose normalize99** | ★★★★★ | ★★★★★ | ★★★★☆ |
| Standard normalization | ★★★☆☆ | ★★★☆☆ | ★★★★★ |
| OpenCV resize | ★★★☆☆ | ★★★☆☆ | ★★★★★ |

---

## Troubleshooting

### Cellpose Not Available

If you see: `"Warning: Cellpose not available"`

**Solution**: Install Cellpose
```bash
pip install cellpose
```

The system will automatically fall back to SLIC segmentation if Cellpose is unavailable.

### Out of Memory

If processing large images causes memory issues:

**Solution 1**: Disable GPU (already default)
```python
self.cellpose_model = models.Cellpose(gpu=False, model_type='cyto2')
```

**Solution 2**: Preprocess images to smaller size
```python
image_small = loader.preprocess_with_cellpose(image, target_size=(512, 512))
```

### Poor Segmentation Quality

If segmentation quality is poor:

**Solution 1**: Adjust diameter parameter
```python
# Try different diameters (typical range: 15-60 pixels)
masks = segmenter.segment(image, diameter=40.0)  # Larger cells
masks = segmenter.segment(image, diameter=20.0)  # Smaller cells
masks = segmenter.segment(image, diameter=None)  # Auto-detect
```

**Solution 2**: Adjust thresholds
```python
# More selective (fewer false positives)
masks = segmenter.segment(image, flow_threshold=0.6, cellprob_threshold=0.5)

# Less selective (catch more cells)
masks = segmenter.segment(image, flow_threshold=0.2, cellprob_threshold=-1.0)
```

### Slow Processing

If processing is too slow:

**Solution 1**: Enable GPU (if available)
```python
self.cellpose_model = models.Cellpose(gpu=True, model_type='cyto2')
```

**Solution 2**: Switch to SLIC for faster processing
```python
segmenter = SegmentationModule(method="SLIC")
```

---

## Advanced Configuration

### Custom Cellpose Parameters

Edit `segmentation.py` to customize Cellpose initialization:

```python
class SegmentationModule:
    def __init__(self, method: str = "CELLPOSE"):
        if self.method == "CELLPOSE":
            self.cellpose_model = models.Cellpose(
                gpu=False,              # Use GPU if available
                model_type='cyto2',     # Model type
                net_avg=True,           # Average 4 network outputs
                device=None             # Specify device
            )
```

### Hybrid Approach

Combine multiple segmentation methods:

```python
# Use Cellpose for accurate cell detection
cellpose_seg = SegmentationModule(method="CELLPOSE")
cell_masks = cellpose_seg.segment(image, diameter=30.0)

# Use SLIC for superpixel graphs
slic_seg = SegmentationModule(method="SLIC")
superpixels = slic_seg.segment(image, n_segments=100)
```

---

## References

- **Cellpose Paper**: Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021). Cellpose: a generalist algorithm for cellular segmentation. *Nature Methods*, 18(1), 100-106.
- **Cellpose GitHub**: https://github.com/MouseLand/cellpose
- **Cellpose Documentation**: https://cellpose.readthedocs.io/

---

## Summary

The Cellpose integration provides:

✅ **Superior segmentation accuracy** for cellular and nuclear structures  
✅ **Robust preprocessing** with percentile-based normalization  
✅ **Easy configuration** through `config.py`  
✅ **Automatic fallback** to SLIC if Cellpose unavailable  
✅ **Full integration** in Jupyter notebook and CLI tools  
✅ **Flexible parameters** for different cell types and imaging conditions  

**Recommendation**: Use Cellpose for production analysis of neuronal TIFF microscopy images. The improved segmentation quality significantly enhances downstream CNN and GNN classification accuracy.

---

*For questions or issues with Cellpose integration, refer to the main README.md or TRAINING_GUIDE.md*
