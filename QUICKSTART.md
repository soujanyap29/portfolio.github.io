# Quick Start Guide

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Verify installation:**
```bash
python -c "import tensorflow; import torch; import streamlit; print('✓ All dependencies installed')"
```

## Running the System

### Option 1: Web Interface (Recommended)

```bash
./run.sh
```

Or manually:
```bash
cd output/frontend
streamlit run streamlit_app.py
```

Then open your browser to: http://localhost:8501

### Option 2: Command Line

**Process single image:**
```bash
python output/backend/pipeline.py --image /path/to/neuron.tif --output /path/to/output
```

**Batch process directory:**
```bash
python output/backend/pipeline.py --batch /mnt/d/5TH_SEM/CELLULAR/input --output /mnt/d/5TH_SEM/CELLULAR/output
```

## Configuration

Edit `output/backend/config.py` to customize:

```python
# Input/Output paths
INPUT_PATH = "/mnt/d/5TH_SEM/CELLULAR/input"
OUTPUT_PATH = "/mnt/d/5TH_SEM/CELLULAR/output"

# Segmentation method
SEGMENTATION_METHOD = "SLIC"  # Options: "UNET", "SLIC", "WATERSHED"

# Model fusion weights
CNN_WEIGHT = 0.6
GNN_WEIGHT = 0.4
```

## Output Files

For each processed image `neuron_001.tif`, the system generates:

```
/output
  /results
    /segmented
      neuron_001_segment.png          # Segmentation visualization
    /reports
      neuron_001_report.json          # Complete analysis results
  /graphs
    neuron_001_overlay.png            # Image + segmentation overlay
    neuron_001_probabilities.png      # Probability distribution
    neuron_001_graph.png              # Graph network visualization
    neuron_001_compartments.png       # Compartment mask map
```

## Example JSON Report

```json
{
  "filename": "neuron_001.tif",
  "timestamp": "2025-11-19T13:15:00",
  "fused": {
    "predicted_class": "Nucleus",
    "confidence": 0.876,
    "probabilities": {
      "Nucleus": 0.876,
      "Cytoplasm": 0.045,
      "Membrane": 0.032,
      ...
    }
  }
}
```

## Protein Localization Classes

1. Nucleus
2. Cytoplasm
3. Membrane
4. Mitochondria
5. Endoplasmic Reticulum
6. Golgi Apparatus
7. Peroxisome
8. Cytoskeleton

## Troubleshooting

**Issue: Module not found**
```bash
pip install -r requirements.txt
```

**Issue: CUDA/GPU errors**
- System works on CPU (slower but functional)
- For GPU: Install CUDA-compatible TensorFlow and PyTorch

**Issue: Out of memory**
- Reduce BATCH_SIZE in config.py
- Process images one at a time instead of batch

**Issue: Segmentation takes too long**
- Use SLIC instead of U-Net (faster)
- Reduce SLIC_N_SEGMENTS in config.py

## Generate Journal Paper

```bash
python output/backend/journal_generator.py
```

Output: `JOURNAL_PAPER.md` (35,000+ words, complete academic paper)

## System Requirements

**Minimum:**
- Python 3.8+
- 8GB RAM
- CPU (any modern processor)

**Recommended:**
- Python 3.8+
- 32GB RAM
- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.x

## Support

For issues or questions:
1. Check README.md for detailed documentation
2. Review JOURNAL_PAPER.md for technical details
3. Open an issue on GitHub

## Citation

```bibtex
@software{protein_localization_2025,
  title = {Protein Sub-Cellular Localization in Neurons},
  author = {Your Name},
  year = {2025},
  institution = {Your Institution}
}
```
