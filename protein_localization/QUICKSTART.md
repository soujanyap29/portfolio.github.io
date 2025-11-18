# Quick Start Guide

Get up and running with the Protein Sub-Cellular Localization System in minutes!

## 🚀 Installation (5 minutes)

### Option 1: Automatic Setup (Recommended)

```bash
cd protein_localization
bash setup.sh
```

This will:
- Create a virtual environment
- Install all dependencies
- Create output directories
- Generate sample data if needed
- Test imports

### Option 2: Manual Setup

```bash
cd protein_localization

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p output/{models,visualizations,segmented,graphs,predictions}
```

## 🎯 Three Ways to Use the System

### 1. Complete Pipeline Script (Fastest - 2 minutes)

Run the entire pipeline with one command:

```bash
python run_pipeline.py
```

This will:
- Load TIFF files
- Segment with Cellpose
- Build biological graphs
- Train a demo model
- Generate all visualizations
- Make predictions

**Output**: All results in `output/` directory

### 2. Jupyter Notebook (Recommended for Learning - 10 minutes)

Interactive step-by-step workflow:

```bash
jupyter lab output/final_pipeline.ipynb
```

Features:
- Detailed explanations
- Cell-by-cell execution
- Customizable parameters
- Visualization inline

### 3. Web Interface (Best for Production - 1 minute)

User-friendly web interface:

```bash
cd frontend
python app.py
```

Then open: http://localhost:5000

Features:
- Drag-and-drop file upload
- Real-time processing
- Interactive results
- Download outputs

## 📁 Your Data

### Using Your Own TIFF Files

Edit `config.yaml`:

```yaml
data:
  input_dir: "/path/to/your/tiff/files"
  output_dir: "/path/to/save/results"
```

The system will:
- Recursively scan all subdirectories
- Process all `.tif` and `.tiff` files
- Save results to output directory

### Sample Data

If you don't have data yet, the system automatically creates sample 4D TIFF files for testing.

## 🧪 Quick Test (30 seconds)

Verify everything works:

```bash
python -c "
from preprocessing import TIFFProcessor
from utils import load_config
config = load_config('config.yaml')
processor = TIFFProcessor(config)
print('✅ System ready!')
"
```

## 📊 Understanding Outputs

After running the pipeline, check these directories:

```
output/
├── models/               # Trained model files (.pth)
├── visualizations/       # All plots and figures (.png)
├── segmented/           # Segmentation results
├── graphs/              # Saved biological graphs
└── predictions/         # Prediction results and metrics
```

### Key Visualizations

1. **segmentation_overlay.png** - Original + segmentation + overlay
2. **graph_visualization.png** - Biological network with nodes/edges
3. **compartment_map.png** - Color-coded regions
4. **confusion_matrix.png** - Model performance
5. **intensity_grouped_bar.png** - Statistical analysis

## 🎨 Customization

### Change Segmentation Parameters

In `config.yaml`:

```yaml
segmentation:
  model_type: "cyto"      # or "nuclei", "cyto2"
  diameter: 30            # cell diameter in pixels
  flow_threshold: 0.4     # sensitivity (0-1)
```

### Change Model Architecture

```yaml
training:
  model_type: "graph_cnn"   # or "vgg16", "hybrid"
  epochs: 100               # more epochs = better accuracy
  batch_size: 16           # larger = faster, needs more memory
  learning_rate: 0.001     # smaller = more stable
```

### Change Output Quality

```yaml
visualization:
  dpi: 300                 # higher = better quality
  figure_size: [10, 8]    # [width, height] in inches
  colormap: "viridis"     # or "plasma", "inferno", etc.
```

## 🔧 Troubleshooting

### Problem: "Module not found"
**Solution**: Activate virtual environment
```bash
source venv/bin/activate
```

### Problem: "CUDA out of memory"
**Solution**: Reduce batch size in config.yaml
```yaml
training:
  batch_size: 8  # or smaller
```

### Problem: "Cellpose not working"
**Solution**: Install with conda
```bash
conda install -c conda-forge cellpose
```

### Problem: Web interface won't start
**Solution**: Check if port 5000 is available
```bash
# Use different port
python app.py --port 8080
```

## 📖 Next Steps

1. **Read the full README**: `README.md`
2. **Explore documentation**: `DOCUMENTATION.md`
3. **Try the Jupyter notebook**: More interactive
4. **Customize for your data**: Edit config.yaml
5. **Train on real labels**: Replace demo labels with actual annotations

## 💡 Tips

- Start with small datasets (5-10 files) for testing
- Use GPU for faster processing if available
- Check visualizations after each run
- Save your trained models for reuse
- Experiment with different parameters

## 🆘 Need Help?

- Check `DOCUMENTATION.md` for detailed API reference
- Look at example code in `run_pipeline.py`
- Review Jupyter notebook cells for step-by-step guidance
- Open an issue on GitHub for bugs

## 🎉 Success!

If you see visualizations in `output/visualizations/`, congratulations! 

Your system is working correctly. You're ready to:
- Process your own TIFF files
- Train models on your data
- Generate publication-quality figures
- Make predictions on new samples

---

**Estimated total time**: 10-15 minutes for complete setup and first run

**Questions?** See DOCUMENTATION.md or README.md
