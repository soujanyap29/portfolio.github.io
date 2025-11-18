# Quick Start Guide - Protein Localization Pipeline

This guide will help you get the protein localization pipeline up and running quickly.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster processing

## Installation

### Option 1: Quick Install (Recommended)

```bash
cd protein_localization
pip install -r requirements.txt
```

### Option 2: Install with GPU Support

```bash
cd protein_localization
pip install -r requirements.txt

# Install PyTorch Geometric with CUDA support
# Replace cu117 with your CUDA version (cu102, cu113, cu116, cu117, etc.)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-1.10.0+cu117.html
```

### Option 3: Development Install

```bash
cd protein_localization
pip install -e .
```

## Quick Test

Run the demo script to verify installation:

```bash
cd protein_localization
python demo.py
```

This will:
1. Create synthetic test data
2. Build biological graphs
3. Train a small model
4. Generate visualizations
5. Make predictions

Expected output directory: `protein_localization/demo_output/`

## Usage Options

### 1. Jupyter Notebook (Easiest)

Open and run the complete pipeline notebook:

```bash
cd protein_localization/notebooks
jupyter lab final_pipeline.ipynb
```

**Features:**
- Step-by-step execution
- Inline documentation
- Interactive visualization
- Supports both real and synthetic data

### 2. Web Interface (Most User-Friendly)

Start the web application:

```bash
cd protein_localization/frontend
python app.py
```

Then open: http://localhost:5000

**Features:**
- Drag-and-drop file upload
- Real-time processing
- Interactive results
- No coding required

### 3. Python API (Most Flexible)

Use the pipeline programmatically:

```python
import sys
sys.path.append('protein_localization/src')

from preprocessing import preprocess_pipeline
from graph_builder import build_graphs_pipeline
from models import train_model_pipeline

# Process your data
INPUT_DIR = "/path/to/your/tiff/files"
OUTPUT_DIR = "/path/to/save/results"

processed = preprocess_pipeline(INPUT_DIR, OUTPUT_DIR)
graphs = build_graphs_pipeline(processed, OUTPUT_DIR)
results = train_model_pipeline(graphs, OUTPUT_DIR)
```

## Directory Setup

The pipeline expects the following structure:

```
/mnt/d/5TH_SEM/CELLULAR/
├── input/          # Place your TIFF files here
│   ├── sample1.tif
│   ├── sample2.tiff
│   └── subfolder/
│       └── more_samples.tif
└── output/         # Results will be saved here
    ├── models/
    ├── graphs/
    └── visualizations/
```

**Note:** If these directories don't exist, the pipeline will create them automatically.

## Testing with Your Data

### Step 1: Prepare Your Data

Place TIFF files in the input directory:

```bash
mkdir -p /mnt/d/5TH_SEM/CELLULAR/input
# Copy your TIFF files there
```

### Step 2: Run the Pipeline

#### Using Jupyter:
```bash
cd protein_localization/notebooks
jupyter lab final_pipeline.ipynb
# Run all cells
```

#### Using Web Interface:
```bash
cd protein_localization/frontend
python app.py
# Upload files via browser
```

#### Using Python Script:
```bash
cd protein_localization
python -c "
from src.preprocessing import preprocess_pipeline
from src.graph_builder import build_graphs_pipeline

INPUT = '/mnt/d/5TH_SEM/CELLULAR/input'
OUTPUT = '/mnt/d/5TH_SEM/CELLULAR/output'

results = preprocess_pipeline(INPUT, OUTPUT)
graphs = build_graphs_pipeline(results, OUTPUT)
print(f'Processed {len(results)} files')
"
```

### Step 3: View Results

Results are saved to `/mnt/d/5TH_SEM/CELLULAR/output/`:

- **Models**: `output/models/graph_cnn_model.pt`
- **Graphs**: `output/graphs/*.gpickle`, `*.pt`
- **Visualizations**: `output/visualizations/*.png`
- **Summary**: `output/pipeline_summary.json`

## Common Issues & Solutions

### Issue: "Cellpose not found"
**Solution:** The pipeline will automatically fall back to threshold-based segmentation. For better results, install Cellpose:
```bash
pip install cellpose
```

### Issue: "CUDA out of memory"
**Solution:** Use CPU mode or reduce batch size:
```python
CONFIG['batch_size'] = 8  # Reduce from 32
# Or use CPU
device = 'cpu'
```

### Issue: "No TIFF files found"
**Solution:** 
- Check the input directory path
- Verify file extensions (.tif or .tiff)
- Ensure files are readable

### Issue: "Module not found"
**Solution:** Make sure you're in the correct directory and have installed requirements:
```bash
cd protein_localization
pip install -r requirements.txt
```

## Performance Tips

1. **Use GPU**: 10-100x faster processing
2. **Batch Processing**: Process multiple files at once
3. **Adjust Parameters**: 
   - `k_neighbors=3-7` for different cell densities
   - `distance_threshold=30-100` based on image scale
4. **Optimize for Speed**:
   - Reduce epochs during development
   - Use smaller image patches
   - Skip visualization during bulk processing

## Next Steps

1. **Read the Documentation**: See [README.md](README.md) for detailed information
2. **Explore the Code**: Check out the source files in `src/`
3. **Customize**: Modify parameters to suit your specific needs
4. **Contribute**: Improve the pipeline and share your enhancements

## Getting Help

- Check the main [README.md](README.md)
- Review code comments and docstrings
- Run the demo script to see expected behavior
- Check the Jupyter notebook for examples

## Example Workflow

Here's a complete workflow from start to finish:

```bash
# 1. Install dependencies
cd protein_localization
pip install -r requirements.txt

# 2. Test installation
python demo.py

# 3. Prepare your data
mkdir -p /mnt/d/5TH_SEM/CELLULAR/input
cp /path/to/your/tiff/files/*.tif /mnt/d/5TH_SEM/CELLULAR/input/

# 4. Run the pipeline
jupyter lab notebooks/final_pipeline.ipynb

# 5. View results
ls /mnt/d/5TH_SEM/CELLULAR/output/

# 6. Use web interface for new predictions
cd frontend
python app.py
# Open http://localhost:5000
```

## Success Indicators

You'll know everything is working when:
- ✅ Demo script completes without errors
- ✅ Graphs are created in `demo_output/graphs/`
- ✅ Visualizations appear in `demo_output/visualizations/`
- ✅ Model is saved to `demo_output/models/`
- ✅ Web interface loads at http://localhost:5000

## Need More Help?

Check these resources:
- Main README: [README.md](README.md)
- Source code documentation in each module
- Jupyter notebook with detailed comments
- Demo script for working example

Happy analyzing! 🧬
