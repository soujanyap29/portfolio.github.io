# Troubleshooting Guide

## Common Issues and Solutions

### 1. ModuleNotFoundError: No module named 'tiff_loader'

**Problem**: The Streamlit app cannot find the pipeline modules.

**Solution**:

1. **Ensure you're in the correct directory**:
   ```bash
   # Navigate to the protein-localization directory
   cd protein-localization
   
   # Verify you're in the right place
   ls -la
   # You should see: scripts/, frontend/, docs/, etc.
   ```

2. **Run from the protein-localization directory**:
   ```bash
   # Always run from here:
   streamlit run frontend/streamlit_app.py
   
   # NOT from the frontend directory
   ```

3. **Check the directory structure**:
   ```bash
   tree -L 2 -I '__pycache__|*.pyc|venv'
   ```
   
   Expected structure:
   ```
   protein-localization/
   ├── scripts/
   │   ├── tiff_loader.py
   │   ├── preprocessing.py
   │   └── ...
   ├── frontend/
   │   └── streamlit_app.py
   └── ...
   ```

4. **Verify dependencies are installed**:
   ```bash
   pip list | grep -E "numpy|pandas|streamlit|tifffile"
   ```

### 2. Dependencies Not Installed

**Problem**: Missing Python packages.

**Solution**:
```bash
# Install all dependencies
pip install -r requirements.txt

# Or install individually
pip install numpy pandas tifffile scikit-image opencv-python
pip install torch torchvision networkx matplotlib seaborn streamlit
```

### 3. Permission Denied on setup.sh

**Problem**: Cannot execute setup script.

**Solution**:
```bash
chmod +x setup.sh
bash setup.sh
```

### 4. CUDA/GPU Issues

**Problem**: GPU not detected or CUDA errors.

**Solution**:
```bash
# Run in CPU mode
python scripts/pipeline.py --input /path/to/tiffs --output /path/to/output
# (GPU flag is disabled by default)

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 5. Cellpose API Error

**Problem**: `AttributeError: module 'cellpose.models' has no attribute 'Cellpose'`

**Solution**:

This is due to Cellpose API changes between versions. The pipeline now handles multiple Cellpose versions automatically.

If you still encounter issues:

```bash
# Check Cellpose version
pip show cellpose

# For Cellpose v2.0+
pip install cellpose>=2.0

# Or for compatibility with older versions
pip install cellpose==1.0.2

# The pipeline will automatically detect and use the correct API
```

The system will fall back to classical segmentation methods if Cellpose fails.

### 6. Cellpose Not Found

**Problem**: Cellpose segmentation not available.

**Solution**:
```bash
# Install Cellpose
pip install cellpose

# The pipeline will use fallback segmentation if Cellpose is not available
```

### 7. Cannot Write to Output Directory

**Problem**: Permission denied when writing outputs.

**Solution**:
```bash
# Create output directory with proper permissions
mkdir -p /mnt/d/5TH_SEM/CELLULAR/output
chmod 755 /mnt/d/5TH_SEM/CELLULAR/output

# Or use a different output directory
python scripts/pipeline.py --output ./my_output
```

### 8. Import Error in Jupyter Notebook

**Problem**: Cannot import modules in Jupyter.

**Solution**:

In the first cell of the notebook, add:
```python
import sys
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path.cwd().parent / 'scripts' if 'protein-localization' not in str(Path.cwd()) else Path.cwd() / 'scripts'
sys.path.insert(0, str(scripts_dir))
```

Or run Jupyter from the protein-localization directory:
```bash
cd protein-localization
jupyter lab
```

### 9. Port Already in Use (Streamlit)

**Problem**: Streamlit port 8501 already in use.

**Solution**:
```bash
# Use a different port
streamlit run frontend/streamlit_app.py --server.port 8502

# Or kill existing Streamlit process
pkill -f streamlit
streamlit run frontend/streamlit_app.py
```

### 10. Slow Processing Time

**Problem**: Pipeline takes a long time to process images.

**Solution**:

Processing time depends on image size and hardware:
- Small images (<512x512): 30-60 seconds
- Medium images (512-1024): 1-3 minutes
- Large images (>1024): 3-5 minutes

**To speed up processing:**

```bash
# 1. Enable GPU acceleration (5-10x faster for segmentation)
# In Streamlit: Check "Use GPU Acceleration" in sidebar
# In CLI: python scripts/pipeline.py --gpu

# 2. Reduce image size before processing
# Resize large images to reasonable dimensions

# 3. Use smaller cell diameter for faster segmentation
# In Streamlit: Reduce "Expected Cell Diameter" slider value

# 4. Process in batches for multiple files
python scripts/pipeline.py --max-files 10
```

**What takes the most time:**
- Cellpose segmentation: 40-60% of time
- Feature extraction: 15-20%
- Graph construction: 10-15%
- Visualization generation: 10-15%

### 11. Out of Memory Error

**Problem**: System runs out of memory processing large images.

**Solution**:
```bash
# Process fewer files at a time
python scripts/pipeline.py --max-files 5

# Or use smaller images
# Resize images before processing
```

### 12. Test Structure Fails

**Problem**: test_structure.py reports errors.

**Solution**:
```bash
# Check which tests are failing
cd scripts
python test_structure.py

# Install missing dependencies
pip install -r ../requirements.txt

# Verify Python version
python --version  # Should be 3.8+
```

## Getting Help

If you encounter issues not covered here:

1. **Check the documentation**:
   - README.md
   - docs/QUICKSTART.md
   - docs/PROJECT_OVERVIEW.md

2. **Run the test suite**:
   ```bash
   cd scripts
   python test_structure.py
   ```

3. **Check Python and dependency versions**:
   ```bash
   python --version
   pip list
   ```

4. **Enable debug mode** (for Streamlit):
   ```bash
   streamlit run frontend/streamlit_app.py --logger.level=debug
   ```

5. **Check the GitHub repository**:
   - Issues: https://github.com/soujanyap29/portfolio.github.io/issues
   - Documentation: https://github.com/soujanyap29/portfolio.github.io/tree/main/protein-localization

## Quick Checklist

Before running the pipeline, verify:

- [ ] Python 3.8+ is installed
- [ ] In the protein-localization directory
- [ ] Virtual environment activated (`source venv/bin/activate`)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Directory structure is intact (scripts/, frontend/, docs/)
- [ ] Input directory exists (or will upload files via web interface)
- [ ] Output directory is writable

## Command Reference

```bash
# Setup
cd protein-localization
bash setup.sh
source venv/bin/activate

# Web Interface
streamlit run frontend/streamlit_app.py

# Command Line
python scripts/pipeline.py --input /path/to/tiffs --output /path/to/output

# Jupyter
jupyter lab final_pipeline.ipynb

# Testing
cd scripts && python test_structure.py
```
