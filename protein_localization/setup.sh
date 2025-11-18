#!/bin/bash

# Protein Localization System - Setup Script
# This script sets up the environment and prepares the system for use

echo "🧬 Protein Sub-Cellular Localization System Setup"
echo "=================================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating output directories..."
mkdir -p output/{models,visualizations,segmented,graphs,predictions}
mkdir -p sample_data

# Generate sample TIFF files if no input directory exists
echo ""
echo "Checking for input data..."
if [ ! -d "/mnt/d/5TH_SEM/CELLULAR/input" ]; then
    echo "⚠️  Input directory not found. Creating sample data..."
    python3 -c "
import numpy as np
import tifffile
import os

os.makedirs('sample_data', exist_ok=True)
for i in range(5):
    # Create 4D sample: (time, z, y, x)
    sample = np.random.randint(0, 255, (5, 10, 256, 256), dtype=np.uint8)
    tifffile.imwrite(f'sample_data/sample_{i}.tif', sample)
print('✓ Created 5 sample TIFF files')
"
else
    echo "✓ Input directory found"
fi

# Test imports
echo ""
echo "Testing imports..."
python3 -c "
import numpy
import torch
import matplotlib
import networkx
import tifffile
import cellpose
print('✓ All core libraries imported successfully')
"

echo ""
echo "=================================================="
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Run Jupyter notebook: jupyter lab output/final_pipeline.ipynb"
echo "  3. Or start web interface: python frontend/app.py"
echo ""
echo "For configuration, edit config.yaml"
echo "=================================================="
