#!/bin/bash

# Setup script for Protein Sub-Cellular Localization Pipeline

echo "=================================================="
echo "Protein Localization Pipeline - Setup"
echo "=================================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if Python 3.8+
required_version="3.8"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✓ Python version is compatible"
else
    echo "✗ Python 3.8 or higher is required"
    exit 1
fi

echo ""
echo "Installing Python dependencies..."
echo ""

# Install core dependencies
echo "Installing core scientific libraries..."
pip3 install numpy scipy scikit-image Pillow

echo ""
echo "Installing graph processing libraries..."
pip3 install networkx

echo ""
echo "Installing visualization libraries..."
pip3 install matplotlib seaborn

echo ""
echo "Installing machine learning libraries..."
pip3 install scikit-learn

echo ""
echo "Installing PyTorch (CPU version)..."
pip3 install torch

echo ""
echo "Installing PyTorch Geometric..."
pip3 install torch-geometric

echo ""
echo "Installing additional PyTorch Geometric dependencies..."
pip3 install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cpu.html

echo ""
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "To verify the installation, run:"
echo "  python3 scripts/test_structure.py"
echo ""
echo "To run the pipeline with demo data:"
echo "  python3 scripts/pipeline.py --output ./output --epochs 20"
echo ""
echo "To start the web interface:"
echo "  cd frontend && python3 -m http.server 8000"
echo "  Then open http://localhost:8000 in your browser"
echo ""
echo "=================================================="
