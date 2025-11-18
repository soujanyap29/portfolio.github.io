#!/bin/bash

# Installation script for Protein Localization Pipeline
# This script automates the installation process

set -e  # Exit on error

echo "=================================="
echo "Protein Localization Pipeline"
echo "Installation Script"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Check if Python 3.8+
if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "✓ Python version is compatible"
else
    echo "✗ Python 3.8+ is required"
    exit 1
fi

echo ""
echo "=================================="
echo "Step 1: Creating virtual environment"
echo "=================================="

if [ -d "venv" ]; then
    echo "Virtual environment already exists"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
    fi
else
    python3 -m venv venv
fi

echo "✓ Virtual environment created"

echo ""
echo "=================================="
echo "Step 2: Activating virtual environment"
echo "=================================="

source venv/bin/activate
echo "✓ Virtual environment activated"

echo ""
echo "=================================="
echo "Step 3: Upgrading pip"
echo "=================================="

pip install --upgrade pip setuptools wheel
echo "✓ pip upgraded"

echo ""
echo "=================================="
echo "Step 4: Installing dependencies"
echo "=================================="

pip install -r requirements.txt
echo "✓ Dependencies installed"

echo ""
echo "=================================="
echo "Step 5: Installing PyTorch Geometric"
echo "=================================="

# Detect if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected. Installing GPU version..."
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
        -f https://data.pyg.org/whl/torch-1.10.0+cu117.html || {
        echo "GPU installation failed, falling back to CPU version"
        pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
            -f https://data.pyg.org/whl/torch-1.10.0+cpu.html
    }
else
    echo "No CUDA detected. Installing CPU version..."
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
        -f https://data.pyg.org/whl/torch-1.10.0+cpu.html
fi

echo "✓ PyTorch Geometric installed"

echo ""
echo "=================================="
echo "Step 6: Running installation tests"
echo "=================================="

# Test imports
python3 -c "
import sys
sys.path.append('src')
print('Testing core imports...')
try:
    import numpy
    import torch
    import networkx
    print('✓ Core libraries imported successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
"

echo "✓ Installation tests passed"

echo ""
echo "=================================="
echo "Step 7: Creating output directories"
echo "=================================="

mkdir -p demo_output
mkdir -p /mnt/d/5TH_SEM/CELLULAR/input 2>/dev/null || echo "Note: Could not create /mnt/d/5TH_SEM/CELLULAR/input (may need manual creation)"
mkdir -p /mnt/d/5TH_SEM/CELLULAR/output 2>/dev/null || echo "Note: Could not create /mnt/d/5TH_SEM/CELLULAR/output (may need manual creation)"

echo "✓ Directories created"

echo ""
echo "=================================="
echo "Installation Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Run the demo:"
echo "   python demo.py"
echo ""
echo "3. Start Jupyter notebook:"
echo "   jupyter lab notebooks/final_pipeline.ipynb"
echo ""
echo "4. Launch web interface:"
echo "   cd frontend && python app.py"
echo ""
echo "For more information, see:"
echo "  - QUICKSTART.md"
echo "  - README.md"
echo "  - docs/DOCUMENTATION.md"
echo ""
echo "Happy analyzing! 🧬"
