#!/bin/bash
# Setup script for Protein Sub-Cellular Localization Pipeline

echo "=========================================="
echo "Protein Localization Pipeline Setup"
echo "=========================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create directory structure
echo ""
echo "Creating directory structure..."
mkdir -p data/{raw,processed,graphs,labels}
mkdir -p outputs/{models,results,visualizations,logs}

echo ""
echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start using the pipeline:"
echo "1. Place your TIFF images in data/raw/"
echo "2. Run: python main.py --input_dir data/raw"
echo ""
echo "For Jupyter Lab:"
echo "  jupyter lab"
echo ""
echo "=========================================="
