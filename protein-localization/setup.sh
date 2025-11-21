#!/bin/bash

echo "=== Protein Sub-Cellular Localization Pipeline Setup ==="
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "Python 3 found: $(python3 --version)"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

echo ""
echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Upgrading pip..."
pip install --upgrade pip

echo ""
echo "Installing required packages..."
pip install -r requirements.txt

echo ""
echo "Creating output directories..."
mkdir -p output/models
mkdir -p output/figures
mkdir -p output/graphs
mkdir -p output/features

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start the Streamlit app, run:"
echo "  streamlit run frontend/streamlit_app.py"
echo ""
echo "To run the complete pipeline, see docs/QUICKSTART.md"
