#!/bin/bash
# Run script for Protein Localization System

echo "========================================"
echo "Protein Sub-Cellular Localization System"
echo "========================================"
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "Error: Streamlit is not installed."
    echo "Please run: pip install -r requirements.txt"
    exit 1
fi

# Navigate to frontend directory
cd "$(dirname "$0")/output/frontend"

# Launch Streamlit app
echo "Launching web interface..."
echo "Open your browser to: http://localhost:8501"
echo ""
streamlit run streamlit_app.py
