# Quick Start Guide

## Getting Started with the Protein Localization Pipeline

### 🎯 Process YOUR TIFF Images to Network Diagrams (New!)

**Generate biological network diagrams from any TIFF image:**

```bash
cd protein-localization/scripts
python generate_biological_network.py --input your_image.tif --output network.png
```

This will:
1. Load your TIFF image (any format: 2D, 3D, 4D)
2. Automatically segment cellular structures
3. Build a graph network from segmented regions
4. Generate a clean biological network diagram with:
   - Soft grey rounded nodes
   - Blue central hub
   - Curved grey connection lines
   - Translucent cluster backgrounds
   - Scientific minimal aesthetic

**📖 See detailed guide:** `docs/TIFF_TO_NETWORK_GUIDE.md`

---

### 🎯 Easiest Way: Jupyter Notebook (Recommended)

**No need to copy-paste code!** Just open the notebook:

1. **Install Jupyter** (if not already installed):
   ```bash
   pip install jupyter
   ```

2. **Open the notebook**:
   ```bash
   cd protein-localization
   jupyter notebook Protein_Localization_Pipeline.ipynb
   ```

3. **Run the cells** to execute the pipeline interactively with visualizations

The notebook includes everything: imports, step-by-step execution, and demos with synthetic data.

---

### Alternative: Command Line

#### 5-Minute Setup

1. **Install Dependencies**
   ```bash
   pip install numpy scipy scikit-image networkx matplotlib seaborn scikit-learn torch torch-geometric
   ```

2. **Run Demo with Synthetic Data**
   ```bash
   cd protein-localization/scripts
   python pipeline.py --output ../output --epochs 20
   ```
   This will create synthetic data for demonstration purposes.

3. **View Results**
   - Check the `output/visualizations/` folder for generated images
   - Open `frontend/index.html` in your browser for the web interface

### Using Your Own Data

1. **Organize Your TIFF Files**
   Place your TIFF files in any directory structure:
   ```
   my_data/
   ├── experiment1/
   │   ├── image1.tif
   │   └── image2.tif
   └── experiment2/
       └── image3.tif
   ```

2. **Run Pipeline**
   ```bash
   python pipeline.py --input /path/to/my_data --output ./results --epochs 50
   ```

3. **Review Outputs**
   - Trained model: `results/models/graph_cnn.pt`
   - Graphs: `results/graphs/*.gml`
   - Visualizations: `results/visualizations/*.png`

### Web Interface Demo

1. **Start Local Server**
   ```bash
   cd protein-localization/frontend
   python -m http.server 8000
   ```

2. **Open Browser**
   Navigate to `http://localhost:8000`

3. **Upload & Classify**
   - Drag and drop a TIFF file
   - View real-time processing
   - Explore interactive visualizations

### Testing Individual Components

#### Test TIFF Loader
```bash
python tiff_loader.py /path/to/tiff/directory
```

#### Test Preprocessing
```bash
python preprocessing.py
```

#### Test Graph Construction
```bash
python graph_construction.py
```

#### Test Model
```bash
python model_training.py
```

#### Test Visualization
```bash
python visualization.py
```

### Expected Outputs

After running the pipeline, you should see:
- Console output showing progress through each step
- Generated graph files (.gml)
- Trained model file (.pt)
- Visualization images (.png)
- Processing statistics

### Next Steps

1. **Tune Hyperparameters**: Adjust learning rate, epochs, model architecture
2. **Add More Data**: Increase training data for better accuracy
3. **Experiment with Features**: Try different node features
4. **Deploy**: Set up as a web service for production use

### Troubleshooting

**Problem**: Import errors for torch_geometric
**Solution**: Install torch-scatter and torch-sparse:
```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cpu.html
```

**Problem**: No TIFF files found
**Solution**: The pipeline will automatically generate synthetic data for demonstration

**Problem**: Out of memory during training
**Solution**: Reduce batch size or use smaller images

### Support

For issues or questions:
1. Check the main README.md
2. Review the code comments
3. Test individual components separately
