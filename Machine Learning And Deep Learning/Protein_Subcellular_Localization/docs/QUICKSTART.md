# Quick Start Guide

## Getting Started in 5 Minutes

### Prerequisites
- Python 3.8+
- 8GB+ RAM
- TIFF microscopy images

### Step 1: Setup (2 minutes)

```bash
# Navigate to project directory
cd "Machine Learning And Deep Learning/Protein_Subcellular_Localization"

# Run setup script
python setup.py
```

The setup script will:
- Check Python version
- Create necessary directories
- Install dependencies
- Verify installations

### Step 2: Prepare Your Data (1 minute)

Place your TIFF microscopy images in:
```
/mnt/d/5TH_SEM/CELLULAR/input/
```

**Note:** If this path doesn't exist on your system, you can modify it in `config.yaml`:
```yaml
paths:
  input_dir: "your/custom/path/input"
  output_dir: "your/custom/path/output"
```

### Step 3: Run Analysis (2 minutes)

#### Option A: Jupyter Notebook (Recommended)
```bash
cd notebooks
jupyter notebook automated_pipeline.ipynb
```

Then click "Run All" or run cells sequentially.

#### Option B: Web Interface
```bash
cd frontend
python app.py
```

Open browser to: `http://localhost:5000`

## What Happens Next?

The pipeline will automatically:

1. **Scan** for TIFF images
2. **Segment** using Cellpose
3. **Analyze** with CNN and GNN
4. **Fuse** predictions
5. **Visualize** results
6. **Generate** reports

Results will be saved to:
```
/mnt/d/5TH_SEM/CELLULAR/output/
├── segmented/          # Segmentation images
├── predictions/        # CSV with predictions
├── reports/           # JSON reports per image
└── graphs/            # All visualizations
```

## Example Output

### For each image, you get:
- ✅ Segmentation overlay
- ✅ CNN predictions with probabilities
- ✅ GNN predictions with probabilities
- ✅ Fused predictions (most accurate)
- ✅ Confusion matrix
- ✅ High-resolution graphs (300 DPI)
- ✅ JSON report with all metrics

### Batch processing produces:
- ✅ Combined predictions CSV
- ✅ Performance comparison charts
- ✅ Summary statistics
- ✅ Complete pipeline notebook

## Customization

### Change Classes
Edit `config.yaml`:
```yaml
classes:
  - "Your_Class_1"
  - "Your_Class_2"
  - "Your_Class_3"
```

### Adjust Model Weights
Edit `config.yaml`:
```yaml
fusion:
  cnn_weight: 0.7  # Increase for more CNN influence
  gnn_weight: 0.3  # Decrease GNN influence
```

### Change Segmentation Parameters
Edit `config.yaml`:
```yaml
segmentation:
  model_type: "cyto"  # or "nuclei"
  diameter: 30        # Expected cell size
```

## Troubleshooting

### Problem: "No module named 'cellpose'"
**Solution:**
```bash
pip install cellpose
```

### Problem: "CUDA out of memory"
**Solution:** Reduce batch size in `config.yaml`:
```yaml
cnn:
  batch_size: 16  # Reduce from 32
```

### Problem: "No images found"
**Solution:** 
1. Check path in `config.yaml`
2. Ensure images have .tif or .tiff extension
3. Check read permissions

### Problem: Web interface doesn't start
**Solution:**
```bash
# Check if Flask is installed
pip install Flask Flask-CORS

# Try different port
python app.py --port 8080
```

## Next Steps

### 1. Train Your Own Models

See `notebooks/training_example.ipynb` for:
- Preparing training data
- Training CNN from scratch
- Training GNN
- Saving trained models

### 2. Customize Visualizations

Edit `backend/utils/visualization.py` to add:
- Custom plots
- Different color schemes
- Additional metrics

### 3. Add New Features

The modular design makes it easy to:
- Add new model architectures
- Implement different fusion strategies
- Extend evaluation metrics

## Performance Tips

### For Faster Processing:
1. Use GPU (CUDA) if available
2. Reduce image resolution in `config.yaml`
3. Decrease number of superpixels
4. Use smaller models

### For Better Accuracy:
1. Increase training epochs
2. Use data augmentation
3. Fine-tune hyperparameters
4. Collect more training data

## Getting Help

### Documentation:
- `README.md` - Overview and features
- `docs/TECHNICAL_DOCUMENTATION.md` - Mathematical details
- `config.yaml` - All configurable parameters

### Code Structure:
```
backend/
├── models/        # CNN and GNN implementations
├── utils/         # Helper functions
└── segmentation/  # Cellpose integration

frontend/
├── app.py         # Web server
└── templates/     # HTML interface

notebooks/
└── *.ipynb        # Jupyter workflows
```

### Common Questions:

**Q: Can I use different image formats?**
A: Currently supports TIFF only. Modify `image_preprocessing.py` for other formats.

**Q: How do I add more GNN architectures?**
A: Implement in `backend/models/gnn_model.py` following the existing pattern.

**Q: Can I deploy this as a web service?**
A: Yes! Use gunicorn or uwsgi to deploy the Flask app.

**Q: What about other cell types?**
A: The system works for any microscopy images. Adjust segmentation parameters.

## Success Checklist

- [ ] Setup completed without errors
- [ ] Sample data in input directory
- [ ] Pipeline runs successfully
- [ ] Results saved to output directory
- [ ] Visualizations look correct
- [ ] Web interface accessible
- [ ] Reports generated

## Resources

- Cellpose documentation: https://cellpose.readthedocs.io/
- PyTorch tutorials: https://pytorch.org/tutorials/
- Graph Neural Networks: https://pytorch-geometric.readthedocs.io/

---

**You're ready to analyze protein localization!** 🔬✨

For questions or issues, please refer to the full documentation or create an issue in the repository.
