# Using the Protein Localization Pipeline

## No Copy-Paste Required! 

You have **3 easy options** to use this pipeline:

---

## Option 1: Jupyter Notebook (Recommended) 🎯

**File**: `Protein_Localization_Pipeline.ipynb`

### Quick Start:
```bash
# 1. Install Jupyter (if needed)
pip install jupyter

# 2. Open the notebook
jupyter notebook Protein_Localization_Pipeline.ipynb

# 3. Run the cells!
```

### What's Inside:
- ✅ **Easy imports** - All modules pre-configured
- ✅ **3 execution modes**:
  - Complete pipeline (one cell)
  - Step-by-step with visualizations
  - Use pre-trained models
- ✅ **Interactive plots** - See results inline
- ✅ **Demo mode** - Works even without TIFF files
- ✅ **Examples for everything** - Loading, preprocessing, training, prediction

### Example Cells:

**Run Complete Pipeline (Easiest):**
```python
from pipeline import ProteinLocalizationPipeline

pipeline = ProteinLocalizationPipeline("input_dir", "output_dir")
model, history = pipeline.run_complete_pipeline(epochs=20)
```

**Or Step-by-Step:**
```python
# Load images
from tiff_loader import TIFFLoader
loader = TIFFLoader("your_data_path")
tiff_files = loader.scan_directory()

# Preprocess
from preprocessing import ImagePreprocessor
preprocessor = ImagePreprocessor()
labeled_regions, features = preprocessor.process_image(image)

# Build graph
from graph_construction import GraphConstructor
constructor = GraphConstructor()
graph = constructor.create_graph_from_regions(features)

# Train model
from model_training import GraphCNN, ModelTrainer
model = GraphCNN(num_features=4, num_classes=5)
trainer = ModelTrainer(model)
# ... and so on
```

---

## Option 2: Command Line (For Automation)

### Quick Start:
```bash
# Run complete pipeline
cd protein-localization/scripts
python pipeline.py --input D:\5TH_SEM\CELLULAR\input --output D:\5TH_SEM\CELLULAR\output --epochs 50
```

### When to Use:
- ✅ Batch processing multiple datasets
- ✅ Running on remote servers
- ✅ Automated workflows
- ✅ No GUI environment

---

## Option 3: Import as Python Module

### Quick Start:
```python
# In your own Python script or notebook
import sys
sys.path.insert(0, 'path/to/protein-localization/scripts')

from pipeline import ProteinLocalizationPipeline

pipeline = ProteinLocalizationPipeline(
    input_dir="D:\\5TH_SEM\\CELLULAR\\input",
    output_dir="D:\\5TH_SEM\\CELLULAR\\output"
)

model, history = pipeline.run_complete_pipeline(epochs=50)
print(f"Final accuracy: {history['test_accuracy'][-1]:.4f}")
```

### When to Use:
- ✅ Integrating into larger projects
- ✅ Custom workflows
- ✅ Building on top of the pipeline

---

## Comparison

| Feature | Jupyter Notebook | Command Line | Python Import |
|---------|-----------------|--------------|---------------|
| **Ease of Use** | ⭐⭐⭐⭐⭐ Easiest | ⭐⭐⭐⭐ Easy | ⭐⭐⭐ Moderate |
| **Visualizations** | ✅ Inline | ❌ Saved to files | ✅ If in notebook |
| **Step-by-step** | ✅ Yes | ❌ All at once | ✅ Yes |
| **Interactivity** | ✅ High | ❌ None | ⭐ Depends |
| **Batch Processing** | ❌ Manual | ✅ Easy | ✅ Easy |
| **Best For** | Learning, Exploration | Automation, Remote | Integration, Custom |

---

## What You Get

All options give you:
- ✅ Automatic TIFF loading from nested directories
- ✅ Image segmentation and feature extraction
- ✅ Graph construction with spatial relationships
- ✅ Graph-CNN training and classification
- ✅ Beautiful visualizations
- ✅ Saved models and results

### Output Structure:
```
output/
├── graphs/              # .gml graph files with node labels
├── models/              # graph_cnn.pt trained model
├── visualizations/      # PNG images of results
│   ├── training_history.png
│   ├── graph_0_prediction.png
│   └── ...
└── data/                # Processed features
```

---

## Frequently Asked Questions

### Q: Do I need to copy-paste all the Python code?
**A: No!** Just use the Jupyter notebook or import the modules.

### Q: What if I don't have TIFF files?
**A: No problem!** The pipeline will generate synthetic data for demonstration.

### Q: Can I use my own data?
**A: Yes!** Just put your TIFF files in any directory structure and point the pipeline to it.

### Q: Which option should I choose?
**A: Start with the Jupyter notebook** - it's the easiest way to learn and see results interactively.

### Q: How do I modify the code?
**A: Edit the files in the `scripts/` directory** - they're well-documented Python modules.

---

## Need Help?

- 📖 **README.md** - Complete documentation
- 🚀 **QUICKSTART.md** - 5-minute setup guide  
- 📊 **PROJECT_OVERVIEW.md** - Technical details
- 💻 **Protein_Localization_Pipeline.ipynb** - Interactive examples
- 🌐 **frontend/index.html** - Web interface

---

## Next Steps

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Choose your option**: Jupyter notebook, command line, or Python import
3. **Run the pipeline**: Process your images and get predictions
4. **Check results**: Look in the output directory for graphs, models, and visualizations

**Remember: You don't need to copy-paste anything!** 🎉
