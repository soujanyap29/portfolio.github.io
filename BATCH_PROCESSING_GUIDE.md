# Batch Processing Guide

## Processing ALL TIFF Files from Multiple Protein Folders

This guide explains how to process all TIFF files from a nested directory structure like:

```
D:\5TH_SEM\CELLULAR\input\
├── AAMP_ENSG00000127837\
│   ├── OC-FOV_AAMP_ENSG00000127837_CID001050_FID00013888_stack.tif
│   ├── OC-FOV_AAMP_ENSG00000127837_CID001050_FID00026369_stack.tif
│   └── ... (more TIFF files)
├── AATF_ENSG00000275700\
│   ├── OC-FOV_AATF_ENSG00000275700_*.tif
│   └── ... (more TIFF files)
├── (many more protein folders)
└── ...
```

## Quick Start

### Process ALL Files (Default Behavior)

The pipeline **automatically processes ALL TIFF files from ALL subdirectories**:

```bash
cd protein-localization/scripts
python pipeline.py --input D:\5TH_SEM\CELLULAR\input --output D:\5TH_SEM\CELLULAR\output --epochs 20
```

**What happens:**
1. ✅ Recursively scans ALL subdirectories (AAMP_*, AATF_*, etc.)
2. ✅ Finds ALL `.tif` and `.tiff` files
3. ✅ Processes each file: load → segment → extract features → build graph
4. ✅ Trains a single model on all processed data
5. ✅ Generates visualizations and saves results

## Usage Options

### 1. Process All Files (Default)

```bash
python pipeline.py --input D:\5TH_SEM\CELLULAR\input --output D:\5TH_SEM\CELLULAR\output
```

- Processes **every TIFF file** in **every subfolder**
- No limit on number of files
- Best for: Final training runs

### 2. Limit Number of Files (Testing)

```bash
python pipeline.py --input D:\5TH_SEM\CELLULAR\input --output D:\5TH_SEM\CELLULAR\output --max-files 50
```

- Processes only first 50 files found
- Useful for quick testing
- Best for: Development and debugging

### 3. Custom Epochs

```bash
python pipeline.py --input D:\5TH_SEM\CELLULAR\input --output D:\5TH_SEM\CELLULAR\output --epochs 100
```

- More epochs = better training (but longer time)
- Recommended: 20-50 epochs for testing, 50-200 for final model

## Output Structure

After processing, your output directory will contain:

```
D:\5TH_SEM\CELLULAR\output\
├── graphs\
│   ├── graph_0.gml         # Graph from first TIFF
│   ├── graph_1.gml         # Graph from second TIFF
│   └── ...                 # One graph per processed TIFF
├── models\
│   └── graph_cnn.pt        # Trained model (single model for all data)
├── visualizations\
│   ├── training_history.png        # Loss and accuracy curves
│   ├── confusion_matrix.png        # Classification performance
│   └── prediction_graphs.png       # Sample predictions
└── data\
    ├── features_0.pkl      # Extracted features from first TIFF
    ├── features_1.pkl      # Extracted features from second TIFF
    └── ...                 # One features file per TIFF
```

## Jupyter Notebook Example

For interactive processing with progress visualization:

```python
from pipeline import ProteinLocalizationPipeline

# Configure paths
input_dir = "D:\\5TH_SEM\\CELLULAR\\input"
output_dir = "D:\\5TH_SEM\\CELLULAR\\output"

# Create pipeline - processes ALL files by default
pipeline = ProteinLocalizationPipeline(input_dir, output_dir)

# Run complete pipeline
model, history = pipeline.run_complete_pipeline(epochs=20)

print(f"✓ Processed {len(pipeline.features_list)} TIFF files")
print(f"✓ Final accuracy: {history['test_accuracy'][-1]:.4f}")
```

**To limit files in Jupyter:**

```python
# Process only first 50 files
pipeline = ProteinLocalizationPipeline(input_dir, output_dir, max_files=50)
model, history = pipeline.run_complete_pipeline(epochs=20)
```

## Generate Network Diagrams for Individual Files

To create biological network diagrams for specific TIFF files:

```bash
# Single file
python generate_biological_network.py \
  --input "D:\5TH_SEM\CELLULAR\input\AAMP_ENSG00000127837\OC-FOV_AAMP_*.tif" \
  --output "D:\5TH_SEM\CELLULAR\output\AAMP_network.png"
```

**Batch generate networks for all files:**

```bash
# Windows PowerShell
Get-ChildItem -Path "D:\5TH_SEM\CELLULAR\input" -Filter *.tif -Recurse | ForEach-Object {
    $outputName = $_.BaseName + "_network.png"
    python generate_biological_network.py --input $_.FullName --output "D:\5TH_SEM\CELLULAR\output\networks\$outputName"
}

# Linux/WSL Bash
find /mnt/d/5TH_SEM/CELLULAR/input -name "*.tif" -type f | while read file; do
    filename=$(basename "$file" .tif)
    python generate_biological_network.py --input "$file" --output "/mnt/d/5TH_SEM/CELLULAR/output/networks/${filename}_network.png"
done
```

## Performance Considerations

### Processing Time Estimates

For a **4D TIFF** image (106 × 2 × 600 × 600):
- Loading: 1-2 seconds
- Segmentation: 5-10 seconds
- Graph construction: 1-2 seconds
- **Total per file: ~10-15 seconds**

For **1000 TIFF files**:
- Preprocessing: ~3-4 hours
- Training (20 epochs): ~10-20 minutes
- **Total: ~4-5 hours**

### Memory Requirements

- **Per TIFF**: ~50-100 MB RAM
- **Batch processing**: Processes files sequentially to manage memory
- **Recommended**: 8GB+ RAM for large datasets

### Optimization Tips

1. **Use max-files for testing**:
   ```bash
   # Test with 10 files first
   python pipeline.py --max-files 10 --epochs 10
   ```

2. **Enable GPU acceleration** (if available):
   - PyTorch automatically uses CUDA if available
   - 5-10x faster training with GPU

3. **Parallel processing** (advanced):
   - Modify `pipeline.py` to use `multiprocessing` for loading
   - Can process multiple TIFFs simultaneously

## Troubleshooting

### Issue: "Too many files, running out of memory"

**Solution:** Use `--max-files` to process in batches:

```bash
# Batch 1: First 100 files
python pipeline.py --input D:\5TH_SEM\CELLULAR\input --max-files 100 --output output_batch1

# Batch 2: Next 100 files (requires code modification to skip first 100)
```

### Issue: "Processing is too slow"

**Solutions:**
1. Reduce image size during loading (modify `tiff_loader.py`)
2. Use fewer epochs: `--epochs 10`
3. Process subset: `--max-files 50`

### Issue: "Segmentation finds too few/many regions"

**Solution:** Adjust thresholds in `preprocessing.py`:
- Increase `threshold_abs` for fewer regions
- Decrease `threshold_abs` for more regions

## Summary

✅ **Default behavior**: Processes ALL TIFF files from ALL subdirectories  
✅ **Recursive scanning**: Automatically finds files in nested folders  
✅ **Progress tracking**: Shows which file is being processed  
✅ **Flexible limits**: Use `--max-files` to control batch size  
✅ **Single model**: Trains one model on all processed data  
✅ **Organized output**: Separate folders for graphs, models, visualizations  

**Key Command:**
```bash
python pipeline.py --input D:\5TH_SEM\CELLULAR\input --output D:\5TH_SEM\CELLULAR\output
```

This single command processes your entire dataset!
