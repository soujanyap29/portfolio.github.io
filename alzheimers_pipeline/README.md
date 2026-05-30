# Alzheimer's MRI Classification Pipeline (PyTorch)

## Selected public dataset
Recommended: **Kaggle Alzheimer's MRI Dataset** (`sachinkumar413/alzheimer-mri-dataset`).

Why:
- Includes 4 practical classes for demos: Non Demented, Very Mild, Mild, Moderate.
- Preorganized image folders for direct deep-learning training with PyTorch.
- Small enough for college project timelines and Streamlit demonstrations.

## What this pipeline does
- Checks `/home/lab3/min/merged_dataset.csv` for usable Alzheimer's labels + image paths.
- If not usable, downloads the public Kaggle MRI dataset automatically.
- Creates stratified train/validation/test splits.
- Trains and compares **EfficientNet-B0**, **ResNet50**, and **ViT-B/16** with transfer learning.
- Uses mixed precision, checkpointing, early stopping, LR scheduling, TensorBoard logging.
- Saves:
  - `/home/lab3/min/output/checkpoints/alzheimers_model.pt`
  - metrics, confusion matrix, ROC curves, and training curves under `/home/lab3/min/output/`
- Includes inference CLI and Streamlit UI with confidence scores and Grad-CAM (CNN models).

## Install
```bash
pip install -r /tmp/workspace/soujanyap29/portfolio.github.io/alzheimers_pipeline/requirements.txt
```

## Prepare data
```bash
python -m alzheimers_pipeline.download_and_prepare_data
```

## Train (and save `.pt`)
```bash
python -m alzheimers_pipeline.train --epochs 10 --batch-size 32
```

## Evaluate
```bash
python -m alzheimers_pipeline.evaluate
```

## Inference
```bash
python -m alzheimers_pipeline.infer --image /path/to/mri_image.jpg
```

## Streamlit
```bash
streamlit run /tmp/workspace/soujanyap29/portfolio.github.io/alzheimers_pipeline/streamlit_app.py
```

## Kaggle API setup (if public fallback is used)
1. Create Kaggle token from your Kaggle account.
2. Place `kaggle.json` in `~/.kaggle/kaggle.json`.
3. Run:
```bash
chmod 600 ~/.kaggle/kaggle.json
```
