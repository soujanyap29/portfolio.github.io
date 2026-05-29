# Protein Subcellular Localization and Alzheimer's Disease Prediction

Production-ready multimodal pipeline using microscopy TIFF images + protein sequence features.

## Highlights
- Multilabel localization prediction for 15 subcellular classes.
- Multimodal fusion options: concatenation and attention.
- ESM-2 sequence embeddings with caching and BiLSTM fallback.
- Future-ready Alzheimer's prediction head (uses labels when available, placeholder otherwise).
- TensorBoard logging, early stopping, checkpoints, evaluation plots, Grad-CAM.
- Conference-quality Streamlit inference dashboard.

## Project Structure
- `data/`
- `notebooks/`
- `src/`
- `app/`
- `outputs/`
- `docs/`

## Required Data Paths
- CSV: `/home/lab3/min/merged_dataset.csv`
- Images: `/home/lab3/min/images`

## Quick Start
```bash
bash setup.sh
source .venv/bin/activate
python -m src.main
streamlit run app/streamlit_app.py
```

## Outputs
Saved under `/home/lab3/min/output`:
- models/checkpoints/logs
- plots/Grad-CAM/evaluation metrics
- predictions/reports/exported CSV

## Alzheimer's Labels Integration
If `alzheimers_label` exists in CSV, the supervised `AlzheimersHead` is used.
If absent, the `AlzheimersPlaceholderModule` keeps architecture future-ready and can be swapped seamlessly once labels are added.
