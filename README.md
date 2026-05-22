# Neural–Symbolic Knowledge Base Construction System

This repository contains a complete end-to-end neural–symbolic pipeline that converts unstructured text into a structured knowledge graph.

## Components

- `backend.py` — Modular backend pipeline with:
  - preprocessing
  - LLM-based neural extraction via local Ollama
  - triplet parsing and cleaning
  - confidence scoring and filtering
  - symbolic reasoning
  - graph construction and JSON storage
  - multi-hop inference
  - query and evaluation modules
- `app.py` — Streamlit frontend connected directly to backend interface functions.
- `neural_symbolic_kb_system.ipynb` — Publication-style notebook that documents architecture, workflow, module behavior, integration, evaluation, and limitations.

## Backend Interface

The backend exposes these frontend-facing functions:

- `process_text(input_text)`
- `get_triples()`
- `get_graph()`
- `query_graph(query)`

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure Ollama is running locally and pull model:
   ```bash
   ollama pull llama3:latest
   ```
4. Launch Streamlit:
   ```bash
   streamlit run app.py
   ```

## Query Examples

- `subject: Alice`
- `object: India`
- `relation: located in`
- `works at`
