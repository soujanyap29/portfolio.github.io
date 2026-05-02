# 🧠 Neural–Symbolic Knowledge Base Construction System

A publication-level, end-to-end Generative AI system that transforms raw unstructured text into a structured, queryable **knowledge graph** using a Neural–Symbolic framework.

---

## System Architecture

```
Frontend (Streamlit UI)  ←──────────────────────────────────────┐
        ↓                                                        │
Backend API (backend.py / notebook)                             │
        ↓                                                        │
Text Preprocessing Layer                                        │
        ↓                                                        │
Neural Extraction  (LLM via Ollama / rule-based fallback)       │
        ↓                                                        │
Triplet Parsing & Structuring                                   │
        ↓                                                        │
Confidence Estimation                                           │
        ↓                                                        │
Symbolic Reasoning Engine                                       │
        ↓                                                        │
Knowledge Graph Construction (NetworkX)                         │
        ↓                                                        │
Storage Layer (JSON on disk)                                    │
        ↓                                                        │
Inference Engine (multi-hop reasoning)                          │
        ↓                                                        │
Query Processing Module                                         │
        ↓                                                        │
Evaluation Module                                               │
        ↓                                                        │
Results returned to Frontend ───────────────────────────────────┘
```

---

## Repository Structure

| File | Purpose |
|------|---------|
| `neural_symbolic_kb.ipynb` | Jupyter Notebook — complete pipeline with markdown documentation, runnable cells, and demo |
| `backend.py` | Importable Python module mirroring the notebook; used by the Streamlit UI |
| `app.py` | Streamlit frontend — fully connected to `backend.py` |
| `requirements.txt` | All Python dependencies |

---

## Setup & Installation

### 1. Prerequisites

- Python 3.9 or higher
- [Ollama](https://ollama.ai/) installed and running locally

### 2. Install Ollama and pull the model

```bash
# Install Ollama (see https://ollama.ai for platform-specific instructions)
ollama pull llama3
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit frontend

```bash
streamlit run app.py
```

### 5. (Optional) Open the Jupyter Notebook

```bash
jupyter notebook neural_symbolic_kb.ipynb
```

---

## Pipeline Phases

| Phase | Module | Description |
|-------|--------|-------------|
| 1 | Initialization | Environment check, Ollama connectivity validation |
| 2 | Preprocessing | Noise removal, normalisation, sentence segmentation |
| 3 | Neural Extraction | LLM prompt engineering, retry logic, rule-based fallback |
| 4 | Triplet Parsing | Regex extraction, normalisation, deduplication |
| 5 | Confidence Estimation | Heuristic scoring, threshold filtering, logging |
| 6 | Symbolic Reasoning | Banned relation removal, entity canonicalisation, deduplication |
| 7 | Knowledge Graph | NetworkX MultiDiGraph with edge confidence attributes |
| 8 | Storage | JSON persistence to `kg_storage/` directory |
| 9 | Inference Engine | Transitive rule application, 2-hop+ multi-hop reasoning |
| 10 | Query Module | Subject / object / relation queries, free-text dispatch |
| 11 | Evaluation | Unsupervised quality metrics, stability analysis |
| 12 | Feedback Loop | Error pattern detection, prompt & rule improvement suggestions |
| 13 | Backend API | `process_text()`, `get_triples()`, `get_graph()`, `query_graph()` |
| 14 | Documentation | This README + notebook markdown cells |

---

## Backend API Reference

These four functions are the integration points between the frontend and the pipeline:

```python
from backend import process_text, get_triples, get_graph, query_graph

# Run the full pipeline on raw text
results = process_text("Albert Einstein was born in Germany...")

# Retrieve triple sets from the last run
triples = get_triples()   # keys: raw, accepted, rejected, refined, inferred

# Get a JSON snapshot of the current knowledge graph
graph = get_graph()       # keys: graph, stats, evaluation

# Query the knowledge graph
answer = query_graph("tell me about Albert Einstein")
print(answer['formatted'])
```

---

## Streamlit Frontend Features

- **Extract Knowledge tab** — text input, example presets, full pipeline execution, triple display
- **Knowledge Graph tab** — interactive Plotly graph visualisation, edge table, JSON export
- **Query tab** — free-text queries, structured subject/object/relation search
- **Evaluation tab** — metric cards, bar chart, raw JSON report, feedback suggestions

---

## Offline / Fallback Mode

When Ollama is unavailable, the system automatically activates a **rule-based extractor** that handles common English sentence patterns (born_in, founded, won, works_at, studied_at, located_in, etc.). All downstream pipeline stages (confidence scoring, symbolic reasoning, graph construction, inference, querying) run identically regardless of extraction mode.

---

## Extending the System

- **Add new extraction patterns** → extend `RuleBasedExtractor._PATTERNS` in `backend.py`
- **Add inference rules** → extend `_INFERENCE_RULES` list in `backend.py`
- **Add entity aliases** → extend `_ENTITY_ALIASES` dict in `backend.py`
- **Tune confidence** → adjust `CONFIDENCE_THRESHOLD` or the scoring heuristics in `ConfidenceEstimator`
- **Swap the LLM** → change `OLLAMA_MODEL` to any model available in your Ollama instance

---

## Requirements

See `requirements.txt` for the full list. Key dependencies:

- `ollama` — local LLM client
- `networkx` — knowledge graph
- `streamlit` — frontend UI
- `plotly` — interactive graph visualisation
- `pandas`, `numpy` — data handling

---

## Limitations

- LLM extraction quality depends on model capability; smaller models may miss subtle relations.
- Rule-based fallback covers common English patterns; domain-specific texts may need additional patterns.
- Confidence scoring uses structural heuristics rather than a trained classifier.
- The inference engine applies forward-chaining rules; complex ontological reasoning is out of scope.

---

## License

This project is part of an academic portfolio. Refer to the repository owner for licensing terms.
