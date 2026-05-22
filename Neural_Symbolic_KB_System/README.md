# Neural–Symbolic Knowledge Base Construction System

A complete end-to-end Generative AI project that converts unstructured text into a structured knowledge base using a neural–symbolic pipeline.

## Project Overview

This implementation includes:

- Modular backend (preprocessing, extraction, parsing, confidence, symbolic reasoning, graph building, inference, querying, evaluation)
- Streamlit frontend connected directly to backend API functions
- JSON-based storage for raw outputs, filtered outputs, and knowledge graph
- Multi-hop inference for knowledge expansion
- Notebook with full executable workflow and explanations

## Folder Structure

```text
Neural_Symbolic_KB_System/
│
├── backend/
│   ├── notebook/
│   │   └── neural_symbolic_kb.ipynb
│   ├── modules/
│   │   ├── preprocessing.py
│   │   ├── extraction.py
│   │   ├── parser.py
│   │   ├── confidence.py
│   │   ├── symbolic_reasoning.py
│   │   ├── graph_builder.py
│   │   ├── inference_engine.py
│   │   ├── query_engine.py
│   │   ├── evaluation.py
│   │   └── storage.py
│   ├── data/
│   │   ├── raw_outputs.json
│   │   ├── filtered_outputs.json
│   │   └── knowledge_graph.json
│   └── api_interface.py
├── frontend/
│   └── streamlit_app.py
├── docs/
│   ├── architecture.md
│   ├── workflow.md
│   ├── methodology.md
│   └── results.md
├── requirements.txt
├── README.md
└── run_project.bat
```

## Setup Instructions

### 1) Create Virtual Environment

**Windows**

```bat
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install Dependencies

```bash
pip install -r requirements.txt
```

### 3) Ollama Setup

Install Ollama from: <https://ollama.com/download>

Then pull and test model:

```bash
ollama pull llama3
ollama run llama3 "Return exactly: (Alice, works at, Acme Corp)"
```

Keep Ollama running locally while using the system.

### 4) Run Frontend

From `Neural_Symbolic_KB_System` folder:

```bash
streamlit run frontend/streamlit_app.py
```

## Backend API Functions

- `process_text(input_text)`
- `get_triples()`
- `get_graph()`
- `query_graph(query)`

## Expected Outputs

- Raw and filtered triples
- Inferred triples
- Query responses
- Interactive graph rendering
- Evaluation metrics

## Screenshots

Add frontend screenshots of:
- input + triples
- query results
- graph visualization
- evaluation panel
