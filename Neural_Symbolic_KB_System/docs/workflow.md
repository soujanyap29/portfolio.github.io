# Workflow

## End-to-End Data Flow

1. User enters text in Streamlit.
2. Frontend calls `process_text(input_text)` in backend API.
3. Backend preprocesses text and runs LLaMA3 extraction via Ollama.
4. Extracted triples are parsed, scored, and filtered.
5. Symbolic reasoning normalizes and validates accepted triples.
6. Knowledge graph is built and saved to JSON.
7. Inference module adds multi-hop inferred triples.
8. Evaluation module computes metrics.
9. Frontend retrieves triples/graph/query responses dynamically.

## Backend–Frontend Communication

- `process_text(input_text)` returns pipeline outputs.
- `get_triples()` returns raw/filtered/refined/inferred views.
- `get_graph()` returns node-link JSON graph payload.
- `query_graph(query)` returns human-readable query results.
