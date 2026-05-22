# System Architecture

The Neural–Symbolic Knowledge Base Construction System combines neural extraction with symbolic reasoning for reliable knowledge graph construction.

## Layered Pipeline

1. **Frontend (Streamlit UI)**
2. **Backend API Interface**
3. **Text Preprocessing Layer**
4. **Neural Extraction Layer (Ollama + LLaMA3)**
5. **Triplet Parsing Layer**
6. **Confidence Estimation Layer**
7. **Symbolic Reasoning Layer**
8. **Knowledge Graph Builder**
9. **Storage Layer (JSON)**
10. **Inference Engine (Multi-hop)**
11. **Query Engine**
12. **Evaluation Engine**
13. **Results back to Frontend**

## Neural–Symbolic Integration

- Neural models provide flexible extraction from unstructured language.
- Symbolic constraints improve consistency by filtering invalid or weak triples.
- Graph reasoning expands knowledge through inferred links.
