from pathlib import Path
import sys
from typing import Any, Dict

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.api_interface import get_graph, get_triples, process_text, query_graph  # noqa: E402


st.set_page_config(page_title="Neural–Symbolic KB System", layout="wide")
st.title("Neural–Symbolic Knowledge Base Construction System")

st.markdown("End-to-end pipeline: raw text → extraction → reasoning → graph → queryable knowledge.")

input_text = st.text_area(
    "Enter raw text",
    height=180,
    placeholder="Alice works at Acme Corp. Acme Corp is located in Bangalore. Bangalore is in India.",
)

if st.button("Process Text", type="primary"):
    try:
        with st.spinner("Running backend pipeline..."):
            st.session_state["pipeline_output"] = process_text(input_text)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Pipeline execution failed: {exc}")

output = st.session_state.get("pipeline_output")
if output:
    st.subheader("Extracted and Processed Triples")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Raw Triples**")
        st.dataframe(pd.DataFrame(output.get("raw_triples", [])), use_container_width=True)

    with col2:
        st.markdown("**Filtered (Accepted) Triples**")
        st.dataframe(pd.DataFrame(output.get("accepted_triples", [])), use_container_width=True)

    with col3:
        st.markdown("**Inferred Triples**")
        st.dataframe(pd.DataFrame(output.get("inferred_triples", [])), use_container_width=True)

    st.subheader("Evaluation")
    st.json(output.get("evaluation", {}))

st.subheader("Query the Knowledge Graph")
query_text = st.text_input(
    "Use `subject:`, `object:`, `relation:` or free text",
    placeholder="relation: located in",
)
if st.button("Run Query"):
    try:
        st.session_state["query_output"] = query_graph(query_text)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Query failed: {exc}")

query_output = st.session_state.get("query_output")
if query_output:
    st.info(query_output.get("message", ""))
    st.dataframe(pd.DataFrame(query_output.get("results", [])), use_container_width=True)


def _escape_dot(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def to_dot(graph_payload: Dict[str, Any]) -> str:
    nodes = graph_payload.get("nodes", [])
    links = graph_payload.get("links", [])
    lines = ["digraph G {", "rankdir=LR;", "node [shape=ellipse];"]

    for node in nodes:
        node_id = _escape_dot(node.get("id", ""))
        lines.append(f'"{node_id}";')

    for edge in links:
        src = _escape_dot(edge.get("source", ""))
        dst = _escape_dot(edge.get("target", ""))
        rel = _escape_dot(edge.get("relation", ""))
        conf = edge.get("confidence", 0.0)
        label = _escape_dot(f"{rel} ({conf})")
        lines.append(f'"{src}" -> "{dst}" [label="{label}"];')

    lines.append("}")
    return "\n".join(lines)


st.subheader("Knowledge Graph Visualization")
try:
    graph_data = get_graph()
    if graph_data.get("nodes"):
        st.graphviz_chart(to_dot(graph_data), use_container_width=True)
    else:
        st.warning("Graph is empty. Process text to populate the graph.")
except Exception as exc:  # noqa: BLE001
    st.error(f"Unable to render graph: {exc}")

with st.expander("Backend API Snapshot"):
    st.json({"triples": get_triples(), "graph": get_graph()})
