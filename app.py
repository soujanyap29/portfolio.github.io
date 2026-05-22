import json
from typing import Any, Dict

import streamlit as st

from backend import get_graph, get_triples, process_text, query_graph


st.set_page_config(page_title="Neural–Symbolic Knowledge Base", layout="wide")
st.title("Neural–Symbolic Knowledge Base Construction System")

st.subheader("Input")
input_text = st.text_area(
    "Enter raw unstructured text",
    height=180,
    placeholder="Example: Alice works at Acme Corp. Acme Corp is located in Bangalore. Bangalore is in India.",
)

if st.button("Process Text", type="primary"):
    with st.spinner("Running end-to-end pipeline..."):
        result = process_text(input_text)
    st.session_state["result"] = result

result = st.session_state.get("result")

if result:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Extracted Triples")
        st.dataframe(result["raw_triples"], use_container_width=True)
        st.subheader("Filtered Triples")
        st.dataframe(result["filtered_triples"], use_container_width=True)
        st.subheader("Refined Triples")
        st.dataframe(result.get("refined_triples", []), use_container_width=True)
    with col2:
        st.subheader("Inferred Triples")
        st.dataframe(result["inferred_triples"], use_container_width=True)
        st.subheader("Evaluation")
        st.json(result["evaluation"])

st.subheader("Query")
query = st.text_input(
    "Use plain text or prefixes like subject:, object:, relation:",
    placeholder="relation: located in",
)
if st.button("Run Query"):
    st.session_state["query_result"] = query_graph(query)

if "query_result" in st.session_state:
    st.dataframe(st.session_state["query_result"]["results"], use_container_width=True)

st.subheader("Knowledge Graph")


def dot_escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("\"", "\\\"")


def to_dot(graph_data: Dict[str, Any]) -> str:
    nodes = graph_data.get("nodes", [])
    links = graph_data.get("links", [])
    lines = ["digraph G {", "rankdir=LR;"]
    for n in nodes:
        node = dot_escape(n.get("id"))
        lines.append(f'"{node}";')
    for e in links:
        src = dot_escape(e.get("source"))
        dst = dot_escape(e.get("target"))
        relation = dot_escape(e.get("relation", ""))
        conf = e.get("confidence", 0.0)
        label = dot_escape(f"{relation} ({conf})")
        lines.append(f'"{src}" -> "{dst}" [label="{label}"];')
    lines.append("}")
    return "\n".join(lines)

try:
    graph_payload = get_graph()
    if graph_payload.get("nodes"):
        st.graphviz_chart(to_dot(graph_payload))
    else:
        st.info("Graph is empty. Process text to build the graph.")
except (KeyError, TypeError, ValueError) as exc:
    st.warning(f"Unable to render graph: {exc}")

with st.expander("Backend Interface Snapshot"):
    st.json({"triples": get_triples(), "graph": get_graph()})
