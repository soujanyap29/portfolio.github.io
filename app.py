"""
Neural–Symbolic Knowledge Base Construction System
Streamlit Frontend — fully connected to the backend pipeline.

Run with:  streamlit run app.py
Requires:  Jupyter kernel NOT needed — the backend modules are imported
           directly from backend.py (auto-generated from the notebook logic).
"""

import json
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Backend loading
# We ship backend.py (a plain-Python copy of the notebook's pipeline cells).
# If it isn't present we fall back to an embedded lightweight version so the
# UI still runs even without the full Ollama stack.
# ─────────────────────────────────────────────────────────────────────────────

BACKEND_PATH = Path(__file__).parent / 'backend.py'

def _load_backend():
    spec = importlib.util.spec_from_file_location('backend', BACKEND_PATH)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@st.cache_resource(show_spinner='Loading backend pipeline...')
def get_backend():
    if not BACKEND_PATH.exists():
        st.error(
            '❌ `backend.py` not found. '
            'Please generate it from the notebook or create it manually.'
        )
        st.stop()
    return _load_backend()


backend = get_backend()

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title='Neural–Symbolic Knowledge Base',
    page_icon='🧠',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.metric-card {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.triple-pill {
    display: inline-block;
    background: #313244;
    color: #cdd6f4;
    border-radius: 20px;
    padding: 4px 12px;
    margin: 3px;
    font-family: monospace;
    font-size: 0.85em;
}
.inferred-pill {
    background: #1e3a5f;
    color: #89dceb;
}
.section-header {
    font-size: 1.1em;
    font-weight: 600;
    color: #cba6f7;
    border-bottom: 1px solid #313244;
    padding-bottom: 6px;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title('🧠 KB System')
    st.caption('Neural–Symbolic Knowledge Base Construction')

    st.divider()

    ollama_ok = getattr(backend, 'OLLAMA_AVAILABLE', False)
    model_name = getattr(backend, 'OLLAMA_MODEL', 'unknown')
    if ollama_ok:
        st.success(f'✅ Ollama online\n\nModel: `{model_name}`')
    else:
        st.warning('⚠️ Ollama offline\n\nUsing rule-based fallback extractor.')

    st.divider()

    accumulate = st.checkbox(
        'Accumulate extractions',
        value=False,
        help='When checked, new text is ADDED to the existing graph rather than replacing it.',
    )

    confidence_threshold = st.slider(
        'Confidence threshold',
        min_value=0.0, max_value=1.0, value=0.40, step=0.05,
        help='Triples scoring below this value are filtered out.',
    )
    # Push threshold update to backend
    if hasattr(backend, '_confidence_est'):
        backend._confidence_est.threshold = confidence_threshold

    st.divider()

    if st.button('🗑️ Reset graph', use_container_width=True):
        if hasattr(backend, '_kg_builder'):
            backend._kg_builder.clear()
        st.session_state.pop('last_results', None)
        st.success('Graph reset.')

    st.divider()
    st.caption('📦 Phases: Preprocessing → LLM → Parse → Confidence → Symbolic → Graph → Inference → Query → Evaluate')

# ─────────────────────────────────────────────────────────────────────────────
# Main tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_extract, tab_graph, tab_query, tab_eval = st.tabs([
    '📥 Extract Knowledge',
    '🕸️ Knowledge Graph',
    '🔍 Query',
    '📊 Evaluation',
])

# ═══════════════════════════════════════════════════════════════════
# TAB 1 — Extract Knowledge
# ═══════════════════════════════════════════════════════════════════

with tab_extract:
    st.header('📥 Extract Knowledge from Text')
    st.caption(
        'Enter any raw text below. The backend pipeline will preprocess it, '
        'extract triples via the LLM (or fallback), score confidence, apply '
        'symbolic rules, and build the knowledge graph.'
    )

    example_texts = {
        'Albert Einstein & Marie Curie': (
            'Albert Einstein was born in Germany in 1879. He developed the theory of '
            'relativity and worked at Princeton University. Einstein won the Nobel Prize '
            'in Physics in 1921. Marie Curie was born in Warsaw, Poland. She discovered '
            'radium and polonium. Marie Curie won the Nobel Prize in Chemistry. She studied '
            'at the University of Paris, which is located in France.'
        ),
        'Tech pioneers': (
            'Tim Berners-Lee invented the World Wide Web. He worked at CERN, which is '
            'located in Switzerland. Steve Jobs co-founded Apple Inc. Linus Torvalds '
            'created the Linux kernel while studying at the University of Helsinki, '
            'which is located in Finland.'
        ),
        'Space exploration': (
            'NASA was founded in 1958 and is located in Washington D.C. Neil Armstrong '
            'was born in Ohio and worked at NASA. He was the first person to walk on '
            'the Moon. Elon Musk founded SpaceX, which is located in Hawthorne, California.'
        ),
    }

    col_ex1, col_ex2 = st.columns([3, 1])
    with col_ex2:
        selected_example = st.selectbox('Load example', ['(custom)'] + list(example_texts))

    default_text = ''
    if selected_example != '(custom)':
        default_text = example_texts.get(selected_example, '')

    input_text = st.text_area(
        'Input text',
        value=default_text,
        height=180,
        placeholder='Paste or type any text here…',
        label_visibility='collapsed',
    )

    run_btn = st.button('🚀 Run Pipeline', type='primary', use_container_width=True)

    if run_btn:
        if not input_text.strip():
            st.warning('Please enter some text first.')
        else:
            with st.spinner('Running full backend pipeline…'):
                try:
                    results = backend.process_text(input_text, accumulate=accumulate)
                    st.session_state['last_results'] = results
                    st.success('Pipeline completed successfully.')
                except Exception as exc:
                    st.error(f'Pipeline error: {exc}')
                    st.stop()

    # ── Display results ───────────────────────────────────────────────────────
    if 'last_results' in st.session_state:
        results = st.session_state['last_results']
        pp      = results.get('preprocessed', {})

        st.divider()

        # Metrics row
        ev = results.get('evaluation', {})
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric('Sentences', pp.get('sent_count', 0))
        m2.metric('Raw triples', ev.get('raw_triples', 0))
        m3.metric('Accepted', ev.get('accepted_triples', 0))
        m4.metric('Refined', ev.get('refined_triples', 0))
        m5.metric('Inferred', ev.get('inferred_triples', 0))

        # Triple display columns
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<p class="section-header">✅ Refined Triples</p>',
                        unsafe_allow_html=True)
            refined = results.get('refined_triples', [])
            if refined:
                for t in refined:
                    conf = t.get('confidence', '')
                    conf_str = f'{conf:.2f}' if isinstance(conf, float) else ''
                    label = f"({t['subject']}, {t['relation']}, {t['object']}) {conf_str}"
                    st.markdown(f'<span class="triple-pill">{label}</span>',
                                unsafe_allow_html=True)
            else:
                st.info('No refined triples yet.')

            st.markdown('<p class="section-header">🔁 Inferred Triples</p>',
                        unsafe_allow_html=True)
            inferred = results.get('inferred_triples', [])
            if inferred:
                for t in inferred:
                    label = f"({t['subject']}, {t['relation']}, {t['object']})"
                    via   = t.get('via', '')
                    st.markdown(
                        f'<span class="triple-pill inferred-pill" title="{via}">'
                        f'{label}</span>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info('No inferred triples (need more connected triples for multi-hop reasoning).')

        with col_b:
            st.markdown('<p class="section-header">❌ Rejected Triples (low confidence)</p>',
                        unsafe_allow_html=True)
            rejected = results.get('rejected_triples', [])
            if rejected:
                for t in rejected:
                    conf = t.get('confidence', '')
                    conf_str = f'{conf:.2f}' if isinstance(conf, float) else ''
                    label = f"({t['subject']}, {t['relation']}, {t['object']}) {conf_str}"
                    st.markdown(f'<span class="triple-pill" style="opacity:0.5">{label}</span>',
                                unsafe_allow_html=True)
            else:
                st.info('No triples were rejected.')

            st.markdown('<p class="section-header">💡 Feedback & Suggestions</p>',
                        unsafe_allow_html=True)
            feedback = results.get('feedback', {})
            suggestions = feedback.get('suggestions', [])
            if suggestions:
                for s in suggestions:
                    st.info(s)
            else:
                st.success('No issues detected — pipeline looks healthy!')

# ═══════════════════════════════════════════════════════════════════
# TAB 2 — Knowledge Graph Visualisation
# ═══════════════════════════════════════════════════════════════════

with tab_graph:
    st.header('🕸️ Knowledge Graph')
    st.caption('Interactive visualisation of the current knowledge graph.')

    graph_data = backend.get_graph()
    stats = graph_data.get('stats', {})

    cs1, cs2, cs3 = st.columns(3)
    cs1.metric('Nodes (entities)', stats.get('nodes', 0))
    cs2.metric('Edges (relations)', stats.get('edges', 0))
    cs3.metric('Total triples logged', stats.get('triples_logged', 0))

    st.divider()

    graph_raw = graph_data.get('graph', {})
    edges     = graph_raw.get('edges', [])

    if not edges:
        st.info('The knowledge graph is empty. Run the pipeline on some text first.')
    else:
        # ── Plotly interactive graph ──────────────────────────────────────────
        try:
            import plotly.graph_objects as go
            import networkx as nx
            import math

            G = nx.DiGraph()
            for e in edges:
                G.add_edge(e['from'], e['to'],
                           label=e.get('relation', ''),
                           weight=e.get('confidence', 0.5))

            pos = nx.spring_layout(G, seed=42, k=2.5)

            # Build edge traces (one per edge for label hover)
            edge_traces = []
            for u, v, data in G.edges(data=True):
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                mid_x  = (x0 + x1) / 2
                mid_y  = (y0 + y1) / 2
                conf   = data.get('weight', 0.5)
                color  = f'rgba(100,200,255,{max(0.2, conf)})'
                edge_traces.append(go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=max(1, conf * 3), color=color),
                    hoverinfo='text',
                    text=f"{u} →[{data.get('label','')}]→ {v}  (conf: {conf:.2f})",
                    showlegend=False,
                ))
                # Label at midpoint
                edge_traces.append(go.Scatter(
                    x=[mid_x], y=[mid_y],
                    mode='text',
                    text=[data.get('label', '')],
                    textfont=dict(size=9, color='#aaaacc'),
                    hoverinfo='none',
                    showlegend=False,
                ))

            # Node trace
            node_x = [pos[n][0] for n in G.nodes()]
            node_y = [pos[n][1] for n in G.nodes()]
            node_labels = list(G.nodes())
            node_degrees = [G.degree(n) for n in G.nodes()]
            node_sizes   = [max(18, 10 + d * 6) for d in node_degrees]

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_labels,
                textposition='top center',
                textfont=dict(size=10),
                marker=dict(
                    size=node_sizes,
                    color=node_degrees,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Degree'),
                    line=dict(width=2, color='#ffffff'),
                ),
                hovertext=[
                    f'{n}  (degree: {G.degree(n)})' for n in G.nodes()
                ],
            )

            fig = go.Figure(
                data=edge_traces + [node_trace],
                layout=go.Layout(
                    title=dict(text='Knowledge Graph', font=dict(size=16)),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=50),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=600,
                ),
            )
            st.plotly_chart(fig, use_container_width=True)

        except ImportError:
            st.warning('Plotly not installed. Showing raw edge table instead.')

        # ── Tabular fallback ─────────────────────────────────────────────────
        st.divider()
        st.subheader('Edge Table')
        import pandas as pd
        rows = [
            {
                'Subject'   : e.get('from', ''),
                'Relation'  : e.get('relation', ''),
                'Object'    : e.get('to', ''),
                'Confidence': round(e.get('confidence', 0), 3),
                'Inferred'  : e.get('inferred', False),
            }
            for e in edges
        ]
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # ── JSON export ───────────────────────────────────────────────────────
        st.download_button(
            label='⬇️ Download graph as JSON',
            data=json.dumps(graph_raw, indent=2),
            file_name='knowledge_graph.json',
            mime='application/json',
        )

# ═══════════════════════════════════════════════════════════════════
# TAB 3 — Query
# ═══════════════════════════════════════════════════════════════════

with tab_query:
    st.header('🔍 Query the Knowledge Graph')
    st.caption(
        'Ask anything about the knowledge graph. Examples: '
        '"tell me about Marie Curie", "who born_in Germany", "what works_at Princeton"'
    )

    query_input = st.text_input(
        'Your query',
        placeholder='e.g. tell me about Albert Einstein',
        key='query_input',
    )

    example_queries = [
        'tell me about Albert Einstein',
        'what do you know about Marie Curie',
        'who born_in Germany',
        'who invented the World Wide Web',
        'what located_in France',
    ]
    chosen_example = st.selectbox('Or pick an example query', [''] + example_queries)
    if chosen_example:
        query_input = chosen_example

    search_btn = st.button('🔎 Search', type='primary')

    if search_btn or (query_input and st.session_state.get('_auto_search') == query_input):
        if not query_input.strip():
            st.warning('Please enter a query.')
        else:
            with st.spinner('Querying graph…'):
                qr = backend.query_graph(query_input)

            st.divider()
            intent  = qr.get('intent', '')
            results = qr.get('results', [])
            formatted = qr.get('formatted', 'No results.')

            st.markdown(f'**Detected intent:** `{intent}`')
            st.markdown(f'**Results found:** {len(results)}')

            if results:
                import pandas as pd
                rows = []
                for r in results:
                    rows.append({
                        'Subject'   : r.get('subject', ''),
                        'Relation'  : r.get('relation', ''),
                        'Object'    : r.get('object', ''),
                        'Confidence': round(r.get('confidence', 0), 3) if isinstance(r.get('confidence'), float) else r.get('confidence', ''),
                        'Inferred'  : r.get('inferred', False),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                st.code(formatted, language='')
            else:
                st.info('No matching triples found. Try running the pipeline on more text first.')

    # ── Structured query helpers ──────────────────────────────────────────────
    st.divider()
    st.subheader('Structured Query Builder')

    sq_col1, sq_col2, sq_col3 = st.columns(3)

    with sq_col1:
        st.markdown('**By Subject**')
        subj_input = st.text_input('Entity name', key='sq_subj',
                                    placeholder='e.g. Albert Einstein')
        if st.button('Search by subject', key='btn_subj'):
            if hasattr(backend, '_kg_builder') and hasattr(backend, 'QueryEngine'):
                eng = backend.QueryEngine(backend._kg_builder)
                res = eng.by_subject(subj_input)
                st.write(eng.format_results(res) or 'No results.')

    with sq_col2:
        st.markdown('**By Object**')
        obj_input = st.text_input('Object entity', key='sq_obj',
                                   placeholder='e.g. Nobel Prize')
        if st.button('Search by object', key='btn_obj'):
            if hasattr(backend, '_kg_builder') and hasattr(backend, 'QueryEngine'):
                eng = backend.QueryEngine(backend._kg_builder)
                res = eng.by_object(obj_input)
                st.write(eng.format_results(res) or 'No results.')

    with sq_col3:
        st.markdown('**By Relation**')
        rel_input = st.text_input('Relation label', key='sq_rel',
                                   placeholder='e.g. born_in')
        if st.button('Search by relation', key='btn_rel'):
            if hasattr(backend, '_kg_builder') and hasattr(backend, 'QueryEngine'):
                eng = backend.QueryEngine(backend._kg_builder)
                res = eng.by_relation(rel_input)
                st.write(eng.format_results(res) or 'No results.')

# ═══════════════════════════════════════════════════════════════════
# TAB 4 — Evaluation
# ═══════════════════════════════════════════════════════════════════

with tab_eval:
    st.header('📊 Pipeline Evaluation')
    st.caption(
        'Quality metrics computed from the last pipeline run. '
        'No ground-truth labels are required.'
    )

    if 'last_results' not in st.session_state:
        st.info('Run the pipeline on some text first (Extract Knowledge tab).')
    else:
        ev = st.session_state['last_results'].get('evaluation', {})

        # ── Metric cards ──────────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        c1.metric('Acceptance rate',
                   f"{ev.get('acceptance_rate_pct', 0):.1f}%",
                   help='Fraction of raw triples passing confidence threshold')
        c2.metric('Refinement rate',
                   f"{ev.get('refinement_rate_pct', 0):.1f}%",
                   help='Fraction of accepted triples surviving symbolic reasoning')
        c3.metric('Knowledge expansion',
                   f"{ev.get('expansion_rate_pct', 0):.1f}%",
                   help='Inferred triples as % of refined triples')

        c4, c5, c6 = st.columns(3)
        c4.metric('Avg confidence', f"{ev.get('avg_confidence', 0):.3f}")
        c5.metric('Min confidence', f"{ev.get('min_confidence', 0):.3f}")
        c6.metric('Max confidence', f"{ev.get('max_confidence', 0):.3f}")

        st.divider()

        # ── Bar chart: triple counts ───────────────────────────────────────────
        try:
            import plotly.graph_objects as go

            categories = ['Raw', 'Accepted', 'Rejected', 'Refined', 'Inferred']
            values     = [
                ev.get('raw_triples', 0),
                ev.get('accepted_triples', 0),
                ev.get('rejected_triples', 0),
                ev.get('refined_triples', 0),
                ev.get('inferred_triples', 0),
            ]
            colors = ['#89b4fa', '#a6e3a1', '#f38ba8', '#cba6f7', '#89dceb']

            fig = go.Figure(go.Bar(
                x=categories,
                y=values,
                marker_color=colors,
                text=values,
                textposition='outside',
            ))
            fig.update_layout(
                title='Triple Counts by Pipeline Stage',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(showgrid=True, gridcolor='#313244'),
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.json(ev)

        st.divider()

        # ── Raw evaluation JSON ───────────────────────────────────────────────
        with st.expander('📄 Full evaluation report (JSON)'):
            st.json(ev)

        # ── Feedback ─────────────────────────────────────────────────────────
        feedback = st.session_state['last_results'].get('feedback', {})
        st.subheader('💡 Feedback & Improvement Suggestions')
        for s in feedback.get('suggestions', ['No issues detected.']):
            st.info(s)

        top_rej = feedback.get('top_rejected_relations', [])
        if top_rej:
            st.subheader('Top Rejected Relations')
            import pandas as pd
            st.dataframe(
                pd.DataFrame(top_rej, columns=['Relation', 'Count']),
                use_container_width=True,
                hide_index=True,
            )
