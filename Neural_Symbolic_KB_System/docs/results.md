# Results and Observations

## Output Artifacts

- `backend/data/raw_outputs.json`
- `backend/data/filtered_outputs.json`
- `backend/data/knowledge_graph.json`

## Typical Outputs

- Raw extracted triples from LLM.
- Confidence-scored accepted/rejected triples.
- Refined triples after symbolic processing.
- Inferred triples from multi-hop reasoning.
- Directed graph visualization in Streamlit.

## Evaluation Highlights

- **Acceptance Ratio**: accepted vs raw triples.
- **Refinement Consistency Ratio**: refined vs accepted triples.
- **Knowledge Expansion Ratio**: inferred vs refined triples.
- **Average Confidence**: mean confidence over accepted triples.

## Discussion

This architecture balances flexibility from neural extraction with structure from symbolic logic. It supports iterative improvements to prompts and rules without changing the full pipeline design.
