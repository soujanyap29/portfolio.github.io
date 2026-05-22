# Methodology

## NLP Preprocessing

- Normalize whitespace and symbols.
- Produce cleaned and lowercased variants.
- Split text into sentence-level units.

## LLM-Based Extraction

- Use strict prompt constraints forcing `(subject, relation, object)` format.
- Call local Ollama endpoint with retries.
- Stabilize noisy model output to valid tuple lines.

## Symbolic Reasoning

- Remove invalid/vague relations.
- Normalize entities and relations.
- Enforce lightweight type constraints.
- Deduplicate triples by `(subject, relation, object)`.

## Graph Inference

- Build directed graph using refined triples.
- Discover multi-hop paths up to 2 hops.
- Add inferred triples if direct edge is absent.

## Query & Evaluation

- Query by subject, object, relation, or free text.
- Evaluate acceptance, refinement consistency, and expansion ratios.
