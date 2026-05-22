from typing import Dict, List

import networkx as nx


class QueryEngine:
    """Runs subject/object/relation queries on the knowledge graph."""

    @staticmethod
    def query(graph: nx.DiGraph, query: str) -> Dict[str, object]:
        query = query.strip()
        if not query:
            return {"query": query, "results": [], "message": "Empty query provided."}

        mode = "contains"
        term = query
        if ":" in query:
            prefix, value = query.split(":", 1)
            prefix = prefix.strip().lower()
            value = value.strip()
            if prefix in {"subject", "object", "relation"}:
                mode = prefix
                term = value

        term_lower = term.lower()
        results: List[Dict[str, object]] = []

        for source, target, data in graph.edges(data=True):
            row = {
                "subject": source,
                "relation": data.get("relation", ""),
                "object": target,
                "confidence": data.get("confidence", 0.0),
                "inferred": data.get("inferred", False),
            }

            if mode == "subject" and term_lower in source.lower():
                results.append(row)
            elif mode == "object" and term_lower in target.lower():
                results.append(row)
            elif mode == "relation" and term_lower in str(data.get("relation", "")).lower():
                results.append(row)
            elif mode == "contains" and (
                term_lower in source.lower()
                or term_lower in target.lower()
                or term_lower in str(data.get("relation", "")).lower()
            ):
                results.append(row)

        return {
            "query": query,
            "mode": mode,
            "result_count": len(results),
            "results": results,
            "message": f"Found {len(results)} matching triple(s).",
        }
