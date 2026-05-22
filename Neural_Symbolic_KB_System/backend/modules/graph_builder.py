from typing import Dict, List

import networkx as nx


class KnowledgeGraphBuilder:
    """Builds a directed knowledge graph from triples."""

    def __init__(self) -> None:
        self.graph = nx.DiGraph()

    def reset(self) -> None:
        self.graph = nx.DiGraph()

    def add_triples(self, triples: List[Dict[str, object]]) -> nx.DiGraph:
        for triple in triples:
            subject = str(triple["subject"])
            relation = str(triple["relation"])
            obj = str(triple["object"])
            confidence = float(triple.get("confidence", 0.0))
            inferred = bool(triple.get("inferred", False))

            self.graph.add_node(subject)
            self.graph.add_node(obj)
            self.graph.add_edge(
                subject,
                obj,
                relation=relation,
                confidence=confidence,
                inferred=inferred,
            )
        return self.graph

    def graph_to_records(self) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        for u, v, data in self.graph.edges(data=True):
            records.append(
                {
                    "subject": u,
                    "relation": data.get("relation", ""),
                    "object": v,
                    "confidence": data.get("confidence", 0.0),
                    "inferred": data.get("inferred", False),
                }
            )
        return records
