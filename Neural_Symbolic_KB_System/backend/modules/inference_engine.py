from typing import Dict, List, Set, Tuple

import networkx as nx


class InferenceEngine:
    """Performs rule-based multi-hop inference on the graph."""

    def infer_multi_hop(self, graph: nx.DiGraph, max_hops: int = 2) -> List[Dict[str, object]]:
        inferred: List[Dict[str, object]] = []
        seen: Set[Tuple[str, str, str]] = set()

        for source in graph.nodes:
            for target in graph.nodes:
                if source == target:
                    continue
                for path in nx.all_simple_paths(graph, source=source, target=target, cutoff=max_hops):
                    if len(path) < 3:
                        continue

                    edge_data = [graph[path[i]][path[i + 1]] for i in range(len(path) - 1)]
                    relations = [str(item.get("relation", "related_to")) for item in edge_data]
                    confidences = [float(item.get("confidence", 0.5)) for item in edge_data]

                    relation_label = "inferred_via_" + "_".join(rel.replace(" ", "_") for rel in relations)
                    confidence = round(sum(confidences) / len(confidences), 2)
                    triple = (path[0], relation_label, path[-1])

                    if graph.has_edge(path[0], path[-1]):
                        continue
                    if triple in seen:
                        continue

                    seen.add(triple)
                    inferred.append(
                        {
                            "subject": path[0],
                            "relation": relation_label,
                            "object": path[-1],
                            "confidence": confidence,
                            "inferred": True,
                            "path": path,
                        }
                    )

        return inferred
