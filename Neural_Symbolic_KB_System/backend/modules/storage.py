import json
from pathlib import Path
from typing import Any, Dict

import networkx as nx


class StorageManager:
    """Saves and loads triples and graph JSON files."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_path = self.data_dir / "raw_outputs.json"
        self.filtered_path = self.data_dir / "filtered_outputs.json"
        self.graph_path = self.data_dir / "knowledge_graph.json"

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _read_json(path: Path, fallback: Any) -> Any:
        if not path.exists():
            return fallback
        return json.loads(path.read_text(encoding="utf-8"))

    def save_raw_outputs(self, payload: Dict[str, object]) -> None:
        self._write_json(self.raw_path, payload)

    def save_filtered_outputs(self, payload: Dict[str, object]) -> None:
        self._write_json(self.filtered_path, payload)

    def save_graph(self, graph: nx.DiGraph) -> None:
        self._write_json(self.graph_path, nx.node_link_data(graph))

    def load_graph(self) -> nx.DiGraph:
        data = self._read_json(self.graph_path, {"nodes": [], "links": []})
        return nx.node_link_graph(data, directed=True)

    def load_raw_outputs(self) -> Dict[str, object]:
        return self._read_json(self.raw_path, {"raw_triples": []})

    def load_filtered_outputs(self) -> Dict[str, object]:
        return self._read_json(self.filtered_path, {"filtered_triples": [], "rejected_triples": []})

    def initialize_files(self) -> None:
        if not self.raw_path.exists():
            self.save_raw_outputs({"raw_triples": []})
        if not self.filtered_path.exists():
            self.save_filtered_outputs({"filtered_triples": [], "rejected_triples": []})
        if not self.graph_path.exists():
            self._write_json(self.graph_path, {"nodes": [], "links": []})
