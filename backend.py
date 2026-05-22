import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import requests


@dataclass
class Triple:
    subject: str
    relation: str
    object: str
    confidence: float = 0.0
    inferred: bool = False


class PreprocessingModule:
    @staticmethod
    def clean_text(text: str) -> List[str]:
        text = re.sub(r"[^\w\s.,;:()\-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]


class NeuralExtractionModule:
    def __init__(self, model: str = "llama3:latest", ollama_url: str = "http://localhost:11434/api/generate"):
        self.model = model
        self.ollama_url = ollama_url

    @staticmethod
    def _prompt(text: str) -> str:
        return (
            "Extract factual knowledge triples from the text.\\n"
            "Return only triples in this exact format, one per line: (subject, relation, object)\\n"
            "No explanations, no bullets, no numbering.\\n"
            f"Text: {text}"
        )

    def extract(self, text: str, retries: int = 2) -> str:
        prompt = self._prompt(text)
        for _ in range(retries + 1):
            try:
                response = requests.post(
                    self.ollama_url,
                    json={"model": self.model, "prompt": prompt, "stream": False},
                    timeout=60,
                )
                response.raise_for_status()
                payload = response.json()
                output = payload.get("response", "").strip()
                if TripletParser.has_valid_pattern(output):
                    return output
            except requests.RequestException:
                continue
        return self._fallback_extract(text)

    @staticmethod
    def _fallback_extract(text: str) -> str:
        triples = []
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            match = re.match(r"\s*([A-Z][\w\s-]+?)\s+(is|are|was|were|works at|located in|part of)\s+([A-Z][\w\s-]+)", sentence)
            if match:
                triples.append(f"({match.group(1).strip()}, {match.group(2).strip()}, {match.group(3).strip()})")
        return "\n".join(triples)


class TripletParser:
    TRIPLE_PATTERN = re.compile(r"\(([^,]+),\s*([^,]+),\s*([^\)]+)\)")

    @classmethod
    def has_valid_pattern(cls, text: str) -> bool:
        return bool(cls.TRIPLE_PATTERN.search(text))

    @classmethod
    def parse(cls, text: str) -> List[Triple]:
        triples: List[Triple] = []
        for match in cls.TRIPLE_PATTERN.findall(text):
            s, r, o = [x.strip(" \t\n\r\"'") for x in match]
            if s and r and o:
                triples.append(Triple(subject=s, relation=r.lower(), object=o))
        return triples


class ConfidenceEstimator:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    @staticmethod
    def score(triple: Triple) -> float:
        score = 0.4
        if len(triple.subject.split()) <= 6:
            score += 0.15
        if len(triple.object.split()) <= 6:
            score += 0.15
        if re.match(r"^[a-z_\-\s]+$", triple.relation):
            score += 0.2
        if len(triple.relation.split()) <= 4:
            score += 0.1
        return round(min(score, 0.99), 2)

    def filter(self, triples: List[Triple]) -> Tuple[List[Triple], List[Triple]]:
        accepted: List[Triple] = []
        rejected: List[Triple] = []
        for triple in triples:
            triple.confidence = self.score(triple)
            (accepted if triple.confidence >= self.threshold else rejected).append(triple)
        return accepted, rejected


class SymbolicReasoningEngine:
    INVALID_RELATIONS = {"it", "this", "that", "related", "thing"}

    @staticmethod
    def normalize_entity(entity: str) -> str:
        return " ".join(word.capitalize() for word in entity.split())

    def refine(self, triples: List[Triple]) -> List[Triple]:
        dedup: Dict[Tuple[str, str, str], Triple] = {}
        for t in triples:
            t.subject = self.normalize_entity(t.subject)
            t.object = self.normalize_entity(t.object)
            t.relation = t.relation.strip().lower()
            if t.relation in self.INVALID_RELATIONS:
                continue
            if len(t.relation) < 2 or t.subject == t.object:
                continue
            key = (t.subject, t.relation, t.object)
            if key not in dedup or dedup[key].confidence < t.confidence:
                dedup[key] = t
        return list(dedup.values())


class KnowledgeGraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()

    def build(self, triples: List[Triple]) -> nx.DiGraph:
        for triple in triples:
            self.graph.add_node(triple.subject)
            self.graph.add_node(triple.object)
            self.graph.add_edge(
                triple.subject,
                triple.object,
                relation=triple.relation,
                confidence=triple.confidence,
                inferred=triple.inferred,
            )
        return self.graph


class StorageManager:
    def __init__(self, storage_path: str = "knowledge_graph.json"):
        self.storage_path = Path(storage_path)

    def save(self, graph: nx.DiGraph) -> None:
        data = nx.node_link_data(graph)
        self.storage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self) -> nx.DiGraph:
        if not self.storage_path.exists():
            return nx.DiGraph()
        data = json.loads(self.storage_path.read_text(encoding="utf-8"))
        return nx.node_link_graph(data, directed=True)


class InferenceEngine:
    @staticmethod
    def infer(graph: nx.DiGraph) -> List[Triple]:
        inferred: List[Triple] = []
        for a in graph.nodes:
            for b in graph.successors(a):
                for c in graph.successors(b):
                    if a == c:
                        continue
                    if graph.has_edge(a, c):
                        continue
                    rel_ab = graph[a][b].get("relation", "related_to")
                    rel_bc = graph[b][c].get("relation", "related_to")
                    inferred.append(
                        Triple(
                            subject=a,
                            relation=f"inferred_{rel_ab}_{rel_bc}"[:60],
                            object=c,
                            confidence=round((graph[a][b].get("confidence", 0.5) + graph[b][c].get("confidence", 0.5)) / 2, 2),
                            inferred=True,
                        )
                    )
        unique = {(t.subject, t.relation, t.object): t for t in inferred}
        return list(unique.values())


class QueryEngine:
    @staticmethod
    def query(graph: nx.DiGraph, query: str) -> Dict[str, List[Dict[str, object]]]:
        query = query.strip()
        if not query:
            return {"results": []}

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
        results = []
        for u, v, d in graph.edges(data=True):
            row = {
                "subject": u,
                "relation": d.get("relation", ""),
                "object": v,
                "confidence": d.get("confidence", 0.0),
                "inferred": d.get("inferred", False),
            }
            if mode == "subject" and term_lower in u.lower():
                results.append(row)
            elif mode == "object" and term_lower in v.lower():
                results.append(row)
            elif mode == "relation" and term_lower in str(d.get("relation", "")).lower():
                results.append(row)
            elif mode == "contains" and term_lower in json.dumps(row).lower():
                results.append(row)
        return {"results": results}


class EvaluationModule:
    @staticmethod
    def evaluate(raw: List[Triple], filtered: List[Triple], rejected: List[Triple], refined: List[Triple]) -> Dict[str, object]:
        validity_ratio = round((len(filtered) / len(raw)) if raw else 0.0, 2)
        consistency_score = round((len(refined) / len(filtered)) if filtered else 0.0, 2)
        return {
            "raw_triples": len(raw),
            "accepted_triples": len(filtered),
            "rejected_triples": len(rejected),
            "refined_triples": len(refined),
            "validity_ratio": validity_ratio,
            "consistency_score": consistency_score,
        }


class NeuralSymbolicKBSystem:
    def __init__(self):
        self.preprocessor = PreprocessingModule()
        self.extractor = NeuralExtractionModule()
        self.parser = TripletParser()
        self.confidence = ConfidenceEstimator()
        self.reasoner = SymbolicReasoningEngine()
        self.builder = KnowledgeGraphBuilder()
        self.storage = StorageManager()
        self.inference = InferenceEngine()
        self.query_engine = QueryEngine()
        self.evaluator = EvaluationModule()

        self.last_raw_text = ""
        self.last_raw_triples: List[Triple] = []
        self.last_filtered_triples: List[Triple] = []
        self.last_rejected_triples: List[Triple] = []
        self.last_refined_triples: List[Triple] = []
        self.last_inferred_triples: List[Triple] = []
        self.last_evaluation: Dict[str, object] = {}

    def process_text(self, input_text: str) -> Dict[str, object]:
        self.last_raw_text = input_text
        sentences = self.preprocessor.clean_text(input_text)
        llm_output = self.extractor.extract(" ".join(sentences))

        self.last_raw_triples = self.parser.parse(llm_output)
        self.last_filtered_triples, self.last_rejected_triples = self.confidence.filter(self.last_raw_triples)
        self.last_refined_triples = self.reasoner.refine(self.last_filtered_triples)

        graph = self.builder.build(self.last_refined_triples)
        self.last_inferred_triples = self.inference.infer(graph)
        self.builder.build(self.last_inferred_triples)
        self.storage.save(self.builder.graph)

        self.last_evaluation = self.evaluator.evaluate(
            self.last_raw_triples,
            self.last_filtered_triples,
            self.last_rejected_triples,
            self.last_refined_triples,
        )

        return {
            "raw_triples": [asdict(t) for t in self.last_raw_triples],
            "filtered_triples": [asdict(t) for t in self.last_refined_triples],
            "rejected_triples": [asdict(t) for t in self.last_rejected_triples],
            "inferred_triples": [asdict(t) for t in self.last_inferred_triples],
            "evaluation": self.last_evaluation,
        }

    def get_triples(self) -> Dict[str, List[Dict[str, object]]]:
        return {
            "raw": [asdict(t) for t in self.last_raw_triples],
            "filtered": [asdict(t) for t in self.last_refined_triples],
            "rejected": [asdict(t) for t in self.last_rejected_triples],
            "inferred": [asdict(t) for t in self.last_inferred_triples],
        }

    def get_graph(self) -> Dict[str, object]:
        graph = self.storage.load() if self.storage.storage_path.exists() else self.builder.graph
        return nx.node_link_data(graph)

    def query_graph(self, query: str) -> Dict[str, List[Dict[str, object]]]:
        graph = self.storage.load() if self.storage.storage_path.exists() else self.builder.graph
        return self.query_engine.query(graph, query)


_SYSTEM = NeuralSymbolicKBSystem()


def process_text(input_text: str) -> Dict[str, object]:
    return _SYSTEM.process_text(input_text)


def get_triples() -> Dict[str, List[Dict[str, object]]]:
    return _SYSTEM.get_triples()


def get_graph() -> Dict[str, object]:
    return _SYSTEM.get_graph()


def query_graph(query: str) -> Dict[str, List[Dict[str, object]]]:
    return _SYSTEM.query_graph(query)
