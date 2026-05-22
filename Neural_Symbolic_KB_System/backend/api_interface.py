from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx

from backend.modules.confidence import ConfidenceEstimator
from backend.modules.evaluation import EvaluationEngine
from backend.modules.extraction import OllamaExtractor
from backend.modules.graph_builder import KnowledgeGraphBuilder
from backend.modules.inference_engine import InferenceEngine
from backend.modules.parser import TripleParser
from backend.modules.preprocessing import TextPreprocessor
from backend.modules.query_engine import QueryEngine
from backend.modules.storage import StorageManager
from backend.modules.symbolic_reasoning import SymbolicReasoner


@dataclass
class TripleRecord:
    subject: str
    relation: str
    object: str
    confidence: float
    inferred: bool = False


class NeuralSymbolicPipeline:
    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parent
        self.storage = StorageManager(project_root / "data")
        self.storage.initialize_files()

        self.preprocessor = TextPreprocessor()
        self.extractor = OllamaExtractor(model_name="llama3")
        self.parser = TripleParser()
        self.confidence = ConfidenceEstimator(threshold=0.55)
        self.reasoner = SymbolicReasoner()
        self.graph_builder = KnowledgeGraphBuilder()
        self.inference = InferenceEngine()
        self.query_engine = QueryEngine()
        self.evaluation = EvaluationEngine()

        self.last_outputs: Dict[str, Any] = {
            "preprocessing": {},
            "raw_triples": [],
            "accepted_triples": [],
            "rejected_triples": [],
            "refined_triples": [],
            "inferred_triples": [],
            "evaluation": {},
        }

    def _convert_tuple_triples(self, triples: List[tuple]) -> List[Dict[str, object]]:
        return [
            {
                "subject": item[0],
                "relation": item[1],
                "object": item[2],
            }
            for item in triples
        ]

    def process_text(self, input_text: str) -> Dict[str, Any]:
        preprocessing = self.preprocessor.preprocess(input_text)
        extracted_text = self.extractor.extract(preprocessing["cleaned_text"])

        parse_result = self.parser.parse_with_errors(extracted_text)
        tuple_triples = parse_result["triples"]
        raw_triples = self._convert_tuple_triples(tuple_triples)

        confidence_outputs = self.confidence.filter_triples(tuple_triples)
        accepted = confidence_outputs["accepted"]
        rejected = confidence_outputs["rejected"]

        refined = self.reasoner.refine(accepted)

        self.graph_builder.reset()
        graph = self.graph_builder.add_triples(refined)

        inferred = self.inference.infer_multi_hop(graph, max_hops=2)
        graph = self.graph_builder.add_triples(inferred)

        evaluation = self.evaluation.evaluate(raw_triples, accepted, rejected, refined, inferred)

        self.storage.save_raw_outputs(
            {
                "preprocessing": preprocessing,
                "raw_output_text": extracted_text,
                "raw_triples": raw_triples,
                "invalid_triple_count": parse_result["invalid_count"],
            }
        )
        self.storage.save_filtered_outputs(
            {
                "filtered_triples": accepted,
                "rejected_triples": rejected,
                "refined_triples": refined,
                "inferred_triples": inferred,
                "evaluation": evaluation,
            }
        )
        self.storage.save_graph(graph)

        self.last_outputs = {
            "preprocessing": preprocessing,
            "raw_triples": raw_triples,
            "accepted_triples": accepted,
            "rejected_triples": rejected,
            "refined_triples": refined,
            "inferred_triples": inferred,
            "evaluation": evaluation,
            "invalid_triple_count": parse_result["invalid_count"],
        }

        return self.last_outputs

    def get_triples(self) -> Dict[str, Any]:
        return {
            "raw_triples": self.last_outputs.get("raw_triples", []),
            "filtered_triples": self.last_outputs.get("accepted_triples", []),
            "rejected_triples": self.last_outputs.get("rejected_triples", []),
            "refined_triples": self.last_outputs.get("refined_triples", []),
            "inferred_triples": self.last_outputs.get("inferred_triples", []),
        }

    def get_graph(self) -> Dict[str, Any]:
        graph = self.storage.load_graph()
        return nx.node_link_data(graph)

    def query_graph(self, query: str) -> Dict[str, Any]:
        graph = self.storage.load_graph()
        return self.query_engine.query(graph, query)


_PIPELINE = NeuralSymbolicPipeline()


def process_text(input_text: str) -> Dict[str, Any]:
    return _PIPELINE.process_text(input_text)


def get_triples() -> Dict[str, Any]:
    return _PIPELINE.get_triples()


def get_graph() -> Dict[str, Any]:
    return _PIPELINE.get_graph()


def query_graph(query: str) -> Dict[str, Any]:
    return _PIPELINE.query_graph(query)
