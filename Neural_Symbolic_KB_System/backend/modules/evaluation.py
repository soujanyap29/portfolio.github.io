from typing import Dict, List


class EvaluationEngine:
    """Computes quality and stability metrics for extracted knowledge."""

    @staticmethod
    def evaluate(
        raw_triples: List[Dict[str, object]],
        accepted_triples: List[Dict[str, object]],
        rejected_triples: List[Dict[str, object]],
        refined_triples: List[Dict[str, object]],
        inferred_triples: List[Dict[str, object]],
    ) -> Dict[str, object]:
        raw_count = len(raw_triples)
        accepted_count = len(accepted_triples)
        rejected_count = len(rejected_triples)
        refined_count = len(refined_triples)
        inferred_count = len(inferred_triples)

        confidence_values = [float(item.get("confidence", 0.0)) for item in accepted_triples]
        average_confidence = round(sum(confidence_values) / len(confidence_values), 3) if confidence_values else 0.0

        acceptance_ratio = round((accepted_count / raw_count), 3) if raw_count else 0.0
        refinement_ratio = round((refined_count / accepted_count), 3) if accepted_count else 0.0
        expansion_ratio = round((inferred_count / refined_count), 3) if refined_count else 0.0

        return {
            "raw_triples": raw_count,
            "accepted_triples": accepted_count,
            "rejected_triples": rejected_count,
            "refined_triples": refined_count,
            "inferred_triples": inferred_count,
            "acceptance_ratio": acceptance_ratio,
            "refinement_consistency_ratio": refinement_ratio,
            "knowledge_expansion_ratio": expansion_ratio,
            "average_confidence": average_confidence,
            "stability_note": "Run process_text multiple times on same input and compare counts for stability analysis.",
        }
