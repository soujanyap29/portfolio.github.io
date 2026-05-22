from typing import Dict, List, Tuple


class ConfidenceEstimator:
    """Assigns confidence scores and filters triples."""

    def __init__(self, threshold: float = 0.55) -> None:
        self.threshold = threshold

    @staticmethod
    def score(triple: Tuple[str, str, str]) -> float:
        subject, relation, obj = triple
        score = 0.35

        if 1 <= len(subject.split()) <= 6:
            score += 0.2
        if 1 <= len(obj.split()) <= 6:
            score += 0.2
        if 1 <= len(relation.split()) <= 4:
            score += 0.15
        if relation.replace("_", " ").replace("-", " ").isprintable():
            score += 0.1

        return round(min(score, 0.99), 2)

    def filter_triples(self, triples: List[Tuple[str, str, str]]) -> Dict[str, List[Dict[str, object]]]:
        accepted: List[Dict[str, object]] = []
        rejected: List[Dict[str, object]] = []

        for triple in triples:
            confidence = self.score(triple)
            record = {
                "subject": triple[0],
                "relation": triple[1],
                "object": triple[2],
                "confidence": confidence,
            }
            if confidence >= self.threshold:
                accepted.append(record)
            else:
                rejected.append(record)

        return {"accepted": accepted, "rejected": rejected}
