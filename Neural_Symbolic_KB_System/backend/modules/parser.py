import re
from typing import Dict, List, Tuple


class TripleParser:
    """Parses `(subject, relation, object)` triples from text."""

    TRIPLE_REGEX = re.compile(r"\(([^,]+),\s*([^,]+),\s*([^\)]+)\)")

    @classmethod
    def parse_triples(cls, text: str) -> List[Tuple[str, str, str]]:
        triples: List[Tuple[str, str, str]] = []
        for subject, relation, obj in cls.TRIPLE_REGEX.findall(text):
            clean = (
                subject.strip(" \t\n\r\"'"),
                relation.strip(" \t\n\r\"'").lower(),
                obj.strip(" \t\n\r\"'"),
            )
            if cls._is_valid(clean):
                triples.append(clean)
        return triples

    @staticmethod
    def _is_valid(triple: Tuple[str, str, str]) -> bool:
        subject, relation, obj = triple
        if not subject or not relation or not obj:
            return False
        if len(relation) < 2:
            return False
        if subject.lower() == obj.lower():
            return False
        return True

    @classmethod
    def parse_with_errors(cls, text: str) -> Dict[str, object]:
        parsed = cls.parse_triples(text)
        all_matches = cls.TRIPLE_REGEX.findall(text)
        invalid_count = max(0, len(all_matches) - len(parsed))
        return {
            "triples": parsed,
            "invalid_count": invalid_count,
        }
