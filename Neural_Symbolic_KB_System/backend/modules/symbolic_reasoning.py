from typing import Dict, List, Set, Tuple


class SymbolicReasoner:
    """Applies symbolic constraints, normalization, and deduplication."""

    INVALID_RELATIONS: Set[str] = {"it", "this", "that", "thing", "related"}

    @staticmethod
    def normalize_entity(entity: str) -> str:
        return " ".join(token.capitalize() for token in entity.split())

    @staticmethod
    def normalize_relation(relation: str) -> str:
        return " ".join(relation.strip().lower().split())

    @staticmethod
    def infer_entity_type(entity: str) -> str:
        lowered = entity.lower()
        if any(x in lowered for x in ["inc", "corp", "company", "ltd", "university"]):
            return "organization"
        if any(x in lowered for x in ["city", "state", "country", "region"]):
            return "location"
        if entity[:1].isupper():
            if " " in entity:
                return "person"
            return "entity"
        return "unknown"

    def _satisfies_type_constraint(self, subject: str, relation: str, obj: str) -> bool:
        s_type = self.infer_entity_type(subject)
        o_type = self.infer_entity_type(obj)

        if relation in {"works at", "employed by"} and o_type not in {"organization", "entity"}:
            return False
        if relation in {"located in", "is in", "is located in"} and o_type not in {"location", "entity"}:
            return False
        if s_type == "unknown" or o_type == "unknown":
            return True
        return True

    def refine(self, accepted_triples: List[Dict[str, object]]) -> List[Dict[str, object]]:
        dedup: Dict[Tuple[str, str, str], Dict[str, object]] = {}

        for item in accepted_triples:
            subject = self.normalize_entity(str(item["subject"]))
            relation = self.normalize_relation(str(item["relation"]))
            obj = self.normalize_entity(str(item["object"]))
            confidence = float(item["confidence"])

            if relation in self.INVALID_RELATIONS:
                continue
            if subject == obj:
                continue
            if not self._satisfies_type_constraint(subject, relation, obj):
                continue

            key = (subject, relation, obj)
            payload = {
                "subject": subject,
                "relation": relation,
                "object": obj,
                "confidence": confidence,
                "inferred": False,
            }
            if key not in dedup or confidence > float(dedup[key]["confidence"]):
                dedup[key] = payload

        return list(dedup.values())
