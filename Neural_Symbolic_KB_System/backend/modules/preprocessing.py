import re
from typing import Dict, List


class TextPreprocessor:
    """Cleans and normalizes raw unstructured text."""

    def __init__(self) -> None:
        self._sentence_split_pattern = re.compile(r"(?<=[.!?])\s+")

    def clean_text(self, text: str) -> str:
        text = text.replace("\n", " ")
        text = re.sub(r"[^\w\s.,;:()\-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def normalize_text(self, text: str) -> str:
        return " ".join(text.lower().split())

    def sentence_tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return [s.strip() for s in self._sentence_split_pattern.split(text) if s.strip()]

    def preprocess(self, text: str) -> Dict[str, object]:
        cleaned = self.clean_text(text)
        normalized = self.normalize_text(cleaned)
        sentences = self.sentence_tokenize(cleaned)
        return {
            "cleaned_text": cleaned,
            "normalized_text": normalized,
            "sentences": sentences,
        }
