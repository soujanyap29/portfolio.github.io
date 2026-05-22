import re
from typing import List

import requests


class OllamaExtractor:
    """Extracts triples using local Ollama LLaMA3 model."""

    def __init__(
        self,
        model_name: str = "llama3",
        base_url: str = "http://localhost:11434/api/generate",
        retries: int = 2,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url
        self.retries = retries

    @staticmethod
    def build_prompt(text: str) -> str:
        return (
            "You are an information extraction engine.\\n"
            "Extract factual triples from the input text and return ONLY in this format:\\n"
            "(subject, relation, object)\\n"
            "Rules:\\n"
            "1) No explanation text.\\n"
            "2) No bullets or numbering.\\n"
            "3) One triple per line.\\n"
            "4) Keep relation concise and meaningful.\\n"
            f"Input: {text}"
        )

    @staticmethod
    def stabilize_output(output: str) -> str:
        lines: List[str] = []
        triple_pattern = re.compile(r"\([^,]+,\s*[^,]+,\s*[^\)]+\)")
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            found = triple_pattern.findall(line)
            if found:
                lines.extend(found)
        return "\n".join(lines)

    def _call_ollama(self, prompt: str) -> str:
        response = requests.post(
            self.base_url,
            json={"model": self.model_name, "prompt": prompt, "stream": False},
            timeout=90,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    def extract(self, text: str) -> str:
        prompt = self.build_prompt(text)
        last_error = ""

        for _ in range(self.retries + 1):
            try:
                raw = self._call_ollama(prompt)
                stabilized = self.stabilize_output(raw)
                if stabilized:
                    return stabilized
            except requests.RequestException as exc:
                last_error = str(exc)

        fallback = self._heuristic_fallback(text)
        if fallback:
            return fallback
        if last_error:
            raise RuntimeError(f"Ollama extraction failed after retries: {last_error}")
        raise RuntimeError("Ollama extraction failed: empty output after retries")

    @staticmethod
    def _heuristic_fallback(text: str) -> str:
        pattern = re.compile(
            r"\b([A-Za-z0-9][\w\s-]*?)\s+(is located in|located in|works at|part of|is in|are|is)\s+([A-Za-z0-9][\w\s-]+)\b",
            re.IGNORECASE,
        )
        triples: List[str] = []
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            match = pattern.search(sentence)
            if match:
                triples.append(f"({match.group(1).strip()}, {match.group(2).strip().lower()}, {match.group(3).strip()})")
        return "\n".join(triples)
