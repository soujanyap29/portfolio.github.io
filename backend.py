"""
backend.py
==========
Standalone Python module containing the full Neural–Symbolic Knowledge Base
Construction pipeline.  This file mirrors the Jupyter Notebook
(neural_symbolic_kb.ipynb) but is importable directly by the Streamlit
frontend without requiring a running Jupyter kernel.

Run the Streamlit UI with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import requests

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Configuration & Ollama check
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL  = 'http://localhost:11434'
OLLAMA_MODEL     = 'llama3'
OLLAMA_TIMEOUT   = 120
CONFIDENCE_THRESHOLD = 0.40


def _check_ollama() -> bool:
    try:
        resp = requests.get(f'{OLLAMA_BASE_URL}/api/tags', timeout=5)
        if resp.status_code != 200:
            return False
        models = [m['name'] for m in resp.json().get('models', [])]
        return any(OLLAMA_MODEL in m for m in models)
    except Exception as exc:
        logger.warning(f'Ollama not reachable: {exc}')
        return False


OLLAMA_AVAILABLE: bool = _check_ollama()

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

class Preprocessor:
    _NOISE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
    _BLANK_RE = re.compile(r'\n{3,}')
    _SENT_RE  = re.compile(r'(?<=[.!?])\s+')

    @classmethod
    def clean(cls, text: str) -> str:
        text = cls._NOISE_RE.sub(' ', text)
        text = text.replace('\t', ' ')
        text = cls._BLANK_RE.sub('\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()

    @classmethod
    def segment(cls, text: str) -> List[str]:
        cleaned = cls.clean(text)
        sentences = cls._SENT_RE.split(cleaned)
        return [s.strip() for s in sentences if len(s.strip()) > 5]

    @classmethod
    def process(cls, text: str) -> Dict[str, Any]:
        original  = text
        cleaned   = cls.clean(text)
        sentences = cls.segment(cleaned)
        return {
            'original'  : original,
            'cleaned'   : cleaned,
            'sentences' : sentences,
            'sent_count': len(sentences),
        }

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Neural Extraction
# ─────────────────────────────────────────────────────────────────────────────

EXTRACTION_PROMPT_TEMPLATE = """\
You are an expert information-extraction engine.
Extract ALL factual knowledge triples from the text below.

STRICT OUTPUT RULES:
- Each triple must be on its own line in EXACTLY this format: (subject, relation, object)
- Use lowercase for relation labels, e.g. 'born_in', 'works_at', 'developed'
- NO explanations, NO numbering, NO markdown, NO extra text — triples only.

TEXT:
{text}

TRIPLES:"""


class NeuralExtractor:
    def __init__(self, model: str = OLLAMA_MODEL,
                 base_url: str = OLLAMA_BASE_URL,
                 max_retries: int = 3):
        self.model       = model
        self.base_url    = base_url
        self.max_retries = max_retries
        self._endpoint   = f'{base_url}/api/generate'

    def _call_llm(self, prompt: str) -> str:
        payload = {
            'model' : self.model,
            'prompt': prompt,
            'stream': False,
            'options': {'temperature': 0.1, 'top_p': 0.9},
        }
        resp = requests.post(self._endpoint, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get('response', '')

    def extract(self, text: str) -> List[str]:
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(text=text)
        for attempt in range(1, self.max_retries + 1):
            try:
                raw    = self._call_llm(prompt)
                lines  = [l.strip() for l in raw.split('\n') if l.strip()]
                triples = [l for l in lines if re.search(r'\(.+,.+,.+\)', l)]
                if triples:
                    return triples
                logger.warning(f'Attempt {attempt}: no valid triples, retrying…')
            except Exception as exc:
                logger.warning(f'Attempt {attempt} failed: {exc}')
            time.sleep(1)
        logger.error('All LLM extraction attempts failed.')
        return []


class RuleBasedExtractor:
    _PATTERNS = [
        (r'([A-Z][\w\s]+?) was born in ([\w\s,]+)',
         lambda m: f'({m.group(1).strip()}, born_in, {m.group(2).strip()})'),
        (r'([A-Z][\w\s]+?) is (?:a |an |the )?([\w\s]+)',
         lambda m: f'({m.group(1).strip()}, is_a, {m.group(2).strip()})'),
        (r'([A-Z][\w\s]+?) was (?:a |an |the )?([\w\s]+?) of ([A-Z][\w\s]+)',
         lambda m: f'({m.group(1).strip()}, {m.group(2).strip().replace(" ","_")}, {m.group(3).strip()})'),
        (r'([A-Z][\w-]{1,30}(?:\s+[\w-]+){0,3}) (founded|created|developed|invented|discovered) ([\w\s]+)',
         lambda m: f'({m.group(1).strip()}, {m.group(2).strip()}, {m.group(3).strip()})'),
        (r'([A-Z][\w\s]+?) (?:works|worked) at ([A-Z][\w\s]+)',
         lambda m: f'({m.group(1).strip()}, works_at, {m.group(2).strip()})'),
        (r'([A-Z][\w\s]+?) won (?:the )?([\w\s]+? (?:Prize|Award|Medal))',
         lambda m: f'({m.group(1).strip()}, won, {m.group(2).strip()})'),
        (r'([A-Z][\w\s]+?) (?:is )?located in ([\w\s,]+)',
         lambda m: f'({m.group(1).strip()}, located_in, {m.group(2).strip()})'),
        (r'([A-Z][\w\s]+?) studied at ([A-Z][\w\s]+)',
         lambda m: f'({m.group(1).strip()}, studied_at, {m.group(2).strip()})'),
    ]

    def extract(self, text: str) -> List[str]:
        triples: List[str] = []
        for pattern, formatter in self._PATTERNS:
            for match in re.finditer(pattern, text):
                triples.append(formatter(match))
        return list(dict.fromkeys(triples))

# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Triplet Parser
# ─────────────────────────────────────────────────────────────────────────────

class TripletParser:
    # Match the content between the outermost parentheses (or the whole line)
    _OUTER_RE = re.compile(r'\(([^()]+)\)')

    @classmethod
    def _normalise(cls, token: str) -> str:
        token = re.sub(r'["\'\'\"[\]`]', '', token)
        token = re.sub(r'\s+', ' ', token)
        return token.strip()

    @classmethod
    def _normalise_relation(cls, relation: str) -> str:
        rel = cls._normalise(relation).lower()
        rel = re.sub(r'[^a-z0-9]+', '_', rel).strip('_')
        return rel

    @classmethod
    def parse_line(cls, line: str) -> Optional[Dict[str, str]]:
        # Try to extract content between parentheses first; fall back to raw line
        m = cls._OUTER_RE.search(line)
        content = m.group(1) if m else line
        # Split into exactly 3 parts on the first two commas
        parts = content.split(',', 2)
        if len(parts) != 3:
            return None
        subj = cls._normalise(parts[0])
        rel  = cls._normalise_relation(parts[1])
        obj  = cls._normalise(parts[2])
        if not (subj and rel and obj):
            return None
        return {'subject': subj, 'relation': rel, 'object': obj}

    @classmethod
    def parse_all(cls, raw_lines: List[str]) -> List[Dict[str, str]]:
        parsed, seen = [], set()
        for line in raw_lines:
            triple = cls.parse_line(line)
            if triple:
                key = (triple['subject'], triple['relation'], triple['object'])
                if key not in seen:
                    seen.add(key)
                    parsed.append(triple)
        return parsed

# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — Confidence Estimation
# ─────────────────────────────────────────────────────────────────────────────

_HIGH_VALUE_RELATIONS = {
    'born_in', 'founded', 'developed', 'invented', 'discovered',
    'won', 'authored', 'works_at', 'studied_at', 'located_in',
    'part_of', 'ceo_of', 'president_of', 'member_of',
}

_LOW_VALUE_RELATIONS = {
    'is_a', 'has', 'have', 'had', 'be', 'are', 'was', 'were',
    'is', 'said', 'says', 'told',
}


class ConfidenceEstimator:
    def __init__(self, threshold: float = CONFIDENCE_THRESHOLD):
        self.threshold = threshold

    def score(self, triple: Dict[str, str]) -> float:
        subj = triple['subject']
        rel  = triple['relation']
        obj  = triple['object']

        score = 0.5
        if subj and subj[0].isupper():
            score += 0.10
        if obj and obj[0].isupper():
            score += 0.05
        if len(subj.split()) > 6:
            score -= 0.10
        if len(obj.split()) > 6:
            score -= 0.10
        if rel in _HIGH_VALUE_RELATIONS:
            score += 0.20
        elif rel in _LOW_VALUE_RELATIONS:
            score -= 0.15
        if len(rel) < 3:
            score -= 0.15
        if len(subj) > 1 and len(obj) > 1:
            score += 0.05

        return round(max(0.0, min(1.0, score)), 3)

    def filter_triples(
        self,
        triples: List[Dict[str, str]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        accepted, rejected = [], []
        for triple in triples:
            s = self.score(triple)
            record = {**triple, 'confidence': s}
            (accepted if s >= self.threshold else rejected).append(record)
        return {'accepted': accepted, 'rejected': rejected}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 6 — Symbolic Reasoning
# ─────────────────────────────────────────────────────────────────────────────

_BANNED_RELATIONS = {
    'is', 'are', 'was', 'were', 'be', 'have', 'has', 'had',
    'said', 'says', 'told', 'think', 'knows',
}

_ENTITY_ALIASES: Dict[str, str] = {
    'einstein'    : 'Albert Einstein',
    'curie'       : 'Marie Curie',
    'marie curie' : 'Marie Curie',
    'usa'         : 'United States',
    'u.s.'        : 'United States',
    'us'          : 'United States',
    'uk'          : 'United Kingdom',
    'u.k.'        : 'United Kingdom',
}


class SymbolicReasoner:
    def __init__(self,
                 banned_relations: Optional[set] = None,
                 aliases: Optional[Dict[str, str]] = None):
        self.banned_relations = banned_relations or set(_BANNED_RELATIONS)
        self.aliases          = aliases or dict(_ENTITY_ALIASES)

    def _canonicalise(self, entity: str) -> str:
        return self.aliases.get(entity.lower(), entity)

    def _is_valid(self, triple: Dict[str, Any]) -> bool:
        rel  = triple['relation']
        subj = triple['subject']
        obj  = triple['object']
        if rel in self.banned_relations:
            return False
        if subj.lower() == obj.lower():
            return False
        if len(subj) < 2 or len(obj) < 2:
            return False
        if subj.isdigit() or obj.isdigit():
            return False
        # Reject subjects that are clearly not named entities (too many words)
        if len(subj.split()) > 5:
            return False
        # Reject pronoun subjects
        if subj.lower() in {'he', 'she', 'it', 'they', 'we', 'i', 'you',
                             'him', 'her', 'them', 'his', 'its'}:
            return False
        return True

    def apply(self, triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        refined, seen = [], set()
        for t in triples:
            t = dict(t)
            t['subject'] = self._canonicalise(t['subject'])
            t['object']  = self._canonicalise(t['object'])
            if not self._is_valid(t):
                continue
            key = (t['subject'], t['relation'], t['object'])
            if key in seen:
                continue
            seen.add(key)
            refined.append(t)
        return refined

# ─────────────────────────────────────────────────────────────────────────────
# Phase 7 — Knowledge Graph Builder
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeGraphBuilder:
    def __init__(self):
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._triple_log: List[Dict[str, Any]] = []

    def add_triples(self, triples: List[Dict[str, Any]]) -> None:
        for t in triples:
            subj = t['subject']
            rel  = t['relation']
            obj  = t['object']
            conf = t.get('confidence', 1.0)
            self.graph.add_node(subj)
            self.graph.add_node(obj)
            self.graph.add_edge(
                subj, obj,
                relation=rel,
                confidence=conf,
                inferred=t.get('inferred', False),
                timestamp=datetime.utcnow().isoformat(),
            )
            self._triple_log.append(t)

    def clear(self) -> None:
        self.graph.clear()
        self._triple_log.clear()

    def get_stats(self) -> Dict[str, int]:
        return {
            'nodes'         : self.graph.number_of_nodes(),
            'edges'         : self.graph.number_of_edges(),
            'triples_logged': len(self._triple_log),
        }

    def to_serialisable(self) -> Dict[str, Any]:
        nodes = list(self.graph.nodes())
        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append({'from': u, 'to': v, **data})
        return {'nodes': nodes, 'edges': edges}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 8 — Storage Manager
# ─────────────────────────────────────────────────────────────────────────────

STORAGE_DIR  = Path('kg_storage')
STORAGE_DIR.mkdir(exist_ok=True)
GRAPH_FILE   = STORAGE_DIR / 'knowledge_graph.json'


class StorageManager:
    @staticmethod
    def save_graph(builder: KnowledgeGraphBuilder) -> Path:
        data = builder.to_serialisable()
        data['saved_at'] = datetime.utcnow().isoformat()
        with open(GRAPH_FILE, 'w') as fh:
            json.dump(data, fh, indent=2)
        return GRAPH_FILE

    @staticmethod
    def load_graph(builder: KnowledgeGraphBuilder) -> bool:
        if not GRAPH_FILE.exists():
            return False
        with open(GRAPH_FILE) as fh:
            data = json.load(fh)
        for edge in data.get('edges', []):
            t = {
                'subject'   : edge['from'],
                'relation'  : edge['relation'],
                'object'    : edge['to'],
                'confidence': edge.get('confidence', 1.0),
            }
            builder.add_triples([t])
        return True

    @staticmethod
    def save_triples(triples: List[Dict[str, Any]], tag: str = 'run') -> Path:
        fname = STORAGE_DIR / f'triples_{tag}_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.json'
        with open(fname, 'w') as fh:
            json.dump(triples, fh, indent=2)
        return fname

# ─────────────────────────────────────────────────────────────────────────────
# Phase 9 — Inference Engine
# ─────────────────────────────────────────────────────────────────────────────

_INFERENCE_RULES: List[Tuple[str, str, str]] = [
    ('born_in',    'located_in',  'born_in_country'),
    ('works_at',   'part_of',     'employed_by'),
    ('studied_at', 'located_in',  'studied_in'),
    ('won',        'given_by',    'recognised_by'),
    ('located_in', 'located_in',  'located_in'),
    ('part_of',    'part_of',     'part_of'),
]


class InferenceEngine:
    def __init__(self, max_hops: int = 3):
        self.max_hops = max_hops
        self.inferred: List[Dict[str, Any]] = []

    def run(self, builder: KnowledgeGraphBuilder) -> List[Dict[str, Any]]:
        graph      = builder.graph
        new_triples: List[Dict[str, Any]] = []
        seen_keys  = set()

        edge_map: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for u, v, data in graph.edges(data=True):
            edge_map[(u, v)].append(data.get('relation', ''))

        for (rel_a, rel_b, inferred_rel) in _INFERENCE_RULES:
            for (a, b), rels_ab in edge_map.items():
                if rel_a not in rels_ab:
                    continue
                for (b2, c), rels_bc in edge_map.items():
                    if b2 != b:
                        continue
                    if rel_b not in rels_bc:
                        continue
                    if a == c:
                        continue
                    key = (a, inferred_rel, c)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    triple = {
                        'subject'   : a,
                        'relation'  : inferred_rel,
                        'object'    : c,
                        'confidence': 0.70,
                        'inferred'  : True,
                        'via'       : f'{a}→[{rel_a}]→{b}→[{rel_b}]→{c}',
                    }
                    new_triples.append(triple)

        if new_triples:
            builder.add_triples(new_triples)
        self.inferred = new_triples
        return new_triples

# ─────────────────────────────────────────────────────────────────────────────
# Phase 10 — Query Engine
# ─────────────────────────────────────────────────────────────────────────────

class QueryEngine:
    def __init__(self, builder: KnowledgeGraphBuilder):
        self.builder = builder

    def _graph(self) -> nx.MultiDiGraph:
        return self.builder.graph

    def by_subject(self, subject: str) -> List[Dict[str, Any]]:
        results = []
        g = self._graph()
        for node in g.nodes():
            if node.lower() == subject.lower():
                for _, obj, data in g.edges(node, data=True):
                    results.append({'subject': node, **data, 'object': obj})
        return results

    def by_object(self, obj: str) -> List[Dict[str, Any]]:
        results = []
        g = self._graph()
        for node in g.nodes():
            if node.lower() == obj.lower():
                for subj, _, data in g.in_edges(node, data=True):
                    results.append({'subject': subj, **data, 'object': node})
        return results

    def by_relation(self, relation: str) -> List[Dict[str, Any]]:
        rel_norm = relation.lower().replace(' ', '_')
        results  = []
        for u, v, data in self._graph().edges(data=True):
            if data.get('relation', '').lower() == rel_norm:
                results.append({'subject': u, **data, 'object': v})
        return results

    def free_text(self, query: str) -> Dict[str, Any]:
        q       = query.strip()
        q_lower = q.lower()

        m = re.search(r'(?:about|tell me about|what is|who is)\s+(.+)', q_lower)
        if m:
            entity  = m.group(1).strip().title()
            results = self.by_subject(entity) or self.by_object(entity)
            return {'intent': 'entity_lookup', 'entity': entity, 'results': results}

        m = re.search(r'(?:who|what)\s+(\w+)\s+(\w+)', q_lower)
        if m:
            rel     = m.group(2)
            results = self.by_relation(rel)
            return {'intent': 'relation_query', 'relation': rel, 'results': results}

        hits = []
        for node in self._graph().nodes():
            if q_lower in node.lower():
                hits.extend(self.by_subject(node))
        return {'intent': 'substring_match', 'query': q, 'results': hits}

    def format_results(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return 'No results found.'
        lines = []
        for r in results:
            subj     = r.get('subject', '?')
            rel      = r.get('relation', '?')
            obj      = r.get('object', '?')
            conf     = r.get('confidence', '–')
            conf_str = f'{conf:.2f}' if isinstance(conf, float) else str(conf)
            lines.append(f'  {subj}  →[{rel}]→  {obj}  (conf: {conf_str})')
        return '\n'.join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# Phase 11 — Evaluation
# ─────────────────────────────────────────────────────────────────────────────

class EvaluationModule:
    @staticmethod
    def compute(run_results: Dict[str, Any]) -> Dict[str, Any]:
        raw      = run_results.get('raw_triples', [])
        accepted = run_results.get('accepted_triples', [])
        rejected = run_results.get('rejected_triples', [])
        refined  = run_results.get('refined_triples', [])
        inferred = run_results.get('inferred_triples', [])
        graph    = run_results.get('graph_stats', {})

        n_raw      = len(raw)
        n_accepted = len(accepted)
        n_rejected = len(rejected)
        n_refined  = len(refined)
        n_inferred = len(inferred)

        acceptance_rate = (n_accepted / n_raw      * 100) if n_raw      else 0
        refinement_rate = (n_refined  / n_accepted * 100) if n_accepted else 0
        expansion_rate  = (n_inferred / n_refined  * 100) if n_refined  else 0

        confidences = [t.get('confidence', 0) for t in accepted]
        avg_conf = float(np.mean(confidences))  if confidences else 0.0
        min_conf = float(np.min(confidences))   if confidences else 0.0
        max_conf = float(np.max(confidences))   if confidences else 0.0

        return {
            'raw_triples'          : n_raw,
            'accepted_triples'     : n_accepted,
            'rejected_triples'     : n_rejected,
            'refined_triples'      : n_refined,
            'inferred_triples'     : n_inferred,
            'acceptance_rate_pct'  : round(acceptance_rate, 1),
            'refinement_rate_pct'  : round(refinement_rate, 1),
            'expansion_rate_pct'   : round(expansion_rate, 1),
            'avg_confidence'       : round(avg_conf, 3),
            'min_confidence'       : round(min_conf, 3),
            'max_confidence'       : round(max_conf, 3),
            'graph_nodes'          : graph.get('nodes', 0),
            'graph_edges'          : graph.get('edges', 0),
        }

    @staticmethod
    def stability_check(results_a: Dict[str, Any],
                        results_b: Dict[str, Any]) -> Dict[str, Any]:
        def triple_set(r):
            return {
                (t['subject'], t['relation'], t['object'])
                for t in r.get('refined_triples', [])
            }
        set_a = triple_set(results_a)
        set_b = triple_set(results_b)
        intersection = set_a & set_b
        union        = set_a | set_b
        jaccard = len(intersection) / len(union) if union else 1.0
        return {
            'run_a_triples'     : len(set_a),
            'run_b_triples'     : len(set_b),
            'common_triples'    : len(intersection),
            'jaccard_similarity': round(jaccard, 3),
        }

# ─────────────────────────────────────────────────────────────────────────────
# Phase 12 — Feedback Loop
# ─────────────────────────────────────────────────────────────────────────────

class FeedbackLoop:
    def analyse(self, run_results: Dict[str, Any]) -> Dict[str, Any]:
        rejected = run_results.get('rejected_triples', [])
        rel_counts: Dict[str, int] = defaultdict(int)
        for t in rejected:
            rel_counts[t.get('relation', 'unknown')] += 1

        top_rejected = sorted(rel_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        suggestions  = []

        if not run_results.get('raw_triples'):
            suggestions.append(
                'LLM returned no triples — check Ollama connectivity or refine the prompt.'
            )
        if len(rejected) > len(run_results.get('accepted_triples', [])):
            suggestions.append(
                'More triples rejected than accepted — consider lowering '
                'CONFIDENCE_THRESHOLD or reviewing banned relations.'
            )
        if top_rejected:
            common = [r for r, _ in top_rejected]
            suggestions.append(
                f'Frequently rejected relations: {common}. '
                'Consider removing from the banned list if they carry useful information.'
            )
        return {
            'top_rejected_relations': top_rejected,
            'suggestions'           : suggestions,
        }

    @staticmethod
    def update_banned_relations(reasoner: SymbolicReasoner,
                                relations_to_allow: List[str]) -> None:
        for rel in relations_to_allow:
            reasoner.banned_relations.discard(rel)

# ─────────────────────────────────────────────────────────────────────────────
# Phase 13 — Backend API (called by Streamlit frontend)
# ─────────────────────────────────────────────────────────────────────────────

_neural_extractor     = NeuralExtractor()
_rule_based_extractor = RuleBasedExtractor()
_confidence_est       = ConfidenceEstimator()
_sym_reasoner         = SymbolicReasoner()
_kg_builder           = KnowledgeGraphBuilder()
_inference_eng        = InferenceEngine()
_evaluator            = EvaluationModule()
_feedback             = FeedbackLoop()
_storage              = StorageManager()

_last_results: Dict[str, Any] = {}


def process_text(input_text: str, accumulate: bool = False) -> Dict[str, Any]:
    """
    Full pipeline: raw text → knowledge graph.

    Parameters
    ----------
    input_text : str  — raw unstructured text
    accumulate : bool — add to existing graph (True) or rebuild (False)

    Returns
    -------
    dict with keys: preprocessed, raw_triples, accepted_triples,
                    rejected_triples, refined_triples, inferred_triples,
                    graph_stats, evaluation, feedback
    """
    global _last_results

    if not accumulate:
        _kg_builder.clear()

    pp = Preprocessor.process(input_text)

    raw_lines: List[str] = []
    for sentence in pp['sentences']:
        if OLLAMA_AVAILABLE:
            raw_lines.extend(_neural_extractor.extract(sentence))
        else:
            raw_lines.extend(_rule_based_extractor.extract(sentence))

    raw_triples = TripletParser.parse_all(raw_lines)
    filtered    = _confidence_est.filter_triples(raw_triples)
    accepted    = filtered['accepted']
    rejected    = filtered['rejected']
    refined     = _sym_reasoner.apply(accepted)

    _kg_builder.add_triples(refined)
    _storage.save_graph(_kg_builder)
    _storage.save_triples(refined, tag='refined')

    inferred = _inference_eng.run(_kg_builder)

    results = {
        'preprocessed'    : pp,
        'raw_triples'     : raw_triples,
        'accepted_triples': accepted,
        'rejected_triples': rejected,
        'refined_triples' : refined,
        'inferred_triples': inferred,
        'graph_stats'     : _kg_builder.get_stats(),
    }
    results['evaluation'] = _evaluator.compute(results)
    results['feedback']   = _feedback.analyse(results)

    _last_results = results
    return results


def get_triples() -> Dict[str, List]:
    """Return the triple sets from the last pipeline run."""
    return {
        'raw'     : _last_results.get('raw_triples', []),
        'accepted': _last_results.get('accepted_triples', []),
        'rejected': _last_results.get('rejected_triples', []),
        'refined' : _last_results.get('refined_triples', []),
        'inferred': _last_results.get('inferred_triples', []),
    }


def get_graph() -> Dict[str, Any]:
    """Return a JSON-serialisable snapshot of the current knowledge graph."""
    return {
        'graph'     : _kg_builder.to_serialisable(),
        'stats'     : _kg_builder.get_stats(),
        'evaluation': _last_results.get('evaluation', {}),
    }


def query_graph(query: str) -> Dict[str, Any]:
    """
    Query the knowledge graph with a natural-language string.
    Returns a dict with 'intent', 'results', and 'formatted' keys.
    """
    engine = QueryEngine(_kg_builder)
    result = engine.free_text(query)
    result['formatted'] = engine.format_results(result.get('results', []))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test when run directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    DEMO = (
        'Albert Einstein was born in Germany. He developed the theory of relativity '
        'and worked at Princeton University. Einstein won the Nobel Prize in Physics. '
        'Marie Curie was born in Warsaw, Poland. She discovered radium and polonium. '
        'Marie Curie won the Nobel Prize in Chemistry. She studied at the University '
        'of Paris, which is located in France.'
    )
    print('Running pipeline…')
    res = process_text(DEMO)
    print(f"\nRefined triples ({len(res['refined_triples'])}):")
    for t in res['refined_triples']:
        print(f"  ({t['subject']}, {t['relation']}, {t['object']})  conf={t.get('confidence','')}")
    print(f"\nInferred triples ({len(res['inferred_triples'])}):")
    for t in res['inferred_triples']:
        print(f"  ({t['subject']}, {t['relation']}, {t['object']})  via: {t.get('via','')}")
    print('\nEvaluation:')
    for k, v in res['evaluation'].items():
        print(f'  {k}: {v}')
    print('\nQuery: tell me about Albert Einstein')
    print(query_graph('tell me about Albert Einstein')['formatted'])
