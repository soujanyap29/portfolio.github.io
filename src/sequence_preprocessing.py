from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWYBXZJUO-"
AA_TO_ID = {aa: i + 1 for i, aa in enumerate(AA_VOCAB)}


@dataclass
class SequenceBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


def tokenize_sequence(seq: str, max_len: int = 2048) -> List[int]:
    ids = [AA_TO_ID.get(ch, 0) for ch in seq[:max_len]]
    return ids


def collate_sequences(sequences: Iterable[str], max_len: int = 2048) -> SequenceBatch:
    tokens = [tokenize_sequence(s, max_len=max_len) for s in sequences]
    max_batch_len = max(len(x) for x in tokens) if tokens else 1
    max_batch_len = min(max_batch_len, max_len)

    input_ids = torch.zeros((len(tokens), max_batch_len), dtype=torch.long)
    attention_mask = torch.zeros((len(tokens), max_batch_len), dtype=torch.long)
    for i, ids in enumerate(tokens):
        ids = ids[:max_batch_len]
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, : len(ids)] = 1
    return SequenceBatch(input_ids=input_ids, attention_mask=attention_mask)
