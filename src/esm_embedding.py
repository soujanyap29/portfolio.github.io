from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from src.sequence_preprocessing import collate_sequences


class ESM2Embedder:
    def __init__(self, model_name: str, cache_dir: str, device: str = "cpu"):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _cache_path(self, sequence: str) -> Path:
        digest = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.npy"

    @torch.no_grad()
    def embed_sequence(self, sequence: str) -> np.ndarray:
        cache_path = self._cache_path(sequence)
        if cache_path.exists():
            return np.load(cache_path)

        toks = self.tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024)
        toks = {k: v.to(self.device) for k, v in toks.items()}
        hidden = self.model(**toks).last_hidden_state
        emb = hidden.mean(dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)
        np.save(cache_path, emb)
        return emb

    def embed_batch(self, sequences: Iterable[str]) -> np.ndarray:
        return np.stack([self.embed_sequence(s) for s in sequences], axis=0)


class BiLSTMSequenceEncoder(nn.Module):
    def __init__(self, vocab_size: int = 40, emb_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=1,
        )

    def forward(self, sequences: List[str]) -> torch.Tensor:
        batch = collate_sequences(sequences)
        x = self.embedding(batch.input_ids.to(self.embedding.weight.device))
        out, _ = self.encoder(x)
        mask = batch.attention_mask.to(out.device).unsqueeze(-1)
        out = out * mask
        denom = mask.sum(dim=1).clamp(min=1)
        pooled = out.sum(dim=1) / denom
        return pooled
