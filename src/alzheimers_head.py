from __future__ import annotations

import torch
import torch.nn as nn


class AlzheimersHead(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x).squeeze(-1)


class AlzheimersPlaceholderModule(nn.Module):
    """Future-ready module when labels are not available."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.projection = nn.Linear(in_dim, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)
