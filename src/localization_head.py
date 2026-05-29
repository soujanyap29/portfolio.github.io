from __future__ import annotations

import torch.nn as nn


class LocalizationHead(nn.Module):
    def __init__(self, in_dim: int, num_labels: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, num_labels),
        )

    def forward(self, x):
        return self.net(x)
