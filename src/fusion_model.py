from __future__ import annotations

from typing import Optional

import timm
import torch
import torch.nn as nn

from src.alzheimers_head import AlzheimersHead, AlzheimersPlaceholderModule
from src.config import LOCALIZATION_LABELS
from src.esm_embedding import BiLSTMSequenceEncoder
from src.localization_head import LocalizationHead


class ImageEncoder(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b0", out_dim: int = 512, pretrained: bool = True):
        super().__init__()
        model_name = "efficientnet_b0" if backbone == "efficientnet_b0" else "resnet50"
        self.encoder = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        in_features = self.encoder.num_features
        self.project = nn.Linear(in_features, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.encoder(x))


class AttentionFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)

    def forward(self, image_emb: torch.Tensor, seq_emb: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack([image_emb, seq_emb], dim=1)
        fused, _ = self.attn(tokens, tokens, tokens)
        return fused.mean(dim=1)


class MultimodalLocalizationModel(nn.Module):
    def __init__(
        self,
        image_backbone: str = "efficientnet_b0",
        image_emb_dim: int = 512,
        seq_emb_dim: int = 480,
        fusion_dim: int = 512,
        fusion_type: str = "attention",
        seq_model: str = "esm2",
        has_alzheimers_labels: bool = False,
    ):
        super().__init__()
        self.seq_model = seq_model
        self.image_encoder = ImageEncoder(backbone=image_backbone, out_dim=image_emb_dim)
        self.seq_encoder = BiLSTMSequenceEncoder(hidden_dim=seq_emb_dim // 2) if seq_model == "bilstm" else None
        self.seq_projection = nn.Linear(seq_emb_dim, fusion_dim)
        self.image_projection = nn.Linear(image_emb_dim, fusion_dim)
        self.fusion_type = fusion_type
        self.fusion = AttentionFusion(fusion_dim) if fusion_type == "attention" else None

        head_in = fusion_dim if fusion_type == "attention" else fusion_dim * 2
        self.localization_head = LocalizationHead(head_in, len(LOCALIZATION_LABELS))
        self.alzheimers_head = AlzheimersHead(head_in) if has_alzheimers_labels else AlzheimersPlaceholderModule(head_in)

    def encode_sequence(self, sequences, seq_embedding: Optional[torch.Tensor] = None):
        if self.seq_model == "bilstm":
            return self.seq_encoder(sequences)
        if seq_embedding is None:
            raise ValueError("ESM mode requires precomputed seq_embedding in batch")
        return seq_embedding

    def forward(self, images: torch.Tensor, sequences, seq_embedding: Optional[torch.Tensor] = None):
        image_emb = self.image_projection(self.image_encoder(images))
        seq_emb = self.seq_projection(self.encode_sequence(sequences, seq_embedding=seq_embedding))

        if self.fusion_type == "attention":
            fused = self.fusion(image_emb, seq_emb)
        else:
            fused = torch.cat([image_emb, seq_emb], dim=-1)

        localization_logits = self.localization_head(fused)
        alzheimers_logits = self.alzheimers_head(fused)
        return {
            "localization_logits": localization_logits,
            "alzheimers_logits": alzheimers_logits,
            "fused_features": fused,
        }
