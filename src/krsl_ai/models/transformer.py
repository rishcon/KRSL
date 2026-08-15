"""Compact Transformer classifier for fixed-length landmark sequences."""

from __future__ import annotations

import torch
from torch import nn


class TransformerClassifier(nn.Module):
    """Encode a masked landmark sequence and classify one isolated sign."""

    def __init__(
        self,
        feature_size: int,
        class_count: int,
        sequence_length: int = 70,
        model_size: int = 128,
        head_count: int = 4,
        layer_count: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(feature_size, model_size)
        self.position = nn.Parameter(torch.zeros(1, sequence_length, model_size))
        layer = nn.TransformerEncoderLayer(
            d_model=model_size,
            nhead=head_count,
            dim_feedforward=model_size * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layer_count)
        self.normalization = nn.LayerNorm(model_size)
        self.classifier = nn.Linear(model_size, class_count)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        sequence_length = features.shape[1]
        positions = torch.arange(sequence_length, device=features.device)[None, :]
        padding_mask = positions >= lengths[:, None]
        encoded = self.input_projection(features) + self.position[:, :sequence_length]
        encoded = self.encoder(encoded, src_key_padding_mask=padding_mask)
        valid = (~padding_mask).unsqueeze(-1)
        pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return self.classifier(self.normalization(pooled))
