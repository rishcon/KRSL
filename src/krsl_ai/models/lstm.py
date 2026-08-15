"""Bidirectional LSTM used by the KRSL isolated-sign recognizer."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class LstmClassifier(nn.Module):
    """Classify one variable-length landmark sequence."""

    def __init__(self, feature_size: int, class_count: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(feature_size, 64, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(128, class_count)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            features, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        return self.classifier(torch.cat([hidden[-2], hidden[-1]], dim=1))
