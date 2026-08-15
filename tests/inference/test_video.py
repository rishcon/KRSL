"""Tests for isolated-video prediction policy."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from krsl_ai.inference.video import load_lstm_checkpoint, predict_sequence


class FixedModel(nn.Module):
    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        assert features.shape == (1, 4, 2)
        assert 1 <= lengths.item() <= 4
        return torch.tensor([[0.0, 2.0, 1.0]])


def test_predict_sequence_returns_ranked_label() -> None:
    result = predict_sequence(
        FixedModel(),
        np.zeros((4, 2), dtype=np.float32),
        np.array([True, True, True, False]),
        {"A": 0, "B": 1, "C": 2},
        threshold=0.6,
    )

    assert result.label == "B"
    assert result.accepted is True
    assert [candidate.label for candidate in result.candidates] == ["B", "C", "A"]


def test_predict_sequence_abstains_below_threshold() -> None:
    result = predict_sequence(
        FixedModel(),
        np.zeros((4, 2), dtype=np.float32),
        np.ones(4, dtype=bool),
        {"A": 0, "B": 1, "C": 2},
        threshold=0.9,
    )

    assert result.label == "UNKNOWN"
    assert result.predicted_label == "B"
    assert result.accepted is False


def test_predict_sequence_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        predict_sequence(
            FixedModel(),
            np.zeros((4, 2), dtype=np.float32),
            np.ones(4, dtype=bool),
            {"A": 0, "B": 1, "C": 2},
            threshold=1.1,
        )


def test_published_v3_checkpoint_contains_runtime_policy() -> None:
    checkpoint = load_lstm_checkpoint(Path("models/lstm-handcentric-v3.pt"))

    assert checkpoint.model_type == "hand-centric-bilstm-v3"
    assert checkpoint.unknown_threshold == pytest.approx(0.59)
    assert checkpoint.model.lstm.input_size == 442
