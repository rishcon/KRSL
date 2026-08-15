"""Class-balancing policies for KRSL model training."""

from __future__ import annotations

import torch


def inverse_frequency_weights(targets: list[int], class_count: int) -> torch.Tensor:
    """Give every class equal total weight in cross-entropy loss."""
    counts = torch.bincount(torch.tensor(targets), minlength=class_count).float()
    if len(counts) != class_count or (counts == 0).any():
        raise ValueError("Every configured class must have at least one training sample.")
    return len(targets) / (class_count * counts)
