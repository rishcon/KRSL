"""Landmark-sequence augmentation used only while training."""

from __future__ import annotations

import torch
from torch.nn import functional as functional

PRESENCE_FEATURE_COUNT = 4


def augment_sequence(
    features: torch.Tensor,
    length: int,
    *,
    generator: torch.Generator | None = None,
    noise_std: float = 0.01,
    minimum_time_scale: float = 0.85,
    maximum_time_scale: float = 1.15,
) -> tuple[torch.Tensor, int]:
    """Apply mild temporal scaling and landmark noise to one valid sequence."""
    if length < 1 or length > len(features):
        raise ValueError("Sequence length must identify at least one valid frame.")
    if features.shape[1] <= PRESENCE_FEATURE_COUNT:
        raise ValueError("Sequence does not contain coordinate and presence features.")

    time_scale = float(
        torch.empty(1).uniform_(minimum_time_scale, maximum_time_scale, generator=generator)
    )
    target_length = min(len(features), max(1, round(length * time_scale)))
    valid = features[:length].transpose(0, 1).unsqueeze(0)
    resampled = functional.interpolate(
        valid,
        size=target_length,
        mode="linear",
        align_corners=length > 1,
    )[0].transpose(0, 1)

    coordinate_count = features.shape[1] - PRESENCE_FEATURE_COUNT
    spatial_scale = float(torch.empty(1).uniform_(0.97, 1.03, generator=generator))
    coordinates = resampled[:, :coordinate_count]
    noise = torch.randn(
        coordinates.shape,
        dtype=coordinates.dtype,
        device=coordinates.device,
        generator=generator,
    )
    resampled[:, :coordinate_count] = coordinates * spatial_scale + noise * noise_std
    resampled[:, coordinate_count:] = (resampled[:, coordinate_count:] >= 0.5).to(resampled.dtype)

    output = torch.zeros_like(features)
    output[:target_length] = resampled
    return output, target_length
