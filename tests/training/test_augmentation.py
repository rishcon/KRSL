"""Tests for landmark-sequence augmentation."""

import torch
from pytest import approx

from krsl_ai.training.augmentation import augment_sequence
from krsl_ai.training.balancing import inverse_frequency_weights


def test_augmentation_preserves_shape_and_presence_flags() -> None:
    features = torch.ones((10, 8), dtype=torch.float32)
    generator = torch.Generator().manual_seed(42)

    augmented, length = augment_sequence(features, 6, generator=generator)

    assert augmented.shape == features.shape
    assert 5 <= length <= 7
    assert set(augmented[:length, -4:].unique().tolist()) <= {0.0, 1.0}
    assert torch.all(augmented[length:] == 0)
    assert not torch.equal(augmented[:length, :-4], features[:length, :-4])


def test_inverse_frequency_weights_balance_class_contributions() -> None:
    targets = [0, 0, 0, 1]

    weights = inverse_frequency_weights(targets, class_count=2)

    assert weights.tolist() == approx([2 / 3, 2.0])
    assert float(weights[0] * 3) == approx(float(weights[1]))
