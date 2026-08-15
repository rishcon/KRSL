import torch

from krsl_ai.models.transformer import TransformerClassifier


def test_transformer_returns_one_logit_vector_per_sequence() -> None:
    model = TransformerClassifier(feature_size=12, class_count=3, model_size=16)
    features = torch.zeros((2, 5, 12))
    lengths = torch.tensor([5, 3])

    logits = model(features, lengths)

    assert logits.shape == (2, 3)
