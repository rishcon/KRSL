"""Run the velocity BiLSTM recognizer on one prepared landmark sequence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from krsl_ai.features.holistic import extract_video
from krsl_ai.features.training import build_sequence
from krsl_ai.models.lstm import LstmClassifier


@dataclass(frozen=True)
class RankedPrediction:
    """One candidate label and its softmax confidence."""

    label: str
    confidence: float


@dataclass(frozen=True)
class VideoPrediction:
    """Recognition result with abstention and ranked candidates."""

    label: str
    predicted_label: str
    confidence: float
    threshold: float
    accepted: bool
    candidates: tuple[RankedPrediction, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def load_lstm_checkpoint(checkpoint_path: Path) -> tuple[LstmClassifier, dict[str, int]]:
    """Load a published BiLSTM checkpoint on CPU."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    labels = checkpoint["labels"]
    feature_size = checkpoint["model"]["lstm.weight_ih_l0"].shape[1]
    model = LstmClassifier(feature_size, len(labels))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, labels


def predict_sequence(
    model: nn.Module,
    features: np.ndarray,
    mask: np.ndarray,
    labels: dict[str, int],
    threshold: float = 0.6,
    top_k: int = 3,
) -> VideoPrediction:
    """Classify one sequence and return UNKNOWN below the confidence threshold."""
    if not 0 <= threshold <= 1:
        raise ValueError("Confidence threshold must be between 0 and 1.")
    length = int(mask.sum())
    if length < 1:
        raise ValueError("The sequence contains no valid frames.")
    names = {index: label for label, index in labels.items()}
    with torch.no_grad():
        logits = model(
            torch.from_numpy(features.astype(np.float32))[None, ...],
            torch.tensor([length], dtype=torch.long),
        )
        probabilities = torch.softmax(logits[0], dim=0)
    count = min(max(top_k, 1), len(labels))
    confidence_values, indices = probabilities.topk(count)
    candidates = tuple(
        RankedPrediction(names[int(index)], float(confidence))
        for confidence, index in zip(confidence_values, indices, strict=True)
    )
    best = candidates[0]
    accepted = best.confidence >= threshold
    return VideoPrediction(
        label=best.label if accepted else "UNKNOWN",
        predicted_label=best.label,
        confidence=best.confidence,
        threshold=threshold,
        accepted=accepted,
        candidates=candidates,
    )


def recognize_video(
    video_path: Path,
    holistic_model_path: Path,
    checkpoint_path: Path,
    threshold: float = 0.6,
    top_k: int = 3,
) -> VideoPrediction:
    """Extract velocity features from one video and recognize its isolated sign."""
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    artifact = extract_video(video_path, holistic_model_path)
    features, mask = build_sequence(artifact.__dict__, include_velocity=True)
    model, labels = load_lstm_checkpoint(checkpoint_path)
    return predict_sequence(model, features, mask, labels, threshold, top_k)
