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


@dataclass(frozen=True)
class LoadedLstmCheckpoint:
    """Runtime model and preprocessing policy stored in one checkpoint."""

    model: LstmClassifier
    labels: dict[str, int]
    model_type: str
    unknown_threshold: float


def load_lstm_checkpoint(checkpoint_path: Path) -> LoadedLstmCheckpoint:
    """Load a published BiLSTM checkpoint on CPU."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    labels = checkpoint["labels"]
    feature_size = checkpoint["model"]["lstm.weight_ih_l0"].shape[1]
    model = LstmClassifier(feature_size, len(labels))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    config = checkpoint.get("config", {})
    return LoadedLstmCheckpoint(
        model=model,
        labels=labels,
        model_type=str(config.get("model_type", "velocity-bilstm-v1")),
        unknown_threshold=float(checkpoint.get("unknown_threshold", 0.6)),
    )


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
    threshold: float | None = None,
    top_k: int = 3,
) -> VideoPrediction:
    """Extract checkpoint-compatible features and recognize one isolated sign."""
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    checkpoint = load_lstm_checkpoint(checkpoint_path)
    artifact = extract_video(video_path, holistic_model_path)
    hand_centric = checkpoint.model_type == "hand-centric-bilstm-v3"
    features, mask = build_sequence(
        artifact.__dict__,
        include_velocity=not hand_centric,
        hand_centric=hand_centric,
    )
    selected_threshold = checkpoint.unknown_threshold if threshold is None else threshold
    return predict_sequence(
        checkpoint.model,
        features,
        mask,
        checkpoint.labels,
        selected_threshold,
        top_k,
    )
