"""Convert holistic cache artifacts into fixed-length training sequences."""

from __future__ import annotations

import numpy as np

SEQUENCE_SCHEMA_VERSION = "training-sequence-v1"
SEQUENCE_LENGTH = 70
POSE_INDICES = np.array([0, 11, 12, 13, 14, 15, 16, 23, 24], dtype=np.intp)
FACE_INDICES = np.linspace(0, 477, 20, dtype=np.intp)


def expected_source_group(label: str) -> str:
    """Return the canonical raw folder name for a manifest label."""
    if label.endswith("Q"):
        return f"_{label[:-1].lower()}_q"
    return f"_{label.lower()}"


def normalize_landmarks(values: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Center landmarks on the shoulders and scale by shoulder width."""
    left_shoulder, right_shoulder = pose[:, 11], pose[:, 12]
    center = (left_shoulder + right_shoulder) / 2
    scale = np.linalg.norm(left_shoulder - right_shoulder, axis=1, keepdims=True)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    return (values - center[:, None, :]) / scale[:, None, :]


def build_sequence(
    artifact: dict[str, np.ndarray], sequence_length: int = SEQUENCE_LENGTH
) -> tuple[np.ndarray, np.ndarray]:
    """Build fixed-length features and a validity mask from one cached artifact."""
    pose = artifact["pose"].astype(np.float32)
    groups = (
        normalize_landmarks(artifact["left_hand"], pose),
        normalize_landmarks(artifact["right_hand"], pose),
        normalize_landmarks(artifact["pose"][:, POSE_INDICES], pose),
        normalize_landmarks(artifact["face"][:, FACE_INDICES], pose),
    )
    coordinates = np.concatenate([group.reshape(len(pose), -1) for group in groups], axis=1)
    presence = np.column_stack(
        [
            artifact["left_hand_present"],
            artifact["right_hand_present"],
            artifact["pose_present"],
            artifact["face_present"],
        ]
    ).astype(np.float32)
    features = np.nan_to_num(np.concatenate([coordinates, presence], axis=1), nan=0.0)
    source_length = len(features)
    output = np.zeros((sequence_length, features.shape[1]), dtype=np.float32)
    mask = np.zeros(sequence_length, dtype=bool)
    if source_length >= sequence_length:
        indices = np.linspace(0, source_length - 1, sequence_length, dtype=np.intp)
        output[:] = features[indices]
        mask[:] = True
    else:
        output[:source_length] = features
        mask[:source_length] = True
    return output, mask
