"""Convert holistic cache artifacts into fixed-length training sequences."""

from __future__ import annotations

import numpy as np

SEQUENCE_SCHEMA_VERSION = "training-sequence-v1"
HAND_CENTRIC_SCHEMA_VERSION = "training-sequence-v3"
SEQUENCE_LENGTH = 70
POSE_INDICES = np.array([0, 11, 12, 13, 14, 15, 16, 23, 24], dtype=np.intp)
FACE_INDICES = np.linspace(0, 477, 20, dtype=np.intp)
SEMANTIC_FACE_INDICES = np.array(
    [70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 33, 263, 1, 152, 61, 291, 13, 14, 0, 17],
    dtype=np.intp,
)


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


def normalize_hand_shape(hand: np.ndarray) -> np.ndarray:
    """Center one hand on its wrist and scale it by wrist-to-middle-palm distance."""
    wrist = hand[:, :1]
    palm_scale = np.linalg.norm(hand[:, 9] - hand[:, 0], axis=1, keepdims=True)
    palm_scale = np.where(np.isfinite(palm_scale) & (palm_scale > 1e-6), palm_scale, 1.0)
    return (hand - wrist) / palm_scale[:, None, :]


def presence_aware_velocity(coordinates: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Calculate velocity only across adjacent frames where a group is detected."""
    clean = np.nan_to_num(coordinates, nan=0.0)
    velocity = np.zeros_like(clean)
    valid_pairs = present[1:] & present[:-1]
    velocity[1:][valid_pairs] = clean[1:][valid_pairs] - clean[:-1][valid_pairs]
    return velocity


def hand_centric_groups(artifact: dict[str, np.ndarray]) -> tuple[tuple[np.ndarray, ...], ...]:
    """Build v3 coordinate groups and matching frame-presence masks."""
    pose = artifact["pose"].astype(np.float32)
    left_hand = normalize_landmarks(artifact["left_hand"], pose)
    right_hand = normalize_landmarks(artifact["right_hand"], pose)
    left_features = np.concatenate(
        [normalize_hand_shape(left_hand).reshape(len(pose), -1), left_hand[:, 0]], axis=1
    )
    right_features = np.concatenate(
        [normalize_hand_shape(right_hand).reshape(len(pose), -1), right_hand[:, 0]], axis=1
    )
    coordinate_groups = (
        left_features,
        right_features,
        normalize_landmarks(pose[:, POSE_INDICES], pose).reshape(len(pose), -1),
        normalize_landmarks(artifact["face"][:, SEMANTIC_FACE_INDICES], pose).reshape(
            len(pose), -1
        ),
    )
    presence_groups = (
        artifact["left_hand_present"],
        artifact["right_hand_present"],
        artifact["pose_present"],
        artifact["face_present"],
    )
    return coordinate_groups, presence_groups


def build_sequence(
    artifact: dict[str, np.ndarray],
    sequence_length: int = SEQUENCE_LENGTH,
    include_velocity: bool = False,
    hand_centric: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build fixed-length features and a validity mask from one cached artifact."""
    pose = artifact["pose"].astype(np.float32)
    presence = np.column_stack(
        [
            artifact["left_hand_present"],
            artifact["right_hand_present"],
            artifact["pose_present"],
            artifact["face_present"],
        ]
    ).astype(np.float32)
    if hand_centric:
        coordinate_groups, presence_groups = hand_centric_groups(artifact)
        coordinates = np.concatenate(
            [np.nan_to_num(group, nan=0.0) for group in coordinate_groups], axis=1
        )
        velocity = np.concatenate(
            [
                presence_aware_velocity(group, group_present)
                for group, group_present in zip(coordinate_groups, presence_groups, strict=True)
            ],
            axis=1,
        )
        features = np.concatenate([coordinates, velocity, presence], axis=1)
    else:
        groups = (
            normalize_landmarks(artifact["left_hand"], pose),
            normalize_landmarks(artifact["right_hand"], pose),
            normalize_landmarks(artifact["pose"][:, POSE_INDICES], pose),
            normalize_landmarks(artifact["face"][:, FACE_INDICES], pose),
        )
        coordinates = np.concatenate([group.reshape(len(pose), -1) for group in groups], axis=1)
        coordinates = np.nan_to_num(coordinates, nan=0.0)
        velocity = np.diff(coordinates, axis=0, prepend=coordinates[:1])
        features = (
            np.concatenate([coordinates, velocity, presence], axis=1)
            if include_velocity
            else np.concatenate([coordinates, presence], axis=1)
        )
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
