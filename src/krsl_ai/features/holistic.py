"""MediaPipe Holistic feature extraction for one decoded video."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FEATURE_SCHEMA_VERSION = "holistic-v1"


@dataclass(frozen=True)
class HolisticFeatures:
    """Fixed-shape landmark arrays with a frame-level presence mask per group."""

    left_hand: np.ndarray
    right_hand: np.ndarray
    pose: np.ndarray
    face: np.ndarray
    left_hand_present: np.ndarray
    right_hand_present: np.ndarray
    pose_present: np.ndarray
    face_present: np.ndarray
    timestamps_ms: np.ndarray


def landmarks_to_array(landmarks: list[Any], landmark_count: int) -> tuple[np.ndarray, bool]:
    """Convert MediaPipe landmarks to a fixed shape, retaining missingness explicitly."""
    values = np.full((landmark_count, 3), np.nan, dtype=np.float32)
    if not landmarks:
        return values, False
    if len(landmarks) != landmark_count:
        raise ValueError(f"Expected {landmark_count} landmarks, got {len(landmarks)}.")
    values[:, :] = [(landmark.x, landmark.y, landmark.z) for landmark in landmarks]
    return values, True


def extract_video(video_path: Path, model_path: Path) -> HolisticFeatures:
    """Extract face, pose, and hand landmarks using MediaPipe VIDEO mode."""
    import cv2
    import mediapipe as mp

    if not model_path.is_file():
        raise FileNotFoundError(f"MediaPipe model asset not found: {model_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp.tasks.vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
    )
    groups: dict[str, list[np.ndarray | bool]] = {
        "left_hand": [],
        "right_hand": [],
        "pose": [],
        "face": [],
        "left_hand_present": [],
        "right_hand_present": [],
        "pose_present": [],
        "face_present": [],
    }
    timestamps_ms: list[int] = []

    with mp.tasks.vision.HolisticLandmarker.create_from_options(options) as landmarker:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_ms = int(capture.get(cv2.CAP_PROP_POS_MSEC))
            timestamp_ms = max(timestamp_ms, frame_index)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect_for_video(image, timestamp_ms)
            for name, landmarks, count in (
                ("left_hand", result.left_hand_landmarks, 21),
                ("right_hand", result.right_hand_landmarks, 21),
                ("pose", result.pose_landmarks, 33),
                ("face", result.face_landmarks, 478),
            ):
                values, present = landmarks_to_array(landmarks, count)
                groups[name].append(values)
                groups[f"{name}_present"].append(present)
            timestamps_ms.append(timestamp_ms)
            frame_index += 1
    capture.release()

    if not timestamps_ms:
        raise ValueError(f"Video contains no decodable frames: {video_path}")
    return HolisticFeatures(
        left_hand=np.stack(groups["left_hand"]),
        right_hand=np.stack(groups["right_hand"]),
        pose=np.stack(groups["pose"]),
        face=np.stack(groups["face"]),
        left_hand_present=np.asarray(groups["left_hand_present"], dtype=bool),
        right_hand_present=np.asarray(groups["right_hand_present"], dtype=bool),
        pose_present=np.asarray(groups["pose_present"], dtype=bool),
        face_present=np.asarray(groups["face_present"], dtype=bool),
        timestamps_ms=np.asarray(timestamps_ms, dtype=np.int64),
    )


def save_features(features: HolisticFeatures, output_path: Path, source_video: Path) -> None:
    """Store a self-describing compressed feature artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema_version=FEATURE_SCHEMA_VERSION,
        source_video=str(source_video),
        **features.__dict__,
    )
