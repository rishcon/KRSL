"""Quality checks for cached landmark artifacts."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

PRESENCE_KEYS = (
    "left_hand_present",
    "right_hand_present",
    "pose_present",
    "face_present",
)


def summarize_artifacts(artifacts: Iterable[dict[str, np.ndarray]]) -> dict[str, object]:
    """Summarize sequence lengths and landmark detection rates."""
    frame_counts: list[int] = []
    presence_totals = {key: 0 for key in PRESENCE_KEYS}
    total_frames = 0

    for artifact in artifacts:
        frame_count = len(artifact["timestamps_ms"])
        frame_counts.append(frame_count)
        total_frames += frame_count
        for key in PRESENCE_KEYS:
            presence_totals[key] += int(np.count_nonzero(artifact[key]))

    if not frame_counts:
        raise ValueError("No valid artifacts were supplied.")
    counts = np.asarray(frame_counts)
    return {
        "artifact_count": len(frame_counts),
        "total_frames": total_frames,
        "frames_per_video": {
            "min": int(counts.min()),
            "p05": float(np.percentile(counts, 5)),
            "median": float(np.median(counts)),
            "p95": float(np.percentile(counts, 95)),
            "max": int(counts.max()),
        },
        "presence_rate": {
            key.removesuffix("_present"): presence_totals[key] / total_frames
            for key in PRESENCE_KEYS
        },
    }
