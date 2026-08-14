import numpy as np

from krsl_ai.features.quality import summarize_artifacts


def test_summarize_artifacts_reports_detection_rates() -> None:
    artifact = {
        "timestamps_ms": np.array([0, 1]),
        "left_hand_present": np.array([True, False]),
        "right_hand_present": np.array([True, True]),
        "pose_present": np.array([True, True]),
        "face_present": np.array([False, False]),
    }

    summary = summarize_artifacts([artifact])

    assert summary["artifact_count"] == 1
    assert summary["total_frames"] == 2
    assert summary["presence_rate"] == {
        "left_hand": 0.5,
        "right_hand": 1.0,
        "pose": 1.0,
        "face": 0.0,
    }
