import numpy as np

from krsl_ai.features.training import (
    SEMANTIC_FACE_INDICES,
    build_sequence,
    expected_source_group,
)


def test_expected_source_group_marks_question_labels() -> None:
    assert expected_source_group("ktoQ") == "_kto_q"
    assert expected_source_group("zachem") == "_zachem"


def test_build_sequence_pads_short_sequences() -> None:
    frames = 2
    artifact = {
        "left_hand": np.ones((frames, 21, 3), dtype=np.float32),
        "right_hand": np.ones((frames, 21, 3), dtype=np.float32),
        "pose": np.ones((frames, 33, 3), dtype=np.float32),
        "face": np.ones((frames, 478, 3), dtype=np.float32),
        "left_hand_present": np.ones(frames, dtype=bool),
        "right_hand_present": np.ones(frames, dtype=bool),
        "pose_present": np.ones(frames, dtype=bool),
        "face_present": np.ones(frames, dtype=bool),
    }

    features, mask = build_sequence(artifact, sequence_length=3)

    assert features.shape == (3, 217)
    assert mask.tolist() == [True, True, False]
    assert np.all(features[2] == 0)


def test_hand_centric_sequence_suppresses_missing_hand_velocity() -> None:
    frames = 3
    pose = np.ones((frames, 33, 3), dtype=np.float32)
    pose[:, 12, 0] = 2
    left_hand = np.arange(frames * 21 * 3, dtype=np.float32).reshape(frames, 21, 3)
    left_hand[1] = np.nan
    artifact = {
        "left_hand": left_hand,
        "right_hand": np.ones((frames, 21, 3), dtype=np.float32),
        "pose": pose,
        "face": np.ones((frames, 478, 3), dtype=np.float32),
        "left_hand_present": np.array([True, False, True]),
        "right_hand_present": np.ones(frames, dtype=bool),
        "pose_present": np.ones(frames, dtype=bool),
        "face_present": np.ones(frames, dtype=bool),
    }

    features, mask = build_sequence(artifact, sequence_length=3, hand_centric=True)

    assert len(SEMANTIC_FACE_INDICES) == 20
    assert features.shape == (3, 442)
    assert mask.tolist() == [True, True, True]
    assert np.all(features[:, 219:285] == 0)
