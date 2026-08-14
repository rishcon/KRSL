from types import SimpleNamespace

import numpy as np

from krsl_ai.features.holistic import landmarks_to_array


def test_landmarks_to_array_preserves_coordinates() -> None:
    landmarks = [SimpleNamespace(x=0.1, y=0.2, z=0.3) for _ in range(2)]

    values, present = landmarks_to_array(landmarks, landmark_count=2)

    assert present
    np.testing.assert_allclose(values, [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])


def test_landmarks_to_array_marks_missing_group() -> None:
    values, present = landmarks_to_array([], landmark_count=2)

    assert not present
    assert np.isnan(values).all()
