"""Tests for desktop result presentation."""

import pytest

from krsl_ai.inference.video import RankedPrediction, VideoPrediction
from krsl_ai.ui.presentation import display_label, present_prediction


def test_display_label_preserves_dataset_variant() -> None:
    assert display_label("gde") == "Где · gde"
    assert display_label("gdeQ") == "Где · gdeQ"
    assert display_label("UNKNOWN") == "Не распознано · UNKNOWN"


def test_present_prediction_formats_accepted_result() -> None:
    prediction = VideoPrediction(
        label="kto",
        predicted_label="kto",
        confidence=0.734,
        threshold=0.59,
        accepted=True,
        candidates=(RankedPrediction("kto", 0.734), RankedPrediction("ktoQ", 0.2)),
    )

    view = present_prediction(prediction)

    assert view.title == "Кто · kto"
    assert view.status == "Жест распознан"
    assert view.confidence_text == "Уверенность 73.4% · порог 59%"
    assert view.confidence_percent == pytest.approx(73.4)
    assert view.candidates[0] == ("Кто · kto", "73.4%")


def test_present_prediction_explains_unknown() -> None:
    prediction = VideoPrediction(
        label="UNKNOWN",
        predicted_label="kak",
        confidence=0.4,
        threshold=0.59,
        accepted=False,
        candidates=(RankedPrediction("kak", 0.4),),
    )

    view = present_prediction(prediction)

    assert view.title == "Не распознано · UNKNOWN"
    assert view.status == "Недостаточно уверенности"
    assert view.accepted is False
