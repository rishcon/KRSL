"""Presentation mapping for recognizer results."""

from __future__ import annotations

from dataclasses import dataclass

from krsl_ai.inference.video import VideoPrediction

RUSSIAN_LABELS = {
    "gde": "Где",
    "kak": "Как",
    "kakoi": "Какой",
    "kogda": "Когда",
    "kotoriy": "Который",
    "kto": "Кто",
    "kuda": "Куда",
    "skolko": "Сколько",
    "what": "Что",
    "zachem": "Зачем",
}


@dataclass(frozen=True)
class PredictionPresentation:
    """Text and values rendered by the desktop application."""

    title: str
    status: str
    confidence_text: str
    confidence_percent: float
    accepted: bool
    candidates: tuple[tuple[str, str], ...]


def display_label(label: str) -> str:
    """Return a readable Russian name while preserving the dataset label."""
    if label == "UNKNOWN":
        return "Не распознано · UNKNOWN"
    base = label[:-1] if label.endswith("Q") else label
    translated = RUSSIAN_LABELS.get(base)
    return f"{translated} · {label}" if translated else label


def present_prediction(prediction: VideoPrediction) -> PredictionPresentation:
    """Convert a domain prediction into stable UI text."""
    return PredictionPresentation(
        title=display_label(prediction.label),
        status="Жест распознан" if prediction.accepted else "Недостаточно уверенности",
        confidence_text=(
            f"Уверенность {prediction.confidence:.1%} · порог {prediction.threshold:.0%}"
        ),
        confidence_percent=prediction.confidence * 100,
        accepted=prediction.accepted,
        candidates=tuple(
            (display_label(candidate.label), f"{candidate.confidence:.1%}")
            for candidate in prediction.candidates
        ),
    )
