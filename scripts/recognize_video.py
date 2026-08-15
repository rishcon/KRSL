"""Recognize one isolated KRSL sign from a video file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from krsl_ai.inference.video import recognize_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--holistic-model",
        type=Path,
        default=Path("models/holistic_landmarker.task"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/lstm-handcentric-v3.pt"),
    )
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    result = recognize_video(
        args.video,
        args.holistic_model,
        args.checkpoint,
        args.threshold,
        args.top_k,
    )
    if args.as_json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return

    print()
    print(f"Результат: {result.label}")
    print(f"Уверенность: {result.confidence:.1%} (порог {result.threshold:.0%})")
    if not result.accepted:
        print(f"Ближайший известный жест: {result.predicted_label}")
    print("Варианты:")
    for candidate in result.candidates:
        print(f"  {candidate.label}: {candidate.confidence:.1%}")


if __name__ == "__main__":
    main()
