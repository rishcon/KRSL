"""Extract one video into a versioned MediaPipe Holistic `.npz` artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from krsl_ai.features.holistic import extract_video, save_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    save_features(extract_video(args.video, args.model), args.output, args.video)


if __name__ == "__main__":
    main()
