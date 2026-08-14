"""Build the Phase 1 KRSL20 manifest and signer-independent split files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from krsl_ai.data.manifest import build_samples, validate_signer_split, write_manifest, write_report

DEFAULT_SPLITS = {
    "P1": "test",
    "P2": "validation",
    "P3": "train",
    "P4": "train",
    "P5": "train",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--keypoint-root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--split-config", type=Path, required=True)
    args = parser.parse_args()

    samples, unmatched = build_samples(args.video_root, DEFAULT_SPLITS)
    validate_signer_split(samples)
    write_manifest(samples, args.manifest)
    write_report(samples, unmatched, args.keypoint_root, args.report)
    args.split_config.parent.mkdir(parents=True, exist_ok=True)
    args.split_config.write_text(
        json.dumps({"schema_version": 1, "split_by_signer": DEFAULT_SPLITS}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
