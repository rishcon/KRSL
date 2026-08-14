"""Batch orchestration for versioned feature extraction."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from krsl_ai.features.holistic import FEATURE_SCHEMA_VERSION, HolisticFeatures


@dataclass(frozen=True)
class BatchSummary:
    attempted: int
    extracted: int
    skipped: int
    failed: int


def is_valid_cache(path: Path) -> bool:
    """Return whether a cache artifact has the expected immutable schema."""
    if not path.is_file():
        return False
    try:
        with np.load(path) as artifact:
            return str(artifact["schema_version"]) == FEATURE_SCHEMA_VERSION
    except (KeyError, OSError, ValueError):
        return False


def run_batch(
    manifest_path: Path,
    video_root: Path,
    output_root: Path,
    failure_log: Path,
    extractor: Callable[[Path], HolisticFeatures],
    save: Callable[[HolisticFeatures, Path, Path], None],
    limit: int | None = None,
    overwrite: bool = False,
) -> BatchSummary:
    """Process manifest rows while preserving failures and reusable cache files."""
    output_root.mkdir(parents=True, exist_ok=True)
    failure_log.parent.mkdir(parents=True, exist_ok=True)
    attempted = extracted = skipped = failed = 0

    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        rows = csv.DictReader(manifest_file)
        for row in rows:
            if limit is not None and attempted >= limit:
                break
            attempted += 1
            sample_id = row["sample_id"]
            video_path = video_root / row["relative_path"]
            output_path = output_root / f"{sample_id}.npz"
            if not overwrite and is_valid_cache(output_path):
                skipped += 1
                continue
            try:
                save(extractor(video_path), output_path, video_path)
                extracted += 1
            except Exception as error:  # Keep a per-sample audit trail and continue the batch.
                failure = {
                    "sample_id": sample_id,
                    "relative_path": row["relative_path"],
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
                with failure_log.open("a", encoding="utf-8") as log_file:
                    log_file.write(json.dumps(failure, ensure_ascii=False) + "\n")
                failed += 1
    return BatchSummary(attempted=attempted, extracted=extracted, skipped=skipped, failed=failed)


def write_summary(summary: BatchSummary, output_path: Path) -> None:
    """Write stable aggregate counters without storing source video data."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")
