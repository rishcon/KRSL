"""Batch-extract MediaPipe Holistic features from a KRSL20 manifest."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from krsl_ai.features.batch import run_batch, write_summary
from krsl_ai.features.holistic import extract_video, save_features


def count_rows(manifest_path: Path, limit: int | None) -> int:
    with manifest_path.open(encoding="utf-8") as manifest_file:
        total = max(sum(1 for _ in manifest_file) - 1, 0)
    return min(total, limit) if limit is not None else total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--failure-log", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    total = count_rows(args.manifest, args.limit)

    def show_progress(summary: object) -> None:
        completed = getattr(summary, "attempted")
        extracted = getattr(summary, "extracted")
        skipped = getattr(summary, "skipped")
        failed = getattr(summary, "failed")
        print(
            f"[{completed}/{total}] extracted={extracted} skipped={skipped} failed={failed}",
            flush=True,
        )

    summary = run_batch(
        manifest_path=args.manifest,
        video_root=args.video_root,
        output_root=args.output_root,
        failure_log=args.failure_log,
        extractor=partial(extract_video, model_path=args.model),
        save=save_features,
        limit=args.limit,
        overwrite=args.overwrite,
        progress_callback=show_progress,
    )
    write_summary(summary, args.summary)
    if summary.failed:
        raise SystemExit("Batch completed with extraction failures; inspect the failure log.")


if __name__ == "__main__":
    main()
