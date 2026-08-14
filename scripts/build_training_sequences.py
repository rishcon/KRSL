"""Build fixed-length training-sequence-v1 files from local holistic artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from krsl_ai.features.training import (
    SEQUENCE_LENGTH,
    SEQUENCE_SCHEMA_VERSION,
    build_sequence,
    expected_source_group,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    included = excluded = failed = 0

    with args.manifest.open(newline="", encoding="utf-8") as manifest_file:
        for row in csv.DictReader(manifest_file):
            if args.limit is not None and included + excluded + failed >= args.limit:
                break
            if row["source_group"] != expected_source_group(row["label"]):
                excluded += 1
                continue
            artifact_path = args.artifact_root / f"{row['sample_id']}.npz"
            output_path = args.output_root / f"{row['sample_id']}.npz"
            try:
                with np.load(artifact_path) as artifact:
                    features, mask = build_sequence({key: artifact[key] for key in artifact.files})
                np.savez_compressed(
                    output_path,
                    schema_version=SEQUENCE_SCHEMA_VERSION,
                    sample_id=row["sample_id"],
                    label=row["label"],
                    signer_id=row["signer_id"],
                    split=row["split"],
                    features=features,
                    sequence_mask=mask,
                )
                included += 1
            except (KeyError, OSError, ValueError) as error:
                print(f"FAILED {row['sample_id']}: {error}", flush=True)
                failed += 1
            if (included + excluded + failed) % 100 == 0:
                print(
                    f"processed={included + excluded + failed} included={included} "
                    f"excluded={excluded} failed={failed}",
                    flush=True,
                )

    report = {
        "schema_version": SEQUENCE_SCHEMA_VERSION,
        "sequence_length": SEQUENCE_LENGTH,
        "included": included,
        "excluded_label_conflicts": excluded,
        "failed": failed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failed:
        raise SystemExit("Sequence building completed with failures.")


if __name__ == "__main__":
    main()
