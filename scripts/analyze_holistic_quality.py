"""Create a quality report for locally cached holistic-v1 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from krsl_ai.features.holistic import FEATURE_SCHEMA_VERSION
from krsl_ai.features.quality import PRESENCE_KEYS, summarize_artifacts


def load_artifact(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as artifact:
        if str(artifact["schema_version"]) != FEATURE_SCHEMA_VERSION:
            raise ValueError("Unexpected schema version")
        required_keys = ("timestamps_ms", *PRESENCE_KEYS)
        return {key: artifact[key] for key in required_keys}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    valid: list[dict[str, np.ndarray]] = []
    invalid: list[dict[str, str]] = []
    for path in sorted(args.artifact_root.glob("*.npz")):
        try:
            valid.append(load_artifact(path))
        except (KeyError, OSError, ValueError) as error:
            invalid.append({"artifact": path.name, "error": str(error)})
    report = summarize_artifacts(valid)
    report["schema_version"] = FEATURE_SCHEMA_VERSION
    report["invalid_artifact_count"] = len(invalid)
    report["invalid_artifacts"] = invalid
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if invalid:
        raise SystemExit("Quality report contains invalid artifacts.")


if __name__ == "__main__":
    main()
