"""Create a deterministic manifest for raw isolated-sign videos."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

VIDEO_NAME_PATTERN = re.compile(
    r"^(?P<signer_id>P\d+)_(?P<label>.+?)\s*\((?P<take_id>\d+)\)\.mov$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Sample:
    """One immutable raw-video record."""

    sample_id: str
    signer_id: str
    label: str
    take_id: int
    relative_path: str
    source_group: str
    file_size_bytes: int
    split: str


def parse_video_name(filename: str) -> tuple[str, str, int] | None:
    """Return signer, label, and take ID for a supported raw-video filename."""
    match = VIDEO_NAME_PATTERN.fullmatch(filename)
    if match is None:
        return None
    return (
        match.group("signer_id").upper(),
        match.group("label").strip(),
        int(match.group("take_id")),
    )


def build_samples(
    video_root: Path,
    split_by_signer: dict[str, str],
) -> tuple[list[Sample], list[str]]:
    """Inventory videos without mutating source data."""
    samples: list[Sample] = []
    unmatched: list[str] = []

    for path in sorted(video_root.rglob("*.mov"), key=lambda item: item.as_posix().lower()):
        parsed = parse_video_name(path.name)
        relative_path = path.relative_to(video_root).as_posix()
        if parsed is None:
            unmatched.append(relative_path)
            continue

        signer_id, label, take_id = parsed
        split = split_by_signer.get(signer_id)
        if split is None:
            raise ValueError(f"No split assigned to signer {signer_id!r} ({relative_path}).")

        path_digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:12]
        sample_id = f"{signer_id}_{label}_{take_id:04d}_{path_digest}"
        samples.append(
            Sample(
                sample_id=sample_id,
                signer_id=signer_id,
                label=label,
                take_id=take_id,
                relative_path=relative_path,
                source_group=path.parent.name,
                file_size_bytes=path.stat().st_size,
                split=split,
            )
        )

    return samples, unmatched


def validate_signer_split(samples: list[Sample]) -> None:
    """Ensure split assignment is signer-independent and label-complete."""
    signers_by_split: dict[str, set[str]] = {}
    labels_by_split: dict[str, set[str]] = {}
    all_labels = {sample.label for sample in samples}

    for sample in samples:
        signers_by_split.setdefault(sample.split, set()).add(sample.signer_id)
        labels_by_split.setdefault(sample.split, set()).add(sample.label)

    if set(labels_by_split) != {"train", "validation", "test"}:
        raise ValueError("Expected exactly train, validation, and test splits.")

    seen_signers: set[str] = set()
    for split, signers in signers_by_split.items():
        overlap = seen_signers & signers
        if overlap:
            raise ValueError(f"Signers assigned to multiple splits: {sorted(overlap)}")
        seen_signers.update(signers)

    for split, labels in labels_by_split.items():
        missing_labels = sorted(all_labels - labels)
        if missing_labels:
            raise ValueError(f"{split} split is missing labels: {', '.join(missing_labels)}")


def write_manifest(samples: list[Sample], output_path: Path) -> None:
    """Write a stable CSV manifest."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(Sample.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(sample) for sample in samples)


def write_report(
    samples: list[Sample],
    unmatched: list[str],
    keypoint_root: Path | None,
    output_path: Path,
) -> None:
    """Write a concise machine-readable inventory report."""
    keypoint_file_count = (
        sum(1 for _ in keypoint_root.rglob("*.json")) if keypoint_root is not None else None
    )
    report = {
        "schema_version": 1,
        "sample_count": len(samples),
        "keypoint_file_count": keypoint_file_count,
        "class_counts": dict(sorted(Counter(sample.label for sample in samples).items())),
        "signer_counts": dict(sorted(Counter(sample.signer_id for sample in samples).items())),
        "split_counts": dict(sorted(Counter(sample.split for sample in samples).items())),
        "source_group_counts": dict(
            sorted(Counter(sample.source_group for sample in samples).items())
        ),
        "labels_by_source_group": {
            source_group: dict(sorted(label_counts.items()))
            for source_group, label_counts in sorted(
                (
                    source_group,
                    Counter(
                        sample.label for sample in samples if sample.source_group == source_group
                    ),
                )
                for source_group in {sample.source_group for sample in samples}
            )
        },
        "logical_sample_collisions": sorted(
            logical_id
            for logical_id, count in Counter(
                f"{sample.signer_id}_{sample.label}_{sample.take_id:04d}" for sample in samples
            ).items()
            if count > 1
        ),
        "unmatched_video_count": len(unmatched),
        "unmatched_videos": unmatched,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
