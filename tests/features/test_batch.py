import csv
from pathlib import Path

from krsl_ai.features.batch import run_batch
from krsl_ai.features.holistic import HolisticFeatures


def test_batch_continues_after_failure(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["sample_id", "relative_path"])
        writer.writeheader()
        writer.writerows(
            [
                {"sample_id": "bad", "relative_path": "bad.mov"},
                {"sample_id": "good", "relative_path": "good.mov"},
            ]
        )

    def extractor(path: Path) -> HolisticFeatures:
        if path.name == "bad.mov":
            raise ValueError("unreadable")
        return None  # type: ignore[return-value]

    def save(features: HolisticFeatures, output: Path, source: Path) -> None:
        output.write_bytes(b"feature")

    summary = run_batch(
        manifest, tmp_path, tmp_path / "output", tmp_path / "failures.jsonl", extractor, save
    )

    assert summary.failed == 1
    assert summary.extracted == 1
    assert summary.attempted == 2
    assert '"sample_id": "bad"' in (tmp_path / "failures.jsonl").read_text()
