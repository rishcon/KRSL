"""Analyze per-class errors of a trained KRSL LSTM on the held-out test signer."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from krsl_ai.features.training import expected_source_group
from krsl_ai.models.lstm import LstmClassifier


class TestDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, rows: list[dict[str, str]], root: Path, labels: dict[str, int]) -> None:
        self.rows, self.root, self.labels = rows, root, labels

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        with np.load(self.root / f"{row['sample_id']}.npz") as item:
            features = torch.from_numpy(item["features"].astype(np.float32))
            length = torch.tensor(int(item["sequence_mask"].sum()), dtype=torch.long)
        return features, length, torch.tensor(self.labels[row["label"]], dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    labels = checkpoint["labels"]
    with args.manifest.open(newline="", encoding="utf-8") as file:
        rows = [
            row
            for row in csv.DictReader(file)
            if row["split"] == "test" and row["source_group"] == expected_source_group(row["label"])
        ]
    loader = DataLoader(TestDataset(rows, args.sequence_root, labels), batch_size=64)
    feature_size = checkpoint["model"]["lstm.weight_ih_l0"].shape[1]
    model = LstmClassifier(feature_size, len(labels))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    confusion = np.zeros((len(labels), len(labels)), dtype=int)
    with torch.no_grad():
        for features, lengths, targets in loader:
            predictions = model(features, lengths).argmax(dim=1)
            for target, prediction in zip(targets.tolist(), predictions.tolist(), strict=True):
                confusion[target, prediction] += 1

    names = [name for name, _ in sorted(labels.items(), key=lambda item: item[1])]
    per_class = []
    for index, name in enumerate(names):
        tp = int(confusion[index, index])
        support = int(confusion[index].sum())
        precision = tp / max(int(confusion[:, index].sum()), 1)
        recall = tp / max(support, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        alternatives = confusion[index].copy()
        alternatives[index] = 0
        confused_with = names[int(alternatives.argmax())] if alternatives.max() else None
        per_class.append(
            {
                "label": name,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "most_confused_with": confused_with,
                "confusion_count": int(alternatives.max()),
            }
        )
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "per_class_metrics.json").write_text(
        json.dumps(sorted(per_class, key=lambda item: item["f1"]), indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.report_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", *names])
        writer.writerows([[name, *confusion[index].tolist()] for index, name in enumerate(names)])
    for result in sorted(per_class, key=lambda item: item["f1"]):
        print(
            f"{result['label']}: f1={result['f1']:.3f}; "
            f"confused_with={result['most_confused_with']}",
        )


if __name__ == "__main__":
    main()
