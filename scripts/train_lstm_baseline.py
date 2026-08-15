"""Train an LSTM baseline on fixed-length KRSL training sequences."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from krsl_ai.features.training import expected_source_group
from krsl_ai.models.lstm import LstmClassifier


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
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


def read_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(newline="", encoding="utf-8") as file:
        return [
            row
            for row in csv.DictReader(file)
            if row["source_group"] == expected_source_group(row["label"])
        ]


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, class_count: int
) -> dict[str, float]:
    model.eval()
    correct = total = 0
    confusion = np.zeros((class_count, class_count), dtype=int)
    with torch.no_grad():
        for features, lengths, targets in loader:
            predictions = model(features.to(device), lengths.to(device)).argmax(dim=1).cpu()
            correct += int((predictions == targets).sum())
            total += len(targets)
            for target, prediction in zip(targets.tolist(), predictions.tolist(), strict=True):
                confusion[target, prediction] += 1
    f1 = []
    for index in range(class_count):
        tp = confusion[index, index]
        precision = tp / max(confusion[:, index].sum(), 1)
        recall = tp / max(confusion[index, :].sum(), 1)
        f1.append(2 * precision * recall / max(precision + recall, 1e-12))
    return {"accuracy": correct / total, "macro_f1": float(np.mean(f1))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = read_rows(args.manifest)
    labels = {label: index for index, label in enumerate(sorted({row["label"] for row in rows}))}
    datasets = {
        split: SequenceDataset(
            [row for row in rows if row["split"] == split], args.sequence_root, labels
        )
        for split in ("train", "validation", "test")
    }
    loaders = {
        split: DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train")
        for split, dataset in datasets.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_size = datasets["train"][0][0].shape[1]
    model = LstmClassifier(feature_size, len(labels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for features, lengths, targets in loaders["train"]:
            optimizer.zero_grad()
            loss = loss_fn(model(features.to(device), lengths.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()
        validation = evaluate(model, loaders["validation"], device, len(labels))
        history.append({"epoch": epoch, **validation})
        print(
            f"epoch={epoch}/{args.epochs} validation_accuracy="
            f"{validation['accuracy']:.3f} macro_f1={validation['macro_f1']:.3f}",
            flush=True,
        )
        if validation["macro_f1"] > best_f1:
            best_f1 = validation["macro_f1"]
            torch.save({"model": model.state_dict(), "labels": labels}, args.report_dir / "best.pt")
    checkpoint = torch.load(args.report_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    report = {
        "device": str(device),
        "feature_size": feature_size,
        "labels": labels,
        "history": history,
        "test": evaluate(model, loaders["test"], device, len(labels)),
    }
    (args.report_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["test"], indent=2))


if __name__ == "__main__":
    main()
