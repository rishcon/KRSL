"""Train an LSTM baseline on fixed-length KRSL training sequences."""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from krsl_ai.features.training import expected_source_group
from krsl_ai.models.lstm import LstmClassifier
from krsl_ai.training.augmentation import augment_sequence
from krsl_ai.training.balancing import inverse_frequency_weights


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        rows: list[dict[str, str]],
        root: Path,
        labels: dict[str, int],
        augment: bool = False,
    ) -> None:
        self.rows, self.root, self.labels, self.augment = rows, root, labels, augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        with np.load(self.root / f"{row['sample_id']}.npz") as item:
            features = torch.from_numpy(item["features"].astype(np.float32))
            length_value = int(item["sequence_mask"].sum())
        if self.augment:
            features, length_value = augment_sequence(features, length_value)
        length = torch.tensor(length_value, dtype=torch.long)
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


def current_git_sha() -> str:
    """Return the code revision recorded with an experiment."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balanced-loss", action="store_true")
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = read_rows(args.manifest)
    labels = {label: index for index, label in enumerate(sorted({row["label"] for row in rows}))}
    split_rows = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    datasets = {
        split: SequenceDataset(
            split_rows[split],
            args.sequence_root,
            labels,
            augment=args.augment and split == "train",
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
    targets = [labels[row["label"]] for row in split_rows["train"]]
    class_weights = (
        inverse_frequency_weights(targets, len(labels)).to(device) if args.balanced_loss else None
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    experiment_config = {
        "model_type": "velocity-bilstm-v2" if args.augment else "bilstm-v1",
        "manifest": str(args.manifest),
        "sequence_root": str(args.sequence_root),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "balanced_loss": args.balanced_loss,
        "augmentation": args.augment,
    }
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
            torch.save(
                {
                    "model": model.state_dict(),
                    "labels": labels,
                    "feature_size": feature_size,
                    "config": experiment_config,
                    "git_sha": current_git_sha(),
                },
                args.report_dir / "best.pt",
            )
    checkpoint = torch.load(args.report_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    report = {
        "device": str(device),
        "feature_size": feature_size,
        "labels": labels,
        "git_sha": current_git_sha(),
        "config": experiment_config,
        "train_class_counts": {
            label: sum(row["label"] == label for row in split_rows["train"]) for label in labels
        },
        "class_weights": class_weights.cpu().tolist() if class_weights is not None else None,
        "history": history,
        "test": evaluate(model, loaders["test"], device, len(labels)),
    }
    (args.report_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["test"], indent=2))


if __name__ == "__main__":
    main()
