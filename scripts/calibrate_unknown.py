"""Calibrate an UNKNOWN confidence threshold on validation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from train_lstm_baseline import SequenceDataset, read_rows

from krsl_ai.models.lstm import LstmClassifier


def predictions(model: LstmClassifier, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    confidence, correct = [], []
    model.eval()
    with torch.no_grad():
        for features, lengths, targets in loader:
            scores = torch.softmax(model(features, lengths), dim=1)
            maximum, predicted = scores.max(dim=1)
            confidence.append(maximum)
            correct.append(predicted.eq(targets))
    return torch.cat(confidence), torch.cat(correct)


def metrics(confidence: torch.Tensor, correct: torch.Tensor, threshold: float) -> dict[str, float]:
    accepted = confidence >= threshold
    return {
        "threshold": threshold,
        "coverage": float(accepted.float().mean()),
        "accepted_accuracy": float(correct[accepted].float().mean()) if accepted.any() else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-coverage", type=float, default=0.6)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    labels = checkpoint["labels"]
    model = LstmClassifier(checkpoint["model"]["lstm.weight_ih_l0"].shape[1], len(labels))
    model.load_state_dict(checkpoint["model"])
    rows = read_rows(args.manifest)
    output = {}
    for split in ("validation", "test"):
        data = SequenceDataset(
            [row for row in rows if row["split"] == split], args.sequence_root, labels
        )
        output[split] = predictions(model, DataLoader(data, batch_size=64))
    choices = [metrics(*output["validation"], threshold / 100) for threshold in range(100)]
    chosen = max(
        (item for item in choices if item["coverage"] >= args.min_coverage),
        key=lambda item: item["accepted_accuracy"],
    )
    report = {
        "selection_split": "validation",
        "min_coverage": args.min_coverage,
        "selected": chosen,
        "test": metrics(*output["test"], chosen["threshold"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
