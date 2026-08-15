"""Train a compact Transformer on KRSL training-sequence-v2."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from train_lstm_baseline import SequenceDataset, evaluate, read_rows

from krsl_ai.models.transformer import TransformerClassifier


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
    feature_size = datasets["train"][0][0].shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerClassifier(feature_size, len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
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
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                    "model_type": "transformer-v1",
                },
                args.report_dir / "best.pt",
            )

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
