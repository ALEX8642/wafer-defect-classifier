"""
evaluate.py — Test-set evaluation: confusion matrix, per-class metrics, macro-F1.

Entry point: python -m wafer.evaluate [--config configs/baseline.yaml] [--checkpoint path]

Metrics note:
    Plain accuracy is misleading on WM-811K — predicting "none" for everything
    scores ~85 % without learning any defect pattern. This script reports
    macro-F1 (equal weight per class regardless of frequency) and balanced
    accuracy as the honest headline numbers.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm

from wafer.config import WaferConfig, build_arg_parser
from wafer.data import get_dataloaders, CLASS_NAMES
from wafer.model import build_model


def evaluate(cfg: WaferConfig, checkpoint_path: Path | None = None) -> None:
    if checkpoint_path is None:
        checkpoint_path = cfg.output_dir / "best.pt"

    ckpt = torch.load(checkpoint_path, map_location=cfg.device, weights_only=False)
    class_to_idx: dict = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes  = len(class_to_idx)
    class_names  = [idx_to_class[i] for i in range(num_classes)]

    model = build_model(cfg, num_classes=num_classes).to(cfg.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Checkpoint : {checkpoint_path}  (epoch {ckpt.get('epoch', '?')}, "
          f"val macro-F1 {ckpt.get('val_macro_f1', float('nan')):.4f})")

    _, _, test_loader, _, _ = get_dataloaders(cfg)

    all_preds: list = []
    all_targets: list = []
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            logits = model(inputs.to(cfg.device, non_blocking=True))
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_targets.extend(targets.numpy())

    preds   = np.array(all_preds)
    targets = np.array(all_targets)

    macro_f1 = float(
        classification_report(targets, preds, target_names=class_names,
                               zero_division=0, output_dict=True)["macro avg"]["f1-score"]
    )
    bal_acc = balanced_accuracy_score(targets, preds)

    print("\n" + "=" * 64)
    print("TEST SET RESULTS")
    print("=" * 64)
    print(f"  Macro-F1          : {macro_f1:.4f}  ← headline metric")
    print(f"  Balanced accuracy : {bal_acc:.4f}")
    print("  (Plain accuracy suppressed — misleading under 85 % class imbalance.)\n")
    print(classification_report(targets, preds, target_names=class_names, zero_division=0))

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    _save_csv(targets, preds, class_names, cfg.output_dir)
    _save_confusion_matrix(targets, preds, class_names, cfg.output_dir)


def _save_csv(targets, preds, class_names, output_dir: Path) -> None:
    report = classification_report(
        targets, preds, target_names=class_names, zero_division=0, output_dict=True
    )
    csv_path = output_dir / "per_class_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1_score", "support"])
        for cls in class_names:
            r = report[cls]
            w.writerow([cls,
                         f"{r['precision']:.4f}",
                         f"{r['recall']:.4f}",
                         f"{r['f1-score']:.4f}",
                         int(r["support"])])
        r = report["macro avg"]
        w.writerow(["macro_avg",
                     f"{r['precision']:.4f}",
                     f"{r['recall']:.4f}",
                     f"{r['f1-score']:.4f}",
                     ""])
    print(f"Per-class CSV : {csv_path}")


def _save_confusion_matrix(targets, preds, class_names, output_dir: Path) -> None:
    cm      = confusion_matrix(targets, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, data, title in zip(
        axes,
        [cm,      cm_norm],
        ["Raw counts", "Row-normalised (recall per class)"],
    ):
        im = ax.imshow(data, interpolation="nearest", cmap="Blues")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ticks = range(len(class_names))
        ax.set_xticks(ticks); ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(ticks); ax.set_yticklabels(class_names, fontsize=8)
        plt.colorbar(im, ax=ax)
        thresh = data.max() / 2.0
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                cell = f"{data[i, j]:.2f}" if data.dtype == float else str(data[i, j])
                ax.text(j, i, cell, ha="center", va="center", fontsize=6,
                        color="white" if data[i, j] > thresh else "black")

    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix : {cm_path}")


if __name__ == "__main__":
    parser = build_arg_parser("wafer evaluate")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to .pt checkpoint (default: outputs/best.pt)")
    args = parser.parse_args()
    cfg  = WaferConfig.from_yaml_and_args(args.config, args)
    evaluate(cfg, checkpoint_path=args.checkpoint)
