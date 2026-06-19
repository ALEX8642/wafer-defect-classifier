"""
evaluate.py — Test-set evaluation: confusion matrix, per-class metrics, macro-F1.

Entry point: python -m wafer.evaluate [--config configs/baseline.yaml] [--checkpoint path]

Metrics note:
    Plain accuracy is misleading on WM-811K — predicting "none" for everything
    scores ~85 % without learning any defect pattern. This script reports
    macro-F1 (equal weight per class regardless of frequency) and balanced
    accuracy as the honest headline numbers.

Improvement options (configured via baseline.yaml or CLI overrides):
    --tta           Average predictions over the 8-element D4 symmetry group.
                    Exploits the rotational/flip symmetry of wafer maps; no retraining.
    thresholds.json Per-class confidence thresholds tuned on the val set (written by
                    calibrate.py). Applied automatically if the file exists.
"""
from __future__ import annotations

import csv
import json
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


# ---------------------------------------------------------------------------
# Test-time augmentation over the D4 symmetry group
# ---------------------------------------------------------------------------

def tta_predict(
    model: torch.nn.Module,
    tensor_batch: torch.Tensor,
    device: str,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Average calibrated softmax probabilities over the 8 elements of D4.

    The dihedral group D4 describes the symmetries of a square: 4 rotations
    (0°, 90°, 180°, 270°) and a horizontal flip of each.  Wafer maps have
    D4 symmetry — a rotated or mirrored defect pattern is physically the same
    defect.  Averaging the model's output across all 8 views reduces prediction
    variance without any retraining.

    Args:
        model: Trained ResNet-18 in eval mode.
        tensor_batch: (B, 3, H, W) one-hot encoded wafer maps.
        device: "cuda" or "cpu".
        temperature: Calibration temperature T; divides logits before softmax.

    Returns:
        Averaged probability matrix, shape (B, num_classes).
    """
    model.eval()
    all_probs = []
    for k in range(4):
        for flip in (False, True):
            x = tensor_batch.clone()
            if flip:
                x = x.flip(3)               # horizontal flip along width axis
            x = torch.rot90(x, k, [2, 3])   # rotate k×90° in the H-W plane
            with torch.no_grad():
                logits = model(x.to(device, non_blocking=True))
            probs = torch.softmax(logits / max(temperature, 0.05), dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.mean(all_probs, axis=0)  # (B, num_classes)


# ---------------------------------------------------------------------------
# Per-class confidence threshold application
# ---------------------------------------------------------------------------

def apply_thresholds(
    probs: np.ndarray,
    thresholds: dict[str, float],
    class_names: list[str],
) -> np.ndarray:
    """
    Apply per-class confidence thresholds to a probability matrix.

    Decision rule for each sample:
      1. Start with the highest-probability class c.
      2. If P(c) >= thresholds[c], predict c.
      3. Otherwise mask c out and try the next-highest class.
      4. If no class meets its threshold, fall back to plain argmax.

    This "threshold cascade" is what makes Scratch precision tunable:
    raising τ_Scratch from 0.50 to (e.g.) 0.72 means the model only flags
    a wafer as Scratch when it is 72 % confident, reducing false alarms.
    """
    preds = np.empty(len(probs), dtype=int)
    for i in range(len(probs)):
        p = probs[i].copy()
        for _ in range(len(class_names)):
            c = int(p.argmax())
            tau = thresholds.get(class_names[c], 0.0)
            if p[c] >= tau:
                preds[i] = c
                break
            p[c] = -np.inf          # mask this class and try the next
        else:
            preds[i] = int(probs[i].argmax())   # fallback: no class met threshold
    return preds


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

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

    # Load calibration temperature (written by calibrate.py)
    temperature = 1.0
    temp_path = cfg.output_dir / "temperature.json"
    if temp_path.exists():
        with open(temp_path) as f:
            temperature = float(json.load(f)["temperature"])
        print(f"Temperature: T={temperature:.4f}  (calibrated)")
    else:
        print("Temperature: T=1.0  (uncalibrated — run python -m wafer.calibrate first)")

    # Load per-class thresholds (written by calibrate.py tune_thresholds step)
    thresholds: dict[str, float] = {}
    thresh_path = cfg.output_dir / "thresholds.json"
    if thresh_path.exists():
        with open(thresh_path) as f:
            thresholds = json.load(f)
        print(f"Thresholds : {len(thresholds)} class thresholds loaded")

    _, _, test_loader, _, _ = get_dataloaders(cfg)

    mode_tag = "TTA×8" if cfg.tta else "standard"
    all_probs: list[np.ndarray] = []
    all_targets: list[int] = []

    for inputs, targets in tqdm(test_loader, desc=f"Evaluating [{mode_tag}]"):
        if cfg.tta:
            probs = tta_predict(model, inputs, cfg.device, temperature)
        else:
            with torch.no_grad():
                logits = model(inputs.to(cfg.device, non_blocking=True))
            probs = torch.softmax(logits / max(temperature, 0.05), dim=1).cpu().numpy()
        all_probs.append(probs)
        all_targets.extend(targets.numpy())

    probs_arr = np.vstack(all_probs)
    targets   = np.array(all_targets)
    preds     = (
        apply_thresholds(probs_arr, thresholds, class_names)
        if thresholds
        else probs_arr.argmax(axis=1)
    )

    macro_f1 = float(
        classification_report(targets, preds, target_names=class_names,
                               zero_division=0, output_dict=True)["macro avg"]["f1-score"]
    )
    bal_acc = balanced_accuracy_score(targets, preds)

    extras = []
    if cfg.tta:
        extras.append("TTA×8")
    if thresholds:
        extras.append("per-class τ")
    header = "TEST SET RESULTS" + (f"  [{', '.join(extras)}]" if extras else "")

    print("\n" + "=" * 64)
    print(header)
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
