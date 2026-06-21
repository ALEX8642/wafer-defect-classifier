"""
pseudo_label.py — Semi-supervised pseudo-labeling over the 638k unlabeled WM-811K maps.

WM-811K contains ~638k wafer maps with no failureType label. These production maps
were never reviewed by a process engineer but come from the same fab and the same
distribution as the labeled set.

This script runs the trained model (with TTA) over all unlabeled maps and saves
high-confidence predictions as pseudo-labels. Only predictions where max-softmax
probability >= min_conf are kept — at 0.95 threshold, the expected error rate on
pseudo-labels is ~5%, well below the noise floor of a real fab label system.

Output: outputs/pseudo_labels.pkl — pickled DataFrame with columns:
    waferMap    original wafer map array (same format as LSWMD.pkl)
    label       predicted class name string
    confidence  max softmax probability after TTA (float)

Usage:
    python -m wafer.pseudo_label
    python -m wafer.pseudo_label --min-conf 0.95 --batch-size 256

Then enable in baseline.yaml and retrain:
    pseudo_label_path: outputs/pseudo_labels.pkl
    python -m wafer.train
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from wafer.config import WaferConfig, build_arg_parser
from wafer.data import _is_labeled, encode_map, resize_map, CLASS_NAMES
from wafer.evaluate import tta_predict
from wafer.model import build_model


# ---------------------------------------------------------------------------
# Dataset for unlabeled maps (tensors only, no labels)
# ---------------------------------------------------------------------------

class UnlabeledDataset(Dataset):
    def __init__(self, maps: list, input_size: int = 224) -> None:
        self.maps = maps
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.maps)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return resize_map(encode_map(self.maps[idx]), self.input_size)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def pseudo_label(cfg: WaferConfig, min_conf: float = 0.95) -> None:
    pkl_path = cfg.data_root / "LSWMD.pkl"
    output_path = cfg.output_dir / "pseudo_labels.pkl"
    ckpt_path = cfg.output_dir / "best.pt"

    print(f"Loading {pkl_path} ...")
    df_full = pd.read_pickle(pkl_path)

    unlabeled_mask = ~df_full["failureType"].apply(_is_labeled)
    df_unlabeled = df_full[unlabeled_mask].copy()
    print(f"Unlabeled maps: {len(df_unlabeled):,}  (labeled: {(~unlabeled_mask).sum():,})")

    # Load model
    ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
    class_to_idx: dict = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    class_names = [idx_to_class[i] for i in range(num_classes)]

    model = build_model(cfg, num_classes=num_classes).to(cfg.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Checkpoint : {ckpt_path}  (epoch {ckpt.get('epoch','?')}, "
          f"val F1 {ckpt.get('val_macro_f1', float('nan')):.4f})")

    # Calibration temperature
    temperature = 1.0
    temp_path = cfg.output_dir / "temperature.json"
    if temp_path.exists():
        with open(temp_path) as f:
            temperature = float(json.load(f)["temperature"])
        print(f"Temperature: T={temperature:.4f}")

    # Build DataLoader
    ds = UnlabeledDataset(df_unlabeled["waferMap"].tolist(), cfg.input_size)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device.startswith("cuda"),
    )

    # Inference with TTA
    print(f"\nRunning TTA inference on {len(ds):,} unlabeled maps "
          f"(min_conf={min_conf}) ...")
    all_probs: list[np.ndarray] = []
    for batch in tqdm(loader, desc="Pseudo-labeling"):
        probs = tta_predict(model, batch, cfg.device, temperature)
        all_probs.append(probs)

    probs_arr = np.vstack(all_probs)            # (N, num_classes)
    max_conf = probs_arr.max(axis=1)            # (N,)
    pred_idx = probs_arr.argmax(axis=1)         # (N,)

    # Filter by confidence threshold
    keep = max_conf >= min_conf
    print(f"\nKept {keep.sum():,} / {len(keep):,} maps "
          f"({100 * keep.mean():.1f}%) at min_conf={min_conf}")

    df_keep = df_unlabeled[keep].copy().reset_index(drop=True)
    df_keep["label"] = [class_names[i] for i in pred_idx[keep]]
    df_keep["confidence"] = max_conf[keep]

    # Per-class breakdown
    print("\nPseudo-label distribution:")
    counts = df_keep["label"].value_counts()
    for cls in CLASS_NAMES:
        n = counts.get(cls, 0)
        conf_mean = df_keep.loc[df_keep["label"] == cls, "confidence"].mean() if n > 0 else 0.0
        print(f"  {cls:<12} {n:>7,}  avg_conf={conf_mean:.3f}")

    # Save
    out_df = df_keep[["waferMap", "label", "confidence"]]
    out_df.to_pickle(output_path)
    print(f"\nSaved → {output_path}  ({len(out_df):,} rows, "
          f"{output_path.stat().st_size / 1e6:.0f} MB)")
    print("\nNext step: set pseudo_label_path in baseline.yaml and retrain.")


if __name__ == "__main__":
    parser = build_arg_parser("wafer pseudo_label")
    parser.add_argument(
        "--min-conf", dest="min_conf", type=float, default=0.95,
        help="Minimum TTA softmax confidence to accept a pseudo-label (default 0.95)",
    )
    args = parser.parse_args()
    cfg = WaferConfig.from_yaml_and_args(args.config, args)
    pseudo_label(cfg, min_conf=args.min_conf)
