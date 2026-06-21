"""
data.py — WM-811K dataset loading, splitting, and DataLoader construction.

Encoding choice (one-hot, 3 channels):
    Pixel values {0, 1, 2} carry distinct physical meanings:
        0 = outside the wafer boundary (no die)
        1 = passing die
        2 = failing die
    Treating these as a continuous scalar (0 → 0.0, 1 → 0.5, 2 → 1.0) implies
    a metric relationship that doesn't exist — "fail" is not twice "pass".
    One-hot encoding into 3 binary channels preserves the discrete semantics and
    keeps the input compatible with a 3-channel ResNet first conv.

Imbalance strategy (class-weighted cross-entropy):
    "none" is ~85 % of labeled rows; Near-full has only 149 samples.
    A model trained with uniform loss quickly learns to predict "none" everywhere
    (>85 % accuracy, 0 recall on rare classes — a degenerate solution).
    Class-weighted CE penalises misclassifying rare classes proportionally to
    their inverse frequency.  This is the escape-risk / false-alarm trade-off
    that quality engineers care about: we pay more loss for defects that escape
    than for over-flagging non-defective wafers.
    Alternatives: focal loss (γ≈2) or WeightedRandomSampler — noted but not used
    in Phase 1 for simplicity.
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from wafer.config import WaferConfig

# Canonical 9-class ordering (alphabetical; 'none' last for readability in reports)
CLASS_NAMES: list[str] = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Near-full", "Random", "Scratch", "none",
]
CLASS_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}


# ---------------------------------------------------------------------------
# Label helpers (shared with scripts/download_data.py)
# ---------------------------------------------------------------------------

def _unwrap_label(val) -> str:
    """Flatten any numpy array nesting and return the scalar label string."""
    arr = np.asarray(val).ravel()
    if arr.size == 0:
        return ""
    return str(arr[0]).strip()


def _is_labeled(val) -> bool:
    """True when val is a non-empty, non-NaN failureType entry."""
    if val is None:
        return False
    try:
        if isinstance(val, float) and math.isnan(val):
            return False
    except TypeError:
        pass
    lbl = _unwrap_label(val)
    return lbl != "" and lbl.lower() != "nan"


# ---------------------------------------------------------------------------
# Map encoding and resizing
# ---------------------------------------------------------------------------

def encode_map(wmap: np.ndarray) -> torch.Tensor:
    """
    Convert a 2D wafer map (values 0/1/2) to a (3, H, W) one-hot float tensor.
    Values outside [0, 2] are clipped before encoding.
    """
    arr = np.clip(np.asarray(wmap, dtype=np.int64), 0, 2)
    t = torch.tensor(arr, dtype=torch.long)
    return F.one_hot(t, num_classes=3).permute(2, 0, 1).float()  # (3, H, W)


def resize_map(tensor: torch.Tensor, size: int) -> torch.Tensor:
    """
    Resize a (3, H, W) one-hot tensor to (3, size, size).
    Nearest-neighbour keeps pixel values binary (no interpolation artefacts).
    """
    return F.interpolate(
        tensor.unsqueeze(0), size=(size, size), mode="nearest"
    ).squeeze(0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WaferDataset(Dataset):
    """
    PyTorch Dataset wrapping the labeled WM-811K wafer maps.

    Args:
        df: DataFrame with 'waferMap' and 'label' columns.
        class_to_idx: Mapping from class name string to integer id.
        input_size: Side length of the square CNN input (pixels).
        augment: Whether to apply training-time augmentation.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        class_to_idx: Dict[str, int],
        input_size: int = 224,
        augment: bool = False,
    ) -> None:
        self.maps = df["waferMap"].tolist()
        self.labels = [class_to_idx[lbl] for lbl in df["label"]]
        self.input_size = input_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        tensor = encode_map(self.maps[idx])
        tensor = resize_map(tensor, self.input_size)
        if self.augment:
            # Wafer has approximate rotational symmetry → 90-degree rotations are valid.
            k = random.randint(0, 3)
            if k:
                tensor = torch.rot90(tensor, k, dims=[1, 2])
            if random.random() < 0.5:
                tensor = torch.flip(tensor, dims=[2])  # horizontal flip
            if random.random() < 0.5:
                tensor = torch.flip(tensor, dims=[1])  # vertical flip
            # No crops: aggressive cropping removes edge-ring / edge-loc patterns.
        return tensor, self.labels[idx]


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------

def load_and_split(cfg: WaferConfig) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    Dict[str, int], torch.Tensor
]:
    """
    Load LSWMD.pkl and return (train_df, val_df, test_df, class_to_idx, class_weights).

    Split strategy:
      - If trainTestLabel populates a meaningful test set (> 1 % of labeled rows),
        use it as-is and carve the val set from the remaining pool.
      - Otherwise, fall back to a reproducible stratified 70 / 10 / 20 split.
    In both cases val is always a fresh stratified split from the training pool so
    the model never sees test labels during training.
    """
    pkl_path = cfg.data_root / "LSWMD.pkl"
    df = pd.read_pickle(pkl_path)

    # 1. Filter to labeled rows and unwrap labels
    labeled_mask = df["failureType"].apply(_is_labeled)
    df = df[labeled_mask].copy()
    df["label"] = df["failureType"].apply(_unwrap_label)

    # 2. Attempt to use the dataset-provided train/test split
    df_test = None
    if "trainTestLabel" in df.columns:
        df["_ttl"] = df["trainTestLabel"].apply(
            lambda v: _unwrap_label(v) if _is_labeled(v) else ""
        )
        test_mask = df["_ttl"].str.lower().str.strip() == "test"
        if test_mask.sum() > 0.01 * len(df):
            df_test = df[test_mask].copy()
            df = df[~test_mask].copy()

    if df_test is None:
        # Stratified 80 / 20 train-pool / test
        df, df_test = train_test_split(
            df, test_size=0.20, stratify=df["label"], random_state=cfg.seed
        )

    # 3. Carve val from train pool: 0.125 × 0.80 ≈ 10 % of total
    df_train, df_val = train_test_split(
        df, test_size=0.125, stratify=df["label"], random_state=cfg.seed
    )

    # 4. Optionally augment training split with pseudo-labeled unlabeled maps.
    #    Pseudo-labels are appended to the training split only — val and test are
    #    never contaminated, so evaluation remains unbiased.
    pseudo_path = getattr(cfg, "pseudo_label_path", "") or ""
    if pseudo_path:
        from pathlib import Path as _Path
        pl_path = _Path(pseudo_path) if _Path(pseudo_path).is_absolute() else \
                  (_Path(__file__).resolve().parents[2] / pseudo_path)
        if pl_path.exists():
            df_pseudo = pd.read_pickle(pl_path)[["waferMap", "label"]]
            df_pseudo = df_pseudo[df_pseudo["label"] != "none"]  # none already dominant; keep defect pseudo-labels only
            n_before = len(df_train)
            df_train = pd.concat([df_train, df_pseudo], ignore_index=True)
            print(f"Pseudo-labels: +{len(df_pseudo):,} rows appended to train "
                  f"({n_before:,} → {len(df_train):,})")
        else:
            print(f"Warning: pseudo_label_path={pl_path!r} not found — skipping.")

    # 5. Class weights for weighted CE (inverse class frequency on labeled train rows)
    labeled_train_y = df_train["label"].map(CLASS_TO_IDX).dropna().astype(int).values
    raw_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(CLASS_NAMES)),
        y=labeled_train_y,
    )
    class_weights = torch.tensor(raw_weights, dtype=torch.float32)

    print(
        f"Split sizes — train: {len(df_train):,}  val: {len(df_val):,}  "
        f"test: {len(df_test):,}"
    )

    return df_train, df_val, df_test, CLASS_TO_IDX, class_weights


def get_dataloaders(cfg: WaferConfig) -> Tuple[
    DataLoader, DataLoader, DataLoader, torch.Tensor, Dict[str, int]
]:
    """Return (train_loader, val_loader, test_loader, class_weights, class_to_idx)."""
    df_train, df_val, df_test, class_to_idx, class_weights = load_and_split(cfg)

    train_ds = WaferDataset(df_train, class_to_idx, cfg.input_size, augment=True)
    val_ds   = WaferDataset(df_val,   class_to_idx, cfg.input_size, augment=False)
    test_ds  = WaferDataset(df_test,  class_to_idx, cfg.input_size, augment=False)

    loader_kw = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device.startswith("cuda"),
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kw)

    return train_loader, val_loader, test_loader, class_weights, class_to_idx
