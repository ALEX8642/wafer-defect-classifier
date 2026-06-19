"""
download_data.py — Verify WM-811K LSWMD.pkl and optionally run a GPU smoke test.

WM-811K is distributed via Kaggle (manual download required — not auto-fetched).
Place LSWMD.pkl in data/raw/ then run:

    python scripts/download_data.py
    python scripts/download_data.py --check-gpu

After download, record the source URL and SHA256 in docs/DATA_SOURCE.md.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_MIN_SIZE_BYTES = 2_500_000_000  # warn if pkl is smaller than ~2.5 GB


def _unwrap_label(val) -> str:
    """Flatten any numpy array nesting and return the scalar label string."""
    arr = np.asarray(val).ravel()
    if arr.size == 0:
        return ""
    return str(arr[0]).strip()


def _is_labeled(val) -> bool:
    """Return True if val represents a non-empty, non-NaN failure-type label."""
    if val is None:
        return False
    try:
        if isinstance(val, float) and math.isnan(val):
            return False
    except TypeError:
        pass
    lbl = _unwrap_label(val)
    return lbl != "" and lbl.lower() != "nan"


def verify_dataset(data_root: Path) -> None:
    data_root.mkdir(parents=True, exist_ok=True)

    pkl_path = data_root / "LSWMD.pkl"
    if not pkl_path.exists():
        print(
            f"\nERROR: {pkl_path} not found.\n"
            "Download WM-811K from Kaggle (search 'WM-811K wafer map dataset')\n"
            "and place LSWMD.pkl in data/raw/. Then record the source URL in\n"
            "docs/DATA_SOURCE.md.",
            file=sys.stderr,
        )
        sys.exit(1)

    size = pkl_path.stat().st_size
    size_gb = size / 1e9
    if size < _MIN_SIZE_BYTES:
        print(
            f"WARNING: {pkl_path.name} is {size_gb:.2f} GB — smaller than expected (~3 GB).\n"
            "The download may be truncated or incomplete.",
            file=sys.stderr,
        )
    else:
        print(f"File: {pkl_path.name}  ({size_gb:.2f} GB)")

    print("Loading LSWMD.pkl (may take ~30 s on first access)...")
    df = pd.read_pickle(pkl_path)

    total = len(df)
    print(f"\nTotal rows:   {total:,}")

    # Inspect a few cells so users can verify the unwrap logic on their actual file
    if "failureType" in df.columns:
        sample = df["failureType"].dropna().head(3).tolist()
        print(f"Sample failureType cells (raw): {sample}")

        labeled_mask = df["failureType"].apply(_is_labeled)
        df_labeled = df[labeled_mask].copy()
        df_labeled["label"] = df_labeled["failureType"].apply(_unwrap_label)

        n_labeled = len(df_labeled)
        print(f"Labeled rows: {n_labeled:,}")
        print(f"\nClass distribution (failureType):")
        dist = df_labeled["label"].value_counts()
        print(dist.to_string())
        print(f"\nTotal classes: {dist.nunique()}")
    else:
        print("WARNING: 'failureType' column not found. Check the pickle structure.")
        print(f"Columns: {list(df.columns)}")


def check_gpu() -> None:
    import torch

    print(f"\ntorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("No CUDA device found. Install cu128 PyTorch wheel:")
        print("  pip install torch>=2.7.1 --extra-index-url https://download.pytorch.org/whl/cu128")
        return

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"Device: {name}")
    print(f"Compute capability: {cap[0]}.{cap[1]}")
    if cap == (12, 0):
        print("(5090 / Blackwell sm_120 detected)")
    elif cap == (8, 9):
        print("(4090 / Ada sm_89 detected)")

    # Kernel smoke test — catches "no kernel image" errors on sm_120 with wrong builds
    a = torch.randn(256, 256, device="cuda")
    b = torch.randn(256, 256, device="cuda")
    c = a @ b
    print(f"Matmul smoke test: {c.shape} — kernel OK")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Path to data/raw/. Defaults to data_root from configs/baseline.yaml.",
    )
    p.add_argument(
        "--check-gpu",
        action="store_true",
        help="Run CUDA smoke test (tensor.cuda() + matmul).",
    )
    args = p.parse_args()

    if args.data_root is not None:
        data_root = args.data_root
    else:
        # Lazy import: works without pip install -e . if PYTHONPATH includes src/
        try:
            from wafer.config import WaferConfig, REPO_ROOT
            cfg = WaferConfig.from_yaml(REPO_ROOT / "configs" / "baseline.yaml")
            data_root = cfg.data_root
        except ImportError:
            # Fallback: resolve relative to this script's repo root
            data_root = Path(__file__).resolve().parents[1] / "data" / "raw"

    verify_dataset(data_root)

    if args.check_gpu:
        check_gpu()
