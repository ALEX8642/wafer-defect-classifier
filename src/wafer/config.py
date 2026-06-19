"""
config.py — WaferConfig dataclass, YAML + CLI loading, device selection.

Usage:
    from wafer.config import WaferConfig, build_arg_parser

    parser = build_arg_parser("wafer train")
    args = parser.parse_args()
    cfg = WaferConfig.from_yaml_and_args(args.config, args)
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import re
from pathlib import Path
from typing import Optional

import yaml

# Invariant: repo root regardless of working directory or symlinks.
REPO_ROOT = Path(__file__).resolve().parents[2]

_VALID_DEVICE = re.compile(r"^(cpu|cuda(:\d+)?|mps)$")


def _resolve_device(hint: str) -> str:
    """
    Priority (highest to lowest):
      1. WAFER_DEVICE environment variable
      2. explicit non-"auto" value from YAML or CLI
      3. cuda if available, else cpu
    torch is imported only when "auto" resolution requires it.
    """
    raw = os.environ.get("WAFER_DEVICE") or (hint if hint != "auto" else None)
    if raw is None:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    if not _VALID_DEVICE.match(raw):
        raise ValueError(
            f"Invalid device {raw!r}. Expected cpu, cuda, cuda:N, or mps. "
            f"Check WAFER_DEVICE env var and --device flag."
        )
    return raw


def _anchor(p: Path) -> Path:
    """Make relative paths absolute relative to REPO_ROOT."""
    return p if p.is_absolute() else REPO_ROOT / p


@dataclasses.dataclass
class WaferConfig:
    # --- paths ---
    data_root: Path = REPO_ROOT / "data" / "raw"
    output_dir: Path = REPO_ROOT / "outputs"

    # --- hardware ---
    device: str = "auto"
    num_workers: int = 4

    # --- data ---
    batch_size: int = 128
    seed: int = 42
    input_size: int = 224          # square resize target (pixels)

    # --- training ---
    num_epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 7              # early-stopping: epochs without val macro-F1 gain

    # --- model (Phase 1) ---
    arch: str = "resnet18"         # resnet18 | resnet50
    pretrained: bool = False       # ImageNet weights transfer weakly to wafer maps

    # --- inference ---
    tta: bool = False              # test-time augmentation over the D4 symmetry group

    # --- loss (Phase C retraining) ---
    loss: str = "ce"               # ce | focal
    focal_gamma: float = 2.0       # focal loss concentration parameter

    def __post_init__(self) -> None:
        self.data_root = _anchor(Path(self.data_root))
        self.output_dir = _anchor(Path(self.output_dir))
        self.device = _resolve_device(self.device)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "WaferConfig":
        with open(yaml_path) as f:
            raw: dict = yaml.safe_load(f)
        return cls(**raw)

    @classmethod
    def from_yaml_and_args(
        cls,
        yaml_path: Path,
        args: Optional[argparse.Namespace] = None,
    ) -> "WaferConfig":
        """Load YAML then overlay non-None CLI args. Constructs cls exactly once."""
        with open(yaml_path) as f:
            merged: dict = yaml.safe_load(f)
        if args is not None:
            cli = vars(args)
            for field in dataclasses.fields(cls):
                val = cli.get(field.name)
                if val is not None:
                    merged[field.name] = val
        return cls(**merged)

    def to_dict(self) -> dict:
        """Serialisable snapshot (Path → str) for checkpoint metadata."""
        d = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            d[f.name] = str(v) if isinstance(v, Path) else v
        return d


def build_arg_parser(description: str = "wafer classifier") -> argparse.ArgumentParser:
    """
    Returns a parser mirroring every WaferConfig field.
    All args default to None so non-supplied flags don't shadow YAML values.
    """
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "baseline.yaml",
        help="Path to YAML config (default: configs/baseline.yaml)",
    )
    p.add_argument("--data-root", dest="data_root", type=Path, default=None)
    p.add_argument("--output-dir", dest="output_dir", type=Path, default=None)
    p.add_argument(
        "--device", type=str, default=None,
        help="cpu | cuda | cuda:N | mps | auto. Env WAFER_DEVICE overrides all.",
    )
    p.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    p.add_argument("--num-workers", dest="num_workers", type=int, default=None)
    p.add_argument("--num-epochs", dest="num_epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--input-size", dest="input_size", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--arch", type=str, default=None, help="resnet18 | resnet50")
    p.add_argument("--pretrained", dest="pretrained", action="store_true", default=None,
                   help="Use ImageNet pretrained weights (transfers weakly to wafer maps)")
    p.add_argument("--tta", action="store_true", default=None,
                   help="Enable test-time augmentation over the D4 symmetry group")
    p.add_argument("--loss", type=str, default=None, help="ce | focal")
    p.add_argument("--focal-gamma", dest="focal_gamma", type=float, default=None,
                   help="Focal loss γ parameter (default 2.0)")
    return p
