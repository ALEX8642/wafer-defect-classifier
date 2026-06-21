"""
train.py — Training loop: AdamW, cosine LR, early stopping on val macro-F1.

Entry point: python -m wafer.train [--config configs/baseline.yaml] [overrides...]
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from wafer.config import WaferConfig, build_arg_parser
from wafer.data import get_dataloaders
from wafer.model import build_model


class FocalLoss(nn.Module):
    """
    Focal loss (Lin et al., 2017 — originally for object detection, effective for
    class-imbalanced classification).

    Focal loss adds a modulating factor (1 - p_t)^gamma to unweighted CE that
    down-weights easy examples (high p_t) and focuses learning on hard ones (low p_t).

    Intentionally does NOT combine with class_weights: focal's modulating factor
    already suppresses the dominant "none" class (easy examples → high p_t → low
    focal weight). Adding class_weights on top of focal double-penalizes rare
    classes, destabilizing early training.

    gamma=0 recovers standard (unweighted) cross-entropy.
    gamma=2 is the default from the original paper.
    """

    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)                          # probability of the correct class
        return ((1.0 - pt) ** self.gamma * ce).mean()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    device_type: str,
) -> tuple[float, float]:
    """Run one train or eval epoch. Returns (avg_loss, macro_f1)."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    all_preds: list = []
    all_targets: list = []

    ctx = torch.enable_grad if training else torch.no_grad
    with ctx():
        for inputs, targets in tqdm(loader, leave=False, desc="train" if training else "val"):
            inputs  = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device_type, enabled=(device_type == "cuda")):
                logits = model(inputs)
                loss   = criterion(logits, targets)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * inputs.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1


def train(cfg: WaferConfig) -> None:
    set_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.device
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    print(f"Device: {device}  |  arch: {cfg.arch}  |  pretrained: {cfg.pretrained}")
    print("Loading data...")
    train_loader, val_loader, _, class_weights, class_to_idx = get_dataloaders(cfg)
    print(f"  Classes: {list(class_to_idx.keys())}")

    # Persist class mapping alongside the checkpoint for portability
    class_map_path = cfg.output_dir / "class_map.json"
    with open(class_map_path, "w") as f:
        json.dump(class_to_idx, f, indent=2)

    model = build_model(cfg).to(device)
    if getattr(cfg, "backbone_ckpt_path", ""):
        bb_ckpt = torch.load(cfg.backbone_ckpt_path, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(bb_ckpt["backbone_state_dict"], strict=False)
        n_loaded = len(bb_ckpt["backbone_state_dict"]) - len(missing)
        print(f"Pretrained backbone: loaded {n_loaded} tensors from {cfg.backbone_ckpt_path}"
              f"  (missing={len(missing)}, unexpected={len(unexpected)})")

    if cfg.loss == "focal":
        criterion = FocalLoss(gamma=cfg.focal_gamma)
        print(f"Loss: FocalLoss (γ={cfg.focal_gamma}, no class weights — focal handles imbalance)")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print("Loss: CrossEntropyLoss (class-weighted)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    scaler    = torch.amp.GradScaler(enabled=(device_type == "cuda"))

    best_val_f1     = 0.0
    patience_count  = 0
    ckpt_path       = cfg.output_dir / "best.pt"

    for epoch in range(1, cfg.num_epochs + 1):
        tr_loss, tr_f1 = _epoch(
            model, train_loader, criterion, device, optimizer, scaler, device_type
        )
        va_loss, va_f1 = _epoch(
            model, val_loader, criterion, device, None, scaler, device_type
        )
        scheduler.step()

        improved = va_f1 > best_val_f1
        marker   = " *" if improved else ""
        print(
            f"Epoch {epoch:3d}/{cfg.num_epochs}  "
            f"train loss {tr_loss:.4f} f1 {tr_f1:.4f}  |  "
            f"val loss {va_loss:.4f} f1 {va_f1:.4f}{marker}"
        )

        if improved:
            best_val_f1    = va_f1
            patience_count = 0
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "val_macro_f1":     best_val_f1,
                    "class_to_idx":     class_to_idx,
                    "cfg":              cfg.to_dict(),
                },
                ckpt_path,
            )
        else:
            patience_count += 1
            if patience_count >= cfg.patience:
                print(f"Early stop: no val macro-F1 gain for {cfg.patience} epochs.")
                break

    print(f"\nDone. Best val macro-F1: {best_val_f1:.4f}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Class map  : {class_map_path}")


if __name__ == "__main__":
    parser = build_arg_parser("wafer train")
    args   = parser.parse_args()
    cfg    = WaferConfig.from_yaml_and_args(args.config, args)
    train(cfg)
