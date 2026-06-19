"""
model.py — ResNet-based wafer defect classifier.

Architecture note:
    ResNet-18 is the default. Research on WM-811K (and the related MixedWM38)
    shows ResNet-18 matches or outperforms ResNet-50 on this task — attributed
    to the relative simplicity of binned wafer map patterns vs. natural images.
    ResNet-50 is available via --arch resnet50 for comparison.

Pretrained weights note:
    ImageNet pretrained weights are available (--pretrained) but transfer weakly
    to one-hot encoded wafer maps. The first-conv weights learn completely
    different low-level features on natural RGB images. Use pretrained=False for
    an honest from-scratch baseline; try pretrained=True to see if initialisation
    helps convergence speed on this dataset.

Input compatibility:
    Both ResNet variants expect 3-channel input. One-hot encoding of {0,1,2}
    pixel values produces exactly 3 channels — no first-conv adaptation needed.
"""
from __future__ import annotations

import torch.nn as nn
import torchvision.models as models

from wafer.config import WaferConfig
from wafer.data import CLASS_NAMES

NUM_CLASSES = len(CLASS_NAMES)  # 9


def build_model(cfg: WaferConfig, num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Build ResNet-18 (default) or ResNet-50 with a num_classes-way head.

    Args:
        cfg: WaferConfig — reads cfg.arch and cfg.pretrained.
        num_classes: Number of output classes (9 for WM-811K).

    Returns:
        nn.Module with the final FC layer replaced.
    """
    arch = cfg.arch.lower()
    pretrained = cfg.pretrained

    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unknown arch {arch!r}. Supported: resnet18, resnet50.")

    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, num_classes)

    return backbone
