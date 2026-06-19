"""
explain.py — Grad-CAM root-cause localisation for wafer defect predictions.

GradCAM algorithm (Selvaraju et al., 2017):
  1. Forward pass → capture activations at target conv layer
  2. Backward pass on target-class score → capture gradients
  3. Global-average-pool gradients → per-channel weights
  4. Weighted sum of activations → class activation map (CAM)
  5. ReLU + normalise → resize to input resolution → overlay

Target layer: model.layer4[-1]  (last residual block of ResNet-18/50)
This captures the highest-level spatial features before global pooling.

Entry point: python -m wafer.explain
  Loads outputs/best.pt, samples correctly-classified test examples for ≥3
  classes, saves overlay PNGs to outputs/grad_cam/.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from wafer.config import WaferConfig, build_arg_parser
from wafer.data import get_dataloaders, CLASS_NAMES
from wafer.model import build_model


# ---------------------------------------------------------------------------
# GradCAM implementation (hook-based, no extra dependencies)
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Context-manager GradCAM for any CNN with a spatial conv layer.

    Usage:
        with GradCAM(model, model.layer4[-1]) as cam:
            heatmap, pred_class, probs = cam.compute(input_tensor)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hooks: list = []

    def __enter__(self) -> "GradCAM":
        self._hooks.append(
            self.target_layer.register_forward_hook(self._save_activations)
        )
        self._hooks.append(
            self.target_layer.register_full_backward_hook(self._save_gradients)
        )
        return self

    def __exit__(self, *_) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _save_activations(self, _module, _input, output) -> None:
        self._activations = output.detach()

    def _save_gradients(self, _module, _grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def compute(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """
        Run GradCAM for one sample.

        Args:
            input_tensor: (1, C, H, W) on the same device as model.
            target_class: class index to explain; None = predicted class.

        Returns:
            heatmap (H, W) in [0, 1], predicted class index, softmax probs (num_classes,)
        """
        self.model.eval()
        x = input_tensor.clone().requires_grad_(True)

        logits = self.model(x)
        probs  = logits.softmax(dim=1).squeeze().detach().cpu().numpy()

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        logits[0, target_class].backward()

        # Gradient-weighted activations
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, Ch, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Normalise to [0, 1] and resize
        cam = cam - cam.min()
        denom = cam.max()
        cam = cam / (denom + 1e-8)
        heatmap = F.interpolate(
            cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy()

        return heatmap, target_class, probs


# ---------------------------------------------------------------------------
# Convenience: revert one-hot tensor → displayable grayscale map
# ---------------------------------------------------------------------------

def tensor_to_display(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a (3, H, W) one-hot tensor back to an (H, W) uint8 image.
    Channel order: [outside=0, pass=1, fail=2].
    Output pixel values: 0 → 40, 1 → 160, 2 → 255  (dark / mid / bright).
    """
    wafer_map = tensor.argmax(dim=0).cpu().numpy().astype(np.uint8)
    lut = np.array([40, 160, 255], dtype=np.uint8)
    return lut[wafer_map]


def save_overlay(
    wafer_img: np.ndarray,
    heatmap: np.ndarray,
    pred_class: int,
    probs: np.ndarray,
    true_class: int,
    save_path: Path,
) -> None:
    """Save a three-panel figure: wafer map | CAM heatmap | overlay."""
    overlay = plt.cm.jet(heatmap)[..., :3]  # (H, W, 3) RGB jet
    wafer_rgb = np.stack([wafer_img] * 3, axis=-1).astype(float) / 255.0
    blended = 0.55 * wafer_rgb + 0.45 * overlay

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["Wafer map", "Grad-CAM", "Overlay"]
    imgs = [wafer_img, heatmap, blended]
    cmaps = ["gray", "jet", None]

    for ax, img, title, cmap in zip(axes, imgs, titles, cmaps):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1 if img.max() <= 1 else 255)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    pred_name = CLASS_NAMES[pred_class]
    true_name = CLASS_NAMES[true_class]
    conf = probs[pred_class]
    match = "✓" if pred_class == true_class else f"✗ (true: {true_name})"
    fig.suptitle(
        f"Predicted: {pred_name}  conf={conf:.2%}  {match}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point: generate examples for ≥3 classes
# ---------------------------------------------------------------------------

# Classes chosen for process-mode richness + contrast; Near-full included
# because it's the rarest and Grad-CAM spatial coverage is informative.
_EXAMPLE_CLASSES = ["Center", "Edge-Ring", "Scratch", "Loc", "Near-full"]


def generate_cam_examples(cfg: WaferConfig, checkpoint_path: Path | None = None) -> None:
    if checkpoint_path is None:
        checkpoint_path = cfg.output_dir / "best.pt"

    ckpt = torch.load(checkpoint_path, map_location=cfg.device, weights_only=False)
    class_to_idx: dict = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = build_model(cfg, num_classes=len(class_to_idx)).to(cfg.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    target_layer = model.layer4[-1]

    _, _, test_loader, _, _ = get_dataloaders(cfg)

    # Collect one correctly-classified example per target class
    targets_wanted = {class_to_idx[c] for c in _EXAMPLE_CLASSES if c in class_to_idx}
    collected: dict[int, tuple[torch.Tensor, int]] = {}

    with torch.no_grad():
        for inputs, labels in test_loader:
            for i in range(len(labels)):
                cls = labels[i].item()
                if cls not in targets_wanted or cls in collected:
                    continue
                inp = inputs[i : i + 1].to(cfg.device)
                logit = model(inp)
                pred = int(logit.argmax(dim=1).item())
                if pred == cls:
                    collected[cls] = (inputs[i], cls)
            if len(collected) == len(targets_wanted):
                break

    out_dir = cfg.output_dir / "grad_cam"
    out_dir.mkdir(parents=True, exist_ok=True)

    with GradCAM(model, target_layer) as cam:
        for cls_idx, (tensor, true_cls) in collected.items():
            inp = tensor.unsqueeze(0).to(cfg.device)
            heatmap, pred_cls, probs = cam.compute(inp, target_class=cls_idx)
            wafer_img = tensor_to_display(tensor)
            class_name = idx_to_class[cls_idx]
            save_path = out_dir / f"gradcam_{class_name.lower().replace('-', '_')}.png"
            save_overlay(wafer_img, heatmap, pred_cls, probs, true_cls, save_path)
            print(f"  Saved: {save_path}  (conf={probs[pred_cls]:.2%})")

    if len(collected) < len(targets_wanted):
        missing = [CLASS_NAMES[i] for i in targets_wanted if i not in collected]
        print(f"  Warning: no correct example found for: {missing}")

    print(f"\nGrad-CAM examples saved to {out_dir}/")


# Backward-compatible stub (used by Phase 3 demo)
def grad_cam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    layer: Optional[nn.Module] = None,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Thin wrapper around GradCAM for single-sample inference."""
    if layer is None:
        layer = model.layer4[-1]
    with GradCAM(model, layer) as gcam:
        return gcam.compute(input_tensor, target_class)


if __name__ == "__main__":
    parser = build_arg_parser("wafer explain")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    cfg  = WaferConfig.from_yaml_and_args(args.config, args)
    print(f"Generating Grad-CAM examples for: {_EXAMPLE_CLASSES}")
    generate_cam_examples(cfg, checkpoint_path=args.checkpoint)
