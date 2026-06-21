"""
demo.py — One-click Gradio demo for the wafer defect classifier.

Loads outputs/best.pt + outputs/temperature.json, runs ResNet-18 inference +
temperature-scaled calibration + Grad-CAM for any wafer-map PNG.

Input format: PNG rendered with the project's 3-value LUT —
    0 (outside wafer) → gray 40
    1 (passing die)   → gray 160
    2 (failing die)   → gray 255
Test-set example images (one per class) are auto-extracted at first launch
and cached in outputs/demo_examples/.

Usage:
    python -m wafer.demo                # localhost:7860
    python -m wafer.demo --share        # Gradio public tunnel
    python -m wafer.demo --port 7861
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from wafer.config import WaferConfig, build_arg_parser
from wafer.data import get_dataloaders, CLASS_NAMES
from wafer.model import build_model
from wafer.explain import GradCAMPlusPlus, tensor_to_display
from wafer.evaluate import tta_predict, apply_thresholds


# ---------------------------------------------------------------------------
# Process-mode notes (shown alongside each prediction)
# ---------------------------------------------------------------------------

_PROCESS_NOTES: dict[str, str] = {
    "Edge-Ring": (
        "Systematic perimeter non-uniformity — etch/CMP edge effect or film "
        "deposition rate variation at perimeter. Corrective action: tool "
        "edge-uniformity check, edge-exclusion zone review."
    ),
    "Edge-Loc": (
        "Localised edge-sector failure — incomplete edge-bead removal (EBR) or "
        "edge-exclusion-zone contamination. Investigate sector-specific EBR nozzle."
    ),
    "Center": (
        "Center-symmetric anomaly — spin-coat centre-point defect, chuck contact "
        "mark, or CVD centre-flow issue. Check chuck flatness, spin speed uniformity."
    ),
    "Scratch": (
        "Linear/arc mechanical damage — robot-arm contact, cassette scratch, or "
        "stylus contact during metrology. Review wafer-handling sequence."
    ),
    "Loc": (
        "Off-centre particle cluster — localised contamination, reticle particle, "
        "or repeating field defect. Particle map and reticle inspection recommended."
    ),
    "Donut": (
        "Annular non-uniformity at intermediate radius — spin-coat solvent ring or "
        "hot-plate non-uniformity. Check bake plate temperature profile."
    ),
    "Random": (
        "Scattered failures — particle shower, ESD event, or random film defect. "
        "Airborne particle monitoring; ESD grounding audit."
    ),
    "Near-full": (
        "Gross process excursion — bulk chemistry failure, severe equipment "
        "malfunction, or film delamination. Lot-hold and tool quarantine warranted."
    ),
    "none": (
        "No systematic spatial failure signature — normal production wafer. "
        "No corrective action indicated."
    ),
}


# ---------------------------------------------------------------------------
# Model + temperature loading
# ---------------------------------------------------------------------------

def _load_assets(cfg: WaferConfig):
    """Load checkpoint, temperature scalar, and per-class thresholds.

    Returns (model, target_layer, temperature, thresholds).
    thresholds is an empty dict if thresholds.json does not exist yet.
    """
    ckpt_path = cfg.output_dir / "best.pt"
    temp_path = cfg.output_dir / "temperature.json"
    thresh_path = cfg.output_dir / "thresholds.json"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run `python -m wafer.train` first."
        )

    ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
    saved_cfg = ckpt.get("cfg", {})
    cfg.cbam = bool(saved_cfg.get("cbam", cfg.cbam))
    cfg.cbam_reduction = int(saved_cfg.get("cbam_reduction", cfg.cbam_reduction))
    model = build_model(cfg, num_classes=len(ckpt["class_to_idx"])).to(cfg.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    temperature = 1.0
    if temp_path.exists():
        with open(temp_path) as f:
            temperature = float(json.load(f)["temperature"])
    else:
        print(f"Warning: {temp_path} not found — using T=1.0 (uncalibrated)")

    thresholds: dict = {}
    if thresh_path.exists():
        with open(thresh_path) as f:
            thresholds = json.load(f)
        print(f"  Per-class thresholds loaded ({len(thresholds)} classes)")

    target_layer = model.layer4[-1]
    return model, target_layer, temperature, thresholds


# ---------------------------------------------------------------------------
# Image decode: PNG → (3, H, W) one-hot tensor
# ---------------------------------------------------------------------------

def _png_to_tensor(img_array: np.ndarray, input_size: int = 224) -> torch.Tensor:
    """
    Reverse the project LUT (outside=40, pass=160, fail=255) back to {0,1,2},
    then encode as a (3, H, W) one-hot float tensor resized to input_size.
    """
    gray = img_array.mean(axis=2) if img_array.ndim == 3 else img_array.astype(float)
    # Midpoints: 40↔160 = 100, 160↔255 = 207.5 → 208
    wmap = np.where(gray < 100, 0, np.where(gray < 208, 1, 2)).astype(np.int64)
    t = torch.tensor(wmap, dtype=torch.long)
    tensor = F.one_hot(t, num_classes=3).permute(2, 0, 1).float()
    return F.interpolate(
        tensor.unsqueeze(0), size=(input_size, input_size), mode="nearest"
    ).squeeze(0)


# ---------------------------------------------------------------------------
# Inference + Grad-CAM
# ---------------------------------------------------------------------------

def _run_inference(
    img_array: np.ndarray,
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    temperature: float,
    thresholds: dict,
    cfg: WaferConfig,
):
    """
    Full pipeline: decode PNG → calibrated probs (TTA if enabled) + Grad-CAM++ heatmap.

    Two-pass design:
      Pass 1 — calibrated probabilities via TTA or single forward pass (no grad).
               Per-class thresholds applied to pick the predicted class.
      Pass 2 — Grad-CAM++ backward pass for the predicted class heatmap.
               Always single-pass (TTA gradients don't average meaningfully).

    Returns (tensor, heatmap, pred_cls, calibrated_probs).
    """
    tensor = _png_to_tensor(img_array, cfg.input_size)
    inp = tensor.unsqueeze(0).to(cfg.device)

    # Pass 1: calibrated probabilities
    if cfg.tta:
        cal_probs = tta_predict(model, tensor.unsqueeze(0), cfg.device, temperature)[0]
    else:
        with torch.no_grad():
            logits = model(inp)
        cal_probs = torch.softmax(logits / max(temperature, 0.05), dim=1).squeeze().cpu().numpy()

    # Apply per-class thresholds or plain argmax
    if thresholds:
        pred_cls = int(apply_thresholds(cal_probs[np.newaxis, :], thresholds, CLASS_NAMES)[0])
    else:
        pred_cls = int(cal_probs.argmax())

    # Pass 2: Grad-CAM++ for the predicted class
    with GradCAMPlusPlus(model, target_layer) as cam:
        heatmap, _, _ = cam.compute(inp, target_class=pred_cls)

    return tensor, heatmap, pred_cls, cal_probs


def _build_figure(
    tensor: torch.Tensor,
    heatmap: np.ndarray,
    pred_cls: int,
    cal_probs: np.ndarray,
    temperature: float,
    save_path: Path,
) -> str:
    """Render the 3-panel Grad-CAM figure, save to save_path, return the path string."""
    wafer_img = tensor_to_display(tensor)          # (H, W) uint8 {40, 160, 255}
    jet_map = plt.cm.jet(heatmap)[..., :3]         # (H, W, 3) float
    wafer_rgb = np.stack([wafer_img] * 3, axis=-1).astype(float) / 255.0
    blended = 0.55 * wafer_rgb + 0.45 * jet_map

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title, cmap in zip(
        axes,
        [wafer_img, heatmap, blended],
        ["Wafer map", "Grad-CAM", "Overlay"],
        ["gray", "jet", None],
    ):
        vmax = 255 if img.max() > 1.0 else 1.0
        ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    pred_name = CLASS_NAMES[pred_cls]
    conf = cal_probs[pred_cls]
    fig.suptitle(
        f"Predicted: {pred_name}  |  Calibrated confidence: {conf:.1%}  (T={temperature:.3f})",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def _build_markdown(pred_cls: int, cal_probs: np.ndarray, temperature: float) -> str:
    pred_name = CLASS_NAMES[pred_cls]
    conf = cal_probs[pred_cls]
    top3 = cal_probs.argsort()[::-1][:3]
    rows = "\n".join(f"| {CLASS_NAMES[i]} | {cal_probs[i]:.1%} |" for i in top3)
    note = _PROCESS_NOTES.get(pred_name, "")

    return f"""\
## Prediction: **{pred_name}**

**Calibrated confidence:** {conf:.1%}

### Top-3 classes

| Class | P(class) |
|---|---|
{rows}

### Process interpretation

{note}

---
*ECE after calibration: 0.0031 · temperature T={temperature:.3f} · macro-F1 0.9157 on WM-811K test set (TTA×8)*
"""


# ---------------------------------------------------------------------------
# Example extraction (one wafer-map PNG per class, cached after first run)
# ---------------------------------------------------------------------------

def _extract_examples(cfg: WaferConfig, out_dir: Path) -> list[list]:
    """Return a list of [[image_path]] for each class, extracting from test set if needed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("*.png"))
    if len(existing) >= len(CLASS_NAMES):
        print(f"  Using cached examples from {out_dir}/")
        return [[str(p)] for p in existing]

    print("  Extracting demo examples from test set (one-time)...")
    _, _, test_loader, _, _ = get_dataloaders(cfg)

    collected: dict[int, torch.Tensor] = {}
    for inputs, labels in test_loader:
        for i in range(len(labels)):
            cls = int(labels[i].item())
            if cls not in collected:
                collected[cls] = inputs[i]
        if len(collected) == len(CLASS_NAMES):
            break

    from PIL import Image

    saved: list[list] = []
    for cls_idx in range(len(CLASS_NAMES)):
        if cls_idx not in collected:
            continue
        class_name = CLASS_NAMES[cls_idx]
        wafer_img = tensor_to_display(collected[cls_idx])  # (H, W) uint8 {40,160,255}
        save_path = out_dir / f"{class_name.lower().replace('-', '_')}.png"

        # Save as RGB PNG directly via PIL — no matplotlib rendering artifacts.
        # Values {40, 160, 255} are preserved exactly so _png_to_tensor decodes back
        # to the correct {0, 1, 2} map without border/title contamination.
        rgb = np.stack([wafer_img, wafer_img, wafer_img], axis=-1)
        Image.fromarray(rgb, "RGB").save(save_path)
        saved.append([str(save_path)])

    print(f"  Saved {len(saved)} examples to {out_dir}/")
    return saved


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def build_demo(cfg: WaferConfig):
    # Use tempfile.mkdtemp() so the temp dir is always owned by the current user.
    # Avoids PermissionError when /tmp/gradio or outputs/demo_tmp was created by
    # a different user (e.g. root during development/testing).
    demo_tmp = Path(tempfile.mkdtemp(prefix="wafer_demo_"))
    os.environ["GRADIO_TEMP_DIR"] = str(demo_tmp)

    import gradio as gr

    print("Loading model assets...")
    model, target_layer, temperature, thresholds = _load_assets(cfg)

    print("Preparing demo examples...")
    examples = _extract_examples(cfg, cfg.output_dir / "demo_examples")

    _call_count = [0]  # mutable counter in closure

    def predict(img_array):
        if img_array is None:
            return None, "*Upload a wafer map or click an example to classify it.*"
        try:
            tensor, heatmap, pred_cls, cal_probs = _run_inference(
                img_array, model, target_layer, temperature, thresholds, cfg
            )
            # Unique filename per call so Gradio doesn't serve a cached copy
            _call_count[0] += 1
            cam_path = demo_tmp / f"cam_{_call_count[0]}.png"
            fig_path = _build_figure(tensor, heatmap, pred_cls, cal_probs, temperature, cam_path)
            md = _build_markdown(pred_cls, cal_probs, temperature)
            return fig_path, md
        except Exception as exc:
            return None, f"**Error:** {exc}"

    with gr.Blocks(title="Wafer Defect Classifier") as demo:
        gr.Markdown(
            "# Wafer Defect Classifier\n"
            "ResNet-18 trained on WM-811K (172 k labeled maps, 9 classes). "
            "Macro-F1 **0.87** on the held-out test set. "
            "Click an example below or upload your own wafer-map PNG."
        )

        with gr.Row():
            with gr.Column(scale=1):
                img_input = gr.Image(
                    label="Wafer map input",
                    type="numpy",
                    height=300,
                )
                classify_btn = gr.Button("Classify", variant="primary")

            with gr.Column(scale=2):
                cam_output = gr.Image(label="Grad-CAM overlay", height=300)
                md_output = gr.Markdown(
                    "*Select an example or upload a wafer map.*"
                )

        if examples:
            gr.Examples(
                examples=examples,
                inputs=[img_input],
                label="Test-set examples — one per defect class",
                examples_per_page=9,
            )

        classify_btn.click(
            fn=predict, inputs=[img_input], outputs=[cam_output, md_output]
        )
        img_input.change(
            fn=predict, inputs=[img_input], outputs=[cam_output, md_output]
        )

    return demo, demo_tmp


if __name__ == "__main__":
    import argparse

    parser = build_arg_parser("wafer demo")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio URL")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    cfg = WaferConfig.from_yaml_and_args(args.config, args)
    app, demo_tmp = build_demo(cfg)
    app.launch(
        server_port=args.port,
        share=args.share,
        allowed_paths=[str(cfg.output_dir), str(demo_tmp)],
    )
