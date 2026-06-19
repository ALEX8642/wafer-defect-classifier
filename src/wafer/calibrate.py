"""
calibrate.py — Temperature scaling, ECE, reliability diagram, cost-of-quality analysis.

Why calibration matters for manufacturing:
    A model that outputs P(Edge-Ring) = 0.95 should be correct ~95 % of the time.
    If it's only correct 70 % of the time, the confidence number is misleading —
    an operator who trusts it will make systematically wrong hold/release decisions.
    Temperature scaling (a single learned scalar T that divides logits) is the
    simplest post-hoc recalibration that preserves ranking while fixing confidence.

Cost-of-quality framing:
    Two error types with very different costs in semiconductor manufacturing:
      Escape    = defect classified as "none" → wafer advances / ships.
                  Cost: yield loss at customer, warranty, potential recall.
      False alarm = "none" classified as defect → unnecessary hold/teardown/scrap.
                  Cost: throughput loss, technician time, potential yield loss from
                  unnecessary rework.
    The model can be tuned to trade one against the other by adjusting the
    decision threshold on P(none).  This is an operating-point decision driven
    by the customer's cost ratio, not a fixed F1-maximisation problem.

Entry point: python -m wafer.calibrate
  Loads outputs/best.pt, runs calibration on val set, reports ECE before/after,
  saves temperature, reliability diagram, and cost-analysis plots to outputs/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from wafer.config import WaferConfig, build_arg_parser
from wafer.data import get_dataloaders, CLASS_NAMES
from wafer.model import build_model


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

class TemperatureScaler(nn.Module):
    """Single scalar applied to logits before softmax. Tuned on validation set."""

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.05)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> "TemperatureScaler":
        """Find T that minimises NLL on the provided logits/labels (val set)."""
        nll = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.1, max_iter=100)

        def _eval():
            optimizer.zero_grad()
            loss = nll(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(_eval)
        return self


# ---------------------------------------------------------------------------
# ECE and reliability diagram
# ---------------------------------------------------------------------------

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error: confidence-weighted mean |confidence - accuracy|.
    Lower is better; perfect calibration = 0.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    corrects    = (predictions == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() > 0:
            bin_conf = confidences[mask].mean()
            bin_acc  = corrects[mask].mean()
            ece += mask.mean() * abs(bin_conf - bin_acc)
    return float(ece)


def plot_reliability_diagram(
    probs_before: np.ndarray,
    probs_after: np.ndarray,
    labels: np.ndarray,
    save_path: Path,
    n_bins: int = 15,
) -> None:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ece_before = compute_ece(probs_before, labels, n_bins)
    ece_after  = compute_ece(probs_after,  labels, n_bins)

    for ax, probs, title, ece in zip(
        axes,
        [probs_before, probs_after],
        ["Before temperature scaling", "After temperature scaling"],
        [ece_before, ece_after],
    ):
        confs = probs.max(axis=1)
        preds = probs.argmax(axis=1)
        accs  = (preds == labels).astype(float)

        bin_accs  = []
        bin_confs = []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (confs > lo) & (confs <= hi)
            if mask.sum() > 0:
                bin_accs.append(accs[mask].mean())
                bin_confs.append(confs[mask].mean())
            else:
                bin_accs.append(0.0)
                bin_confs.append(bin_centres[len(bin_confs)])

        ax.bar(bin_centres, bin_accs, width=1 / n_bins, alpha=0.7,
               color="steelblue", label="Accuracy")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
        ax.set_title(f"{title}\nECE = {ece:.4f}")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Reliability diagram: {save_path}")
    print(f"  ECE before: {ece_before:.4f}")
    print(f"  ECE after : {ece_after:.4f}")
    return ece_before, ece_after


# ---------------------------------------------------------------------------
# Cost-of-quality analysis
# ---------------------------------------------------------------------------

# Notional cost assumptions (relative units).
# Escape cost >> false-alarm cost: an undetected defect that reaches the customer
# risks warranty claims, yield loss at packaging/test, and potential recall.
# A false alarm only costs an unnecessary hold and inspection — recoverable.
# A 10:1 ratio is conservative for high-volume semiconductor manufacturing;
# the point is the framework, not the exact multiplier.
ESCAPE_COST    = 10   # relative units per escaped defect
FALSE_ALARM_COST = 1  # relative units per false alarm

NONE_IDX = CLASS_NAMES.index("none")


def cost_weighted_error(
    probs: np.ndarray,
    labels: np.ndarray,
    escape_cost: float = ESCAPE_COST,
    fa_cost: float = FALSE_ALARM_COST,
) -> dict:
    """
    Compute escape count, false-alarm count, and cost-weighted error.

    Escape:     true defect (label ≠ none) predicted as none
    False alarm: true none predicted as a defect class
    """
    preds = probs.argmax(axis=1)

    escapes    = int(((labels != NONE_IDX) & (preds == NONE_IDX)).sum())
    false_alarms = int(((labels == NONE_IDX) & (preds != NONE_IDX)).sum())
    n = len(labels)

    cost = (escapes * escape_cost + false_alarms * fa_cost) / n

    return {
        "escapes":      escapes,
        "false_alarms": false_alarms,
        "n":            n,
        "escape_rate":  escapes / max((labels != NONE_IDX).sum(), 1),
        "fa_rate":      false_alarms / max((labels == NONE_IDX).sum(), 1),
        "cost_weighted_error": cost,
    }


def plot_threshold_sensitivity(
    logits: torch.Tensor,
    labels: np.ndarray,
    temperature: float,
    save_path: Path,
) -> None:
    """
    Show how escape rate and false-alarm rate trade off as the none-class
    confidence threshold varies.  Decision rule: if P(none) > τ → predict none;
    otherwise predict the highest-confidence defect class.
    """
    scaled_logits = logits / max(temperature, 0.05)
    probs = torch.softmax(scaled_logits, dim=1).numpy()

    thresholds = np.linspace(0.05, 0.99, 120)
    escape_rates, fa_rates, f1s = [], [], []

    defect_mask = labels != NONE_IDX
    none_mask   = labels == NONE_IDX

    for tau in thresholds:
        p_none = probs[:, NONE_IDX]
        # Predict none where P(none) > tau, else pick best non-none class
        non_none_probs = probs.copy()
        non_none_probs[:, NONE_IDX] = -np.inf
        defect_pred = non_none_probs.argmax(axis=1)
        preds = np.where(p_none > tau, NONE_IDX, defect_pred)

        er = (defect_mask & (preds == NONE_IDX)).sum() / max(defect_mask.sum(), 1)
        fr = (none_mask   & (preds != NONE_IDX)).sum() / max(none_mask.sum(), 1)
        # macro-F1 (simplified: harmonic mean of per-class F1s)
        from sklearn.metrics import f1_score
        f1 = f1_score(labels, preds, average="macro", zero_division=0)

        escape_rates.append(er)
        fa_rates.append(fr)
        f1s.append(f1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: escape vs false-alarm rate (operating curve)
    axes[0].plot(thresholds, escape_rates,    color="crimson",   label="Escape rate")
    axes[0].plot(thresholds, fa_rates,         color="steelblue", label="False-alarm rate")
    axes[0].axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8, label="τ=0.50 (default)")
    axes[0].set_xlabel("None-class confidence threshold τ")
    axes[0].set_ylabel("Rate")
    axes[0].set_title("Escape vs False-alarm rate by threshold\n"
                       f"(escape cost={ESCAPE_COST}×, FA cost={FALSE_ALARM_COST}×)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Right: macro-F1 vs threshold
    axes[1].plot(thresholds, f1s, color="darkorange", label="Macro-F1")
    axes[1].axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8, label="τ=0.50")
    best_tau = thresholds[np.argmax(f1s)]
    axes[1].axvline(x=best_tau, color="green", linestyle=":", linewidth=1.2,
                    label=f"Best F1 τ={best_tau:.2f}")
    axes[1].set_xlabel("None-class confidence threshold τ")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_title("Macro-F1 by threshold")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Threshold sensitivity plot: {save_path}")
    print(f"  Best macro-F1 threshold: τ={best_tau:.2f} → F1={max(f1s):.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def calibrate(cfg: WaferConfig, checkpoint_path: Path | None = None) -> None:
    if checkpoint_path is None:
        checkpoint_path = cfg.output_dir / "best.pt"

    ckpt = torch.load(checkpoint_path, map_location=cfg.device, weights_only=False)
    class_to_idx: dict = ckpt["class_to_idx"]

    model = build_model(cfg, num_classes=len(class_to_idx)).to(cfg.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    train_loader, val_loader, test_loader, _, _ = get_dataloaders(cfg)

    def _collect(loader, desc):
        all_logits, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc=desc, leave=False):
                logits = model(inputs.to(cfg.device))
                all_logits.append(logits.cpu())
                all_labels.append(labels)
        return torch.cat(all_logits), torch.cat(all_labels)

    print("Collecting val logits for calibration...")
    val_logits,  val_labels  = _collect(val_loader,  "val")
    print("Collecting test logits for evaluation...")
    test_logits, test_labels = _collect(test_loader, "test")

    val_labels_np  = val_labels.numpy()
    test_labels_np = test_labels.numpy()

    # --- Pre-calibration metrics ---
    probs_before = torch.softmax(test_logits, dim=1).numpy()
    ece_before = compute_ece(probs_before, test_labels_np)
    print(f"\nECE before calibration: {ece_before:.4f}")

    # --- Fit temperature on val set ---
    scaler = TemperatureScaler()
    scaler.fit(val_logits, val_labels)
    T = float(scaler.temperature.item())
    print(f"Temperature T = {T:.4f}")

    # --- Post-calibration metrics ---
    probs_after = torch.softmax(test_logits / max(T, 0.05), dim=1).numpy()
    ece_after = compute_ece(probs_after, test_labels_np)
    print(f"ECE after  calibration: {ece_after:.4f}")

    # --- Save temperature ---
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    temp_path = cfg.output_dir / "temperature.json"
    with open(temp_path, "w") as f:
        json.dump({"temperature": T, "ece_before": ece_before, "ece_after": ece_after}, f, indent=2)
    print(f"Temperature saved: {temp_path}")

    # --- Reliability diagram ---
    plot_reliability_diagram(
        probs_before, probs_after, test_labels_np,
        cfg.output_dir / "reliability_diagram.png",
    )

    # --- Cost-of-quality analysis ---
    cost_before = cost_weighted_error(probs_before, test_labels_np)
    cost_after  = cost_weighted_error(probs_after,  test_labels_np)

    print(f"\nCost-of-quality analysis (escape cost={ESCAPE_COST}×, FA cost={FALSE_ALARM_COST}×):")
    print(f"  Escapes     : {cost_before['escapes']:5d}  escape rate {cost_before['escape_rate']:.3f}")
    print(f"  False alarms: {cost_before['false_alarms']:5d}  FA rate     {cost_before['fa_rate']:.3f}")
    print(f"  Cost-weighted error: {cost_before['cost_weighted_error']:.4f}")
    print(f"  (calibration changes confidence not argmax predictions, so escape/FA counts unchanged)")

    # --- Threshold sensitivity ---
    plot_threshold_sensitivity(
        test_logits, test_labels_np, T,
        cfg.output_dir / "threshold_sensitivity.png",
    )

    print(f"\nAll Phase 2 calibration outputs saved to {cfg.output_dir}/")


if __name__ == "__main__":
    parser = build_arg_parser("wafer calibrate")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    cfg  = WaferConfig.from_yaml_and_args(args.config, args)
    calibrate(cfg, checkpoint_path=args.checkpoint)
