# Wafer Defect Classifier

ResNet-18 trained on the public **WM-811K** wafer map dataset. 9-class spatial
defect classification with calibrated confidence, Grad-CAM interpretability, and
a one-click Gradio demo.

**Macro-F1 0.87 · Balanced accuracy 0.93 · ECE 0.0034 (after temperature scaling)**

---

## Results

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Edge-Ring | 0.98 | 0.98 | **0.98** |
| Near-full | 1.00 | 0.87 | **0.93** |
| Center | 0.85 | 0.97 | **0.90** |
| Random | 0.79 | 0.97 | **0.87** |
| Donut | 0.87 | 0.90 | **0.88** |
| Edge-Loc | 0.73 | 0.92 | **0.82** |
| Loc | 0.64 | 0.88 | **0.74** |
| Scratch | 0.55 | 0.92 | **0.69** |
| none | 1.00 | 0.97 | **0.98** |
| **Macro avg** | | | **0.87** |

Plain accuracy (0.97) is suppressed — a constant "none" predictor scores 0.85
while catching zero defects. Macro-F1 and balanced accuracy are the right metrics
under 85% class imbalance.

---

## Grad-CAM spatial interpretability

Does the model key on the physically meaningful region? Three examples:

| Scratch (99.99%) | Edge-Ring (100%) | Center (54%) |
|---|---|---|
| ![Scratch](outputs/grad_cam/gradcam_scratch.png) | ![Edge-Ring](outputs/grad_cam/gradcam_edge_ring.png) | ![Center](outputs/grad_cam/gradcam_center.png) |

**Scratch**: activation tightly follows the linear/arc streak — the model has learned
the mechanical-damage spatial signature.

**Edge-Ring**: activation concentrates on the interior passing-die zone (the boundary
between the intact die region and the failing ring). The model has learned
"Edge-Ring = large passing interior bounded by a failing perimeter" — an inverted but
valid representation.

**Center**: correct localisation to the lower-center cluster at 54% confidence,
reflecting genuine ambiguity between Center and Loc.

---

## Demo

```bash
pip install -r requirements.txt
pip install -e .

# Place LSWMD.pkl in data/raw/ (download from Kaggle: wafer-map-dataset)
# then train (takes ~10 min on a 4090):
python -m wafer.train

# Run the Gradio demo:
python -m wafer.demo
# → http://localhost:7860
```

The demo loads 9 test-set examples (one per class) at startup.
Click any example to see the predicted class, calibrated confidence, process-mode
interpretation, and Grad-CAM overlay in one view.

---

## Approach

**Dataset**: WM-811K (Wu et al., 2015) — 811k wafer maps, 172k labeled across 9
failure-pattern classes. Binned maps (0=outside, 1=pass, 2=fail). No optical/SEM
imagery — spatial defect classification only.

**Preprocessing**: One-hot encode {0,1,2} into 3 channels (preserves discrete
semantics; scalar normalisation would imply "fail = 2× pass"). Nearest-neighbour
resize to 224×224 (preserves binary values; bilinear would create intermediate
values that don't exist in the domain).

**Architecture**: ResNet-18 from scratch — research consistently shows ResNet-18
matches or outperforms ResNet-50 on this task, attributed to the relative simplicity
of binned spatial patterns vs. natural images. Three-channel first conv reused
unchanged; head replaced with a 9-class linear layer.

**Imbalance (85% "none")**: Class-weighted cross-entropy with weights from
`sklearn.compute_class_weight('balanced')`. Directly encodes the domain intuition:
an escaped defect costs more than a false alarm.

**Calibration**: Temperature scaling (T=1.10, fit on val set via LBFGS). Halves ECE
(0.0067 → 0.0034). T > 1 confirms mild overconfidence typical of from-scratch
training.

**Cost-of-quality framing**: Two error types with different operational costs —
escape (defect predicted as none) vs. false alarm (none predicted as defect).
At the default threshold: 59 escapes (1.2% of defect samples), 917 false alarms
(3.1% of none samples), cost-weighted error 0.0436 at 10:1 escape/FA cost ratio.
Threshold sensitivity plot shows how the operating point shifts across τ ∈ [0.05, 0.99].

---

## Repository layout

```
src/wafer/
  config.py      — WaferConfig dataclass, YAML + CLI merge
  data.py        — WM-811K loading, 70/10/20 stratified split, DataLoaders
  model.py       — ResNet-18 builder
  train.py       — AdamW + CosineAnnealingLR + early stopping on val macro-F1
  evaluate.py    — Test-set metrics, per-class breakdown, confusion matrix
  calibrate.py   — Temperature scaling, ECE, reliability diagram, cost analysis
  explain.py     — GradCAM (hook-based, no extra deps), overlay figures
  demo.py        — Gradio demo

docs/
  ANALYSIS.md         — full narrative for technical audience
  process_modes.md    — 9-class defect → process failure mode table
  phase1_results.md   — Phase 1 test metrics

configs/baseline.yaml — training hyperparameters
```

---

## Limitations

- **Binned maps only**: 0/1/2 per die, not pixel-level inspection images. Defect
  boundaries are at die pitch resolution.
- **No fab ground truth**: process-mode interpretations are illustrative QE reasoning
  from spatial geometry, not cause-verified claims.
- **Near-full sample size**: 30 test samples; precision/recall carry wide CIs.
- **Single seed**: results reflect seed=42. Macro-F1 variance is typically ±0.01–0.02.

---

## References

Wu, M.-J., Jang, J.-S. R., Chen, J.-L. (2015). Wafer Map Failure Pattern Recognition
and Similarity Ranking for Large-Scale Data Sets. *IEEE Trans. Semiconductor
Manufacturing*, 28(1), 1–12.

Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization. *ICCV 2017*.
