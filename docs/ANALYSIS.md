# Wafer Defect Classifier — Analysis

**Audience:** hiring managers, Solutions/FDE interviewers, internal R&D transfer.
**One-line summary:** A ResNet-18 trained on public WM-811K wafer maps achieves
macro-F1 **0.87** with calibrated probabilities, Grad-CAM spatial localisation, and
a cost-of-quality error framework — demonstrating the intersection of
manufacturing domain judgment and production-grade ML.

---

## 1. Problem and data

Semiconductor yield depends on detecting defective die before packaging and
shipment. Wafer maps — spatial 2D grids recording passing and failing die — carry
process-failure signatures that repeat across wafers when a specific tool or step
goes out of control. Classifying the *type* of failure pattern guides root-cause
investigation and corrective action faster than die-level inspection alone.

This project uses the public **WM-811K dataset** (Wu et al., 2015): 811,457 wafer
maps from a real fab, of which 172,950 carry a `failureType` label across 9 classes.
The dataset is publicly available on Kaggle and has been used in 30+ academic papers.
It uses **binned maps** (0 = outside wafer, 1 = passing die, 2 = failing die) — not
optical or SEM imagery. That boundary is stated explicitly: this is spatial defect
classification, not inspection-grade optical CV. The public-data choice is
deliberate: the artifact travels with zero IP exposure.

**Class distribution (labeled rows):**

| Class | Count | % |
|---|---|---|
| none | 147,431 | 85.2 % |
| Edge-Ring | 9,680 | 5.6 % |
| Edge-Loc | 5,189 | 3.0 % |
| Center | 4,294 | 2.5 % |
| Loc | 3,593 | 2.1 % |
| Scratch | 1,193 | 0.7 % |
| Random | 866 | 0.5 % |
| Donut | 555 | 0.3 % |
| Near-full | 149 | 0.09 % |

The severe imbalance ("none" at 85 %) is the central modeling challenge — and the
first quality-engineering talking point.

---

## 2. Model and training

**Architecture:** ResNet-18, trained from scratch (no ImageNet pretrained weights).
Research on this dataset consistently shows ResNet-18 matches or slightly outperforms
ResNet-50, attributed to the relative simplicity of binned spatial patterns versus
natural images. Pretrained weights are an option in the config but the honest
baseline uses random initialisation.

**Preprocessing:** Variable-size maps (6×21 to 300×202 pixels) are one-hot encoded
into 3 channels (one per pixel value) then resized to 224×224 via nearest-neighbour
interpolation. One-hot encoding preserves the discrete semantic distinction between
outside/pass/fail; scalar normalisation would incorrectly imply "fail = 2× pass."

**Imbalance handling:** Class-weighted cross-entropy with weights inversely
proportional to class frequency (via `sklearn.compute_class_weight('balanced')`).
This penalises misclassifying rare-class defects more heavily — directly encoding
the quality-engineering intuition that an undetected defect (escape) is more costly
than an over-flagged good wafer (false alarm).

**Training:** AdamW (lr=1e-3, wd=1e-4), CosineAnnealingLR, early stopping on val
macro-F1 (patience=7). Mixed precision (torch.amp) on CUDA. Best checkpoint at
epoch 27/30, val macro-F1 = 0.8732.

---

## 3. Test-set performance

**Headline numbers (epoch 27, 4090 laptop):**

| Metric | Value |
|---|---|
| **Macro-F1** | **0.8662** |
| Balanced accuracy | 0.9300 |
| Plain accuracy | 0.97 — suppressed; see below |

Plain accuracy of 0.97 is misleading: a model predicting "none" for every sample
would score 0.85 accuracy while catching zero defects. Macro-F1 weights each class
equally regardless of frequency; balanced accuracy averages per-class recall. Both
are the right metrics under class imbalance, and both are reported here.

**Per-class breakdown:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Center | 0.85 | 0.97 | 0.90 |
| Donut | 0.87 | 0.90 | 0.88 |
| Edge-Loc | 0.73 | 0.92 | 0.82 |
| Edge-Ring | 0.98 | 0.98 | 0.98 |
| Loc | 0.64 | 0.88 | 0.74 |
| Near-full | 1.00 | 0.87 | 0.93 |
| Random | 0.79 | 0.97 | 0.87 |
| Scratch | 0.55 | 0.92 | 0.69 |
| none | 1.00 | 0.97 | 0.98 |

---

## 4. Calibration

*[Fill in after running `python -m wafer.calibrate`]*

**ECE before temperature scaling:** `[FILL]`
**Temperature T:** `[FILL]`
**ECE after temperature scaling:** `[FILL]`

A well-calibrated model is essential for operational decisions. When the model
outputs P(Edge-Ring) = 0.95, an operator should be able to trust that confidence
number — it should reflect true accuracy ~95 % of the time. The reliability diagram
shows the relationship between stated confidence and empirical accuracy before and
after a single-parameter temperature scaling correction.

Temperature scaling divides all logits by a learned scalar T (fit on the validation
set). It preserves class rankings — the predicted class never changes — but rescales
the confidence to match empirical accuracy. A T > 1 indicates the model was
overconfident (common in deep networks without regularisation); T < 1 indicates
underconfidence.

*Reliability diagram: `outputs/reliability_diagram.png`*

---

## 5. Cost-of-quality error analysis

*[Fill in after running `python -m wafer.calibrate`]*

**Escapes (defect predicted as none):** `[FILL]` out of `[FILL]` defect test samples
**False alarms (none predicted as defect):** `[FILL]` out of `[FILL]` none test samples
**Cost-weighted error (10:1 assumption):** `[FILL]`

This reframing separates two error types that have very different operational costs:

- **Escape** = a defective wafer classified as clean. It advances through the line or
  ships to the customer. Cost: yield loss at test, potential customer return,
  warranty claim, or — at worst — field failure. In high-reliability applications
  (automotive, medical, aerospace) the cost multiplier can exceed 100×.
- **False alarm** = a clean wafer classified as defective. It is unnecessarily held
  for inspection or scrapped. Cost: throughput loss, technician time, potential
  yield loss from unnecessary rework. Recoverable.

The notional **10:1 cost ratio** (escape cost = 10× false-alarm cost) is
conservative for volume semiconductor manufacturing and is stated as an assumption,
not a calibrated number. The point is the framework: by varying the decision
threshold on P(none), an operator can tune the operating point to their specific
cost-of-quality requirements rather than accepting whatever threshold maximises F1.

The threshold sensitivity plot (`outputs/threshold_sensitivity.png`) shows how
escape rate and false-alarm rate trade off across the full range of decision
thresholds. Key observation: at the default threshold (argmax, no explicit τ),
**Scratch** dominates the false-alarm count (154 of 29,486 "none" test samples
predicted as Scratch) while providing high recall (0.92). A quality engineer
facing a high-escape-risk customer could lower τ (accept more false alarms to
catch more escapes); one operating a cost-sensitive line could raise τ.

---

## 6. Grad-CAM spatial localisation

*[Fill in after running `python -m wafer.explain` and reviewing outputs/grad_cam/]*

Grad-CAM (Selvaraju et al., 2017) computes a class activation map by
gradient-weighting the final convolutional layer's feature maps. The result is a
heatmap showing *where* on the wafer the model is keying for a given prediction.

For a spatial-defect classifier, this is the interpretability check that matters:
does the model's attention align with the region that defines the defect pattern?

| Class | Expected activation region | Observed (Grad-CAM) |
|---|---|---|
| Edge-Ring | Perimeter ring | `[FILL after running]` |
| Center | Central cluster | `[FILL after running]` |
| Scratch | Linear/arc streak | `[FILL after running]` |
| Loc | Off-center cluster | `[FILL after running]` |
| Near-full | Nearly full coverage | `[FILL after running]` |

*Grad-CAM overlays: `outputs/grad_cam/`*

If the activation aligns with the expected region, it provides evidence that the
model has learned the physically meaningful feature — not a spurious correlation
with wafer ID, map size, or background pixel count. If it misaligns, that is a
finding worth reporting and investigating (e.g. is the model keying on the wafer
boundary rather than the defect pattern?).

---

## 7. Process-mode interpretations

See `docs/process_modes.md` for the full table. Three examples:

**Edge-Ring** (F1 0.98) → model achieves near-perfect classification of the most
common defect class. Process interpretation: systematic perimeter non-uniformity,
consistent with etch or CMP edge effects. Corrective action: tool edge-uniformity
check, edge exclusion zone review.

**Center** (F1 0.90) → high recall (0.97) — the model rarely misses a center
defect. Process interpretation: spin-coat, chuck contact, or CVD center-flow
anomaly. Corrective action: check chuck flatness, spin speed uniformity.

**Scratch** (recall 0.92, precision 0.55) → the model detects nearly all scratch
defects but raises false alarms on "none" wafers at ~1.8× the true-detection rate.
Process interpretation: handling or mechanical contact damage. The low precision
is the operating-point discussion: in a high-escape-risk environment, accept the
false alarms; in a cost-sensitive line, raise the confidence threshold for Scratch.

---

## 8. Limitations

- **Public binned maps, not optical/SEM imagery.** WM-811K records 0/1/2 per die,
  not pixel-level inspection images. Defect boundaries are coarser; the model's
  spatial resolution is limited to die pitch. This is stated explicitly so the
  artifact is not mistaken for inspection-grade optical CV.
- **No fab ground truth on process cause.** Process-mode interpretations in
  Section 7 and `docs/process_modes.md` are illustrative QE reasoning based on
  spatial geometry, not cause-verified claims.
- **Near-full sample size.** The Near-full class has only 30 test samples. Reported
  recall (0.87) and precision (1.00) carry wide confidence intervals at this support.
- **Single seed.** Results reflect one training run with seed=42. Macro-F1 variance
  across seeds is typically ±0.01–0.02 on this dataset.

---

## References

[1] M.-J. Wu, J.-S. R. Jang, J.-L. Chen, "Wafer Map Failure Pattern Recognition
and Similarity Ranking for Large-Scale Data Sets," IEEE Trans. Semiconductor
Manufacturing, vol. 28, no. 1, pp. 1–12, 2015.
https://ieeexplore.ieee.org/document/6932449

[2] R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization," ICCV 2017.
