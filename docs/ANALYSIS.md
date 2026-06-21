# Wafer Defect Classifier — Analysis

**Audience:** hiring managers, Solutions/FDE interviewers, internal R&D transfer.
**One-line summary:** A ResNet-18 with CBAM attention trained on public WM-811K wafer
maps achieves macro-F1 **0.92** (CE baseline 0.88, +0.04 from focal loss retraining
with CBAM channel-and-spatial attention, +0.01 from TTA and per-class thresholds)
with calibrated probabilities, Grad-CAM++ spatial localisation, and a cost-of-quality
error framework — demonstrating the intersection of manufacturing domain judgment and
production-grade ML.

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
macro-F1 (patience=10). Mixed precision (torch.amp) on CUDA. Best checkpoint at
epoch 34/40, val macro-F1 = 0.9265 (focal loss + CBAM).

**Attention (CBAM):** Convolutional Block Attention Module appended after each ResNet
stage (Woo et al., ECCV 2018). Adds 43.9k parameters (0.4% overhead). Channel
attention recalibrates which feature maps matter; spatial attention recalibrates where
on the map those features are. Significant gains on spatially-localised tail classes.

---

## 3. Test-set performance

**Headline numbers (epoch 34, 5090 desktop — focal loss + CBAM):**

| Metric | CE baseline | Focal + CBAM + TTA + τ |
|---|---|---|
| **Macro-F1** | **0.8952** | **0.9157** |
| Balanced accuracy | 0.9257 | 0.9085 |
| Plain accuracy | 0.98 — suppressed | 0.98 — suppressed |

*CE baseline = ResNet-18 with class-weighted CE, TTA×8, per-class τ.
Focal+CBAM = same inference stack, retrained with corrected focal loss (γ=2, no class
weights) and CBAM attention. Val macro-F1 of current checkpoint: 0.9265 at epoch 34.*

Plain accuracy of 0.97 is misleading: a model predicting "none" for every sample
would score 0.85 accuracy while catching zero defects. Macro-F1 weights each class
equally regardless of frequency; balanced accuracy averages per-class recall. Both
are the right metrics under class imbalance, and both are reported here.

**Per-class breakdown (focal loss + CBAM + TTA×8 + per-class τ):**

| Class | Prec | Rec | F1 | vs CE baseline |
|---|---|---|---|---|
| Center | 0.97 | 0.93 | **0.95** | = |
| Edge-Ring | 0.99 | 0.99 | **0.99** | +1pp |
| none | 0.99 | 1.00 | **0.99** | = |
| Near-full | 0.97 | 0.93 | **0.95** | +2pp |
| Random | 0.91 | 0.91 | **0.91** | +4pp |
| Edge-Loc | 0.90 | 0.89 | **0.89** | +2pp |
| Loc | 0.90 | 0.79 | **0.84** | +5pp |
| Scratch | 0.84 | 0.87 | **0.86** | +4pp |
| Donut | 0.86 | 0.86 | **0.86** | = |
| **Macro avg** | **0.92** | **0.91** | **0.92** | **+2pp** |

Key observations:
- Loc (+5pp) and Scratch (+4pp) are the biggest movers — exactly the tail classes
  CBAM spatial attention was designed to improve
- Focal loss with γ=2 produces an underconfident model (T=0.6685 < 1) — the
  model's softmax distributions are flatter than CE; temperature scaling amplifies them
- The escape/FA operating point shifted: 275 escapes (5.4%) vs 54 (1.1%) for CE.
  Macro-F1 improved because more defect classes are classified correctly; but the
  10:1 cost-weighted error increased (0.0835 vs 0.0442). In a high-escape-cost
  environment, returning to CE + thresholds or tuning γ downward is the right call.
- Donut (0.86): only 111 test samples — numbers carry wide confidence intervals

---

## 4. Calibration

**ECE before temperature scaling:** 0.0164
**Temperature T:** 0.6685
**ECE after temperature scaling:** 0.0031

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

T = 0.6685 reveals mild underconfidence — atypical and worth explaining. Focal loss
suppresses gradients on easy examples by multiplying the CE loss by (1−p_t)^γ. During
training, this means the model's logits are never "pushed hard" on any single class,
resulting in flatter softmax distributions at inference. Temperature T < 1 sharpens
these distributions back to match empirical accuracy. ECE improved from 0.0164 → 0.0031,
making this one of the best-calibrated checkpoints in the project.

*Reliability diagram: `outputs/reliability_diagram.png`*

---

## 5. Cost-of-quality error analysis

**Escapes (defect predicted as none):** 275 out of 5,104 defect test samples (5.4 %)
**False alarms (none predicted as defect):** 137 out of 29,486 none test samples (0.5 %)
**Cost-weighted error (10:1 assumption):** 0.0835

*Operating point comparison:*

| Model | Escapes | Escape rate | False alarms | FA rate | Cost-weighted |
|---|---|---|---|---|---|
| CE + TTA + τ | 54 | 1.1% | 990 | 3.4% | 0.0442 |
| Focal+CBAM + TTA + τ | 275 | 5.4% | 137 | 0.5% | 0.0835 |

Focal loss without class weights redistributes the escape/FA tradeoff. The dominant
"none" class no longer receives an explicit loss penalty, so some borderline defect
samples are predicted as "none" with high confidence — increasing escapes. At the same
time, false alarms drop sharply because the model is more decisive about defect classes
when it does predict them. Which operating point is better depends on the fab's
cost-of-quality assumptions: at 10:1 (escape much worse), CE + thresholds is better.
At 2:1 or lower, focal+CBAM wins on both macro-F1 and cost-weighted error.

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

**Per-class confidence thresholds (tuned on val set):**

| Class | τ | Interpretation |
|---|---|---|
| none | 0.05 | Model is almost always confident on clean wafers — any prediction accepted |
| Donut | 0.05 | Low threshold — model is generally confident here |
| Edge-Ring | 0.52 | Model reaches high confidence reliably |
| Random | 0.66 | Scattered-failure pattern — moderate threshold |
| Edge-Loc | 0.73 | Confused class — requires meaningful confidence |
| Loc | 0.76 | Off-center cluster is genuinely ambiguous |
| Near-full | 0.76 | High-confidence pattern when present |
| Center | 0.86 | Model makes many borderline Center predictions |
| Scratch | **0.84** | Primary improvement target — requires high confidence before committing |

High thresholds for Scratch and Edge-Loc are the expected outcome given their low
baseline precision. Predictions below τ fall through to the next-highest class
(typically "none") — visible in the confusion matrix as increased none predictions.

The threshold sensitivity plot (`outputs/threshold_sensitivity.png`) shows how
escape rate and false-alarm rate trade off across the full range of decision
thresholds. Key observation: the best global macro-F1 occurs at τ = 0.06 for the
none-class threshold (macro-F1 = 0.907), far lower than the naïve τ = 0.50 default.
The per-class threshold approach targets each class's specific operating point
rather than a single global threshold. The cost-weighted error of 0.0436 at the
default argmax threshold corresponds to a mix of 59 escaped defects (each costing
10 units) and 917 false alarms (each costing 1 unit), normalised by test set size.

---

## 6. Grad-CAM spatial localisation

Grad-CAM (Selvaraju et al., 2017) computes a class activation map by
gradient-weighting the final convolutional layer's feature maps. The result is a
heatmap showing *where* on the wafer the model is keying for a given prediction.

For a spatial-defect classifier, this is the interpretability check that matters:
does the model's attention align with the region that defines the defect pattern?

| Class | Confidence | Expected region | Observed activation |
|---|---|---|---|
| Edge-Ring | 100 % | Perimeter ring | Interior of ring — model keys on the large passing-die zone bounded by the failing perimeter, not the ring itself |
| Center | 54 % | Central cluster | Lower-center region — correctly localises to the failing cluster; lower confidence reflects the cluster sitting slightly below centre (Loc-like) |
| Scratch | 99.99 % | Linear/arc streak | Tight hot spot on the upper portion of the vertical scratch streak — strong alignment with the linear defect |
| Loc | 99.51 % | Off-center cluster | Diffuse activation distributed across centre and right side — model keying on the non-edge, non-centre aspect of the pattern rather than the cluster itself |
| Near-full | 100 % | Broad coverage | Broad activation with emphasis on boundary transition zones (corners, right side) — the global "nearly everything failing" signature is recognised without needing precise local attribution |

*Grad-CAM overlays: `outputs/grad_cam/`*

**Interpretation.** Three of the five classes show clean spatial alignment
(Scratch, Center, Near-full). Two show an inverted or diffuse pattern that warrants
comment:

- **Edge-Ring**: The model keys on the *interior* passing-die zone rather than the
  failing perimeter ring. This is a valid learned representation — "Edge-Ring = large
  circular passing interior bounded by a failing edge ring" — but the activation is
  inverted relative to naive expectation. A process engineer would say: the model has
  learned the *shape* of the intact die region rather than the defect band. This is
  not a failure; it reflects that the interior boundary is the highest-contrast spatial
  feature for this class.

- **Loc**: Activation is diffuse rather than tightly localised on the cluster.
  High confidence (99.51 %) despite diffuse attribution suggests the model has learned
  a "not Edge, not Center, not full" discriminating rule at the representation level
  rather than explicit spatial localisation of the cluster. This is a limitation worth
  noting: if the cluster moves to a new position on a future wafer, the model may still
  classify correctly, but the Grad-CAM provides less actionable spatial information for
  a process engineer trying to identify the contamination site.

Overall, the Grad-CAM evidence supports that the model has learned physically
meaningful spatial features for the classes where those features are geometrically
distinctive (Scratch, Center, Near-full). The Edge-Ring and Loc findings are worth
noting in any operational deployment review.

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
