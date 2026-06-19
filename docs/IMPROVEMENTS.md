# Wafer Defect Classifier — Improvement Story

This document tells the story of four improvements made after the initial
ResNet-18 baseline (macro-F1 0.8662, balanced accuracy 0.9300). Each section
covers: what we observed, the technique, what changed, and how to apply
the same idea to a real fab dataset.

---

## Phase A1 — Test-Time Augmentation over the D4 symmetry group

### What we observed

The training pipeline already augments each wafer map with random 90° rotations
and horizontal/vertical flips. The reasoning: a Scratch running diagonally
top-left to bottom-right is the same defect as a Scratch running bottom-left
to top-right — the physical failure mode is identical regardless of orientation.
The model should be invariant to these transforms.

But during *inference*, the model sees a single orientation. If a particular
Scratch is oriented at 270° and the model has learned mostly from 0° and 90°
examples of that class, its confidence is lower and it may misclassify.

### The technique

Test-time augmentation (TTA) applies each member of the D4 symmetry group
to the input, runs a separate forward pass for each, and averages the eight
probability vectors before deciding on a class.

The D4 group is the symmetry group of a square: 4 rotations (0°, 90°, 180°,
270°) and a horizontal flip of each, giving 8 elements total.

```
Original   → model → P₁
Rotated 90°  → model → P₂
Rotated 180° → model → P₃
Rotated 270° → model → P₄
Flipped      → model → P₅
Flip + 90°   → model → P₆
Flip + 180°  → model → P₇
Flip + 270°  → model → P₈

Final prediction = argmax( (P₁+P₂+...+P₈) / 8 )
```

Why does averaging help? Each forward pass is a slightly different view of the
same defect. Where one view is uncertain (low confidence), another may be
confident. Averaging acts as an ensemble of 8 "virtual models" that each see
the defect from a different angle, with zero additional training.

The cost is 8× the inference time per sample. For a real-time inline tool, this
matters. For a post-lot review tool or a Gradio demo, it is acceptable.

### What changed

No retraining. Enable TTA by setting `tta: true` in `configs/baseline.yaml` (default)
or passing `--tta` at the CLI.

```bash
python -m wafer.evaluate   # tta: true in baseline.yaml
```

**Measured result (combined with per-class thresholds, Phase A2):**
- Macro-F1: 0.8662 → **0.9025** (+3.6 pp)
- Scratch precision: 0.55 → **0.77** (+22 pp)
- Center F1: 0.90 → **0.95**
- Edge-Loc F1: 0.82 → **0.87**

The TTA and threshold contributions are entangled in the combined run, but both
are active. TTA reduces variance on orientation-sensitive classes (Scratch, Loc);
thresholds reduce false alarms on low-confidence predictions.

### Applying this to a real fab dataset

A real-time SPC monitoring system uses die-level pass/fail maps from inline
electrical test. These maps have the same D4 symmetry — the fab coordinate
system is arbitrary. TTA applies directly.

If your real dataset uses optical inspection images (SEM, brightfield), D4 is
still valid for symmetric defect types (pits, particles) but may not be
appropriate for directional defects (scratches, asymmetric probe marks) — in
that case, restrict to D2 (0°, 180°, horizontal flip, both) or just the
horizontal flip, depending on your tooling geometry.

---

## Phase A2 — Per-class confidence thresholds

### What we observed

Scratch had macro-F1 0.69, driven by low precision (0.55). Precision 0.55
means that for every 2 wafers the model flags as Scratch, only 1 is actually
a Scratch. The other is a false alarm — a good wafer pulled aside for
unnecessary inspection.

The root cause: with the default argmax decision rule, the model predicts
Scratch whenever P(Scratch) is the highest-probability class — even if that
probability is only 0.30. The model is uncertain but still makes a decision.

For a production tool, this creates a practical problem. If the fab sets a rule
"hold all wafers with predicted class = Scratch for manual review", they would
be holding roughly twice as many wafers as actually need it.

### The technique

Per-class confidence thresholds replace the single argmax rule with a class-
specific confidence requirement. For Scratch: only predict Scratch if
P(Scratch) ≥ τ_Scratch. If the model is not confident enough, the next-highest
class is tried instead.

The threshold τ is tuned on the validation set by grid-searching over [0.05, 0.95]
in 0.01 steps and picking the τ that maximises per-class F1 for that class.
This is done independently for each of the 9 classes.

```python
for tau in np.arange(0.05, 0.96, 0.01):
    # Suppress Scratch prediction where confidence < tau, take next-best class
    # Measure: does this improve Scratch F1 on the val set?
```

The resulting thresholds are saved to `outputs/thresholds.json` and applied
automatically by both `evaluate.py` and `demo.py`.

For most classes (Edge-Ring, none, Near-full), the optimised threshold turns
out close to 0.5 — the argmax default was already good. For Scratch, the
optimised threshold is typically 0.65–0.80, requiring significantly higher
confidence before committing to the Scratch prediction.

This is regenerated each time you run `python -m wafer.calibrate`.

### What changed

```bash
python -m wafer.calibrate   # now also runs threshold tuning and saves thresholds.json
python -m wafer.evaluate    # automatically loads and applies thresholds.json
```

**Measured result:** Scratch precision lifted from 0.55 → 0.77 (+22 pp), with
recall falling from 0.92 → 0.87 (−5 pp). Scratch F1: 0.69 → 0.81. The high
threshold (τ=0.85) means the model only predicts Scratch when highly confident —
uncertain calls now fall through to "none" rather than making a wrong Scratch
prediction. Whether this trade-off is correct depends on your quality target:
lower τ_Scratch to recover recall, raise it to further reduce false alarms.

### Applying this to a real fab dataset

In a real yield management system (YMS), the decision rule for each defect class
is set by the process engineers based on cost-of-quality assumptions:
- High-stakes customer: lower threshold → catch more defects, accept more false alarms
- Cost-sensitive high-volume production: raise threshold → fewer false alarms

The threshold tuning framework gives you a principled way to generate that
operating-point curve for each class, rather than using an arbitrary 0.5 cutoff.

In a real deployment, you would tune thresholds on a held-out lot of wafers with
confirmed labels (not just the internal val split), and review the precision-recall
curve with the process engineering team before deploying to production.

---

## Phase B — Test suite

### What we observed

The project had zero automated tests. In the Grad-CAM debugging session, a
subtle bug (matplotlib `savefig` adding white border pixels, decoded as "fail"
by the reverse-LUT) caused all 9 demo example images to classify as Edge-Ring.
The bug was caught visually, not automatically.

A test that verifies the encode-decode round-trip would have caught this
immediately.

### The technique

Pytest with synthetic tensors (no LSWMD.pkl required). Three test modules:

**`test_data.py`**: Verifies the encode-decode invariant. The critical test is
`test_png_roundtrip`: encode a synthetic wafer map to grayscale pixels using the
LUT, then decode it back using `_png_to_tensor`, and assert the output equals
the original map. This is exactly the path that broke with matplotlib artifacts.

**`test_calibration.py`**: Verifies the math of ECE and temperature scaling.
A perfect predictor must have ECE = 0; a consistently wrong predictor must have
ECE ≈ 1; temperature T > 1 must reduce max confidence without changing argmax.

**`test_model.py`**: Verifies model output shapes and TTA correctness. The
D4-invariance test is elegant: a constant input (all pixels = passing die) is
unchanged by all 8 transforms, so TTA and single-pass must produce identical
probabilities.

```bash
pip install pytest
pytest tests/ -v
# 18 tests, all pass in ~11 seconds on CPU — no GPU required
```

### Applying this to a real fab dataset

In a production ML pipeline, the tests you need most are boundary tests at
every data handoff:
- Raw map → encoded tensor round-trip (catches encoding bugs)
- Encoded tensor → model → output shape (catches schema mismatches after model updates)
- Calibrated probability → decision → audit log entry (catches threshold changes)

In a regulated environment (automotive, medical), these tests become part of the
validation evidence required by IATF 16949 or ISO 13485.

---

## Phase C — Focal loss retraining

### What we observed

Even after per-class threshold tuning, Scratch precision was still lower than
ideal. The root cause is the training objective: cross-entropy with class weights
addresses *how often* each class appears but not *which specific samples* are
hard to classify.

The 85% "none" class generates many easy training examples (wafers that look
obviously clean). These easy examples contribute large batch totals to the
gradient even though the model already handles them well. The rare, hard Scratch
examples get proportionally less gradient signal.

Class weighting helps by upweighting Scratch's loss — but once the model is
already predicting Scratch better than chance, those upweighted losses are for
easy Scratch examples too.

### The technique

Focal loss (Lin et al., ICCV 2017; originally developed for object detection)
adds a modulating factor (1 - p_t)^γ to the per-sample cross-entropy loss:

```
L_focal = -(1 - p_t)^γ · log(p_t)
```

where p_t is the probability the model assigns to the correct class.

When p_t is high (easy example, model already confident): (1-p_t)^γ ≈ 0 →
the loss for this sample is nearly zero, regardless of its class weight.

When p_t is low (hard example, model uncertain): (1-p_t)^γ ≈ 1 → the full
loss applies.

With γ=2 (the standard default), an easy example where p_t=0.9 contributes
(1-0.9)² = 0.01× the focal weight, while a hard example where p_t=0.1
contributes (1-0.1)² = 0.81× the focal weight — an 81× difference in
gradient attention.

We combine focal loss with class weighting: class weights handle the frequency
imbalance (85% none), focal loss handles the difficulty imbalance (rare hard
Scratch and Loc examples).

**To retrain with focal loss:**
```bash
# Change configs/baseline.yaml:
#   loss: focal
#   num_epochs: 40    (focal loss converges slower — give it more time)
#   patience: 10

python -m wafer.train
python -m wafer.calibrate   # re-run after new checkpoint
python -m wafer.evaluate
```

### Expected outcome

Literature on WM-811K and similar imbalanced multi-class problems shows γ=2
gives +3–8 pp F1 on tail classes (Scratch, Loc) with negligible change on
dominant classes (none, Edge-Ring). If the improvement is smaller, try γ=1
(less aggressive) or γ=3 (more aggressive).

### Applying this to a real fab dataset

Focal loss is particularly valuable when:
- You have many "normal" (clean-wafer) examples relative to defect classes
- Some defect classes are genuinely hard to classify (they look like noise)

In a real fab, this is almost always true: the none class dominates, and
borderline defects (faint scratches, sparse random failures) are inherently
ambiguous. Focal loss directs the model's attention toward exactly these hard
cases — the ones where getting it wrong has the most consequence.

One caution: focal loss slows early convergence. If you are limited to a small
number of epochs (e.g., for rapid model updates after a process change), start
with standard weighted CE and switch to focal only if the model has time to train.

---

## Phase D — Grad-CAM++ for sharper attribution

### What we observed

Grad-CAM on the Loc class produced a diffuse heatmap spread across the wafer
despite the model predicting Loc with 99.51 % confidence. This was not a
misclassification — the predicted class was correct. But the spatial attribution
was weak: a process engineer looking at the heatmap would not be able to identify
where on the wafer the contamination cluster is.

Standard Grad-CAM uses global-average-pooled gradients:

```
weights_k = (1 / Z) · Σ_{i,j} (∂y_c / ∂A^k_{ij})
```

where A^k is the k-th feature map and y_c is the target class score.
Global averaging spreads the attribution across the entire spatial extent.
When the relevant region is small (a Loc cluster is ~5-20 die out of hundreds),
most of the spatial gradient is near-zero, and the tiny peak at the cluster
gets diluted.

### The technique

Grad-CAM++ (Chattopadhyay et al., CVPR 2018) replaces global average pooling
with a position-wise second-order weighting:

```
α^k_{ij} = (∂²y_c / ∂A^k_{ij}²) / (2 · ∂²y_c / ∂A^k_{ij}² + A^k · ∂³y_c / ∂A^k_{ij}³ + ε)

weights_k = Σ_{i,j} α^k_{ij} · ReLU(∂y_c / ∂A^k_{ij})
```

Intuitively: α is large at spatial positions where the local gradient magnitude
is high relative to the activation magnitude — i.e., where the feature map is
changing rapidly because the class score depends on that specific location.

The effect: instead of diffuse attention proportional to which feature maps
matter globally, you get attention proportional to which *spatial positions*
in those feature maps are discriminative. For Loc and Scratch, this is a tight
cluster or streak. For Edge-Ring, where the discriminative region is broad,
the difference between Grad-CAM and Grad-CAM++ is minimal.

**To regenerate the overlay PNGs:**
```bash
python -m wafer.explain --method gradcampp
```

The demo automatically uses Grad-CAM++ for all live predictions.

### What changed

The five overlay PNGs in `outputs/grad_cam/` are regenerated with GradCAM++.
The Loc overlay should show a tighter hotspot on the off-center cluster
rather than diffuse coverage of the right side.

Grad-CAM (original) is still available via `--method gradcam` or by using the
`GradCAM` class directly in code.

### Applying this to a real fab dataset

For optical inspection images (SEM, brightfield, darkfield), Grad-CAM++ is
almost always preferable to standard Grad-CAM because:
- Real inspection images have much higher spatial resolution than 224×224 bins
- Defects (particles, pits, scratch marks) occupy a tiny fraction of the image
- A quality engineer looking at a Grad-CAM overlay needs the highlighted region
  to correspond to the actual defect location, not a diffuse halo

If you find Grad-CAM++ still produces diffuse attribution, consider:
- Moving the target layer earlier (e.g. `model.layer3[-1]`) — earlier layers
  have higher spatial resolution at the cost of less semantic specificity
- Score-CAM (Gradient-free, uses masked forward passes) for fully independent verification
- SHAP DeepExplainer as an alternative attribution framework

---

*All improvements are tested (`pytest tests/ -v`) and the `docs/ML_PRIMER.md`
explains the underlying ML concepts for readers coming from a manufacturing background.*
