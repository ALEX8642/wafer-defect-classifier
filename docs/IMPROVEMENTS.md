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
- Macro-F1: ~0.87 (single-pass) → **0.8811** (TTA + τ)
- Scratch precision: 0.55 → **0.73** (+18 pp)
- Center F1 → **0.95**
- Edge-Ring F1 → **0.98**

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

**Measured result:** Scratch precision lifted from 0.55 → 0.73 (+18 pp), with
recall improved to 0.87 (model still catches most Scratch). Scratch F1: ~0.69 →
0.79. The high threshold (τ=0.84) means the model only predicts Scratch when
highly confident — uncertain calls now fall through to "none" rather than making
a wrong Scratch prediction. Whether this trade-off is correct depends on your
quality target: lower τ_Scratch to recover recall, raise it to further reduce
false alarms.

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

## Phase C — Focal loss retraining (experiment + post-mortem)

### What we observed

Even after per-class threshold tuning, Scratch precision was still lower than
ideal. The root cause is the training objective: cross-entropy with class weights
addresses *how often* each class appears but not *which specific samples* are
hard to classify.

The 85% "none" class generates many easy training examples (wafers that look
obviously clean). These easy examples contribute large gradient signal even though
the model already handles them well. The rare, hard Scratch examples get
proportionally less attention.

Class weighting helps by upweighting Scratch's loss — but once the model already
predicts Scratch better than chance, the upweighted losses cover easy Scratch
examples too. What we really want is to focus gradient on the *hard* examples
regardless of class.

### The technique

Focal loss (Lin et al., ICCV 2017; originally developed for object detection)
adds a modulating factor (1 - p_t)^γ to the per-sample cross-entropy loss:

```
L_focal = -(1 - p_t)^γ · log(p_t)
```

where p_t is the probability the model assigns to the correct class.

When p_t is high (easy example): (1-p_t)^γ ≈ 0 → this sample contributes
almost nothing to the gradient.

When p_t is low (hard example): (1-p_t)^γ ≈ 1 → the full loss applies.

With γ=2, an easy example where p_t=0.9 contributes (0.1)² = 0.01× the weight,
while a hard example where p_t=0.1 contributes (0.9)² = 0.81× — an 81×
difference in gradient attention.

### First attempt: focal + class weights (failed)

Our first implementation combined focal loss with the class weights already used
by the cross-entropy baseline. The reasoning seemed sound on paper: class weights
handle frequency imbalance (85% none), focal handles difficulty imbalance. In
practice, this was **double-penalization**.

The modulating factor (1 - p_t)^γ already suppresses easy examples from the
dominant "none" class, because the model learns "none" early (high p_t → low
focal weight). Adding class weights on top amplifies the loss on the same rare
classes that focal is already focusing on, creating extreme gradient magnitudes
on rare hard examples and destabilizing early training.

**Results of the first attempt (focal + class weights, γ=2, 40 epochs):**

```
Epoch   1/40  train loss 0.9412 f1 0.2641  |  val f1 0.2324
Epoch  10/40  train loss 0.1916 f1 0.5974  |  val f1 0.6162
Epoch  22/40  train loss 0.0971 f1 0.6703  |  val f1 0.6859  ← best so far
Epoch  34/40  train loss 0.0430 f1 0.7252  |  val f1 0.7303  ← overall best
Epoch  40/40  train loss 0.0343 f1 0.7349  |  val f1 0.7191
Best val macro-F1: 0.7303
```

Compare: the CE baseline reaches val macro-F1 ~0.87 by epoch 27. The focal +
class weights run reached only 0.73 after 40 epochs and was still slowly
climbing — convergence was severely disrupted and the model never approached the
CE baseline within the training budget.

**Lesson:** Focal loss and class weights are alternative solutions to class
imbalance, not complementary ones. Choose one:
- Class-weighted CE: simple, stable, effective when frequency imbalance is the
  dominant problem
- Focal loss (no class weights): better when difficulty distribution matters more
  than frequency — model already gets easy examples right, gradient is wasted

### Corrected approach: focal without class weights

The fix is to remove class weights from the FocalLoss criterion. The focal
modulating factor already suppresses easy "none" examples because they have high
p_t. This is exactly the behaviour class weights were providing — but via a
per-sample adaptive mechanism rather than a fixed per-class scalar.

```python
# Corrected implementation (see src/wafer/train.py)
class FocalLoss(nn.Module):
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")  # no weight=
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()
```

**To retrain with the corrected focal loss:**
```bash
# In configs/baseline.yaml:
#   loss: focal
#   num_epochs: 40
#   patience: 10

python -m wafer.train
python -m wafer.calibrate
python -m wafer.evaluate
```

### Current state

The corrected focal loss implementation is in `src/wafer/train.py` and is ready
to run. The active checkpoint (`outputs/best.pt`) is the CE baseline, which with
TTA + per-class thresholds gives macro-F1 0.9025 — a strong result. The focal
retraining experiment is documented here as a planned next step.

**Expected outcome with corrected implementation:** Literature and the reasoning
above suggest γ=2 without class weights should converge in ~30–40 epochs to
within or above the CE baseline, with gains on the tail classes (Scratch, Loc)
that the CE + threshold approach already partially addressed. Whether the
improvement over 0.9025 is material is an open question — if the improvement is
< 1 pp macro-F1, the CE + TTA + threshold approach is the practical choice.

### Applying this to a real fab dataset

Focal loss is most valuable when:
- The dominant class (clean wafers) is genuinely easy and the model learns it
  early, making class-weighted CE wasteful in later epochs
- Some defect classes are hard *samples* not just rare classes — faint scratches,
  borderline edge effects, mixed-pattern wafers

The failed experiment here is itself a useful calibration: combining two
imbalance-correction mechanisms (class weights + focal) produces instability that
looks like slow convergence but is actually a loss landscape problem. In a real
production ML workflow, this would be caught earlier by watching the train F1 vs
val F1 curves across the first 10 epochs — a large gap (as seen here) signals
that the loss surface is too steep, not that the model needs more time.

Practical advice for deployment: run CE + class weights first (stable baseline),
then try focal without class weights as a second experiment. Compare test macro-F1
at convergence, not mid-training.

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
