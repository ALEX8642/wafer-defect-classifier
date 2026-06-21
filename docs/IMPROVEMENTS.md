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
TTA + per-class thresholds gives macro-F1 0.8811 — a strong result. The focal
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

## Phase F — Focal loss (corrected) + CBAM attention

### What we observed

After the CE baseline with TTA and thresholds reached macro-F1 0.8952, three tail
classes still lagged: Loc (0.79), Scratch (0.82), Random (0.87). The pattern across
all three was the same: high recall, low-to-moderate precision. The model could detect
these defects but confused them with similar-looking classes.

Two root causes:
1. **Loss focus**: class-weighted CE upweights rare classes by frequency, but treats
   all samples of a class equally. Hard examples (faint scratches, borderline Loc
   clusters, ambiguous Random patterns) receive the same gradient weight as easy ones.
2. **No spatial attention**: the ResNet feature maps attend to channels uniformly.
   For spatially-localised defects (Scratch = a streak, Loc = a cluster), the model
   needs to know *where* on the map to look, not just *which features* to activate.

### The techniques

**Focal loss (corrected, γ=2, no class weights):**

Focal loss modulates the per-sample CE loss by (1 − p_t)^γ where p_t is the
model's probability for the correct class. Easy examples where the model is already
confident (high p_t) contribute (1−p_t)^γ ≈ 0 to the gradient — nearly nothing.
Hard examples (low p_t) contribute (1−p_t)^γ ≈ 1 — the full gradient.

The corrected implementation deliberately omits class weights. The focal modulator
already suppresses the dominant "none" class (model learns it early → high p_t → low
focal weight). Adding class weights on top creates double-penalization that destabilises
training — this was the root cause of the Phase C failure, documented in detail above.

**CBAM — Convolutional Block Attention Module (Woo et al., ECCV 2018):**

CBAM appends two lightweight attention operations after each ResNet stage:

```
Stage output → Channel attention → Spatial attention → next stage
```

*Channel attention* asks: which feature maps (out of 64/128/256/512) carry
discriminative information for this prediction? A shared MLP operates on both
average-pooled and max-pooled channel statistics and produces a per-channel
weight in [0, 1].

*Spatial attention* asks: where on the map are those features activated? It
pools across channels (avg + max) and runs a 7×7 convolution to produce a
spatial weight map of the same H×W as the feature maps.

The two operations cost 43,912 parameters (0.4% of ResNet-18's 11.2M) — nearly
free, but the spatial attention is exactly what's missing for streak and cluster
defects.

### Results

Training ran 40 epochs with patience=10 on the 5090 (batch_size=128). Best
checkpoint at epoch 34.

```
Epoch  1/40  train f1 0.641  |  val f1 0.732
Epoch  8/40  train f1 0.870  |  val f1 0.884  ← already past CE best
Epoch 19/40  train f1 0.910  |  val f1 0.903  ← past original Phase A result
Epoch 34/40  train f1 0.955  |  val f1 0.927  ← checkpoint saved
```

**Test set (TTA×8 + per-class τ):**

| Class | F1 (CE baseline) | F1 (Focal + CBAM) | Δ |
|---|---|---|---|
| Loc | 0.79 | **0.84** | +5pp |
| Scratch | 0.82 | **0.86** | +4pp |
| Random | 0.87 | **0.91** | +4pp |
| Edge-Loc | 0.87 | **0.89** | +2pp |
| Near-full | 0.93 | **0.95** | +2pp |
| Edge-Ring | 0.98 | **0.99** | +1pp |
| Center | 0.95 | **0.95** | = |
| Donut | 0.86 | **0.86** | = |
| none | 0.99 | **0.99** | = |
| **Macro-F1** | **0.8952** | **0.9157** | **+2.0pp** |

Notable calibration result: T=0.6685 (less than 1). Focal loss produces an
underconfident model — the softmax distributions are flatter than CE because
easy examples are suppressed during training and never push the logits to
high-confidence values. Temperature scaling T<1 amplifies logits to match
empirical accuracy; ECE improved from 0.0164 → 0.0031.

**Operating point tradeoff:** The escape/FA distribution shifted compared to CE:
275 escapes (5.4%) vs 54 (1.1%), with far fewer false alarms (137 vs 990). At
the 10:1 cost assumption, cost-weighted error increased (0.0835 vs 0.0442). At
cost ratios below ~4:1, focal+CBAM dominates on both metrics. The appropriate
operating point depends on the fab's quality target.

### Applying this to a real fab dataset

CBAM is particularly valuable when:
- Defect signatures are spatially small relative to the wafer (Loc clusters,
  Scratch streaks) — the spatial attention module learns to focus on the relevant
  region regardless of where it appears on the map
- Multiple defect types can produce similar global statistics — channel attention
  helps distinguish them by re-weighting the most discriminative feature maps

Focal loss is most beneficial when:
- A large fraction of training samples are "easy" (the model gets them right from
  epoch 1) — class-weighted CE wastes gradient on those examples in every epoch
- Hard examples are spread across classes, not concentrated in one rare class

In a production ML pipeline, the escape/FA tradeoff documented here would be
resolved by working with process engineers to establish the actual cost ratio for
the specific product and customer context, then selecting the model and threshold
accordingly.

---

## Phase S — Semi-supervised pseudo-labeling (experiment; slight regression)

### What we observed

WM-811K contains 638,507 wafer maps with no `failureType` label — production wafers
that were never reviewed by a process engineer. These come from the same fab and the
same distribution as the labeled 172k subset. If the trained model can accurately
pseudo-label a large fraction of them, that additional data could improve tail-class
performance (Donut, Scratch, Loc) by giving the model more examples of each rare
defect pattern.

### The technique

1. **Pseudo-label generation**: Run the focal+CBAM teacher model (val F1 0.9265) on all
   638k unlabeled maps using TTA×8 + temperature calibration (T=0.6685). Retain only
   predictions where the maximum softmax probability ≥ 0.95.

2. **Filter none**: 514k of the 543k accepted maps were predicted as "none" (94.6%).
   The training set already has ~117k "none" examples — adding 514k more would make
   per-epoch compute ~5× heavier with no benefit to tail classes. Only the **29,100
   defect-class pseudo-labels** were appended to the training set.

3. **Retrain**: Same focal+CBAM config (γ=2, no class weights, 40 epochs, patience=10).
   Val and test splits untouched — only the training split is augmented.

**Pseudo-label distribution (defect classes only):**

| Class | Count | Avg confidence |
|---|---|---|
| Center | 8,356 | 0.992 |
| Edge-Ring | 6,254 | 0.993 |
| Edge-Loc | 5,602 | 0.984 |
| Loc | 2,095 | 0.982 |
| Near-full | 2,490 | 0.995 |
| Random | 3,419 | 0.988 |
| Scratch | 735 | 0.981 |
| Donut | 149 | 0.987 |
| **Total** | **29,100** | — |

### Results

**Outcome: slight regression across all tail classes.**

| Class | Focal+CBAM (Phase F) | +Pseudo-labels (Phase S) | Δ |
|---|---|---|---|
| Center | 0.95 | 0.95 | = |
| Donut | 0.86 | 0.86 | = |
| Edge-Loc | **0.89** | 0.88 | −1pp |
| Edge-Ring | 0.99 | 0.99 | = |
| Loc | **0.84** | 0.83 | −1pp |
| Near-full | **0.95** | 0.93 | −2pp |
| Random | **0.91** | 0.89 | −2pp |
| Scratch | **0.86** | 0.85 | −1pp |
| none | 0.99 | 0.99 | = |
| **Macro-F1** | **0.9157** | **0.9085** | **−0.7pp** |

Val F1 also regressed (0.9265 → 0.9140) and the model converged earlier (epoch 21
vs epoch 34), suggesting noisier gradient signal from pseudo-labeled examples.

### Why it regressed

At 0.95 confidence threshold the expected pseudo-label error rate is ~5%. For Scratch
(735 pseudo-labels), that implies ~37 mislabeled examples — significant relative to
the ~1,900 labeled Scratch training samples. Focal loss downweights high-confidence
examples so those 37 mislabeled maps receive normal gradient weight, quietly degrading
Scratch recall.

A second contributing factor: the pseudo-label distribution for rare classes is small
(Donut: 149, Scratch: 735). The teacher model's errors on these classes are
concentrated — a near-Donut misclassified as Donut in pseudo-labels appears many
times given the model's consistent behavior, creating systematic noise rather than
random noise.

### Conclusion

Phase F (focal+CBAM) remains the best model. Phase S is retained in the codebase as
`src/wafer/pseudo_label.py` — the infrastructure is sound and would benefit from:

- A higher confidence threshold (0.99 instead of 0.95) to reduce pseudo-label noise
- Iterative pseudo-labeling (train on pseudo-labels → re-generate with new model)
- Consistency regularization (MeanTeacher, FixMatch) rather than hard pseudo-labels

These are left as documented future work. The current portfolio headline is Phase F:
macro-F1 **0.9157**, ECE 0.0031, T=0.6685.

### Applying this to a real fab dataset

Pseudo-labeling is most effective when the teacher model's confidence threshold can be
set high enough to suppress noise without discarding too many examples. In this case,
raising from 0.95 to 0.99 would retain ~40% fewer maps but cut estimated error rate
from 5% to ~1% — potentially enough to avoid the regression. Experiment with both
thresholds and compare val F1 before committing to a retrain.

---
