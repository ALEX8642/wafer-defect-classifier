# ML Primer for the Wafer Defect Classifier

**Audience:** Engineers and scientists with semiconductor manufacturing expertise
who want to understand what the ML model is doing, why the design choices were
made, and how to reason about its outputs — without needing a formal ML background.

---

## What is a CNN doing?

A convolutional neural network (CNN) is a mathematical function with millions of
parameters (weights). Given an input image, it produces a probability distribution
over a fixed set of classes.

The "convolutional" part refers to how the network processes space. Instead of
looking at every pixel independently, it applies small filters (typically 3×3 or
5×5 pixels) repeatedly across the image. Each filter detects a specific local
pattern — an edge, a gradient, a corner. Stacking many layers of filters lets the
network detect increasingly complex patterns: edges → shapes → textures → objects.

For a wafer map classifier:
- Early layers learn to detect edges in the 0/1/2 map (transitions between outside,
  passing, and failing die)
- Middle layers detect shapes — clusters, rings, lines
- The final layer combines shapes into class scores: "this spatial arrangement of
  failure regions matches the Edge-Ring class"

Nothing in the network is hand-coded. It learns all of this from labelled examples
during training. The "learning" is gradient descent: repeatedly computing how much
each weight contributed to each error, and adjusting weights to reduce that error.

**In manufacturing terms:** A CNN is like a yield engineer who has reviewed 100,000
wafer maps and developed an intuition for what each defect pattern looks like. The
key difference is that the CNN can't articulate its reasoning the way a human can —
it just produces a probability score. That's why we add Grad-CAM: to show *where*
on the wafer the model is looking.

---

## What does the loss function do?

During training, the model makes predictions on batches of labelled wafer maps.
The loss function measures how wrong those predictions are. The training loop
adjusts the model's weights to minimize this loss.

### Cross-entropy loss

The most common loss for classification. For each sample, it computes:

```
loss = -log(P(correct class))
```

If the model assigns probability 0.9 to the correct class: loss = -log(0.9) = 0.10
If the model assigns probability 0.1 to the correct class: loss = -log(0.1) = 2.30

The model is pushed to increase its probability for the correct class. Over many
training iterations, this adjusts the weights until the model reliably assigns
high probability to the right class.

### Class-weighted cross-entropy

The WM-811K dataset has 85% "none" (clean wafers) and only 0.7% Scratch. With
standard cross-entropy, the model can achieve 85% accuracy by predicting "none"
for every sample — it never learns to detect defects.

Class weighting solves this by multiplying the loss for each sample by the
inverse frequency of its class:

```
weight(none)   = 1.0   (common — low weight)
weight(Scratch)= ~10.0  (rare — high weight)
```

Each Scratch mistake now costs ~10× more than a "none" mistake. The model is
forced to learn Scratch detection to avoid these high-cost errors.

### Focal loss (Phase C improvement)

Class weighting fixes the frequency imbalance, but there is still a *difficulty*
imbalance. Once the model learns to handle most "none" samples well, those easy
examples continue to dominate the gradient even though the model has nothing left
to learn from them. Focal loss adds a term that automatically down-weights easy
examples:

```
L_focal = -(1 - p_t)^γ · log(p_t)

where p_t = probability assigned to the correct class
      γ   = concentration parameter (γ=2 is standard)
```

When p_t = 0.9 (easy): focal weight = (1-0.9)² = 0.01 → almost zero contribution
When p_t = 0.1 (hard): focal weight = (1-0.1)² = 0.81 → strong contribution

The model is directed to focus on the hard examples — the borderline Scratch cases
that are genuinely ambiguous — rather than wasting gradient capacity on the many
clean wafers it has already mastered.

---

## What does calibration mean?

A model's raw output is a *logit vector* — one number per class. Logits are
converted to probabilities by the softmax function:

```
P(class i) = exp(logit_i) / Σ_j exp(logit_j)
```

The probabilities always sum to 1. But are they *meaningful* probabilities?
If the model says P(Edge-Ring) = 0.92, does that mean it's correct ~92% of
the time when it makes that prediction?

**Calibration** measures this correspondence between stated confidence and
empirical accuracy. It is quantified by the Expected Calibration Error (ECE):

```
ECE = Σ_bins (|bin|/N) × |avg_confidence_in_bin - avg_accuracy_in_bin|
```

A perfectly calibrated model has ECE = 0. Our baseline had ECE = 0.0067
(0.67% miscalibration) — already quite good, but still systematically
overconfident (the model's average stated confidence was slightly higher than
its actual accuracy in each confidence bin).

**Why does this matter for manufacturing?** Consider a yield monitoring alert:
"Model confidence for Edge-Ring fell from 0.95 to 0.70 over the last 50 wafers."
If the model is poorly calibrated, that 0.70 might correspond to only 50% actual
accuracy, and the alert is meaningless. If the model is well calibrated, a 0.70
confidence genuinely means the defect pattern looks like Edge-Ring 70% of the time
across similar wafers in the past.

### Temperature scaling

We correct overconfidence with a single learnable scalar T (the "temperature"):

```
calibrated probability = softmax(logits / T)
```

T is fit on the validation set by minimizing the NLL loss with T as the only
parameter. Our baseline gives T = 1.1036, indicating mild overconfidence
(T > 1 softens the probabilities; T < 1 would sharpen them).

Temperature scaling preserves the *ranking* of classes (the argmax doesn't change),
but rescales confidence to match empirical accuracy. After scaling, ECE drops from
0.0067 to 0.0034 — a halving of calibration error for a single-parameter fix.

**In manufacturing terms:** Temperature scaling is like a bias correction in a
measurement system. The model's raw output is systematically slightly too confident
(a repeatable bias). Temperature scaling corrects that bias without changing which
defect class is predicted — it only adjusts how much the model is saying it believes
in that prediction.

---

## What does Grad-CAM show?

Grad-CAM (Gradient-weighted Class Activation Mapping) answers: **which spatial
regions on the wafer did the model key on to make its prediction?**

The algorithm:
1. Run a forward pass through the model for the target class.
2. Backpropagate the class score to the last convolutional layer.
3. Average the gradient signals across all channels of that layer to get a spatial
   importance map.
4. Multiply by the feature activations and apply ReLU (keep only positive
   contributions).
5. Resize to the original input resolution and overlay on the wafer map.

A red/hot region in the overlay means: "the model placed high importance on this
part of the wafer map when deciding the predicted class."

### What the Grad-CAM results tell us

**Scratch (99.99% confidence):** Tight activation on the vertical streak.
The model has learned the linear/arc spatial signature of mechanical damage.
This is the ideal result — the attribution matches what a process engineer would
highlight.

**Edge-Ring (100% confidence):** Activation on the *interior* passing-die zone,
not the failing perimeter ring. This sounds inverted, but is a valid learned
representation: "Edge-Ring = large circular passing interior bounded by a failing
edge." The interior boundary is the highest-contrast spatial feature for this class.

**Loc (99.51% confidence):** Diffuse activation. The model is highly confident but
the attribution doesn't point clearly to the off-center cluster. This is a known
limitation: the model has learned "not Edge, not Center, not full" as the Loc
discriminating rule rather than explicit cluster localisation. Grad-CAM++ (our
Phase D improvement) partially addresses this with sharper gradient weighting.

### Why the diffuse Loc attribution is a genuine limitation

A quality engineer using this tool to identify the contamination site responsible
for a Loc failure would look at the Grad-CAM overlay and need it to point to the
cluster. If the overlay is diffuse, the tool is not useful for root-cause
localisation — only for defect classification. This is an honest limitation:
the model's *decision* is reliable (99.51% confidence, correct class), but its
*spatial reasoning* doesn't fully match the physical definition of the class.

In a production deployment, this distinction matters: use the model's classification
for automatic hold/release decisions, but don't rely on its Grad-CAM for particle
map correlation without manual verification.

---

## Why macro-F1 and balanced accuracy, not plain accuracy?

With 85% "none" samples, a model that predicts "none" for every single wafer
achieves **85% accuracy**. It has learned nothing about defects but scores
better than our trained model on plain accuracy (97% vs 97% — the trained model
only slightly outperforms on this metric).

**Macro-F1** treats each of the 9 classes equally regardless of how often it
appears. It is the average of per-class F1 scores:

```
F1_class = 2 × (precision × recall) / (precision + recall)
Macro-F1 = average(F1_class) over all 9 classes
```

A constant "none" predictor has F1 = 0 for all 8 defect classes → Macro-F1 ≈ 0.11.
Our trained model achieves Macro-F1 = 0.8662 — a genuine improvement.

**Balanced accuracy** averages per-class recall (what fraction of each class is
correctly detected), which is another way to treat classes equally.

**In manufacturing terms:** Macro-F1 is equivalent to asking "is the system
effective at detecting and distinguishing all failure modes, not just the most
common one?" This is the right question for a yield management tool. A system
that perfectly identifies clean wafers but misses every Scratch failure is useless.

---

## How to use this model responsibly

1. **The model classifies spatial *patterns*, not root causes.** "Edge-Ring" means
   the wafer matches the spatial signature of past Edge-Ring failures. It is a
   hypothesis about process cause, not a confirmed finding.

2. **Calibrated confidence is a flag for human review, not an autoreleasing signal.**
   At 70% confidence, the model is saying "this looks like Edge-Ring to me, but
   there is a 30% chance I am wrong or it is an atypical example." In a low-escape-
   risk environment, that might trigger a hold for engineer review; in a cost-
   sensitive line, you might set the hold threshold at 90%.

3. **The per-class threshold is an operational decision, not a technical one.**
   The engineer who sets `τ_Scratch = 0.70` vs `0.50` is making a quality engineering
   judgment: "I want fewer false alarms, and I accept the corresponding increase in
   escape risk." That judgment belongs with the process team, not in the model.

4. **Monitor for distribution shift.** The model was trained on WM-811K (a real fab
   dataset from ~2015). If your fab's photolithography node, equipment generation, or
   die layout is substantially different, the spatial patterns may differ. Periodic
   re-evaluation against a labelled holdout from your fab is good practice.
