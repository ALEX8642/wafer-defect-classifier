# Phase 2 — The Differentiator Layer (Weeks 4–7)

**Goal:** the part that separates this from a Kaggle notebook. Two additions plus
a narrative artifact. This is where manufacturing/quality judgment gets layered
onto the ML, which is the entire reason the project is persuasive.

## 2.1 Explainability — Grad-CAM root-cause localization
- Implement Grad-CAM (use a maintained lib or a clean forward/backward hook on
  the last conv block) to overlay class activation on the wafer map.
- For each prediction the demo (Phase 3) will show *where* on the wafer the model
  is keying. This maps defect class → physical/process meaning.
- Tie at least three classes to plausible process-failure modes, e.g.:
  - **Edge-Ring** → edge-localized etch / CMP / film-uniformity effects at wafer
    perimeter.
  - **Center** → spin-coat / chuck-contact / center-of-wafer process variation.
  - **Scratch** → handling / mechanical contact damage (often linear).
  Write these as a small `process_modes.md` table the demo and writeup both read.
  (Frame as illustrative QE reasoning, not a verified physical claim — public
  data, no fab ground truth on cause.)

## 2.2 Calibration & cost-of-quality framing (the signature move)
This is the thing no pure-ML candidate does and every quality engineer should.

- **Reliability diagram** + Expected Calibration Error (ECE) on test predictions.
  Apply temperature scaling if miscalibrated; show before/after.
- **Cost-framed error analysis.** Reframe the confusion matrix in
  escape-vs-false-alarm terms:
  - An **escape** = a real defect class misclassified as "none" (defective wafer
    ships / advances). High cost-of-quality — external failure.
  - A **false alarm** = "none" or benign pattern flagged as a defect (unnecessary
    hold, teardown, scrap). Internal cost.
  - Assign a *notional* relative cost to each error type (state assumptions
    plainly; the point is the framework, not the exact dollars) and compute a
    cost-weighted error from the confusion matrix.
  - Show how the decision threshold / class weighting trades escapes against
    false alarms — i.e., you can tune the operating point to the customer's
    cost-of-quality, not just maximize F1.

## 2.3 Narrative artifact
Produce `docs/ANALYSIS.md`: the metrics, the calibration story, the cost framing,
and the process-mode interpretations, written for a hiring/transfer audience.
This is the source text you'll compress into the README and the slide in Phase 4.

## Acceptance criteria
- [ ] Grad-CAM overlay generated for any test wafer; saved examples for ≥3 classes.
- [ ] `process_modes.md` table exists, linking ≥3 classes to process modes.
- [ ] Reliability diagram + ECE reported; temperature scaling shown before/after.
- [ ] Cost-weighted error metric computed from the confusion matrix with stated
      cost assumptions and a threshold/operating-point sensitivity illustration.
- [ ] `ANALYSIS.md` written and coherent to a non-ML reader.

## Anti-goals
No demo UI yet (Phase 3). Don't chase a higher F1 here — Phase 2's value is
interpretation and framing on top of the Phase 1 model, not squeezing accuracy.
