# Phase 1 — Baseline Pipeline (Weeks 1–3)

**Goal:** end-to-end, runnable. Load → split → train an imbalance-aware ResNet
baseline → emit confusion matrix, per-class precision/recall/F1, and a headline
**macro-F1**. Nothing fancy. This phase exists to produce a real number.

## Known data friction (pre-warned — these trip everyone)

1. **Filter to labeled rows first.** `failureType` is populated on only ~173k of
   ~811k rows. Many rows have empty/`[]` labels — drop them before anything else.
2. **Labels are nested arrays, not strings.** `failureType` and `trainTestLabel`
   come wrapped (e.g. `array(['Center'], dtype=object)`). Unwrap to scalar
   strings. Expect stray empty arrays and at least one near-empty class.
3. **Wafer maps are variable-dimension** 2D arrays with values `{0,1,2}`
   (0 = outside wafer, 1 = pass die, 2 = fail die). They must be resized/padded
   to a fixed input (e.g. 224×224) before a CNN. Preserve the 3-value semantics
   — do **not** naively normalize as if continuous; map to channels or scale
   deliberately and document the choice.
4. **Severe class imbalance.** "none" dominates (~85%+ of labeled). Donut,
   Near-full, Scratch are rare. This is the central modeling challenge and the
   first quality-engineering talking point.

## Tasks

### 1.1 `data.py`
- Load `LSWMD.pkl`, filter to labeled rows, unwrap labels.
- Map 9 class names → integer ids; persist the mapping.
- **Stratified** train/val/test split (use the provided `trainTestLabel` where
  present, but also create a stratified val split from train). Fixed seed.
- Resize/pad maps to fixed size; convert 3-value maps to model input
  (recommend: one-hot the 3 states into 3 channels, or a documented scalar map).
- `torch Dataset`/`DataLoader`; `num_workers` and `batch_size` from config.
- Augmentation appropriate to wafer maps: rotations/flips are valid (a wafer has
  rotational symmetry); do **not** use augmentations that destroy spatial defect
  meaning (no aggressive crops that cut off edge-ring patterns).

### 1.2 Imbalance handling (pick one, document why)
- Class-weighted cross-entropy (weights ∝ inverse frequency), **or**
- Focal loss (γ≈2), **or**
- Targeted oversampling of rare classes via `WeightedRandomSampler`.
Start with class-weighted CE for simplicity; note the alternatives in a comment.
The *reasoning* about escape risk vs. false alarms is the talking point — capture
it in a docstring for reuse in Phase 2 and the writeup.

### 1.3 `model.py`
- `torchvision` ResNet-18 to start (fast iteration on 4090), with a flag to swap
  to ResNet-50. Adapt the first conv if using a non-3-channel input. Replace the
  final FC with a 9-way head. Pretrained weights optional — note that ImageNet
  pretraining transfers weakly to binned wafer maps; try both, report both.

### 1.4 `train.py`
- Standard loop: AdamW, cosine or step LR, early stopping on val macro-F1.
- Mixed precision (`torch.amp`) — works on both Ada and Blackwell.
- Deterministic seeds. Save best checkpoint (portable `.pt`, includes class map).
- Log per-epoch train/val loss and val macro-F1.

### 1.5 `evaluate.py`
- On the held-out **test** split, produce:
  - Confusion matrix (raw + row-normalized), saved as PNG.
  - Per-class precision/recall/F1 table (`sklearn.metrics.classification_report`),
    saved as CSV and printed.
  - **Headline macro-F1** and balanced accuracy (both matter under imbalance;
    plain accuracy is misleading here — say so).

## Acceptance criteria
- [ ] `python -m wafer.train --config configs/baseline.yaml` runs start→finish on
      the 5090 and produces a checkpoint.
- [ ] Same command runs on the 4090 with only a batch-size config change.
- [ ] `python -m wafer.evaluate` emits confusion matrix PNG, per-class CSV, and a
      printed macro-F1 + balanced accuracy.
- [ ] Macro-F1 is a real, non-degenerate number (model is not just predicting
      "none" for everything — verify rare-class recall > 0).
- [ ] Run is reproducible: same seed → same headline metric within noise.

## Anti-goals for Phase 1
No Grad-CAM, no calibration, no demo UI, no ensembles, no hyperparameter sweep
beyond a sane default. Those are later phases. Ship the number first.
