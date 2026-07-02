# Project Status

Last updated: 2026-06-20

## Phase completion

| Phase | Status | Key output |
|-------|--------|------------|
| 0 — Environment & scaffold | **DONE** | Repo, deps, `scripts/download_data.py` |
| 1 — Baseline pipeline | **DONE** | macro-F1 ~0.87 (seed-dependent), balanced acc ~0.93 |
| 2 — Differentiator layer | **DONE** | ECE calibrated, cost-weighted error, Grad-CAM overlays |
| 3 — One-click demo | **DONE** | `python -m wafer.demo` — Gradio on localhost:7860 |
| 4 — Package & position | **DONE** | README.md, resume bullet, slide outline |

## Current headline numbers

**Checkpoint:** `outputs/best.pt` — epoch 34, val macro-F1 0.9265 (focal loss + CBAM, 5090)
**Test set (TTA×8 + per-class τ):** macro-F1 **0.9157**, balanced accuracy 0.9085
**Calibration:** T=0.6685, ECE 0.0164→0.0031
**Cost-weighted error:** 0.0835 (275 escapes, 137 false alarms, 10:1 assumption)

*Note on cost-weighted error: focal loss without class weights trades escape rate
for false-alarm rate relative to the CE baseline (54 escapes / 990 FA → 0.0442).
The macro-F1 improvement is real (+2pp) but the escape/FA operating point shifted.
See IMPROVEMENTS.md Phase F for full analysis.*

*Note on reproducibility: `set_seed()` now sets `cudnn.deterministic=True` and
`benchmark=False`. Results are locked across sessions with seed=42.*

---

## Improvement phases (post-baseline)

### Phase A — No-retrain metric wins (DONE)

- [x] **A1 — TTA**: `tta_predict()` in evaluate.py; D4 group (8 transforms); `tta: true` in baseline.yaml
- [x] **A2 — Per-class thresholds**: `tune_thresholds()` in calibrate.py → `outputs/thresholds.json`

Measured on CE baseline: macro-F1 0.8952 (5090), Scratch precision 0.55→0.73 (+18 pp).

```bash
python -m wafer.calibrate       # writes temperature.json + thresholds.json
python -m wafer.evaluate        # TTA + thresholds active
```

### Phase B — Test suite (DONE — 18 tests, all green)

- [x] `tests/test_data.py` — encode_map round-trip, PNG round-trip, augmentation shape/values
- [x] `tests/test_calibration.py` — ECE math, temperature scaling direction
- [x] `tests/test_model.py` — output shape, TTA D4 invariance, probability sums
- [x] `pyproject.toml` — pytest config
- [x] `requirements-dev.txt` — pytest>=8.0

```bash
pip install pytest && pytest tests/ -v      # no GPU or LSWMD.pkl required
```

### Phase C — Focal loss retraining (NEGATIVE RESULT → FIXED, see Phase F)

**First attempt failed:** Focal + class weights combined → val macro-F1 0.7303 (double-penalization bug).
**Corrected focal loss** (no class weights) validated in Phase F combined with CBAM.
Full experiment post-mortem in `docs/IMPROVEMENTS.md` Phase C section.

### Phase D — Grad-CAM++ overlays (DONE)

- [x] `GradCAMPlusPlus` class in explain.py; second-order gradient weighting
- [x] `--method gradcam|gradcampp` CLI flag; default gradcampp
- [x] demo.py switched to GradCAMPlusPlus
- [x] 5 overlay PNGs regenerated in `outputs/grad_cam/`

```bash
python -m wafer.explain --method gradcampp    # regenerate overlay PNGs
```

### Phase E1 — Deterministic training mode (DONE)

`set_seed()` in `train.py` now sets `torch.backends.cudnn.deterministic = True`
and `torch.backends.cudnn.benchmark = False`. Results locked across GPU sessions.

### Phase F — Focal loss (corrected) + CBAM attention (DONE)

**What changed:** Combined two improvements in one training run:
- Corrected focal loss (γ=2.0, no class weights) — validated after Phase C post-mortem
- CBAM attention after each ResNet stage (43.9k extra parameters, 0.4% overhead)

**Results (5090, 40 epochs):**

| Metric | CE baseline | Focal + CBAM |
|--------|-------------|-------------|
| Val macro-F1 | 0.8713 | **0.9265** |
| Test macro-F1 | 0.8952 | **0.9157** |
| Loc F1 | 0.79 | **0.84** (+5pp) |
| Scratch F1 | 0.82 | **0.86** (+4pp) |
| Random F1 | 0.87 | **0.91** (+4pp) |
| Edge-Loc F1 | 0.87 | **0.89** (+2pp) |

Full per-class breakdown and narrative in `docs/IMPROVEMENTS.md` Phase F section.

```bash
# To replicate (configs/baseline.yaml now defaults to this recipe: loss focal, cbam true, 40 epochs):
python -m wafer.train && python -m wafer.calibrate && python -m wafer.evaluate
```

### Phase S — Semi-supervised pseudo-labeling (DONE — experiment; Phase F remains best)

Pseudo-labeled 638k unlabeled WM-811K maps with focal+CBAM teacher at 0.95 confidence.
Filtered "none" pseudo-labels (514k maps) to keep only 29,100 defect-class additions.

**Result:** test macro-F1 0.9085 vs Phase F 0.9157 — slight regression across all tail
classes. Root cause: ~5% noise at 0.95 threshold degraded Scratch/Loc/Random recall.
Phase F checkpoint restored as the production model. Full analysis in IMPROVEMENTS.md Phase S.

### Narrative documentation (DONE through Phase S)

- [x] `docs/IMPROVEMENTS.md` — Phases A–F + Phase S (experiment)
- [x] `docs/ML_PRIMER.md` — CNN, loss, calibration, Grad-CAM for manufacturing audience

---

## Running the project

```bash
cd /home/waferclassifier/wafer-defect-classifier
source .venv/bin/activate

python -m wafer.train          # ~15 min on 5090 (focal+CBAM, 40 epochs)
python -m wafer.calibrate      # temperature + thresholds
python -m wafer.evaluate       # test metrics
python -m wafer.pseudo_label   # generate pseudo-labels from unlabeled maps
python -m wafer.demo           # Gradio demo → localhost:7860
python -m wafer.demo --share   # public URL
```

**Important:** Always activate `.venv` before running. System python3 does not have
wafer installed. Or use `.venv/bin/python -m wafer.<module>` directly.

**GPU note:** When running training as a background task from the root shell,
confirm GPU is being used: `nvidia-smi` should show the training PID with non-zero
memory usage within 30 seconds of start. If GPU shows 0% utilization,
the shell environment may lack CUDA paths — run in a foreground terminal instead.

## Environment

- Python 3.12, PyTorch 2.8.0+cu128
- Venv: `.venv/` (gitignored)
- 4090 laptop: batch_size=64 | 5090 desktop: batch_size=128
- LSWMD.pkl: `data/raw/LSWMD.pkl` (2.0 GiB, gitignored)

## Push to GitHub

Root does not have GitHub credentials. In your alex8642 terminal:
```bash
cd /home/waferclassifier/wafer-defect-classifier
gh auth setup-git
git push
```
