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

**Checkpoint:** `outputs/best.pt` — epoch 27, val macro-F1 0.8627  
**Test set (TTA×8 + per-class τ):** macro-F1 **0.8811**, balanced accuracy 0.9213  
**Calibration:** T=1.1344, ECE 0.0098→0.0033  
**Cost-weighted error:** 0.0472 (53 escapes, 1101 false alarms, 10:1 assumption)

*Note on reproducibility: CUDA floating-point non-determinism means training results
vary slightly between sessions even with seed=42. The original Phase 1 run achieved
val F1 0.8732 / test macro-F1 0.8662; the current checkpoint has val F1 0.8627.
To lock results across runs, add `torch.backends.cudnn.deterministic = True` and
`torch.backends.cudnn.benchmark = False` to `train.py` — accepted ~5% speed cost.*

---

## Improvement phases (post-baseline)

### Phase A — No-retrain metric wins (DONE)

- [x] **A1 — TTA**: `tta_predict()` in evaluate.py; D4 group (8 transforms); `tta: true` in baseline.yaml
- [x] **A2 — Per-class thresholds**: `tune_thresholds()` in calibrate.py → `outputs/thresholds.json`

Measured: macro-F1 0.8811, Scratch precision 0.55→0.73 (+18 pp).

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

### Phase C — Focal loss retraining (NEGATIVE RESULT — documented)

**What happened:** Focal loss with class weights combined (first attempt) peaked at
val macro-F1 0.7303 after 40 epochs — significantly worse than CE baseline 0.8627.
Root cause: double-penalization (class weights + focal modulator both address
imbalance; combining them destabilises early training).

**Fix in code:** `FocalLoss` in `train.py` no longer accepts class weights — the
focal modulator handles imbalance directly. This corrected version is ready to run.

**To try corrected focal loss retraining:**
```bash
# Edit configs/baseline.yaml: loss: focal, num_epochs: 40, patience: 10
python -m wafer.train
python -m wafer.calibrate
python -m wafer.evaluate
```

**Expected:** With corrected implementation, focal should converge in ~30–40 epochs
to at or above the CE baseline. Whether it beats TTA+thresholds is an open question.

Full experiment post-mortem in `docs/IMPROVEMENTS.md` Phase C section.

### Phase D — Grad-CAM++ overlays (DONE)

- [x] `GradCAMPlusPlus` class in explain.py; second-order gradient weighting
- [x] `--method gradcam|gradcampp` CLI flag; default gradcampp
- [x] demo.py switched to GradCAMPlusPlus
- [x] 5 overlay PNGs regenerated in `outputs/grad_cam/`

```bash
python -m wafer.explain --method gradcampp    # regenerate overlay PNGs
```

### Narrative documentation (DONE)

- [x] `docs/IMPROVEMENTS.md` — full story: Phase A (TTA + thresholds), Phase B (tests),
  Phase C (focal loss experiment + post-mortem), Phase D (Grad-CAM++)
- [x] `docs/ML_PRIMER.md` — CNN, loss, calibration, Grad-CAM for manufacturing audience

---

## Running the project

```bash
cd /home/waferclassifier/wafer-defect-classifier
source .venv/bin/activate

python -m wafer.train       # ~10 min on 4090 (uses .venv/bin/python, not system python)
python -m wafer.calibrate   # temperature + thresholds
python -m wafer.evaluate    # test metrics
python -m wafer.demo        # Gradio demo → localhost:7860
python -m wafer.demo --share  # public URL
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
