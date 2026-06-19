# Project Status

Last updated: 2026-06-19

## Phase completion

| Phase | Status | Key output |
|-------|--------|------------|
| 0 — Environment & scaffold | **DONE** | Repo, deps, `scripts/download_data.py` |
| 1 — Baseline pipeline | **DONE** | macro-F1 0.8662, balanced acc 0.9300 |
| 2 — Differentiator layer | **DONE** | ECE 0.0067→0.0034, T=1.10, cost-weighted error 0.0436, Grad-CAM overlays |
| 3 — One-click demo | **DONE** | `python -m wafer.demo` — Gradio on localhost:7860 |
| 4 — Package & position | **DONE** | README.md, resume bullet, slide outline |

## Phase 1 headline numbers (commit d2a67f7)

- **Macro-F1: 0.8662** — test set, ResNet-18 epoch 27/30
- **Balanced accuracy: 0.9300**
- Checkpoint: `outputs/best.pt` (gitignored, reproducible — `python -m wafer.train`)
- Full breakdown: `docs/phase1_results.md`

### Key per-class notes for Phase 2
- **Scratch** precision 0.55 / recall 0.92 → the operating-point / cost-of-quality story
- **Near-full** precision 1.00 / recall 0.87 (30 test samples — weighted CE worked)
- **Edge-Loc/Edge-Ring** confusion → Grad-CAM spatial analysis candidate

## Phase 2 results (commit dc95132 + Phase 2 closure commit)

- [x] `src/wafer/explain.py` — Grad-CAM overlays for 5 classes; `outputs/grad_cam/`
- [x] `docs/process_modes.md` — 9-class defect → process failure mode table
- [x] `src/wafer/calibrate.py` — ECE 0.0067 → 0.0034, T=1.1036; `outputs/reliability_diagram.png`
- [x] Cost-weighted error 0.0436 (59 escapes, 917 false alarms); `outputs/threshold_sensitivity.png`
- [x] `docs/ANALYSIS.md` — full narrative complete with real numbers

## Phase 3 (DONE)

- [x] `src/wafer/demo.py` — Gradio Blocks app
- [x] Loads best.pt + temperature.json at startup
- [x] Auto-extracts 9 example wafer maps (one per class) to outputs/demo_examples/
- [x] Inference: PNG → reverse LUT → 3-channel tensor → calibrated probs + GradCAM
- [x] Output: 3-panel Grad-CAM figure + prediction markdown with process interpretation
- [x] `--share` and `--port` CLI flags

## Phase 4 next steps — Package & position

- [ ] `README.md` — project overview, quickstart, results table, Grad-CAM sample images
- [ ] 90-second screen capture of the Gradio demo
- [ ] Portfolio slide (1 page)
- [ ] Resume bullet

## Resume

```bash
cd /home/waferclassifier/wafer-defect-classifier
source .venv/bin/activate
python -m wafer.demo           # localhost:7860
python -m wafer.demo --share   # public URL
```

## Resuming after a session break

```bash
cd /home/waferclassifier/wafer-defect-classifier
source .venv/bin/activate

# Phase 3: Gradio demo
pip install gradio
python -m wafer.demo
```

## Environment

- Python 3.12, PyTorch 2.8.0+cu128
- Venv: `.venv/` (gitignored)
- 4090 laptop: batch_size=64 | 5090 desktop: batch_size=128
- LSWMD.pkl: `data/raw/LSWMD.pkl` (2.0 GiB, gitignored)
