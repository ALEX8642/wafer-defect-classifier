# Project Status

Last updated: 2026-06-19

## Phase completion

| Phase | Status | Key output |
|-------|--------|------------|
| 0 — Environment & scaffold | **DONE** | Repo, deps, `scripts/download_data.py` |
| 1 — Baseline pipeline | **DONE** | macro-F1 0.8662, balanced acc 0.9300 |
| 2 — Differentiator layer | **IN PROGRESS** | — |
| 3 — One-click demo | Not started | — |
| 4 — Package & position | Not started | — |

## Phase 1 headline numbers (commit d2a67f7)

- **Macro-F1: 0.8662** — test set, ResNet-18 epoch 27/30
- **Balanced accuracy: 0.9300**
- Checkpoint: `outputs/best.pt` (gitignored, reproducible — `python -m wafer.train`)
- Full breakdown: `docs/phase1_results.md`

### Key per-class notes for Phase 2
- **Scratch** precision 0.55 / recall 0.92 → the operating-point / cost-of-quality story
- **Near-full** precision 1.00 / recall 0.87 (30 test samples — weighted CE worked)
- **Edge-Loc/Edge-Ring** confusion → Grad-CAM spatial analysis candidate

## Phase 2 remaining work (`docs/02-differentiator.md`)

- [ ] `src/wafer/explain.py` — Grad-CAM on last ResNet conv block; save overlays for ≥3 classes
- [ ] `docs/process_modes.md` — table: ≥3 defect classes → process failure modes
- [ ] `src/wafer/calibrate.py` — reliability diagram, ECE, temperature scaling before/after
- [ ] Cost-weighted error metric — escape vs false-alarm framing with stated cost assumptions
- [ ] `docs/ANALYSIS.md` — full narrative for hiring/transfer audience

## Resuming after a session break

```bash
cd /home/waferclassifier/wafer-defect-classifier
source .venv/bin/activate

# Verify checkpoint is present
ls -lh outputs/best.pt

# Re-run evaluate if needed
python -m wafer.evaluate

# Continue Phase 2
# edit src/wafer/explain.py and src/wafer/calibrate.py
```

## Environment

- Python 3.12, PyTorch 2.8.0+cu128
- Venv: `.venv/` (gitignored)
- 4090 laptop: batch_size=64 | 5090 desktop: batch_size=128
- LSWMD.pkl: `data/raw/LSWMD.pkl` (2.0 GiB, gitignored)
