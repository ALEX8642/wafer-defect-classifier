# Phase 0 — Environment & Data Acquisition

**Goal:** a reproducible environment that runs identically on the 4090 laptop and
the 5090 desktop, plus the WM-811K dataset on disk and verified.

## Tasks

### 0.1 Repo scaffold
```
wafer-defect-classifier/
├── PLAN.md
├── docs/
├── src/wafer/
│   ├── __init__.py
│   ├── config.py        # dataclass config, CLI/env override, device select
│   ├── data.py          # Phase 1
│   ├── model.py         # Phase 1
│   ├── train.py         # Phase 1
│   ├── evaluate.py      # Phase 1
│   ├── explain.py       # Phase 2
│   └── calibrate.py     # Phase 2
├── scripts/
│   └── download_data.py
├── configs/
│   └── baseline.yaml
├── tests/
├── requirements.txt
├── pyproject.toml
└── README.md            # Phase 4
```

### 0.2 Python & dependency pinning
- Python 3.11.
- Use a venv or uv. Pin everything in `requirements.txt`.
- Core: `torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`,
  `matplotlib`, `pyyaml`, `tqdm`. (Phase 2 adds `grad-cam` or a hand-rolled
  hook; Phase 3 adds `gradio` or `fastapi`+`uvicorn`.)

### 0.3 Blackwell CUDA caveat (the one gotcha that costs an hour)
The 5090 is `sm_120` (Blackwell). Stable cu121 PyTorch wheels will **not**
initialize it. Install a CUDA 12.8+ build:
```
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
The 4090 (`sm_89`, Ada) runs on cu121 or cu128 fine, so a cu128 build is the
common denominator — use it on both for portability. Verify:
```python
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))   # (12,0) on 5090, (8,9) on 4090
```
If `cuda.is_available()` is True but a kernel launch errors with "no kernel
image is available for execution on the device," the wheel lacks `sm_120` —
reinstall the cu128/nightly build.

### 0.4 Device-agnostic config
`config.py` selects `cuda` if available else `cpu`, overridable via
`WAFER_DEVICE` env var or `--device` CLI flag. No hardcoded paths: data root,
output dir, batch size, and num_workers all come from `configs/baseline.yaml`
with CLI override. Batch size should be a config value so the 4090 (24GB) and
5090 (32GB) can differ without code changes.

### 0.5 Data acquisition
WM-811K is distributed as `LSWMD.pkl` (a single pickle, ~3GB) via the MIR
Corpus / Kaggle mirrors.
- `scripts/download_data.py` should accept a path, verify the file exists and
  its size/row count, and print a class-distribution summary.
- Do **not** commit the data. Add `data/` to `.gitignore`.
- Record the exact source URL used in `docs/DATA_SOURCE.md` for reproducibility.

## Acceptance criteria
- [ ] `torch.cuda.get_device_name(0)` prints the correct GPU on both machines.
- [ ] A trivial tensor `.cuda()` op + matmul runs without kernel-image errors.
- [ ] `LSWMD.pkl` loads; script prints total rows (~811k) and labeled rows (~173k).
- [ ] Class-distribution summary printed (expect heavy "none" dominance).
- [ ] Repo scaffold committed; data gitignored.
