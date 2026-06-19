# Phase 1 Results

**Model:** ResNet-18, from scratch (pretrained=false), one-hot 3-channel encoding, 224×224  
**Training:** AdamW lr=1e-3 wd=1e-4, CosineAnnealingLR, class-weighted CE, patience=7  
**Hardware:** NVIDIA GeForce RTX 4090 Laptop GPU (sm_89, cu128), batch_size=64  
**Best checkpoint:** epoch 27 / 30, val macro-F1 = 0.8732  
**Split:** stratified 70 / 10 / 20 — train 121,065 / val 17,295 / test 34,590  
**Seed:** 42

## Test-set headline metrics

| Metric | Value |
|---|---|
| **Macro-F1** | **0.8662** |
| Balanced accuracy | 0.9300 |
| Plain accuracy | 0.97 (suppressed — misleading under 85 % imbalance) |

## Per-class breakdown

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Center | 0.85 | 0.97 | 0.90 | 859 |
| Donut | 0.87 | 0.90 | 0.88 | 111 |
| Edge-Loc | 0.73 | 0.92 | 0.82 | 1038 |
| Edge-Ring | 0.98 | 0.98 | 0.98 | 1936 |
| Loc | 0.64 | 0.88 | 0.74 | 718 |
| Near-full | 1.00 | 0.87 | 0.93 | 30 |
| Random | 0.79 | 0.97 | 0.87 | 173 |
| Scratch | 0.55 | 0.92 | 0.69 | 239 |
| none | 1.00 | 0.97 | 0.98 | 29486 |

## Notable patterns

**Scratch** (precision 0.55, recall 0.92): the model aggressively catches real Scratch
defects but raises ~1.8× as many false alarms as true detections. 154 `none` wafers
predicted as Scratch in the test set. This is the central operating-point trade-off:
high recall protects against escapes; low precision drives unnecessary hold/inspection
cost. Phase 2 will quantify this via the cost-of-quality framework.

**Near-full** (only 30 test samples): precision 1.00, recall 0.87. The weighted CE loss
handled the rarest class (149 training samples) without any oversampling.

**Edge-Loc / Edge-Ring confusion**: 30 Edge-Ring wafers predicted as Edge-Loc (1.5 %
of Edge-Ring test set). Both classes are perimeter-localised — the hardest spatial
boundary in the taxonomy. Grad-CAM in Phase 2 will show whether the model is keying
on the right spatial region for each.

## Reproducibility

Artifacts in `outputs/` (gitignored, reproducible):
- `best.pt` — checkpoint (epoch 27, val macro-F1 0.8732)
- `class_map.json` — class name → index mapping
- `per_class_metrics.csv` — per-class precision/recall/F1
- `confusion_matrix.png` — raw counts + row-normalised

To reproduce: `python -m wafer.train` then `python -m wafer.evaluate` with seed=42.
