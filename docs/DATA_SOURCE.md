# WM-811K Data Source

## Download

Manually downloaded from: Kaggle — WM-811K wafer map dataset
Date downloaded: 2026-06-18
Download archive size: ~150 MB (compressed)
File: `LSWMD.pkl` (extracted from archive)
Size on disk: 2.10 GB (2.0 GiB) — normal; Kaggle distributes a compressed archive that extracts to ~2 GB
SHA256: <!-- run `sha256sum data/raw/LSWMD.pkl` and fill in -->

## Verification

Output of `python scripts/download_data.py`:

```
Total rows:   811,457
Sample failureType cells (raw): [array([['none']], dtype='<U4'), array([['none']], dtype='<U4'), array([['none']], dtype='<U4')]
Labeled rows: 172,950

Class distribution (failureType):
label
none         147431
Edge-Ring      9680
Edge-Loc       5189
Center         4294
Loc            3593
Scratch        1193
Random          866
Donut           555
Near-full       149

Total classes: 9
```

**Note on file size:** The file is ~2.1 GB in the current pickle format, smaller than the ~3 GB figure cited in older references (which used an earlier serialization). Row counts match the published dataset exactly — the file is complete.

**Note on label structure:** `failureType` cells are 2-D numpy object arrays of shape `(1, 1)` (e.g. `array([['Center']], dtype='<U4')`). The data pipeline uses `np.asarray(val).ravel()[0]` to unwrap labels at any nesting depth.

## GPU verification

Output of `python scripts/download_data.py --check-gpu` (4090 laptop):

```
torch version: 2.8.0+cu128
CUDA available: True
Device: NVIDIA GeForce RTX 4090 Laptop GPU
Compute capability: 8.9
(4090 / Ada sm_89 detected)
Matmul smoke test: torch.Size([256, 256]) — kernel OK
```

## License / Terms

WM-811K is distributed for academic/research use.

Source paper:
> M.-J. Wu, J.-S. R. Jang, J.-L. Chen, "Wafer Map Failure Pattern Recognition
> and Similarity Ranking for Large-Scale Data Sets," IEEE Trans. Semiconductor
> Manufacturing, vol. 28, no. 1, pp. 1–12, 2015.
> https://ieeexplore.ieee.org/document/6932449

This project uses the public WM-811K dataset (binned wafer maps). It does not
contain or use any proprietary fab data or internal inspection imagery.
