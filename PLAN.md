# Wafer Defect Classifier — Master Plan

A portfolio artifact: 9-class wafer-map defect classification on the public
WM-811K dataset, built to demonstrate the intersection of manufacturing /
supplier-quality judgment and hands-on ML. Audience-agnostic (external
Solutions/Forward-Deployed Engineer applications **and** internal R&D transfer
case). The persuasive power comes from domain framing — cost-of-quality error
analysis and root-cause localization — not from model size.

## Operating principle for execution

Execute **one phase at a time**. Each phase has its own doc in `docs/` with
acceptance criteria. Do not start a phase until the prior phase's criteria are
met and committed. Do not gold-plate: a working baseline beats a clever
unfinished pipeline. Resist ensembles, NAS, multi-node training, and any
showpiece that does not increase the artifact's clarity or persuasiveness.

## Phase index

| Phase | Doc | Outcome | Target |
|-------|-----|---------|--------|
| 0 | `docs/00-environment.md` | Reproducible env across 4090/5090; data acquired | Week 1 |
| 1 | `docs/01-baseline.md` | End-to-end training → confusion matrix + per-class metrics + macro-F1 | Weeks 1–3 |
| 2 | `docs/02-differentiator.md` | Grad-CAM localization + calibration + cost-of-quality framing | Weeks 4–7 |
| 3 | `docs/03-demo.md` | One-click local demo (upload/pick → class + confidence + Grad-CAM + process interpretation) | Weeks 8–10 |
| 4 | `docs/04-package.md` | README, 90-sec capture, portfolio slide, resume bullet | Weeks 11–13 |

## Hardware policy (portability)

Design and code target **CPU + single CUDA GPU**, parameterized by device.
The same code must run on:

- **4090 laptop** — portable dev/debug, small experiments.
- **5090 desktop** — primary full-training box (fastest iteration, 32GB ample).

**GB10 (Grace-Blackwell) nodes are explicitly OUT OF SCOPE for training here.**
Rationale: model quality ("best results") is set by data handling, loss,
augmentation, and architecture — not by which GPU computes the gradients. A
converged ResNet is identical regardless of trainer. The GB10s' advantage is
128GB unified memory, which serves large-model *inference* (existing vLLM/MiniMax
duty), not a small conv net on 224×224 maps. On this compute-bound small-tensor
workload the 5090's raw throughput wins per-step; the GB10 memory is wasted.
Keep the GB10s on serving duty. They re-enter scope only if the project later
adds a large optical-defect vision model or a VLM fine-tune.

Portability requirements baked into every phase:
- No hardcoded device; select `cuda` if available else `cpu`, allow override.
- No hardcoded absolute paths; use a config file / env vars / CLI args.
- Pin dependencies; document the Blackwell CUDA caveat (see `docs/00-environment.md`).
- Deterministic seeds; checkpoints portable between machines.

## Dataset (summary; full detail in Phase 0/1 docs)

WM-811K: ~811,457 wafer maps; only ~172,950 carry a `failureType` label across
9 classes (Center, Donut, Edge-Loc, Edge-Ring, Loc, Random, Scratch, Near-full,
none). This is **spatial defect-pattern classification on binned wafer maps**,
not optical/SEM die imagery — state this boundary explicitly in all writeups so
the artifact is not mistaken for inspection-grade optical CV. Built on public
data deliberately, so it travels with zero IP exposure.

## Reference

[1] M.-J. Wu, J.-S. R. Jang, J.-L. Chen, "Wafer Map Failure Pattern Recognition
and Similarity Ranking for Large-Scale Data Sets," IEEE Trans. Semiconductor
Manufacturing, vol. 28, no. 1, pp. 1–12, 2015.
https://ieeexplore.ieee.org/document/6932449
