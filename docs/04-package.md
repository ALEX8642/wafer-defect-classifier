# Phase 4 — Package & Position (Weeks 11–13)

**Goal:** turn a working project into a persuasive, portable artifact an
opportunist can deploy the moment an opening appears.

## 4.1 README (leads with problem + result, not architecture)
Structure:
1. One-paragraph problem statement (wafer defect classification, why it matters
   for yield/cost-of-quality).
2. **Headline result** up top: macro-F1, balanced accuracy, calibrated.
3. Three example outputs (Grad-CAM overlays with interpretation).
4. The cost-of-quality framing in 3–4 sentences — the differentiator.
5. How to run (one command for the demo).
6. **Honest limitations:** public WM-811K binned maps, not optical/SEM die
   imagery; not the author's fab data; process-mode interpretations are
   illustrative QE reasoning, not cause-verified. Built on public data
   deliberately so it travels with zero IP exposure — frame this as a feature.
7. Reference [1].

## 4.2 90-second screen capture
Pick a sample → prediction + confidence → Grad-CAM → process interpretation →
the one calibration/cost slide. Narrate the quality-engineering reasoning, not
the code. Host the clip (unlisted) and link from README + resume.

## 4.3 Portfolio slide (single)
One slide: problem, headline metric, one Grad-CAM example, one line on the
cost-of-quality framing. Reusable in the existing portfolio deck and as an
interview leave-behind.

## 4.4 Resume bullet (measured, honest)
Template:
> Built a 9-class wafer-defect classifier on public WM-811K (~173k labeled maps);
> achieved macro-F1 **X.XX** with calibrated probabilities (ECE **Y.YY** after
> temperature scaling), Grad-CAM root-cause localization, and a cost-of-quality
> error framework trading defect escapes against false-alarm scrap. Portable
> across CUDA GPUs; one-command local demo.

Fill X/Y from real Phase 1–2 numbers. Never round up or invent.

## 4.5 Positioning notes (for the opportunist)
- Lead every conversation with the **intersection**: floor-level supplier-quality
  experience + a model you built, calibrated, and framed in cost-of-quality
  terms. That overlap is the scarce asset; the F1 is table stakes.
- For Solutions/FDE roles: emphasize the demo, the framing, the ability to
  translate a model into operational decisions a fab cares about.
- For an internal R&D/transfer case: emphasize yield/cost linkage and that it was
  built independently on public data without touching restricted internal data.

## Acceptance criteria
- [ ] README complete with real numbers and the limitations section.
- [ ] 90-second capture recorded and linked.
- [ ] One portfolio slide produced.
- [ ] Resume bullet finalized with actual metrics.
- [ ] Repo is clean, runs from a fresh clone per the README on both machines.
