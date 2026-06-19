# Phase 3 — One-Click Demo (Weeks 8–10)

**Goal:** an interviewer or hiring manager clicks once and sees the whole story.
Self-contained, runs locally, no setup ceremony. Demoable in under two minutes.

## Design
Single-page app. User picks a sample wafer map (ship a curated set of test
examples covering each class) or uploads one. Output panel shows, for that map:

1. Predicted class + confidence (calibrated probability from Phase 2).
2. Top-3 class probabilities.
3. Grad-CAM overlay (where the model is keying).
4. The process-mode interpretation line for the predicted class.
5. A one-line cost-of-quality note if the prediction is a high-escape-risk class.

## Implementation choice
- **Gradio** if you want it done in a day (recommended for speed; it gives
  upload, gallery, and image output for free).
- **FastAPI + minimal single-page frontend** if you want it to look more like a
  product / bespoke (more polish, more time). Given your stack either is trivial;
  default to Gradio unless you specifically want the FastAPI look on your resume.
- Must run with one command (`python -m wafer.demo`) and load the portable
  checkpoint from Phase 1. Device-agnostic (runs on the 4090 laptop for live
  demos away from the desktop).

## Acceptance criteria
- [ ] One command launches the app locally.
- [ ] Curated sample wafers for all 9 classes are selectable.
- [ ] For any input: class + calibrated confidence + top-3 + Grad-CAM overlay +
      process interpretation all render together.
- [ ] Runs on the 4090 laptop (portable for in-person/remote demos).
- [ ] Cold-start to first prediction < ~10s; per-inference interactive.

## Anti-goals
No auth, no database, no deployment to cloud, no multi-user concerns. It's a
local demo artifact, not a SaaS. Don't over-build the frontend.
