# Wafer Defect Classes — Process Mode Interpretations

These interpretations map WM-811K defect patterns to plausible semiconductor
manufacturing process failure modes. They are **illustrative QE reasoning** based
on the spatial geometry of each pattern, not cause-verified claims — this project
uses public binned maps, not fab inspection data with confirmed root causes.

| Defect class | Spatial pattern | Plausible process failure modes |
|---|---|---|
| **Edge-Ring** | Continuous ring at wafer perimeter | Edge-localized etch non-uniformity; CMP over-/under-polish at edge; film deposition rate variation at perimeter (edge effect); edge-exclusion zone contamination |
| **Edge-Loc** | Localized cluster near edge | Edge-bead removal (EBR) incomplete; edge-exclusion-zone proximity; localized edge contamination or handling damage near one sector of the perimeter |
| **Center** | Cluster at wafer center | Spin-coat center-point defect; chuck contact mark; CVD / ALD center-of-wafer flow-rate anomaly; center-symmetric photomask defect |
| **Scratch** | Linear or arc-shaped streak | Wafer handling damage (robot arm, cassette contact); diamond stylus scratch during metrology; mechanical contact during chuck load/unload |
| **Loc** | Small localized cluster (non-edge) | Particle contamination at a specific die site; localized photomask defect; reticle contamination repeating at one field position |
| **Donut** | Ring with clear center | Spin-coat bead or solvent non-uniformity at intermediate radius; possible hot-plate non-uniformity annulus |
| **Random** | Scattered across die | Particle shower (airborne contamination event); electrostatic discharge damage; random film defects |
| **Near-full** | Defects covering nearly all die | Gross process excursion (bulk chemistry failure, severe equipment malfunction); total film delamination or etch stop failure |
| **none** | No pattern / all passing | Normal production wafer; no systematic spatial failure signature |

## Notes for Phase 3 demo and writeup

- The **Scratch** class is the strongest visual signal for Grad-CAM: the model
  should key on the linear/arc spatial feature, and the activation map will
  highlight it clearly.
- **Edge-Ring vs Edge-Loc** are the hardest pair to separate visually and in the
  model (~30/1936 Edge-Ring predicted as Edge-Loc at epoch 27). The process
  distinction matters: Edge-Ring implies a *systematic* perimeter process; Edge-Loc
  implies a *localised* sector-specific failure. Both warrant different corrective
  actions.
- **Near-full** is the most serious escape risk: if a Near-full wafer is
  misclassified as "none", nearly the entire lot is contaminated. The model achieved
  recall 0.87 on this class (30 test samples) — note the small test set when
  discussing confidence in this number.
- These interpretations read to the **Grad-CAM story**: the activation should
  concentrate on the region that defines the spatial pattern (perimeter for
  Edge-Ring, linear streak for Scratch, centre for Center). If it does, the model
  has learned the physically meaningful feature. If it doesn't, that's a finding
  worth noting.
