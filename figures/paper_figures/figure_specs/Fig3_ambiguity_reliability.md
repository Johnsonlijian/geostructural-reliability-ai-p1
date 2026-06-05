# Fig. 3 Critical-State Decision Reliability

## Scientific Question

Where does conformal reliability fail under marginal calibration, where is the remaining ambiguity
physically located, and how should it be converted into an engineering decision object?

## One-Sentence Takeaway

Plain marginal conformal coverage hides regime-level failure; the remaining ambiguity has a
critical-state address, and margin-band conformal calibration converts it into singleton or two-label
decision sets.

## Figure Archetype

Uncertainty / reliability synthesis.

## Panels

| Panel | Job | Evidence |
|---|---|---|
| A | Show coverage failure: plain conformal over-covers the safe/critical bands and under-covers high-risk tails. | `innovation_analysis.json` |
| B | Give the failure a mechanism address using observed frequency vs mechanistic probability and the critical band. | `innovation_analysis.json`, `ambiguity_floor_sensitivity.json` |
| C | Show repair: mechanism-band conformal moves regime coverage toward the target. | `innovation_analysis.json` |
| D | Show decision handoff: singleton, two-label, and critical-band uncertain set rates. | `conformal_decision_metrics.json` |

## Caption First Sentence

Plain marginal coverage hides regime-level failure; mechanism-band conformal calibration repairs the
decision object and exposes critical-band two-label uncertainty.
