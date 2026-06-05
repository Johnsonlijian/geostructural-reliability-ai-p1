# Fig. 2 Measured Bound And Excluded-Gain Audit

## Scientific Question

How much practically meaningful transferable gain can be excluded beyond the effective-stress-derived
margin under earthquake-grouped validation?

## One-Sentence Takeaway

Random validation inflates apparent ML skill, while grouped likelihood, residual audits, and
practical-improvement checks define a measured upper bound on recoverable gain from the tested
public predictors.

## Figure Archetype

Integrated result figure / evidence-chain figure.

## Panels

| Panel | Job | Evidence |
|---|---|---|
| A | Show why grouped validation defines the measured bound by contrasting 100-repeat random optimism against the physics baseline. | `random_split_sensitivity.json`, `reliability_upgrade.json` |
| B | Show that full-feature likelihood gains are not supported. | `sufficiency_likelihood.json` |
| C | Show that residual/raw/groundwater terms test what the margin leaves behind. | `residual_sufficiency_audit.json` |
| D | Present a 2 x 3 excluded-gain matrix; SPT residual log-loss gain is the explicit caveat. | `practical_equivalence_audit.json` |

## Caption First Sentence

The practical-improvement audit defines a measured upper bound on recoverable ML gain, with the SPT
residual boundary made explicit.
