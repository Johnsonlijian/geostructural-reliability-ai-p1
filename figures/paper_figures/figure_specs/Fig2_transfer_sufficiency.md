# Fig. 2 Operational Non-Inferiority Evidence

## Scientific Question

Do flexible data-driven models provide a practically meaningful transferable gain beyond the
effective-stress-derived margin under earthquake-grouped validation?

## One-Sentence Takeaway

Random validation inflates apparent ML skill, while grouped likelihood, residual audits, and
practical-improvement checks bound the gain available beyond the mechanistic coordinate.

## Figure Archetype

Integrated result figure / evidence-chain figure.

## Panels

| Panel | Job | Evidence |
|---|---|---|
| A | Show random-vs-grouped optimism against physics baseline. | `cetin2018_grouped_validation.json`, `geyin2021_cpt_results.json`, `reliability_upgrade.json` |
| B | Show full-model log-loss deltas relative to margin-only. | `sufficiency_likelihood.json` |
| C | Show residual/raw/groundwater variants after the margin. | `residual_sufficiency_audit.json` |
| D | Show which practical gains are excluded and which residual boundary remains. | `practical_equivalence_audit.json` |

## Caption First Sentence

Operational non-inferiority is supported under event-aware validation, with the residual boundary
made explicit.
