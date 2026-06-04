# Fig. 2 Transferable Sufficiency Evidence

## Scientific Question

Do flexible data-driven models add transferable predictive information beyond the effective-stress-derived margin under earthquake-grouped validation?

## One-Sentence Takeaway

Random validation inflates apparent ML skill, while grouped likelihood and residual audits show no statistically supported OOS gain beyond the mechanistic coordinate.

## Figure Archetype

Integrated result figure / evidence-chain figure.

## Panels

| Panel | Job | Evidence |
|---|---|---|
| A | Show random-vs-grouped optimism against physics baseline. | `cetin2018_grouped_validation.json`, `geyin2021_cpt_results.json`, `reliability_upgrade.json` |
| B | Show full-model log-loss deltas relative to margin-only. | `sufficiency_likelihood.json` |
| C | Show residual/raw/groundwater variants after the margin. | `residual_sufficiency_audit.json` |
| D | Show leave-one-region-out transfer. | `cross_region_transfer.json` |

## Caption First Sentence

Transferable performance is bounded by the mechanistic coordinate under event-aware validation.

