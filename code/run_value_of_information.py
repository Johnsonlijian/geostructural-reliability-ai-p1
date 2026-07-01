"""Value of information / measurement value for liquefaction screening — v2 (debiased).

Cold-read fixes (Cycle 1): value of information cannot be negative for a Bayes-optimal agent
(Good / Blackwell); the earlier "negative second-measurement VoI" was an artefact of (a) an in-sample
min-over-threshold operating point and (b) an overfit, uncalibrated multivariate model. This version
(i) sets the decision threshold from THEORY — the Bayes rule p* = 1/(1+R), R = C_FN/C_FP — not from the
data; (ii) uses out-of-fold, calibrated logistic posteriors (1-D mechanistic margin vs the full
standard-predictor set, both naturally calibrated); (iii) reports the in-sample optimism gap explicitly.
The honest question becomes: do the extra standard predictors add ANY decision value beyond the
mechanistic margin? (Expected: no — Δ ≈ 0, consistent with the AUC results; NOT negative.)

NOT a site-investigation prescription. This concerns extra PREDICTORS on one labelled record and says
NOTHING about multi-method investigation (profiling, layer/seam detection, redundancy), which remain
standard practice. Cross-modality numbers are per arm (different sites/responses; shared earthquakes).

VoI is in units of C_FP (with C_FN = R*C_FP): expected screening-cost reduction per site from resolving
the test, relative to the prior-only decision min(R*pi, 1-pi). Outputs value_of_information.json. Seed 0.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_predictability_ceiling import load_spt, load_cpt, load_vs

BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, "data", "processed")
SEED = 0
N_BOOT = 800
PI_GRID = [round(x, 3) for x in np.linspace(0.05, 0.60, 12)]
R_GRID = [2, 5, 10, 20]
OPERATING = [(0.20, 5), (0.20, 10), (0.35, 10)]


def oof_logit(X, y, groups):
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    cv = list(GroupKFold(min(5, len(np.unique(groups)))).split(X, y, groups))
    return cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                             X, y, cv=cv, method="predict_proba")[:, 1]


def _rates(y, p, thr):
    y = np.asarray(y, int); p = np.asarray(p, float)
    pos, neg = y == 1, y == 0
    fnr = float(np.mean(p[pos] < thr)) if pos.any() else 0.0
    fpr = float(np.mean(p[neg] >= thr)) if neg.any() else 0.0
    return fnr, fpr


def voi_oof(y, p, pi, R):
    """Bayes-threshold (theory, not in-sample) VoI on out-of-fold calibrated posteriors."""
    thr = 1.0 / (1.0 + R)
    fnr, fpr = _rates(y, p, thr)
    return min(R * pi, 1.0 - pi) - (R * pi * fnr + (1.0 - pi) * fpr)


def voi_naive(y, p, pi, R):
    """In-sample min-over-threshold VoI (for the optimism-gap comparison only)."""
    y = np.asarray(y, int)
    if len(np.unique(y)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(y, p)
    return max(min(R * pi, 1.0 - pi) - float(np.min(R * pi * (1.0 - tpr) + (1.0 - pi) * fpr)), 0.0)


def voi_oof_ci(y, p, groups, pi, R, n_boot=N_BOOT, seed=SEED):
    y = np.asarray(y, int); p = np.asarray(p, float); groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    gids = np.unique(groups); mem = {g: np.where(groups == g)[0] for g in gids}
    bo = []
    for _ in range(n_boot):
        idx = np.concatenate([mem[g] for g in rng.choice(gids, len(gids), replace=True)])
        if len(np.unique(y[idx])) > 1:
            bo.append(voi_oof(y[idx], p[idx], pi, R))
    return {"voi": round(voi_oof(y, p, pi, R), 4),
            "lo": round(float(np.percentile(bo, 2.5)), 4),
            "hi": round(float(np.percentile(bo, 97.5)), 4)}


def delta_ci(y, p_full, p_margin, groups, pi, R, n_boot=N_BOOT, seed=SEED):
    """Δ = VoI(full predictors) − VoI(margin), out-of-fold. Expected ≈ 0 (not negative)."""
    y = np.asarray(y, int); groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    gids = np.unique(groups); mem = {g: np.where(groups == g)[0] for g in gids}
    bo = []
    for _ in range(n_boot):
        idx = np.concatenate([mem[g] for g in rng.choice(gids, len(gids), replace=True)])
        if len(np.unique(y[idx])) > 1:
            bo.append(voi_oof(y[idx], p_full[idx], pi, R) - voi_oof(y[idx], p_margin[idx], pi, R))
    d = voi_oof(y, p_full, pi, R) - voi_oof(y, p_margin, pi, R)
    return {"delta_full_minus_margin": round(d, 4),
            "lo": round(float(np.percentile(bo, 2.5)), 4),
            "hi": round(float(np.percentile(bo, 97.5)), 4),
            "P(full_adds_value)": round(float(np.mean(np.array(bo) > 0)), 3)}


def analyze(name, d):
    y = np.asarray(d["y"], int); groups = np.asarray(d["groups"])
    p_margin = oof_logit(d["margin_s"], y, groups)
    p_full = oof_logit(d["X_full"], y, groups)
    operating = {f"pi={pi},R={R}": voi_oof_ci(y, p_margin, groups, pi, R) for (pi, R) in OPERATING}
    optimism = {f"pi={pi},R={R}": {
        "voi_oof": round(voi_oof(y, p_margin, pi, R), 4),
        "voi_in_sample": round(voi_naive(y, p_margin, pi, R), 4),
        "optimism_gap": round(voi_naive(y, p_margin, pi, R) - voi_oof(y, p_margin, pi, R), 4)}
        for (pi, R) in OPERATING}
    second = {f"pi={pi},R={R}": delta_ci(y, p_full, p_margin, groups, pi, R) for (pi, R) in OPERATING}
    surface = {str(R): [{"pi": pi, "voi": round(voi_oof(y, p_margin, pi, R), 4)} for pi in PI_GRID]
               for R in R_GRID}
    return {"n": int(len(y)), "n_earthquakes": int(len(np.unique(groups))),
            "base_rate": round(float(np.mean(y)), 4),
            "margin_voi_oof_at_operating_points": operating,
            "in_sample_optimism": optimism,
            "extra_predictors_added_value": second,
            "margin_voi_surface": surface}


def main():
    out = {"method": ("Out-of-fold calibrated logistic posteriors; Bayes threshold p*=1/(1+R) from theory "
                      "(no in-sample thresholding); VoI in units of C_FP. Δ tests whether the FULL standard "
                      "predictor set adds decision value beyond the 1-D mechanistic margin."),
           "safety_note": ("Δ concerns extra PREDICTORS on one labelled record; it is NOT a statement about "
                           "multi-method site investigation (profiling, layer/seam detection, redundancy), "
                           "which remain standard practice and are not substitutable."),
           "caveats": ("VoI uses class-conditional rates so π can be swept, but covariate shift across "
                       "populations is untested; π and R are decision inputs, reported as a sensitivity not a "
                       "per-site rule; cross-modality arms are different sites/responses with shared earthquakes."),
           "seed": SEED, "n_boot": N_BOOT}
    loaders = [("SPT_Cetin2018", load_spt), ("CPT_Geyin2021", load_cpt), ("Vs_Kayen2013", load_vs)]
    present = []
    for name, ld in loaders:
        r = ld()
        if r is None:
            continue
        _, d = r
        out[name] = analyze(name, d)
        present.append(name)
        a = out[name]
        print("=" * 80); print(f"{name} (n={a['n']}, EQ={a['n_earthquakes']}, base={a['base_rate']})")
        for k, v in a["margin_voi_oof_at_operating_points"].items():
            g = a["in_sample_optimism"][k]["optimism_gap"]
            print(f"  margin VoI {k:<14} {v['voi']:.3f} [{v['lo']:.3f},{v['hi']:.3f}]  (in-sample optimism +{g:.3f})")
        s = a["extra_predictors_added_value"]["pi=0.2,R=10"]
        print(f"  extra predictors add value (π=0.2,R=10): Δ={s['delta_full_minus_margin']:+.3f} "
              f"[{s['lo']:+.3f},{s['hi']:+.3f}] P(adds>0)={s['P(full_adds_value)']:.2f}")
    path = os.path.join(PROC, "value_of_information.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
