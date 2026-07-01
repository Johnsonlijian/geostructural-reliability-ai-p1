"""Tuned-ML challenge: does HYPERPARAMETER-TUNED ML beat the mechanistic margin? (P1 upgrade)

A cold-read flagged that the sufficiency claim ("no flexible ML beats the 1-D critical-state margin")
used untuned models, so "ML loses" might just measure modest ML. This script gives ML its best shot:
properly regularised / tuned challengers selected by NESTED earthquake-grouped cross-validation
(inner GroupKFold for hyperparameters, outer GroupKFold for honest out-of-fold prediction), then a
paired cluster-bootstrap comparison against the zero-shot mechanistic margin. If even tuned ML does not
exceed the margin under honest grouping, the critical-state-sufficiency claim is robust to ML capacity.

Uses all cores (GridSearchCV n_jobs=-1). Outputs data/processed/tuned_ml_challenge.json. Seed 0.
"""

import json
import os
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from geoliq.reliability import bootstrap_auc_ci, paired_auc_diff_ci
from run_predictability_ceiling import load_spt, load_cpt, load_vs

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, "data", "processed")
SEED = 0
N_BOOT = 1000

GRIDS = {
    "logreg_full": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                    {"logisticregression__C": [0.01, 0.1, 1.0, 10.0]}),
    "knn": (make_pipeline(StandardScaler(), KNeighborsClassifier()),
            {"kneighborsclassifier__n_neighbors": [5, 9, 15, 21, 31],
             "kneighborsclassifier__weights": ["uniform", "distance"]}),
    "hgb": (HistGradientBoostingClassifier(random_state=SEED),
            {"learning_rate": [0.03, 0.1], "max_depth": [2, 3],
             "l2_regularization": [0.0, 1.0], "max_iter": [200, 400]}),
    "rf": (RandomForestClassifier(random_state=SEED, n_jobs=1),
           {"n_estimators": [400], "max_depth": [3, 5, None], "min_samples_leaf": [1, 5]}),
}


def nested_oof(est, grid, X, y, groups, n_splits):
    """Outer-GroupKFold OOF probabilities; inner-GroupKFold GridSearch tuning per outer fold."""
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    groups = np.asarray(groups)
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        inner = GroupKFold(min(4, len(np.unique(groups[tr]))))
        gs = GridSearchCV(est, grid, scoring="roc_auc", cv=inner, n_jobs=-1, refit=True)
        gs.fit(X[tr], y[tr], groups=groups[tr])
        oof[te] = gs.predict_proba(X[te])[:, 1]
    return oof


def margin_oof(margin_s, y, groups, n_splits):
    cv = list(GroupKFold(n_splits).split(np.asarray(margin_s).reshape(-1, 1), y, groups))
    return cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                             np.asarray(margin_s, float).reshape(-1, 1), y, cv=cv,
                             method="predict_proba")[:, 1]


def analyze(name, d):
    y = np.asarray(d["y"], int)
    groups = np.asarray(d["groups"])
    n_splits = min(5, len(np.unique(groups)))
    p_margin = margin_oof(d["margin_s"], y, groups, n_splits)
    auc_margin = bootstrap_auc_ci(y, p_margin, groups=groups, seed=SEED, n_boot=N_BOOT)

    ml = {}
    for mname, (est, grid) in GRIDS.items():
        p = nested_oof(est, grid, d["X_full"], y, groups, n_splits)
        ml[mname] = {"oof": p, "auc": bootstrap_auc_ci(y, p, groups=groups, seed=SEED, n_boot=N_BOOT)}

    best = max(ml, key=lambda m: ml[m]["auc"]["auc"])
    dpair = paired_auc_diff_ci(y, ml[best]["oof"], p_margin, groups=groups, seed=SEED, n_boot=N_BOOT)
    beats = dpair["lo"] > 0
    return {
        "n": int(len(y)), "n_earthquakes": int(len(np.unique(groups))),
        "margin_grouped_auc": {k: round(v, 4) for k, v in auc_margin.items()},
        "tuned_ml_grouped_auc": {m: {k: round(v, 4) for k, v in ml[m]["auc"].items()} for m in ml},
        "best_tuned_ml": best,
        "best_tuned_ml_minus_margin_AUC": {k: round(v, 4) for k, v in dpair.items()},
        "tuned_ml_beats_margin": bool(beats),
        "margin_sufficient_vs_tuned_ml": bool(not beats),
    }


def main():
    out = {"method": ("Nested earthquake-grouped CV (inner GroupKFold GridSearch, outer GroupKFold OOF) "
                      "for tuned logreg/kNN/HGB/RF on the full predictor set; paired cluster-bootstrap "
                      "vs the zero-shot mechanistic margin. Margin never refit on labels."),
           "seed": SEED, "n_boot": N_BOOT}
    for name, loader in (("SPT_Cetin2018", load_spt), ("CPT_Geyin2021", load_cpt), ("Vs_Kayen2013", load_vs)):
        r = loader()
        if r is None:
            continue
        _, d = r
        out[name] = analyze(name, d)
        a = out[name]
        print("=" * 78)
        print(f"{name}: margin AUC {a['margin_grouped_auc']['auc']:.3f} "
              f"[{a['margin_grouped_auc']['lo']:.3f},{a['margin_grouped_auc']['hi']:.3f}]")
        for m, v in a["tuned_ml_grouped_auc"].items():
            print(f"    tuned {m:<11} {v['auc']:.3f} [{v['lo']:.3f},{v['hi']:.3f}]")
        dd = a["best_tuned_ml_minus_margin_AUC"]
        print(f"  best tuned ML = {a['best_tuned_ml']}; Δ(ML-margin) {dd['delta']:+.3f} "
              f"[{dd['lo']:+.3f},{dd['hi']:+.3f}] P={dd['P(delta>0)']:.2f} "
              f"-> margin sufficient vs tuned ML: {a['margin_sufficient_vs_tuned_ml']}")
    path = os.path.join(PROC, "tuned_ml_challenge.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
