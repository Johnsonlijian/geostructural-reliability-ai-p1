"""Honest-evaluation (over-validation) benchmark for liquefaction-ML (P1 upgrade, 2026-06-30).

Quantifies how much naive RANDOM cross-validation INFLATES apparent skill versus honest
earthquake-GROUPED cross-validation, because random folds leak within-earthquake structure
(neighbouring soundings from the same event in train and test). For each modality and model
we report out-of-fold AUC under random KFold vs GroupKFold(earthquake) and the optimism gap,
with bootstrap CIs.

The methodological message: under random CV a flexible ML model appears to match or beat the
zero-shot mechanistic margin; under earthquake-grouped CV that apparent advantage disappears.
The widely reported "ML beats the simplified procedure" is therefore in large part an
over-validation artifact, and the mechanistic margin -- which is not fit on labels and so
barely moves between the two protocols -- is the honest reference.

Outputs data/processed/overvalidation_benchmark.json. Deterministic (seed 0).
"""

import json
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from geoliq.reliability import bootstrap_auc_ci
from run_predictability_ceiling import load_spt, load_cpt, load_vs

BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, "data", "processed")
SEED = 0
N_BOOT = 1000


def _oof(estimator, X, y, groups, mode):
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if mode == "random":
        cv = list(KFold(5, shuffle=True, random_state=SEED).split(X, y))
    else:
        cv = list(GroupKFold(min(5, len(np.unique(groups)))).split(X, y, groups))
    return cross_val_predict(estimator, X, y, cv=cv, method="predict_proba")[:, 1]


def _models(margin_s, X_full):
    s = np.asarray(margin_s, float).reshape(-1, 1)
    return {
        "margin_1d": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), s),
        "logistic_full": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), X_full),
        "knn": (make_pipeline(StandardScaler(), KNeighborsClassifier(
            n_neighbors=max(5, int(round(np.sqrt(len(margin_s))))))), X_full),
        "gb": (HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
               l2_regularization=1.0, random_state=SEED), X_full),
    }


def analyze(y, groups, margin_s, X_full, response):
    y = np.asarray(y, int)
    groups = np.asarray(groups)
    rows = {}
    for name, (est, X) in _models(margin_s, X_full).items():
        p_rand = _oof(est, X, y, groups, "random")
        p_grp = _oof(est, X, y, groups, "grouped")
        a_rand = bootstrap_auc_ci(y, p_rand, seed=SEED, n_boot=N_BOOT)              # iid bootstrap
        a_grp = bootstrap_auc_ci(y, p_grp, groups=groups, seed=SEED, n_boot=N_BOOT)  # cluster bootstrap
        rows[name] = {
            "auc_random_cv": {k: round(v, 4) for k, v in a_rand.items()},
            "auc_grouped_cv": {k: round(v, 4) for k, v in a_grp.items()},
            "optimism_random_minus_grouped": round(a_rand["auc"] - a_grp["auc"], 4),
        }
    best_ml = max(("logistic_full", "knn", "gb"), key=lambda m: rows[m]["auc_random_cv"]["auc"])
    flip = {
        "best_ml_model": best_ml,
        "under_random_cv_best_ml_minus_margin_AUC":
            round(rows[best_ml]["auc_random_cv"]["auc"] - rows["margin_1d"]["auc_random_cv"]["auc"], 4),
        "under_grouped_cv_best_ml_minus_margin_AUC":
            round(rows[best_ml]["auc_grouped_cv"]["auc"] - rows["margin_1d"]["auc_grouped_cv"]["auc"], 4),
    }
    return {"response": response, "n": int(len(y)), "n_earthquakes": int(len(np.unique(groups))),
            "models": rows, "ml_vs_margin_flip": flip}


def main():
    out = {"method": ("Out-of-fold AUC under random KFold(5) vs earthquake-GroupKFold; optimism = "
                      "random - grouped; iid bootstrap CI for random, cluster bootstrap (by earthquake) "
                      "for grouped. Margins never refit on labels."),
           "seed": SEED, "n_boot": N_BOOT}
    loaders = [load_spt(), load_cpt()]
    vs = load_vs()
    if vs is not None:
        loaders.append(vs)
    for name, d in loaders:
        out[name] = analyze(d["y"], d["groups"], d["margin_s"], d["X_full"], d["response"])

    for name, _ in loaders:
        r = out[name]
        f = r["ml_vs_margin_flip"]
        print("=" * 80)
        print(f"{name} (n={r['n']}, EQ={r['n_earthquakes']}, {r['response']})")
        for m, mr in r["models"].items():
            print(f"  {m:<14} random AUC {mr['auc_random_cv']['auc']:.3f} | grouped "
                  f"{mr['auc_grouped_cv']['auc']:.3f} | optimism {mr['optimism_random_minus_grouped']:+.3f}")
        print(f"  FLIP: best ML ({f['best_ml_model']}) vs margin: "
              f"random {f['under_random_cv_best_ml_minus_margin_AUC']:+.3f}  ->  "
              f"grouped {f['under_grouped_cv_best_ml_minus_margin_AUC']:+.3f}")
    path = os.path.join(PROC, "overvalidation_benchmark.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
