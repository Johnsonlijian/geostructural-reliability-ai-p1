"""P1 signature result on REAL data: random vs earthquake-grouped CV optimism gap,
and whether ML beats the Boulanger-Idriss (2014) zero-shot physics baseline.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE, "data", "processed", "cetin2018_baseline_records.csv"))

FEAT = ["Mw", "amax_g", "sigma_v_kpa", "sigma_eff_kpa", "rd", "FC",
        "N1_60", "N1_60cs", "crit_depth_m", "water_depth_ft", "CSR_cetin"]
d = df.dropna(subset=FEAT + ["y", "earthquake", "CRR_BI2014"]).copy()
d = d[np.isfinite(d[FEAT].to_numpy(float)).all(axis=1)]
X = d[FEAT].to_numpy(float)
y = d["y"].astype(int).to_numpy()
groups = d["earthquake"].astype(str).to_numpy()
n_groups = int(len(np.unique(groups)))
nsplit = 5

models = {
    "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    "random_forest": RandomForestClassifier(n_estimators=400, random_state=0),
    "grad_boost": GradientBoostingClassifier(random_state=0),
}

phys_score = (d["CSR_cetin"] / d["CRR_BI2014"]).to_numpy(float)
auc_phys = float(roc_auc_score(y, phys_score))

rows = {}
for name, mdl in models.items():
    skf = StratifiedKFold(n_splits=nsplit, shuffle=True, random_state=0)
    oof_r = cross_val_predict(mdl, X, y, cv=skf, method="predict_proba")[:, 1]
    auc_r = float(roc_auc_score(y, oof_r))
    gkf = GroupKFold(n_splits=nsplit)
    oof_g = cross_val_predict(mdl, X, y, cv=gkf.split(X, y, groups), method="predict_proba")[:, 1]
    auc_g = float(roc_auc_score(y, oof_g))
    rows[name] = {"random_cv_auc": round(auc_r, 4), "event_grouped_auc": round(auc_g, 4),
                  "optimism_gap": round(auc_r - auc_g, 4),
                  "beats_BI2014_under_grouped": bool(auc_g > auc_phys)}

out = {"n_used": int(len(d)), "n_earthquakes": n_groups, "n_splits": nsplit,
       "BI2014_physics_baseline_auc": round(auc_phys, 4), "models": rows}
with open(os.path.join(BASE, "data", "processed", "cetin2018_grouped_validation.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print("=" * 70)
print("OPTIMISM GAP — random K-fold vs earthquake-grouped CV (real Cetin 2018)")
print("=" * 70)
print(f"n={len(d)} cases across {n_groups} earthquakes; {nsplit}-fold")
print(f"Boulanger-Idriss (2014) zero-shot physics baseline AUC = {auc_phys:.3f}\n")
print(f"{'model':<16}{'random AUC':>12}{'grouped AUC':>13}{'gap':>8}{'>BI2014?':>10}")
for name, r in rows.items():
    print(f"{name:<16}{r['random_cv_auc']:>12.3f}{r['event_grouped_auc']:>13.3f}"
          f"{r['optimism_gap']:>8.3f}{str(r['beats_BI2014_under_grouped']):>10}")
print("\nwrote: data/processed/cetin2018_grouped_validation.json")
