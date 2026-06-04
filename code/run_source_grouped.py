"""(3) Cross-source generalization on REAL Cetin 2018: random vs earthquake-grouped vs
SOURCE(reference)-grouped CV. Crux: zero-shot physics AUC is split-invariant (not fitted),
ML degrades as grouping tightens -> the cross-source/cross-DB thesis.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from geoliq.mechanics import crr_boulanger_idriss as BI

PSF_TO_KPA = 0.0478802589
XLS = os.path.join(BASE, "data", "raw", "cetin2018_spt", "mmc2.xls")
raw = pd.read_excel(XLS, sheet_name="CETIN_2018", header=0)


def col(i):
    return raw.iloc[:, i]


d = pd.DataFrame({
    "earthquake": col(1).astype(str), "reference": col(50).astype(str), "liq": col(3),
    "Mw": pd.to_numeric(col(25), errors="coerce"),
    "amax_g": pd.to_numeric(col(23), errors="coerce"),
    "sigma_v_kpa": pd.to_numeric(col(16), errors="coerce") * PSF_TO_KPA,
    "sigma_eff_kpa": pd.to_numeric(col(18), errors="coerce") * PSF_TO_KPA,
    "rd": pd.to_numeric(col(21), errors="coerce"),
    "FC": pd.to_numeric(col(33), errors="coerce"),
    "N1_60": pd.to_numeric(col(48), errors="coerce"),
    "N1_60cs": pd.to_numeric(col(56), errors="coerce"),
    "crit_depth_m": pd.to_numeric(col(8), errors="coerce"),
    "water_depth_ft": pd.to_numeric(col(10), errors="coerce"),
    "CSR": pd.to_numeric(col(27), errors="coerce"),
})
lab = d["liq"].astype(str).str.strip().str.lower()
d["y"] = np.where(lab.isin(["yes"]), 1.0, np.where(lab.isin(["no"]), 0.0, np.nan))
d["CRR"] = BI.crr_insitu(d["Mw"].to_numpy(float), d["sigma_eff_kpa"].to_numpy(float),
                         d["N1_60cs"].to_numpy(float), mode="spt")

FEAT = ["Mw", "amax_g", "sigma_v_kpa", "sigma_eff_kpa", "rd", "FC", "N1_60", "N1_60cs", "crit_depth_m", "water_depth_ft"]
d = d.dropna(subset=FEAT + ["y", "CRR", "CSR"]).copy()
d = d[np.isfinite(d[FEAT].to_numpy(float)).all(axis=1) & (d["CRR"] > 0) & (d["CSR"] > 0)]
X = d[FEAT].to_numpy(float)
y = d["y"].astype(int).to_numpy()
eq = d["earthquake"].to_numpy()
ref = d["reference"].to_numpy()
auc_phys = float(roc_auc_score(y, (d["CSR"] / d["CRR"]).to_numpy(float)))  # split-invariant

splits = {
    "random_kfold": StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
    "earthquake_grouped": list(GroupKFold(n_splits=5).split(X, y, eq)),
    "source(reference)_grouped": list(GroupKFold(n_splits=5).split(X, y, ref)),
}


def oof_auc(est, cv):
    p = cross_val_predict(est, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


rows = {}
for sname, cv in splits.items():
    rows[sname] = {
        "grad_boost_auc": round(oof_auc(GradientBoostingClassifier(random_state=0), cv), 4),
        "logistic_auc": round(oof_auc(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), cv), 4),
    }

out = {"n_used": int(len(d)), "n_earthquakes": int(len(np.unique(eq))), "n_sources": int(len(np.unique(ref))),
       "physics_BI2014_auc_split_invariant": round(auc_phys, 4), "by_split": rows}
with open(os.path.join(BASE, "data", "processed", "cetin2018_source_grouped.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print("=" * 70)
print("(3) CROSS-SOURCE GENERALIZATION (real Cetin 2018)")
print("=" * 70)
print(f"n={len(d)}  earthquakes={out['n_earthquakes']}  source-studies={out['n_sources']}")
print(f"physics BI2014 AUC (SPLIT-INVARIANT, zero-shot) = {auc_phys:.3f}\n")
print(f"{'split (increasing strictness)':<30}{'GBM':>8}{'logit':>8}{'physics':>9}")
for sname, r in rows.items():
    print(f"{sname:<30}{r['grad_boost_auc']:>8.3f}{r['logistic_auc']:>8.3f}{auc_phys:>9.3f}")
print("\nwrote: data/processed/cetin2018_source_grouped.json")
