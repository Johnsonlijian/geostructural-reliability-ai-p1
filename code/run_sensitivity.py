"""A5 robustness: decision-threshold sensitivity and model-class transfer checks."""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
PROC = os.path.join(BASE, "data", "processed")

spt = pd.read_csv(os.path.join(PROC, "cetin2018_baseline_records.csv"))
FS = ["Mw", "amax_g", "sigma_v_kpa", "sigma_eff_kpa", "rd", "FC", "N1_60", "N1_60cs", "crit_depth_m", "water_depth_ft"]
spt = spt.dropna(subset=FS + ["y", "earthquake", "CRR_BI2014", "CSR_cetin"]).copy()
spt = spt[np.isfinite(spt[FS].to_numpy(float)).all(1) & (spt["CRR_BI2014"] > 0) & (spt["CSR_cetin"] > 0)]
fs_ratio = (spt["CRR_BI2014"] / spt["CSR_cetin"]).to_numpy(float)
yS = spt["y"].astype(int).to_numpy()
cpt = pd.read_csv(os.path.join(PROC, "geyin2021_cpt_records.csv"))
yC = cpt["y"].astype(int).to_numpy(); LPI = cpt["LPI"].to_numpy(float)

out = {"threshold_sensitivity": {}, "sufficiency_across_models": {}}

# (i) decision-threshold sensitivity
out["threshold_sensitivity"]["SPT_FS_threshold"] = {
    str(t): round(float(np.mean((fs_ratio < t).astype(int) == yS)), 3) for t in [0.9, 1.0, 1.1, 1.2]}
out["threshold_sensitivity"]["CPT_LPI_threshold"] = {
    str(t): round(float(np.mean((LPI >= t).astype(int) == yC)), 3) for t in [4, 5, 6, 7]}

# (ii) model-class transfer checks (margin-only vs full, earthquake-grouped OOF AUC)
models = {
    "logistic": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=5000)),
    "random_forest": lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=400, random_state=0)),
    "grad_boost": lambda: make_pipeline(SimpleImputer(strategy="median"), GradientBoostingClassifier(random_state=0)),
    "hist_gbt": lambda: HistGradientBoostingClassifier(random_state=0),
}
DB = {
    "SPT/Cetin2018": (spt[FS].to_numpy(float), yS, spt["earthquake"].to_numpy(),
                      np.log((spt["CSR_cetin"] / spt["CRR_BI2014"]).to_numpy(float))),
    "CPT/Geyin2021": (cpt[["Mw", "PGA", "GWT", "LPI", "crit_FS_capped", "crit_depth", "qc1Ncs_crit", "Ic_crit", "sev_crit"]].to_numpy(float),
                      yC, cpt["event"].to_numpy(), np.log1p(LPI)),
}
for dbname, (X, y, g, s) in DB.items():
    cv = list(GroupKFold(5).split(X, y, g))
    cvm = list(GroupKFold(5).split(s.reshape(-1, 1), y, g))
    row = {}
    for mname, mk in models.items():
        auc_full = roc_auc_score(y, cross_val_predict(mk(), X, y, cv=cv, method="predict_proba")[:, 1])
        auc_marg = roc_auc_score(y, cross_val_predict(mk(), s.reshape(-1, 1), y, cv=cvm, method="predict_proba")[:, 1])
        row[mname] = {"full": round(float(auc_full), 3), "margin_only": round(float(auc_marg), 3),
                      "margin_minus_full": round(float(auc_marg - auc_full), 3)}
    out["sufficiency_across_models"][dbname] = row

with open(os.path.join(PROC, "sensitivity.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print("THRESHOLD SENSITIVITY")
print(" SPT acc by FS thr:", out["threshold_sensitivity"]["SPT_FS_threshold"])
print(" CPT acc by LPI thr:", out["threshold_sensitivity"]["CPT_LPI_threshold"])
print("\nMODEL-CLASS TRANSFER CHECKS (margin_only - full, earthquake-grouped; >=0 => no full-feature gain)")
for db, row in out["sufficiency_across_models"].items():
    print(f" {db}:")
    for m, v in row.items():
        flag = "OK" if v["margin_minus_full"] >= -0.01 else "**"
        print(f"   {m:<14} full={v['full']:.3f}  margin={v['margin_only']:.3f}  delta={v['margin_minus_full']:+.3f} {flag}")
print("\nwrote data/processed/sensitivity.json")
