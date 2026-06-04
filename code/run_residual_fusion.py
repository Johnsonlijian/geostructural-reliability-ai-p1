"""(1) Residual / physics-data fusion on REAL Cetin 2018 data, earthquake-grouped CV.

Question: can ML that USES the BI2014 physics margin beat the code baseline under
honest (event-grouped) validation, where black-box ML did not?
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE, "data", "processed", "cetin2018_baseline_records.csv"))
FEAT = ["Mw", "amax_g", "sigma_v_kpa", "sigma_eff_kpa", "rd", "FC", "N1_60", "N1_60cs", "crit_depth_m", "water_depth_ft"]
d = df.dropna(subset=FEAT + ["y", "earthquake", "CRR_BI2014", "CSR_cetin"]).copy()
d = d[np.isfinite(d[FEAT].to_numpy(float)).all(axis=1) & (d["CRR_BI2014"] > 0) & (d["CSR_cetin"] > 0)]
y = d["y"].astype(int).to_numpy()
groups = d["earthquake"].astype(str).to_numpy()
zphys = np.log(d["CSR_cetin"].to_numpy(float) / d["CRR_BI2014"].to_numpy(float))  # physics margin (>0 -> liq)
X2 = d[FEAT].to_numpy(float)
X3 = np.column_stack([X2, zphys])

gkf = list(GroupKFold(n_splits=5).split(X2, y, groups))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


def oof_auc(est, Xmat, cv):
    p = cross_val_predict(est, Xmat, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


auc_phys = float(roc_auc_score(y, zphys))  # M0 physics-only, no fit
auc_M1 = oof_auc(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), zphys.reshape(-1, 1), gkf)
auc_M2 = oof_auc(GradientBoostingClassifier(random_state=0), X2, gkf)
auc_M3_g = oof_auc(GradientBoostingClassifier(random_state=0), X3, gkf)
auc_M3_r = oof_auc(GradientBoostingClassifier(random_state=0), X3, skf)

out = {
    "n_used": int(len(d)), "n_earthquakes": int(len(np.unique(groups))),
    "M0_physics_only_BI2014": round(auc_phys, 4),
    "M1_calibrated_physics_grouped": round(auc_M1, 4),
    "M2_ML_features_only_grouped": round(auc_M2, 4),
    "M3_fusion_features+physics_grouped": round(auc_M3_g, 4),
    "M3_fusion_random": round(auc_M3_r, 4),
    "M3_optimism_gap": round(auc_M3_r - auc_M3_g, 4),
    "fusion_beats_code_grouped": bool(auc_M3_g > auc_phys),
    "fusion_gain_over_code_grouped": round(auc_M3_g - auc_phys, 4),
    "fusion_gain_over_blackbox_grouped": round(auc_M3_g - auc_M2, 4),
}
with open(os.path.join(BASE, "data", "processed", "cetin2018_residual_fusion.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print("=" * 66)
print("(1) RESIDUAL / PHYSICS-DATA FUSION  (real Cetin 2018, earthquake-grouped)")
print("=" * 66)
print(f"n={out['n_used']} cases / {out['n_earthquakes']} earthquakes")
print(f"M0 physics-only BI2014           AUC = {auc_phys:.3f}")
print(f"M1 calibrated physics (grouped)  AUC = {auc_M1:.3f}")
print(f"M2 ML features-only  (grouped)   AUC = {auc_M2:.3f}")
print(f"M3 FUSION feat+physics (grouped) AUC = {auc_M3_g:.3f}   (random {auc_M3_r:.3f}, gap {auc_M3_r-auc_M3_g:.3f})")
print(f"  fusion vs code   : {auc_M3_g-auc_phys:+.3f}   beats code under grouped CV? {out['fusion_beats_code_grouped']}")
print(f"  fusion vs blackbox: {auc_M3_g-auc_M2:+.3f}")
