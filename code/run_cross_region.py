"""Literal same-modality cross-population transfer (welds cross-DB evidence):
leave-one-REGION-out on the global SPT database. Train ML on all other regions' earthquakes,
test on the held-out region (independent earthquakes, no shared sites). The zero-shot physics
margin is computed per region without any training. Expectation: physics transfers (stable AUC),
ML trained elsewhere degrades.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from geoliq import reliability as R

PROC = os.path.join(BASE, "data", "processed")

REGION_KEYS = {
    "Japan": ["niigata", "kobe", "kushiro", "tohoku", "miyagi", "nihonkai", "chiba", "hyogo", "tokachi", "off-", "fukushima", "kocaeli_no"],
    "USA": ["imperial", "loma", "prieta", "northridge", "san fernando", "superstition", "morgan", "alaska", "whittier", "coalinga", "westmorland", "borrego", "elcentro", "el centro"],
    "Taiwan": ["chi-chi", "chichi", "chi chi"],
    "Turkey": ["kocaeli", "izmit", "duzce", "düzce", "erzincan", "adapazari", "adapazarı"],
    "China": ["tangshan", "haicheng"],
    "NewZealand": ["darfield", "christchurch", "canterbury"],
    "Other": [],
}


def region_of(name):
    s = str(name).lower()
    for reg, keys in REGION_KEYS.items():
        if any(k in s for k in keys):
            return reg
    return "Other"


spt = pd.read_csv(os.path.join(PROC, "cetin2018_baseline_records.csv"))
FS = ["Mw", "amax_g", "sigma_v_kpa", "sigma_eff_kpa", "rd", "FC", "N1_60", "N1_60cs", "crit_depth_m", "water_depth_ft"]
spt = spt.dropna(subset=FS + ["y", "earthquake", "CRR_BI2014", "CSR_cetin"]).copy()
spt = spt[np.isfinite(spt[FS].to_numpy(float)).all(1) & (spt["CRR_BI2014"] > 0) & (spt["CSR_cetin"] > 0)]
spt["region"] = spt["earthquake"].map(region_of)

print("=== earthquake -> region inference (verify) ===")
chk = spt.groupby(["region", "earthquake"]).size().reset_index(name="n")
for reg in chk["region"].unique():
    sub = chk[chk["region"] == reg]
    print(f"[{reg}] n={int(sub['n'].sum())}: " + ", ".join(f"{r.earthquake}({r.n})" for r in sub.itertuples()))
print("\n=== region summary ===")
summ = spt.groupby("region").agg(n=("y", "size"), n_liq=("y", "sum")).reset_index()
print(summ.to_string(index=False))

X = spt[FS].to_numpy(float)
y = spt["y"].astype(int).to_numpy()
reg = spt["region"].to_numpy()
margin = (spt["CSR_cetin"] / spt["CRR_BI2014"]).to_numpy(float)

# leave-one-region-out OOF ML predictions (each region predicted by model trained on all others)
logo = LeaveOneGroupOut()
p_ml = cross_val_predict(HistGradientBoostingClassifier(random_state=0), X, y, groups=reg, cv=logo, method="predict_proba")[:, 1]
p_lr = cross_val_predict(make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=5000)),
                         X, y, groups=reg, cv=logo, method="predict_proba")[:, 1]

rows = []
for rname in sorted(set(reg)):
    m = reg == rname
    if m.sum() < 8 or len(np.unique(y[m])) < 2:
        rows.append({"region": rname, "n": int(m.sum()), "n_liq": int(y[m].sum()), "note": "AUC undefined (single class or n<8)"})
        continue
    rows.append({"region": rname, "n": int(m.sum()), "n_liq": int(y[m].sum()),
                 "physics_AUC": round(float(roc_auc_score(y[m], margin[m])), 3),
                 "ML_trained_elsewhere_AUC": round(float(roc_auc_score(y[m], p_ml[m])), 3),
                 "logistic_elsewhere_AUC": round(float(roc_auc_score(y[m], p_lr[m])), 3)})

pooled = {
    "physics_pooled_AUC": round(float(roc_auc_score(y, margin)), 3),
    "ML_crossregion_pooled_AUC": round(float(roc_auc_score(y, p_ml)), 3),
    "logistic_crossregion_pooled_AUC": round(float(roc_auc_score(y, p_lr)), 3),
}
pair = R.paired_auc_diff_ci(y, margin, p_ml, groups=reg, n_boot=2000)
pooled["paired_physics_minus_ML"] = {"delta": round(pair["delta"], 3), "CI95": [round(pair["lo"], 3), round(pair["hi"], 3)], "P(phys>ML)": round(pair["P(delta>0)"], 3)}

out = {"design": "leave-one-region-out (train other regions' earthquakes, test held-out region)",
       "by_region": rows, "pooled": pooled}
json.dump(out, open(os.path.join(PROC, "cross_region_transfer.json"), "w"), indent=2, ensure_ascii=False)

print("\n=== LEAVE-ONE-REGION-OUT TRANSFER ===")
print(f"{'region':<12}{'n':>5}{'n_liq':>6}{'physics':>9}{'ML(else)':>10}{'logit(else)':>12}")
for r in rows:
    if "physics_AUC" in r:
        print(f"{r['region']:<12}{r['n']:>5}{r['n_liq']:>6}{r['physics_AUC']:>9.3f}{r['ML_trained_elsewhere_AUC']:>10.3f}{r['logistic_elsewhere_AUC']:>12.3f}")
    else:
        print(f"{r['region']:<12}{r['n']:>5}{r['n_liq']:>6}   {r['note']}")
print(f"\nPOOLED: physics={pooled['physics_pooled_AUC']}  ML(cross-region)={pooled['ML_crossregion_pooled_AUC']}  logit={pooled['logistic_crossregion_pooled_AUC']}")
print(f"paired physics-ML: {pooled['paired_physics_minus_ML']}")
print("\nwrote cross_region_transfer.json")
