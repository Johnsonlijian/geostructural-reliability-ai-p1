"""(2) Groundwater / effective-stress control on REAL Cetin 2018 data.

(2a) Mechanistic counterfactual: move the water table (surface / actual / deep) through the
     VALIDATED engine and watch triggering move.  (2b) ML drop-column ablation, event-grouped.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from geoliq.mechanics import crr_boulanger_idriss as BI
from geoliq.mechanics import effective_stress as ES

PSF_TO_KPA = 0.0478802589
GAMMA_W_PCF = 62.4
XLS = os.path.join(BASE, "data", "raw", "cetin2018_spt", "mmc2.xls")
raw = pd.read_excel(XLS, sheet_name="CETIN_2018", header=0)


def col(i):
    return raw.iloc[:, i]


d = pd.DataFrame({
    "earthquake": col(1), "liq": col(3),
    "z_ft": pd.to_numeric(col(7), errors="coerce"),
    "dw_ft": pd.to_numeric(col(10), errors="coerce"),
    "g1": pd.to_numeric(col(12), errors="coerce"),
    "g2": pd.to_numeric(col(14), errors="coerce"),
    "sv_psf": pd.to_numeric(col(16), errors="coerce"),
    "sveff_psf": pd.to_numeric(col(18), errors="coerce"),
    "rd": pd.to_numeric(col(21), errors="coerce"),
    "amax": pd.to_numeric(col(23), errors="coerce"),
    "Mw": pd.to_numeric(col(25), errors="coerce"),
    "FC": pd.to_numeric(col(33), errors="coerce"),
    "N1_60cs": pd.to_numeric(col(56), errors="coerce"),
})
lab = d["liq"].astype(str).str.strip().str.lower()
d["y"] = np.where(lab.isin(["yes"]), 1.0, np.where(lab.isin(["no"]), 0.0, np.nan))
d = d.dropna(subset=["z_ft", "dw_ft", "g1", "g2", "rd", "amax", "Mw", "N1_60cs", "y", "sveff_psf"]).copy()
y = d["y"].astype(int).to_numpy()

# engine self-check: recompute effective stress from depth/dw/unit weights vs dataset column
sveff_calc = ES.effective_vertical_stress(d["z_ft"].to_numpy(float), d["dw_ft"].to_numpy(float),
                                          d["g1"].to_numpy(float), d["g2"].to_numpy(float), gamma_w=GAMMA_W_PCF)
r_self = float(np.corrcoef(sveff_calc, d["sveff_psf"].to_numpy(float))[0, 1])
mae_self = float(np.mean(np.abs(sveff_calc - d["sveff_psf"].to_numpy(float))))


def fs_for_dw(dw):
    z = d["z_ft"].to_numpy(float)
    sv = ES.total_vertical_stress(z, dw, d["g1"].to_numpy(float), d["g2"].to_numpy(float))
    sveff = ES.effective_vertical_stress(z, dw, d["g1"].to_numpy(float), d["g2"].to_numpy(float), gamma_w=GAMMA_W_PCF)
    sveff = np.maximum(sveff, 1.0)
    csr = 0.65 * d["amax"].to_numpy(float) * (sv / sveff) * d["rd"].to_numpy(float)
    crr = BI.crr_insitu(d["Mw"].to_numpy(float), sveff * PSF_TO_KPA, d["N1_60cs"].to_numpy(float), mode="spt")
    return crr / csr

z = d["z_ft"].to_numpy(float)
scen = {"surface(dw=0)": np.zeros_like(z), "actual": d["dw_ft"].to_numpy(float), "deep/dry(dw=z)": z}
scen_rows = {}
for name, dw in scen.items():
    fs = fs_for_dw(dw)
    fsv = fs[np.isfinite(fs)]
    scen_rows[name] = {"median_FS": round(float(np.median(fsv)), 3),
                       "p25_FS": round(float(np.percentile(fsv, 25)), 3),
                       "p75_FS": round(float(np.percentile(fsv, 75)), 3),
                       "pred_liq_rate_FS<1": round(float(np.mean(fsv < 1)), 3)}

fs_actual = fs_for_dw(d["dw_ft"].to_numpy(float))
fs_surface = fs_for_dw(np.zeros_like(z))
acc_actual = float(accuracy_score(y, (fs_actual < 1).astype(int)))
acc_surface = float(accuracy_score(y, (fs_surface < 1).astype(int)))

# (2b) ML drop-column ablation, earthquake-grouped
d["sveff_kpa"] = d["sveff_psf"] * PSF_TO_KPA
d["sv_kpa"] = d["sv_psf"] * PSF_TO_KPA
FULL = ["Mw", "amax", "sv_kpa", "sveff_kpa", "rd", "FC", "N1_60cs", "z_ft", "dw_ft"]
NOGW = ["Mw", "amax", "sv_kpa", "rd", "FC", "N1_60cs", "z_ft"]  # drop sveff & water depth
groups = d["earthquake"].astype(str).to_numpy()
gkf = list(GroupKFold(n_splits=5).split(d[FULL].to_numpy(float), y, groups))
auc_full = float(roc_auc_score(y, cross_val_predict(GradientBoostingClassifier(random_state=0), d[FULL].to_numpy(float), y, cv=gkf, method="predict_proba")[:, 1]))
auc_nogw = float(roc_auc_score(y, cross_val_predict(GradientBoostingClassifier(random_state=0), d[NOGW].to_numpy(float), y, cv=gkf, method="predict_proba")[:, 1]))

out = {"n_used": int(len(d)),
       "engine_selfcheck_effstress": {"pearson_r": round(r_self, 5), "MAE_psf": round(mae_self, 4)},
       "counterfactual_water_table": scen_rows,
       "hazard_call_accuracy": {"actual_wt": round(acc_actual, 3), "wrong_surface_wt": round(acc_surface, 3)},
       "ML_grouped_AUC": {"with_groundwater": round(auc_full, 4), "without_groundwater": round(auc_nogw, 4),
                          "drop": round(auc_full - auc_nogw, 4)}}
with open(os.path.join(BASE, "data", "processed", "cetin2018_groundwater_ablation.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print("=" * 66)
print("(2) GROUNDWATER / EFFECTIVE-STRESS CONTROL  (real Cetin 2018)")
print("=" * 66)
print(f"engine self-check eff.stress vs dataset: r={r_self:.5f}, MAE={mae_self:.3f} psf  (n={len(d)})")
print("\n(2a) mechanistic counterfactual water table -> triggering:")
for name, r in scen_rows.items():
    print(f"   {name:<16} median FS={r['median_FS']:.3f}  [p25 {r['p25_FS']:.2f}, p75 {r['p75_FS']:.2f}]  pred-liq rate={r['pred_liq_rate_FS<1']:.3f}")
print(f"   hazard-call accuracy: actual WT={acc_actual:.3f}  vs  wrong surface WT={acc_surface:.3f}")
print("\n(2b) ML drop-column (earthquake-grouped AUC):")
print(f"   with groundwater={auc_full:.3f}   without groundwater={auc_nogw:.3f}   drop={auc_full-auc_nogw:+.3f}")
