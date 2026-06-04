"""P1 mechanistic baseline on REAL data: Cetin et al. (2018) SPT liquefaction case histories.

(1) Validate geoliq Seed-Idriss CSR against the dataset's own CSR column.
(2) Run Boulanger & Idriss (2014) zero-shot CRR -> FS -> classification vs observed labels.
NO label fitting anywhere. Outputs to data/processed/.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geoliq.mechanics import csr as CSR
from geoliq.mechanics import crr_boulanger_idriss as BI

BASE = os.path.dirname(os.path.abspath(__file__))
XLS = os.path.join(BASE, "data", "raw", "cetin2018_spt", "mmc2.xls")
OUT = os.path.join(BASE, "data", "processed")
os.makedirs(OUT, exist_ok=True)
PSF_TO_KPA = 0.0478802589

raw = pd.read_excel(XLS, sheet_name="CETIN_2018", header=0)


def col(i):
    return raw.iloc[:, i]


df = pd.DataFrame({
    "case": col(0), "earthquake": col(1), "site": col(2),
    "liquefied_raw": col(3), "data_class": col(4),
    "crit_depth_m": pd.to_numeric(col(8), errors="coerce"),
    "water_depth_ft": pd.to_numeric(col(10), errors="coerce"),
    "sigma_v_psf": pd.to_numeric(col(16), errors="coerce"),
    "sigma_eff_psf": pd.to_numeric(col(18), errors="coerce"),
    "rd": pd.to_numeric(col(21), errors="coerce"),
    "amax_g": pd.to_numeric(col(23), errors="coerce"),
    "Mw": pd.to_numeric(col(25), errors="coerce"),
    "CSR_cetin": pd.to_numeric(col(27), errors="coerce"),
    "FC": pd.to_numeric(col(33), errors="coerce"),
    "N1_60": pd.to_numeric(col(48), errors="coerce"),
    "N1_60cs": pd.to_numeric(col(56), errors="coerce"),
})

df["sigma_eff_kpa"] = df["sigma_eff_psf"] * PSF_TO_KPA
df["sigma_v_kpa"] = df["sigma_v_psf"] * PSF_TO_KPA

lab = df["liquefied_raw"].astype(str).str.strip().str.lower()
df["y"] = np.where(lab.isin(["yes", "y", "liquefied", "1"]), 1.0,
                   np.where(lab.isin(["no", "n", "non-liquefied", "0"]), 0.0, np.nan))

# (1) geoliq Seed-Idriss CSR (stress ratio is unitless -> psf is fine)
df["CSR_mine"] = CSR.csr_seed_idriss(df["amax_g"].to_numpy(float), df["sigma_v_psf"].to_numpy(float),
                                     df["sigma_eff_psf"].to_numpy(float), df["rd"].to_numpy(float))
# (2) BI2014 zero-shot CRR (in-situ) from clean-sand N1,60,cs, Mw, sigma'_v[kPa]
df["CRR_BI2014"] = BI.crr_insitu(df["Mw"].to_numpy(float), df["sigma_eff_kpa"].to_numpy(float),
                                 df["N1_60cs"].to_numpy(float), mode="spt")
df["FS_BI2014"] = df["CRR_BI2014"] / df["CSR_cetin"]
df["pred_liq"] = (df["FS_BI2014"] < 1.0).astype("Int64")
df["score_liq"] = df["CSR_cetin"] / df["CRR_BI2014"]  # higher -> more likely to liquefy

# ---- CSR validation ----
v = df.dropna(subset=["CSR_mine", "CSR_cetin"])
v = v[np.isfinite(v["CSR_mine"]) & np.isfinite(v["CSR_cetin"])]
csr_mae = float(np.mean(np.abs(v["CSR_mine"] - v["CSR_cetin"])))
csr_max = float(np.max(np.abs(v["CSR_mine"] - v["CSR_cetin"])))
csr_corr = float(np.corrcoef(v["CSR_mine"], v["CSR_cetin"])[0, 1])

# ---- BI2014 baseline vs observed labels ----
m = df.dropna(subset=["y"]).copy()
m = m[np.isfinite(m["CRR_BI2014"]) & np.isfinite(m["CSR_cetin"]) & (m["CSR_cetin"] > 0)]
y = m["y"].astype(int).to_numpy()
pred = m["pred_liq"].astype(int).to_numpy()
auc = float(roc_auc_score(y, m["score_liq"].to_numpy(float)))
acc = float(accuracy_score(y, pred))
prec = float(precision_score(y, pred))
rec = float(recall_score(y, pred))
cm = confusion_matrix(y, pred)  # [[TN,FP],[FN,TP]]
tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
false_alarm = fp / (fp + tn) if (fp + tn) else float("nan")  # predict liq but didn't
miss_rate = fn / (fn + tp) if (fn + tp) else float("nan")    # missed real liquefaction

n_total = len(df)
n_liq = int((df["y"] == 1).sum())
n_non = int((df["y"] == 0).sum())
n_marg = int(df["y"].isna().sum())

results = {
    "dataset": "Cetin et al. (2018) Data in Brief, DOI 10.1016/j.dib.2018.08.043",
    "n_total_rows": n_total, "n_liquefied": n_liq, "n_nonliquefied": n_non, "n_marginal_or_unmapped": n_marg,
    "n_used_in_metrics": int(len(m)),
    "csr_validation_vs_cetin": {"pearson_r": round(csr_corr, 5), "MAE": round(csr_mae, 5), "max_abs_err": round(csr_max, 5)},
    "BI2014_zero_shot_baseline": {
        "FS_threshold": 1.0, "accuracy": round(acc, 4), "roc_auc": round(auc, 4),
        "precision_liq": round(prec, 4), "recall_liq": round(rec, 4),
        "confusion_matrix_[[TN,FP],[FN,TP]]": [[tn, fp], [fn, tp]],
        "false_alarm_rate_FP_over_FP+TN": round(false_alarm, 4),
        "miss_rate_FN_over_FN+TP": round(miss_rate, 4),
    },
}

df.to_csv(os.path.join(OUT, "cetin2018_baseline_records.csv"), index=False)
with open(os.path.join(OUT, "cetin2018_baseline_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("=" * 64)
print("CETIN 2018 — REAL DATA, ZERO-SHOT MECHANISTIC BASELINES")
print("=" * 64)
print(f"rows={n_total}  liquefied={n_liq}  non={n_non}  marginal/unmapped={n_marg}  used={len(m)}")
print("\n[1] geoliq Seed-Idriss CSR  vs  Cetin published CSR column:")
print(f"    Pearson r = {csr_corr:.5f}   MAE = {csr_mae:.5f}   max|err| = {csr_max:.5f}")
print("\n[2] Boulanger-Idriss (2014) zero-shot, FS=CRR/CSR, predict liq if FS<1:")
print(f"    accuracy = {acc:.3f}   ROC-AUC = {auc:.3f}   precision(liq) = {prec:.3f}   recall(liq) = {rec:.3f}")
print(f"    confusion [[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]")
print(f"    false-alarm rate = {false_alarm:.3f}   miss rate = {miss_rate:.3f}")
print("\nwrote: data/processed/cetin2018_baseline_records.csv  +  cetin2018_baseline_metrics.json")
