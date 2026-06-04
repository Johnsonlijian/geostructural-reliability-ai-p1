"""④ + rigor upgrade: bootstrap AUC CIs, cluster-bootstrap paired physics-vs-ML test,
calibration (Brier/ECE), and conformal coverage (random vs earthquake-grouped) on BOTH DBs.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from geoliq import reliability as R

PROC = os.path.join(BASE, "data", "processed")
HGB = lambda: HistGradientBoostingClassifier(random_state=0)


def battery(name, X, y, groups, phys_score, label):
    gkf = list(GroupKFold(n_splits=5).split(X, y, groups))
    ml_oof = cross_val_predict(HGB(), X, y, cv=gkf, method="predict_proba")[:, 1]
    phys_log = np.log(np.clip(phys_score, 1e-6, None)).reshape(-1, 1)
    phys_cal = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                                 phys_log, y, cv=gkf, method="predict_proba")[:, 1]
    res = {
        "n": int(len(y)), "n_groups": int(len(np.unique(groups))), "label": label,
        "physics_auc_ci": R.bootstrap_auc_ci(y, phys_score, groups=groups),
        "ml_grouped_auc_ci": R.bootstrap_auc_ci(y, ml_oof, groups=groups),
        "paired_physics_minus_ml": R.paired_auc_diff_ci(y, phys_score, ml_oof, groups=groups),
        "calibration": {"physics": {"brier": round(R.brier(y, phys_cal), 4), "ece": round(R.ece(y, phys_cal), 4)},
                        "ml_grouped": {"brier": round(R.brier(y, ml_oof), 4), "ece": round(R.ece(y, ml_oof), 4)}},
        "conformal_coverage_alpha0.10": {
            "random_calib": R.conformal_coverage_experiment(X, y, groups, HGB, mode="random"),
            "earthquake_grouped_calib": R.conformal_coverage_experiment(X, y, groups, HGB, mode="grouped")},
        "conformal_conditional_grouped": R.conformal_conditional_coverage(X, y, groups, HGB),
    }
    return res


# ---- SPT (Cetin 2018), triggering ----
spt = pd.read_csv(os.path.join(PROC, "cetin2018_baseline_records.csv"))
FS = ["Mw", "amax_g", "sigma_v_kpa", "sigma_eff_kpa", "rd", "FC", "N1_60", "N1_60cs", "crit_depth_m", "water_depth_ft"]
spt = spt.dropna(subset=FS + ["y", "earthquake", "CRR_BI2014", "CSR_cetin"]).copy()
spt = spt[np.isfinite(spt[FS].to_numpy(float)).all(axis=1) & (spt["CRR_BI2014"] > 0) & (spt["CSR_cetin"] > 0)]
spt_res = battery("SPT", spt[FS].to_numpy(float), spt["y"].astype(int).to_numpy(),
                  spt["earthquake"].to_numpy(), (spt["CSR_cetin"] / spt["CRR_BI2014"]).to_numpy(float), "triggering")

# ---- CPT (Geyin & Maurer 2021), manifestation ----
cpt = pd.read_csv(os.path.join(PROC, "geyin2021_cpt_records.csv"))
FC = ["Mw", "PGA", "GWT", "LPI", "crit_FS_capped", "crit_depth", "qc1Ncs_crit", "Ic_crit", "sev_crit"]
cpt_res = battery("CPT", cpt[FC].to_numpy(float), cpt["y"].astype(int).to_numpy(),
                  cpt["event"].to_numpy(), cpt["LPI"].to_numpy(float), "manifestation")

out = {"SPT_Cetin2018": spt_res, "CPT_Geyin2021": cpt_res}
with open(os.path.join(PROC, "reliability_upgrade.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)


def show(tag, r):
    p, m = r["physics_auc_ci"], r["ml_grouped_auc_ci"]
    d = r["paired_physics_minus_ml"]
    cr, cg = r["conformal_coverage_alpha0.10"]["random_calib"], r["conformal_coverage_alpha0.10"]["earthquake_grouped_calib"]
    print(f"\n=== {tag}  (n={r['n']}, {r['n_groups']} earthquakes, label={r['label']}) ===")
    print(f"  physics AUC      = {p['auc']:.3f}  [95% CI {p['lo']:.3f}, {p['hi']:.3f}]")
    print(f"  ML grouped AUC   = {m['auc']:.3f}  [95% CI {m['lo']:.3f}, {m['hi']:.3f}]")
    print(f"  physics - ML     = {d['delta']:+.3f}  [95% CI {d['lo']:+.3f}, {d['hi']:+.3f}]   P(physics>ML)={d['P(delta>0)']:.3f}")
    print(f"  Brier  physics={r['calibration']['physics']['brier']:.3f}  ML={r['calibration']['ml_grouped']['brier']:.3f}")
    print(f"  ECE    physics={r['calibration']['physics']['ece']:.3f}  ML={r['calibration']['ml_grouped']['ece']:.3f}")
    print(f"  conformal coverage (target 0.90):  random={cr['mean_coverage']:.3f}±{cr['std_coverage']:.3f}  |  earthquake-grouped={cg['mean_coverage']:.3f}±{cg['std_coverage']:.3f}")
    cc = r["conformal_conditional_grouped"]
    print(f"  conformal CONDITIONAL per-earthquake: mean={cc['mean_conditional_coverage']:.3f}  p10={cc['p10_conditional_coverage']:.3f}  min={cc['min_conditional_coverage']:.3f}  frac under-covered={cc['frac_events_undercovered']:.3f}")


print("=" * 74)
print("RELIABILITY UPGRADE — bootstrap CIs, paired physics-vs-ML test, calibration, conformal")
print("=" * 74)
show("SPT / Cetin 2018", spt_res)
show("CPT / Geyin-Maurer 2021", cpt_res)
print("\nwrote: data/processed/reliability_upgrade.json")
