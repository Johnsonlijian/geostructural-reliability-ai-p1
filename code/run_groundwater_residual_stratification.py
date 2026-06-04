"""Groundwater-depth residual stratification for P1.

This is a conservative diagnostic for the groundwater claim. It asks whether
margin-only predictions leave systematic residual bias across groundwater-depth
bins, and whether a flexible full-feature ML model removes such bias under the
same earthquake-grouped validation.

The output is descriptive and diagnostic, not causal identification.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, "data", "processed")


def margin_model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=5000))


def full_model():
    return make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(random_state=0))


def oof_probs(model, X, y, groups):
    cv = list(GroupKFold(5).split(X, y, groups))
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]


def cluster_ci(values, groups, n=2000, seed=7):
    values = np.asarray(values, float)
    groups = np.asarray(groups)
    gids = np.unique(groups)
    if len(gids) < 3 or len(values) == 0:
        mean = float(np.mean(values)) if len(values) else float("nan")
        return {"mean": round(mean, 5), "lo": None, "hi": None, "n_groups": int(len(gids))}
    mem = {gid: np.where(groups == gid)[0] for gid in gids}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n):
        take = rng.choice(gids, size=len(gids), replace=True)
        idx = np.concatenate([mem[gid] for gid in take])
        boot.append(float(values[idx].mean()))
    return {
        "mean": round(float(values.mean()), 5),
        "lo": round(float(np.percentile(boot, 2.5)), 5),
        "hi": round(float(np.percentile(boot, 97.5)), 5),
        "n_groups": int(len(gids)),
    }


def bin_rows(df, group_col, depth_col, y_col, p_margin, p_full):
    bins = [-np.inf, 2.0, 5.0, 10.0, np.inf]
    labels = ["<=2 m", "2-5 m", "5-10 m", ">10 m"]
    out = []
    work = df.copy()
    work["_bin"] = pd.cut(work[depth_col].to_numpy(float), bins=bins, labels=labels)
    work["_p_margin"] = p_margin
    work["_p_full"] = p_full
    work["_resid_margin"] = work[y_col].to_numpy(float) - p_margin
    work["_resid_full"] = work[y_col].to_numpy(float) - p_full
    for label in labels:
        sub = work[work["_bin"].astype(str) == label].copy()
        if sub.empty:
            continue
        groups = sub[group_col].astype(str).to_numpy()
        margin_ci = cluster_ci(sub["_resid_margin"].to_numpy(float), groups)
        full_ci = cluster_ci(sub["_resid_full"].to_numpy(float), groups)
        out.append(
            {
                "groundwater_bin": label,
                "n": int(len(sub)),
                "n_groups": int(sub[group_col].nunique()),
                "mean_depth_m": round(float(sub[depth_col].mean()), 3),
                "observed_rate": round(float(sub[y_col].mean()), 5),
                "margin_pred_mean": round(float(sub["_p_margin"].mean()), 5),
                "full_ml_pred_mean": round(float(sub["_p_full"].mean()), 5),
                "margin_residual_y_minus_p": margin_ci,
                "full_ml_residual_y_minus_p": full_ci,
                "abs_residual_reduction_full_minus_margin": round(
                    float(abs(full_ci["mean"]) - abs(margin_ci["mean"])), 5
                ),
            }
        )
    return out


def summarize(rows):
    return {
        "max_abs_margin_bin_residual": round(float(max(abs(r["margin_residual_y_minus_p"]["mean"]) for r in rows)), 5),
        "max_abs_full_ml_bin_residual": round(float(max(abs(r["full_ml_residual_y_minus_p"]["mean"]) for r in rows)), 5),
        "n_bins": int(len(rows)),
        "bins": rows,
    }


def main():
    spt = pd.read_csv(os.path.join(PROC, "cetin2018_baseline_records.csv"))
    spt_features = ["Mw", "amax_g", "sigma_v_kpa", "sigma_eff_kpa", "rd", "FC", "N1_60", "N1_60cs", "crit_depth_m", "water_depth_ft"]
    spt = spt.dropna(subset=spt_features + ["y", "earthquake", "CSR_cetin", "CRR_BI2014"]).copy()
    spt = spt[np.isfinite(spt[spt_features].to_numpy(float)).all(1) & (spt["CRR_BI2014"] > 0) & (spt["CSR_cetin"] > 0)].copy()
    spt["groundwater_depth_m"] = spt["water_depth_ft"].astype(float) * 0.3048
    spt_margin = np.log((spt["CSR_cetin"] / spt["CRR_BI2014"]).to_numpy(float)).reshape(-1, 1)
    spt_full = spt[spt_features].to_numpy(float)
    spt_y = spt["y"].astype(int).to_numpy()
    spt_groups = spt["earthquake"].astype(str).to_numpy()
    spt_pm = oof_probs(margin_model(), spt_margin, spt_y, spt_groups)
    spt_pf = oof_probs(full_model(), spt_full, spt_y, spt_groups)

    cpt = pd.read_csv(os.path.join(PROC, "geyin2021_cpt_records.csv")).copy()
    cpt_features = ["Mw", "PGA", "GWT", "LPI", "crit_FS_capped", "crit_depth", "qc1Ncs_crit", "Ic_crit", "sev_crit"]
    cpt_margin = np.log1p(cpt["LPI"].to_numpy(float)).reshape(-1, 1)
    cpt_full = cpt[cpt_features].to_numpy(float)
    cpt_y = cpt["y"].astype(int).to_numpy()
    cpt_groups = cpt["event"].astype(str).to_numpy()
    cpt_pm = oof_probs(margin_model(), cpt_margin, cpt_y, cpt_groups)
    cpt_pf = oof_probs(full_model(), cpt_full, cpt_y, cpt_groups)

    out = {
        "interpretation": (
            "Residuals are observed label minus grouped out-of-fold predicted probability. "
            "Bins are by groundwater depth. Values near zero indicate little systematic bias "
            "in that groundwater-depth stratum. This is a diagnostic for residual bias, not "
            "causal identification."
        ),
        "SPT_Cetin2018": summarize(
            bin_rows(spt, "earthquake", "groundwater_depth_m", "y", spt_pm, spt_pf)
        ),
        "CPT_Geyin2021": summarize(
            bin_rows(cpt, "event", "GWT", "y", cpt_pm, cpt_pf)
        ),
    }
    path = os.path.join(PROC, "groundwater_residual_stratification.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    for dataset in ["SPT_Cetin2018", "CPT_Geyin2021"]:
        print("=" * 78)
        print(dataset, {k: out[dataset][k] for k in ["max_abs_margin_bin_residual", "max_abs_full_ml_bin_residual", "n_bins"]})
        for row in out[dataset]["bins"]:
            mr = row["margin_residual_y_minus_p"]
            fr = row["full_ml_residual_y_minus_p"]
            print(
                f"{row['groundwater_bin']:<7} n={row['n']:<3} groups={row['n_groups']:<2} "
                f"obs={row['observed_rate']:.3f} margin_p={row['margin_pred_mean']:.3f} "
                f"resid={mr['mean']:+.3f} CI[{mr['lo']},{mr['hi']}] "
                f"full_resid={fr['mean']:+.3f}"
            )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

