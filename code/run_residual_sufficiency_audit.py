"""Residual sufficiency audit for P1.

This script is deliberately more conservative than the headline sufficiency
script. It tests whether standard public case-history variables add
out-of-distribution information after the mechanistic coordinate is already in
the model, and it separates CPT direct LPI-derived variables from raw-ish
profile/event descriptors.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, "data", "processed")


def cluster_ci(values, groups, n=2000, seed=0):
    values = np.asarray(values, float)
    groups = np.asarray(groups)
    gids = np.unique(groups)
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
        "P_gt_0": round(float(np.mean(np.asarray(boot) > 0)), 4),
    }


def oof_probs(model, X, y, groups):
    cv = list(GroupKFold(5).split(X, y, groups))
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]


def logit():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=5000))


def hgb():
    return make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(random_state=0))


def metric_row(y, p):
    return {
        "logloss": round(float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), 5),
        "auc": round(float(roc_auc_score(y, p)), 5),
    }


def audit_dataset(name, y, groups, margin, variants):
    y = np.asarray(y, int)
    groups = np.asarray(groups)
    margin = np.asarray(margin, float).reshape(-1, 1)

    out = {"n": int(len(y)), "n_groups": int(len(np.unique(groups))), "variants": {}}
    pm = oof_probs(logit(), margin, y, groups)
    out["margin_only_logistic"] = metric_row(y, pm)

    for variant_name, X in variants.items():
        X = np.asarray(X, float)
        X_margin = np.column_stack([margin.ravel(), X])
        for family, model_fn, feat in [
            ("logistic", logit, X_margin),
            ("hist_gbt", hgb, X_margin),
            ("hist_gbt_without_margin", hgb, X),
        ]:
            p = oof_probs(model_fn(), feat, y, groups)
            row = metric_row(y, p)
            diff = -(y * np.log(np.clip(p, 1e-6, 1 - 1e-6)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1 - 1e-6))) - (
                -(y * np.log(np.clip(pm, 1e-6, 1 - 1e-6)) + (1 - y) * np.log(np.clip(1 - pm, 1e-6, 1 - 1e-6)))
            )
            row["dlogloss_variant_minus_margin"] = cluster_ci(diff, groups)
            row["variant_improves_OOS_logloss"] = bool(row["dlogloss_variant_minus_margin"]["hi"] < 0)
            out["variants"][f"{variant_name}::{family}"] = row

    # Narrow-band residual test: can residual variables classify labels where the
    # margin-only predicted probability is ambiguous?
    amb = (pm >= 0.30) & (pm <= 0.70)
    out["ambiguous_band"] = {"n": int(amb.sum()), "prob_range": [0.30, 0.70]}
    if amb.sum() >= 30 and len(np.unique(y[amb])) == 2 and len(np.unique(groups[amb])) >= 5:
        for variant_name, X in variants.items():
            Xb = np.asarray(X, float)[amb]
            yb, gb = y[amb], groups[amb]
            try:
                pb = oof_probs(hgb(), Xb, yb, gb)
                out["ambiguous_band"][variant_name] = metric_row(yb, pb)
            except Exception as exc:
                out["ambiguous_band"][variant_name] = {"error": str(exc)}
    return out


def main():
    spt = pd.read_csv(os.path.join(PROC, "cetin2018_baseline_records.csv"))
    spt_features = ["Mw", "amax_g", "sigma_v_kpa", "sigma_eff_kpa", "rd", "FC", "N1_60", "N1_60cs", "crit_depth_m", "water_depth_ft"]
    spt_raw_no_effstress = ["Mw", "amax_g", "rd", "FC", "N1_60", "N1_60cs", "crit_depth_m"]
    spt = spt.dropna(subset=spt_features + ["y", "earthquake", "CRR_BI2014", "CSR_cetin"]).copy()
    spt = spt[np.isfinite(spt[spt_features].to_numpy(float)).all(1) & (spt["CRR_BI2014"] > 0) & (spt["CSR_cetin"] > 0)]
    spt_margin = np.log((spt["CSR_cetin"] / spt["CRR_BI2014"]).to_numpy(float))

    cpt = pd.read_csv(os.path.join(PROC, "geyin2021_cpt_records.csv"))
    cpt_all = ["Mw", "PGA", "GWT", "LPI", "crit_FS_capped", "crit_depth", "qc1Ncs_crit", "Ic_crit", "sev_crit"]
    cpt_no_lpi_or_fs = ["Mw", "PGA", "GWT", "crit_depth", "qc1Ncs_crit", "Ic_crit"]
    cpt_no_gwt = ["Mw", "PGA", "crit_depth", "qc1Ncs_crit", "Ic_crit"]
    cpt_margin = np.log1p(cpt["LPI"].to_numpy(float))

    out = {
        "interpretation": (
            "Positive dlogloss means the tested variant is worse than the margin-only logistic "
            "coordinate under earthquake-grouped CV. A negative CI entirely below zero would be "
            "evidence that residual/raw variables improve OOS likelihood after the margin."
        ),
        "SPT_Cetin2018": audit_dataset(
            "SPT_Cetin2018",
            spt["y"].astype(int).to_numpy(),
            spt["earthquake"].astype(str).to_numpy(),
            spt_margin,
            {
                "standard_public_features": spt[spt_features].to_numpy(float),
                "raw_no_explicit_effstress_or_gwt": spt[spt_raw_no_effstress].to_numpy(float),
                "groundwater_only_after_margin": spt[["water_depth_ft"]].to_numpy(float),
            },
        ),
        "CPT_Geyin2021": audit_dataset(
            "CPT_Geyin2021",
            cpt["y"].astype(int).to_numpy(),
            cpt["event"].astype(str).to_numpy(),
            cpt_margin,
            {
                "all_features_including_LPI_FS": cpt[cpt_all].to_numpy(float),
                "rawish_no_LPI_or_FS": cpt[cpt_no_lpi_or_fs].to_numpy(float),
                "rawish_no_LPI_FS_or_GWT": cpt[cpt_no_gwt].to_numpy(float),
                "groundwater_only_after_margin": cpt[["GWT"]].to_numpy(float),
            },
        ),
    }

    path = os.path.join(PROC, "residual_sufficiency_audit.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    for dataset in ["SPT_Cetin2018", "CPT_Geyin2021"]:
        print("=" * 78)
        print(dataset, out[dataset]["margin_only_logistic"])
        for key, row in out[dataset]["variants"].items():
            ci = row["dlogloss_variant_minus_margin"]
            print(f"{key:<46} logloss={row['logloss']:.4f} auc={row['auc']:.3f} dLL={ci['mean']:+.4f} CI[{ci['lo']:+.4f},{ci['hi']:+.4f}] improves={row['variant_improves_OOS_logloss']}")
        print("ambiguous band:", out[dataset]["ambiguous_band"])
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
