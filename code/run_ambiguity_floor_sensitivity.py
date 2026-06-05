"""Mechanistic-coordinate ambiguity floor sensitivity for P1.

Earlier drafts used stronger theoretical language. This audit reports a
more defensible empirical lower-bound proxy: label ambiguity after conditioning
on the mechanistic coordinate, under several bin counts and an isotonic-smoothed
coordinate probability.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, "data", "processed")


def grouped_margin_prob(s, y, groups):
    s = np.asarray(s, float).reshape(-1, 1)
    return cross_val_predict(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        s,
        y,
        cv=list(GroupKFold(5).split(s, y, groups)),
        method="predict_proba",
    )[:, 1]


def bin_floor(p, y, n_bins):
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    err = 0.0
    rows = []
    for b in bins:
        freq = float(np.mean(y[b]))
        be = min(freq, 1.0 - freq)
        err += be * len(b)
        rows.append({"p_mean": round(float(np.mean(p[b])), 4), "label_freq": round(freq, 4), "ambiguity": round(be, 4), "n": int(len(b))})
    err /= len(y)
    band_err = sum(r["ambiguity"] * r["n"] for r in rows if 0.30 <= r["p_mean"] <= 0.70)
    return {
        "n_bins": int(n_bins),
        "coordinate_ambiguity_floor": round(float(err), 5),
        "frac_floor_in_0.30_0.70_band": round(float(band_err / (err * len(y) + 1e-12)), 5),
        "bins": rows,
    }


def isotonic_floor(s, y):
    order = np.argsort(s)
    # Fit only for descriptive sensitivity, not as an OOS classifier.
    iso = IsotonicRegression(out_of_bounds="clip").fit(np.asarray(s)[order], np.asarray(y)[order])
    p = iso.predict(np.asarray(s))
    return {
        "descriptive_isotonic_logloss": round(float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), 5),
        "descriptive_isotonic_auc": round(float(roc_auc_score(y, p)), 5),
        "mean_min_p_1minus_p": round(float(np.mean(np.minimum(p, 1.0 - p))), 5),
    }


def audit(name, y, groups, s):
    y = np.asarray(y, int)
    groups = np.asarray(groups)
    p = grouped_margin_prob(s, y, groups)
    out = {
        "n": int(len(y)),
        "n_groups": int(len(np.unique(groups))),
        "margin_grouped_logistic": {
            "auc": round(float(roc_auc_score(y, p)), 5),
            "logloss": round(float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), 5),
        },
        "bin_count_sensitivity": [bin_floor(p, y, nb) for nb in [6, 8, 10, 12, 15]],
        "isotonic_descriptive_sensitivity": isotonic_floor(s, y),
    }
    return out


def main():
    spt = pd.read_csv(os.path.join(PROC, "cetin2018_baseline_records.csv"))
    keep = ["Mw", "amax_g", "sigma_v_kpa", "sigma_eff_kpa", "rd", "FC", "N1_60", "N1_60cs", "crit_depth_m", "water_depth_ft"]
    spt = spt.dropna(subset=keep + ["y", "earthquake", "CRR_BI2014", "CSR_cetin"]).copy()
    spt = spt[np.isfinite(spt[keep].to_numpy(float)).all(1) & (spt["CRR_BI2014"] > 0) & (spt["CSR_cetin"] > 0)]
    spt_s = np.log((spt["CSR_cetin"] / spt["CRR_BI2014"]).to_numpy(float))

    cpt = pd.read_csv(os.path.join(PROC, "geyin2021_cpt_records.csv"))
    cpt_s = np.log1p(cpt["LPI"].to_numpy(float))

    out = {
        "interpretation": (
            "These values are empirical mechanistic-coordinate ambiguity floors, not strict theoretical "
            "errors for the full physical state space. They show how much binary label ambiguity "
            "remains after reducing the available public predictors to the margin coordinate."
        ),
        "SPT_Cetin2018": audit("SPT_Cetin2018", spt["y"].astype(int).to_numpy(), spt["earthquake"].astype(str).to_numpy(), spt_s),
        "CPT_Geyin2021": audit("CPT_Geyin2021", cpt["y"].astype(int).to_numpy(), cpt["event"].astype(str).to_numpy(), cpt_s),
    }
    path = os.path.join(PROC, "ambiguity_floor_sensitivity.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    for dataset in ["SPT_Cetin2018", "CPT_Geyin2021"]:
        print("=" * 72)
        print(dataset, out[dataset]["margin_grouped_logistic"])
        for row in out[dataset]["bin_count_sensitivity"]:
            print(f"bins={row['n_bins']:<2} floor={row['coordinate_ambiguity_floor']:.4f} frac_in_amb_band={row['frac_floor_in_0.30_0.70_band']:.3f}")
        print("isotonic:", out[dataset]["isotonic_descriptive_sensitivity"])
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
