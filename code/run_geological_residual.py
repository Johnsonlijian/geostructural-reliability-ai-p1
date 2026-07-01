"""Geological structure of liquefaction-triggering predictability (P1 upgrade, honest-B(b)).

WHERE does the effective-stress margin fail? We decompose the margin's out-of-fold calibrated
ambiguity a = min(p, 1-p) (whose case-weighted mean IS the error floor) by engineering-geology
variables: soil behaviour type (CPT I_c), critical-layer depth, groundwater-table depth, penetration
density, and fines content. This turns the predictability ceiling into an engineering-geology
statement — triggering predictability is geologically STRUCTURED, not uniform — using only the
existing case-history variables (no state-parameter assumption).

Outputs data/processed/geological_residual.json. Deterministic (seed 0).
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, "data", "processed")
SEED = 0
EPS = 1e-6


def oof_ambiguity(s, y, groups):
    """Calibrated out-of-fold ambiguity a = min(p_cal, 1-p_cal) for the 1-D margin."""
    s = np.asarray(s, float).reshape(-1, 1)
    y = np.asarray(y, int)
    cv = list(GroupKFold(min(5, len(np.unique(groups)))).split(s, y, groups))
    p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                          s, y, cv=cv, method="predict_proba")[:, 1]
    pc = IsotonicRegression(out_of_bounds="clip").fit(np.clip(p, EPS, 1 - EPS), y.astype(float)).predict(p)
    pc = np.clip(pc, EPS, 1 - EPS)
    return np.minimum(pc, 1 - pc)


def boot_mean(vals, groups, n_boot=1000, seed=SEED):
    vals = np.asarray(vals, float); groups = np.asarray(groups)
    if len(vals) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    gids = np.unique(groups); mem = {g: np.where(groups == g)[0] for g in gids}
    bo = []
    for _ in range(n_boot):
        idx = np.concatenate([mem[g] for g in rng.choice(gids, len(gids), replace=True)])
        bo.append(float(np.mean(vals[idx])))
    return {"mean": round(float(np.mean(vals)), 4), "lo": round(float(np.percentile(bo, 2.5)), 4),
            "hi": round(float(np.percentile(bo, 97.5)), 4), "n": int(len(vals))}


def stratify(a, factor, groups, edges, labels):
    factor = np.asarray(factor, float)
    out = {}
    for i, lab in enumerate(labels):
        m = (factor >= edges[i]) & (factor < edges[i + 1])
        out[lab] = boot_mean(a[m], np.asarray(groups)[m])
    return out


def contrast(a, factor, groups, mA, mB, n_boot=1000, seed=SEED):
    """Difference in mean ambiguity between two masks, cluster-bootstrap by event."""
    factor = np.asarray(factor, float); a = np.asarray(a, float); groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    gids = np.unique(groups); mem = {g: np.where(groups == g)[0] for g in gids}
    bo = []
    for _ in range(n_boot):
        idx = np.concatenate([mem[g] for g in rng.choice(gids, len(gids), replace=True)])
        fa, aa = factor[idx], a[idx]
        A, B = mA(fa), mB(fa)
        if A.sum() and B.sum():
            bo.append(float(np.mean(aa[A]) - np.mean(aa[B])))
    d = float(np.mean(a[mA(factor)]) - np.mean(a[mB(factor)]))
    return {"diff": round(d, 4), "lo": round(float(np.percentile(bo, 2.5)), 4),
            "hi": round(float(np.percentile(bo, 97.5)), 4), "P(diff>0)": round(float(np.mean(np.array(bo) > 0)), 3)}


def main():
    out = {"method": ("Calibrated out-of-fold margin ambiguity a=min(p,1-p) stratified by engineering-"
                      "geology variables; cluster-bootstrap CIs by earthquake. Mean a = error floor."),
           "seed": SEED}

    # ---- CPT geology — soil behaviour type I_c, depth, GWT, density — on TWO independent releases ----
    def cpt_geology(fname):
        cpt = pd.read_csv(os.path.join(PROC, fname))
        cpt = cpt.dropna(subset=["LPI", "y", "event", "Ic_crit", "crit_depth", "GWT", "qc1Ncs_crit"]).copy()
        a = oof_ambiguity(np.log1p(cpt["LPI"].to_numpy(float)), cpt["y"].to_numpy(int), cpt["event"].to_numpy(str))
        g = cpt["event"].to_numpy(str)
        return {
            "overall_floor": boot_mean(a, g),
            "by_soil_behaviour_type_Ic": stratify(a, cpt["Ic_crit"], g, [0, 2.05, 2.60, 10],
                ["sand-like (Ic<2.05)", "transitional (2.05-2.60)", "clay-like (Ic>2.60)"]),
            "by_critical_depth_m": stratify(a, cpt["crit_depth"], g, [0, 3, 6, 100],
                ["shallow (<3m)", "mid (3-6m)", "deep (>6m)"]),
            "by_groundwater_depth_m": stratify(a, cpt["GWT"], g, [0, 1.5, 3, 100],
                ["very shallow (<1.5m)", "shallow (1.5-3m)", "deeper (>3m)"]),
            "by_density_qc1Ncs": stratify(a, cpt["qc1Ncs_crit"], g, [0, 100, 150, 1000],
                ["loose (<100)", "medium (100-150)", "dense (>150)"]),
            "contrast_transitional_vs_sandlike_Ic": contrast(a, cpt["Ic_crit"].to_numpy(float), g,
                lambda f: (f >= 2.05) & (f < 2.60), lambda f: f < 2.05),
        }
    out["CPT_Geyin2021"] = cpt_geology("geyin2021_cpt_records.csv")
    out["CPT_Rateria2024_independent_replication"] = cpt_geology("cpt2024_records_measured.csv")

    # ---- SPT (Cetin 2018): fines, depth, water table, density ----
    spt = pd.read_csv(os.path.join(PROC, "cetin2018_baseline_records.csv"))
    spt = spt.dropna(subset=["CSR_cetin", "CRR_BI2014", "y", "earthquake", "FC", "crit_depth_m",
                             "water_depth_ft", "N1_60cs"]).copy()
    spt = spt[(spt["CRR_BI2014"] > 0) & (spt["CSR_cetin"] > 0)]
    s = np.log((spt["CSR_cetin"] / spt["CRR_BI2014"]).to_numpy(float))
    a2 = oof_ambiguity(s, spt["y"].to_numpy(int), spt["earthquake"].to_numpy(str))
    g2 = spt["earthquake"].to_numpy(str)
    out["SPT_Cetin2018"] = {
        "overall_floor": boot_mean(a2, g2),
        "by_fines_content_FC": stratify(a2, spt["FC"], g2, [0, 5, 35, 1000],
            ["clean sand (FC<5)", "silty (5-35)", "fines-rich (>35)"]),
        "by_critical_depth_m": stratify(a2, spt["crit_depth_m"], g2, [0, 4, 8, 100],
            ["shallow (<4m)", "mid (4-8m)", "deep (>8m)"]),
        "by_density_N1_60cs": stratify(a2, spt["N1_60cs"], g2, [0, 12, 20, 1000],
            ["loose (<12)", "medium (12-20)", "dense (>20)"]),
        "contrast_silty_vs_cleansand_FC": contrast(a2, spt["FC"].to_numpy(float), g2,
            lambda f: (f >= 5) & (f < 35), lambda f: f < 5),
    }

    path = os.path.join(PROC, "geological_residual.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    for arm in ("CPT_Geyin2021", "CPT_Rateria2024_independent_replication", "SPT_Cetin2018"):
        print("=" * 78); print(arm, "overall floor", out[arm]["overall_floor"]["mean"])
        for k, v in out[arm].items():
            if k.startswith("by_"):
                print(f"  {k}:")
                for lab, st in v.items():
                    print(f"      {lab:26s} floor={st['mean']:.3f} [{st['lo']:.3f},{st['hi']:.3f}] (n={st['n']})")
        c = [v for k, v in out[arm].items() if k.startswith("contrast")][0]
        print(f"  CONTRAST {[k for k in out[arm] if k.startswith('contrast')][0]}: "
              f"Δ={c['diff']:+.3f} [{c['lo']:+.3f},{c['hi']:+.3f}] P(>0)={c['P(diff>0)']:.2f}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
