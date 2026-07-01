"""Rigorous predictability ceiling for liquefaction triggering (P1 upgrade, 2026-06-30).

For each measurement modality we estimate the predictability ceiling of triggering
prediction and test whether the 1-D mechanistic margin coordinate already *reaches* it --
i.e. whether flexible multivariate models can do better under honest earthquake-grouped
validation. This replaces the earlier binned "ambiguity floor" (no CI, single estimator,
partly circular) with a calibration-robust, multi-metric, CI-bearing analysis.

Why calibration-robust: the naive Bayes-error proxy E[min(p,1-p)] is only valid for
*calibrated* posteriors. A sharp/overconfident model (e.g. gradient boosting) drives
min(p,1-p) down without discriminating better, faking "epistemic headroom." We therefore
(a) lead with the calibration-free discrimination ceiling (grouped AUC), (b) report the
proper-score floor (out-of-fold log-loss / Brier), and (c) report an irreducible-error
floor computed on ISOTONIC-CALIBRATED out-of-fold posteriors (same treatment for every
model, so the comparison is fair). All carry cluster-bootstrap CIs by earthquake.

Sufficiency test (the "critical-state control" claim): does the best multivariate model
beat the 1-D mechanistic margin on grouped AUC or on the calibrated error floor? If not
(CIs do not show the multivariate model strictly better), the mechanistic / effective-
stress coordinate -- not model class -- sets the ceiling. This is the mechanistic
explanation for the ~20-year model plateau of Geyin, Baird & Maurer (2020, Earthq.
Spectra).

Mechanistic margin coordinates (never refit on labels):
  SPT (Cetin 2018, triggering)        s = ln(CSR_cetin / CRR_BI2014)
  CPT (Geyin-Maurer 2021, manifest.)  s = ln(1 + LPI)

Outputs data/processed/predictability_ceiling.json. Deterministic (seed 0).
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from geoliq.reliability import bootstrap_auc_ci, paired_auc_diff_ci

BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, "data", "processed")
SEED = 0
N_BOOT = 1000
EPS = 1e-6


def _oof_proba(estimator, X, y, groups, n_splits):
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    cv = list(GroupKFold(n_splits).split(X, y, groups))
    return cross_val_predict(estimator, X, y, cv=cv, method="predict_proba")[:, 1]


def _calibrated_ambiguity(p, y):
    """Per-point min(p_cal, 1-p_cal) after isotonic calibration of OOF posteriors.
    Isotonic is monotone (AUC-preserving) and maps scores to observed frequencies, so this
    is a fair, overconfidence-proof irreducible-error proxy applied identically to every
    model."""
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    pc = IsotonicRegression(out_of_bounds="clip").fit(p, np.asarray(y, float)).predict(p)
    pc = np.clip(pc, EPS, 1 - EPS)
    return np.minimum(pc, 1 - pc)


def _logloss_points(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _min_error(p, y):
    """Out-of-fold misclassification error at the error-minimizing threshold."""
    p = np.asarray(p, float)
    y = np.asarray(y, int)
    thr = np.unique(p)
    best = 1.0
    for t in thr:
        best = min(best, float(np.mean((p >= t).astype(int) != y)))
    return best


def _boot_mean_ci(per_point, groups, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    vals = np.asarray(per_point, float)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    gids = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in gids}
    boots = []
    for _ in range(n_boot):
        sg = rng.choice(gids, len(gids), replace=True)
        idx = np.concatenate([members[g] for g in sg])
        boots.append(float(np.mean(vals[idx])))
    return {"est": round(float(np.mean(vals)), 4),
            "lo": round(float(np.percentile(boots, 100 * alpha / 2)), 4),
            "hi": round(float(np.percentile(boots, 100 * (1 - alpha / 2))), 4)}


def _boot_paired_floor(per_point_a, per_point_b, groups, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    """gap = floor(a) - floor(b), cluster-bootstrap. Here a = best multivariate, b = margin;
    gap < 0 means the multivariate model has a LOWER (better) error floor than the margin."""
    d = np.asarray(per_point_a, float) - np.asarray(per_point_b, float)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    gids = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in gids}
    boots = []
    for _ in range(n_boot):
        sg = rng.choice(gids, len(gids), replace=True)
        idx = np.concatenate([members[g] for g in sg])
        boots.append(float(np.mean(d[idx])))
    boots = np.array(boots)
    return {"gap_mv_minus_margin": round(float(np.mean(d)), 4),
            "lo": round(float(np.percentile(boots, 2.5)), 4),
            "hi": round(float(np.percentile(boots, 97.5)), 4),
            "P(mv_lower_floor)": round(float(np.mean(boots < 0)), 3)}


def _label_noise_decomp(e_obs, rhos=(0.0, 0.05, 0.10, 0.15)):
    """Symmetric label noise rho: observed floor e_obs = rho + (1-2 rho) e_int -> solve e_int."""
    return [{"rho": rho, "intrinsic_floor": round(max((e_obs - rho) / max(1 - 2 * rho, EPS), 0.0), 4)}
            for rho in rhos]


def analyze(y, groups, margin_s, X_full, feat_names, response):
    y = np.asarray(y, int)
    groups = np.asarray(groups)
    n_splits = min(5, len(np.unique(groups)))

    models = {
        "margin_1d": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                      np.asarray(margin_s, float).reshape(-1, 1)),
        "logistic_full": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), X_full),
        "knn": (make_pipeline(StandardScaler(),
                KNeighborsClassifier(n_neighbors=max(5, int(round(np.sqrt(len(y))))))), X_full),
        "gb": (HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
               l2_regularization=1.0, random_state=SEED), X_full),
    }

    res, oof, cal_amb = {}, {}, {}
    for name, (est, X) in models.items():
        p = _oof_proba(est, X, y, groups, n_splits)
        oof[name] = p
        cal_amb[name] = _calibrated_ambiguity(p, y)
        res[name] = {
            "grouped_auc": {k: round(v, 4) for k, v in
                            bootstrap_auc_ci(y, p, groups=groups, seed=SEED, n_boot=N_BOOT).items()},
            "logloss_floor": _boot_mean_ci(_logloss_points(p, y), groups),
            "calibrated_error_floor": _boot_mean_ci(cal_amb[name], groups),
            "min_error_rate": round(_min_error(p, y), 4),
        }

    # best multivariate model = highest grouped AUC among the non-margin models
    mv = max(("logistic_full", "knn", "gb"), key=lambda m: res[m]["grouped_auc"]["auc"])
    d_auc = paired_auc_diff_ci(y, oof[mv], oof["margin_1d"], groups=groups, seed=SEED, n_boot=N_BOOT)
    d_floor = _boot_paired_floor(cal_amb[mv], cal_amb["margin_1d"], groups)
    beats_auc = d_auc["lo"] > 0          # mv strictly higher AUC than margin
    beats_floor = d_floor["hi"] < 0      # mv strictly lower error floor than margin
    margin_sufficient = not (beats_auc or beats_floor)

    e = res["margin_1d"]["calibrated_error_floor"]["est"]
    return {
        "response": response,
        "n": int(len(y)), "n_earthquakes": int(len(np.unique(groups))),
        "base_rate": round(float(np.mean(y)), 4),
        "models": res,
        "best_multivariate": mv,
        "sufficiency_test": {
            "best_mv_minus_margin_AUC": {k: round(v, 4) for k, v in d_auc.items()},
            "best_mv_minus_margin_error_floor": d_floor,
            "multivariate_beats_margin_on_AUC": bool(beats_auc),
            "multivariate_beats_margin_on_error_floor": bool(beats_floor),
            "margin_is_sufficient_coordinate": bool(margin_sufficient),
        },
        "predictability_ceiling": {
            "discrimination_AUC_ceiling": res["margin_1d"]["grouped_auc"],
            "irreducible_error_floor_calibrated": res["margin_1d"]["calibrated_error_floor"],
            "min_error_rate": res["margin_1d"]["min_error_rate"],
        },
        "label_noise_sensitivity_on_margin_floor": _label_noise_decomp(e),
        "features": feat_names,
    }


def load_spt():
    df = pd.read_csv(os.path.join(PROC, "cetin2018_baseline_records.csv"))
    feats = ["Mw", "amax_g", "sigma_v_kpa", "sigma_eff_kpa", "rd", "FC",
             "N1_60", "N1_60cs", "crit_depth_m", "water_depth_ft"]
    df = df.dropna(subset=feats + ["y", "earthquake", "CRR_BI2014", "CSR_cetin"]).copy()
    df = df[(df["CRR_BI2014"] > 0) & (df["CSR_cetin"] > 0)
            & np.isfinite(df[feats].to_numpy(float)).all(1)]
    s = np.log((df["CSR_cetin"] / df["CRR_BI2014"]).to_numpy(float))
    return ("SPT_Cetin2018", dict(y=df["y"].astype(int).to_numpy(),
            groups=df["earthquake"].astype(str).to_numpy(), margin_s=s,
            X_full=df[feats].to_numpy(float), feat_names=feats, response="triggering"))


def load_cpt():
    df = pd.read_csv(os.path.join(PROC, "geyin2021_cpt_records.csv"))
    feats = ["Mw", "PGA", "GWT", "crit_FS", "crit_depth", "qc1Ncs_crit",
             "Ic_crit", "sev_crit", "n_liq_layers", "LPI"]
    df = df.dropna(subset=feats + ["y", "event"]).copy()
    df = df[np.isfinite(df[feats].to_numpy(float)).all(1)]
    s = np.log1p(df["LPI"].to_numpy(float))
    return ("CPT_Geyin2021", dict(y=df["y"].astype(int).to_numpy(),
            groups=df["event"].astype(str).to_numpy(), margin_s=s,
            X_full=df[feats].to_numpy(float), feat_names=feats, response="surface_manifestation"))


def load_vs():
    path = os.path.join(PROC, "vs_kayen2013_records.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    feats = ["vs1", "csr", "mw", "sigma_eff_kpa", "fc"]
    return ("Vs_Kayen2013", dict(y=df["y"].astype(int).to_numpy(),
            groups=df["event"].astype(str).to_numpy(), margin_s=df["s_Vs"].to_numpy(float),
            X_full=df[feats].to_numpy(float), feat_names=feats, response="triggering"))


def main():
    out = {"method": ("Calibration-robust predictability ceiling: grouped-AUC discrimination "
                      "ceiling + OOF log-loss floor + isotonic-calibrated irreducible-error floor, "
                      "all cluster-bootstrapped by earthquake. Sufficiency test asks whether the best "
                      "multivariate model beats the 1-D mechanistic margin. Margins never refit on labels."),
           "seed": SEED, "n_boot": N_BOOT}
    for loader in (load_spt, load_cpt, load_vs):
        r = loader()
        if r is None:
            out["Vs_Kayen2013"] = "absent (vs_kayen2013_records.csv not built)"
            continue
        name, d = r
        out[name] = analyze(d["y"], d["groups"], d["margin_s"], d["X_full"],
                            d["feat_names"], d["response"])
    path = os.path.join(PROC, "predictability_ceiling.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    for name in ("SPT_Cetin2018", "CPT_Geyin2021", "Vs_Kayen2013"):
        if not isinstance(out.get(name), dict):
            continue
        r = out[name]
        st = r["sufficiency_test"]
        pc = r["predictability_ceiling"]
        print("=" * 80)
        print(f"{name} (n={r['n']}, EQ={r['n_earthquakes']}, base={r['base_rate']}, {r['response']})")
        print(f"  margin   AUC {r['models']['margin_1d']['grouped_auc']['auc']:.3f}"
              f"  | best-mv ({r['best_multivariate']}) AUC {r['models'][r['best_multivariate']]['grouped_auc']['auc']:.3f}")
        print(f"  ceiling: discrimination AUC {pc['discrimination_AUC_ceiling']['auc']:.3f}"
              f" [{pc['discrimination_AUC_ceiling']['lo']:.3f},{pc['discrimination_AUC_ceiling']['hi']:.3f}]")
        print(f"           irreducible error floor {pc['irreducible_error_floor_calibrated']['est']:.3f}"
              f" [{pc['irreducible_error_floor_calibrated']['lo']:.3f},{pc['irreducible_error_floor_calibrated']['hi']:.3f}]"
              f"  (min-error {pc['min_error_rate']:.3f})")
        print(f"  sufficiency: mv beats margin on AUC? {st['multivariate_beats_margin_on_AUC']}"
              f" | on error floor? {st['multivariate_beats_margin_on_error_floor']}"
              f"  =>  margin sufficient: {st['margin_is_sufficient_coordinate']}")
        print(f"    dAUC(mv-margin) {st['best_mv_minus_margin_AUC']['delta']:+.3f}"
              f" [{st['best_mv_minus_margin_AUC']['lo']:+.3f},{st['best_mv_minus_margin_AUC']['hi']:+.3f}]")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
