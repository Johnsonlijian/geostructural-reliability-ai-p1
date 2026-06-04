"""Decision-oriented conformal metrics for liquefaction reliability.

Reports efficiency and abstention-like behavior of binary conformal prediction
sets under plain and mechanism-band calibration.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from geoliq.reliability import split_conformal_sets


BASE = Path(__file__).resolve().parent
PROC = BASE / "data" / "processed"


def hgb():
    return HistGradientBoostingClassifier(random_state=0)


def load_spt():
    fs_cols = [
        "Mw",
        "amax_g",
        "sigma_v_kpa",
        "sigma_eff_kpa",
        "rd",
        "FC",
        "N1_60",
        "N1_60cs",
        "crit_depth_m",
        "water_depth_ft",
    ]
    df = pd.read_csv(PROC / "cetin2018_baseline_records.csv")
    df = df.dropna(subset=fs_cols + ["y", "earthquake", "CRR_BI2014", "CSR_cetin"]).copy()
    df = df[
        np.isfinite(df[fs_cols].to_numpy(float)).all(axis=1)
        & (df["CRR_BI2014"] > 0)
        & (df["CSR_cetin"] > 0)
    ]
    return {
        "name": "SPT_Cetin2018",
        "X": df[fs_cols].to_numpy(float),
        "y": df["y"].astype(int).to_numpy(),
        "groups": df["earthquake"].to_numpy(),
    }


def load_cpt():
    cols = ["Mw", "PGA", "GWT", "LPI", "crit_FS_capped", "crit_depth", "qc1Ncs_crit", "Ic_crit", "sev_crit"]
    df = pd.read_csv(PROC / "geyin2021_cpt_records.csv")
    return {
        "name": "CPT_Geyin2021",
        "X": df[cols].to_numpy(float),
        "y": df["y"].astype(int).to_numpy(),
        "groups": df["event"].to_numpy(),
    }


def band_of(prob: np.ndarray) -> np.ndarray:
    return np.clip((prob * 3).astype(int), 0, 2)


def summarize(coverages, sizes):
    sizes = np.asarray(sizes, dtype=float)
    coverages = np.asarray(coverages, dtype=float)
    return {
        "coverage": float(np.mean(coverages)) if len(coverages) else None,
        "mean_set_size": float(np.mean(sizes)) if len(sizes) else None,
        "singleton_rate": float(np.mean(sizes == 1)) if len(sizes) else None,
        "two_label_rate": float(np.mean(sizes == 2)) if len(sizes) else None,
        "empty_rate": float(np.mean(sizes == 0)) if len(sizes) else None,
        "n_predictions": int(len(sizes)),
    }


def evaluate_dataset(D, alpha_values=(0.05, 0.10, 0.20), n_rep=60, seed=11):
    X, y, groups = D["X"], D["y"], D["groups"]
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    result = {"n": int(len(y)), "n_groups": int(len(unique_groups)), "alpha": {}}
    for alpha in alpha_values:
        plain_cov, plain_size = [], []
        band_cov = {0: [], 1: [], 2: []}
        band_size = {0: [], 1: [], 2: []}
        mond_cov, mond_size = [], []
        event_cov = []

        for _ in range(n_rep):
            gids = rng.permutation(unique_groups)
            ng = len(gids)
            train_groups = gids[: int(0.4 * ng)]
            calib_groups = gids[int(0.4 * ng) : int(0.7 * ng)]
            test_groups = gids[int(0.7 * ng) :]
            tr = np.where(np.isin(groups, train_groups))[0]
            ca = np.where(np.isin(groups, calib_groups))[0]
            te = np.where(np.isin(groups, test_groups))[0]
            if len(np.unique(y[tr])) < 2 or len(ca) < 8 or len(te) < 8:
                continue

            model = hgb().fit(X[tr], y[tr])
            p_cal = model.predict_proba(X[ca])[:, 1]
            p_test = model.predict_proba(X[te])[:, 1]

            inc0, inc1 = split_conformal_sets(p_cal, y[ca], p_test, alpha=alpha)
            sizes = inc0.astype(int) + inc1.astype(int)
            cov = np.where(y[te] == 1, inc1, inc0)
            plain_cov.extend(cov.astype(float).tolist())
            plain_size.extend(sizes.astype(int).tolist())

            test_bands = band_of(p_test)
            calib_bands = band_of(p_cal)
            mond_cov_rep = np.zeros(len(te), dtype=bool)
            mond_size_rep = np.zeros(len(te), dtype=int)
            for bb in [0, 1, 2]:
                mca = calib_bands == bb
                mte = test_bands == bb
                if not np.any(mte):
                    continue
                if np.sum(mca) >= 5:
                    j0, j1 = split_conformal_sets(p_cal[mca], y[ca][mca], p_test[mte], alpha=alpha)
                    js = j0.astype(int) + j1.astype(int)
                    jc = np.where(y[te][mte] == 1, j1, j0)
                else:
                    js = sizes[mte]
                    jc = cov[mte]
                mond_size_rep[mte] = js
                mond_cov_rep[mte] = jc
                band_size[bb].extend(js.astype(int).tolist())
                band_cov[bb].extend(jc.astype(float).tolist())
            mond_cov.extend(mond_cov_rep.astype(float).tolist())
            mond_size.extend(mond_size_rep.astype(int).tolist())

            for g in np.unique(groups[te]):
                idx = groups[te] == g
                if np.sum(idx) >= 3:
                    event_cov.append(float(mond_cov_rep[idx].mean()))

        result["alpha"][f"{alpha:.2f}"] = {
            "target_coverage": 1 - alpha,
            "plain": summarize(plain_cov, plain_size),
            "mechanism_band": summarize(mond_cov, mond_size),
            "mechanism_band_by_band": {str(bb): summarize(band_cov[bb], band_size[bb]) for bb in [0, 1, 2]},
            "per_event_mechanism_band": {
                "n_event_evals": int(len(event_cov)),
                "mean": float(np.mean(event_cov)) if event_cov else None,
                "p10": float(np.percentile(event_cov, 10)) if event_cov else None,
                "min": float(np.min(event_cov)) if event_cov else None,
                "frac_below_target": float(np.mean(np.asarray(event_cov) < (1 - alpha))) if event_cov else None,
            },
        }
    return result


def main() -> None:
    out = {
        "interpretation": (
            "Set size 1 is a decisive singleton prediction; set size 2 is an abstention-like "
            "ambiguous prediction set. Empty sets are reported if they occur."
        ),
        "datasets": {},
    }
    for D in [load_spt(), load_cpt()]:
        out["datasets"][D["name"]] = evaluate_dataset(D)
    path = PROC / "conformal_decision_metrics.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for name, res in out["datasets"].items():
        a = res["alpha"]["0.10"]
        print(
            name,
            "plain size", round(a["plain"]["mean_set_size"], 3),
            "band size", round(a["mechanism_band"]["mean_set_size"], 3),
            "band singleton", round(a["mechanism_band"]["singleton_rate"], 3),
            "band two-label", round(a["mechanism_band"]["two_label_rate"], 3),
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
