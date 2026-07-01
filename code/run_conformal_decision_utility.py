"""Decision-utility audit for mechanism-band conformal prediction sets.

Coverage and mean set size are useful but incomplete for engineering use.
This script converts conformal sets into a conservative decision policy.
Mechanism bands are defined from a train-only logistic calibration of the
published margin coordinate:

* singleton set {0} or {1}: make the indicated triggering/non-triggering call;
* two-label set {0,1}: send the case to additional investigation or conservative
  review;
* empty set: treat as failed calibration and investigate.

The utility calculation is deliberately simple and transparent. It is not a
design-code cost model; it asks whether conformal abstention can reduce costly
misclassification relative to always forcing a binary ML decision.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from geoliq.reliability import split_conformal_sets


BASE = Path(__file__).resolve().parent
PROC = BASE / "data" / "processed"
ALPHAS = (0.05, 0.10, 0.20)
N_REP = 60

COST_SCENARIOS = {
    "fn5_fp1_review0.25": {"false_negative": 5.0, "false_positive": 1.0, "review": 0.25, "empty": 1.0},
    "fn5_fp1_review0.50": {"false_negative": 5.0, "false_positive": 1.0, "review": 0.50, "empty": 1.25},
    "fn10_fp1_review0.50": {"false_negative": 10.0, "false_positive": 1.0, "review": 0.50, "empty": 1.50},
}


def hgb() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(random_state=0)


def load_spt() -> dict:
    cols = [
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
    df = df.dropna(subset=cols + ["y", "earthquake"]).copy()
    df = df[np.isfinite(df[cols].to_numpy(float)).all(axis=1)].copy()
    return {
        "name": "SPT_Cetin2018",
        "X": df[cols].to_numpy(float),
        "y": df["y"].astype(int).to_numpy(),
        "groups": df["earthquake"].to_numpy(str),
        "margin": np.log((df["CSR_cetin"] / df["CRR_BI2014"]).to_numpy(float)),
    }


def load_cpt() -> dict:
    cols = ["Mw", "PGA", "GWT", "LPI", "crit_FS_capped", "crit_depth", "qc1Ncs_crit", "Ic_crit", "sev_crit"]
    df = pd.read_csv(PROC / "geyin2021_cpt_records.csv")
    df = df.dropna(subset=cols + ["y", "event"]).copy()
    df = df[np.isfinite(df[cols].to_numpy(float)).all(axis=1)].copy()
    return {
        "name": "CPT_Geyin2021",
        "X": df[cols].to_numpy(float),
        "y": df["y"].astype(int).to_numpy(),
        "groups": df["event"].to_numpy(str),
        "margin": np.log1p(np.clip(df["LPI"].to_numpy(float), 0, None)),
    }


def band_of_margin_probability(prob: np.ndarray) -> np.ndarray:
    return np.clip((prob * 3).astype(int), 0, 2)


def fit_margin_probability(margin, y, train_idx, eval_idx):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    model.fit(np.asarray(margin[train_idx], float).reshape(-1, 1), y[train_idx])
    return model.predict_proba(np.asarray(margin[eval_idx], float).reshape(-1, 1))[:, 1]


def summarize_binary_policy(y: np.ndarray, pred: np.ndarray, costs: dict) -> float:
    false_negative = (pred == 0) & (y == 1)
    false_positive = (pred == 1) & (y == 0)
    return float(np.mean(false_negative * costs["false_negative"] + false_positive * costs["false_positive"]))


def summarize_sets(y: np.ndarray, p: np.ndarray, inc0: np.ndarray, inc1: np.ndarray, bands: np.ndarray | None) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    inc0 = np.asarray(inc0, dtype=bool)
    inc1 = np.asarray(inc1, dtype=bool)
    sizes = inc0.astype(int) + inc1.astype(int)
    singleton0 = inc0 & ~inc1
    singleton1 = inc1 & ~inc0
    singleton = singleton0 | singleton1
    two_label = inc0 & inc1
    empty = ~inc0 & ~inc1
    pred = np.where(singleton1, 1, np.where(singleton0, 0, -1))
    singleton_errors = singleton & (pred != y)
    singleton_count = int(np.sum(singleton))
    hard_pred = (p >= 0.5).astype(int)

    out = {
        "coverage": float(np.mean(np.where(y == 1, inc1, inc0))) if len(y) else None,
        "mean_set_size": float(np.mean(sizes)) if len(y) else None,
        "singleton_rate": float(np.mean(singleton)) if len(y) else None,
        "two_label_rate": float(np.mean(two_label)) if len(y) else None,
        "empty_rate": float(np.mean(empty)) if len(y) else None,
        "singleton_error_rate": float(np.sum(singleton_errors) / singleton_count) if singleton_count else None,
        "singleton_false_negative_rate_per_case": float(np.mean(singleton0 & (y == 1))) if len(y) else None,
        "singleton_false_positive_rate_per_case": float(np.mean(singleton1 & (y == 0))) if len(y) else None,
        "hard_binary_error_rate": float(np.mean(hard_pred != y)) if len(y) else None,
        "hard_false_negative_rate_per_case": float(np.mean((hard_pred == 0) & (y == 1))) if len(y) else None,
        "hard_false_positive_rate_per_case": float(np.mean((hard_pred == 1) & (y == 0))) if len(y) else None,
        "n_predictions": int(len(y)),
        "cost_scenarios": {},
    }
    for name, costs in COST_SCENARIOS.items():
        hard_cost = summarize_binary_policy(y, hard_pred, costs)
        singleton_cost = (
            ((singleton0 & (y == 1)) * costs["false_negative"])
            + ((singleton1 & (y == 0)) * costs["false_positive"])
            + (two_label * costs["review"])
            + (empty * costs["empty"])
        )
        conformal_cost = float(np.mean(singleton_cost))
        out["cost_scenarios"][name] = {
            "always_classify_cost_per_case": hard_cost,
            "conformal_policy_cost_per_case": conformal_cost,
            "cost_reduction_per_case": hard_cost - conformal_cost,
            "relative_cost_reduction": (hard_cost - conformal_cost) / hard_cost if hard_cost > 0 else None,
        }

    if bands is not None:
        out["by_mechanism_band"] = {}
        for band in [0, 1, 2]:
            mask = bands == band
            if not np.any(mask):
                continue
            band_out = summarize_sets(y[mask], p[mask], inc0[mask], inc1[mask], None)
            out["by_mechanism_band"][str(band)] = {
                key: band_out[key]
                for key in [
                    "coverage",
                    "mean_set_size",
                    "singleton_rate",
                    "two_label_rate",
                    "empty_rate",
                    "singleton_error_rate",
                    "hard_binary_error_rate",
                    "n_predictions",
                ]
            }
    return out


def aggregate(records: list[dict]) -> dict:
    if not records:
        return {}
    keys = [
        "coverage",
        "mean_set_size",
        "singleton_rate",
        "two_label_rate",
        "empty_rate",
        "singleton_error_rate",
        "singleton_false_negative_rate_per_case",
        "singleton_false_positive_rate_per_case",
        "hard_binary_error_rate",
        "hard_false_negative_rate_per_case",
        "hard_false_positive_rate_per_case",
    ]
    out = {"n_repeats_used": len(records)}
    for key in keys:
        values = [r[key] for r in records if r.get(key) is not None]
        out[key] = float(np.mean(values)) if values else None
    out["n_predictions_total"] = int(sum(r["n_predictions"] for r in records))
    out["cost_scenarios"] = {}
    for scenario in COST_SCENARIOS:
        out["cost_scenarios"][scenario] = {}
        for key in ["always_classify_cost_per_case", "conformal_policy_cost_per_case", "cost_reduction_per_case", "relative_cost_reduction"]:
            values = [r["cost_scenarios"][scenario][key] for r in records if r["cost_scenarios"][scenario][key] is not None]
            out["cost_scenarios"][scenario][key] = float(np.mean(values)) if values else None
    if any("by_mechanism_band" in r for r in records):
        out["by_mechanism_band"] = {}
        for band in ["0", "1", "2"]:
            band_records = [r["by_mechanism_band"][band] for r in records if r.get("by_mechanism_band", {}).get(band)]
            if not band_records:
                continue
            out["by_mechanism_band"][band] = {}
            for key in [
                "coverage",
                "mean_set_size",
                "singleton_rate",
                "two_label_rate",
                "empty_rate",
                "singleton_error_rate",
                "hard_binary_error_rate",
            ]:
                values = [r[key] for r in band_records if r.get(key) is not None]
                out["by_mechanism_band"][band][key] = float(np.mean(values)) if values else None
            out["by_mechanism_band"][band]["n_predictions_total"] = int(sum(r["n_predictions"] for r in band_records))
    return out


def evaluate_dataset(D: dict, seed: int = 43) -> dict:
    X, y, groups, margin = D["X"], D["y"], D["groups"], D["margin"]
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    out = {"n": int(len(y)), "n_groups": int(len(unique_groups)), "alpha": {}}

    for alpha in ALPHAS:
        plain_records = []
        band_records = []
        for _ in range(N_REP):
            gids = rng.permutation(unique_groups)
            ng = len(gids)
            train_groups = gids[: int(0.4 * ng)]
            calib_groups = gids[int(0.4 * ng) : int(0.7 * ng)]
            test_groups = gids[int(0.7 * ng) :]
            tr = np.where(np.isin(groups, train_groups))[0]
            ca = np.where(np.isin(groups, calib_groups))[0]
            te = np.where(np.isin(groups, test_groups))[0]
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[ca])) < 2 or len(te) < 8:
                continue

            model = hgb().fit(X[tr], y[tr])
            p_cal = model.predict_proba(X[ca])[:, 1]
            p_test = model.predict_proba(X[te])[:, 1]
            pm_cal = fit_margin_probability(margin, y, tr, ca)
            pm_test = fit_margin_probability(margin, y, tr, te)

            inc0, inc1 = split_conformal_sets(p_cal, y[ca], p_test, alpha=alpha)
            plain_records.append(summarize_sets(y[te], p_test, inc0, inc1, band_of_margin_probability(pm_test)))

            calib_bands = band_of_margin_probability(pm_cal)
            test_bands = band_of_margin_probability(pm_test)
            b0 = np.zeros(len(te), dtype=bool)
            b1 = np.zeros(len(te), dtype=bool)
            for band in [0, 1, 2]:
                mca = calib_bands == band
                mte = test_bands == band
                if not np.any(mte):
                    continue
                if np.sum(mca) >= 5 and len(np.unique(y[ca][mca])) >= 2:
                    j0, j1 = split_conformal_sets(p_cal[mca], y[ca][mca], p_test[mte], alpha=alpha)
                else:
                    j0, j1 = inc0[mte], inc1[mte]
                b0[mte] = j0
                b1[mte] = j1
            band_records.append(summarize_sets(y[te], p_test, b0, b1, test_bands))

        out["alpha"][f"{alpha:.2f}"] = {
            "target_coverage": 1 - alpha,
            "plain": aggregate(plain_records),
            "mechanism_band": aggregate(band_records),
        }
    return out


def rounded(obj):
    if isinstance(obj, dict):
        return {k: rounded(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rounded(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


def main() -> None:
    out = {
        "interpretation": (
            "Decision utility uses a transparent policy: singleton conformal sets trigger a binary "
            "decision; two-label and empty sets are routed to review/investigation. Positive cost "
            "reduction means the conformal policy is cheaper than forcing every HGB prediction into "
            "a binary decision under that cost scenario. Mechanism bands use train-only logistic "
            "calibration of SPT log(CSR/CRR) or CPT log(1+LPI), not the full ML score."
        ),
        "settings": {"n_repeats": N_REP, "alpha_values": list(ALPHAS), "cost_scenarios": COST_SCENARIOS},
        "datasets": {},
    }
    for D in [load_spt(), load_cpt()]:
        print(f"evaluating {D['name']}...")
        out["datasets"][D["name"]] = evaluate_dataset(D)

    path = PROC / "conformal_decision_utility.json"
    path.write_text(json.dumps(rounded(out), indent=2), encoding="utf-8")

    for name, res in out["datasets"].items():
        a10 = res["alpha"]["0.10"]["mechanism_band"]
        cost = a10["cost_scenarios"]["fn5_fp1_review0.50"]
        print(
            name,
            "singleton", f"{a10['singleton_rate']:.3f}",
            "two-label", f"{a10['two_label_rate']:.3f}",
            "singleton error", f"{a10['singleton_error_rate']:.3f}",
            "cost reduction", f"{cost['cost_reduction_per_case']:.3f}",
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
