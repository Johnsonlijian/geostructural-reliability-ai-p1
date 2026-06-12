"""Split sensitivity for mechanism-band conformal decision metrics.

Repeats earthquake/event-grouped train/calibration/test splits and reports the
distribution of mechanism-band conformal coverage, efficiency, ambiguity, and
per-event undercoverage. Mechanism bands are defined from train-only logistic
calibration of the published margin coordinate, not from the full ML score.
This is a robustness companion to run_conformal_decision_metrics.py; it does
not touch manuscript or figure files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from geoliq.reliability import split_conformal_sets


BASE = Path(__file__).resolve().parent
PROC = BASE / "data" / "processed"
OUT = PROC / "conformal_split_sensitivity.json"


SPT_FEATURES = [
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
CPT_FEATURES = ["Mw", "PGA", "GWT", "LPI", "crit_FS_capped", "crit_depth", "qc1Ncs_crit", "Ic_crit", "sev_crit"]


def hgb() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(random_state=0)


def load_spt() -> dict[str, Any]:
    df = pd.read_csv(PROC / "cetin2018_baseline_records.csv")
    df = df.dropna(subset=SPT_FEATURES + ["y", "earthquake", "CRR_BI2014", "CSR_cetin"]).copy()
    df = df[
        np.isfinite(df[SPT_FEATURES].to_numpy(float)).all(axis=1)
        & (df["CRR_BI2014"] > 0)
        & (df["CSR_cetin"] > 0)
    ]
    return {
        "name": "SPT_Cetin2018",
        "label": "triggering",
        "group_label": "earthquake",
        "X": df[SPT_FEATURES].to_numpy(float),
        "y": df["y"].astype(int).to_numpy(),
        "groups": df["earthquake"].astype(str).to_numpy(),
        "margin": np.log((df["CSR_cetin"] / df["CRR_BI2014"]).to_numpy(float)),
    }


def load_cpt() -> dict[str, Any]:
    df = pd.read_csv(PROC / "geyin2021_cpt_records.csv")
    return {
        "name": "CPT_Geyin2021",
        "label": "manifestation",
        "group_label": "event",
        "X": df[CPT_FEATURES].to_numpy(float),
        "y": df["y"].astype(int).to_numpy(),
        "groups": df["event"].astype(str).to_numpy(),
        "margin": np.log1p(np.clip(df["LPI"].to_numpy(float), 0, None)),
    }


def band_of_margin_probability(prob: np.ndarray) -> np.ndarray:
    return np.clip((np.asarray(prob, float) * 3).astype(int), 0, 2)


def fit_margin_probability(margin, y, train_idx, eval_idx) -> np.ndarray:
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    model.fit(np.asarray(margin[train_idx], float).reshape(-1, 1), y[train_idx])
    return model.predict_proba(np.asarray(margin[eval_idx], float).reshape(-1, 1))[:, 1]


def finite(values: list[float | int | None]) -> np.ndarray:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    return arr


def dist(values: list[float | int | None]) -> dict[str, Any]:
    arr = finite(values)
    if len(arr) == 0:
        return {"n": 0, "median": None, "iqr": [None, None], "interval_95": [None, None]}
    return {
        "n": int(len(arr)),
        "median": float(np.median(arr)),
        "iqr": [float(np.percentile(arr, 25)), float(np.percentile(arr, 75))],
        "interval_95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def prediction_summary(covered: np.ndarray, sizes: np.ndarray) -> dict[str, Any]:
    covered = np.asarray(covered, dtype=bool)
    sizes = np.asarray(sizes, dtype=int)
    if len(sizes) == 0:
        return {
            "coverage": None,
            "mean_set_size": None,
            "singleton_rate": None,
            "two_label_rate": None,
            "empty_rate": None,
            "n_predictions": 0,
        }
    return {
        "coverage": float(np.mean(covered)),
        "mean_set_size": float(np.mean(sizes)),
        "singleton_rate": float(np.mean(sizes == 1)),
        "two_label_rate": float(np.mean(sizes == 2)),
        "empty_rate": float(np.mean(sizes == 0)),
        "n_predictions": int(len(sizes)),
    }


def grouped_indices(groups: np.ndarray, train_groups: np.ndarray, calib_groups: np.ndarray, test_groups: np.ndarray):
    tr = np.where(np.isin(groups, train_groups))[0]
    ca = np.where(np.isin(groups, calib_groups))[0]
    te = np.where(np.isin(groups, test_groups))[0]
    return tr, ca, te


def split_signature(train_groups: np.ndarray, calib_groups: np.ndarray, test_groups: np.ndarray) -> tuple[tuple[str, ...], ...]:
    return (
        tuple(sorted(map(str, train_groups))),
        tuple(sorted(map(str, calib_groups))),
        tuple(sorted(map(str, test_groups))),
    )


def evaluate_split(
    D: dict[str, Any],
    train_groups: np.ndarray,
    calib_groups: np.ndarray,
    test_groups: np.ndarray,
    alpha: float,
    min_band_calib: int,
    min_event_cases: int,
) -> dict[str, Any] | None:
    X = np.asarray(D["X"], float)
    y = np.asarray(D["y"], int)
    groups = np.asarray(D["groups"])
    margin = np.asarray(D["margin"], float)
    tr, ca, te = grouped_indices(groups, train_groups, calib_groups, test_groups)
    if len(np.unique(y[tr])) < 2 or len(ca) < 8 or len(te) < 8:
        return None

    model = hgb().fit(X[tr], y[tr])
    p_cal = model.predict_proba(X[ca])[:, 1]
    p_test = model.predict_proba(X[te])[:, 1]
    pm_cal = fit_margin_probability(margin, y, tr, ca)
    pm_test = fit_margin_probability(margin, y, tr, te)

    plain0, plain1 = split_conformal_sets(p_cal, y[ca], p_test, alpha=alpha)
    plain_sizes = plain0.astype(int) + plain1.astype(int)
    plain_covered = np.where(y[te] == 1, plain1, plain0)

    cal_bands = band_of_margin_probability(pm_cal)
    test_bands = band_of_margin_probability(pm_test)
    band0 = np.zeros(len(te), dtype=bool)
    band1 = np.zeros(len(te), dtype=bool)
    fallback_predictions = 0
    band_calib_counts: dict[str, int] = {}
    band_test_counts: dict[str, int] = {}

    for bb in [0, 1, 2]:
        mca = cal_bands == bb
        mte = test_bands == bb
        band_calib_counts[str(bb)] = int(np.sum(mca))
        band_test_counts[str(bb)] = int(np.sum(mte))
        if not np.any(mte):
            continue
        if np.sum(mca) >= min_band_calib:
            inc0, inc1 = split_conformal_sets(p_cal[mca], y[ca][mca], p_test[mte], alpha=alpha)
        else:
            inc0, inc1 = plain0[mte], plain1[mte]
            fallback_predictions += int(np.sum(mte))
        band0[mte] = inc0
        band1[mte] = inc1

    band_sizes = band0.astype(int) + band1.astype(int)
    band_covered = np.where(y[te] == 1, band1, band0)

    by_band: dict[str, Any] = {}
    for bb in [0, 1, 2]:
        mte = test_bands == bb
        by_band[str(bb)] = prediction_summary(band_covered[mte], band_sizes[mte])

    event_rows = []
    for gg in np.unique(groups[te]):
        idx = groups[te] == gg
        if np.sum(idx) < min_event_cases:
            continue
        event_coverage = float(np.mean(band_covered[idx]))
        event_rows.append(
            {
                "event": str(gg),
                "n": int(np.sum(idx)),
                "coverage": event_coverage,
                "undercovered": bool(event_coverage < 1.0 - alpha),
                "undercoverage_gap": float(max(0.0, (1.0 - alpha) - event_coverage)),
                "mean_set_size": float(np.mean(band_sizes[idx])),
                "singleton_rate": float(np.mean(band_sizes[idx] == 1)),
                "two_label_rate": float(np.mean(band_sizes[idx] == 2)),
            }
        )

    event_coverages = [row["coverage"] for row in event_rows]
    event_gaps = [row["undercoverage_gap"] for row in event_rows]
    event_under = [row["undercovered"] for row in event_rows]
    critical = by_band["1"]

    return {
        "split_sizes": {
            "train_cases": int(len(tr)),
            "calibration_cases": int(len(ca)),
            "test_cases": int(len(te)),
            "train_groups": int(len(train_groups)),
            "calibration_groups": int(len(calib_groups)),
            "test_groups": int(len(test_groups)),
        },
        "band_calibration_counts": band_calib_counts,
        "band_test_counts": band_test_counts,
        "fallback_predictions": int(fallback_predictions),
        "plain": prediction_summary(plain_covered, plain_sizes),
        "mechanism_band": prediction_summary(band_covered, band_sizes),
        "mechanism_band_by_band": by_band,
        "critical_band_two_label_rate": critical["two_label_rate"],
        "per_event": {
            "n_event_evals": int(len(event_rows)),
            "undercoverage_rate": float(np.mean(event_under)) if event_rows else None,
            "mean_undercoverage_gap": float(np.mean(event_gaps)) if event_rows else None,
            "max_undercoverage_gap": float(np.max(event_gaps)) if event_rows else None,
            "min_event_coverage": float(np.min(event_coverages)) if event_rows else None,
            "p10_event_coverage": float(np.percentile(event_coverages, 10)) if event_rows else None,
            "events": event_rows,
        },
    }


def collect_metric(repetitions: list[dict[str, Any]], path: tuple[str, ...]) -> list[float | int | None]:
    values = []
    for rep in repetitions:
        cur: Any = rep
        for key in path:
            cur = cur.get(key) if isinstance(cur, dict) else None
            if cur is None:
                break
        values.append(cur)
    return values


def summarize_repetitions(repetitions: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "mechanism_band_coverage": dist(collect_metric(repetitions, ("mechanism_band", "coverage"))),
        "mechanism_band_mean_set_size": dist(collect_metric(repetitions, ("mechanism_band", "mean_set_size"))),
        "mechanism_band_singleton_rate": dist(collect_metric(repetitions, ("mechanism_band", "singleton_rate"))),
        "mechanism_band_two_label_rate": dist(collect_metric(repetitions, ("mechanism_band", "two_label_rate"))),
        "mechanism_band_empty_rate": dist(collect_metric(repetitions, ("mechanism_band", "empty_rate"))),
        "critical_band_two_label_rate": dist(collect_metric(repetitions, ("critical_band_two_label_rate",))),
        "per_event_undercoverage_rate": dist(collect_metric(repetitions, ("per_event", "undercoverage_rate"))),
        "per_event_mean_undercoverage_gap": dist(collect_metric(repetitions, ("per_event", "mean_undercoverage_gap"))),
        "per_event_max_undercoverage_gap": dist(collect_metric(repetitions, ("per_event", "max_undercoverage_gap"))),
        "min_event_coverage": dist(collect_metric(repetitions, ("per_event", "min_event_coverage"))),
        "p10_event_coverage": dist(collect_metric(repetitions, ("per_event", "p10_event_coverage"))),
    }
    summary["plain_reference"] = {
        "coverage": dist(collect_metric(repetitions, ("plain", "coverage"))),
        "mean_set_size": dist(collect_metric(repetitions, ("plain", "mean_set_size"))),
    }
    summary["mechanism_band_by_band"] = {}
    for bb in ["0", "1", "2"]:
        summary["mechanism_band_by_band"][bb] = {
            "coverage": dist(collect_metric(repetitions, ("mechanism_band_by_band", bb, "coverage"))),
            "mean_set_size": dist(collect_metric(repetitions, ("mechanism_band_by_band", bb, "mean_set_size"))),
            "singleton_rate": dist(collect_metric(repetitions, ("mechanism_band_by_band", bb, "singleton_rate"))),
            "two_label_rate": dist(collect_metric(repetitions, ("mechanism_band_by_band", bb, "two_label_rate"))),
        }

    event_rows = [event for rep in repetitions for event in rep["per_event"]["events"]]
    event_under = [float(event["undercovered"]) for event in event_rows]
    summary["pooled_per_event"] = {
        "n_event_evals": int(len(event_rows)),
        "coverage": dist([event["coverage"] for event in event_rows]),
        "undercoverage_gap": dist([event["undercoverage_gap"] for event in event_rows]),
        "undercovered_rate": float(np.mean(event_under)) if event_under else None,
    }
    return summary


def evaluate_dataset(
    D: dict[str, Any],
    n_rep: int,
    alpha: float,
    seed: int,
    min_band_calib: int,
    min_event_cases: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    groups = np.asarray(D["groups"])
    gids = np.unique(groups)
    repetitions: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, ...], ...]] = set()
    attempts = 0
    max_attempts = n_rep * 50

    while len(repetitions) < n_rep and attempts < max_attempts:
        attempts += 1
        shuffled = rng.permutation(gids)
        ng = len(shuffled)
        train_groups = shuffled[: int(0.4 * ng)]
        calib_groups = shuffled[int(0.4 * ng) : int(0.7 * ng)]
        test_groups = shuffled[int(0.7 * ng) :]
        signature = split_signature(train_groups, calib_groups, test_groups)
        if signature in seen:
            continue
        seen.add(signature)
        rep = evaluate_split(D, train_groups, calib_groups, test_groups, alpha, min_band_calib, min_event_cases)
        if rep is None:
            continue
        rep["rep"] = int(len(repetitions) + 1)
        repetitions.append(rep)

    return {
        "n": int(len(D["y"])),
        "n_groups": int(len(gids)),
        "label": D["label"],
        "group_label": D["group_label"],
        "n_requested_splits": int(n_rep),
        "n_valid_splits": int(len(repetitions)),
        "n_attempted_splits": int(attempts),
        "summary": summarize_repetitions(repetitions),
        "repetitions": repetitions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rep", type=int, default=100, help="Number of valid grouped splits per dataset.")
    parser.add_argument("--alpha", type=float, default=0.10, help="Split-conformal alpha.")
    parser.add_argument("--seed", type=int, default=20260605, help="Random seed for split generation.")
    parser.add_argument("--min-band-calib", type=int, default=5, help="Minimum calibration cases for a band-specific quantile.")
    parser.add_argument("--min-event-cases", type=int, default=3, help="Minimum held-out cases for per-event coverage.")
    parser.add_argument("--output", type=Path, default=OUT, help="Output JSON path.")
    args = parser.parse_args()

    out = {
        "interpretation": (
            "Each repetition is a valid earthquake/event-grouped train/calibration/test split. "
            "Distributions summarize split-level mechanism-band conformal decision metrics by "
            "median, IQR, and empirical 95% interval. Band 1 is the critical middle "
            "probability band [1/3, 2/3). Per-event undercoverage is descriptive; split "
            "conformal guarantees marginal, not event-conditional, coverage."
        ),
        "settings": {
            "alpha": float(args.alpha),
            "target_coverage": float(1.0 - args.alpha),
            "n_rep": int(args.n_rep),
            "seed": int(args.seed),
            "split_group_fractions": {"train": 0.4, "calibration": 0.3, "test": 0.3},
            "mechanism_band_rule": "train-only margin probability band = clip(int(p_margin * 3), 0, 2)",
            "margin_coordinates": {
                "SPT_Cetin2018": "log(CSR/CRR_BI2014)",
                "CPT_Geyin2021": "log(1+LPI)",
            },
            "critical_band": {"band": 1, "probability_range": [1.0 / 3.0, 2.0 / 3.0]},
            "min_band_calibration_cases": int(args.min_band_calib),
            "min_event_cases": int(args.min_event_cases),
        },
        "datasets": {},
    }

    for offset, D in enumerate([load_spt(), load_cpt()]):
        out["datasets"][D["name"]] = evaluate_dataset(
            D,
            n_rep=args.n_rep,
            alpha=args.alpha,
            seed=args.seed + 1009 * offset,
            min_band_calib=args.min_band_calib,
            min_event_cases=args.min_event_cases,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    for name, result in out["datasets"].items():
        s = result["summary"]
        print("=" * 72)
        print(f"{name}: valid_splits={result['n_valid_splits']} attempted={result['n_attempted_splits']}")
        print(
            " mechanism-band coverage median/IQR/95% = "
            f"{s['mechanism_band_coverage']['median']:.3f} / "
            f"{s['mechanism_band_coverage']['iqr']} / "
            f"{s['mechanism_band_coverage']['interval_95']}"
        )
        print(
            " mean set size median/IQR/95% = "
            f"{s['mechanism_band_mean_set_size']['median']:.3f} / "
            f"{s['mechanism_band_mean_set_size']['iqr']} / "
            f"{s['mechanism_band_mean_set_size']['interval_95']}"
        )
        print(
            " singleton/two-label median = "
            f"{s['mechanism_band_singleton_rate']['median']:.3f} / "
            f"{s['mechanism_band_two_label_rate']['median']:.3f}; "
            f"critical-band two-label median={s['critical_band_two_label_rate']['median']:.3f}"
        )
        print(
            " per-event undercoverage-rate median/IQR/95% = "
            f"{s['per_event_undercoverage_rate']['median']:.3f} / "
            f"{s['per_event_undercoverage_rate']['iqr']} / "
            f"{s['per_event_undercoverage_rate']['interval_95']}"
        )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
