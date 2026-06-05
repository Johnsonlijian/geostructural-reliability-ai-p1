"""Repeated random-split sensitivity for the P1 optimism-gap claim.

The existing P1 scripts report one random K-fold AUC and one earthquake/event-
grouped AUC. This audit repeats random stratified K-fold CV and compares the
random AUC distribution with the fixed grouped reference already written by the
main validation pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE = Path(__file__).resolve().parent
PROC = BASE / "data" / "processed"

SPT_FEAT = [
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
    "CSR_cetin",
]
CPT_FEAT = ["Mw", "PGA", "GWT", "LPI", "crit_FS_capped", "crit_depth", "qc1Ncs_crit", "Ic_crit", "sev_crit"]


def load_json(name: str) -> dict:
    return json.loads((PROC / name).read_text(encoding="utf-8"))


def label_counts(y: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(y.astype(int), return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def load_spt() -> tuple[np.ndarray, np.ndarray, dict]:
    df = pd.read_csv(PROC / "cetin2018_baseline_records.csv")
    keep = SPT_FEAT + ["y", "earthquake", "CRR_BI2014"]
    d = df.dropna(subset=keep).copy()
    d = d[
        np.isfinite(d[SPT_FEAT].to_numpy(float)).all(axis=1)
        & (d["CRR_BI2014"] > 0)
        & (d["CSR_cetin"] > 0)
    ]
    y = d["y"].astype(int).to_numpy()
    meta = {
        "records_file": "cetin2018_baseline_records.csv",
        "n_cases": int(len(d)),
        "n_groups": int(d["earthquake"].astype(str).nunique()),
        "group_field": "earthquake",
        "label_counts": label_counts(y),
        "features": SPT_FEAT,
    }
    return d[SPT_FEAT].to_numpy(float), y, meta


def load_cpt() -> tuple[np.ndarray, np.ndarray, dict]:
    df = pd.read_csv(PROC / "geyin2021_cpt_records.csv")
    keep = CPT_FEAT + ["y", "event"]
    d = df.dropna(subset=keep).copy()
    d = d[np.isfinite(d[CPT_FEAT].to_numpy(float)).all(axis=1)]
    y = d["y"].astype(int).to_numpy()
    meta = {
        "records_file": "geyin2021_cpt_records.csv",
        "n_cases": int(len(d)),
        "n_groups": int(d["event"].astype(str).nunique()),
        "group_field": "event",
        "label_counts": label_counts(y),
        "features": CPT_FEAT,
    }
    return d[CPT_FEAT].to_numpy(float), y, meta


def spt_model_factories() -> dict[str, Callable[[], object]]:
    return {
        "hist_gbt": lambda: HistGradientBoostingClassifier(random_state=0),
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    }


def cpt_model_factories() -> dict[str, Callable[[], object]]:
    return {
        "hist_gbt": lambda: HistGradientBoostingClassifier(random_state=0),
        "logistic": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=5000),
        ),
    }


def random_cv_auc(
    estimator_factory: Callable[[], object],
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int,
    seed: int,
) -> float:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    prob = cross_val_predict(estimator_factory(), X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    return float(roc_auc_score(y, prob))


def summarize(values: list[float]) -> dict[str, object]:
    arr = np.asarray(values, dtype=float)
    q05, q25, q50, q75, q95 = np.quantile(arr, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": int(arr.size),
        "mean": round(float(np.mean(arr)), 5),
        "sd": round(float(np.std(arr, ddof=1)), 5) if arr.size > 1 else 0.0,
        "median": round(float(q50), 5),
        "q05": round(float(q05), 5),
        "q25": round(float(q25), 5),
        "q75": round(float(q75), 5),
        "q95": round(float(q95), 5),
        "min": round(float(np.min(arr)), 5),
        "max": round(float(np.max(arr)), 5),
    }


def rounded(values: list[float]) -> list[float]:
    return [round(float(v), 5) for v in values]


def run_dataset(
    *,
    X: np.ndarray,
    y: np.ndarray,
    meta: dict,
    model_factories: dict[str, Callable[[], object]],
    grouped_reference: dict[str, float],
    physics_reference: dict,
    repeats: int,
    n_splits: int,
    seed_start: int,
) -> dict:
    seeds = list(range(seed_start, seed_start + repeats))
    models: dict[str, dict] = {}
    for model_name, factory in model_factories.items():
        aucs = [random_cv_auc(factory, X, y, n_splits=n_splits, seed=seed) for seed in seeds]
        grouped_auc = float(grouped_reference[model_name])
        gaps = [auc - grouped_auc for auc in aucs]
        models[model_name] = {
            "grouped_auc_reference": round(grouped_auc, 5),
            "random_auc": summarize(aucs),
            "optimism_gap_random_minus_grouped": summarize(gaps),
        }

        gap_summary = models[model_name]["optimism_gap_random_minus_grouped"]
        print(
            f"{model_name:<14} random median={models[model_name]['random_auc']['median']:.3f} "
            f"grouped={grouped_auc:.3f} gap median={gap_summary['median']:.3f}"
        )

    return {
        **meta,
        "n_splits": int(n_splits),
        "n_repeats": int(repeats),
        "random_cv": "Repeated StratifiedKFold(shuffle=True)",
        "repeat_seeds": seeds,
        "physics_baseline_reference": physics_reference,
        "models": models,
    }


def build_output(repeats: int, n_splits: int, seed_start: int) -> dict:
    spt_grouped = load_json("cetin2018_grouped_validation.json")
    cpt_grouped = load_json("geyin2021_cpt_results.json")
    likelihood = load_json("sufficiency_likelihood.json")
    spt_X, spt_y, spt_meta = load_spt()
    cpt_X, cpt_y, cpt_meta = load_cpt()

    spt_grouped_auc = {
        name: float(likelihood["SPT/Cetin2018"][name]["auc"]) for name in ["hist_gbt", "logistic"]
    }
    cpt_grouped_auc = {name: float(row["event_grouped"]) for name, row in cpt_grouped["ML_optimism_gap"].items()}

    print("=" * 78)
    print("Repeated random split sensitivity: random K-fold optimism vs grouped reference")
    print("=" * 78)
    print(f"repeats={repeats} folds={n_splits} seeds={seed_start}..{seed_start + repeats - 1}")
    print("\nSPT_Cetin2018")
    spt = run_dataset(
        X=spt_X,
        y=spt_y,
        meta=spt_meta,
        model_factories=spt_model_factories(),
        grouped_reference=spt_grouped_auc,
        physics_reference={
            "name": "BI2014_physics_baseline_auc",
            "auc": round(float(spt_grouped["BI2014_physics_baseline_auc"]), 5),
            "source": "cetin2018_grouped_validation.json; grouped ML reference from sufficiency_likelihood.json",
        },
        repeats=repeats,
        n_splits=n_splits,
        seed_start=seed_start,
    )

    print("\nCPT_Geyin2021")
    cpt = run_dataset(
        X=cpt_X,
        y=cpt_y,
        meta=cpt_meta,
        model_factories=cpt_model_factories(),
        grouped_reference=cpt_grouped_auc,
        physics_reference={
            "name": "BI2014_CPT_LPI_zero_shot.roc_auc_LPI",
            "auc": round(float(cpt_grouped["BI2014_CPT_LPI_zero_shot"]["roc_auc_LPI"]), 5),
            "source": "geyin2021_cpt_results.json",
        },
        repeats=repeats,
        n_splits=n_splits,
        seed_start=seed_start,
    )

    return {
        "interpretation": (
            "Positive optimism_gap_random_minus_grouped means repeated random K-fold AUC exceeds "
            "the fixed earthquake/event-grouped AUC reference. Grouped ML and physics values are "
            "read from existing processed JSON files and are not recomputed by this script."
        ),
        "random_split_protocol": {
            "n_splits": int(n_splits),
            "n_repeats": int(repeats),
            "seed_start": int(seed_start),
            "seed_stop_inclusive": int(seed_start + repeats - 1),
            "auc_scoring": "OOF predict_proba[:, 1] -> roc_auc_score",
        },
        "datasets": {
            "SPT_Cetin2018": spt,
            "CPT_Geyin2021": cpt,
        },
        "headline_models": {
            "SPT_Cetin2018": "hist_gbt",
            "CPT_Geyin2021": "hist_gbt",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=100, help="Number of random K-fold repeats.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of stratified folds per repeat.")
    parser.add_argument("--seed-start", type=int, default=0, help="First integer seed for repeated splits.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROC / "random_split_sensitivity.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = build_output(args.repeats, args.n_splits, args.seed_start)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote: {args.output}")


if __name__ == "__main__":
    main()
