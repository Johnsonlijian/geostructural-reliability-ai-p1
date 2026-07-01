"""Computers and Geotechnics targeted validation stress tests.

This script adds reviewer-facing checks that are stricter than the headline
random-vs-earthquake validation comparison:

* source-study and site grouped validation for the SPT data;
* group-aware nested tuning for strong tabular learners;
* monotonic gradient boosting as a mechanics-constrained ML baseline;
* a physics-margin fusion screen.

The output is intentionally compact so manuscript and SI tables can quote it
without over-claiming mathematical sufficiency.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, ParameterGrid, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE = Path(__file__).resolve().parent
PROC = BASE / "data" / "processed"
RAW = BASE / "data" / "raw"
sys.path.insert(0, str(BASE))

from geoliq.mechanics import crr_boulanger_idriss as BI  # noqa: E402


PSF_TO_KPA = 0.0478802589

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
SPT_MONOTONIC = [1, 1, 1, 0, 1, 0, -1, -1, 0, -1]

CPT_FEATURES = [
    "Mw",
    "PGA",
    "GWT",
    "LPI",
    "crit_FS_capped",
    "crit_depth",
    "qc1Ncs_crit",
    "Ic_crit",
    "sev_crit",
]
CPT_MONOTONIC = [1, 1, -1, 1, -1, 0, -1, 0, 1]

HGB_GRID = list(
    ParameterGrid(
        {
            "learning_rate": [0.05],
            "max_iter": [80, 160],
            "max_leaf_nodes": [7, 15],
            "l2_regularization": [0.0],
            "min_samples_leaf": [8],
        }
    )
)

N_BOOT = 400


def _col(raw: pd.DataFrame, i: int) -> pd.Series:
    return raw.iloc[:, i]


def load_spt() -> dict:
    xls = RAW / "cetin2018_spt" / "mmc2.xls"
    raw = pd.read_excel(xls, sheet_name="CETIN_2018", header=0)
    d = pd.DataFrame(
        {
            "case": _col(raw, 0),
            "earthquake": _col(raw, 1).astype(str),
            "site": _col(raw, 2).astype(str),
            "source": _col(raw, 50).astype(str),
            "liq": _col(raw, 3),
            "Mw": pd.to_numeric(_col(raw, 25), errors="coerce"),
            "amax_g": pd.to_numeric(_col(raw, 23), errors="coerce"),
            "sigma_v_kpa": pd.to_numeric(_col(raw, 16), errors="coerce") * PSF_TO_KPA,
            "sigma_eff_kpa": pd.to_numeric(_col(raw, 18), errors="coerce") * PSF_TO_KPA,
            "rd": pd.to_numeric(_col(raw, 21), errors="coerce"),
            "FC": pd.to_numeric(_col(raw, 33), errors="coerce"),
            "N1_60": pd.to_numeric(_col(raw, 48), errors="coerce"),
            "N1_60cs": pd.to_numeric(_col(raw, 56), errors="coerce"),
            "crit_depth_m": pd.to_numeric(_col(raw, 8), errors="coerce"),
            "water_depth_ft": pd.to_numeric(_col(raw, 10), errors="coerce"),
            "CSR": pd.to_numeric(_col(raw, 27), errors="coerce"),
        }
    )
    lab = d["liq"].astype(str).str.strip().str.lower()
    d["y"] = np.where(lab.isin(["yes"]), 1.0, np.where(lab.isin(["no"]), 0.0, np.nan))
    d["CRR"] = BI.crr_insitu(
        d["Mw"].to_numpy(float),
        d["sigma_eff_kpa"].to_numpy(float),
        d["N1_60cs"].to_numpy(float),
        mode="spt",
    )
    d = d.dropna(subset=SPT_FEATURES + ["y", "CRR", "CSR", "earthquake", "site", "source"]).copy()
    d = d[
        np.isfinite(d[SPT_FEATURES].to_numpy(float)).all(axis=1)
        & (d["CRR"] > 0)
        & (d["CSR"] > 0)
    ].copy()
    d["margin"] = np.log(d["CSR"].to_numpy(float) / d["CRR"].to_numpy(float))
    return {
        "name": "SPT_Cetin2018",
        "frame": d,
        "features": SPT_FEATURES,
        "monotonic": SPT_MONOTONIC,
        "score_name": "log(CSR/CRR_BI2014)",
        "published_score": d["margin"].to_numpy(float),
        "protocol_groups": {
            "earthquake_grouped": d["earthquake"].to_numpy(str),
            "source_grouped": d["source"].to_numpy(str),
            "site_grouped": d["site"].to_numpy(str),
            "source_earthquake_grouped": (
                d["source"].astype(str) + " | " + d["earthquake"].astype(str)
            ).to_numpy(str),
        },
        "counts": {
            "n": int(len(d)),
            "n_earthquakes": int(d["earthquake"].nunique()),
            "n_sources": int(d["source"].nunique()),
            "n_sites": int(d["site"].nunique()),
        },
    }


def load_cpt() -> dict:
    df = pd.read_csv(PROC / "geyin2021_cpt_records.csv")
    d = df.dropna(subset=CPT_FEATURES + ["y", "event", "LPI"]).copy()
    d = d[np.isfinite(d[CPT_FEATURES].to_numpy(float)).all(axis=1)].copy()
    d["margin"] = np.log1p(np.clip(d["LPI"].to_numpy(float), 0, None))
    return {
        "name": "CPT_Geyin2021",
        "frame": d,
        "features": CPT_FEATURES,
        "monotonic": CPT_MONOTONIC,
        "score_name": "log(1+LPI)",
        "published_score": d["margin"].to_numpy(float),
        "protocol_groups": {"event_grouped": d["event"].to_numpy(str)},
        "counts": {
            "n": int(len(d)),
            "n_events": int(d["event"].nunique()),
            "source_or_site_groups": "not available in processed Geyin-Maurer 2021 table",
        },
    }


def has_two_classes(y: np.ndarray, idx: np.ndarray) -> bool:
    return len(np.unique(y[idx])) == 2


def valid_group_splits(y: np.ndarray, groups: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]] | None, int | None]:
    for n_splits in [5, 4, 3]:
        if len(np.unique(groups)) < n_splits:
            continue
        splits = list(GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups))
        if all(has_two_classes(y, tr) and has_two_classes(y, te) for tr, te in splits):
            return splits, n_splits
    return None, None


def valid_random_splits(y: np.ndarray, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def clip_prob(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y, dtype=int)
    p = clip_prob(p)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not np.any(mask):
            continue
        total += float(np.mean(mask) * abs(np.mean(y[mask]) - np.mean(p[mask])))
    return total


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = clip_prob(p)
    return {
        "auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece_10bin": float(ece(y, p)),
    }


def fold_diagnostics(
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    groups: np.ndarray | None,
) -> list[dict]:
    rows = []
    for fold, (tr, te) in enumerate(splits, start=1):
        row = {
            "fold": fold,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "train_pos": int(np.sum(y[tr] == 1)),
            "train_neg": int(np.sum(y[tr] == 0)),
            "test_pos": int(np.sum(y[te] == 1)),
            "test_neg": int(np.sum(y[te] == 0)),
        }
        if groups is not None:
            row["test_group_count"] = int(len(np.unique(groups[te])))
            row["test_groups"] = sorted(map(str, np.unique(groups[te])))[:12]
        rows.append(row)
    return rows


def _metric_gain(y: np.ndarray, p_margin: np.ndarray, p_model: np.ndarray, metric: str) -> float:
    p_margin = clip_prob(p_margin)
    p_model = clip_prob(p_model)
    if metric == "auc":
        return float(roc_auc_score(y, p_model) - roc_auc_score(y, p_margin))
    if metric == "log_loss":
        return float(log_loss(y, p_margin, labels=[0, 1]) - log_loss(y, p_model, labels=[0, 1]))
    if metric == "brier":
        return float(brier_score_loss(y, p_margin) - brier_score_loss(y, p_model))
    if metric == "ece_10bin":
        return float(ece(y, p_margin) - ece(y, p_model))
    raise KeyError(metric)


def paired_bootstrap_gain_ci(
    y: np.ndarray,
    p_margin: np.ndarray,
    p_model: np.ndarray,
    groups: np.ndarray | None,
    metric: str,
    n_boot: int = N_BOOT,
    seed: int = 7301,
) -> dict:
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    p_margin = clip_prob(p_margin)
    p_model = clip_prob(p_model)
    gains = []

    if groups is None:
        units = np.arange(len(y))
        for _ in range(n_boot):
            idx = rng.choice(units, size=len(units), replace=True)
            if metric == "auc" and len(np.unique(y[idx])) < 2:
                continue
            gains.append(_metric_gain(y[idx], p_margin[idx], p_model[idx], metric))
    else:
        groups = np.asarray(groups).astype(str)
        unit_names = np.unique(groups)
        group_to_idx = {g: np.where(groups == g)[0] for g in unit_names}
        for _ in range(n_boot):
            sampled = rng.choice(unit_names, size=len(unit_names), replace=True)
            idx = np.concatenate([group_to_idx[g] for g in sampled])
            if metric == "auc" and len(np.unique(y[idx])) < 2:
                continue
            gains.append(_metric_gain(y[idx], p_margin[idx], p_model[idx], metric))

    if not gains:
        return {"point": _metric_gain(y, p_margin, p_model, metric), "ci95": [None, None], "n_valid": 0}

    arr = np.asarray(gains, dtype=float)
    return {
        "point": _metric_gain(y, p_margin, p_model, metric),
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "n_valid": int(len(arr)),
    }


def gain_report(
    y: np.ndarray,
    p_margin: np.ndarray,
    p_model: np.ndarray,
    groups: np.ndarray | None,
) -> dict:
    out = {
        metric: paired_bootstrap_gain_ci(y, p_margin, p_model, groups, metric)
        for metric in ["auc", "log_loss", "brier", "ece_10bin"]
    }
    auc_hi = out["auc"]["ci95"][1]
    ll_hi = out["log_loss"]["ci95"][1]
    out["practical_exclusion"] = {
        "exclude_auc_gain_ge_0.02": bool(auc_hi is not None and auc_hi < 0.02),
        "exclude_logloss_gain_ge_0.02": bool(ll_hi is not None and ll_hi < 0.02),
        "exclude_logloss_gain_ge_0.05": bool(ll_hi is not None and ll_hi < 0.05),
        "exclude_logloss_gain_ge_0.10": bool(ll_hi is not None and ll_hi < 0.10),
    }
    return out


def oof_predict(
    estimator_factory: Callable[[], object],
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    out = np.full(len(y), np.nan, dtype=float)
    for tr, te in splits:
        est = estimator_factory()
        est.fit(X[tr], y[tr])
        out[te] = est.predict_proba(X[te])[:, 1]
    if np.any(~np.isfinite(out)):
        raise RuntimeError("OOF prediction did not fill every row")
    return clip_prob(out)


def inner_splits(
    y_train: np.ndarray,
    groups_train: np.ndarray | None,
) -> list[tuple[np.ndarray, np.ndarray]] | None:
    if groups_train is not None and len(np.unique(groups_train)) >= 3:
        for n_splits in [3, 2]:
            if len(np.unique(groups_train)) < n_splits:
                continue
            splits = list(GroupKFold(n_splits=n_splits).split(np.zeros(len(y_train)), y_train, groups_train))
            if all(has_two_classes(y_train, tr) and has_two_classes(y_train, va) for tr, va in splits):
                return splits
    if len(y_train) >= 20:
        return list(StratifiedKFold(n_splits=3, shuffle=True, random_state=17).split(np.zeros(len(y_train)), y_train))
    return None


def tune_hgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray | None,
    monotonic: list[int] | None,
) -> dict:
    splits = inner_splits(y_train, groups_train)
    if splits is None:
        return HGB_GRID[0]

    best_score = -math.inf
    best_params = HGB_GRID[0]
    for params in HGB_GRID:
        fold_scores = []
        for tr, va in splits:
            if not has_two_classes(y_train, tr) or not has_two_classes(y_train, va):
                continue
            model = HistGradientBoostingClassifier(
                random_state=0,
                early_stopping=False,
                monotonic_cst=monotonic,
                **params,
            )
            model.fit(X_train[tr], y_train[tr])
            p = model.predict_proba(X_train[va])[:, 1]
            fold_scores.append(roc_auc_score(y_train[va], p))
        if fold_scores and float(np.mean(fold_scores)) > best_score:
            best_score = float(np.mean(fold_scores))
            best_params = params
    return dict(best_params)


def nested_hgb_predict(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    splits: list[tuple[np.ndarray, np.ndarray]],
    monotonic: list[int] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    out = np.full(len(y), np.nan, dtype=float)
    chosen = []
    for tr, te in splits:
        groups_train = groups[tr] if groups is not None else None
        params = tune_hgb(X[tr], y[tr], groups_train, monotonic)
        model = HistGradientBoostingClassifier(
            random_state=0,
            early_stopping=False,
            monotonic_cst=monotonic,
            **params,
        )
        model.fit(X[tr], y[tr])
        out[te] = model.predict_proba(X[te])[:, 1]
        chosen.append(params)
    return clip_prob(out), chosen


def group_platt_hgb_predict(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int = 29,
) -> np.ndarray:
    out = np.full(len(y), np.nan, dtype=float)
    rng = np.random.default_rng(seed)
    for fold, (tr, te) in enumerate(splits):
        base_idx, calib_idx = split_train_calibration(y, tr, groups, rng, fold)
        base = HistGradientBoostingClassifier(random_state=0, early_stopping=False)
        base.fit(X[base_idx], y[base_idx])
        p_cal = clip_prob(base.predict_proba(X[calib_idx])[:, 1])
        p_test = clip_prob(base.predict_proba(X[te])[:, 1])
        if len(np.unique(y[calib_idx])) < 2:
            out[te] = p_test
            continue
        z_cal = np.log(p_cal / (1 - p_cal)).reshape(-1, 1)
        z_test = np.log(p_test / (1 - p_test)).reshape(-1, 1)
        cal = LogisticRegression(max_iter=5000).fit(z_cal, y[calib_idx])
        out[te] = cal.predict_proba(z_test)[:, 1]
    return clip_prob(out)


def split_train_calibration(
    y: np.ndarray,
    train_idx: np.ndarray,
    groups: np.ndarray | None,
    rng: np.random.Generator,
    fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    if groups is not None:
        group_ids = np.unique(groups[train_idx]).astype(str)
        for _ in range(100):
            shuffled = np.array(group_ids, copy=True)
            rng.shuffle(shuffled)
            n_cal = max(1, int(round(0.25 * len(shuffled))))
            calib_groups = shuffled[:n_cal]
            calib_idx = train_idx[np.isin(groups[train_idx].astype(str), calib_groups)]
            base_idx = train_idx[~np.isin(groups[train_idx].astype(str), calib_groups)]
            if has_two_classes(y, base_idx) and has_two_classes(y, calib_idx):
                return base_idx, calib_idx
    # Deterministic fallback when group calibration cannot be class-balanced.
    local = np.array(train_idx, copy=True)
    local_rng = np.random.default_rng(1000 + fold)
    local_rng.shuffle(local)
    for frac in [0.25, 0.33, 0.2]:
        n_cal = max(4, int(round(frac * len(local))))
        calib_idx = local[:n_cal]
        base_idx = local[n_cal:]
        if has_two_classes(y, base_idx) and has_two_classes(y, calib_idx):
            return base_idx, calib_idx
    return train_idx, train_idx


def summarize_params(params_by_fold: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for params in params_by_fold:
        key = json.dumps(params, sort_keys=True)
        counts[key] = counts.get(key, 0) + 1
    return {"by_fold": params_by_fold, "frequency": counts}


def evaluate_protocol(
    D: dict,
    protocol: str,
    groups: np.ndarray | None,
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_splits: int,
) -> dict:
    d = D["frame"]
    X = d[D["features"]].to_numpy(float)
    y = d["y"].astype(int).to_numpy()
    margin = d["margin"].to_numpy(float).reshape(-1, 1)
    fusion = np.column_stack([X, d["margin"].to_numpy(float)])

    result: dict[str, object] = {
        "n_splits": int(n_splits),
        "n_tested": int(len(y)),
        "group_count": int(len(np.unique(groups))) if groups is not None else None,
        "fold_diagnostics": fold_diagnostics(y, splits, groups),
        "bootstrap": {"paired_unit": "row" if groups is None else "group", "n_bootstrap": N_BOOT},
        "published_margin_score_auc": float(roc_auc_score(y, D["published_score"])),
        "models": {},
    }
    predictions: dict[str, np.ndarray] = {}

    models: dict[str, Callable[[], object]] = {
        "margin_logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "full_logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "full_hgb_default": lambda: HistGradientBoostingClassifier(random_state=0, early_stopping=False),
    }
    model_inputs = {
        "margin_logistic": margin,
        "full_logistic": X,
        "full_hgb_default": X,
    }
    for name, factory in models.items():
        p = oof_predict(factory, model_inputs[name], y, splits)
        predictions[name] = p
        result["models"][name] = metrics(y, p)

    p_cal = group_platt_hgb_predict(X, y, groups, splits)
    predictions["full_hgb_group_platt"] = p_cal
    result["models"]["full_hgb_group_platt"] = metrics(y, p_cal)

    if protocol != "random_kfold_context":
        p_nested, params_nested = nested_hgb_predict(X, y, groups, splits)
        predictions["full_hgb_nested_group_tuned"] = p_nested
        result["models"]["full_hgb_nested_group_tuned"] = metrics(y, p_nested)
        result["models"]["full_hgb_nested_group_tuned"]["selected_params"] = summarize_params(params_nested)

        p_mono, params_mono = nested_hgb_predict(X, y, groups, splits, monotonic=D["monotonic"])
        predictions["full_hgb_monotonic_nested"] = p_mono
        result["models"]["full_hgb_monotonic_nested"] = metrics(y, p_mono)
        result["models"]["full_hgb_monotonic_nested"]["monotonic_cst"] = D["monotonic"]
        result["models"]["full_hgb_monotonic_nested"]["selected_params"] = summarize_params(params_mono)

        p_fusion, params_fusion = nested_hgb_predict(
            fusion,
            y,
            groups,
            splits,
            monotonic=D["monotonic"] + [1],
        )
        predictions["fusion_margin_plus_full_hgb_nested"] = p_fusion
        result["models"]["fusion_margin_plus_full_hgb_nested"] = metrics(y, p_fusion)
        result["models"]["fusion_margin_plus_full_hgb_nested"]["selected_params"] = summarize_params(params_fusion)

    margin_auc = result["models"]["margin_logistic"]["auc"]
    margin_ll = result["models"]["margin_logistic"]["log_loss"]
    comparable = {
        k: v
        for k, v in result["models"].items()
        if k != "margin_logistic" and isinstance(v, dict) and "auc" in v
    }
    best_auc_name = max(comparable, key=lambda k: comparable[k]["auc"])
    best_ll_name = min(comparable, key=lambda k: comparable[k]["log_loss"])
    margin_prediction = predictions["margin_logistic"]
    for name in comparable:
        result["models"][name]["gain_vs_margin_logistic"] = gain_report(
            y,
            margin_prediction,
            predictions[name],
            groups,
        )
    result["comparison_to_margin_logistic"] = {
        "best_nonmargin_auc_model": best_auc_name,
        "best_nonmargin_auc": comparable[best_auc_name]["auc"],
        "best_nonmargin_auc_gain": comparable[best_auc_name]["auc"] - margin_auc,
        "best_nonmargin_auc_gain_ci95": result["models"][best_auc_name]["gain_vs_margin_logistic"]["auc"]["ci95"],
        "best_nonmargin_logloss_model": best_ll_name,
        "best_nonmargin_logloss": comparable[best_ll_name]["log_loss"],
        "best_nonmargin_logloss_gain": margin_ll - comparable[best_ll_name]["log_loss"],
        "best_nonmargin_logloss_gain_ci95": result["models"][best_ll_name]["gain_vs_margin_logistic"]["log_loss"]["ci95"],
        "full_or_fusion_auc_gain_gt_0.02": bool((comparable[best_auc_name]["auc"] - margin_auc) > 0.02),
        "full_or_fusion_logloss_gain_gt_0.02": bool((margin_ll - comparable[best_ll_name]["log_loss"]) > 0.02),
        "exclude_auc_gain_ge_0.02_for_best_auc_model": result["models"][best_auc_name]["gain_vs_margin_logistic"][
            "practical_exclusion"
        ]["exclude_auc_gain_ge_0.02"],
        "exclude_logloss_gain_ge_0.05_for_best_logloss_model": result["models"][best_ll_name][
            "gain_vs_margin_logistic"
        ]["practical_exclusion"]["exclude_logloss_gain_ge_0.05"],
    }
    return result


def evaluate_dataset(D: dict) -> dict:
    d = D["frame"]
    y = d["y"].astype(int).to_numpy()
    out = {
        "counts": D["counts"],
        "score_name": D["score_name"],
        "feature_names": D["features"],
        "protocols": {},
        "skipped_protocols": {},
    }

    random_splits = valid_random_splits(y)
    out["protocols"]["random_kfold_context"] = evaluate_protocol(D, "random_kfold_context", None, random_splits, 5)

    for protocol, groups in D["protocol_groups"].items():
        splits, n_splits = valid_group_splits(y, groups)
        if splits is None or n_splits is None:
            out["skipped_protocols"][protocol] = (
                "No 3- to 5-fold grouped split kept two classes in every train/test fold."
            )
            continue
        out["protocols"][protocol] = evaluate_protocol(D, protocol, groups, splits, n_splits)
    return out


def build_summary(datasets: dict) -> dict:
    summary = {}
    for dname, dres in datasets.items():
        rows = {}
        for pname, pres in dres["protocols"].items():
            cmp = pres["comparison_to_margin_logistic"]
            rows[pname] = {
                "margin_auc": pres["models"]["margin_logistic"]["auc"],
                "published_score_auc": pres["published_margin_score_auc"],
                "best_nonmargin_auc_model": cmp["best_nonmargin_auc_model"],
                "best_nonmargin_auc": cmp["best_nonmargin_auc"],
                "best_nonmargin_auc_gain": cmp["best_nonmargin_auc_gain"],
                "best_nonmargin_auc_gain_ci95": cmp["best_nonmargin_auc_gain_ci95"],
                "best_nonmargin_logloss_model": cmp["best_nonmargin_logloss_model"],
                "best_nonmargin_logloss_gain": cmp["best_nonmargin_logloss_gain"],
                "best_nonmargin_logloss_gain_ci95": cmp["best_nonmargin_logloss_gain_ci95"],
                "auc_gain_gt_0.02": cmp["full_or_fusion_auc_gain_gt_0.02"],
                "logloss_gain_gt_0.02": cmp["full_or_fusion_logloss_gain_gt_0.02"],
                "exclude_auc_gain_ge_0.02_for_best_auc_model": cmp[
                    "exclude_auc_gain_ge_0.02_for_best_auc_model"
                ],
                "exclude_logloss_gain_ge_0.05_for_best_logloss_model": cmp[
                    "exclude_logloss_gain_ge_0.05_for_best_logloss_model"
                ],
            }
        summary[dname] = rows
    return summary


def rounded(obj):
    if isinstance(obj, dict):
        return {k: rounded(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rounded(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


def main() -> None:
    datasets = {}
    for D in [load_spt(), load_cpt()]:
        print(f"evaluating {D['name']}...")
        datasets[D["name"]] = evaluate_dataset(D)

    out = {
        "interpretation": (
            "C&G targeted stress tests. Source/site grouping is available for SPT only; "
            "CPT is event-grouped because source/site identifiers are not present in the "
            "processed public table. Gains are reported against an out-of-fold logistic "
            "calibration of the published mechanistic margin, so positive values indicate "
            "recoverable information beyond the margin under the specified validation protocol."
        ),
        "model_fairness_checks": [
            "random and grouped validation protocols",
            "source-study and site grouped SPT sensitivity",
            "group-aware nested HGB tuning",
            "mechanics-sign monotonic HGB",
            "group-heldout Platt calibration for HGB",
            "physics-margin plus full-feature fusion HGB",
        ],
        "datasets": datasets,
        "summary": build_summary(datasets),
    }
    path = PROC / "cg_targeted_stress_tests.json"
    path.write_text(json.dumps(rounded(out), indent=2), encoding="utf-8")

    for dname, rows in out["summary"].items():
        print(f"\n{dname}")
        print(f"{'protocol':<28}{'margin':>9}{'best ML/fusion':>15}{'gain':>9}{'model':>34}")
        for pname, row in rows.items():
            print(
                f"{pname:<28}"
                f"{row['margin_auc']:>9.3f}"
                f"{row['best_nonmargin_auc']:>15.3f}"
                f"{row['best_nonmargin_auc_gain']:>9.3f}"
                f"{row['best_nonmargin_auc_model']:>34}"
            )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
