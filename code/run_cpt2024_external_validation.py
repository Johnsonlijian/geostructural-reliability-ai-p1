"""Rebuild the Rateria-Geyin-Maurer 2024 CPT validation artifacts.

Raw third-party data are not redistributed. Place DesignSafe PRJ-5746
`GLOBALDATASET.mat` under `data/raw/rateria_geyin_maurer_2024_cpt/` before
running this script.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from geoliq.mechanics import cpt_profile as CP  # noqa: E402

RAW = BASE / "data" / "raw" / "rateria_geyin_maurer_2024_cpt" / "GLOBALDATASET.mat"
PROC = BASE / "data" / "processed"


ARCHIVED_HEADLINES = {
    "inverse": {
        "event_grouped": {
            "margin_auc": 0.762,
            "best_nonmargin_model": "monotonic nested HGB",
            "best_nonmargin_auc_gain": -0.071,
            "best_nonmargin_auc_gain_ci95": [-0.141, -0.009],
            "best_nonmargin_logloss_gain": -0.092,
            "best_nonmargin_logloss_gain_ci95": [-0.130, -0.051],
            "exclude_auc_gain_ge_0.02": True,
            "exclude_logloss_gain_ge_0.05": True,
        },
        "leave_nisqually": {
            "split": "fixed leave-Nisqually-out",
            "n_test": 24,
            "n_test_events": 1,
            "margin_auc": 0.970,
            "best_nonmargin_model": "full logistic / nested HGB sensitivity",
            "best_nonmargin_auc_gain": 0.030,
            "best_nonmargin_auc_gain_ci95": [0.000, 0.130],
            "best_nonmargin_logloss_gain": 0.119,
            "best_nonmargin_logloss_gain_ci95": [0.021, 0.236],
        },
    },
    "measured": {
        "event_grouped": {
            "margin_auc": 0.787,
            "best_nonmargin_model": "full logistic",
            "best_nonmargin_auc_gain": -0.051,
            "best_nonmargin_auc_gain_ci95": [-0.084, -0.009],
            "best_nonmargin_logloss_gain": -0.116,
            "best_nonmargin_logloss_gain_ci95": [-0.163, -0.055],
            "exclude_auc_gain_ge_0.02": True,
            "exclude_logloss_gain_ge_0.05": True,
        },
        "leave_nisqually": {
            "split": "fixed leave-Nisqually-out",
            "n_test": 24,
            "n_test_events": 1,
            "margin_auc": 0.970,
            "best_nonmargin_model": "nested/default HGB sensitivity",
            "best_nonmargin_auc_gain": -0.011,
            "best_nonmargin_auc_gain_ci95": [-0.093, 0.037],
            "best_nonmargin_logloss_gain": 0.106,
            "best_nonmargin_logloss_gain_ci95": [-0.230, 0.352],
        },
    },
}


def scalar(case, name: str) -> float:
    return float(np.asarray(getattr(case, name)).ravel()[0])


def load_cases() -> np.ndarray:
    if not RAW.exists():
        raise FileNotFoundError(
            f"Missing {RAW}. Download DesignSafe DOI 10.17603/ds2-8hvd-hd43 "
            "and place GLOBALDATASET.mat in the raw-data directory."
        )
    data = sio.loadmat(RAW, squeeze_me=True, struct_as_record=False)
    key = "GLOBALDATASET_matchreadme" if "GLOBALDATASET_matchreadme" in data else "GLOBALDATASET"
    return np.atleast_1d(data[key])


def build_records(cases: np.ndarray, profile: str) -> pd.DataFrame:
    qc_name = "qc_inv" if profile == "inverse" else "qc"
    fs_name = "fs_inv" if profile == "inverse" else "fs"
    rows = []
    for case in cases:
        result = CP.process_case(
            np.asarray(getattr(case, "depth"), dtype=float),
            np.asarray(getattr(case, qc_name), dtype=float),
            np.asarray(getattr(case, fs_name), dtype=float),
            scalar(case, "GWT"),
            scalar(case, "Magnitude"),
            scalar(case, "PGA"),
        )
        rows.append(
            {
                "id": getattr(case, "ID", ""),
                "event": getattr(case, "EventName"),
                "Mw": scalar(case, "Magnitude"),
                "PGA": scalar(case, "PGA"),
                "GWT": scalar(case, "GWT"),
                "y": int(scalar(case, "Manifestation")),
                "profile": profile,
                **result,
            }
        )
    df = pd.DataFrame(rows)
    df["crit_FS_capped"] = np.where(np.isfinite(df["crit_FS"]), df["crit_FS"], 5.0)
    df["log_LPI"] = np.log1p(df["LPI"])
    return df


def computed_protocol_check(df: pd.DataFrame) -> dict:
    """Independent check of the current records, not the archival headline table."""
    y = df["y"].to_numpy(int)
    groups = df["event"].astype(str).to_numpy()
    cv = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=6).split(df, y, groups))

    margin = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=5000))
    p_margin = cross_val_predict(margin, df[["log_LPI"]].to_numpy(float), y, cv=cv, method="predict_proba")[:, 1]

    nonmargin_features = ["Mw", "PGA", "GWT", "crit_depth", "qc1Ncs_crit", "Ic_crit", "sev_crit", "n_liq_layers"]
    nonmargin = RandomForestClassifier(n_estimators=300, random_state=0, min_samples_leaf=5)
    p_nonmargin = cross_val_predict(
        nonmargin, df[nonmargin_features].to_numpy(float), y, cv=cv, method="predict_proba"
    )[:, 1]

    return {
        "protocol": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=6)",
        "margin_model": "logistic calibration of log1p(LPI)",
        "nonmargin_check_model": "random forest on demand and critical-layer summary features, excluding LPI",
        "computed_margin_auc": round(float(roc_auc_score(y, p_margin)), 6),
        "computed_nonmargin_auc_gain": round(float(roc_auc_score(y, p_nonmargin) - roc_auc_score(y, p_margin)), 6),
        "computed_nonmargin_logloss_gain": round(
            float(log_loss(y, p_margin, labels=[0, 1]) - log_loss(y, p_nonmargin, labels=[0, 1])), 6
        ),
    }


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    inverse = build_records(cases, "inverse")
    measured = build_records(cases, "measured")

    inverse.to_csv(PROC / "cpt2024_records_inverse.csv", index=False)
    measured.to_csv(PROC / "cpt2024_records_measured.csv", index=False)

    event_counts = inverse["event"].value_counts()
    counts = {
        "n": int(len(inverse)),
        "n_events": int(inverse["event"].nunique()),
        "n_manifestation": int((inverse["y"] == 1).sum()),
        "n_no_manifestation": int((inverse["y"] == 0).sum()),
        "nisqually_cases": int(event_counts.get("Nisqually, Washington", 0)),
    }

    out = {
        "dataset": "Rateria, Geyin, and Maurer (2024) CPT DesignSafe release",
        "doi": "10.17603/ds2-8hvd-hd43",
        "source_project": "DesignSafe PRJ-5746",
        "counts": counts,
        "summary": {
            "CPT_Rateria2024_inverse": {
                "event_grouped": {
                    **ARCHIVED_HEADLINES["inverse"]["event_grouped"],
                    "computed_protocol_check": computed_protocol_check(inverse),
                }
            },
            "CPT_Rateria2024_measured": {
                "event_grouped": {
                    **ARCHIVED_HEADLINES["measured"]["event_grouped"],
                    "computed_protocol_check": computed_protocol_check(measured),
                }
            },
            "leave_nisqually_out_inverse": ARCHIVED_HEADLINES["inverse"]["leave_nisqually"],
            "leave_nisqually_out_measured": ARCHIVED_HEADLINES["measured"]["leave_nisqually"],
        },
        "interpretation": (
            "The 2024 current-release CPT event-grouped results remain margin dominated. "
            "Nisqually is included in the 2024 release and is therefore not treated as independent; "
            "it is retained only as a fixed held-out event caveat."
        ),
        "reconstruction_note": (
            "Per-case processed CSV files are rebuilt from the official GLOBALDATASET.mat. "
            "The headline summary values mirror the R08 manuscript/Table S14 gate values used "
            "for Fig. 2 and the submission text."
        ),
    }
    (PROC / "cpt2024_external_validation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("wrote data/processed/cpt2024_records_inverse.csv")
    print("wrote data/processed/cpt2024_records_measured.csv")
    print("wrote data/processed/cpt2024_external_validation.json")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
