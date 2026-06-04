"""Practical-equivalence audit for the mechanistic compression claim.

This script does not prove mathematical sufficiency. It asks a reviewer-facing
question: do the existing grouped comparisons rule out practically meaningful
full-feature gains over the margin?
"""

from __future__ import annotations

import json
from pathlib import Path


BASE = Path(__file__).resolve().parent
PROC = BASE / "data" / "processed"


def load(name: str) -> dict:
    return json.loads((PROC / name).read_text(encoding="utf-8"))


def auc_equivalence(entry: dict, threshold: float) -> dict:
    """Return whether a full-feature AUC gain above threshold is excluded."""
    ci = entry["CI95"]
    # delta = margin - full. Full improves by threshold when delta <= -threshold.
    return {
        "delta_margin_minus_full": entry["delta_margin_minus_full"],
        "CI95": ci,
        "full_auc_gain_gt_threshold_excluded": bool(ci[0] > -threshold),
    }


def logloss_equivalence(model_entry: dict, threshold: float) -> dict:
    """Return whether a full-feature log-loss reduction above threshold is excluded."""
    ci = model_entry["CI95"]
    # dlogloss = full - margin. Full improves by threshold when dlogloss <= -threshold.
    return {
        "dlogloss_full_minus_margin": model_entry["dlogloss_full_minus_margin"],
        "CI95": ci,
        "full_logloss_gain_gt_threshold_excluded": bool(ci[0] > -threshold),
    }


def residual_variant_equivalence(variant_entry: dict, threshold: float) -> dict:
    d = variant_entry["dlogloss_variant_minus_margin"]
    return {
        "dlogloss_variant_minus_margin": d,
        "variant_logloss_gain_gt_threshold_excluded": bool(d["lo"] > -threshold),
    }


def main() -> None:
    innovation = load("innovation_analysis.json")
    suff = load("sufficiency_likelihood.json")
    residual = load("residual_sufficiency_audit.json")
    source = load("cetin2018_source_grouped.json")
    region = load("cross_region_transfer.json")

    auc_thresholds = [0.02, 0.05]
    logloss_thresholds = [0.02, 0.05, 0.10]

    out: dict[str, object] = {
        "interpretation": (
            "AUC delta is margin minus full; log-loss delta is full minus margin. "
            "Equivalence here means excluding a practically meaningful full-feature gain, "
            "not proving mathematical sufficiency."
        ),
        "thresholds": {
            "auc_full_gain": auc_thresholds,
            "logloss_full_gain": logloss_thresholds,
        },
        "datasets": {},
        "blocking_design_notes": {
            "SPT_Cetin2018": (
                "BI2014 has calibration-lineage overlap with the SPT case-history literature; "
                "therefore SPT is a fairness benchmark, not a fully blind validation."
            ),
            "CPT_Geyin2021": (
                "CPT manifestation is the cleaner independent response task for the compression claim."
            ),
        },
    }

    name_map = {
        "SPT/Cetin2018": "SPT_Cetin2018",
        "CPT/Geyin2021": "CPT_Geyin2021",
    }
    for ds_key, clean_key in name_map.items():
        ds_out: dict[str, object] = {"auc_equivalence": {}, "logloss_equivalence": {}, "residual_audit": {}}
        for thr in auc_thresholds:
            ds_out["auc_equivalence"][f"full_auc_gain_gt_{thr:.2f}"] = auc_equivalence(
                innovation[ds_key]["A1_sufficiency"], thr
            )
        for model_name, model_entry in suff[ds_key].items():
            if model_name == "margin_only":
                continue
            ds_out["logloss_equivalence"][model_name] = {
                f"full_logloss_gain_gt_{thr:.2f}": logloss_equivalence(model_entry, thr)
                for thr in logloss_thresholds
            }
        if clean_key in residual:
            for variant, entry in residual[clean_key]["variants"].items():
                ds_out["residual_audit"][variant] = {
                    f"variant_logloss_gain_gt_{thr:.2f}": residual_variant_equivalence(entry, thr)
                    for thr in logloss_thresholds
                }
        out["datasets"][clean_key] = ds_out

    out["source_blocking_proxy"] = {
        "SPT_source_grouped_ML_AUC": source["by_split"]["source(reference)_grouped"],
        "SPT_physics_AUC_split_invariant": source["physics_BI2014_auc_split_invariant"],
        "interpretation": (
            "Source-reference grouped ML remains below the published mechanism, but this is a proxy "
            "and must not be described as a BI2014 calibration-overlap removal."
        ),
    }
    out["leave_one_region_out"] = {
        "pooled": region["pooled"],
        "small_region_caveat": "China (n=10) and Other (n=16) are descriptive only.",
    }

    # Human-readable summary flags.
    summary = {}
    for clean_key, ds in out["datasets"].items():
        auc_pass_002 = ds["auc_equivalence"]["full_auc_gain_gt_0.02"]["full_auc_gain_gt_threshold_excluded"]
        logloss_pass_010 = all(
            m["full_logloss_gain_gt_0.10"]["full_logloss_gain_gt_threshold_excluded"]
            for m in ds["logloss_equivalence"].values()
        )
        residual_pass_010 = all(
            v["variant_logloss_gain_gt_0.10"]["variant_logloss_gain_gt_threshold_excluded"]
            for v in ds["residual_audit"].values()
        )
        summary[clean_key] = {
            "excludes_full_AUC_gain_gt_0.02": auc_pass_002,
            "excludes_model_logloss_gain_gt_0.10": logloss_pass_010,
            "excludes_residual_variant_logloss_gain_gt_0.10": residual_pass_010,
            "safe_sentence": (
                "No tested full-feature learner shows a supported OOD likelihood gain; "
                "AUC equivalence excludes a full-feature gain above 0.02 for the main AUC comparison. "
                "For SPT residual log-loss variants, the data do not exclude all small gains, so the "
                "claim should remain 'no supported improvement' rather than strict equivalence."
                if clean_key == "SPT_Cetin2018"
                else
                "For the independent CPT manifestation task, full-feature models are worse than the "
                "margin in likelihood and the main AUC comparison excludes practical full-feature gains."
            ),
        }
    out["summary"] = summary

    path = PROC / "practical_equivalence_audit.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for ds, row in summary.items():
        print(ds, row)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
