import json
from pathlib import Path


PROC = Path(__file__).resolve().parents[1] / "data" / "processed"


def load(name):
    return json.loads((PROC / name).read_text(encoding="utf-8"))


def test_p1_headline_processed_numbers_are_stable():
    baseline = load("cetin2018_baseline_metrics.json")
    assert baseline["n_used_in_metrics"] == 208
    assert baseline["n_liquefied"] == 113
    assert baseline["n_nonliquefied"] == 95
    assert baseline["csr_validation_vs_cetin"]["pearson_r"] == 1.0
    assert baseline["csr_validation_vs_cetin"]["MAE"] == 0.0
    assert round(baseline["BI2014_zero_shot_baseline"]["roc_auc"], 3) == 0.923

    cpt = load("geyin2021_cpt_results.json")
    assert cpt["n_cases"] == 275
    assert cpt["n_events"] == 21
    assert round(cpt["BI2014_CPT_LPI_zero_shot"]["roc_auc_LPI"], 3) == 0.750

    reliability = load("reliability_upgrade.json")
    assert round(reliability["SPT_Cetin2018"]["paired_physics_minus_ml"]["delta"], 3) == 0.058
    assert round(reliability["CPT_Geyin2021"]["paired_physics_minus_ml"]["delta"], 3) == 0.137
    assert round(reliability["SPT_Cetin2018"]["conformal_conditional_grouped"]["frac_events_undercovered"], 3) == 0.303
    assert round(reliability["CPT_Geyin2021"]["conformal_conditional_grouped"]["frac_events_undercovered"], 3) == 0.387


def test_p1_conservative_audits_exist_and_keep_claim_bounded():
    residual = load("residual_sufficiency_audit.json")
    assert residual["SPT_Cetin2018"]["n"] == 208
    assert residual["CPT_Geyin2021"]["n"] == 275
    for dataset in ["SPT_Cetin2018", "CPT_Geyin2021"]:
        for row in residual[dataset]["variants"].values():
            assert row["variant_improves_OOS_logloss"] is False

    ambiguity = load("ambiguity_floor_sensitivity.json")
    assert "not strict Bayes" in ambiguity["interpretation"]
    assert ambiguity["SPT_Cetin2018"]["bin_count_sensitivity"][2]["n_bins"] == 10
    assert ambiguity["CPT_Geyin2021"]["bin_count_sensitivity"][2]["n_bins"] == 10

    practical = load("practical_equivalence_audit.json")
    summary = practical["summary"]
    assert summary["SPT_Cetin2018"]["excludes_full_AUC_gain_gt_0.02"] is True
    assert summary["SPT_Cetin2018"]["excludes_model_logloss_gain_gt_0.10"] is True
    assert summary["SPT_Cetin2018"]["excludes_residual_variant_logloss_gain_gt_0.10"] is False
    assert summary["CPT_Geyin2021"]["excludes_residual_variant_logloss_gain_gt_0.10"] is True


def test_conformal_decision_metrics_make_uncertainty_operational():
    decision = load("conformal_decision_metrics.json")
    spt = decision["datasets"]["SPT_Cetin2018"]["alpha"]["0.10"]
    cpt = decision["datasets"]["CPT_Geyin2021"]["alpha"]["0.10"]
    assert round(spt["mechanism_band"]["singleton_rate"], 3) == 0.431
    assert spt["mechanism_band_by_band"]["1"]["two_label_rate"] > 0.80
    assert cpt["mechanism_band"]["two_label_rate"] > 0.65
    assert cpt["mechanism_band_by_band"]["1"]["two_label_rate"] > 0.80


def test_groundwater_residual_stratification_stays_diagnostic_not_causal():
    gw = load("groundwater_residual_stratification.json")
    assert "not causal identification" in gw["interpretation"]
    assert gw["SPT_Cetin2018"]["n_bins"] == 3
    assert gw["CPT_Geyin2021"]["n_bins"] == 3
    assert gw["SPT_Cetin2018"]["max_abs_margin_bin_residual"] <= 0.06
    assert gw["CPT_Geyin2021"]["max_abs_margin_bin_residual"] <= 0.09
