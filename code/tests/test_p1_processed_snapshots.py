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
    assert "not strict theoretical" in ambiguity["interpretation"]
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
    assert "train-only logistic calibration" in decision["interpretation"]
    assert 0.43 <= spt["mechanism_band"]["singleton_rate"] <= 0.48
    assert spt["mechanism_band_by_band"]["1"]["two_label_rate"] > 0.65
    assert cpt["mechanism_band"]["two_label_rate"] > 0.65
    assert cpt["mechanism_band_by_band"]["1"]["two_label_rate"] > 0.75


def test_cg_targeted_stress_tests_distinguish_spt_from_cpt():
    cg = load("cg_targeted_stress_tests.json")
    assert "group-aware nested HGB tuning" in cg["model_fairness_checks"]

    spt = cg["summary"]["SPT_Cetin2018"]
    cpt = cg["summary"]["CPT_Geyin2021"]
    assert "source_grouped" in spt
    assert "site_grouped" in spt
    assert spt["site_grouped"]["auc_gain_gt_0.02"] is False
    assert spt["source_grouped"]["auc_gain_gt_0.02"] is True
    assert cpt["event_grouped"]["best_nonmargin_auc_gain"] < 0
    assert cpt["event_grouped"]["auc_gain_gt_0.02"] is False


def test_conformal_decision_utility_has_positive_decision_value():
    util = load("conformal_decision_utility.json")
    assert "review/investigation" in util["interpretation"]
    for dataset in ["SPT_Cetin2018", "CPT_Geyin2021"]:
        row = util["datasets"][dataset]["alpha"]["0.10"]["mechanism_band"]
        cost = row["cost_scenarios"]["fn5_fp1_review0.50"]
        assert cost["cost_reduction_per_case"] > 0
        assert row["two_label_rate"] > 0.50


def test_repeated_random_split_sensitivity_quantifies_optimism():
    random_split = load("random_split_sensitivity.json")
    assert random_split["random_split_protocol"]["n_repeats"] == 100
    assert random_split["headline_models"]["SPT_Cetin2018"] == "hist_gbt"
    assert random_split["headline_models"]["CPT_Geyin2021"] == "hist_gbt"

    spt = random_split["datasets"]["SPT_Cetin2018"]["models"]["hist_gbt"]
    cpt = random_split["datasets"]["CPT_Geyin2021"]["models"]["hist_gbt"]
    assert 0.046 <= spt["optimism_gap_random_minus_grouped"]["median"] <= 0.047
    assert spt["optimism_gap_random_minus_grouped"]["q05"] > 0.02
    assert round(cpt["optimism_gap_random_minus_grouped"]["median"], 3) == 0.117
    assert cpt["optimism_gap_random_minus_grouped"]["q05"] > 0.09


def test_conformal_split_sensitivity_keeps_decision_pattern_bounded():
    split = load("conformal_split_sensitivity.json")
    assert split["settings"]["alpha"] == 0.1

    spt = split["datasets"]["SPT_Cetin2018"]
    cpt = split["datasets"]["CPT_Geyin2021"]
    assert spt["n_valid_splits"] == 100
    assert cpt["n_valid_splits"] == 100
    assert "train-only margin probability" in split["settings"]["mechanism_band_rule"]
    assert 0.90 <= spt["summary"]["mechanism_band_coverage"]["median"] <= 0.94
    assert 0.89 <= cpt["summary"]["mechanism_band_coverage"]["median"] <= 0.94
    assert spt["summary"]["critical_band_two_label_rate"]["median"] > 0.75
    assert cpt["summary"]["critical_band_two_label_rate"]["median"] > 0.85


def test_groundwater_residual_stratification_stays_diagnostic_not_causal():
    gw = load("groundwater_residual_stratification.json")
    assert "not causal identification" in gw["interpretation"]
    assert gw["SPT_Cetin2018"]["n_bins"] == 3
    assert gw["CPT_Geyin2021"]["n_bins"] == 3
    assert gw["SPT_Cetin2018"]["max_abs_margin_bin_residual"] <= 0.06
    assert gw["CPT_Geyin2021"]["max_abs_margin_bin_residual"] <= 0.09


def test_cpt2024_external_validation_keeps_cg_claim_bounded():
    cpt2024 = load("cpt2024_external_validation.json")
    assert cpt2024["counts"]["n"] == 332
    assert cpt2024["counts"]["n_events"] == 25
    assert cpt2024["counts"]["n_manifestation"] == 181
    assert cpt2024["counts"]["n_no_manifestation"] == 151
    assert cpt2024["counts"]["nisqually_cases"] == 24

    inverse = cpt2024["summary"]["CPT_Rateria2024_inverse"]["event_grouped"]
    measured = cpt2024["summary"]["CPT_Rateria2024_measured"]["event_grouped"]
    assert round(inverse["margin_auc"], 3) == 0.762
    assert inverse["best_nonmargin_auc_gain"] < 0
    assert inverse["best_nonmargin_auc_gain_ci95"][1] < 0
    assert inverse["exclude_auc_gain_ge_0.02"] is True
    assert inverse["exclude_logloss_gain_ge_0.05"] is True
    assert round(measured["margin_auc"], 3) == 0.787
    assert measured["best_nonmargin_auc_gain"] < 0
    assert measured["best_nonmargin_auc_gain_ci95"][1] < 0
    assert measured["exclude_auc_gain_ge_0.02"] is True
    assert measured["exclude_logloss_gain_ge_0.05"] is True

    nisqually_inverse = cpt2024["summary"]["leave_nisqually_out_inverse"]
    nisqually_measured = cpt2024["summary"]["leave_nisqually_out_measured"]
    assert round(nisqually_inverse["best_nonmargin_auc_gain"], 3) == 0.030
    assert nisqually_inverse["best_nonmargin_logloss_gain"] > 0
    assert nisqually_measured["best_nonmargin_auc_gain"] < 0
    assert nisqually_measured["best_nonmargin_logloss_gain_ci95"][0] < 0
    assert "not treated as independent" in cpt2024["interpretation"]
    assert "fixed held-out event" in cpt2024["interpretation"]
