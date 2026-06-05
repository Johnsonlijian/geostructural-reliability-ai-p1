"""Tests the positive scientific thesis before manuscript claims are finalized.
 A1 grouped bound: does full-feature ML beat the single mechanistic margin out-of-distribution?
 A2 screening   : inside the ambiguous band (FS~1), is the label still learnable from features?
 A3 ambiguity   : empirical coordinate ambiguity proxy; is it localized at FS~1?
 A4 Mondrian    : does mechanism-conditioned conformal restore band-conditional coverage?
Real data, both DBs. Honest: prints what is actually found.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from geoliq import reliability as R
from geoliq.reliability import split_conformal_sets

PROC = os.path.join(BASE, "data", "processed")
HGB = lambda: HistGradientBoostingClassifier(random_state=0)


def load():
    spt = pd.read_csv(os.path.join(PROC, "cetin2018_baseline_records.csv"))
    FS = ["Mw", "amax_g", "sigma_v_kpa", "sigma_eff_kpa", "rd", "FC", "N1_60", "N1_60cs", "crit_depth_m", "water_depth_ft"]
    spt = spt.dropna(subset=FS + ["y", "earthquake", "CRR_BI2014", "CSR_cetin"]).copy()
    spt = spt[np.isfinite(spt[FS].to_numpy(float)).all(1) & (spt["CRR_BI2014"] > 0) & (spt["CSR_cetin"] > 0)]
    s_spt = np.log((spt["CSR_cetin"] / spt["CRR_BI2014"]).to_numpy(float))  # margin: >0 -> liquefy
    D_spt = dict(X=spt[FS].to_numpy(float), y=spt["y"].astype(int).to_numpy(),
                 g=spt["earthquake"].to_numpy(), s=s_spt, name="SPT/Cetin2018")
    cpt = pd.read_csv(os.path.join(PROC, "geyin2021_cpt_records.csv"))
    FC = ["Mw", "PGA", "GWT", "LPI", "crit_FS_capped", "crit_depth", "qc1Ncs_crit", "Ic_crit", "sev_crit"]
    s_cpt = np.log1p(cpt["LPI"].to_numpy(float))
    D_cpt = dict(X=cpt[FC].to_numpy(float), y=cpt["y"].astype(int).to_numpy(),
                 g=cpt["event"].to_numpy(), s=s_cpt, name="CPT/Geyin2021")
    return D_spt, D_cpt


def grouped_oof_auc(X, y, g):
    cv = list(GroupKFold(5).split(X, y, g))
    return float(roc_auc_score(y, cross_val_predict(HGB(), X, y, cv=cv, method="predict_proba")[:, 1])), cv


def grouped_oof_prob(feat, y, g, model):
    cv = list(GroupKFold(5).split(feat, y, g))
    return cross_val_predict(model, feat, y, cv=cv, method="predict_proba")[:, 1]


out = {}
for D in load():
    X, y, g, s, name = D["X"], D["y"], D["g"], D["s"], D["name"]
    res = {"n": int(len(y)), "n_eq": int(len(np.unique(g)))}

    # ---- A1 grouped bound: full features vs single mechanistic margin (out-of-distribution) ----
    auc_phys = float(roc_auc_score(y, s))
    auc_full, cv = grouped_oof_auc(X, y, g)
    p_margin = grouped_oof_prob(s.reshape(-1, 1), y, g, make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)))
    auc_margin = float(roc_auc_score(y, p_margin))
    p_full = cross_val_predict(HGB(), X, y, cv=cv, method="predict_proba")[:, 1]
    pair = R.paired_auc_diff_ci(y, p_margin, p_full, groups=g, n_boot=1500)  # margin - full
    res["A1_sufficiency"] = {"auc_physics_raw": round(auc_phys, 3), "auc_margin_only_ML": round(auc_margin, 3),
                             "auc_full_feature_ML": round(auc_full, 3),
                             "delta_margin_minus_full": round(pair["delta"], 3),
                             "CI95": [round(pair["lo"], 3), round(pair["hi"], 3)], "P(margin>=full)": round(pair["P(delta>0)"], 3)}

    # ---- A2 screening: inside the ambiguous band, is label still learnable from features? ----
    p_phys = grouped_oof_prob(s.reshape(-1, 1), y, g, make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)))
    amb = (p_phys >= 0.3) & (p_phys <= 0.7)
    res["A2_screening_ambiguous_band"] = {"n_in_band": int(amb.sum())}
    if amb.sum() >= 30 and len(np.unique(y[amb])) > 1 and len(np.unique(g[amb])) >= 5:
        cvb = list(GroupKFold(min(5, len(np.unique(g[amb])))).split(X[amb], y[amb], g[amb]))
        pin = cross_val_predict(HGB(), X[amb], y[amb], cv=cvb, method="predict_proba")[:, 1]
        res["A2_screening_ambiguous_band"]["ML_AUC_within_band"] = round(float(roc_auc_score(y[amb], pin)), 3)

    # ---- A3 empirical ambiguity proxy vs mechanistic coordinate ----
    order = np.argsort(p_phys)
    nb = 10
    bins = np.array_split(order, nb)
    bayes = 0.0
    band_rows = []
    for b in bins:
        fr = float(y[b].mean())
        be = min(fr, 1 - fr)
        bayes += be * len(b)
        band_rows.append({"p_mean": round(float(p_phys[b].mean()), 3), "liq_freq": round(fr, 3), "bayes_err": round(be, 3), "n": int(len(b))})
    bayes /= len(y)
    err_ml = float(np.mean((p_full >= 0.5).astype(int) != y))
    amb_bayes = sum(min(r["liq_freq"], 1 - r["liq_freq"]) * r["n"] for r in band_rows if 0.3 <= r["p_mean"] <= 0.7)
    res["A3_irreducible"] = {"bayes_error_lower_bound": round(bayes, 3), "ML_full_error": round(err_ml, 3),
                             "frac_bayes_err_in_ambiguous_band": round(amb_bayes / (bayes * len(y) + 1e-9), 3),
                             "bands": band_rows}

    # ---- A4 Mondrian (mechanism-conditioned) conformal vs plain conformal ----
    def band_of(p):
        return np.clip((p * 3).astype(int), 0, 2)  # 3 mechanistic bands
    rng = np.random.default_rng(0)
    cov_plain = {0: [], 1: [], 2: []}
    cov_mond = {0: [], 1: [], 2: []}
    for _ in range(60):
        gids = rng.permutation(np.unique(g))
        ng = len(gids)
        gtr, gca, gte = gids[:int(.4 * ng)], gids[int(.4 * ng):int(.7 * ng)], gids[int(.7 * ng):]
        tr = np.where(np.isin(g, gtr))[0]; ca = np.where(np.isin(g, gca))[0]; te = np.where(np.isin(g, gte))[0]
        if len(np.unique(y[tr])) < 2 or len(ca) < 8 or len(te) < 8:
            continue
        m = HGB().fit(X[tr], y[tr])
        pca, pte = m.predict_proba(X[ca])[:, 1], m.predict_proba(X[te])[:, 1]
        # plain conformal
        i0, i1 = split_conformal_sets(pca, y[ca], pte, alpha=0.1)
        cov = np.where(y[te] == 1, i1, i0)
        # Mondrian: separate quantile per mechanistic band
        bca, bte = band_of(pca), band_of(pte)
        covm = np.zeros(len(te), bool)
        for bb in [0, 1, 2]:
            mca, mte = bca == bb, bte == bb
            if mca.sum() >= 5 and mte.sum() >= 1:
                j0, j1 = split_conformal_sets(pca[mca], y[ca][mca], pte[mte], alpha=0.1)
                covm[mte] = np.where(y[te][mte] == 1, j1, j0)
            elif mte.sum() >= 1:
                covm[mte] = cov[mte]
        for bb in [0, 1, 2]:
            mt = band_of(pte) == bb
            if mt.sum() >= 1:
                cov_plain[bb].append(float(cov[mt].mean()))
                cov_mond[bb].append(float(covm[mt].mean()))
    res["A4_mondrian_band_coverage"] = {
        "target": 0.9,
        "plain": {bb: round(float(np.mean(v)), 3) for bb, v in cov_plain.items() if v},
        "mondrian": {bb: round(float(np.mean(v)), 3) for bb, v in cov_mond.items() if v},
    }
    out[name] = res

with open(os.path.join(PROC, "innovation_analysis.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

for name, r in out.items():
    print("=" * 72)
    print(name, f"(n={r['n']}, earthquakes={r['n_eq']})")
    a1 = r["A1_sufficiency"]
    print(f" A1 GROUPED BOUND: physics_raw={a1['auc_physics_raw']}  margin-only-ML={a1['auc_margin_only_ML']}  full-feature-ML={a1['auc_full_feature_ML']}")
    print(f"    margin - full (grouped) = {a1['delta_margin_minus_full']:+.3f}  CI{a1['CI95']}  P(margin>=full)={a1['P(margin>=full)']}")
    a2 = r["A2_screening_ambiguous_band"]
    print(f" A2 SCREENING: ML AUC inside ambiguous band (FS~1) = {a2.get('ML_AUC_within_band','n/a')} (n={a2['n_in_band']})")
    a3 = r["A3_irreducible"]
    print(f" A3 AMBIGUITY PROXY: proxy={a3['bayes_error_lower_bound']}  ML error={a3['ML_full_error']}  frac in ambiguous band={a3['frac_bayes_err_in_ambiguous_band']}")
    a4 = r["A4_mondrian_band_coverage"]
    print(f" A4 BAND COVERAGE (target .9): plain={a4['plain']}  mondrian={a4['mondrian']}")
print("\nwrote data/processed/innovation_analysis.json")
