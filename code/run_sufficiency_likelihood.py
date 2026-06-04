"""Likelihood/information sufficiency (stronger than AUC parity):
does ANY model (incl. a neural net) improve OUT-OF-SAMPLE log-loss over the single
mechanistic margin under earthquake-grouped CV? Per-sample paired cluster bootstrap.
Fair multi-model comparison to preempt 'your ML was weak'.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
PROC = os.path.join(BASE, "data", "processed")


def ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))  # per-sample deviance


def load():
    spt = pd.read_csv(os.path.join(PROC, "cetin2018_baseline_records.csv"))
    FS = ["Mw", "amax_g", "sigma_v_kpa", "sigma_eff_kpa", "rd", "FC", "N1_60", "N1_60cs", "crit_depth_m", "water_depth_ft"]
    spt = spt.dropna(subset=FS + ["y", "earthquake", "CRR_BI2014", "CSR_cetin"]).copy()
    spt = spt[np.isfinite(spt[FS].to_numpy(float)).all(1) & (spt["CRR_BI2014"] > 0) & (spt["CSR_cetin"] > 0)]
    yield dict(name="SPT/Cetin2018", X=spt[FS].to_numpy(float), y=spt["y"].astype(int).to_numpy(),
               g=spt["earthquake"].to_numpy(), s=np.log((spt["CSR_cetin"] / spt["CRR_BI2014"]).to_numpy(float)))
    cpt = pd.read_csv(os.path.join(PROC, "geyin2021_cpt_records.csv"))
    FC = ["Mw", "PGA", "GWT", "LPI", "crit_FS_capped", "crit_depth", "qc1Ncs_crit", "Ic_crit", "sev_crit"]
    yield dict(name="CPT/Geyin2021", X=cpt[FC].to_numpy(float), y=cpt["y"].astype(int).to_numpy(),
               g=cpt["event"].to_numpy(), s=np.log1p(cpt["LPI"].to_numpy(float)))


MODELS = {
    "logistic": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=5000)),
    "random_forest": lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=500, random_state=0)),
    "grad_boost": lambda: make_pipeline(SimpleImputer(strategy="median"), GradientBoostingClassifier(random_state=0)),
    "hist_gbt": lambda: HistGradientBoostingClassifier(random_state=0),
    "neural_net": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=3000, random_state=0)),
}


def oof(est, X, y, g):
    cv = list(GroupKFold(5).split(X, y, g))
    return cross_val_predict(est, X, y, cv=cv, method="predict_proba")[:, 1]


def cluster_boot(diff, g, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    g = np.asarray(g); gids = np.unique(g)
    mem = {k: np.where(g == k)[0] for k in gids}
    vals = [diff[np.concatenate([mem[k] for k in rng.choice(gids, len(gids), True)])].mean() for _ in range(n)]
    return float(diff.mean()), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


res = {}
for D in load():
    X, y, g, s, name = D["X"], D["y"], D["g"], D["s"], D["name"]
    pm = oof(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), s.reshape(-1, 1), y, g)
    llm = ll(y, pm)
    amb = (pm >= 0.3) & (pm <= 0.7)
    row = {"margin_only": {"logloss": round(float(llm.mean()), 4), "auc": round(float(roc_auc_score(y, pm)), 3)}}
    for mname, mk in MODELS.items():
        pf = oof(mk(), X, y, g); llf = ll(y, pf)
        d, lo, hi = cluster_boot(llf - llm, g)
        row[mname] = {"logloss": round(float(llf.mean()), 4), "auc": round(float(roc_auc_score(y, pf)), 3),
                      "dlogloss_full_minus_margin": round(d, 4), "CI95": [round(lo, 4), round(hi, 4)],
                      "full_improves_OOS_loglik": bool(hi < 0),
                      "band_dll": round(float((llf[amb] - llm[amb]).mean()), 4) if amb.sum() > 10 else None}
    res[name] = row

json.dump(res, open(os.path.join(PROC, "sufficiency_likelihood.json"), "w"), indent=2)
for name, row in res.items():
    print("=" * 74); print(name, f"(margin-only: logloss={row['margin_only']['logloss']}, auc={row['margin_only']['auc']})")
    for m in ["logistic", "random_forest", "grad_boost", "hist_gbt", "neural_net"]:
        e = row[m]; flag = "full BETTER (OOS loglik)" if e["full_improves_OOS_loglik"] else "margin >= full"
        print(f"  {m:<13} ll={e['logloss']:.4f} auc={e['auc']:.3f}  d_ll(full-margin)={e['dlogloss_full_minus_margin']:+.4f} CI{e['CI95']} -> {flag}")
print("\nwrote sufficiency_likelihood.json")
