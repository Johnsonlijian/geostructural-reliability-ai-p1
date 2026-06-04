"""Publication figures for P1 (real data, both DBs). Outputs 300-dpi PNGs to figures/."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from geoliq import reliability as R
from geoliq.reliability import split_conformal_sets

PROC = os.path.join(BASE, "data", "processed")
FIG = os.path.join(BASE, "..", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 120})
C_PHYS, C_ML, C_GW = "#1f3b73", "#c0392b", "#2e8b57"
HGB = lambda: HistGradientBoostingClassifier(random_state=0)

# ---------- load both DBs ----------
spt = pd.read_csv(os.path.join(PROC, "cetin2018_baseline_records.csv"))
FS = ["Mw", "amax_g", "sigma_v_kpa", "sigma_eff_kpa", "rd", "FC", "N1_60", "N1_60cs", "crit_depth_m", "water_depth_ft"]
spt = spt.dropna(subset=FS + ["y", "earthquake", "CRR_BI2014", "CSR_cetin"]).copy()
spt = spt[np.isfinite(spt[FS].to_numpy(float)).all(axis=1) & (spt["CRR_BI2014"] > 0) & (spt["CSR_cetin"] > 0)]
Xs, ys, gs = spt[FS].to_numpy(float), spt["y"].astype(int).to_numpy(), spt["earthquake"].to_numpy()
phs = (spt["CSR_cetin"] / spt["CRR_BI2014"]).to_numpy(float)

cpt = pd.read_csv(os.path.join(PROC, "geyin2021_cpt_records.csv"))
FC = ["Mw", "PGA", "GWT", "LPI", "crit_FS_capped", "crit_depth", "qc1Ncs_crit", "Ic_crit", "sev_crit"]
Xc, yc, gc = cpt[FC].to_numpy(float), cpt["y"].astype(int).to_numpy(), cpt["event"].to_numpy()
phc = cpt["LPI"].to_numpy(float)


def ml_grouped_oof(X, y, g):
    gkf = list(GroupKFold(n_splits=5).split(X, y, g))
    return cross_val_predict(HGB(), X, y, cv=gkf, method="predict_proba")[:, 1]


def ml_random_auc(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    return roc_auc_score(y, cross_val_predict(HGB(), X, y, cv=skf, method="predict_proba")[:, 1])


def phys_cal_oof(score, y, g):
    gkf = list(GroupKFold(n_splits=5).split(score.reshape(-1, 1), y, g))
    Xl = np.log(np.clip(score, 1e-6, None)).reshape(-1, 1)
    return cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), Xl, y, cv=gkf, method="predict_proba")[:, 1]


mls, mlc = ml_grouped_oof(Xs, ys, gs), ml_grouped_oof(Xc, yc, gc)
phs_cal, phc_cal = phys_cal_oof(phs, ys, gs), phys_cal_oof(phc, yc, gc)
saved = []


def done(fig, name):
    p = os.path.join(FIG, name)
    fig.tight_layout()
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved.append(name)


# ---------- Fig 1: study-design schematic ----------
try:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axis("off")
    boxes = [("Hazard\n(Mw, PGA)", 0.02, C_PHYS), ("Groundwater /\neffective stress", 0.21, C_GW),
             ("Mechanism baseline\n(BI2014 / LPI), zero-shot", 0.40, C_PHYS),
             ("Residual ML\n(data-driven)", 0.60, C_ML),
             ("Source-aware\nvalidation + conformal", 0.79, "#444444")]
    for txt, x, col in boxes:
        ax.add_patch(plt.Rectangle((x, 0.35), 0.165, 0.32, fc="white", ec=col, lw=2))
        ax.text(x + 0.082, 0.51, txt, ha="center", va="center", fontsize=9.5, color=col)
    for x in [0.185, 0.375, 0.565, 0.755]:
        ax.annotate("", xy=(x + 0.03, 0.51), xytext=(x, 0.51), arrowprops=dict(arrowstyle="->", lw=1.6))
    ax.text(0.5, 0.86, "From hazard to engineering failure: groundwater as the mechanistic hidden state;\nreliability under source-aware validation", ha="center", fontsize=10)
    ax.text(0.5, 0.12, "Reliability boundary: marginal coverage holds, per-earthquake coverage fails", ha="center", fontsize=9, style="italic", color="#777777")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    done(fig, "fig1_design.png")
except Exception as e:
    print("fig1 FAIL", repr(e))

# ---------- Fig 2: optimism gap ----------
try:
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (X, y, g, ph, title) in zip(axs, [(Xs, ys, gs, phs, "SPT / Cetin 2018 (triggering)"),
                                              (Xc, yc, gc, phc, "CPT / Geyin-Maurer 2021 (manifestation)")]):
        rnd = ml_random_auc(X, y)
        grp = roc_auc_score(y, ml_grouped_oof(X, y, g))
        phy = roc_auc_score(y, ph)
        ax.bar([0, 1], [rnd, grp], color=[C_ML, "#e8a0a0"], width=0.55, edgecolor="k")
        ax.axhline(phy, color=C_PHYS, ls="--", lw=2, label=f"zero-shot physics ({phy:.3f})")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["random\nK-fold", "earthquake\ngrouped"])
        for i, v in enumerate([rnd, grp]):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
        ax.annotate(f"optimism\ngap {rnd-grp:+.3f}", xy=(0.5, (rnd+grp)/2), ha="center", fontsize=9, color="#333")
        ax.set_ylim(0.5, 1.0); ax.set_ylabel("ROC-AUC"); ax.set_title(title, fontsize=10); ax.legend(loc="lower left", fontsize=9)
    fig.suptitle("ML AUC inflated by random splitting; never beats zero-shot physics", fontsize=11)
    done(fig, "fig2_optimism_gap.png")
except Exception as e:
    print("fig2 FAIL", repr(e))

# ---------- Fig 3: ROC physics vs ML ----------
try:
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.3))
    for ax, (y, ph, ml, title) in zip(axs, [(ys, phs, mls, "SPT / Cetin 2018"), (yc, phc, mlc, "CPT / Geyin-Maurer 2021")]):
        cip = R.bootstrap_auc_ci(y, ph); cim = R.bootstrap_auc_ci(y, ml)
        fpr, tpr, _ = roc_curve(y, ph); ax.plot(fpr, tpr, color=C_PHYS, lw=2.2, label=f"physics  AUC={cip['auc']:.3f} [{cip['lo']:.2f},{cip['hi']:.2f}]")
        fpr, tpr, _ = roc_curve(y, ml); ax.plot(fpr, tpr, color=C_ML, lw=2.0, ls="--", label=f"ML (grouped)  AUC={cim['auc']:.3f} [{cim['lo']:.2f},{cim['hi']:.2f}]")
        ax.plot([0, 1], [0, 1], color="gray", lw=1, ls=":")
        ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate"); ax.set_title(title, fontsize=10); ax.legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Zero-shot mechanism vs source-aware ML (95% bootstrap CI)", fontsize=11)
    done(fig, "fig3_roc.png")
except Exception as e:
    print("fig3 FAIL", repr(e))

# ---------- Fig 4: groundwater counterfactual (SPT) ----------
try:
    gw = json.load(open(os.path.join(PROC, "cetin2018_groundwater_ablation.json"), encoding="utf-8"))
    sc = gw["counterfactual_water_table"]
    order = ["surface(dw=0)", "actual", "deep/dry(dw=z)"]
    medfs = [sc[k]["median_FS"] for k in order]
    rate = [sc[k]["pred_liq_rate_FS<1"] for k in order]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(range(3), medfs, color=C_GW, alpha=0.8, edgecolor="k", width=0.55, label="median FS")
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(["water at\nsurface", "actual\nwater table", "deep /\ndry"])
    ax.set_ylabel("median factor of safety", color=C_GW)
    ax2 = ax.twinx()
    ax2.plot(range(3), rate, color=C_ML, marker="o", lw=2, label="predicted liquefaction rate")
    ax2.set_ylabel("predicted liquefaction rate (FS<1)", color=C_ML); ax2.set_ylim(0, 1); ax2.grid(False)
    for i, v in enumerate(rate):
        ax2.text(i, v + 0.03, f"{v:.2f}", ha="center", color=C_ML, fontsize=9)
    ax.set_title("Groundwater mechanistically controls triggering (SPT, validated engine)", fontsize=10)
    done(fig, "fig4_groundwater.png")
except Exception as e:
    print("fig4 FAIL", repr(e))

# ---------- Fig 5: reliability (calibration) diagram ----------
try:
    def rel_curve(p, y, bins=8):
        edges = np.linspace(0, 1, bins + 1); xs, ysv = [], []
        for i in range(bins):
            m = (p >= edges[i]) & (p <= edges[i + 1] if i == bins - 1 else p < edges[i + 1])
            if m.sum() >= 5:
                xs.append(p[m].mean()); ysv.append(y[m].mean())
        return xs, ysv
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.3))
    for ax, (y, pc, ml, title) in zip(axs, [(ys, phs_cal, mls, "SPT / Cetin 2018"), (yc, phc_cal, mlc, "CPT / Geyin-Maurer 2021")]):
        ax.plot([0, 1], [0, 1], ":", color="gray", lw=1)
        x1, y1 = rel_curve(pc, y); ax.plot(x1, y1, "-o", color=C_PHYS, lw=2, label=f"physics (ECE={R.ece(y,pc):.3f})")
        x2, y2 = rel_curve(ml, y); ax.plot(x2, y2, "--s", color=C_ML, lw=2, label=f"ML (ECE={R.ece(y,ml):.3f})")
        ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed frequency"); ax.set_title(title, fontsize=10); ax.legend(loc="upper left", fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.suptitle("Calibration: physics well-calibrated; ML miscalibrated out-of-distribution", fontsize=11)
    done(fig, "fig5_calibration.png")
except Exception as e:
    print("fig5 FAIL", repr(e))

# ---------- Fig 6 (money): conformal conditional coverage ----------
try:
    def cond_samples(X, y, g, n_rep=40, seed=0, alpha=0.1, min_pts=3):
        rng = np.random.default_rng(seed); pgc = []
        gar = np.asarray(g)
        for _ in range(n_rep):
            gids = rng.permutation(np.unique(gar)); ng = len(gids)
            gtr, gca, gte = gids[:int(0.4*ng)], gids[int(0.4*ng):int(0.7*ng)], gids[int(0.7*ng):]
            tr = np.where(np.isin(gar, gtr))[0]; ca = np.where(np.isin(gar, gca))[0]; te = np.where(np.isin(gar, gte))[0]
            if len(np.unique(y[tr])) < 2 or len(ca) < 5 or len(te) < 5:
                continue
            mdl = HGB(); mdl.fit(X[tr], y[tr])
            inc0, inc1 = split_conformal_sets(mdl.predict_proba(X[ca])[:, 1], y[ca], mdl.predict_proba(X[te])[:, 1], alpha)
            cov = np.where(y[te] == 1, inc1, inc0)
            for gg in np.unique(gar[te]):
                m = gar[te] == gg
                if m.sum() >= min_pts:
                    pgc.append(float(cov[m].mean()))
        return np.array(pgc)
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.3))
    for ax, (X, y, g, title) in zip(axs, [(Xs, ys, gs, "SPT / Cetin 2018"), (Xc, yc, gc, "CPT / Geyin-Maurer 2021")]):
        s = cond_samples(X, y, g)
        _rk = "SPT_Cetin2018" if title.startswith("SPT") else "CPT_Geyin2021"
        _rr = json.load(open(os.path.join(PROC, "reliability_upgrade.json"), encoding="utf-8"))[_rk]["conformal_conditional_grouped"]
        frac, meanc = _rr["frac_events_undercovered"], _rr["mean_conditional_coverage"]
        ax.hist(s, bins=np.linspace(0, 1, 21), color="#888", edgecolor="k", alpha=0.85)
        ax.axvline(0.90, color=C_PHYS, ls="--", lw=2, label="target coverage 0.90")
        ax.axvline(meanc, color=C_ML, lw=2, label=f"mean cond. coverage {meanc:.2f}")
        ax.set_xlabel("per-earthquake conformal coverage"); ax.set_ylabel("count (bootstrap evaluations)")
        ax.set_title(f"{title}\n{frac*100:.1f}% of new earthquakes under-covered (primary run)", fontsize=10); ax.legend(fontsize=8.5)
    fig.suptitle("Reliability boundary: marginal coverage holds, per-earthquake coverage fails", fontsize=11)
    done(fig, "fig6_conformal_coverage.png")
except Exception as e:
    print("fig6 FAIL", repr(e))

print("SAVED:", saved)
