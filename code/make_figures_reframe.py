"""Figure system for the re-framed liquefaction paper (P1 upgrade, Engineering Geology).

All four figures are generated from the audited JSON outputs (no fabricated data):
  Fig 1  opener  — the over-validation flip (mechanism schematic + measured sign flip)
  Fig 2  ceiling — leave-event-out AUC ceiling + label-conditioned error floor, 3 arms
  Fig 3  no-ML   — tested tuned ML does not beat the zero-shot mechanistic margin, 3 arms
  Fig 4  VoI     — margin value-of-information surface + the ~0 added-predictor delta

Saves vector (PDF + SVG) and 600-dpi PNG to figures/reframe_2026-06-30/.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

# Journal styling: try the project tooling, else a clean fallback.
try:
    sys.path.insert(0, r"R:\AcademicWorkspace\tools")
    from pyplot_cjk import set_style
    set_style(lang="en")
except Exception:
    pass
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "axes.linewidth": 0.8,
    "font.family": "DejaVu Sans",
})

BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, "data", "processed")
FIGDIR = os.path.join(BASE, "..", "figures", "reframe_2026-06-30")
os.makedirs(FIGDIR, exist_ok=True)

CMARGIN, CML, CRAND, CGRP = "#1f4e79", "#c0392b", "#e08214", "#2166ac"
ARMS = [("SPT_Cetin2018", "SPT", "triggering"),
        ("Vs_Kayen2013", "V$_s$", "triggering"),
        ("CPT_Geyin2021", "CPT", "manifestation")]


def load(name):
    with open(os.path.join(PROC, name), encoding="utf-8") as f:
        return json.load(f)


def save(fig, stem):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(os.path.join(FIGDIR, f"{stem}.{ext}"), dpi=600)
    plt.close(fig)
    print("wrote", stem)


def fig1_flip():
    ov = load("overvalidation_benchmark.json")
    fig = plt.figure(figsize=(7.2, 3.2))
    gsA = fig.add_axes([0.045, 0.12, 0.40, 0.78]); gsA.axis("off")
    # --- Panel A: leakage schematic ---
    gsA.set_title("a  Why apparent ML skill is inflated", loc="left", fontweight="bold")
    rng = np.random.default_rng(0)
    cols = ["#4477aa", "#ee6677", "#228833", "# ccbb44".replace(" ", "")]
    for j, (cx, lab) in enumerate([(0.27, "random split\n(leaks within event)"),
                                   (0.74, "event-grouped split")]):
        gsA.text(cx, 0.96, lab, ha="center", va="top", fontsize=7.6)
        for k in range(3):  # three earthquakes = three clusters
            cxk, cyk = cx + (k - 1) * 0.12, 0.62 - k * 0.18
            pts = rng.normal([cxk, cyk], 0.028, size=(6, 2))
            if j == 0:   # random: color = train/test mixed inside each cluster
                c = ["#999999" if rng.random() < 0.5 else "#000000" for _ in range(6)]
            else:        # grouped: whole cluster is train(grey) or test(black)
                c = ["#000000" if k == 1 else "#999999"] * 6
            gsA.scatter(pts[:, 0], pts[:, 1], s=7, c=c, edgecolors="none")
            gsA.add_patch(Rectangle((cxk - 0.05, cyk - 0.05), 0.10, 0.10, fill=False,
                                    ec=cols[k], lw=0.8))
    gsA.scatter([], [], s=10, c="#000000", label="held-out (test)")
    gsA.scatter([], [], s=10, c="#999999", label="train")
    gsA.legend(loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04),
               handletextpad=0.2, columnspacing=1.0)
    gsA.set_xlim(0, 1); gsA.set_ylim(0, 1)

    # --- Panel B: the measured flip ---
    ax = fig.add_axes([0.56, 0.17, 0.40, 0.70])
    ax.set_title("b  The flip: best ML $-$ margin AUC", loc="left", fontweight="bold")
    ys = np.arange(len(ARMS))[::-1]
    for y, (key, lab, resp) in zip(ys, ARMS):
        f = ov[key]["ml_vs_margin_flip"]
        xr = f["under_random_cv_best_ml_minus_margin_AUC"]
        xg = f["under_grouped_cv_best_ml_minus_margin_AUC"]
        ax.annotate("", xy=(xg, y), xytext=(xr, y),
                    arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.6))
        ax.scatter([xr], [y], s=42, color=CRAND, zorder=5)
        ax.scatter([xg], [y], s=42, color=CGRP, zorder=5)
        ax.text(-0.165, y, f"{lab}", va="center", ha="left", fontsize=8.5)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.scatter([], [], color=CRAND, label="random CV (illusory +)")
    ax.scatter([], [], color=CGRP, label="event-grouped CV")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    ax.set_yticks([]); ax.set_xlabel(r"$\Delta$AUC (ML $-$ mechanistic margin)")
    ax.set_ylim(-0.6, len(ARMS) - 0.2); ax.set_xlim(-0.17, 0.07)
    ax.text(0.045, -0.5, "margin\nwins", color=CGRP, fontsize=7, ha="center")
    ax.text(0.03, len(ARMS) - 0.45, "ML 'wins'", color=CRAND, fontsize=7, ha="center")
    save(fig, "fig1_overvalidation_flip")


def fig2_ceiling():
    pc = load("predictability_ceiling.json")
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 2.7))
    ys = np.arange(len(ARMS))[::-1]
    for y, (key, lab, resp) in zip(ys, ARMS):
        c = pc[key]["predictability_ceiling"]["discrimination_AUC_ceiling"]
        axA.errorbar(c["auc"], y, xerr=[[c["auc"] - c["lo"]], [c["hi"] - c["auc"]]],
                     fmt="o", color=CMARGIN, capsize=2.5, ms=5)
        axA.text(c["auc"], y + 0.18, f"{c['auc']:.3f}", ha="center", fontsize=7.5)
        fl = pc[key]["predictability_ceiling"]["irreducible_error_floor_calibrated"]
        ln = pc[key]["label_noise_sensitivity_on_margin_floor"]
        lo10 = [d["intrinsic_floor"] for d in ln if d["rho"] == 0.1][0]
        axB.plot([lo10, fl["est"]], [y, y], color="#bbbbbb", lw=4, solid_capstyle="round")
        axB.scatter([fl["est"]], [y], color="#8c510a", s=30, zorder=5)
        axB.text(fl["est"], y + 0.18, f"{fl['est']:.3f}", ha="center", fontsize=7.5)
        for ax in (axA, axB):
            ax.text(ax.get_xlim()[0], y, "", va="center")
    for ax, ttl, xl in [(axA, "a  Discrimination ceiling", "leave-event-out AUC"),
                        (axB, "b  Label-conditioned error floor", "min error (band: 0–10% label noise)")]:
        ax.set_title(ttl, loc="left", fontweight="bold")
        ax.set_yticks(ys); ax.set_yticklabels([f"{l}\n({r})" for _, l, r in ARMS])
        ax.set_xlabel(xl); ax.set_ylim(-0.5, len(ARMS) - 0.3)
    axA.axvline(0.78, color=CML, ls=":", lw=1.2)
    axA.text(0.775, -0.42, "lit. plateau ~0.78", color=CML, fontsize=6.6, ha="center", va="center")
    axA.set_xlim(0.6, 0.98); axA.set_ylim(-0.6, len(ARMS) - 0.3)
    save(fig, "fig2_predictability_ceiling")


def fig3_noml():
    tm = load("tuned_ml_challenge.json")
    order = ["logreg_full", "knn", "hgb", "rf"]
    labs = {"logreg_full": "logistic", "knn": "kNN", "hgb": "grad-boost", "rf": "rand-forest"}
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 3.1), sharey=False)
    fig.subplots_adjust(top=0.74, bottom=0.17, wspace=0.28)
    for ax, (key, lab, resp) in zip(axes, ARMS):
        m = tm[key]["margin_grouped_auc"]
        ax.axvspan(m["lo"], m["hi"], color=CMARGIN, alpha=0.12)
        ax.axvline(m["auc"], color=CMARGIN, lw=1.8, label="mechanistic margin")
        for i, mod in enumerate(order):
            v = tm[key]["tuned_ml_grouped_auc"][mod]
            ax.errorbar(v["auc"], i, xerr=[[v["auc"] - v["lo"]], [v["hi"] - v["auc"]]],
                        fmt="s", color=CML, capsize=2, ms=4)
        ax.set_yticks(range(len(order))); ax.set_yticklabels([labs[m] for m in order])
        d = tm[key]["best_tuned_ml_minus_margin_AUC"]
        ax.set_title(f"{lab} ({resp}); margin {m['auc']:.3f}", fontsize=7.6)
        ax.set_xlabel("leave-event-out AUC"); ax.set_ylim(-0.6, len(order) - 0.4)
        ax.invert_yaxis()
    axes[0].legend(loc="lower left", frameon=False, fontsize=7)
    fig.suptitle("Tuned ML challengers do not exceed the mechanism-based margin",
                 fontsize=8.3, y=0.985)
    save(fig, "fig3_tuned_ml_challenge")


def fig4_voi():
    vi = load("value_of_information.json")
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    Rsel = "10"
    for (key, lab, resp), col in zip(ARMS, [CMARGIN, "#2c7fb8", "#7a4fa0"]):
        surf = vi[key]["margin_voi_surface"][Rsel]
        axA.plot([d["pi"] for d in surf], [d["voi"] for d in surf], "-o", ms=3, color=col, label=lab)
    axA.set_title("a  Value of resolving the margin (R=10)", loc="left", fontweight="bold")
    axA.set_xlabel(r"prior triggering probability $\pi$")
    axA.set_ylabel("VoI  (units of $C_{FP}$)"); axA.legend(frameon=False)
    # Panel B: added-predictor delta at pi=0.2,R=10
    ys = np.arange(len(ARMS))[::-1]
    for y, (key, lab, resp) in zip(ys, ARMS):
        s = vi[key]["extra_predictors_added_value"]["pi=0.2,R=10"]
        d = s["delta_full_minus_margin"]
        axB.errorbar(d, y, xerr=[[d - s["lo"]], [s["hi"] - d]], fmt="o", color="#444444", capsize=2.5, ms=5)
        axB.text(d, y + 0.16, f"{d:+.3f}", ha="center", fontsize=7.5)
    axB.axvline(0, color="k", ls="--", lw=0.8)
    axB.set_title("b  Do extra predictors add decision value?", loc="left", fontweight="bold")
    axB.set_yticks(ys); axB.set_yticklabels([f"{l}" for _, l, r in ARMS])
    axB.set_xlabel(r"$\Delta$VoI (full predictors $-$ margin), $\pi$=0.2, R=10")
    axB.set_ylim(-0.5, len(ARMS) - 0.3); axB.set_xlim(-0.2, 0.2)
    axB.text(0.10, len(ARMS) - 0.5, "≈ 0\n(no added value)", fontsize=7, ha="center", color="#444")
    save(fig, "fig4_value_of_information")


def fig5_geology():
    gr = load("geological_residual.json")
    fig, ax = plt.subplots(figsize=(3.95, 2.8))
    GE, RA = "CPT_Geyin2021", "CPT_Rateria2024_independent_replication"
    rows = [(GE, "sand-like (Ic<2.05)", "Geyin · sand-like", CMARGIN, 3),
            (GE, "transitional (2.05-2.60)", "Geyin · transitional", CML, 2),
            (RA, "sand-like (Ic<2.05)", "Rateria · sand-like", CMARGIN, 1),
            (RA, "transitional (2.05-2.60)", "Rateria · transitional", CML, 0)]
    yt, ylab = [], []
    for dbkey, skey, lab, col, yy in rows:
        st = gr[dbkey]["by_soil_behaviour_type_Ic"][skey]
        ax.errorbar(st["mean"], yy, xerr=[[st["mean"] - st["lo"]], [st["hi"] - st["mean"]]],
                    fmt="o", color=col, capsize=2.5, ms=5.5)
        yt.append(yy); ylab.append(lab)
    for dbkey, yy in [(GE, 2.0), (RA, 0.0)]:
        c = gr[dbkey]["contrast_transitional_vs_sandlike_Ic"]
        st = gr[dbkey]["by_soil_behaviour_type_Ic"]["transitional (2.05-2.60)"]
        ax.annotate(f"$\\Delta$=+{c['diff']:.3f}*", xy=(st["hi"] + 0.006, yy), fontsize=7.2, va="center", color="#333")
    ax.set_yticks(yt); ax.set_yticklabels(ylab, fontsize=7.6); ax.set_ylim(-0.6, 3.6); ax.invert_yaxis()
    ax.set_xlim(0.15, 0.45); ax.set_xlabel(r"label-conditioned residual floor  min$(p,1-p)$")
    ax.set_title("Residual uncertainty is geologically controlled:\nconcentrated in transitional silt-sand soils",
                 fontsize=8.6, fontweight="bold")
    ax.scatter([], [], color=CMARGIN, label="sand-like (Ic<2.05)")
    ax.scatter([], [], color=CML, label="transitional (Ic 2.05–2.60)")
    ax.legend(loc="lower right", frameon=False, fontsize=6.8)
    ax.text(0.16, 3.5, "* p<0.05, cluster-bootstrap", fontsize=6, color="#777")
    save(fig, "fig5_geological_structure")


def fig1_mechanism():
    """Opening mechanism figure (机理图): three in-situ probes -> one effective-stress margin ->
    a predictability ceiling at FS=1 (concentrated in transitional soils); plus the over-validation flip.
    Panel (a) uses the REAL SPT cases on the margin coordinate (not a cartoon)."""
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    ov = load("overvalidation_benchmark.json")
    spt = pd.read_csv(os.path.join(PROC, "cetin2018_baseline_records.csv"))
    spt = spt.dropna(subset=["CSR_cetin", "CRR_BI2014", "y"])
    spt = spt[(spt["CSR_cetin"] > 0) & (spt["CRR_BI2014"] > 0)]
    M = np.log((spt["CRR_BI2014"] / spt["CSR_cetin"]).to_numpy(float))   # ln FS; >0 = safe
    y = spt["y"].astype(int).to_numpy()                                  # 1 = liquefied
    lr = LogisticRegression(max_iter=5000).fit(M.reshape(-1, 1), y)
    g = np.linspace(np.percentile(M, 1) - 0.1, np.percentile(M, 99) + 0.1, 250)
    p = lr.predict_proba(g.reshape(-1, 1))[:, 1]

    fig = plt.figure(figsize=(7.3, 3.35))
    ax = fig.add_axes([0.075, 0.18, 0.55, 0.58])
    ax.set_title(r"a  Three probes $\to$ one effective-stress margin $\to$ a predictability ceiling",
                 loc="left", fontweight="bold", fontsize=8.2)
    band = g[(p > 0.3) & (p < 0.7)]
    if len(band):
        ax.axvspan(band.min(), band.max(), color="#9e9e9e", alpha=0.28, zorder=0)
        ax.text(band.mean(), 1.05, "margin boundary\n(FS$\\approx$1; residual floor)", ha="center",
                va="bottom", fontsize=6.4, color="#444")
    rng = np.random.default_rng(0)
    ax.scatter(M[y == 1], 1 - rng.uniform(0, 0.08, int((y == 1).sum())), s=8, c=CML, alpha=0.5,
               edgecolors="none", label="liquefied")
    ax.scatter(M[y == 0], rng.uniform(0, 0.08, int((y == 0).sum())), s=8, c=CMARGIN, alpha=0.5,
               edgecolors="none", label="no liquefaction")
    ax.plot(g, p, color="k", lw=1.9, zorder=5)
    ax.axvline(0, color="k", ls="--", lw=0.9)
    ax.set_xlabel(r"effective-stress margin  $M=\ln(\mathrm{CRR}/\mathrm{CSR})=\ln\,\mathrm{FS}$", fontsize=8)
    ax.set_ylabel("P(liquefaction)", fontsize=8)
    ax.set_ylim(-0.1, 1.16); ax.set_xlim(max(g.min(), -2.0), 7.0)
    ax.legend(loc=(0.62, 0.40), frameon=False, fontsize=6.5, handletextpad=0.2)
    if len(band):
        ax.annotate("transitional soils\nIc 2.05–2.60", xy=(band.mean(), 0.55),
                    xytext=(2.3, 0.74), ha="left", fontsize=6.3, color="#8c510a",
                    arrowprops=dict(arrowstyle="->", color="#8c510a", lw=0.7))
    ax.text(6.8, 0.22, "standard features\nlose resolution", ha="right", fontsize=6.3, color="#333")

    axB = fig.add_axes([0.72, 0.22, 0.26, 0.44])
    axB.set_title("b  over-validation flip", loc="left", fontweight="bold", fontsize=8.2)
    ys = np.arange(len(ARMS))[::-1]
    for yy, (key, lab, resp) in zip(ys, ARMS):
        f = ov[key]["ml_vs_margin_flip"]
        xr = f["under_random_cv_best_ml_minus_margin_AUC"]
        xg = f["under_grouped_cv_best_ml_minus_margin_AUC"]
        axB.annotate("", xy=(xg, yy), xytext=(xr, yy), arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.3))
        axB.scatter([xr], [yy], s=26, color=CRAND, zorder=5)
        axB.scatter([xg], [yy], s=26, color=CGRP, zorder=5)
        axB.text(-0.185, yy, lab, fontsize=7, va="center")
    axB.axvline(0, color="k", ls="--", lw=0.8)
    axB.set_xlabel(r"$\Delta$AUC (ML$-$margin)", fontsize=7); axB.set_yticks([])
    axB.set_xlim(-0.185, 0.06); axB.set_ylim(-0.7, len(ARMS) - 0.2)
    axB.scatter([], [], color=CRAND, label="random CV"); axB.scatter([], [], color=CGRP, label="grouped CV")
    axB.legend(loc="lower left", frameon=False, fontsize=6, handletextpad=0.2)
    save(fig, "fig1_overvalidation_flip")


def fig4_voi():
    vi = load("value_of_information.json")
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    Rsel = "10"
    for (key, lab, resp), col in zip(ARMS, [CMARGIN, "#2c7fb8", "#7a4fa0"]):
        surf = vi[key]["margin_voi_surface"][Rsel]
        axA.plot([d["pi"] for d in surf], [d["voi"] for d in surf], "-o",
                 ms=3, color=col, label=lab)
    axA.set_title("a  Fixed-rule value of the margin (R=10)", loc="left", fontweight="bold")
    axA.set_xlabel(r"prior triggering probability $\pi$")
    axA.set_ylabel("VoI  (units of $C_{FP}$)")
    axA.legend(frameon=False)

    ys = np.arange(len(ARMS))[::-1]
    for y, (key, lab, resp) in zip(ys, ARMS):
        s = vi[key]["extra_predictors_added_value"]["pi=0.2,R=10"]
        d = s["delta_full_minus_margin"]
        axB.errorbar(d, y, xerr=[[d - s["lo"]], [s["hi"] - d]],
                     fmt="o", color="#444444", capsize=2.5, ms=5)
        axB.text(d, y + 0.16, f"{d:+.3f}", ha="center", fontsize=7.5)
    axB.axvline(0, color="k", ls="--", lw=0.8)
    axB.set_title("b  Do extra predictors add decision value?", loc="left", fontweight="bold")
    axB.set_yticks(ys)
    axB.set_yticklabels([f"{l}" for _, l, r in ARMS])
    axB.set_xlabel(r"$\Delta$VoI (full predictors $-$ margin), $\pi$=0.2, R=10")
    axB.set_ylim(-0.5, len(ARMS) - 0.3)
    axB.set_xlim(-0.2, 0.2)
    axB.text(0.10, len(ARMS) - 0.5, "approx. 0\n(no added value)",
             fontsize=7, ha="center", color="#444")
    save(fig, "fig4_value_of_information")


def fig5_geology():
    gr = load("geological_residual.json")
    fig, ax = plt.subplots(figsize=(3.95, 2.8))
    ge_key = "CPT_Geyin2021"
    ra_key = "CPT_Rateria2024_independent_replication"
    rows = [
        (ge_key, "sand-like (Ic<2.05)", "Geyin - sand-like", CMARGIN, 3),
        (ge_key, "transitional (2.05-2.60)", "Geyin - transitional", CML, 2),
        (ra_key, "sand-like (Ic<2.05)", "Rateria - sand-like", CMARGIN, 1),
        (ra_key, "transitional (2.05-2.60)", "Rateria - transitional", CML, 0),
    ]
    yt, ylab = [], []
    for dbkey, skey, lab, col, yy in rows:
        st = gr[dbkey]["by_soil_behaviour_type_Ic"][skey]
        ax.errorbar(st["mean"], yy,
                    xerr=[[st["mean"] - st["lo"]], [st["hi"] - st["mean"]]],
                    fmt="o", color=col, capsize=2.5, ms=5.5)
        yt.append(yy)
        ylab.append(lab)
    for dbkey, yy in [(ge_key, 2.0), (ra_key, 0.0)]:
        c = gr[dbkey]["contrast_transitional_vs_sandlike_Ic"]
        st = gr[dbkey]["by_soil_behaviour_type_Ic"]["transitional (2.05-2.60)"]
        ax.annotate(f"$\\Delta$=+{c['diff']:.3f}*",
                    xy=(st["hi"] + 0.006, yy), fontsize=7.2,
                    va="center", color="#333")
    ax.set_yticks(yt)
    ax.set_yticklabels(ylab, fontsize=7.6)
    ax.set_ylim(-0.6, 3.6)
    ax.invert_yaxis()
    ax.set_xlim(0.15, 0.45)
    ax.set_xlabel(r"label-conditioned residual floor  min$(p,1-p)$")
    ax.set_title("Residual uncertainty is geologically controlled:\nconcentrated in transitional silt-sand soils",
                 fontsize=8.6, fontweight="bold")
    ax.text(0.16, 3.5, "* cluster-bootstrap sign probability >0.99 for both CPT contrasts",
            fontsize=6, color="#777")
    save(fig, "fig5_geological_structure")


def fig1_mechanism():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    ov = load("overvalidation_benchmark.json")
    spt = pd.read_csv(os.path.join(PROC, "cetin2018_baseline_records.csv"))
    spt = spt.dropna(subset=["CSR_cetin", "CRR_BI2014", "y"])
    spt = spt[(spt["CSR_cetin"] > 0) & (spt["CRR_BI2014"] > 0)]
    margin = np.log((spt["CRR_BI2014"] / spt["CSR_cetin"]).to_numpy(float))
    label = spt["y"].astype(int).to_numpy()
    lr = LogisticRegression(max_iter=5000).fit(margin.reshape(-1, 1), label)
    grid = np.linspace(np.percentile(margin, 1) - 0.1,
                       np.percentile(margin, 99) + 0.1, 250)
    prob = lr.predict_proba(grid.reshape(-1, 1))[:, 1]

    fig = plt.figure(figsize=(7.3, 3.35))
    ax = fig.add_axes([0.075, 0.18, 0.55, 0.58])
    ax.set_title(r"a  Three probes $\to$ one effective-stress margin $\to$ residual floor",
                 loc="left", fontweight="bold", fontsize=8.2)
    band = grid[(prob > 0.3) & (prob < 0.7)]
    if len(band):
        ax.axvspan(band.min(), band.max(), color="#9e9e9e", alpha=0.28, zorder=0)
        ax.text(band.mean(), 1.05, "margin boundary\n(FS$\\approx$1; residual floor)",
                ha="center", va="bottom", fontsize=6.4, color="#444")
    rng = np.random.default_rng(0)
    ax.scatter(margin[label == 1], 1 - rng.uniform(0, 0.08, int((label == 1).sum())),
               s=8, c=CML, alpha=0.5, edgecolors="none", label="liquefied")
    ax.scatter(margin[label == 0], rng.uniform(0, 0.08, int((label == 0).sum())),
               s=8, c=CMARGIN, alpha=0.5, edgecolors="none", label="no liquefaction")
    ax.plot(grid, prob, color="k", lw=1.9, zorder=5)
    ax.axvline(0, color="k", ls="--", lw=0.9)
    ax.set_xlabel(r"effective-stress margin  $M=\ln(\mathrm{CRR}/\mathrm{CSR})=\ln\,\mathrm{FS}$",
                  fontsize=8)
    ax.set_ylabel("P(liquefaction)", fontsize=8)
    ax.set_ylim(-0.1, 1.16)
    ax.set_xlim(max(grid.min(), -2.0), 3.2)
    ax.legend(loc=(0.62, 0.40), frameon=False, fontsize=6.5, handletextpad=0.2)
    if len(band):
        ax.annotate("transitional soils\nIc 2.05-2.60", xy=(band.mean(), 0.55),
                    xytext=(1.7, 0.74), ha="left", fontsize=6.3, color="#8c510a",
                    arrowprops=dict(arrowstyle="->", color="#8c510a", lw=0.7))
    ax.text(3.05, 0.22, "standard features\nlose resolution",
            ha="right", fontsize=6.3, color="#333")

    axB = fig.add_axes([0.72, 0.22, 0.26, 0.44])
    axB.set_title("b  event grouping removes apparent gains",
                  loc="left", fontweight="bold", fontsize=7.4)
    ys = np.arange(len(ARMS))[::-1]
    for yy, (key, lab, resp) in zip(ys, ARMS):
        f = ov[key]["ml_vs_margin_flip"]
        xr = f["under_random_cv_best_ml_minus_margin_AUC"]
        xg = f["under_grouped_cv_best_ml_minus_margin_AUC"]
        axB.annotate("", xy=(xg, yy), xytext=(xr, yy),
                     arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.3))
        axB.scatter([xr], [yy], s=26, color=CRAND, zorder=5)
        axB.scatter([xg], [yy], s=26, color=CGRP, zorder=5)
        axB.text(-0.185, yy, lab, fontsize=7, va="center")
    axB.axvline(0, color="k", ls="--", lw=0.8)
    axB.set_xlabel(r"$\Delta$AUC (ML$-$margin)", fontsize=7)
    axB.set_yticks([])
    axB.set_xlim(-0.185, 0.06)
    axB.set_ylim(-0.7, len(ARMS) - 0.2)
    axB.scatter([], [], color=CRAND, label="random CV")
    axB.scatter([], [], color=CGRP, label="grouped CV")
    axB.legend(loc="lower left", frameon=False, fontsize=6, handletextpad=0.2)
    save(fig, "fig1_overvalidation_flip")


if __name__ == "__main__":
    fig1_mechanism(); fig2_ceiling(); fig3_noml(); fig4_voi(); fig5_geology()
    print("\nFigures ->", os.path.abspath(FIGDIR))
