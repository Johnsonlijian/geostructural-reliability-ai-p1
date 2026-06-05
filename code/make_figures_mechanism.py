"""Mechanism schematic + graphical abstract (publication-grade, not bar charts)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from geoliq.mechanics import crr_boulanger_idriss as bi

PROC = os.path.join(BASE, "data", "processed")
FIG = os.path.join(BASE, "..", "figures")
plt.rcParams.update({"font.size": 10})
C_PHYS, C_ML, C_GW, C_CRIT = "#1f3b73", "#c0392b", "#1f77b4", "#f2c200"

spt = pd.read_csv(os.path.join(PROC, "cetin2018_baseline_records.csv"))
spt = spt.dropna(subset=["N1_60cs", "CSR_cetin", "y"])
N1, CSR, yl = spt["N1_60cs"].to_numpy(float), spt["CSR_cetin"].to_numpy(float), spt["y"].to_numpy(int)
J = json.load(open(os.path.join(PROC, "innovation_analysis.json"), encoding="utf-8"))

# ============ FIG 1: mechanism (3 panels) ============
fig = plt.figure(figsize=(13, 4.7))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.15, 1.05], wspace=0.25)

# (a) effective-stress mechanism + groundwater
axa = fig.add_subplot(gs[0, 0]); axa.axis("off"); axa.set_xlim(0, 1); axa.set_ylim(0, 1)
axa.add_patch(Rectangle((0.16, 0.08), 0.52, 0.74, fc="#e8dcc0", ec="k", zorder=1))
axa.add_patch(Rectangle((0.16, 0.30), 0.52, 0.15, fc=C_CRIT, ec="k", alpha=0.85, zorder=2))
axa.text(0.42, 0.375, "liquefiable\nlayer", ha="center", va="center", fontsize=8)
axa.plot([0.16, 0.68], [0.60, 0.60], color=C_GW, lw=1.6)
axa.add_patch(Polygon([[0.40, 0.60], [0.435, 0.655], [0.365, 0.655]], closed=True, fc=C_GW))
axa.text(0.70, 0.605, "GWT", color=C_GW, fontsize=8)
xx = np.linspace(0.16, 0.68, 120)
axa.plot(xx, 0.90 + 0.028 * np.sin((xx - 0.16) * 42), color=C_ML, lw=1.6)
axa.annotate("", xy=(0.72, 0.90), xytext=(0.12, 0.90), arrowprops=dict(arrowstyle="<->", color=C_ML))
axa.text(0.42, 0.955, "cyclic shaking", ha="center", color=C_ML, fontsize=8.5)
axa.annotate("", xy=(0.28, 0.40), xytext=(0.28, 0.80), arrowprops=dict(arrowstyle="->", color="k", lw=1.4))
axa.text(0.295, 0.66, r"$\sigma_v$", fontsize=11)
axa.annotate("", xy=(0.56, 0.56), xytext=(0.56, 0.40), arrowprops=dict(arrowstyle="->", color=C_GW, lw=1.4))
axa.text(0.575, 0.47, r"$u$", color=C_GW, fontsize=11)
axa.text(0.42, 0.155, r"$\sigma'_v=\sigma_v-u$", ha="center", fontsize=10.5,
         bbox=dict(boxstyle="round", fc="white", ec="k"))
axa.set_title("(a) Effective-stress mechanism;\ngroundwater is the hidden state", fontsize=10)

# (b) BI2014 boundary + real cases
axb = fig.add_subplot(gs[0, 1])
nn = np.linspace(1, 40, 240); crr = bi.crr75_spt(nn)
axb.fill_between(nn, crr, 0.6, color="#f6c6c6", alpha=0.55)
axb.fill_between(nn, 0, crr, color="#cfe8cf", alpha=0.65)
axb.plot(nn, crr, color=C_PHYS, lw=2.6, label="BI2014 boundary (FS=1)")
m = yl == 1
axb.scatter(N1[m], CSR[m], facecolors=C_ML, edgecolors="k", s=24, lw=0.4, label="liquefied", zorder=5)
axb.scatter(N1[~m], CSR[~m], facecolors="none", edgecolors=C_PHYS, s=24, lw=0.9, label="no liquefaction", zorder=5)
axb.set_xlim(0, 40); axb.set_ylim(0, 0.6)
axb.set_xlabel(r"clean-sand penetration resistance $(N_1)_{60,cs}$"); axb.set_ylabel("cyclic stress ratio, CSR")
axb.text(33, 0.52, "LIQUEFY", color=C_ML, fontsize=9, ha="center", weight="bold")
axb.text(33, 0.05, "STABLE", color="#2e7d32", fontsize=9, ha="center", weight="bold")
axb.set_title("(b) Triggering chart: real SPT cases\nagainst the mechanistic boundary", fontsize=10)
axb.legend(fontsize=7.8, loc="upper left", framealpha=0.95)

# (c) information bottleneck
axc = fig.add_subplot(gs[0, 2]); axc.axis("off"); axc.set_xlim(0, 1); axc.set_ylim(0, 1)
feats = ["$M_w$", "$a_{max}$", r"$\sigma_v$", r"$\sigma'_v$", "FC", "$(N_1)_{60}$", "depth", "GWT"]
for i, f in enumerate(feats):
    yy = 0.90 - i * 0.095
    axc.add_patch(FancyBboxPatch((0.02, yy - 0.033), 0.19, 0.066, boxstyle="round,pad=0.008", fc="#eef2f7", ec="#7f8c9b"))
    axc.text(0.115, yy, f, ha="center", va="center", fontsize=8)
    axc.annotate("", xy=(0.44, 0.5), xytext=(0.21, yy), arrowprops=dict(arrowstyle="-", color="#cccccc", lw=0.7))
axc.add_patch(FancyBboxPatch((0.44, 0.40), 0.28, 0.20, boxstyle="round,pad=0.02", fc=C_PHYS, ec="k"))
axc.text(0.58, 0.50, "effective-stress\nmargin\n" + r"$\ln(\mathrm{CSR}/\mathrm{CRR})$", ha="center", va="center", color="white", fontsize=8.5)
axc.annotate("", xy=(0.92, 0.5), xytext=(0.72, 0.5), arrowprops=dict(arrowstyle="->", lw=2.2))
axc.add_patch(FancyBboxPatch((0.80, 0.42), 0.17, 0.16, boxstyle="round,pad=0.02", fc=C_CRIT, ec="k"))
axc.text(0.885, 0.50, "liquefy?", ha="center", va="center", fontsize=8.5)
axc.text(0.5, 0.93, "(c) Effective-stress coordinate:\ntested features add little grouped-transfer gain", ha="center", fontsize=9.5)
axc.text(0.5, 0.14, "ambiguity proxy peaks at FS$\\approx$1", ha="center", fontsize=8.3, style="italic", color="#555")
fig.savefig(os.path.join(FIG, "fig1_mechanism.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

# ============ FIG 0: graphical abstract ============
fig = plt.figure(figsize=(12, 4.2))
gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 1, 1, 1], wspace=0.33)
fig.suptitle("The effective-stress margin bounds tested transferable ML gain in seismic liquefaction:\n"
             "full-feature ML does not exceed it under grouped validation, with ambiguity concentrated near the critical state",
             fontsize=11, y=1.06)
# left: pipeline
ax0 = fig.add_subplot(gs[0, 0]); ax0.axis("off"); ax0.set_xlim(0, 1); ax0.set_ylim(0, 1)
for (yy, t, c) in [(0.78, "hazard\n$M_w,\\ a_{max}$", C_ML), (0.50, "groundwater\n$\\Rightarrow\\ \\sigma'_v$", C_GW), (0.22, "margin\n$\\ln(\\mathrm{CSR}/\\mathrm{CRR})$", C_PHYS)]:
    ax0.add_patch(FancyBboxPatch((0.2, yy - 0.1), 0.6, 0.2, boxstyle="round,pad=0.02", fc=c, ec="k"))
    ax0.text(0.5, yy, t, ha="center", va="center", color="white", fontsize=8.5)
ax0.annotate("", xy=(0.5, 0.60), xytext=(0.5, 0.68), arrowprops=dict(arrowstyle="->", lw=1.6))
ax0.annotate("", xy=(0.5, 0.32), xytext=(0.5, 0.40), arrowprops=dict(arrowstyle="->", lw=1.6))
ax0.set_title("mechanism", fontsize=9.5)
# 1 grouped-transfer comparison (CPT, larger gap)
a1 = fig.add_subplot(gs[0, 1])
a1.bar([0, 1], [J["CPT/Geyin2021"]["A1_sufficiency"]["auc_full_feature_ML"], J["CPT/Geyin2021"]["A1_sufficiency"]["auc_margin_only_ML"]],
       color=[C_ML, C_PHYS], edgecolor="k", width=0.6)
a1.set_xticks([0, 1]); a1.set_xticklabels(["full\nML", "margin"], fontsize=8.5); a1.set_ylim(0.5, 0.8); a1.set_ylabel("OOD AUC", fontsize=9)
a1.set_title("1) grouped transfer\nmargin >= ML", fontsize=9.5)
# 2 ambiguity proxy (SPT bands)
a2 = fig.add_subplot(gs[0, 2])
bands = J["SPT/Cetin2018"]["A3_irreducible"]["bands"]
a2.axvspan(0.3, 0.7, color=C_CRIT, alpha=0.2)
a2.bar([b["p_mean"] for b in bands], [b["bayes_err"] for b in bands], width=0.08, color="#999", edgecolor="k")
a2.set_xlim(0, 1); a2.set_ylim(0, 0.55); a2.set_xlabel("margin (P)", fontsize=9); a2.set_ylabel("ambiguity proxy", fontsize=9)
a2.set_title("2) ambiguity\npeaks at FS$\\approx$1", fontsize=9.5)
# 3 reliability (SPT mid band plain vs mondrian)
a3 = fig.add_subplot(gs[0, 3])
pl = J["SPT/Cetin2018"]["A4_mondrian_band_coverage"]["plain"]
mo = J["SPT/Cetin2018"]["A4_mondrian_band_coverage"]["mondrian"]
bb = ["0", "1", "2"]; xb = np.arange(3); w = 0.38
a3.bar(xb - w / 2, [pl[k] for k in bb], w, color="#bbb", edgecolor="k", label="plain")
a3.bar(xb + w / 2, [mo[k] for k in bb], w, color=C_PHYS, edgecolor="k", label="Mondrian")
a3.axhline(0.9, color=C_ML, ls="--", lw=1.5, label="target")
a3.set_xticks(xb); a3.set_xticklabels(["safe", "FS$\\approx$1", "liq."], fontsize=8)
a3.set_ylim(0.8, 1.02); a3.set_ylabel("band coverage", fontsize=9)
a3.legend(fontsize=6.3, loc="lower center", ncol=1); a3.set_title("3) reliability\nmade uniform", fontsize=9.5)
fig.savefig(os.path.join(FIG, "fig0_graphical_abstract.png"), dpi=300, bbox_inches="tight"); plt.close(fig)
print("saved fig0_graphical_abstract.png, fig1_mechanism.png")
