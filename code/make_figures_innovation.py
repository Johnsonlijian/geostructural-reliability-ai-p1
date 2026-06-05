"""Legacy innovation figures: grouped-transfer bound, ambiguity proxy, Mondrian coverage."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, "data", "processed")
FIG = os.path.join(BASE, "..", "figures")
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
C_PHYS, C_ML, C_MARG = "#1f3b73", "#c0392b", "#2e8b57"
J = json.load(open(os.path.join(PROC, "innovation_analysis.json"), encoding="utf-8"))
names = list(J.keys())

# ---- fig7 grouped-transfer bound ----
fig, ax = plt.subplots(figsize=(7.2, 4.3))
x = np.arange(len(names)); w = 0.26
full = [J[n]["A1_sufficiency"]["auc_full_feature_ML"] for n in names]
marg = [J[n]["A1_sufficiency"]["auc_margin_only_ML"] for n in names]
phys = [J[n]["A1_sufficiency"]["auc_physics_raw"] for n in names]
ax.bar(x - w, full, w, label="full-feature ML", color=C_ML, edgecolor="k")
ax.bar(x, marg, w, label="margin-only ML", color=C_MARG, edgecolor="k")
ax.bar(x + w, phys, w, label="raw mechanistic margin", color=C_PHYS, edgecolor="k")
for i in range(len(names)):
    for off, v in [(-w, full[i]), (0, marg[i]), (w, phys[i])]:
        ax.text(x[i] + off, v + 0.008, f"{v:.3f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([n.split("/")[0] + "\n" + n.split("/")[1] for n in names])
ax.set_ylabel("out-of-distribution ROC-AUC (earthquake-grouped)")
ax.set_ylim(0.5, 1.0); ax.legend(loc="lower right", fontsize=9)
ax.set_title("Grouped-transfer check: the margin is the strongest tested coordinate")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig7_sufficiency.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

# ---- fig8 ambiguity proxy ----
fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.3))
for ax, n in zip(axs, names):
    bands = J[n]["A3_irreducible"]["bands"]
    pm = [b["p_mean"] for b in bands]; fr = [b["liq_freq"] for b in bands]; be = [b["bayes_err"] for b in bands]
    ax.axvspan(0.3, 0.7, color="#f2c200", alpha=0.18, label="critical-state band")
    ax.bar(pm, be, width=0.07, color="#aaaaaa", edgecolor="k", label="coordinate ambiguity proxy")
    ax.plot(pm, fr, "-o", color=C_PHYS, lw=2, label="observed liquefaction freq.")
    ax.plot([0, 1], [0.5, 0.5], ":", color="gray", lw=1)
    a3 = J[n]["A3_irreducible"]
    ax.set_title(f"{n}\nML err {a3['ML_full_error']} vs ambiguity proxy {a3['bayes_error_lower_bound']}; {int(a3['frac_bayes_err_in_ambiguous_band']*100)}% near FS~1", fontsize=9)
    ax.set_xlabel("mechanistic margin (calibrated P)"); ax.set_ylabel("frequency / error"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5, loc="upper left")
fig.suptitle("Coordinate ambiguity is localized at the critical state FS~1", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig8_irreducibility.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

# ---- fig9 Mondrian ----
fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.3))
lab = {0: "low (safe)", 1: "mid (FS~1)", 2: "high (liquefy)"}
for ax, n in zip(axs, names):
    a4 = J[n]["A4_mondrian_band_coverage"]
    bb = sorted(set(int(k) for k in a4["plain"]) & set(int(k) for k in a4["mondrian"]))
    xx = np.arange(len(bb)); w = 0.36
    pl = [a4["plain"][str(b)] if str(b) in a4["plain"] else a4["plain"][b] for b in bb]
    mo = [a4["mondrian"][str(b)] if str(b) in a4["mondrian"] else a4["mondrian"][b] for b in bb]
    ax.bar(xx - w / 2, pl, w, label="plain conformal", color="#bbbbbb", edgecolor="k")
    ax.bar(xx + w / 2, mo, w, label="mechanism-conditioned (Mondrian)", color=C_PHYS, edgecolor="k")
    ax.axhline(0.9, color=C_ML, ls="--", lw=2, label="target 0.90")
    ax.set_xticks(xx); ax.set_xticklabels([lab[b] for b in bb], fontsize=8.5)
    ax.set_ylim(0.7, 1.02); ax.set_ylabel("band-conditional coverage"); ax.set_title(n, fontsize=10); ax.legend(fontsize=8, loc="lower center")
fig.suptitle("Mechanism-conditioned conformal restores coverage where it matters (the FS~1 band)", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig9_mondrian.png"), dpi=300, bbox_inches="tight"); plt.close(fig)
print("saved fig7_sufficiency.png, fig8_irreducibility.png, fig9_mondrian.png")
