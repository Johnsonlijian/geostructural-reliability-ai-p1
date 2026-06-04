"""fig10: cross-population (leave-one-region-out) transfer, dumbbell plot."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, "data", "processed")
FIG = os.path.join(BASE, "..", "figures")
C_PHYS, C_ML = "#1f3b73", "#c0392b"
J = json.load(open(os.path.join(PROC, "cross_region_transfer.json"), encoding="utf-8"))

rows = [r for r in J["by_region"] if "physics_AUC" in r]
rows = sorted(rows, key=lambda r: -r["n"])
labels = [f"{r['region']}\n(n={r['n']})" for r in rows] + ["POOLED\n(n=208)"]
phys = [r["physics_AUC"] for r in rows] + [J["pooled"]["physics_pooled_AUC"]]
ml = [r["ML_trained_elsewhere_AUC"] for r in rows] + [J["pooled"]["ML_crossregion_pooled_AUC"]]
x = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(8.4, 4.6))
for xi, p, m in zip(x, phys, ml):
    ax.plot([xi, xi], [m, p], color="#9aa3ad", lw=2, zorder=1)
ax.scatter(x, phys, s=110, color=C_PHYS, zorder=3, label="zero-shot physics margin (not trained)")
ax.scatter(x, ml, s=90, facecolors="white", edgecolors=C_ML, linewidths=2, zorder=3, label="ML trained on the OTHER regions")
for xi, p, m in zip(x, phys, ml):
    ax.text(xi, p + 0.008, f"{p:.2f}", ha="center", color=C_PHYS, fontsize=8.5)
    ax.text(xi, m - 0.022, f"{m:.2f}", ha="center", color=C_ML, fontsize=8.5)
ax.axvline(len(x) - 1.5, color="#ccc", ls=":", lw=1)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("ROC-AUC on the held-out population"); ax.set_ylim(0.74, 1.02)
ax.set_title("Cross-population transfer (leave-one-region-out, all SPT):\nthe mechanism transfers; ML trained on other regions does not")
ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig10_cross_region.png"), dpi=300, bbox_inches="tight"); plt.close(fig)
print("saved fig10_cross_region.png")
