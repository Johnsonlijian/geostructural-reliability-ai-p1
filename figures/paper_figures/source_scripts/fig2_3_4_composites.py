"""Generate composite main figures 2-4 for P1.

The figures are data-driven from code/data/processed JSON/CSV outputs and
exported as SVG/PDF/PNG previews.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


SCRIPT = Path(__file__).resolve()
PROJECT = SCRIPT.parents[3]
FIG_ROOT = SCRIPT.parents[1]
DATA = PROJECT / "code" / "data" / "processed"
OUT = FIG_ROOT / "output"

BLUE = "#2f80ed"
RED = "#d1495b"
GREEN = "#27ae60"
PURPLE = "#6f4e9b"
AMBER = "#f2c94c"
GRAY = "#9aa0a6"
DARK = "#1f2933"


def load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def ensure_dirs() -> None:
    for sub in ["svg", "pdf", "png"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)


def setup():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 320,
        }
    )


def label(ax, letter: str, title: str):
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", ha="left", va="bottom")
    ax.text(0.035, 1.04, title, transform=ax.transAxes, fontsize=10.4, fontweight="bold", ha="left", va="bottom")


def save(fig, stem: str):
    for ext in ["svg", "pdf", "png"]:
        path = OUT / ext / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"wrote {path}")


def panel_validation(ax, rel, random_sens):
    label(ax, "A", "Grouped validation defines the measured bound")
    datasets = ["SPT triggering", "CPT manifestation"]
    xbase = np.arange(2)
    spt_rand = random_sens["datasets"]["SPT_Cetin2018"]["models"]["hist_gbt"]
    cpt_rand = random_sens["datasets"]["CPT_Geyin2021"]["models"]["hist_gbt"]
    vals = {
        "mechanism": [rel["SPT_Cetin2018"]["physics_auc_ci"]["auc"], rel["CPT_Geyin2021"]["physics_auc_ci"]["auc"]],
        "random ML": [
            spt_rand["random_auc"]["median"],
            cpt_rand["random_auc"]["median"],
        ],
        "grouped ML": [
            spt_rand["grouped_auc_reference"],
            cpt_rand["grouped_auc_reference"],
        ],
    }
    markers = {"mechanism": "o", "random ML": "s", "grouped ML": "^"}
    colors = {"mechanism": BLUE, "random ML": GRAY, "grouped ML": RED}
    for i, ds in enumerate(datasets):
        ax.plot([i - 0.16, i + 0.16], [vals["random ML"][i], vals["grouped ML"][i]], color="#6b7280", lw=1.2, alpha=0.8)
        for j, key in enumerate(["mechanism", "random ML", "grouped ML"]):
            ax.scatter(i + [-0.26, -0.05, 0.16][j], vals[key][i], s=58, marker=markers[key], color=colors[key], edgecolor=DARK, zorder=3, label=key if i == 0 else None)
    ax.set_xticks(xbase)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.56, 0.96)
    ax.grid(axis="y", color="#e5e7eb")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.text(0.52, 0.91, "100-repeat optimism gap\n+0.047 / +0.117", transform=ax.transAxes, fontsize=8.5, color="#374151")


def panel_logloss(ax, suff):
    label(ax, "B", "Full-feature gains are not supported")
    models = ["logistic", "random_forest", "grad_boost", "hist_gbt", "neural_net"]
    y_positions = np.arange(len(models) * 2)
    rows = []
    for ds, short, color in [("SPT/Cetin2018", "SPT", BLUE), ("CPT/Geyin2021", "CPT", PURPLE)]:
        for m in models:
            r = suff[ds][m]
            rows.append((f"{short} {m.replace('_', ' ')}", r["dlogloss_full_minus_margin"], r["CI95"][0], r["CI95"][1], color))
    for y, (name, mean, lo, hi, color) in zip(y_positions, rows):
        ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]], fmt="o", color=color, ecolor=color, capsize=2.5, ms=4.5)
    ax.axvline(0, color="#111827", lw=1)
    ax.axvline(-0.10, color="#c0392b", lw=1, ls="--")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.2)
    ax.invert_yaxis()
    ax.set_xlabel("Delta log-loss (candidate - margin; <0 favors candidate)")
    xmin = min(-0.13, min(r[2] for r in rows) - 0.03)
    xmax = max(r[3] for r in rows) + 0.08
    ax.set_xlim(xmin, xmax)
    ax.grid(axis="x", color="#e5e7eb")


def panel_residual(ax, residual):
    label(ax, "C", "Residual and groundwater terms test what the margin leaves behind")
    pick = [
        ("SPT", "standard_public_features::logistic", BLUE),
        ("SPT", "raw_no_explicit_effstress_or_gwt::logistic", BLUE),
        ("SPT", "groundwater_only_after_margin::logistic", BLUE),
        ("CPT", "rawish_no_LPI_or_FS::logistic", PURPLE),
        ("CPT", "rawish_no_LPI_FS_or_GWT::logistic", PURPLE),
        ("CPT", "groundwater_only_after_margin::logistic", PURPLE),
    ]
    source = {"SPT": "SPT_Cetin2018", "CPT": "CPT_Geyin2021"}
    labels = {
        "standard_public_features::logistic": "standard features",
        "raw_no_explicit_effstress_or_gwt::logistic": "raw no effstress/GWT",
        "groundwater_only_after_margin::logistic": "GWT only",
        "rawish_no_LPI_or_FS::logistic": "raw no LPI/FS",
        "rawish_no_LPI_FS_or_GWT::logistic": "raw no LPI/FS/GWT",
    }
    for y, (ds, key, color) in enumerate(pick):
        ci = residual[source[ds]]["variants"][key]["dlogloss_variant_minus_margin"]
        mean, lo, hi = ci["mean"], ci["lo"], ci["hi"]
        ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]], fmt="D", ms=4.5, color=color, ecolor=color, capsize=3)
    ax.axvline(0, color="#111827", lw=1)
    ax.axvline(-0.10, color="#c0392b", lw=1, ls="--")
    ax.set_yticks(range(len(pick)))
    ax.set_yticklabels([f"{ds}: {labels.get(key, key)}" for ds, key, _ in pick], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Delta log-loss (candidate - margin; <0 favors candidate)")
    ax.set_xlim(-0.13, 0.50)
    ax.grid(axis="x", color="#e5e7eb")
    ax.text(0.54, 0.08, "No CI is entirely below zero", transform=ax.transAxes, fontsize=8.4, color="#374151")


def panel_practical_gain(ax, practical):
    label(ax, "D", "Excluded-gain audit defines the bound")
    rows = [
        ("SPT", practical["summary"]["SPT_Cetin2018"]),
        ("CPT", practical["summary"]["CPT_Geyin2021"]),
    ]
    cols = [
        ("AUC gain\n> 0.02", "excludes_full_AUC_gain_gt_0.02"),
        ("full log-loss\ngain > 0.10", "excludes_model_logloss_gain_gt_0.10"),
        ("residual log-loss\ngain > 0.10", "excludes_residual_variant_logloss_gain_gt_0.10"),
    ]
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels([c[0] for c in cols], fontsize=8)
    ax.set_yticks([1.5, 0.5])
    ax.set_yticklabels([r[0] for r in rows], fontsize=9, fontweight="bold")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, (_, summary) in enumerate(rows):
        y = 1 - i
        for j, (_, key) in enumerate(cols):
            passed = summary[key]
            face = GREEN if passed else AMBER
            box = FancyBboxPatch((j + 0.06, y + 0.10), 0.88, 0.78, boxstyle="round,pad=0.01,rounding_size=0.025", facecolor=face, edgecolor=DARK, lw=1.1)
            ax.add_patch(box)
            ax.text(
                j + 0.50,
                y + (0.55 if not passed else 0.50),
                "excluded" if passed else "not excluded",
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color="#0f172a",
            )
            if not passed:
                ax.text(j + 0.50, y + 0.32, "SPT residual caveat", ha="center", va="center", fontsize=7.5, color="#7b5a00")


def fig2():
    rel = load("reliability_upgrade.json")
    random_sens = load("random_split_sensitivity.json")
    suff = load("sufficiency_likelihood.json")
    residual = load("residual_sufficiency_audit.json")
    practical = load("practical_equivalence_audit.json")
    fig, axs = plt.subplots(2, 2, figsize=(13.2, 8.7))
    panel_validation(axs[0, 0], rel, random_sens)
    panel_logloss(axs[0, 1], suff)
    panel_residual(axs[1, 0], residual)
    panel_practical_gain(axs[1, 1], practical)
    fig.suptitle("Measured bound and excluded-gain audit beyond the effective-stress margin", fontsize=15, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    save(fig, "Fig2_transfer_sufficiency")


def panel_plain_failure(ax, innov):
    label(ax, "A", "Plain marginal coverage hides regime-level failure")
    bands = ["safe", "critical", "high"]
    x = np.arange(3)
    w = 0.26
    for ds, color, off in [("SPT/Cetin2018", BLUE, -w / 2), ("CPT/Geyin2021", PURPLE, w / 2)]:
        plain = innov[ds]["A4_mondrian_band_coverage"]["plain"]
        vals = [plain[str(j)] for j in range(3)]
        ax.bar(x + off, vals, width=w, color=color, edgecolor=DARK, alpha=0.88, label=f"{ds.split('/')[0]} plain")
        for xi, val in zip(x + off, vals):
            ax.text(xi, val + 0.006, f"{val:.3f}", ha="center", fontsize=7.2, color="#374151")
    ax.axhline(0.90, color="#c0392b", ls="--", lw=1.2)
    ax.text(2.08, 0.902, "target 0.90", fontsize=8, color="#7b241c")
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_ylabel("plain conformal coverage")
    ax.set_ylim(0.84, 1.02)
    ax.grid(axis="y", color="#e5e7eb")
    ax.legend(frameon=False, fontsize=8, loc="lower center", ncol=2)
    ax.text(0.02, 0.08, "Marginal validity does not imply regime reliability.", transform=ax.transAxes, fontsize=8.2, color="#374151")


def panel_ambiguity(ax, innov):
    label(ax, "B", "Critical-state band locates the remaining ambiguity")
    for ds, color, marker in [("SPT/Cetin2018", BLUE, "o"), ("CPT/Geyin2021", PURPLE, "s")]:
        rows = innov[ds]["A3_irreducible"]["bands"]
        p = [r["p_mean"] for r in rows]
        freq = [r["liq_freq"] for r in rows]
        sizes = [max(35, r["n"] * 2.0) for r in rows]
        ax.scatter(p, freq, s=sizes, color=color, marker=marker, alpha=0.82, edgecolor=DARK, label=ds.split("/")[0])
    ax.axvspan(0.3, 0.7, color=AMBER, alpha=0.22)
    ax.plot([0, 1], [0, 1], color="#9ca3af", lw=1, ls="--")
    ax.set_xlabel("mechanistic probability")
    ax.set_ylabel("observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color="#e5e7eb")
    ax.text(0.32, 0.08, "82% / 52% of coordinate ambiguity\nlies in the critical band", fontsize=8.2, color="#7b5a00")


def panel_reliability(ax, rel):
    label(ax, "B", "Miscalibration and per-earthquake undercoverage")
    ds = ["SPT", "CPT"]
    keys = ["SPT_Cetin2018", "CPT_Geyin2021"]
    x = np.arange(2)
    w = 0.28
    phys_ece = [rel[k]["calibration"]["physics"]["ece"] for k in keys]
    ml_ece = [rel[k]["calibration"]["ml_grouped"]["ece"] for k in keys]
    under = [rel[k]["conformal_conditional_grouped"]["frac_events_undercovered"] for k in keys]
    ax.bar(x - w / 2, phys_ece, width=w, color=BLUE, edgecolor=DARK, label="mechanism ECE")
    ax.bar(x + w / 2, ml_ece, width=w, color=RED, edgecolor=DARK, label="ML ECE")
    ax.set_xticks(x)
    ax.set_xticklabels(ds)
    ax.set_ylabel("expected calibration error")
    ax.set_ylim(0, max(ml_ece + phys_ece) * 1.45)
    ax.grid(axis="y", color="#e5e7eb")
    ax2 = ax.twinx()
    ax2.plot(x, under, marker="^", color="#111827", lw=1.8, label="undercovered events")
    ax2.set_ylim(0, 0.50)
    ax2.set_ylabel("fraction of events undercovered")
    lines, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels1 + labels2, frameon=False, fontsize=7.7, loc="upper left")


def panel_mondrian(ax, innov):
    label(ax, "C", "Mechanism-band conformal repairs regime coverage")
    bands = ["safe", "critical", "high"]
    x = np.arange(3)
    w = 0.18
    for i, (ds, color, off) in enumerate([("SPT/Cetin2018", BLUE, -0.27), ("CPT/Geyin2021", PURPLE, 0.05)]):
        plain = innov[ds]["A4_mondrian_band_coverage"]["plain"]
        mon = innov[ds]["A4_mondrian_band_coverage"]["mondrian"]
        ax.plot(x + off, [plain[str(j)] for j in range(3)], color=color, marker="o", ls="--", lw=1.4, label=f"{ds.split('/')[0]} plain")
        ax.plot(x + off + w, [mon[str(j)] for j in range(3)], color=color, marker="s", lw=2.1, label=f"{ds.split('/')[0]} band")
    ax.axhline(0.90, color="#c0392b", ls="--", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_ylabel("band coverage")
    ax.set_ylim(0.84, 1.01)
    ax.grid(axis="y", color="#e5e7eb")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower center")


def panel_decision_sets(ax, decision):
    label(ax, "D", "Prediction sets flag critical-band decisions")
    labels = ["SPT", "CPT"]
    keys = ["SPT_Cetin2018", "CPT_Geyin2021"]
    x = np.arange(2)
    w = 0.22
    singleton = []
    two_label = []
    critical_two = []
    for key in keys:
        a = decision["datasets"][key]["alpha"]["0.10"]
        singleton.append(a["mechanism_band"]["singleton_rate"])
        two_label.append(a["mechanism_band"]["two_label_rate"])
        critical_two.append(a["mechanism_band_by_band"]["1"]["two_label_rate"])
    ax.bar(x - w, singleton, width=w, color=BLUE, edgecolor=DARK, label="singleton decision")
    ax.bar(x, two_label, width=w, color=GRAY, edgecolor=DARK, label="two-label uncertain")
    ax.bar(x + w, critical_two, width=w, color=AMBER, edgecolor=DARK, label="critical-band uncertain")
    for xs, vals in [(x, two_label), (x + w, critical_two)]:
        for xi, val in zip(xs, vals):
            ax.text(xi, val + 0.025, f"{val*100:.1f}%", ha="center", fontsize=7.8, color="#374151")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("prediction-set rate")
    ax.grid(axis="y", color="#e5e7eb")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")


def fig3():
    innov = load("innovation_analysis.json")
    rel = load("reliability_upgrade.json")
    ambsens = load("ambiguity_floor_sensitivity.json")
    decision = load("conformal_decision_metrics.json")
    fig, axs = plt.subplots(2, 2, figsize=(13.2, 8.7))
    panel_plain_failure(axs[0, 0], innov)
    panel_ambiguity(axs[0, 1], innov)
    panel_mondrian(axs[1, 0], innov)
    panel_decision_sets(axs[1, 1], decision)
    fig.suptitle("Critical-state decision reliability", fontsize=15, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    save(fig, "Fig3_ambiguity_reliability")


def round_box(ax, xy, text, fc):
    box = FancyBboxPatch(xy, 0.22, 0.16, boxstyle="round,pad=0.02,rounding_size=0.03", facecolor=fc, edgecolor=DARK, lw=1.2)
    ax.add_patch(box)
    ax.text(xy[0] + 0.11, xy[1] + 0.08, text, ha="center", va="center", fontsize=9)
    return box


def arr(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, lw=1.6, color="#374151"))


def panel_gw_path(ax):
    label(ax, "A", "Groundwater changes the margin through effective stress")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    round_box(ax, (0.05, 0.64), "water table\ndepth", "#9dc3e6")
    round_box(ax, (0.38, 0.64), "effective\nstress", "#c6e0b4")
    round_box(ax, (0.70, 0.64), "CSR/CRR\nmargin", "#f7f3d7")
    round_box(ax, (0.38, 0.28), "residual\ncheck", "#f4cccc")
    arr(ax, (0.27, 0.72), (0.38, 0.72))
    arr(ax, (0.60, 0.72), (0.70, 0.72))
    arr(ax, (0.81, 0.64), (0.52, 0.44))
    ax.text(0.50, 0.12, "Claim boundary: residual discrimination, not full causal identification.", ha="center", fontsize=8.6, color="#374151")


def panel_counterfactual(ax, gw):
    label(ax, "B", "Water-table counterfactual strongly shifts FS and risk")
    cf = gw["counterfactual_water_table"]
    names = ["surface", "actual", "deep/dry"]
    keys = ["surface(dw=0)", "actual", "deep/dry(dw=z)"]
    fs = [cf[k]["median_FS"] for k in keys]
    rate = [cf[k]["pred_liq_rate_FS<1"] for k in keys]
    x = np.arange(3)
    ax.plot(x, fs, color=BLUE, marker="o", lw=2.2, label="median FS")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("median FS")
    ax.axhline(1, color="#111827", lw=1, ls="--")
    ax2 = ax.twinx()
    ax2.plot(x, rate, color=RED, marker="s", lw=2.2, label="predicted liquefaction rate")
    ax2.set_ylabel("pred. rate FS<1")
    ax.set_ylim(0.45, 1.38)
    ax2.set_ylim(0.20, 0.82)
    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, frameon=False, fontsize=8, loc="center right")
    ax.grid(axis="y", color="#e5e7eb")


def panel_gw_resid(ax, rows, title, color):
    label(ax, title[0], title[1])
    labels = [r["groundwater_bin"] for r in rows]
    x = np.arange(len(rows))
    means = [r["margin_residual_y_minus_p"]["mean"] for r in rows]
    lo = [m - r["margin_residual_y_minus_p"]["lo"] for m, r in zip(means, rows)]
    hi = [r["margin_residual_y_minus_p"]["hi"] - m for m, r in zip(means, rows)]
    ax.errorbar(x, means, yerr=[lo, hi], fmt="o", color=color, ecolor=color, capsize=3, lw=1.6)
    ax.axhline(0, color="#111827", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean residual y - p")
    ax.set_ylim(-0.24, 0.28)
    ax.grid(axis="y", color="#e5e7eb")
    for i, r in enumerate(rows):
        ax.text(i, -0.225, f"n={r['n']}", ha="center", fontsize=7.5, color="#374151")


def fig4():
    gw = load("cetin2018_groundwater_ablation.json")
    strat = load("groundwater_residual_stratification.json")
    fig, axs = plt.subplots(2, 2, figsize=(13.2, 8.5))
    panel_gw_path(axs[0, 0])
    panel_counterfactual(axs[0, 1], gw)
    panel_gw_resid(axs[1, 0], strat["SPT_Cetin2018"]["bins"], ("C", "SPT residuals by water-depth bin"), BLUE)
    panel_gw_resid(axs[1, 1], strat["CPT_Geyin2021"]["bins"], ("D", "CPT residuals by groundwater-depth bin"), PURPLE)
    fig.suptitle("Groundwater mechanism and residual boundary", fontsize=15, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    save(fig, "Fig4_groundwater_boundary")


def main():
    ensure_dirs()
    setup()
    fig2()
    fig3()
    fig4()


if __name__ == "__main__":
    main()
