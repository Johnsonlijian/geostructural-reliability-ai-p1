"""Generate Fig. 1 central mechanism-evidence synthesis for P1.

Outputs SVG/PDF/PNG from processed JSON files. The schematic panel is conceptual;
data panels trace to code/data/processed outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


SCRIPT = Path(__file__).resolve()
PROJECT = SCRIPT.parents[3]
FIG_ROOT = SCRIPT.parents[1]
DATA = PROJECT / "code" / "data" / "processed"
OUT = FIG_ROOT / "output"


def load_json(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def ensure_dirs() -> None:
    for sub in ["svg", "pdf", "png"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)


def add_panel_label(ax, label: str, title: str) -> None:
    ax.text(
        -0.018,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
    )
    ax.text(
        0.035,
        1.04,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )


def rounded(ax, xy, width, height, text, fc, ec="#1f2933", fs=9):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.2,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color="#102027",
        wrap=True,
    )
    return box


def arrow(ax, start, end, color="#34495e", lw=1.7):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=lw,
            color=color,
            shrinkA=4,
            shrinkB=4,
        )
    )


def panel_a(ax) -> None:
    add_panel_label(ax, "A", "Mechanism: groundwater is absorbed through effective stress")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(Rectangle((0.02, 0.08), 0.96, 0.55, facecolor="#d9c29c", edgecolor="#6b5d45", lw=1.0))
    ax.add_patch(Rectangle((0.02, 0.55), 0.96, 0.08, facecolor="#9cc7de", edgecolor="none", alpha=0.9))
    xs = np.linspace(0.02, 0.98, 160)
    ax.plot(xs, 0.63 + 0.012 * np.sin(xs * 12 * np.pi), color="#2c3e50", lw=1.4)
    ax.text(0.04, 0.66, "ground surface", fontsize=8, color="#2c3e50")
    ax.text(0.04, 0.57, "water table", fontsize=8, color="#145a7a")

    rounded(ax, (0.08, 0.78), 0.20, 0.11, "earthquake\ndemand\nCSR", "#f4b183")
    rounded(ax, (0.38, 0.78), 0.22, 0.11, "soil\nresistance\nCRR", "#a9d18e")
    rounded(ax, (0.70, 0.78), 0.22, 0.11, "effective\nstress\nstate", "#9dc3e6")
    rounded(ax, (0.29, 0.37), 0.38, 0.13, "margin coordinate\ns = ln(CSR/CRR) or ln(1+LPI)", "#f7f3d7", fs=9)
    rounded(ax, (0.70, 0.36), 0.24, 0.13, "GWT shift\nFS 0.64 -> 0.90 -> 1.25", "#dceefb", fs=8)

    arrow(ax, (0.18, 0.78), (0.38, 0.50))
    arrow(ax, (0.49, 0.78), (0.49, 0.50))
    arrow(ax, (0.81, 0.78), (0.60, 0.50))
    arrow(ax, (0.71, 0.43), (0.66, 0.43), color="#145a7a", lw=1.4)

    ax.plot([0.20, 0.78], [0.24, 0.24], color="#2c3e50", lw=2.0)
    ax.add_patch(Polygon([[0.48, 0.21], [0.52, 0.21], [0.52, 0.27], [0.48, 0.27]], color="#c0392b"))
    ax.text(0.50, 0.30, "critical\nboundary", ha="center", fontsize=8, color="#7b241c")
    ax.text(0.22, 0.18, "stable", fontsize=9, color="#1f618d")
    ax.text(0.72, 0.18, "triggered/\nmanifested", ha="center", fontsize=9, color="#922b21")
    ax.text(
        0.50,
        0.04,
        "Known physics enters the margin; data audit tests only what remains.",
        ha="center",
        fontsize=8.5,
        color="#37474f",
    )


def panel_b(ax, rel: dict) -> None:
    add_panel_label(ax, "B", "Evidence audit: no supported OOD gain beyond the margin")
    datasets = ["SPT\ntriggering", "CPT\nmanifestation"]
    phys = [
        rel["SPT_Cetin2018"]["physics_auc_ci"],
        rel["CPT_Geyin2021"]["physics_auc_ci"],
    ]
    ml = [
        rel["SPT_Cetin2018"]["ml_grouped_auc_ci"],
        rel["CPT_Geyin2021"]["ml_grouped_auc_ci"],
    ]
    x = np.arange(len(datasets))
    w = 0.32
    colors = {"phys": "#2f80ed", "ml": "#d1495b"}
    for vals, offset, label, color in [
        (phys, -w / 2, "mechanistic margin", colors["phys"]),
        (ml, w / 2, "best grouped ML", colors["ml"]),
    ]:
        auc = [v["auc"] for v in vals]
        lo = [v["auc"] - v["lo"] for v in vals]
        hi = [v["hi"] - v["auc"] for v in vals]
        ax.bar(x + offset, auc, width=w, color=color, alpha=0.88, edgecolor="#1b1f23", label=label)
        ax.errorbar(x + offset, auc, yerr=[lo, hi], fmt="none", ecolor="#1b1f23", lw=1.2, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("AUC, earthquake-grouped")
    ax.set_ylim(0.50, 1.00)
    ax.axhline(0.5, color="#9aa0a6", lw=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(axis="y", color="#e5e7eb", lw=0.8)
    ax.text(0.02, 0.94, "100-repeat random optimism:\n+0.047 SPT, +0.117 CPT", transform=ax.transAxes, fontsize=8.5)


def panel_c(ax, amb: dict) -> None:
    add_panel_label(ax, "C", "Failure locus: ambiguity concentrates near the boundary")
    colors = {"SPT_Cetin2018": "#2f80ed", "CPT_Geyin2021": "#6f4e9b"}
    labels = {"SPT_Cetin2018": "SPT triggering", "CPT_Geyin2021": "CPT manifestation"}
    for key in ["SPT_Cetin2018", "CPT_Geyin2021"]:
        rows = amb[key]["bin_count_sensitivity"]
        bins = [r["n_bins"] for r in rows]
        floor = [r["coordinate_ambiguity_floor"] for r in rows]
        frac = [r["frac_floor_in_0.30_0.70_band"] for r in rows]
        ax.plot(bins, floor, marker="o", lw=2.2, color=colors[key], label=labels[key])
        idx = min(range(len(bins)), key=lambda i: abs(bins[i] - 10))
        ax.text(
            bins[idx] + 0.2,
            floor[idx],
            f"{frac[idx]*100:.0f}% in critical band\n(10-bin estimate)",
            fontsize=8,
            color=colors[key],
        )
    ax.set_xlabel("number of probability bins")
    ax.set_ylabel("coordinate ambiguity floor")
    ax.set_ylim(0.12, 0.34)
    ax.set_xlim(5.5, 15.8)
    ax.grid(color="#e5e7eb", lw=0.8)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.text(
        0.02,
        0.06,
        "This is a coordinate-level ambiguity bound, not a full-state limit.",
        transform=ax.transAxes,
        fontsize=8.4,
        color="#37474f",
    )


def panel_d(ax, decision: dict) -> None:
    add_panel_label(ax, "D", "Decision repair: critical band becomes a two-label flag")
    keys = ["SPT_Cetin2018", "CPT_Geyin2021"]
    labels = ["SPT\ntriggering", "CPT\nmanifestation"]
    coverage = []
    critical_two = []
    two_label = []
    for key in keys:
        a = decision["datasets"][key]["alpha"]["0.10"]
        coverage.append(a["mechanism_band"]["coverage"])
        two_label.append(a["mechanism_band"]["two_label_rate"])
        critical_two.append(a["mechanism_band_by_band"]["1"]["two_label_rate"])
    x = np.arange(2)
    w = 0.24
    ax.bar(x - w, coverage, width=w, color="#27ae60", edgecolor="#1b1f23", label="band coverage")
    ax.bar(x, two_label, width=w, color="#b7bdc5", edgecolor="#1b1f23", label="two-label all")
    ax.bar(x + w, critical_two, width=w, color="#f2c94c", edgecolor="#1b1f23", label="two-label critical")
    ax.axhline(0.90, color="#c0392b", ls="--", lw=1.2)
    ax.text(1.29, 0.905, "target 0.90", fontsize=8, color="#7b241c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("coverage or set rate")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", color="#e5e7eb", lw=0.8)
    ax.legend(frameon=False, fontsize=7.4, loc="lower right")
    for xi, val in zip(x + w, critical_two):
        ax.text(xi, val + 0.025, f"{val*100:.1f}%", ha="center", fontsize=8, color="#7b5a00")


def main() -> None:
    ensure_dirs()
    rel = load_json("reliability_upgrade.json")
    amb = load_json("ambiguity_floor_sensitivity.json")
    decision = load_json("conformal_decision_metrics.json")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlepad": 8,
            "figure.dpi": 160,
            "savefig.dpi": 320,
        }
    )

    fig = plt.figure(figsize=(13.2, 9.0), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.0], height_ratios=[1.0, 1.0], hspace=0.34, wspace=0.28)
    panel_a(fig.add_subplot(gs[0, 0]))
    panel_b(fig.add_subplot(gs[0, 1]), rel)
    panel_c(fig.add_subplot(gs[1, 0]), amb)
    panel_d(fig.add_subplot(gs[1, 1]), decision)

    fig.suptitle(
        "Mechanism-evidence-decision loop for reliable liquefaction AI",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.50,
        0.942,
        "mechanism -> measured bound -> failure locus -> decision repair -> bounded claim",
        ha="center",
        fontsize=9,
        color="#37474f",
    )
    paths = {
        "svg": OUT / "svg" / "Fig1_central_mechanism_evidence.svg",
        "pdf": OUT / "pdf" / "Fig1_central_mechanism_evidence.pdf",
        "png": OUT / "png" / "Fig1_central_mechanism_evidence.png",
    }
    for ext, path in paths.items():
        fig.savefig(path, bbox_inches="tight")
        print(f"wrote {ext}: {path}")


if __name__ == "__main__":
    main()
