"""Number-consistency verifier for the re-framed manuscript.

Every headline number in the manuscript text must match the computed JSON outputs (no drift, typo or
fabrication). Reports GREEN/TOTAL. Robust to LaTeX/unicode minus signs and spacing.

Run:  python verify_manuscript.py [path-to-manuscript.(md|tex)]
"""
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, "data", "processed")


def L(f):
    with open(os.path.join(PROC, f), encoding="utf-8") as fh:
        return json.load(fh)


pc, tm, ov, vi, gr = (L("predictability_ceiling.json"), L("tuned_ml_challenge.json"),
                      L("overvalidation_benchmark.json"), L("value_of_information.json"),
                      L("geological_residual.json"))

if len(sys.argv) <= 1:
    print("Usage: python verify_manuscript.py <path-to-submitted-manuscript.md-or-tex>")
    sys.exit(2)

path = sys.argv[1]
raw = open(path, encoding="utf-8").read()
raw = raw.split("Changelog")[0]  # ignore the deletable changelog


def norm(s):
    s = str(s).replace("−", "-").replace("–", "-").replace("—", "-")
    return s.replace("$", "").replace("\\,", "").replace("\\", "").replace(" ", "").replace("+", "")


NTEX = norm(raw)
checks = []


def chk(name, *vals):
    missing = [v for v in vals if norm(v) not in NTEX]
    checks.append((not missing, name, missing))


ARMS = [("SPT_Cetin2018", "SPT"), ("CPT_Geyin2021", "CPT"), ("Vs_Kayen2013", "Vs")]

# 1. Predictability ceiling: AUC + CI + floor
for k, lab in ARMS:
    c = pc[k]["predictability_ceiling"]["discrimination_AUC_ceiling"]
    f = pc[k]["predictability_ceiling"]["irreducible_error_floor_calibrated"]
    chk(f"{lab} ceiling AUC+CI", f"{c['auc']:.3f}", f"{c['lo']:.3f}", f"{c['hi']:.3f}")
    chk(f"{lab} error floor", f"{f['est']:.3f}")

# 2. Tuned-ML challenge: ΔAUC, P, upper limit, margin AUC
for k, lab in ARMS:
    d = tm[k]["best_tuned_ml_minus_margin_AUC"]
    chk(f"{lab} tuned ΔAUC", f"{d['delta']:.3f}")
    chk(f"{lab} tuned P", f"{d['P(delta>0)']:.3f}")
    chk(f"{lab} tuned upper limit", f"{d['hi']:.3f}")

# 3. Over-validation flip (grouped) + optimism
for k, lab in ARMS:
    fl = ov[k]["ml_vs_margin_flip"]
    chk(f"{lab} flip grouped", f"{fl['under_grouped_cv_best_ml_minus_margin_AUC']:.3f}")
    chk(f"{lab} best-ML optimism", f"{ov[k]['models'][fl['best_ml_model']]['optimism_random_minus_grouped']:.3f}")

# 4. Value of information
chk("SPT VoI", f"{vi['SPT_Cetin2018']['margin_voi_oof_at_operating_points']['pi=0.2,R=10']['voi']:.3f}")
for k, lab in ARMS:
    d = vi[k]["extra_predictors_added_value"]["pi=0.2,R=10"]["delta_full_minus_margin"]
    chk(f"{lab} added-predictor Δ", f"{d:.3f}")

# 5. Geological structure: transitional contrast (both databases) + floors
for k, lab in [("CPT_Geyin2021", "Geyin"), ("CPT_Rateria2024_independent_replication", "Rateria")]:
    c = gr[k]["contrast_transitional_vs_sandlike_Ic"]
    chk(f"{lab} transitional Δ+CI", f"{c['diff']:.3f}", f"{c['lo']:.3f}", f"{c['hi']:.3f}")

# 6. Figures exist + referenced
FIGDIR = os.path.join(BASE, "..", "figures", "reframe_2026-06-30")
for i, stem in enumerate(["fig1_overvalidation_flip", "fig2_predictability_ceiling", "fig3_tuned_ml_challenge",
                          "fig4_value_of_information", "fig5_geological_structure"], 1):
    exists = os.path.exists(os.path.join(FIGDIR, stem + ".pdf"))
    referenced = (f"Fig. {i}" in raw) or (f"Fig.~{i}" in raw) or (f"Figure {i}" in raw) or (f"fig:{i}" in raw)
    checks.append((exists and referenced, f"Fig {i} exists+referenced",
                   [] if exists and referenced else [f"exists={exists}", f"ref={referenced}"]))

# 7. No leftover placeholders (the reproducibility DOI is the only allowed TODO)
bad = re.findall(r"\[EG-\d|Engineering Geology cite|to.?fill|placeholder|XXXX|\bTBD\b", raw)
checks.append((len(bad) == 0, "no unfilled placeholders", bad[:5]))

green = sum(1 for ok, _, _ in checks if ok)
print(f"MANUSCRIPT VERIFIER  ({os.path.basename(path)}):  {green}/{len(checks)} GREEN\n")
for ok, name, missing in checks:
    if not ok:
        print(f"  FAIL  {name}   missing: {missing}")
if green == len(checks):
    print("  ALL CHECKS GREEN — every headline number traces to a computed output.")
sys.exit(0 if green == len(checks) else 1)
