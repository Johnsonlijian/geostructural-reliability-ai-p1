"""E2 amendment #2 execution: Moss-CPT axis + pooled closure + post-hoc robustness.

Moss engine: PEER 2005/15 Eq. (5.3)/(5.4), Table 5.1 mean Theta values
(transcribed from `incoming/moss2006_peer_2005-15.pdf` pp. 59-60; zero-shot,
never refit). Units per Table 4.1: qc1 in MPa, Rf in %, sigma'_v in kPa.

MOSS_ALIAS below was adjudicated label-blind from the names-only candidate list
(results/moss_match_candidates_blind.csv). Rejections recorded in REJECT_NOTE.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vs_kayen2013 as vk  # noqa: E402
from run_closure import event_bootstrap_delta, metrics, oof_probs  # noqa: E402
from e2_closure import build_pairs as build_spt_pairs  # noqa: E402

MODULE = Path(__file__).resolve().parent
CODE_ROOT = MODULE.parent
RESULTS = MODULE
VS_CSV = MODULE / "incoming" / "kayen2013_table_s1.csv"
MOSS_CSV = MODULE / "incoming" / "moss2006_table41.csv"
CETIN_CSV = CODE_ROOT / "data" / "processed" / "cetin2018_baseline_records.csv"

BAND = (0.3, 0.7)
H1_DNLL = 0.05

# Moss et al. (2006) / PEER 2005/15 published coefficients (Table 5.1 means).
T1, T2, T3, T4, T5, T6, T7, SE = 0.110, 0.001, 0.850, 7.177, 0.848, 0.002, 20.923, 1.632
QEXP = 1.045  # concise Eq. (5.3) exponent


def moss_margin(qc1, rf, c, csr, mw, sig_kpa):
    g = (np.asarray(qc1, float) ** QEXP
         + np.asarray(qc1, float) * (T1 * np.asarray(rf, float))
         + (T2 * np.asarray(rf, float))
         + np.asarray(c, float) * (1 + T3 * np.asarray(rf, float))
         - T4 * np.log(np.asarray(csr, float))
         - T5 * np.log(np.asarray(mw, float))
         - T6 * np.log(np.asarray(sig_kpa, float))
         - T7)
    return -g / SE  # P_L = Phi(margin)


def moss_self_test():
    rng = np.random.default_rng(3)
    qc1 = rng.uniform(0.5, 20, 100); rf = rng.uniform(0.1, 3, 100)
    c = rng.uniform(0.3, 1.0, 100); csr = rng.uniform(0.05, 0.6, 100)
    mw = rng.uniform(5.9, 8.0, 100); sig = rng.uniform(15, 180, 100)
    s = moss_margin(qc1, rf, c, csr, mw, sig)
    # Eq. (5.4) round trip via margin identity
    num = (qc1 ** QEXP + qc1 * (T1 * rf) + (T2 * rf) + c * (1 + T3 * rf)
           - T5 * np.log(mw) - T6 * np.log(sig) - T7 + SE * s)
    csr_back = np.exp(num / T4)
    assert np.max(np.abs(csr_back / csr - 1)) < 1e-9
    base = moss_margin(8, 1, 0.6, 0.2, 7, 60)
    assert moss_margin(12, 1, 0.6, 0.2, 7, 7) < base      # stronger soil -> safer
    assert moss_margin(8, 1, 0.6, 0.3, 7, 60) > base      # more demand -> riskier
    print("[moss engine] self-tests pass")


# Label-blind adjudication (names only): (vs_cases, year, moss-site regex, tier, reason)
MOSS_ALIAS = [
    (["9016"], 1975, r"^Paper Mill", "A", "Ying-Kou Paper Mill"),
    (["9018"], 1975, r"^Const\. Com\. Building", "A", "Construction Building, Ying-Kou"),
    (["9020"], 1975, r"^17th Middle School", "A", "Middle School, Ying-Kou"),
    (["9021"], 1975, r"^Chemical Fiber", "A", "Chemical Fiber, Ying-Kou"),
    (["9022", "9023", "9024"], 1979, r"^Wildlife", "A", "Wildlife array"),
    (["9025"], 1979, r"^Radio Tower", "A", "Radio Tower array"),
    (["9026"], 1979, r"^McKim", "A", "McKim Ranch"),
    (["9028"], 1979, r"^Kornbloom", "A", "Kornbloom"),
    (["9033", "9034", "9034B"], 1981, r"^Wildlife", "A", "Wildlife array"),
    (["9035"], 1981, r"^Radio Tower", "A", "Radio Tower array"),
    (["9036"], 1981, r"^McKim", "A", "McKim Ranch"),
    (["9038"], 1981, r"^Kornbloom", "A", "Kornbloom"),
    (["9043", "9044", "9045", "9046", "9047", "9048", "9049", "9050", "9051", "9052", "9053"],
     1983, r"^Pence Ranch", "A", "Pense=Pence Ranch spelling variant, Borah Peak"),
    (["9061", "9062"], 1983, r"^Whiskey Springs", "A", "Whiskey Springs site, Borah Peak"),
    (["9093", "9094", "9095", "9104", "9105", "9106"], 1987, r"^Wildlife", "A", "Wildlife; ER/SH via Mw"),
    (["509", "516"], 1989, r"^Leonardini", "A", "Lenardini=Leonardini Farm"),
    (["510"], 1989, r"^Leonardini 53", "A", "boring LEN 53"),
    (["511"], 1989, r"^Model Airport 18", "A", "boring AIR 18"),
    (["512"], 1989, r"^Model Airport 21", "A", "boring AIR 21"),
    (["513", "514"], 1989, r"^Farris 61", "A", "boring FAR 61"),
    (["515"], 1989, r"^Leonardini", "A", "Leonardini Farm"),
    (["521"], 1989, r"^Leonardini 39", "A", "boring LEN 39"),
    (["517", "518"], 1989, r"^Miller Fa(r)?m CMF", "A", "Clint Miller Farm"),
    (["519"], 1989, r"^Miller Farm CMF 10", "A", "boring CMF10"),
    (["520"], 1989, r"^Miller Fam CMF 8|^Miller Farm CMF 8", "A", "boring CMF8 (source typo 'Fam')"),
    (["522"], 1989, r"^Sea Mist 31", "A", "SEAMIST FARM 31"),
    (["524"], 1989, r"^Granite Const\. 123", "A", "GRA 123"),
    (["525"], 1989, r"^Marinovich 65", "A", "MRR 65"),
    (["526"], 1989, r"^Marinovich 67", "A", "MRR 67"),
    (["527"], 1989, r"^Radovich", "A", "Radovich Farm 98/99, lowest"),
    (["528"], 1989, r"^McGowan Farm 136", "A", "MCG 136"),
    (["530"], 1989, r"^SP Bridge", "A", "Southern Pacific Bridge"),
    (["531"], 1989, r"^Kett", "A", "Kett Ranch"),
    (["538", "539", "540", "9174", "9175"], 1989, r"^Moss Landing S\.? ?B", "A", "Moss Landing State Beach"),
    (["9176", "9178"], 1989, r"^Sandholdt", "A", "Sandholdt Road"),
    (["9180"], 1989, r"^Woodward Marine", "A", "Woodward Marine facility"),
    (["165"], 1995, r"^Imazu Elementary", "A", "Imazu Sho Gakku = Imazu Elementary School"),
    # tier B (area/facility-level)
    (["9116", "9117", "9118", "9119", "9122"], 1989, r"^T\.I\. Naval Station", "B", "fire station within T.I. Naval Station"),
    (["9160V"], 1989, r"^Salinas River", "B", "river reach vs bridge"),
    (["529"], 1989, r"^McGowan Farm", "B", "MCG 138 vs McGowan 136"),
    (["176"], 1995, r"^Hamakoshienn", "B", "Hama Koshien area"),
    (["135"], 1999, r"^Yuanlin", "B", "Yuanlin town-level"),
    (["9188", "9189"], 1999, r"^Adapazari", "B", "Adapazari site codes unmappable"),
]

REJECT_NOTE = ("Rejected: Kobe Port/Rokko park-name non-matches, Wufeng/Nantou town-level letters, "
               "Goddard/Larter ranches and 523-SIL levee (no Moss counterpart), Tangshan district-level, "
               "school-vs-park name mismatches.")

YR = lambda s: (m.group(0) if (m := re.search(r"(19|20)\d{2}", str(s))) else None)
SUFFIX = re.compile(r"(\d+)\s*$")


def build_moss_pairs() -> pd.DataFrame:
    vs = pd.read_csv(VS_CSV, dtype={"case_id": str})
    moss = pd.read_csv(MOSS_CSV)
    rows = []
    for vs_ids, year, pat, tier, reason in MOSS_ALIAS:
        for vid in vs_ids:
            r = vs[vs["case_id"] == vid]
            if r.empty:
                continue
            r = r.iloc[0]
            if YR(r["earthquake"]) != str(year):
                continue
            cands = moss[(moss["year"] == year) & moss["site"].astype(str).str.match(pat)]
            cands = cands[(cands["eq_mw"] - float(r["mw"])).abs() <= 0.35]
            if cands.empty:
                continue
            sfx = SUFFIX.search(str(r["site"]))
            chosen = None
            if sfx is not None:
                exact = cands[cands["site"].astype(str).str.contains(rf"\b{sfx.group(1)}\b")]
                if len(exact):
                    chosen = exact.sort_values("moss_id").iloc[0]
            if chosen is None:
                cands = cands.assign(_d=(cands["eq_mw"] - float(r["mw"])).abs()).sort_values(["_d", "moss_id"])
                chosen = cands.iloc[0]
            rows.append({"tier": tier, "year": year, "vs_case": vid, "vs_site": r["site"],
                         "moss_id": int(chosen["moss_id"]), "moss_site": chosen["site"],
                         "moss_eq": chosen["eq_name"], "reason": reason})
    return pd.DataFrame(rows)


def closure(vs_idx, pen_y, pen_margin, groups, s_vs, label):
    XA = pen_margin.reshape(-1, 1)
    XB = np.column_stack([pen_margin, s_vs])
    pA = oof_probs(lambda: LogisticRegression(max_iter=2000), XA, pen_y, groups)
    pB = oof_probs(lambda: LogisticRegression(max_iter=2000), XB, pen_y, groups)
    out = {"n": int(len(pen_y)), "n_events": int(len(np.unique(groups))),
           "pen_only": metrics(pen_y, pA), "pen_plus_vs": metrics(pen_y, pB),
           "paired_overall": event_bootstrap_delta(pen_y, pA, pB, groups)}
    inb = (pA >= BAND[0]) & (pA <= BAND[1])
    out["n_inband"] = int(inb.sum())
    if inb.sum() >= 20 and len(set(pen_y[inb])) == 2:
        out["paired_inband"] = event_bootstrap_delta(pen_y[inb], pA[inb], pB[inb], groups[inb])
        out["inband_pen_only"] = metrics(pen_y[inb], pA[inb])
        out["inband_pen_plus_vs"] = metrics(pen_y[inb], pB[inb])
        ci = out["paired_inband"]["d_nll_ci"]
        out["H1_supported"] = bool(ci[0] > 0 and out["paired_inband"]["d_nll"] > H1_DNLL)
        out["resolved_fraction"] = float(((pB[inb] < BAND[0]) | (pB[inb] > BAND[1])).mean())
    else:
        out["H1_supported"] = None
    print(f"[{label}] n={out['n']} ev={out['n_events']} inband={out['n_inband']} "
          f"dNLL_in={out.get('paired_inband',{}).get('d_nll','--')} H1={out['H1_supported']}")
    return out


def main():
    moss_self_test()
    vs = pd.read_csv(VS_CSV, dtype={"case_id": str}).set_index("case_id")
    res = {}

    # ---- Moss axis ----
    mp = build_moss_pairs()
    mp.to_csv(RESULTS / "e2_moss_pairs_adjudicated.csv", index=False)
    moss = pd.read_csv(MOSS_CSV).set_index("moss_id")
    res["moss_n_tierA"] = int((mp["tier"] == "A").sum())
    res["moss_n_events_tierA"] = mp.loc[mp["tier"] == "A", "moss_eq"].nunique()
    res["reject_note"] = REJECT_NOTE
    print(f"[moss pairs] total={len(mp)} tierA={res['moss_n_tierA']} eventsA={res['moss_n_events_tierA']}")

    def moss_arrays(sub):
        m = moss.loc[sub["moss_id"]]
        v = vs.loc[sub["vs_case"]]
        y = m["liquefied"].map({"Yes": 1.0, "No": 0.0}).to_numpy(float)
        groups = m["eq_name"].astype(str).to_numpy()
        s_pen = moss_margin(m["qc1_mpa"], m["rf_pct"], m["c_fines"], m["csr"], m["eq_mw"], m["sigma_eff_kpa"])
        s_v = vk.margin(v["vs1_ms"].to_numpy(float), v["csr"].to_numpy(float),
                        v["mw"].to_numpy(float), v["sigma_eff_kpa"].to_numpy(float), 0.0)
        return y, np.asarray(s_pen), groups, np.asarray(s_v)

    if res["moss_n_tierA"] >= 60 and res["moss_n_events_tierA"] >= 8:
        yA, spA, gA, svA = moss_arrays(mp[mp["tier"] == "A"])
        res["moss_tierA"] = closure(None, yA, spA, gA, svA, "moss tierA")
        yB, spB, gB, svB = moss_arrays(mp)
        res["moss_tierAB"] = closure(None, yB, spB, gB, svB, "moss tierA+B")
    else:
        res["moss_status"] = "UNPOWERED_ALONE"

    # ---- SPT axis (from e2_closure) + pooled ----
    sp = build_spt_pairs()
    cet = pd.read_csv(CETIN_CSV).set_index("case")
    spA_df = sp[sp["tier"] == "A"]
    c = cet.loc[spA_df["cet_case"]]
    v = vs.loc[spA_df["vs_case"]]
    y1 = c["y"].to_numpy(float)
    g1 = ("SPT|" + c["earthquake"].astype(str)).to_numpy()
    sp1 = c["score_liq"].to_numpy(float)
    sv1 = vk.margin(v["vs1_ms"].to_numpy(float), v["csr"].to_numpy(float),
                    v["mw"].to_numpy(float), v["sigma_eff_kpa"].to_numpy(float), 0.0)

    mpA = mp[mp["tier"] == "A"]
    y2, sp2, g2raw, sv2 = moss_arrays(mpA)
    g2 = np.array(["CPT|" + g for g in g2raw])

    # pooled with per-instrument standardized margins + instrument indicator
    def z(a): return (a - a.mean()) / a.std()
    y = np.concatenate([y1, y2])
    pen = np.concatenate([z(sp1), z(sp2)])
    svv = np.concatenate([sv1, sv2])
    inst = np.concatenate([np.zeros(len(y1)), np.ones(len(y2))])
    groups = np.concatenate([g1, g2])
    XA = np.column_stack([pen, inst])
    XB = np.column_stack([pen, inst, svv])
    pA = oof_probs(lambda: LogisticRegression(max_iter=2000), XA, y, groups)
    pB = oof_probs(lambda: LogisticRegression(max_iter=2000), XB, y, groups)
    pooled = {"n": int(len(y)), "n_events": int(len(np.unique(groups))),
              "pen_only": metrics(y, pA), "pen_plus_vs": metrics(y, pB),
              "paired_overall": event_bootstrap_delta(y, pA, pB, groups)}
    inb = (pA >= BAND[0]) & (pA <= BAND[1])
    pooled["n_inband"] = int(inb.sum())
    pooled["paired_inband"] = event_bootstrap_delta(y[inb], pA[inb], pB[inb], groups[inb])
    pooled["inband_pen_only"] = metrics(y[inb], pA[inb])
    pooled["inband_pen_plus_vs"] = metrics(y[inb], pB[inb])
    ci = pooled["paired_inband"]["d_nll_ci"]
    pooled["H1_supported"] = bool(ci[0] > 0 and pooled["paired_inband"]["d_nll"] > H1_DNLL)
    pooled["resolved_fraction"] = float(((pB[inb] < BAND[0]) | (pB[inb] > BAND[1])).mean())
    print(f"[pooled] n={pooled['n']} ev={pooled['n_events']} inband={pooled['n_inband']} "
          f"dNLL_in={pooled['paired_inband']['d_nll']:+.3f} {ci} H1={pooled['H1_supported']}")
    res["pooled_tierA"] = pooled

    # ---- post-hoc robustness: SPT axis excluding label-disagreement pairs ----
    agree = c["y"].to_numpy(float) == v["liquefied"].map({"Yes": 1.0, "No": 0.0}).to_numpy(float)
    res["spt_label_agreement"] = float(agree.mean())
    if agree.sum() >= 40:
        XA1 = sp1[agree].reshape(-1, 1); XB1 = np.column_stack([sp1[agree], sv1[agree]])
        pa = oof_probs(lambda: LogisticRegression(max_iter=2000), XA1, y1[agree], g1[agree])
        pb = oof_probs(lambda: LogisticRegression(max_iter=2000), XB1, y1[agree], g1[agree])
        inb1 = (pa >= BAND[0]) & (pa <= BAND[1])
        rb = {"n": int(agree.sum()), "n_inband": int(inb1.sum())}
        if inb1.sum() >= 15 and len(set(y1[agree][inb1])) == 2:
            rb["paired_inband"] = event_bootstrap_delta(y1[agree][inb1], pa[inb1], pb[inb1], g1[agree][inb1])
        res["spt_posthoc_agreement_only"] = rb
        print(f"[post-hoc agree-only] n={rb['n']} inband={rb['n_inband']} "
              f"dNLL_in={rb.get('paired_inband',{}).get('d_nll','--')}")

    (RESULTS / "e2_closure2_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("[done] results/e2_closure2_results.json")


if __name__ == "__main__":
    main()
