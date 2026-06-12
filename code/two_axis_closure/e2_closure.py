"""SPT--Vs site-paired critical-band closure test.

Stage 1 (label-blind): adjudicated site-alias table below was written by reading
ONLY the site-name strings in e2_match_candidates_blind.csv (no labels, margins,
or outcomes were visible). Each entry carries a tier and a reason. Cetin-side
case resolution is deterministic: among rows of the matched event-year whose
site matches the pattern, prefer an exact boring-suffix match, else the row with
minimal |Mw difference| (must be <= 0.35), else lowest case number.

Stage 2 (outcomes joined only after the audit file is frozen): pre-specified
readout. Model A: logistic(y ~ s_pen). Model B: logistic(y ~ s_pen + s_vs).
Event-grouped CV (GroupKFold by Cetin earthquake string). H1 criterion:
in-band (p_A in [0.3, 0.7], n >= 20, both classes) paired Delta NLL interval
excludes zero below AND median gain > +0.05.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vs_kayen2013 as vk  # noqa: E402
from run_closure import event_bootstrap_delta, metrics, oof_probs  # noqa: E402

MODULE = Path(__file__).resolve().parent
CODE_ROOT = MODULE.parent
RESULTS = MODULE
VS_CSV = MODULE / "incoming" / "kayen2013_table_s1.csv"
CETIN_CSV = CODE_ROOT / "data" / "processed" / "cetin2018_baseline_records.csv"

BAND = (0.3, 0.7)
H1_DNLL = 0.05

# (vs_case_ids, year, cetin-site regex, tier, reason) - written label-blind.
ALIAS = [
    (["95"], 1964, r"^Showa Br", "A", "Showa Bridge / Shinano River south bank, Niigata: unique landmark site"),
    (["56A"], 1968, r"^Nanaehama", "A", "Nanaehama Beach = Cetin Nanaehama1-2-3"),
    (["64B", "65B"], 1968, r"^Aomori Station", "A", "Aomori Station, exact name (two Vs casings, one Cetin site)"),
    (["9016"], 1975, r"^Ying Kou P\. P\.", "A", "Ying-Kou Paper Mill = Ying Kou Paper Plant"),
    (["9017"], 1975, r"^Ying Kou G\. F\. P\.", "A", "Ying-Kou Glass Fiber = Ying Kou Glass Fiber Plant"),
    (["203", "205"], 1976, r"^Luan Nan", "A", "LUANNAN = Luan Nan county site"),
    (["75A", "75B"], 1978, r"^Yuriagekami-2", "A", "Yuriage Kami-2, exact boring name; A/B = M7.4/M6.7 events via Mw"),
    (["77A", "77B"], 1978, r"^Nakamura", "A", "Nakamura dike borings, same levee site"),
    (["78B"], 1978, r"^Shiomi", "B", "Shiomi-Minami Wharf vs Shiomi-6: same wharf area, boring unconfirmed"),
    (["79A", "79B"], 1978, r"^Nakajima", "A", "Nakajima Wharf borings, same port site"),
    (["81A", "81B", "82A", "82B"], 1978, r"^Kitawabuchi", "A", "Kitawabuchi, Eai River: unique site name"),
    (["9025"], 1979, r"^Radio Tower", "A", "Imperial Valley Radio Tower array, reoccupied liquefaction site"),
    (["9026"], 1979, r"^McKim", "A", "McKim Ranch, Imperial Valley"),
    (["9028"], 1979, r"^Kornbloom", "A", "Kornbloom site, Imperial Valley"),
    (["9029", "9030", "9031", "9032"], 1979, r"^Heber Road", "A", "Heber Road array (channel fill / point bar sub-areas)"),
    (["9033", "9034", "9034B"], 1981, r"^Wildlife", "A", "Wildlife liquefaction array"),
    (["9035"], 1981, r"^Radio Tower", "A", "Radio Tower array"),
    (["9036"], 1981, r"^McKim", "A", "McKim Ranch"),
    (["9038"], 1981, r"^Kornbloom", "A", "Kornbloom site"),
    (["64A", "64C"], 1983, r"^Aomori Station", "A", "Aomori Station; main shock vs aftershock via Mw"),
    (["67A", "67B"], 1983, r"^Takeda", "A", "Takeda Elementary School"),
    (["72"], 1983, r"^Gaiko Wharf", "A", "Gaiko Wharf, Akita"),
    (["73"], 1983, r"^Arayamotomachi$", "A", "Araya-Motomachi, exact name"),
    (["9093", "9094", "9095", "9104", "9105", "9106"], 1987, r"^Wildlife", "A", "Wildlife array; SH vs ER events via Mw"),
    (["9096", "9107"], 1987, r"^Radio Tower", "A", "Radio Tower array"),
    (["9097", "9108"], 1987, r"^McKim", "A", "McKim Ranch"),
    (["9099", "9110"], 1987, r"^Kornbloom", "A", "Kornbloom site"),
    (["9100", "9101", "9102", "9103", "9111", "9112", "9113", "9114"], 1987, r"^Heber Road", "A", "Heber Road array"),
    (["513", "514"], 1989, r"^Farris Farm", "A", "Faris/Farris Farm spelling variant, Pajaro Valley"),
    (["517", "518"], 1989, r"^Miller Farm$", "A", "Clint Miller Farm = Miller Farm"),
    (["519"], 1989, r"^Miller Farm CMF10", "A", "boring-level match CMF10"),
    (["520"], 1989, r"^Miller Farm CMF8", "A", "boring-level match CMF8"),
    (["9115", "9116", "9117", "9118", "9119", "9122", "9124", "9125", "9126", "9127", "9128"], 1989,
     r"^Treasure Island", "A", "Treasure Island fill (single Cetin case)"),
    (["538", "539", "540", "9174", "9175"], 1989, r"^State Beach", "A", "Moss Landing State Beach UC borings"),
    (["9176", "9178"], 1989, r"^Sandholdt|^Sandholt", "A", "Sandholdt Road, Moss Landing"),
    (["9180"], 1989, r"^Marine Laboratory", "B", "Moss Landing Marine Lab area; 'Woodward Marine' boring unconfirmed"),
    (["28"], 1993, r"^Kushiro Port Seismo", "A", "Kushiro Port seismometer station"),
    (["34"], 1993, r"^Kushiro Port Site D", "B", "south end of port vs Site D: area-level"),
    (["114", "115", "116"], 1948, r"^Takaya", "B", "Takaya area, Kuzuryu River: area-level only"),
]

REJECT_NOTE = ("1995 Kobe island-level candidates (Port/Rokko Island), 1976 Fengnan village "
               "cross-matches, Leonardini/Seamist/Marinovich/Radovich/McGowan farms (no Cetin "
               "counterpart), and all sub-0.5-similarity leftovers were rejected as not "
               "auditable same-site pairs.")

YR = lambda s: (m.group(0) if (m := re.search(r"(19|20)\d{2}", str(s))) else None)
SUFFIX = re.compile(r"(\d+)\s*$")


def build_pairs() -> pd.DataFrame:
    vs = pd.read_csv(VS_CSV, dtype={"case_id": str})
    cet = pd.read_csv(CETIN_CSV)
    vs["y4"] = vs["earthquake"].map(YR).astype(float)
    cet["y4"] = cet["earthquake"].map(YR).astype(float)
    cet["mw_num"] = pd.to_numeric(cet["Mw"], errors="coerce")
    rows = []
    for vs_ids, year, pat, tier, reason in ALIAS:
        for vid in vs_ids:
            r = vs[vs["case_id"] == vid]
            if r.empty:
                continue
            r = r.iloc[0]
            cands = cet[(cet["y4"] == year) & cet["site"].astype(str).str.match(pat)]
            if cands.empty:
                continue
            dmw = (cands["mw_num"] - float(r["mw"])).abs()
            cands = cands[dmw <= 0.35]
            if cands.empty:
                continue
            sfx = SUFFIX.search(str(r["site"]))
            chosen = None
            if sfx:
                exact = cands[cands["site"].astype(str).str.rstrip().str.endswith(sfx.group(1))]
                if len(exact):
                    chosen = exact.sort_values("case").iloc[0]
            if chosen is None:
                dmw = (cands["mw_num"] - float(r["mw"])).abs()
                cands = cands.assign(_d=dmw).sort_values(["_d", "case"])
                chosen = cands.iloc[0]
            rows.append({
                "tier": tier, "year": year,
                "vs_case": vid, "vs_site": r["site"], "vs_eq": r["earthquake"], "vs_mw": float(r["mw"]),
                "cet_case": int(chosen["case"]), "cet_site": chosen["site"],
                "cet_eq": chosen["earthquake"], "cet_mw": float(chosen["mw_num"]),
                "reason": reason,
            })
    return pd.DataFrame(rows)


def main() -> None:
    pairs = build_pairs()
    audit = RESULTS / "e2_pairs_adjudicated.csv"
    pairs.to_csv(audit, index=False)
    nA = int((pairs["tier"] == "A").sum())
    n_events = pairs.loc[pairs["tier"] == "A", "cet_eq"].nunique()
    print(f"[e2] adjudicated pairs: {len(pairs)} (tier A: {nA}, events A: {n_events})")
    print(f"[e2] reject note: {REJECT_NOTE}")

    res = {"n_pairs_total": int(len(pairs)), "n_tierA": nA, "n_events_tierA": int(n_events),
           "amendment": "pre-specified frozen amendment; gate unchanged (>=60 tier-A pairs, >=8 events, in-band n>=20)"}
    if nA < 60 or n_events < 8:
        res["status"] = "UNPOWERED"
        (RESULTS / "e2_closure_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print("[e2] still UNPOWERED under amendment; stopping per frozen analysis plan.")
        return

    # ---- outcomes joined only from here on ----
    vs = pd.read_csv(VS_CSV, dtype={"case_id": str}).set_index("case_id")
    cet = pd.read_csv(CETIN_CSV).set_index("case")

    def run(subset: pd.DataFrame, label: str) -> dict:
        v = vs.loc[subset["vs_case"]]
        c = cet.loc[subset["cet_case"]]
        y = c["y"].to_numpy(float)
        groups = c["earthquake"].astype(str).to_numpy()
        s_pen = c["score_liq"].to_numpy(float)
        s_vs = vk.margin(v["vs1_ms"].to_numpy(float), v["csr"].to_numpy(float),
                         v["mw"].to_numpy(float), v["sigma_eff_kpa"].to_numpy(float), 0.0)
        lab_agree = float((c["y"].to_numpy(float) == v["liquefied"].map({"Yes": 1.0, "No": 0.0}).to_numpy(float)).mean())
        XA = s_pen.reshape(-1, 1)
        XB = np.column_stack([s_pen, s_vs])
        pA = oof_probs(lambda: LogisticRegression(max_iter=2000), XA, y, groups)
        pB = oof_probs(lambda: LogisticRegression(max_iter=2000), XB, y, groups)
        out = {"n": int(len(y)), "n_events": int(len(np.unique(groups))),
               "label_agreement_vs_pen": lab_agree,
               "pen_only": metrics(y, pA), "pen_plus_vs": metrics(y, pB),
               "paired_overall": event_bootstrap_delta(y, pA, pB, groups)}
        inb = (pA >= BAND[0]) & (pA <= BAND[1])
        out["n_inband"] = int(inb.sum())
        out["inband_classes"] = sorted(set(y[inb].tolist()))
        if inb.sum() >= 20 and len(set(y[inb])) == 2:
            out["inband_pen_only"] = metrics(y[inb], pA[inb])
            out["inband_pen_plus_vs"] = metrics(y[inb], pB[inb])
            out["paired_inband"] = event_bootstrap_delta(y[inb], pA[inb], pB[inb], groups[inb])
            ci = out["paired_inband"]["d_nll_ci"]
            out["H1_supported"] = bool(ci[0] > 0 and out["paired_inband"]["d_nll"] > H1_DNLL)
            out["resolved_fraction"] = float(((pB[inb] < BAND[0]) | (pB[inb] > BAND[1])).mean())
        else:
            out["H1_supported"] = None
            out["note"] = "in-band gate not met"
        print(f"[e2:{label}] n={out['n']} events={out['n_events']} "
              f"overall dNLL={out['paired_overall']['d_nll']:+.3f} {out['paired_overall']['d_nll_ci']} "
              f"inband n={out['n_inband']} H1={out['H1_supported']}")
        return out

    res["tierA"] = run(pairs[pairs["tier"] == "A"], "tierA")
    res["tierAB_sensitivity"] = run(pairs, "tierA+B")
    res["status"] = "DONE"
    res["H1_supported_primary"] = res["tierA"]["H1_supported"]
    (RESULTS / "e2_closure_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
