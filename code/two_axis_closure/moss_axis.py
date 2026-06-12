"""Moss-CPT axis (E2 amendment #2): extract PEER 2005/15 Table 4.1 and emit
label-blind pairing candidates against the KEA2013 Vs catalog.

Sources (both archived in ../incoming/):
- moss2006_peer_2005-15.pdf, Table 4.1 pp. 41-45 (row grammar: Site, Liquefied?,
  Data Class, Crit. Depth Range, GWT, svo+-, s'vo+-, amax+-, rd+-, CSR+-, c,
  qc1+-, Rf+-) and Eqs. (5.3)/(5.4) + Table 5.1 (p. 59-60).
Stage 1 output is name-only (no labels in the candidates file used for
adjudication view... labels are retained in the raw extraction CSV but the
candidate file carries names/events only).
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path

import pandas as pd
import pdfplumber

ROUND = Path(__file__).resolve().parents[1]
SRC = ROUND / "incoming" / "moss2006_peer_2005-15.pdf"
OUT = ROUND / "incoming" / "moss2006_table41.csv"
CAND = ROUND / "results" / "moss_match_candidates_blind.csv"
VS_CSV = ROUND / "incoming" / "kayen2013_table_s1.csv"

PM = r"(?:\u00b1|\ufffd+)"
NUM = r"[\d.]+"

ROW = re.compile(
    rf"^(?P<site>.+?)\s+"
    rf"(?P<liq>Yes|No|YES|NO|Yes/No)\s+"
    rf"(?P<dclass>[A-E])\s+"
    rf"(?P<d1>{NUM})-(?P<d2>{NUM})\s+(?P<gwt>{NUM})\s+"
    rf"(?P<svo>{NUM})\s*{PM}\s*(?P<svo_sd>{NUM})\s+"
    rf"(?P<sveff>{NUM})\s*{PM}\s*(?P<sveff_sd>{NUM})\s+"
    rf"(?P<amax>{NUM})\s*{PM}\s*(?P<amax_sd>{NUM})\s+"
    rf"(?P<rd>{NUM})\s*{PM}\s*(?P<rd_sd>{NUM})\s+"
    rf"(?P<csr>{NUM})\s*{PM}\s*(?P<csr_sd>{NUM})\s+"
    rf"(?P<c>{NUM})\s+"
    rf"(?P<qc1>{NUM})\s*{PM}\s*(?P<qc1_sd>{NUM})\s+"
    rf"(?P<rf>{NUM})\s*{PM}\s*(?P<rf_sd>{NUM})\s*$"
)
HEADER = re.compile(rf"^(?P<year>(?:18|19|20)\d{{2}})\s+(?P<name>.+?)\s+(?P<mw>{NUM})\s*{PM}\s*(?P<mw_sd>{NUM})")
SKIP = ("Table 4.1", "Site Liquefied?", "Class Range", "Earthquake Mw", "Loma Prieta continued")


def extract() -> pd.DataFrame:
    rows, unparsed = [], []
    eq = None
    with pdfplumber.open(SRC) as pdf:
        for i in range(40, 45):
            for raw in (pdf.pages[i].extract_text() or "").splitlines():
                line = raw.strip()
                if not line or any(line.startswith(s) for s in SKIP) or re.match(r"^\d+$", line):
                    continue
                h = HEADER.match(line)
                if h and not ROW.match(line):
                    eq = {"year": int(h.group("year")),
                          "eq_name": f"{h.group('year')} {h.group('name')}",
                          "eq_mw": float(h.group("mw"))}
                    continue
                m = ROW.match(line)
                if m and eq:
                    g = m.groupdict()
                    rows.append({
                        "site": g["site"].strip(), "liquefied": g["liq"].capitalize(),
                        "data_class": g["dclass"], **eq,
                        "crit_lo": float(g["d1"]), "crit_hi": float(g["d2"]),
                        "gwt_m": float(g["gwt"]),
                        "sigma_eff_kpa": float(g["sveff"]),
                        "amax_g": float(g["amax"]), "rd": float(g["rd"]),
                        "csr": float(g["csr"]), "c_fines": float(g["c"]),
                        "qc1_mpa": float(g["qc1"]), "rf_pct": float(g["rf"]),
                    })
                elif re.search(rf"{PM}", line):
                    unparsed.append(line)
    df = pd.DataFrame(rows)
    df.insert(0, "moss_id", range(1, len(df) + 1))
    df.to_csv(OUT, index=False)
    print(f"moss rows: {len(df)} | events: {df['eq_name'].nunique()} | labels: {df['liquefied'].value_counts().to_dict()}")
    print(f"unparsed candidate lines: {len(unparsed)}")
    for u in unparsed[:10]:
        print("  UNPARSED:", u[:140])
    return df


def candidates(moss: pd.DataFrame) -> None:
    vs = pd.read_csv(VS_CSV, dtype={"case_id": str})
    yr = lambda s: (m.group(0) if (m := re.search(r"(19|20)\d{2}", str(s))) else None)
    vs["y4"] = vs["earthquake"].map(yr).astype(float)
    nk = lambda s: re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()
    rows = []
    for _, r in vs.iterrows():
        sub = moss[moss["year"] == r["y4"]]
        if sub.empty:
            continue
        a = nk(r["site"])
        atok = set(a.split()) - {"the", "of", "at", "site", "city", "pref", "et", "al", "this", "study"}
        for _, c in sub.iterrows():
            b = nk(c["site"])
            btok = set(b.split()) - {"site", "the", "of"}
            sim = difflib.SequenceMatcher(None, re.sub(" ", "", a), re.sub(" ", "", b)).ratio()
            tok = len({t for t in (atok & btok) if len(t) >= 4})
            if sim >= 0.5 or tok >= 1:
                rows.append({"year": int(r["y4"]), "vs_case": r["case_id"], "vs_site": r["site"],
                             "moss_id": int(c["moss_id"]), "moss_site": c["site"],
                             "sim": round(sim, 3), "tok": tok})
    df = pd.DataFrame(rows).sort_values(["year", "vs_case", "sim"], ascending=[True, True, False])
    df.to_csv(CAND, index=False)
    print(f"candidates: {len(df)} | vs cases: {df['vs_case'].nunique()} | years: {sorted(df['year'].unique())}")
    for y in sorted(df["year"].unique()):
        for _, r in df[df["year"] == y].groupby("vs_case").head(2).iterrows():
            print(f"{y} vs[{r['vs_case']}] {str(r['vs_site'])[:48]:48s} | m[{r['moss_id']}] {str(r['moss_site'])[:34]:34s} sim={r['sim']} tok={r['tok']}")


if __name__ == "__main__":
    candidates(extract())
