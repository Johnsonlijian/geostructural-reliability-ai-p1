"""Extract the KEA2013 Table S1 catalog PDF into a tidy CSV.

Input : ../incoming/kayen2013_table_s1_supplemental.pdf
Output: ../incoming/kayen2013_table_s1.csv  (+ extraction report on stdout)

Row grammar (one case per line in the PDF text layer):
  <case_id> <LOCATION text> <Mw> ± <sd> <Yes|No|...> <d1> - <d2> <gwt>
  <sigma_vo> ± <sd> <sigma_eff> ± <sd> <amax> ± <sd> <rd> ± <sd> <csr> ± <sd>
  <vs1> ± <sd> <reference>
The plus-minus glyph extracts as garbled bytes; we match a tolerant separator.
Earthquake/region come from section header lines like
  "1948 Fukui M7.1 Earthquake, Japan".
Erratum (DOI 10.1061/(ASCE)GT.1943-5606.0001390) site-name fixes for 171-173
are applied after parsing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pdfplumber

ROUND = Path(__file__).resolve().parents[1]
SRC = ROUND / "incoming" / "kayen2013_table_s1_supplemental.pdf"
OUT = ROUND / "incoming" / "kayen2013_table_s1.csv"

PM = r"(?:\u00b1|\ufffd+|\?\?)"  # plus-minus glyph or its mojibake
NUM = r"[\d.]+"

ROW = re.compile(
    rf"^(?P<case_id>\d+[A-Za-z]?)\s+(?P<site>.+?)\s+"
    rf"(?P<mw>{NUM})\s*{PM}\s*(?P<mw_sd>{NUM})\s+"
    rf"(?P<liq>YES/Marg\.?|NO/Marg\.?|Yes/Marg\.?|No/Marg\.?|YES|NO|N0|Yes|No|yes|no|Marginal|MARGINAL|Yes/No)\s+"
    rf"(?P<d1>{NUM})\s*-\s*(?P<d2>{NUM})\s+(?P<gwt>{NUM})\s+"
    rf"(?P<svo>{NUM})\s*{PM}\s*(?P<svo_sd>{NUM})\s+"
    rf"(?P<sveff>{NUM})\s*{PM}\s*(?P<sveff_sd>{NUM})\s+"
    rf"(?P<amax>{NUM})\s*{PM}\s*(?P<amax_sd>{NUM})\s+"
    rf"(?P<rd>{NUM})\s*{PM}\s*(?P<rd_sd>{NUM})\s+"
    rf"(?P<csr>{NUM})\s*{PM}\s*(?P<csr_sd>{NUM})\s+"
    rf"(?P<vs1>{NUM})\s*{PM}\s*(?P<vs1_sd>{NUM})\s+"
    rf"(?P<ref>.+)$"
)

HEADER = re.compile(
    r"^(?P<year>(?:18|19|20)\d{2})\s+(?P<body>.*?(?:Earthquakes?|Aftershocks?|Aftershook|Mainshock|Events).*)$")
DATE_LIKE = re.compile(r"^(January|February|March|April|May|June|July|August|September|October|November|December|\d)")

SKIP_PREFIXES = (
    "SUPPLEMENTAL DATA", "ASCE Journal", "Shear Wave Velocity", "R. Kayen",
    "DOI:", "site ID", "Range (m)", "w vo vo", "Table S1", "Page ",
)

ERRATUM_SITE_FIX = {  # Kayen et al. (2015) erratum, Nishinomiya City parks
    "171": "Kotsu Koen, Nishinomiya",
    "172": "Miyamea Koen, Nishinomiya",
    "173": "Kawazoe Koen, Nishinomiya",
}


def main() -> None:
    rows, unparsed = [], []
    eq_name, eq_region = None, None
    with pdfplumber.open(SRC) as pdf:
        for pg in pdf.pages:
            text = pg.extract_text() or ""
            for raw in text.splitlines():
                line = raw.strip()
                if not line or any(line.startswith(p) for p in SKIP_PREFIXES):
                    continue
                h = HEADER.match(line)
                if h and not ROW.match(line):
                    parts = [p.strip() for p in line.split(",")]
                    tail = parts[-1] if len(parts) > 1 else ""
                    if tail and not DATE_LIKE.match(tail):
                        eq_region = tail
                    eq_name = parts[0]
                    continue
                m = ROW.match(line)
                if m:
                    g = m.groupdict()
                    site = g["site"].strip()
                    if g["case_id"] in ERRATUM_SITE_FIX and (eq_region or "").endswith("Japan"):
                        site = ERRATUM_SITE_FIX[g["case_id"]]
                    liq = g["liq"].strip().lower()
                    label = {"yes": "Yes", "no": "No", "n0": "No"}.get(liq, "Marginal")
                    d1, d2 = float(g["d1"]), float(g["d2"])
                    rows.append({
                        "case_id": g["case_id"],
                        "site": site,
                        "earthquake": eq_name,
                        "region": eq_region,
                        "mw": float(g["mw"]), "mw_sd": float(g["mw_sd"]),
                        "liquefied": label,
                        "crit_depth_lo_m": d1, "crit_depth_hi_m": d2,
                        "crit_depth_mid_m": 0.5 * (d1 + d2),
                        "gwt_m": float(g["gwt"]),
                        "total_stress_kpa": float(g["svo"]), "total_stress_sd": float(g["svo_sd"]),
                        "sigma_eff_kpa": float(g["sveff"]), "sigma_eff_sd": float(g["sveff_sd"]),
                        "amax_g": float(g["amax"]), "amax_sd": float(g["amax_sd"]),
                        "rd": float(g["rd"]), "rd_sd": float(g["rd_sd"]),
                        "csr": float(g["csr"]), "csr_sd": float(g["csr_sd"]),
                        "vs1_ms": float(g["vs1"]), "vs1_sd": float(g["vs1_sd"]),
                        "reference": g["ref"].strip(),
                    })
                else:
                    unparsed.append(line)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"parsed rows: {len(df)}")
    print(f"events: {df['earthquake'].nunique()}")
    print(f"labels: {df['liquefied'].value_counts().to_dict()}")
    print(f"unparsed candidate lines: {len(unparsed)}")
    for u in unparsed[:15]:
        print("  UNPARSED:", u[:160])


if __name__ == "__main__":
    main()
